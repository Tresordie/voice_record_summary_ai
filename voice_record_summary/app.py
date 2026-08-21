import asyncio
import json
import logging
import os
import re
import struct
import subprocess
import threading
import time
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path

import jieba
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("voice_app")

app = FastAPI(title="语音录制 & 总结")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
SUMMARIES_DIR = BASE_DIR / "summaries"

for d in [RECORDINGS_DIR, TRANSCRIPTS_DIR, SUMMARIES_DIR]:
    d.mkdir(exist_ok=True)

_funasr_models = {}  # cache offline FunASR models keyed by model path
_streaming_models = {}  # cache streaming models keyed by model path
_stream_generate_locks = {}  # per-model locks: FunASR streaming generate is not thread-safe

# Legacy Whisper model names, kept only to remap old saved settings.
_LEGACY_WHISPER_NAMES = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
}

DEFAULT_LOCAL_MODEL = "iic/SenseVoiceSmall"


def _normalize_local_model(model_path):
    """Remap legacy Whisper model names to the default local FunASR model."""
    name = (model_path or "").strip()
    if not name or name in _LEGACY_WHISPER_NAMES:
        return DEFAULT_LOCAL_MODEL
    return name


def _modelscope_cache_dir():
    """Locate the ModelScope local model cache directory (if present)."""
    env = os.environ.get("MODELSCOPE_CACHE")
    if env:
        hub = Path(env) / "hub" / "models"
        if hub.is_dir():
            return hub
    hub = Path.home() / ".cache" / "modelscope" / "hub" / "models"
    return hub if hub.is_dir() else None


def _is_valid_model_dir(p):
    """A usable FunASR model dir must contain config.yaml plus weight files."""
    try:
        if not p.is_dir():
            return False
        if not (p / "config.yaml").exists():
            return False
        return any(
            f.suffix in (".pt", ".onnx", ".bin")
            for f in p.iterdir()
            if f.is_file()
        )
    except OSError:
        return False


def _find_aux_models(model_path):
    """Locate local FSMN-VAD / CT-PUNC auxiliary models for ASR.

    Search order:
    1. sibling directories next to the model itself;
    2. the ModelScope cache under iic/ and damo/ namespaces.

    Returns (vad_path_or_None, punc_path_or_None).
    """
    vad_names = ["speech_fsmn_vad_zh-cn-16k-common-pytorch", "fsmn-vad"]
    punc_names = [
        "punc_ct-transformer_cn-en-common-vocab471067-large",
        "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "ct-punc",
    ]
    candidates = []
    p = Path(model_path)
    if p.parent.exists():
        candidates.append(p.parent)
    cache = _modelscope_cache_dir()
    if cache:
        candidates.extend([cache / "iic", cache / "damo"])

    vad_path = punc_path = None
    for base in candidates:
        if vad_path is None:
            for name in vad_names:
                if _is_valid_model_dir(base / name):
                    vad_path = str(base / name)
                    break
        if punc_path is None:
            for name in punc_names:
                if _is_valid_model_dir(base / name):
                    punc_path = str(base / name)
                    break
        if vad_path and punc_path:
            break
    return vad_path, punc_path


def get_funasr_model(model_path):
    """Load and cache an offline FunASR model (SenseVoice / Paraformer).

    Local FSMN-VAD / CT-PUNC auxiliary models are attached only when found
    on disk (offline-friendly: never triggers a download).
    """
    if model_path not in _funasr_models:
        from funasr import AutoModel

        vad_path, punc_path = _find_aux_models(model_path)
        _funasr_models[model_path] = AutoModel(
            model=model_path,
            vad_model=vad_path,
            punc_model=punc_path,
            trust_remote_code=True,
            disable_update=True,
        )
    return _funasr_models[model_path]


def get_streaming_model(model_path="paraformer-zh-streaming"):
    """Load and cache FunASR streaming ASR model (lazy, keyed by path).

    Streaming (online) models such as paraformer-realtime /
    speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
    are loaded WITHOUT VAD/PUNC: FunASR's inference_with_vad pipeline is
    whole-utterance oriented (it reloads the full input per generate call
    and expects a scalar ms ``chunk_size`` for VAD), which conflicts with
    per-chunk WebSocket streaming and crashes on the paraformer
    ``chunk_size=[l,c,r]`` list; the punc stage only runs inside that
    pipeline anyway. Punctuation and hallucination filtering for streaming
    sessions are provided by the offline 2-pass refinement (SenseVoice +
    FSMN-VAD + CT-PUNC) when the recording stops.
    """
    if model_path not in _streaming_models:
        from funasr import AutoModel

        _streaming_models[model_path] = AutoModel(
            model=model_path,
            trust_remote_code=True,
            disable_update=True,
        )
    return _streaming_models[model_path]


def _find_offline_refine_model():
    """Locate a local offline model for 2-pass refinement after streaming.

    Preference: SenseVoiceSmall (multilingual) > paraformer-large with
    built-in VAD/PUNC > plain paraformer-large.
    """
    cache = _modelscope_cache_dir()
    if not cache:
        return None
    names = [
        "iic/SenseVoiceSmall",
        "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    ]
    for name in names:
        p = cache / name
        if _is_valid_model_dir(p):
            return str(p)
    return None


def _offline_refine(wav_path, language="zh"):
    """Re-transcribe the saved WAV with a local offline model (2-pass).

    Returns refined text, or None when no offline model is available or
    refinement fails (caller keeps the streaming text).
    """
    model_path = _find_offline_refine_model()
    if not model_path:
        return None
    try:
        text, _ = transcribe_local(str(wav_path), model_path, language)
        return text or None
    except Exception as e:
        logger.warning("Offline refinement failed: %s", e)
        return None


def transcribe_sensevoice(audio_path, model_path):
    model = get_funasr_model(model_path)
    result = model.generate(input=str(audio_path))
    if result and len(result) > 0:
        text = result[0].get("text", "").strip()
        # SenseVoice emits event tokens like <|zh|><|Speech|><|withitn|>;
        # strip them (including space-broken variants) from the output.
        text = re.sub(r"<\s*\|[^|]*\|\s*>", "", text)
        return text.strip()
    return ""


def transcribe_local(audio_path, model_path, language="zh"):
    """Transcribe with the local FunASR engine (SenseVoice / Paraformer)."""
    model_path = _normalize_local_model(model_path)
    engine = (
        "sensevoice"
        if "sensevoice" in Path(str(model_path)).name.lower()
        else "funasr"
    )
    return transcribe_sensevoice(audio_path, model_path), engine


def split_sentences(text):
    parts = re.split(
        r"(?:[。！？；\n，]|[.?!](?<!\d\.\d)(?=\s+|$))(?![a-z])",
        text,
    )
    result = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) > 3 and any(c in buf for c in "。！？；\n.?!。"):
            if buf.strip():
                result.append(buf.strip())
            buf = ""
    if buf.strip():
        result.append(buf.strip())
    return result


def _jieba_tokenize(text):
    """Tokenize with jieba, filtering stopwords and short tokens."""
    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "他", "她", "它", "们", "那", "这个",
        "那个", "什么", "怎么", "哪", "吗", "啊", "嗯", "呢", "吧", "还",
        "能", "可以", "但", "一个", "我们", "他们", "所以", "因为", "不过",
        "然后", "就是", "这边", "那边", "知道", "应该", "需要", "可能",
        "已经", "比较", "如果", "或者", "还是",
    }
    return [
        w for w in jieba.cut(text) if len(w) > 1 and w.strip() and w not in stopwords
    ]


def summarize_local(text, summary_type="会议总结"):
    sentences = split_sentences(text)
    if not sentences:
        return text
    if len(sentences) <= 4:
        chinese_part = "\n".join(f"- {s}" for s in sentences)
        return f"## {summary_type}\n\n{chinese_part}\n\n## English Summary\n\n(Local mode: Chinese content extracted. Use online mode for full bilingual output.)"

    # Build TF-IDF weights
    word_df = {}
    sent_words = []
    for i, sent in enumerate(sentences):
        ws = _jieba_tokenize(sent)
        sent_words.append(ws)
        for w in set(ws):
            word_df[w] = word_df.get(w, 0) + 1

    N = len(sentences)
    import math

    # Score each sentence: sum of TF*IDF per word / sqrt(len) for length normalization
    scores = []
    for ws in sent_words:
        if not ws:
            scores.append(0.0)
            continue
        score = sum(ws.count(w) * math.log((N + 1) / (1 + word_df[w])) for w in set(ws))
        scores.append(score / math.sqrt(len(ws)))

    # Select top ~35% sentences, minimum 5
    top_k = max(5, int(N * 0.35))
    ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:top_k])

    # Extract keywords
    keywords = jieba.analyse.extract_tags(text, topK=6)

    # Group consecutive sentences into paragraphs for readability
    groups = []
    prev = -2
    for idx in top_indices:
        if idx == prev + 1 and groups:
            groups[-1].append(sentences[idx])
        else:
            groups.append([sentences[idx]])
        prev = idx

    chinese_lines = []
    for g in groups:
        merged = " ".join(g)
        chinese_lines.append(f"- {merged}")

    chinese_content = "\n".join(chinese_lines)

    result = [
        f"## {summary_type}",
        "",
    ]
    if keywords:
        result.append(f"**关键词:** {' | '.join(keywords)}")
        result.append("")

    result.append(chinese_content)
    result.append("")
    result.append("## English Summary")
    result.append("")
    result.append(
        "(Local mode: Chinese content extracted. Use online mode with AI for full bilingual translation.)"
    )

    return "\n".join(result)


def _make_minimal_wav():
    """Create a minimal valid WAV file (0.1s silence) for testing STT endpoint."""
    sample_rate = 16000
    num_samples = int(sample_rate * 0.1)
    samples = b"\x00\x00" * num_samples
    buf = BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(samples)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<H", 2))
    buf.write(struct.pack("<H", 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(samples)))
    buf.write(samples)
    buf.seek(0)
    return buf


_DASHSCOPE_FORMAT_MAP = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".opus": "opus",
    ".ogg": "opus",
    ".aac": "aac",
    ".m4a": "aac",
    ".amr": "amr",
    ".speex": "speex",
}


def _ensure_supported_format(audio_path):
    """Convert audio to 16kHz mono WAV if format is not directly supported by DashScope."""
    ext = Path(audio_path).suffix.lower()
    if ext in _DASHSCOPE_FORMAT_MAP:
        return audio_path, _DASHSCOPE_FORMAT_MAP[ext]

    wav_path = Path(audio_path).with_suffix(".wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(wav_path),
        ],
        capture_output=True,
        check=True,
    )
    return wav_path, "wav"


def transcribe_dashscope(audio_path, api_key, model, language):
    """Use Alibaba Cloud DashScope native Real-time ASR for local files."""
    from http import HTTPStatus

    import dashscope
    from dashscope.audio.asr import Recognition

    dashscope.api_key = api_key

    model = model or "fun-asr-realtime"

    lang_hints_map = {
        "zh": ["zh"], "en": ["en"], "ja": ["ja"], "ko": ["ko"],
        "yue": ["yue"], "de": ["de"], "fr": ["fr"], "ru": ["ru"],
    }
    lang_hints = lang_hints_map.get(language, ["zh", "en"])

    src_path, fmt = _ensure_supported_format(audio_path)

    recognition = Recognition(
        model=model,
        format=fmt,
        sample_rate=16000,
        language_hints=lang_hints,
        callback=None,
    )
    result = recognition.call(str(src_path))

    if result.status_code == HTTPStatus.OK:
        sentence = result.get_sentence()
        if isinstance(sentence, dict):
            return sentence.get("text", "").strip()
        if isinstance(sentence, list):
            return "".join(s.get("text", "") for s in sentence).strip()
        return ""

    raise Exception(f"DashScope ASR 失败: {result.message}")


def _is_dashscope(api_base):
    return "dashscope" in (api_base or "").lower()


def _chat_completion(api_key, api_base, model, messages, temperature=0.1, max_tokens=4096):
    """Call an OpenAI-compatible chat/completions endpoint via plain HTTP.

    DeepSeek V4 (``deepseek-v4-flash`` / ``deepseek-v4-pro``) defaults to
    thinking mode ON: the model spends the ``max_tokens`` budget on
    ``reasoning_content`` and leaves the final ``content`` empty, so the
    summary/correction comes back blank. We explicitly disable thinking for
    DeepSeek so the answer lands in ``content`` (and ``temperature`` becomes
    effective again). As a safety net, fall back to ``reasoning_content`` if
    ``content`` is somehow still empty.
    """
    import requests

    base = (api_base or "https://api.deepseek.com").rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    is_deepseek = (
        "deepseek" in (api_base or "").lower()
        or str(model or "").lower().startswith("deepseek-v4")
    )
    if is_deepseek:
        payload["thinking"] = {"type": "disabled"}
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    message = resp.json()["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    if not content:
        content = (message.get("reasoning_content") or "").strip()
    return content


def transcribe_online(audio_path, api_key, api_base, model, language):
    """Use an OpenAI-compatible audio/transcriptions endpoint."""
    import requests

    base = (api_base or "https://api.deepseek.com").rstrip("/")
    data = {"model": model or "fun-asr-realtime", "response_format": "text"}
    if language and language != "auto":
        data["language"] = language
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files={"file": (Path(audio_path).name, f, "audio/wav")},
            timeout=600,
        )
    resp.raise_for_status()
    body = resp.text.strip()
    if body.startswith("{"):
        return resp.json().get("text", "").strip()
    return body


# Core principles injected into every system prompt
_FAITHFULNESS_RULES = (
    "严格遵守以下原则：\n"
    "1. 忠实原文——只基于原文内容进行总结，绝不添加原文中没有的信息、数据、人名或结论；\n"
    "2. 语音纠错——输入来自语音识别，可能含有同音错字（如'攻破'实为'工装'）、缺标点、口语填充词，需根据上下文推断正确含义，修正明显的识别错误；\n"
    "3. 先理解后输出——先通读全文判断主题和意图，再提取关键信息，最后按格式输出；\n"
    "4. 尽力提取——输入是口语化的会议实录，即使表达碎片化、有识别错误，也要尽最大努力提取讨论要点、决定和待办；只有当原文确实完全没有相关内容时，才允许在某部分标注（无/None），绝不允许所有部分全部留空。"
)

SUMMARY_PROMPTS = {
    "会议总结": {
        "system": (
            "你是专业的会议记录总结助手。你的任务是将语音识别的原始文本整理为结构化的详细会议纪要，"
            "先输出完整的中文版本，再输出完整的英文版本。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 用1-2句话概括会议核心主题和目的\n"
            "2. 按话题分组提取讨论要点（每个话题2-4句话，包含各方观点）\n"
            "3. 列出明确做出的决定（如有）\n"
            "4. 列出待跟进事项及负责人（如原文提及）\n\n"
            "输出格式（严格按以下顺序）：\n"
            "## 会议总结\n\n"
            "### 会议主题\n"
            "### 关键讨论\n"
            "### 决策事项\n"
            "### 待办事项\n\n"
            "## English Summary\n\n"
            "### Meeting Topic\n"
            "### Key Discussions\n"
            "### Decisions\n"
            "### Action Items\n\n"
            "注意：某部分无内容则标注（无/None），切勿编造。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "今日计划": {
        "system": (
            "你是专业的待办事项和计划整理助手。你的任务是从语音识别文本中提取所有计划、任务和安排，"
            "先输出完整的中文版本，再输出完整的英文版本。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 先判断文本中是否包含计划/任务/安排类内容\n"
            "2. 提取所有明确的任务项，区分今日任务和近期安排\n"
            "3. 仅提取原文明确提及的任务，不要推测或补充\n\n"
            "输出格式（严格按以下顺序）：\n"
            "## 今日计划\n\n"
            "### 今日任务\n"
            "（按优先级排列，标注预计时间——仅当原文提及）\n"
            "### 近期安排\n"
            "（非今日但近期的事项，标注大致时间节点——仅当原文提及）\n"
            "### 备注\n"
            "（其他提醒和想法）\n\n"
            "## English Summary\n\n"
            "### Today's Tasks\n"
            "### Upcoming\n"
            "### Notes\n\n"
            "注意：原文未提及的时间/优先级不要编造，某部分无内容则标注（无/None）。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "学习笔记": {
        "system": (
            "你是专业的学习笔记整理助手。你的任务是从语音识别文本中提取知识点、概念和关键信息，"
            "先输出完整的中文版本，再输出完整的英文版本。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 用1-2句话概括学习主题\n"
            "2. 提取核心概念并简要解释（保留原文专业术语）\n"
            "3. 按逻辑层次归纳主要知识点\n"
            "4. 标注文中提及但未深入、值得后续学习的内容\n\n"
            "输出格式（严格按以下顺序）：\n"
            "## 学习笔记\n\n"
            "### 主题\n"
            "### 核心概念\n"
            "### 要点归纳\n"
            "### 待深入\n\n"
            "## English Summary\n\n"
            "### Topic\n"
            "### Core Concepts\n"
            "### Key Points\n"
            "### Further Study\n\n"
            "注意：概念解释必须基于原文，不要曲解或添加原文未提及的定义。某部分无内容则标注（无/None）。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "快速摘要": {
        "system": (
            "你是专业的文本摘要助手。你的任务是用简洁的语言概括语音识别文本的核心内容，"
            "先输出完整的中文摘要（3-5句话），再输出完整的英文摘要（3-5 sentences）。\n"
            + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 先判断文本的主题和最核心的3-5个信息点\n"
            "2. 用自己的话简洁概括，不要摘抄原文句子\n"
            "3. 中文3-5句话，英文3-5 sentences\n\n"
            "输出格式（严格按以下顺序）：\n"
            "## 快速摘要\n\n"
            "（3-5句话）\n\n"
            "## English Summary\n\n"
            "（3-5 sentences）\n\n"
            "注意：只保留最重要的信息，忽略口语填充和重复内容，不要添加原文未提及的信息。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "待办事项": {
        "system": (
            "你是专业的任务管理助手。你的任务是从语音识别文本中精确提取所有待办事项和任务，"
            "先输出完整的中文清单，再输出完整的英文清单。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 区分明确的任务项和一般性讨论——只提取真正需要执行的事项\n"
            "2. 每条任务提取：行动描述、负责人（如有）、时间节点（如有）\n"
            "3. 按优先级排列\n\n"
            "输出格式（严格按以下顺序）：\n"
            "## 待办事项\n\n"
            "- [ ] 任务描述（负责人: XX，截止: XX）\n\n"
            "### 备注\n"
            "（补充说明等非任务信息）\n\n"
            "## English Summary\n\n"
            "- [ ] Task description（person: XX，due: XX）\n\n"
            "### Notes\n"
            "（补充说明等非任务信息）\n\n"
            "注意：原文未提及的负责人或时间标注「未提及/N/A」，不要编造。某部分无内容则标注（无/None）。\n\n"
            "原始文本:\n{text}"
        ),
    },
}


def _build_custom_prompt(summary_type):
    """Build prompt template for a custom summary type not in predefined list."""
    system = (
        f"你是专业的「{summary_type}」整理助手。你的任务是将语音识别的原始文本"
        f"按照「{summary_type}」的格式进行整理和总结，先输出完整的中文版本，再输出完整的英文版本。\n"
        + _FAITHFULNESS_RULES
    )
    prompt = (
        f"请将以下语音识别文本整理为「{summary_type}」格式的结构化内容。\n\n"
        "步骤：\n"
        "1. 用1-2句话概括文本主题\n"
        "2. 提取关键信息点\n"
        f"3. 按「{summary_type}」的格式合理分段输出\n\n"
        "输出格式（严格按以下顺序）：\n"
        f"## {summary_type}\n\n"
        "（根据主题合理分段，确保信息完整、结构清晰）\n\n"
        "## English Summary\n\n"
        "（Corresponding English version with the same structure）\n\n"
        "注意：不添加原文中不存在的信息，某部分无内容则标注（无/None）。\n\n"
        "原始文本:\n{text}"
    )
    return system, prompt


def _build_context_hint(context):
    """Build a system-prompt hint describing the user-provided context.

    The context (meeting topic, domain jargon, proper nouns, participants,
    background...) is comprehension aid ONLY: it must never be copied into
    the output as if it were part of the transcript.
    """
    return (
        "\n\n【语境背景】\n"
        "用户提供了以下当前语境说明（如主题、行业背景、专有名词、人名、讲话背景等），"
        "用于帮助你准确理解原文的主题、术语和意图：\n"
        f"{context}\n\n"
        "注意：语境仅用于辅助理解，总结内容必须仍然只来源于原文，"
        "不得把语境信息本身当作原文内容写入总结。"
    )


def _summarize_single(text, api_key, api_base, model, summary_type, context=None):
    """One-shot structured bilingual summarization."""
    if summary_type in SUMMARY_PROMPTS:
        cfg = SUMMARY_PROMPTS[summary_type]
        system = cfg["system"]
        prompt = cfg["prompt"].format(text=text)
    else:
        system, prompt_template = _build_custom_prompt(summary_type)
        prompt = prompt_template.format(text=text)

    if context:
        system += _build_context_hint(context)

    return _chat_completion(
        api_key,
        api_base,
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )


_SUMMARY_DIRECT_LIMIT = 6000  # chars: below this, summarize in a single call
_SUMMARY_CHUNK_SIZE = 6000    # chars: max chunk size for long-text map phase
_SUMMARY_MAX_CHUNKS = 10      # bound cost for extremely long recordings


def _chunk_text(text, chunk_size):
    """Split text into chunks at sentence boundaries (hard-cut if needed)."""
    sentences = split_sentences(text)
    chunks, cur = [], ""
    for s in sentences:
        if len(s) > chunk_size:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(s), chunk_size):
                chunks.append(s[i:i + chunk_size])
            continue
        if cur and len(cur) + len(s) + 1 > chunk_size:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur)
    if len(chunks) > _SUMMARY_MAX_CHUNKS:
        step = len(chunks) / _SUMMARY_MAX_CHUNKS
        merged = []
        for k in range(_SUMMARY_MAX_CHUNKS):
            i0 = int(k * step)
            i1 = len(chunks) if k == _SUMMARY_MAX_CHUNKS - 1 else int((k + 1) * step)
            merged.append("\n".join(chunks[i0:i1]))
        chunks = merged
    return chunks


def summarize_online(text, api_key, api_base, model, summary_type="会议总结", context=None):
    """Use OpenAI-compatible Chat API for structured bilingual summarization.

    Long transcripts (e.g. hour-long recordings) are summarized with a
    map-reduce scheme: each chunk is condensed in parallel, then the
    condensed points are fed into the final structured summarization.
    This avoids context-overflow/timeout failures on very long input.
    """
    model = model or "deepseek-v4-flash"
    logger.info("Summary request: %s chars, model=%s", len(text), model)
    if len(text) <= _SUMMARY_DIRECT_LIMIT:
        return _summarize_single(text, api_key, api_base, model, summary_type, context)

    from concurrent.futures import ThreadPoolExecutor

    chunks = _chunk_text(text, _SUMMARY_CHUNK_SIZE)
    logger.info("Long-text summary: %s chars -> %s chunks", len(text), len(chunks))

    map_system = (
        "你是长文本总结的预处理模块。用户会给你一份长录音转写文本中的一个片段，"
        "请详细整理该片段包含的全部信息要点（话题、观点、数据、决定、待办、时间、人物），"
        "按要点列表输出，不得遗漏、不得编造。"
    )
    if context:
        map_system += _build_context_hint(context)

    def map_one(args):
        i, chunk = args
        return _chat_completion(
            api_key,
            api_base,
            model,
            [
                {
                    "role": "system",
                    "content": map_system,
                },
                {
                    "role": "user",
                    "content": (
                        f"这是长录音转写文本的第 {i + 1}/{len(chunks)} 段，请整理要点：\n\n{chunk}"
                    ),
                },
            ],
            max_tokens=1500,
        )

    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as ex:
        partials = list(ex.map(map_one, enumerate(chunks)))

    merged = "\n\n".join(
        f"【第 {i + 1} 段要点】\n{p}" for i, p in enumerate(partials)
    )
    logger.info("Long-text summary: merged points %s chars", len(merged))
    return _summarize_single(merged, api_key, api_base, model, summary_type, context)


def correct_asr_text(text, api_key, api_base, model, context=None, context_prompt=None):
    """Use LLM to correct and polish raw ASR output.

    When ``context`` is given, only ``text`` (the new delta) is polished while
    ``context`` serves purely for comprehension — this keeps the call fast
    and prevents previously polished content from being rewritten or lost.

    ``context_prompt`` is the user-provided description of the current
    situation (topic, jargon, participants...) and is injected as background
    so homophone errors are resolved against the right domain.
    """
    ctx_hint = ""
    if context_prompt:
        ctx_hint = (
            "\n\n【语境背景】\n"
            "用户提供了以下当前语境说明（如主题、行业背景、专有名词、人名等），"
            "请结合该语境理解对话主题，并据此推断同音错字的正确写法：\n"
            f"{context_prompt}"
        )
    if context is not None:
        return _chat_completion(
            api_key,
            api_base,
            model or "deepseek-v4-flash",
            [
                {
                    "role": "system",
                    "content": (
                        "你是专业的语音识别文本纠错助手。用户会提供 <context>（已处理过的前文，仅用于理解语境）和 <new>（新增的语音识别原始文本）。\n"
                        "你只需处理 <new>：根据语义补充标点和停顿、修正同音错字、去除无意义的语气填充词（嗯、啊等）。\n"
                        "严格做到：不删除、不遗漏、不压缩、不改写 <new> 中的任何语义信息；不输出 <context> 的任何内容。\n"
                        "只输出纠正后的 <new> 纯文本，不要任何额外解释。"
                        + ctx_hint
                    ),
                },
                {
                    "role": "user",
                    "content": f"<context>\n{context}\n</context>\n<new>\n{text}\n</new>",
                },
            ],
            max_tokens=2048,
        )
    return _chat_completion(
        api_key,
        api_base,
        model or "deepseek-v4-flash",
        [
            {
                "role": "system",
                "content": (
                    "你是专业的语音识别文本纠错助手。语音识别输出的原始文本没有标点符号、"
                    "没有停顿、没有分段，你的核心任务就是将它变成通顺易读的自然文本。\n\n"
                    "工作步骤（严格按顺序执行）：\n"
                    "1. 通读全文，判断语境、场景和意图（会议？日常对话？学习笔记？）\n"
                    "2. 根据语义补充标点符号——这是最重要的步骤：\n"
                    "   - 每个句子末尾加句号（。）、问号（？）或感叹号（！）\n"
                    "   - 句中停顿处加逗号（，）、顿号（、）、分号（；）\n"
                    "   - 直接引语需加引号\n"
                    "   - 宁可多加逗号，不可让长句没有停顿\n"
                    "3. 合理分段，不同话题之间用空行分隔\n"
                    "4. 根据语境修正同音错字（如技术会议中'攻破'→'工装'）\n"
                    "5. 去除口语填充词（嗯、啊、那个、就是说、然后呢等）\n"
                    "6. 修正明显的语法不通顺\n\n"
                    "严格遵循：忠实原意，不添加原文没有的信息，不改变表达风格，不删除、不遗漏、不压缩原文任何信息。\n"
                    "只输出纠错后的纯文本，不要任何额外解释。"
                    + ctx_hint
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请为以下无标点的语音识别文本添加标点符号和停顿，并修正识别错误：\n\n{text}"
                ),
            },
        ],
    )


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    mode: str = Form("local"),
    language: str = Form("zh"),
    model: str = Form(""),
    api_key_stt: str = Form(""),
    api_key: str = Form(""),
    api_base_stt: str = Form(""),
    api_base: str = Form(""),
    stt_model: str = Form("fun-asr-realtime"),
    summary_type: str = Form("会议总结"),
    api_key_summary: str = Form(""),
    api_base_summary: str = Form(""),
    summary_model: str = Form("deepseek-v4-flash"),
    context: str = Form(""),
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = _normalize_local_model(model)

    ext = os.path.splitext(audio.filename or "recording.webm")[1] or ".webm"
    audio_path = RECORDINGS_DIR / f"{timestamp}{ext}"

    transcript = ""
    stt_engine = "unknown"
    stt_error = None
    try:
        audio_path.write_bytes(await audio.read())
        logger.info(
            "[STT] Saved audio: %s (%s bytes), model: %s, mode: %s",
            audio_path,
            audio_path.stat().st_size,
            model_path,
            mode,
        )

        if mode == "online":
            key = api_key_stt or api_key
            base = api_base_stt or api_base
            if key:
                try:
                    if _is_dashscope(base):
                        transcript = await asyncio.to_thread(
                            transcribe_dashscope, audio_path, key, stt_model, language
                        )
                    else:
                        transcript = await asyncio.to_thread(
                            transcribe_online, audio_path, key, base, stt_model, language
                        )
                    stt_engine = "online"
                except Exception as e:
                    stt_error = str(e)
                    logger.warning("Online STT failed, falling back to local: %s", e)

            if not transcript:
                transcript, stt_engine = await asyncio.to_thread(
                    transcribe_local, audio_path, model_path, language
                )
        else:
            transcript, stt_engine = await asyncio.to_thread(
                transcribe_local, audio_path, model_path, language
            )

        logger.info(
            "[STT] Engine: %s, transcript length: %s chars", stt_engine, len(transcript)
        )
    except Exception as e:
        logger.error("[STT] Error: %s", e)
        return JSONResponse({"error": f"语音识别失败: {str(e)}"}, status_code=500)

    # --- Summarization ---
    summary = ""
    if transcript:
        try:
            if mode in ("online", "hybrid") and api_key_summary:
                summary = await asyncio.to_thread(
                    summarize_online,
                    transcript,
                    api_key_summary,
                    api_base_summary,
                    summary_model,
                    summary_type,
                    (context or "").strip() or None,
                )
            else:
                summary = await asyncio.to_thread(
                    summarize_local, transcript, summary_type
                )
        except Exception as e:
            summary = f"(总结生成失败: {e})"

    (TRANSCRIPTS_DIR / f"{timestamp}.txt").write_text(transcript, encoding="utf-8")
    (SUMMARIES_DIR / f"{timestamp}.txt").write_text(summary, encoding="utf-8")

    return {
        "transcript": transcript,
        "summary": summary,
        "audio_file": str(audio_path.name),
        "timestamp": timestamp,
        "stt_engine": stt_engine,
        "stt_error": stt_error,
        "summary_type": summary_type,
    }


@app.post("/api/summarize")
async def summarize_text(request: Request):
    """Re-summarize text (e.g. after user edits the transcript)."""
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "文本为空"}, status_code=400)

    mode = data.get("mode", "local")
    summary_type = data.get("summary_type", "会议总结")
    context = (data.get("context") or "").strip() or None
    try:
        if mode in ("online", "hybrid") and data.get("api_key_summary"):
            summary = await asyncio.to_thread(
                summarize_online,
                text,
                data.get("api_key_summary"),
                data.get("api_base_summary", ""),
                data.get("summary_model", "deepseek-v4-flash"),
                summary_type,
                context,
            )
        else:
            summary = await asyncio.to_thread(summarize_local, text, summary_type)
    except Exception as e:
        return JSONResponse({"error": f"总结生成失败: {e}"}, status_code=500)

    return {"summary": summary, "summary_type": summary_type}


@app.post("/api/correct")
async def correct_text(request: Request):
    """AI correction/polishing for raw ASR output."""
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "文本为空"}, status_code=400)

    api_key = (data.get("api_key") or "").strip()
    if not api_key:
        return JSONResponse({"error": "未提供 API Key"}, status_code=400)

    context_prompt = (data.get("context_prompt") or "").strip() or None
    try:
        corrected = await asyncio.to_thread(
            correct_asr_text,
            text,
            api_key,
            data.get("api_base", ""),
            data.get("model", "deepseek-v4-flash"),
            data.get("context"),
            context_prompt,
        )
        return {"corrected": corrected}
    except Exception as e:
        return JSONResponse({"error": f"纠错失败: {e}"}, status_code=500)


@app.post("/api/save-summary")
async def save_summary(request: Request):
    """Save edited summary text back to file."""
    data = await request.json()
    timestamp = (data.get("timestamp") or "").strip()
    summary_text = data.get("summary") or ""
    if not timestamp:
        return JSONResponse({"error": "缺少时间戳"}, status_code=400)

    (SUMMARIES_DIR / f"{timestamp}.txt").write_text(summary_text, encoding="utf-8")
    return {"success": True}


@app.post("/api/test-connection")
async def test_connection(request: Request):
    """Test if the API key and base URL can connect successfully."""
    data = await request.json()
    api_key = data.get("api_key", "")
    api_base = data.get("api_base", "")
    test_type = data.get("type", "chat")
    model = data.get("model", "")

    if not api_key:
        return {"success": False, "message": "请先输入 API Key"}

    def _run():
        if test_type == "stt" and _is_dashscope(api_base):
            import tempfile
            from http import HTTPStatus

            import dashscope
            from dashscope.audio.asr import Recognition

            dashscope.api_key = api_key
            test_audio = _make_minimal_wav()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(test_audio.read())
                tmp_path = tmp.name
            try:
                recognition = Recognition(
                    model=model or "fun-asr-realtime",
                    format="wav",
                    sample_rate=16000,
                    language_hints=["zh", "en"],
                    callback=None,
                )
                result = recognition.call(tmp_path)
                if result.status_code != HTTPStatus.OK:
                    raise Exception(f"ASR 失败: {result.message}")
            finally:
                os.unlink(tmp_path)
            return "连接成功 (阿里云 ASR 可用)"

        if test_type == "stt":
            import requests

            base = (api_base or "https://api.deepseek.com").rstrip("/")
            resp = requests.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data={"model": model or "fun-asr-realtime", "response_format": "text"},
                files={"file": ("test.wav", _make_minimal_wav(), "audio/wav")},
                timeout=60,
            )
            resp.raise_for_status()
            return "连接成功 (STT API 可用)"

        _chat_completion(
            api_key,
            api_base,
            model or "deepseek-v4-flash",
            [{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return (
            f"连接成功 (模型 {model} 可用)" if model else "连接成功 (Chat API 可用)"
        )

    try:
        msg = await asyncio.to_thread(_run)
        return {"success": True, "message": msg}
    except Exception as e:
        err = str(e)
        if len(err) > 300:
            err = err[:300] + "..."
        return {"success": False, "message": err}


@app.get("/api/history")
async def history():
    items = []
    for f in sorted(RECORDINGS_DIR.glob("*.*"), reverse=True):
        ts = f.stem
        transcript_file = TRANSCRIPTS_DIR / f"{ts}.txt"
        summary_file = SUMMARIES_DIR / f"{ts}.txt"
        items.append(
            {
                "timestamp": ts,
                "audio": f.name,
                "transcript": transcript_file.read_text(encoding="utf-8")
                if transcript_file.exists()
                else "",
                "summary": summary_file.read_text(encoding="utf-8")
                if summary_file.exists()
                else "",
            }
        )
    return items


@app.delete("/api/history/{timestamp}")
async def delete_history_item(timestamp: str):
    """Delete a single recording and its transcript/summary files."""
    deleted = 0
    for pattern, dir_path in [
        ("*", RECORDINGS_DIR),
        (f"{timestamp}.txt", TRANSCRIPTS_DIR),
        (f"{timestamp}.txt", SUMMARIES_DIR),
    ]:
        for f in dir_path.glob(pattern):
            if f.stem == timestamp:
                f.unlink()
                deleted += 1
    if deleted:
        return {"success": True}
    return JSONResponse({"error": "记录不存在"}, status_code=404)


@app.delete("/api/history")
async def clear_history():
    """Delete all recordings, transcripts, and summaries."""
    count = 0
    for dir_path in [RECORDINGS_DIR, TRANSCRIPTS_DIR, SUMMARIES_DIR]:
        for f in dir_path.iterdir():
            f.unlink()
            count += 1
    return {"success": True, "deleted": count}


@app.get("/api/download-summary/{timestamp}")
async def download_summary(timestamp: str):
    """Download summary as a markdown file."""
    summary_path = SUMMARIES_DIR / f"{timestamp}.txt"
    if not summary_path.exists():
        return JSONResponse({"error": "文件不存在"}, status_code=404)

    content = summary_path.read_text(encoding="utf-8")
    md_path = SUMMARIES_DIR / f"{timestamp}.md"
    md_path.write_text(content, encoding="utf-8")
    return FileResponse(
        md_path,
        media_type="text/markdown",
        filename=f"summary_{timestamp}.md",
    )


@app.get("/api/local-models")
async def local_models():
    """List locally cached ModelScope models usable for ASR."""
    cache = _modelscope_cache_dir()
    streaming, offline = [], []
    if cache:
        for ns_dir in sorted(cache.iterdir()):
            if not ns_dir.is_dir() or ns_dir.name.startswith("."):
                continue
            for mdir in sorted(ns_dir.iterdir()):
                if not _is_valid_model_dir(mdir):
                    continue
                name = mdir.name.lower()
                entry = {
                    "path": str(mdir),
                    "id": f"{ns_dir.name}/{mdir.name}",
                }
                if "online" in name or "streaming" in name or "realtime" in name:
                    streaming.append(entry)
                elif (
                    "paraformer" in name or "sensevoice" in name
                ) and "vad" not in name.replace("-vad-punc", ""):
                    offline.append(entry)
    return {
        "cache_dir": str(cache) if cache else None,
        "streaming": streaming,
        "offline": offline,
    }


@app.post("/api/preload-streaming-model")
async def preload_streaming_model(request: Request):
    """Pre-load the streaming ASR model so that the first recognition starts instantly."""
    data = await request.json() or {}
    model_path = data.get("model", "paraformer-zh-streaming")
    try:
        await asyncio.to_thread(get_streaming_model, model_path)
        return {"success": True, "model": model_path}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# WebSocket: Local streaming ASR (FunASR paraformer streaming)
# ---------------------------------------------------------------------------
def _is_sentence_final(result_dict):
    """Check if a streaming result represents a completed sentence.

    FunASR streaming models return a ``timestamp`` field whose last entry is
    ``[start, end]``.  When ``end > -1`` the sentence is considered final.
    """
    ts = result_dict.get("timestamp")
    if ts and isinstance(ts, list) and len(ts) > 0:
        last = ts[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            return last[1] > -1
    return False


def _stream_lock(model_path):
    """Per-model lock serializing FunASR streaming ``generate`` calls.

    A finishing session's final flush and a new session's chunks may hit the
    same cached AutoModel instance concurrently from different threads when
    the user immediately starts the next segment; FunASR generate is not
    thread-safe, so serialize access.
    """
    return _stream_generate_locks.setdefault(model_path, threading.Lock())


def _finish_stream_session(
    model, cache, pcm_buffer, wav_buffer, all_text, sentence_text, lang, chunk_size, lock=None
):
    """Flush the streaming session and save files.

    The slow 2-pass offline refinement deliberately runs AFTER the
    ``completed`` message has been delivered (see the caller), so the UI is
    released immediately and the user can start the next segment right away.

    Returns (outbound_messages, wav_path_or_None, timestamp, full_text).
    """
    msgs = []
    try:
        if len(pcm_buffer) > 0:
            kwargs = dict(
                input=pcm_buffer,
                cache=cache,
                is_final=True,
                language=lang,
                chunk_size=chunk_size,
            )
            if lock:
                with lock:
                    res = model.generate(**kwargs)
            else:
                res = model.generate(**kwargs)
            if res and len(res) > 0:
                text = res[0].get("text", "").strip()
                if text:
                    all_text += text
                    msgs.append(
                        {"text": all_text + sentence_text, "final": True, "is_last": True}
                    )
    except Exception as e:
        logger.warning("Flush error: %s", e)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = RECORDINGS_DIR / f"{timestamp}.wav"
    if wav_buffer:
        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(bytes(wav_buffer))
        except Exception:
            pass
    if not wav_path.exists():
        wav_path = None

    full_text = all_text + sentence_text
    (TRANSCRIPTS_DIR / f"{timestamp}.txt").write_text(full_text, encoding="utf-8")
    msgs.append(
        {
            "status": "completed",
            "full_text": full_text,
            "timestamp": timestamp,
            "refined": False,
        }
    )
    return msgs, wav_path, timestamp, full_text


@app.websocket("/ws/stream")
async def stream_transcribe(ws: WebSocket):
    """Local streaming speech recognition via FunASR streaming model."""
    await ws.accept()
    init_msg = json.loads(await ws.receive_text())
    model_path = init_msg.get("model", "paraformer-zh-streaming")
    language = init_msg.get("language", "zh")
    lang = language if language != "auto" else "zh"

    try:
        model = await asyncio.to_thread(get_streaming_model, model_path)
        await ws.send_text(json.dumps({"status": "ready"}))
    except Exception as e:
        await ws.send_text(
            json.dumps({"status": "error", "message": f"模型加载失败: {e}"})
        )
        await ws.close()
        return

    chunk_size = [5, 10, 5]  # [left_ctx, cur_chunk, right_ctx] in encoder frames
    chunk_stride = chunk_size[1] * 960  # samples per model chunk (9600 @ 16kHz)
    lock = _stream_lock(model_path)  # serialize generate calls across sessions
    cache = {}
    all_text = ""  # committed (finalized) text across all sentences
    sentence_text = ""  # current sentence cumulative text
    pcm_buffer = np.array([], dtype=np.float32)
    wav_buffer = bytearray()  # raw PCM bytes for WAV saving
    last_voice_ts = time.monotonic()  # last chunk containing speech
    silence_flushes = 0  # silent chunks fed since last speech

    async def emit(out):
        nonlocal all_text, sentence_text
        if not out:
            return
        if out["final"]:
            all_text += out["text"]
            sentence_text = ""
            await ws.send_text(json.dumps({"text": all_text, "final": True}))
        else:
            sentence_text += out["text"]
            await ws.send_text(
                json.dumps({"text": all_text + sentence_text, "final": False})
            )

    def run_chunk(chunk):
        with lock:
            res = model.generate(
                input=chunk,
                cache=cache,
                is_final=False,
                language=lang,
                chunk_size=chunk_size,
            )
        if res and len(res) > 0:
            r0 = res[0]
            text = r0.get("text", "").strip()
            if text:
                is_final = _is_sentence_final(r0)
                if not is_final and "is_final" in r0:
                    is_final = bool(r0["is_final"])
                if (
                    not is_final
                    and cache
                    and "encoder" not in cache
                    and "all_fea" not in cache
                ):
                    is_final = True
                return {"final": bool(is_final), "text": text}
        return None

    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                # No WS traffic at all: feed a silent chunk to flush the
                # encoder's right context so trailing words surface promptly.
                silence = np.zeros(chunk_stride, dtype=np.float32)
                try:
                    out = await asyncio.to_thread(run_chunk, silence)
                except Exception as e:
                    logger.warning("Silence flush error: %s", e)
                    continue
                await emit(out)
                continue

            if message["type"] != "websocket.receive":
                break

            # Text message: control command
            if message.get("text"):
                msg = json.loads(message["text"])
                if msg.get("action") == "stop":
                    stop_msgs, wav_path, timestamp, full_text = await asyncio.to_thread(
                        _finish_stream_session,
                        model,
                        cache,
                        pcm_buffer,
                        wav_buffer,
                        all_text,
                        sentence_text,
                        lang,
                        chunk_size,
                        lock,
                    )
                    for m in stop_msgs:
                        await ws.send_text(json.dumps(m))
                    # 2-pass offline refinement AFTER the UI is released:
                    # the user can already start the next segment while this
                    # runs; if it yields better text, push a follow-up.
                    if wav_path:
                        better = await asyncio.to_thread(_offline_refine, str(wav_path), lang)
                        if better and better != full_text:
                            (TRANSCRIPTS_DIR / f"{timestamp}.txt").write_text(
                                better, encoding="utf-8"
                            )
                            try:
                                await ws.send_text(
                                    json.dumps(
                                        {
                                            "status": "refined",
                                            "full_text": better,
                                            "timestamp": timestamp,
                                        }
                                    )
                                )
                            except Exception:
                                logger.warning("Failed to send refined text", exc_info=True)
                    break
                continue

            # Binary message: PCM 16bit 16kHz audio data
            data = message.get("bytes")
            if not data:
                continue
            wav_buffer.extend(data)
            new_pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            pcm_buffer = np.concatenate([pcm_buffer, new_pcm])

            # Voice-activity tracking: the microphone keeps streaming during
            # silence, so detect speech by amplitude rather than by message
            # arrival.
            if float(np.abs(new_pcm).max()) > 0.008:
                last_voice_ts = time.monotonic()
                silence_flushes = 0

            # Process complete chunks from the buffer
            while len(pcm_buffer) >= chunk_stride:
                chunk = pcm_buffer[:chunk_stride]
                pcm_buffer = pcm_buffer[chunk_stride:]
                try:
                    out = await asyncio.to_thread(run_chunk, chunk)
                except Exception as e:
                    logger.warning("Streaming chunk error: %s", e)
                    continue
                await emit(out)

            # During a speech pause, feed up to 3 silent chunks so the
            # encoder's right context is flushed and the sentence's last
            # word(s) surface immediately instead of waiting for the next
            # utterance.
            if (
                silence_flushes < 3
                and time.monotonic() - last_voice_ts >= 0.4
            ):
                silence = np.zeros(chunk_stride, dtype=np.float32)
                silence_flushes += 1
                try:
                    out = await asyncio.to_thread(run_chunk, silence)
                except Exception as e:
                    logger.warning("Silence flush error: %s", e)
                    continue
                await emit(out)
    except Exception as e:
        logger.error("WebSocket stream error: %s", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket: Online streaming ASR (DashScope Recognition)
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream-online")
async def stream_transcribe_online(ws: WebSocket):
    """Online streaming speech recognition via DashScope Recognition API."""
    await ws.accept()
    init_msg = json.loads(await ws.receive_text())
    api_key = init_msg.get("api_key", "")
    model = init_msg.get("model", "fun-asr-realtime")
    language = init_msg.get("language", "zh")

    if not api_key:
        await ws.send_text(json.dumps({"status": "error", "message": "未提供 API Key"}))
        await ws.close()
        return

    import threading

    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue = asyncio.Queue()

    def push(m):
        loop.call_soon_threadsafe(out_queue.put_nowait, m)

    state = {"all": "", "cur": ""}
    completed = threading.Event()

    class StreamCallback(RecognitionCallback):
        def on_event(self, result):
            sentence = result.get_sentence()
            if sentence and "text" in sentence:
                is_end = RecognitionResult.is_sentence_end(sentence)
                text = sentence["text"].strip()
                if text:
                    if is_end:
                        state["all"] += text
                        state["cur"] = ""
                        push({"text": state["all"], "final": True})
                    else:
                        state["cur"] = text
                        push({"text": state["all"] + text, "final": False})

        def on_error(self, result):
            push(
                {
                    "status": "error",
                    "message": str(getattr(result, "message", "Unknown error")),
                }
            )

        def on_complete(self):
            completed.set()

    lang_hints_map = {"zh": ["zh"], "en": ["en"], "ja": ["ja"], "ko": ["ko"]}
    lang_hints = lang_hints_map.get(language, ["zh", "en"])

    try:
        recognition = Recognition(
            model=model,
            format="pcm",
            sample_rate=16000,
            language_hints=lang_hints,
            callback=StreamCallback(),
        )
        await asyncio.to_thread(recognition.start)
    except Exception as e:
        await ws.send_text(json.dumps({"status": "error", "message": str(e)}))
        await ws.close()
        return

    audio_buffer = bytearray()
    done_event = asyncio.Event()

    async def sender():
        while True:
            m = await out_queue.get()
            if m.get("__complete"):
                done_event.set()
                continue
            try:
                await ws.send_text(json.dumps(m))
            except Exception:
                break

    sender_task = asyncio.create_task(sender())
    await ws.send_text(json.dumps({"status": "ready"}))

    try:
        while True:
            message = await ws.receive()
            if message["type"] != "websocket.receive":
                break
            if message.get("text"):
                msg = json.loads(message["text"])
                if msg.get("action") == "stop":
                    await asyncio.to_thread(recognition.stop)
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(completed.wait), 10
                        )
                    except asyncio.TimeoutError:
                        pass

                    full_text = state["all"] + state["cur"]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    if audio_buffer:
                        wav_path = RECORDINGS_DIR / f"{timestamp}.wav"
                        try:
                            with wave.open(str(wav_path), "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                wf.writeframes(bytes(audio_buffer))
                        except Exception:
                            pass

                    (TRANSCRIPTS_DIR / f"{timestamp}.txt").write_text(
                        full_text, encoding="utf-8"
                    )
                    await ws.send_text(
                        json.dumps(
                            {
                                "status": "completed",
                                "full_text": full_text,
                                "timestamp": timestamp,
                            }
                        )
                    )
                    break
                continue

            # Binary PCM data
            data = message.get("bytes")
            if not data:
                continue
            audio_buffer.extend(data)
            try:
                await asyncio.to_thread(recognition.send_audio_frame, data)
            except Exception as e:
                logger.warning("DashScope send_audio_frame error: %s", e)
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        sender_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Optional HTTPS (self-signed) so the mic works over LAN access
# ---------------------------------------------------------------------------
def _ensure_self_signed_cert():
    """Create (once) a self-signed cert covering localhost + local IPs.

    Returns (cert_path, key_path) or None when ``cryptography`` is missing.
    """
    try:
        import datetime as _dt
        import ipaddress
        import socket

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None

    cert_dir = BASE_DIR / "certs"
    cert_dir.mkdir(exist_ok=True)
    cert_file, key_file = cert_dir / "cert.pem", cert_dir / "key.pem"
    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "voice-record-summary-local")]
    )
    san_entries = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        _, _, ips = socket.gethostbyname_ex(socket.gethostname())
        for ip in ips:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except OSError:
        pass

    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))
        .not_valid_after(now + _dt.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_file), str(key_file)


if __name__ == "__main__":
    import socket
    import sys
    import threading

    import uvicorn

    logging.basicConfig(level=logging.INFO)

    def _port_free(port):
        s = socket.socket()
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def _kill_stale_servers(ports):
        """Kill leftover ``python app.py`` processes holding the given ports.

        Only processes whose command line references app.py are terminated,
        so unrelated applications are never touched.
        """
        import subprocess as _sp

        killed = []
        pids = set()
        try:
            out = _sp.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=15
            ).stdout
        except Exception:
            return killed
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 5 or "LISTENING" not in line:
                continue
            local_addr = parts[1]
            if any(local_addr.endswith(f":{p}") for p in ports):
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    pass
        for pid in pids:
            if pid == os.getpid():
                continue
            is_our_app = False
            for cmd in (
                ["wmic", "process", "where", f"ProcessId={pid}", "get",
                 "CommandLine", "/format:list"],
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
            ):
                try:
                    cout = _sp.run(
                        cmd, capture_output=True, text=True, timeout=15
                    ).stdout or ""
                except Exception:
                    continue
                if "app.py" in cout:
                    is_our_app = True
                    break
            if not is_our_app:
                continue
            try:
                _sp.run(
                    ["taskkill", "/F", "/PID", str(pid), "/T"],
                    capture_output=True,
                    timeout=15,
                )
                killed.append(pid)
            except Exception:
                pass
        return killed

    cert = _ensure_self_signed_cert()
    http_port = int(os.environ.get("PORT", "5000"))
    https_port = int(os.environ.get("HTTPS_PORT", "5443"))

    busy = [p for p in (http_port, https_port) if not _port_free(p)]
    if busy:
        killed = _kill_stale_servers(busy)
        if killed:
            print(f"已结束占用端口的旧 app.py 进程: {killed}")
            import time as _time

            for _ in range(20):
                if all(_port_free(p) for p in (http_port, https_port)):
                    break
                _time.sleep(0.3)
        busy = [p for p in (http_port, https_port) if not _port_free(p)]
    if busy:
        print(
            f"\n错误: 端口 {busy} 已被占用（通常是上次运行的旧进程）。\n"
            "请先在终端执行以下命令结束旧进程后重试：\n"
            "  netstat -ano | findstr \":5000 :5443\"\n"
            "  taskkill /F /PID <对应PID> /T\n"
            f"或通过环境变量改用其他端口: set PORT=5001 && set HTTPS_PORT=5444 && python app.py"
        )
        sys.exit(1)

    if cert:
        https_cfg = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=https_port,
            ssl_certfile=cert[0],
            ssl_keyfile=cert[1],
            log_level="warning",
        )
        threading.Thread(target=uvicorn.Server(https_cfg).run, daemon=True).start()

    print()
    print("=" * 60)
    print("  语音录制 & 总结工具 已启动")
    print()
    print("  本机访问:  http://127.0.0.1:%d" % http_port)
    print("  局域网访问: http://localhost:%d" % http_port)
    if cert:
        print("  加密访问:  https://127.0.0.1:%d  (自签名证书)" % https_port)
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=http_port)
