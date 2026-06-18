import json
import os
import re
import struct
import subprocess
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

import jieba
import numpy as np
import whisper
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
SUMMARIES_DIR = BASE_DIR / "summaries"

for d in [RECORDINGS_DIR, TRANSCRIPTS_DIR, SUMMARIES_DIR]:
    d.mkdir(exist_ok=True)

_local_model = None
_sensevoice_model = None

WHISPER_NAMES = {
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


def detect_engine(model_path):
    """Auto-detect engine type from model name or folder path."""
    name = str(Path(model_path).name)
    # Known Whisper model names
    if model_path in WHISPER_NAMES:
        return "whisper"
    # ModelScope SenseVoice ID
    if model_path.startswith("iic/"):
        return "sensevoice"
    # Check filesystem path for characteristic files
    p = Path(model_path)
    if p.exists() and p.is_dir():
        files = {f.name.lower() for f in p.iterdir()} if p.is_dir() else set()
        # SenseVoice: config.yaml + onnx/pt model files
        has_sv = any(name in files for name in ["config.yaml", "config.yml"])
        has_sv = has_sv or any(
            f.endswith(".onnx") for f in files if "sense" in f.lower()
        )
        if has_sv:
            return "sensevoice"
        # Whisper: tokenizer.json or .pt files with model name pattern
        if "tokenizer.json" in files or any(f.endswith(".pt") for f in files):
            return "whisper"
    if p.is_file() and p.suffix in (".pt", ".bin"):
        return "whisper"
    # Default: Whisper (more common, handles model names)
    return "whisper"


def get_local_model(model_name="base"):
    global _local_model
    if _local_model is None:
        _local_model = whisper.load_model(model_name)
    return _local_model


def get_sensevoice_model(model_path):
    global _sensevoice_model
    if _sensevoice_model is None:
        from funasr import AutoModel

        # Try to find auxiliary models (VAD, punctuation) locally
        parent = Path(model_path).parent
        vad_local = parent / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        punc_local = parent / "punc_ct-transformer_cn-en-common-vocab471067-large"

        vad = str(vad_local) if vad_local.exists() else "fsmn-vad"
        punc = str(punc_local) if punc_local.exists() else "ct-punc"

        _sensevoice_model = AutoModel(
            model=model_path,
            vad_model=vad,
            punc_model=punc,
            trust_remote_code=True,
            disable_update=True,
        )
    return _sensevoice_model


def transcribe_sensevoice(audio_path, model_path):
    model = get_sensevoice_model(model_path)
    result = model.generate(input=str(audio_path))
    if result and len(result) > 0:
        return result[0].get("text", "").strip()
    return ""


def transcribe_local(audio_path, model_path, language="zh"):
    """Transcribe with auto-detected local engine."""
    engine = detect_engine(model_path)
    if engine == "sensevoice":
        return transcribe_sensevoice(audio_path, model_path), engine
    else:
        m = get_local_model(model_path)
        result = m.transcribe(
            str(audio_path), language=language if language != "auto" else None
        )
        return result["text"].strip(), engine


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
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "他",
        "她",
        "它",
        "们",
        "那",
        "这个",
        "那个",
        "什么",
        "怎么",
        "哪",
        "吗",
        "啊",
        "嗯",
        "呢",
        "吧",
        "还",
        "能",
        "可以",
        "但",
        "一个",
        "我们",
        "他们",
        "所以",
        "因为",
        "不过",
        "然后",
        "就是",
        "这个",
        "那个",
        "这边",
        "那边",
        "知道",
        "应该",
        "需要",
        "可能",
        "已经",
        "比较",
        "如果",
        "或者",
        "还是",
    }
    return [
        w for w in jieba.cut(text) if len(w) > 1 and w.strip() and w not in stopwords
    ]


def summarize_local(text, summary_type="会议总结"):
    sentences = split_sentences(text)
    if not sentences:
        return text
    if len(sentences) <= 4:
        return "\n".join(f"- {s}" for s in sentences)

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

    result = [f"## {summary_type}", ""]
    if keywords:
        result.append(f"**关键词:** {' | '.join(keywords)}")
        result.append("")

    # Group consecutive sentences into paragraphs for readability
    groups = []
    prev = -2
    for idx in top_indices:
        if idx == prev + 1 and groups:
            groups[-1].append(sentences[idx])
        else:
            groups.append([sentences[idx]])
        prev = idx

    for g in groups:
        merged = " ".join(g)
        result.append(f"- {merged}")

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
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
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
        "zh": ["zh"],
        "en": ["en"],
        "ja": ["ja"],
        "ko": ["ko"],
        "yue": ["yue"],
        "de": ["de"],
        "fr": ["fr"],
        "ru": ["ru"],
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


def transcribe_online(audio_path, api_key, api_base, model, language):
    """Use OpenAI-compatible API for transcription."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base or None)
    lang_param = language if language != "auto" else None

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=model or "whisper-1",
            file=f,
            language=lang_param,
            response_format="text",
        )
    return transcript.strip()


# Core principles injected into every system prompt
_FAITHFULNESS_RULES = (
    "严格遵守以下原则：\n"
    "1. 忠实原文——只基于原文内容进行总结，绝不添加原文中没有的信息、数据、人名或结论；\n"
    "2. 语音纠错——输入来自语音识别，可能含有同音错字（如'攻破'实为'工装'）、缺标点、口语填充词，需根据上下文推断正确含义，修正明显的识别错误；\n"
    "3. 先理解后输出——先通读全文判断主题和意图，再提取关键信息，最后按格式输出。"
)

SUMMARY_PROMPTS = {
    "会议总结": {
        "system": (
            "你是专业的会议记录总结助手。你的任务是将语音识别的原始文本整理为结构化的详细会议纪要，"
            "输出中文和英文两个版本。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 用1-2句话概括会议核心主题和目的\n"
            "2. 按话题分组提取讨论要点（每个话题2-4句话，包含各方观点）\n"
            "3. 列出明确做出的决定（如有）\n"
            "4. 列出待跟进事项及负责人（如原文提及）\n\n"
            "输出格式：\n"
            "## 会议主题 / Meeting Topic\n"
            "## 关键讨论 / Key Discussions\n"
            "## 决策事项 / Decisions\n"
            "## 待办事项 / Action Items\n\n"
            "注意：某部分无内容则标注（无/None），切勿编造。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "今日计划": {
        "system": (
            "你是专业的待办事项和计划整理助手。你的任务是从语音识别文本中提取所有计划、任务和安排，"
            "整理成清晰的中文和英文待办清单。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 先判断文本中是否包含计划/任务/安排类内容\n"
            "2. 提取所有明确的任务项，区分今日任务和近期安排\n"
            "3. 仅提取原文明确提及的任务，不要推测或补充\n\n"
            "输出格式：\n"
            "## 今日计划 / Today's Plan\n"
            "（按优先级排列，标注预计时间——仅当原文提及）\n"
            "## 近期安排 / Upcoming\n"
            "（非今日但近期的事项，标注大致时间节点——仅当原文提及）\n"
            "## 备注 / Notes\n"
            "（其他提醒或想法）\n\n"
            "注意：原文未提及的时间/优先级不要编造，某部分无内容则标注（无/None）。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "学习笔记": {
        "system": (
            "你是专业的学习笔记整理助手。你的任务是从语音识别文本中提取知识点、概念和关键信息，"
            "整理成结构化的学习笔记，输出中文和英文两个版本。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 用1-2句话概括学习主题\n"
            "2. 提取核心概念并简要解释（保留原文专业术语）\n"
            "3. 按逻辑层次归纳主要知识点\n"
            "4. 标注文中提及但未深入、值得后续学习的内容\n\n"
            "输出格式：\n"
            "## 主题 / Topic\n"
            "## 核心概念 / Core Concepts\n"
            "## 要点归纳 / Key Points\n"
            "## 待深入 / Further Study\n\n"
            "注意：概念解释必须基于原文，不要曲解或添加原文未提及的定义。某部分无内容则标注（无/None）。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "快速摘要": {
        "system": (
            "你是专业的文本摘要助手。你的任务是用简洁的语言概括语音识别文本的核心内容，"
            "输出中文和英文两个版本，每个版本3-5句话。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 先判断文本的主题和最核心的3-5个信息点\n"
            "2. 用自己的话简洁概括，不要摘抄原文句子\n"
            "3. 中英文各3-5句话\n\n"
            "输出格式：\n"
            "## 中文摘要\n"
            "（3-5句话）\n\n"
            "## English Summary\n"
            "（3-5 sentences）\n\n"
            "注意：只保留最重要的信息，忽略口语填充和重复内容，不要添加原文未提及的信息。\n\n"
            "原始文本:\n{text}"
        ),
    },
    "待办事项": {
        "system": (
            "你是专业的任务管理助手。你的任务是从语音识别文本中精确提取所有待办事项和任务，"
            "整理成中英文对照的清晰清单。\n" + _FAITHFULNESS_RULES
        ),
        "prompt": (
            "请按以下步骤处理文本：\n"
            "1. 区分明确的任务项和一般性讨论——只提取真正需要执行的事项\n"
            "2. 每条任务提取：行动描述、负责人（如有）、时间节点（如有）\n"
            "3. 按优先级排列\n\n"
            "输出格式：\n"
            "## 待办事项 / Action Items\n"
            "- [ ] 任务描述 / Task description（负责人/person: XX，截止/due: XX）\n\n"
            "## 备注 / Notes\n"
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
        f"按照「{summary_type}」的格式进行整理和总结，输出中文和英文两个版本。\n"
        + _FAITHFULNESS_RULES
    )
    prompt = (
        f"请将以下语音识别文本整理为「{summary_type}」格式的结构化内容。\n\n"
        "步骤：\n"
        "1. 用1-2句话概括文本主题\n"
        "2. 提取关键信息点\n"
        f"3. 按「{summary_type}」的格式合理分段输出\n\n"
        "输出格式：\n"
        "## 中文\n"
        "（根据主题合理分段，确保信息完整、结构清晰）\n\n"
        "## English\n"
        "（Corresponding English version with the same structure）\n\n"
        "注意：不添加原文中不存在的信息，某部分无内容则标注（无/None）。\n\n"
        "原始文本:\n{text}"
    )
    return system, prompt


def summarize_online(text, api_key, api_base, model, summary_type="会议总结"):
    """Use OpenAI-compatible Chat API for structured bilingual summarization."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base or None)

    if summary_type in SUMMARY_PROMPTS:
        cfg = SUMMARY_PROMPTS[summary_type]
        system = cfg["system"]
        prompt = cfg["prompt"].format(text=text)
    else:
        system, prompt_template = _build_custom_prompt(summary_type)
        prompt = prompt_template.format(text=text)

    resp = client.chat.completions.create(
        model=model or "deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return resp.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "未收到音频文件"}), 400

    audio_file = request.files["audio"]
    mode = request.form.get("mode", "local")
    lang = request.form.get("language", "zh")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = request.form.get("model", "base")

    ext = os.path.splitext(audio_file.filename)[1] or ".webm"
    audio_path = RECORDINGS_DIR / f"{timestamp}{ext}"

    # --- Transcription ---
    transcript = ""
    stt_engine = "unknown"
    stt_error = None
    try:
        audio_file.save(str(audio_path))
        print(
            f"[STT] Saved audio: {audio_path} ({audio_path.stat().st_size} bytes), model: {model_path}, mode: {mode}"
        )

        if mode == "online":
            api_key = request.form.get("api_key_stt", "") or request.form.get(
                "api_key", ""
            )
            api_base = request.form.get("api_base_stt", "") or request.form.get(
                "api_base", ""
            )
            stt_model = request.form.get("stt_model", "whisper-1")

            if api_key:
                try:
                    if _is_dashscope(api_base):
                        transcript = transcribe_dashscope(
                            audio_path, api_key, stt_model, lang
                        )
                    else:
                        transcript = transcribe_online(
                            audio_path, api_key, api_base, stt_model, lang
                        )
                    stt_engine = "online"
                except Exception as e:
                    stt_error = str(e)
                    app.logger.warning(
                        "Online STT failed, falling back to local: %s", e
                    )

            if not transcript:
                transcript, stt_engine = transcribe_local(audio_path, model_path, lang)
        else:
            transcript, stt_engine = transcribe_local(audio_path, model_path, lang)

        print(f"[STT] Engine: {stt_engine}, transcript length: {len(transcript)} chars")

    except Exception as e:
        print(f"[STT] Error: {e}")
        return jsonify({"error": f"语音识别失败: {str(e)}"}), 500

    # --- Summarization ---
    summary_type = request.form.get("summary_type", "会议总结")
    summary = ""
    if transcript:
        try:
            if mode in ("online", "hybrid"):
                api_key = request.form.get("api_key_summary", "")
                api_base = request.form.get("api_base_summary", "")
                summary_model = request.form.get("summary_model", "deepseek-chat")
                if api_key:
                    summary = summarize_online(
                        transcript, api_key, api_base, summary_model, summary_type
                    )
                else:
                    summary = summarize_local(transcript, summary_type)
            else:
                summary = summarize_local(transcript, summary_type)
        except Exception as e:
            summary = f"(总结生成失败: {e})"

    # Save files
    transcript_path = TRANSCRIPTS_DIR / f"{timestamp}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")

    summary_path = SUMMARIES_DIR / f"{timestamp}.txt"
    summary_path.write_text(summary, encoding="utf-8")

    return jsonify(
        {
            "transcript": transcript,
            "summary": summary,
            "audio_file": str(audio_path.name),
            "timestamp": timestamp,
            "stt_engine": stt_engine,
            "stt_error": stt_error,
            "summary_type": summary_type,
        }
    )


@app.route("/api/summarize", methods=["POST"])
def summarize_text():
    """Re-summarize text (e.g. after user edits the transcript)."""
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "文本为空"}), 400

    mode = data.get("mode", "local")
    summary_type = data.get("summary_type", "会议总结")
    try:
        if mode in ("online", "hybrid"):
            api_key = data.get("api_key_summary", "")
            api_base = data.get("api_base_summary", "")
            summary_model = data.get("summary_model", "deepseek-chat")
            if api_key:
                summary = summarize_online(
                    text, api_key, api_base, summary_model, summary_type
                )
            else:
                summary = summarize_local(text, summary_type)
        else:
            summary = summarize_local(text, summary_type)
    except Exception as e:
        return jsonify({"error": f"总结生成失败: {e}"}), 500

    return jsonify({"summary": summary, "summary_type": summary_type})


@app.route("/api/save-summary", methods=["POST"])
def save_summary():
    """Save edited summary text back to file."""
    data = request.get_json()
    timestamp = (data.get("timestamp") or "").strip()
    summary_text = data.get("summary") or ""
    if not timestamp:
        return jsonify({"error": "缺少时间戳"}), 400

    summary_path = SUMMARIES_DIR / f"{timestamp}.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    return jsonify({"success": True})


@app.route("/api/test-connection", methods=["POST"])
def test_connection():
    """Test if the API key and base URL can connect successfully."""
    data = request.get_json()
    api_key = data.get("api_key", "")
    api_base = data.get("api_base", "")
    test_type = data.get("type", "chat")
    model = data.get("model", "")

    if not api_key:
        return jsonify({"success": False, "message": "请先输入 API Key"})

    try:
        if test_type == "stt" and _is_dashscope(api_base):
            # Test DashScope native realtime ASR
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
                if result.status_code == HTTPStatus.OK:
                    msg = "连接成功 (阿里云 ASR 可用)"
                else:
                    return jsonify(
                        {"success": False, "message": f"ASR 失败: {result.message}"}
                    )
            finally:
                os.unlink(tmp_path)

        elif test_type == "stt":
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=api_base or None)
            test_audio = _make_minimal_wav()
            client.audio.transcriptions.create(
                model=model or "whisper-1",
                file=("test.wav", test_audio, "audio/wav"),
                response_format="text",
            )
            msg = "连接成功 (STT API 可用)"
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=api_base or None)
            # Minimal chat completion to verify
            client.chat.completions.create(
                model=model or "deepseek-chat",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            msg = (
                f"连接成功 (模型 {model} 可用)" if model else "连接成功 (Chat API 可用)"
            )

        return jsonify({"success": True, "message": msg})
    except Exception as e:
        err = str(e)
        # Trim overly long error messages
        if len(err) > 300:
            err = err[:300] + "..."
        return jsonify({"success": False, "message": err})


@app.route("/api/history")
def history():
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
    return jsonify(items)


@app.route("/api/history/<timestamp>", methods=["DELETE"])
def delete_history_item(timestamp):
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
        return jsonify({"success": True})
    return jsonify({"error": "记录不存在"}), 404


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Delete all recordings, transcripts, and summaries."""
    count = 0
    for dir_path in [RECORDINGS_DIR, TRANSCRIPTS_DIR, SUMMARIES_DIR]:
        for f in dir_path.iterdir():
            f.unlink()
            count += 1
    return jsonify({"success": True, "deleted": count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
