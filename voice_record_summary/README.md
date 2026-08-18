# Voice Record Summary — 语音录制与总结工具

浏览器端录音 / 流式识别 / 上传音频文件 → 语音识别 → AI 大模型自动总结要点，一站式工具。

> **当前版本：v1.4** | [English](README_EN.md)

## 功能

- **录音识别**：点击麦克风按钮直接录制，结束后自动转写
- **流式识别**：麦克风一边说话一边实时转写，再次点击结束，结束后自动总结
- **文件上传**：支持拖放或选择本地音频文件（WAV / WebM / MP3 / M4A / OGG / FLAC / MP4 等）
- **三种工作模式**：本地识别、在线识别、混合模式
- **本地语音识别**：FunASR 引擎（SenseVoice 离线识别、Paraformer 流式识别）
- **在线语音识别**：OpenAI 兼容 `/audio/transcriptions` 端点、阿里云百炼 DashScope 原生 ASR（fun-asr-realtime，支持流式 WebSocket）
- **多种总结模板**：会议总结、今日计划、学习笔记、快速摘要、待办事项，支持自定义总结方式
- **AI 文本总结**：通过在线大模型（DeepSeek V4 / Qwen / GLM / Claude 等，任意 OpenAI 兼容端点）或本地 TF-IDF 算法归纳整理，输出中英双语结构化纪要
- **语音纠错**：AI 自动修正同音错字、补充标点分段（流式识别支持实时纠错）
- **服务商预设**：一键切换 DeepSeek / 阿里云 / 自定义，自动填充 API 地址和推荐模型
- **连接测试**：可单独测试 STT 和 Chat API 的连通性
- **手动编辑**：识别文本和总结均可直接编辑，修改后可重新生成总结
- **Markdown 预览与下载**：总结支持编辑/预览切换，可下载为 `.md` 文件
- **设置记忆**：所有配置自动保存到浏览器 localStorage，刷新不丢失
- **历史记录**：自动保存每次的录音、识别文本和总结，支持查看和单条/全部删除
- **音频回放**：录音或上传后可直接在页面中播放

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 本地语音识别（FunASR）需要 PyTorch，请根据你的环境安装对应版本：`pip install torch`
>
> 阿里云 ASR / 部分音频格式转换需要 ffmpeg：[ffmpeg.org](https://ffmpeg.org/)

### 2. 启动服务

```bash
python app.py
```

启动后终端会打印可直接点击的访问链接，点击即可进入应用：

```
  本机访问:  http://127.0.0.1:5000
  局域网访问: http://localhost:5000
  加密访问:  https://127.0.0.1:5443  (自签名证书)
```

也可手动打开浏览器访问 `http://localhost:5000`

## 识别方式

| 识别方式 | 说明 |
|----------|------|
| **录音识别** | 说完一段话后点击停止，再统一转写 |
| **流式识别** | 边说话边实时转写，识别内容实时显示在文本框，结束后自动总结 |

## 工作模式

| 模式 | 语音识别 | 文本总结 |
|------|----------|----------|
| **本地** | 本地 FunASR 引擎 | 本地 TF-IDF 算法 |
| **混合** | 本地 FunASR 引擎 | 在线大模型 |
| **在线** | 在线 API | 在线大模型 |

### 本地模式

在「模型路径」中填入 ModelScope 模型 ID 或本地文件夹路径：

- **离线识别**：`iic/SenseVoiceSmall`（多语言，推荐默认值）或其他 FunASR 模型
- **流式识别**：`paraformer-zh-streaming` 或其他 Paraformer 流式模型

引擎类型会根据模型名称和文件夹内容自动检测。

### 在线模式

点击预设按钮快速切换服务商，填入 API Key 即可使用：

| 预设 | 语音识别 | 文本总结 |
|------|----------|----------|
| DeepSeek | OpenAI 兼容转录端点 | deepseek-v4-flash / deepseek-v4-pro |
| 阿里云 | fun-asr-realtime（原生 WebSocket 流式 ASR） | qwen-plus |
| 自定义 | 任意 OpenAI 兼容端点 | 任意 OpenAI 兼容端点（如 glm-4-flash、claude-sonnet-4-7 等） |

点击「测试连接」按钮可以验证 STT 或 Chat API Key 和端点是否可用。

## 总结模板

| 模板 | 说明 |
|------|------|
| 会议总结 | 提取会议主题、关键讨论、决策事项、待办事项，中英双语 |
| 今日计划 | 提取今日任务、近期安排、备注提醒，按优先级排序 |
| 学习笔记 | 提取核心概念、要点归纳、待深入学习的内容 |
| 快速摘要 | 3-5 句话概括全文核心内容，中英双语 |
| 待办事项 | 精确提取所有任务，含负责人和时间节点 |
| 自定义 | 输入任意总结方式名称，AI 自动适配格式 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| POST | `/api/transcribe` | 上传音频并执行识别 + 总结 |
| POST | `/api/summarize` | 对文本重新生成总结 |
| POST | `/api/correct` | AI 纠错/润色原始识别文本 |
| POST | `/api/save-summary` | 保存编辑后的总结内容 |
| POST | `/api/test-connection` | 测试 API 连接（支持 STT 和 Chat） |
| GET | `/api/history` | 获取历史记录列表 |
| GET | `/api/download-summary/<timestamp>` | 下载单条记录的 Markdown 总结 |
| GET | `/api/local-models` | 获取本地已缓存的 FunASR 模型列表 |
| POST | `/api/preload-streaming-model` | 预加载流式识别模型 |
| DELETE | `/api/history` | 清空全部历史 |
| DELETE | `/api/history/<timestamp>` | 删除单条记录 |
| WS | `/ws/stream` | 本地流式识别 WebSocket |
| WS | `/ws/stream-online` | 阿里云在线流式识别 WebSocket |

## 更新日志

### v1.4 (2026-08)
- 后端从 Flask 迁移到 FastAPI + Uvicorn，新增 HTTPS（自签名）支持
- 本地识别引擎从 Whisper 迁移到 FunASR（SenseVoice / Paraformer）
- 新增流式识别（录音识别 / 流式识别两种模式），支持本地与在线（阿里云 WebSocket）流式转写
- 新增 `/api/correct` 语音纠错、`/api/download-summary` Markdown 下载、`/api/local-models`、`/api/preload-streaming-model` 等端点
- 服务商预设更新：默认使用 DeepSeek V4（deepseek-v4-flash / deepseek-v4-pro），移除 OpenAI 预设
- 修复 DeepSeek V4 无法总结的问题：请求时显式关闭思考模式（`thinking: {"type": "disabled"}`），并在 `content` 为空时回退 `reasoning_content`

### v1.3 (2026-06-19)
- 重写 AI 总结提示词，增加三条强制原则：忠实原文不编造、语音识别纠错、先理解后总结
- 优化提示词结构，改为分步骤执行（概括主题→提取要点→结构化输出），提升总结准确度
- max_tokens 从 3000 提升至 4096，避免长文本双语输出被截断
- temperature 从 0.3 降至 0.1，减少模型编造内容的可能性
- 修复 API Key 回退缺陷：STT Key 不再被错误用作 Summary Key
- 修复自定义总结方式字符串格式化在特殊字符下崩溃的问题
- 默认总结模型从 gpt-3.5-turbo 改为 deepseek-chat
- 结果卡片标题动态显示当前使用的总结方式
- 删除重复的 app.run() 冗余代码

### v1.2 (2026-06-14)
- 新增多种总结模板：今日计划、学习笔记、快速摘要、待办事项，支持自定义总结方式
- 新增大模型在线总结，支持中英双语结构化工整输出
- 新增服务商预设：一键切换 OpenAI / DeepSeek / 阿里云 / 自定义
- 新增 API 连接测试功能，可分别测试 STT 和 Chat 端点
- 新增阿里云百炼 DashScope 原生 ASR 集成（fun-asr-realtime）
- 新增引擎自动检测：Whisper 和 SenseVoice 模型路径自动识别
- 新增拖放上传音频文件
- 新增音频格式自动转换（ffmpeg），扩展 DashScope 兼容性
- 新增浏览器 localStorage 设置自动记忆
- 识别文本区域改为可编辑，支持修改后重新总结
- 音频录制/上传后支持页面内直接回放

### v1.1 (2026-06-08)
- 在线总结支持中英双语输出（中文摘要 + English Summary）
- 总结区域改为可编辑文本框，支持手动修改内容
- 新增预览/编辑切换，可查看 Markdown 渲染效果
- 新增保存按钮，编辑后的总结可持久化到服务器
- 新增 `POST /api/save-summary` 端点

### v1.0
- 初始版本：浏览器录音、文件上传、语音识别、AI 总结、历史记录

## 项目结构

```
voice_record_summary/
├── app.py               # FastAPI 后端主程序
├── requirements.txt     # Python 依赖
├── templates/
│   └── index.html       # 前端页面（单文件）
├── recordings/          # 录音文件存储（自动创建）
├── transcripts/         # 识别文本存储（自动创建）
└── summaries/           # 总结文本存储（自动创建）
```

## 系统要求

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/)（阿里云 ASR 模式下，浏览器录制的 WebM 格式需转码为 WAV）
- 本地 FunASR 识别需要 PyTorch
