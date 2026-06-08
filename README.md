# Voice Record Summary — 语音录制与总结工具

浏览器端录音或上传音频文件 → 语音识别 → AI 大模型自动总结要点，一站式工具。

> **当前版本：v1.1**

## 功能

- **浏览器录音**：点击麦克风按钮直接在浏览器中录音
- **文件上传**：支持拖放或选择本地音频文件（WAV / WebM / MP3 / M4A / OGG / FLAC 等）
- **三种工作模式**：本地识别、在线识别、混合模式
- **多引擎语音识别**：
  - 本地：OpenAI Whisper、FunASR SenseVoice（自动检测引擎类型）
  - 在线：OpenAI 兼容 API、阿里云百炼 DashScope 原生 ASR
- **AI 文本总结**：通过在线大模型（GPT / DeepSeek / Qwen 等）对识别结果归纳整理，输出中英双语结构化纪要
- **手动编辑总结**：总结内容可在浏览器中直接修改，支持预览/编辑切换，修改后可保存回服务器
- **重新总结**：修改识别文本后，可点击按钮重新生成总结
- **历史记录**：自动保存每次的录音、识别文本和总结，支持查看和单条/全部删除

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 本地语音识别需要 PyTorch，请根据你的环境安装对应版本：`pip install torch`

### 2. 启动服务

```bash
python app.py
```

打开浏览器访问 `http://localhost:5000`

## 工作模式

| 模式 | 语音识别 | 文本总结 |
|------|----------|----------|
| **本地** | 本地引擎（Whisper / SenseVoice） | 本地分句 |
| **混合** | 本地引擎 | 在线大模型 |
| **在线** | 在线 API | 在线大模型 |

### 本地模式

在「模型路径」中填入引擎名称或路径：

- **Whisper**：`tiny` / `base` / `small` / `medium` / `large-v3` / `turbo`
- **FunASR SenseVoice**：`iic/SenseVoiceSmall`（ModelScope 模型 ID）或本地文件夹路径

引擎类型会根据模型名称和文件夹内容自动检测。

### 在线模式

点击预设按钮快速切换服务商，填入 API Key 即可使用：

| 预设 | 语音识别 | 文本总结 |
|------|----------|----------|
| OpenAI | whisper-1 / gpt-4o-transcribe | gpt-4o-mini |
| DeepSeek | 不支持（自动回退本地） | deepseek-chat |
| 阿里云 | fun-asr-realtime（原生 WebSocket API） | qwen-plus |
| 自定义 | 任意 OpenAI 兼容端点 | 任意 OpenAI 兼容端点 |

点击「测试连接」按钮可以验证 API Key 和端点是否可用。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| POST | `/api/transcribe` | 上传音频并执行识别+总结 |
| POST | `/api/summarize` | 对文本重新生成总结 |
| POST | `/api/save-summary` | 保存编辑后的总结内容 |
| POST | `/api/test-connection` | 测试 API 连接 |
| GET | `/api/history` | 获取历史记录列表 |
| DELETE | `/api/history` | 清空全部历史 |
| DELETE | `/api/history/<timestamp>` | 删除单条记录 |

## 更新日志

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
├── app.py               # Flask 后端主程序
├── requirements.txt     # Python 依赖
├── templates/
│   └── index.html       # 前端页面（单文件）
├── recordings/          # 录音文件存储（自动创建）
├── transcripts/         # 识别文本存储（自动创建）
└── summaries/           # 总结文本存储（自动创建）
```

## 系统要求

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/)（在线阿里云 ASR 模式下，浏览器录制的 WebM 格式需转码为 WAV）
- 本地 Whisper 使用需要 PyTorch
