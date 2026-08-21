# Voice Record Summary — 语音录制与总结工具

浏览器端录音 / 流式识别 / 上传音频文件 → 语音识别 → AI 大模型自动总结要点，一站式工具。

> **当前版本：v1.6.0** | [English](README_EN.md)

## 功能

- **录音识别**：点击麦克风按钮直接录制，结束后自动转写
- **流式识别**：麦克风一边说话一边实时转写，再次点击结束，结束后自动总结
- **文件上传**：支持拖放或选择本地音频文件（WAV / WebM / MP3 / M4A / OGG / FLAC / MP4 等）
- **三种工作模式**：本地识别、在线识别、混合模式
- **本地语音识别**：FunASR 引擎（SenseVoice 离线识别、Paraformer 流式识别）
- **在线语音识别**：OpenAI 兼容 `/audio/transcriptions` 端点、阿里云百炼 DashScope 原生 ASR（fun-asr-realtime，支持流式 WebSocket）
- **多种总结模板**：会议总结、今日计划、学习笔记、快速摘要、待办事项，支持自定义总结方式
- **语境提示**：可填写当前语境的背景说明（会议主题、行业术语、专有名词等），AI 总结与实时纠错时结合该语境更准确地理解内容
- **AI 文本总结**：通过在线大模型（DeepSeek V4 / Qwen / GLM / Claude 等，任意 OpenAI 兼容端点）或本地 TF-IDF 算法归纳整理，输出中英双语结构化纪要
- **语音纠错**：AI 自动修正同音错字、补充标点分段（流式识别支持实时纠错）
- **流式实时纠错**：混合/在线模式下，大模型可用时会在说话停顿和结束时结合语境自动纠正识别文本
- **服务商预设**：一键切换 DeepSeek / 阿里云 / 自定义，自动填充 API 地址和推荐模型
- **连接测试**：可单独测试 STT 和 Chat API 的连通性
- **手动编辑**：识别文本和总结均可直接编辑，修改后可重新生成总结
- **Markdown 预览与下载**：总结支持编辑/预览切换，可下载为 `.md` 文件
- **设置记忆**：所有配置自动保存到浏览器 localStorage，刷新不丢失
- **多主题界面**：Catppuccin 全套 12 主题——Mocha / Macchiato / Frappé / Latte 四种风味 × Blue / Mauve / Peach 三种强调色，严格采用官方调色板，一键切换自动记忆
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
| POST | `/api/transcribe` | 上传音频并执行识别 + 总结（支持 `context` 语境提示） |
| POST | `/api/summarize` | 对文本重新生成总结（支持 `context` 语境提示） |
| POST | `/api/correct` | AI 纠错/润色原始识别文本（支持 `context` 前文与 `context_prompt` 语境提示） |
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

### v1.6.0 (2026-08)
- 主题体系重构为 Catppuccin 全套：删除深空蓝/月光白/清新绿/天湖蓝/樱花粉，保留原 Catppuccin Mocha（Blue）
- 新增 Latte、Frappé、Macchiato、Mocha 四种风味，每种提供 Blue / Mauve / Peach 三种强调色，共 12 套主题
- 全部颜色严格取自 catppuccin/palette 官方调色板（base、surface、overlay、text 及各强调色）
- 主题选择器按风味分组（Mocha / Macchiato / Frappé / Latte），每组 3 个颜色圆点

### v1.5.6 (2026-08)
- 新增 Catppuccin Mocha 主题，严格采用官方调色板（base #1e1e2e、blue #89b4fa、mauve #cba6f7、green #a6e3a1 等，源自 catppuccin/palette）
- 主题圆点使用 Catppuccin 标志性的蓝/绿/桃/紫四色环

### v1.5.5 (2026-08)
- 删除星夜紫、极光绿、落日橙 3 套暗色主题，保留深空蓝 + 4 套浅色，共 5 套（已保存被删主题的用户自动回退深空蓝）
- 页面最大宽度从 1120px 提升至 1560px，宽屏下两侧留白大幅减少，双栏比例调整为 1.15:1
- 主题选择器增加暗色/浅色分隔线

### v1.5.4 (2026-08)
- 暗色主题降低饱和度：深空蓝/星夜紫/极光绿/落日橙改为低饱和的钢蓝、灰紫、灰青、黄铜色调，告别霓虹感
- 发光强度整体下调（光晕、背景光斑、焦点环），边框改为中性偏色调的柔和色，Markdown 高亮改为暗金
- 标题渐变动画放慢至 14s，视觉更沉稳；浅色主题保持不变

### v1.5.3 (2026-08)
- 页面宽度从 720px 提升至 1120px，宽屏下采用双栏布局（左侧录制/配置，右侧结果/历史），窄屏自动回到单栏
- 主题重构为 8 套：暗色 4 套（深空蓝、星夜紫、极光绿、落日橙）+ 浅色 4 套（月光白、清新绿、天湖蓝、樱花粉）
- 整体配色提亮：暗色主题卡片背景从近黑改为深蓝/紫/绿/棕色半透明，文字提亮，发光效果增强；背景光晕更丰富
- 新增清新绿、天湖蓝、樱花粉三套浅色主题，白底彩边、柔和阴影，视觉清爽

### v1.5.2 (2026-08)
- 界面质感升级：卡片悬浮抬升与光晕、标题渐变流光动效、录音中旋转光环、按钮悬浮背景光、焦点可见环、文本选区配色
- 新增多主题系统：深空青（默认）、星夜紫、翡翠绿、落日橙、赛博玫红、月光白（浅色）6 套主题，页面顶部圆点一键切换，选择自动保存到 localStorage
- 全组件颜色改为 CSS 变量驱动，下拉箭头、滚动条、Markdown 渲染配色均随主题适配

### v1.5.1 (2026-08)
- 修复流式识别结束后无法立即开始下一段录音的问题：停止后按钮立刻恢复可用，不再等待收尾完成
- 后端将耗时的 2-pass 离线精修移到 `completed` 消息之后，精修结果通过新增的 `refined` 跟进消息推送（不影响历史保存）；UI 释放不再被精修阻塞
- 修复快速连续识别时的竞态：为每次流式会话分配递增 sessionId，旧会话的迟到消息/连接关闭事件不会干扰新会话；纠错中标记改为按会话隔离
- 后端为 FunASR 流式模型增加每模型锁，避免旧会话收尾与新会话分块并发调用 `generate` 导致的线程安全问题
- 新会话开始后旧会话的总结仍会在后台完成并保存到服务器历史记录，但不干扰当前界面

### v1.5 (2026-08)
- 新增「语境提示」区域：用户可填写当前语境的背景说明（主题、行业术语、专有名词、参会人等），AI 总结与纠错时将其作为理解辅助注入提示词，且明确要求语境不写入总结内容
- 流式识别增强：混合/在线模式下自动探测大模型可用性（结果缓存 5 分钟），可用后在说话停顿（1.5s）及识别结束时结合语境实时纠错；纠错失败自动暂停，不再重复请求
- 结束纠错优化：若停顿期间已纠错过，结束时仅纠错新增的原始片段并拼接，避免重写全文
- 上下文提示词贯通 `/api/transcribe`、`/api/summarize`、`/api/correct` 三个端点

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
