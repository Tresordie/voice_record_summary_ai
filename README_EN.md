# Voice Record Summary — Speech Recording & Summarization Tool

A one-stop tool: record / stream / upload audio in the browser → speech recognition → AI-powered structured summary.

> **Current version: v1.4** | [中文](README.md)

## Features

- **Recording recognition**: Click the microphone to record; transcription runs after you stop
- **Streaming recognition**: Real-time transcription while speaking; click again to stop, then auto-summarize
- **File upload**: Drag-and-drop or select local audio files (WAV / WebM / MP3 / M4A / OGG / FLAC / MP4, etc.)
- **Three working modes**: Local, Online, Hybrid
- **Local speech recognition**: FunASR engine (SenseVoice for offline, Paraformer for streaming)
- **Online speech recognition**: OpenAI-compatible `/audio/transcriptions` endpoint, and Alibaba Cloud DashScope native ASR (fun-asr-realtime, with streaming WebSocket)
- **Multiple summary templates**: Meeting Summary, Today's Plan, Study Notes, Quick Summary, Action Items, plus custom summary types
- **AI summarization**: Structured bilingual (Chinese + English) summaries via online LLMs (DeepSeek V4 / Qwen / GLM / Claude, etc., any OpenAI-compatible endpoint) or the local TF-IDF algorithm
- **ASR correction**: AI fixes homophone errors and adds punctuation (streaming mode supports real-time correction)
- **Provider presets**: One-click switching between DeepSeek / Alibaba Cloud / Custom, with auto-filled API URLs and recommended models
- **Connection testing**: Independently test STT and Chat API connectivity
- **Manual editing**: Both transcript and summary are editable; re-summarize after modifying the transcript
- **Markdown preview & download**: Toggle edit/preview, and download the summary as a `.md` file
- **Settings persistence**: All settings auto-saved to browser localStorage, survives page refresh
- **History**: Automatically saves recordings, transcripts, and summaries; supports viewing and single/clear-all deletion
- **Audio playback**: Play back recordings or uploaded files directly in the page

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Local speech recognition (FunASR) requires PyTorch. Install the version matching your environment: `pip install torch`
>
> Alibaba Cloud ASR / some audio format conversions require ffmpeg: [ffmpeg.org](https://ffmpeg.org/)

### 2. Start the Server

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## Recognition Modes

| Mode | Description |
|------|-------------|
| **Recording** | Speak, then click stop; transcription runs once |
| **Streaming** | Real-time transcription while speaking, displayed live; auto-summarize when stopped |

## Working Modes

| Mode | Speech Recognition | Summarization |
|------|-------------------|---------------|
| **Local** | Local FunASR engine | Local TF-IDF algorithm |
| **Hybrid** | Local FunASR engine | Online LLM |
| **Online** | Online API | Online LLM |

### Local Mode

Enter a ModelScope model ID or a local folder path in the "Model Path" field:

- **Offline**: `iic/SenseVoiceSmall` (multilingual, recommended default) or another FunASR model
- **Streaming**: `paraformer-zh-streaming` or another Paraformer streaming model

Engine type is auto-detected from the model name and folder contents.

### Online Mode

Click a provider preset to quickly fill in settings, then enter your API Key:

| Preset | Speech Recognition | Summarization |
|--------|-------------------|---------------|
| DeepSeek | OpenAI-compatible transcription endpoint | deepseek-v4-flash / deepseek-v4-pro |
| Alibaba Cloud | fun-asr-realtime (native WebSocket streaming ASR) | qwen-plus |
| Custom | Any OpenAI-compatible endpoint | Any OpenAI-compatible endpoint (e.g. glm-4-flash, claude-sonnet-4-7) |

Use the "Test Connection" button to verify your STT or Chat API Key and endpoint.

## Summary Templates

| Template | Description |
|----------|-------------|
| Meeting Summary | Extracts meeting topic, key discussions, decisions, action items — bilingual |
| Today's Plan | Extracts today's tasks, upcoming items, and notes, sorted by priority |
| Study Notes | Extracts core concepts, key points, and topics for further study |
| Quick Summary | 3-5 sentence bilingual summary capturing the core content |
| Action Items | Precisely extracts all tasks with owners and timelines |
| Custom | Enter any summary type name; the AI adapts the format automatically |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Main page |
| POST | `/api/transcribe` | Upload audio, run recognition + summarization |
| POST | `/api/summarize` | Re-generate summary from text |
| POST | `/api/correct` | Correct/polish raw ASR text |
| POST | `/api/save-summary` | Save edited summary content |
| POST | `/api/test-connection` | Test API connection (supports STT and Chat) |
| GET | `/api/history` | Get history list |
| GET | `/api/download-summary/<timestamp>` | Download a record's Markdown summary |
| GET | `/api/local-models` | List locally cached FunASR models |
| POST | `/api/preload-streaming-model` | Preload a streaming recognition model |
| DELETE | `/api/history` | Clear all history |
| DELETE | `/api/history/<timestamp>` | Delete a single record |
| WS | `/ws/stream` | Local streaming recognition WebSocket |
| WS | `/ws/stream-online` | Alibaba Cloud online streaming recognition WebSocket |

## Changelog

### v1.4 (2026-08)
- Migrated backend from Flask to FastAPI + Uvicorn, added HTTPS (self-signed) support
- Migrated local recognition engine from Whisper to FunASR (SenseVoice / Paraformer)
- Added streaming recognition (recording / streaming modes) with local and online (Alibaba Cloud WebSocket) real-time transcription
- Added `/api/correct`, `/api/download-summary`, `/api/local-models`, `/api/preload-streaming-model` endpoints
- Updated provider presets: defaulting to DeepSeek V4 (deepseek-v4-flash / deepseek-v4-pro), removed the OpenAI preset
- Fixed DeepSeek V4 summarization returning empty results: requests now explicitly disable thinking mode (`thinking: {"type": "disabled"}`) and fall back to `reasoning_content` when `content` is empty

### v1.3 (2026-06-19)
- Rewrote AI summarization prompts with three mandatory principles: faithfulness (no fabrication), ASR error correction, and topic-first reasoning
- Optimized prompt structure to step-by-step execution (understand topic → extract key points → structured output), improving summary accuracy
- Increased max_tokens from 3000 to 4096 to prevent bilingual output truncation for long transcripts
- Lowered temperature from 0.3 to 0.1 for more consistent, faithful summaries with less hallucination
- Fixed API key fallback bug: STT key no longer incorrectly used as Summary key
- Fixed custom summary type string formatting crash when summary type contains special characters
- Changed default summarization model from gpt-3.5-turbo to deepseek-chat
- Result card header now dynamically displays the current summary type
- Removed duplicate app.run() code

### v1.2 (2026-06-14)
- Added multiple summary templates: Today's Plan, Study Notes, Quick Summary, Action Items, plus custom summary types
- Added online LLM summarization with structured bilingual (Chinese + English) output
- Added provider presets: one-click switching between OpenAI / DeepSeek / Alibaba Cloud / Custom
- Added API connection testing for STT and Chat endpoints independently
- Added Alibaba Cloud DashScope native ASR integration (fun-asr-realtime)
- Added auto engine detection for Whisper and SenseVoice model paths
- Added drag-and-drop audio file upload
- Added automatic audio format conversion (ffmpeg) for expanded DashScope compatibility
- Added browser localStorage settings persistence
- Transcript textarea is now editable, supporting re-summarization after modifications
- Audio playback available directly in-page after recording or upload

### v1.1 (2026-06-08)
- Online summarization now outputs bilingual results (Chinese + English Summary)
- Summary area changed to editable textarea for manual editing
- Added preview/edit toggle to view Markdown rendered output
- Added save button to persist edited summaries to the server
- Added `POST /api/save-summary` endpoint

### v1.0
- Initial release: browser recording, file upload, speech recognition, AI summarization, history

## Project Structure

```
voice_record_summary/
├── app.py               # FastAPI backend
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Frontend (single-file SPA)
├── recordings/          # Audio file storage (auto-created)
├── transcripts/         # Transcript text storage (auto-created)
└── summaries/           # Summary text storage (auto-created)
```

## System Requirements

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/) (required for Alibaba Cloud ASR — converts browser-recorded WebM to WAV)
- PyTorch (required for local FunASR recognition)
