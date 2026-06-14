# Voice Record Summary — Speech Recording & Summarization Tool

A one-stop tool: record or upload audio in the browser → speech recognition → AI-powered structured summary.

> **Current version: v1.2** | [中文](README.md)

## Features

- **Browser recording**: Click the microphone button to record directly in the browser
- **File upload**: Drag-and-drop or select local audio files (WAV / WebM / MP3 / M4A / OGG / FLAC / MP4, etc.)
- **Three working modes**: Local, Online, Hybrid
- **Multi-engine speech recognition**:
  - Local: OpenAI Whisper (tiny~large-v3/turbo), FunASR SenseVoice (auto-detected engine type)
  - Online: OpenAI-compatible API, Alibaba Cloud DashScope native ASR (fun-asr-realtime)
- **Multiple summary templates**: Meeting Summary, Today's Plan, Study Notes, Quick Summary, Action Items, plus custom summary types
- **AI summarization**: Structured bilingual (Chinese + English) summaries via online LLMs (GPT / DeepSeek / Qwen, etc.) or local TF-IDF algorithm
- **Provider presets**: One-click switching between OpenAI / DeepSeek / Alibaba Cloud / Custom, with auto-filled API URLs and recommended models
- **Connection testing**: Independently test STT and Chat API connectivity
- **Manual editing**: Both transcript and summary are editable; re-summarize after modifying the transcript
- **Markdown preview**: Toggle between edit and preview modes for the summary
- **Settings persistence**: All settings auto-saved to browser localStorage, survives page refresh
- **History**: Automatically saves recordings, transcripts, and summaries; supports viewing and single/clear-all deletion
- **Audio playback**: Play back recordings or uploaded files directly in the page

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Local speech recognition requires PyTorch. Install the version matching your environment: `pip install torch`
>
> Alibaba Cloud ASR mode requires ffmpeg for audio format conversion: [ffmpeg.org](https://ffmpeg.org/)

### 2. Start the Server

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## Working Modes

| Mode | Speech Recognition | Summarization |
|------|-------------------|---------------|
| **Local** | Local engine (Whisper / SenseVoice) | Local TF-IDF algorithm |
| **Hybrid** | Local engine | Online LLM |
| **Online** | Online API | Online LLM |

### Local Mode

Enter the engine name or path in the "Model Path" field:

- **Whisper**: `tiny` / `base` / `small` / `medium` / `large-v3` / `turbo`
- **FunASR SenseVoice**: `iic/SenseVoiceSmall` (ModelScope model ID) or a local folder path

Engine type is auto-detected from the model name and folder contents — no manual selection needed.

### Online Mode

Click a provider preset to quickly fill in settings, then enter your API Key:

| Preset | Speech Recognition | Summarization |
|--------|-------------------|---------------|
| OpenAI | whisper-1 / gpt-4o-transcribe | gpt-4o-mini |
| DeepSeek | Not supported (falls back to local) | deepseek-chat |
| Alibaba Cloud | fun-asr-realtime (native WebSocket API) | qwen-plus |
| Custom | Any OpenAI-compatible endpoint | Any OpenAI-compatible endpoint |

Use the "Test Connection" button to verify your API Key and endpoint.

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
| POST | `/api/save-summary` | Save edited summary content |
| POST | `/api/test-connection` | Test API connection (supports STT and Chat) |
| GET | `/api/history` | Get history list |
| DELETE | `/api/history` | Clear all history |
| DELETE | `/api/history/<timestamp>` | Delete a single record |

## Changelog

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
├── app.py               # Flask backend
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
- PyTorch (required for local Whisper)
