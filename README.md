<div align="center">

# 👁️ TRUSIGHT
### Multimodal Forensic Intelligence & Credibility Assessment Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75C2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://aistudio.google.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0075FF?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue?style=for-the-badge)](https://github.com)

<p align="center">
  <strong>An integrated investigative intelligence platform that unifies real-time facial micro-cue extraction, vocal acoustic deception telemetry, and AI-driven textual evidence & statement contradiction analysis inside a single high-performance dashboard.</strong>
</p>

[Key Features](#-core-capabilities) • [System Architecture](#-system-architecture) • [Text & Evidence Feature](#-text--statement-contradiction-feature-gemini-rag) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Directory Layout](#-directory-structure)

---

</div>

## 📌 Executive Summary

**TRUSIGHT** bridges the gap between physiological signals, vocal acoustics, and textual evidence consistency. Rather than relying on a single fallible indicator, investigators and researchers can examine synchronized tri-modal evidence:

1. **Facial Micro-Cue Telemetry**: Real-time facial landmarking (478 points), Action Units (FACS), 3D head pose orientation, and micro-expression sentiment analysis.
2. **Vocal Acoustic Deception Analysis**: Physical glottal measurements (fundamental frequency F0, jitter, shimmer), articulation pacing, speech pauses, and ensemble machine learning predictions.
3. **Text & Statement Contradiction Verification (Gemini RAG Engine)**: Ingestion of case files (FIRs, witness statements, depositions in PDF format), live verbatim speech transcription, and automated RAG-based semantic contradiction analysis against recorded facts.

All streams are unified on a **single high-performance tactical dashboard (`http://localhost:8000`)** powered by FastAPI reverse-proxying, WebSockets, and zero external GUI dependencies.

---

## ⚡ Core Capabilities

### 1. 👁️ Face Analysis Engine (`:8002`)
* **478-Landmark Mesh**: MediaPipe FaceLandmarker tracking sub-millimeter facial mesh geometry.
* **FACS Action Units (AUs)**: Live extraction of Action Units including AU01 (Inner Brow), AU02 (Outer Brow), AU04 (Brow Lowerer), AU12 (Lip Corner Puller), AU15 (Lip Corner Depressor), AU25/AU26 (Lips Part/Jaw Drop), and AU45 (Blink rate).
* **3D Head Pose Estimation**: Perspective-n-Point (PnP) solver yielding Pitch, Yaw, and Roll orientation angles.
* **Sentiment & Emotion Classifier**: EfficientNet-B0 trained on AffectNet (8 emotion classes).
* **Facial Deception Neural Network**: 52-cue truth/lie probability classifier.
* **Direct Hardware Control**: Start, stop, and snapshot the webcam feed directly from the browser HUD.

### 2. 🎙️ Voice Analysis Engine (`:8001`)
* **Glottal Pitch Variations (F0)**: Fundamental frequency mean, standard deviation, and intonation contour via Praat/Parselmouth.
* **Vocal Stability Metrics**: Local Jitter (% cycle-to-cycle frequency variation) and Local Shimmer (% amplitude perturbation).
* **Tempo & Phrasing Pacing**: Syllable onset rate, pauses count, hesitation duration, and silent interval ratio via Librosa.
* **Acoustic Deception Model**: 200-tree Random Forest classifier yielding calibrated Deception Scores.
* **Dual Audio Ingestion**:
  * **Live Microphone**: In-browser MediaRecorder (`15s`, `20s`, `30s` sampling windows) with Web Audio API waveform visualizer.
  * **File Upload**: Drag-and-drop support for WAV, MP3, FLAC, OGG, M4A, and WEBM.

### 3. 📄 Text & Statement Contradiction Engine (`:8003`)
* **Case Evidence PDF Ingestion**: High-speed local text extraction with PyMuPDF, preserving layout and structural context without external OCR dependencies.
* **Semantic Vector Retrieval (RAG)**: Chunking and local dense embedding retrieval using `gemini-embedding-001` with cosine similarity ranking to isolate relevant evidentiary facts.
* **Real-time Live Audio Transcription**: Audio stream forwarded via WebSocket to `gemini-3.5-transcribe-live` for interim and finalized speech-to-text.
* **Contradiction & Credibility Assessment**: `gemini-3.8-flash` cross-references candidate utterances against extracted case facts to highlight:
  * Contradiction probability and risk score.
  * Verbatim excerpt matches from official case files.
  * Chain-of-thought explanation of factual discrepancies.

---

## 🏛️ System Architecture

```
                                  ┌──────────────────────────────────────────────────┐
                                  │           BROWSER TACTICAL DASHBOARD             │
                                  │             http://localhost:8000                │
                                  └────────────────────────┬─────────────────────────┘
                                                           │
                                                           ▼
                                  ┌──────────────────────────────────────────────────┐
                                  │          TRUSIGHT GATEWAY (FastAPI :8000)        │
                                  │            trusight_server.py / static/          │
                                  └───────────┬────────────────┬─────────────────┬───┘
                                              │                │                 │
                Live MJPEG & REST Telemetry   │                │ WebSockets & REST│ Binary PCM & Case Upload
                                              ▼                ▼                 ▼
             ┌──────────────────────────────────┐  ┌────────────────────────┐  ┌─────────────────────────────────┐
             │       FACE ENGINE (:8002)        │  │  VOICE ENGINE (:8001)  │  │    GEMINI / TEXT ENGINE (:8003) │
             │  RealTime-Facial-Expression-Rec  │  │ truesight-voice-analys │  │       TruSight_Gemini_API       │
             │                                  │  │                        │  │                                 │
             │ - MediaPipe 478 Mesh             │  │ - Parselmouth (F0,     │  │ - PyMuPDF Case PDF Indexing     │
             │ - AffectNet Emotion CNN          │  │   Jitter, Shimmer)     │  │ - Gemini Dense Embeddings & RAG │
             │ - 52-Cue Deception MLP           │  │ - Librosa Audio Pacing │  │ - Live Verbatim Transcription   │
             │ - 3D PnP Head Pose (Yaw/Pitch)   │  │ - 200-Tree RF Model    │  │ - Statement Contradiction Model │
             │ - FACS Action Units Engine       │  │ - Acoustic Evidence    │  │ - Fact Verification Explanation │
             └──────────────────────────────────┘  └────────────────────────┘  └─────────────────────────────────┘
```

---

## 📝 Text & Statement Contradiction Feature (Gemini RAG)

The **Text & Evidence Analysis Engine** enables cross-referencing between what an interviewee says in real time and what is officially documented in case reports, deposition records, or FIRs.

```
       [Case PDF] ──────► PyMuPDF Parsing ──────► Gemini Embeddings ──────► Evidence Vector Store
                                                                                   │
                                                                                   ▼
[Live Microphone] ──────► Speech-to-Text ──────► Relevant Excerpt Search ──► Gemini 3.8 Flash
                          (Gemini Live)                 (Cosine Sim)         (Contradiction & Reasoning)
                                                                                   │
                                                                                   ▼
                                                                        [Discrepancy Report & Score]
```

### How the Text Feature Works:
1. **Case Document Ingestion**: Upload any investigative report, FIR, police charge sheet, or contract PDF via the dashboard interface or `POST /api/gemini/case`.
2. **Evidence Chunking & Indexing**: The document is partitioned into semantic paragraphs and indexed with dense embeddings (`gemini-embedding-001`).
3. **Live Interview Session**: As the subject speaks, audio is transcribed utterance-by-utterance over a secure WebSocket pipeline.
4. **Automated Evidence Retrieval**: Each finalized sentence automatically queries the indexed case documents to retrieve relevant historical claims.
5. **Contradiction Assessment**: The model evaluates whether the utterance supports, is neutral to, or explicitly contradicts established case facts, generating:
   * **Contradiction Score** (0% - 100%)
   * **Verbatim Retrieved Citation**
   * **Forensic Reasoning Breakdown**

---

## 🚀 Quick Start

### Prerequisites
* **OS**: Windows 10/11 or Linux
* **Python**: `3.10` or `3.11` (recommended)
* **Webcam & Microphone**: Connected and permissions granted
* **Google Gemini API Key**: [Get your key here](https://aistudio.google.com/apikey)

---

### Step 1: Clone Repository & Configure Environment
```bash
git clone https://github.com/Likk18/Truusight.git
cd Truusight
```

Configure your Gemini API key in `TruSight_Gemini_API/TruSight_Gemini_API/.env`:
```dotenv
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_TEXT_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_LIVE_TRANSCRIPTION_MODEL=gemini-3.5-transcribe-live
```

---

### Step 2: Automated One-Click Setup (Windows)

Run the included setup script to install all dependencies for the Gateway, Face, and Voice engines:
```bat
.\setup_trusight.bat
```

*Or install manually using pip:*
```bash
# Gateway, Face, and Text dependencies
pip install -r requirements.txt

# Voice engine dedicated environment
cd truesight-voice-analysis/voice_service
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
deactivate
cd ../..
```

---

### Step 3: Launch the Platform

Launch all coordinated engines and open the unified dashboard with one command:

```bat
.\run_trusight.bat
```

*Or launch directly with Python:*
```bash
python trusight_launcher.py
```

#### What the Launcher Does:
1. Validates and clears ports `8000`, `8001`, `8002`, and `8003`.
2. Launches the **Voice Analysis Engine** on port `8001`.
3. Launches the **Face Analysis Engine** on port `8002`.
4. Launches the **Gemini Text & Evidence Engine** on port `8003`.
5. Launches the **Unified Gateway** on port `8000`.
6. Opens `http://localhost:8000` automatically in your default browser.
7. Shuts down all processes cleanly when you press `Ctrl+C`.

---

## 📊 Dashboard Overview

| Interface Section | Modality | Output Metrics & Capabilities |
| :--- | :--- | :--- |
| **Top Tactical Bar** | Tri-Modal Cross-Check | Side-by-side Face Verdict, Voice Deception Score, and Gemini Contradiction Risk with correlation assessment. |
| **Left Column** | Face Analysis | Live MJPEG video stream, 3D Head Pose (Pitch/Yaw/Roll), FACS Action Unit intensity bars, AffectNet emotion sentiment, Truth/Lie score. |
| **Middle Column** | Voice Analysis | Dual inputs (Mic / File upload), F0 pitch variation, vocal stability (jitter & shimmer), tempo/pauses, timeline history chart. |
| **Right Column** | Text & Evidence Analysis | PDF case upload, live speech transcription stream, retrieved evidence excerpts, and Gemini contradiction analysis. |

---

## 🌐 API Reference

### Gateway Unified Endpoints (Port 8000)

| Endpoint | Method | Backend Target | Description |
| :--- | :---: | :--- | :--- |
| `GET /` | `GET` | Gateway | Unified TRUSIGHT Forensic Dashboard HTML |
| `GET /api/health` | `GET` | All Engines | Aggregated health check for Face, Voice, and Gemini engines |
| `GET /api/face/video_feed` | `GET` | Face (`:8002`) | Live MJPEG camera stream with tactical overlays |
| `GET /api/face/live` | `GET` | Face (`:8002`) | Current frame telemetry (AUs, head pose, truth score) |
| `POST /api/face/camera/start` | `POST` | Face (`:8002`) | Starts webcam hardware feed |
| `POST /api/face/camera/stop` | `POST` | Face (`:8002`) | Releases webcam hardware feed |
| `POST /api/voice/analyze` | `POST` | Voice (`:8001`) | Acoustic analysis of an uploaded audio file |
| `POST /api/voice/live/analyze` | `POST` | Voice (`:8001`) | Chunk-based microphone audio analysis |
| `WS /ws/voice/live` | `WS` | Voice (`:8001`) | Bidirectional WebSocket for real-time acoustic telemetry |
| `POST /api/gemini/case` | `POST` | Gemini (`:8003`) | Multipart PDF upload and semantic indexing |
| `WS /ws/gemini/audio/{case_id}`| `WS` | Gemini (`:8003`) | Live PCM streaming for verbatim transcription and contradiction check |

---

## 📁 Directory Structure

```text
Truusight/
├── run_trusight.bat                             # One-click startup script (Windows)
├── setup_trusight.bat                           # Automated environment installer
├── trusight_launcher.py                         # Multi-process orchestrator & health manager
├── trusight_server.py                           # Unified FastAPI Gateway (port 8000)
├── requirements.txt                             # Top-level dependencies
├── README.md                                    # Comprehensive system documentation
│
├── static/                                      # Single-Page Forensic Dashboard UI
│   ├── index.html                               # Responsive three-column tactical layout
│   ├── css/
│   │   └── dashboard.css                        # Professional dark-mode intelligence theme
│   └── js/
│       ├── dashboard.js                         # Master synchronization & cross-modal logic
│       ├── face_module.js                       # Face telemetry & camera controller
│       ├── voice_module.js                      # Voice recording, audio upload & chart logic
│       └── gemini_module.js                     # Evidence PDF upload & text contradiction client
│
├── RealTime-Facial-Expression-Recognition/      # Face Analysis Subsystem (Port 8002)
│   └── AU/
│       ├── run_api.bat                          # Standalone Face API launcher
│       └── facial_cues/
│           ├── api/server.py                    # Face FastAPI streaming backend
│           ├── pipeline/orchestrator.py         # MediaPipe & PyTorch inference loop
│           └── models/                          # AffectNet & Deception neural weights
│
├── truesight-voice-analysis/                    # Voice Analysis Subsystem (Port 8001)
│   └── voice_service/
│       ├── run_server.bat                       # Standalone Voice service launcher
│       ├── .venv/                               # Isolated virtualenv (Parselmouth, Scikit-Learn)
│       └── app/
│           ├── main.py                          # Voice FastAPI endpoints
│           ├── predictor.py                     # Random Forest acoustic deception model
│           └── explanation.py                   # Physiological evidence generator
│
└── TruSight_Gemini_API/                         # Text & Evidence Subsystem (Port 8003)
    └── TruSight_Gemini_API/
        ├── .env.example                         # Template for Gemini API configuration
        ├── main.py                              # Gemini FastAPI server & WebSocket handler
        ├── rag.py                               # PDF parsing, semantic chunking & cosine search
        └── pipeline.py                          # Verification and contradiction reasoning
```

---

## 🔬 Scientific & Forensic Disclaimer

> [!WARNING]
> **RESEARCH USE NOTICE**: TRUSIGHT is an experimental multimodal investigative intelligence research platform.
>
> 1. **No Calibration for Infallible Deception Detection**: Physiological indicators (e.g., elevated heart rate, pitch variations, gaze aversion, Action Unit micro-movements) can be triggered by stress, cognitive load, anxiety, neurodiversity, or conversational context.
> 2. **Not Standalone Evidence**: Outputs produced by TRUSIGHT must **never** be used as the sole or definitive basis for legal, judicial, or consequential decisions.
> 3. **Human Review Obligatory**: All contradiction assessments and acoustic scores require rigorous verification by qualified forensic specialists and investigators.

---

## 🤝 Contributing

Contributions, issues, and feature proposals are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewForensicModality`).
3. Commit your Changes (`git commit -m 'Add NewForensicModality'`).
4. Push to the Branch (`git push origin feature/NewForensicModality`).
5. Open a Pull Request.

---

<div align="center">
  <sub>Built for advanced multimodal research and forensic intelligence.</sub>
</div>
