# TruSight — Gemini API edition

TruSight uses hosted Google Gemini APIs for text analysis, evidence embeddings, and real-time transcription. It does not load local Whisper, DeBERTa, SentenceTransformer, or Ollama models. The case PDF is still extracted locally with PyMuPDF; its text and interview excerpts are sent to Google Gemini for hosted inference/embeddings.

## Requirements

- Python 3.11 or newer
- A Google Gemini API key with access to the configured models
- A browser with microphone support (localhost or HTTPS)

## Install on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your key from https://aistudio.google.com/apikey. Never put the key in Python files, JavaScript, source control, screenshots, or public logs.

```dotenv
GEMINI_API_KEY=your_actual_key_here
GEMINI_TEXT_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_LIVE_TRANSCRIPTION_MODEL=gemini-3.5-transcribe-live
```

Run the server from this directory:

```powershell
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/live, upload a text-based PDF, then start the interview. The browser streams mono PCM audio to the FastAPI WebSocket, which forwards it to Gemini Live Transcription. Interim text is shown while speaking; finalized utterances trigger evidence retrieval, analysis, and explanation.

## API routes

- `GET /` — service health
- `GET /live` — embedded browser interface
- `POST /case` — upload and index a PDF case report
- `WS /ws/audio/{case_id}` — continuous live audio transcription and analysis

## Model and score notes

- `gemini-3.8-flash` analyzes interview statements against retrieved report excerpts.
- `gemini-embedding-001` creates document and query embeddings; NumPy cosine similarity performs local ranking.
- `gemini-3.5-transcribe-live` supplies live interim and finalized transcript events.
- The returned confidence is model-reported confidence in the textual relationship, not a calibrated probability that someone is lying. Contradiction means a candidate explicit conflict between a statement and the retrieved report excerpt; it is not a determination of deception, guilt, or credibility. Human review is required.

## Troubleshooting

- If case upload fails immediately, verify `.env` exists and `GEMINI_API_KEY` is set.
- If Gemini reports an unknown model or access error, check model availability for your key and update the corresponding model variable in `.env`.
- If microphone access fails, use `http://127.0.0.1:8000` or HTTPS and grant browser permission.
- If the PDF is scanned/image-only, PyMuPDF may extract no text; OCR would need to be added separately.
- Never commit `.env`. `.gitignore` excludes it.

## Tests

Run `python -m unittest discover -s tests -v` for offline unit tests. Live API behavior still requires a valid key and a real microphone/browser integration test.
