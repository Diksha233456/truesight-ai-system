"""Shared Gemini client configuration. API keys are read from environment only."""
import os
from functools import lru_cache
from dotenv import load_dotenv
from google import genai

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
LIVE_TRANSCRIPTION_MODEL = os.getenv(
    "GEMINI_LIVE_TRANSCRIPTION_MODEL", "gemini-3.5-transcribe-live"
)
