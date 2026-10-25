"""Shared Groq API helper. Reads GROQ_API_KEY from environment."""
import json
import os
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=api_key)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def groq_chat_json(prompt):
    response = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return only valid JSON. Ground every conclusion in the provided evidence. Do not invent facts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Groq returned invalid JSON: {exc}") from exc
