#!/usr/bin/env python3
"""
download_multispeaker_dataset.py

Downloads the Santhosh-kumar/Indian-Accent-Dataset (Apache 2.0) into
datasets/external/indian_english_multispeaker/ with preserved transcripts and manifest.
"""

import os
import csv
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_multispeaker")

REPO = "Santhosh-kumar/Indian-Accent-Dataset"
API_URL = f"https://huggingface.co/api/datasets/{REPO}"
BASE_RAW_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"

TARGET_DIR = os.path.join("datasets", "external", "indian_english_multispeaker")
AUDIO_DIR = os.path.join(TARGET_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(TARGET_DIR, "transcripts")
MANIFEST_PATH = os.path.join(TARGET_DIR, "manifest.csv")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

logger.info("Fetching repository metadata from Hugging Face...")
res = requests.get(API_URL, headers={"User-Agent": "Mozilla/5.0"}).json()
siblings = res.get("siblings", [])

wav_files = [s["rfilename"] for s in siblings if s["rfilename"].endswith(".wav")]
logger.info(f"Found {len(wav_files)} WAV files to download.")

manifest_rows = []

for idx, wav_rel in enumerate(wav_files, 1):
    fn = os.path.basename(wav_rel)
    txt_rel = wav_rel.replace(".wav", ".txt")
    txt_fn = os.path.basename(txt_rel)

    wav_dest = os.path.join(AUDIO_DIR, fn)
    txt_dest = os.path.join(TRANSCRIPT_DIR, txt_fn)

    # Download WAV if not present
    if not os.path.exists(wav_dest):
        wav_url = f"{BASE_RAW_URL}/{wav_rel}"
        r_wav = requests.get(wav_url, headers={"User-Agent": "Mozilla/5.0"})
        with open(wav_dest, "wb") as f:
            f.write(r_wav.content)

    # Download TXT if not present
    transcript = ""
    if not os.path.exists(txt_dest):
        txt_url = f"{BASE_RAW_URL}/{txt_rel}"
        r_txt = requests.get(txt_url, headers={"User-Agent": "Mozilla/5.0"})
        with open(txt_dest, "wb") as f:
            f.write(r_txt.content)
        txt_content = r_txt.text.strip()
    else:
        with open(txt_dest, "r", encoding="utf-8", errors="ignore") as f:
            txt_content = f.read().strip()

    # Parse transcript line: format usually "audio/xxx.wav\t<prompt_id>\t<text>"
    parts = txt_content.split("\t")
    if len(parts) >= 3:
        transcript = parts[2]
        prompt_id = parts[1]
    elif len(parts) == 2:
        transcript = parts[1]
        prompt_id = parts[0]
    else:
        transcript = txt_content
        prompt_id = "UNKNOWN"

    # Filename format: e.g. 281474976879627_f1171_chunk_0.wav
    # parts: [session_id, prompt_token, chunk_info]
    name_parts = fn.replace(".wav", "").split("_")
    session_id = name_parts[0] if len(name_parts) >= 1 else f"SPK_{idx:03d}"
    prompt_token = name_parts[1] if len(name_parts) >= 2 else prompt_id

    file_size = os.path.getsize(wav_dest)

    manifest_rows.append({
        "filename": fn,
        "audio_path": os.path.relpath(wav_dest, "."),
        "session_id": session_id,
        "speaker_id": session_id,
        "prompt_token": prompt_token,
        "prompt_id": prompt_id,
        "transcript": transcript,
        "file_size_bytes": file_size,
        "license": "Apache-2.0",
        "dataset_source": REPO
    })

    if idx % 10 == 0 or idx == len(wav_files):
        logger.info(f"Progress: {idx}/{len(wav_files)} files downloaded.")

# Write manifest.csv
with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename", "audio_path", "session_id", "speaker_id", "prompt_token",
        "prompt_id", "transcript", "file_size_bytes", "license", "dataset_source"
    ])
    writer.writeheader()
    writer.writerows(manifest_rows)

logger.info(f"Manifest written to {MANIFEST_PATH} with {len(manifest_rows)} entries.")
print(f"SUCCESS: Ingested {len(manifest_rows)} files into {TARGET_DIR}")
