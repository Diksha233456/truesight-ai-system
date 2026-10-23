"""
process_reallife_and_retrain.py

Extracts audio from the Real-Life Trial Deception Detection dataset (.mp4 files),
merges them into dolos_dataset/truthful/ and dolos_dataset/deceptive/,
retrains the Random Forest classifier on the combined dataset, and reloads the API.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("process_reallife")

BASE_DIR = Path(__file__).parent
REALLIFE_DIR = BASE_DIR / "reallife_dataset_raw" / "Real-life_Deception_Detection_2016" / "Clips"
DATASET_DIR = BASE_DIR / "dolos_dataset"
TRUTHFUL_DIR = DATASET_DIR / "truthful"
DECEPTIVE_DIR = DATASET_DIR / "deceptive"

# Ensure ffmpeg is on PATH
EXTRA_PATHS = [
    r"C:\Users\diksh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin",
    r"C:\Program Files\FFmpeg\bin",
    r"C:\ProgramData\chocolatey\bin",
    r"C:\ffmpeg\bin",
]
os.environ["PATH"] = os.pathsep.join(EXTRA_PATHS) + os.pathsep + os.environ.get("PATH", "")


def extract_audio_from_mp4(mp4_path: Path, wav_path: Path) -> bool:
    """Convert MP4 video to 16kHz mono WAV audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(wav_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        return res.returncode == 0
    except Exception as e:
        logger.warning("Failed to extract audio from %s: %s", mp4_path.name, e)
        return False


def process_dataset():
    TRUTHFUL_DIR.mkdir(parents=True, exist_ok=True)
    DECEPTIVE_DIR.mkdir(parents=True, exist_ok=True)

    mappings = [
        (REALLIFE_DIR / "Deceptive", DECEPTIVE_DIR, "reallife_deceptive"),
        (REALLIFE_DIR / "Truthful", TRUTHFUL_DIR, "reallife_truthful"),
    ]

    total_extracted = 0
    for src_dir, target_dir, prefix in mappings:
        if not src_dir.exists():
            logger.error("Source directory missing: %s", src_dir)
            continue

        mp4_files = list(src_dir.glob("*.mp4"))
        logger.info("Found %d MP4 files in %s", len(mp4_files), src_dir.name)

        for mp4_file in mp4_files:
            wav_name = f"{prefix}_{mp4_file.stem}.wav"
            wav_path = target_dir / wav_name

            if wav_path.exists():
                logger.info("Already extracted: %s", wav_name)
                total_extracted += 1
                continue

            if extract_audio_from_mp4(mp4_file, wav_path):
                logger.info("✓ Extracted: %s -> %s", mp4_file.name, wav_name)
                total_extracted += 1
            else:
                logger.warning("✗ Failed: %s", mp4_file.name)

    logger.info("Total Real-Life audio clips ready: %d", total_extracted)


def retrain_model():
    train_script = BASE_DIR / "train_model.py"
    logger.info("Retraining Random Forest model on combined dataset (%s)...", DATASET_DIR)
    res = subprocess.run(
        [sys.executable, str(train_script), "--data-dir", str(DATASET_DIR)],
        cwd=str(BASE_DIR),
    )
    if res.returncode == 0:
        logger.info("✅ Model retrained successfully!")
    else:
        logger.error("Model training failed with exit code %d", res.returncode)
        sys.exit(1)


def reload_server():
    url = "http://127.0.0.1:8000/reload-model"
    logger.info("Reloading model on live server (%s)...", url)
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode()
            logger.info("Server reload response: %s", data)
    except Exception as e:
        logger.warning("Could not automatically reload live server: %s", e)


def main():
    logger.info("============================================================")
    logger.info("Processing Real-Life Trial Dataset & Merging with DOLOS")
    logger.info("============================================================")
    process_dataset()
    retrain_model()
    reload_server()
    logger.info("🎉 All done!")


if __name__ == "__main__":
    main()
