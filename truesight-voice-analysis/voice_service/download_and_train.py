"""
download_and_train.py

Full pipeline to:
1. Download DOLOS dataset audio clips from YouTube using yt-dlp
2. Trim clips to exact timestamps
3. Organize into truthful/ and deceptive/ folders
4. Retrain the Random Forest classifier on real data

Usage:
    python download_and_train.py

Requirements (auto-installed if missing):
    yt-dlp, ffmpeg (must be on PATH)

The DOLOS dataset is from the paper:
  "Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient
   Crossmodal Learning" (ICCV 2023)
  https://arxiv.org/abs/2303.12745
"""

from __future__ import annotations

import csv
import io
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("download_and_train")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dolos_dataset"
TRUTHFUL_DIR = DATASET_DIR / "truthful"
DECEPTIVE_DIR = DATASET_DIR / "deceptive"
TEMP_DIR = DATASET_DIR / "_temp_full_videos"

# ── Inline CSV (DOLOS timestamps recovered from git history) ───────────────
# Columns: YT_Video_ID, file_name, start_time, end_time, label
DOLOS_CSV_INLINE = """\
YT_Video_ID,file_name,start_time,end_time,label
UCPIBQDI0QM,AN_WILTY_EP15_truth1,2:06,2:11,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth2,2:25,2:32,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth3,5:50,5:55,truth
UCPIBQDI0QM,AN_WILTY_EP15_lie4,7:03,7:06,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie5,7:32,7:36,deception
UCPIBQDI0QM,AN_WILTY_EP15_truth6,8:58,9:04,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth7,9:48,9:51,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth8,17:04,17:08,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth9,17:44,17:47,truth
UCPIBQDI0QM,AN_WILTY_EP15_lie10,18:17,18:20,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie11,18:30,18:33,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie12,18:41,18:45,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie13,18:49,18:53,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie14,19:01,19:07,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie15,19:22,19:26,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie16,19:29,19:34,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie17,19:48,19:53,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie18,23:38,23:42,deception
UCPIBQDI0QM,AN_WILTY_EP15_lie19,23:44,23:47,deception
UCPIBQDI0QM,AN_WILTY_EP15_truth20,25:26,25:31,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth21,25:35,25:39,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth22,25:44,25:47,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth23,26:14,26:23,truth
UCPIBQDI0QM,AN_WILTY_EP15_truth24,26:28,26:37,truth
hhSNKJ_lcNU,AN_WILTY_EP16_lie1,2:04,2:07,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie2,2:21,2:27,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie3,2:42,2:46,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie4,2:50,2:54,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie5,3:01,3:05,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie6,3:11,3:15,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie7,3:33,3:36,deception
hhSNKJ_lcNU,AN_WILTY_EP16_truth8,5:29,5:37,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth9,5:55,6:00,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth10,6:04,6:09,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth11,6:14,6:17,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth12,8:51,8:58,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth13,9:07,9:11,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth14,9:40,9:47,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth15,9:50,9:53,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth16,15:57,16:06,truth
hhSNKJ_lcNU,AN_WILTY_EP16_lie17,16:11,16:21,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie18,16:23,16:27,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie19,16:36,16:40,deception
hhSNKJ_lcNU,AN_WILTY_EP16_lie20,16:51,16:55,deception
hhSNKJ_lcNU,AN_WILTY_EP16_truth21,20:10,20:22,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth22,20:24,20:31,truth
hhSNKJ_lcNU,AN_WILTY_EP16_truth23,20:33,20:40,truth
YS-EkPwZmpY,CS_WILTY_EP2_lie1,0:09,0:16,deception
YS-EkPwZmpY,CS_WILTY_EP2_lie2,0:19,0:28,deception
YS-EkPwZmpY,CS_WILTY_EP2_lie3,1:01,1:07,deception
YS-EkPwZmpY,CS_WILTY_EP2_truth4,2:12,2:19,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth5,2:25,2:32,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth6,2:36,2:40,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth7,4:34,4:45,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth8,5:11,5:17,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth9,7:26,7:34,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth10,7:38,7:44,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth11,7:51,7:57,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth12,8:02,8:07,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth13,9:41,9:46,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth14,9:49,9:54,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth15,9:58,10:02,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth16,10:06,10:09,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth17,10:13,10:16,truth
YS-EkPwZmpY,CS_WILTY_EP2_lie18,11:01,11:10,deception
YS-EkPwZmpY,CS_WILTY_EP2_lie19,11:13,11:23,deception
YS-EkPwZmpY,CS_WILTY_EP2_lie20,11:25,11:38,deception
YS-EkPwZmpY,CS_WILTY_EP2_truth21,12:39,12:47,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth22,12:51,12:57,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth23,13:00,13:06,truth
YS-EkPwZmpY,CS_WILTY_EP2_truth24,13:08,13:12,truth
"""


def timestamp_to_seconds(ts: str) -> float:
    """Convert 'M:SS' or 'MM:SS' timestamp to seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def ensure_yt_dlp() -> None:
    """Check yt-dlp is available (should be pre-installed in .venv)."""
    try:
        # Try as a module first (works even if not on PATH)
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True, check=True
        )
        logger.info("yt-dlp is available: %s", result.stdout.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("Installing yt-dlp into current venv...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yt-dlp", "--quiet"],
            check=True
        )


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is on PATH (check refreshed env)."""
    # Build a PATH that includes the winget-installed location
    extra_paths = [
        r"C:\Users\diksh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin",
        r"C:\Program Files\FFmpeg\bin",
        r"C:\ProgramData\chocolatey\bin",
        r"C:\ffmpeg\bin",
    ]
    os.environ["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + os.environ.get("PATH", "")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_video_audio(yt_id: str, out_dir: Path) -> Path | None:
    """
    Download the full audio track of a YouTube video as WAV.
    Returns the path to the downloaded file, or None on failure.
    """
    out_path = out_dir / f"{yt_id}.wav"
    if out_path.exists():
        logger.info("Already downloaded: %s", yt_id)
        return out_path

    url = f"https://www.youtube.com/watch?v={yt_id}"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(out_dir / f"{yt_id}.%(ext)s"),
        "--quiet",
        "--no-warnings",
        url,
    ]
    try:
        logger.info("Downloading YT video: %s", yt_id)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning("yt-dlp failed for %s: %s", yt_id, result.stderr[:200])
            return None
        # yt-dlp may use different extension; find the file
        for f in out_dir.glob(f"{yt_id}.*"):
            if f.suffix in (".wav", ".webm", ".m4a", ".mp3", ".opus"):
                if f.suffix != ".wav":
                    wav_path = out_dir / f"{yt_id}.wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(f), str(wav_path)],
                        capture_output=True, check=True
                    )
                    f.unlink()
                    return wav_path
                return f
        logger.warning("Downloaded file not found for %s", yt_id)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Timeout downloading %s", yt_id)
        return None
    except Exception as e:
        logger.warning("Error downloading %s: %s", yt_id, e)
        return None


def trim_clip(source_wav: Path, start_sec: float, end_sec: float, out_path: Path) -> bool:
    """Trim a WAV file to [start_sec, end_sec] using ffmpeg."""
    duration = end_sec - start_sec
    if duration <= 0:
        logger.warning("Invalid duration for clip %s", out_path.name)
        return False
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(source_wav),
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz to match TARGET_SR
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        logger.warning("ffmpeg trim failed for %s: %s", out_path.name, e)
        return False


def load_timestamps() -> list[dict]:
    """Parse the inline DOLOS timestamps CSV."""
    rows = []
    reader = csv.DictReader(io.StringIO(DOLOS_CSV_INLINE.strip()))
    for row in reader:
        row = {k.lstrip("\ufeff"): v for k, v in row.items()}
        rows.append(row)
    return rows


def build_dataset() -> tuple[int, int, int]:
    """Download and trim all DOLOS audio clips."""
    TRUTHFUL_DIR.mkdir(parents=True, exist_ok=True)
    DECEPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_timestamps()
    logger.info("Found %d clips in DOLOS timestamps.", len(rows))

    # Group by YT video ID so we download each video only once
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        vid = row["YT_Video_ID"].strip()
        by_video.setdefault(vid, []).append(row)

    total = len(rows)
    downloaded = 0
    skipped = 0

    for yt_id, clips in by_video.items():
        full_audio = download_video_audio(yt_id, TEMP_DIR)
        if full_audio is None:
            logger.warning("Skipping %d clips from %s (download failed)", len(clips), yt_id)
            skipped += len(clips)
            continue

        for clip in clips:
            label = clip["label"].strip().lower()
            fname = clip["file_name"].strip()
            start = timestamp_to_seconds(clip["start_time"])
            end = timestamp_to_seconds(clip["end_time"])

            out_dir = TRUTHFUL_DIR if label == "truth" else DECEPTIVE_DIR
            out_path = out_dir / f"{fname}.wav"

            if out_path.exists():
                logger.info("Clip already exists: %s", fname)
                downloaded += 1
                continue

            ok = trim_clip(full_audio, start, end, out_path)
            if ok:
                logger.info("✓ %s → %s", fname, "truthful" if label == "truth" else "deceptive")
                downloaded += 1
            else:
                logger.warning("✗ Failed to trim: %s", fname)
                skipped += 1

        time.sleep(1)

    return total, downloaded, skipped


def retrain_model() -> None:
    """Run train_model.py pointing at the downloaded dataset."""
    train_script = BASE_DIR / "train_model.py"
    logger.info("Retraining model on real DOLOS data...")
    result = subprocess.run(
        [sys.executable, str(train_script), "--data-dir", str(DATASET_DIR)],
        cwd=str(BASE_DIR),
    )
    if result.returncode == 0:
        logger.info("Model retrained successfully!")
    else:
        logger.error("Model training failed with exit code %d", result.returncode)
        sys.exit(1)


def main():
    logger.info("=" * 60)
    logger.info("DOLOS Dataset Download + Retrain Pipeline")
    logger.info("=" * 60)

    # 1. Check dependencies
    ensure_yt_dlp()
    if not check_ffmpeg():
        logger.error(
            "ffmpeg not found on PATH!\n"
            "Install it with:  winget install ffmpeg\n"
            "Then restart your terminal and re-run this script."
        )
        sys.exit(1)

    # 2. Download & trim clips
    logger.info("\nStep 1: Downloading DOLOS audio clips from YouTube...")
    total, downloaded, skipped = build_dataset()

    logger.info("\nDownload Summary:")
    logger.info("  Total clips:     %d", total)
    logger.info("  Downloaded:      %d", downloaded)
    logger.info("  Skipped/failed:  %d", skipped)

    truth_count = len(list(TRUTHFUL_DIR.glob("*.wav")))
    decep_count = len(list(DECEPTIVE_DIR.glob("*.wav")))
    logger.info("  Truthful clips:  %d", truth_count)
    logger.info("  Deceptive clips: %d", decep_count)

    if truth_count == 0 or decep_count == 0:
        logger.error(
            "Not enough clips to train (truth=%d, deceptive=%d).\n"
            "Some YouTube videos may be private or deleted.\n"
            "Try requesting the full dataset from: https://rose1.ntu.edu.sg/dataset/DOLOS/",
            truth_count, decep_count,
        )
        sys.exit(1)

    # 3. Retrain
    logger.info("\nStep 2: Retraining model on real DOLOS data...")
    retrain_model()

    logger.info("\nDone! Model is now trained on real DOLOS speech data.")
    logger.info("Hit POST /reload-model in Swagger to load it without restarting.")


if __name__ == "__main__":
    main()
