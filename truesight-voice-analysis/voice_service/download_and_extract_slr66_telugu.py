"""
download_and_extract_slr66_telugu.py

Resumable automated downloader and extractor for OpenSLR SLR66 Telugu female partition:
- Supports HTTP 206 partial content resume.
- Uses the verified ELDA mirror (https://openslr.elda.org/resources/66/te_in_female.zip).
- Safe extraction preventing Zip Slip vulnerabilities.
- Idempotent and preserves all source archives and extracted files.
"""

import os
import shutil
import sys
import time
import urllib.request
import zipfile

DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "datasets", "external", "slr66_telugu")
)
ZIP_NAME = "te_in_female.zip"
ZIP_PATH = os.path.join(DATASET_DIR, ZIP_NAME)
EXTRACT_DIR = os.path.join(DATASET_DIR, "extracted")
MIRROR_URL = "https://openslr.elda.org/resources/66/te_in_female.zip"
BACKUP_URL = "https://openslr.trmal.net/resources/66/te_in_female.zip"
TSV_URL = "https://openslr.elda.org/resources/66/line_index_female.tsv"
EXPECTED_TOTAL_BYTES = 505_680_804


def check_disk_space(required_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
    _, _, free = shutil.disk_usage(DATASET_DIR if os.path.exists(DATASET_DIR) else ".")
    if free < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space: {free / (1024**3):.2f} GB free, "
            f"requires at least {required_bytes / (1024**3):.2f} GB."
        )
    print(f"Disk space verified: {free / (1024**3):.2f} GB available.")


def safe_extract_zip(zip_path: str, target_dir: str) -> None:
    target_dir_abs = os.path.abspath(target_dir)
    os.makedirs(target_dir_abs, exist_ok=True)

    print(f"Extracting {zip_path} into {target_dir_abs} safely with Zip Slip checks...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_members = len(members)
        for idx, member in enumerate(members, 1):
            target_path = os.path.abspath(os.path.join(target_dir_abs, member.filename))
            if not target_path.startswith(target_dir_abs + os.sep) and target_path != target_dir_abs:
                raise RuntimeError(
                    f"Security violation: Zip Slip path traversal detected in member: {member.filename}"
                )
            zf.extract(member, target_dir_abs)
            if idx % 500 == 0 or idx == total_members:
                print(f"Extracted {idx}/{total_members} files...")
    print("Extraction completed successfully.")


def download_resumable(url: str, output_path: str, expected_size: int = EXPECTED_TOTAL_BYTES) -> None:
    current_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    if current_size == expected_size:
        print(f"Archive is already fully downloaded ({current_size} bytes).")
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if current_size > 0:
        if current_size < expected_size:
            print(f"Resuming download from byte {current_size} / {expected_size}...")
            headers["Range"] = f"bytes={current_size}-"
            mode = "ab"
        else:
            print(f"Local file size ({current_size}) > expected ({expected_size}). Restarting fresh.")
            current_size = 0

    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    last_print = start_time

    with urllib.request.urlopen(req, timeout=30) as resp, open(output_path, mode) as out_f:
        chunk_size = 1024 * 1024  # 1 MB
        downloaded = current_size
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 5.0 or downloaded >= expected_size:
                elapsed = max(0.1, now - start_time)
                speed_mb = ((downloaded - current_size) / (1024 * 1024)) / elapsed
                percent = (downloaded / expected_size) * 100.0 if expected_size > 0 else 0
                print(
                    f"Progress: {downloaded / (1024*1024):.1f}/{expected_size / (1024*1024):.1f} MB "
                    f"({percent:.1f}%) @ {speed_mb:.2f} MB/s"
                )
                last_print = now

    final_size = os.path.getsize(output_path)
    print(f"Download completed: {final_size} bytes in {time.time() - start_time:.1f}s.")
    if final_size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch: expected {expected_size}, got {final_size}")


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    check_disk_space()

    # 1. Download line index TSV if missing
    tsv_path = os.path.join(DATASET_DIR, "line_index_female.tsv")
    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        print(f"Downloading TSV from {TSV_URL}...")
        urllib.request.urlretrieve(TSV_URL, tsv_path)
        print(f"TSV downloaded: {os.path.getsize(tsv_path)} bytes.")
    else:
        print(f"TSV already present: {os.path.getsize(tsv_path)} bytes.")

    # 2. Resumable download of te_in_female.zip
    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) != EXPECTED_TOTAL_BYTES:
        download_resumable(MIRROR_URL, ZIP_PATH, EXPECTED_TOTAL_BYTES)
    else:
        print(f"Valid archive already exists at {ZIP_PATH} ({os.path.getsize(ZIP_PATH)} bytes).")

    # 3. Extract safely if not already extracted
    if not os.path.isdir(EXTRACT_DIR) or not os.listdir(EXTRACT_DIR):
        safe_extract_zip(ZIP_PATH, EXTRACT_DIR)
    else:
        print(f"Extracted directory already exists at {EXTRACT_DIR} with {len(os.listdir(EXTRACT_DIR))} items.")


if __name__ == "__main__":
    main()
