"""
prepare_cbu5201.py

Safe automated preparation script for the CBU5201 Deception Dataset:
1. Validates and extracts the CBU5201 archive with Zip Slip path-traversal protection.
2. Inspects real metadata CSV and audio files.
3. Generates normalized manifest CSV (datasets/external/cbu5201/manifest.csv).
4. Produces a comprehensive dataset inventory report.
5. Preserves all original downloaded archives, audio files, and metadata without alteration.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_cbu5201")

DEFAULT_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "datasets", "external", "cbu5201")
)
DEFAULT_ZIP_URL = "https://github.com/CBU5201Datasets/Deception/archive/refs/heads/main.zip"
DEFAULT_ZIP_NAME = "cbu5201_main.zip"
DEFAULT_EXTRACT_SUBDIR = "extracted"


class CBU5201PreparationError(Exception):
    """Raised when dataset preparation or validation fails."""


def safe_extract_zip(zip_path: str, target_dir: str) -> None:
    """
    Safely extract a zip archive preventing Zip Slip (path traversal) vulnerabilities.
    """
    if not os.path.exists(zip_path):
        raise CBU5201PreparationError(f"ZIP archive not found at: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise CBU5201PreparationError(f"File at {zip_path} is not a valid zip archive.")

    target_dir_abs = os.path.abspath(target_dir)
    os.makedirs(target_dir_abs, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = os.path.abspath(os.path.join(target_dir_abs, member.filename))
            # Verify target path is strictly within target directory
            if not member_path.startswith(target_dir_abs + os.sep) and member_path != target_dir_abs:
                raise CBU5201PreparationError(
                    f"Security violation: Zip Slip path traversal detected in member '{member.filename}'"
                )
            zf.extract(member, target_dir_abs)
    logger.info("Safely extracted %s into %s", zip_path, target_dir_abs)


def download_dataset_if_needed(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    zip_url: str = DEFAULT_ZIP_URL,
    zip_name: str = DEFAULT_ZIP_NAME,
) -> str:
    """
    Download the CBU5201 zip archive if not already downloaded and valid.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    zip_path = os.path.join(dataset_dir, zip_name)
    extract_dir = os.path.join(dataset_dir, DEFAULT_EXTRACT_SUBDIR)

    # Check if already extracted
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        logger.info("Extracted dataset directory already exists at %s", extract_dir)
        return extract_dir

    # Check if zip already exists
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        logger.info("Downloading CBU5201 dataset from %s...", zip_url)
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
            logger.info("Download completed: %d bytes saved to %s", os.path.getsize(zip_path), zip_path)
        except Exception as e:
            raise CBU5201PreparationError(f"Failed to download dataset from {zip_url}: {e}") from e

    # Extract
    safe_extract_zip(zip_path, extract_dir)
    return extract_dir


def map_story_type_to_label(story_type: str) -> Optional[int]:
    """
    Map CBU5201 Story_type string to binary integer label:
    - 'True Story' -> 0 (Truthful)
    - 'Deceptive Story' -> 1 (Deceptive)
    - Any other string or None -> None (ambiguous/missing, must NOT be treated as valid label)
    """
    if not story_type or not isinstance(story_type, str):
        return None
    normalized = story_type.strip().lower()
    if normalized == "true story" or normalized == "truthful":
        return 0
    if normalized == "deceptive story" or normalized == "deceptive":
        return 1
    return None


def locate_dataset_files(extract_dir: str) -> Tuple[str, str]:
    """
    Locate the metadata CSV and the audio stories folder inside the extracted directory.
    """
    csv_path = None
    audio_dir = None

    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith("_attributes.csv") or f == "attributes.csv" or (f.endswith(".csv") and "stories" in f.lower()):
                csv_path = os.path.join(root, f)
        for d in dirs:
            if "stories" in d.lower():
                audio_dir = os.path.join(root, d)

    if not csv_path or not os.path.exists(csv_path):
        raise CBU5201PreparationError(f"Metadata CSV not found inside {extract_dir}")
    if not audio_dir or not os.path.exists(audio_dir):
        raise CBU5201PreparationError(f"Audio directory not found inside {extract_dir}")

    return csv_path, audio_dir


def generate_manifest_and_inventory(
    csv_path: str,
    audio_dir: str,
    output_manifest_path: str,
    project_root: Optional[str] = None,
) -> Dict:
    """
    Parse original metadata CSV, check audio existence, map labels, and output
    a normalized manifest CSV along with an inventory dictionary.
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    rows_out: List[Dict] = []
    inventory = {
        "total_metadata_rows": 0,
        "total_audio_files_found_on_disk": 0,
        "valid_labelled_samples": 0,
        "truthful_count": 0,
        "deceptive_count": 0,
        "language_distribution": {},
        "missing_audio_files": [],
        "missing_or_ambiguous_labels": [],
        "speaker_identifier_present_in_metadata": False,
        "speaker_distribution_notes": (
            "No speaker or participant IDs are provided in CBU0521DD_stories_attributes.csv. "
            "Speaker identifiers are marked as 'UNKNOWN'. Cross-speaker independence cannot be "
            "verified from metadata alone; evaluating on this dataset cannot guarantee speaker-independent results."
        ),
        "dataset_limitation_notes": (
            "CBU5201 labels represent whether participants were recounting a truthful vs deceptive "
            "personal story in an academic experiment. Laboratory story-telling deception does not "
            "equate to real-world criminal or high-stakes deception detection. Furthermore, 0 Indian "
            "languages are present in this dataset (only Chinese and English)."
        ),
    }

    if not os.path.exists(csv_path):
        raise CBU5201PreparationError(f"Metadata CSV does not exist: {csv_path}")

    # Inspect all wav files present in audio_dir
    disk_files = {f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")}
    inventory["total_audio_files_found_on_disk"] = len(disk_files)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_speaker_col = any("speaker" in col.lower() or "participant" in col.lower() for col in fieldnames)
        inventory["speaker_identifier_present_in_metadata"] = has_speaker_col

        for row in reader:
            inventory["total_metadata_rows"] += 1
            filename = row.get("filename", "").strip()
            raw_language = row.get("Language", "").strip() or "UNKNOWN"
            raw_story = row.get("Story_type", "").strip()

            label = map_story_type_to_label(raw_story)
            abs_audio_path = os.path.join(audio_dir, filename)
            rel_audio_path = os.path.relpath(abs_audio_path, project_root).replace("\\", "/")

            file_exists = os.path.exists(abs_audio_path)
            file_size = os.path.getsize(abs_audio_path) if file_exists else 0

            # Language tracking
            inventory["language_distribution"][raw_language] = (
                inventory["language_distribution"].get(raw_language, 0) + 1
            )

            # Label tracking
            if label is None:
                inventory["missing_or_ambiguous_labels"].append(
                    {"filename": filename, "raw_story_type": raw_story}
                )
            elif not file_exists:
                inventory["missing_audio_files"].append(filename)
            else:
                inventory["valid_labelled_samples"] += 1
                if label == 0:
                    inventory["truthful_count"] += 1
                elif label == 1:
                    inventory["deceptive_count"] += 1

            speaker_id = row.get("speaker_id", "UNKNOWN") if has_speaker_col else "UNKNOWN"

            rows_out.append(
                {
                    "filename": filename,
                    "relative_audio_path": rel_audio_path,
                    "absolute_audio_path": os.path.abspath(abs_audio_path),
                    "label": label if label is not None else "",
                    "original_story_type": raw_story,
                    "language": raw_language,
                    "speaker_id": speaker_id,
                    "file_exists": file_exists,
                    "file_size_bytes": file_size,
                    "is_valid_labelled_sample": file_exists and (label is not None),
                }
            )

    # Write normalized manifest CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_manifest_path)), exist_ok=True)
    manifest_headers = [
        "filename",
        "relative_audio_path",
        "absolute_audio_path",
        "label",
        "original_story_type",
        "language",
        "speaker_id",
        "file_exists",
        "file_size_bytes",
        "is_valid_labelled_sample",
    ]
    with open(output_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_headers)
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info("Wrote normalized manifest with %d rows to %s", len(rows_out), output_manifest_path)
    return inventory


def main() -> None:
    logger.info("=== Starting CBU5201 Dataset Preparation ===")
    extract_dir = download_dataset_if_needed()
    csv_path, audio_dir = locate_dataset_files(extract_dir)
    manifest_path = os.path.join(DEFAULT_DATASET_DIR, "manifest.csv")
    inventory = generate_manifest_and_inventory(csv_path, audio_dir, manifest_path)

    # Save inventory JSON
    inventory_path = os.path.join(DEFAULT_DATASET_DIR, "inventory.json")
    with open(inventory_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    logger.info("=== Dataset Preparation Completed Successfully ===")
    logger.info("Total audio files found on disk: %d", inventory["total_audio_files_found_on_disk"])
    logger.info("Total metadata rows: %d", inventory["total_metadata_rows"])
    logger.info("Valid labelled samples: %d", inventory["valid_labelled_samples"])
    logger.info("Truthful (label=0): %d", inventory["truthful_count"])
    logger.info("Deceptive (label=1): %d", inventory["deceptive_count"])
    logger.info("Language breakdown: %s", inventory["language_distribution"])
    logger.info("Missing audio files: %d", len(inventory["missing_audio_files"]))
    logger.info("Missing/ambiguous labels: %d", len(inventory["missing_or_ambiguous_labels"]))
    logger.info("Manifest saved: %s", manifest_path)
    logger.info("Inventory saved: %s", inventory_path)


if __name__ == "__main__":
    main()
