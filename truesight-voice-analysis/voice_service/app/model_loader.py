"""
app/model_loader.py

Robust model loading, downloading, caching, and integrity validation for TRUSIGHT.
Ensures the voice deception analysis model is trained once and reused everywhere
without retraining on client machines or during application startup.

Key Features:
- Persistent local cache (default: ~/.cache/truesight/models/<version>)
- Pinned model versioning and SHA-256 integrity validation
- Safe atomic download writing to prevent corrupted/partial files
- Public downloading from Hugging Face Hub without requiring user credentials
- Configurable cache directory, repository, revision, and direct local overrides
- NEVER silently falls back to training when download or loading fails
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

logger = logging.getLogger("voice_service.model_loader")

# ── Defaults & Configuration ─────────────────────────────────────────────────
DEFAULT_MODEL_REPO = "sofi233/truesight-voice-deception"
DEFAULT_MODEL_REVISION = "v1.0.0"
MODEL_FILENAME = "deception_model.pkl"
METADATA_FILENAME = "model_metadata.json"

# Pinned SHA-256 checksums for verified releases (prevents supply-chain/network corruption)
PINNED_CHECKSUMS: Dict[str, Dict[str, str]] = {
    "v1.0.0": {
        "deception_model.pkl": "f2ad274377a33c5511ed1e01452113f75874f912818faeea95776cdd934f0b88",
    }
}


class ModelLoaderError(RuntimeError):
    """Base exception for model resolution and loading failures."""


class ModelNotFoundError(ModelLoaderError):
    """Raised when the model cannot be found in cache or local paths and auto-download is disabled."""


class ModelDownloadError(ModelLoaderError):
    """Raised when downloading the pretrained model fails due to network or repository errors."""


class ModelIntegrityError(ModelLoaderError):
    """Raised when downloaded or cached model file fails SHA-256 integrity check."""


def get_default_cache_dir() -> Path:
    """Determine the root cache directory for pretrained models."""
    env_cache = os.environ.get("TRUSIGHT_MODEL_CACHE_DIR")
    if env_cache:
        return Path(env_cache).expanduser().resolve()

    # Use standard user home cache directory
    try:
        user_cache = Path.home() / ".cache" / "truesight" / "models"
        user_cache.mkdir(parents=True, exist_ok=True)
        return user_cache
    except Exception:
        # Fallback to local project directory if home directory is not writable
        local_cache = Path(__file__).resolve().parent / "models" / "cache"
        local_cache.mkdir(parents=True, exist_ok=True)
        return local_cache


def compute_sha256(filepath: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_integrity(filepath: str | Path, expected_sha256: str) -> bool:
    """Check whether a file matches the expected SHA-256 digest."""
    if not os.path.isfile(filepath):
        return False
    actual_sha256 = compute_sha256(filepath)
    return actual_sha256.lower() == expected_sha256.lower()


def download_file_atomic(
    url: str,
    destination: Path,
    expected_sha256: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """
    Download a remote file atomically: streams to a temporary file,
    validates checksum, and renames only on success.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = destination.parent / f"{destination.name}.tmp.{uuid.uuid4().hex}"

    try:
        req = urllib.request.Request(url, headers=headers or {"User-Agent": "TRUSIGHT-Model-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_file, "wb") as out_f:
            shutil.copyfileobj(response, out_f)

        if expected_sha256:
            actual_sha256 = compute_sha256(temp_file)
            if actual_sha256.lower() != expected_sha256.lower():
                raise ModelIntegrityError(
                    f"Integrity check failed for downloaded file {destination.name}. "
                    f"Expected SHA-256: {expected_sha256}, got: {actual_sha256}."
                )

        # Atomic replacement
        temp_file.replace(destination)
    except Exception as e:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        if isinstance(e, ModelIntegrityError):
            raise
        raise ModelDownloadError(
            f"Failed to download {url} to {destination}: {e}. "
            "Please check your internet connection or configure a manual model path via TRUSIGHT_MODEL_PATH."
        ) from e


def fetch_from_huggingface(
    repo_id: str,
    filename: str,
    revision: str,
    destination: Path,
    expected_sha256: Optional[str] = None,
) -> None:
    """
    Download a file from Hugging Face Hub using huggingface_hub if available,
    falling back to direct HTTPS download.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s (revision: %s) from Hugging Face repo %s via hf_hub_download...", filename, revision, repo_id)
        downloaded_cache_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
        )
        if expected_sha256 and not verify_file_integrity(downloaded_cache_path, expected_sha256):
            raise ModelIntegrityError(
                f"Downloaded artifact {filename} failed SHA-256 check. Expected: {expected_sha256}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_cache_path, destination)
        return
    except ImportError:
        logger.debug("huggingface_hub not installed; falling back to direct Hugging Face HTTPS download.")
    except Exception as e:
        if isinstance(e, ModelIntegrityError):
            raise
        logger.warning("hf_hub_download failed (%s); trying direct HTTPS fallback...", e)

    # Direct HTTPS fallback from Hugging Face Hub resolve endpoint
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    download_file_atomic(url=url, destination=destination, expected_sha256=expected_sha256, headers=headers)


def resolve_model(
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
    force_download: bool = False,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """
    Resolve the pretrained model file path and metadata.

    Order of precedence:
    1. TRUSIGHT_MODEL_PATH environment variable (explicit file override)
    2. Local persistent cache (~/.cache/truesight/models/<revision>/deception_model.pkl)
    3. Bundled local application model (app/models/deception_model.pkl) if valid
    4. Automatic download from Hugging Face Hub (cached persistently for subsequent runs)

    Returns:
        (model_path, metadata_dict)
    """
    resolved_repo = repo_id or os.environ.get("TRUSIGHT_MODEL_REPO", DEFAULT_MODEL_REPO)
    resolved_revision = revision or os.environ.get("TRUSIGHT_MODEL_REVISION", DEFAULT_MODEL_REVISION)
    root_cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    version_cache_dir = root_cache / resolved_revision
    cached_model_path = version_cache_dir / MODEL_FILENAME
    cached_metadata_path = version_cache_dir / METADATA_FILENAME

    # 1. Explicit local path override
    explicit_path = os.environ.get("TRUSIGHT_MODEL_PATH")
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.is_file():
            logger.info("Using explicit local model specified by TRUSIGHT_MODEL_PATH: %s", p)
            meta = None
            meta_path = p.parent / METADATA_FILENAME
            if meta_path.is_file():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            return p, meta
        raise ModelNotFoundError(
            f"Environment variable TRUSIGHT_MODEL_PATH is set to '{explicit_path}', "
            "but this file does not exist."
        )

    expected_sha256 = PINNED_CHECKSUMS.get(resolved_revision, {}).get(MODEL_FILENAME)

    # 2. Check persistent local cache
    if not force_download and cached_model_path.is_file():
        # Validate checksum if known
        if expected_sha256:
            if verify_file_integrity(cached_model_path, expected_sha256):
                logger.info(
                    "Loaded pretrained model from persistent cache: %s (revision: %s)",
                    cached_model_path,
                    resolved_revision,
                )
                meta = _read_metadata_if_present(cached_metadata_path)
                return cached_model_path, meta
            else:
                logger.warning(
                    "Cached model at %s is corrupted (SHA-256 mismatch). Invalidating cache and re-fetching.",
                    cached_model_path,
                )
                try:
                    cached_model_path.unlink()
                except Exception:
                    pass
        else:
            logger.info("Loaded pretrained model from persistent cache: %s", cached_model_path)
            meta = _read_metadata_if_present(cached_metadata_path)
            return cached_model_path, meta

    # 3. Check bundled project models directory (e.g. app/models/deception_model.pkl)
    bundled_model_path = Path(__file__).resolve().parent / "models" / MODEL_FILENAME
    if not force_download and bundled_model_path.is_file():
        if not expected_sha256 or verify_file_integrity(bundled_model_path, expected_sha256):
            logger.info("Found verified model in local project repository: %s", bundled_model_path)
            # Copy to persistent cache to ensure subsequent runs use the cached directory
            try:
                version_cache_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bundled_model_path, cached_model_path)
                bundled_meta = Path(__file__).resolve().parent / "models" / METADATA_FILENAME
                if bundled_meta.is_file():
                    shutil.copy2(bundled_meta, cached_metadata_path)
            except Exception as e:
                logger.debug("Could not copy bundled model to cache: %s", e)
            meta = _read_metadata_if_present(Path(__file__).resolve().parent / "models" / METADATA_FILENAME)
            return bundled_model_path, meta

    # 4. Automatic Download from Hugging Face Hub
    auto_download = os.environ.get("TRUSIGHT_AUTO_DOWNLOAD", "true").lower() in ("true", "1", "yes")
    if not auto_download:
        raise ModelNotFoundError(
            f"Pretrained model '{MODEL_FILENAME}' revision '{resolved_revision}' is not in local cache "
            f"({version_cache_dir}) and TRUSIGHT_AUTO_DOWNLOAD is disabled. "
            "Set TRUSIGHT_AUTO_DOWNLOAD=true, or download the model manually and set TRUSIGHT_MODEL_PATH."
        )

    logger.info(
        "Pretrained model not found locally. Downloading %s (revision %s) from %s ...",
        MODEL_FILENAME,
        resolved_revision,
        resolved_repo,
    )
    version_cache_dir.mkdir(parents=True, exist_ok=True)

    # Download model artifact
    fetch_from_huggingface(
        repo_id=resolved_repo,
        filename=MODEL_FILENAME,
        revision=resolved_revision,
        destination=cached_model_path,
        expected_sha256=expected_sha256,
    )

    # Optionally download metadata
    try:
        fetch_from_huggingface(
            repo_id=resolved_repo,
            filename=METADATA_FILENAME,
            revision=resolved_revision,
            destination=cached_metadata_path,
            expected_sha256=None,
        )
    except Exception as e:
        logger.debug("Metadata artifact could not be downloaded (%s), continuing with model.", e)

    meta = _read_metadata_if_present(cached_metadata_path)
    logger.info("Successfully downloaded and cached pretrained model to %s", cached_model_path)
    return cached_model_path, meta


def _read_metadata_if_present(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("Failed to read metadata at %s: %s", path, e)
    return None


def clear_cache(revision: Optional[str] = None) -> None:
    """
    Remove cached models. If revision is provided, clears only that version;
    otherwise clears all cached model versions.
    """
    root_cache = get_default_cache_dir()
    if revision:
        target = root_cache / revision
        if target.exists():
            shutil.rmtree(target)
            logger.info("Cleared cache for model revision %s (%s)", revision, target)
    else:
        if root_cache.exists():
            shutil.rmtree(root_cache)
            logger.info("Cleared entire model cache at %s", root_cache)


def main():
    """CLI to inspect, download, or clear the model cache."""
    import argparse
    parser = argparse.ArgumentParser(description="TRUSIGHT Model Loader & Cache Manager")
    parser.add_argument("--download", action="store_true", help="Download and cache the configured pretrained model.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the local model cache.")
    parser.add_argument("--repo-id", default=None, help="Hugging Face repository ID.")
    parser.add_argument("--revision", default=None, help="Model revision string (e.g. v1.0.0).")
    parser.add_argument("--force", action="store_true", help="Force re-download even if already cached.")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache(args.revision)
        print("Model cache cleared.")
        return

    try:
        path, meta = resolve_model(
            repo_id=args.repo_id,
            revision=args.revision,
            force_download=args.force,
        )
        print(f"Pretrained Model Resolved Successfully:")
        print(f"  Path: {path}")
        print(f"  Size: {os.path.getsize(path)} bytes")
        print(f"  SHA-256: {compute_sha256(path)}")
        if meta:
            print(f"  Version: {meta.get('model_version')}")
            print(f"  Architecture: {meta.get('architecture')}")
            print(f"  Held-out Accuracy: {meta.get('evaluation_metrics', {}).get('held_out_accuracy')}")
    except Exception as e:
        print(f"Error resolving model: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
