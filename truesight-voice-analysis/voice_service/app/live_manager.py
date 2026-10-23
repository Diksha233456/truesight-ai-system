"""
live_manager.py

Browser-driven live recording session manager for TRUSIGHT.

Architecture change from v1
---------------------------
Previously the manager spawned a background thread that used sounddevice
(a Python PortAudio wrapper) to capture the SERVER's microphone. This is
wrong for a web application — the user's microphone is in their browser,
not on the server.

New flow (v2):
  1. Browser calls POST /live/start  → resets session state, marks running.
  2. Browser uses getUserMedia + MediaRecorder to record N-second segments.
  3. Browser calls POST /live/analyze with each completed audio blob.
  4. Backend decodes blob → existing ML pipeline → SegmentResult.
  5. Backend calls submit_segment() which stores the result AND broadcasts
     it to all connected WebSocket clients at /ws/live.
  6. Browser calls POST /live/stop when the user clicks Stop.

All WebSocket push, session history, and summary logic is preserved.
No sounddevice / PortAudio dependency remains in this module.

Thread-safety contract
----------------------
- _lock (threading.Lock) guards all mutable state reads/writes.
- WebSocket queues are per-connection asyncio.Queue objects; submit_segment()
  uses loop.call_soon_threadsafe so it is safe to call from any thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional
import numpy as np

from app.live_audio import VALID_WINDOWS, DEFAULT_WINDOW
from app.live_analyzer import SegmentResult
from app.utils import LoadedAudio, TARGET_SR

logger = logging.getLogger("voice_service.live_manager")

# Rolling buffer settings (10-12s audio memory at 16kHz)
MAX_ROLLING_BUFFER_SECONDS: float = 12.0
MAX_ROLLING_SAMPLES: int = int(MAX_ROLLING_BUFFER_SECONDS * TARGET_SR)  # 192,000 samples
MIN_VOICED_SECONDS_FOR_ROLLING: float = 2.5


class LiveSessionManager:
    """
    Singleton-style manager for a browser-driven live recording session.

    Only one session can run at a time. Calling start() while already running
    is a no-op (returns False). Call stop() explicitly before starting a new
    session with different parameters.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Session state (all guarded by _lock)
        self._running: bool = False
        self._window_seconds: int = DEFAULT_WINDOW
        self._segments: List[SegmentResult] = []
        self._session_start_time: float = 0.0
        self._current_status: str = "Stopped"  # "Listening" | "Processing" | "Stopped"

        # Rolling decoded audio sample buffer (float32 mono at TARGET_SR)
        self._rolling_samples: np.ndarray = np.empty(0, dtype=np.float32)
        self._last_chunk_fingerprint: Optional[int] = None

        # WebSocket subscriber queues  {client_id: asyncio.Queue}
        self._ws_queues: Dict[str, asyncio.Queue] = {}
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, window_seconds: int = DEFAULT_WINDOW) -> bool:
        """
        Mark a new live session as started.

        The browser is responsible for capturing audio via getUserMedia and
        submitting each completed segment via POST /live/analyze.

        Parameters
        ----------
        window_seconds:
            Duration of each analysis window (must be in VALID_WINDOWS).
            Stored for reference; the browser enforces the actual timing.

        Returns
        -------
        bool
            True if the session was started; False if already running.

        Raises
        ------
        ValueError
            If window_seconds is not a valid choice.
        """
        if window_seconds not in VALID_WINDOWS:
            raise ValueError(
                f"Invalid window: {window_seconds}s. "
                f"Choose from {sorted(VALID_WINDOWS)}."
            )

        with self._lock:
            if self._running:
                logger.info("start() called while session already running — restarting fresh session.")

            self._window_seconds = window_seconds
            self._segments = []
            self._session_start_time = time.monotonic()
            self._current_status = "Listening"
            self._rolling_samples = np.empty(0, dtype=np.float32)
            self._last_chunk_fingerprint = None
            self._running = True

        logger.info("Live session started (browser-driven, window=%ds).", window_seconds)
        return True

    @property
    def is_running(self) -> bool:
        """Return True if a live session is currently active."""
        with self._lock:
            return self._running

    def stop(self) -> None:
        """
        Mark the session as stopped and broadcast the final summary.

        Safe to call when not running (no-op).
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._current_status = "Stopped"
            self._rolling_samples = np.empty(0, dtype=np.float32)
            self._last_chunk_fingerprint = None

        logger.info("Live session stopped.")

        # Final broadcast so the frontend knows recording ended.
        self._broadcast(
            {
                "type": "session_stopped",
                "status": self.get_status(),
                "summary": self.get_session_summary(),
            }
        )

    def append_audio_samples(self, new_samples: np.ndarray, sample_rate: int = TARGET_SR) -> LoadedAudio:
        """
        Append decoded audio samples from a live chunk to the session's rolling buffer.
        
        Guarantees:
        1. Thread-safe concatenation under self._lock.
        2. Deduplication check: ignores duplicate identical chunk submissions.
        3. Hard FIFO cap: maintains rolling window of at most MAX_ROLLING_BUFFER_SECONDS.
        4. Returns a LoadedAudio representation of the consolidated rolling window.
        """
        with self._lock:
            if not self._running or new_samples.size == 0:
                # If not running or empty, return single-chunk LoadedAudio
                dur = len(new_samples) / float(sample_rate) if sample_rate > 0 else 0.0
                return LoadedAudio(
                    samples=new_samples,
                    sample_rate=sample_rate,
                    duration_seconds=dur,
                    source_filename="live_rolling.wav",
                )

            # Check fingerprint to prevent accidental duplicate submission
            # Fast fingerprint using sample length and head/tail/mean values
            chunk_fp = hash((len(new_samples), round(float(np.mean(new_samples[:50])), 4), round(float(np.mean(new_samples[-50:])), 4)))
            if self._last_chunk_fingerprint == chunk_fp:
                logger.warning("Duplicate audio chunk detected in live buffer; ignoring duplicate samples.")
            else:
                self._last_chunk_fingerprint = chunk_fp
                self._rolling_samples = np.concatenate([self._rolling_samples, new_samples])
                # Trim to FIFO maximum
                if len(self._rolling_samples) > MAX_ROLLING_SAMPLES:
                    self._rolling_samples = self._rolling_samples[-MAX_ROLLING_SAMPLES:]

            current_window = self._rolling_samples.copy()

        dur = len(current_window) / float(sample_rate) if sample_rate > 0 else 0.0
        raw_rms = float(np.sqrt(np.mean(current_window ** 2))) if current_window.size else 0.0
        raw_peak = float(np.max(np.abs(current_window))) if current_window.size else 0.0

        snr_estimate_db = 0.0
        if current_window.size > int(sample_rate * 0.03):
            try:
                frame_len = int(sample_rate * 0.03)
                hop_len = int(sample_rate * 0.015)
                num_frames = 1 + (len(current_window) - frame_len) // hop_len
                frame_energies = np.array([
                    np.sqrt(np.mean(current_window[i * hop_len : i * hop_len + frame_len] ** 2))
                    for i in range(num_frames)
                ])
                noise_floor = float(np.percentile(frame_energies, 10))
                signal_peak = float(np.percentile(frame_energies, 95))
                effective_floor = max(noise_floor, 1e-5)
                if signal_peak > effective_floor:
                    snr_estimate_db = float(20.0 * np.log10(signal_peak / effective_floor))
            except Exception:
                snr_estimate_db = 0.0

        return LoadedAudio(
            samples=current_window,
            sample_rate=sample_rate,
            duration_seconds=dur,
            source_filename="live_rolling.wav",
            raw_rms=raw_rms,
            raw_peak=raw_peak,
            snr_estimate_db=snr_estimate_db,
        )

    def get_rolling_buffer_duration(self) -> float:
        """Return the current duration in seconds of accumulated audio in the rolling buffer."""
        with self._lock:
            return len(self._rolling_samples) / float(TARGET_SR)

    def record_silence_chunk(self, reason: str = "Voice not detected") -> None:
        """
        Record that a silent / non-speech chunk arrived during live recording.
        Does NOT append to self._segments to prevent inflating segment counts.
        Resets the rolling buffer so stale speech is not carried forward across pauses.
        Updates status and broadcasts a status update.
        """
        with self._lock:
            if not self._running:
                return
            # Reset rolling buffer on pause to avoid stale speech contamination
            self._rolling_samples = np.empty(0, dtype=np.float32)
            self._last_chunk_fingerprint = None
            self._current_status = "Voice not detected — speak into your microphone."

        logger.info("Live pause/silence recorded: %s", reason)
        self._broadcast(
            {
                "type": "voice_status",
                "status": self.get_status(),
                "speech_detected": False,
                "message": "Voice not detected — speak into your microphone.",
                "detail": reason,
            }
        )

    def submit_segment(self, result: SegmentResult) -> None:
        """
        Accept a completed segment result (analysed by POST /live/analyze),
        store it, and broadcast it to all WebSocket clients.

        This is the primary way results enter the manager in browser-driven mode.
        Thread-safe: may be called from any thread or async context.
        """
        with self._lock:
            if not self._running:
                logger.warning("submit_segment() called when session is not running — ignoring.")
                return
            # Only store valid speech segments in history to prevent silence from inflating counts
            if result.speech_detected and result.verdict in ("Truthful", "Deceptive") and result.score is not None:
                self._segments.append(result)
            self._current_status = "Listening"

        logger.info(
            "Segment %s stored: %s score=%s conf=%s",
            result.segment,
            result.verdict,
            result.score,
            result.confidence,
        )

        self._broadcast(
            {
                "type": "segment_result",
                "data": result.to_dict(),
                "status": self.get_status(),
            }
        )

    def get_next_segment_index(self) -> int:
        """Return the 1-based index for the next segment to be submitted."""
        with self._lock:
            return len(self._segments) + 1

    def get_session_start_time(self) -> float:
        """Return the monotonic clock time when this session started."""
        with self._lock:
            return self._session_start_time

    def get_status(self) -> dict:
        """
        Return a snapshot of the current session state.

        Returns dict matching GET /live/status response schema.
        """
        with self._lock:
            running = self._running
            window = self._window_seconds
            n_segments = len(self._segments)
            elapsed = time.monotonic() - self._session_start_time if running else 0.0
            status_str = self._current_status

        elapsed_label = _fmt_seconds(elapsed) if running else "00:00"
        return {
            "running": running,
            "window": window,
            "segments_processed": n_segments,
            "elapsed_time": elapsed_label,
            "current_status": status_str,
        }

    def get_results(self) -> List[dict]:
        """Return a JSON-serialisable list of all completed segment results."""
        with self._lock:
            return [seg.to_dict() for seg in self._segments]

    def get_session_summary(self) -> dict:
        """Generate a summary of the completed (or ongoing) session."""
        with self._lock:
            segments = list(self._segments)

        if not segments:
            return {
                "total_segments": 0,
                "total_time": "00:00",
                "avg_score": None,
                "avg_confidence": None,
                "highest_risk_segment": None,
                "lowest_risk_segment": None,
                "overall_verdict": "No speech segments analyzed",
                "common_indicators": [],
            }

        speech_segments = [s for s in segments if s.verdict in ("Truthful", "Deceptive") and s.score is not None]
        total_secs = segments[-1].end_seconds if segments else 0.0

        if not speech_segments:
            return {
                "total_segments": 0,
                "total_time": _fmt_seconds(total_secs),
                "avg_score": None,
                "avg_confidence": None,
                "highest_risk_segment": None,
                "lowest_risk_segment": None,
                "overall_verdict": "No speech segments analyzed",
                "common_indicators": ["No usable human speech segments recorded."],
            }

        scores = [s.score for s in speech_segments if s.score is not None]
        confidences = [s.confidence for s in speech_segments if s.confidence is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        hr_seg = max(speech_segments, key=lambda s: s.score)
        lr_seg = min(speech_segments, key=lambda s: s.score)

        # Aggregate explanation phrases
        indicator_counts: Dict[str, int] = {}
        for seg in speech_segments:
            for note in seg.explanation:
                indicator_counts[note] = indicator_counts.get(note, 0) + 1
        common = sorted(indicator_counts, key=lambda k: -indicator_counts[k])[:5]

        deceptive_count = sum(1 for s in speech_segments if s.verdict == "Deceptive")
        overall = (
            "Experimental Model Output (Deceptive-Leaning)"
            if deceptive_count > len(speech_segments) / 2
            else "Experimental Model Output (Truthful-Leaning)"
        )

        return {
            "total_segments": len(speech_segments),
            "total_time": _fmt_seconds(total_secs),
            "avg_score": round(avg_score, 4),
            "avg_confidence": round(avg_conf, 4),
            "highest_risk_segment": hr_seg.to_dict(),
            "lowest_risk_segment": lr_seg.to_dict(),
            "overall_verdict": overall,
            "common_indicators": common,
        }

    # ------------------------------------------------------------------
    # WebSocket subscriber management
    # ------------------------------------------------------------------

    def register_ws_client(
        self, client_id: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Register a new WebSocket client to receive live push updates."""
        with self._lock:
            self._ws_queues[client_id] = queue
            self._ws_loop = loop
        logger.debug("WS client registered: %s", client_id)

    def unregister_ws_client(self, client_id: str) -> None:
        """Remove a WebSocket client (called on disconnect)."""
        with self._lock:
            self._ws_queues.pop(client_id, None)
        logger.debug("WS client unregistered: %s", client_id)

    def _broadcast(self, payload: dict) -> None:
        """
        Push a payload dict to all connected WebSocket clients.

        Safe to call from any thread (uses call_soon_threadsafe).
        """
        with self._lock:
            queues = list(self._ws_queues.values())
            loop = self._ws_loop

        if not queues or loop is None:
            return

        for q in queues:
            try:
                loop.call_soon_threadsafe(q.put_nowait, payload)
            except Exception as exc:
                logger.debug("Failed to enqueue WS broadcast: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_seconds(total: float) -> str:
    mins = int(total) // 60
    secs = int(total) % 60
    return f"{mins:02d}:{secs:02d}"
