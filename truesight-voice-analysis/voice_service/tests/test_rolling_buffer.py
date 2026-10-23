"""
tests/test_rolling_buffer.py

Unit tests for Stage 3: Multi-Scale Rolling Audio Buffer in live_manager and main pipeline:
1. Buffer accumulation across multiple chunks.
2. FIFO trimming when buffer exceeds MAX_ROLLING_BUFFER_SECONDS (12.0s).
3. Session reset clearing the buffer completely on start/stop.
4. Deduplication: identical audio chunks are detected and not duplicated.
5. Handling insufficient speech vs sufficient speech across rolling chunks.
6. Ensuring live session lifecycle and segment results remain intact and schema-compliant.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.live_manager import (
    LiveSessionManager,
    MAX_ROLLING_BUFFER_SECONDS,
    MAX_ROLLING_SAMPLES,
)
from app.utils import TARGET_SR, LoadedAudio
from app.live_analyzer import SegmentResult


class TestRollingAudioBuffer(unittest.TestCase):

    def setUp(self):
        self.manager = LiveSessionManager()

    def test_buffer_accumulation_and_fifo_trimming(self):
        """Buffer should accumulate incoming samples up to MAX_ROLLING_SAMPLES and trim older samples."""
        self.manager.start(window_seconds=15)

        # Append 3 distinct 2.0s chunks (6.0s total)
        for i in range(3):
            samples = np.full(TARGET_SR * 2, fill_value=0.1 * (i + 1), dtype=np.float32)
            rolling = self.manager.append_audio_samples(samples, TARGET_SR)

        self.assertAlmostEqual(self.manager.get_rolling_buffer_duration(), 6.0, places=2)
        self.assertAlmostEqual(rolling.duration_seconds, 6.0, places=2)

        # Now append 4 more 2.0s chunks (8.0s more, total 14.0s) -> should cap at MAX_ROLLING_BUFFER_SECONDS (12.0s)
        for i in range(4):
            samples = np.full(TARGET_SR * 2, fill_value=0.2 * (i + 1), dtype=np.float32)
            rolling = self.manager.append_audio_samples(samples, TARGET_SR)

        self.assertAlmostEqual(self.manager.get_rolling_buffer_duration(), MAX_ROLLING_BUFFER_SECONDS, places=2)
        self.assertEqual(len(self.manager._rolling_samples), MAX_ROLLING_SAMPLES)
        self.assertEqual(len(rolling.samples), MAX_ROLLING_SAMPLES)

    def test_buffer_reset_on_stop_and_start(self):
        """Calling stop() or starting a new session should reset the rolling audio buffer completely."""
        self.manager.start(window_seconds=15)
        samples = np.full(TARGET_SR * 3, fill_value=0.25, dtype=np.float32)
        self.manager.append_audio_samples(samples, TARGET_SR)
        self.assertGreater(self.manager.get_rolling_buffer_duration(), 0.0)

        # Stop session
        self.manager.stop()
        self.assertEqual(self.manager.get_rolling_buffer_duration(), 0.0)
        self.assertEqual(len(self.manager._rolling_samples), 0)

        # Restart session
        self.manager.start(window_seconds=15)
        self.assertEqual(self.manager.get_rolling_buffer_duration(), 0.0)

    def test_duplicate_chunk_deduplication(self):
        """Identical audio chunks submitted consecutively should be caught and not duplicated."""
        self.manager.start(window_seconds=15)
        samples = np.linspace(-0.5, 0.5, TARGET_SR * 2, dtype=np.float32)

        # First submission
        self.manager.append_audio_samples(samples, TARGET_SR)
        dur1 = self.manager.get_rolling_buffer_duration()
        self.assertAlmostEqual(dur1, 2.0, places=2)

        # Duplicate submission of the exact same audio
        self.manager.append_audio_samples(samples, TARGET_SR)
        dur2 = self.manager.get_rolling_buffer_duration()

        # Duration should remain 2.0s, not 4.0s
        self.assertAlmostEqual(dur2, 2.0, places=2)

    def test_not_running_behavior(self):
        """Appending samples when session is not running returns single-chunk LoadedAudio without accumulating."""
        samples = np.full(TARGET_SR * 2, fill_value=0.1, dtype=np.float32)
        rolling = self.manager.append_audio_samples(samples, TARGET_SR)
        self.assertEqual(rolling.duration_seconds, 2.0)
        self.assertEqual(self.manager.get_rolling_buffer_duration(), 0.0)

    def test_segment_result_submission_and_summary(self):
        """Submitting SegmentResults updates session count and aggregate metrics correctly."""
        self.manager.start(window_seconds=15)

        seg1 = SegmentResult(
            segment=1,
            start_seconds=0.0,
            end_seconds=2.0,
            verdict="Truthful",
            score=0.35,
            confidence=0.65,
            explanation=["Normal speech tempo."],
        )
        self.manager.submit_segment(seg1)

        results = self.manager.get_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["verdict"], "Truthful")

        summary = self.manager.get_session_summary()
        self.assertEqual(summary["total_segments"], 1)
        self.assertAlmostEqual(summary["avg_score"], 0.35, places=2)

        self.manager.stop()


if __name__ == "__main__":
    unittest.main()
