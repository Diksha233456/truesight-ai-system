"""
Video source abstraction for webcam, video files, and RTSP streams.
Provides a unified interface for frame acquisition.
"""

import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Union

from facial_cues.config.schema import CaptureConfig

logger = logging.getLogger(__name__)


class VideoSource:
    """
    Unified video source that handles:
    - Local webcam (source=0, 1, ...) with background threading for full FPS
    - Video files (source="path/to/video.mp4")
    - RTSP/HTTP streams (source="rtsp://...")

    Usage:
        source = VideoSource(config)
        source.open()
        while True:
            ret, frame, timestamp = source.read()
            if not ret:
                break
        source.release()
    """

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._is_file: bool = False
        self._thread: Optional[threading.Thread] = None
        self._thread_running: bool = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ret = False
        self._latest_ts = 0.0

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        source = self.config.source

        # Determine source type
        if isinstance(source, int):
            # Webcam index
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # DirectShow on Windows
            self._is_file = False
            logger.info(f"Opening webcam index {source}")
        elif isinstance(source, str):
            if source.startswith(("rtsp://", "http://", "https://")):
                # Network stream
                self._cap = cv2.VideoCapture(source)
                self._is_file = False
                logger.info(f"Opening network stream: {source}")
            else:
                # Video file
                self._cap = cv2.VideoCapture(source)
                self._is_file = True
                logger.info(f"Opening video file: {source}")
        else:
            logger.error(f"Invalid source type: {type(source)}")
            return False

        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return False

        # Configure capture properties for live sources
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        # Read actual properties
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video source opened: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

        self._frame_count = 0
        self._start_time = time.perf_counter()

        # Start non-blocking capture thread for live webcams
        if not self._is_file:
            self._thread_running = True
            self._thread = threading.Thread(target=self._capture_worker, name="videosource-worker", daemon=True)
            self._thread.start()

        return True

    def _capture_worker(self):
        """Continuously reads frames in a background thread to prevent camera hardware sync blocking."""
        while self._thread_running and self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            ts = (time.perf_counter() - self._start_time) * 1000.0 if self._start_time > 0 else 0.0
            with self._lock:
                self._latest_ret = ret
                self._latest_frame = frame
                self._latest_ts = ts
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional["numpy.ndarray"], float]:
        """
        Read the next frame.

        Returns:
            (success, frame_bgr, timestamp_ms)
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None, 0.0

        if not self._is_file:
            # For live webcam, fetch the latest frame captured by the worker thread
            with self._lock:
                ret = self._latest_ret
                frame = self._latest_frame.copy() if (ret and self._latest_frame is not None) else None
                ts = self._latest_ts

            if not ret or frame is None:
                # Fallback to direct read if worker hasn't grabbed a frame yet
                ret, frame = self._cap.read()
                ts = (time.perf_counter() - self._start_time) * 1000.0

            if ret and frame is not None:
                self._frame_count += 1
                return True, frame, ts
            return False, None, 0.0

        # For video files, read sequentially
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None, 0.0

        timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        self._frame_count += 1
        return True, frame, timestamp_ms

    def release(self):
        """Release the video source."""
        self._thread_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"Video source released after {self._frame_count} frames")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        """Get the actual FPS of the source."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return self.config.fps

    @property
    def total_frames(self) -> Optional[int]:
        """Total frames in the video (only for files)."""
        if self._is_file and self._cap is not None:
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return total if total > 0 else None
        return None

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of frames."""
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self.config.width, self.config.height)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()
