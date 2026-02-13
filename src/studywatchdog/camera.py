"""Webcam capture and frame management via OpenCV.

Handles camera device selection, frame capture, and resource cleanup.
"""

import logging
import time

import cv2
import numpy as np

from studywatchdog.config import CameraConfig

logger = logging.getLogger(__name__)


def list_cameras(max_index: int = 10) -> list[int]:
    """Probe available camera device indices.

    Args:
        max_index: Maximum device index to check.

    Returns:
        List of valid camera indices.
    """
    available: list[int] = []
    # Suppress noisy V4L2 and FFMPEG warnings during probe
    prev_level = cv2.getLogLevel()
    cv2.setLogLevel(0)  # LOG_LEVEL_SILENT
    try:
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
    finally:
        cv2.setLogLevel(prev_level)
    return available


class Camera:
    """Webcam capture wrapper around OpenCV VideoCapture.

    Args:
        config: Camera configuration settings.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._last_capture_time: float = 0.0

    def open(self) -> None:
        """Open the camera device.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        logger.info("Opening camera %d ...", self._config.camera_index)
        self._cap = cv2.VideoCapture(self._config.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self._config.camera_index}. "
                f"Available cameras: {list_cameras()}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

        # Read actual resolution (camera may not support requested size)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera %d opened at %dx%d", self._config.camera_index, actual_w, actual_h)

    def read_frame(self) -> np.ndarray | None:
        """Read a single frame from the camera.

        Returns:
            BGR frame as numpy array, or None if read failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera %d", self._config.camera_index)
            return None
        return frame

    def should_capture(self) -> bool:
        """Check if enough time has passed since the last capture.

        Returns:
            True if a new frame should be captured for analysis.
        """
        now = time.monotonic()
        if now - self._last_capture_time >= self._config.capture_interval:
            self._last_capture_time = now
            return True
        return False

    def close(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera %d released", self._config.camera_index)

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
