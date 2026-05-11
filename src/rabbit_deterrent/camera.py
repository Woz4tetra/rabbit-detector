from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._cam = None

    def capture(self) -> np.ndarray:
        """One-shot still capture for SCANNING mode (opens and closes the camera each call)."""
        from picamera2 import Picamera2

        cam = Picamera2()
        # OV5647 via Picamera2 returns RGB despite BGR888 being requested; use RGB888
        # explicitly and convert to BGR so all downstream code (cv2, detector) is consistent.
        config = cam.create_still_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        cam.stop()
        cam.close()
        logger.debug("Captured frame %dx%d", frame.shape[1], frame.shape[0])
        return frame

    def start_stream(self, frame_rate: float) -> None:
        """Open camera in continuous video mode for ALERT state recording."""
        from picamera2 import Picamera2

        self._cam = Picamera2()
        frame_duration_us = int(1_000_000 / frame_rate)
        config = self._cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        )
        self._cam.configure(config)
        self._cam.start()
        logger.debug("Camera stream started at %.1f fps", frame_rate)

    def capture_frame(self) -> np.ndarray:
        """Capture the next frame from the active stream."""
        return cv2.cvtColor(self._cam.capture_array(), cv2.COLOR_RGB2BGR)

    def stop_stream(self) -> None:
        """Stop and close the camera stream."""
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None
            logger.debug("Camera stream stopped")
