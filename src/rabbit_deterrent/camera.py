from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(self, width: int, height: int, max_exposure_seconds: float = 3.0) -> None:
        self.width = width
        self.height = height
        self._max_exposure_us = int(max_exposure_seconds * 1_000_000)
        self._cam = None

    def capture(self) -> np.ndarray:
        """One-shot still capture for SCANNING mode (opens and closes the camera each call)."""
        from picamera2 import Picamera2

        cam = Picamera2()
        # Picamera2 "RGB888" on OV5647/Trixie yields BGR byte order — the format name
        # is misleading but the raw array is directly usable by cv2 without conversion.
        # FrameDurationLimits sets the AE ceiling. AE uses shorter exposures in daylight
        # automatically, so a high max (e.g. 3s) does not slow down daytime captures.
        config = cam.create_still_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, self._max_exposure_us)},
        )
        cam.configure(config)
        cam.start()
        cam.capture_array()  # discard: let AE settle before the real shot
        frame = cam.capture_array()
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
            main={"size": (self.width, self.height), "format": "RGB888"},  # yields BGR bytes, see capture()
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        )
        self._cam.configure(config)
        self._cam.start()
        logger.debug("Camera stream started at %.1f fps", frame_rate)

    def capture_frame(self) -> np.ndarray:
        """Capture the next frame from the active stream."""
        return self._cam.capture_array()

    def stop_stream(self) -> None:
        """Stop and close the camera stream."""
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None
            logger.debug("Camera stream stopped")
