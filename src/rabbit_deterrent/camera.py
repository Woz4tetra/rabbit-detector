from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def capture(self) -> np.ndarray:
        from picamera2 import Picamera2

        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        cam.configure(config)
        cam.start()
        frame = cam.capture_array()
        cam.stop()
        cam.close()
        logger.debug("Captured frame %dx%d", frame.shape[1], frame.shape[0])
        return frame
