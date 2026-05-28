from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import CameraConfig

logger = logging.getLogger(__name__)

# Hysteresis thresholds for day/night profile switching. Picamera2 reports Lux
# from sensor metering on every capture. The gap (5–50) prevents oscillation
# during dawn/dusk transitions. Values are deliberately conservative: switching
# to "day" too eagerly at dawn would starve the IR LEDs of long-shutter time.
LUX_DAY_THRESHOLD = 50.0   # in night profile: switch to day if Lux > this
LUX_NIGHT_THRESHOLD = 5.0  # in day profile: switch to night if Lux < this


class CameraCapture:
    def __init__(self, width: int, height: int, day_config: "CameraConfig", night_config: "CameraConfig") -> None:
        self.width = width
        self.height = height
        self._day_cfg = day_config
        self._night_cfg = night_config
        self._profile = "night"  # safer default; first capture's Lux flips us to day if it's actually bright
        self._cam = None

    def _active_cfg(self) -> "CameraConfig":
        return self._day_cfg if self._profile == "day" else self._night_cfg

    def _max_exposure_us(self) -> int:
        return int(self._active_cfg().max_exposure_seconds * 1_000_000)

    def _extra_controls(self) -> dict:
        c = self._active_cfg()
        ctrl: dict = {
            "AeEnable": c.ae_enable,
            "AwbEnable": c.awb_enable,
            "AwbMode": c.awb_mode,
            "Brightness": c.brightness,
            "Contrast": c.contrast,
            "Saturation": c.saturation,
            "Sharpness": c.sharpness,
            "NoiseReductionMode": c.noise_reduction_mode,
        }
        if not c.ae_enable:
            ctrl["ExposureTime"] = c.exposure_time_us
            ctrl["AnalogueGain"] = c.analogue_gain
        if not c.awb_enable:
            ctrl["ColourGains"] = (c.red_gain, c.blue_gain)
        return ctrl

    def _update_profile_from_lux(self, lux: float) -> None:
        if self._profile == "night" and lux > LUX_DAY_THRESHOLD:
            logger.info("Lux=%.1f → switching to day profile", lux)
            self._profile = "day"
        elif self._profile == "day" and lux < LUX_NIGHT_THRESHOLD:
            logger.info("Lux=%.1f → switching to night profile", lux)
            self._profile = "night"

    @property
    def profile(self) -> str:
        return self._profile

    def capture(self) -> np.ndarray:
        """One-shot still capture for SCANNING mode (opens and closes the camera each call)."""
        from picamera2 import Picamera2

        cam = Picamera2()
        # Picamera2 "RGB888" on OV5647/Trixie yields BGR byte order — the format name
        # is misleading but the raw array is directly usable by cv2 without conversion.
        config = cam.create_still_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, self._max_exposure_us())},
        )
        cam.configure(config)
        cam.start()
        extra = self._extra_controls()
        if extra:
            cam.set_controls(extra)
        cam.capture_array()  # discard: let AE settle before the real shot
        req = cam.capture_request()
        frame = req.make_array("main")
        meta = req.get_metadata()
        req.release()
        cam.stop()
        cam.close()
        logger.debug("Captured frame %dx%d (profile=%s)", frame.shape[1], frame.shape[0], self._profile)
        # Lux from current frame drives the NEXT capture's profile. A saturated
        # night-profile frame in daylight still reports a high Lux, which correctly
        # triggers the switch on the following tick.
        self._update_profile_from_lux(float(meta.get("Lux", 0.0)))
        return frame

    def start_stream(self, frame_rate: float) -> None:
        """Open camera in continuous video mode for ALERT state recording.

        Profile selection is fixed for the lifetime of the stream — ALERT bursts
        are short-lived and we don't want to reconfigure the camera mid-burst.
        """
        from picamera2 import Picamera2

        self._cam = Picamera2()
        frame_duration_us = int(1_000_000 / frame_rate)
        config = self._cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},  # yields BGR bytes, see capture()
            controls={"FrameDurationLimits": (frame_duration_us, self._max_exposure_us())},
        )
        self._cam.configure(config)
        self._cam.start()
        extra = self._extra_controls()
        if extra:
            self._cam.set_controls(extra)
        logger.debug("Camera stream started at %.1f fps (profile=%s)", frame_rate, self._profile)

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
