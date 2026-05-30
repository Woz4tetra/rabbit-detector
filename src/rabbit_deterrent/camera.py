from __future__ import annotations

import logging
import time
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

# Auto-exposure railing is the reliable day→night trigger. Lux alone fails on the
# Arducam day-night camera: at night its IR LEDs flood the scene and the IR-cut
# filter opens, so day-profile metering reports Lux in the 5–50 dead zone and
# never crosses LUX_NIGHT_THRESHOLD — the camera stays stuck in day exposure until
# a restart forces profile="night" from a cold libcamera. When day-profile AE
# instead pushes ExposureTime to its ceiling, the scene is genuinely dark no
# matter what Lux claims, so we switch on that. In daylight (even heavy overcast)
# AE picks a few ms, nowhere near the 100ms ceiling, so this won't false-trigger.
DAY_AE_RAIL_FRACTION = 0.9   # ExposureTime >= this * day ceiling means AE ran out of light
DAY_TO_NIGHT_FRAMES = 3      # consecutive dark/railed frames before switching to night

STATUS_LOG_INTERVAL_S = 60.0  # throttle for the INFO heartbeat that logs lux/exposure/gain

# Dark-frame recovery. A wedged sensor or manual exposure controls that haven't
# settled can return near-black night frames even though capture() reopens the
# device on every call. After several consecutive dark night frames we reopen
# with extra settle frames and a short delay, giving AeEnable/ExposureTime/
# AnalogueGain time to actually take effect — the single settle frame on the
# normal path is not always enough right after start(), which is the soft-wedge
# that produced the all-black night image.
DARK_MEAN_THRESHOLD = 12.0       # mean 8-bit pixel value below this is effectively black
DARK_FRAMES_BEFORE_RECOVERY = 3  # consecutive dark night frames before forcing recovery
NORMAL_SETTLE_FRAMES = 1         # discard frames before the real shot in steady state
RECOVERY_SETTLE_FRAMES = 6       # discard frames on a recovery capture


class CameraCapture:
    def __init__(self, width: int, height: int, day_config: "CameraConfig", night_config: "CameraConfig") -> None:
        self.width = width
        self.height = height
        self._day_cfg = day_config
        self._night_cfg = night_config
        self._profile = "night"  # safer default; first capture's Lux flips us to day if it's actually bright
        self._cam = None
        self._consecutive_dark = 0   # consecutive near-black night frames
        self._dark_ae_ticks = 0      # consecutive day frames where AE has railed / Lux is low
        self._recover_next = False   # next capture() should reopen with extra settle frames
        self._last_status_log = 0.0  # monotonic time of the last lux/exposure heartbeat

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

    def _update_profile(self, lux: float, exposure_us: float, gain: float) -> None:
        """Pick the day/night profile for the NEXT capture from this frame's metadata.

        Night→day keys off Lux: daylight saturates the meter so Lux > 50 is
        unambiguous. Day→night cannot trust Lux alone (see DAY_AE_RAIL_FRACTION),
        so it also fires when day-profile AE has railed its exposure for several
        consecutive frames. Either way, a switch arms a recovery reopen so the new
        profile's controls actually apply on the next capture — the same thing a
        manual restart was doing by hand.
        """
        if self._profile == "night":
            if lux > LUX_DAY_THRESHOLD:
                logger.info("Lux=%.1f → switching to day profile", lux)
                self._profile = "day"
                self._dark_ae_ticks = 0
                self._recover_next = True
            return

        # Day profile: decide whether it has gotten dark.
        ceiling_us = self._day_cfg.max_exposure_seconds * 1_000_000
        ae_railed = exposure_us >= DAY_AE_RAIL_FRACTION * ceiling_us
        if lux < LUX_NIGHT_THRESHOLD or ae_railed:
            self._dark_ae_ticks += 1
            logger.info(
                "Dark day frame (Lux=%.1f, exp=%.0fus, gain=%.2f, %d/%d)",
                lux, exposure_us, gain, self._dark_ae_ticks, DAY_TO_NIGHT_FRAMES,
            )
            if self._dark_ae_ticks >= DAY_TO_NIGHT_FRAMES:
                logger.info("Lux=%.1f, exp=%.0fus → switching to night profile", lux, exposure_us)
                self._profile = "night"
                self._dark_ae_ticks = 0
                self._recover_next = True
        else:
            self._dark_ae_ticks = 0

    def _maybe_log_status(self, lux: float, exposure_us: float, gain: float, mean: float) -> None:
        """Throttled INFO heartbeat so the day/night transition is observable in journalctl."""
        now = time.monotonic()
        if now - self._last_status_log < STATUS_LOG_INTERVAL_S:
            return
        self._last_status_log = now
        logger.info(
            "Camera status: profile=%s lux=%.1f exp=%.0fus gain=%.2f mean=%.1f",
            self._profile, lux, exposure_us, gain, mean,
        )

    @property
    def profile(self) -> str:
        return self._profile

    def capture(self) -> np.ndarray:
        """One-shot still capture for SCANNING mode (opens and closes the camera each call)."""
        from picamera2 import Picamera2

        recovering = self._recover_next
        self._recover_next = False
        settle = RECOVERY_SETTLE_FRAMES if recovering else NORMAL_SETTLE_FRAMES

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
        if recovering:
            logger.warning("Reopening camera with %d settle frames (profile switch or dark-frame recovery)", settle)
            time.sleep(0.5)  # give the freshly applied manual controls a moment before discarding
        for _ in range(settle):
            cam.capture_array()  # discard: let AE / manual controls settle before the real shot
        req = cam.capture_request()
        frame = req.make_array("main")
        meta = req.get_metadata()
        req.release()
        cam.stop()
        cam.close()
        logger.debug("Captured frame %dx%d (profile=%s)", frame.shape[1], frame.shape[0], self._profile)
        lux = float(meta.get("Lux", 0.0))
        exposure_us = float(meta.get("ExposureTime", 0.0))
        gain = float(meta.get("AnalogueGain", 0.0))
        self._maybe_log_status(lux, exposure_us, gain, float(frame.mean()))
        # Darkness is judged against the profile used for THIS frame, before the
        # metadata below may flip the profile for the next capture.
        self._check_dark_frame(frame)
        # This frame's metadata drives the NEXT capture's profile. A saturated
        # night-profile frame in daylight reports a high Lux; a day-profile frame
        # whose AE has railed its exposure reports the scene is dark. Either one
        # correctly triggers a switch on the following tick.
        self._update_profile(lux, exposure_us, gain)
        return frame

    def _check_dark_frame(self, frame: np.ndarray) -> None:
        """Track consecutive near-black night frames and arm a recovery capture.

        Only night frames are checked: a dark day frame is normal dusk, not a
        fault. Recovery is the extra-settle-frame reopen in capture(); reopening
        alone (which capture() already does every call) does not clear the wedge,
        so the extra settle frames are what actually let the controls apply.
        """
        if self._profile != "night":
            self._consecutive_dark = 0
            return
        mean = float(frame.mean())
        if mean >= DARK_MEAN_THRESHOLD:
            self._consecutive_dark = 0
            return
        self._consecutive_dark += 1
        logger.warning("Dark night frame (mean=%.1f, %d consecutive)", mean, self._consecutive_dark)
        if self._consecutive_dark >= DARK_FRAMES_BEFORE_RECOVERY:
            self._recover_next = True
            self._consecutive_dark = 0

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
