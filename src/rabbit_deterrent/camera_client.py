from __future__ import annotations

import collections
import datetime
import json
import logging
import time
from enum import Enum
from pathlib import Path

import numpy as np
import requests

from .audio import AudioPlayer
from .camera import CameraCapture
from .config import ClipConfig, Config, ServerConfig

logger = logging.getLogger(__name__)

STATE_FILE = Path("/var/lib/rabbit-deterrent/state.json")
CLEAR_THRESHOLD = 3  # consecutive clear responses before leaving ALERT


class State(str, Enum):
    SCANNING = "scanning"
    ALERT = "alert"
    OFFLINE = "offline"  # server unreachable; transient, not persisted


class ClipRecorder:
    """Buffers a pre-roll ring buffer and writes MJPEG AVI clips on demand."""

    def __init__(self, config: ClipConfig, fps: float) -> None:
        self._config = config
        self._fps = fps
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=config.pre_roll_frames)
        self._writer = None
        self._current_path: Path | None = None
        self._recording = False
        self._clip_start: float = 0.0

    def push_frame(self, frame: np.ndarray) -> None:
        """Always add frame to pre-roll buffer; write to disk only during recording."""
        self._buffer.append(frame)
        if self._recording:
            if self._writer is not None:
                self._writer.write(frame)
            if time.monotonic() - self._clip_start >= self._config.max_clip_seconds:
                logger.info("Clip reached max duration (%.0fs), stopping", self._config.max_clip_seconds)
                self.stop()

    def start(self, path: Path) -> None:
        if self._recording:
            return
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        if self._buffer:
            h, w = next(iter(self._buffer)).shape[:2]
        else:
            h, w = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
        for frame in self._buffer:
            self._writer.write(frame)
        self._current_path = path
        self._recording = True
        self._clip_start = time.monotonic()
        logger.info("Started clip: %s", path)

    def stop(self) -> Path | None:
        if not self._recording:
            return None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        path = self._current_path
        self._current_path = None
        self._recording = False
        if path:
            logger.info("Stopped clip: %s", path)
            self._prune()
        return path

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _prune(self) -> None:
        if self._config.max_clips <= 0:
            return
        clip_dir = self._config.resolved_clip_dir()
        clips = sorted(clip_dir.glob("*.avi"))
        for old in clips[: -self._config.max_clips]:
            old.unlink(missing_ok=True)


class ServerClient:
    """Thin HTTP wrapper around the Moondream2 detection server."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self._config.url}/health", timeout=self._config.timeout_seconds)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def detect(self, jpeg_bytes: bytes) -> dict | None:
        """POST a JPEG frame. Returns {"rabbit": bool, "confidence": float} or None on error."""
        try:
            r = requests.post(
                f"{self._config.url}/detect",
                files={"frame": ("frame.jpg", jpeg_bytes, "image/jpeg")},
                timeout=self._config.timeout_seconds,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.warning("Server request failed: %s", exc)
            return None

    @staticmethod
    def encode_frame(frame: np.ndarray) -> bytes:
        import cv2
        ok, buf = cv2.imencode(".jpg", _correct_ir_frame(frame), [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()


def _correct_ir_frame(frame: np.ndarray) -> np.ndarray:
    """Convert IR-mode frames to grayscale with contrast enhancement.

    When the IR-cut filter disengages at night the OV5647 sees only 850nm IR
    light from the built-in LEDs. AWB produces a strong blue cast because it
    assumes broadband illumination. Detecting IR mode by the blue/red ratio and
    converting to grayscale gives Moondream2 a cleaner image. CLAHE then boosts
    local contrast so detail is visible despite weak IR illumination.
    """
    import cv2
    b, r = frame[:, :, 0].mean(), frame[:, :, 2].mean()
    if r < 1 or b / r > 1.5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


class CameraClient:
    """Pi-side main loop: captures frames, queries server, plays audio, records clips."""

    def __init__(self, config: Config, camera: CameraCapture, audio: AudioPlayer) -> None:
        self._config = config
        self._camera = camera
        self._audio = audio
        self._server = ServerClient(config.server)
        self._clip = ClipRecorder(
            config=config.clip,
            fps=1.0 / max(config.detection.alert_poll_interval_seconds, 1.0),
        )
        self._state, self._clear_streak = self._load_state()
        self._failure_count = 0
        self._log_dir = config.resolved_log_dir()
        self._last_audio_time: float = 0.0

    def run(self) -> None:
        if not self._server.health_check():
            logger.warning("Server unreachable at startup — will retry during loop")

        if self._state == State.ALERT:
            logger.info("Resuming ALERT state from previous session")
            self._camera.start_stream(1.0 / self._config.detection.alert_poll_interval_seconds)

        try:
            while True:
                if self._state == State.OFFLINE:
                    self._run_offline_tick()
                elif self._state == State.SCANNING:
                    self._run_scanning_tick()
                else:
                    self._run_alert_tick()
        finally:
            self._clip.stop()
            self._camera.stop_stream()

    def _run_scanning_tick(self) -> None:
        frame = self._camera.capture()
        self._clip.push_frame(frame)
        jpeg = ServerClient.encode_frame(frame)
        result = self._server.detect(jpeg)

        if result is None:
            self._failure_count += 1
            logger.warning("Server failure %d/%d", self._failure_count, self._config.server.max_failures)
            if self._failure_count >= self._config.server.max_failures:
                logger.error("Too many failures — entering OFFLINE mode")
                self._state = State.OFFLINE
                return
        else:
            self._failure_count = 0
            rabbit = result.get("rabbit", False)
            confidence = result.get("confidence", 0.0)
            self._log_event(rabbit, confidence, result.get("raw_response", ""))
            if rabbit and confidence >= self._config.detection.confidence_threshold:
                logger.info("Rabbit detected (conf=%.2f), entering ALERT", confidence)
                self._state = State.ALERT
                self._clear_streak = 0
                self._save_state()
                self._play_audio_if_due()
                self._start_clip()
                self._camera.start_stream(1.0 / max(self._config.detection.alert_poll_interval_seconds, 1.0))
                return
            else:
                logger.info("No rabbit (conf=%.2f)", confidence)

        self._save_state()
        time.sleep(self._config.detection.scanning_interval_seconds)

    def _run_alert_tick(self) -> None:
        frame = self._camera.capture_frame()
        self._clip.push_frame(frame)
        jpeg = ServerClient.encode_frame(frame)
        result = self._server.detect(jpeg)

        if result is None:
            self._failure_count += 1
            logger.warning("Server failure %d/%d in ALERT", self._failure_count, self._config.server.max_failures)
            if self._failure_count >= self._config.server.max_failures:
                logger.error("Too many failures — entering OFFLINE mode")
                self._clip.stop()
                self._camera.stop_stream()
                self._state = State.OFFLINE
                return
        else:
            self._failure_count = 0
            rabbit = result.get("rabbit", False)
            confidence = result.get("confidence", 0.0)
            self._log_event(rabbit, confidence, result.get("raw_response", ""))

            if rabbit and confidence >= self._config.detection.confidence_threshold:
                self._clear_streak = 0
                logger.info("Rabbit still present (conf=%.2f)", confidence)
                self._play_audio_if_due()
            else:
                self._clear_streak += 1
                logger.info("Clear streak %d/%d", self._clear_streak, CLEAR_THRESHOLD)
                if self._clear_streak >= CLEAR_THRESHOLD:
                    logger.info("Rabbit gone, returning to SCANNING")
                    self._clip.stop()
                    self._camera.stop_stream()
                    self._state = State.SCANNING
                    self._clear_streak = 0

        self._save_state()
        time.sleep(self._config.detection.alert_poll_interval_seconds)

    def _run_offline_tick(self) -> None:
        frame = self._camera.capture()
        self._clip.push_frame(frame)
        logger.debug("OFFLINE: waiting for server")
        time.sleep(self._config.server.retry_delay_seconds)
        if self._server.health_check():
            logger.info("Server is back — resuming SCANNING")
            self._failure_count = 0
            self._state = State.SCANNING

    def _play_audio_if_due(self) -> None:
        interval = self._config.audio.alert_interval_seconds
        now = time.monotonic()
        if now - self._last_audio_time >= interval:
            self._audio.play()
            self._last_audio_time = now
            logger.info("Audio played (next due in %.0fs)", interval)
        else:
            remaining = interval - (now - self._last_audio_time)
            logger.debug("Audio suppressed (%.0fs until next play)", remaining)

    def _start_clip(self) -> None:
        if not self._config.clip.enabled:
            return
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self._config.clip.resolved_clip_dir() / f"{ts}_alert.avi"
        self._clip.start(path)

    def _log_event(self, rabbit: bool, confidence: float, raw_response: str) -> None:
        if not self._config.log_detections:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "state": self._state.value,
            "rabbit_present": rabbit,
            "confidence": confidence,
            "raw_response": raw_response,
        }
        with open(self._log_dir / "detections.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_state(self) -> tuple[State, int]:
        try:
            data = json.loads(STATE_FILE.read_text())
            state_val = data.get("state", State.SCANNING)
            if state_val == State.OFFLINE:
                state_val = State.SCANNING
            return State(state_val), int(data.get("clear_streak", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return State.SCANNING, 0

    def _save_state(self) -> None:
        persisted_state = self._state if self._state != State.OFFLINE else State.SCANNING
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps({"state": persisted_state.value, "clear_streak": self._clear_streak})
        )
