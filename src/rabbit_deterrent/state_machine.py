from __future__ import annotations

import json
import logging
import time
from enum import Enum
from pathlib import Path

import numpy as np

from .audio import AudioPlayer
from .camera import CameraCapture
from .config import Config
from .detector import Detection, OnnxRabbitDetector
from .notifier import EmailNotifier

logger = logging.getLogger(__name__)

STATE_FILE = Path("/var/lib/rabbit-deterrent/state.json")
SCANNING_INTERVAL = 30.0
ALERT_INTERVAL = 5.0
CLEAR_THRESHOLD = 3  # consecutive clear frames before leaving ALERT


class State(str, Enum):
    SCANNING = "scanning"
    ALERT = "alert"


class DetectionStateMachine:
    def __init__(
        self,
        config: Config,
        camera: CameraCapture,
        detector: OnnxRabbitDetector,
        audio: AudioPlayer,
        notifier: EmailNotifier,
        log_dir: Path,
    ) -> None:
        self._config = config
        self._camera = camera
        self._detector = detector
        self._audio = audio
        self._notifier = notifier
        self._log_dir = log_dir
        self._state, self._clear_streak = self._load_state()

    def run(self) -> None:
        while True:
            frame = self._camera.capture()
            detections = self._detector.detect(frame)
            rabbit_present = len(detections) > 0

            if rabbit_present:
                best = max(detections, key=lambda d: d.confidence)
                logger.info(
                    "Rabbit detected (conf=%.2f) in state=%s", best.confidence, self._state
                )
                self._clear_streak = 0
                self._handle_detection(frame, best)
                self._state = State.ALERT
            else:
                if self._state == State.ALERT:
                    self._clear_streak += 1
                    logger.info(
                        "No rabbit (clear streak %d/%d)", self._clear_streak, CLEAR_THRESHOLD
                    )
                    if self._clear_streak >= CLEAR_THRESHOLD:
                        logger.info("Rabbit gone, returning to SCANNING")
                        self._state = State.SCANNING
                        self._clear_streak = 0
                else:
                    logger.debug("No rabbit detected")

            self._save_state()
            self._log_event(rabbit_present, detections)

            interval = ALERT_INTERVAL if self._state == State.ALERT else SCANNING_INTERVAL
            time.sleep(interval)

    def _handle_detection(self, frame: np.ndarray, detection: Detection) -> None:
        self._audio.play()
        self._notifier.send(
            subject="Rabbit detected in garden!",
            body=f"Confidence: {detection.confidence:.0%}\nBounding box: ({detection.x1:.0f},{detection.y1:.0f}) -> ({detection.x2:.0f},{detection.y2:.0f})",
            image=frame,
        )

    def _load_state(self) -> tuple[State, int]:
        try:
            data = json.loads(STATE_FILE.read_text())
            return State(data.get("state", State.SCANNING)), int(data.get("clear_streak", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return State.SCANNING, 0

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps({"state": self._state.value, "clear_streak": self._clear_streak})
        )

    def _log_event(self, rabbit_present: bool, detections: list[Detection]) -> None:
        if not self._config.log_detections:
            return
        import datetime

        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "state": self._state.value,
            "rabbit_present": rabbit_present,
            "detections": [
                {
                    "confidence": d.confidence,
                    "bbox": [d.x1, d.y1, d.x2, d.y2],
                }
                for d in detections
            ],
        }
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self._log_dir / "detections.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
