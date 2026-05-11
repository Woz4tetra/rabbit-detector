from __future__ import annotations

import json
import logging
import threading
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
CLEAR_THRESHOLD = 3  # consecutive clear inference results before leaving ALERT


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
        self._storage = config.storage
        self._image_dir = config.storage.resolved_image_dir()
        self._video_dir = config.storage.resolved_video_dir()
        self._video_writer = None
        self._video_thread: threading.Thread | None = None
        self._video_stop = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._state, self._clear_streak = self._load_state()

    def run(self) -> None:
        if self._state == State.ALERT:
            self._start_alert_recording()

        while True:
            if self._state == State.SCANNING:
                frame = self._camera.capture()
                detections = self._detector.detect(frame)
                rabbit_present = len(detections) > 0

                self._save_frame(frame, rabbit_present)

                if rabbit_present:
                    best = max(detections, key=lambda d: d.confidence)
                    logger.info("Rabbit detected (conf=%.2f), entering ALERT", best.confidence)
                    self._handle_detection(frame, best)
                    self._clear_streak = 0
                    self._state = State.ALERT
                    self._start_alert_recording()
                else:
                    logger.debug("No rabbit detected")

                self._save_state()
                self._log_event(rabbit_present, detections)
                time.sleep(SCANNING_INTERVAL)

            else:  # ALERT
                frame = self._get_latest_alert_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                detections = self._detector.detect(frame)
                rabbit_present = len(detections) > 0

                self._save_frame(frame, rabbit_present)

                if rabbit_present:
                    best = max(detections, key=lambda d: d.confidence)
                    logger.info("Rabbit detected (conf=%.2f) in ALERT", best.confidence)
                    self._handle_detection(frame, best)
                    self._clear_streak = 0
                else:
                    self._clear_streak += 1
                    logger.info(
                        "No rabbit (clear streak %d/%d)", self._clear_streak, CLEAR_THRESHOLD
                    )
                    if self._clear_streak >= CLEAR_THRESHOLD:
                        logger.info("Rabbit gone, returning to SCANNING")
                        self._stop_alert_recording()
                        self._state = State.SCANNING
                        self._clear_streak = 0

                self._save_state()
                self._log_event(rabbit_present, detections)

    def _handle_detection(self, frame: np.ndarray, detection: Detection) -> None:
        self._audio.play()
        self._notifier.send(
            subject="Rabbit detected in garden!",
            body=f"Confidence: {detection.confidence:.0%}\nBounding box: ({detection.x1:.0f},{detection.y1:.0f}) -> ({detection.x2:.0f},{detection.y2:.0f})",
            image=frame,
        )

    # --- Alert recording (streaming camera + video writer thread) ---

    def _start_alert_recording(self) -> None:
        if self._video_thread is not None and self._video_thread.is_alive():
            return
        self._camera.start_stream(self._config.detection.frame_rate)
        self._video_stop.clear()
        self._latest_frame = None
        self._video_thread = threading.Thread(
            target=self._alert_recording_loop, daemon=True, name="video-capture"
        )
        self._video_thread.start()

    def _alert_recording_loop(self) -> None:
        while not self._video_stop.is_set():
            frame = self._camera.capture_frame()
            with self._frame_lock:
                self._latest_frame = frame
            self._write_video_frame(frame)

    def _get_latest_alert_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _stop_alert_recording(self) -> None:
        self._video_stop.set()
        if self._video_thread is not None:
            self._video_thread.join(timeout=5.0)
            self._video_thread = None
        self._camera.stop_stream()
        self._stop_video()

    # --- Image saving ---

    def _save_frame(self, frame: np.ndarray, rabbit_present: bool) -> None:
        if not self._storage.save_images:
            return
        import cv2
        import datetime

        self._image_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        tag = "rabbit" if rabbit_present else "clear"
        path = self._image_dir / f"{ts}_{tag}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._prune_images()

    def _prune_images(self) -> None:
        max_images = self._storage.max_images
        if max_images <= 0:
            return
        images = sorted(self._image_dir.glob("*.jpg"))
        for old in images[:-max_images]:
            old.unlink(missing_ok=True)

    # --- Video writing (called from recording thread) ---

    def _write_video_frame(self, frame: np.ndarray) -> None:
        if not self._storage.save_video:
            return
        import cv2
        import datetime

        if self._video_writer is None:
            self._video_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = self._video_dir / f"{ts}_alert.avi"
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._video_writer = cv2.VideoWriter(
                str(path), fourcc, self._config.detection.frame_rate, (w, h)
            )
            logger.info("Started recording video: %s", path)
        self._video_writer.write(frame)

    def _stop_video(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            logger.info("Stopped recording video")
            self._video_writer = None

    # --- State persistence ---

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
