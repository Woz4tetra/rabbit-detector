from __future__ import annotations

import argparse
import logging

from .audio import AudioPlayer
from .camera import CameraCapture
from .config import load_config
from .detector import OnnxRabbitDetector
from .logger_setup import setup_logging
from .notifier import EmailNotifier
from .power import apply_power_optimizations
from .state_machine import DetectionStateMachine

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rabbit deterrent detection loop")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    config = load_config(args.config)
    logger.info("Starting rabbit deterrent system")

    apply_power_optimizations()

    camera = CameraCapture(
        width=config.detection.capture_width,
        height=config.detection.capture_height,
    )
    detector = OnnxRabbitDetector(
        model_path=config.detection.resolved_model_path(),
        confidence_threshold=config.detection.confidence_threshold,
        image_size=config.detection.image_size,
    )
    audio = AudioPlayer(
        sounds_dir=config.audio.resolved_sounds_dir(),
        volume=config.audio.volume,
    )
    notifier = EmailNotifier(config.email)

    sm = DetectionStateMachine(
        config=config,
        camera=camera,
        detector=detector,
        audio=audio,
        notifier=notifier,
        log_dir=config.resolved_log_dir(),
    )
    sm.run()


if __name__ == "__main__":
    main()
