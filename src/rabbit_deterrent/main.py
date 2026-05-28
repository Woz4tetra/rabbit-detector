from __future__ import annotations

import argparse
import logging

from .audio import AudioPlayer
from .camera import CameraCapture
from .camera_client import CameraClient
from .config import load_config
from .hotspot import wait_for_network
from .logger_setup import setup_logging
from .power import apply_power_optimizations

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rabbit deterrent camera client")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    config = load_config(args.config)
    logger.info("Starting rabbit deterrent camera client")
    logger.info("Server: %s", config.server.url)

    apply_power_optimizations()

    if config.hotspot.enabled:
        wait_for_network(
            timeout=config.hotspot.timeout_seconds,
            ssid=config.hotspot.ssid,
            password=config.hotspot.password,
        )

    camera = CameraCapture(
        width=config.detection.capture_width,
        height=config.detection.capture_height,
        camera_config=config.camera,
    )
    audio = AudioPlayer(
        sounds_dir=config.audio.resolved_sounds_dir(),
        volume=config.audio.volume,
    )

    client = CameraClient(config=config, camera=camera, audio=audio)
    client.run()


if __name__ == "__main__":
    main()
