from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_initialized = False


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        import pygame

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        _initialized = True


class AudioPlayer:
    def __init__(self, sound_path: Path, volume: float) -> None:
        import pygame

        _ensure_init()
        logger.info("Loading sound from %s", sound_path)
        self._sound = pygame.mixer.Sound(str(sound_path))
        self._sound.set_volume(max(0.0, min(1.0, volume)))

    def play(self) -> None:
        self._sound.play()
        logger.info("Playing deterrent sound")
