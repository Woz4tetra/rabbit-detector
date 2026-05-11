from __future__ import annotations

import logging
import random
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
    def __init__(self, sounds_dir: Path, volume: float) -> None:
        import pygame

        _ensure_init()
        volume = max(0.0, min(1.0, volume))
        files = sorted(sounds_dir.glob("*.wav"))
        if not files:
            logger.warning("No .wav files found in %s — audio disabled", sounds_dir)
        else:
            logger.info("Loaded %d sound(s) from %s", len(files), sounds_dir)
        self._sounds: list[pygame.mixer.Sound] = []
        for f in files:
            sound = pygame.mixer.Sound(str(f))
            sound.set_volume(volume)
            self._sounds.append(sound)

    def play(self) -> None:
        if not self._sounds:
            return
        sound = random.choice(self._sounds)
        sound.play()
        logger.info("Playing deterrent sound")
