from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

_initialized = False
_init_failed = False


def _ensure_init() -> None:
    global _initialized, _init_failed
    if _initialized or _init_failed:
        return
    import pygame

    try:
        # USB speaker (USB2.0 Device) only supports 48000 Hz stereo.
        # Matching hardware exactly avoids ALSA resampling, which causes crackling.
        pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=2048)
        _initialized = True
    except pygame.error as exc:
        logger.warning("Audio init failed (non-fatal): %s", exc)
        _init_failed = True


class AudioPlayer:
    def __init__(self, sounds_dir: Path, volume: float) -> None:
        import pygame

        _ensure_init()
        volume = max(0.0, min(1.0, volume))
        self._sounds: list[pygame.mixer.Sound] = []
        if _init_failed:
            logger.warning("Audio unavailable — sound playback disabled")
            return
        files = sorted(sounds_dir.glob("*.wav"))
        if not files:
            logger.warning("No .wav files found in %s — audio disabled", sounds_dir)
        else:
            logger.info("Loaded %d sound(s) from %s", len(files), sounds_dir)
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
