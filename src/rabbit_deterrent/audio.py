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


def _load_sound(path: Path, gain: float) -> "pygame.mixer.Sound":
    """Load a WAV and apply gain. gain > 1.0 amplifies by multiplying samples and
    clipping to int16 range — intentional hard saturation for maximum loudness."""
    import pygame

    if gain <= 1.0:
        sound = pygame.mixer.Sound(str(path))
        sound.set_volume(gain)
        return sound

    import wave
    import numpy as np

    with wave.open(str(path)) as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        # Non-16-bit file: can't amplify in-place, use hardware volume at max
        logger.warning("%s is not 16-bit; capping volume at 1.0", path.name)
        sound = pygame.mixer.Sound(str(path))
        sound.set_volume(1.0)
        return sound

    samples = np.frombuffer(raw, dtype=np.int16).copy()
    samples = np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    return pygame.sndarray.make_sound(samples)


class AudioPlayer:
    def __init__(self, sounds_dir: Path, volume: float) -> None:
        _ensure_init()
        self._sounds: list = []
        if _init_failed:
            logger.warning("Audio unavailable — sound playback disabled")
            return
        files = sorted(sounds_dir.glob("*.wav"))
        if not files:
            logger.warning("No .wav files found in %s — audio disabled", sounds_dir)
        else:
            logger.info("Loaded %d sound(s) from %s (gain=%.1fx)", len(files), sounds_dir, volume)
        for f in files:
            self._sounds.append(_load_sound(f, volume))

    def play(self) -> None:
        if not self._sounds:
            return
        random.choice(self._sounds).play()
        logger.info("Playing deterrent sound")
