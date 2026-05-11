"""Play each deterrent sound in data/sounds/ via the USB speaker."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SOUNDS_DIR = Path(__file__).parent.parent / "data" / "sounds"

wav_files = sorted(SOUNDS_DIR.glob("*.wav"))
if not wav_files:
    print(f"ERROR: No .wav files found in {SOUNDS_DIR}")
    print("Add .wav files to data/sounds/ and try again.")
    sys.exit(1)

import pygame

pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

print(f"Found {len(wav_files)} sound(s) in {SOUNDS_DIR}:")
for f in wav_files:
    print(f"  {f.name}")
print()

for f in wav_files:
    print(f"Playing {f.name}...")
    sound = pygame.mixer.Sound(str(f))
    sound.set_volume(0.9)
    sound.play()
    time.sleep(5)

print("Done.")
