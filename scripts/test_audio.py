"""Play the deterrent sound via the USB speaker and wait for it to finish."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SOUND_FILE = Path(__file__).parent.parent / "data" / "sounds" / "deterrent.wav"

if not SOUND_FILE.exists():
    print(f"ERROR: Sound file not found at {SOUND_FILE}")
    print("Place a deterrent.wav file in data/sounds/ and try again.")
    sys.exit(1)

from rabbit_deterrent.audio import AudioPlayer

player = AudioPlayer(sound_path=SOUND_FILE, volume=0.9)
print(f"Playing {SOUND_FILE.name} — you should hear the sound now...")
player.play()
time.sleep(5)
print("Done.")
