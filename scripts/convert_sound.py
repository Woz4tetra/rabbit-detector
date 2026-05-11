"""Convert audio files to WAV format compatible with AudioPlayer (44100 Hz, 16-bit, mono).

Usage:
    python scripts/convert_sound.py <file> [<file> ...]
    python scripts/convert_sound.py <file> [<file> ...] --out-dir data/sounds
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install it with:")
        print("  sudo apt install ffmpeg")
        sys.exit(1)


def convert(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + ".wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ar", "44100",   # sample rate matches pygame mixer
            "-ac", "1",       # mono
            "-sample_fmt", "s16",  # 16-bit signed
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR converting {src.name}:")
        print(result.stderr)
        sys.exit(1)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert audio files to WAV for use as deterrent sounds")
    parser.add_argument("files", nargs="+", help="Input audio files")
    parser.add_argument("--out-dir", default="data/sounds", help="Output directory (default: data/sounds)")
    args = parser.parse_args()

    check_ffmpeg()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    for path_str in args.files:
        src = Path(path_str)
        if not src.exists():
            print(f"ERROR: {src} not found")
            sys.exit(1)
        dst = convert(src, out_dir)
        print(f"{src.name} -> {dst}")


if __name__ == "__main__":
    main()
