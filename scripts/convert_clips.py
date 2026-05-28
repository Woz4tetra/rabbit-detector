#!/usr/bin/env python3
"""Convert AVI clips to MP4. Accepts one or more AVI files or directories."""

import argparse
import subprocess
import sys
from pathlib import Path


def convert(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        print(f"skip {dst} (already exists, use --overwrite)")
        return False
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(src),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"error converting {src}:\n{result.stderr}", file=sys.stderr)
        return False
    print(f"ok {src} -> {dst}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert AVI clips to MP4")
    parser.add_argument("inputs", nargs="+", help="AVI files or directories containing AVIs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP4s")
    parser.add_argument("--delete-source", action="store_true", help="Delete source AVI after successful conversion")
    args = parser.parse_args()

    paths: list[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.avi")))
        elif p.suffix.lower() == ".avi":
            paths.append(p)
        else:
            print(f"skip {p} (not an AVI)", file=sys.stderr)

    if not paths:
        print("no AVI files found", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for src in paths:
        dst = src.with_suffix(".mp4")
        if convert(src, dst, args.overwrite):
            ok += 1
            if args.delete_source:
                src.unlink()
                print(f"deleted {src}")

    print(f"\n{ok}/{len(paths)} converted")


if __name__ == "__main__":
    main()
