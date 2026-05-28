#!/usr/bin/env python3
"""
Scan existing frames in data/server-frames/ and generate data/detections.jsonl
for all frames where the model detects a rabbit.

Run on the A6000 cluster (uses cuda:1 to avoid conflicting with the running server on cuda:0).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Generate detections.jsonl from existing frames")
    parser.add_argument("--device", default="cuda:1", help="Torch device (default: cuda:1)")
    parser.add_argument("--frames-dir", default="data/server-frames", help="Frames directory relative to project root")
    parser.add_argument("--output", default="data/detections.jsonl", help="Output JSONL file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    frames_dir = PROJECT_ROOT / args.frames_dir
    output_path = PROJECT_ROOT / args.output

    skip = {"latest.jpg", "last_detection.jpg"}
    frames = sorted(f for f in frames_dir.glob("*.jpg") if f.name not in skip)
    if not frames:
        print("No frames found in", frames_dir)
        sys.exit(1)

    if output_path.exists() and not args.overwrite:
        print(f"{output_path} already exists. Use --overwrite to replace it.")
        sys.exit(1)

    print(f"Found {len(frames)} frames. Loading model on {args.device}…")

    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from server.moondream_loader import load_moondream
    from PIL import Image

    model, tokenizer = load_moondream(device=args.device)
    prompt = "Is there a rabbit in this image? Reply with only 'yes' or 'no'."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    with open(output_path, "w") as out:
        for i, frame_path in enumerate(frames):
            # Parse timestamp from filename: 20260527T091818Z.jpg -> ISO
            stem = frame_path.stem  # e.g. 20260527T091818Z
            try:
                from datetime import datetime, timezone
                dt = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                ts = dt.isoformat().replace("+00:00", "Z")
            except ValueError:
                ts = stem

            try:
                image = Image.open(frame_path).convert("RGB")
                enc = model.encode_image(image)
                raw: str = model.query(enc, prompt)["answer"]
            except Exception as e:
                print(f"  error on {frame_path.name}: {e}", file=sys.stderr)
                continue

            cleaned = raw.strip().lower().rstrip(".,!? \t\n")
            rabbit = cleaned.startswith("yes")

            if rabbit:
                record = {
                    "timestamp": ts,
                    "rabbit": True,
                    "confidence": 1.0,
                    "raw_response": raw.strip(),
                    "frame": frame_path.name,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                found += 1
                print(f"  [{i+1}/{len(frames)}] RABBIT: {frame_path.name}")
            elif (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(frames)}] {found} rabbits so far…")

    print(f"\nDone. {found} rabbit detections written to {output_path}")


if __name__ == "__main__":
    main()
