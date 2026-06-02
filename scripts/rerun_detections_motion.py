#!/usr/bin/env python3
"""Replay stored server frames through the motion-gated detect() pipeline.

Mirrors server._run_inference: running-average background subtraction picks the
changed regions, Moondream detect() decides whether each crop holds an animal.
Frames are processed in chronological order because the motion gate is stateful.

Run on a spare GPU so the live server (cuda:0) is undisturbed:

    server/.venv/bin/python scripts/rerun_detections_motion.py --device cuda:1

Outputs:
  - a fresh detections JSONL (default data/detections_motion.jsonl)
  - annotated copies of every frame the new pipeline flags that the existing
    (yes/no) log did not, so the additions can be eyeballed for false positives
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.motion import MotionDetector  # noqa: E402


def parse_ts(stem: str) -> str:
    try:
        dt = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return stem


def detect_count(model, image, objects) -> tuple[int, list[str]]:
    """Return (#boxes, [labels that fired]) for the given crop."""
    total = 0
    hits: list[str] = []
    for obj in objects:
        try:
            res = model.detect(image, obj)
        except Exception as e:  # noqa: BLE001
            print(f"  detect() failed for {obj!r}: {e}", file=sys.stderr)
            continue
        n = len(res.get("objects", []))
        if n:
            hits.append(obj)
        total += n
    return total, hits


def load_frames(set_of_frames: Path) -> set[str]:
    frames: set[str] = set()
    if not set_of_frames.exists():
        return frames
    for line in set_of_frames.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line).get("frame")
        except json.JSONDecodeError:
            continue
        if frame:
            frames.add(frame)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames-dir", default="data/server-frames")
    ap.add_argument("--output", default="data/detections_motion.jsonl")
    ap.add_argument(
        "--compare",
        default="data/detections.jsonl",
        help="Existing detection log to diff against",
    )
    ap.add_argument("--review-dir", default="data/rerun_review")
    ap.add_argument("--objects", nargs="+", default=["animal"])
    ap.add_argument("--limit", type=int, default=0, help="Process at most N frames (0 = all)")
    args = ap.parse_args()

    frames_dir = PROJECT_ROOT / args.frames_dir
    skip = {"latest.jpg", "last_detection.jpg"}
    frames = sorted(f for f in frames_dir.glob("*.jpg") if f.name not in skip)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print("No frames found", file=sys.stderr)
        sys.exit(1)

    old = load_frames(PROJECT_ROOT / args.compare)

    print(
        f"{len(frames)} frames, {len(old)} existing detections. "
        f"Loading model on {args.device}…",
        flush=True,
    )
    from server.moondream_loader import load_moondream

    model, _ = load_moondream(device=args.device)
    motion = MotionDetector()  # server defaults

    review_dir = PROJECT_ROOT / args.review_dir
    review_dir.mkdir(parents=True, exist_ok=True)
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    detected: list[str] = []
    detect_calls = 0

    with open(out_path, "w") as out:
        for i, fp in enumerate(frames):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = motion.update(np.ascontiguousarray(bgr))

            present = False
            regions_hit: list[list[int]] = []
            objs_hit: list[str] = []

            if res.warming:
                n, hits = detect_count(model, image, args.objects)
                detect_calls += 1
                if n:
                    present = True
                    objs_hit += hits
                    regions_hit.append([0, 0, image.width, image.height])
            else:
                for (x, y, w, h) in res.regions:
                    n, hits = detect_count(model, image.crop((x, y, x + w, y + h)), args.objects)
                    detect_calls += 1
                    if n:
                        present = True
                        objs_hit += hits
                        regions_hit.append([x, y, w, h])

            if present:
                rec = {
                    "timestamp": parse_ts(fp.stem),
                    "rabbit_present": True,
                    "confidence": 1.0,
                    "objects": sorted(set(objs_hit)),
                    "regions": regions_hit,
                    "frame": fp.name,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                detected.append(fp.name)

                if fp.name not in old:
                    annotated = bgr.copy()
                    for (x, y, w, h) in regions_hit:
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        annotated,
                        ",".join(sorted(set(objs_hit))),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imwrite(str(review_dir / f"NEW_{fp.name}"), annotated)

            if (i + 1) % 500 == 0:
                print(
                    f"  [{i + 1}/{len(frames)}] {len(detected)} detections, "
                    f"{detect_calls} detect() calls",
                    flush=True,
                )

    new = set(detected)
    new_only = sorted(new - old)
    lost = sorted(old - new)

    print("\n=== SUMMARY ===")
    print(f"frames processed:   {len(frames)}")
    print(f"detect() calls:     {detect_calls}")
    print(f"old detections:     {len(old)}")
    print(f"new detections:     {len(new)}")
    print(f"new-only (added):   {len(new_only)}")
    print(f"lost (old not new): {len(lost)}")
    print(f"annotated new-only frames in: {review_dir}")
    if new_only:
        print("\nNEW-ONLY frames:")
        for n in new_only:
            print("  ", n)
    if lost:
        print("\nLOST frames (old detected, new missed):")
        for n in lost:
            print("  ", n)


if __name__ == "__main__":
    main()
