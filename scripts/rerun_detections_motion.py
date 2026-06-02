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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.motion import MotionDetector  # noqa: E402
from server.pipeline import DEFAULT_CONFIRM_PROMPT, run_pipeline  # noqa: E402


def parse_dt(stem: str) -> datetime | None:
    try:
        return datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_ts(stem: str) -> str:
    dt = parse_dt(stem)
    return dt.isoformat().replace("+00:00", "Z") if dt else stem


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
    ap.add_argument("--objects", nargs="+", default=["rabbit", "chipmunk", "squirrel"])
    ap.add_argument("--no-confirm", action="store_true", help="Skip the stage-2 yes/no confirmation")
    ap.add_argument("--confirm-prompt", default=DEFAULT_CONFIRM_PROMPT)
    ap.add_argument(
        "--since-hours",
        type=float,
        default=0,
        help="Only process frames from the last N hours (0 = all)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Process at most N frames (0 = all)")
    args = ap.parse_args()

    frames_dir = PROJECT_ROOT / args.frames_dir
    skip = {"latest.jpg", "last_detection.jpg"}
    frames = sorted(f for f in frames_dir.glob("*.jpg") if f.name not in skip)

    cutoff: datetime | None = None
    if args.since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
        frames = [f for f in frames if (dt := parse_dt(f.stem)) and dt >= cutoff]
        print(f"Window: frames since {cutoff.isoformat()} ({len(frames)} frames)")

    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print("No frames found", file=sys.stderr)
        sys.exit(1)

    old = load_frames(PROJECT_ROOT / args.compare)
    if cutoff is not None:
        # Restrict the baseline to the same window so new-only/lost are meaningful.
        old = {f for f in old if (dt := parse_dt(Path(f).stem)) and dt >= cutoff}

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

    with open(out_path, "w") as out:
        for i, fp in enumerate(frames):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            outcome = run_pipeline(
                model,
                image,
                motion=motion,
                objects=args.objects,
                confirm_enabled=not args.no_confirm,
                confirm_prompt=args.confirm_prompt,
            )

            if outcome.present:
                rec = {
                    "timestamp": parse_ts(fp.stem),
                    "rabbit_present": True,
                    "confidence": 1.0,
                    "objects": outcome.objects,
                    "regions": outcome.regions,
                    "frame": fp.name,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                detected.append(fp.name)

                if fp.name not in old:
                    annotated = bgr.copy()
                    for (x, y, w, h) in outcome.regions:
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        annotated,
                        ",".join(outcome.objects),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imwrite(str(review_dir / f"NEW_{fp.name}"), annotated)

            if (i + 1) % 500 == 0:
                print(
                    f"  [{i + 1}/{len(frames)}] {len(detected)} detections",
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
