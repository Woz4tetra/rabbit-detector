#!/usr/bin/env python3
"""Create a timelapse MP4 for each 24-hour window of server frames.
Windows run from 8 AM Eastern to 8 AM Eastern the next day."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).parent.parent


def parse_timestamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(Path(name).stem, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def load_detection_frames(path: Path) -> set[str]:
    """Return the set of frame filenames labeled as rabbit detections."""
    detected: set[str] = set()
    if not path.exists():
        return detected
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = record.get("frame")
            if frame:
                detected.add(frame)
    return detected


def window_start(dt: datetime) -> datetime:
    """Return the 8 AM ET anchor of the 24-hour window this timestamp falls in."""
    et = dt.astimezone(EASTERN)
    anchor = et.replace(hour=8, minute=0, second=0, microsecond=0)
    if et < anchor:
        anchor -= timedelta(days=1)
    return anchor


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def read_manifest(output: Path) -> dict | None:
    path = manifest_path(output)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(output: Path, frames: list[Path], complete: bool) -> None:
    manifest_path(output).write_text(
        json.dumps(
            {
                "frame_count": len(frames),
                "last_frame": frames[-1].name if frames else None,
                "complete": complete,
            }
        )
    )


def make_timelapse(frames: list[Path], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for frame in frames:
            f.write(f"file '{frame.absolute()}'\n")
            f.write(f"duration {1 / fps:.6f}\n")
        list_path = Path(f.name)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "server-frames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timelapses",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=PROJECT_ROOT / "data" / "detections.jsonl",
        help="JSONL of rabbit detections; their frames are excluded from the timelapse",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=10,
        help="Skip windows with fewer than this many frames",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every window, ignoring the up-to-date manifest cache",
    )
    args = parser.parse_args()

    skip = {"latest", "last_detection"}
    detected_frames = load_detection_frames(args.detections)
    windows: dict[datetime, list[tuple[datetime, Path]]] = {}
    excluded = 0

    for path in sorted(args.frames_dir.glob("*.jpg")):
        if path.stem in skip:
            continue
        if path.name in detected_frames:
            excluded += 1
            continue
        ts = parse_timestamp(path.name)
        if ts is None:
            continue
        key = window_start(ts)
        windows.setdefault(key, []).append((ts, path))

    if excluded:
        print(f"Excluded {excluded} frames labeled as rabbit detections")

    if not windows:
        print(f"No frames found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc)

    for window_dt, pairs in sorted(windows.items()):
        date_str = window_dt.strftime("%Y%m%d")
        output = args.output_dir / f"timelapse_{date_str}.mp4"

        if len(pairs) < args.min_frames:
            print(
                f"Skipping {date_str}: {len(pairs)} frames (min {args.min_frames})"
            )
            continue

        frames = [p for _, p in sorted(pairs)]
        # The window has fully elapsed once 24 hours past its 8 AM anchor, so no
        # more frames will ever land in it.
        complete = now >= window_dt + timedelta(days=1)

        manifest = read_manifest(output)
        if not args.force and output.exists() and manifest is not None:
            unchanged = (
                manifest.get("frame_count") == len(frames)
                and manifest.get("last_frame") == frames[-1].name
            )
            if unchanged:
                print(f"Skipping {date_str}: up to date ({len(frames)} frames)")
                continue

        status = "complete" if complete else "in progress, will rebuild next run"
        print(
            f"Building {output.name}: {len(frames)} frames at {args.fps} fps "
            f"({status})"
        )
        make_timelapse(frames, output, args.fps)
        write_manifest(output, frames, complete)
        print(f"  -> {output}")


if __name__ == "__main__":
    main()
