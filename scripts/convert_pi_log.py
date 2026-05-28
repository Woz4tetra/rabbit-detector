"""Convert a Pi-side detections.jsonl to the server schema.

The Pi log mixes two formats:
  - Old: {timestamp, state, rabbit_present, confidence, raw_response}
  - New: {timestamp, state, rabbit_present, detections: [{confidence, bbox}]}

This script:
  1. Keeps only rabbit_present=true entries
  2. Normalises confidence to a top-level float
  3. Matches each entry to the nearest saved server frame (within FRAME_TOLERANCE_S)
  4. Writes the result to OUTPUT_PATH

Run from the project root:
  python scripts/convert_pi_log.py [--input data/detections.jsonl] [--output data/detections.jsonl]
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "server-frames"
FRAME_TOLERANCE_S = 10


def load_frame_index(frames_dir: Path) -> list[tuple[datetime.datetime, str]]:
    skip = {"latest.jpg", "last_detection.jpg"}
    result: list[tuple[datetime.datetime, str]] = []
    for f in frames_dir.glob("*.jpg"):
        if f.name in skip:
            continue
        try:
            dt = datetime.datetime.strptime(f.stem, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=datetime.timezone.utc
            )
            result.append((dt, f.name))
        except ValueError:
            pass
    result.sort()
    return result


def find_nearest_frame(
    dt: datetime.datetime,
    frame_index: list[tuple[datetime.datetime, str]],
) -> str | None:
    if not frame_index:
        return None
    dts = [ft[0] for ft in frame_index]
    idx = bisect.bisect_left(dts, dt)
    tolerance = datetime.timedelta(seconds=FRAME_TOLERANCE_S)
    best_name: str | None = None
    best_diff = tolerance
    for i in (idx - 1, idx):
        if 0 <= i < len(frame_index):
            diff = abs(frame_index[i][0] - dt)
            if diff <= best_diff:
                best_diff = diff
                best_name = frame_index[i][1]
    return best_name


def convert(input_path: Path, output_path: Path) -> None:
    frame_index = load_frame_index(FRAMES_DIR)
    print(f"Loaded {len(frame_index)} server frames from {FRAMES_DIR}")

    entries: list[dict] = []
    skipped = 0
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1

    print(f"Read {len(entries)} entries ({skipped} skipped as malformed)")

    rabbit_entries = [e for e in entries if e.get("rabbit_present")]
    print(f"Rabbit-present entries: {len(rabbit_entries)}")

    matched = 0
    output_records: list[dict] = []
    for e in rabbit_entries:
        ts = e.get("timestamp", "")

        # Normalise confidence from either schema variant
        if "detections" in e and e["detections"]:
            confidence = e["detections"][0].get("confidence", 0.0)
        else:
            confidence = e.get("confidence", 0.0)

        record: dict = {
            "timestamp": ts,
            "rabbit_present": True,
            "confidence": confidence,
            "state": e.get("state", ""),
        }

        # Try to match a server frame
        frame = e.get("frame")
        if not frame and ts and frame_index:
            try:
                dt = datetime.datetime.fromisoformat(ts.rstrip("Z")).replace(
                    tzinfo=datetime.timezone.utc
                )
                frame = find_nearest_frame(dt, frame_index)
            except ValueError:
                pass

        if frame:
            record["frame"] = frame
            matched += 1

        output_records.append(record)

    print(f"Frame-matched: {matched} / {len(rabbit_entries)}")

    with open(output_path, "w") as f:
        for r in output_records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(output_records)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "detections.jsonl"),
        help="Input Pi log path",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "detections.jsonl"),
        help="Output path (default overwrites input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        raise SystemExit(1)

    if input_path.resolve() == output_path.resolve():
        backup = output_path.with_suffix(".jsonl.bak")
        import shutil
        shutil.copy2(input_path, backup)
        print(f"Backed up original to {backup}")

    convert(input_path, output_path)


if __name__ == "__main__":
    main()
