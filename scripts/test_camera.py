"""Test the CSI camera using the production CameraCapture class.

Checks: camera detection, frame capture at multiple resolutions,
frame validity (not all-black or all-white), and saves a sample frame.

Usage:
    python scripts/test_camera.py [--width W] [--height H] [--output /tmp/test_frame.jpg]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from rabbit_deterrent.camera import CameraCapture


def check_libcamera_detect() -> bool:
    try:
        result = subprocess.run(
            ["libcamera-hello", "--list-cameras"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout + result.stderr
        if "Available cameras" in output and "No cameras available" not in output:
            print("libcamera-hello: camera detected")
            return True
        print(f"libcamera-hello: no camera found\n{output.strip()}")
        return False
    except FileNotFoundError:
        print("libcamera-hello not found — skipping")
        return True
    except subprocess.TimeoutExpired:
        print("libcamera-hello timed out")
        return False


def check_frame(frame: np.ndarray, label: str) -> bool:
    mean, std = frame.mean(), frame.std()
    print(f"  [{label}] mean={mean:.1f}  std={std:.1f}  min={frame.min()}  max={frame.max()}")
    ok = True
    if mean < 5:
        print(f"  WARNING: nearly all-black — check ribbon cable and lens cap")
        ok = False
    if mean > 250:
        print(f"  WARNING: nearly all-white — overexposure or sensor fault")
        ok = False
    if std < 2:
        print(f"  WARNING: no variation — sensor may be stuck")
        ok = False
    return ok


def capture(width: int, height: int) -> np.ndarray | None:
    print(f"\nCapturing at {width}x{height}...")
    cam = CameraCapture(width=width, height=height)
    try:
        t0 = time.time()
        frame = cam.capture()
        elapsed = time.time() - t0
        print(f"  shape={frame.shape}  time={elapsed*1000:.0f}ms")
        return frame
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", default="/tmp/test_frame.jpg")
    args = parser.parse_args()

    failures: list[str] = []

    print("=== libcamera detection ===")
    if not check_libcamera_detect():
        failures.append("libcamera cannot see the camera")

    print(f"\n=== capture at {args.width}x{args.height} ===")
    frame = capture(args.width, args.height)
    if frame is None:
        print("\nFAILED — capture returned no frame.")
        sys.exit(1)
    if not check_frame(frame, f"{args.width}x{args.height}"):
        failures.append(f"frame at {args.width}x{args.height} looks invalid")

    if (args.width, args.height) != (320, 320):
        print("\n=== capture at 320x320 (detection resolution) ===")
        small = capture(320, 320)
        if small is not None:
            if not check_frame(small, "320x320"):
                failures.append("frame at 320x320 looks invalid")
        else:
            failures.append("capture at 320x320 failed")

    out = Path(args.output)
    cv2.imwrite(str(out), frame)
    print(f"\nSaved {frame.shape[1]}x{frame.shape[0]} frame to {out}")

    print("\n=== summary ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("  All checks passed.")


if __name__ == "__main__":
    main()
