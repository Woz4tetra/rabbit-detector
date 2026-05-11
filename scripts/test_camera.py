"""Test the CSI camera via picamera2/libcamera.

Checks: camera detection, frame capture at multiple resolutions,
frame validity (not all-black or all-white), and saves a sample frame.

Usage:
    python scripts/test_camera.py [--width W] [--height H] [--output /tmp/test_frame.jpg]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_libcamera_detect() -> bool:
    """Return True if libcamera sees at least one camera."""
    import subprocess

    try:
        result = subprocess.run(
            ["libcamera-hello", "--list-cameras"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        if "Available cameras" in output and "No cameras available" not in output:
            print("libcamera-hello: camera detected")
            for line in output.splitlines():
                if line.strip():
                    print(f"  {line.strip()}")
            return True
        print(f"libcamera-hello: no camera found\n{output.strip()}")
        return False
    except FileNotFoundError:
        print("libcamera-hello not found — skipping libcamera detection check")
        return True  # not a blocking failure
    except subprocess.TimeoutExpired:
        print("libcamera-hello timed out — camera may be unresponsive")
        return False


def capture_frame(width: int, height: int) -> "np.ndarray | None":
    import numpy as np

    try:
        from picamera2 import Picamera2
    except ImportError:
        print("ERROR: picamera2 not importable. Run install_pi.sh first.")
        return None

    print(f"\nOpening camera at {width}x{height}...")
    cam = Picamera2()

    # Print camera properties
    camera_properties = cam.camera_properties
    print(f"  Model:           {camera_properties.get('Model', 'unknown')}")
    print(f"  PixelArraySize:  {camera_properties.get('PixelArraySize', 'unknown')}")
    print(f"  Location:        {camera_properties.get('Location', 'unknown')}")

    config = cam.create_still_configuration(
        main={"size": (width, height), "format": "BGR888"}
    )
    cam.configure(config)
    cam.start()

    # Let auto-exposure settle
    time.sleep(2)

    t0 = time.time()
    frame = cam.capture_array()
    elapsed = time.time() - t0

    cam.stop()
    cam.close()

    print(f"  Captured in {elapsed*1000:.0f} ms, shape {frame.shape}")
    return frame


def check_frame_validity(frame: "np.ndarray", label: str) -> bool:
    import numpy as np

    mean = frame.mean()
    std = frame.std()
    min_val = frame.min()
    max_val = frame.max()

    print(f"  [{label}] mean={mean:.1f}  std={std:.1f}  min={min_val}  max={max_val}")

    ok = True
    if mean < 5:
        print(f"  WARNING: frame is nearly all-black (mean={mean:.1f}) — check camera ribbon cable and lens cap")
        ok = False
    if mean > 250:
        print(f"  WARNING: frame is nearly all-white (mean={mean:.1f}) — severe overexposure or sensor fault")
        ok = False
    if std < 2:
        print(f"  WARNING: frame has no variation (std={std:.1f}) — sensor may be stuck")
        ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", default="/tmp/test_frame.jpg")
    args = parser.parse_args()

    import cv2
    import numpy as np

    failures: list[str] = []

    # 1. libcamera detection
    print("=== libcamera detection ===")
    if not check_libcamera_detect():
        failures.append("libcamera cannot see the camera")

    # 2. Primary resolution capture
    print(f"\n=== capture at {args.width}x{args.height} ===")
    frame = capture_frame(args.width, args.height)
    if frame is None:
        failures.append("picamera2 capture failed")
        print("\nFAILED — camera did not capture a frame.")
        sys.exit(1)

    if not check_frame_validity(frame, f"{args.width}x{args.height}"):
        failures.append(f"frame at {args.width}x{args.height} looks invalid")

    # 3. Detection resolution (what the model actually uses)
    if (args.width, args.height) != (320, 320):
        print("\n=== capture at 320x320 (detection resolution) ===")
        frame_small = capture_frame(320, 320)
        if frame_small is not None:
            check_frame_validity(frame_small, "320x320")
        else:
            failures.append("picamera2 capture at 320x320 failed")

    # 4. Save output
    out = Path(args.output)
    cv2.imwrite(str(out), frame)
    print(f"\nSaved {frame.shape[1]}x{frame.shape[0]} frame to {out}")

    # 5. Summary
    print("\n=== summary ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    else:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
