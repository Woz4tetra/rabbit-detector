"""Run the ONNX detector on a test image and report inference latency.

Usage:
    python scripts/test_detector.py [--image path/to/rabbit.jpg]

If no image is provided, a synthetic noise image is used (won't detect anything,
but confirms the model loads and runs without error).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "rabbit_detector.onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run training/export_onnx.py and deploy the model first.")
        sys.exit(1)

    import cv2
    import numpy as np

    from rabbit_deterrent.detector import OnnxRabbitDetector

    detector = OnnxRabbitDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.3,
        image_size=320,
    )

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Could not read image {args.image}")
            sys.exit(1)
    else:
        print("No image provided, using synthetic noise image (no detections expected)")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print(f"Running inference on {frame.shape[1]}x{frame.shape[0]} image...")
    t0 = time.time()
    detections = detector.detect(frame)
    elapsed = time.time() - t0

    print(f"Inference time: {elapsed:.2f}s")
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  confidence={d.confidence:.2f}  bbox=({d.x1:.0f},{d.y1:.0f})->({d.x2:.0f},{d.y2:.0f})")

    # Pi Zero W (ARMv6, single-core, cv2.dnn): expect 15-60s per frame.
    # Pi Zero W (ARMv7, quad-core): expect 2-4s per frame.
    if elapsed > 60.0:
        print(f"WARNING: Inference took {elapsed:.1f}s — significantly above expected range (15-60s).")


if __name__ == "__main__":
    main()
