"""Run validation on the test split and report mAP50.

Usage:
    python training/validate.py [--model data/models/rabbit_detector_best.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PT = PROJECT_ROOT / "data" / "models" / "rabbit_detector_best.pt"
DATA_YAML = Path(__file__).parent / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_PT))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml not found at {DATA_YAML}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    metrics = model.val(data=str(DATA_YAML), split=args.split, imgsz=320)
    map50 = metrics.box.map50
    print(f"\nmAP50 on '{args.split}' split: {map50:.4f}")
    if map50 < 0.70:
        print("WARNING: mAP50 below 0.70. Consider collecting more data or tuning augmentation.")
    elif map50 >= 0.80:
        print("Model looks good. Proceed to export_onnx.py")


if __name__ == "__main__":
    main()
