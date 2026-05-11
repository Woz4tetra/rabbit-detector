"""Export the trained .pt model to ONNX for deployment on the Raspberry Pi.

Usage:
    python training/export_onnx.py [--model data/models/rabbit_detector_best.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PT = PROJECT_ROOT / "data" / "models" / "rabbit_detector_best.pt"
DEFAULT_ONNX = PROJECT_ROOT / "data" / "models" / "rabbit_detector.onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_PT))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--output", default=str(DEFAULT_ONNX))
    # opset=12: compatible with system OpenCV 4.5.x on Raspberry Pi OS Bullseye (armv6l).
    # Raise to 17 only if targeting a newer OpenCV build.
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Run train.py first.")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        simplify=True,
        opset=args.opset,
    )

    # Ultralytics places the onnx next to the .pt by default; move it
    exported = Path(str(export_path))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported != output:
        exported.rename(output)

    print(f"ONNX model written to {output}")
    print(f"File size: {output.stat().st_size / 1e6:.1f} MB")

    # Quick sanity check
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dummy = np.zeros(inp.shape, dtype=np.float32)
    out = sess.run(None, {inp.name: dummy})
    print(f"Input shape:  {inp.shape}")
    print(f"Output shape: {out[0].shape}")
    print("ONNX export verified OK")


if __name__ == "__main__":
    main()
