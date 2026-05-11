"""Fine-tune YOLOv8n on the rabbit dataset using DDP across all available GPUs.

Usage:
    python training/train.py [--gpus 0,1,2] [--epochs 100] [--batch 192]

Run from the project root or the training/ directory.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

_IR_PROB = 0.4  # fraction of batches simulated as Arducam IR night-vision output


def _ir_augment_callback(trainer) -> None:
    """Convert a random fraction of batches to simulated IR greyscale.

    The Arducam auto-switches to IR mode at night, producing greyscale images.
    Boosting the green channel weight approximates near-IR vegetation response
    (chlorophyll reflection makes foliage appear bright in NIR).
    """
    if random.random() >= _IR_PROB:
        return
    imgs = trainer.batch["img"]  # [B, 3, H, W], float32 in [0, 1]
    weights = imgs.new_tensor([0.20, 0.70, 0.10]).view(1, 3, 1, 1)
    grey = (imgs * weights).sum(dim=1, keepdim=True).expand_as(imgs)
    # Mild contrast stretch to mimic IR LED illumination falloff
    grey = ((grey - 0.5) * 1.25 + 0.5).clamp(0.0, 1.0)
    trainer.batch["img"] = grey


TRAINING_DIR = Path(__file__).parent
PROJECT_ROOT = TRAINING_DIR.parent
RUNS_DIR = TRAINING_DIR / "runs"
OUTPUT_MODEL = PROJECT_ROOT / "data" / "models" / "rabbit_detector_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2", help="Comma-separated GPU IDs or 'cpu'")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--name", default="rabbit_v1")
    parser.add_argument("--data-yaml", default=str(TRAINING_DIR / "data.yaml"),
                        help="Path to data.yaml (default: training/data.yaml)")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found at {data_yaml}. Run download_dataset.py first.")

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.add_callback("on_train_batch_start", _ir_augment_callback)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.gpus,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        # Augmentation: keep defaults but disable vertical flip (rabbits are upright)
        flipud=0.0,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        verbose=True,
    )

    # With DDP, only rank-0 returns a results object; other workers return None.
    # Use the known save path directly rather than results.save_dir.
    save_dir = RUNS_DIR / args.name
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise SystemExit(f"Expected best.pt not found at {best_pt}")

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, OUTPUT_MODEL)
    print(f"\nBest weights saved to {OUTPUT_MODEL}")

    if results is not None:
        print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print("\nNext step: run training/export_onnx.py")


if __name__ == "__main__":
    main()
