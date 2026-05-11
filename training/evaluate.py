"""Evaluate model on the test split and save annotated images.

Metrics (mAP50, mAP50-95, precision, recall) are computed via ultralytics.
One annotated image per test sample is saved to the output directory, with
ground-truth boxes in green and predicted boxes in blue.

Usage:
    python training/evaluate.py [--model <path>] [--out <dir>] [--conf 0.25]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

TRAINING_DIR = Path(__file__).parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_PT = PROJECT_ROOT / "data" / "models" / "rabbit_detector_best.pt"
DEFAULT_OUT = TRAINING_DIR / "runs" / "evaluate"


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Read YOLO-format label file and return (class_id, x1, y1, x2, y2) in pixels."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             label: str, color: tuple, thickness: int = 2) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    bg_y1 = max(y1 - th - 6, 0)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_image(img_path: Path, label_path: Path, predictions, class_names: list[str]) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Ground truth — green
    for cls_id, x1, y1, x2, y2 in load_gt_boxes(label_path, w, h):
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        draw_box(img, x1, y1, x2, y2, f"GT:{name}", color=(0, 180, 0))

    # Predictions — blue
    if predictions is not None and len(predictions.boxes):
        for box in predictions.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            draw_box(img, x1, y1, x2, y2, f"{name} {conf:.2f}", color=(200, 80, 0))

    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_PT))
    parser.add_argument("--data-yaml", default=str(TRAINING_DIR / "data.yaml"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found at {data_yaml}")

    meta = yaml.safe_load(data_yaml.read_text())
    class_names = meta.get("names", [])
    test_images_dir = Path(meta["test"])
    test_labels_dir = test_images_dir.parent / "labels"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.model)

    # --- Metrics on test split ---
    print("Computing metrics on test split...")
    metrics = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, verbose=False)
    mp, mr, map50, map50_95 = (
        metrics.box.mp,
        metrics.box.mr,
        metrics.box.map50,
        metrics.box.map,
    )
    print(f"\n{'Metric':<20} {'Value':>8}")
    print("-" * 30)
    print(f"{'Precision':<20} {mp:>8.4f}")
    print(f"{'Recall':<20} {mr:>8.4f}")
    print(f"{'mAP50':<20} {map50:>8.4f}")
    print(f"{'mAP50-95':<20} {map50_95:>8.4f}")
    if map50 < 0.70:
        print("\nWARNING: mAP50 below 0.70.")
    elif map50 >= 0.80:
        print("\nModel looks good for deployment.")

    # --- Annotated images ---
    image_paths = sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))
    print(f"\nAnnotating {len(image_paths)} test images -> {out_dir}")

    tp = fp = fn = 0
    for img_path in image_paths:
        label_path = test_labels_dir / (img_path.stem + ".txt")
        result = model.predict(img_path, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        annotated = annotate_image(img_path, label_path, result, class_names)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        n_gt = sum(1 for _ in load_gt_boxes(label_path, *annotated.shape[1::-1]))
        n_pred = len(result.boxes) if result.boxes is not None else 0
        tp += min(n_gt, n_pred)
        fp += max(0, n_pred - n_gt)
        fn += max(0, n_gt - n_pred)

    print(f"Done. Images saved to {out_dir}")
    print(f"\nNaive box counts across test set (not IoU-matched):")
    print(f"  GT boxes:   {tp + fn}")
    print(f"  Pred boxes: {tp + fp}")
    print(f"  Over-detections (FP proxy):  {fp}")
    print(f"  Missed detections (FN proxy): {fn}")


if __name__ == "__main__":
    main()
