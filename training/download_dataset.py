"""Download rabbit detection dataset from Roboflow.

Usage:
    ROBOFLOW_API_KEY=<key> python training/download_dataset.py
    # or store key in ~/ROBOFLOW_API_KEY and run without the env var prefix

The downloaded dataset lands in training/data/ and a data.yaml is written
that YOLO training scripts can consume directly.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

TRAINING_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="yolodemov5")
    parser.add_argument("--project", default="rabbit-n4bj4")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        key_file = Path.home() / "ROBOFLOW_API_KEY"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        raise SystemExit(
            "No API key found. Set ROBOFLOW_API_KEY env var or write it to ~/ROBOFLOW_API_KEY"
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    dataset = version.download("yolov8")

    # Move downloaded data into training/data/, removing the roboflow staging dir.
    src_dir = Path(dataset.location)
    dst_dir = TRAINING_DIR / "data"
    if src_dir != dst_dir:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        shutil.rmtree(src_dir)

    # Rewrite data.yaml with absolute paths so YOLO can find images regardless
    # of working directory. Roboflow exports relative paths (e.g. ../train/images)
    # that only work inside the zip structure.
    import yaml

    yaml_src = dst_dir / "data.yaml"
    meta = yaml.safe_load(yaml_src.read_text())
    for split in ("train", "val", "test"):
        if split in meta:
            # Roboflow's paths (e.g. "../valid/images") are relative to a layout
            # that doesn't match our extraction directory. Take only the last two
            # path components (e.g. "valid/images") and root them in dst_dir.
            tail = Path(meta[split]).parts[-2:]
            meta[split] = str(dst_dir / Path(*tail))

    yaml_dst = TRAINING_DIR / "data.yaml"
    yaml_dst.write_text(yaml.dump(meta))
    print(f"Dataset at {dst_dir}")
    print(f"data.yaml written to {yaml_dst}")
    print("Class names:", meta.get("names", []))


if __name__ == "__main__":
    main()
