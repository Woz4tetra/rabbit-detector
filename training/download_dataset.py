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
    parser.add_argument("--data-yaml", default=str(TRAINING_DIR / "data.yaml"),
                        help="Path to write the data.yaml for training")
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

    # Copy downloaded data into training/data/ and write training/data.yaml
    src_dir = Path(dataset.location)
    dst_dir = TRAINING_DIR / "data"
    if src_dir != dst_dir:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

    yaml_src = dst_dir / "data.yaml"
    yaml_dst = Path(args.data_yaml)
    shutil.copy(yaml_src, yaml_dst)

    import yaml
    meta = yaml.safe_load(yaml_dst.read_text())
    print(f"Dataset at {dst_dir}")
    print("Class names:", meta.get("names", []))


if __name__ == "__main__":
    main()
