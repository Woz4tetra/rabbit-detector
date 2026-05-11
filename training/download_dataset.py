"""Download rabbit detection dataset from Roboflow Universe.

Usage:
    ROBOFLOW_API_KEY=<key> python training/download_dataset.py [--workspace <ws>] [--project <proj>] [--version <n>]

The downloaded dataset lands in training/data/ and a data.yaml is written
that YOLO training scripts can consume directly.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="roboflow-universe-projects")
    parser.add_argument("--project", default="rabbit-n4bj4")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY environment variable")

    import zipfile

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)

    available = [v.version for v in project.versions()]
    print(f"Available versions: {available}")
    if str(args.version) not in available:
        raise SystemExit(
            f"Version {args.version} not found. Pick one from: {available}"
        )

    try:
        dataset = project.version(args.version).download("yolov8", location=str(DATA_DIR))
    except zipfile.BadZipFile:
        raise SystemExit(
            "Download returned a bad zip — the API likely rejected the request. "
            "Check that your API key has access to this project and version."
        )
    print(f"Dataset downloaded to {dataset.location}")

    # Some roboflow versions download the zip but skip extraction.
    # Extract manually if data.yaml is missing.
    dataset_dir = Path(dataset.location)
    if not (dataset_dir / "data.yaml").exists():
        zip_path = dataset_dir / "roboflow.zip"
        if not zip_path.exists():
            raise SystemExit(f"Neither data.yaml nor roboflow.zip found in {dataset_dir}")
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_dir)

    # Write the data.yaml pointer used by train.py
    yaml_src = dataset_dir / "data.yaml"
    yaml_dst = TRAINING_DIR / "data.yaml"
    yaml_dst.write_text(yaml_src.read_text())
    print(f"data.yaml written to {yaml_dst}")

    import yaml
    meta = yaml.safe_load(yaml_dst.read_text())
    print("Class names:", meta.get("names", []))


if __name__ == "__main__":
    main()
