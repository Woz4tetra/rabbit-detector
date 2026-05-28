#!/usr/bin/env python3
"""Capture a single still with rpicam-still using camera settings from config.yaml.

This bypasses Picamera2 / CameraCapture and goes straight to the rpicam-apps
CLI. Useful for diffing "what config.yaml says" against "what the sensor
actually produces" without the Python pipeline in the way.

Run on the Pi. Output is a JPEG you can scp back for visual inspection.

    python scripts/test_raspi_still.py                       # uses config.yaml
    python scripts/test_raspi_still.py --output /tmp/x.jpg
    python scripts/test_raspi_still.py --shutter 2000000     # override exposure
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT = Path("/tmp/raspi_test.jpg")

AWB_MODES = {
    0: "auto",
    1: "incandescent",
    2: "tungsten",
    3: "fluorescent",
    4: "indoor",
    5: "daylight",
    6: "cloudy",
}

DENOISE_MODES = {
    0: "off",
    1: "cdn_fast",
    2: "cdn_hq",
}


def build_command(cfg: dict, output: Path, shutter_override: int | None, gain_override: float | None) -> list[str]:
    camera = cfg.get("camera", {})
    detection = cfg.get("detection", {})

    width = int(detection.get("capture_width", 1280))
    height = int(detection.get("capture_height", 720))

    ae_enable = bool(camera.get("ae_enable", True))
    exposure_time_us = int(camera.get("exposure_time_us", 20000))
    analogue_gain = float(camera.get("analogue_gain", 1.0))
    awb_enable = bool(camera.get("awb_enable", True))
    awb_mode = int(camera.get("awb_mode", 0))
    red_gain = float(camera.get("red_gain", 1.5))
    blue_gain = float(camera.get("blue_gain", 1.5))
    brightness = float(camera.get("brightness", 0.0))
    contrast = float(camera.get("contrast", 1.0))
    saturation = float(camera.get("saturation", 1.0))
    sharpness = float(camera.get("sharpness", 1.0))
    nr_mode = int(camera.get("noise_reduction_mode", 1))

    if shutter_override is not None:
        ae_enable = False
        exposure_time_us = shutter_override
    if gain_override is not None:
        ae_enable = False
        analogue_gain = gain_override

    cmd = [
        "rpicam-still",
        "--nopreview",
        "--width", str(width),
        "--height", str(height),
        "--output", str(output),
        "--brightness", str(brightness),
        "--contrast", str(contrast),
        "--saturation", str(saturation),
        "--sharpness", str(sharpness),
        "--denoise", DENOISE_MODES.get(nr_mode, "cdn_fast"),
    ]

    if ae_enable:
        # Give AE ~3 s to converge before the shot.
        cmd += ["--timeout", "3000"]
    else:
        # --immediate skips the preview wait; the shutter still takes its full time.
        cmd += ["--immediate", "--shutter", str(exposure_time_us), "--gain", str(analogue_gain)]

    if awb_enable:
        cmd += ["--awb", AWB_MODES.get(awb_mode, "auto")]
    else:
        cmd += ["--awbgains", f"{red_gain},{blue_gain}"]

    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config.yaml path (default: project root)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"output JPEG path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--shutter", type=int, default=None, help="override exposure_time_us (forces AE off)")
    p.add_argument("--gain", type=float, default=None, help="override analogue_gain (forces AE off)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cmd = build_command(cfg, args.output, args.shutter, args.gain)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"rpicam-still exited {result.returncode}", file=sys.stderr)
        return result.returncode

    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
