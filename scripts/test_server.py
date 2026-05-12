#!/usr/bin/env python3
"""Test the rabbit-detector server endpoints.

Usage:
    python scripts/test_server.py [--url http://host:8000] [--image path/to/image.jpg]
"""
from __future__ import annotations

import argparse
import sys
import time

import requests


def check_health(base_url: str) -> bool:
    print(f"GET {base_url}/health … ", end="", flush=True)
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        assert data.get("status") == "ok", f"unexpected body: {data}"
        print(f"OK  ({data})")
        return True
    except Exception as exc:
        print(f"FAIL  ({exc})")
        return False


def check_detect(base_url: str, image_path: str | None) -> bool:
    if image_path:
        with open(image_path, "rb") as f:
            jpeg_bytes = f.read()
        label = image_path
    else:
        import cv2
        import numpy as np
        noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", noise)
        jpeg_bytes = buf.tobytes()
        label = "synthetic noise frame"

    print(f"POST {base_url}/detect ({label}) … ", end="", flush=True)
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/detect",
            files={"frame": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()
        assert "rabbit" in data, f"missing 'rabbit' key: {data}"
        rabbit = data["rabbit"]
        raw = data.get("raw_response", "")
        print(f"OK  rabbit={rabbit!s:<5} raw={raw!r}  ({elapsed:.2f}s)")
        return True
    except Exception as exc:
        print(f"FAIL  ({exc})")
        return False


def check_latest_frame(base_url: str) -> None:
    print(f"GET {base_url}/latest-frame … ", end="", flush=True)
    try:
        r = requests.get(f"{base_url}/latest-frame", timeout=10)
        if r.status_code == 404:
            print("404 (no frames yet — expected before first detect call)")
        else:
            r.raise_for_status()
            print(f"OK  ({len(r.content)} bytes)")
    except Exception as exc:
        print(f"FAIL  ({exc})")


def check_dashboard(base_url: str) -> None:
    print(f"GET {base_url}/ … ", end="", flush=True)
    try:
        r = requests.get(f"{base_url}/", timeout=10)
        r.raise_for_status()
        assert "Rabbit Detector" in r.text, "title not found in dashboard HTML"
        print(f"OK  ({len(r.text)} chars)")
    except Exception as exc:
        print(f"FAIL  ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the rabbit-detector server")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--image", default=None, help="Path to a JPEG image to POST to /detect")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    failures = 0

    if not check_health(base):
        print("\nServer not reachable — is it running?")
        sys.exit(1)

    check_dashboard(base)
    check_latest_frame(base)

    if not check_detect(base, args.image):
        failures += 1

    if args.image:
        check_latest_frame(base)

    if failures:
        print(f"\n{failures} check(s) failed.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
