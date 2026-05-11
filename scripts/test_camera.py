"""Capture one frame from the camera and save it to /tmp/test_frame.jpg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rabbit_deterrent.camera import CameraCapture
import cv2

cam = CameraCapture(width=640, height=480)
print("Capturing frame...")
frame = cam.capture()
out = "/tmp/test_frame.jpg"
cv2.imwrite(out, frame)
print(f"Saved {frame.shape[1]}x{frame.shape[0]} frame to {out}")
