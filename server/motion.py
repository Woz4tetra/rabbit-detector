"""Running-average background subtraction for the static patio camera.

The detector flags regions that changed relative to a slowly-adapting
background. It does not decide whether a region contains an animal; it only
narrows down where the vision model should look. A wind-moved plant registers
as motion but gets rejected by the downstream ``detect()`` pass, so the cost of
a false motion region is one extra inference, not a false alert.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionResult:
    # Padded crop boxes as (x, y, w, h) in pixel coordinates.
    regions: list[tuple[int, int, int, int]]
    # True while the background model is still being seeded; callers should fall
    # back to whole-frame inference instead of trusting an empty region list.
    warming: bool


class MotionDetector:
    def __init__(
        self,
        *,
        threshold: int = 25,
        min_area_frac: float = 0.0002,
        bg_alpha: float = 0.05,
        pad_frac: float = 0.6,
        min_crop_px: int = 160,
        warmup_frames: int = 3,
        max_regions: int = 6,
    ):
        self.threshold = threshold
        self.min_area_frac = min_area_frac
        self.bg_alpha = bg_alpha
        self.pad_frac = pad_frac
        self.min_crop_px = min_crop_px
        self.warmup_frames = warmup_frames
        self.max_regions = max_regions
        self._bg: np.ndarray | None = None  # float32 grayscale background
        self._frames_seen = 0

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def update(self, frame_bgr: np.ndarray) -> MotionResult:
        h, w = frame_bgr.shape[:2]
        gray = self._prep(frame_bgr)
        self._frames_seen += 1

        if self._bg is None:
            self._bg = gray.astype(np.float32)
            return MotionResult(regions=[], warming=True)

        bg_u8 = cv2.convertScaleAbs(self._bg)
        delta = cv2.absdiff(gray, bg_u8)
        _, mask = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)

        # Adapt the background toward the current frame for the next comparison.
        cv2.accumulateWeighted(gray, self._bg, self.bg_alpha)

        if self._frames_seen <= self.warmup_frames:
            return MotionResult(regions=[], warming=True)

        min_area = self.min_area_frac * w * h
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((x, y, bw, bh, area))

        boxes.sort(key=lambda b: b[4], reverse=True)
        regions = [
            self._pad((x, y, bw, bh), w, h)
            for x, y, bw, bh, _ in boxes[: self.max_regions]
        ]
        return MotionResult(regions=regions, warming=False)

    def _pad(
        self, box: tuple[int, int, int, int], w: int, h: int
    ) -> tuple[int, int, int, int]:
        x, y, bw, bh = box
        pad_x = int(bw * self.pad_frac)
        pad_y = int(bh * self.pad_frac)
        x0, y0 = x - pad_x, y - pad_y
        x1, y1 = x + bw + pad_x, y + bh + pad_y

        # Enforce a minimum crop so a few-pixel blob still gives the model context.
        if x1 - x0 < self.min_crop_px:
            cx = (x0 + x1) // 2
            x0, x1 = cx - self.min_crop_px // 2, cx + self.min_crop_px // 2
        if y1 - y0 < self.min_crop_px:
            cy = (y0 + y1) // 2
            y0, y1 = cy - self.min_crop_px // 2, cy + self.min_crop_px // 2

        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        return (x0, y0, x1 - x0, y1 - y0)
