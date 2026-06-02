#!/usr/bin/env python3
"""Build contact sheets of detection regions for false-positive review.

For each frame flagged by the motion pipeline but not by the old log, crop the
region(s) detect() fired on (with context padding), label with the timestamp,
and tile into pages. Looking at the crop, not the whole frame, makes it obvious
whether there is really an animal there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def load_frames(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                f = json.loads(line).get("frame")
                if f:
                    out.add(f)
            except json.JSONDecodeError:
                pass
    return out


def load_records(path: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("frame"):
            recs[r["frame"]] = r
    return recs


def crop_region(frame_bgr, region, pad_frac=0.4):
    h, w = frame_bgr.shape[:2]
    x, y, rw, rh = region
    px, py = int(rw * pad_frac), int(rh * pad_frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + rw + px), min(h, y + rh + py)
    return frame_bgr[y0:y1, x0:x1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", default="data/detections_motion.jsonl")
    ap.add_argument("--old", default="data/detections.jsonl")
    ap.add_argument("--frames-dir", default="data/server-frames")
    ap.add_argument("--out", default="/tmp/review")
    ap.add_argument("--which", choices=["new_only", "lost"], default="new_only")
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--rows", type=int, default=7)
    args = ap.parse_args()

    new_recs = load_records(PROJECT_ROOT / args.new)
    old = load_frames(PROJECT_ROOT / args.old)
    new = set(new_recs)

    if args.which == "new_only":
        targets = sorted(new - old)
    else:
        targets = sorted(old - new)

    frames_dir = PROJECT_ROOT / args.frames_dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile, cols, rows = args.tile, args.cols, args.rows
    per_page = cols * rows
    label_h = 22
    cell_h = tile + label_h

    print(f"{len(targets)} {args.which} frames -> {(len(targets)+per_page-1)//per_page} page(s)")

    page_imgs: list[np.ndarray] = []
    page_no = 0

    def flush(cells):
        nonlocal page_no
        if not cells:
            return
        canvas = np.full((rows * cell_h, cols * tile, 3), 30, np.uint8)
        for idx, (label, img) in enumerate(cells):
            r, c = divmod(idx, cols)
            y0 = r * cell_h
            x0 = c * tile
            cv2.putText(canvas, label, (x0 + 2, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            if img is not None and img.size:
                ih, iw = img.shape[:2]
                scale = min(tile / iw, tile / ih)
                rw, rh = max(1, int(iw * scale)), max(1, int(ih * scale))
                resized = cv2.resize(img, (rw, rh))
                yy = y0 + label_h + (tile - rh) // 2
                xx = x0 + (tile - rw) // 2
                canvas[yy:yy + rh, xx:xx + rw] = resized
        page_no += 1
        outp = out_dir / f"{args.which}_page_{page_no:02d}.png"
        cv2.imwrite(str(outp), canvas)
        print(f"  wrote {outp}")

    cells: list = []
    for frame in targets:
        rec = new_recs.get(frame) if args.which == "new_only" else None
        fp = frames_dir / frame
        bgr = cv2.imread(str(fp))
        # Label: HHMMSS of the day + objects
        stem = Path(frame).stem  # 20260530T090641Z
        label = stem[9:15] if len(stem) >= 15 else stem
        if rec and rec.get("objects"):
            label += " " + ",".join(rec["objects"])[:10]
        crop = None
        if bgr is not None:
            if rec and rec.get("regions"):
                crop = crop_region(bgr, rec["regions"][0])
            else:
                crop = bgr  # lost frames: no region info, show whole frame
        cells.append((label, crop))
        if len(cells) == per_page:
            flush(cells)
            cells = []
    flush(cells)


if __name__ == "__main__":
    main()
