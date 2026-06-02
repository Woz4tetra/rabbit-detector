"""Shared motion-gated, two-stage detection used by the live server and the
offline replay script, so the two never drift.

Stage 1 (recall): motion gating picks changed regions; detect() localizes any
target species in each region crop.
Stage 2 (precision): a yes/no query() on the same crop confirms a real animal is
present, which rejects detect()'s hallucinations on leaves, furniture, and
shadows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

DEFAULT_CONFIRM_PROMPT = (
    "Does this image clearly show a live animal such as a rabbit, chipmunk, "
    "squirrel, or bird? Answer with only 'yes' or 'no'."
)


@dataclass
class DetectionOutcome:
    present: bool
    objects: list[str] = field(default_factory=list)
    regions: list[list[int]] = field(default_factory=list)  # (x, y, w, h) px in frame
    detail: str = ""


def _detect_labels(model, enc, objects: list[str]) -> list[str]:
    """Return the target labels detect() localizes at least one box for.

    Takes a pre-encoded image so the same crop is encoded once and reused across
    every species, instead of re-encoding per detect() call.
    """
    labels: list[str] = []
    for obj in objects:
        try:
            res = model.detect(enc, obj)
        except Exception:  # noqa: BLE001 — a bad label shouldn't kill the frame
            continue
        if res.get("objects"):
            labels.append(obj)
    return labels


def _confirm(model, enc, prompt: str) -> bool:
    ans = str(model.query(enc, prompt)["answer"]).strip().lower()
    return ans.startswith("yes")


def run_pipeline(
    model,
    image: Image.Image,
    *,
    motion=None,
    objects: list[str],
    confirm_enabled: bool = True,
    confirm_prompt: str = DEFAULT_CONFIRM_PROMPT,
) -> DetectionOutcome:
    """Run the full pipeline on one RGB frame and return what was confirmed."""
    if motion is None:
        regions: list[tuple[int, int, int, int]] = [(0, 0, image.width, image.height)]
    else:
        frame_bgr = np.ascontiguousarray(np.array(image)[:, :, ::-1])
        res = motion.update(frame_bgr)
        if res.warming:
            regions = [(0, 0, image.width, image.height)]
        elif not res.regions:
            return DetectionOutcome(present=False, detail="no motion")
        else:
            regions = res.regions

    hit_objs: list[str] = []
    hit_regions: list[list[int]] = []

    for (x, y, w, h) in regions:
        # Encode the crop once; detect() and query() both accept the EncodedImage.
        enc = model.encode_image(image.crop((x, y, x + w, y + h)))
        labels = _detect_labels(model, enc, objects)
        if not labels:
            continue
        if confirm_enabled and not _confirm(model, enc, confirm_prompt):
            continue
        hit_objs += labels
        hit_regions.append([x, y, w, h])

    if hit_regions:
        objs = sorted(set(hit_objs))
        return DetectionOutcome(
            present=True,
            objects=objs,
            regions=hit_regions,
            detail=f"confirmed {','.join(objs)} in {len(hit_regions)} region(s)",
        )
    return DetectionOutcome(present=False, detail="motion but no confirmed animal")
