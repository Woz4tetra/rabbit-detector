from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    model_id: str = "vikhyatk/moondream2"
    model_revision: str = "2025-01-09"
    device: str = "cuda:0"
    detection_prompt: str = "Is there a rabbit in this image? Reply with only 'yes' or 'no'."
    # Objects passed to Moondream's detect() head (open-vocabulary). Concrete
    # species hallucinate far less than the generic "animal" on leaves/furniture.
    detection_objects: list[str] = field(
        default_factory=lambda: ["rabbit", "chipmunk", "squirrel"]
    )
    # Stage-2 confirmation: after detect() localizes a candidate, a yes/no query()
    # on the region crop rejects false positives (leaves, shadows, furniture).
    confirm_enabled: bool = True
    confirm_prompt: str = (
        "Does this image clearly show a live animal such as a rabbit, chipmunk, "
        "squirrel, or bird? Answer with only 'yes' or 'no'."
    )
    # Motion gating: only run detect() on frame regions that changed against a
    # rolling background. Dramatically improves recall on small, distant subjects.
    motion_enabled: bool = True
    motion_threshold: int = 25
    motion_min_area_frac: float = 0.0002
    motion_bg_alpha: float = 0.05
    motion_pad_frac: float = 0.6
    motion_min_crop_px: int = 160
    motion_warmup_frames: int = 3
    motion_max_regions: int = 6
    frames_dir: str = "data/server-frames"
    max_frames: int = 5000


def load_server_settings(path: Path | None = None):
    """Return (ServerSettings, EmailConfig) loaded from config.yaml."""
    from rabbit_deterrent.config import EmailConfig

    if path is None:
        path = PROJECT_ROOT / "config.yaml"

    with open(path) as f:
        raw = yaml.safe_load(f)

    sa = raw.get("server_app", {})
    settings = ServerSettings(
        host=sa.get("host", "0.0.0.0"),
        port=int(sa.get("port", 8000)),
        model_id=sa.get("model_id", "vikhyatk/moondream2"),
        model_revision=sa.get("model_revision", "2025-01-09"),
        device=sa.get("device", "cuda:0"),
        detection_prompt=sa.get(
            "detection_prompt",
            "Is there a rabbit in this image? Reply with only 'yes' or 'no'.",
        ),
        detection_objects=list(
            sa.get("detection_objects", ["rabbit", "chipmunk", "squirrel"])
        ),
        confirm_enabled=bool(sa.get("confirm_enabled", True)),
        confirm_prompt=sa.get("confirm_prompt", ServerSettings.confirm_prompt),
        motion_enabled=bool(sa.get("motion_enabled", True)),
        motion_threshold=int(sa.get("motion_threshold", 25)),
        motion_min_area_frac=float(sa.get("motion_min_area_frac", 0.0002)),
        motion_bg_alpha=float(sa.get("motion_bg_alpha", 0.05)),
        motion_pad_frac=float(sa.get("motion_pad_frac", 0.6)),
        motion_min_crop_px=int(sa.get("motion_min_crop_px", 160)),
        motion_warmup_frames=int(sa.get("motion_warmup_frames", 3)),
        motion_max_regions=int(sa.get("motion_max_regions", 6)),
        frames_dir=sa.get("frames_dir", "data/server-frames"),
        max_frames=int(sa.get("max_frames", 5000)),
    )

    e = raw.get("email", {})
    email = EmailConfig(
        enabled=bool(e.get("enabled", True)),
        host=e.get("host", ""),
        port=int(e.get("port", 587)),
        username=e.get("username", ""),
        password=e.get("password", ""),
        from_addr=e.get("from_addr", ""),
        to_addr=e.get("to_addr", ""),
        cooldown_seconds=int(e.get("cooldown_seconds", 300)),
    )

    return settings, email
