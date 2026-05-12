from __future__ import annotations

from dataclasses import dataclass
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
