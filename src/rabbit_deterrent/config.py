from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class DetectionConfig:
    model_path: str
    confidence_threshold: float
    image_size: int
    capture_width: int
    capture_height: int

    def resolved_model_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class AudioConfig:
    sound_file: str
    volume: float

    def resolved_sound_path(self) -> Path:
        p = Path(self.sound_file)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class EmailConfig:
    host: str
    port: int
    username: str
    password: str
    from_addr: str
    to_addr: str
    cooldown_seconds: int


@dataclass
class Config:
    detection: DetectionConfig
    audio: AudioConfig
    email: EmailConfig
    log_detections: bool
    log_dir: str

    def resolved_log_dir(self) -> Path:
        p = Path(self.log_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    d = raw["detection"]
    a = raw["audio"]
    e = raw["email"]

    return Config(
        detection=DetectionConfig(
            model_path=d["model_path"],
            confidence_threshold=float(d["confidence_threshold"]),
            image_size=int(d["image_size"]),
            capture_width=int(d["capture_width"]),
            capture_height=int(d["capture_height"]),
        ),
        audio=AudioConfig(
            sound_file=a["sound_file"],
            volume=float(a["volume"]),
        ),
        email=EmailConfig(
            host=e["host"],
            port=int(e["port"]),
            username=e["username"],
            password=e["password"],
            from_addr=e["from_addr"],
            to_addr=e["to_addr"],
            cooldown_seconds=int(e.get("cooldown_seconds", 300)),
        ),
        log_detections=bool(raw.get("log_detections", True)),
        log_dir=raw.get("log_dir", "data/logs"),
    )
