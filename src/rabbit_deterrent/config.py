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
    frame_rate: float  # camera stream fps during ALERT; also sets video output fps

    def resolved_model_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class AudioConfig:
    sounds_dir: str
    volume: float

    def resolved_sounds_dir(self) -> Path:
        p = Path(self.sounds_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class EmailConfig:
    enabled: bool
    host: str
    port: int
    username: str
    password: str
    from_addr: str
    to_addr: str
    cooldown_seconds: int


@dataclass
class StorageConfig:
    save_images: bool
    image_dir: str
    max_images: int  # 0 = unlimited
    save_video: bool
    video_dir: str

    def resolved_image_dir(self) -> Path:
        p = Path(self.image_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def resolved_video_dir(self) -> Path:
        p = Path(self.video_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Config:
    detection: DetectionConfig
    audio: AudioConfig
    email: EmailConfig
    storage: StorageConfig
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
    s = raw.get("storage", {})

    return Config(
        detection=DetectionConfig(
            model_path=d["model_path"],
            confidence_threshold=float(d["confidence_threshold"]),
            image_size=int(d["image_size"]),
            capture_width=int(d["capture_width"]),
            capture_height=int(d["capture_height"]),
            frame_rate=float(d.get("frame_rate", 1.0)),
        ),
        audio=AudioConfig(
            sounds_dir=a.get("sounds_dir", "data/sounds"),
            volume=float(a["volume"]),
        ),
        email=EmailConfig(
            enabled=bool(e.get("enabled", True)),
            host=e.get("host", ""),
            port=int(e.get("port", 587)),
            username=e.get("username", ""),
            password=e.get("password", ""),
            from_addr=e.get("from_addr", ""),
            to_addr=e.get("to_addr", ""),
            cooldown_seconds=int(e.get("cooldown_seconds", 300)),
        ),
        storage=StorageConfig(
            save_images=bool(s.get("save_images", False)),
            image_dir=s.get("image_dir", "data/images"),
            max_images=int(s.get("max_images", 1000)),
            save_video=bool(s.get("save_video", False)),
            video_dir=s.get("video_dir", "data/videos"),
        ),
        log_detections=bool(raw.get("log_detections", True)),
        log_dir=raw.get("log_dir", "data/logs"),
    )
