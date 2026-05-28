from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class CameraConfig:
    max_exposure_seconds: float = 3.0
    ae_enable: bool = True
    exposure_time_us: int = 20000
    analogue_gain: float = 1.0
    awb_enable: bool = True
    awb_mode: int = 0
    red_gain: float = 1.5
    blue_gain: float = 1.5
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    noise_reduction_mode: int = 1


@dataclass
class DetectionConfig:
    confidence_threshold: float
    capture_width: int
    capture_height: int
    scanning_interval_seconds: float  # pause between server queries in SCANNING state
    alert_poll_interval_seconds: float  # pause between server queries in ALERT state


@dataclass
class ServerConfig:
    url: str  # e.g. "http://192.168.50.XXX:8000"
    timeout_seconds: float = 10.0
    max_failures: int = 5  # consecutive failures before entering OFFLINE mode
    retry_delay_seconds: float = 5.0


@dataclass
class ClipConfig:
    enabled: bool
    clip_dir: str
    max_clips: int  # 0 = unlimited; prune oldest when exceeded
    pre_roll_frames: int  # frames buffered before detection starts
    max_clip_seconds: float  # hard stop per clip

    def resolved_clip_dir(self) -> Path:
        p = Path(self.clip_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class HotspotConfig:
    enabled: bool = True
    timeout_seconds: float = 120.0
    ssid: str = "RabbitDetector"
    password: str = "rabbitdet"  # min 8 chars required by WPA2


@dataclass
class AudioConfig:
    sounds_dir: str
    volume: float
    alert_interval_seconds: float = 30.0  # min seconds between audio plays while rabbit is present

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
class Config:
    detection: DetectionConfig
    server: ServerConfig
    clip: ClipConfig
    audio: AudioConfig
    email: EmailConfig
    hotspot: HotspotConfig
    camera_day: CameraConfig
    camera_night: CameraConfig
    log_detections: bool
    log_dir: str

    def resolved_log_dir(self) -> Path:
        p = Path(self.log_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _load_camera_profile(section: dict) -> CameraConfig:
    return CameraConfig(
        max_exposure_seconds=float(section.get("max_exposure_seconds", 3.0)),
        ae_enable=bool(section.get("ae_enable", True)),
        exposure_time_us=int(section.get("exposure_time_us", 20000)),
        analogue_gain=float(section.get("analogue_gain", 1.0)),
        awb_enable=bool(section.get("awb_enable", True)),
        awb_mode=int(section.get("awb_mode", 0)),
        red_gain=float(section.get("red_gain", 1.5)),
        blue_gain=float(section.get("blue_gain", 1.5)),
        brightness=float(section.get("brightness", 0.0)),
        contrast=float(section.get("contrast", 1.0)),
        saturation=float(section.get("saturation", 1.0)),
        sharpness=float(section.get("sharpness", 1.0)),
        noise_reduction_mode=int(section.get("noise_reduction_mode", 1)),
    )


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    d = raw.get("detection", {})
    s = raw.get("server", {})
    c = raw.get("clip", {})
    a = raw.get("audio", {})
    e = raw.get("email", {})
    h = raw.get("hotspot", {})
    cam_day = raw.get("camera_day", {})
    cam_night = raw.get("camera_night", {})

    return Config(
        detection=DetectionConfig(
            confidence_threshold=float(d.get("confidence_threshold", 0.5)),
            capture_width=int(d.get("capture_width", 640)),
            capture_height=int(d.get("capture_height", 480)),
            scanning_interval_seconds=float(d.get("scanning_interval_seconds", 2.0)),
            alert_poll_interval_seconds=float(d.get("alert_poll_interval_seconds", 1.0)),
        ),
        server=ServerConfig(
            url=s["url"],
            timeout_seconds=float(s.get("timeout_seconds", 10.0)),
            max_failures=int(s.get("max_failures", 5)),
            retry_delay_seconds=float(s.get("retry_delay_seconds", 5.0)),
        ),
        clip=ClipConfig(
            enabled=bool(c.get("enabled", True)),
            clip_dir=c.get("clip_dir", "data/clips"),
            max_clips=int(c.get("max_clips", 50)),
            pre_roll_frames=int(c.get("pre_roll_frames", 3)),
            max_clip_seconds=float(c.get("max_clip_seconds", 120.0)),
        ),
        audio=AudioConfig(
            sounds_dir=a.get("sounds_dir", "data/sounds"),
            volume=float(a.get("volume", 0.9)),
            alert_interval_seconds=float(a.get("alert_interval_seconds", 30.0)),
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
        hotspot=HotspotConfig(
            enabled=bool(h.get("enabled", True)),
            timeout_seconds=float(h.get("timeout_seconds", 120.0)),
            ssid=h.get("ssid", "RabbitDetector"),
            password=h.get("password", "rabbitdet"),
        ),
        camera_day=_load_camera_profile(cam_day),
        camera_night=_load_camera_profile(cam_night),
        log_detections=bool(raw.get("log_detections", True)),
        log_dir=raw.get("log_dir", "data/logs"),
    )
