# Rabbit Deterrent — Project Reference

## What this is

Raspberry Pi Zero 2W system that detects rabbits via a CSI camera, plays a deterrent sound, and emails a photo. Runs 24/7 on a 12V 6Ah battery (~6-7 days runtime).

## Hardware

- **Pi**: Zero 2W (quad-core Cortex-A53, 512MB RAM, WiFi built-in)
- **Camera**: Arducam Day-Night (B07X1VGQSL) — OV5647, 5MP, automatic IR-cut filter switching, built-in IR LEDs, M12 lens mount, MIPI CSI ribbon cable (needs Pi Zero mini-CSI adapter cable)
- **Speaker**: USB speaker via micro-USB OTG adapter (the only USB-A port on the Pi; nothing else can share it)
- **Power**: 12V 6Ah battery → 5V/3A buck converter → Pi
- **No I2S DAC, no Witty Pi HAT**

## Where things run

| Task | Where |
|------|-------|
| `training/` scripts | A6000 cluster (3× GPUs, 48GB VRAM each) |
| `src/rabbit_deterrent/` | Raspberry Pi Zero 2W |
| `scripts/test_*.py` | Raspberry Pi Zero 2W |
| `scripts/deploy.sh` | Developer machine (rsyncs to Pi) |

## Key commands

```bash
# Training (on A6000 cluster)
pip install -r training/requirements.txt
ROBOFLOW_API_KEY=<key> python training/download_dataset.py
python training/train.py                  # DDP across 3 GPUs, ~5 min
python training/validate.py               # check mAP50 > 0.80
python training/export_onnx.py            # writes data/models/rabbit_detector.onnx

# Deploy to Pi
cp config.yaml.example config.yaml        # fill in email credentials
PI_HOST=raspberrypi.local bash scripts/deploy.sh

# Pi: one-time setup (then reboot)
bash scripts/optimize_pi.sh
bash scripts/install_pi.sh
bash scripts/install_service.sh

# Pi: hardware verification
python scripts/test_camera.py             # saves /tmp/test_frame.jpg
python scripts/test_audio.py              # plays deterrent.wav
python scripts/test_detector.py --image <rabbit.jpg>
python scripts/test_notify.py             # sends test email

# Pi: service monitoring
journalctl -u rabbit-deterrent -f
sudo systemctl restart rabbit-deterrent
```

## Python environment on the Pi

Use `--system-site-packages` when creating the venv. `picamera2` is not on PyPI; it comes from the system apt package `python3-picamera2`. Without the flag, imports fail.

```bash
python3 -m venv --system-site-packages .venv
```

`install_pi.sh` handles this automatically.

## ALSA / USB speaker

`optimize_pi.sh` writes `/etc/asound.conf` pointing at card index 1. If the USB speaker enumerates as a different card index, check with `aplay -l` and update the card number in `/etc/asound.conf`. The card index is not stable across reboots if other USB devices are plugged in (they shouldn't be).

## Config

- `config.yaml` is **gitignored**. It holds real email credentials.
- `config.yaml.example` is the checked-in template.
- All paths in config are resolved relative to the project root by `Config.resolved_*` methods.
- `detection.image_size` (default 320) must match the `imgsz` used in `export_onnx.py`. Changing one without the other degrades accuracy.

## ONNX model output format

The model is exported with `end2end=False` (raw anchors, not decoded boxes). `detector.py` applies NMS manually. Output tensor shape: `[1, num_classes+4, num_anchors]`. Transposing to `[num_anchors, num_classes+4]` before processing.

## State persistence

Detection state (`SCANNING` / `ALERT`) survives reboots via `/var/lib/rabbit-deterrent/state.json`. `install_service.sh` creates this directory with the right ownership. If the file is missing or corrupt, the state machine defaults to `SCANNING`.

## Systemd service

`systemd/rabbit-deterrent.service` contains the literal string `__PROJECT_ROOT__`. `install_service.sh` substitutes the real path with `sed` when it copies the file to `/etc/systemd/system/`. Do not edit the installed copy directly; edit the source file and re-run `install_service.sh`.

## Power optimizations

`power.py` runs at startup via `main.py` and disables HDMI (`tvservice -o`), the ACT LED, and WiFi power-save mode. These are non-fatal if they fail (logged as warnings). The `/boot/firmware/config.txt` changes from `optimize_pi.sh` are permanent and require a reboot.

## Email cooldown

`EmailNotifier` suppresses duplicate emails within `cooldown_seconds` (default 300s). The cooldown resets when the process restarts. A rabbit that triggers an alert shortly after a restart can send an email even if one was sent before the restart.

## Detection loop timing

| State | Sleep between captures |
|-------|----------------------|
| `SCANNING` | 30 seconds |
| `ALERT` | 5 seconds |

The system returns from `ALERT` to `SCANNING` after 3 consecutive frames with no rabbit detected.

## Inference performance target

`test_detector.py` warns if inference exceeds 5 seconds. At `image_size=320` on a Pi Zero 2W, expect 2-4 seconds. If it runs slower, verify the model is at imgsz=320 (not 640) and that only `CPUExecutionProvider` is active.
