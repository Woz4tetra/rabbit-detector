# Rabbit Deterrent — Project Reference

## What this is

Raspberry Pi Zero W system that detects rabbits via a CSI camera, plays a deterrent sound, and emails a photo. Runs 24/7 on a 12V 6Ah battery (~6-7 days runtime).

## Hardware

- **Pi**: Zero W (single-core ARM1176JZF-S ARMv6, 512MB RAM, WiFi built-in)
- **OS**: Raspbian GNU/Linux 13 (Trixie), kernel 6.12.75+rpt-rpi-v6, Python 3.13.5
- **Camera**: Arducam Day-Night (B07X1VGQBL) — OV5647, 5MP, automatic IR-cut filter switching, built-in IR LEDs, M12 lens mount, MIPI CSI ribbon cable (needs Pi Zero mini-CSI adapter cable)
- **Speaker**: USB speaker via micro-USB OTG adapter (the only USB-A port on the Pi; nothing else can share it)
- **Power**: 12V 6Ah battery → 5V/3A buck converter → Pi
- **No I2S DAC, no Witty Pi HAT**

## Where things run

| Task | Where |
|------|-------|
| `training/` scripts | A6000 cluster (3× GPUs, 48GB VRAM each) |
| `src/rabbit_deterrent/` | Raspberry Pi Zero W |
| `scripts/test_*.py` | Raspberry Pi Zero W |
| `scripts/deploy.sh` | Developer machine (rsyncs to Pi) |

## Deploying changes to the Pi

Do not rsync individual files. Commit changes to the repo, then pull on the Pi:

```bash
# On developer machine: commit and push
git add <files>
git commit -m "..."
git push

# On Pi: pull and reinstall
ssh ben@192.168.50.252
cd ~/rabbit-detector
git pull
bash scripts/install_pi.sh   # if dependencies changed
```

Pi IP: `192.168.50.252`

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

The Pi Zero W runs ARMv6 (`armv6l`). PyPI has no binary wheels for `numpy`, `opencv-python-headless`, `pygame`, or `onnxruntime` on ARMv6. These all come from apt:

```
python3-numpy  python3-opencv  python3-pygame  python3-picamera2
```

The venv must use `--system-site-packages` so these apt packages are importable inside it. The inference engine is `cv2.dnn` (OpenCV's built-in DNN module, v4.10.0 on Trixie), not onnxruntime.

```bash
python3 -m venv --system-site-packages .venv
```

`install_pi.sh` handles this automatically. Trixie enforces PEP 668 (no pip installs into system Python), but the venv pip is unaffected.

## ALSA / USB speaker

`optimize_pi.sh` writes `/etc/asound.conf` pointing at card index 1. If the USB speaker enumerates as a different card index, check with `aplay -l` and update the card number in `/etc/asound.conf`. The card index is not stable across reboots if other USB devices are plugged in (they shouldn't be).

## Config

- `config.yaml` is **gitignored**. It holds real email credentials.
- `config.yaml.example` is the checked-in template.
- All paths in config are resolved relative to the project root by `Config.resolved_*` methods.
- `detection.image_size` (default 320) must match the `imgsz` used in `export_onnx.py`. Changing one without the other degrades accuracy.

## ONNX model output format

The model is exported with `end2end=False` (raw anchors, not decoded boxes). `detector.py` applies NMS manually via `cv2.dnn.NMSBoxes`. Output tensor shape: `[1, num_classes+4, num_anchors]`. Transposing to `[num_anchors, num_classes+4]` before processing. Export uses opset=12 for compatibility with OpenCV 4.5.x; 4.10.0 on Trixie supports higher opsets too.

## State persistence

Detection state (`SCANNING` / `ALERT`) survives reboots via `/var/lib/rabbit-deterrent/state.json`. `install_service.sh` creates this directory with the right ownership. If the file is missing or corrupt, the state machine defaults to `SCANNING`.

## Systemd service

`systemd/rabbit-deterrent.service` contains `__PROJECT_ROOT__` and `__USER__` placeholders. `install_service.sh` substitutes both with `sed` when copying to `/etc/systemd/system/`. Do not edit the installed copy directly; edit the source file and re-run `install_service.sh`.

## Power optimizations

`power.py` runs at startup via `main.py`. On Trixie:
- HDMI: `vcgencmd display_power 0` (replaces `tvservice -o` which was removed)
- ACT LED: `/sys/class/leds/ACT/brightness` (was `led0` on older Pi OS)
- WiFi power save: disabled permanently by `optimize_pi.sh` via `/etc/NetworkManager/conf.d/powersave-off.conf`

All power calls are non-fatal (logged as warnings on failure). The `/boot/firmware/config.txt` changes from `optimize_pi.sh` are permanent and require a reboot. Do NOT set `arm_freq` below 1000MHz — single-core ARMv6 inference is already slow; underclocking makes it unusable.

## Email cooldown

`EmailNotifier` suppresses duplicate emails within `cooldown_seconds` (default 300s). The cooldown resets when the process restarts. A rabbit that triggers an alert shortly after a restart can send an email even if one was sent before the restart.

## Detection loop timing

| State | Sleep between captures |
|-------|----------------------|
| `SCANNING` | 30 seconds |
| `ALERT` | 5 seconds |

The system returns from `ALERT` to `SCANNING` after 3 consecutive frames with no rabbit detected.

## Inference performance target

`test_detector.py` warns if inference exceeds 5 or 30 seconds. On the Pi Zero W (ARMv6, single-core, `cv2.dnn`), expect 15-60 seconds per frame at `image_size=320`. The ARMv6 has no NEON SIMD, so there is no fast path for convolution. This means the ALERT-state 5-second polling interval will be exceeded; the system simply runs as fast as it can. Verify imgsz=320 in both `export_onnx.py` and `config.yaml`.
