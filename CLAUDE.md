# Rabbit Deterrent — Project Reference

## What this is

Raspberry Pi Zero W WiFi camera client + A6000 cluster inference server. The Pi captures frames and POSTs them to the server; the server runs Moondream2 to detect rabbits, sends email alerts, and serves a live web dashboard. The Pi plays audio locally and records short video clips to SD card on detection. Runs 24/7 on a 12V 6Ah battery (~6-7 days runtime).

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
| `server/` | A6000 cluster (Moondream2 inference + web dashboard + email) |
| `src/rabbit_deterrent/` | Raspberry Pi Zero W (camera capture, audio, clip recording) |
| `scripts/test_*.py` | Raspberry Pi Zero W (except `test_server.py` — run anywhere) |
| `scripts/deploy.sh` | Developer machine (rsyncs to Pi) |

## Deploying changes to the Pi

Do not rsync individual files. Commit changes to the repo, then pull on the Pi:

```bash
# On developer machine: commit and push
git add <files>
git commit -m "..."
git push

# Reach the Pi via the megamind jump host (the Pi is not reachable directly
# from the developer machine — pathfinder key lives on megamind, not locally):
ssh ben@megamind
ssh -i ~/.ssh/pathfinder ben@192.168.1.174   # from megamind
cd ~/rabbit-detector
git pull
bash scripts/install_pi.sh   # if dependencies changed
```

For one-shot commands from the developer machine:

```bash
ssh ben@megamind 'ssh -i ~/.ssh/pathfinder ben@192.168.1.174 "<command>"'
```

To copy a file to the Pi, stage it on megamind first (ProxyJump can't reach
the Pi because the pathfinder key isn't local):

```bash
scp <file> ben@megamind:/tmp/
ssh ben@megamind 'scp -i ~/.ssh/pathfinder /tmp/<file> ben@192.168.1.174:<dest> && rm /tmp/<file>'
```

Pi IP: `192.168.1.174` (jump host: `megamind`, inner key: `~/.ssh/pathfinder` on megamind)

## Key commands

```bash
# Server: one-time setup on A6000 cluster
cp config.yaml.example config.yaml        # fill in email credentials and server_app.device
bash scripts/install_server.sh            # creates server/.venv, downloads Moondream2 (~4 GB)
bash scripts/install_server_service.sh    # installs systemd service

# Server: start manually (dev/test)
server/.venv/bin/uvicorn server.server:app --host 0.0.0.0 --port 8000 --workers 1

# Server: test endpoints
python scripts/test_server.py --url http://localhost:8000
python scripts/test_server.py --url http://localhost:8000 --image /path/to/rabbit.jpg

# Server: service monitoring
journalctl -u rabbit-server -f
sudo systemctl restart rabbit-server

# Pi: one-time setup (then reboot)
bash scripts/optimize_pi.sh
bash scripts/install_pi.sh
bash scripts/install_service.sh

# Pi: hardware verification
python scripts/test_camera.py             # saves /tmp/test_frame.jpg
python scripts/test_audio.py              # plays deterrent.wav
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

USB speaker hardware spec (USB2.0 Device, card 1): S16_LE, 48000 Hz, stereo only. Pygame mixer and `convert_sound.py` are both configured to match this exactly. Mismatching frequency or channels forces ALSA to resample in software, causing crackling.

## Config

- `config.yaml` is **gitignored**. It holds real email credentials.
- `config.yaml.example` is the checked-in template.
- All paths in config are resolved relative to the project root by `Config.resolved_*` methods.
- Pi reads: `detection`, `server`, `clip`, `audio`, `email`, `log_detections`, `log_dir`
- Server reads: `email`, `server_app` (everything else is ignored by the server)

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

`gpu_mem` must be at least 128. Values below that cause the firmware to load `bcm2835_unicam_legacy` (V4L2-only) instead of the libcamera unicam driver, making the camera invisible to Picamera2. The Pi Zero W has 512MB RAM so 128MB GPU split leaves 384MB for the CPU.

## Email cooldown

`EmailNotifier` suppresses duplicate emails within `cooldown_seconds` (default 300s). The cooldown resets when the process restarts. A rabbit that triggers an alert shortly after a restart can send an email even if one was sent before the restart.

## Detection loop timing

| State | Poll interval | Server round-trip | Effective cycle |
|-------|--------------|-------------------|----------------|
| `SCANNING` | 2s | ~1–2s | ~2–4s per check |
| `ALERT` | 1s | ~1–2s | ~1–2s per check |
| `OFFLINE` | retry_delay_seconds (5s) | — | health-check only |

The system returns from `ALERT` to `SCANNING` after 3 consecutive clear responses.

## Rabbit deterrent sound research

**Target species**: Eastern cottontail (*Sylvilagus floridanus*) — solitary, flees to cover when startled. If in Europe/Australia, target is European rabbit (*Oryctolagus cuniculus*), which lives in warrens and has a group alarm response. Both respond to the same predator sounds.

**Hearing range**: 96 Hz – 49,000 Hz. Sharpest sensitivity at 1,000–16,000 Hz. This overlaps directly with audible predator calls, so a standard USB speaker is the right tool.

**What works**:
- Predator vocalizations in the audible range: hawk screams, fox barks, coyote calls, owl hoots
- Red-tailed hawk scream is ideal for cottontails — direct aerial predator, sharp call peaks in the 1–4 kHz sensitivity band
- Distress calls (baby rabbit screams) are effective but exceed 50,000 Hz — a USB speaker cannot reproduce them

**What doesn't work long-term**:
- Ultrasonic devices (15,000–25,000 Hz): rabbits habituate within days to weeks
- Any single fixed sound played repeatedly: rabbits adapt quickly

**Habituation mitigation**: Rotate between 2–3 different sounds (e.g., hawk + coyote) and randomize playback. `AudioPlayer` already does this — drop multiple `.wav` files into `data/sounds/` and each detection plays a random one.

**Sound sources (free)**:
- [Freesound.org — Red-Tailed Hawk, CC0](https://freesound.org/people/craigsmith/sounds/479610/)
- [Pixabay — Red-Tailed Hawk sounds, royalty-free](https://pixabay.com/sound-effects/search/red-tailed-hawk/)
- [SoundBible — Hawk sounds, WAV + MP3](https://soundbible.com/tags-hawk.html)

## Inference performance target

Moondream2 on a single A6000 GPU: ~0.3–0.8s per frame. Total server round-trip including network: ~1–2s. The Pi no longer does local inference.
