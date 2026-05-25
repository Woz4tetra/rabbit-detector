# Rabbit Deterrent

Detects rabbits in your garden using a Raspberry Pi Zero W and a camera. When a rabbit appears, it plays a deterrent sound and emails you a photo. Runs 24/7 on a 12V battery, day and night.

**Architecture:** the Pi captures frames and sends them over WiFi to an inference server (A6000 cluster) running Moondream2. The server detects rabbits, saves frames, sends email alerts, and serves a live web dashboard. The Pi plays audio locally and records short video clips to SD card on detection.

- Scanning mode: checks for rabbits every 2 seconds
- Alert mode: checks every 1 second until the rabbit leaves, then returns to scanning
- Sends one email per detection event (5-minute cooldown) with a photo attached

**Expected battery life:** ~6-7 days on a 12V 6Ah battery.

---

## Hardware

| Part | Notes |
|------|-------|
| Raspberry Pi Zero W | Must be the W (has WiFi) |
| [Arducam Day-Night Camera (B07X1VGQBL)](https://www.amazon.com/dp/B07X1VGQBL) | OV5647, 5MP, automatic IR-cut filter switching, built-in IR LEDs, M12 lens mount |
| Pi Zero CSI adapter cable | 22-pin mini to 15-pin standard (Arducam ships a standard cable; you need the Pi Zero variant) |
| USB speaker | Any USB speaker; connects via micro-USB OTG adapter |
| Micro-USB OTG adapter | USB-A female to micro-USB male |
| 12V battery + 5V/3A buck converter | e.g., a 12V 6Ah LiFePO4 + Pololu D24V90F5 |
| MicroSD card, 32GB | Samsung PRO Endurance recommended (high write-cycle rated) |
| IP66 waterproof enclosure with clear lid | e.g., Hammond 1554T2GYCL |
| Speaker cable gland | Seal the speaker wire entry point |
| GPU inference server | Any Linux machine with a CUDA GPU; A6000 recommended |

The Pi Zero W has one micro-USB data port. The OTG adapter uses it for the speaker. Do not attach any other USB devices.

---

## Setup

There are two machines to set up: the **inference server** (GPU machine) and the **Raspberry Pi**.

---

### Server setup (GPU machine)

#### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rabbit-detector
```

#### 2. Create `config.yaml`

```bash
cp config.yaml.example config.yaml
nano config.yaml
```

Fill in the `email` section and set `server_app.device` to `cuda:0` (or whichever GPU you want to use).

#### 3. Install server dependencies

This creates a Python venv at `server/.venv` and downloads the Moondream2 weights (~4 GB, one-time):

```bash
bash scripts/install_server.sh
```

#### 4. Test the server manually

```bash
server/.venv/bin/uvicorn server.server:app --host 0.0.0.0 --port 8000 --workers 1
```

In another terminal:

```bash
# Basic health check
python scripts/test_server.py --url http://localhost:8000

# With a rabbit image for a real inference test
python scripts/test_server.py --url http://localhost:8000 --image /path/to/rabbit.jpg
```

#### 5. Install as a systemd service

```bash
bash scripts/install_server_service.sh
```

Monitor it:

```bash
journalctl -u rabbit-server -f
sudo systemctl restart rabbit-server
```

Note the server machine's IP address — you'll need it when configuring the Pi.

---

### Pi setup

#### 1. Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Choose **Raspberry Pi OS Lite (64-bit)** as the OS.
3. Before writing, open settings (Ctrl+Shift+X) and configure:
   - Set a hostname (e.g., `rabbitpi`)
   - Enable SSH with password authentication
   - Set a username and password
   - Enter your WiFi network name and password
4. Write to the MicroSD card and insert it into the Pi.

#### 2. Connect the camera

With the Pi powered off, connect the Arducam to the CSI ribbon cable slot (the small white connector near the power port, not the HDMI port) using the Pi Zero mini-CSI adapter cable. The ribbon's contacts face away from the Pi board.

The Arducam switches automatically between color mode (daylight) and IR greyscale mode (low light). No software configuration is needed for this.

#### 3. Boot and SSH in

Power on the Pi and wait about 60 seconds for first boot. Then:

```bash
ssh <username>@rabbitpi.local
```

If `rabbitpi.local` doesn't resolve, find the Pi's IP from your router's device list and SSH to that directly.

#### 4. Clone the repository

```bash
git clone <your-repo-url>
cd rabbit-detector
```

#### 5. Run one-time Pi optimizations

This tunes the Pi for lower power draw and sets the USB speaker as the default audio device:

```bash
bash scripts/optimize_pi.sh
sudo reboot
```

SSH back in after the reboot:

```bash
ssh <username>@rabbitpi.local
cd rabbit-detector
```

After rebooting, plug in the USB speaker via the OTG adapter. Verify it shows up:

```bash
aplay -l
```

You should see a USB audio device in the list. If it is not card index 1, edit `/etc/asound.conf` and change `card 1` to the correct index.

#### 6. Install Pi software

```bash
bash scripts/install_pi.sh
```

This installs system packages, creates a Python virtual environment, and installs the package. It takes a few minutes on the Pi Zero W.

#### 7. Add a deterrent sound

Place one or more `.wav` files in `data/sounds/`. Any sound works — predator calls, loud noises, etc. On each detection the Pi picks one at random, so adding multiple sounds helps prevent habituation.

```bash
# Copy from your PC
scp your_sound.wav <username>@rabbitpi.local:~/rabbit-detector/data/sounds/
```

Predator calls (hawk screams, coyote calls) are most effective. See [Freesound.org](https://freesound.org) for free CC0 options.

#### 8. Configure

```bash
cp config.yaml.example config.yaml
nano config.yaml
```

Key settings to fill in:

```yaml
server:
  url: "http://<server-ip>:8000"   # IP of your GPU inference server

email:
  host: smtp.gmail.com
  port: 587
  username: you@gmail.com
  password: "xxxx-xxxx-xxxx-xxxx"  # Gmail App Password (not your account password)
  from_addr: you@gmail.com
  to_addr: you@gmail.com
```

For Gmail, generate an App Password at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords). This requires 2-factor authentication to be enabled on your Google account.

#### 9. Verify hardware

Run each test script before installing the service. Fix any failures before continuing.

```bash
# Verify camera captures a frame
python scripts/test_camera.py
# Saves /tmp/test_frame.jpg — copy to your PC to inspect:
# scp <username>@rabbitpi.local:/tmp/test_frame.jpg .

# Verify USB speaker plays sound
python scripts/test_audio.py

# Verify email sends
python scripts/test_notify.py

# Verify the server is reachable and inference works
python scripts/test_server.py --url http://<server-ip>:8000
```

#### 10. Install the service

```bash
bash scripts/install_service.sh
```

The system is now running. Watch the logs:

```bash
journalctl -u rabbit-deterrent -f
```

You should see frames being sent to the server and detection results logged every few seconds.

---

## Hotspot fallback

If the Pi can't connect to WiFi within 2 minutes of booting, it automatically creates a WiFi hotspot so you can SSH in and fix the network config:

- **SSID:** `RabbitDetector`
- **Password:** `rabbitdet`
- **Pi address:** `192.168.4.1`

This is configured under the `hotspot` section of `config.yaml` and can be disabled by setting `enabled: false`.

---

## Enclosure

Mount the Pi, camera, and buck converter inside the waterproof enclosure. Route the speaker cable through a cable gland and seal it. Position the clear lid to face the garden area, with the camera directly behind the clear panel.

Keep the battery outside the sealed enclosure if it needs ventilation, connected via waterproof cable connectors.

---

## Updating

To push code changes to the Pi from your developer machine:

```bash
# Commit and push first, then on the Pi:
git pull
bash scripts/install_pi.sh  # only if dependencies changed
sudo systemctl restart rabbit-deterrent
```

---

## Training your own model

The server uses Moondream2 with a text prompt — no retraining required for most use cases. If you want to tune the detection prompt, edit `detection_prompt` in `config.yaml` under `server_app`.

If you want to fine-tune Moondream2 on your own rabbit images, a training pipeline is in `training/`. Requirements: Python 3.9+, CUDA GPU.

```bash
# Install training dependencies (not on the Pi)
pip install -r training/requirements.txt

# Download dataset from Roboflow (free API key at roboflow.com)
ROBOFLOW_API_KEY=<your_key> python training/download_dataset.py

# Train
python training/train.py --gpus 0

# Validate (target: mAP50 > 0.80)
python training/validate.py
```

---

## Troubleshooting

**Camera not detected**
Run `libcamera-hello` on the Pi. If it fails, check that the CSI ribbon cable is seated fully with the contacts facing away from the board. Ensure `camera_auto_detect=1` is in `/boot/firmware/config.txt` (added by `optimize_pi.sh`).

**No sound from speaker**
Run `aplay -l` and note the card index of the USB audio device. If it is not 1, update `/etc/asound.conf` to use the correct index and reboot. Run `python scripts/test_audio.py` to confirm.

**Email not sending**
Double-check the App Password in `config.yaml` (16 characters, no spaces). Verify your Google account has 2FA enabled. Test manually: `python scripts/test_notify.py`. The cooldown is 5 minutes by default — wait before testing again.

**Pi can't reach the server**
Run `python scripts/test_server.py --url http://<server-ip>:8000` from the Pi. Check that the server firewall allows port 8000 and that `rabbit-server` is running (`systemctl status rabbit-server`).

**Service not starting**
Check `journalctl -u rabbit-deterrent -n 50` for the error. Common causes: `config.yaml` missing (copy from `config.yaml.example`), `data/sounds/` is empty, or the server URL is wrong.

**Hotspot not appearing after moving to a new network**
The hotspot takes up to 2 minutes after boot. If it still doesn't appear, pull the SD card, mount the root partition on another Linux machine, and add a new NetworkManager connection file at `/etc/NetworkManager/system-connections/`. See the [NetworkManager docs](https://networkmanager.dev/docs/api/latest/nm-settings-nmcli.html) for the `.nmconnection` file format.

---

## Battery life

| Optimization | Approximate saving |
|---|---|
| Disable HDMI | 25 mA |
| Disable Bluetooth | 10–15 mA |
| Reduce GPU/CPU frequency | 8–10 mA |
| Disable unused services | 5 mA |

With all optimizations applied, the Pi Zero W draws roughly 70–80 mA at idle. On a 12V 6Ah battery with an 85% efficient buck converter (~12.2 Ah available at 5V), expect 150–175 hours, or about 6–7 days. A larger battery or solar panel extends this proportionally.
