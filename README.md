# Rabbit Deterrent

Detects rabbits in your garden using a Raspberry Pi Zero W and a camera. When a rabbit appears, it plays a deterrent sound and emails you a photo. Runs 24/7 on a 12V battery, day and night.

- Normal mode: checks for rabbits every 30 seconds
- Alert mode: checks every 5 seconds until the rabbit leaves, then returns to normal
- Sends one email per detection event (5-minute cooldown) with a photo attached

**Expected battery life:** ~6-7 days on a 12V 6Ah battery.

---

## Hardware

| Part | Notes |
|------|-------|
| Raspberry Pi Zero W | Must be the W (has WiFi and enough CPU for inference) |
| [Arducam Day-Night Camera (B07X1VGQSL)](https://www.amazon.com/dp/B07X1VGQSL) | OV5647, 5MP, automatic IR-cut filter switching, built-in IR LEDs, M12 lens mount |
| Pi Zero CSI adapter cable | 22-pin mini to 15-pin standard (Arducam ships a standard cable; you need the Pi Zero variant) |
| USB speaker | Any USB speaker; connects via micro-USB OTG adapter |
| Micro-USB OTG adapter | USB-A female to micro-USB male |
| 12V battery + 5V/3A buck converter | e.g., a 12V 6Ah LiFePO4 + Pololu D24V90F5 |
| MicroSD card, 32GB | Samsung PRO Endurance recommended (high write-cycle rated) |
| IP66 waterproof enclosure with clear lid | e.g., Hammond 1554T2GYCL |
| Speaker cable gland | Seal the speaker wire entry point |

The Pi Zero W has one micro-USB data port. The OTG adapter uses it for the speaker. Do not attach any other USB devices.

---

## Setup

### 1. Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Choose **Raspberry Pi OS Lite (64-bit)** as the OS.
3. Before writing, click the gear icon (or press Ctrl+Shift+X) and configure:
   - Set a hostname (e.g., `rabbitpi`)
   - Enable SSH with password authentication
   - Set username `pi` and a password
   - Enter your WiFi network name and password
4. Write to the MicroSD card and insert it into the Pi.

### 2. Connect the camera

With the Pi powered off, connect the Arducam to the CSI ribbon cable slot (the small white connector near the power port, not the HDMI port) using the Pi Zero mini-CSI adapter cable. The ribbon's contacts face away from the Pi board.

The Arducam switches automatically between color mode (daylight) and IR greyscale mode (low light). No software configuration is needed for this.

### 3. Boot and SSH in

Power on the Pi and wait about 60 seconds for first boot. Then:

```bash
ssh pi@rabbitpi.local
```

If `rabbitpi.local` doesn't resolve, find the IP address from your router's device list and SSH to that directly.

### 4. Clone the repository

```bash
git clone https://github.com/woz4tetra/rabbit-deterrent.git
cd rabbit-deterrent
```

### 5. Run one-time Pi optimization

This tunes the Pi for lower power draw and sets the USB speaker as the default audio device.

```bash
bash scripts/optimize_pi.sh
sudo reboot
```

SSH back in after the reboot:

```bash
ssh pi@rabbitpi.local
cd rabbit-deterrent
```

After rebooting, plug in the USB speaker via the OTG adapter. Verify it shows up:

```bash
aplay -l
```

You should see a USB audio device in the list. If it is not card index 1, edit `/etc/asound.conf` and change `card 1` to the correct index.

### 6. Install software

```bash
bash scripts/install_pi.sh
```

This installs system packages, creates a Python virtual environment, and installs the package. It takes a few minutes on the Pi Zero W.

### 7. Download the detection model

Go to the [Releases page](https://github.com/woz4tetra/rabbit-deterrent/releases) and download `rabbit_detector.onnx` from the latest release. Copy it to the Pi:

```bash
# Run this on your PC, not the Pi
scp rabbit_detector.onnx pi@rabbitpi.local:~/rabbit-deterrent/data/models/
```

Or download it directly on the Pi if you have a browser or `wget`:

```bash
# On the Pi — replace the URL with the actual release asset URL
wget -O data/models/rabbit_detector.onnx \
  https://github.com/woz4tetra/rabbit-deterrent/releases/download/v1.0/rabbit_detector.onnx
```

### 8. Add a deterrent sound

Place a `.wav` file at `data/sounds/deterrent.wav`. Any sound works: a predator call, a loud noise, an ultrasonic tone. The file must be WAV format. MP3 is not supported.

```bash
# Copy from your PC
scp your_sound.wav pi@rabbitpi.local:~/rabbit-deterrent/data/sounds/deterrent.wav
```

### 9. Configure email

```bash
cp config.yaml.example config.yaml
nano config.yaml
```

Fill in the `email` section. For Gmail, use an [App Password](https://myaccount.google.com/apppasswords), not your account password. App Passwords require 2-factor authentication to be enabled on your Google account.

```yaml
email:
  host: smtp.gmail.com
  port: 587
  username: you@gmail.com
  password: "xxxx-xxxx-xxxx-xxxx"   # App Password, 16 characters
  from_addr: you@gmail.com
  to_addr: you@gmail.com            # Can be any address
  cooldown_seconds: 300
```

### 10. Verify hardware

Run each test script before installing the service. Fix any failures before continuing.

```bash
# Verify camera captures a frame
python scripts/test_camera.py
# Output: saves /tmp/test_frame.jpg — copy it back to your PC to inspect:
# scp pi@rabbitpi.local:/tmp/test_frame.jpg .

# Verify USB speaker plays sound
python scripts/test_audio.py

# Verify model loads and runs inference (use a rabbit photo for a real check)
python scripts/test_detector.py
python scripts/test_detector.py --image /path/to/rabbit.jpg

# Verify email sends
python scripts/test_notify.py
```

### 11. Install the service

```bash
bash scripts/install_service.sh
```

The system is now running. Watch the logs:

```bash
journalctl -u rabbit-deterrent -f
```

You should see a `No rabbit detected` or similar log line every 30 seconds.

---

## Enclosure

Mount the Pi, camera, and buck converter inside the waterproof enclosure. Route the speaker cable through a cable gland and seal it. Position the clear lid to face the garden area, with the camera directly behind the clear panel.

Keep the battery outside the sealed enclosure if it needs ventilation, connected via waterproof cable connectors.

---

## Updating

To update the software from your PC after making code changes:

```bash
bash scripts/deploy.sh
```

To update only the model file (after retraining):

```bash
bash scripts/deploy.sh --model-only
```

---

## Training your own model

If you want to retrain the model (for example, to improve accuracy in your specific lighting or background conditions), you need a machine with a GPU. The default model was trained on a Roboflow dataset of ~4,000 rabbit images.

Requirements: Python 3.9+, CUDA-capable GPU.

```bash
# Install training dependencies (not on the Pi)
pip install -r training/requirements.txt

# Download the dataset from Roboflow
# Get a free API key at https://roboflow.com
ROBOFLOW_API_KEY=<your_key> python training/download_dataset.py

# Train (defaults to GPUs 0,1,2 — adjust --gpus for your setup)
python training/train.py --gpus 0

# Check accuracy on the test split (target: mAP50 > 0.80)
python training/validate.py

# Export to ONNX for Pi deployment
python training/export_onnx.py
```

Training automatically applies IR simulation to 40% of batches so the model handles both daytime color images and nighttime IR greyscale images from the Arducam's automatic night mode.

The exported model lands at `data/models/rabbit_detector.onnx`. Deploy it with:

```bash
bash scripts/deploy.sh --model-only
```

---

## Troubleshooting

**Camera not detected**
Run `libcamera-hello` on the Pi. If it fails, check that the CSI ribbon cable is seated fully with the contacts facing away from the board. Ensure `camera_auto_detect=1` is in `/boot/firmware/config.txt` (added by `optimize_pi.sh`).

**No sound from speaker**
Run `aplay -l` and note the card index of the USB audio device. If it is not 1, update `/etc/asound.conf` to use the correct index and reboot. Run `python scripts/test_audio.py` to confirm.

**Email not sending**
Double-check the App Password in `config.yaml` (16 characters, no spaces). Verify your Google account has 2FA enabled. Test manually: `python scripts/test_notify.py`. The cooldown is 5 minutes by default; if you just sent a test email, wait before testing again.

**Inference too slow (over 5 seconds)**
Confirm the model was exported at `imgsz=320`, not 640. Check `detection.image_size` in `config.yaml` matches. Run `python scripts/test_detector.py` and check the reported latency.

**Service not starting**
Check `journalctl -u rabbit-deterrent -n 50` for the error. Common causes: `config.yaml` missing (copy from `config.yaml.example`), model file missing from `data/models/`, or `data/sounds/deterrent.wav` missing.

**High false positive rate**
Lower `detection.confidence_threshold` in `config.yaml` if you're missing rabbits, raise it if other animals are triggering alerts. Default is 0.45. Values between 0.35 and 0.60 are typical.

---

## Battery life

| Optimization | Approximate saving |
|---|---|
| Disable HDMI (`tvservice -o`) | 25mA |
| Disable Bluetooth (`dtoverlay=disable-bt`) | 10-15mA |
| Reduce GPU/CPU frequency | 8-10mA |
| Disable unused services | 5mA |

With all optimizations applied, the Pi Zero W draws roughly 70-80mA at idle. On a 12V 6Ah battery with an 85% efficient buck converter (~12.2Ah available at 5V), expect 150-175 hours, or about 6-7 days. A larger battery or solar panel extends this proportionally.
