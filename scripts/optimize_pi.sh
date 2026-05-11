#!/usr/bin/env bash
# One-time power optimization for Raspberry Pi Zero W.
# Run once after first boot, then reboot.
set -euo pipefail

CONFIG=/boot/firmware/config.txt
# Older Pi OS uses /boot/config.txt
[ -f "$CONFIG" ] || CONFIG=/boot/config.txt

echo "Applying power optimizations to $CONFIG"

append_if_missing() {
    local line="$1"
    grep -qxF "$line" "$CONFIG" || echo "$line" | sudo tee -a "$CONFIG" > /dev/null
}

# 128MB minimum for libcamera/Picamera2 to load the unicam driver.
# Lower values cause the firmware to fall back to bcm2835_unicam_legacy,
# which is V4L2-only and invisible to libcamera.
append_if_missing "gpu_mem=128"
# Do not set arm_freq below 1000MHz on the Zero W — single-core ARMv6 inference
# is already slow (~15-60s per frame); underclocking makes it unusable.
append_if_missing "dtoverlay=disable-bt"
append_if_missing "camera_auto_detect=1"

echo "Disabling WiFi power save (via NetworkManager)..."
sudo mkdir -p /etc/NetworkManager/conf.d
cat <<'EOF' | sudo tee /etc/NetworkManager/conf.d/powersave-off.conf > /dev/null
[connection]
wifi.powersave = 2
EOF

echo "Disabling unused systemd services..."
for svc in bluetooth avahi-daemon triggerhappy ModemManager; do
    if systemctl list-unit-files "${svc}.service" &>/dev/null; then
        sudo systemctl disable --now "$svc" || true
    fi
done

echo "Setting ALSA default to USB audio device..."
cat <<'EOF' | sudo tee /etc/asound.conf
pcm.!default {
    type hw
    card 1
}
ctl.!default {
    type hw
    card 1
}
EOF
echo "(Card index may differ — run 'aplay -l' after plugging in USB speaker to verify)"

echo ""
echo "Done. Reboot for changes to take effect: sudo reboot"
