#!/usr/bin/env bash
# Install and enable the rabbit-deterrent systemd service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_SRC="$PROJECT_ROOT/systemd/rabbit-deterrent.service"
SERVICE_DST=/etc/systemd/system/rabbit-deterrent.service

echo "Installing service to $SERVICE_DST"
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo sed -i "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" "$SERVICE_DST"
sudo sed -i "s|__USER__|${USER}|g" "$SERVICE_DST"

sudo mkdir -p /var/lib/rabbit-deterrent
sudo chown "${USER}:${USER}" /var/lib/rabbit-deterrent

# Allow the service user to manage NetworkManager (needed for hotspot mode)
sudo usermod -a -G netdev "${USER}"

sudo systemctl daemon-reload
sudo systemctl enable rabbit-deterrent
sudo systemctl start rabbit-deterrent

echo "Service status:"
sudo systemctl status rabbit-deterrent --no-pager
