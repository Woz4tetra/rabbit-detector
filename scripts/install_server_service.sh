#!/usr/bin/env bash
# Install and enable the rabbit-server systemd service on the A6000 cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_SRC="$PROJECT_ROOT/systemd/rabbit-server.service"
SERVICE_DST=/etc/systemd/system/rabbit-server.service

echo "Installing service to $SERVICE_DST"
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo sed -i "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" "$SERVICE_DST"
sudo sed -i "s|__USER__|${USER}|g" "$SERVICE_DST"

sudo systemctl daemon-reload
sudo systemctl enable rabbit-server
sudo systemctl start rabbit-server

echo "Service status:"
sudo systemctl status rabbit-server --no-pager
