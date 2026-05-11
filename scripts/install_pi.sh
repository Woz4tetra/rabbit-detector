#!/usr/bin/env bash
# Install system packages and Python environment on the Raspberry Pi.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Updating apt..."
sudo apt-get update -qq

echo "Installing system dependencies..."
# numpy, opencv, and pygame are installed from apt because pip has no armv6l wheels.
# The venv uses --system-site-packages so they are importable inside it.
sudo apt-get install -y \
    python3-venv \
    python3-pip \
    python3-libcamera \
    python3-picamera2 \
    python3-numpy \
    python3-opencv \
    python3-pygame \
    libcamera-dev \
    libasound2-dev

echo "Creating Python venv (with system site-packages for picamera2)..."
cd "$PROJECT_ROOT"
python3 -m venv --system-site-packages .venv

echo "Installing Python packages..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[pi]"

echo ""
echo "Installation complete. Next: copy config.yaml.example -> config.yaml and fill in credentials."
echo "Then run: bash scripts/install_service.sh"
