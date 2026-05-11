#!/usr/bin/env bash
# Sync code and model from this machine to the Raspberry Pi.
# Usage: bash scripts/deploy.sh [--model-only]
#
# Environment variables:
#   PI_HOST  (default: raspberrypi.local)
#   PI_USER  (default: pi)
#   PI_PATH  (default: ~/rabbit-deterrent)
set -euo pipefail

PI_HOST="${PI_HOST:-raspberrypi.local}"
PI_USER="${PI_USER:-pi}"
PI_PATH="${PI_PATH:-~/rabbit-deterrent}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_ONLY=false
for arg in "$@"; do
    [ "$arg" = "--model-only" ] && MODEL_ONLY=true
done

if [ "$MODEL_ONLY" = false ]; then
    echo "Syncing code to ${PI_USER}@${PI_HOST}:${PI_PATH}/"
    rsync \
        --archive \
        --verbose \
        --delete \
        --filter=':- .gitignore' \
        --exclude='.git/' \
        --exclude='training/data/' \
        --exclude='training/runs/' \
        --exclude='training/.venv/' \
        "${PROJECT_ROOT}/" \
        "${PI_USER}@${PI_HOST}:${PI_PATH}/"
fi

echo "Syncing model..."
rsync \
    --archive \
    --verbose \
    "${PROJECT_ROOT}/data/models/" \
    "${PI_USER}@${PI_HOST}:${PI_PATH}/data/models/"

if [ "$MODEL_ONLY" = false ]; then
    echo "Re-installing package on Pi..."
    ssh "${PI_USER}@${PI_HOST}" \
        "cd ${PI_PATH} && .venv/bin/pip install -e '.[pi]' -q"

    echo "Restarting service..."
    ssh "${PI_USER}@${PI_HOST}" \
        "sudo systemctl restart rabbit-deterrent"
fi

echo "Deploy complete."
