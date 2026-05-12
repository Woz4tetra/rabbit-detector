#!/usr/bin/env bash
# Set up the Moondream2 inference server on the A6000 cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_ROOT/server/.venv"

echo "Creating server venv at $VENV ..."
python3 -m venv "$VENV"

echo "Installing server dependencies ..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$PROJECT_ROOT"           # install rabbit-deterrent base package
"$VENV/bin/pip" install -r "$PROJECT_ROOT/server/requirements.txt"

echo ""
echo "Pre-downloading Moondream2 weights (~4 GB, runs once) ..."
"$VENV/bin/python" -c "
from server.moondream_loader import load_moondream
import sys, os
os.chdir('$PROJECT_ROOT')
sys.path.insert(0, '$PROJECT_ROOT')
model, tokenizer = load_moondream()
print('Model loaded OK on', next(model.parameters()).device)
"

echo ""
echo "Installation complete."
echo "To start the server:"
echo "  cd $PROJECT_ROOT"
echo "  $VENV/bin/uvicorn server.server:app --host 0.0.0.0 --port 8000 --workers 1"
echo ""
echo "To install as a systemd service:"
echo "  bash $SCRIPT_DIR/install_server_service.sh"
