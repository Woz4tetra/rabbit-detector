#!/usr/bin/env bash
# SSH local port forward: localhost:8000 -> pi:8000
LOCAL_PORT=${1:-8000}
REMOTE_PORT=${2:-8000}
PI=ben@192.168.50.252

echo "Tunneling localhost:$LOCAL_PORT -> $PI:$REMOTE_PORT"
echo "Open http://localhost:$LOCAL_PORT in your browser"
echo "Press Ctrl+C to stop."

ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "$PI"
