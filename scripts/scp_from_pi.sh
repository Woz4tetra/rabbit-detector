#!/bin/bash
# Usage: scp_from_pi.sh <remote_path> [local_dest]
#   remote_path: absolute path on the Pi, e.g. /tmp/test_frame.jpg
#   local_dest:  local destination file or directory (default: current dir)
set -e

REMOTE_PATH="${1:?Usage: $0 <remote_path> [local_dest]}"
LOCAL_DEST="${2:-.}"

scp -J ben@megamind -i ~/.ssh/pathfinder -r "ben@192.168.1.174:${REMOTE_PATH}" "${LOCAL_DEST}"
