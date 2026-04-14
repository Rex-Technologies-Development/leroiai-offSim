#!/bin/bash
# Pull match logs from Jetson to data/matches/
# Usage: ./pull_match_data.sh [jetson_host]

JETSON="${1:-jetson}"
DEST="$(dirname "$0")/../matches/"

echo "Pulling match data from ${JETSON}:/data/match_logs/ ..."
scp "${JETSON}:/data/match_logs/*.npz" "$DEST"
echo "Done. Files in $DEST:"
ls -la "$DEST"/*.npz 2>/dev/null || echo "  (no .npz files found)"
