#!/bin/bash
# Export ONNX + SCP to Jetson + restart ROS 2 node
# Usage: ./deploy.sh <model_path> [jetson_host]

set -e
MODEL="${1:?Usage: deploy.sh <model_path> [jetson_host]}"
JETSON="${2:-jetson}"
ONNX="models/onnx/model.onnx"

cd "$(dirname "$0")/.."

echo "=== Export ONNX ==="
python main.py export --model "$MODEL" --output "$ONNX"

echo "=== Copy to Jetson ==="
scp "$ONNX" "${JETSON}:/ros2_ws/src/rl_decision/models/model.onnx"

echo "=== Restart node ==="
ssh "$JETSON" "ros2 lifecycle set /rl_decision shutdown && sleep 1 && ros2 lifecycle set /rl_decision configure && ros2 lifecycle set /rl_decision activate"

echo "=== Done ==="
