#!/bin/bash
# Convenience wrapper: train PPO with default args
cd "$(dirname "$0")/.."
python main.py train --timesteps 2000000 --n-envs 4 "$@"
