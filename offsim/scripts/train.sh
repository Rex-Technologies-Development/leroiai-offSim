#!/usr/bin/env bash
set -euo pipefail
python offsim/main.py train --chassis "${CHASSIS:-tank}" --timesteps "${TIMESTEPS:-1000000}" "$@"
