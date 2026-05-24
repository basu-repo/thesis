#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/basudeo/miniconda3/envs/tct/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "/home/basudeo/Documents/Thesis/simulation_dashboard/dashboard_server.py" "$@"
