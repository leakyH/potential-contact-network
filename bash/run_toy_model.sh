#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p toyoutput/pkls toyoutput/scalars tmp .mplconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-.mplconfig}"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" ODtoy/toy_daily_wave_multinomial_PCF_over_flow.py
