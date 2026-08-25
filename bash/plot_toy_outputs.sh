#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p .mplconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-.mplconfig}"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" ODtoy/plot_toy_outputs.py "$@"
