#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p .mplconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-.mplconfig}"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" graphs/graph_start_end/plot_start_end_aftershutdown_simple.py \
  --alpha_ratio_name 2 --flow_ratio_name 1 --period preCovid \
  --init_method airport5k --beta_density fit --process_threshold 120 \
  --process_method 33 --process_inf --Rtype empirical --Pratio 0p632 \
  --output_root .
