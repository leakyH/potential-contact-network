#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p graphs/graph_start_end/artifacts .mplconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-.mplconfig}"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" graphs/graph_start_end/plot_multiple_process_peak_flow.py \
  --alpha_ratio_name 2 --flow_ratio_name 1 --period preCovid \
  --init_method airport5k --beta_density fit --process_threshold 120 \
  --process_inf --Rtype empirical --Pratio 0p632 \
  --figure_preset fig4h \
  --output_stem graphs/graph_start_end/artifacts/multiple_process_prc31-40-58_peak_after_Fig4h
