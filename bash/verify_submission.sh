#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VERIFY_PYCACHE="$(mktemp -d)"
VERIFY_MPLCONFIG="$(mktemp -d)"
trap 'rm -rf "$VERIFY_PYCACHE" "$VERIFY_MPLCONFIG"' EXIT
export PYTHONPYCACHEPREFIX="$VERIFY_PYCACHE"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$VERIFY_MPLCONFIG}"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

PKL="graphs/graph1/average_graph_full_daily_preCovid_workday.pkl"
ZIP="graphs/graph1/average_graph_full_daily_preCovid_workday.zip"

if [ ! -f "$PKL" ]; then
    if [ -f "$ZIP" ]; then
        echo "Preparing mobility input from $ZIP"
        unzip -o "$ZIP" -d graphs/graph1
    else
        echo "Mobility input archive was not found at $ZIP." >&2
        exit 1
    fi
fi

test -f "$PKL"

for script in bash/*.sh; do
  bash -n "$script"
done

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m compileall -q us_reopen ODtoy graphs paper_repro_paths.py
"$PYTHON_BIN" -m us_reopen.cli --help >/dev/null
"$PYTHON_BIN" graphs/graph_start_end/plot_multiple_process_peak_flow.py --help >/dev/null
"$PYTHON_BIN" graphs/graph_start_end/plot_start_end_aftershutdown_simple.py --help >/dev/null
"$PYTHON_BIN" ODtoy/plot_toy_outputs.py --help >/dev/null

FIG5C="graphs/graph_start_end/artifacts/multiple_process_prc31-33-40-58_peak_after_Fig5c"
test -f "$FIG5C.jpg"
test -f "$FIG5C.svg"

FIG5_BDE="graphs/graph_start_end/us_preCovid_fit_empirical_P0p632_a2_fit_f1/curve_preCovid_Iairport5k_empirical_P0p632_a2_fit_f1_aftershutdown_v1_prc33_ar0p60120_1_TO5_fitted500kIa0p9initPF120_120_inf"
for product in realtime_random_both realtime_county_pchDextS_both realtime_county_pchDextS_both_Q; do
  test -f "$FIG5_BDE/$product.jpg"
  test -f "$FIG5_BDE/$product.svg"
done

echo "Submission structure and entry points passed verification."
