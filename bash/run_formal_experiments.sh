#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs/experiments output/pkls output/csvs output/scalars .mplconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-.mplconfig}"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

# CPU is the portable default. Set US_REOPEN_USE_CUDA=1 and the CUDA device
# variables before launching to enable GPU execution.
export US_REOPEN_USE_CUDA="${US_REOPEN_USE_CUDA:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CURVE_CODES="${CURVE_CODES:-31 33 40 58}"
CURVE_POINTS="${CURVE_POINTS:-21}"
CURVE_POOLSIZE="${CURVE_POOLSIZE:-8}"
CURVE_SEED="${CURVE_SEED:-0}"
WEEKCOUNT="${WEEKCOUNT:-60}"
RUN_CURVES="${RUN_CURVES:-1}"
RUN_MULTIPLE="${RUN_MULTIPLE:-1}"
REPEAT_COUNT="${REPEAT_COUNT:-50}"
MULTIPLE_POOLSIZE="${MULTIPLE_POOLSIZE:-8}"
MULTIPLE_SEED="${MULTIPLE_SEED:-0}"

BASE_ARGS=(
  --alpha_ratio_name 2
  --flow_ratio_name 1
  --period preCovid
  --init_method airport5k
  --beta_density fit
  --process_threshold 120
  --process_inf
  --Rtype empirical
  --Pratio 0p632
  --weekcount "$WEEKCOUNT"
)

run_logged() {
  local label="$1"
  shift
  local logfile="logs/experiments/${label}.log"
  echo "[$(date '+%F %T')] start ${label}"
  "$@" >"$logfile" 2>&1
  echo "[$(date '+%F %T')] done ${label}"
}

if [ "$RUN_CURVES" = "1" ]; then
  for code in $CURVE_CODES; do
    run_logged "curve_code${code}" \
      "$PYTHON_BIN" -m us_reopen.cli \
        --mode curve \
        "${BASE_ARGS[@]}" \
        --process_method "$code" \
        --curve_points "$CURVE_POINTS" \
        --poolsize "$CURVE_POOLSIZE" \
        --seed "$CURVE_SEED"
  done
fi

if [ "$RUN_MULTIPLE" = "1" ]; then
  # Repeated endpoints and PCH checkpoints used by the Fig. 5c plot.
  run_logged multiple_code31_linear \
    "$PYTHON_BIN" -m us_reopen.cli \
      --mode multiple "${BASE_ARGS[@]}" \
      --process_method 31 --process_fn_names linear \
      --multiple_idx 20 --repeat_count "$REPEAT_COUNT" \
      --multiple_poolsize "$MULTIPLE_POOLSIZE" --seed "$MULTIPLE_SEED"

  run_logged multiple_code31_population \
    "$PYTHON_BIN" -m us_reopen.cli \
      --mode multiple "${BASE_ARGS[@]}" \
      --process_method 31 --process_fn_names pop_both_inverse \
      --multiple_idx 12 --repeat_count "$REPEAT_COUNT" \
      --multiple_poolsize "$MULTIPLE_POOLSIZE" --seed "$MULTIPLE_SEED"

  run_logged multiple_code33_pch \
    "$PYTHON_BIN" -m us_reopen.cli \
      --mode multiple "${BASE_ARGS[@]}" \
      --process_method 33 --process_fn_names realtime_county_pchDextS_both \
      --multiple_idx 6 7 8 9 10 --repeat_count "$REPEAT_COUNT" \
      --multiple_poolsize "$MULTIPLE_POOLSIZE" --seed "$MULTIPLE_SEED"
fi

echo "[$(date '+%F %T')] formal experiments complete"
