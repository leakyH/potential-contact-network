#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${RUN_TOY:-1}" = "1" ]; then
  bash bash/run_toy_model.sh
  bash bash/plot_toy_outputs.sh
fi

bash bash/run_formal_experiments.sh
bash bash/plot_fig4h.sh
bash bash/plot_fig5c.sh
bash bash/plot_fig5_bde.sh
