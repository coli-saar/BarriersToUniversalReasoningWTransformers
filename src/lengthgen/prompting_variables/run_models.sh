#!/usr/bin/env bash

set -euo pipefail
MODEL="${1:-mistral24B}"
LENGTHS=(10 15 20 25 30)
VARIANTS=(none linenums linenums+value_change)
for n_ops in "${LENGTHS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "============================================================"
    echo "Running: n_ops=${n_ops}, variant=${variant}, model=${MODEL}"
    echo "============================================================"
    python3 run_api.py \
      --n-ops "${n_ops}" \
      --variant "${variant}" \
      --model "${MODEL}"
  done
done
echo "All runs completed."