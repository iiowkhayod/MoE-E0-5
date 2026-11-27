#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
base=/root/data1/expert_E1-5/E3_index
e2=/root/data1/expert_E1-5/E2_addRes/runs/e2_baseline_seed43_attempt01/conditions
prepared=/root/data1/expert_E1-5/E2_addRes/prepared/cross_year_v1
work="$base/comparison/seed43_attempt01"
test -f "$base/runs/index_seed43_attempt01/run.json"
test ! -e "$work"
mkdir -p "$work/conditions/e1_direct_a4" "$work/conditions/e2_a4" "$work/conditions/e3_index"
for split in validation test; do
  cp "$e2/e1_direct_a4/${split}_predictions.csv" "$work/conditions/e1_direct_a4/"
  cp "$e2/e2_a4/${split}_predictions.csv" "$work/conditions/e2_a4/"
  cp "$base/runs/index_seed43_attempt01/${split}_predictions.csv" "$work/conditions/e3_index/"
done
python "$project/E2/evaluate.py" --config "$project/E2/configs/base.json" \
  --prepared "$prepared" --runs "$work/conditions" --output "$work/evaluation" \
  > "$work/evaluate.log" 2>&1
