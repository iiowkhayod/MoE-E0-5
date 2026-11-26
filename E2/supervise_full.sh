#!/usr/bin/env bash
set -euo pipefail

source_commit=${SOURCE_COMMIT:?Set SOURCE_COMMIT to the immutable 40-character Git commit}
workspace=/root/data1/expert_E1-5/E2_addRes
project="$workspace/source/$source_commit/E2"
cache="$workspace/cache/rgb64_v2"
prepared="$workspace/prepared/cross_year_v1"
python=/root/data1/miniconda3/envs/folsom-lab/bin/python
export PYTHONPATH="$project:/root/data1/igarss_t18_ramp_pilot_20261114/vendor"

test "${#source_commit}" -eq 40
test -f "$project/run_campaign.sh"
for year in 2014 2015 2016; do
  test -d "$cache/$year"
done

while true; do
  remaining=0
  for year in 2014 2015 2016; do
    if test -f "$cache/$year/complete.json"; then
      continue
    fi
    remaining=$((remaining + 1))
    pid=$(cat "$cache/$year.pid")
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "cache worker for $year exited without a complete marker" >&2
      exit 1
    fi
  done
  if test "$remaining" -eq 0; then
    break
  fi
  echo "$(date -u +%FT%TZ) waiting for $remaining cache workers"
  sleep 60
done

test ! -e "$prepared"
"$python" -u "$project/prepare_data.py" --config "$project/configs/base.json" \
  --output "$prepared"
test -f "$prepared/complete.json"

SOURCE_COMMIT="$source_commit" bash "$project/run_campaign.sh"
