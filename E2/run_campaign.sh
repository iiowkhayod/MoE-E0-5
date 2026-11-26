#!/usr/bin/env bash
set -euo pipefail

project=$(cd "$(dirname "$0")" && pwd)
source_commit=${SOURCE_COMMIT:?Set SOURCE_COMMIT to the immutable 40-character Git commit}
config="$project/configs/base.json"
conditions="$project/configs/conditions.json"
prepared=/root/data1/expert_E1-5/E2_addRes/prepared/cross_year_v1
run_root=/root/data1/expert_E1-5/E2_addRes/runs/e2_baseline_seed43_attempt01
test "${#source_commit}" -eq 40
test -f "$prepared/complete.json"
for year in 2014 2015 2016; do
  test -f "/root/data1/expert_E1-5/E2_addRes/cache/rgb64_v2/$year/complete.json"
done
test ! -e "$run_root"
mkdir -p "$run_root/conditions"

date -u +%FT%TZ > "$run_root/started_utc.txt"
uname -a > "$run_root/hardware.txt"
lscpu >> "$run_root/hardware.txt"
free -b >> "$run_root/hardware.txt"
nvidia-smi -q >> "$run_root/hardware.txt"
python -VV > "$run_root/environment.txt"
python -m pip freeze >> "$run_root/environment.txt"
python "$project/audit_data.py" --config "$config" --prepared "$prepared" \
  --output "$run_root/data_audit"

for condition in e1_direct_a4 e2_a1 e2_a2 e2_a3 e2_a4 e2_a1_gru e2_a3_3dcnn e2_a4_gate; do
  output="$run_root/conditions/$condition"
  test ! -e "$output"
  python -u "$project/train.py" --config "$config" --conditions "$conditions" \
    --condition "$condition" --prepared "$prepared" --output "$output" --source-commit "$source_commit" \
    2>&1 | tee "$run_root/$condition.log"
done
python -u "$project/evaluate.py" --config "$config" --prepared "$prepared" \
  --runs "$run_root/conditions" --output "$run_root/evaluation" 2>&1 | tee "$run_root/evaluate.log"
python "$project/plot_results.py" --forecasts "$run_root/evaluation/forecasts.csv" \
  --output "$run_root/plots" --lead 15 --threshold 100
python "$project/report.py" --run-root "$run_root"
date -u +%FT%TZ > "$run_root/finished_utc.txt"
python "$project/finalize.py" --run-root "$run_root" --prepared "$prepared" \
  --source-commit "$source_commit" --protocol "$project/experiments/e2_baseline/protocol.md" \
  --config "$config"
python - <<'PY' "$run_root"
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
files={}
for path in sorted(p for p in root.rglob('*') if p.is_file()):
    files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'complete.json').write_text(json.dumps({'status':'complete','sha256_before_complete_marker':files},indent=2))
PY

