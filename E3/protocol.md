# E3 clear-sky index experiment

Use E2-A4 inputs, architecture, prepared scopes, seed and training hyperparameters unchanged. The target is GHI / future clear GHI, with identity normalization; MSE is in index space. Reconstruct GHI as predicted index times clear GHI, then apply E2's [0,1600] W/m² output bounds. Raw index predictions are saved without [0,1] clipping. Index-space MSE is not equivalent to E2's GHI-scale residual MSE: it weights squared GHI errors inversely by clear GHI squared.

All prepared clear targets must be finite and strictly positive; do not change the E2 sample scope. The full E2 scopes observed minima 86.52 W/m² or higher. No future measured weather enters the inputs. The exact E2 split and image selection and normalization rules apply.

On the server, from an immutable checkout (set `COMMIT` to its full SHA):

```bash
export PATH=/root/data1/miniconda3/envs/folsom-lab/bin:$PATH
export PYTHONPATH=/root/data1/igarss_t18_ramp_pilot_20261114/vendor
python E3/train_index.py --config E2/configs/base.json --conditions E2/configs/conditions.json --condition e2_a4 --prepared /root/data1/expert_E1-5/E2_addRes/prepared/cross_year_v1 --output /root/data1/expert_E1-5/E3_index/runs/index_seed43_attempt01 --source-commit "$COMMIT"
```

Validation and test predictions contain issue timestamps, observations, reconstructed GHI, clear GHI and raw index predictions. Evaluation uses E2/evaluate.py on a separate directory containing copies of E1, E2 and E3 prediction files.

Run the paired validation and test evaluator with `bash E3/evaluate_paired.sh` in the same Python environment. The evaluator reproduces E2's Smart Persistence, Ramp thresholds, event rules and metrics, and rejects mismatched issue timestamps.
