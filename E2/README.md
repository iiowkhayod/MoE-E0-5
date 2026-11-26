# E2 fixed clear-sky plus additive residual forecasting

E2 evaluates whether separating deterministic solar geometry from weather-related deviations improves Folsom GHI forecasts. The paired protocol uses 2014 for training, 2015 for validation and 2016 for testing. Forecast horizons are +5, +10, +15 and +30 minutes.

## Components

- `prepare_cache.py` streams each official sky-image archive into a verified 64×64 RGB memory map while preserving archive mtime.
- `prepare_data.py` creates the complete common population for every condition.
- `train.py` trains one immutable E1, E2, input-ablation or model-family condition.
- `evaluate.py` computes overall, Ramp, non-Ramp, direction and event metrics against Smart Persistence.
- `plot_results.py` produces paired test trajectories for representative stable and Ramp-rich days.
- `finalize.py` hashes scopes, environments, predictions, checkpoints and metrics into an append-only ledger.

The primary input ablation uses one controlled model family. A1 enables residual history, A2 adds weather, A3 adds six causal images, and A4 enables both weather and images. The model-family follow-up changes one component at a time.

## Server paths

Raw data are expected at `/root/data1/folsom_multimodal/data/raw`. The requested workspace is `/root/data1/expert_E1-5/E2_addRes`. The frozen protocol is in `experiments/e2_baseline/protocol.md`.

The scientific run starts only after all three cache `complete.json` markers exist and `prepare_data.py` has emitted all three full-year scope manifests. From an immutable source checkout:

```bash
export PYTHONPATH=/root/data1/igarss_t18_ramp_pilot_20261114/vendor
export SOURCE_COMMIT=<full Git commit>
bash E2/run_campaign.sh
python E2/report.py --run-root /root/data1/expert_E1-5/E2_addRes/runs/e2_baseline_seed43_attempt01
```

Engineering fixtures live under `engineering/` on the server and are excluded from scientific results.
