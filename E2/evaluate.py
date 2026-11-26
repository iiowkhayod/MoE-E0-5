"""Evaluate all E2 conditions on identical validation and test scopes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_config


def errors(target, prediction):
    difference = prediction - target
    return float(np.sqrt(np.mean(difference ** 2))), float(np.mean(np.abs(difference))), float(np.mean(difference))


def event_summary(frame, leads, threshold):
    observed = np.column_stack([(frame[f"ghi_{lead}"] - frame.current_ghi).abs() >= threshold
                                for lead in leads])
    predicted = np.column_stack([(frame[f"prediction_{lead}"] - frame.current_ghi).abs() >= threshold
                                 for lead in leads])
    observed_any, predicted_any = observed.any(axis=1), predicted.any(axis=1)
    tp = int((observed_any & predicted_any).sum())
    precision = tp / predicted_any.sum() if predicted_any.sum() else np.nan
    recall = tp / observed_any.sum() if observed_any.sum() else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    observed_first = np.argmax(observed, axis=1)
    predicted_first = np.argmax(predicted, axis=1)
    rows = np.flatnonzero(observed_any & predicted_any)
    direction_match = []
    lags = []
    for row in rows:
        oi, pi = observed_first[row], predicted_first[row]
        observed_delta = frame.iloc[row][f"ghi_{leads[oi]}"] - frame.current_ghi.iloc[row]
        predicted_delta = frame.iloc[row][f"prediction_{leads[pi]}"] - frame.current_ghi.iloc[row]
        if np.sign(observed_delta) == np.sign(predicted_delta):
            direction_match.append(row)
            lags.append(leads[pi] - leads[oi])
    return {"event_n": int(observed_any.sum()), "predicted_event_n": int(predicted_any.sum()),
            "precision": precision, "recall": recall, "f1": f1,
            "direction_matched_n": len(direction_match),
            "mean_detection_lag_min": float(np.mean(lags)) if lags else np.nan,
            "mean_abs_detection_lag_min": float(np.mean(np.abs(lags))) if lags else np.nan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    leads = config["lead_minutes"]
    threshold = float(config["ramp_threshold_wm2"])
    metric_rows, event_rows, forecasts = [], [], []
    for role in ("validation", "test"):
        scope = pd.read_csv(args.prepared / role / "scope.csv", parse_dates=["issue_utc"])
        current = np.load(args.prepared / role / "current_ghi.npy")
        clear_current = np.load(args.prepared / role / "current_clear.npy")
        clear_target = np.load(args.prepared / role / "clear_targets.npy")
        smart = np.maximum(0, current[:, None] / np.maximum(clear_current[:, None], 1) * clear_target)
        model_frames = {"smart_persistence": scope[["issue_utc"]].copy()}
        for column, lead in enumerate(leads):
            model_frames["smart_persistence"][f"ghi_{lead}"] = scope[f"ghi_{lead}"]
            model_frames["smart_persistence"][f"clear_{lead}"] = scope[f"clear_{lead}"]
            model_frames["smart_persistence"][f"prediction_{lead}"] = smart[:, column]
        for folder in sorted(path for path in args.runs.iterdir() if path.is_dir()):
            if folder.name == "smart_persistence":
                continue
            path = folder / f"{role}_predictions.csv"
            if path.exists():
                model_frames[folder.name] = pd.read_csv(path, parse_dates=["issue_utc"])
        reference_rmse = {}
        for model, frame in model_frames.items():
            if not frame.issue_utc.equals(scope.issue_utc):
                raise ValueError(f"scope mismatch: {role}/{model}")
            frame["current_ghi"] = current
            for lead in leads:
                frame[f"clear_{lead}"] = scope[f"clear_{lead}"].to_numpy()
            frame["model"] = model
            frame["split"] = role
            forecasts.append(frame)
            for lead in leads:
                target = frame[f"ghi_{lead}"].to_numpy()
                prediction = frame[f"prediction_{lead}"].to_numpy()
                delta = target - current
                groups = {"all": np.ones(len(frame), dtype=bool),
                          "ramp": np.abs(delta) >= threshold,
                          "non_ramp": np.abs(delta) < threshold,
                          "ramp_up": delta >= threshold,
                          "ramp_down": delta <= -threshold}
                for group, chosen in groups.items():
                    rmse, mae, mbe = errors(target[chosen], prediction[chosen])
                    metric_rows.append({"split": role, "model": model, "lead_min": lead,
                                        "group": group, "n": int(chosen.sum()),
                                        "rmse": rmse, "mae": mae, "mbe": mbe})
                if model == "smart_persistence":
                    reference_rmse[lead] = metric_rows[-5]["rmse"]
            event_rows.append({"split": role, "model": model,
                               **event_summary(frame, leads, threshold)})
        for row in metric_rows:
            if row["split"] == role and row["group"] == "all":
                row["forecast_skill_percent"] = 100 * (1 - row["rmse"] / reference_rmse[row["lead_min"]])
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.output / "events.csv", index=False)
    forecast = pd.concat(forecasts, ignore_index=True)
    forecast.to_csv(args.output / "forecasts.csv", index=False)
    smart_folder = args.runs / "smart_persistence"
    smart_folder.mkdir(exist_ok=False)
    for role in ("validation", "test"):
        smart = forecast[(forecast.model == "smart_persistence") & (forecast.split == role)]
        columns = ["issue_utc"] + [item for lead in leads for item in
                                    (f"ghi_{lead}", f"prediction_{lead}")]
        smart[columns].to_csv(smart_folder / f"{role}_predictions.csv", index=False)
    paired = []
    for split in ("validation", "test"):
        for lead in leads:
            for group in ("all", "ramp", "non_ramp", "ramp_up", "ramp_down"):
                part = metrics[(metrics.split == split) & (metrics.lead_min == lead) &
                               (metrics.group == group)].set_index("model")
                if {"e1_direct_a4", "e2_a4"}.issubset(part.index):
                    paired.append({"split": split, "lead_min": lead, "group": group,
                                   "n": int(part.loc["e2_a4", "n"]),
                                   "e2_minus_e1_rmse": part.loc["e2_a4", "rmse"] - part.loc["e1_direct_a4", "rmse"],
                                   "e2_minus_e1_mae": part.loc["e2_a4", "mae"] - part.loc["e1_direct_a4", "mae"]})
    pd.DataFrame(paired).to_csv(args.output / "e2_vs_e1.csv", index=False)
    test = metrics[(metrics.split == "test") & (metrics.group.isin(["all", "ramp", "non_ramp"]))]
    pivot = test.pivot_table(index=["model", "lead_min"], columns="group", values=["rmse", "mae", "n"])
    (args.output / "summary.txt").write_text(pivot.to_string())
    print(pivot.to_string())


if __name__ == "__main__":
    main()

