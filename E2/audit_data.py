"""Audit the official Folsom observations, image clock and prepared scopes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import WEATHER, load_config, load_observations


def quantiles(values):
    values = np.asarray(values)
    if not len(values):
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {name: float(value) for name, value in zip(
        ("p50", "p90", "p95", "p99", "max"),
        np.quantile(values, (0.50, 0.90, 0.95, 0.99, 1.00)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = Path(config["raw_dir"])
    observation_rows, image_rows, scope_rows = [], [], []

    raw_tables = {}
    for filename in ("Folsom_irradiance.csv", "Folsom_weather.csv"):
        frame = pd.read_csv(raw / filename, na_values=["NAN"])
        parsed = pd.to_datetime(frame["timeStamp"], utc=True, errors="coerce")
        raw_tables[filename] = (frame, parsed)

    for role, year in config["years"].items():
        grid, ghi, weather = load_observations(raw, year)
        start, finish = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(
            f"{year + 1}-01-01", tz="UTC")
        row = {"role": role, "year": year, "expected_minutes": len(grid),
               "valid_ghi_minutes": int(np.isfinite(ghi).sum()),
               "missing_ghi_minutes": int((~np.isfinite(ghi)).sum())}
        for filename, (frame, parsed) in raw_tables.items():
            selected = frame.loc[(parsed >= start) & (parsed < finish)]
            selected_time = parsed.loc[selected.index]
            prefix = "irradiance" if filename.startswith("Folsom_irradiance") else "weather"
            row[f"{prefix}_raw_rows"] = len(selected)
            row[f"{prefix}_invalid_timestamps_total_file"] = int(parsed.isna().sum())
            row[f"{prefix}_duplicate_timestamp_rows"] = int(selected_time.duplicated(keep=False).sum())
        for column, name in enumerate(WEATHER):
            row[f"valid_{name}_minutes"] = int(np.isfinite(weather[:, column]).sum())
        observation_rows.append(row)

        cache = Path(config["cache_dir"]) / str(year)
        complete = json.loads((cache / "complete.json").read_text())
        frames = pd.read_csv(cache / "frames.csv")
        seconds = np.sort(frames.mtime.to_numpy(dtype=np.int64))
        positive_intervals = np.diff(seconds)
        positive_intervals = positive_intervals[positive_intervals > 0]
        image_rows.append({"role": role, "year": year,
                           "readable_frames": int(complete["readable_frames"]),
                           "unreadable_frames": int(complete["unreadable_frames"]),
                           "first_mtime_utc": pd.to_datetime(seconds[0], unit="s", utc=True).isoformat(),
                           "last_mtime_utc": pd.to_datetime(seconds[-1], unit="s", utc=True).isoformat(),
                           "duplicate_mtime_rows": int(pd.Series(seconds).duplicated(keep=False).sum()),
                           **{f"interval_seconds_{key}": value
                              for key, value in quantiles(positive_intervals).items()}})

        folder = args.prepared / role
        manifest = json.loads((folder / "scope_manifest.json").read_text())
        ages = np.load(folder / "image_age_seconds.npy", mmap_mode="r")
        scope_rows.append({"role": role, "year": year, "eligible_samples": manifest["count"],
                           "first_issue_utc": manifest["first_issue_utc"],
                           "last_issue_utc": manifest["last_issue_utc"],
                           "months": ",".join(map(str, manifest["months"])),
                           **{f"image_age_seconds_{key}": value
                              for key, value in quantiles(ages.ravel()).items()}})

    pd.DataFrame(observation_rows).to_csv(args.output / "observations.csv", index=False)
    pd.DataFrame(image_rows).to_csv(args.output / "image_clock.csv", index=False)
    pd.DataFrame(scope_rows).to_csv(args.output / "prepared_scopes.csv", index=False)
    summary = {"frame_clock": "archive mtime", "image_matching": "latest frame no later than request",
               "maximum_allowed_image_age_seconds": config["max_image_age_seconds"],
               "observation_rows": observation_rows, "image_rows": image_rows,
               "scope_rows": scope_rows}
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
