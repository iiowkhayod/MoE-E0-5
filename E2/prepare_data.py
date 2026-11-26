"""Prepare the complete paired cross-year E2 sample population."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import FrameCache, load_config, load_observations, solar_components


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def save_array(folder, name, value):
    np.save(folder / f"{name}.npy", value, allow_pickle=False)


def prepare_year(config, year, role, output):
    grid, ghi, weather = load_observations(config["raw_dir"], year)
    clear, zenith = solar_components(config, grid)
    cache = FrameCache(Path(config["cache_dir"]) / str(year))
    history = int(config["history_minutes"])
    leads = np.asarray(config["lead_minutes"], dtype=np.int64)
    image_offsets = np.asarray(config["image_offsets_minutes"], dtype=np.int64)
    positions = np.arange(history - 1, len(grid) - leads.max(), dtype=np.int64)
    history_ids = positions[:, None] - np.arange(history - 1, -1, -1)
    target_ids = positions[:, None] + leads
    valid = np.isfinite(ghi[history_ids]).all(axis=1)
    valid &= np.isfinite(ghi[target_ids]).all(axis=1)
    max_zenith = float(config["maximum_solar_zenith_degrees"])
    valid &= zenith[positions] < max_zenith
    valid &= (zenith[target_ids] < max_zenith).all(axis=1)
    positions, history_ids, target_ids = positions[valid], history_ids[valid], target_ids[valid]

    issue_seconds = grid[positions].asi8 // 10**9
    requested = issue_seconds[:, None] + image_offsets * 60
    flat = requested.ravel()
    image_ids = np.searchsorted(cache.seconds, flat, side="right") - 1
    image_age = flat - cache.seconds[np.maximum(image_ids, 0)]
    image_ids = image_ids.reshape(len(positions), -1)
    image_age = image_age.reshape(len(positions), -1)
    image_valid = (image_ids >= 0).all(axis=1)
    image_valid &= (image_age >= 0).all(axis=1)
    image_valid &= (image_age <= int(config["max_image_age_seconds"])).all(axis=1)
    positions, history_ids, target_ids = (x[image_valid] for x in
                                           (positions, history_ids, target_ids))
    image_ids, image_age = image_ids[image_valid], image_age[image_valid]
    residual = ghi[history_ids] - clear[history_ids]
    folder = output / role
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {
        "issue_ns": grid[positions].asi8,
        "history_ids": history_ids.astype(np.int32),
        "frame_ids": image_ids.astype(np.int32),
        "image_age_seconds": image_age.astype(np.int16),
        "residual_history": residual.astype(np.float32),
        "clear_history": clear[history_ids].astype(np.float32),
        "weather_history": weather[history_ids].astype(np.float32),
        "targets": ghi[target_ids].astype(np.float32),
        "clear_targets": clear[target_ids].astype(np.float32),
        "current_ghi": ghi[positions].astype(np.float32),
        "current_clear": clear[positions].astype(np.float32),
    }
    for name, value in arrays.items():
        save_array(folder, name, value)
    scope = pd.DataFrame({"issue_utc": grid[positions], "max_image_age_seconds": image_age.max(axis=1)})
    for column, lead in enumerate(leads):
        scope[f"ghi_{lead}"] = arrays["targets"][:, column]
        scope[f"clear_{lead}"] = arrays["clear_targets"][:, column]
    scope.to_csv(folder / "scope.csv", index=False)
    manifest = {"scope_id": role, "year": year, "count": len(positions),
                "first_issue_utc": str(grid[positions[0]]), "last_issue_utc": str(grid[positions[-1]]),
                "months": sorted(np.unique(grid[positions].month).tolist()),
                "arrays": {name: list(value.shape) for name, value in arrays.items()},
                "frame_clock": "archive mtime", "causal": True,
                "max_observed_image_age_seconds": int(image_age.max())}
    (folder / "scope_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    roles = config["years"]
    manifests = [prepare_year(config, year, role, args.output) for role, year in roles.items()]
    raw = Path(config["raw_dir"])
    identity = {name: {"bytes": (raw / name).stat().st_size, "sha256": file_sha256(raw / name)}
                for name in ("Folsom_irradiance.csv", "Folsom_weather.csv")}
    (args.output / "data_identity.json").write_text(json.dumps(identity, indent=2))
    (args.output / "complete.json").write_text(json.dumps({"scopes": manifests}, indent=2))


if __name__ == "__main__":
    main()

