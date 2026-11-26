"""Shared Folsom data definitions for the E2 campaign."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.location import Location


WEATHER = ["air_temp", "relhum", "press", "windsp", "wind_sin", "wind_cos",
           "max_windsp", "precipitation"]


class FrameCache:
    def __init__(self, root):
        root = Path(root)
        info = json.loads((root / "complete.json").read_text())
        frame = pd.read_csv(root / "frames.csv")
        count = len(frame)
        if count != info["readable_frames"]:
            raise ValueError(f"manifest count mismatch in {root}")
        expected = count * 64 * 64 * 3
        if (root / "rgb.bin").stat().st_size != expected:
            raise ValueError(f"cache size mismatch in {root}")
        order = np.lexsort((frame.path.to_numpy(), frame.mtime.to_numpy()))
        self.names = frame.path.to_numpy()[order]
        self.seconds = frame.mtime.to_numpy(dtype=np.int64)[order]
        self.original_ids = order
        self.rgb = np.memmap(root / "rgb.bin", dtype=np.uint8, mode="r",
                             shape=(count, 64, 64, 3))

    def images(self, ordered_ids):
        original = self.original_ids[np.asarray(ordered_ids)]
        return np.asarray(self.rgb[original])


def load_config(path):
    return json.loads(Path(path).read_text())


def load_observations(raw_dir, year):
    raw_dir = Path(raw_dir)
    grid = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", inclusive="left",
                         freq="min", tz="UTC")
    tables = []
    for filename in ("Folsom_irradiance.csv", "Folsom_weather.csv"):
        frame = pd.read_csv(raw_dir / filename, na_values=["NAN"])
        frame.index = pd.to_datetime(frame.pop("timeStamp"), utc=True, errors="coerce")
        frame = frame.loc[frame.index.notna()]
        frame = frame.loc[~frame.index.duplicated(keep=False)]
        tables.append(frame.reindex(grid))
    irradiance, weather = tables
    ghi = pd.to_numeric(irradiance.ghi, errors="coerce").where(lambda x: x.between(0, 1600))
    direction = np.deg2rad(pd.to_numeric(weather.winddir, errors="coerce").where(
        lambda x: x.between(0, 360)))
    conventional = pd.DataFrame(index=grid)
    conventional["air_temp"] = pd.to_numeric(weather.air_temp, errors="coerce").where(
        lambda x: x.between(-50, 60))
    conventional["relhum"] = pd.to_numeric(weather.relhum, errors="coerce").where(
        lambda x: x.between(0, 100))
    conventional["press"] = pd.to_numeric(weather.press, errors="coerce").where(
        lambda x: x.between(800, 1100))
    conventional["windsp"] = pd.to_numeric(weather.windsp, errors="coerce").where(
        lambda x: x.between(0, 80))
    conventional["wind_sin"], conventional["wind_cos"] = np.sin(direction), np.cos(direction)
    conventional["max_windsp"] = pd.to_numeric(weather.max_windsp, errors="coerce").where(
        lambda x: x.between(0, 100))
    conventional["precipitation"] = pd.to_numeric(weather.precipitation, errors="coerce").where(
        lambda x: x.between(0, 500))
    return grid, ghi.to_numpy(np.float32), conventional[WEATHER].to_numpy(np.float32)


def solar_components(config, grid):
    site = config["site"]
    location = Location(site["latitude"], site["longitude"], tz="UTC",
                        altitude=site["altitude_m"])
    clear = location.get_clearsky(grid, model=config["clear_sky_model"]).ghi.to_numpy(np.float32)
    zenith = location.get_solarposition(grid).zenith.to_numpy(np.float32)
    return clear, zenith

