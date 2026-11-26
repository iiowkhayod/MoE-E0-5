"""Plot paired E1/E2/clear-sky/persistence trajectories for representative test days."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_day(data, day, lead, output, title):
    models = {name: data[data.model == name].set_index("issue_utc") for name in
              ("smart_persistence", "e1_direct_a4", "e2_a4")}
    index = models["e2_a4"].index.normalize() == day
    frame = models["e2_a4"][index]
    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(frame.index, frame[f"ghi_{lead}"], label="Observed GHI", color="black", linewidth=1.5)
    axis.plot(frame.index, frame[f"prediction_{lead}"], label="E2 residual", linewidth=1.2)
    axis.plot(frame.index, frame[f"clear_{lead}"], label="Clear-sky GHI", linewidth=1,
              color="tab:orange", alpha=.8)
    for name, label in (("e1_direct_a4", "E1 direct"), ("smart_persistence", "Smart persistence")):
        if name not in models or models[name].empty:
            continue
        other = models[name].loc[frame.index]
        axis.plot(frame.index, other[f"prediction_{lead}"], label=label, linewidth=1, alpha=.85)
    axis.set(title=title, ylabel="GHI (W/m²)", xlabel="Issue time (UTC)")
    axis.legend(ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lead", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(args.forecasts, parse_dates=["issue_utc"])
    test = data[(data.split == "test") & (data.model == "e2_a4")].copy()
    delta = (test[f"ghi_{args.lead}"] - test.current_ghi).abs()
    counts = delta.ge(args.threshold).groupby(test.issue_utc.dt.normalize()).sum()
    ramp_day, stable_day = counts.idxmax(), counts.idxmin()
    plot_day(data[data.split == "test"], ramp_day, args.lead, args.output / "ramp_day.png",
             f"Ramp-rich test day, +{args.lead} min")
    plot_day(data[data.split == "test"], stable_day, args.lead, args.output / "stable_day.png",
             f"Stable test day, +{args.lead} min")
    pd.DataFrame({"selection": ["ramp-rich", "stable"], "utc_day": [ramp_day, stable_day],
                  "ramp_samples": [int(counts.loc[ramp_day]), int(counts.loc[stable_day])]}
                 ).to_csv(args.output / "selected_days.csv", index=False)


if __name__ == "__main__":
    main()

