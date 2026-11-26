"""Train one immutable E1/E2 condition and emit paired validation/test forecasts."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import FrameCache, load_config
from model import Forecaster


def load_arrays(root, role):
    folder = Path(root) / role
    names = ("issue_ns", "frame_ids", "residual_history", "clear_history", "weather_history",
             "targets", "clear_targets", "current_ghi", "current_clear")
    return {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in names}


def fit_normalizers(train):
    residual = np.asarray(train["residual_history"])
    clear = np.asarray(train["clear_history"])
    weather = np.asarray(train["weather_history"])
    median = np.nanmedian(weather.reshape(-1, weather.shape[-1]), axis=0)
    filled = np.where(np.isfinite(weather), weather, median)
    return {
        "residual_mean": float(residual.mean()), "residual_std": float(residual.std().clip(1e-6)),
        "clear_mean": float(clear.mean()), "clear_std": float(clear.std().clip(1e-6)),
        "weather_median": median.tolist(),
        "weather_mean": filled.reshape(-1, filled.shape[-1]).mean(axis=0).tolist(),
        "weather_std": filled.reshape(-1, filled.shape[-1]).std(axis=0).clip(1e-6).tolist(),
        "future_clear_mean": np.asarray(train["clear_targets"]).mean(axis=0).tolist(),
        "future_clear_std": np.asarray(train["clear_targets"]).std(axis=0).clip(1e-6).tolist(),
    }


class ForecastDataset(Dataset):
    def __init__(self, arrays, cache, normalizers, condition, loss_scale):
        self.arrays, self.cache, self.condition = arrays, cache, condition
        self.loss_scale = loss_scale
        self.norm = {key: np.asarray(value, dtype=np.float32) for key, value in normalizers.items()}

    def __len__(self):
        return len(self.arrays["issue_ns"])

    def __getitem__(self, index):
        residual = (self.arrays["residual_history"][index] - self.norm["residual_mean"]) / self.norm["residual_std"]
        clear = (self.arrays["clear_history"][index] - self.norm["clear_mean"]) / self.norm["clear_std"]
        weather = np.asarray(self.arrays["weather_history"][index])
        missing = ~np.isfinite(weather)
        weather = np.where(missing, self.norm["weather_median"], weather)
        weather = (weather - self.norm["weather_mean"]) / self.norm["weather_std"]
        weather = np.clip(weather, -10, 10)
        if not self.condition["weather"]:
            weather = np.zeros_like(weather)
            missing = np.zeros_like(missing)
        sequence = np.column_stack((residual, clear, weather, missing.astype(np.float32))).astype(np.float32)
        if self.condition["images"]:
            image = self.cache.images(self.arrays["frame_ids"][index]).transpose(0, 3, 1, 2)
            image = image.astype(np.float32) / 127.5 - 1
        else:
            image = np.zeros((6, 3, 64, 64), dtype=np.float32)
        future_clear = ((self.arrays["clear_targets"][index] - self.norm["future_clear_mean"])
                        / self.norm["future_clear_std"])
        if self.condition["target"] == "direct":
            target = self.arrays["targets"][index] / self.loss_scale
        else:
            target = (self.arrays["targets"][index] - self.arrays["clear_targets"][index]) / self.loss_scale
        return sequence, image, future_clear.astype(np.float32), target.astype(np.float32)


def predict(model, loader, device, condition, loss_scale, clear_targets):
    model.eval()
    values = []
    with torch.inference_mode():
        for sequence, images, future_clear, _ in loader:
            output = model(sequence.to(device), images.to(device), future_clear.to(device),
                           condition["images"])
            values.append(output.cpu().numpy() * loss_scale)
    values = np.concatenate(values)
    restored = values if condition["target"] == "direct" else values + clear_targets
    return np.clip(restored, 0, 1600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    condition = json.loads(args.conditions.read_text())[args.condition]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    arrays = {role: load_arrays(args.prepared, role) for role in ("train", "validation", "test")}
    normalizers = fit_normalizers(arrays["train"])
    caches = {role: FrameCache(Path(config["cache_dir"]) / str(config["years"][role]))
              for role in arrays}
    datasets = {role: ForecastDataset(arrays[role], caches[role], normalizers, condition,
                                      float(config["loss_scale_wm2"])) for role in arrays}
    generator = torch.Generator().manual_seed(seed)
    loaders = {role: DataLoader(data, batch_size=int(config["batch_size"]),
                                shuffle=role == "train", generator=generator,
                                num_workers=int(config["num_workers"]), pin_memory=True,
                                persistent_workers=int(config["num_workers"]) > 0)
               for role, data in datasets.items()}
    model = Forecaster(18, len(config["lead_minutes"]), condition["temporal"],
                       condition["visual"], condition["fusion"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    best, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        total = count = 0
        for sequence, images, future_clear, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence.to(device), images.to(device), future_clear.to(device),
                               condition["images"])
            loss = torch.nn.functional.mse_loss(prediction, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += float(loss) * len(sequence)
            count += len(sequence)
        model.eval()
        val_total = val_count = 0
        with torch.inference_mode():
            for sequence, images, future_clear, target in loaders["validation"]:
                prediction = model(sequence.to(device), images.to(device), future_clear.to(device),
                                   condition["images"])
                loss = torch.nn.functional.mse_loss(prediction, target.to(device))
                val_total += float(loss) * len(sequence)
                val_count += len(sequence)
        train_loss, val_loss = total / count, val_total / val_count
        history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss})
        print(args.condition, epoch, train_loss, val_loss, flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "condition": condition, "epoch": best_epoch,
                "normalizers": normalizers, "config": config}, args.output / "best.pt")
    pd.DataFrame(history).to_csv(args.output / "training.csv", index=False)
    for role in ("validation", "test"):
        prediction = predict(model, loaders[role], device, condition, float(config["loss_scale_wm2"]),
                             np.asarray(arrays[role]["clear_targets"]))
        frame = pd.DataFrame({"issue_utc": pd.to_datetime(arrays[role]["issue_ns"], utc=True)})
        for column, lead in enumerate(config["lead_minutes"]):
            frame[f"ghi_{lead}"] = arrays[role]["targets"][:, column]
            frame[f"prediction_{lead}"] = prediction[:, column]
        frame.to_csv(args.output / f"{role}_predictions.csv", index=False)
    metadata = {"condition_id": args.condition, "condition": condition, "seed": seed,
                "best_epoch": best_epoch, "best_validation_mse_scaled": best,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "runtime_seconds": time.time() - started, "hostname": platform.node(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0)}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

