"""Bind completed E2 attempts to hashes, environments, metrics and an append-only ledger."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import pandas as pd
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--conditions", required=True, type=Path)
    args = parser.parse_args()
    metrics = pd.read_csv(args.run_root / "evaluation/metrics.csv")
    events = pd.read_csv(args.run_root / "evaluation/events.csv")
    hardware_path = args.run_root / "hardware.txt"
    environment_path = args.run_root / "environment.txt"
    ledger = []
    memory_bytes = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory_bytes = int(line.split()[1]) * 1024
    scope = {}
    for role in ("train", "validation", "test"):
        path = args.prepared / role / "scope_manifest.json"
        scope[role] = {"path": str(path), "sha256": sha256(path),
                       "count": json.loads(path.read_text())["count"]}
    for folder in sorted(path for path in (args.run_root / "conditions").iterdir() if path.is_dir()):
        model = folder.name
        model_metrics = metrics[metrics.model == model]
        model_events = events[events.model == model]
        if model_metrics.empty or model_events.empty:
            raise ValueError(f"missing evaluation records for {model}")
        model_metrics.to_csv(folder / "metrics.csv", index=False)
        model_events.to_csv(folder / "events.csv", index=False)
        run_path = folder / "run.json"
        if run_path.exists():
            run = json.loads(run_path.read_text())
            seed = run["seed"]
            start, finish, runtime = run["started_utc"], run["finished_utc"], run["runtime_seconds"]
            parameter_count, best_epoch = run["parameter_count"], run["best_epoch"]
        else:
            seed = None
            start = (args.run_root / "started_utc.txt").read_text().strip()
            finish = (args.run_root / "finished_utc.txt").read_text().strip()
            runtime = (pd.Timestamp(finish) - pd.Timestamp(start)).total_seconds()
            parameter_count, best_epoch = 0, None
            run = {"condition_id": model, "deterministic": True, "source_commit": args.source_commit}
            run_path.write_text(json.dumps(run, indent=2))
        environment = {
            "schema_version": "1.1", "run_id": f"{model}_seed{seed}" if seed is not None else model,
            "attempt_id": f"{model}_attempt01",
            "platform": {"operating_system": platform.platform(), "architecture": platform.machine(),
                         "kernel": platform.release()},
            "hardware": {"cpu_model": platform.processor() or "see hardware evidence",
                         "logical_cpu_count": os.cpu_count(), "system_memory_bytes": memory_bytes,
                         "accelerators": [{"kind": "gpu", "model": torch.cuda.get_device_name(0),
                                           "device_count": 1,
                                           "memory_bytes_per_device": torch.cuda.get_device_properties(0).total_memory}]},
            "hardware_evidence": [{"command": ["uname", "lscpu", "free", "nvidia-smi"],
                                   "path": str(hardware_path), "sha256": sha256(hardware_path)}],
            "software_versions": {"python": platform.python_version(), "pytorch": torch.__version__,
                                  "cuda_runtime": torch.version.cuda},
            "dependency_lock": {"path": str(environment_path), "sha256": sha256(environment_path)}}
        (folder / "environment.json").write_text(json.dumps(environment, indent=2))
        artifacts = {}
        for name in ("metrics.csv", "events.csv", "test_predictions.csv", "best.pt"):
            path = folder / name
            if path.exists():
                artifacts[name] = {"path": str(path), "sha256": sha256(path)}
        ledger.append({"run_id": environment["run_id"], "attempt_id": environment["attempt_id"],
                       "condition_id": model, "seed": seed, "status": "complete",
                       "source_commit": args.source_commit, "protocol_sha256": sha256(args.protocol),
                       "config_sha256": sha256(args.config), "started_utc": start,
                       "conditions_sha256": sha256(args.conditions),
                       "finished_utc": finish, "runtime_seconds": runtime,
                       "parameter_count": parameter_count, "best_epoch": best_epoch,
                       "scopes": scope, "environment_path": str(folder / "environment.json"),
                       "environment_sha256": sha256(folder / "environment.json"), "artifacts": artifacts})
    ledger_path = args.run_root / "run_ledger.jsonl"
    if ledger_path.exists():
        raise FileExistsError(ledger_path)
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
    manifest = {str(path.relative_to(args.run_root)): sha256(path)
                for path in sorted(args.run_root.rglob("*")) if path.is_file()}
    (args.run_root / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"completed_attempts": len(ledger), "finalized_utc": datetime.now(timezone.utc).isoformat()}))


if __name__ == "__main__":
    main()
