"""Generate a concise evidence-linked E2 analysis report from completed artifacts."""
import argparse
import json
from pathlib import Path

import pandas as pd


def markdown_table(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.astype(str).itertuples(index=False):
        rows.append("| " + " | ".join(record) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    evaluation = args.run_root / "evaluation"
    metrics = pd.read_csv(evaluation / "metrics.csv")
    comparisons = pd.read_csv(evaluation / "e2_vs_e1.csv")
    events = pd.read_csv(evaluation / "events.csv")
    test = metrics[(metrics.split == "test") & metrics.model.isin(
        ["smart_persistence", "e1_direct_a4", "e2_a4"]) &
        metrics.group.isin(["all", "ramp", "non_ramp"])].copy()
    test["RMSE"] = test.rmse.round(2)
    test["MAE"] = test.mae.round(2)
    table = test[["model", "lead_min", "group", "n", "RMSE", "MAE"]]
    paired = comparisons[(comparisons.split == "test") &
                         comparisons.group.isin(["all", "ramp", "non_ramp"])].copy()
    paired["ΔRMSE(E2−E1)"] = paired.e2_minus_e1_rmse.round(2)
    paired["ΔMAE(E2−E1)"] = paired.e2_minus_e1_mae.round(2)
    paired_table = paired[["lead_min", "group", "n", "ΔRMSE(E2−E1)", "ΔMAE(E2−E1)"]]
    improvement = paired.groupby("group").e2_minus_e1_rmse.mean().to_dict()
    direction = {group: ("下降" if value < 0 else "上升") for group, value in improvement.items()}
    parameters = []
    for folder in sorted((args.run_root / "conditions").iterdir()):
        path = folder / "run.json"
        if path.exists():
            run = json.loads(path.read_text())
            parameters.append({"model": folder.name, "parameters": run.get("parameter_count", 0),
                               "best_epoch": run.get("best_epoch"),
                               "runtime_seconds": round(run.get("runtime_seconds", 0), 1)})
    text = f"""# E2 加性残差预测实验报告

## 结论

本报告对应单种子探索性跨年实验。E2 相对 E1 的测试 RMSE 在整体样本上平均{direction.get('all', '未知')}，在 Ramp 样本上平均{direction.get('ramp', '未知')}，在非 Ramp 样本上平均{direction.get('non_ramp', '未知')}。具体差值见配对表。该结果用于判断残差分解在当前协议下的效果，不支持跨种子显著性结论。

## 主对照

{markdown_table(table)}

## E2 相对 E1 的配对差值

负值表示 E2 误差较低。

{markdown_table(paired_table)}

## Ramp 事件

{markdown_table(events[events.split == 'test'].round(3))}

## 参数量与运行开销

{markdown_table(pd.DataFrame(parameters))}

## 产物与限制

- 所有模型共享同一测试起报时刻和目标，测试结果未参与权重选择。
- `plots/ramp_day.png` 与 `plots/stable_day.png` 分别按测试真值选择 Ramp 最密集和最稀疏的日期，仅用于事后可视化。
- 图像时钟采用原始归档 mtime，并只向后匹配起报时可用帧。
- Folsom 数据发布不规定本研究的跨年预测任务和模型评估器，因此结果属于本仓库冻结的自定义协议。
"""
    (args.run_root / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()

