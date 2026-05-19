#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence


def _load_manifest(study_dir: Path) -> Dict:
    return json.loads((study_dir / "manifest.json").read_text())


def _read_results_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _as_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except Exception:
        return 0


def _stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": mean, "std": math.sqrt(var), "min": min(vals), "max": max(vals)}


def _plot(study_dir: Path, payloads: List[Dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_defs = [
        ("time_to_goal", "Time To Goal [s]"),
        ("path_length", "Path Length [m]"),
        ("min_dist", "Min Distance [m]"),
        ("avg_dist", "Avg Distance [m]"),
        ("collision_count", "Collisions"),
        ("viol_time", "Violation Time [s]"),
        ("avg_robot_influence", "Robot Influence"),
        ("nearest_robot_influence", "Nearest Robot Influence"),
    ]

    labels = [f"{item['force']:.2f}" for item in payloads]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.reshape(-1)

    for ax, (metric_key, title) in zip(axes, metric_defs):
        series = [item["metrics"][metric_key] for item in payloads]
        bp = ax.boxplot(series, patch_artist=True, labels=labels, widths=0.6, showfliers=True)
        palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=palette[i % len(palette)], alpha=0.45, edgecolor="#333333")
        for median in bp["medians"]:
            median.set(color="#111111", linewidth=1.5)

        means = [item["stats"][metric_key]["mean"] for item in payloads]
        ax.scatter(range(1, len(labels) + 1), means, color="#d62728", marker="D", s=32, zorder=4, label="mean")
        for idx, mu in enumerate(means, start=1):
            if math.isfinite(mu):
                ax.text(idx, mu, f"{mu:.2f}", ha="center", va="bottom", fontsize=8, color="#8b0000")

        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xlabel("robot_force_scale")
        if metric_key in ("avg_robot_influence", "nearest_robot_influence"):
            ax.legend(loc="upper left", fontsize=8)

    success_lines = [
        f"force={item['force']:.2f}: {item['successes']}/{item['episodes']} succeeded"
        for item in payloads
    ]
    fig.suptitle("Benchmark Episode Metrics By Robot Force", fontsize=15)
    fig.text(0.5, 0.02, " | ".join(success_lines), ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.96))

    out_path = study_dir / "benchmark_metrics_boxplot.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _write_report(study_dir: Path, payloads: List[Dict]) -> Path:
    metric_defs = [
        ("time_to_goal", "time_to_goal"),
        ("path_length", "path_length"),
        ("min_dist", "min_dist"),
        ("avg_dist", "avg_distance"),
        ("collision_count", "collisions"),
        ("viol_time", "violation_time"),
        ("avg_robot_influence", "robot_influence"),
        ("nearest_robot_influence", "nearest_robot_influence"),
    ]
    lines: List[str] = []
    lines.append(f"Study: {study_dir.name}")
    lines.append("")
    for item in payloads:
        lines.append(f"Force {item['force']:.2f}")
        lines.append(f"  success: {item['successes']}/{item['episodes']}")
        for key, label in metric_defs:
            s = item["stats"][key]
            lines.append(
                f"  {label}: {s['mean']:.3f} ± {s['std']:.3f}  "
                f"[{s['min']:.3f} .. {s['max']:.3f}]"
            )
        lines.append("")
    out_path = study_dir / "benchmark_metrics_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-episode benchmark metrics for a robot-force sweep study.")
    parser.add_argument("--study-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = Path(args.study_dir).expanduser().resolve()
    manifest = _load_manifest(study_dir)

    payloads: List[Dict] = []
    for force_item in manifest.get("forces", []):
        run_dir = Path(force_item["run_dir"])
        rows = _read_results_csv(run_dir / "results.csv")
        metric_values = {
            "time_to_goal": [_as_float(r, "time_to_goal") for r in rows],
            "path_length": [_as_float(r, "path_length") for r in rows],
            "min_dist": [_as_float(r, "min_dist") for r in rows],
            "avg_dist": [_as_float(r, "avg_dist") for r in rows],
            "collision_count": [float(_as_int(r, "collision_count")) for r in rows],
            "viol_time": [_as_float(r, "viol_time") for r in rows],
            "avg_robot_influence": [_as_float(r, "avg_robot_influence") for r in rows],
            "nearest_robot_influence": [_as_float(r, "nearest_robot_influence") for r in rows],
        }
        payloads.append(
            {
                "force": float(force_item["force"]),
                "run_dir": str(run_dir),
                "episodes": len(rows),
                "successes": sum(1 for r in rows if str(r.get("status", "")).upper() == "SUCCEEDED"),
                "metrics": metric_values,
                "stats": {k: _stats(v) for k, v in metric_values.items()},
            }
        )

    plot_path = _plot(study_dir, payloads)
    report_path = _write_report(study_dir, payloads)
    print(f"Plot saved to {plot_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
