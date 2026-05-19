#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = [
    "linear",
    "interaction",
    "dense_interaction",
    "turning",
    "stop_go",
    "complex",
]


def _stats(vals: List[float]) -> Dict[str, float]:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(xs),
        "max": max(xs),
    }


def _load_manifest(study_dir: Path) -> Dict[str, Any]:
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest.json not found in {study_dir}")
    return json.loads(manifest_path.read_text())


def _load_run_payload(run_dir: Path) -> Dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text())
    interaction_summary = json.loads((run_dir / "interaction_summary.json").read_text())
    return {"summary": summary, "interaction_summary": interaction_summary}


def _format_stats(label: str, values: List[float]) -> str:
    s = _stats(values)
    if not math.isfinite(s["mean"]):
        return f"{label}: n/a"
    return f"{label}: {s['mean']:.3f} ± {s['std']:.3f} [{s['min']:.3f} .. {s['max']:.3f}]"


def main():
    parser = argparse.ArgumentParser(description="Build text report and category bar chart for robot-force interaction study.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--forces", default="0.0,1.0",
                        help="Comma-separated forces to compare in one plot/report, e.g. 0.0,0.5,1.0")
    args = parser.parse_args()

    study_dir = Path(args.study_dir).resolve()
    manifest = _load_manifest(study_dir)
    requested_forces = [float(x.strip()) for x in str(args.forces).split(",") if x.strip()]
    if len(requested_forces) < 2:
        raise SystemExit("--forces must contain at least two values")

    run_by_force = {float(item["force"]): Path(item["run_dir"]) for item in manifest.get("forces", [])}
    missing = [force for force in requested_forces if force not in run_by_force]
    if missing:
        raise SystemExit(f"Missing runs for forces: {missing}")

    payloads = {force: _load_run_payload(run_by_force[force]) for force in requested_forces}

    report_lines: List[str] = []
    report_lines.append(f"Study: {manifest.get('study_name', study_dir.name)}")
    report_lines.append(f"Predictor: {manifest.get('predictor', 'unknown')}")
    report_lines.append(f"Episodes per force: {manifest.get('episode_count', 'unknown')}")
    report_lines.append("")

    for force in requested_forces:
        payload = payloads[force]
        rows = payload["summary"]
        ok = [row for row in rows if row.get("status") == "SUCCEEDED"]
        counts = payload["interaction_summary"].get("counts", {})
        report_lines.append(f"Force {force:.2f}")
        report_lines.append(f"  episodes: {len(rows)}")
        report_lines.append(f"  succeeded: {len(ok)}")
        report_lines.append(f"  {_format_stats('time_to_goal', [row.get('time_to_goal') for row in ok])}")
        report_lines.append(f"  {_format_stats('path_length', [row.get('path_length') for row in ok])}")
        report_lines.append(f"  {_format_stats('min_dist', [row.get('min_dist') for row in ok])}")
        report_lines.append(f"  {_format_stats('viol_time', [row.get('viol_time') for row in ok])}")
        report_lines.append(f"  {_format_stats('avg_robot_influence', [row.get('avg_robot_influence') for row in ok])}")
        report_lines.append("  interaction category counts:")
        for category in CATEGORY_ORDER:
            report_lines.append(f"    {category}: {int(counts.get(category, 0))}")
        report_lines.append("")

    report_path = study_dir / "force_interaction_report.txt"
    report_path.write_text("\n".join(report_lines))

    x = np.arange(len(CATEGORY_ORDER))
    n_forces = len(requested_forces)
    width = min(0.8 / max(1, n_forces), 0.36)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, force in enumerate(requested_forces):
        counts = payloads[force]["interaction_summary"].get("counts", {})
        total = sum(int(counts.get(category, 0)) for category in CATEGORY_ORDER)
        y = [100.0 * int(counts.get(category, 0)) / max(1, total) for category in CATEGORY_ORDER]
        offset = (idx - (n_forces - 1) / 2.0) * width
        ax.bar(x + offset, y, width=width, label=f"force={force:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Visible Human Interaction Categories Near Robot")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    plot_path = study_dir / "force_interaction_counts.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Report saved to {report_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
