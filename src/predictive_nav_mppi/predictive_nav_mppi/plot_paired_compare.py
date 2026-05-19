#!/usr/bin/env python3
"""Aggregate cached_summary.json across all variants of a paired study and produce
   boxplots + cross-variant comparison tables.

Usage:
  python3 -m predictive_nav_mppi.plot_paired_compare --study-dir benchmark_paired/<study_id>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


METRICS = [
    ("time_to_goal", "lower"),
    ("path_length", "lower"),
    ("min_dist", "higher"),
    ("avg_dist", "higher"),
    ("collision_count", "lower"),
    ("viol_time", "lower"),
    ("avg_robot_influence", "lower"),
    ("nearest_robot_influence", "lower"),
    ("peak_robot_influence", "lower"),
    ("close_robot_influence", "lower"),
    ("robot_influence_auc", "lower"),
]


def _load_variant_results(study_dir: Path) -> Dict[str, List[Dict]]:
    variants = {}
    for cache_path in sorted(study_dir.glob("*/cached_summary.json")):
        variant_id = cache_path.parent.name
        data = json.loads(cache_path.read_text())
        results = data if isinstance(data, list) else data.get("results", [])
        variants[variant_id] = results
    return variants


def _metric_value(result: Dict, key: str, nav_timeout: float) -> float:
    if key == "time_to_goal" and result.get("status") != "SUCCEEDED":
        return float(nav_timeout)
    v = result.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _summary(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _permutation_p(diffs: Sequence[float], samples: int = 20000, seed: int = 19) -> float:
    diffs = [d for d in diffs if np.isfinite(d)]
    if not diffs:
        return float("nan")
    observed = abs(np.mean(diffs))
    n = len(diffs)
    rng = np.random.default_rng(seed)
    if n <= 16:
        total = 1 << n
        extreme = 0
        diffs_arr = np.asarray(diffs)
        for mask in range(total):
            signs = np.array([1 if (mask & (1 << i)) else -1 for i in range(n)])
            if abs((signs * diffs_arr).mean()) >= observed - 1e-12:
                extreme += 1
        return extreme / total
    extreme = 0
    diffs_arr = np.asarray(diffs)
    for _ in range(samples):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((signs * diffs_arr).mean()) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def _build_boxplots(variants: Dict[str, List[Dict]], nav_timeout: float, out_path: Path) -> None:
    if not MPL_AVAILABLE:
        print("matplotlib not available; skipping boxplots")
        return
    variant_names = sorted(variants.keys())
    n_metrics = len(METRICS)
    cols = 3
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_2d(axes).reshape(-1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_names)))
    for ax, (metric, direction) in zip(axes, METRICS):
        data = []
        labels = []
        for vname in variant_names:
            vals = [_metric_value(r, metric, nav_timeout) for r in variants[vname]]
            vals = [v for v in vals if np.isfinite(v)]
            data.append(vals)
            labels.append(vname)
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 6})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title(f"{metric} ({'lower=better' if direction == 'lower' else 'higher=better'})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        # annotate means
        for i, vals in enumerate(data):
            if vals:
                mean = float(np.mean(vals))
                ax.text(i + 1, mean, f"{mean:.3g}", fontsize=7, ha="center", va="bottom", color="red")
    for ax in axes[len(METRICS):]:
        ax.axis("off")
    fig.suptitle(f"Navigation benchmark — boxplots ({out_path.parent.name})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def _build_pairwise_table(variants: Dict[str, List[Dict]], nav_timeout: float, out_dir: Path) -> None:
    names = sorted(variants.keys())
    by_key = {n: {(int(r["goal_idx"]), int(r["repeat"])): r for r in variants[n] if "goal_idx" in r and "repeat" in r} for n in names}

    lines: List[str] = []
    lines.append(f"Pairwise comparison — {len(names)} variants: {names}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            keys = sorted(set(by_key[a]) & set(by_key[b]))
            lines.append("")
            lines.append(f"### {a}  vs  {b}   (paired n={len(keys)})")
            lines.append(f"{'metric':<22} {'direction':<8} {a+'_mean':>14} {b+'_mean':>14} {'Δ(b-a)':>12} {'rel%':>8} {'p':>10}")
            for metric, direction in METRICS:
                a_vals = []; b_vals = []; diffs = []
                for key in keys:
                    av = _metric_value(by_key[a][key], metric, nav_timeout)
                    bv = _metric_value(by_key[b][key], metric, nav_timeout)
                    if np.isfinite(av) and np.isfinite(bv):
                        a_vals.append(av); b_vals.append(bv); diffs.append(bv - av)
                if not diffs:
                    continue
                a_mean = float(np.mean(a_vals)); b_mean = float(np.mean(b_vals))
                d_mean = float(np.mean(diffs))
                rel = (100.0 * d_mean / a_mean) if abs(a_mean) > 1e-9 else float("nan")
                p = _permutation_p(diffs)
                lines.append(f"{metric:<22} {direction:<8} {a_mean:>14.4g} {b_mean:>14.4g} {d_mean:>12.4g} {rel:>7.1f}% {p:>10.4g}")
    out_txt = out_dir / "pairwise_comparison.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"[saved] {out_txt}")
    print("\n" + "\n".join(lines))


def _build_summary_json(variants: Dict[str, List[Dict]], nav_timeout: float, out_path: Path) -> None:
    summary = {"variants": {}}
    for vname, results in variants.items():
        success = sum(1 for r in results if r.get("status") == "SUCCEEDED")
        block = {"episodes": len(results), "success": success}
        for metric, _ in METRICS:
            vals = [_metric_value(r, metric, nav_timeout) for r in results]
            block[metric] = _summary(vals)
        summary["variants"][vname] = block
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(f"[saved] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build boxplots + pairwise table from paired-benchmark cache")
    parser.add_argument("--study-dir", required=True, help="Path like benchmark_paired/<study_id>")
    parser.add_argument("--nav-timeout", type=float, default=120.0,
                        help="Time-to-goal used when status != SUCCEEDED (matches benchmark_session nav_timeout)")
    args = parser.parse_args()

    study_dir = Path(args.study_dir).resolve()
    if not study_dir.is_dir():
        raise SystemExit(f"--study-dir not found: {study_dir}")

    variants = _load_variant_results(study_dir)
    if not variants:
        raise SystemExit(f"No cached_summary.json under {study_dir}")
    print(f"[load] {len(variants)} variants: {list(variants.keys())}")
    print(f"[load] episodes per variant: " + ", ".join(f"{k}={len(v)}" for k, v in variants.items()))

    _build_summary_json(variants, args.nav_timeout, study_dir / "compare_summary.json")
    _build_pairwise_table(variants, args.nav_timeout, study_dir)
    _build_boxplots(variants, args.nav_timeout, study_dir / "metrics_boxplots.png")


if __name__ == "__main__":
    main()
