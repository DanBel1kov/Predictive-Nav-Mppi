#!/usr/bin/env python3
"""End-to-end training + evaluation runner for residual predictor.

Creates a timestamped run directory containing:
  - best_residual_model.pt + train_history.json (from training)
  - plots/loss.png, plots/ade_fde.png, plots/jerk_acc.png  (training curves)
  - command.txt  (the exact command that was launched)
  - val_benchmark/  (benchmark on curated val/benchmark cases; per-tag ADE/FDE plots)
  - test_benchmark/ (benchmark on raw test dataset, streaming metrics, plots)  [optional]

Usage:
    ros2 run predictive_nav_mppi train_and_evaluate \
        --run_tag react_v3_smooth_jerk05 \
        --train_dataset .../train_residual_cases.json \
        --val_dataset   .../benchmark_residual_cases.json \
        --raw_test_dataset .../raw/long_corridor_react_better_rft.json \
        --include_robot --lambda_jerk 0.05 --lambda_acc_match 0.5 \
        --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")


def _plot_training(history: List[Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping training plots")
        return

    epochs = [r["epoch"] for r in history]

    # === Total loss (train vs val, single panel) ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train"]["loss"] for r in history], label="train (eval mode)", color="#1f77b4")
    ax.plot(epochs, [r["val"]["loss"] for r in history], "--", label="val", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title("Total loss (frozen-weights eval; train ≤ val = good)")
    ax.legend()
    ax.grid(alpha=0.3)
    _safe_savefig(fig, out_dir / "loss_total.png")
    plt.close(fig)

    # === Each component on its own subplot ===
    components = [
        ("loss_pos", "Position SmoothL1"),
        ("loss_vel", "Velocity match"),
        ("loss_acc", "Acceleration match"),
        ("loss_jerk", "Jerk smoothness"),
        ("loss_res_mag", "Residual magnitude"),
        ("loss_risk", "Risk-weighted pos"),
        ("loss_safety", "Asymmetric safety"),
    ]
    active = [(k, t) for (k, t) in components if any(r["train"].get(k, 0) > 0 or r["val"].get(k, 0) > 0 for r in history)]
    n = len(active)
    if n > 0:
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
        for i, (key, title) in enumerate(active):
            ax = axes[i // cols][i % cols]
            tr = [r["train"].get(key, 0) for r in history]
            va = [r["val"].get(key, 0) for r in history]
            ax.plot(epochs, tr, label="train", color="#1f77b4")
            ax.plot(epochs, va, "--", label="val", color="#d62728")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            # Auto-clip y to robust range (ignore outliers in epoch 1)
            all_vals = tr + va
            if len(all_vals) > 5:
                lo = min(all_vals)
                hi = sorted(all_vals)[max(0, int(len(all_vals) * 0.95) - 1)]
                if hi > lo:
                    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.15 * (hi - lo))
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")
        _safe_savefig(fig, out_dir / "loss.png")
        plt.close(fig)

    # ADE/FDE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, [r["train"]["hybrid_ade"] for r in history], label="train hybrid_ADE")
    axes[0].plot(epochs, [r["val"]["hybrid_ade"] for r in history], label="val hybrid_ADE")
    axes[0].plot(epochs, [r["val"]["kalman_ade"] for r in history], "--", label="val kalman_ADE", color="grey")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("ADE (local frame)")
    axes[0].set_title("Average Displacement Error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [r["train"]["hybrid_fde"] for r in history], label="train hybrid_FDE")
    axes[1].plot(epochs, [r["val"]["hybrid_fde"] for r in history], label="val hybrid_FDE")
    axes[1].plot(epochs, [r["val"]["kalman_fde"] for r in history], "--", label="val kalman_FDE", color="grey")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("FDE")
    axes[1].set_title("Final Displacement Error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _safe_savefig(fig, out_dir / "ade_fde.png")
    plt.close(fig)

    # Jerk / acc-match (if used)
    if any(r["train"].get("loss_jerk", 0) > 0 or r["train"].get("loss_acc", 0) > 0 for r in history):
        fig, ax = plt.subplots(figsize=(8, 5))
        if any(r["train"].get("loss_jerk", 0) > 0 for r in history):
            ax.plot(epochs, [r["train"].get("loss_jerk", 0) for r in history], label="train loss_jerk")
            ax.plot(epochs, [r["val"].get("loss_jerk", 0) for r in history], "--", label="val loss_jerk")
        if any(r["train"].get("loss_acc", 0) > 0 for r in history):
            ax.plot(epochs, [r["train"].get("loss_acc", 0) for r in history], label="train loss_acc")
            ax.plot(epochs, [r["val"].get("loss_acc", 0) for r in history], "--", label="val loss_acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("auxiliary loss")
        ax.set_title("Smoothness regularizers")
        ax.legend()
        ax.grid(alpha=0.3)
        _safe_savefig(fig, out_dir / "jerk_acc.png")
        plt.close(fig)


def _plot_benchmark(summary_path: Path, out_dir: Path, label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not summary_path.exists():
        print(f"{summary_path} missing; skipping {label} plots")
        return

    data = json.loads(summary_path.read_text())
    metrics = data.get("metrics", {})

    # Per-horizon, per-model ADE/FDE bar chart
    for hk, block in metrics.items():
        models = list(block.keys())
        ades = [block[m]["ade"]["mean"] for m in models]
        fdes = [block[m]["fde"]["mean"] for m in models]
        x = range(len(models))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(x, ades, color=["#888"] + ["#3a86ff"] * (len(models) - 1))
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(models, rotation=15)
        axes[0].set_ylabel("ADE")
        axes[0].set_title(f"{label}: ADE at horizon {hk}")
        for i, v in enumerate(ades):
            axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
        axes[1].bar(x, fdes, color=["#888"] + ["#3a86ff"] * (len(models) - 1))
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(models, rotation=15)
        axes[1].set_ylabel("FDE")
        axes[1].set_title(f"{label}: FDE at horizon {hk}")
        for i, v in enumerate(fdes):
            axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
        _safe_savefig(fig, out_dir / f"ade_fde_{hk}.png")
        plt.close(fig)

    # Per-tag bar chart on biggest horizon
    splits = data.get("metrics_by_split", {})
    if splits:
        hks = list(metrics.keys())
        if hks:
            hk = sorted(hks, key=lambda s: -int(s.lstrip("h")))[0]
            split_block = splits.get(hk, {})
            tags = list(split_block.keys())
            if tags:
                models = list(metrics[hk].keys())
                fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.2), 5))
                bar_w = 0.8 / max(1, len(models))
                for mi, m in enumerate(models):
                    vals = [split_block[t].get(m, {}).get("ade", {}).get("mean", 0) for t in tags]
                    pos = [i + (mi - len(models) / 2) * bar_w + bar_w / 2 for i in range(len(tags))]
                    ax.bar(pos, vals, width=bar_w, label=m)
                ax.set_xticks(range(len(tags)))
                ax.set_xticklabels(tags, rotation=20, ha="right")
                ax.set_ylabel("ADE")
                ax.set_title(f"{label}: ADE per tag (horizon {hk})")
                ax.legend()
                _safe_savefig(fig, out_dir / f"ade_per_tag_{hk}.png")
                plt.close(fig)


def _run(cmd: List[str], cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_tag", default="run", help="Tag for the run directory; timestamp is always appended.")
    parser.add_argument("--runs_root", default="/home/danbel1kov/predictive-nav-mppi/models/residual_predictor/runs")
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", required=True)
    parser.add_argument("--raw_test_dataset", default="",
                        help="Optional raw JSON for streaming metrics benchmark on test set (uses --frame_start_fraction 0.75).")

    # Pass-through training args (most important ones; rest can be added)
    parser.add_argument("--include_robot", action="store_true")
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=26)
    parser.add_argument("--obs_dt", type=float, default=0.4)
    parser.add_argument("--pred_dt", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--k_neighbors", type=int, default=3)
    parser.add_argument("--lambda_jerk", type=float, default=0.0)
    parser.add_argument("--lambda_acc_match", type=float, default=0.0)
    parser.add_argument("--lambda_vel", type=float, default=0.0)
    parser.add_argument("--lambda_res_mag", type=float, default=0.0)
    parser.add_argument("--lambda_risk", type=float, default=0.0)
    parser.add_argument("--lambda_safety", type=float, default=0.0)
    parser.add_argument("--risk_sigma", type=float, default=1.5)
    parser.add_argument("--safety_radius", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")

    # Benchmark args for test (raw) run
    parser.add_argument("--test_max_cases", type=int, default=10000)
    parser.add_argument("--test_residual_alpha", type=float, default=1.0,
                        help="Residual alpha for test benchmark (use 1.0 for pure model, 0.6 for MID, etc.)")
    parser.add_argument("--test_smoothing_beta", type=float, default=0.0)
    parser.add_argument("--test_clip_norm", type=float, default=0.0)
    parser.add_argument("--test_disable_turn_gate", action="store_true", default=True)
    parser.add_argument("--report_splits", default="all,turning,interaction,dense_interaction,stop_go,complex,very_complex")

    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_name = f"{args.run_tag}_{_now_tag()}"
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(c) for c in sys.argv))

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    val_dir = run_dir / "val_benchmark"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir = run_dir / "test_benchmark"

    # 1. Train
    train_cmd = [
        "ros2", "run", "predictive_nav_mppi", "train_residual_predictor",
        "--train_dataset", args.train_dataset,
        "--val_dataset", args.val_dataset,
        "--output_dir", str(run_dir),
        "--obs_len", str(args.obs_len),
        "--pred_len", str(args.pred_len),
        "--obs_dt", str(args.obs_dt),
        "--pred_dt", str(args.pred_dt),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--k_neighbors", str(args.k_neighbors),
        "--lambda_jerk", str(args.lambda_jerk),
        "--lambda_acc_match", str(args.lambda_acc_match),
        "--lambda_vel", str(args.lambda_vel),
        "--lambda_res_mag", str(args.lambda_res_mag),
        "--lambda_risk", str(args.lambda_risk),
        "--lambda_safety", str(args.lambda_safety),
        "--risk_sigma", str(args.risk_sigma),
        "--safety_radius", str(args.safety_radius),
        "--seed", str(args.seed),
    ]
    if args.include_robot:
        train_cmd.append("--include_robot")
    if args.device:
        train_cmd += ["--device", args.device]
    rc = _run(train_cmd)
    if rc != 0:
        print(f"Training failed with code {rc}", file=sys.stderr)
        sys.exit(rc)

    # 2. Training plots
    hist_path = run_dir / "train_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        _plot_training(history, plots_dir)
    else:
        print(f"train_history.json not found in {run_dir}, skipping plots")

    ckpt_path = run_dir / "best_residual_model.pt"
    if not ckpt_path.exists():
        print(f"WARNING: {ckpt_path} not found; benchmarks skipped")
        return

    # 3. Val benchmark (curated cases, deterministic)
    val_cmd = [
        "ros2", "run", "predictive_nav_mppi", "benchmark_people_predictors",
        "--datasets", args.val_dataset,
        "--output_dir", str(val_dir),
        "--residual_model_weights", str(ckpt_path),
        "--residual_alpha", str(args.test_residual_alpha),
        "--residual_smoothing_beta", str(args.test_smoothing_beta),
        "--residual_clip_norm", str(args.test_clip_norm),
        "--obs_len", str(args.obs_len),
        "--obs_dt", str(args.obs_dt),
        "--pred_steps", str(args.pred_len),
        "--pred_dt", str(args.pred_dt),
        "--stride", "1",
        "--report_splits", args.report_splits,
    ]
    if args.test_disable_turn_gate:
        val_cmd.append("--disable_residual_turn_gate")
    _run(val_cmd)
    _plot_benchmark(val_dir / "summary.json", val_dir, label="val")

    # 4. Test benchmark on raw dataset (streaming metrics)
    if args.raw_test_dataset:
        test_dir.mkdir(parents=True, exist_ok=True)
        test_cmd = [
            "ros2", "run", "predictive_nav_mppi", "benchmark_people_predictors",
            "--datasets", args.raw_test_dataset,
            "--output_dir", str(test_dir),
            "--residual_model_weights", str(ckpt_path),
            "--residual_alpha", str(args.test_residual_alpha),
            "--residual_smoothing_beta", str(args.test_smoothing_beta),
            "--residual_clip_norm", str(args.test_clip_norm),
            "--obs_len", str(args.obs_len),
            "--obs_dt", str(args.obs_dt),
            "--pred_steps", f"12,{args.pred_len}",
            "--pred_dt", str(args.pred_dt),
            "--stride", "1",
            "--frame_start_fraction", "0.75",
            "--streaming_metrics",
            "--max_cases", str(args.test_max_cases),
            "--report_splits", args.report_splits,
        ]
        if args.test_disable_turn_gate:
            test_cmd.append("--disable_residual_turn_gate")
        _run(test_cmd)
        _plot_benchmark(test_dir / "summary.json", test_dir, label="test")

    print(f"\n=== run finished: {run_dir} ===")


if __name__ == "__main__":
    main()
