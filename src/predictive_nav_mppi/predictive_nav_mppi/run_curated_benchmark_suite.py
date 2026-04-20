#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_dataset = repo_root / "datasets" / "curated_people" / "benchmark_cases.json"
    default_gru = repo_root / "src" / "predictive_nav_mppi" / "predictive_nav_mppi" / "models" / "best_model.pt"
    default_vae_ckpt = repo_root / "src" / "predictive_nav_mppi" / "predictive_nav_mppi" / "models" / "vae_hotel"
    default_residual = repo_root / "models" / "residual_predictor" / "best_residual_model.pt"

    parser = argparse.ArgumentParser(
        description="Run offline predictor benchmark on curated cases for horizons 12 and 26."
    )
    parser.add_argument("--dataset", default=str(default_dataset))
    parser.add_argument("--output_dir", default=str(repo_root / "benchmark_people_predictors" / "curated_suite"))
    parser.add_argument("--pred_steps", default="12,26")
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--obs_dt", type=float, default=0.4)
    parser.add_argument("--pred_dt", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--neighbor_radius", type=float, default=2.0)
    parser.add_argument("--max_neighbors", type=int, default=16)
    parser.add_argument("--progress_step_pct", type=int, default=10)
    parser.add_argument("--social_gru_weights", default=str(default_gru))
    parser.add_argument("--social_gru_device", default="")
    parser.add_argument("--disable_social_gru", action="store_true")
    parser.add_argument("--social_vae_repo_path", default="")
    parser.add_argument("--social_vae_ckpt_path", default=str(default_vae_ckpt))
    parser.add_argument("--social_vae_config_path", default="")
    parser.add_argument("--social_vae_samples", type=int, default=20)
    parser.add_argument("--social_vae_device", default="")
    parser.add_argument("--residual_model_weights", default=str(default_residual))
    parser.add_argument("--residual_model_device", default="")
    parser.add_argument("--residual_alpha", type=float, default=0.3)
    parser.add_argument("--residual_smoothing_beta", type=float, default=0.8)
    parser.add_argument("--residual_clip_norm", type=float, default=0.35)
    parser.add_argument("--disable_residual_turn_gate", action="store_true")
    parser.add_argument("--residual_turn_gate_tau", type=float, default=0.1)
    parser.add_argument("--residual_turn_gate_alpha", type=float, default=30.0)
    parser.add_argument("--report_splits", default="all,interaction,turning,dense_interaction,complex,interaction_turning")
    parser.add_argument("--min_split_cases", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    dataset = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "predictive_nav_mppi.benchmark_people_predictors",
        "--dataset",
        str(dataset),
        "--output_dir",
        str(output_dir),
        "--pred_steps",
        str(args.pred_steps),
        "--obs_len",
        str(args.obs_len),
        "--obs_dt",
        str(args.obs_dt),
        "--pred_dt",
        str(args.pred_dt),
        "--batch_size",
        str(args.batch_size),
        "--neighbor_radius",
        str(args.neighbor_radius),
        "--max_neighbors",
        str(args.max_neighbors),
        "--progress_step_pct",
        str(args.progress_step_pct),
        "--report_splits",
        str(args.report_splits),
        "--min_split_cases",
        str(args.min_split_cases),
    ]

    gru_path = Path(args.social_gru_weights).expanduser()
    if (not args.disable_social_gru) and gru_path.is_file():
        cmd.extend(["--social_gru_weights", str(gru_path.resolve())])
        if args.social_gru_device:
            cmd.extend(["--social_gru_device", args.social_gru_device])
    else:
        print(f"[suite] skip SocialGRU: {'disabled' if args.disable_social_gru else f'weights not found at {gru_path}'}")

    if args.social_vae_repo_path:
        cmd.extend(["--social_vae_repo_path", str(Path(args.social_vae_repo_path).expanduser().resolve())])
        cmd.extend(["--social_vae_ckpt_path", str(Path(args.social_vae_ckpt_path).expanduser().resolve())])
        if args.social_vae_config_path:
            cmd.extend(["--social_vae_config_path", str(Path(args.social_vae_config_path).expanduser().resolve())])
        cmd.extend(["--social_vae_samples", str(args.social_vae_samples)])
        if args.social_vae_device:
            cmd.extend(["--social_vae_device", args.social_vae_device])
    else:
        print("[suite] skip SocialVAE: --social_vae_repo_path not provided")

    residual_path = Path(args.residual_model_weights).expanduser()
    if residual_path.is_file():
        cmd.extend(["--residual_model_weights", str(residual_path.resolve())])
        cmd.extend(["--residual_alpha", str(args.residual_alpha)])
        cmd.extend(["--residual_smoothing_beta", str(args.residual_smoothing_beta)])
        cmd.extend(["--residual_clip_norm", str(args.residual_clip_norm)])
        if args.disable_residual_turn_gate:
            cmd.append("--disable_residual_turn_gate")
        cmd.extend(["--residual_turn_gate_tau", str(args.residual_turn_gate_tau)])
        cmd.extend(["--residual_turn_gate_alpha", str(args.residual_turn_gate_alpha)])
        if args.residual_model_device:
            cmd.extend(["--residual_model_device", args.residual_model_device])
    else:
        print(f"[suite] skip residual model: weights not found at {residual_path}")

    print("[suite] running command:")
    print("PYTHONPATH={} {}".format(repo_root / "src" / "predictive_nav_mppi", " ".join(cmd)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src" / "predictive_nav_mppi")
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
