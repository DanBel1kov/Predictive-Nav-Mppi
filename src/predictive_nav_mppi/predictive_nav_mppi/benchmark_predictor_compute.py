#!/usr/bin/env python3
"""Measure compute footprint of trajectory predictors: params, latency, throughput, memory.

Usage:
  ros2 run predictive_nav_mppi benchmark_predictor_compute \
    --residual_ckpt /path/to/best_residual_model.pt \
    --social_vae_repo /home/danbel1kov/SocialVAE \
    --social_vae_ckpt /path/to/social_vae_finetuned_force0p3_stable \
    --batch_size 1 --warmup 20 --iters 200 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _timeit(fn, *, warmup: int, iters: int, device: torch.device) -> Dict[str, float]:
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples: List[float] = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples)
    out = {
        "mean_ms": float(arr.mean() * 1000.0),
        "std_ms": float(arr.std(ddof=1) * 1000.0) if arr.size > 1 else 0.0,
        "median_ms": float(np.median(arr) * 1000.0),
        "p90_ms": float(np.percentile(arr, 90) * 1000.0),
        "p99_ms": float(np.percentile(arr, 99) * 1000.0),
        "min_ms": float(arr.min() * 1000.0),
    }
    if is_cuda:
        out["peak_mem_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    return out


def _bench_residual(ckpt_path: str, device: torch.device, batch_size: int,
                    warmup: int, iters: int) -> Dict[str, Any]:
    from predictive_nav_mppi.models.kalman_residual_net import load_checkpoint
    model, cfg = load_checkpoint(ckpt_path, device=str(device))
    model.eval()
    params = _count_params(model)

    pred_len = int(cfg.get("pred_len", 12))
    obs_len = int(cfg.get("obs_len", 8))
    target_dim = int(cfg.get("target_input_dim", 8))
    neigh_dim = int(cfg.get("neighbor_input_dim", 5))
    k_neigh = int(cfg.get("k_neighbors", 3))
    scene_size = int(cfg.get("scene_size", 32))

    B = batch_size
    tgt = torch.randn(B, obs_len, target_dim, device=device)
    nbr = torch.randn(B, k_neigh, obs_len, neigh_dim, device=device)
    msk = torch.ones(B, k_neigh, device=device)
    soc = torch.randn(B, 11, device=device)
    klf = torch.randn(B, 2 * pred_len + 4, device=device)
    sce = torch.randn(B, 1, scene_size, scene_size, device=device)

    has_scene = hasattr(model, "scene_cnn")

    @torch.no_grad()
    def fwd():
        if has_scene:
            _ = model(tgt, nbr, msk, soc, klf, sce)
        else:
            _ = model(tgt, nbr, msk, soc, klf)

    timing = _timeit(fwd, warmup=warmup, iters=iters, device=device)
    return {
        "name": "residual_mlp",
        "params": params,
        "size_mb_fp32": float(params["total"] * 4 / (1024 * 1024)),
        "batch": B,
        "obs_len": obs_len,
        "pred_len": pred_len,
        "timing": timing,
        "throughput_preds_per_sec": float(B * 1000.0 / max(1e-9, timing["mean_ms"])),
    }


def _bench_social_vae(repo: str, ckpt_dir: str, device: torch.device,
                      batch_size: int, warmup: int, iters: int,
                      obs_len: int, pred_len: int, n_samples: int,
                      max_neighbors: int, hidden_dim: int, ob_radius: float) -> Dict[str, Any]:
    repo_p = str(Path(repo).expanduser().resolve())
    if repo_p not in sys.path:
        sys.path.insert(0, repo_p)
    from social_vae import SocialVAE

    model = SocialVAE(horizon=pred_len, ob_radius=ob_radius, hidden_dim=hidden_dim).to(device)
    ckpt_dir_p = Path(ckpt_dir).expanduser().resolve()
    src = None
    for name in ("ckpt-best", "ckpt-last"):
        cand = ckpt_dir_p / name
        if cand.exists():
            src = cand
            break
    if src is None and ckpt_dir_p.is_file():
        src = ckpt_dir_p
    if src is None:
        raise FileNotFoundError(f"SocialVAE checkpoint not found in {ckpt_dir_p}")
    state = torch.load(str(src), map_location=device)
    sd = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    model.eval()
    params = _count_params(model)

    B = batch_size
    L = obs_len
    Nn = max_neighbors
    x = torch.randn(L, B, 6, device=device)
    neighbor = torch.full((L, B, Nn, 6), 1e9, device=device)
    neighbor[:, :, :max(1, Nn // 2), :] = torch.randn(L, B, max(1, Nn // 2), 6, device=device)

    @torch.no_grad()
    def fwd():
        _ = model(x, neighbor, n_predictions=n_samples)

    timing = _timeit(fwd, warmup=warmup, iters=iters, device=device)
    return {
        "name": "social_vae",
        "params": params,
        "size_mb_fp32": float(params["total"] * 4 / (1024 * 1024)),
        "batch": B,
        "obs_len": L,
        "pred_len": pred_len,
        "n_samples_per_query": n_samples,
        "timing": timing,
        "throughput_preds_per_sec": float(B * 1000.0 / max(1e-9, timing["mean_ms"])),
    }


def _bench_kalman_baseline(device: torch.device, batch_size: int, warmup: int, iters: int,
                           pred_len: int) -> Dict[str, Any]:
    """Constant-velocity Kalman baseline — pure NumPy reference."""
    B = batch_size
    obs = np.random.randn(B, 8, 2).astype(np.float64)
    dt = 0.4

    def fwd():
        # Predict last_pos + last_velocity * k for k=1..pred_len
        last = obs[:, -1, :]
        prev = obs[:, -2, :]
        v = (last - prev) / dt
        out = np.zeros((B, pred_len, 2))
        for k in range(pred_len):
            out[:, k, :] = last + v * dt * (k + 1)
        return out

    # warmup
    for _ in range(warmup):
        fwd()
    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fwd()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples)
    timing = {
        "mean_ms": float(arr.mean() * 1000.0),
        "std_ms": float(arr.std(ddof=1) * 1000.0) if arr.size > 1 else 0.0,
        "median_ms": float(np.median(arr) * 1000.0),
        "p90_ms": float(np.percentile(arr, 90) * 1000.0),
        "p99_ms": float(np.percentile(arr, 99) * 1000.0),
        "min_ms": float(arr.min() * 1000.0),
    }
    return {
        "name": "kalman_cv",
        "params": {"total": 0, "trainable": 0},
        "size_mb_fp32": 0.0,
        "batch": B,
        "obs_len": 8,
        "pred_len": pred_len,
        "timing": timing,
        "throughput_preds_per_sec": float(B * 1000.0 / max(1e-9, timing["mean_ms"])),
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
    print()
    print(f"{'model':<14} {'params':>10} {'size_MB':>9} {'batch':>6} "
          f"{'mean_ms':>10} {'std_ms':>8} {'p99_ms':>9} {'thr/sec':>10} {'peak_MB':>10}")
    print("-" * 96)
    for r in rows:
        n = r["params"]["total"]
        t = r["timing"]
        peak = t.get("peak_mem_mb", float("nan"))
        print(f"{r['name']:<14} "
              f"{_format_count(n):>10} "
              f"{r['size_mb_fp32']:>8.2f} "
              f"{r['batch']:>6} "
              f"{t['mean_ms']:>9.3f} "
              f"{t['std_ms']:>7.3f} "
              f"{t['p99_ms']:>8.3f} "
              f"{r['throughput_preds_per_sec']:>10.1f} "
              f"{peak:>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute footprint of trajectory predictors")
    parser.add_argument("--residual_ckpt", default="", help="Path to KalmanResidualNet checkpoint")
    parser.add_argument("--social_vae_repo", default="", help="Path to SocialVAE repo")
    parser.add_argument("--social_vae_ckpt", default="", help="Path to SocialVAE ckpt dir")
    parser.add_argument("--social_vae_obs_len", type=int, default=8)
    parser.add_argument("--social_vae_pred_len", type=int, default=12)
    parser.add_argument("--social_vae_samples", type=int, default=20)
    parser.add_argument("--social_vae_max_neighbors", type=int, default=8)
    parser.add_argument("--social_vae_hidden_dim", type=int, default=256)
    parser.add_argument("--social_vae_ob_radius", type=float, default=2.0)
    parser.add_argument("--include_kalman", action="store_true", default=True,
                        help="Also benchmark Kalman CV baseline")
    parser.add_argument("--kalman_pred_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference. Real-time prediction usually uses B equal to number of tracked people.")
    parser.add_argument("--batches", default="",
                        help="Optional: comma-separated list of batch sizes to sweep (e.g. '1,4,8,16'). Overrides --batch_size.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default="",
                        help="Where to write JSON results (default: prints only).")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}  cuda_available={torch.cuda.is_available()}")
    batch_sizes: List[int] = (
        [int(b) for b in str(args.batches).split(",") if b.strip()]
        if args.batches.strip() else [int(args.batch_size)]
    )

    all_results: List[Dict[str, Any]] = []
    for B in batch_sizes:
        rows: List[Dict[str, Any]] = []
        if args.include_kalman:
            print(f"\n[bench] kalman_cv  batch={B}")
            rows.append(_bench_kalman_baseline(device=device, batch_size=B,
                                               warmup=args.warmup, iters=args.iters,
                                               pred_len=int(args.kalman_pred_len)))
        if args.residual_ckpt:
            print(f"[bench] residual_mlp  batch={B}  ckpt={args.residual_ckpt}")
            rows.append(_bench_residual(args.residual_ckpt, device=device, batch_size=B,
                                        warmup=args.warmup, iters=args.iters))
        if args.social_vae_repo and args.social_vae_ckpt:
            print(f"[bench] social_vae  batch={B}  ckpt={args.social_vae_ckpt}  "
                  f"samples={args.social_vae_samples}")
            rows.append(_bench_social_vae(
                repo=args.social_vae_repo,
                ckpt_dir=args.social_vae_ckpt,
                device=device,
                batch_size=B,
                warmup=args.warmup,
                iters=args.iters,
                obs_len=int(args.social_vae_obs_len),
                pred_len=int(args.social_vae_pred_len),
                n_samples=int(args.social_vae_samples),
                max_neighbors=int(args.social_vae_max_neighbors),
                hidden_dim=int(args.social_vae_hidden_dim),
                ob_radius=float(args.social_vae_ob_radius),
            ))
        _print_table(rows)
        all_results.append({"batch": B, "rows": rows})

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "device": str(device),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "results": all_results,
        }, indent=2))
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
