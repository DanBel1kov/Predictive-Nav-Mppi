#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from predictive_nav_mppi.kf_cv import predict_state_cov, update_state_cov
from predictive_nav_mppi.models.social_gru import load_checkpoint, predict_trajectories_world
from predictive_nav_mppi.models.social_vae import load_external_social_vae, predict_social_vae_samples


@dataclass
class Case:
    t: float
    person_id: int
    obs_xy: np.ndarray
    neigh_xy: List[np.ndarray]
    gt_xy: np.ndarray


def _interp_xy(track_txy: np.ndarray, t: float) -> Optional[np.ndarray]:
    ts = track_txy[:, 0]
    if t < ts[0] or t > ts[-1]:
        return None
    idx = np.searchsorted(ts, t, side="left")
    if idx == 0:
        return track_txy[0, 1:3].copy()
    if idx >= len(ts):
        return track_txy[-1, 1:3].copy()
    t0, t1 = ts[idx - 1], ts[idx]
    if t1 <= t0:
        return track_txy[idx, 1:3].copy()
    a = float((t - t0) / (t1 - t0))
    p0 = track_txy[idx - 1, 1:3]
    p1 = track_txy[idx, 1:3]
    return (1.0 - a) * p0 + a * p1


def _sample_obs(track_txy: np.ndarray, t_now: float, obs_len: int, obs_dt: float) -> Optional[np.ndarray]:
    target_t = [t_now - (obs_len - 1 - i) * obs_dt for i in range(obs_len)]
    out = []
    for t in target_t:
        p = _interp_xy(track_txy, t)
        if p is None:
            return None
        out.append(p)
    return np.asarray(out, dtype=np.float64)


def _sample_gt(track_txy: np.ndarray, t_now: float, pred_steps_max: int, pred_dt: float) -> Optional[np.ndarray]:
    target_t = [t_now + (i + 1) * pred_dt for i in range(pred_steps_max)]
    out = []
    for t in target_t:
        p = _interp_xy(track_txy, t)
        if p is None:
            return None
        out.append(p)
    return np.asarray(out, dtype=np.float64)


def _xy_to_state6(xy: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros((xy.shape[0], 6), dtype=np.float32)
    out[:, 0:2] = xy.astype(np.float32)
    if xy.shape[0] < 2:
        return out
    vel = np.zeros_like(xy, dtype=np.float32)
    vel[1:] = (xy[1:] - xy[:-1]) / max(1e-6, dt)
    vel[0] = vel[1]
    acc = np.zeros_like(xy, dtype=np.float32)
    acc[1:] = (vel[1:] - vel[:-1]) / max(1e-6, dt)
    acc[0] = acc[1]
    out[:, 2:4] = vel
    out[:, 4:6] = acc
    return out


def _extend_cv(pred_xy: np.ndarray, target_steps: int) -> np.ndarray:
    if pred_xy.shape[0] >= target_steps:
        return pred_xy[:target_steps].copy()
    if pred_xy.shape[0] == 0:
        return np.zeros((target_steps, 2), dtype=np.float64)
    out = [pred_xy[i].copy() for i in range(pred_xy.shape[0])]
    while len(out) < target_steps:
        if len(out) >= 2:
            v = out[-1] - out[-2]
        else:
            v = np.array([0.0, 0.0], dtype=np.float64)
        out.append(out[-1] + v)
    return np.asarray(out, dtype=np.float64)


def _kalman_predict(obs_xy: np.ndarray, pred_steps: int, obs_dt: float, pred_dt: float) -> np.ndarray:
    sigma_acc = 0.06
    sigma_meas = 0.08
    sigma_p0 = 0.06
    sigma_v0 = 0.8
    mu = [float(obs_xy[0, 0]), float(obs_xy[0, 1]), 0.0, 0.0]
    sigma = [
        [sigma_p0 ** 2, 0.0, 0.0, 0.0],
        [0.0, sigma_p0 ** 2, 0.0, 0.0],
        [0.0, 0.0, sigma_v0 ** 2, 0.0],
        [0.0, 0.0, 0.0, sigma_v0 ** 2],
    ]
    for i in range(1, obs_xy.shape[0]):
        mu, sigma = predict_state_cov(mu, sigma, obs_dt, sigma_acc)
        mu, sigma = update_state_cov(mu, sigma, float(obs_xy[i, 0]), float(obs_xy[i, 1]), sigma_meas)
    out = []
    mu_h = list(mu)
    sigma_h = [r[:] for r in sigma]
    for _ in range(pred_steps):
        mu_h, sigma_h = predict_state_cov(mu_h, sigma_h, pred_dt, sigma_acc)
        out.append([float(mu_h[0]), float(mu_h[1])])
    return np.asarray(out, dtype=np.float64)


def _ade_fde(pred_xy: np.ndarray, gt_xy: np.ndarray, horizon: int) -> Tuple[float, float]:
    p = pred_xy[:horizon]
    g = gt_xy[:horizon]
    d = np.linalg.norm(p - g, axis=1)
    return float(np.mean(d)), float(d[-1])


def _permutation_paired(diff: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(diff)))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.shape[0]))
    vals = np.abs(np.mean(signs * diff[None, :], axis=1))
    return float((1 + np.sum(vals >= obs)) / (n_perm + 1))


def _summary(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _build_cases(
    dataset: Dict[str, Any],
    obs_len: int,
    obs_dt: float,
    pred_steps_max: int,
    pred_dt: float,
    neighbor_radius: float,
    max_neighbors: int,
    stride: int,
    max_cases: int,
) -> List[Case]:
    frames = dataset["frames"]
    track_map: Dict[int, List[Tuple[float, float, float]]] = {}
    for fr in frames:
        t = float(fr["t"])
        for p in fr["people"]:
            pid = int(p["id"])
            track_map.setdefault(pid, []).append((t, float(p["x"]), float(p["y"])))
    track_np = {
        pid: np.asarray(sorted(samples, key=lambda it: it[0]), dtype=np.float64)
        for pid, samples in track_map.items()
        if len(samples) >= 2
    }
    out: List[Case] = []
    for idx in range(0, len(frames), max(1, stride)):
        fr = frames[idx]
        t0 = float(fr["t"])
        present = fr["people"]
        by_id = {int(p["id"]): p for p in present}
        ids = list(by_id.keys())
        for pid in ids:
            tr = track_np.get(pid)
            if tr is None:
                continue
            obs = _sample_obs(tr, t0, obs_len, obs_dt)
            gt = _sample_gt(tr, t0, pred_steps_max, pred_dt)
            if obs is None or gt is None:
                continue
            px = float(by_id[pid]["x"])
            py = float(by_id[pid]["y"])
            neigh_list: List[np.ndarray] = []
            for oid in ids:
                if oid == pid:
                    continue
                ox = float(by_id[oid]["x"])
                oy = float(by_id[oid]["y"])
                if math.hypot(ox - px, oy - py) > neighbor_radius:
                    continue
                otr = track_np.get(oid)
                if otr is None:
                    continue
                oobs = _sample_obs(otr, t0, obs_len, obs_dt)
                if oobs is None:
                    continue
                neigh_list.append(oobs)
            if len(neigh_list) > max_neighbors:
                neigh_list = neigh_list[:max_neighbors]
            out.append(Case(t=t0, person_id=pid, obs_xy=obs, neigh_xy=neigh_list, gt_xy=gt))
            if max_cases > 0 and len(out) >= max_cases:
                return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline benchmark for people trajectory predictors.")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset from record_people_dataset.")
    parser.add_argument("--output_dir", default="benchmark_people_predictors", help="Output folder.")
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--obs_dt", type=float, default=0.4)
    parser.add_argument("--pred_dt", type=float, default=0.1)
    parser.add_argument("--pred_steps", default="12,26", help="Comma-separated horizons in steps.")
    parser.add_argument("--neighbor_radius", type=float, default=2.0)
    parser.add_argument("--max_neighbors", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--social_gru_weights", default="", help="Path to SocialGRU checkpoint (.pt).")
    parser.add_argument("--social_gru_device", default="")
    parser.add_argument("--social_vae_repo_path", default="")
    parser.add_argument("--social_vae_ckpt_path", default="")
    parser.add_argument("--social_vae_config_path", default="")
    parser.add_argument("--social_vae_samples", type=int, default=20)
    parser.add_argument("--social_vae_device", default="")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for model inference.")
    parser.add_argument("--progress_step_pct", type=int, default=10, help="Progress print period in percent.")
    parser.add_argument("--n_permutations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text())
    horizons = [int(x.strip()) for x in args.pred_steps.split(",") if x.strip()]
    horizons = sorted(set([h for h in horizons if h > 0]))
    if not horizons:
        raise ValueError("pred_steps must contain at least one positive integer")
    max_h = max(horizons)

    cases = _build_cases(
        dataset=data,
        obs_len=args.obs_len,
        obs_dt=args.obs_dt,
        pred_steps_max=max_h,
        pred_dt=args.pred_dt,
        neighbor_radius=args.neighbor_radius,
        max_neighbors=args.max_neighbors,
        stride=args.stride,
        max_cases=args.max_cases,
    )
    if not cases:
        raise RuntimeError("No valid cases built from dataset. Increase recording length or relax settings.")

    use_gru = bool(args.social_gru_weights)
    use_vae = bool(args.social_vae_repo_path and args.social_vae_ckpt_path)

    gru_model = None
    gru_device = args.social_gru_device.strip()
    if use_gru:
        gru_model, _ = load_checkpoint(args.social_gru_weights, device=gru_device or None)
        if not gru_device:
            try:
                gru_device = str(next(gru_model.parameters()).device)
            except Exception:
                gru_device = "cpu"

    vae_model = None
    vae_device = args.social_vae_device.strip()
    if use_vae:
        vae_model, _ = load_external_social_vae(
            repo_path=args.social_vae_repo_path,
            ckpt_path=args.social_vae_ckpt_path,
            device=vae_device or None,
            config_path=args.social_vae_config_path or None,
            ob_horizon=args.obs_len,
            pred_horizon=12,
            ob_radius=args.neighbor_radius,
            hidden_dim=256,
        )
        if not vae_device:
            try:
                vae_device = str(next(vae_model.parameters()).device)
            except Exception:
                vae_device = "cpu"

    metric: Dict[int, Dict[str, Dict[str, List[float]]]] = {
        h: {
            "kalman": {"ade": [], "fde": []},
            "social_gru": {"ade": [], "fde": []},
            "social_vae": {"ade": [], "fde": []},
        }
        for h in horizons
    }

    total_cases = len(cases)
    batch_size = max(1, int(args.batch_size))
    progress_step_pct = min(100, max(1, int(args.progress_step_pct)))
    progress_next = progress_step_pct
    processed = 0

    for start in range(0, total_cases, batch_size):
        batch = cases[start : start + batch_size]
        batch_n = len(batch)

        kalman_preds: List[np.ndarray] = [
            _kalman_predict(case.obs_xy, max_h, args.obs_dt, args.pred_dt) for case in batch
        ]

        gru_preds: List[Optional[np.ndarray]] = [None] * batch_n
        if use_gru and gru_model is not None:
            tracks_obs_xy = [case.obs_xy.astype(np.float32) for case in batch]
            tracks_neigh_xy = [[n.astype(np.float32) for n in case.neigh_xy] for case in batch]
            tracks_velocity_xy = []
            for case in batch:
                vel = case.obs_xy[-1] - case.obs_xy[-2] if case.obs_xy.shape[0] >= 2 else np.array([0.0, 0.0])
                tracks_velocity_xy.append((float(vel[0]), float(vel[1])))
            batch_pred = predict_trajectories_world(
                gru_model,
                tracks_obs_xy=tracks_obs_xy,
                tracks_neigh_xy=tracks_neigh_xy,
                device=gru_device or "cpu",
                tracks_velocity_xy=tracks_velocity_xy,
                flip_forward_axis=False,
            )
            for i, p in enumerate(batch_pred):
                gru_preds[i] = _extend_cv(np.asarray(p, dtype=np.float64), max_h)

        vae_preds: List[Optional[np.ndarray]] = [None] * batch_n
        if use_vae and vae_model is not None:
            x = np.zeros((args.obs_len, batch_n, 6), dtype=np.float32)
            neigh = np.full(
                (args.obs_len, batch_n, max(1, args.max_neighbors), 6),
                1e9,
                dtype=np.float32,
            )
            for i, case in enumerate(batch):
                x[:, i, :] = _xy_to_state6(case.obs_xy.astype(np.float64), args.obs_dt)
                for j, nxy in enumerate(case.neigh_xy[: max(1, args.max_neighbors)]):
                    neigh[:, i, j, :] = _xy_to_state6(nxy.astype(np.float64), args.obs_dt)
            samples = predict_social_vae_samples(
                model=vae_model,
                x=x,
                neighbor=neigh,
                device=vae_device or "cpu",
                n_predictions=max(1, int(args.social_vae_samples)),
                expected_horizon=12,
            )
            mu_batch = np.mean(samples, axis=0)  # [T, N, 2]
            for i in range(batch_n):
                vae_preds[i] = _extend_cv(np.asarray(mu_batch[:, i, :], dtype=np.float64), max_h)

        for i, case in enumerate(batch):
            pred_kalman = kalman_preds[i]
            pred_gru = gru_preds[i]
            pred_vae = vae_preds[i]
            for h in horizons:
                ade, fde = _ade_fde(pred_kalman, case.gt_xy, h)
                metric[h]["kalman"]["ade"].append(ade)
                metric[h]["kalman"]["fde"].append(fde)
                if pred_gru is not None:
                    ade, fde = _ade_fde(pred_gru, case.gt_xy, h)
                    metric[h]["social_gru"]["ade"].append(ade)
                    metric[h]["social_gru"]["fde"].append(fde)
                if pred_vae is not None:
                    ade, fde = _ade_fde(pred_vae, case.gt_xy, h)
                    metric[h]["social_vae"]["ade"].append(ade)
                    metric[h]["social_vae"]["fde"].append(fde)

        processed += batch_n
        pct = int((processed * 100) / total_cases)
        if pct >= progress_next:
            print(f"[progress] {pct}% ({processed}/{total_cases})")
            while progress_next <= pct:
                progress_next += progress_step_pct

    summary: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "n_cases": len(cases),
        "settings": {
            "obs_len": args.obs_len,
            "obs_dt": args.obs_dt,
            "pred_dt": args.pred_dt,
            "horizons": horizons,
        },
        "metrics": {},
        "paired_permutation_p": {},
    }

    for h in horizons:
        key = f"h{h}"
        summary["metrics"][key] = {}
        for model_name in ("kalman", "social_gru", "social_vae"):
            if len(metric[h][model_name]["ade"]) == 0:
                continue
            summary["metrics"][key][model_name] = {
                "ade": _summary(metric[h][model_name]["ade"]),
                "fde": _summary(metric[h][model_name]["fde"]),
            }

        pairs = [("kalman", "social_gru"), ("kalman", "social_vae"), ("social_gru", "social_vae")]
        pvals = {}
        for a, b in pairs:
            if len(metric[h][a]["ade"]) != len(metric[h][b]["ade"]) or len(metric[h][a]["ade"]) == 0:
                continue
            da = np.asarray(metric[h][a]["ade"], dtype=np.float64)
            db = np.asarray(metric[h][b]["ade"], dtype=np.float64)
            fa = np.asarray(metric[h][a]["fde"], dtype=np.float64)
            fb = np.asarray(metric[h][b]["fde"], dtype=np.float64)
            p_ade = _permutation_paired(da - db, args.n_permutations, args.seed + h)
            p_fde = _permutation_paired(fa - fb, args.n_permutations, args.seed + h + 1000)
            pvals[f"{a}_vs_{b}"] = {"ade_p": p_ade, "fde_p": p_fde}
        summary["paired_permutation_p"][key] = pvals

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = []
    lines.append(f"Cases: {len(cases)}")
    for h in horizons:
        hk = f"h{h}"
        lines.append(f"\nHorizon {h} steps:")
        for model_name, block in summary["metrics"].get(hk, {}).items():
            ade = block["ade"]
            fde = block["fde"]
            lines.append(
                f"  {model_name:10s} ADE={ade['mean']:.3f}±{ade['std']:.3f}  "
                f"FDE={fde['mean']:.3f}±{fde['std']:.3f}  n={ade['n']}"
            )
        for pair, pv in summary["paired_permutation_p"].get(hk, {}).items():
            lines.append(f"  p({pair}) ADE={pv['ade_p']:.4f} FDE={pv['fde_p']:.4f}")
    report = "\n".join(lines) + "\n"
    print(report)
    (out_dir / "report.txt").write_text(report)


if __name__ == "__main__":
    main()
