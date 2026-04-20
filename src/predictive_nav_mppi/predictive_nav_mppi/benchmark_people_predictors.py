#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predictive_nav_mppi.kf_cv import predict_state_cov, update_state_cov
from predictive_nav_mppi.models.kalman_residual_net import compute_turn_gate
from predictive_nav_mppi.models.kalman_residual_net import load_checkpoint as load_residual_checkpoint
from predictive_nav_mppi.models.kalman_residual_net import predict_residual_world
from predictive_nav_mppi.models.social_gru import load_checkpoint, predict_trajectories_world
from predictive_nav_mppi.models.social_vae import load_external_social_vae, predict_social_vae_samples
from predictive_nav_mppi.scene_context import ScenePatchConfig


@dataclass
class Case:
    t: float
    person_id: int
    obs_xy: np.ndarray
    neigh_xy: List[np.ndarray]
    gt_xy: np.ndarray
    source_name: str = ""


def _speed_magnitudes(obs_xy: np.ndarray, dt: float) -> np.ndarray:
    if obs_xy.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    v = (obs_xy[1:] - obs_xy[:-1]) / max(1e-6, dt)
    return np.linalg.norm(v, axis=1)


def _heading_change(obs_xy: np.ndarray) -> float:
    if obs_xy.shape[0] < 3:
        return 0.0
    step = obs_xy[1:] - obs_xy[:-1]
    ang = np.arctan2(step[:, 1], step[:, 0])
    if ang.shape[0] < 2:
        return 0.0
    d = np.diff(ang)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


def _min_neighbor_distance(case: Case) -> float:
    if not case.neigh_xy:
        return float("inf")
    p = case.obs_xy[-1]
    dmin = float("inf")
    for nxy in case.neigh_xy:
        if nxy.shape[0] == 0:
            continue
        q = nxy[-1]
        d = float(np.linalg.norm(p - q))
        if d < dmin:
            dmin = d
    return dmin


def _classify_case_tags(
    case: Case,
    obs_dt: float,
    interaction_dist: float,
    dense_neighbors_min: int,
    turn_threshold_deg: float,
    stop_speed_thresh: float,
    stop_go_delta: float,
    moving_speed_min: float,
) -> Set[str]:
    tags: Set[str] = {"all"}

    n_neigh = len(case.neigh_xy)
    min_neigh_dist = _min_neighbor_distance(case)
    if n_neigh > 0 and min_neigh_dist <= interaction_dist:
        tags.add("interaction")
    if n_neigh >= dense_neighbors_min:
        tags.add("dense_interaction")

    turn_rad = math.radians(max(0.0, turn_threshold_deg))
    if _heading_change(case.obs_xy) >= turn_rad:
        tags.add("turning")

    sp = _speed_magnitudes(case.obs_xy, obs_dt)
    if sp.size > 0:
        s_min = float(np.min(sp))
        s_max = float(np.max(sp))
        if s_max >= moving_speed_min and s_min <= stop_speed_thresh and (s_max - s_min) >= stop_go_delta:
            tags.add("stop_go")

    complexity_axes = (
        int("interaction" in tags)
        + int("dense_interaction" in tags)
        + int("turning" in tags)
        + int("stop_go" in tags)
    )
    if complexity_axes >= 2:
        tags.add("complex")
    if complexity_axes >= 3:
        tags.add("very_complex")

    return tags


def _case_in_split(tags: Set[str], split_name: str) -> bool:
    name = split_name.strip()
    if not name:
        return False
    if name == "all":
        return True
    if "+" in name:
        return all(tok in tags for tok in name.split("+") if tok)
    if "_" in name and name not in tags:
        return all(tok in tags for tok in name.split("_") if tok)
    return name in tags


def _stage_segments(horizon: int) -> Dict[str, Tuple[int, int]]:
    if horizon <= 0:
        return {}
    if horizon < 3:
        return {"full": (0, horizon)}
    s1 = max(1, horizon // 3)
    s2 = max(s1 + 1, (2 * horizon) // 3)
    s2 = min(s2, horizon - 1)
    return {
        "early": (0, s1),
        "mid": (s1, s2),
        "late": (s2, horizon),
    }


def _ade_fde_segment(pred_xy: np.ndarray, gt_xy: np.ndarray, start: int, end: int) -> Tuple[float, float]:
    p = pred_xy[start:end]
    g = gt_xy[start:end]
    d = np.linalg.norm(p - g, axis=1)
    return float(np.mean(d)), float(d[-1])


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


def _apply_residual_runtime_postprocess(
    kalman_pred_xy: np.ndarray,
    pred_world: np.ndarray,
    smoothing_state: Dict[Tuple[str, int], np.ndarray],
    case_key: Tuple[str, int],
    residual_clip_norm: float,
    residual_smoothing_beta: float,
) -> np.ndarray:
    correction = np.asarray(pred_world, dtype=np.float64) - np.asarray(kalman_pred_xy, dtype=np.float64)

    clip_norm = float(max(0.0, residual_clip_norm))
    if clip_norm > 0.0 and correction.size:
        norms = np.linalg.norm(correction, axis=1, keepdims=True)
        scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-6))
        correction = correction * scale

    beta = float(min(0.999, max(0.0, residual_smoothing_beta)))
    if beta > 0.0:
        prev = smoothing_state.get(case_key)
        if prev is not None and prev.shape == correction.shape:
            correction = beta * prev + (1.0 - beta) * correction
        smoothing_state[case_key] = correction.copy()
    else:
        smoothing_state.pop(case_key, None)

    return np.asarray(kalman_pred_xy, dtype=np.float64) + correction


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
            out.append(Case(t=t0, person_id=pid, obs_xy=obs, neigh_xy=neigh_list, gt_xy=gt, source_name=""))
            if max_cases > 0 and len(out) >= max_cases:
                return out
    return out


def _load_curated_cases(dataset: Dict[str, Any]) -> List[Case]:
    raw_cases = dataset.get("cases", [])
    out: List[Case] = []
    for item in raw_cases:
        obs_xy = np.asarray(item["obs_xy"], dtype=np.float64)
        gt_xy = np.asarray(item["gt_xy"], dtype=np.float64)
        neigh_xy = [np.asarray(arr, dtype=np.float64) for arr in item.get("neigh_xy", [])]
        out.append(
            Case(
                t=float(item.get("t", 0.0)),
                person_id=int(item.get("person_id", 0)),
                obs_xy=obs_xy,
                neigh_xy=neigh_xy,
                gt_xy=gt_xy,
                source_name=str(item.get("source_name", "")),
            )
        )
    return out


def _filter_cases_by_horizon(cases: Sequence[Case], max_horizon: int) -> List[Case]:
    return [case for case in cases if int(case.gt_xy.shape[0]) >= int(max_horizon)]


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
    parser.add_argument("--residual_model_weights", default="", help="Backward-compatible alias for --residual_scene_model_weights.")
    parser.add_argument("--residual_old_model_weights", default="", help="Path to old residual predictor checkpoint without scene branch (.pt).")
    parser.add_argument("--residual_scene_model_weights", default="", help="Path to scene-aware residual predictor checkpoint (.pt).")
    parser.add_argument("--residual_model_device", default="")
    parser.add_argument("--residual_alpha", type=float, default=0.3)
    parser.add_argument("--residual_smoothing_beta", type=float, default=0.8)
    parser.add_argument("--residual_clip_norm", type=float, default=0.35)
    parser.add_argument("--residual_turn_gate_enable", action="store_true", default=True)
    parser.add_argument("--disable_residual_turn_gate", action="store_true")
    parser.add_argument("--residual_turn_gate_tau", type=float, default=0.1)
    parser.add_argument("--residual_turn_gate_alpha", type=float, default=30.0)
    parser.add_argument("--scene_patch_size_m", type=float, default=6.0)
    parser.add_argument("--scene_patch_pixels", type=int, default=32)
    parser.add_argument("--scene_patch_align_to_heading", action="store_true", default=True)
    parser.add_argument("--disable_scene_patch_align_to_heading", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for model inference.")
    parser.add_argument("--progress_step_pct", type=int, default=10, help="Progress print period in percent.")
    parser.add_argument("--n_permutations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--report_splits",
        default="all,interaction,turning,stop_go,dense_interaction,complex,interaction_turning",
        help="Comma-separated split names; combine tags with '_' or '+', e.g. interaction_turning.",
    )
    parser.add_argument(
        "--only_split",
        default="",
        help="If set, evaluate only this split (e.g. complex or interaction_turning).",
    )
    parser.add_argument("--min_split_cases", type=int, default=100)
    parser.add_argument("--interaction_dist", type=float, default=1.5)
    parser.add_argument("--dense_neighbors_min", type=int, default=3)
    parser.add_argument("--turn_threshold_deg", type=float, default=45.0)
    parser.add_argument("--stop_speed_thresh", type=float, default=0.10)
    parser.add_argument("--moving_speed_min", type=float, default=0.25)
    parser.add_argument("--stop_go_delta", type=float, default=0.25)
    args = parser.parse_args()
    if args.disable_residual_turn_gate:
        args.residual_turn_gate_enable = False
    if args.disable_scene_patch_align_to_heading:
        args.scene_patch_align_to_heading = False

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text())
    horizons = [int(x.strip()) for x in args.pred_steps.split(",") if x.strip()]
    horizons = sorted(set([h for h in horizons if h > 0]))
    if not horizons:
        raise ValueError("pred_steps must contain at least one positive integer")
    max_h = max(horizons)

    if "cases" in data:
        cases = _load_curated_cases(data)
    else:
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
    cases = _filter_cases_by_horizon(cases, max_h)
    if not cases:
        raise RuntimeError(
            f"No valid cases built from dataset for max horizon={max_h}. "
            "For curated files, regenerate with pred_len >= requested horizon."
        )

    split_names = [s.strip() for s in str(args.report_splits).split(",") if s.strip()]
    if "all" not in split_names:
        split_names = ["all"] + split_names

    case_tags: List[Set[str]] = [
        _classify_case_tags(
            case=case,
            obs_dt=args.obs_dt,
            interaction_dist=args.interaction_dist,
            dense_neighbors_min=args.dense_neighbors_min,
            turn_threshold_deg=args.turn_threshold_deg,
            stop_speed_thresh=args.stop_speed_thresh,
            stop_go_delta=args.stop_go_delta,
            moving_speed_min=args.moving_speed_min,
        )
        for case in cases
    ]

    only_split = str(args.only_split).strip()
    if only_split:
        filtered = [(c, t) for c, t in zip(cases, case_tags) if _case_in_split(t, only_split)]
        if not filtered:
            raise RuntimeError(f"No cases left after --only_split={only_split}")
        cases = [ct[0] for ct in filtered]
        case_tags = [ct[1] for ct in filtered]
        print(f"[split] only_split={only_split}: using {len(cases)} cases")

    use_gru = bool(args.social_gru_weights)
    use_vae = bool(args.social_vae_repo_path and args.social_vae_ckpt_path)
    residual_scene_path = str(args.residual_scene_model_weights or args.residual_model_weights).strip()
    residual_old_path = str(args.residual_old_model_weights).strip()
    use_residual_old = bool(residual_old_path)
    use_residual_scene = bool(residual_scene_path)

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

    residual_old_model = None
    residual_old_device = args.residual_model_device.strip()
    residual_old_k_neighbors = 3
    residual_old_pred_len = max_h
    residual_old_smoothing_state: Dict[Tuple[str, int], np.ndarray] = {}
    residual_scene_model = None
    residual_scene_device = args.residual_model_device.strip()
    residual_scene_k_neighbors = 3
    residual_scene_pred_len = max_h
    residual_scene_smoothing_state: Dict[Tuple[str, int], np.ndarray] = {}
    residual_scene_cfg = ScenePatchConfig(
        size_m=float(args.scene_patch_size_m),
        pixels=int(args.scene_patch_pixels),
        align_to_heading=bool(args.scene_patch_align_to_heading),
    )
    if use_residual_old:
        residual_old_model, residual_old_cfg = load_residual_checkpoint(residual_old_path, device=residual_old_device or None)
        residual_old_k_neighbors = int(residual_old_cfg.get("k_neighbors", 3))
        residual_old_pred_len = int(residual_old_cfg.get("pred_len", max_h))
        if not residual_old_device:
            try:
                residual_old_device = str(next(residual_old_model.parameters()).device)
            except Exception:
                residual_old_device = "cpu"
    if use_residual_scene:
        residual_scene_model, residual_scene_cfg_ckpt = load_residual_checkpoint(residual_scene_path, device=residual_scene_device or None)
        residual_scene_k_neighbors = int(residual_scene_cfg_ckpt.get("k_neighbors", 3))
        residual_scene_pred_len = int(residual_scene_cfg_ckpt.get("pred_len", max_h))
        if not residual_scene_device:
            try:
                residual_scene_device = str(next(residual_scene_model.parameters()).device)
            except Exception:
                residual_scene_device = "cpu"

    kalman_rollout_steps = max(max_h, residual_old_pred_len, residual_scene_pred_len)

    model_names = ["kalman", "social_gru", "social_vae", "residual_old", "residual_scene"]

    metric: Dict[int, Dict[str, Dict[str, List[float]]]] = {
        h: {name: {"ade": [], "fde": []} for name in model_names}
        for h in horizons
    }
    split_metric: Dict[int, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        h: {
            split: {name: {"ade": [], "fde": []} for name in model_names}
            for split in split_names
        }
        for h in horizons
    }
    stage_metric: Dict[int, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        h: {name: {stage: {"ade": [], "fde": []} for stage in _stage_segments(h).keys()} for name in model_names}
        for h in horizons
    }
    split_stage_metric: Dict[int, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = {
        h: {
            split: {name: {stage: {"ade": [], "fde": []} for stage in _stage_segments(h).keys()} for name in model_names}
            for split in split_names
        }
        for h in horizons
    }
    source_metric: Dict[int, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        h: {}
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
            _kalman_predict(case.obs_xy, kalman_rollout_steps, args.obs_dt, args.pred_dt) for case in batch
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

        residual_old_preds: List[Optional[np.ndarray]] = [None] * batch_n
        if use_residual_old and residual_old_model is not None:
            for i, case in enumerate(batch):
                residual_old_preds[i] = predict_residual_world(
                    model=residual_old_model,
                    obs_xy=case.obs_xy.astype(np.float32),
                    neigh_xy=[n.astype(np.float32) for n in case.neigh_xy],
                    kalman_pred_xy=kalman_preds[i].astype(np.float32),
                    obs_dt=args.obs_dt,
                    device=residual_old_device or "cpu",
                    k_neighbors=residual_old_k_neighbors,
                    residual_alpha=float(args.residual_alpha)
                    * (
                        compute_turn_gate(
                            obs_xy=case.obs_xy,
                            tau=float(args.residual_turn_gate_tau),
                            alpha=float(args.residual_turn_gate_alpha),
                        )
                        if args.residual_turn_gate_enable
                        else 1.0
                    ),
                )
                residual_old_preds[i] = _apply_residual_runtime_postprocess(
                    kalman_pred_xy=kalman_preds[i],
                    pred_world=np.asarray(residual_old_preds[i], dtype=np.float64),
                    smoothing_state=residual_old_smoothing_state,
                    case_key=(str(case.source_name or ""), int(case.person_id)),
                    residual_clip_norm=float(args.residual_clip_norm),
                    residual_smoothing_beta=float(args.residual_smoothing_beta),
                )
        residual_scene_preds: List[Optional[np.ndarray]] = [None] * batch_n
        if use_residual_scene and residual_scene_model is not None:
            for i, case in enumerate(batch):
                residual_scene_preds[i] = predict_residual_world(
                    model=residual_scene_model,
                    obs_xy=case.obs_xy.astype(np.float32),
                    neigh_xy=[n.astype(np.float32) for n in case.neigh_xy],
                    kalman_pred_xy=kalman_preds[i].astype(np.float32),
                    obs_dt=args.obs_dt,
                    device=residual_scene_device or "cpu",
                    k_neighbors=residual_scene_k_neighbors,
                    scene_source_name=str(case.source_name or ""),
                    scene_patch_cfg=residual_scene_cfg,
                    residual_alpha=float(args.residual_alpha)
                    * (
                        compute_turn_gate(
                            obs_xy=case.obs_xy,
                            tau=float(args.residual_turn_gate_tau),
                            alpha=float(args.residual_turn_gate_alpha),
                        )
                        if args.residual_turn_gate_enable
                        else 1.0
                    ),
                )
                residual_scene_preds[i] = _apply_residual_runtime_postprocess(
                    kalman_pred_xy=kalman_preds[i],
                    pred_world=np.asarray(residual_scene_preds[i], dtype=np.float64),
                    smoothing_state=residual_scene_smoothing_state,
                    case_key=(str(case.source_name or ""), int(case.person_id)),
                    residual_clip_norm=float(args.residual_clip_norm),
                    residual_smoothing_beta=float(args.residual_smoothing_beta),
                )

        for i, case in enumerate(batch):
            pred_kalman = kalman_preds[i]
            pred_gru = gru_preds[i]
            pred_vae = vae_preds[i]
            pred_residual_old = residual_old_preds[i]
            pred_residual_scene = residual_scene_preds[i]
            tags = case_tags[start + i]
            active_splits = [sp for sp in split_names if _case_in_split(tags, sp)]
            source_name = case.source_name.strip() or "dataset"
            for h in horizons:
                stage_seg = _stage_segments(h)
                source_metric[h].setdefault(
                source_name,
                {name: {"ade": [], "fde": []} for name in model_names},
            )

                ade, fde = _ade_fde(pred_kalman, case.gt_xy, h)
                metric[h]["kalman"]["ade"].append(ade)
                metric[h]["kalman"]["fde"].append(fde)
                source_metric[h][source_name]["kalman"]["ade"].append(ade)
                source_metric[h][source_name]["kalman"]["fde"].append(fde)
                for sp in active_splits:
                    split_metric[h][sp]["kalman"]["ade"].append(ade)
                    split_metric[h][sp]["kalman"]["fde"].append(fde)
                for st, (s0, s1) in stage_seg.items():
                    sade, sfde = _ade_fde_segment(pred_kalman, case.gt_xy, s0, s1)
                    stage_metric[h]["kalman"][st]["ade"].append(sade)
                    stage_metric[h]["kalman"][st]["fde"].append(sfde)
                    for sp in active_splits:
                        split_stage_metric[h][sp]["kalman"][st]["ade"].append(sade)
                        split_stage_metric[h][sp]["kalman"][st]["fde"].append(sfde)

                if pred_gru is not None:
                    ade, fde = _ade_fde(pred_gru, case.gt_xy, h)
                    metric[h]["social_gru"]["ade"].append(ade)
                    metric[h]["social_gru"]["fde"].append(fde)
                    source_metric[h][source_name]["social_gru"]["ade"].append(ade)
                    source_metric[h][source_name]["social_gru"]["fde"].append(fde)
                    for sp in active_splits:
                        split_metric[h][sp]["social_gru"]["ade"].append(ade)
                        split_metric[h][sp]["social_gru"]["fde"].append(fde)
                    for st, (s0, s1) in stage_seg.items():
                        sade, sfde = _ade_fde_segment(pred_gru, case.gt_xy, s0, s1)
                        stage_metric[h]["social_gru"][st]["ade"].append(sade)
                        stage_metric[h]["social_gru"][st]["fde"].append(sfde)
                        for sp in active_splits:
                            split_stage_metric[h][sp]["social_gru"][st]["ade"].append(sade)
                            split_stage_metric[h][sp]["social_gru"][st]["fde"].append(sfde)
                if pred_vae is not None:
                    ade, fde = _ade_fde(pred_vae, case.gt_xy, h)
                    metric[h]["social_vae"]["ade"].append(ade)
                    metric[h]["social_vae"]["fde"].append(fde)
                    source_metric[h][source_name]["social_vae"]["ade"].append(ade)
                    source_metric[h][source_name]["social_vae"]["fde"].append(fde)
                    for sp in active_splits:
                        split_metric[h][sp]["social_vae"]["ade"].append(ade)
                        split_metric[h][sp]["social_vae"]["fde"].append(fde)
                    for st, (s0, s1) in stage_seg.items():
                        sade, sfde = _ade_fde_segment(pred_vae, case.gt_xy, s0, s1)
                        stage_metric[h]["social_vae"][st]["ade"].append(sade)
                        stage_metric[h]["social_vae"][st]["fde"].append(sfde)
                        for sp in active_splits:
                            split_stage_metric[h][sp]["social_vae"][st]["ade"].append(sade)
                            split_stage_metric[h][sp]["social_vae"][st]["fde"].append(sfde)
                if pred_residual_old is not None:
                    ade, fde = _ade_fde(pred_residual_old, case.gt_xy, h)
                    metric[h]["residual_old"]["ade"].append(ade)
                    metric[h]["residual_old"]["fde"].append(fde)
                    source_metric[h][source_name]["residual_old"]["ade"].append(ade)
                    source_metric[h][source_name]["residual_old"]["fde"].append(fde)
                    for sp in active_splits:
                        split_metric[h][sp]["residual_old"]["ade"].append(ade)
                        split_metric[h][sp]["residual_old"]["fde"].append(fde)
                    for st, (s0, s1) in stage_seg.items():
                        sade, sfde = _ade_fde_segment(pred_residual_old, case.gt_xy, s0, s1)
                        stage_metric[h]["residual_old"][st]["ade"].append(sade)
                        stage_metric[h]["residual_old"][st]["fde"].append(sfde)
                        for sp in active_splits:
                            split_stage_metric[h][sp]["residual_old"][st]["ade"].append(sade)
                            split_stage_metric[h][sp]["residual_old"][st]["fde"].append(sfde)
                if pred_residual_scene is not None:
                    ade, fde = _ade_fde(pred_residual_scene, case.gt_xy, h)
                    metric[h]["residual_scene"]["ade"].append(ade)
                    metric[h]["residual_scene"]["fde"].append(fde)
                    source_metric[h][source_name]["residual_scene"]["ade"].append(ade)
                    source_metric[h][source_name]["residual_scene"]["fde"].append(fde)
                    for sp in active_splits:
                        split_metric[h][sp]["residual_scene"]["ade"].append(ade)
                        split_metric[h][sp]["residual_scene"]["fde"].append(fde)
                    for st, (s0, s1) in stage_seg.items():
                        sade, sfde = _ade_fde_segment(pred_residual_scene, case.gt_xy, s0, s1)
                        stage_metric[h]["residual_scene"][st]["ade"].append(sade)
                        stage_metric[h]["residual_scene"][st]["fde"].append(sfde)
                        for sp in active_splits:
                            split_stage_metric[h][sp]["residual_scene"][st]["ade"].append(sade)
                            split_stage_metric[h][sp]["residual_scene"][st]["fde"].append(sfde)

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
        "split_metrics": {},
        "stage_metrics": {},
        "split_stage_metrics": {},
        "source_metrics": {},
        "case_split_counts": {},
        "case_tag_thresholds": {
            "interaction_dist": args.interaction_dist,
            "dense_neighbors_min": args.dense_neighbors_min,
            "turn_threshold_deg": args.turn_threshold_deg,
            "stop_speed_thresh": args.stop_speed_thresh,
            "moving_speed_min": args.moving_speed_min,
            "stop_go_delta": args.stop_go_delta,
        },
    }

    for h in horizons:
        key = f"h{h}"
        summary["metrics"][key] = {}
        summary["split_metrics"][key] = {}
        summary["stage_metrics"][key] = {}
        summary["split_stage_metrics"][key] = {}
        summary["source_metrics"][key] = {}
        for model_name in model_names:
            if len(metric[h][model_name]["ade"]) == 0:
                continue
            summary["metrics"][key][model_name] = {
                "ade": _summary(metric[h][model_name]["ade"]),
                "fde": _summary(metric[h][model_name]["fde"]),
            }
            summary["stage_metrics"][key][model_name] = {}
            for stage_name, block in stage_metric[h][model_name].items():
                if len(block["ade"]) == 0:
                    continue
                summary["stage_metrics"][key][model_name][stage_name] = {
                    "ade": _summary(block["ade"]),
                    "fde": _summary(block["fde"]),
                }

        for source_name, source_block in source_metric[h].items():
            summary["source_metrics"][key][source_name] = {}
            for model_name in model_names:
                if len(source_block[model_name]["ade"]) == 0:
                    continue
                summary["source_metrics"][key][source_name][model_name] = {
                    "ade": _summary(source_block[model_name]["ade"]),
                    "fde": _summary(source_block[model_name]["fde"]),
                }

        for split in split_names:
            summary["split_metrics"][key][split] = {}
            summary["split_stage_metrics"][key][split] = {}
            for model_name in model_names:
                if len(split_metric[h][split][model_name]["ade"]) == 0:
                    continue
                summary["split_metrics"][key][split][model_name] = {
                    "ade": _summary(split_metric[h][split][model_name]["ade"]),
                    "fde": _summary(split_metric[h][split][model_name]["fde"]),
                }
                summary["split_stage_metrics"][key][split][model_name] = {}
                for stage_name, block in split_stage_metric[h][split][model_name].items():
                    if len(block["ade"]) == 0:
                        continue
                    summary["split_stage_metrics"][key][split][model_name][stage_name] = {
                        "ade": _summary(block["ade"]),
                        "fde": _summary(block["fde"]),
                    }

        pairs = []
        for idx_a, a in enumerate(model_names):
            for b in model_names[idx_a + 1 :]:
                pairs.append((a, b))
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

    for split in split_names:
        summary["case_split_counts"][split] = int(sum(1 for t in case_tags if _case_in_split(t, split)))

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

        lines.append("  Stage ADE/FDE (all cases):")
        for model_name, stages in summary["stage_metrics"].get(hk, {}).items():
            st_parts = []
            for st_name in ("early", "mid", "late", "full"):
                if st_name not in stages:
                    continue
                ade_s = stages[st_name]["ade"]["mean"]
                fde_s = stages[st_name]["fde"]["mean"]
                st_parts.append(f"{st_name}:ADE={ade_s:.3f}/FDE={fde_s:.3f}")
            if st_parts:
                lines.append(f"    {model_name:10s} " + " | ".join(st_parts))

        lines.append("  Hard-case splits:")
        for split in split_names:
            split_n = summary["case_split_counts"].get(split, 0)
            if split == "all" or split_n < args.min_split_cases:
                continue
            lines.append(f"    [{split}] n={split_n}")
            for model_name, block in summary["split_metrics"].get(hk, {}).get(split, {}).items():
                ade = block["ade"]
                fde = block["fde"]
                lines.append(
                    f"      {model_name:10s} ADE={ade['mean']:.3f}±{ade['std']:.3f} "
                    f"FDE={fde['mean']:.3f}±{fde['std']:.3f} n={ade['n']}"
                )
        if summary["source_metrics"].get(hk):
            lines.append("  Per-scene metrics:")
            for source_name, models in sorted(summary["source_metrics"][hk].items()):
                lines.append(f"    [{source_name}]")
                for model_name, block in models.items():
                    ade = block["ade"]
                    fde = block["fde"]
                    lines.append(
                        f"      {model_name:10s} ADE={ade['mean']:.3f}±{ade['std']:.3f} "
                        f"FDE={fde['mean']:.3f}±{fde['std']:.3f} n={ade['n']}"
                    )
    report = "\n".join(lines) + "\n"
    print(report)
    (out_dir / "report.txt").write_text(report)


if __name__ == "__main__":
    main()
