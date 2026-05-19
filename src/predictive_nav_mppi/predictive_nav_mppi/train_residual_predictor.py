#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from predictive_nav_mppi.models.kalman_residual_net import KalmanResidualNet
from predictive_nav_mppi.scene_context import ScenePatchConfig, extract_patch_from_source

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _plot_training_summary(history: List[Dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss.png")
        return

    if not history:
        return

    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.reshape(-1)

    # Total loss
    ax = axes[0]
    ax.plot(epochs, [row["train"]["loss"] for row in history], label="train", color="#1f77b4")
    ax.plot(epochs, [row["val"]["loss"] for row in history], label="val", color="#d62728", linestyle="--")
    ax.set_title("Total Loss")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    # ADE
    ax = axes[1]
    ax.plot(epochs, [row["train"]["hybrid_ade"] for row in history], label="train hybrid", color="#1f77b4")
    ax.plot(epochs, [row["val"]["hybrid_ade"] for row in history], label="val hybrid", color="#d62728", linestyle="--")
    ax.plot(epochs, [row["val"]["kalman_ade"] for row in history], label="val kalman", color="grey", linestyle=":")
    ax.set_title("ADE")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # FDE
    ax = axes[2]
    ax.plot(epochs, [row["train"]["hybrid_fde"] for row in history], label="train hybrid", color="#1f77b4")
    ax.plot(epochs, [row["val"]["hybrid_fde"] for row in history], label="val hybrid", color="#d62728", linestyle="--")
    ax.plot(epochs, [row["val"]["kalman_fde"] for row in history], label="val kalman", color="grey", linestyle=":")
    ax.set_title("FDE")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Position loss
    ax = axes[3]
    ax.plot(epochs, [row["train"].get("loss_pos", 0.0) for row in history], label="train", color="#1f77b4")
    ax.plot(epochs, [row["val"].get("loss_pos", 0.0) for row in history], label="val", color="#d62728", linestyle="--")
    ax.set_title("Position SmoothL1")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    # Smoothness
    ax = axes[4]
    ax.plot(epochs, [row["train"].get("loss_jerk", 0.0) for row in history], label="jerk train", color="#1f77b4")
    ax.plot(epochs, [row["val"].get("loss_jerk", 0.0) for row in history], label="jerk val", color="#d62728", linestyle="--")
    ax.set_title("Smoothness (Jerk)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Velocity match
    ax = axes[5]
    ax.plot(epochs, [row["train"].get("loss_vel", 0.0) for row in history], label="train", color="#1f77b4")
    ax.plot(epochs, [row["val"].get("loss_vel", 0.0) for row in history], label="val", color="#d62728", linestyle="--")
    ax.set_title("Velocity Match")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved plot: {out_path}")


def _rotation(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _to_local(points_xy: np.ndarray, origin_xy: np.ndarray, rot_world_to_local: np.ndarray) -> np.ndarray:
    return ((rot_world_to_local @ (points_xy - origin_xy).T).T).astype(np.float32)


def _speed_and_acc(points_xy: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    vel = np.zeros_like(points_xy, dtype=np.float32)
    if points_xy.shape[0] >= 2:
        vel[1:] = (points_xy[1:] - points_xy[:-1]) / max(1e-6, dt)
        vel[0] = vel[1]
    acc = np.zeros_like(points_xy, dtype=np.float32)
    if points_xy.shape[0] >= 2:
        acc[1:] = (vel[1:] - vel[:-1]) / max(1e-6, dt)
        acc[0] = acc[1]
    return vel, acc


def _heading_delta(points_xy: np.ndarray) -> np.ndarray:
    out = np.zeros((points_xy.shape[0], 1), dtype=np.float32)
    if points_xy.shape[0] < 3:
        return out
    step = points_xy[1:] - points_xy[:-1]
    ang = np.arctan2(step[:, 1], step[:, 0])
    if ang.shape[0] < 2:
        return out
    d = np.diff(ang)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    out[2:, 0] = d.astype(np.float32)
    out[1, 0] = out[2, 0] if points_xy.shape[0] > 2 else 0.0
    return out


def _min_ttc(target_pos: np.ndarray, target_vel: np.ndarray, neigh_pos: np.ndarray, neigh_vel: np.ndarray) -> float:
    rel_p = neigh_pos - target_pos
    rel_v = neigh_vel - target_vel
    vv = float(np.dot(rel_v, rel_v))
    if vv < 1e-6:
        return 999.0
    t = -float(np.dot(rel_p, rel_v)) / vv
    if t <= 0.0:
        return 999.0
    return min(t, 999.0)


class ResidualTrajectoryDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        obs_len: int = 8,
        pred_len: int = 26,
        obs_dt: float = 0.4,
        k_neighbors: int = 3,
        scene_patch_size_m: float = 6.0,
        scene_patch_pixels: int = 32,
        scene_patch_align_to_heading: bool = True,
        include_robot: bool = False,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install torch.")
        payload = json.loads(Path(path).expanduser().read_text())
        self.cases = payload["cases"]
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.obs_dt = float(obs_dt)
        self.k_neighbors = int(k_neighbors)
        self.include_robot = bool(include_robot)
        self.neighbor_feat_dim = 6 if self.include_robot else 5
        self.total_slots = self.k_neighbors + (1 if self.include_robot else 0)
        self.scene_patch_cfg = ScenePatchConfig(
            size_m=float(scene_patch_size_m),
            pixels=int(scene_patch_pixels),
            align_to_heading=bool(scene_patch_align_to_heading),
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case = self.cases[index]
        obs_xy = np.asarray(case["obs_xy"], dtype=np.float32)[-self.obs_len :]
        gt_xy = np.asarray(case["gt_xy"], dtype=np.float32)[: self.pred_len]
        kalman_xy = np.asarray(case["kalman_pred_xy"], dtype=np.float32)[: self.pred_len]
        residual_xy = np.asarray(case["residual_xy"], dtype=np.float32)[: self.pred_len]
        neigh_list = [np.asarray(arr, dtype=np.float32)[-self.obs_len :] for arr in case.get("neigh_xy", [])]

        origin = obs_xy[-1].copy()
        heading = obs_xy[-1] - obs_xy[-2] if obs_xy.shape[0] >= 2 else np.zeros((2,), dtype=np.float32)
        theta = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1e-6 else 0.0
        rot = _rotation(-theta)
        scene_patch = extract_patch_from_source(
            source_name=str(case["source_name"]),
            center_xy=obs_xy[-1],
            heading_rad=theta,
            cfg=self.scene_patch_cfg,
        )

        obs_local = _to_local(obs_xy, origin, rot)
        gt_local = _to_local(gt_xy, origin, rot)
        kalman_local = _to_local(kalman_xy, origin, rot)
        residual_local = (gt_local - kalman_local).astype(np.float32)

        vel_local, acc_local = _speed_and_acc(obs_local, self.obs_dt)
        speed = np.linalg.norm(vel_local, axis=1, keepdims=True).astype(np.float32)
        heading_delta = _heading_delta(obs_local)
        target_seq = np.concatenate([obs_local, vel_local, speed, acc_local, heading_delta], axis=1).astype(np.float32)

        neigh_seq = np.zeros((self.total_slots, self.obs_len, self.neighbor_feat_dim), dtype=np.float32)
        neigh_mask = np.zeros((self.total_slots,), dtype=np.float32)

        target_vel_now = vel_local[-1]
        num_r1 = 0.0
        num_r2 = 0.0
        min_dist = 999.0
        min_ttc = 999.0
        closest_rel = np.zeros((4,), dtype=np.float32)
        front_density = 0.0
        left_density = 0.0
        right_density = 0.0

        def _make_feat(local_xy: np.ndarray, is_robot: float) -> np.ndarray:
            vel, _ = _speed_and_acc(local_xy, self.obs_dt)
            base = np.concatenate(
                [local_xy, vel, np.linalg.norm(vel, axis=1, keepdims=True).astype(np.float32)],
                axis=1,
            ).astype(np.float32)
            if self.include_robot:
                flag = np.full((base.shape[0], 1), is_robot, dtype=np.float32)
                base = np.concatenate([base, flag], axis=1)
            return base

        if self.include_robot:
            robot_obs_xy = case.get("robot_obs_xy")
            if robot_obs_xy is not None:
                robot_arr = np.asarray(robot_obs_xy, dtype=np.float32)[-self.obs_len :]
                if robot_arr.shape[0] == self.obs_len:
                    robot_local = _to_local(robot_arr, origin, rot)
                    neigh_seq[0] = _make_feat(robot_local, is_robot=1.0)
                    neigh_mask[0] = 1.0
        slot_offset = 1 if self.include_robot else 0

        ranked_neighbors: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for neigh_xy in neigh_list:
            neigh_local = _to_local(neigh_xy, origin, rot)
            neigh_vel, _ = _speed_and_acc(neigh_local, self.obs_dt)
            rel_now = neigh_local[-1]
            dist = float(np.linalg.norm(rel_now))
            if dist <= 1.0:
                num_r1 += 1.0
            if dist <= 2.0:
                num_r2 += 1.0
            min_dist = min(min_dist, dist)
            ttc = _min_ttc(np.zeros((2,), dtype=np.float32), target_vel_now, rel_now, neigh_vel[-1])
            min_ttc = min(min_ttc, ttc)
            if dist < float(np.linalg.norm(closest_rel[:2])) or closest_rel[:2].sum() == 0.0:
                closest_rel = np.asarray([rel_now[0], rel_now[1], neigh_vel[-1, 0], neigh_vel[-1, 1]], dtype=np.float32)
            if rel_now[0] >= 0.0:
                front_density += 1.0
            if rel_now[1] >= 0.0:
                left_density += 1.0
            else:
                right_density += 1.0
            neigh_feat = _make_feat(neigh_local, is_robot=0.0)
            ranked_neighbors.append((dist, neigh_local, neigh_feat))

        ranked_neighbors.sort(key=lambda item: item[0])
        for i, (_, _, feat) in enumerate(ranked_neighbors[: self.k_neighbors]):
            neigh_seq[slot_offset + i] = feat
            neigh_mask[slot_offset + i] = 1.0

        social_summary = np.asarray(
            [
                num_r1,
                num_r2,
                min_dist if min_dist < 999.0 else 9.99,
                closest_rel[0],
                closest_rel[1],
                closest_rel[2],
                closest_rel[3],
                min_ttc if min_ttc < 999.0 else 9.99,
                front_density,
                left_density,
                right_density,
            ],
            dtype=np.float32,
        )
        kalman_vel_now = (
            (kalman_local[1] - kalman_local[0]) / max(1e-6, self.obs_dt) if kalman_local.shape[0] >= 2 else np.zeros((2,), dtype=np.float32)
        )
        kalman_features = np.concatenate(
            [
                kalman_local.reshape(-1),
                kalman_vel_now.astype(np.float32),
                kalman_local[-1].astype(np.float32),
            ]
        ).astype(np.float32)

        # Future robot trajectory in target's local frame, by CV-extrapolation
        # from robot_obs_xy. Used for robot-aware losses. This is independent of
        # include_robot: a model can be trained with robot-aware loss without
        # receiving the robot as an input feature.
        robot_future_local = np.zeros((self.pred_len, 2), dtype=np.float32)
        robot_future_valid = np.float32(0.0)
        robot_obs_xy = case.get("robot_obs_xy")
        if robot_obs_xy is not None:
            robot_arr = np.asarray(robot_obs_xy, dtype=np.float32)
            if robot_arr.shape[0] >= 2:
                robot_last_local = _to_local(robot_arr[-1:], origin, rot)[0]
                robot_prev_local = _to_local(robot_arr[-2:-1], origin, rot)[0]
                rv = (robot_last_local - robot_prev_local) / max(1e-6, float(self.obs_dt))
                pred_dt = float(self.obs_dt)  # assumes pred_dt == obs_dt; OK for our setup
                for k in range(self.pred_len):
                    robot_future_local[k] = robot_last_local + rv * (pred_dt * (k + 1))
                robot_future_valid = np.float32(1.0)

        return {
            "target_seq": torch.from_numpy(target_seq),
            "neighbor_seq": torch.from_numpy(neigh_seq),
            "neighbor_mask": torch.from_numpy(neigh_mask),
            "social_summary": torch.from_numpy(social_summary),
            "kalman_features": torch.from_numpy(kalman_features),
            "scene_patch": torch.from_numpy(scene_patch),
            "residual_target": torch.from_numpy(residual_local),
            "kalman_local": torch.from_numpy(kalman_local),
            "gt_local": torch.from_numpy(gt_local),
            "robot_future_local": torch.from_numpy(robot_future_local),
            "robot_future_valid": torch.tensor(robot_future_valid, dtype=torch.float32),
        }


def _ade_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> Tuple[float, float]:
    dist = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    return float(dist.mean().item()), float(dist[:, -1].mean().item())


def _smooth_kinematics(xy: "torch.Tensor", dt: float) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Return (acceleration, jerk) computed via finite differences along time axis.

    xy: [B, T, 2]. Returns acc [B, T-2, 2] and jerk [B, T-3, 2].
    """
    vel = (xy[:, 1:] - xy[:, :-1]) / max(1e-6, float(dt))
    acc = (vel[:, 1:] - vel[:, :-1]) / max(1e-6, float(dt))
    if acc.shape[1] < 2:
        jerk = torch.zeros((xy.shape[0], 0, 2), device=xy.device, dtype=xy.dtype)
    else:
        jerk = (acc[:, 1:] - acc[:, :-1]) / max(1e-6, float(dt))
    return acc, jerk


def _run_epoch(
    model: KalmanResidualNet,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    lambda_jerk: float = 0.0,
    lambda_acc_match: float = 0.0,
    lambda_vel: float = 0.0,
    lambda_res_mag: float = 0.0,
    lambda_risk: float = 0.0,
    lambda_safety: float = 0.0,
    lambda_robot_radial: float = 0.0,
    robot_radial_weight: float = 3.0,
    robot_tangential_weight: float = 1.0,
    robot_radial_away_weight: float = 2.0,
    robot_radial_sigma: float = 0.0,
    risk_sigma: float = 1.5,
    safety_radius: float = 0.5,
    pred_dt: float = 0.4,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_loss_pos = 0.0
    total_loss_jerk = 0.0
    total_loss_acc = 0.0
    total_loss_vel = 0.0
    total_loss_res_mag = 0.0
    total_loss_risk = 0.0
    total_loss_safety = 0.0
    total_loss_robot_radial = 0.0
    total_hybrid_ade = 0.0
    total_hybrid_fde = 0.0
    total_kf_ade = 0.0
    total_kf_fde = 0.0
    n_batches = 0

    for batch in loader:
        target_seq = batch["target_seq"].to(device).float()
        neighbor_seq = batch["neighbor_seq"].to(device).float()
        neighbor_mask = batch["neighbor_mask"].to(device).float()
        social_summary = batch["social_summary"].to(device).float()
        kalman_features = batch["kalman_features"].to(device).float()
        scene_patch = batch["scene_patch"].to(device).float()
        residual_target = batch["residual_target"].to(device).float()
        kalman_local = batch["kalman_local"].to(device).float()
        gt_local = batch["gt_local"].to(device).float()

        robot_future_local = batch.get("robot_future_local")
        robot_future_valid = batch.get("robot_future_valid")
        if robot_future_local is not None:
            robot_future_local = robot_future_local.to(device).float()
            robot_future_valid = robot_future_valid.to(device).float()

        pred_residual = model(target_seq, neighbor_seq, neighbor_mask, social_summary, kalman_features, scene_patch)
        loss_pos = loss_fn(pred_residual, residual_target)

        pred_hybrid = kalman_local + pred_residual
        loss_jerk = torch.tensor(0.0, device=device)
        loss_acc = torch.tensor(0.0, device=device)
        loss_vel = torch.tensor(0.0, device=device)
        loss_res_mag = torch.tensor(0.0, device=device)
        loss_risk = torch.tensor(0.0, device=device)
        loss_safety = torch.tensor(0.0, device=device)
        loss_robot_radial = torch.tensor(0.0, device=device)

        if lambda_jerk > 0.0 or lambda_acc_match > 0.0:
            pred_acc, pred_jerk = _smooth_kinematics(pred_hybrid, pred_dt)
            if lambda_jerk > 0.0 and pred_jerk.shape[1] > 0:
                loss_jerk = (pred_jerk ** 2).sum(dim=-1).mean()
            if lambda_acc_match > 0.0 and pred_acc.shape[1] > 0:
                gt_acc, _ = _smooth_kinematics(gt_local, pred_dt)
                loss_acc = loss_fn(pred_acc, gt_acc)

        if lambda_vel > 0.0:
            pred_vel = (pred_hybrid[:, 1:] - pred_hybrid[:, :-1]) / max(1e-6, pred_dt)
            gt_vel = (gt_local[:, 1:] - gt_local[:, :-1]) / max(1e-6, pred_dt)
            loss_vel = loss_fn(pred_vel, gt_vel)

        if lambda_res_mag > 0.0:
            loss_res_mag = (pred_residual ** 2).sum(dim=-1).mean()

        if (lambda_risk > 0.0 or lambda_safety > 0.0) and robot_future_local is not None:
            # Distance from each predicted/gt person position to the (extrapolated) robot position per step.
            d_pred = torch.linalg.norm(pred_hybrid - robot_future_local, dim=-1)  # [B, T]
            d_gt   = torch.linalg.norm(gt_local    - robot_future_local, dim=-1)  # [B, T]
            valid = robot_future_valid.unsqueeze(-1)  # [B, 1]

            if lambda_risk > 0.0:
                w_risk = torch.exp(-d_gt / max(1e-3, risk_sigma))  # bigger weight where person is close to robot
                pos_err = torch.linalg.norm(pred_hybrid - gt_local, dim=-1)  # [B, T]
                loss_risk = ((w_risk * pos_err ** 2) * valid).sum() / (valid.sum() * pos_err.shape[-1] + 1e-6)

            if lambda_safety > 0.0:
                # Penalize when model thinks person is FURTHER from robot than they actually are.
                # under_risk = ReLU(d_pred - d_gt) — only positive when pred overestimates distance.
                # Weighted by 1{d_gt < safety_radius * 3} so we focus on close-encounter cases.
                close_mask = (d_gt < (safety_radius * 3.0)).float()
                under = torch.relu(d_pred - d_gt)
                loss_safety = ((under ** 2) * close_mask * valid).sum() / (close_mask.sum() * valid.sum().clamp_min(1.0) + 1e-6)

        if lambda_robot_radial > 0.0 and robot_future_local is not None:
            # Decompose prediction error relative to robot-person radial axis.
            # radial_dir points from GT person position toward robot. Negative radial
            # scalar means prediction is farther from robot than GT, i.e. optimistic
            # for personal-space cost; robot_radial_away_weight penalizes that more.
            valid = robot_future_valid.unsqueeze(-1)  # [B, 1]
            robot_to_gt = robot_future_local - gt_local
            d_gt = torch.linalg.norm(robot_to_gt, dim=-1, keepdim=True)  # [B, T, 1]
            radial_dir = robot_to_gt / d_gt.clamp_min(1e-6)
            err = pred_hybrid - gt_local
            err_radial_scalar = (err * radial_dir).sum(dim=-1, keepdim=True)
            err_radial_sq = err_radial_scalar.squeeze(-1) ** 2
            err_total_sq = (err ** 2).sum(dim=-1)
            err_tangential_sq = torch.clamp(err_total_sq - err_radial_sq, min=0.0)
            radial_weight_eff = torch.full_like(err_radial_sq, float(robot_radial_weight))
            away_mask = (err_radial_scalar.squeeze(-1) < 0.0).float()
            radial_weight_eff = radial_weight_eff * (
                1.0 + away_mask * max(0.0, float(robot_radial_away_weight) - 1.0)
            )
            if robot_radial_sigma > 0.0:
                close_w = torch.exp(-d_gt.squeeze(-1) / max(1e-3, float(robot_radial_sigma)))
            else:
                close_w = torch.ones_like(err_radial_sq)
            w = close_w * valid
            per_step = (
                radial_weight_eff * err_radial_sq
                + float(robot_tangential_weight) * err_tangential_sq
            )
            loss_robot_radial = (per_step * w).sum() / w.sum().clamp_min(1.0)

        loss = (
            loss_pos
            + lambda_jerk * loss_jerk
            + lambda_acc_match * loss_acc
            + lambda_vel * loss_vel
            + lambda_res_mag * loss_res_mag
            + lambda_risk * loss_risk
            + lambda_safety * loss_safety
            + lambda_robot_radial * loss_robot_radial
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        hybrid_ade, hybrid_fde = _ade_fde(pred_hybrid, gt_local)
        kf_ade, kf_fde = _ade_fde(kalman_local, gt_local)

        total_loss += float(loss.item())
        total_loss_pos += float(loss_pos.item())
        total_loss_jerk += float(loss_jerk.item())
        total_loss_acc += float(loss_acc.item())
        total_loss_vel += float(loss_vel.item())
        total_loss_res_mag += float(loss_res_mag.item())
        total_loss_risk += float(loss_risk.item())
        total_loss_safety += float(loss_safety.item())
        total_loss_robot_radial += float(loss_robot_radial.item())
        total_hybrid_ade += hybrid_ade
        total_hybrid_fde += hybrid_fde
        total_kf_ade += kf_ade
        total_kf_fde += kf_fde
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "loss_pos": total_loss_pos / max(1, n_batches),
        "loss_jerk": total_loss_jerk / max(1, n_batches),
        "loss_acc": total_loss_acc / max(1, n_batches),
        "loss_vel": total_loss_vel / max(1, n_batches),
        "loss_res_mag": total_loss_res_mag / max(1, n_batches),
        "loss_risk": total_loss_risk / max(1, n_batches),
        "loss_safety": total_loss_safety / max(1, n_batches),
        "loss_robot_radial": total_loss_robot_radial / max(1, n_batches),
        "hybrid_ade": total_hybrid_ade / max(1, n_batches),
        "hybrid_fde": total_hybrid_fde / max(1, n_batches),
        "kalman_ade": total_kf_ade / max(1, n_batches),
        "kalman_fde": total_kf_fde / max(1, n_batches),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Train residual predictor over Kalman baseline.")
    parser.add_argument("--train_dataset", default=str(repo_root / "datasets" / "curated_people" / "train_residual_cases.json"))
    parser.add_argument("--val_dataset", default=str(repo_root / "datasets" / "curated_people" / "benchmark_residual_cases.json"))
    parser.add_argument("--output_dir", default=str(repo_root / "models" / "residual_predictor"))
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=26)
    parser.add_argument("--obs_dt", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--social_hidden", type=int, default=32)
    parser.add_argument("--k_neighbors", type=int, default=3)
    parser.add_argument("--scene_patch_size_m", type=float, default=6.0)
    parser.add_argument("--scene_patch_pixels", type=int, default=32)
    parser.add_argument("--scene_hidden", type=int, default=32)
    parser.add_argument("--scene_pool_size", type=int, default=4)
    parser.add_argument("--scene_patch_align_to_heading", action="store_true", default=True)
    parser.add_argument("--disable_scene_patch_align_to_heading", action="store_true")
    parser.add_argument(
        "--include_robot",
        action="store_true",
        default=False,
        help="Include robot as a special neighbor in slot 0 with is_robot flag (requires robot_obs_xy in cases).",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_jerk", type=float, default=0.0,
                        help="Weight of smoothness penalty (mean ||jerk||^2 of pred_hybrid). Try 0.01–0.1.")
    parser.add_argument("--lambda_acc_match", type=float, default=0.0,
                        help="Weight of GT-acceleration matching penalty (SmoothL1 of pred_acc vs gt_acc). Try 0.1–1.0.")
    parser.add_argument("--lambda_vel", type=float, default=0.0,
                        help="Weight of velocity matching penalty (SmoothL1 of pred_vel vs gt_vel). Try 0.1–0.3.")
    parser.add_argument("--lambda_res_mag", type=float, default=0.0,
                        help="Weight of residual L2 magnitude penalty. Keeps residual small unless needed. Try 0.01–0.05.")
    parser.add_argument("--lambda_risk", type=float, default=0.0,
                        help="Weight of risk-weighted pos loss (close to robot weighs more). Try 0.5–1.0.")
    parser.add_argument("--lambda_safety", type=float, default=0.0,
                        help="Weight of asymmetric safety loss (penalize predicting person further from robot than GT). Try 0.5–2.0.")
    parser.add_argument("--lambda_robot_radial", type=float, default=0.0,
                        help="Weight of robot radial/tangential error loss. 0 disables.")
    parser.add_argument("--robot_radial_weight", type=float, default=3.0,
                        help="Weight for error along the GT-person-to-robot axis.")
    parser.add_argument("--robot_tangential_weight", type=float, default=1.0,
                        help="Weight for error tangent to the robot-person axis.")
    parser.add_argument("--robot_radial_away_weight", type=float, default=2.0,
                        help="Extra multiplier for radial errors that predict the person farther from robot than GT.")
    parser.add_argument("--robot_radial_sigma", type=float, default=0.0,
                        help="If >0, weight radial loss by exp(-distance_to_robot/sigma) to focus on close encounters.")
    parser.add_argument("--risk_sigma", type=float, default=1.5,
                        help="Sigma for risk weight w=exp(-d_gt/sigma) in meters. Smaller = focus on closer encounters.")
    parser.add_argument("--safety_radius", type=float, default=0.5,
                        help="Radius below which safety loss kicks in (robot+person inflation in meters).")
    parser.add_argument("--pred_dt", type=float, default=0.4,
                        help="Time step between predicted frames in seconds (must match the dataset's pred_dt).")
    parser.add_argument("--stable_checkpoint_start_epoch", type=int, default=30,
                        help="Start considering best_residual_model_stable.pt from this epoch.")
    parser.add_argument("--stable_max_ade_rel_drop", type=float, default=0.01,
                        help="Max relative ADE degradation vs best ADE allowed for stable checkpoint selection.")
    return parser.parse_args()


def main() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training. Install torch.")
    args = parse_args()
    if args.disable_scene_patch_align_to_heading:
        args.scene_patch_align_to_heading = False
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ResidualTrajectoryDataset(
        path=args.train_dataset,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        obs_dt=args.obs_dt,
        k_neighbors=args.k_neighbors,
        scene_patch_size_m=args.scene_patch_size_m,
        scene_patch_pixels=args.scene_patch_pixels,
        scene_patch_align_to_heading=args.scene_patch_align_to_heading,
        include_robot=args.include_robot,
    )
    val_ds = ResidualTrajectoryDataset(
        path=args.val_dataset,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        obs_dt=args.obs_dt,
        k_neighbors=args.k_neighbors,
        scene_patch_size_m=args.scene_patch_size_m,
        scene_patch_pixels=args.scene_patch_pixels,
        scene_patch_align_to_heading=args.scene_patch_align_to_heading,
        include_robot=args.include_robot,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_eval_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    kalman_input_dim = 2 * int(args.pred_len) + 4
    neighbor_input_dim = 6 if args.include_robot else 5
    total_neighbor_slots = int(args.k_neighbors) + (1 if args.include_robot else 0)
    model = KalmanResidualNet(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        target_input_dim=8,
        neighbor_input_dim=neighbor_input_dim,
        social_summary_dim=11,
        kalman_input_dim=kalman_input_dim,
        hidden=args.hidden,
        social_hidden=args.social_hidden,
        k_neighbors=total_neighbor_slots,
        scene_channels=1,
        scene_size=int(args.scene_patch_pixels),
        scene_hidden=int(args.scene_hidden),
        scene_pool_size=int(args.scene_pool_size),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    cfg = {
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "obs_dt": float(args.obs_dt),
        "target_input_dim": 8,
        "neighbor_input_dim": neighbor_input_dim,
        "social_summary_dim": 11,
        "kalman_input_dim": kalman_input_dim,
        "hidden": int(args.hidden),
        "social_hidden": int(args.social_hidden),
        "k_neighbors": total_neighbor_slots,
        "k_humans": int(args.k_neighbors),
        "include_robot": bool(args.include_robot),
        "scene_channels": 1,
        "scene_size": int(args.scene_patch_pixels),
        "scene_hidden": int(args.scene_hidden),
        "scene_pool_size": int(args.scene_pool_size),
        "scene_patch_size_m": float(args.scene_patch_size_m),
        "scene_patch_align_to_heading": bool(args.scene_patch_align_to_heading),
        "lambda_jerk": float(args.lambda_jerk),
        "lambda_acc_match": float(args.lambda_acc_match),
        "lambda_vel": float(args.lambda_vel),
        "lambda_res_mag": float(args.lambda_res_mag),
        "lambda_risk": float(args.lambda_risk),
        "lambda_safety": float(args.lambda_safety),
        "lambda_robot_radial": float(args.lambda_robot_radial),
        "robot_radial_weight": float(args.robot_radial_weight),
        "robot_tangential_weight": float(args.robot_tangential_weight),
        "robot_radial_away_weight": float(args.robot_radial_away_weight),
        "robot_radial_sigma": float(args.robot_radial_sigma),
    }

    best_val_ade = float("inf")
    best_stable_val_jerk = float("inf")
    best_stable_epoch = 0
    history: List[Dict[str, Any]] = []
    common_kwargs = dict(
        lambda_jerk=float(args.lambda_jerk),
        lambda_acc_match=float(args.lambda_acc_match),
        lambda_vel=float(args.lambda_vel),
        lambda_res_mag=float(args.lambda_res_mag),
        lambda_risk=float(args.lambda_risk),
        lambda_safety=float(args.lambda_safety),
        lambda_robot_radial=float(args.lambda_robot_radial),
        robot_radial_weight=float(args.robot_radial_weight),
        robot_tangential_weight=float(args.robot_tangential_weight),
        robot_radial_away_weight=float(args.robot_radial_away_weight),
        robot_radial_sigma=float(args.robot_radial_sigma),
        risk_sigma=float(args.risk_sigma),
        safety_radius=float(args.safety_radius),
        pred_dt=float(args.pred_dt),
    )

    for epoch in range(1, int(args.epochs) + 1):
        # 1) Optimization pass: loss recorded here is the noisy running-average DURING updates.
        _ = _run_epoch(model, train_dl, device, optimizer, loss_fn, **common_kwargs)
        # 2) Honest train eval pass: frozen weights, same data → loss directly comparable to val.
        with torch.no_grad():
            train_stats = _run_epoch(model, train_eval_dl, device, None, loss_fn, **common_kwargs)
            val_stats = _run_epoch(model, val_dl, device, None, loss_fn, **common_kwargs)

        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_stats['loss']:.4f} ADE={train_stats['hybrid_ade']:.4f} "
            f"FDE={train_stats['hybrid_fde']:.4f} | "
            f"val loss={val_stats['loss']:.4f} ADE={val_stats['hybrid_ade']:.4f} "
            f"FDE={val_stats['hybrid_fde']:.4f} | "
            f"val kalman_ADE={val_stats['kalman_ade']:.4f} kalman_FDE={val_stats['kalman_fde']:.4f}"
        )

        if val_stats["hybrid_ade"] < best_val_ade:
            best_val_ade = val_stats["hybrid_ade"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_val_hybrid_ade": best_val_ade,
                    "history": history,
                },
                out_dir / "best_residual_model.pt",
            )
        stable_start_epoch = max(1, int(args.stable_checkpoint_start_epoch))
        stable_max_ade = best_val_ade * (1.0 + max(0.0, float(args.stable_max_ade_rel_drop)))
        val_ade = float(val_stats["hybrid_ade"])
        val_jerk = float(val_stats.get("loss_jerk", 0.0))
        if epoch >= stable_start_epoch and val_ade <= stable_max_ade and val_jerk < best_stable_val_jerk:
            best_stable_val_jerk = val_jerk
            best_stable_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_stable_val_jerk": best_stable_val_jerk,
                    "stable_start_epoch": stable_start_epoch,
                    "stable_max_ade_rel_drop": float(args.stable_max_ade_rel_drop),
                    "stable_max_ade": stable_max_ade,
                    "val_hybrid_ade": val_ade,
                    "val_hybrid_fde": float(val_stats["hybrid_fde"]),
                    "history": history,
                },
                out_dir / "best_residual_model_stable.pt",
            )

    (out_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
    _plot_training_summary(history, out_dir / "loss.png")
    print(f"best val hybrid ADE: {best_val_ade:.4f}")
    print(f"saved checkpoint: {out_dir / 'best_residual_model.pt'}")
    if best_stable_epoch > 0:
        print(
            f"best stable val jerk: {best_stable_val_jerk:.4f} "
            f"at epoch {best_stable_epoch}; saved checkpoint: {out_dir / 'best_residual_model_stable.pt'}"
        )
    else:
        print("stable checkpoint was not saved; no epoch satisfied the ADE tolerance criterion")


if __name__ == "__main__":
    main()
