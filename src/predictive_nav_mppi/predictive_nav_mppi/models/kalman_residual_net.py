"""Residual predictor over Kalman baseline for pedestrian trajectory forecasting."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from predictive_nav_mppi.scene_context import ScenePatchConfig, extract_patch_from_source, load_occupancy_scene_map, extract_scene_patch

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class KalmanResidualNetLegacy(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 26,
        target_input_dim: int = 8,
        neighbor_input_dim: int = 5,
        social_summary_dim: int = 11,
        kalman_input_dim: int = 2 * 26 + 4,
        hidden: int = 64,
        social_hidden: int = 32,
        k_neighbors: int = 3,
        scene_channels: int = 1,
        scene_size: int = 32,
        scene_hidden: int = 32,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for KalmanResidualNet. Install torch.")
        super().__init__()
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.target_input_dim = int(target_input_dim)
        self.neighbor_input_dim = int(neighbor_input_dim)
        self.social_summary_dim = int(social_summary_dim)
        self.kalman_input_dim = int(kalman_input_dim)
        self.hidden = int(hidden)
        self.social_hidden = int(social_hidden)
        self.k_neighbors = int(k_neighbors)
        self.scene_channels = int(scene_channels)
        self.scene_size = int(scene_size)
        self.scene_hidden = int(scene_hidden)

        self.target_gru = nn.GRU(
            input_size=self.target_input_dim,
            hidden_size=self.hidden,
            batch_first=True,
        )
        self.neighbor_gru = nn.GRU(
            input_size=self.neighbor_input_dim,
            hidden_size=self.hidden,
            batch_first=True,
        )
        self.social_summary_mlp = nn.Sequential(
            nn.Linear(self.social_summary_dim, self.social_hidden),
            nn.ReLU(),
            nn.Linear(self.social_hidden, self.social_hidden),
            nn.ReLU(),
        )
        self.kalman_mlp = nn.Sequential(
            nn.Linear(self.kalman_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.social_hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden + self.hidden + self.social_hidden + self.social_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.pred_len),
        )

    def forward(
        self,
        target_seq: "torch.Tensor",
        neighbor_seq: "torch.Tensor",
        neighbor_mask: "torch.Tensor",
        social_summary: "torch.Tensor",
        kalman_features: "torch.Tensor",
    ) -> "torch.Tensor":
        _, h_tgt = self.target_gru(target_seq)
        h_tgt = h_tgt.squeeze(0)

        bsz, k_neigh, steps, nfeat = neighbor_seq.shape
        neigh_flat = neighbor_seq.reshape(bsz * k_neigh, steps, nfeat)
        _, h_neigh = self.neighbor_gru(neigh_flat)
        h_neigh = h_neigh.squeeze(0).reshape(bsz, k_neigh, self.hidden)
        mask = neighbor_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        h_social = (h_neigh * mask).sum(dim=1) / denom

        social_vec = self.social_summary_mlp(social_summary)
        kalman_vec = self.kalman_mlp(kalman_features)
        fused = torch.cat([h_tgt, h_social, social_vec, kalman_vec], dim=-1)
        out = self.fusion(fused)
        return out.view(target_seq.shape[0], self.pred_len, 2)


class KalmanResidualNet(KalmanResidualNetLegacy):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 26,
        target_input_dim: int = 8,
        neighbor_input_dim: int = 5,
        social_summary_dim: int = 11,
        kalman_input_dim: int = 2 * 26 + 4,
        hidden: int = 64,
        social_hidden: int = 32,
        k_neighbors: int = 3,
        scene_channels: int = 1,
        scene_size: int = 32,
        scene_hidden: int = 32,
        scene_pool_size: int = 4,
    ) -> None:
        super().__init__(
            obs_len=obs_len,
            pred_len=pred_len,
            target_input_dim=target_input_dim,
            neighbor_input_dim=neighbor_input_dim,
            social_summary_dim=social_summary_dim,
            kalman_input_dim=kalman_input_dim,
            hidden=hidden,
            social_hidden=social_hidden,
            k_neighbors=k_neighbors,
        )
        self.scene_channels = int(scene_channels)
        self.scene_size = int(scene_size)
        self.scene_hidden = int(scene_hidden)
        self.scene_pool_size = int(scene_pool_size)
        self.scene_cnn = nn.Sequential(
            nn.Conv2d(self.scene_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.scene_pool_size, self.scene_pool_size)),
        )
        self.scene_mlp = nn.Sequential(
            nn.Linear(64 * self.scene_pool_size * self.scene_pool_size, self.scene_hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden + self.hidden + self.social_hidden + self.social_hidden + self.scene_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.pred_len),
        )

    def forward(
        self,
        target_seq: "torch.Tensor",
        neighbor_seq: "torch.Tensor",
        neighbor_mask: "torch.Tensor",
        social_summary: "torch.Tensor",
        kalman_features: "torch.Tensor",
        scene_patch: "torch.Tensor",
    ) -> "torch.Tensor":
        # target_seq: [B,T,Ft]
        # neighbor_seq: [B,K,T,Fn], neighbor_mask: [B,K]
        _, h_tgt = self.target_gru(target_seq)
        h_tgt = h_tgt.squeeze(0)

        bsz, k_neigh, steps, nfeat = neighbor_seq.shape
        neigh_flat = neighbor_seq.reshape(bsz * k_neigh, steps, nfeat)
        _, h_neigh = self.neighbor_gru(neigh_flat)
        h_neigh = h_neigh.squeeze(0).reshape(bsz, k_neigh, self.hidden)
        mask = neighbor_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        h_social = (h_neigh * mask).sum(dim=1) / denom

        social_vec = self.social_summary_mlp(social_summary)
        kalman_vec = self.kalman_mlp(kalman_features)
        scene_feat = self.scene_cnn(scene_patch).flatten(start_dim=1)
        scene_vec = self.scene_mlp(scene_feat)
        fused = torch.cat([h_tgt, h_social, social_vec, kalman_vec, scene_vec], dim=-1)
        out = self.fusion(fused)
        return out.view(target_seq.shape[0], self.pred_len, 2)


def load_checkpoint(path: str | Path, device: str | None = None) -> Tuple[nn.Module, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install torch.")
    ckpt = torch.load(str(Path(path).expanduser()), map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    has_scene = any(key in cfg for key in ("scene_channels", "scene_size", "scene_hidden", "scene_patch_size_m"))
    base_kwargs = dict(
        obs_len=int(cfg.get("obs_len", 8)),
        pred_len=int(cfg.get("pred_len", 26)),
        target_input_dim=int(cfg.get("target_input_dim", 8)),
        neighbor_input_dim=int(cfg.get("neighbor_input_dim", 5)),
        social_summary_dim=int(cfg.get("social_summary_dim", 11)),
        kalman_input_dim=int(cfg.get("kalman_input_dim", 56)),
        hidden=int(cfg.get("hidden", 64)),
        social_hidden=int(cfg.get("social_hidden", 32)),
        k_neighbors=int(cfg.get("k_neighbors", 3)),
    )
    if has_scene:
        model = KalmanResidualNet(
            **base_kwargs,
            scene_channels=int(cfg.get("scene_channels", 1)),
            scene_size=int(cfg.get("scene_size", 32)),
            scene_hidden=int(cfg.get("scene_hidden", 32)),
            scene_pool_size=int(cfg.get("scene_pool_size", 4)),
        )
        cfg.setdefault("residual_variant", "scene")
    else:
        model = KalmanResidualNetLegacy(**base_kwargs)
        cfg.setdefault("residual_variant", "legacy")
    model.load_state_dict(ckpt["model"], strict=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, cfg


def _rotation(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def compute_turn_gate(
    obs_xy: np.ndarray,
    tau: float = 0.1,
    alpha: float = 30.0,
) -> float:
    obs_xy = np.asarray(obs_xy, dtype=np.float64)
    if obs_xy.shape[0] < 4:
        return 1.0
    step = obs_xy[1:] - obs_xy[:-1]
    step_norm = np.linalg.norm(step, axis=1)
    if float(np.max(step_norm)) < 1e-6:
        return 0.0
    headings = np.arctan2(step[:, 1], step[:, 0])
    d1 = abs(_wrap_to_pi(float(headings[-1] - headings[-2])))
    d2 = abs(_wrap_to_pi(float(headings[-2] - headings[-3])))
    s_turn = 0.5 * (d1 + d2)
    z = float(alpha) * (s_turn - float(tau))
    z = max(-60.0, min(60.0, z))
    return float(1.0 / (1.0 + math.exp(-z)))


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


def _min_ttc(target_vel: np.ndarray, neigh_pos: np.ndarray, neigh_vel: np.ndarray) -> float:
    rel_p = neigh_pos
    rel_v = neigh_vel - target_vel
    vv = float(np.dot(rel_v, rel_v))
    if vv < 1e-6:
        return 999.0
    t = -float(np.dot(rel_p, rel_v)) / vv
    if t <= 0.0:
        return 999.0
    return min(t, 999.0)


def build_residual_features(
    obs_xy: np.ndarray,
    neigh_xy: List[np.ndarray],
    kalman_pred_xy: np.ndarray,
    obs_dt: float,
    k_neighbors: int,
    scene_patch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_xy = np.asarray(obs_xy, dtype=np.float32)
    kalman_pred_xy = np.asarray(kalman_pred_xy, dtype=np.float32)
    origin = obs_xy[-1].copy()
    heading = obs_xy[-1] - obs_xy[-2] if obs_xy.shape[0] >= 2 else np.zeros((2,), dtype=np.float32)
    theta = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1e-6 else 0.0
    rot = _rotation(-theta)
    rot_inv = rot.T

    obs_local = _to_local(obs_xy, origin, rot)
    kalman_local = _to_local(kalman_pred_xy, origin, rot)
    vel_local, acc_local = _speed_and_acc(obs_local, obs_dt)
    speed = np.linalg.norm(vel_local, axis=1, keepdims=True).astype(np.float32)
    heading_delta = _heading_delta(obs_local)
    target_seq = np.concatenate([obs_local, vel_local, speed, acc_local, heading_delta], axis=1).astype(np.float32)

    neigh_seq = np.zeros((k_neighbors, obs_local.shape[0], 5), dtype=np.float32)
    neigh_mask = np.zeros((k_neighbors,), dtype=np.float32)
    target_vel_now = vel_local[-1]
    num_r1 = 0.0
    num_r2 = 0.0
    min_dist = 999.0
    min_ttc = 999.0
    closest_rel = np.zeros((4,), dtype=np.float32)
    front_density = 0.0
    left_density = 0.0
    right_density = 0.0
    ranked_neighbors: List[Tuple[float, np.ndarray]] = []

    for neigh in neigh_xy:
        neigh = np.asarray(neigh, dtype=np.float32)
        neigh_local = _to_local(neigh, origin, rot)
        neigh_vel, _ = _speed_and_acc(neigh_local, obs_dt)
        rel_now = neigh_local[-1]
        dist = float(np.linalg.norm(rel_now))
        if dist <= 1.0:
            num_r1 += 1.0
        if dist <= 2.0:
            num_r2 += 1.0
        min_dist = min(min_dist, dist)
        min_ttc = min(min_ttc, _min_ttc(target_vel_now, rel_now, neigh_vel[-1]))
        if dist < float(np.linalg.norm(closest_rel[:2])) or closest_rel[:2].sum() == 0.0:
            closest_rel = np.asarray([rel_now[0], rel_now[1], neigh_vel[-1, 0], neigh_vel[-1, 1]], dtype=np.float32)
        if rel_now[0] >= 0.0:
            front_density += 1.0
        if rel_now[1] >= 0.0:
            left_density += 1.0
        else:
            right_density += 1.0
        neigh_feat = np.concatenate(
            [neigh_local, neigh_vel, np.linalg.norm(neigh_vel, axis=1, keepdims=True).astype(np.float32)],
            axis=1,
        ).astype(np.float32)
        ranked_neighbors.append((dist, neigh_feat))

    ranked_neighbors.sort(key=lambda item: item[0])
    for i, (_, feat) in enumerate(ranked_neighbors[:k_neighbors]):
        neigh_seq[i] = feat
        neigh_mask[i] = 1.0

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
        (kalman_local[1] - kalman_local[0]) / max(1e-6, obs_dt)
        if kalman_local.shape[0] >= 2
        else np.zeros((2,), dtype=np.float32)
    )
    kalman_features = np.concatenate(
        [kalman_local.reshape(-1), kalman_vel_now.astype(np.float32), kalman_local[-1].astype(np.float32)]
    ).astype(np.float32)
    scene_patch = np.asarray(scene_patch, dtype=np.float32)
    return target_seq, neigh_seq, neigh_mask, social_summary, kalman_features, scene_patch, origin, rot, rot_inv


def predict_residual_world(
    model: nn.Module,
    obs_xy: np.ndarray,
    neigh_xy: List[np.ndarray],
    kalman_pred_xy: np.ndarray,
    obs_dt: float,
    device: str,
    k_neighbors: int = 3,
    residual_alpha: float = 1.0,
    scene_patch: np.ndarray | None = None,
    scene_source_name: str = "",
    scene_patch_cfg: ScenePatchConfig | None = None,
    scene_map_yaml: str = "",
) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install torch.")
    obs_xy = np.asarray(obs_xy, dtype=np.float32)
    expects_scene = hasattr(model, "scene_cnn")
    if scene_patch is None and expects_scene:
        heading = obs_xy[-1] - obs_xy[-2] if obs_xy.shape[0] >= 2 else np.zeros((2,), dtype=np.float32)
        heading_rad = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1e-6 else 0.0
        cfg = scene_patch_cfg or ScenePatchConfig()
        if scene_map_yaml:
            scene_map = load_occupancy_scene_map(scene_map_yaml)
            scene_patch = extract_scene_patch(
                scene_map=scene_map,
                center_xy=obs_xy[-1],
                heading_rad=heading_rad,
                cfg=cfg,
            )
        else:
            if not scene_source_name:
                raise ValueError("scene_source_name or scene_map_yaml is required when scene_patch is not provided")
            scene_patch = extract_patch_from_source(
                source_name=scene_source_name,
                center_xy=obs_xy[-1],
                heading_rad=heading_rad,
                cfg=cfg,
            )

    if expects_scene:
        target_seq, neigh_seq, neigh_mask, social_summary, kalman_features, scene_patch, origin, rot, rot_inv = build_residual_features(
            obs_xy=obs_xy,
            neigh_xy=neigh_xy,
            kalman_pred_xy=kalman_pred_xy,
            obs_dt=obs_dt,
            k_neighbors=k_neighbors,
            scene_patch=scene_patch,
        )
    else:
        dummy_scene = np.zeros((1, 1, 1), dtype=np.float32)
        target_seq, neigh_seq, neigh_mask, social_summary, kalman_features, _, origin, rot, rot_inv = build_residual_features(
            obs_xy=obs_xy,
            neigh_xy=neigh_xy,
            kalman_pred_xy=kalman_pred_xy,
            obs_dt=obs_dt,
            k_neighbors=k_neighbors,
            scene_patch=dummy_scene,
        )
    with torch.no_grad():
        if expects_scene:
            pred_res = model(
                torch.from_numpy(target_seq[None, ...]).to(device).float(),
                torch.from_numpy(neigh_seq[None, ...]).to(device).float(),
                torch.from_numpy(neigh_mask[None, ...]).to(device).float(),
                torch.from_numpy(social_summary[None, ...]).to(device).float(),
                torch.from_numpy(kalman_features[None, ...]).to(device).float(),
                torch.from_numpy(scene_patch[None, ...]).to(device).float(),
            )[0].detach().cpu().numpy()
        else:
            pred_res = model(
                torch.from_numpy(target_seq[None, ...]).to(device).float(),
                torch.from_numpy(neigh_seq[None, ...]).to(device).float(),
                torch.from_numpy(neigh_mask[None, ...]).to(device).float(),
                torch.from_numpy(social_summary[None, ...]).to(device).float(),
                torch.from_numpy(kalman_features[None, ...]).to(device).float(),
            )[0].detach().cpu().numpy()
    pred_local = _to_local(np.asarray(kalman_pred_xy, dtype=np.float32), origin, rot)
    alpha = float(max(0.0, residual_alpha))
    hybrid_local = pred_local + alpha * pred_res.astype(np.float32)
    pred_world = (rot_inv @ hybrid_local.T).T + origin
    return pred_world.astype(np.float64)
