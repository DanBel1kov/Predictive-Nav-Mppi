"""Social GRU predictor — load and run inference for pedestrian trajectory prediction."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _rotation_matrix_2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


class SocialGRUPredictor(nn.Module if TORCH_AVAILABLE else object):
    """GRU-based social predictor: obs trajectory + neighbors -> future trajectory (agent-centric)."""

    def __init__(
        self,
        hidden: int = 128,
        obs_len: int = 8,
        pred_len: int = 12,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for SocialGRUPredictor. Install torch.")
        super().__init__()
        self.hidden = hidden
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.enc_tgt = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.enc_neigh = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden, 256),
            nn.ReLU(),
            nn.Linear(256, hidden),
            nn.ReLU(),
        )
        self.dec = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2)

    def forward(
        self,
        tgt_obs: "torch.Tensor",
        neigh_obs: "torch.Tensor",
        neigh_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        # tgt_obs: [B,T,2], neigh_obs: [B,N,T,2], neigh_mask: [B,N]
        B, T, _ = tgt_obs.shape
        _, N, _, _ = neigh_obs.shape

        _, h_tgt = self.enc_tgt(tgt_obs)
        h_tgt = h_tgt.squeeze(0)

        neigh_flat = neigh_obs.reshape(B * N, T, 2)
        _, h_neigh = self.enc_neigh(neigh_flat)
        h_neigh = h_neigh.squeeze(0).reshape(B, N, self.hidden)

        m = neigh_mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        h_social = (h_neigh * m).sum(dim=1) / denom

        h0 = self.fuse(torch.cat([h_tgt, h_social], dim=-1))
        h = h0.unsqueeze(0)

        last = tgt_obs[:, -1, :]
        preds = []
        x = last
        for _ in range(self.pred_len):
            out, h = self.dec(x.unsqueeze(1), h)
            dx = self.out(out.squeeze(1))
            x = x + dx
            preds.append(x.unsqueeze(1))
        return torch.cat(preds, dim=1)


def load_checkpoint(
    path: str | Path,
    device: Optional[str] = None,
) -> Tuple[SocialGRUPredictor, Dict[str, Any]]:
    """Load model from .pt checkpoint. Returns (model, cfg_dict)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install torch.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or ckpt.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    obs_len = int(cfg.get("obs_len", 8))
    pred_len = int(cfg.get("pred_len", 12))
    model = SocialGRUPredictor(hidden=128, obs_len=obs_len, pred_len=pred_len)
    state = ckpt.get("model") or ckpt.get("model_state")
    if state is None:
        raise KeyError("Checkpoint must contain 'model' or 'model_state'")
    model.load_state_dict(state, strict=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, cfg


def to_agent_centric(
    obs_xy: np.ndarray,
    world_velocity: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """obs_xy: [T,2]. world_velocity: optional (vx, vy) in world frame for heading.
    Returns (obs_rel, origin, theta, R)."""
    origin = np.asarray(obs_xy[-1], dtype=np.float32)
    theta = 0.0
    if world_velocity is not None:
        vx, vy = float(world_velocity[0]), float(world_velocity[1])
        if abs(vx) + abs(vy) > 1e-6:
            theta = float(math.atan2(vy, vx))
    if theta == 0.0 and obs_xy.shape[0] >= 2:
        v = obs_xy[-1] - obs_xy[-2]
        if abs(v[0]) + abs(v[1]) > 1e-6:
            theta = float(math.atan2(v[1], v[0]))
    R = _rotation_matrix_2d(-theta)
    obs_rel = (R @ (obs_xy - origin).T).T.astype(np.float32)
    return obs_rel, origin, theta, R


def predict_trajectories_world(
    model: SocialGRUPredictor,
    tracks_obs_xy: List[np.ndarray],
    tracks_neigh_xy: List[List[np.ndarray]],
    device: str = "cpu",
    tracks_velocity_xy: Optional[List[Optional[Tuple[float, float]]]] = None,
    flip_forward_axis: bool = False,
) -> List[np.ndarray]:
    """
    Predict future trajectories in world frame.
    - tracks_obs_xy: list of [T_obs, 2] arrays (world frame) per person.
    - tracks_neigh_xy: for each person, list of [T_obs, 2] neighbor trajectories in world frame.
    - tracks_velocity_xy: optional list of (vx, vy) per person (world frame); used for heading when norm > 0.
    - flip_forward_axis: if True, flip agent-centric x before transform (for models trained with opposite forward).
    Returns list of [T_pred, 2] arrays in world frame.
    """
    if not tracks_obs_xy:
        return []
    obs_len = model.obs_len
    pred_len = model.pred_len
    n_max = 10
    if tracks_velocity_xy is None:
        tracks_velocity_xy = [None] * len(tracks_obs_xy)

    batch_tgt_rel = []
    batch_neigh_rel = []
    batch_neigh_mask = []
    origins = []
    R_matrices = []

    for i, tgt_xy in enumerate(tracks_obs_xy):
        # Pad or trim to obs_len
        if tgt_xy.shape[0] < obs_len:
            pad = np.repeat(tgt_xy[:1], obs_len - tgt_xy.shape[0], axis=0)
            tgt_xy = np.concatenate([pad, tgt_xy], axis=0)
        tgt_xy = tgt_xy[-obs_len:].astype(np.float32)
        vel = tracks_velocity_xy[i] if i < len(tracks_velocity_xy) else None
        tgt_rel, origin, _, R = to_agent_centric(tgt_xy, world_velocity=vel)
        origins.append(origin)
        R_matrices.append(R)
        batch_tgt_rel.append(tgt_rel)

        neigh_list = tracks_neigh_xy[i] if i < len(tracks_neigh_xy) else []
        neigh_rel = np.zeros((n_max, obs_len, 2), dtype=np.float32)
        neigh_mask = np.zeros((n_max,), dtype=np.float32)
        for j, n_xy in enumerate(neigh_list[:n_max]):
            if n_xy.shape[0] < obs_len:
                pad = np.repeat(n_xy[:1], obs_len - n_xy.shape[0], axis=0)
                n_xy = np.concatenate([pad, n_xy], axis=0)
            n_xy = n_xy[-obs_len:].astype(np.float32)
            n_rel = (R @ (n_xy - origin).T).T.astype(np.float32)
            neigh_rel[j] = n_rel
            neigh_mask[j] = 1.0
        batch_neigh_rel.append(neigh_rel)
        batch_neigh_mask.append(neigh_mask)

    tgt_t = torch.from_numpy(np.stack(batch_tgt_rel)).to(device).float()
    neigh_t = torch.from_numpy(np.stack(batch_neigh_rel)).to(device).float()
    mask_t = torch.from_numpy(np.stack(batch_neigh_mask)).to(device).float()

    with torch.no_grad():
        pred_rel = model(tgt_t, neigh_t, mask_t)
    pred_rel_np = pred_rel.cpu().numpy()

    if flip_forward_axis:
        pred_rel_np = pred_rel_np.copy()
        pred_rel_np[:, :, 0] *= -1.0

    out = []
    for i in range(len(tracks_obs_xy)):
        R_inv = R_matrices[i].T
        pred_world = (R_inv @ pred_rel_np[i].T).T + origins[i]
        out.append(pred_world.astype(np.float64))
    return out
