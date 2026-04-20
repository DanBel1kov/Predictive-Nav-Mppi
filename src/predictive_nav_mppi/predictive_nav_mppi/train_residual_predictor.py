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
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install torch.")
        payload = json.loads(Path(path).expanduser().read_text())
        self.cases = payload["cases"]
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.obs_dt = float(obs_dt)
        self.k_neighbors = int(k_neighbors)
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

        neigh_seq = np.zeros((self.k_neighbors, self.obs_len, 5), dtype=np.float32)
        neigh_mask = np.zeros((self.k_neighbors,), dtype=np.float32)

        target_vel_now = vel_local[-1]
        num_r1 = 0.0
        num_r2 = 0.0
        min_dist = 999.0
        min_ttc = 999.0
        closest_rel = np.zeros((4,), dtype=np.float32)
        front_density = 0.0
        left_density = 0.0
        right_density = 0.0

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
            neigh_feat = np.concatenate(
                [neigh_local, neigh_vel, np.linalg.norm(neigh_vel, axis=1, keepdims=True).astype(np.float32)],
                axis=1,
            ).astype(np.float32)
            ranked_neighbors.append((dist, neigh_local, neigh_feat))

        ranked_neighbors.sort(key=lambda item: item[0])
        for i, (_, _, feat) in enumerate(ranked_neighbors[: self.k_neighbors]):
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
            (kalman_local[1] - kalman_local[0]) / max(1e-6, self.obs_dt) if kalman_local.shape[0] >= 2 else np.zeros((2,), dtype=np.float32)
        )
        kalman_features = np.concatenate(
            [
                kalman_local.reshape(-1),
                kalman_vel_now.astype(np.float32),
                kalman_local[-1].astype(np.float32),
            ]
        ).astype(np.float32)

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
        }


def _ade_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> Tuple[float, float]:
    dist = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    return float(dist.mean().item()), float(dist[:, -1].mean().item())


def _run_epoch(
    model: KalmanResidualNet,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
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

        pred_residual = model(target_seq, neighbor_seq, neighbor_mask, social_summary, kalman_features, scene_patch)
        loss = loss_fn(pred_residual, residual_target)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        pred_hybrid = kalman_local + pred_residual
        hybrid_ade, hybrid_fde = _ade_fde(pred_hybrid, gt_local)
        kf_ade, kf_fde = _ade_fde(kalman_local, gt_local)

        total_loss += float(loss.item())
        total_hybrid_ade += hybrid_ade
        total_hybrid_fde += hybrid_fde
        total_kf_ade += kf_ade
        total_kf_fde += kf_fde
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
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
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
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
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    kalman_input_dim = 2 * int(args.pred_len) + 4
    model = KalmanResidualNet(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        target_input_dim=8,
        neighbor_input_dim=5,
        social_summary_dim=11,
        kalman_input_dim=kalman_input_dim,
        hidden=args.hidden,
        social_hidden=args.social_hidden,
        k_neighbors=args.k_neighbors,
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
        "neighbor_input_dim": 5,
        "social_summary_dim": 11,
        "kalman_input_dim": kalman_input_dim,
        "hidden": int(args.hidden),
        "social_hidden": int(args.social_hidden),
        "k_neighbors": int(args.k_neighbors),
        "scene_channels": 1,
        "scene_size": int(args.scene_patch_pixels),
        "scene_hidden": int(args.scene_hidden),
        "scene_pool_size": int(args.scene_pool_size),
        "scene_patch_size_m": float(args.scene_patch_size_m),
        "scene_patch_align_to_heading": bool(args.scene_patch_align_to_heading),
    }

    best_val_ade = float("inf")
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = _run_epoch(model, train_dl, device, optimizer, loss_fn)
        with torch.no_grad():
            val_stats = _run_epoch(model, val_dl, device, None, loss_fn)

        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_stats['loss']:.4f} hybrid_ADE={train_stats['hybrid_ade']:.4f} "
            f"hybrid_FDE={train_stats['hybrid_fde']:.4f} | "
            f"val loss={val_stats['loss']:.4f} hybrid_ADE={val_stats['hybrid_ade']:.4f} "
            f"hybrid_FDE={val_stats['hybrid_fde']:.4f} | "
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

    (out_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
    print(f"best val hybrid ADE: {best_val_ade:.4f}")
    print(f"saved checkpoint: {out_dir / 'best_residual_model.pt'}")


if __name__ == "__main__":
    main()
