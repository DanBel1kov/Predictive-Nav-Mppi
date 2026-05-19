#!/usr/bin/env python3
"""Fine-tune SocialVAE on curated people cases.

Loads a pre-trained SocialVAE checkpoint (e.g. vae_hotel), freezes encoder +
latent layers, and only trains the decoder head (rnn_fy_init + dec) on our
curated trajectory cases. Honest fight with KalmanResidualNet: similar trainable
parameter budget (~271k vs ~131k).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _import_social_vae(repo_path: str) -> Any:
    repo_path = str(Path(repo_path).expanduser().resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from social_vae import SocialVAE
    return SocialVAE


class CuratedVAEDataset(Dataset):
    """Adapter: curated residual cases → SocialVAE-style (x, y, neighbor) tensors.

    SocialVAE expects:
      x:        [L1+1, 6] = (px, py, vx, vy, ax, ay) — observation incl. last frame
      y:        [L2,   2] = future (px, py) ground truth
      neighbor: [L1+L2+1, Nn, 6] = same 6-d state per neighbor at every step,
                padded with large value (1e9) to mark "no neighbor".
    """

    def __init__(
        self,
        path: str | Path,
        obs_len: int = 8,
        pred_len: int = 12,
        obs_dt: float = 0.4,
        max_neighbors: int = 8,
        pad_value: float = 1e9,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required.")
        payload = json.loads(Path(path).expanduser().read_text())
        self.cases = payload["cases"]
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.obs_dt = float(obs_dt)
        self.max_neighbors = int(max_neighbors)
        self.pad_value = float(pad_value)

    def __len__(self) -> int:
        return len(self.cases)

    @staticmethod
    def _state_seq(xy: np.ndarray, dt: float) -> np.ndarray:
        """Build (px, py, vx, vy, ax, ay) per timestep from positions."""
        T = xy.shape[0]
        out = np.zeros((T, 6), dtype=np.float32)
        out[:, 0:2] = xy
        if T >= 2:
            v = (xy[1:] - xy[:-1]) / max(1e-6, dt)
            out[1:, 2:4] = v
            out[0, 2:4] = v[0]
        if T >= 3:
            a = (out[1:, 2:4] - out[:-1, 2:4]) / max(1e-6, dt)
            out[1:, 4:6] = a
            out[0, 4:6] = a[0]
        return out

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case = self.cases[index]
        obs_xy = np.asarray(case["obs_xy"], dtype=np.float32)[-self.obs_len:]
        gt_xy = np.asarray(case["gt_xy"], dtype=np.float32)[: self.pred_len]
        neigh_list = [
            np.asarray(arr, dtype=np.float32)[-self.obs_len:]
            for arr in case.get("neigh_xy", [])
        ]

        # x is observation states with shape [L1+1, 6]; SocialVAE expects the
        # last frame of x to coincide with the "now" frame, so we take obs_len.
        x_state = self._state_seq(obs_xy, self.obs_dt)  # [obs_len, 6]
        y_state = gt_xy[: self.pred_len].astype(np.float32)  # [pred_len, 2]

        # Pad with the last obs frame if shorter than expected.
        if x_state.shape[0] < self.obs_len:
            pad = np.repeat(x_state[:1], self.obs_len - x_state.shape[0], axis=0)
            x_state = np.concatenate([pad, x_state], axis=0)
        if y_state.shape[0] < self.pred_len:
            pad = np.repeat(y_state[-1:], self.pred_len - y_state.shape[0], axis=0)
            y_state = np.concatenate([y_state, pad], axis=0)

        L1 = self.obs_len  # SocialVAE uses L1+1 obs frames; here we treat obs_len as L1+1
        Lp = self.pred_len
        Nn = self.max_neighbors

        # neighbor tensor: [L1+Lp, Nn, 6], padded with 1e9 for missing slots
        neighbor = np.full((L1 + Lp, Nn, 6), self.pad_value, dtype=np.float32)
        n_used = min(len(neigh_list), Nn)
        for i in range(n_used):
            ne_obs_xy = neigh_list[i]
            # build observation portion
            if ne_obs_xy.shape[0] < L1:
                pad = np.repeat(ne_obs_xy[:1], L1 - ne_obs_xy.shape[0], axis=0)
                ne_obs_xy = np.concatenate([pad, ne_obs_xy], axis=0)
            ne_obs_state = self._state_seq(ne_obs_xy, self.obs_dt)  # [L1, 6]

            # for future portion: extrapolate position via last velocity (CV)
            last_vel = ne_obs_state[-1, 2:4]
            future_xy = np.zeros((Lp, 2), dtype=np.float32)
            for t in range(Lp):
                future_xy[t] = ne_obs_xy[-1] + last_vel * (self.obs_dt * (t + 1))
            ne_future_state = np.zeros((Lp, 6), dtype=np.float32)
            ne_future_state[:, 0:2] = future_xy
            ne_future_state[:, 2:4] = last_vel  # constant velocity assumption
            # ax, ay stay 0

            neighbor[:L1, i] = ne_obs_state
            neighbor[L1:L1 + Lp, i] = ne_future_state

        return {
            "x": torch.from_numpy(x_state),                    # [L1, 6]
            "y": torch.from_numpy(y_state),                    # [Lp, 2]
            "neighbor": torch.from_numpy(neighbor),            # [L1+Lp, Nn, 6]
        }


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SocialVAE wants time-first tensors: [L, N, ...]."""
    x = torch.stack([b["x"] for b in batch], dim=1)            # [L1, N, 6]
    y = torch.stack([b["y"] for b in batch], dim=1)            # [Lp, N, 2]
    neighbor = torch.stack([b["neighbor"] for b in batch], dim=1)  # [L1+Lp, N, Nn, 6]
    return x, y, neighbor


def _freeze_for_finetune(model: nn.Module) -> Tuple[int, int]:
    """Freeze encoder + latent. Train only rnn_fy_init + dec (~271k params).

    Returns (total_params, trainable_params).
    """
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze decoder head only
    for p in model.rnn_fy_init.parameters():
        p.requires_grad = True
    for p in model.dec.parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    """pred: [n_samples, Lp, N, 2] OR [Lp, N, 2]; gt: [Lp, N, 2]. Returns minADE/minFDE."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    err = torch.linalg.norm(pred - gt.unsqueeze(0), dim=-1)   # [S, Lp, N]
    ade_per_sample = err.mean(dim=1)                          # [S, N]
    fde_per_sample = err[:, -1, :]                            # [S, N]
    ade = ade_per_sample.min(dim=0).values.mean()
    fde = fde_per_sample.min(dim=0).values.mean()
    return float(ade.item()), float(fde.item())


def _kl_weight(epoch: int, warmup: int, target: float) -> float:
    if warmup <= 0:
        return target
    return target * min(1.0, epoch / float(warmup))


def _custom_learn(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                  neighbor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate SocialVAE.learn() but also return predicted trajectory in world coords.

    Returns (err [Lp,N,2], kl [Lp,N,z_dim], pred_world [Lp,N,2]).
    """
    C = x.dim()
    if C < 3:
        x = x.unsqueeze(1)
        neighbor = neighbor.unsqueeze(1)
        y = y.unsqueeze(1)
    N = x.size(1)

    h, b = model.enc(x, neighbor, y=y)
    h = model.rnn_fy_init(h)
    h = h.view(N, -1, model.rnn_fy.num_layers)
    h = h.permute(2, 0, 1).contiguous()

    P, Q, D, Z = [], [], [], []
    for t in range(model.horizon):
        p_z = model.p_z(h[-1])
        q_z = model.q_z(h[-1], b[t])
        z = q_z.rsample()
        d = model.dec(z, h[-1])
        P.append(p_z); Q.append(q_z); D.append(d); Z.append(z)
        if t == model.horizon - 1:
            break
        zd = model.embed_zd(z, d)
        _, h = model.rnn_fy(zd.unsqueeze(0), h)

    d = torch.stack(D)                                       # [Lp, N, 2] velocity deltas
    y_local = y - x[-1, ..., :2].unsqueeze(0)
    pred_local = torch.cumsum(d, 0)                          # [Lp, N, 2]
    pred_world = pred_local + x[-1, ..., :2].unsqueeze(0)    # [Lp, N, 2]
    err = (pred_local - y_local).square()

    kl = []
    for p, q, z in zip(P, Q, Z):
        kl.append(q.log_prob(z) - p.log_prob(z))
    kl = torch.stack(kl)
    return err, kl, pred_world


def _vel_jerk_losses(pred_world: torch.Tensor, y: torch.Tensor, x_last: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute velocity-match and jerk-penalty losses.

    pred_world, y: [Lp, N, 2] (world coords).
    x_last:        [N, 2]     (last obs frame, world coords) — anchors first velocity step.
    """
    # Prepend the last observation so that the first predicted velocity is well-defined.
    pred_path = torch.cat([x_last.unsqueeze(0), pred_world], dim=0)  # [Lp+1, N, 2]
    gt_path = torch.cat([x_last.unsqueeze(0), y], dim=0)              # [Lp+1, N, 2]

    pred_vel = (pred_path[1:] - pred_path[:-1]) / max(1e-6, dt)
    gt_vel = (gt_path[1:] - gt_path[:-1]) / max(1e-6, dt)
    loss_vel = (pred_vel - gt_vel).pow(2).mean()

    if pred_vel.size(0) >= 3:
        pred_acc = (pred_vel[1:] - pred_vel[:-1]) / max(1e-6, dt)
        pred_jerk = (pred_acc[1:] - pred_acc[:-1]) / max(1e-6, dt)
        loss_jerk = pred_jerk.pow(2).mean()
    else:
        loss_jerk = torch.zeros((), device=pred_world.device)
    return loss_vel, loss_jerk


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             pred_samples: int = 20) -> Dict[str, float]:
    model.eval()
    ades, fdes, ns = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, neighbor in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            neighbor = neighbor.to(device).float()
            pred = model(x, neighbor, n_predictions=pred_samples)  # [S, Lp, N, 2]
            ade, fde = _ade_fde(pred, y)
            bsz = x.size(1)
            ades += ade * bsz
            fdes += fde * bsz
            ns += bsz
    return {"ade": ades / max(1, ns), "fde": fdes / max(1, ns)}


def main() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install torch.")
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Fine-tune SocialVAE on curated cases.")
    parser.add_argument("--social_vae_repo", required=True,
                        help="Path to SocialVAE repo (containing social_vae.py)")
    parser.add_argument("--social_vae_ckpt", required=True,
                        help="Path to pre-trained SocialVAE checkpoint dir (with ckpt-best inside)")
    parser.add_argument("--train_dataset", required=True,
                        help="Curated train_residual_cases.json")
    parser.add_argument("--val_dataset", required=True,
                        help="Curated benchmark_residual_cases.json (used as val/test)")
    parser.add_argument("--output_dir", default=str(repo_root / "models" / "social_vae_finetuned"))
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12,
                        help="SocialVAE PRED_HORIZON. Must match config used at training/inference.")
    parser.add_argument("--obs_dt", type=float, default=0.4)
    parser.add_argument("--ob_radius", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_neighbors", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=0.2)
    parser.add_argument("--kl_warmup_epochs", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--pred_samples_eval", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_vel", type=float, default=0.0,
                        help="Weight of velocity-match loss (MSE of pred vs gt velocity). Try 0.1-0.3 to stabilize.")
    parser.add_argument("--lambda_jerk", type=float, default=0.0,
                        help="Weight of jerk-smoothness penalty (mean ||jerk||^2 of pred). Try 0.01-0.05 to smooth.")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    SocialVAE = _import_social_vae(args.social_vae_repo)
    model = SocialVAE(horizon=args.pred_len, ob_radius=args.ob_radius, hidden_dim=args.hidden_dim).to(device)

    ckpt_dir = Path(args.social_vae_ckpt).expanduser().resolve()
    ckpt_best = ckpt_dir / "ckpt-best"
    ckpt_last = ckpt_dir / "ckpt-last"
    src = ckpt_best if ckpt_best.exists() else ckpt_last
    if not src.exists():
        raise SystemExit(f"No ckpt-best or ckpt-last under {ckpt_dir}")
    state = torch.load(str(src), map_location=device)
    model.load_state_dict(state["model"])
    print(f"[init] loaded SocialVAE weights from {src}")

    total, trainable = _freeze_for_finetune(model)
    print(f"[freeze] total params={total:,}  trainable={trainable:,}  "
          f"({100 * trainable / max(1, total):.1f}% of model)")

    train_ds = CuratedVAEDataset(args.train_dataset, obs_len=args.obs_len, pred_len=args.pred_len,
                                 obs_dt=args.obs_dt, max_neighbors=args.max_neighbors)
    val_ds = CuratedVAEDataset(args.val_dataset, obs_len=args.obs_len, pred_len=args.pred_len,
                               obs_dt=args.obs_dt, max_neighbors=args.max_neighbors)
    print(f"[data] train cases={len(train_ds)}  val cases={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    history: List[Dict[str, Any]] = []
    best_ade = float("inf")
    bad_epochs = 0
    best_ckpt_path = out_dir / "social_vae_finetuned.pt"

    use_aux = float(args.lambda_vel) > 0.0 or float(args.lambda_jerk) > 0.0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        kl_w = _kl_weight(epoch, args.kl_warmup_epochs, args.kl_weight)
        n_b = 0
        sum_loss = sum_rec = sum_kl = sum_vel = sum_jerk = 0.0
        for x, y, neighbor in train_dl:
            x = x.to(device).float()
            y = y.to(device).float()
            neighbor = neighbor.to(device).float()
            if use_aux:
                err, kl, pred_world = _custom_learn(model, x, y, neighbor)
                rec = err.mean()
                klm = kl.mean()
                # x has shape [L1, N, 6]; last obs position is x[-1, :, :2]
                loss_vel, loss_jerk = _vel_jerk_losses(
                    pred_world, y, x[-1, ..., :2], float(args.obs_dt))
                loss = rec + kl_w * klm + float(args.lambda_vel) * loss_vel + float(args.lambda_jerk) * loss_jerk
                sum_vel += float(loss_vel.item())
                sum_jerk += float(loss_jerk.item())
            else:
                err, kl = model(x=x, y=y, neighbor=neighbor)
                rec = err.mean()
                klm = kl.mean()
                loss = rec + kl_w * klm
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            sum_loss += float(loss.item())
            sum_rec += float(rec.item())
            sum_kl += float(klm.item())
            n_b += 1
        train_loss = sum_loss / max(1, n_b)
        train_rec = sum_rec / max(1, n_b)
        train_kl = sum_kl / max(1, n_b)
        train_vel = sum_vel / max(1, n_b)
        train_jerk = sum_jerk / max(1, n_b)

        val_stats = evaluate(model, val_dl, device, pred_samples=int(args.pred_samples_eval))
        row = {
            "epoch": epoch,
            "kl_weight": kl_w,
            "train_loss": train_loss,
            "train_rec": train_rec,
            "train_kl": train_kl,
            "train_vel": train_vel,
            "train_jerk": train_jerk,
            "val_ade": val_stats["ade"],
            "val_fde": val_stats["fde"],
        }
        history.append(row)
        aux_str = f" vel={train_vel:.4f} jerk={train_jerk:.4f}" if use_aux else ""
        print(f"epoch {epoch:02d}/{args.epochs} | kl_w={kl_w:.2f} | "
              f"train loss={train_loss:.4f} rec={train_rec:.4f} kl={train_kl:.4f}{aux_str} | "
              f"val ADE={val_stats['ade']:.4f} FDE={val_stats['fde']:.4f}")

        if val_stats["ade"] < best_ade - 1e-4:
            best_ade = val_stats["ade"]
            bad_epochs = 0
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_ade": best_ade,
                        "val_fde": val_stats["fde"],
                        "args": vars(args)},
                       best_ckpt_path)
            print(f"  ↳ saved best to {best_ckpt_path} (val_ade={best_ade:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.early_stop_patience):
                print(f"  ↳ early stop: no improvement for {bad_epochs} epochs")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done] best val ADE={best_ade:.4f}; checkpoint={best_ckpt_path}")


if __name__ == "__main__":
    main()
