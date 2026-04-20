#!/usr/bin/env python3
"""Unified people predictor: switch between Kalman, Social GRU and SocialVAE."""
from __future__ import annotations

import math
import struct
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rclpy
import tf2_ros
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from people_msgs.msg import People, Person
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from predictive_nav_mppi.kf_cv import (
    TrackState,
    clamp_dt,
    predict_state_cov,
    prune_stale_tracks,
    update_state_cov,
)
from predictive_nav_mppi.models.kalman_residual_net import compute_turn_gate
from predictive_nav_mppi.models.kalman_residual_net import load_checkpoint as load_residual_checkpoint
from predictive_nav_mppi.models.kalman_residual_net import predict_residual_world
from predictive_nav_mppi.scene_context import ScenePatchConfig

# (mu, sigma or None) per step; sigma is 4x4 or None for model backend
HorizonStep = Tuple[List[float], Optional[List[List[float]]]]


def _float_or_default(value: float, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    return float(value)


def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _cov2d_to_ellipse(
    cov_xx: float,
    cov_xy: float,
    cov_yy: float,
    n_sigma: float = 2.0,
) -> Tuple[float, float, float]:
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    tmp = max(0.0, trace * trace * 0.25 - det)
    root = math.sqrt(tmp)
    l1 = max(0.0, 0.5 * trace + root)
    l2 = max(0.0, 0.5 * trace - root)
    yaw = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    sx = max(1e-3, 2.0 * n_sigma * math.sqrt(l1))
    sy = max(1e-3, 2.0 * n_sigma * math.sqrt(l2))
    return sx, sy, yaw


# --- Kalman backend (same logic as people_kf_predictor) ---


class _KalmanBackend:
    def __init__(self, node: "PeoplePredictor") -> None:
        self.node = node
        self.tracks: Dict[int, TrackState] = {}
        self.name_to_track_id: Dict[str, int] = {}
        self.next_track_id = 1

    def _init_track(self, now_sec: float, px: float, py: float, vx: float, vy: float) -> TrackState:
        sp2 = self.node.sigma_p0 ** 2
        sv2 = self.node.sigma_v0 ** 2
        sigma = [
            [sp2, 0.0, 0.0, 0.0],
            [0.0, sp2, 0.0, 0.0],
            [0.0, 0.0, sv2, 0.0],
            [0.0, 0.0, 0.0, sv2],
        ]
        return TrackState(mu=[px, py, vx, vy], sigma=sigma, last_update_sec=now_sec)

    def _person_id(self, person: Any) -> int:
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))
        name = str(getattr(person, "name", "")) or f"anon_{self.next_track_id}"
        # Robot injected by predict_robot_as_agent gets name "robot_0" → fixed ID 0.
        if name.startswith("robot_0"):
            return 0
        if name not in self.name_to_track_id:
            self.name_to_track_id[name] = self.next_track_id
            self.next_track_id += 1
        return self.name_to_track_id[name]

    def update(self, msg: People, now_sec: float) -> None:
        n = min(len(msg.people), max(0, self.node.max_people))
        for i in range(n):
            p = msg.people[i]
            pid = self._person_id(p)
            px = _float_or_default(p.position.x, 0.0)
            py = _float_or_default(p.position.y, 0.0)
            vx = _float_or_default(p.velocity.x, 0.0)
            vy = _float_or_default(p.velocity.y, 0.0)
            if pid not in self.tracks:
                self.tracks[pid] = self._init_track(now_sec, px, py, vx, vy)
                continue
            tr = self.tracks[pid]
            dt = clamp_dt(now_sec - tr.last_update_sec, self.node.min_dt, self.node.max_dt)
            mu_p, sig_p = predict_state_cov(tr.mu, tr.sigma, dt, self.node.sigma_acc)
            mu_u, sig_u = update_state_cov(mu_p, sig_p, px, py, self.node.sigma_meas)
            tr.mu, tr.sigma, tr.last_update_sec = mu_u, sig_u, now_sec
        prune_stale_tracks(self.tracks, now_sec, self.node.track_timeout)

    def get_horizons(self, now_sec: float) -> Dict[int, List[HorizonStep]]:
        out: Dict[int, List[HorizonStep]] = {}
        for pid, tr in list(self.tracks.items()):
            dt = now_sec - tr.last_update_sec
            if dt > 0:
                dt = clamp_dt(dt, self.node.min_dt, self.node.max_dt)
                tr.mu, tr.sigma = predict_state_cov(tr.mu, tr.sigma, dt, self.node.sigma_acc)
                tr.last_update_sec = now_sec
            mu_h, sig_h = list(tr.mu), [r[:] for r in tr.sigma]
            horizon = []
            for _ in range(self.node.pred_steps):
                mu_h, sig_h = predict_state_cov(mu_h, sig_h, self.node.pred_dt, self.node.sigma_acc)
                horizon.append((list(mu_h), [r[:] for r in sig_h]))
            out[pid] = horizon
        return out


# --- Social GRU backend (history + learned model) ---


class _ModelBackend:
    def __init__(self, node: "PeoplePredictor") -> None:
        self.node = node
        self.tracks_xy: Dict[int, deque] = {}
        self.tracks_velocity: Dict[int, Tuple[float, float]] = {}
        self.name_to_track_id: Dict[str, int] = {}
        self.next_track_id = 1
        self.last_update_sec: Dict[int, float] = {}
        self._model = None
        self._model_cfg = {}
        self._load_model()

    def _load_model(self) -> None:
        path = self.node.model_weights_path
        if not path:
            self.node.get_logger().warn("model backend: model_weights_path is empty, predictor will not run")
            return
        try:
            from predictive_nav_mppi.models.social_gru import load_checkpoint, SocialGRUPredictor
            self._model, self._model_cfg = load_checkpoint(path, device=self.node.model_device)
            self.node.get_logger().info(f"Loaded model from {path} (obs_len={self._model.obs_len}, pred_len={self._model.pred_len})")
        except Exception as e:
            self.node.get_logger().error(f"Failed to load model from {path}: {e}")
            self._model = None

    def _person_id(self, person: Any) -> int:
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))
        name = str(getattr(person, "name", "")) or f"anon_{self.next_track_id}"
        if name.startswith("robot_0"):
            return 0
        if name not in self.name_to_track_id:
            self.name_to_track_id[name] = self.next_track_id
            self.next_track_id += 1
        return self.name_to_track_id[name]

    def update(self, msg: People, now_sec: float) -> None:
        obs_len = self.node.model_obs_len
        n = min(len(msg.people), max(0, self.node.max_people))
        for i in range(n):
            p = msg.people[i]
            pid = self._person_id(p)
            px = _float_or_default(p.position.x, 0.0)
            py = _float_or_default(p.position.y, 0.0)
            vx = _float_or_default(getattr(p, "velocity", None) and getattr(p.velocity, "x", None), 0.0)
            vy = _float_or_default(getattr(p, "velocity", None) and getattr(p.velocity, "y", None), 0.0)
            if pid not in self.tracks_xy:
                # Store (t, x, y) for temporal subsampling to match TrajNet obs_dt (~0.4 s)
                self.tracks_xy[pid] = deque(maxlen=64)
                self.last_update_sec[pid] = now_sec
            self.tracks_xy[pid].append((now_sec, px, py))
            self.tracks_velocity[pid] = (vx, vy)
            self.last_update_sec[pid] = now_sec
        timeout = self.node.track_timeout
        stale = [pid for pid, t in self.last_update_sec.items() if now_sec - t > timeout]
        for pid in stale:
            self.tracks_xy.pop(pid, None)
            self.tracks_velocity.pop(pid, None)
            self.last_update_sec.pop(pid, None)

    def _sample_obs_at_dt(
        self, history: deque, obs_len: int, obs_dt: float, now_sec: float
    ) -> Optional[np.ndarray]:
        """Sample obs_len points at times now_sec, now_sec - obs_dt, ..., now_sec - (obs_len-1)*obs_dt.
        history contains (t, x, y). Returns [obs_len, 2] or None if not enough data."""
        if len(history) < 1:
            return None
        times = np.array([h[0] for h in history], dtype=np.float64)
        xy = np.array([[h[1], h[2]] for h in history], dtype=np.float64)
        # target times: newest first, then older
        t_end = min(now_sec, float(times[-1]))
        target_t = np.array(
            [t_end - (obs_len - 1 - i) * obs_dt for i in range(obs_len)],
            dtype=np.float64,
        )
        out = np.zeros((obs_len, 2), dtype=np.float64)
        for i in range(obs_len):
            t = target_t[i]
            if t > times[-1]:
                out[i] = xy[-1]
            elif t < times[0]:
                out[i] = xy[0]
            else:
                idx = np.searchsorted(times, t, side="left")
                if idx == 0:
                    out[i] = xy[0]
                elif idx >= len(times):
                    out[i] = xy[-1]
                else:
                    a = (t - times[idx - 1]) / max(1e-9, times[idx] - times[idx - 1])
                    out[i] = (1.0 - a) * xy[idx - 1] + a * xy[idx]
        return out

    def get_horizons(self, now_sec: float) -> Dict[int, List[HorizonStep]]:
        if self._model is None:
            return {}
        obs_len = self.node.model_obs_len
        pred_len = self.node.pred_steps
        obs_dt = max(1e-6, getattr(self.node, "model_obs_dt", 0.4))
        timeout = self.node.track_timeout
        tracks_obs = {}
        for pid, q in list(self.tracks_xy.items()):
            if now_sec - self.last_update_sec.get(pid, 0) > timeout:
                continue
            if len(q) < 1:
                continue
            arr = self._sample_obs_at_dt(q, obs_len, obs_dt, now_sec)
            if arr is None:
                continue
            tracks_obs[pid] = arr
        if not tracks_obs:
            return {}

        from predictive_nav_mppi.models.social_gru import predict_trajectories_world

        pids = list(tracks_obs.keys())
        tracks_obs_xy = [tracks_obs[pid] for pid in pids]
        tracks_velocity_xy: List[Optional[Tuple[float, float]]] = [
            self.tracks_velocity.get(pid) for pid in pids
        ]
        tracks_neigh_xy: List[List[np.ndarray]] = []
        for pid in pids:
            others = [tracks_obs[k] for k in pids if k != pid]
            tracks_neigh_xy.append(others)

        try:
            preds = predict_trajectories_world(
                self._model,
                tracks_obs_xy,
                tracks_neigh_xy,
                device=self.node.model_device,
                tracks_velocity_xy=tracks_velocity_xy,
                flip_forward_axis=getattr(self.node, "model_flip_forward_axis", False),
            )
        except Exception as e:
            self.node.get_logger().warn(f"Model inference failed: {e}")
            return {}

        steps_use = min(
            pred_len,
            getattr(self.node, "model_pred_steps_use", 12),
            max(1, preds[0].shape[0] if preds else 1),
        )
        out: Dict[int, List[HorizonStep]] = {}
        for idx, pid in enumerate(pids):
            if idx >= len(preds):
                break
            traj = preds[idx]
            horizon = []
            for k in range(min(steps_use, traj.shape[0])):
                x, y = float(traj[k, 0]), float(traj[k, 1])
                horizon.append(([x, y, 0.0, 0.0], None))
            out[pid] = horizon
        return out


# --- SocialVAE backend (external repo + multi-sample predictions) ---


class _SocialVAEBackend:
    def __init__(self, node: "PeoplePredictor") -> None:
        self.node = node
        self.tracks_xy: Dict[int, deque] = {}
        self.name_to_track_id: Dict[str, int] = {}
        self.next_track_id = 1
        self.last_update_sec: Dict[int, float] = {}
        self._model = None
        self._model_meta: Dict[str, Any] = {}
        self._load_model()

    def _person_id(self, person: Any) -> int:
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))
        name = str(getattr(person, "name", "")) or f"anon_{self.next_track_id}"
        if name.startswith("robot_0"):
            return 0
        if name not in self.name_to_track_id:
            self.name_to_track_id[name] = self.next_track_id
            self.next_track_id += 1
        return self.name_to_track_id[name]

    def _load_model(self) -> None:
        repo_path = self.node.social_vae_repo_path
        ckpt_path = self.node.social_vae_ckpt_path
        if not repo_path:
            self.node.get_logger().warn(
                "social_vae backend: social_vae_repo_path is empty, predictor will not run"
            )
            return
        if not ckpt_path:
            self.node.get_logger().warn(
                "social_vae backend: social_vae_ckpt_path is empty, predictor will not run"
            )
            return
        try:
            from predictive_nav_mppi.models.social_vae import load_external_social_vae

            self._model, self._model_meta = load_external_social_vae(
                repo_path=repo_path,
                ckpt_path=ckpt_path,
                device=self.node.social_vae_device,
                config_path=self.node.social_vae_config_path,
                ob_horizon=self.node.social_vae_ob_horizon,
                pred_horizon=self.node.social_vae_pred_horizon,
                ob_radius=self.node.social_vae_ob_radius,
                hidden_dim=self.node.social_vae_hidden_dim,
            )
            self.node.get_logger().info(
                "Loaded SocialVAE from "
                f"{ckpt_path} (obs={self._model_meta.get('ob_horizon')}, "
                f"pred={self._model_meta.get('pred_horizon')}, "
                f"radius={self._model_meta.get('ob_radius')})"
            )
        except Exception as e:
            self.node.get_logger().error(f"Failed to load SocialVAE: {e}")
            self._model = None
            self._model_meta = {}

    def update(self, msg: People, now_sec: float) -> None:
        n = min(len(msg.people), max(0, self.node.max_people))
        for i in range(n):
            p = msg.people[i]
            pid = self._person_id(p)
            px = _float_or_default(p.position.x, 0.0)
            py = _float_or_default(p.position.y, 0.0)
            if pid not in self.tracks_xy:
                self.tracks_xy[pid] = deque(maxlen=96)
                self.last_update_sec[pid] = now_sec
            self.tracks_xy[pid].append((now_sec, px, py))
            self.last_update_sec[pid] = now_sec
        timeout = self.node.track_timeout
        stale = [pid for pid, t in self.last_update_sec.items() if now_sec - t > timeout]
        for pid in stale:
            self.tracks_xy.pop(pid, None)
            self.last_update_sec.pop(pid, None)

    def _sample_obs_at_dt(
        self, history: deque, obs_len: int, obs_dt: float, now_sec: float
    ) -> Optional[np.ndarray]:
        if len(history) < 1:
            return None
        times = np.array([h[0] for h in history], dtype=np.float64)
        xy = np.array([[h[1], h[2]] for h in history], dtype=np.float64)
        t_end = min(now_sec, float(times[-1]))
        target_t = np.array(
            [t_end - (obs_len - 1 - i) * obs_dt for i in range(obs_len)],
            dtype=np.float64,
        )
        out = np.zeros((obs_len, 2), dtype=np.float64)
        for i in range(obs_len):
            t = target_t[i]
            if t > times[-1]:
                out[i] = xy[-1]
            elif t < times[0]:
                out[i] = xy[0]
            else:
                idx = np.searchsorted(times, t, side="left")
                if idx == 0:
                    out[i] = xy[0]
                elif idx >= len(times):
                    out[i] = xy[-1]
                else:
                    a = (t - times[idx - 1]) / max(1e-9, times[idx] - times[idx - 1])
                    out[i] = (1.0 - a) * xy[idx - 1] + a * xy[idx]
        return out


class _ResidualBackend:
    def __init__(self, node: "PeoplePredictor") -> None:
        self.node = node
        self._kalman = _KalmanBackend(node)
        self.tracks_xy: Dict[int, deque] = {}
        self.tracks_velocity: Dict[int, Tuple[float, float]] = {}
        self.name_to_track_id: Dict[str, int] = {}
        self.next_track_id = 1
        self.last_update_sec: Dict[int, float] = {}
        self._smoothed_residual_world: Dict[int, np.ndarray] = {}
        self._model = None
        self._k_neighbors = 3
        self._pred_len = 26
        self._scene_patch_cfg = ScenePatchConfig(
            size_m=float(self.node.scene_patch_size_m),
            pixels=int(self.node.scene_patch_pixels),
            align_to_heading=bool(self.node.scene_patch_align_to_heading),
        )
        self._load_model()

    def _person_id(self, person: Any) -> int:
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))
        name = str(getattr(person, "name", "")) or f"anon_{self.next_track_id}"
        if name.startswith("robot_0"):
            return 0
        if name not in self.name_to_track_id:
            self.name_to_track_id[name] = self.next_track_id
            self.next_track_id += 1
        return self.name_to_track_id[name]

    def _load_model(self) -> None:
        path = self.node.residual_model_weights
        if not path:
            self.node.get_logger().warn("residual backend: residual_model_weights is empty, predictor will not run")
            return
        try:
            self._model, cfg = load_residual_checkpoint(path, device=self.node.residual_model_device)
            self._k_neighbors = int(cfg.get("k_neighbors", 3))
            self._pred_len = int(cfg.get("pred_len", 26))
            self.node.get_logger().info(
                f"Loaded residual model from {path} "
                f"(obs_len={cfg.get('obs_len')}, pred_len={self._pred_len}, k_neighbors={self._k_neighbors})"
            )
        except Exception as e:
            self.node.get_logger().error(f"Failed to load residual model from {path}: {e}")
            self._model = None

    def update(self, msg: People, now_sec: float) -> None:
        self._kalman.update(msg, now_sec)
        n = min(len(msg.people), max(0, self.node.max_people))
        for i in range(n):
            p = msg.people[i]
            pid = self._person_id(p)
            px = _float_or_default(p.position.x, 0.0)
            py = _float_or_default(p.position.y, 0.0)
            vx = _float_or_default(getattr(p, "velocity", None) and getattr(p.velocity, "x", None), 0.0)
            vy = _float_or_default(getattr(p, "velocity", None) and getattr(p.velocity, "y", None), 0.0)
            if pid not in self.tracks_xy:
                self.tracks_xy[pid] = deque(maxlen=64)
            self.tracks_xy[pid].append((now_sec, px, py))
            self.tracks_velocity[pid] = (vx, vy)
            self.last_update_sec[pid] = now_sec
        timeout = self.node.track_timeout
        stale = [pid for pid, t in self.last_update_sec.items() if now_sec - t > timeout]
        for pid in stale:
            self.tracks_xy.pop(pid, None)
            self.tracks_velocity.pop(pid, None)
            self.last_update_sec.pop(pid, None)
            self._smoothed_residual_world.pop(pid, None)

    def _sample_obs_at_dt(self, history: deque, obs_len: int, obs_dt: float, now_sec: float) -> Optional[np.ndarray]:
        if len(history) < 1:
            return None
        times = np.array([h[0] for h in history], dtype=np.float64)
        xy = np.array([[h[1], h[2]] for h in history], dtype=np.float64)
        t_end = min(now_sec, float(times[-1]))
        target_t = np.array([t_end - (obs_len - 1 - i) * obs_dt for i in range(obs_len)], dtype=np.float64)
        out = np.zeros((obs_len, 2), dtype=np.float64)
        for i, t in enumerate(target_t):
            if t > times[-1]:
                out[i] = xy[-1]
            elif t < times[0]:
                out[i] = xy[0]
            else:
                idx = np.searchsorted(times, t, side="left")
                if idx == 0:
                    out[i] = xy[0]
                elif idx >= len(times):
                    out[i] = xy[-1]
                else:
                    a = (t - times[idx - 1]) / max(1e-9, times[idx] - times[idx - 1])
                    out[i] = (1.0 - a) * xy[idx - 1] + a * xy[idx]
        return out

    def get_horizons(self, now_sec: float) -> Dict[int, List[HorizonStep]]:
        if self._model is None:
            return {}
        kalman_horizons = self._kalman.get_horizons(now_sec)
        obs_len = self.node.model_obs_len
        obs_dt = max(1e-6, getattr(self.node, "model_obs_dt", 0.4))
        timeout = self.node.track_timeout
        tracks_obs: Dict[int, np.ndarray] = {}
        for pid, q in list(self.tracks_xy.items()):
            if now_sec - self.last_update_sec.get(pid, 0.0) > timeout:
                continue
            arr = self._sample_obs_at_dt(q, obs_len, obs_dt, now_sec)
            if arr is not None:
                tracks_obs[pid] = arr
        out: Dict[int, List[HorizonStep]] = {}
        pids = list(tracks_obs.keys())
        for pid in pids:
            kalman_h = kalman_horizons.get(pid)
            if not kalman_h:
                continue
            kalman_pred_xy = np.asarray([[step[0][0], step[0][1]] for step in kalman_h[: self._pred_len]], dtype=np.float32)
            if kalman_pred_xy.shape[0] < self._pred_len:
                continue
            neigh_xy = [tracks_obs[oid] for oid in pids if oid != pid]
            try:
                pred_world = predict_residual_world(
                    model=self._model,
                    obs_xy=tracks_obs[pid].astype(np.float32),
                    neigh_xy=[n.astype(np.float32) for n in neigh_xy],
                    kalman_pred_xy=kalman_pred_xy,
                    obs_dt=obs_dt,
                    device=self.node.residual_model_device,
                    k_neighbors=self._k_neighbors,
                    scene_patch_cfg=self._scene_patch_cfg,
                    scene_map_yaml=self.node.scene_map_yaml,
                    residual_alpha=self._effective_residual_alpha(tracks_obs[pid]),
                )
            except Exception as e:
                self.node.get_logger().warn(f"Residual inference failed: {e}")
                continue
            pred_world = self._postprocess_prediction(pid, kalman_pred_xy, pred_world)
            steps_use = min(int(self.node.pred_steps), pred_world.shape[0])
            horizon: List[HorizonStep] = []
            for k in range(steps_use):
                x, y = float(pred_world[k, 0]), float(pred_world[k, 1])
                horizon.append(([x, y, 0.0, 0.0], None))
            out[pid] = horizon
        return out

    def _effective_residual_alpha(self, obs_xy: np.ndarray) -> float:
        alpha = float(max(0.0, self.node.residual_alpha))
        if not getattr(self.node, "residual_turn_gate_enable", True):
            return alpha
        gate = compute_turn_gate(
            obs_xy=obs_xy,
            tau=float(self.node.residual_turn_gate_tau),
            alpha=float(self.node.residual_turn_gate_alpha),
        )
        return alpha * gate

    def _postprocess_prediction(
        self,
        pid: int,
        kalman_pred_xy: np.ndarray,
        pred_world: np.ndarray,
    ) -> np.ndarray:
        correction = np.asarray(pred_world, dtype=np.float32) - np.asarray(kalman_pred_xy, dtype=np.float32)

        clip_norm = float(max(0.0, getattr(self.node, "residual_clip_norm", 0.0)))
        if clip_norm > 0.0 and correction.size:
            norms = np.linalg.norm(correction, axis=1, keepdims=True)
            scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-6))
            correction = correction * scale

        beta = float(min(0.999, max(0.0, getattr(self.node, "residual_smoothing_beta", 0.0))))
        if beta > 0.0:
            prev = self._smoothed_residual_world.get(pid)
            if prev is not None and prev.shape == correction.shape:
                correction = beta * prev + (1.0 - beta) * correction
            self._smoothed_residual_world[pid] = correction.copy()
        else:
            self._smoothed_residual_world.pop(pid, None)

        return np.asarray(kalman_pred_xy, dtype=np.float32) + correction


# --- Unified node ---


class PeoplePredictor(Node):
    def __init__(self) -> None:
        super().__init__("people_predictor")

        self.input_topic = self.declare_parameter("input_topic", "/people").value
        self.output_cloud_topic = self.declare_parameter("output_cloud_topic", "/predicted_people_cloud").value
        self.output_markers_topic = self.declare_parameter("output_markers_topic", "/predicted_people_markers").value
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 10.0).value)
        self.pred_dt = float(self.declare_parameter("pred_dt", 0.1).value)
        self.pred_steps = int(self.declare_parameter("pred_steps", 10).value)
        self.track_timeout = float(self.declare_parameter("track_timeout", 1.0).value)
        self.max_people = int(self.declare_parameter("max_people", 100).value)
        self.publish_markers = bool(self.declare_parameter("publish_markers", True).value)
        self.publish_ellipses = bool(self.declare_parameter("publish_ellipses", True).value)
        self.ellipse_steps = int(self.declare_parameter("ellipse_steps", 3).value)
        self.frame_id_override = str(self.declare_parameter("frame_id_override", "").value)

        # Predictor type: "kalman" | "model"/"social_gru" | "social_vae"/"socialvae" | "residual"
        predictor_type_raw = str(self.declare_parameter("predictor_type", "kalman").value).lower().strip()
        type_aliases = {
            "kalman": "kalman",
            "model": "model",
            "social_gru": "model",
            "social_vae": "social_vae",
            "socialvae": "social_vae",
            "residual": "residual",
        }
        self.predictor_type = type_aliases.get(predictor_type_raw, "")
        if not self.predictor_type:
            self.get_logger().warn(f"predictor_type '{predictor_type_raw}' unknown, using 'kalman'")
            self.predictor_type = "kalman"

        # Kalman-only params
        self.sigma_meas = float(self.declare_parameter("sigma_meas", 0.08).value)
        self.sigma_acc = float(self.declare_parameter("sigma_acc", 0.06).value)
        self.sigma_p0 = float(self.declare_parameter("sigma_p0", 0.06).value)
        self.sigma_v0 = float(self.declare_parameter("sigma_v0", 0.8).value)
        self.min_dt = float(self.declare_parameter("min_dt", 0.02).value)
        self.max_dt = float(self.declare_parameter("max_dt", 0.3).value)

        # Model-only params
        self.model_weights_path = str(self.declare_parameter("model_weights_path", "").value).strip()
        self.model_obs_len = int(self.declare_parameter("model_obs_len", 8).value)
        self.model_device = str(self.declare_parameter("model_device", "").value).strip() or ("cuda" if self._torch_cuda() else "cpu")
        # Эллипсы для cost: у модели нет ковариации; задаём радиус неопределённости (м),
        # чтобы MPPI учитывал зону в chance-constraint. Совпадает с dyn_people_default_std в контроллере.
        self.model_ellipse_std = float(self.declare_parameter("model_ellipse_std", 0.08).value)
        self.max_ellipse_scale = float(self.declare_parameter("max_ellipse_scale", 0.25).value)
        _flip = self.declare_parameter("model_flip_forward_axis", False).value
        self.model_flip_forward_axis = (
            _flip if isinstance(_flip, bool) else str(_flip).lower() not in ("false", "0", "no", "")
        )
        # TrajNet: ~2.5 fps → ~0.4 s between frames. Match so model sees same temporal scale as in training.
        self.model_obs_dt = float(self.declare_parameter("model_obs_dt", 0.4).value)
        # Use only first N steps of model output — shorter, less "straight into the distance" predictions.
        self.model_pred_steps_use = int(self.declare_parameter("model_pred_steps_use", 5).value)
        self.residual_model_weights = str(self.declare_parameter("residual_model_weights", "").value).strip()
        self.residual_model_device = str(self.declare_parameter("residual_model_device", "").value).strip() or self.model_device
        self.residual_alpha = float(self.declare_parameter("residual_alpha", 0.3).value)
        self.residual_smoothing_beta = float(self.declare_parameter("residual_smoothing_beta", 0.8).value)
        self.residual_clip_norm = float(self.declare_parameter("residual_clip_norm", 0.35).value)
        self.residual_turn_gate_enable = bool(self.declare_parameter("residual_turn_gate_enable", True).value)
        self.residual_turn_gate_tau = float(self.declare_parameter("residual_turn_gate_tau", 0.1).value)
        self.residual_turn_gate_alpha = float(self.declare_parameter("residual_turn_gate_alpha", 30.0).value)
        self.scene_map_yaml = str(self.declare_parameter("scene_map_yaml", "").value).strip()
        self.scene_patch_size_m = float(self.declare_parameter("scene_patch_size_m", 6.0).value)
        self.scene_patch_pixels = int(self.declare_parameter("scene_patch_pixels", 32).value)
        self.scene_patch_align_to_heading = bool(self.declare_parameter("scene_patch_align_to_heading", True).value)

        # SocialVAE-only params
        self.social_vae_repo_path = str(self.declare_parameter("social_vae_repo_path", "").value).strip()
        self.social_vae_ckpt_path = str(self.declare_parameter("social_vae_ckpt_path", "").value).strip()
        self.social_vae_config_path = str(self.declare_parameter("social_vae_config_path", "").value).strip()
        self.social_vae_ob_horizon = int(self.declare_parameter("social_vae_ob_horizon", 8).value)
        self.social_vae_pred_horizon = int(self.declare_parameter("social_vae_pred_horizon", 12).value)
        self.social_vae_ob_radius = float(self.declare_parameter("social_vae_ob_radius", 2.0).value)
        self.social_vae_hidden_dim = int(self.declare_parameter("social_vae_hidden_dim", 256).value)
        self.social_vae_obs_dt = float(self.declare_parameter("social_vae_obs_dt", 0.4).value)
        self.social_vae_pred_steps_use = int(self.declare_parameter("social_vae_pred_steps_use", 26).value)
        self.social_vae_pred_samples = int(self.declare_parameter("social_vae_pred_samples", 20).value)
        self.social_vae_max_neighbors = int(self.declare_parameter("social_vae_max_neighbors", 16).value)
        self.social_vae_neighbor_pad = float(self.declare_parameter("social_vae_neighbor_pad", 1e9).value)
        self.social_vae_cov_std_floor = float(self.declare_parameter("social_vae_cov_std_floor", 0.08).value)
        self.social_vae_device = str(self.declare_parameter("social_vae_device", "").value).strip() or self.model_device

        # Robot-as-agent injection
        self.predict_robot_as_agent = bool(
            self.declare_parameter("predict_robot_as_agent", False).value)
        self.robot_frame = str(
            self.declare_parameter("robot_frame", "base_footprint").value)
        self.global_frame = str(
            self.declare_parameter("global_frame", "map").value)
        # Special fake person ID for the robot (excluded from output predictions).
        self._ROBOT_FAKE_PID = 0
        self._robot_prev_xy: Optional[Tuple[float, float]] = None
        self._robot_prev_t: float = 0.0
        if self.predict_robot_as_agent:
            self._tf_buf = tf2_ros.Buffer()
            self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)
            self.get_logger().info(
                f"predict_robot_as_agent=True: robot ({self.robot_frame}) "
                "will be injected as agent 0 into predictor")

        self.latest_frame_id = "map"
        self._logged_frame_id = False

        if self.predictor_type == "kalman":
            self._backend = _KalmanBackend(self)
        elif self.predictor_type == "social_vae":
            self._backend = _SocialVAEBackend(self)
        elif self.predictor_type == "residual":
            self._backend = _ResidualBackend(self)
        else:
            self._backend = _ModelBackend(self)

        self.sub_people = self.create_subscription(People, self.input_topic, self._people_cb, 10)
        self.pub_cloud = self.create_publisher(PointCloud2, self.output_cloud_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_markers_topic, 10)
        timer_period = max(0.01, 1.0 / max(1e-3, self.publish_rate_hz))
        self.timer = self.create_timer(timer_period, self._publish_prediction)

        self.get_logger().info(
            f"people_predictor started: type={self.predictor_type}, in={self.input_topic}, "
            f"cloud={self.output_cloud_topic}, markers={self.output_markers_topic}"
        )

    def _torch_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _get_robot_person(self, now_sec: float) -> Optional[Person]:
        """Look up robot TF and return it as a fake Person, or None on failure."""
        try:
            tf = self._tf_buf.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time())
            rx = tf.transform.translation.x
            ry = tf.transform.translation.y
        except Exception:
            return None

        # Estimate velocity from position difference
        dt = now_sec - self._robot_prev_t
        if self._robot_prev_xy is not None and 0.01 < dt < 1.0:
            vx = (rx - self._robot_prev_xy[0]) / dt
            vy = (ry - self._robot_prev_xy[1]) / dt
        else:
            vx, vy = 0.0, 0.0
        self._robot_prev_xy = (rx, ry)
        self._robot_prev_t = now_sec

        p = Person()
        p.name = f"robot_{self._ROBOT_FAKE_PID}"
        p.position.x = rx
        p.position.y = ry
        p.position.z = 0.0
        p.velocity.x = vx
        p.velocity.y = vy
        p.velocity.z = 0.0
        p.reliability = 1.0
        return p

    def _people_cb(self, msg: People) -> None:
        if self.frame_id_override:
            self.latest_frame_id = self.frame_id_override
        elif msg.header.frame_id:
            self.latest_frame_id = msg.header.frame_id
        if not self._logged_frame_id:
            self._logged_frame_id = True
            self.get_logger().info(
                f"People/predictions frame_id: {self.latest_frame_id!r} "
                "(set frame_id_override in config if RViz Fixed Frame differs)"
            )
        now_sec = self._now_sec()
        if self.predict_robot_as_agent:
            robot_person = self._get_robot_person(now_sec)
            if robot_person is not None:
                # Inject robot as first person so backends treat it as a neighbour.
                # Make a shallow copy to avoid mutating the original message.
                from copy import copy
                msg_with_robot = copy(msg)
                msg_with_robot.people = [robot_person] + list(msg.people)
                self._backend.update(msg_with_robot, now_sec)
                return
        self._backend.update(msg, now_sec)

    def _create_cloud_xyz32(self, header: Header, points_xyz: List[Tuple[float, float, float]]) -> PointCloud2:
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points_xyz)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        if points_xyz:
            data = bytearray()
            for x, y, z in points_xyz:
                data.extend(struct.pack("<fff", float(x), float(y), float(z)))
            msg.data = bytes(data)
        else:
            msg.data = b""
        return msg

    def _build_markers(
        self,
        header: Header,
        horizon_by_id: Dict[int, List[HorizonStep]],
    ) -> MarkerArray:
        out = MarkerArray()
        delete_all = Marker()
        delete_all.header = header
        delete_all.action = Marker.DELETEALL
        out.markers.append(delete_all)
        if not self.publish_markers:
            return out

        lifetime = Duration(sec=0, nanosec=int(2e8))
        ellipse_max = max(0, min(self.ellipse_steps, self.pred_steps))

        for person_id, horizon in horizon_by_id.items():
            line = Marker()
            line.header = header
            line.ns = "predicted_people_path"
            line.id = int(person_id)
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.035
            line.color.r = 0.1
            line.color.g = 0.8
            line.color.b = 0.2
            line.color.a = 0.95
            line.lifetime = lifetime
            for mu_t, _ in horizon:
                p = Point()
                p.x = float(mu_t[0])
                p.y = float(mu_t[1])
                p.z = 0.05
                line.points.append(p)
            out.markers.append(line)

            if not self.publish_ellipses:
                continue
            # n_sigma в контроллере (dyn_people_cov_n_sigma) обычно 2.0 → scale = 2*n_sigma*std
            n_sigma_vis = 2.0
            cap = max(1e-3, getattr(self, "max_ellipse_scale", 0.25))
            for step_idx in range(ellipse_max):
                if step_idx >= len(horizon):
                    break
                mu_t, sigma_t = horizon[step_idx]
                if sigma_t is not None:
                    cov_xx, cov_xy, cov_yy = sigma_t[0][0], sigma_t[0][1], sigma_t[1][1]
                    sx, sy, yaw = _cov2d_to_ellipse(cov_xx, cov_xy, cov_yy)
                    sx = min(sx, cap)
                    sy = min(sy, cap)
                    qx, qy, qz, qw = _yaw_to_quat(yaw)
                else:
                    # Модель без ковариации: синтетический круг model_ellipse_std для cost (зона в MPPI)
                    std = max(1e-3, self.model_ellipse_std)
                    sx = sy = min(2.0 * n_sigma_vis * std, cap)
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                ell = Marker()
                ell.header = header
                ell.ns = "predicted_people_cov"
                ell.id = int(person_id * 100 + step_idx)
                ell.type = Marker.CYLINDER
                ell.action = Marker.ADD
                ell.pose.position.x = float(mu_t[0])
                ell.pose.position.y = float(mu_t[1])
                ell.pose.position.z = 0.01
                ell.pose.orientation.x = qx
                ell.pose.orientation.y = qy
                ell.pose.orientation.z = qz
                ell.pose.orientation.w = qw
                ell.scale.x = float(sx)
                ell.scale.y = float(sy)
                ell.scale.z = 0.02
                ell.color.r = 0.2
                ell.color.g = 0.4
                ell.color.b = 1.0
                ell.color.a = 0.30
                ell.lifetime = lifetime
                out.markers.append(ell)
        return out

    def _publish_prediction(self) -> None:
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        horizon_by_id = self._backend.get_horizons(now_sec)

        # Exclude robot's fake track from published predictions.
        if self.predict_robot_as_agent:
            horizon_by_id.pop(self._ROBOT_FAKE_PID, None)

        points_xyz: List[Tuple[float, float, float]] = []
        for _, horizon in horizon_by_id.items():
            for mu_t, _ in horizon:
                points_xyz.append((mu_t[0], mu_t[1], 0.0))

        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = self.frame_id_override or self.latest_frame_id

        self.pub_cloud.publish(self._create_cloud_xyz32(header, points_xyz))
        self.pub_markers.publish(self._build_markers(header, horizon_by_id))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = PeoplePredictor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
