from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


Vector4 = List[float]
Matrix4 = List[List[float]]


@dataclass
class TrackState:
    mu: Vector4
    sigma: Matrix4
    last_update_sec: float


def clamp_dt(dt: float, min_dt: float, max_dt: float) -> float:
    return max(min_dt, min(max_dt, dt))


def build_f_q(dt: float, sigma_acc: float) -> Tuple[Matrix4, Matrix4]:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    q = sigma_acc * sigma_acc

    f = [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    q_mat = [[0.0] * 4 for _ in range(4)]
    q_mat[0][0] = 0.25 * dt4 * q
    q_mat[0][2] = 0.5 * dt3 * q
    q_mat[2][0] = 0.5 * dt3 * q
    q_mat[2][2] = dt2 * q

    q_mat[1][1] = 0.25 * dt4 * q
    q_mat[1][3] = 0.5 * dt3 * q
    q_mat[3][1] = 0.5 * dt3 * q
    q_mat[3][3] = dt2 * q

    return f, q_mat


def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def _mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(a[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = a[i][j] + b[i][j]
    return out


def _mat_transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def predict_state_cov(
    mu: Vector4,
    sigma: Matrix4,
    dt: float,
    sigma_acc: float,
) -> Tuple[Vector4, Matrix4]:
    f, q = build_f_q(dt, sigma_acc)
    mu_pred = [
        mu[0] + dt * mu[2],
        mu[1] + dt * mu[3],
        mu[2],
        mu[3],
    ]
    sigma_pred = _mat_add(_mat_mul(_mat_mul(f, sigma), _mat_transpose(f)), q)
    # Keep covariance symmetric to reduce numeric drift.
    for i in range(4):
        for j in range(i + 1, 4):
            s = 0.5 * (sigma_pred[i][j] + sigma_pred[j][i])
            sigma_pred[i][j] = s
            sigma_pred[j][i] = s
    return mu_pred, sigma_pred


def update_state_cov(
    mu: Vector4,
    sigma: Matrix4,
    z_x: float,
    z_y: float,
    sigma_meas: float,
) -> Tuple[Vector4, Matrix4]:
    r = sigma_meas * sigma_meas
    s00 = sigma[0][0] + r
    s01 = sigma[0][1]
    s10 = sigma[1][0]
    s11 = sigma[1][1] + r
    det = s00 * s11 - s01 * s10
    if abs(det) < 1e-12:
        return mu, sigma

    inv_s = [
        [s11 / det, -s01 / det],
        [-s10 / det, s00 / det],
    ]
    pht = [
        [sigma[0][0], sigma[0][1]],
        [sigma[1][0], sigma[1][1]],
        [sigma[2][0], sigma[2][1]],
        [sigma[3][0], sigma[3][1]],
    ]

    k = [[0.0, 0.0] for _ in range(4)]
    for i in range(4):
        k[i][0] = pht[i][0] * inv_s[0][0] + pht[i][1] * inv_s[1][0]
        k[i][1] = pht[i][0] * inv_s[0][1] + pht[i][1] * inv_s[1][1]

    innov0 = z_x - mu[0]
    innov1 = z_y - mu[1]
    mu_upd = [
        mu[0] + k[0][0] * innov0 + k[0][1] * innov1,
        mu[1] + k[1][0] * innov0 + k[1][1] * innov1,
        mu[2] + k[2][0] * innov0 + k[2][1] * innov1,
        mu[3] + k[3][0] * innov0 + k[3][1] * innov1,
    ]

    kh = [
        [k[0][0], k[0][1], 0.0, 0.0],
        [k[1][0], k[1][1], 0.0, 0.0],
        [k[2][0], k[2][1], 0.0, 0.0],
        [k[3][0], k[3][1], 0.0, 0.0],
    ]
    i_minus_kh = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        i_minus_kh[i][i] = 1.0
        for j in range(4):
            i_minus_kh[i][j] -= kh[i][j]

    sigma_upd = _mat_mul(i_minus_kh, sigma)
    for i in range(4):
        for j in range(i + 1, 4):
            s = 0.5 * (sigma_upd[i][j] + sigma_upd[j][i])
            sigma_upd[i][j] = s
            sigma_upd[j][i] = s
    return mu_upd, sigma_upd


def prune_stale_tracks(
    tracks: Dict[int, TrackState],
    now_sec: float,
    timeout_sec: float,
) -> None:
    stale_ids = [
        track_id for track_id, t in tracks.items()
        if now_sec - t.last_update_sec > timeout_sec
    ]
    for track_id in stale_ids:
        del tracks[track_id]
