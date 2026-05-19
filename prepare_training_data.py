#!/usr/bin/env python3
"""Prepare benchmark data for residual model training by adding Kalman baseline predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Kalman filter parameters (from people_predictor.py defaults)
SIGMA_P0 = 0.1      # Initial position uncertainty
SIGMA_V0 = 0.2      # Initial velocity uncertainty
SIGMA_ACC = 0.5     # Process noise (acceleration)
SIGMA_MEAS = 0.1    # Measurement noise


def build_f_q(dt: float, sigma_acc: float) -> Tuple[List[List[float]], List[List[float]]]:
    """Build Kalman F matrix and Q noise matrix."""
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


def mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication."""
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


def mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Matrix addition."""
    rows = len(a)
    cols = len(a[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = a[i][j] + b[i][j]
    return out


def mat_transpose(a: List[List[float]]) -> List[List[float]]:
    """Matrix transpose."""
    return [list(row) for row in zip(*a)]


def predict_state_cov(
    mu: List[float],
    sigma: List[List[float]],
    dt: float,
    sigma_acc: float,
) -> Tuple[List[float], List[List[float]]]:
    """Kalman filter prediction step."""
    f, q = build_f_q(dt, sigma_acc)
    mu_pred = [
        mu[0] + dt * mu[2],
        mu[1] + dt * mu[3],
        mu[2],
        mu[3],
    ]
    sigma_pred = mat_add(mat_mul(mat_mul(f, sigma), mat_transpose(f)), q)
    # Keep symmetric
    for i in range(4):
        for j in range(i + 1, 4):
            s = 0.5 * (sigma_pred[i][j] + sigma_pred[j][i])
            sigma_pred[i][j] = s
            sigma_pred[j][i] = s
    return mu_pred, sigma_pred


def update_state_cov(
    mu: List[float],
    sigma: List[List[float]],
    z_x: float,
    z_y: float,
    sigma_meas: float,
) -> Tuple[List[float], List[List[float]]]:
    """Kalman filter update step."""
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

    sigma_upd = mat_mul(i_minus_kh, sigma)
    for i in range(4):
        for j in range(i + 1, 4):
            s = 0.5 * (sigma_upd[i][j] + sigma_upd[j][i])
            sigma_upd[i][j] = s
            sigma_upd[j][i] = s
    return mu_upd, sigma_upd


def compute_kalman_prediction(obs_xy: List[List[float]], gt_xy: List[List[float]], dt: float = 0.1) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute Kalman baseline prediction for a trajectory.

    Args:
        obs_xy: Observation trajectory (list of [x, y])
        gt_xy: Ground truth future trajectory
        dt: Time step

    Returns:
        Tuple of (kalman_predictions, residuals)
    """
    obs_xy = np.asarray(obs_xy, dtype=np.float32)
    gt_xy = np.asarray(gt_xy, dtype=np.float32)

    # Initialize Kalman state from first two observations
    if obs_xy.shape[0] < 2:
        return [], []

    # Initial position and velocity estimate
    px, py = float(obs_xy[0, 0]), float(obs_xy[0, 1])
    vx = float(obs_xy[1, 0] - obs_xy[0, 0]) / dt
    vy = float(obs_xy[1, 1] - obs_xy[0, 1]) / dt

    # Initialize covariance
    sp2 = SIGMA_P0 ** 2
    sv2 = SIGMA_V0 ** 2
    sigma = [
        [sp2, 0.0, 0.0, 0.0],
        [0.0, sp2, 0.0, 0.0],
        [0.0, 0.0, sv2, 0.0],
        [0.0, 0.0, 0.0, sv2],
    ]
    mu = [px, py, vx, vy]

    # Process observations through Kalman filter
    for i in range(1, obs_xy.shape[0]):
        # Predict
        mu, sigma = predict_state_cov(mu, sigma, dt, SIGMA_ACC)
        # Update
        z_x, z_y = float(obs_xy[i, 0]), float(obs_xy[i, 1])
        mu, sigma = update_state_cov(mu, sigma, z_x, z_y, SIGMA_MEAS)

    # Now predict into the future
    kalman_predictions = []
    for i in range(len(gt_xy)):
        mu, sigma = predict_state_cov(mu, sigma, dt, SIGMA_ACC)
        kalman_predictions.append([mu[0], mu[1]])

    # Compute residuals
    residuals = []
    for i in range(len(gt_xy)):
        if i < len(kalman_predictions):
            residual = [
                float(gt_xy[i, 0]) - kalman_predictions[i][0],
                float(gt_xy[i, 1]) - kalman_predictions[i][1],
            ]
            residuals.append(residual)

    return kalman_predictions, residuals


def process_benchmark_file(input_path: Path, output_path: Path) -> None:
    """Process a benchmark JSON file to add Kalman predictions and residuals."""
    print(f"Reading: {input_path}")
    data = json.loads(input_path.read_text())
    cases = data.get("cases", [])
    print(f"Processing {len(cases)} cases...")

    added_count = 0
    error_count = 0

    for case_idx, case in enumerate(cases):
        try:
            if case_idx % 100 == 0:
                print(f"  {case_idx}/{len(cases)}: added={added_count}, errors={error_count}")

            # Skip if already has Kalman predictions
            if "kalman_pred_xy" in case:
                continue

            obs_xy = case.get("obs_xy", [])
            gt_xy = case.get("gt_xy", [])

            if not obs_xy or not gt_xy:
                continue

            # Compute Kalman predictions
            kalman_pred, residuals = compute_kalman_prediction(obs_xy, gt_xy)

            if len(kalman_pred) == len(gt_xy):
                case["kalman_pred_xy"] = kalman_pred
                case["residual_xy"] = residuals
                added_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"  Error processing case {case_idx}: {e}")
            error_count += 1

    print(f"\nWriting: {output_path}")
    output_path.write_text(json.dumps(data, indent=2))
    print(f"✓ Added Kalman predictions to {added_count} cases")
    if error_count > 0:
        print(f"⚠ Errors processing {error_count} cases")


def main():
    # Find all benchmark JSON files
    base_path = Path("/home/danbel1kov/predictive-nav-mppi/benchmark_force_sweep")
    benchmark_files = list(base_path.rglob("benchmark_cases.json"))

    if not benchmark_files:
        print("No benchmark_cases.json files found!")
        sys.exit(1)

    print(f"Found {len(benchmark_files)} benchmark files")

    for bench_file in benchmark_files:
        output_file = bench_file.parent / "benchmark_cases_with_kalman.json"
        try:
            process_benchmark_file(bench_file, output_file)
        except Exception as e:
            print(f"Failed to process {bench_file}: {e}")


if __name__ == "__main__":
    main()
