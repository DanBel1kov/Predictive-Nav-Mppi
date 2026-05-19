#!/usr/bin/env python3
"""Prepare complete training dataset with Kalman baseline predictions."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Kalman filter implementation
def build_f_q(dt: float, sigma_acc: float) -> Tuple[List[List[float]], List[List[float]]]:
    dt2, dt3, dt4 = dt * dt, dt**3, dt**4
    q = sigma_acc * sigma_acc
    f = [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
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
    rows, cols, inner = len(a), len(b[0]), len(b)
    out = [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]
    return out


def mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def predict_state_cov(mu: List[float], sigma: List[List[float]], dt: float, sigma_acc: float) -> Tuple[List[float], List[List[float]]]:
    f, q = build_f_q(dt, sigma_acc)
    mu_pred = [mu[0] + dt * mu[2], mu[1] + dt * mu[3], mu[2], mu[3]]
    sigma_pred = mat_add(mat_mul(mat_mul(f, sigma), mat_transpose(f)), q)
    for i in range(4):
        for j in range(i + 1, 4):
            s = 0.5 * (sigma_pred[i][j] + sigma_pred[j][i])
            sigma_pred[i][j] = s
            sigma_pred[j][i] = s
    return mu_pred, sigma_pred


def update_state_cov(mu: List[float], sigma: List[List[float]], z_x: float, z_y: float, sigma_meas: float) -> Tuple[List[float], List[List[float]]]:
    r = sigma_meas * sigma_meas
    s00, s01, s10, s11 = sigma[0][0] + r, sigma[0][1], sigma[1][0], sigma[1][1] + r
    det = s00 * s11 - s01 * s10
    if abs(det) < 1e-12:
        return mu, sigma
    inv_s = [[s11 / det, -s01 / det], [-s10 / det, s00 / det]]
    pht = [[sigma[i][0], sigma[i][1]] for i in range(4)]
    k = [[pht[i][0] * inv_s[0][0] + pht[i][1] * inv_s[1][0], pht[i][0] * inv_s[0][1] + pht[i][1] * inv_s[1][1]] for i in range(4)]
    innov0, innov1 = z_x - mu[0], z_y - mu[1]
    mu_upd = [mu[0] + k[0][0] * innov0 + k[0][1] * innov1, mu[1] + k[1][0] * innov0 + k[1][1] * innov1,
              mu[2] + k[2][0] * innov0 + k[2][1] * innov1, mu[3] + k[3][0] * innov0 + k[3][1] * innov1]
    kh = [[k[i][0], k[i][1], 0.0, 0.0] for i in range(4)]
    i_minus_kh = [[1.0 - kh[i][j] if i == j else -kh[i][j] for j in range(4)] for i in range(4)]
    sigma_upd = mat_mul(i_minus_kh, sigma)
    for i in range(4):
        for j in range(i + 1, 4):
            s = 0.5 * (sigma_upd[i][j] + sigma_upd[j][i])
            sigma_upd[i][j] = s
            sigma_upd[j][i] = s
    return mu_upd, sigma_upd


def compute_kalman_prediction(obs_xy: List[List[float]], gt_xy: List[List[float]], dt: float = 0.1) -> Tuple[List[List[float]], List[List[float]]]:
    obs_xy = np.asarray(obs_xy, dtype=np.float32)
    gt_xy = np.asarray(gt_xy, dtype=np.float32)
    if obs_xy.shape[0] < 2:
        return [], []

    px, py = float(obs_xy[0, 0]), float(obs_xy[0, 1])
    vx = float(obs_xy[1, 0] - obs_xy[0, 0]) / dt
    vy = float(obs_xy[1, 1] - obs_xy[0, 1]) / dt
    sp2, sv2 = 0.01, 0.04
    sigma = [[sp2, 0.0, 0.0, 0.0], [0.0, sp2, 0.0, 0.0], [0.0, 0.0, sv2, 0.0], [0.0, 0.0, 0.0, sv2]]
    mu = [px, py, vx, vy]

    for i in range(1, obs_xy.shape[0]):
        mu, sigma = predict_state_cov(mu, sigma, dt, 0.5)
        z_x, z_y = float(obs_xy[i, 0]), float(obs_xy[i, 1])
        mu, sigma = update_state_cov(mu, sigma, z_x, z_y, 0.1)

    kalman_predictions = []
    for i in range(len(gt_xy)):
        mu, sigma = predict_state_cov(mu, sigma, dt, 0.5)
        kalman_predictions.append([mu[0], mu[1]])

    residuals = [[float(gt_xy[i, 0]) - kalman_predictions[i][0], float(gt_xy[i, 1]) - kalman_predictions[i][1]]
                 for i in range(len(gt_xy))]
    return kalman_predictions, residuals


def main():
    print("=" * 70)
    print("RESIDUAL MODEL TRAINING DATA PREPARATION")
    print("=" * 70)

    # Find all benchmark files
    base_path = Path("/home/danbel1kov/predictive-nav-mppi/benchmark_force_sweep")
    bench_files = list(base_path.rglob("benchmark_cases.json"))
    bench_files = [f for f in bench_files if f.parent.name == "curated_near_robot"]

    if not bench_files:
        print("❌ No benchmark files found!")
        sys.exit(1)

    print(f"\nFound {len(bench_files)} benchmark files")

    # Load and combine all cases
    all_cases = []
    dist = defaultdict(int)

    for bench_file in sorted(bench_files):
        data = json.loads(bench_file.read_text())
        cases = data.get("cases", [])

        # Extract force value from path
        force_val = bench_file.parent.parent.parent.name
        force_val = force_val.replace("force_0p", "0.").replace("force_", "")

        for case in cases:
            if "force_sweep" not in case:
                case["force_sweep"] = force_val
            all_cases.append(case)

        # Track distribution
        for case in cases:
            tags = set(case.get("tags", []))
            tags.discard("all")
            priority = ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"]
            for tag in priority:
                if tag in tags:
                    dist[tag] += 1
                    break

    print(f"\n✓ Loaded {len(all_cases)} total cases")
    print("\nCombined distribution:")
    total = sum(dist.values())
    for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]:
        count = dist.get(cat, 0)
        pct = 100.0 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Add Kalman predictions
    print("\n" + "=" * 70)
    print("COMPUTING KALMAN BASELINES...")
    print("=" * 70)

    cases_with_kalman = 0
    cases_skipped = 0

    for idx, case in enumerate(all_cases):
        if idx % 100 == 0:
            print(f"  {idx}/{len(all_cases)}: added_kalman={cases_with_kalman}, skipped={cases_skipped}")

        try:
            if "kalman_pred_xy" in case:
                continue

            obs_xy = case.get("obs_xy", [])
            gt_xy = case.get("gt_xy", [])

            if not obs_xy or not gt_xy:
                cases_skipped += 1
                continue

            kalman_pred, residuals = compute_kalman_prediction(obs_xy, gt_xy)
            if len(kalman_pred) == len(gt_xy):
                case["kalman_pred_xy"] = kalman_pred
                case["residual_xy"] = residuals
                cases_with_kalman += 1
            else:
                cases_skipped += 1
        except Exception as e:
            cases_skipped += 1

    print(f"✓ Added Kalman to {cases_with_kalman} cases")
    if cases_skipped > 0:
        print(f"⚠ Skipped {cases_skipped} cases")

    # Split into train (75%) and test (25%)
    train_size = int(0.75 * len(all_cases))
    train_cases = all_cases[:train_size]
    test_cases = all_cases[train_size:]

    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"Train: {len(train_cases)} cases (75%)")
    print(f"Test:  {len(test_cases)} cases (25%)")

    # Save files
    output_dir = Path("/home/danbel1kov/predictive-nav-mppi/datasets")
    output_dir.mkdir(exist_ok=True)

    train_file = output_dir / "residual_train.json"
    test_file = output_dir / "residual_test.json"

    train_file.write_text(json.dumps({"cases": train_cases}, indent=2))
    test_file.write_text(json.dumps({"cases": test_cases}, indent=2))

    print(f"\n✓ Train saved: {train_file}")
    print(f"✓ Test saved:  {test_file}")

    # Print training command
    print("\n" + "=" * 70)
    print("TRAINING COMMAND")
    print("=" * 70)
    print()
    print("cd /home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi")
    print()
    print("python3 -m predictive_nav_mppi.train_residual_predictor \\")
    print(f"  --train_file ../../datasets/residual_train.json \\")
    print(f"  --test_file ../../datasets/residual_test.json \\")
    print("  --obs_len 8 \\")
    print("  --pred_len 26 \\")
    print("  --batch_size 32 \\")
    print("  --epochs 100 \\")
    print("  --learning_rate 0.001 \\")
    print("  --k_neighbors 3 \\")
    print("  --scene_patch_size_m 6.0 \\")
    print("  --scene_patch_pixels 32 \\")
    print("  --include_robot \\")
    print("  --output_dir ../../models/residual_baseline")
    print()


if __name__ == "__main__":
    main()
