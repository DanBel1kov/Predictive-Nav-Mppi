#!/usr/bin/env python3
"""Extract training cases from raw simulation datasets with Kalman baseline."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ============= Kalman Filter Implementation =============
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
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


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


# ============= Case Extraction =============
def extract_cases_from_frames(frames: List, source_name: str, obs_len: int = 8, pred_len: int = 26) -> List[dict]:
    """Extract trajectory cases from frame sequences."""
    cases = []

    for frame_idx in range(len(frames) - obs_len - pred_len):
        frame = frames[frame_idx + obs_len - 1]
        people_in_frame = {p.get('id'): (p.get('x'), p.get('y')) for p in frame.get('people', [])}

        for person_id, (px, py) in people_in_frame.items():
            # Get observation trajectory
            obs_traj = []
            for i in range(frame_idx, frame_idx + obs_len):
                for p in frames[i].get('people', []):
                    if p.get('id') == person_id:
                        x, y = p.get('x'), p.get('y')
                        if x is not None and y is not None:
                            obs_traj.append((x, y))
                        break

            if len(obs_traj) < obs_len:
                continue

            # Get ground truth future trajectory
            gt_traj = []
            for i in range(frame_idx + obs_len, min(frame_idx + obs_len + pred_len, len(frames))):
                found = False
                for p in frames[i].get('people', []):
                    if p.get('id') == person_id:
                        x, y = p.get('x'), p.get('y')
                        if x is not None and y is not None:
                            gt_traj.append((x, y))
                        found = True
                        break
                if not found:
                    break

            if len(gt_traj) < pred_len:
                continue

            # Get robot trajectory
            robot_traj = []
            for i in range(frame_idx, frame_idx + obs_len):
                frame = frames[i]
                robot = frame.get('robot', {})
                robot_traj.append([robot.get('x', 0), robot.get('y', 0)])

            # Get other people trajectories
            neigh_xy = []
            for i in range(frame_idx, frame_idx + obs_len):
                frame = frames[i]
                for p in frame.get('people', []):
                    oid = p.get('id')
                    if oid != person_id and oid is not None:
                        x, y = p.get('x'), p.get('y')
                        if x is not None and y is not None:
                            # Find or create entry for this neighbor
                            pass

            # Simplified neighbor extraction: just current frame neighbors
            neigh_xy = []
            for other_id, (ox, oy) in people_in_frame.items():
                if other_id != person_id:
                    neigh_traj = []
                    for i in range(frame_idx, frame_idx + obs_len):
                        for p in frames[i].get('people', []):
                            if p.get('id') == other_id:
                                x, y = p.get('x'), p.get('y')
                                if x is not None and y is not None:
                                    neigh_traj.append((x, y))
                                break
                    if len(neigh_traj) == obs_len:
                        neigh_xy.append(neigh_traj)

            # Create case
            case = {
                'person_id': person_id,
                'frame_index': frame_idx,
                'obs_xy': obs_traj,
                'gt_xy': gt_traj,
                'robot_obs_xy': robot_traj,
                'neigh_xy': neigh_xy,
                'source_name': source_name,
            }
            cases.append(case)

    return cases


def main():
    print("=" * 70)
    print("PREPARING TRAINING DATA FROM RAW SIMULATION DATASETS")
    print("=" * 70)

    obs_len = 8
    pred_len = 26
    target_train_size = 10000
    target_test_size = 2000

    # Load raw datasets
    raw_files = [
        ("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_long_corridor_react_0p5.json", "long_corridor"),
        ("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_nonlinear_corridor_react_0p5.json", "nonlinear_corridor"),
        ("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_labyrinth_turns_react_0p5.json", "labyrinth_turns"),
    ]

    all_cases = []

    for fpath, source_name in raw_files:
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"⚠ Skipping {fpath.name} (not found)")
            continue

        print(f"\nLoading {source_name}...", end=" ")
        data = json.loads(fpath.read_text())
        frames = data.get('frames', [])
        print(f"{len(frames)} frames")

        print(f"  Extracting cases (obs_len={obs_len}, pred_len={pred_len})...", end=" ")
        cases = extract_cases_from_frames(frames, source_name, obs_len, pred_len)
        print(f"extracted {len(cases)} cases")

        all_cases.extend(cases)

    print(f"\n✓ Total cases extracted: {len(all_cases)}")

    # Shuffle and split
    random.shuffle(all_cases)

    # Target split: ~10k train, ~2k test (adjust if needed)
    if len(all_cases) < target_train_size + target_test_size:
        print(f"⚠ Only {len(all_cases)} cases, less than target {target_train_size + target_test_size}")
        train_size = int(0.83 * len(all_cases))  # 83% train, 17% test
    else:
        train_size = target_train_size

    train_cases = all_cases[:train_size]
    test_cases = all_cases[train_size:]

    print(f"\nTrain/Test split:")
    print(f"  Train: {len(train_cases)} cases")
    print(f"  Test:  {len(test_cases)} cases")

    # Add Kalman baselines
    print(f"\nAdding Kalman baselines...")
    for idx, case in enumerate(all_cases):
        if idx % 2000 == 0:
            print(f"  {idx}/{len(all_cases)}")

        try:
            obs_xy = case.get('obs_xy', [])
            gt_xy = case.get('gt_xy', [])

            if obs_xy and gt_xy:
                kalman_pred, residuals = compute_kalman_prediction(obs_xy, gt_xy)
                if len(kalman_pred) == len(gt_xy):
                    case['kalman_pred_xy'] = kalman_pred
                    case['residual_xy'] = residuals
        except:
            pass

    print("✓ Kalman baselines added")

    # Save train/test
    output_dir = Path("/home/danbel1kov/predictive-nav-mppi/datasets")
    output_dir.mkdir(exist_ok=True)

    train_file = output_dir / "training_raw_10k.json"
    test_file = output_dir / "testing_raw_2k.json"

    train_file.write_text(json.dumps({"cases": train_cases}, indent=2))
    test_file.write_text(json.dumps({"cases": test_cases}, indent=2))

    print(f"\n✓ Train saved: {train_file}")
    print(f"✓ Test saved:  {test_file}")

    # Print command
    print("\n" + "=" * 70)
    print("TRAINING COMMAND")
    print("=" * 70)
    print()
    print("cd /home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi")
    print()
    print("python3 -m predictive_nav_mppi.train_residual_predictor \\")
    print(f"  --train_dataset ../../datasets/training_raw_10k.json \\")
    print(f"  --val_dataset ../../datasets/testing_raw_2k.json \\")
    print("  --output_dir ../../models/residual_raw_10k \\")
    print("  --obs_len 8 --pred_len 26 --obs_dt 0.1 --pred_dt 0.1 \\")
    print("  --batch_size 32 --epochs 100 --lr 0.001 \\")
    print("  --k_neighbors 3 --include_robot \\")
    print("  --lambda_jerk 0.05 --lambda_acc_match 0.1 \\")
    print("  --lambda_vel 0.05 --lambda_res_mag 0.01 \\")
    print("  --lambda_risk 0.5 --lambda_safety 1.0")
    print()


if __name__ == "__main__":
    random.seed(42)
    main()
