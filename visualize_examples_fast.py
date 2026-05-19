#!/usr/bin/env python3
"""Fast visualization of trajectory examples."""
from __future__ import annotations

import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def classify_obs(obs_xy: np.ndarray, nearby_count: int, min_neighbor_dist: float,
                 interaction_dist: float = 1.5) -> str:
    """Quick classification."""
    obs_dt = 0.1

    # Heading change
    heading_deg = 0.0
    if obs_xy.shape[0] >= 3:
        step = obs_xy[1:] - obs_xy[:-1]
        ang = np.arctan2(step[:, 1], step[:, 0])
        if ang.shape[0] >= 2:
            d = np.diff(ang)
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            heading_deg = math.degrees(float(np.sum(np.abs(d))))

    # Speed
    speeds = np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1) / obs_dt
    min_speed = float(np.min(speeds)) if speeds.size > 0 else 0.0
    max_speed = float(np.max(speeds)) if speeds.size > 0 else 0.0

    if (max_speed >= 0.25 and min_speed <= 0.10 and
        (max_speed - min_speed) >= 0.25 and nearby_count >= 1):
        return "stop_go"

    if nearby_count >= 4 and min_neighbor_dist <= interaction_dist:
        return "dense_interaction"

    if 1 <= nearby_count <= 3 and min_neighbor_dist <= interaction_dist:
        if heading_deg >= 45.0:
            return "complex"
        return "interaction"

    if heading_deg >= 45.0:
        return "turning"

    return "linear"


def main():
    current_path = Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_merged_react_0p5.json")

    print("Loading dataset...")
    with open(current_path) as f:
        data = json.load(f)

    frames = data['frames']
    print(f"Loaded {len(frames)} frames")

    obs_len = 8
    pred_len = 12

    # Find examples by category
    examples_by_cat = {cat: [] for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]}

    # Sample random frame indices to check
    sample_indices = sorted(random.sample(range(len(frames) - obs_len - pred_len), min(1000, len(frames) - obs_len - pred_len)))

    print(f"Sampling {len(sample_indices)} frame positions...")

    checked = 0
    for frame_idx in sample_indices:
        if checked % 100 == 0:
            print(f"  Checked {checked}... Found: {sum(len(v) for v in examples_by_cat.values())}")
        checked += 1

        # Quick scan: find people and their neighbors in this frame
        frame = frames[frame_idx + obs_len - 1]
        people_in_frame = {p.get('id'): (p.get('x'), p.get('y')) for p in frame.get('people', [])}

        for person_id, (px, py) in people_in_frame.items():
            # Count neighbors
            neighbor_dists = []
            for other_id, (ox, oy) in people_in_frame.items():
                if other_id != person_id:
                    dist = math.sqrt((px - ox)**2 + (py - oy)**2)
                    neighbor_dists.append(dist)

            neighbor_count = sum(1 for d in neighbor_dists if d <= 1.5)
            min_neighbor_dist = min(neighbor_dists) if neighbor_dists else 999.0

            # Get trajectory for this person
            traj = []
            for i in range(frame_idx, frame_idx + obs_len):
                found = False
                for p in frames[i].get('people', []):
                    if p.get('id') == person_id:
                        x, y = p.get('x'), p.get('y')
                        if x is not None and y is not None:
                            traj.append((x, y))
                            found = True
                        break
                if not found:
                    break

            if len(traj) == obs_len:
                obs_xy = np.array(traj, dtype=np.float64)
                cat = classify_obs(obs_xy, neighbor_count, min_neighbor_dist)

                if len(examples_by_cat[cat]) < 4:
                    # Get full data
                    pred_traj = []
                    for i in range(frame_idx + obs_len, min(frame_idx + obs_len + pred_len, len(frames))):
                        found = False
                        for p in frames[i].get('people', []):
                            if p.get('id') == person_id:
                                x, y = p.get('x'), p.get('y')
                                if x is not None and y is not None:
                                    pred_traj.append((x, y))
                                found = True
                                break
                        if not found:
                            break

                    pred_xy = np.array(pred_traj, dtype=np.float64) if pred_traj else np.zeros((0, 2))

                    # Get robot trajectory
                    robot_traj = []
                    nearby_people = []
                    for i in range(frame_idx, frame_idx + obs_len):
                        frame = frames[i]
                        robot = frame.get('robot', {})
                        robot_traj.append([robot.get('x', 0), robot.get('y', 0)])

                    # Get nearby people at final obs frame
                    for other_id, (ox, oy) in people_in_frame.items():
                        if other_id != person_id:
                            nearby_people.append((ox, oy))

                    examples_by_cat[cat].append({
                        'person_id': person_id,
                        'frame_idx': frame_idx,
                        'obs_xy': obs_xy,
                        'pred_xy': pred_xy,
                        'robot_xy': np.array(robot_traj),
                        'nearby_people': nearby_people,
                    })

    print(f"Found examples:")
    for cat, exs in examples_by_cat.items():
        print(f"  {cat:20s}: {len(exs)}/4")

    # Plot
    fig, axes = plt.subplots(6, 4, figsize=(20, 16))
    fig.suptitle("Current Dataset Examples\nBlue=Person obs, Green=Future, Red=Robot, Gray=Others",
                fontsize=14, fontweight='bold')

    categories = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]

    for cat_idx, cat in enumerate(categories):
        for ex_idx in range(4):
            ax = axes[cat_idx, ex_idx]
            exs = examples_by_cat[cat]

            if ex_idx < len(exs):
                ex = exs[ex_idx]

                # Plot nearby people
                if ex['nearby_people']:
                    nearby_xy = np.array(ex['nearby_people'])
                    ax.scatter(nearby_xy[:, 0], nearby_xy[:, 1], c='gray', s=50, alpha=0.5)

                # Plot trajectory
                ax.plot(ex['obs_xy'][:, 0], ex['obs_xy'][:, 1], 'b-', linewidth=2)
                ax.scatter(ex['obs_xy'][:, 0], ex['obs_xy'][:, 1], c='blue', s=30)

                if ex['pred_xy'].size > 0:
                    ax.plot(ex['pred_xy'][:, 0], ex['pred_xy'][:, 1], 'g--', linewidth=2)
                    ax.scatter(ex['pred_xy'][:, 0], ex['pred_xy'][:, 1], c='green', s=20)

                # Plot robot
                ax.plot(ex['robot_xy'][:, 0], ex['robot_xy'][:, 1], 'r-', linewidth=2.5)
                ax.scatter(ex['robot_xy'][0, 0], ex['robot_xy'][0, 1], c='red', s=150, marker='^')

                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{cat} #{ex_idx+1}\nP{ex['person_id']}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

    output = Path("/tmp/trajectory_examples.png")
    plt.tight_layout()
    plt.savefig(output, dpi=100, bbox_inches='tight')
    print(f"\n✓ Saved: {output}")
    plt.close()


if __name__ == "__main__":
    main()
