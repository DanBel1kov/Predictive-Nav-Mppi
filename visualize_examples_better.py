#!/usr/bin/env python3
"""Better visualization with full trajectories."""
from __future__ import annotations

import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def classify_obs(obs_xy: np.ndarray, nearby_count: int, min_neighbor_dist: float,
                 interaction_dist: float = 1.5) -> str:
    """Quick classification."""
    obs_dt = 0.1

    heading_deg = 0.0
    if obs_xy.shape[0] >= 3:
        step = obs_xy[1:] - obs_xy[:-1]
        ang = np.arctan2(step[:, 1], step[:, 0])
        if ang.shape[0] >= 2:
            d = np.diff(ang)
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            heading_deg = math.degrees(float(np.sum(np.abs(d))))

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
    display_len = 40  # Show longer trajectory context

    examples_by_cat = {cat: [] for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]}

    sample_indices = sorted(random.sample(range(len(frames) - obs_len - pred_len), min(1000, len(frames) - obs_len - pred_len)))

    print(f"Sampling {len(sample_indices)} frame positions...")

    checked = 0
    for frame_idx in sample_indices:
        if checked % 100 == 0:
            print(f"  Checked {checked}... Found: {sum(len(v) for v in examples_by_cat.values())}")
        checked += 1

        frame = frames[frame_idx + obs_len - 1]
        people_in_frame = {p.get('id'): (p.get('x'), p.get('y')) for p in frame.get('people', [])}

        for person_id, (px, py) in people_in_frame.items():
            neighbor_dists = []
            for other_id, (ox, oy) in people_in_frame.items():
                if other_id != person_id:
                    dist = math.sqrt((px - ox)**2 + (py - oy)**2)
                    neighbor_dists.append(dist)

            neighbor_count = sum(1 for d in neighbor_dists if d <= 1.5)
            min_neighbor_dist = min(neighbor_dists) if neighbor_dists else 999.0

            # Get main person trajectory
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
                    # Get full trajectory for display (longer context)
                    full_traj = []
                    for i in range(max(0, frame_idx - display_len), min(frame_idx + obs_len + pred_len, len(frames))):
                        found = False
                        for p in frames[i].get('people', []):
                            if p.get('id') == person_id:
                                x, y = p.get('x'), p.get('y')
                                if x is not None and y is not None:
                                    full_traj.append((x, y))
                                found = True
                                break
                        if not found:
                            break

                    full_xy = np.array(full_traj, dtype=np.float64) if full_traj else np.zeros((0, 2))

                    # Get robot trajectory (full context)
                    robot_traj = []
                    for i in range(max(0, frame_idx - display_len), min(frame_idx + obs_len + pred_len, len(frames))):
                        frame = frames[i]
                        robot = frame.get('robot', {})
                        robot_traj.append([robot.get('x', 0), robot.get('y', 0)])

                    # Get other people trajectories (full context)
                    other_people_trajs = {}
                    for i in range(max(0, frame_idx - display_len), min(frame_idx + obs_len + pred_len, len(frames))):
                        frame = frames[i]
                        for p in frame.get('people', []):
                            oid = p.get('id')
                            if oid != person_id and oid is not None:
                                x, y = p.get('x'), p.get('y')
                                if x is not None and y is not None:
                                    if oid not in other_people_trajs:
                                        other_people_trajs[oid] = []
                                    other_people_trajs[oid].append([x, y])

                    examples_by_cat[cat].append({
                        'person_id': person_id,
                        'frame_idx': frame_idx,
                        'obs_xy': obs_xy,
                        'full_xy': full_xy,
                        'robot_xy': np.array(robot_traj),
                        'obs_start_idx': frame_idx - max(0, frame_idx - display_len),
                        'obs_end_idx': frame_idx - max(0, frame_idx - display_len) + obs_len,
                        'pred_end_idx': min(frame_idx + obs_len + pred_len, len(frames)) - max(0, frame_idx - display_len),
                        'other_people_trajs': {oid: np.array(traj) for oid, traj in other_people_trajs.items()},
                    })

    print(f"Found examples:")
    for cat, exs in examples_by_cat.items():
        print(f"  {cat:20s}: {len(exs)}/4")

    # Plot
    fig, axes = plt.subplots(6, 4, figsize=(24, 20))
    fig.suptitle("Current Dataset - Trajectory Examples\n" +
                "Blue circle=Person start, Blue square=Person end, Green=Future pred, Red=Robot, Gray=Other people",
                fontsize=16, fontweight='bold')

    categories = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]

    for cat_idx, cat in enumerate(categories):
        for ex_idx in range(4):
            ax = axes[cat_idx, ex_idx]
            exs = examples_by_cat[cat]

            if ex_idx < len(exs):
                ex = exs[ex_idx]

                # Plot other people trajectories
                for oid, traj in ex['other_people_trajs'].items():
                    if len(traj) > 1:
                        ax.plot(traj[:, 0], traj[:, 1], 'o-', color='lightgray', linewidth=1.5, markersize=3, alpha=0.6)

                # Plot main person full trajectory
                if ex['full_xy'].size > 0:
                    ax.plot(ex['full_xy'][:, 0], ex['full_xy'][:, 1], 'b-', linewidth=3, label='Person trajectory', zorder=5)

                    # Highlight observation window
                    obs_start = ex['obs_start_idx']
                    obs_end = ex['obs_end_idx']
                    if obs_end <= len(ex['full_xy']):
                        obs_segment = ex['full_xy'][obs_start:obs_end]
                        ax.plot(obs_segment[:, 0], obs_segment[:, 1], 'b-', linewidth=5, alpha=0.8, zorder=6, label='Observation')

                    # Mark start and end of observation
                    ax.scatter(ex['full_xy'][obs_start, 0], ex['full_xy'][obs_start, 1], c='blue', s=250, marker='o',
                              edgecolor='black', linewidth=2, zorder=10, label='Obs start')
                    if obs_end - 1 < len(ex['full_xy']):
                        ax.scatter(ex['full_xy'][obs_end-1, 0], ex['full_xy'][obs_end-1, 1], c='blue', s=250, marker='s',
                                  edgecolor='black', linewidth=2, zorder=10, label='Obs end')

                    # Highlight prediction window
                    pred_end = ex['pred_end_idx']
                    if obs_end < pred_end <= len(ex['full_xy']):
                        pred_segment = ex['full_xy'][obs_end:pred_end]
                        if len(pred_segment) > 0:
                            ax.plot(pred_segment[:, 0], pred_segment[:, 1], 'g--', linewidth=3, zorder=4, label='Future')

                # Plot robot trajectory
                ax.plot(ex['robot_xy'][:, 0], ex['robot_xy'][:, 1], 'r-', linewidth=3, label='Robot path', zorder=3)
                ax.scatter(ex['robot_xy'][0, 0], ex['robot_xy'][0, 1], c='red', s=200, marker='^',
                          edgecolor='darkred', linewidth=2, zorder=9, label='Robot start')
                ax.scatter(ex['robot_xy'][-1, 0], ex['robot_xy'][-1, 1], c='darkred', s=200, marker='v',
                          edgecolor='darkred', linewidth=2, zorder=9, label='Robot end')

                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{cat.upper()} #{ex_idx+1} (P{ex['person_id']})", fontsize=11, fontweight='bold')
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)

                if cat_idx == 0 and ex_idx == 0:
                    ax.legend(loc='best', fontsize=8)
            else:
                ax.text(0.5, 0.5, f"No example found", ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='red', fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

    output = Path("/home/danbel1kov/predictive-nav-mppi/trajectory_examples_current.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output}")
    plt.close()


if __name__ == "__main__":
    main()
