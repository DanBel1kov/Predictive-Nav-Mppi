#!/usr/bin/env python3
"""Clean visualization for both benchmark and raw datasets."""
from __future__ import annotations

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def classify_case(tags_list: List[str]) -> str:
    """Classify based on tags."""
    tags = set(tags_list) - {"all"}
    priority = ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"]
    for tag in priority:
        if tag in tags:
            return tag
    return "unknown"


def has_movement(obs_xy: np.ndarray, robot_xy: np.ndarray) -> bool:
    """Check if both person and robot moved."""
    if obs_xy.size < 2 or robot_xy.size < 2:
        return False

    person_disp = np.linalg.norm(obs_xy[-1] - obs_xy[0])
    robot_disp = np.linalg.norm(robot_xy[-1] - robot_xy[0])

    # More lenient thresholds for raw data
    return person_disp > 0.05 and robot_disp > 0.02


def visualize_benchmark(output_path: Path):
    """Visualize benchmark dataset."""
    print("\n" + "="*70)
    print("BENCHMARK DATASET")
    print("="*70)

    bench_cases = Path("./benchmark_force_sweep/kalman_react_force_0_025_05_near_robot3/force_0p25/20260513_005606/curated_near_robot/benchmark_cases.json")

    with open(bench_cases) as f:
        data = json.load(f)

    cases = data.get('cases', [])
    print(f"Loaded {len(cases)} cases")

    examples_by_cat = {cat: [] for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]}

    # Filter and classify
    for case in cases:
        obs_xy = np.array(case.get('obs_xy', []), dtype=np.float64)
        robot_xy = np.array(case.get('robot_obs_xy', []), dtype=np.float64)

        if not has_movement(obs_xy, robot_xy):
            continue

        cat = classify_case(case.get('tags', []))
        if cat in examples_by_cat and len(examples_by_cat[cat]) < 4:
            examples_by_cat[cat].append(case)

    print(f"Found moving examples:")
    for cat, exs in examples_by_cat.items():
        print(f"  {cat:20s}: {len(exs)}/4")

    # Plot
    fig, axes = plt.subplots(6, 4, figsize=(24, 20))
    fig.suptitle("BENCHMARK Force=0.25 - Trajectory Examples\nBlue=Person, Green=Future, Red=Robot, Gray=Others (Lines)",
                fontsize=16, fontweight='bold')

    categories = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]

    for cat_idx, cat in enumerate(categories):
        for ex_idx in range(4):
            ax = axes[cat_idx, ex_idx]
            exs = examples_by_cat[cat]

            if ex_idx < len(exs):
                case = exs[ex_idx]

                obs_xy = np.array(case.get('obs_xy', []), dtype=np.float64)
                gt_xy = np.array(case.get('gt_xy', []), dtype=np.float64)
                robot_xy = np.array(case.get('robot_obs_xy', []), dtype=np.float64)
                neigh_xy = case.get('neigh_xy', [])

                # Plot neighbors as lines
                for neighbor_traj in neigh_xy:
                    if isinstance(neighbor_traj, list) and len(neighbor_traj) > 1:
                        neigh_array = np.array(neighbor_traj, dtype=np.float64)
                        ax.plot(neigh_array[:, 0], neigh_array[:, 1], '-', color='#CCCCCC',
                               linewidth=2, alpha=0.7)

                # Plot person
                if obs_xy.size > 0:
                    ax.plot(obs_xy[:, 0], obs_xy[:, 1], 'b-', linewidth=3.5)
                    ax.scatter(obs_xy[0, 0], obs_xy[0, 1], c='blue', s=300, marker='o',
                              edgecolor='black', linewidth=2, zorder=10)
                    ax.scatter(obs_xy[-1, 0], obs_xy[-1, 1], c='blue', s=300, marker='s',
                              edgecolor='black', linewidth=2, zorder=10)

                # Plot future
                if gt_xy.size > 0:
                    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'g--', linewidth=2.5, alpha=0.8)

                # Plot robot
                if robot_xy.size > 0:
                    ax.plot(robot_xy[:, 0], robot_xy[:, 1], 'r-', linewidth=3.5)
                    ax.scatter(robot_xy[0, 0], robot_xy[0, 1], c='red', s=280, marker='^',
                              edgecolor='darkred', linewidth=2, zorder=9)
                    ax.scatter(robot_xy[-1, 0], robot_xy[-1, 1], c='darkred', s=280, marker='v',
                              edgecolor='darkred', linewidth=2, zorder=9)

                tags_str = ', '.join([t for t in case.get('tags', []) if t != 'all'][:2])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{cat.upper()} #{ex_idx+1}\n{tags_str}", fontsize=11, fontweight='bold')
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
            else:
                ax.text(0.5, 0.5, "Not enough\nexamples", ha='center', va='center',
                       transform=ax.transAxes, fontsize=11, color='orange')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_raw(output_path: Path):
    """Visualize raw merged dataset."""
    print("\n" + "="*70)
    print("RAW MERGED DATASET")
    print("="*70)

    raw_path = Path("./datasets/raw_react_0p5/people_dataset_merged_react_0p5.json")

    with open(raw_path) as f:
        data = json.load(f)

    frames = data['frames']
    print(f"Loaded {len(frames)} frames")

    obs_len = 8
    pred_len = 12
    display_obs_len = 30  # Show longer trajectory context

    examples_by_cat = {cat: [] for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]}

    # Collect person IDs and their frames
    person_frames = {}
    for frame_idx, frame in enumerate(frames):
        for person in frame.get('people', []):
            pid = person.get('id')
            if pid not in person_frames:
                person_frames[pid] = []
            person_frames[pid].append((frame_idx, frame))

    print(f"Found {len(person_frames)} unique people")

    # Find examples
    for pid, pframes in person_frames.items():
        pframes.sort(key=lambda x: x[0])

        for i in range(len(pframes) - obs_len - pred_len):
            # Get longer display window (context before obs)
            display_start = max(0, i - display_obs_len)
            obs_start_in_display = i - display_start

            display_frames = pframes[display_start:i+obs_len]
            pred_frames = pframes[i+obs_len:i+obs_len+pred_len]

            # Extract FULL display trajectory
            display_xy = []
            robot_display_xy = []
            for frame_idx, frame in display_frames:
                for p in frame.get('people', []):
                    if p.get('id') == pid:
                        x, y = p.get('x'), p.get('y')
                        if x is not None and y is not None:
                            display_xy.append([x, y])
                        break

                robot = frame.get('robot', {})
                robot_display_xy.append([robot.get('x', 0), robot.get('y', 0)])

            if len(display_xy) != len(display_frames):
                continue

            display_xy = np.array(display_xy, dtype=np.float64)
            robot_display_xy = np.array(robot_display_xy, dtype=np.float64)

            # Extract actual obs window for metric calculation
            obs_xy = display_xy[obs_start_in_display:obs_start_in_display+obs_len]
            robot_obs_xy = robot_display_xy[obs_start_in_display:obs_start_in_display+obs_len]
            obs_end = obs_start_in_display + obs_len

            if len(obs_xy) != obs_len:
                continue

            # Check movement
            if not has_movement(obs_xy, robot_obs_xy):
                continue

            # Extract pred trajectory
            gt_xy = []
            for frame_idx, frame in pred_frames:
                for p in frame.get('people', []):
                    if p.get('id') == pid:
                        x, y = p.get('x'), p.get('y')
                        if x is not None and y is not None:
                            gt_xy.append([x, y])
                        break

            gt_xy = np.array(gt_xy, dtype=np.float64) if gt_xy else np.zeros((0, 2))

            # Get neighbors at final obs frame
            final_frame = display_frames[obs_end - 1][1]
            final_x, final_y = obs_xy[-1]
            neigh_xy = []

            for p in final_frame.get('people', []):
                oid = p.get('id')
                if oid != pid:
                    ox, oy = p.get('x'), p.get('y')
                    if ox is not None and oy is not None:
                        # Get full neighbor trajectory
                        neigh_traj = []
                        for frame_idx, frame in display_frames:
                            for op in frame.get('people', []):
                                if op.get('id') == oid:
                                    nx, ny = op.get('x'), op.get('y')
                                    if nx is not None and ny is not None:
                                        neigh_traj.append([nx, ny])
                                    break
                        if neigh_traj:
                            neigh_xy.append(neigh_traj)

            # Classify
            heading_deg = 0.0
            if obs_xy.shape[0] >= 3:
                step = obs_xy[1:] - obs_xy[:-1]
                ang = np.arctan2(step[:, 1], step[:, 0])
                if ang.shape[0] >= 2:
                    d = np.diff(ang)
                    d = (d + np.pi) % (2.0 * np.pi) - np.pi
                    heading_deg = math.degrees(float(np.sum(np.abs(d))))

            speeds = np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1) / 0.1
            min_speed = float(np.min(speeds)) if speeds.size > 0 else 0.0
            max_speed = float(np.max(speeds)) if speeds.size > 0 else 0.0

            neighbor_count = len(neigh_xy)
            min_neighbor_dist = float('inf')
            for neigh_traj in neigh_xy:
                neigh_array = np.array(neigh_traj)
                dist = np.linalg.norm(neigh_array[-1] - obs_xy[-1])
                min_neighbor_dist = min(min_neighbor_dist, dist)

            tags = []
            if heading_deg < 30:
                tags.append("linear")
            if heading_deg >= 45:
                tags.append("turning")
            if max_speed >= 0.25 and min_speed <= 0.10 and (max_speed - min_speed) >= 0.25:
                tags.append("stop_go")
            if 1 <= neighbor_count <= 3 and min_neighbor_dist <= 1.5:
                tags.append("interaction")
            if neighbor_count >= 4 and min_neighbor_dist <= 1.5:
                tags.append("dense_interaction")

            complexity_axes = sum([int(t in tags) for t in ["turning", "stop_go", "interaction", "dense_interaction"]])
            if complexity_axes >= 2:
                tags.append("complex")

            cat = next((t for t in ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"] if t in tags), "unknown")

            if cat in examples_by_cat and len(examples_by_cat[cat]) < 4:
                examples_by_cat[cat].append({
                    'obs_xy': obs_xy,
                    'display_xy': display_xy,
                    'robot_display_xy': robot_display_xy,
                    'obs_start_idx': obs_start_in_display,
                    'obs_end_idx': obs_start_in_display + obs_len,
                    'gt_xy': gt_xy,
                    'neigh_xy': neigh_xy,
                    'tags': tags,
                })

    print(f"Found moving examples:")
    for cat, exs in examples_by_cat.items():
        print(f"  {cat:20s}: {len(exs)}/4")

    # Plot
    fig, axes = plt.subplots(6, 4, figsize=(24, 20))
    fig.suptitle("RAW Merged Dataset - Trajectory Examples\nBlue=Person, Green=Future, Red=Robot, Gray=Others (Lines)",
                fontsize=16, fontweight='bold')

    categories = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]

    for cat_idx, cat in enumerate(categories):
        for ex_idx in range(4):
            ax = axes[cat_idx, ex_idx]
            exs = examples_by_cat[cat]

            if ex_idx < len(exs):
                ex = exs[ex_idx]

                display_xy = ex['display_xy']
                robot_display_xy = ex['robot_display_xy']
                obs_start = ex['obs_start_idx']
                obs_end = ex['obs_end_idx']
                gt_xy = ex['gt_xy']
                neigh_xy = ex['neigh_xy']

                # Plot neighbors (make less visible)
                for neigh_traj in neigh_xy:
                    neigh_array = np.array(neigh_traj, dtype=np.float64)
                    ax.plot(neigh_array[:, 0], neigh_array[:, 1], '-', color='#EEEEEE',
                           linewidth=1, alpha=0.4)

                # Plot full trajectory (faint)
                ax.plot(display_xy[:, 0], display_xy[:, 1], 'b-', linewidth=2, alpha=0.3)
                ax.plot(robot_display_xy[:, 0], robot_display_xy[:, 1], 'r-', linewidth=2, alpha=0.3)

                # Highlight observation window
                obs_segment = display_xy[obs_start:obs_end]
                robot_obs_segment = robot_display_xy[obs_start:obs_end]

                ax.plot(obs_segment[:, 0], obs_segment[:, 1], 'b-', linewidth=3.5)
                ax.scatter(obs_segment[0, 0], obs_segment[0, 1], c='blue', s=300, marker='o',
                          edgecolor='black', linewidth=2, zorder=10)
                ax.scatter(obs_segment[-1, 0], obs_segment[-1, 1], c='blue', s=300, marker='s',
                          edgecolor='black', linewidth=2, zorder=10)

                # Plot future
                if gt_xy.size > 0:
                    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'g--', linewidth=2.5, alpha=0.8)

                # Plot robot
                ax.plot(robot_obs_segment[:, 0], robot_obs_segment[:, 1], 'r-', linewidth=3.5)
                ax.scatter(robot_obs_segment[0, 0], robot_obs_segment[0, 1], c='red', s=280, marker='^',
                          edgecolor='darkred', linewidth=2, zorder=9)
                ax.scatter(robot_obs_segment[-1, 0], robot_obs_segment[-1, 1], c='darkred', s=280, marker='v',
                          edgecolor='darkred', linewidth=2, zorder=9)

                tags_str = ', '.join(ex['tags'][:2])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{cat.upper()} #{ex_idx+1}\n{tags_str}", fontsize=11, fontweight='bold')
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
            else:
                ax.text(0.5, 0.5, "Not enough\nexamples", ha='center', va='center',
                       transform=ax.transAxes, fontsize=11, color='orange')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    visualize_benchmark(Path("./docs/figures/analysis/examples_benchmark.png"))
    visualize_raw(Path("./docs/figures/analysis/examples_raw_merged.png"))

    print("\n" + "="*70)
    print("Generated files:")
    print("  ./docs/figures/analysis/examples_benchmark.png")
    print("  ./docs/figures/analysis/examples_raw_merged.png")
    print("="*70)
