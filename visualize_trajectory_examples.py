#!/usr/bin/env python3
"""Visualize trajectory examples from different interaction categories."""
from __future__ import annotations

import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def get_person_trajectory(
    frames: List[Dict[str, Any]],
    person_id: int,
    start_frame_idx: int,
    obs_len: int = 8,
    pred_len: int = 12,
    obs_dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract person trajectory segments: obs_xy, pred_xy, robot_xy."""

    # Collect all frames for this person
    person_frames = []
    for frame_idx, frame in enumerate(frames):
        for person in frame.get('people', []):
            if person.get('id') == person_id:
                x, y = person.get('x'), person.get('y')
                if x is not None and y is not None:
                    person_frames.append((frame_idx, x, y, frame))
                break

    if len(person_frames) < start_frame_idx + obs_len + pred_len:
        return None, None, None, {}

    # Get observation window
    obs_xy = np.array([
        [person_frames[start_frame_idx + i][1], person_frames[start_frame_idx + i][2]]
        for i in range(obs_len)
    ], dtype=np.float64)

    # Get prediction window
    pred_xy = np.array([
        [person_frames[start_frame_idx + obs_len + i][1], person_frames[start_frame_idx + obs_len + i][2]]
        for i in range(min(pred_len, len(person_frames) - start_frame_idx - obs_len))
    ], dtype=np.float64)

    # Get robot trajectory during observation window
    robot_xy = []
    for i in range(obs_len):
        frame = person_frames[start_frame_idx + i][3]
        robot_data = frame.get('robot', {})
        robot_xy.append([robot_data.get('x', 0), robot_data.get('y', 0)])
    robot_xy = np.array(robot_xy, dtype=np.float64)

    # Get nearby people during final observation frame
    frame = person_frames[start_frame_idx + obs_len - 1][3]
    final_x, final_y = obs_xy[-1]

    nearby_people = []
    for person in frame.get('people', []):
        if person.get('id') != person_id:
            x, y = person.get('x'), person.get('y')
            if x is not None and y is not None:
                nearby_people.append((x, y))

    metrics = {
        'nearby_people_count': len(nearby_people),
        'nearby_people': nearby_people,
    }

    return obs_xy, pred_xy, robot_xy, metrics


def classify_trajectory(obs_xy: np.ndarray, nearby_people: List[Tuple[float, float]],
                       interaction_dist: float = 1.5) -> str:
    """Classify a trajectory based on metrics."""

    # Heading change
    if obs_xy.shape[0] >= 3:
        step = obs_xy[1:] - obs_xy[:-1]
        ang = np.arctan2(step[:, 1], step[:, 0])
        if ang.shape[0] >= 2:
            d = np.diff(ang)
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            heading_change_deg = math.degrees(float(np.sum(np.abs(d))))
        else:
            heading_change_deg = 0.0
    else:
        heading_change_deg = 0.0

    # Speed
    obs_dt = 0.1
    speeds = np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1) / obs_dt
    min_speed = float(np.min(speeds)) if speeds.size > 0 else 0.0
    max_speed = float(np.max(speeds)) if speeds.size > 0 else 0.0

    # Neighbors
    neighbor_count = 0
    min_neighbor_dist = float('inf')
    final_pos = obs_xy[-1]
    for nx, ny in nearby_people:
        dist = math.sqrt((final_pos[0] - nx)**2 + (final_pos[1] - ny)**2)
        if dist <= interaction_dist:
            neighbor_count += 1
        min_neighbor_dist = min(min_neighbor_dist, dist)

    tags = set()
    if heading_change_deg < 30.0:
        tags.add("linear")
    if heading_change_deg >= 45.0:
        tags.add("turning")
    if max_speed >= 0.25 and min_speed <= 0.10 and (max_speed - min_speed) >= 0.25:
        tags.add("stop_go")
    if 1 <= neighbor_count <= 3 and min_neighbor_dist <= interaction_dist:
        tags.add("interaction")
    if neighbor_count >= 4 and min_neighbor_dist <= interaction_dist:
        tags.add("dense_interaction")

    complexity_axes = sum([
        int("turning" in tags),
        int("stop_go" in tags),
        int("interaction" in tags),
        int("dense_interaction" in tags),
    ])
    if complexity_axes >= 2:
        tags.add("complex")

    # Return primary tag
    priority = ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"]
    for tag in priority:
        if tag in tags:
            return tag
    return "unknown"


def plot_trajectory_example(ax, obs_xy: np.ndarray, pred_xy: np.ndarray,
                           robot_xy: np.ndarray, nearby_people: List[Tuple[float, float]],
                           title: str):
    """Plot a single trajectory example."""

    # Plot nearby people (static)
    if nearby_people:
        nearby_xy = np.array(nearby_people)
        ax.scatter(nearby_xy[:, 0], nearby_xy[:, 1], c='gray', s=50, alpha=0.5, label='Others')

    # Plot observation trajectory (main person)
    ax.plot(obs_xy[:, 0], obs_xy[:, 1], 'b-', linewidth=2, label='Observation (8 frames)')
    ax.scatter(obs_xy[:, 0], obs_xy[:, 1], c='blue', s=30)

    # Plot prediction trajectory
    if pred_xy.size > 0:
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'g--', linewidth=2, label='Future (12 frames)')
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], c='green', s=20)

    # Plot robot trajectory
    ax.plot(robot_xy[:, 0], robot_xy[:, 1], 'r-', linewidth=2.5, label='Robot')
    ax.scatter(robot_xy[0, 0], robot_xy[0, 1], c='red', s=150, marker='^', label='Robot start')
    ax.scatter(robot_xy[-1, 0], robot_xy[-1, 1], c='darkred', s=150, marker='v', label='Robot end')

    # Mark start and end of main person
    ax.scatter(obs_xy[0, 0], obs_xy[0, 1], c='blue', s=200, marker='o', edgecolor='black', linewidth=2)
    ax.scatter(obs_xy[-1, 0], obs_xy[-1, 1], c='blue', s=200, marker='s', edgecolor='black', linewidth=2)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')


def find_examples_by_category(frames: List[Dict[str, Any]], num_examples: int = 4) -> Dict[str, List[Tuple[int, int]]]:
    """Find example trajectories for each category."""

    # Collect all valid trajectories
    all_trajectories = []
    person_ids = set()
    for frame in frames:
        for person in frame.get('people', []):
            person_ids.add(person.get('id'))

    obs_len = 8
    pred_len = 12

    for person_id in list(person_ids)[:50]:  # Limit search
        for frame_idx in range(len(frames) - obs_len - pred_len):
            obs_xy, pred_xy, robot_xy, metrics = get_person_trajectory(
                frames, person_id, frame_idx, obs_len, pred_len
            )

            if obs_xy is None:
                continue

            nearby_people = metrics.get('nearby_people', [])
            category = classify_trajectory(obs_xy, nearby_people)

            all_trajectories.append((category, person_id, frame_idx, obs_xy, pred_xy, robot_xy, nearby_people))

    # Group by category and select examples
    by_category = {}
    for category, person_id, frame_idx, obs_xy, pred_xy, robot_xy, nearby_people in all_trajectories:
        if category not in by_category:
            by_category[category] = []
        if len(by_category[category]) < num_examples:
            by_category[category].append((person_id, frame_idx, obs_xy, pred_xy, robot_xy, nearby_people))

    return by_category


def main():
    # Load both datasets
    benchmark_curated = Path("/home/danbel1kov/predictive-nav-mppi/benchmark_people_predictors/curated_benchmark/benchmark_cases.json")
    current_merged = Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_merged_react_0p5.json")

    datasets = [
        ("Benchmark (curated)", benchmark_curated),
        ("Current (merged raw)", current_merged),
    ]

    for dataset_name, dataset_path in datasets:
        print(f"\nProcessing {dataset_name}...")

        if not dataset_path.exists():
            print(f"  ❌ Not found: {dataset_path}")
            continue

        with open(dataset_path) as f:
            data = json.load(f)

        # For benchmark: cases format, for raw: frames format
        if 'cases' in data:
            print(f"  ⚠️  Benchmark format not yet supported (need to extract frames)")
            continue

        frames = data.get('frames', [])
        if not frames:
            print(f"  ❌ No frames found")
            continue

        print(f"  Found {len(frames)} frames")

        # Find examples
        by_category = find_examples_by_category(frames, num_examples=4)

        # Create visualization
        categories = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]
        fig, axes = plt.subplots(len(categories), 4, figsize=(20, 16))
        fig.suptitle(f"{dataset_name}\nBlue=Person obs (8f), Green=Future (12f), Red=Robot, Gray=Others",
                    fontsize=14, fontweight='bold')

        for cat_idx, category in enumerate(categories):
            examples = by_category.get(category, [])

            for ex_idx in range(4):
                ax = axes[cat_idx, ex_idx]

                if ex_idx < len(examples):
                    person_id, frame_idx, obs_xy, pred_xy, robot_xy, nearby_people = examples[ex_idx]
                    plot_trajectory_example(ax, obs_xy, pred_xy, robot_xy, nearby_people,
                                          f"{category} #{ex_idx+1}\nPerson {person_id}")
                else:
                    ax.text(0.5, 0.5, f"No example\nfound", ha='center', va='center',
                           transform=ax.transAxes, fontsize=10, color='red')
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)

        # Save figure
        output_path = Path(f"/tmp/{dataset_name.replace(' ', '_').replace('(', '').replace(')', '')}_examples.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


if __name__ == "__main__":
    main()
