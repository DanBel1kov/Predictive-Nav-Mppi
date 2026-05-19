#!/usr/bin/env python3
"""Analyze merged dataset - only observations near robot."""
from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def compute_full_metrics_near_robot(
    frames: List[Dict[str, Any]],
    interaction_dist: float = 1.5,
    visibility_radius: float = 6.0,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Compute metrics for observations near robot only."""

    # Time-indexed person positions for neighbor lookup
    time_indexed = {}
    for frame in frames:
        t = frame.get('t')
        if t not in time_indexed:
            time_indexed[t] = {}
        for person in frame.get('people', []):
            pid = person.get('id')
            x, y = person.get('x'), person.get('y')
            if pid is not None and x is not None and y is not None:
                time_indexed[t][pid] = (x, y)

    # Build person trajectories
    person_data: Dict[int, List[Tuple[float, float, float, int]]] = {}

    for frame_idx, frame in enumerate(frames):
        t = frame.get('t', 0)
        for person in frame.get('people', []):
            pid = person.get('id')
            x, y = person.get('x'), person.get('y')
            if pid is not None and x is not None and y is not None:
                if pid not in person_data:
                    person_data[pid] = []
                person_data[pid].append((t, x, y, frame_idx))

    metrics = {}
    obs_len = 8
    obs_dt = 0.1
    near_robot_count = 0
    total_count = 0

    for pid, trajectory in person_data.items():
        trajectory.sort(key=lambda x: x[0])

        if len(trajectory) < obs_len:
            continue

        # Extract observation windows
        for i in range(obs_len, len(trajectory)):
            obs_xy = np.array([[t[1], t[2]] for t in trajectory[i-obs_len:i]], dtype=np.float64)

            if obs_xy.shape[0] < 2:
                continue

            total_count += 1

            # Get final position and corresponding frame
            final_t = trajectory[i][0]
            final_pos = obs_xy[-1]
            frame_idx = trajectory[i][3]

            # Find robot position at this time
            frame = frames[frame_idx]
            robot_data = frame.get('robot', {})
            robot_x = robot_data.get('x')
            robot_y = robot_data.get('y')

            if robot_x is None or robot_y is None:
                continue

            # Check if near robot
            dist_to_robot = math.sqrt((final_pos[0] - robot_x)**2 + (final_pos[1] - robot_y)**2)
            if dist_to_robot > visibility_radius:
                continue

            near_robot_count += 1

            # Compute metrics
            path_len = float(np.sum(np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1)))
            displacement = float(np.linalg.norm(obs_xy[-1] - obs_xy[0]))

            # Heading change
            heading_change_deg = 0.0
            if obs_xy.shape[0] >= 3:
                step = obs_xy[1:] - obs_xy[:-1]
                ang = np.arctan2(step[:, 1], step[:, 0])
                if ang.shape[0] >= 2:
                    d = np.diff(ang)
                    d = (d + np.pi) % (2.0 * np.pi) - np.pi
                    heading_change_deg = math.degrees(float(np.sum(np.abs(d))))

            # Speed
            speeds = np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1) / obs_dt
            min_speed = float(np.min(speeds)) if speeds.size > 0 else 0.0
            max_speed = float(np.max(speeds)) if speeds.size > 0 else 0.0
            mean_speed = float(np.mean(speeds)) if speeds.size > 0 else 0.0

            # Neighbor detection
            neighbors = []
            if final_t in time_indexed:
                positions = time_indexed[final_t]
                for other_pid, (other_x, other_y) in positions.items():
                    if other_pid != pid:
                        dist = math.sqrt((final_pos[0] - other_x)**2 + (final_pos[1] - other_y)**2)
                        neighbors.append((dist, other_pid))

            neighbors.sort()
            min_neighbor_dist = neighbors[0][0] if neighbors else float('inf')
            neighbor_count = sum(1 for d, _ in neighbors if d <= interaction_dist)

            key = (pid, i)
            metrics[key] = {
                'heading_change_deg': heading_change_deg,
                'min_speed': min_speed,
                'max_speed': max_speed,
                'mean_speed': mean_speed,
                'path_len': path_len,
                'displacement': displacement,
                'neighbor_count': neighbor_count,
                'min_neighbor_dist': min_neighbor_dist if math.isfinite(min_neighbor_dist) else 999.0,
                'dist_to_robot': dist_to_robot,
                'frame_idx': frame_idx,
            }

    return metrics, near_robot_count, total_count


def classify_interaction(obs_metrics: Dict[str, Any], interaction_dist: float = 1.5) -> Set[str]:
    """Classify interaction category."""
    tags = {"all"}

    heading_change_deg = obs_metrics.get('heading_change_deg', 0.0)
    min_speed = obs_metrics.get('min_speed', 0.0)
    max_speed = obs_metrics.get('max_speed', 0.0)
    neighbor_count = obs_metrics.get('neighbor_count', 0)
    min_neighbor_dist = obs_metrics.get('min_neighbor_dist', 999.0)

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

    complexity_axes = (
        int("turning" in tags)
        + int("stop_go" in tags)
        + int("interaction" in tags)
        + int("dense_interaction" in tags)
    )
    if complexity_axes >= 2:
        tags.add("complex")

    return tags


def main():
    merged_path = Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_merged_react_0p5.json")

    print("Loading merged dataset...")
    with open(merged_path) as f:
        data = json.load(f)

    frames = data['frames']
    print(f"Loaded {len(frames)} frames")

    print("\nComputing metrics (near robot only, radius=6.0m)...")
    metrics, near_robot_count, total_count = compute_full_metrics_near_robot(frames)
    print(f"Total observations: {total_count}")
    print(f"Near robot: {near_robot_count} ({100*near_robot_count/total_count:.1f}%)")
    print(f"With valid metrics: {len(metrics)}")

    # Classify
    print("\nClassifying interactions...")
    tag_counts: Dict[str, int] = {}
    for obs_metrics in metrics.values():
        tags = classify_interaction(obs_metrics)
        for tag in tags:
            if tag != "all":
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    total = sum(tag_counts.values())

    print(f"\n{'='*70}")
    print(f"Interaction distribution (observations near robot)")
    print(f"Total: {total} observations")
    print(f"{'='*70}")

    order = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]
    for tag in order:
        if tag in tag_counts:
            count = tag_counts[tag]
            pct = 100.0 * count / total if total > 0 else 0.0
            print(f"  {tag:20s}: {count:7d} ({pct:5.1f}%)")

    # Comparison with benchmark
    print(f"\n{'='*70}")
    print("Comparison with benchmark (force=0.5):")
    print(f"{'='*70}")
    benchmark = {
        'linear': 4.9,
        'interaction': 12.2,
        'dense_interaction': 8.3,
        'turning': 22.2,
        'stop_go': 26.4,
        'complex': 25.9,
    }
    for tag in order:
        if tag in tag_counts:
            current_pct = 100.0 * tag_counts[tag] / total
            benchmark_pct = benchmark.get(tag, 0)
            diff = current_pct - benchmark_pct
            bar = "█" * max(0, int(diff/2)) if diff > 0 else "▓" * max(0, int(-diff/2))
            print(f"  {tag:20s}: {current_pct:5.1f}% (bench {benchmark_pct:5.1f}%) {bar}")

    # Save analysis
    analysis = {
        'total_observations': len(metrics),
        'near_robot_observations': near_robot_count,
        'total_frames': len(frames),
        'unique_people': len(set(k[0] for k in metrics.keys())),
        'tag_counts': tag_counts,
        'duration_seconds': frames[-1]['t'] if frames else 0,
        'visibility_radius': 6.0,
        'tag_order': order,
    }

    output_path = merged_path.parent / "analysis_merged_near_robot_react_0p5.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Analysis saved: {output_path}")


if __name__ == "__main__":
    main()
