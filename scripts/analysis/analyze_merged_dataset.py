#!/usr/bin/env python3
"""Analyze merged dataset and compute interaction statistics."""
from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def compute_metrics(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Compute trajectory metrics for each person."""
    person_data: Dict[int, List[Tuple[float, float, float, float]]] = {}  # pid -> [(t, x, y, obs_idx)]

    for frame_idx, frame in enumerate(frames):
        t = frame.get('t', 0)
        for person in frame.get('people', []):
            pid = person.get('id')
            x, y = person.get('x'), person.get('y')
            if pid is not None and x is not None and y is not None:
                if pid not in person_data:
                    person_data[pid] = []
                person_data[pid].append((t, x, y, frame_idx))

    # Compute metrics for continuous segments
    metrics = {}
    obs_len = 8  # observation window
    obs_dt = 0.1  # ~100ms per frame

    for pid, trajectory in person_data.items():
        # Sort by time
        trajectory.sort(key=lambda x: x[0])

        if len(trajectory) < obs_len:
            continue

        # Extract observation windows
        for i in range(obs_len, len(trajectory)):
            obs_xy = np.array([[t[1], t[2]] for t in trajectory[i-obs_len:i]], dtype=np.float64)

            # Compute metrics
            if obs_xy.shape[0] < 2:
                continue

            path_len = float(np.sum(np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1)))
            displacement = float(np.linalg.norm(obs_xy[-1] - obs_xy[0]))

            # Heading change
            if obs_xy.shape[0] >= 3:
                step = obs_xy[1:] - obs_xy[:-1]
                ang = np.arctan2(step[:, 1], step[:, 0])
                if ang.shape[0] >= 2:
                    d = np.diff(ang)
                    d = (d + np.pi) % (2.0 * np.pi) - np.pi
                    heading_change_rad = float(np.sum(np.abs(d)))
                    heading_change_deg = math.degrees(heading_change_rad)
                else:
                    heading_change_deg = 0.0
            else:
                heading_change_deg = 0.0

            # Speed
            speeds = np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1) / obs_dt
            min_speed = float(np.min(speeds)) if speeds.size > 0 else 0.0
            max_speed = float(np.max(speeds)) if speeds.size > 0 else 0.0
            mean_speed = float(np.mean(speeds)) if speeds.size > 0 else 0.0

            key = (pid, i)
            metrics[key] = {
                'heading_change_deg': heading_change_deg,
                'min_speed': min_speed,
                'max_speed': max_speed,
                'mean_speed': mean_speed,
                'path_len': path_len,
                'displacement': displacement,
                'frame_idx': trajectory[i][3],
            }

    return metrics


def classify_interaction(person_metrics: Dict[str, Any]) -> Set[str]:
    """Classify interaction category based on metrics and context."""
    tags = {"all"}

    heading_change_deg = person_metrics.get('heading_change_deg', 0.0)
    min_speed = person_metrics.get('min_speed', 0.0)
    max_speed = person_metrics.get('max_speed', 0.0)

    # LINEAR: continuous movement in one direction (small heading changes)
    if heading_change_deg < 30.0:
        tags.add("linear")

    # TURNING: large heading change
    if heading_change_deg >= 45.0:
        tags.add("turning")

    # STOP_GO: significant speed variations
    if max_speed >= 0.25 and min_speed <= 0.10 and (max_speed - min_speed) >= 0.25:
        tags.add("stop_go")

    return tags


def main():
    merged_path = Path("./datasets/raw_react_0p5/people_dataset_merged_react_0p5.json")

    print("Loading merged dataset...")
    with open(merged_path) as f:
        data = json.load(f)

    frames = data['frames']
    print(f"Loaded {len(frames)} frames")

    print("\nComputing metrics...")
    metrics = compute_metrics(frames)
    print(f"Computed metrics for {len(metrics)} observations")

    # Classify
    print("\nClassifying interactions...")
    tag_counts: Dict[str, int] = {}
    for key, obs_metrics in metrics.items():
        tags = classify_interaction(obs_metrics)
        for tag in tags:
            if tag != "all":
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    total = sum(tag_counts.values())
    print(f"\nInteraction category distribution (total: {total}):")
    for tag in sorted(tag_counts.keys()):
        count = tag_counts[tag]
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {tag:20s}: {count:6d} ({pct:5.1f}%)")

    # Cases with no special tags
    no_tag = len(metrics) - total
    if no_tag > 0:
        pct = 100.0 * no_tag / len(metrics)
        print(f"  {'(other)':20s}: {no_tag:6d} ({pct:5.1f}%)")

    # Save analysis
    analysis = {
        'total_observations': len(metrics),
        'total_frames': len(frames),
        'unique_people': len(set(k[0] for k in metrics.keys())),
        'tag_counts': tag_counts,
        'duration_seconds': frames[-1]['t'] if frames else 0,
    }

    output_path = merged_path.parent / "analysis_merged_react_0p5.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Analysis saved: {output_path}")


if __name__ == "__main__":
    main()
