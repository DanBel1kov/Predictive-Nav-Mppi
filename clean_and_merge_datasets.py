#!/usr/bin/env python3
"""Clean datasets by removing reset artifacts and merging multiple scenarios."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple


def split_by_resets(frames: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split frames into segments separated by large backward time jumps."""
    if not frames:
        return []

    segments = []
    current_segment = [frames[0]]

    for i in range(1, len(frames)):
        time_diff = frames[i]['t'] - frames[i-1]['t']
        if time_diff < -0.5:  # Large backward jump
            segments.append(current_segment)
            current_segment = [frames[i]]
        else:
            current_segment.append(frames[i])

    if current_segment:
        segments.append(current_segment)

    return segments


def clean_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove frames with suspicious timing or invalid data."""
    cleaned = []

    for i, frame in enumerate(frames):
        # Skip if time is invalid
        if not isinstance(frame.get('t'), (int, float)) or not np.isfinite(frame['t']):
            continue

        # Skip if no people data
        people = frame.get('people', [])
        if not people:
            continue

        # Check people validity
        valid_people = []
        for person in people:
            x, y = person.get('x'), person.get('y')
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                valid_people.append(person)

        if valid_people:
            frame['people'] = valid_people
            cleaned.append(frame)

    return cleaned


def merge_scenarios(scenario_files: List[Tuple[str, Path]]) -> Dict[str, Any]:
    """Merge multiple scenario datasets into one."""
    merged_frames = []
    meta_info = {}

    for scenario_name, file_path in scenario_files:
        print(f"Loading {scenario_name}...")
        with open(file_path) as f:
            data = json.load(f)

        frames = data.get('frames', [])
        meta = data.get('meta', {})

        # Split by resets
        segments = split_by_resets(frames)
        print(f"  Found {len(segments)} segment(s)")

        # Use all segments, clean them
        for seg_idx, segment in enumerate(segments):
            cleaned = clean_frames(segment)
            print(f"  Segment {seg_idx}: {len(segment)} → {len(cleaned)} frames")
            merged_frames.extend(cleaned)

        # Store metadata
        meta_info[scenario_name] = {
            'original_frames': len(frames),
            'segments': len(segments),
            'meta': meta
        }

    # Sort by time
    merged_frames.sort(key=lambda f: f['t'])

    # Renormalize time to start from 0
    if merged_frames:
        t_start = merged_frames[0]['t']
        for frame in merged_frames:
            frame['t'] -= t_start

    return {
        'meta': {
            'source': 'merged_scenarios',
            'original_meta': meta_info,
            'total_frames': len(merged_frames),
        },
        'frames': merged_frames
    }


def main():
    scenarios = [
        ("long_corridor", Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_long_corridor_react_0p5.json")),
        ("nonlinear_corridor", Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_nonlinear_corridor_react_0p5.json")),
        ("labyrinth_turns", Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_labyrinth_turns_react_0p5.json")),
    ]

    print("Cleaning and merging datasets...")
    merged = merge_scenarios(scenarios)

    # Save merged dataset
    output_path = Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_merged_react_0p5.json")
    with open(output_path, 'w') as f:
        json.dump(merged, f)

    print(f"\n✓ Merged dataset saved: {output_path}")
    print(f"  Total frames: {len(merged['frames'])}")
    print(f"  Duration: {merged['frames'][-1]['t']:.2f}s")

    # Report statistics
    all_people = {}
    for frame in merged['frames']:
        for person in frame.get('people', []):
            pid = person.get('id')
            if pid not in all_people:
                all_people[pid] = 0
            all_people[pid] += 1

    frames_per_person = list(all_people.values())
    print(f"  Unique people: {len(all_people)}")
    print(f"  Frames per person: min={min(frames_per_person)}, max={max(frames_per_person)}, mean={np.mean(frames_per_person):.0f}")


if __name__ == "__main__":
    main()
