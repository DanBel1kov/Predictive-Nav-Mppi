#!/usr/bin/env python3
"""Visualize trajectory examples from benchmark dataset."""
from __future__ import annotations

import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def classify_case(case: Dict) -> str:
    """Classify case based on tags."""
    tags = set(case.get('tags', []))

    # Remove 'all' tag
    tags.discard('all')

    # Return primary tag
    priority = ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"]
    for tag in priority:
        if tag in tags:
            return tag
    return "unknown"


def main():
    benchmark_cases = Path("./benchmark_force_sweep/kalman_react_force_0_025_05_near_robot3/force_0p25/20260513_005606/curated_near_robot/benchmark_cases.json")

    print("Loading benchmark dataset...")
    with open(benchmark_cases) as f:
        data = json.load(f)

    cases = data.get('cases', [])
    print(f"Loaded {len(cases)} cases")

    examples_by_cat = {cat: [] for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]}

    # Classify and collect examples
    for case in cases:
        cat = classify_case(case)
        if cat in examples_by_cat and len(examples_by_cat[cat]) < 4:
            examples_by_cat[cat].append(case)

    print(f"Found examples:")
    for cat, exs in examples_by_cat.items():
        print(f"  {cat:20s}: {len(exs)}/4")

    # Plot
    fig, axes = plt.subplots(6, 4, figsize=(24, 20))
    fig.suptitle("BENCHMARK Dataset - Trajectory Examples\n" +
                "Blue circle=Person start, Blue square=Person end, Green=Future pred, Red=Robot, Gray=Other people",
                fontsize=16, fontweight='bold')

    categories = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]

    for cat_idx, cat in enumerate(categories):
        for ex_idx in range(4):
            ax = axes[cat_idx, ex_idx]
            exs = examples_by_cat[cat]

            if ex_idx < len(exs):
                case = exs[ex_idx]

                # Extract data
                obs_xy = np.array(case.get('obs_xy', []), dtype=np.float64)
                pred_xy = np.array(case.get('pred_xy', []), dtype=np.float64)
                robot_xy = np.array(case.get('robot_obs_xy', []), dtype=np.float64)
                neigh_xy = case.get('neigh_xy', [])

                # Plot other people
                for neighbor_traj in neigh_xy:
                    if isinstance(neighbor_traj, list) and len(neighbor_traj) > 1:
                        neigh_array = np.array(neighbor_traj, dtype=np.float64)
                        ax.plot(neigh_array[:, 0], neigh_array[:, 1], 'o-', color='lightgray',
                               linewidth=1.5, markersize=3, alpha=0.6)

                # Plot main person trajectory
                if obs_xy.size > 0:
                    ax.plot(obs_xy[:, 0], obs_xy[:, 1], 'b-', linewidth=3, label='Person obs', zorder=5)
                    ax.scatter(obs_xy[0, 0], obs_xy[0, 1], c='blue', s=250, marker='o',
                              edgecolor='black', linewidth=2, zorder=10, label='Start')
                    ax.scatter(obs_xy[-1, 0], obs_xy[-1, 1], c='blue', s=250, marker='s',
                              edgecolor='black', linewidth=2, zorder=10, label='End obs')

                # Plot prediction
                if pred_xy.size > 0:
                    ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'g--', linewidth=3, label='Future pred', zorder=4)
                    ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], c='green', s=150, marker='*', zorder=9)

                # Plot robot
                if robot_xy.size > 0:
                    ax.plot(robot_xy[:, 0], robot_xy[:, 1], 'r-', linewidth=3, label='Robot path', zorder=3)
                    ax.scatter(robot_xy[0, 0], robot_xy[0, 1], c='red', s=200, marker='^',
                              edgecolor='darkred', linewidth=2, zorder=9, label='Robot start')
                    ax.scatter(robot_xy[-1, 0], robot_xy[-1, 1], c='darkred', s=200, marker='v',
                              edgecolor='darkred', linewidth=2, zorder=9, label='Robot end')

                tags_str = ', '.join([t for t in case.get('tags', []) if t != 'all'][:3])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{cat.upper()} #{ex_idx+1}\n{tags_str}", fontsize=11, fontweight='bold')
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)

                if cat_idx == 0 and ex_idx == 0:
                    ax.legend(loc='best', fontsize=8)
            else:
                ax.text(0.5, 0.5, f"No example found", ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='red', fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

    output = Path("docs/figures/analysis/trajectory_examples_benchmark.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output}")
    plt.close()


if __name__ == "__main__":
    main()
