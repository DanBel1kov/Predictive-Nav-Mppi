#!/usr/bin/env python3
"""Visualize trajectory examples from force sweep benchmark."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def classify_case(case: Dict) -> str:
    """Classify case based on tags."""
    tags = set(case.get('tags', []))
    tags.discard('all')

    priority = ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"]
    for tag in priority:
        if tag in tags:
            return tag
    return "unknown"


def create_visualization(force_value: float, cases_path: Path, output_path: Path):
    """Create visualization for a specific force value."""

    print(f"\nProcessing force={force_value}...")

    if not cases_path.exists():
        print(f"  ❌ File not found: {cases_path}")
        return

    with open(cases_path) as f:
        data = json.load(f)

    cases = data.get('cases', [])
    print(f"  Loaded {len(cases)} cases")

    examples_by_cat = {cat: [] for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]}

    # Classify and collect examples
    for case in cases:
        cat = classify_case(case)
        if cat in examples_by_cat and len(examples_by_cat[cat]) < 4:
            examples_by_cat[cat].append(case)

    print(f"  Found examples:")
    for cat, exs in examples_by_cat.items():
        print(f"    {cat:20s}: {len(exs)}/4")

    # Plot
    fig, axes = plt.subplots(6, 4, figsize=(24, 20))
    fig.suptitle(f"Force Sweep Benchmark (force={force_value}) - Trajectory Examples\n" +
                "Blue=Person, Green=Future, Red=Robot, Gray=Others",
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
                              edgecolor='black', linewidth=2, zorder=10)
                    ax.scatter(obs_xy[-1, 0], obs_xy[-1, 1], c='blue', s=250, marker='s',
                              edgecolor='black', linewidth=2, zorder=10)

                # Plot prediction
                if pred_xy.size > 0:
                    ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'g--', linewidth=3, label='Future pred', zorder=4)
                    ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], c='green', s=150, marker='*', zorder=9)

                # Plot robot
                if robot_xy.size > 0:
                    ax.plot(robot_xy[:, 0], robot_xy[:, 1], 'r-', linewidth=3, label='Robot path', zorder=3)
                    ax.scatter(robot_xy[0, 0], robot_xy[0, 1], c='red', s=200, marker='^',
                              edgecolor='darkred', linewidth=2, zorder=9)
                    ax.scatter(robot_xy[-1, 0], robot_xy[-1, 1], c='darkred', s=200, marker='v',
                              edgecolor='darkred', linewidth=2, zorder=9)

                tags_str = ', '.join([t for t in case.get('tags', []) if t != 'all'][:3])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{cat.upper()} #{ex_idx+1}\n{tags_str}", fontsize=11, fontweight='bold')
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)

                if cat_idx == 0 and ex_idx == 0:
                    ax.legend(loc='best', fontsize=8)
            else:
                ax.text(0.5, 0.5, f"N/A", ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='red', fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    base_path = Path("./benchmark_force_sweep/kalman_react_force_0_025_05_near_robot3")

    forces = [
        (0.0, "force_0p00/20260513_005323/curated_near_robot/benchmark_cases.json"),
        (0.25, "force_0p25/20260513_005606/curated_near_robot/benchmark_cases.json"),
        (0.5, "force_0p50/20260513_005848/curated_near_robot/benchmark_cases.json"),
    ]

    for force_value, rel_path in forces:
        cases_path = base_path / rel_path
        output_path = Path(f"docs/figures/analysis/trajectory_examples_force_{force_value:.2f}.png")
        create_visualization(force_value, cases_path, output_path)

    print("\n" + "="*70)
    print("All visualizations completed!")
    print("="*70)
    print("\nGenerated files:")
    for force_value, _ in forces:
        output = Path(f"docs/figures/analysis/trajectory_examples_force_{force_value:.2f}.png")
        print(f"  Force {force_value}: {output}")


if __name__ == "__main__":
    main()
