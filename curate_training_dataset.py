#!/usr/bin/env python3
"""Create balanced training dataset matching benchmark distribution."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze_distribution(cases: list, name: str = "Dataset") -> dict:
    """Analyze distribution of cases by category."""
    dist = defaultdict(int)
    for case in cases:
        tags = set(case.get("tags", []))
        tags.discard("all")

        # Classify
        priority = ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"]
        category = "unknown"
        for tag in priority:
            if tag in tags:
                category = tag
                break
        dist[category] += 1

    total = sum(dist.values())
    print(f"\n{name}:")
    for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]:
        count = dist.get(cat, 0)
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  {cat:20s}: {count:6d} ({pct:5.1f}%)")

    return dict(dist)


def classify_case(case: dict) -> str:
    """Classify a case by category."""
    tags = set(case.get("tags", []))
    tags.discard("all")

    priority = ["complex", "dense_interaction", "stop_go", "interaction", "turning", "linear"]
    for tag in priority:
        if tag in tags:
            return tag
    return "unknown"


def create_balanced_dataset():
    """Create balanced training dataset."""

    # Load benchmark cases to understand target distribution
    benchmark_files = [
        Path("/home/danbel1kov/predictive-nav-mppi/benchmark_force_sweep/kalman_react_force_0_025_05_near_robot3/force_0p50/20260513_005848/curated_near_robot/benchmark_cases.json"),
    ]

    benchmark_cases = []
    for bf in benchmark_files:
        if bf.exists():
            data = json.loads(bf.read_text())
            benchmark_cases.extend(data.get("cases", []))

    if not benchmark_cases:
        print("❌ No benchmark cases found!")
        return

    # Analyze benchmark distribution (target distribution)
    benchmark_dist = analyze_distribution(benchmark_cases, "Benchmark (target)")

    # Load raw dataset
    raw_file = Path("/home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_merged_react_0p5.json")
    if not raw_file.exists():
        print(f"❌ Raw dataset not found: {raw_file}")
        return

    print(f"\nLoading raw dataset from {raw_file}...")
    raw_data = json.loads(raw_file.read_text())
    raw_cases = raw_data.get("cases", [])

    # Analyze raw distribution
    analyze_distribution(raw_cases, "Raw dataset (before balancing)")

    # Target distribution (from benchmark force=0.5)
    target_dist = {
        "linear": 0.049,
        "interaction": 0.122,
        "dense_interaction": 0.083,
        "turning": 0.222,
        "stop_go": 0.264,
        "complex": 0.259,
    }

    print("\n" + "="*70)
    print("Target distribution (benchmark force=0.5):")
    for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]:
        pct = target_dist[cat] * 100
        print(f"  {cat:20s}: {pct:5.1f}%")

    # Group raw cases by category
    cases_by_cat = defaultdict(list)
    for case in raw_cases:
        cat = classify_case(case)
        if cat != "unknown":
            cases_by_cat[cat].append(case)

    # Calculate target counts - aim for ~2000 total samples for balanced training
    target_total = 2000
    target_counts = {}
    for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]:
        target_counts[cat] = max(1, int(target_total * target_dist[cat]))

    print("\n" + "="*70)
    print(f"Target counts (total={target_total}):")
    for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]:
        print(f"  {cat:20s}: {target_counts[cat]:4d}")

    # Sample and combine
    sampled_cases = []
    sampling_report = []

    for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]:
        available = cases_by_cat.get(cat, [])
        target = target_counts[cat]

        if len(available) >= target:
            # Undersample
            sampled = random.sample(available, target)
            sampling_report.append(f"  {cat:20s}: sampled {len(sampled):4d} from {len(available):6d} (undersample)")
        else:
            # Oversample with replacement
            sampled = random.choices(available, k=target)
            sampling_report.append(f"  {cat:20s}: sampled {len(sampled):4d} from {len(available):6d} (oversample with replacement)")

        sampled_cases.extend(sampled)

    # Shuffle
    random.shuffle(sampled_cases)

    print("\n" + "="*70)
    print("Sampling strategy:")
    for line in sampling_report:
        print(line)

    # Verify distribution
    final_dist = defaultdict(int)
    for case in sampled_cases:
        cat = classify_case(case)
        final_dist[cat] += 1

    print("\n" + "="*70)
    print("Final balanced dataset distribution:")
    total = len(sampled_cases)
    for cat in ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]:
        count = final_dist[cat]
        pct = 100.0 * count / total if total > 0 else 0
        target_pct = target_dist[cat] * 100
        print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%) [target: {target_pct:5.1f}%]")

    # Split into train (75%) and test (25%)
    split_idx = int(0.75 * len(sampled_cases))
    train_cases = sampled_cases[:split_idx]
    test_cases = sampled_cases[split_idx:]

    print("\n" + "="*70)
    print(f"Train/Test split:")
    print(f"  Train: {len(train_cases)} cases (75%)")
    print(f"  Test:  {len(test_cases)} cases (25%)")

    # Save datasets
    output_base = Path("/home/danbel1kov/predictive-nav-mppi/datasets")
    output_base.mkdir(exist_ok=True)

    # Add source_name for scene context
    for case in sampled_cases:
        if "source_name" not in case:
            case["source_name"] = "synthetic"  # Fallback

    # Save train set
    train_file = output_base / "train_balanced_distribution.json"
    train_data = {"cases": train_cases}
    train_file.write_text(json.dumps(train_data, indent=2))
    print(f"\n✓ Train set saved: {train_file}")

    # Save test set
    test_file = output_base / "test_balanced_distribution.json"
    test_data = {"cases": test_cases}
    test_file.write_text(json.dumps(test_data, indent=2))
    print(f"✓ Test set saved: {test_file}")

    # Print training command
    print("\n" + "="*70)
    print("Training command:")
    print()
    print("python3 -m predictive_nav_mppi.train_residual_predictor \\")
    print(f"  --train_file {train_file} \\")
    print(f"  --test_file {test_file} \\")
    print("  --obs_len 8 \\")
    print("  --pred_len 26 \\")
    print("  --batch_size 32 \\")
    print("  --epochs 50 \\")
    print("  --learning_rate 0.001 \\")
    print("  --k_neighbors 3 \\")
    print("  --scene_patch_size_m 6.0 \\")
    print("  --scene_patch_pixels 32 \\")
    print("  --output_dir ./models/residual_balanced_dist")
    print()


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    create_balanced_dataset()
