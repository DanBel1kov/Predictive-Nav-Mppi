#!/usr/bin/env python3
"""Reclassify curated dataset cases with new classification rules."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set


def _classify_tags(metrics: Dict[str, float], neighbor_count: int, min_neighbor_distance: float,
                   interaction_dist: float, turn_threshold_deg: float,
                   stop_speed_thresh: float, stop_go_delta: float, moving_speed_min: float) -> Set[str]:
    """Classify case tags using new rules."""
    tags: Set[str] = {"all"}

    heading_change_deg = float(metrics.get("heading_change_deg", 0.0))
    curvature_ratio = float(metrics.get("curvature_ratio", 999.0))
    min_speed = float(metrics.get("min_speed", 0.0))
    max_speed = float(metrics.get("max_speed", 0.0))

    # LINEAR: continuous movement in one direction (small heading changes)
    if heading_change_deg < 30.0:
        tags.add("linear")

    # TURNING: large heading change
    if heading_change_deg >= turn_threshold_deg:
        tags.add("turning")

    # STOP_GO: motion pattern with significant speed variations (stop and go)
    if max_speed >= moving_speed_min and min_speed <= stop_speed_thresh and (max_speed - min_speed) >= stop_go_delta:
        tags.add("stop_go")

    # INTERACTION: 1-3 neighbors at close distance
    if 1 <= neighbor_count <= 3 and min_neighbor_distance <= interaction_dist:
        tags.add("interaction")

    # DENSE_INTERACTION: 4+ neighbors at close distance
    if neighbor_count >= 4 and min_neighbor_distance <= interaction_dist:
        tags.add("dense_interaction")

    # COMPLEX: 2 or more of [turning, stop_go, interaction, dense_interaction]
    complexity_axes = (
        int("turning" in tags)
        + int("stop_go" in tags)
        + int("interaction" in tags)
        + int("dense_interaction" in tags)
    )
    if complexity_axes >= 2:
        tags.add("complex")

    return tags


def reclassify_dataset(curated_dir: Path, interaction_dist: float = 1.5,
                       turn_threshold_deg: float = 45.0,
                       stop_speed_thresh: float = 0.10,
                       stop_go_delta: float = 0.25,
                       moving_speed_min: float = 0.25) -> None:
    """Reclassify all cases in a curated dataset directory."""

    train_path = curated_dir / "train_cases.json"
    bench_path = curated_dir / "benchmark_cases.json"
    summary_path = curated_dir / "summary.json"

    for cases_path in [train_path, bench_path]:
        if not cases_path.is_file():
            print(f"Skipping {cases_path} (not found)")
            continue

        print(f"Processing {cases_path}...")
        payload = json.loads(cases_path.read_text())
        cases = payload.get("cases", [])

        # Reclassify each case
        for case in cases:
            metrics = case.get("metrics", {})
            neighbor_count = int(metrics.get("neighbor_count", 0))
            min_neighbor_distance = float(metrics.get("min_neighbor_distance", 999.0))

            new_tags = _classify_tags(
                metrics, neighbor_count, min_neighbor_distance,
                interaction_dist, turn_threshold_deg,
                stop_speed_thresh, stop_go_delta, moving_speed_min
            )
            case["tags"] = sorted(list(new_tags))

        # Write back
        cases_path.write_text(json.dumps(payload, indent=2))
        print(f"  Updated {len(cases)} cases")

    # Recount tags in summary
    if summary_path.is_file():
        print(f"Updating {summary_path}...")
        summary = json.loads(summary_path.read_text())

        # Recount tags from actual cases
        tag_counts: Dict[str, int] = {}
        for cases_path in [train_path, bench_path]:
            if cases_path.is_file():
                payload = json.loads(cases_path.read_text())
                for case in payload.get("cases", []):
                    for tag in case.get("tags", []):
                        if tag != "all":
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        selected = summary.get("selected", {})
        for split_key in ["train", "benchmark"]:
            if split_key in selected:
                selected[split_key]["tag_counts"] = {}

        # Distribute counts proportionally
        for split_key, cases_path in [("train", train_path), ("benchmark", bench_path)]:
            if cases_path.is_file() and split_key in selected:
                payload = json.loads(cases_path.read_text())
                split_tag_counts: Dict[str, int] = {}
                for case in payload.get("cases", []):
                    for tag in case.get("tags", []):
                        if tag != "all":
                            split_tag_counts[tag] = split_tag_counts.get(tag, 0) + 1
                selected[split_key]["tag_counts"] = split_tag_counts

        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  Updated tag counts in summary")


def main():
    parser = argparse.ArgumentParser(description="Reclassify curated dataset with new rules")
    parser.add_argument("curated_dir", type=Path, help="Path to curated dataset directory")
    parser.add_argument("--interaction-dist", type=float, default=1.5)
    parser.add_argument("--turn-threshold-deg", type=float, default=45.0)
    parser.add_argument("--stop-speed-thresh", type=float, default=0.10)
    parser.add_argument("--stop-go-delta", type=float, default=0.25)
    parser.add_argument("--moving-speed-min", type=float, default=0.25)
    args = parser.parse_args()

    curated_dir = args.curated_dir.resolve()
    if not curated_dir.is_dir():
        raise SystemExit(f"Not a directory: {curated_dir}")

    reclassify_dataset(
        curated_dir,
        interaction_dist=args.interaction_dist,
        turn_threshold_deg=args.turn_threshold_deg,
        stop_speed_thresh=args.stop_speed_thresh,
        stop_go_delta=args.stop_go_delta,
        moving_speed_min=args.moving_speed_min,
    )
    print("Done!")


if __name__ == "__main__":
    main()
