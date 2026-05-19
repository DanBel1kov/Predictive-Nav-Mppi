#!/usr/bin/env python3
"""Regenerate interaction_summary.json from reclassified curated dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _sum_tag_counts(*dicts: Dict[str, int]) -> Dict[str, int]:
    """Sum tag counts from multiple dicts."""
    result: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            result[k] = result.get(k, 0) + int(v)
    return result


def update_interaction_summary(run_dir: Path, curated_dir: Path) -> None:
    """Update interaction_summary.json from reclassified curated dataset."""
    summary_path = curated_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"summary.json not found in {curated_dir}")

    summary = json.loads(summary_path.read_text())
    train_cases_path = curated_dir / "train_cases.json"
    bench_cases_path = curated_dir / "benchmark_cases.json"

    selected = summary.get("selected", {})
    train = selected.get("train", {})
    bench = selected.get("benchmark", {})

    counts = _sum_tag_counts(
        train.get("tag_counts", {}),
        bench.get("tag_counts", {}),
    )

    # Count linear from actual saved tags (now it's explicit)
    linear_count = 0
    for cases_path in [train_cases_path, bench_cases_path]:
        if cases_path.is_file():
            payload = json.loads(cases_path.read_text())
            for case in payload.get("cases", []):
                if "linear" in case.get("tags", []):
                    linear_count += 1
    counts["linear"] = linear_count
    counts = {k: int(v) for k, v in counts.items() if int(v) > 0}

    examples: List[Dict[str, Any]] = []
    for cases_path in [train_cases_path, bench_cases_path]:
        if cases_path.is_file():
            payload = json.loads(cases_path.read_text())
            for case in payload.get("cases", []):
                tags = [str(t) for t in case.get("tags", []) if str(t) != "all"]
                if not tags:
                    continue
                examples.append({
                    "case_id": case.get("case_id", ""),
                    "source_name": case.get("source_name", ""),
                    "frame_index": int(case.get("frame_index", -1)),
                    "person_id": int(case.get("person_id", -1)),
                    "t": float(case.get("t", 0.0)),
                    "categories": sorted(set(tags)),
                    "metrics": case.get("metrics", {}),
                })

    payload = {
        "counts": counts,
        "examples": examples,
        "source": "offline_curate_people_dataset",
        "curated_dir": str(curated_dir),
    }

    out_path = run_dir / "interaction_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Updated {out_path}")
    print(f"  Counts: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Update interaction_summary from reclassified curated dataset")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("curated_dir", type=Path, help="Path to curated dataset directory")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    curated_dir = args.curated_dir.resolve()

    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    if not curated_dir.is_dir():
        raise SystemExit(f"Not a directory: {curated_dir}")

    update_interaction_summary(run_dir, curated_dir)


if __name__ == "__main__":
    main()
