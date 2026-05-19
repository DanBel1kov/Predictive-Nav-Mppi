#!/usr/bin/env python3
"""Generate distribution plots for a curated people dataset.

Usage:
    python -m predictive_nav_mppi.report_curated_dataset \
        --curated_dir /home/danbel1kov/predictive-nav-mppi/datasets/curated_people_react_v2

Reads <curated_dir>/summary.json (produced by curate_people_dataset) and produces:
  - sources_train.png / sources_benchmark.png   — bar chart of samples per source scene
  - tags_train.png / tags_benchmark.png         — bar chart of movement-type tag counts
  - summary.txt                                  — text report with percentages

If matplotlib is unavailable, only summary.txt is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _bar_chart(counts: Dict[str, int], total: int, title: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping {out_path.name}")
        return

    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    # sort descending
    order = sorted(range(len(values)), key=lambda i: -values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    pct = [100.0 * v / total if total > 0 else 0.0 for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    bars = ax.bar(labels, values, color="#3a86ff")
    for bar, v, p in zip(bars, values, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("samples")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _format_block(name: str, total: int, counts: Dict[str, int]) -> str:
    lines = [f"\n== {name} (total: {total}) =="]
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * v / total if total > 0 else 0.0
        lines.append(f"  {k:<28} {v:>6}  ({pct:5.1f}%)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated_dir", type=Path, required=True,
                        help="Output dir of curate_people_dataset (must contain summary.json)")
    args = parser.parse_args()

    manifest_path = args.curated_dir / "summary.json"
    if not manifest_path.exists():
        raise SystemExit(f"summary.json not found in {args.curated_dir}")

    manifest = json.loads(manifest_path.read_text())
    selected = manifest.get("selected", {})

    out_lines = [f"Curated dataset report: {args.curated_dir}"]

    for split in ("train", "benchmark"):
        block = selected.get(split, {})
        total = int(block.get("count", 0))
        tag_counts = block.get("tag_counts", {})
        source_counts = block.get("source_counts", {})
        linear = int(block.get("linear_cases", 0))
        out_lines.append(f"\n### {split.upper()}: {total} samples (avg_score={block.get('avg_score', 0):.4f}, linear_cases={linear})")
        out_lines.append(_format_block(f"{split} — by source scene", total, source_counts))
        out_lines.append(_format_block(f"{split} — by movement tag", total, tag_counts))

        _bar_chart(source_counts, total,
                   f"{split} samples per scene (n={total})",
                   args.curated_dir / f"sources_{split}.png")
        _bar_chart(tag_counts, total,
                   f"{split} samples per movement tag (n={total})",
                   args.curated_dir / f"tags_{split}.png")

    summary_path = args.curated_dir / "report.txt"
    summary_path.write_text("\n".join(out_lines))
    print(f"\nWrote {summary_path}")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
