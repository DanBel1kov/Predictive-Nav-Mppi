#!/usr/bin/env python3
"""Analyze how category distribution changes with radius AND robot force factor.

Usage:
  python3 analyze_force_radius.py <benchmark_dir>
  python3 -m predictive_nav_mppi.analyze_force_radius --study-dir <study_dir>

Example:
  python3 analyze_force_radius.py benchmark_force_sweep/kalman_force_0_vs_1_fixed2
  python3 -m predictive_nav_mppi.analyze_force_radius --study-dir benchmark_force_sweep/kalman_force_0_vs_1_fixed2

Reads people_dataset_force_*.json from each force subfolder,
computes category share at r=1,2,3,6m for each force,
saves two PNGs and boxplot metrics into <benchmark_dir>/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OBS_LEN = 8
OBS_DT = 0.4
NEIGHBOR_RADIUS = 2.0
TURN_THRESH_DEG = 45.0
STOP_SPEED_THRESH = 0.10
MOVING_SPEED_MIN = 0.25
STOP_GO_DELTA = 0.25
TIME_STRIDE = 0.5
RADII = [1.0, 2.0, 3.0, 6.0]
CATEGORIES = ["linear", "turning", "stop_go", "interaction", "dense_interaction", "complex"]


# ── trajectory classification ──────────────────────────────────────────────

def heading_change(obs_xy: np.ndarray) -> float:
    if obs_xy.shape[0] < 3:
        return 0.0
    step = obs_xy[1:] - obs_xy[:-1]
    ang = np.arctan2(step[:, 1], step[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


def classify(obs_xy: np.ndarray, neigh_xy: list) -> set:
    tags = {"all"}
    hc_deg = math.degrees(heading_change(obs_xy))
    if hc_deg < 30.0:
        tags.add("linear")
    if hc_deg >= TURN_THRESH_DEG:
        tags.add("turning")
    if obs_xy.shape[0] >= 2:
        v = (obs_xy[1:] - obs_xy[:-1]) / OBS_DT
        sp = np.linalg.norm(v, axis=1)
        s_min, s_max = float(sp.min()), float(sp.max())
        if s_max >= MOVING_SPEED_MIN and s_min <= STOP_SPEED_THRESH and (s_max - s_min) >= STOP_GO_DELTA:
            tags.add("stop_go")
    if neigh_xy:
        p = obs_xy[-1]
        dmin = min(float(np.linalg.norm(p - n[-1])) for n in neigh_xy)
        n_n = len(neigh_xy)
        if 1 <= n_n <= 3 and dmin <= 1.5:
            tags.add("interaction")
        if n_n >= 4 and dmin <= 1.5:
            tags.add("dense_interaction")
    axes = sum(t in tags for t in ("turning", "stop_go", "interaction", "dense_interaction"))
    if axes >= 2:
        tags.add("complex")
    return tags


# ── data loading ────────────────────────────────────────────────────────────

def build_tracks(frames):
    by_id: Dict[int, list] = {}
    rt = []
    for fr in frames:
        t = float(fr["t"])
        for p in fr.get("people", []):
            by_id.setdefault(int(p["id"]), []).append((t, float(p["x"]), float(p["y"])))
        rb = fr.get("robot")
        if rb:
            rt.append((t, float(rb["x"]), float(rb["y"])))
    tracks = {pid: np.asarray(rows) for pid, rows in by_id.items()}
    robot = np.asarray(rt) if rt else None
    return tracks, robot


def interp2(arr: np.ndarray, t: float) -> Optional[np.ndarray]:
    if arr is None or arr.shape[0] == 0 or t < arr[0, 0] or t > arr[-1, 0]:
        return None
    idx = int(np.searchsorted(arr[:, 0], t))
    if idx == 0:
        return arr[0, 1:]
    if idx >= arr.shape[0]:
        return arr[-1, 1:]
    t0, t1 = arr[idx - 1, 0], arr[idx, 0]
    a = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
    return (1 - a) * arr[idx - 1, 1:] + a * arr[idx, 1:]


def sample_obs(track: np.ndarray, t_now: float) -> Optional[np.ndarray]:
    pts = []
    for i in range(OBS_LEN):
        p = interp2(track, t_now - (OBS_LEN - 1 - i) * OBS_DT)
        if p is None:
            return None
        pts.append(p)
    return np.asarray(pts)


# ── analysis ────────────────────────────────────────────────────────────────

def analyze(frames, tracks, robot, radius: float) -> Dict[str, int]:
    counts = {c: 0 for c in CATEGORIES + ["all"]}
    t0, t_end = float(frames[0]["t"]), float(frames[-1]["t"])
    t = t0 + OBS_DT * (OBS_LEN - 1)
    while t <= t_end:
        rb = interp2(robot, t)
        if rb is None:
            t += TIME_STRIDE
            continue
        candidates = [pid for pid, tr in tracks.items() if tr[0, 0] <= t <= tr[-1, 0]]
        cur = {}
        for pid in candidates:
            p = interp2(tracks[pid], t)
            if p is not None:
                cur[pid] = p
        for pid, p_now in cur.items():
            if math.hypot(p_now[0] - rb[0], p_now[1] - rb[1]) > radius:
                continue
            obs_xy = sample_obs(tracks[pid], t)
            if obs_xy is None:
                continue
            neigh = []
            for oid, q_now in cur.items():
                if oid == pid or math.hypot(q_now[0] - p_now[0], q_now[1] - p_now[1]) > NEIGHBOR_RADIUS:
                    continue
                o_obs = sample_obs(tracks[oid], t)
                if o_obs is not None:
                    neigh.append(o_obs)
            for c in CATEGORIES + ["all"]:
                if c in classify(obs_xy, neigh):
                    counts[c] += 1
        t += TIME_STRIDE
    return counts


# ── main ────────────────────────────────────────────────────────────────────

def _extract_force_from_dirname(force_dir: Path) -> Optional[float]:
    """Parse force value from directory name: force_0p25 -> 0.25"""
    name = force_dir.name
    if not name.startswith("force_"):
        return None
    try:
        val_str = name.replace("force_", "").replace("p", ".")
        return float(val_str)
    except ValueError:
        return None


def _collect_metrics_from_episodes(force_dir: Path) -> Dict[str, List[float]]:
    """Collect metrics from summary.json files in timestamp subdirs"""
    metrics_by_key = {
        "time_to_goal": [],
        "path_length": [],
        "min_dist": [],
        "avg_dist": [],
        "viol_time": [],
        "avg_robot_influence": [],
        "nearest_robot_influence": [],
    }
    # Look in timestamp subdirectories for summary.json
    for timestamp_dir in sorted(force_dir.iterdir()):
        if not timestamp_dir.is_dir() or timestamp_dir.name.startswith("people_"):
            continue
        summary_path = timestamp_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            episodes = json.loads(summary_path.read_text())
            if not isinstance(episodes, list):
                continue
            for ep in episodes:
                for key in metrics_by_key:
                    if key in ep:
                        metrics_by_key[key].append(float(ep[key]))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return {k: v for k, v in metrics_by_key.items() if v}


def main():
    parser = argparse.ArgumentParser(description="Analyze category distribution vs radius & force")
    parser.add_argument("bench_dir", nargs="?", default=None)
    parser.add_argument("--study-dir", default=None)
    args = parser.parse_args()

    # Determine which directory to analyze
    if args.study_dir:
        bench_dir = Path(args.study_dir).expanduser().resolve()
    elif args.bench_dir:
        bench_dir = Path(args.bench_dir).expanduser().resolve()
    else:
        parser.print_help()
        sys.exit(1)

    if not bench_dir.exists():
        print(f"Not found: {bench_dir}")
        sys.exit(1)

    # Find force dirs: subdirs with people_dataset_force_*.json
    force_entries = []
    for subdir in sorted(bench_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ds_files = sorted(subdir.glob("people_dataset_force_*.json"))
        if not ds_files:
            continue
        ds_file = ds_files[0]
        # Parse force value from filename: people_dataset_force_0p25.json -> 0.25
        stem = ds_file.stem  # people_dataset_force_0p25
        val_str = stem.split("force_")[-1].replace("p", ".")
        try:
            force_val = float(val_str)
        except ValueError:
            continue
        force_entries.append((force_val, ds_file))

    if not force_entries:
        print(f"No people_dataset_force_*.json found in subdirs of {bench_dir}")
        sys.exit(1)

    print(f"Found {len(force_entries)} force levels: {[f for f, _ in force_entries]}")

    # Load datasets
    loaded = []
    for force_val, ds_file in force_entries:
        data = json.loads(ds_file.read_text())
        frames = data["frames"]
        tracks, robot = build_tracks(frames)
        loaded.append((force_val, frames, tracks, robot))
        print(f"  force={force_val:.2f}: {len(frames)} frames, {len(tracks)} people tracks")

    # Compute counts: results[(force, radius)] = counts dict
    results = {}
    for force_val, frames, tracks, robot in loaded:
        for r in RADII:
            print(f"  force={force_val:.2f} r={r}m ...", end=" ", flush=True)
            counts = analyze(frames, tracks, robot, r)
            results[(force_val, r)] = counts
            print(f"N={counts['all']}")

    forces = [f for f, *_ in loaded]
    n_forces = len(forces)
    cmap = plt.cm.viridis
    force_colors = [cmap(i / max(1, n_forces - 1)) for i in range(n_forces)]

    # ── Figure 1: one subplot per category, x=radius, line per force ──────
    n_cat = len(CATEGORIES)
    fig1, axes1 = plt.subplots(1, n_cat, figsize=(3.5 * n_cat, 5), sharey=False)
    for ax, cat in zip(axes1, CATEGORIES):
        for fi, force_val in enumerate(forces):
            shares, ns = [], []
            for r in RADII:
                cnt = results[(force_val, r)]
                shares.append(100.0 * cnt[cat] / max(1, cnt["all"]))
                ns.append(cnt["all"])
            ax.plot(RADII, shares, "o-", color=force_colors[fi],
                    label=f"f={force_val:.2f}", linewidth=2, markersize=6)
        ax.set_title(cat, fontsize=11)
        ax.set_xlabel("radius (m)")
        if cat == CATEGORIES[0]:
            ax.set_ylabel("% of near-robot cases")
        ax.set_xticks(RADII)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes1[0].legend(loc="best", fontsize=8)
    fig1.suptitle(
        f"Category share vs near-robot radius, by force factor\n({bench_dir.name})",
        fontsize=11, y=1.01
    )
    fig1.tight_layout()

    # ── Figure 2: one subplot per radius, bars per category, grouped by force ──
    fig2, axes2 = plt.subplots(1, len(RADII), figsize=(5 * len(RADII), 5), sharey=True)
    width = 0.8 / n_forces
    x = np.arange(n_cat)
    for ri, (r, ax) in enumerate(zip(RADII, axes2)):
        for fi, force_val in enumerate(forces):
            cnt = results[(force_val, r)]
            shares = [100.0 * cnt[c] / max(1, cnt["all"]) for c in CATEGORIES]
            offset = (fi - (n_forces - 1) / 2) * width
            ax.bar(x + offset, shares, width, label=f"f={force_val:.2f}",
                   color=force_colors[fi], alpha=0.85)
        ax.set_title(f"r = {r:.0f} m", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if ri == 0:
            ax.set_ylabel("% of near-robot cases")
            ax.legend(fontsize=8, loc="upper right")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig2.suptitle(
        f"Category share by radius & force factor\n({bench_dir.name})",
        fontsize=11, y=1.01
    )
    fig2.tight_layout()

    out1 = bench_dir / "category_vs_radius_by_force.png"
    out2 = bench_dir / "category_by_radius_grouped.png"
    fig1.savefig(out1, dpi=130, bbox_inches="tight")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

    # ── Boxplot metrics: episode navigation and robot-influence metrics
    print(f"\nCollecting metrics from episodes...")
    metrics_by_force: Dict[float, Dict[str, List[float]]] = {}
    run_dirs = []
    for force_dir in sorted(bench_dir.iterdir()):
        if not force_dir.is_dir() or force_dir.name.startswith("_"):
            continue
        force_val = _extract_force_from_dirname(force_dir)
        if force_val is None:
            continue
        # Look in timestamp subdirectories
        metrics = _collect_metrics_from_episodes(force_dir)
        if metrics:
            metrics_by_force[force_val] = metrics
            run_dirs.append((force_val, force_dir))
            print(f"  force={force_val:.2f}: {force_dir.name} → {len(metrics.get('time_to_goal', []))} episodes")

    if metrics_by_force and len(metrics_by_force) >= 2:
        metric_keys = [
            "time_to_goal",
            "path_length",
            "min_dist",
            "avg_dist",
            "viol_time",
            "avg_robot_influence",
            "nearest_robot_influence",
        ]
        metric_labels = {
            "time_to_goal": "Time to Goal (s)",
            "path_length": "Path Length (m)",
            "min_dist": "Min Distance (m)",
            "avg_dist": "Avg Distance (m)",
            "viol_time": "Violation Time (s)",
            "avg_robot_influence": "Avg Robot Influence",
            "nearest_robot_influence": "Nearest Robot Influence",
        }
        force_vals_sorted = sorted(metrics_by_force.keys())
        fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
        axes3 = axes3.flatten()
        for idx, metric in enumerate(metric_keys):
            ax = axes3[idx]
            data_by_force = [metrics_by_force[f].get(metric, []) for f in force_vals_sorted]
            bp = ax.boxplot(data_by_force, labels=[f"{f:.2f}" for f in force_vals_sorted],
                            patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], force_colors[:len(force_vals_sorted)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(metric_labels[metric], fontsize=10, fontweight="bold")
            ax.set_xlabel("robot_force_scale")
            ax.grid(True, alpha=0.3, axis="y")
        for ax in axes3[len(metric_keys):]:
            ax.axis("off")
        fig3.suptitle(f"Episode metrics by force factor\n({bench_dir.name})",
                      fontsize=12, fontweight="bold")
        fig3.tight_layout()
        out3 = bench_dir / "metrics_boxplot_by_force.png"
        fig3.savefig(out3, dpi=130, bbox_inches="tight")
        print(f"Saved: {out3}")

    # Summary table
    print(f"\nCategory share (%) — pooled by force, r=3m:")
    print(f"  {'force':>6} | " + " | ".join(f"{c:>18}" for c in CATEGORIES) + " |    N")
    for force_val in forces:
        cnt = results[(force_val, 3.0)]
        row = " | ".join(f"{100*cnt[c]/max(1,cnt['all']):17.1f}%" for c in CATEGORIES)
        print(f"  {force_val:>5.2f}f | {row} | {cnt['all']:>4}")


if __name__ == "__main__":
    main()


def __main__():
    main()


__all__ = ["main"]
