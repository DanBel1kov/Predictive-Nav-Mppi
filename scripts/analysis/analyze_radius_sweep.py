"""Show how category distribution changes with near-robot radius.

For each dataset (scene), compute category share at r=1,2,3,6m
and plot all scenes on one figure side-by-side per radius.
"""
from __future__ import annotations

import json
import math
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


def heading_change(obs_xy: np.ndarray) -> float:
    if obs_xy.shape[0] < 3:
        return 0.0
    step = obs_xy[1:] - obs_xy[:-1]
    ang = np.arctan2(step[:, 1], step[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


def speed_mags(obs_xy: np.ndarray) -> np.ndarray:
    if obs_xy.shape[0] < 2:
        return np.zeros(0)
    v = (obs_xy[1:] - obs_xy[:-1]) / OBS_DT
    return np.linalg.norm(v, axis=1)


def classify(obs_xy: np.ndarray, neigh_xy: list) -> set:
    tags = {"all"}
    hc_deg = math.degrees(heading_change(obs_xy))
    if hc_deg < 30.0:
        tags.add("linear")
    if hc_deg >= TURN_THRESH_DEG:
        tags.add("turning")
    sp = speed_mags(obs_xy)
    if sp.size > 0:
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


def build_tracks(frames):
    by_id: Dict[int, list] = {}
    rt = []
    for fr in frames:
        t = float(fr["t"])
        for p in fr.get("people", []):
            by_id.setdefault(int(p["id"]), []).append((t, float(p["x"]), float(p["y"])))
        rb = fr.get("robot")
        if rb:
            rt.append((t, float(rb["x"]), float(rb["y"]), float(rb.get("yaw", 0.0))))
    tracks = {pid: np.asarray(rows) for pid, rows in by_id.items()}
    robot = np.asarray(rt) if rt else None
    return tracks, robot


def interp_xy(track, t):
    if track.shape[0] == 0 or t < track[0, 0] or t > track[-1, 0]:
        return None
    idx = int(np.searchsorted(track[:, 0], t))
    if idx == 0:
        return track[0, 1:3]
    if idx >= track.shape[0]:
        return track[-1, 1:3]
    t0, t1 = track[idx - 1, 0], track[idx, 0]
    a = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
    return (1 - a) * track[idx - 1, 1:3] + a * track[idx, 1:3]


def interp_robot(robot, t):
    if robot is None or t < robot[0, 0] or t > robot[-1, 0]:
        return None
    idx = int(np.searchsorted(robot[:, 0], t))
    if idx == 0:
        return robot[0, 1:3]
    if idx >= robot.shape[0]:
        return robot[-1, 1:3]
    t0, t1 = robot[idx - 1, 0], robot[idx, 0]
    a = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
    return (1 - a) * robot[idx - 1, 1:3] + a * robot[idx, 1:3]


def sample_obs(track, t_now):
    pts = []
    for i in range(OBS_LEN):
        t = t_now - (OBS_LEN - 1 - i) * OBS_DT
        p = interp_xy(track, t)
        if p is None:
            return None
        pts.append(p)
    return np.asarray(pts)


def analyze_radius(frames, tracks, robot, radius: float) -> Dict[str, int]:
    counts = {c: 0 for c in CATEGORIES + ["all"]}
    t0 = float(frames[0]["t"])
    t_end = float(frames[-1]["t"])
    t = t0 + OBS_DT * (OBS_LEN - 1)
    while t <= t_end:
        rb = interp_robot(robot, t)
        if rb is None:
            t += TIME_STRIDE
            continue
        candidates = [pid for pid, tr in tracks.items() if tr[0, 0] <= t <= tr[-1, 0]]
        cur = {pid: p for pid in candidates if (p := interp_xy(tracks[pid], t)) is not None}
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
            tags = classify(obs_xy, neigh)
            for c in CATEGORIES + ["all"]:
                if c in tags:
                    counts[c] += 1
        t += TIME_STRIDE
    return counts


def main():
    datasets = [
        ("long_corridor", "./datasets/raw_benchmark_force/people_dataset_long_corridor.json"),
        ("labyrinth_turns", "./datasets/raw_benchmark_force/people_dataset_labyrinth_turns.json"),
        ("nonlinear_corridor", "./datasets/raw_benchmark_force/people_dataset_nonlinear_corridor.json"),
    ]

    # Pre-load
    loaded = []
    for name, path in datasets:
        data = json.loads(Path(path).read_text())
        frames = data["frames"]
        tracks, robot = build_tracks(frames)
        loaded.append((name, frames, tracks, robot))
        print(f"Loaded {name}: {len(frames)} frames")

    # Compute counts for each (dataset, radius)
    results = {}  # (name, radius) -> counts dict
    for name, frames, tracks, robot in loaded:
        for r in RADII:
            print(f"  {name} r={r}m ...", end=" ", flush=True)
            counts = analyze_radius(frames, tracks, robot, r)
            results[(name, r)] = counts
            print(f"N={counts['all']}")

    # --- Plot: one subplot per category, x=radius, lines=scenes ---
    n_cat = len(CATEGORIES)
    fig, axes = plt.subplots(1, n_cat, figsize=(4 * n_cat, 5), sharey=False)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    scene_names = [name for name, *_ in loaded]

    for ax, cat in zip(axes, CATEGORIES):
        for si, name in enumerate(scene_names):
            shares = []
            ns = []
            for r in RADII:
                cnt = results[(name, r)]
                share = 100.0 * cnt[cat] / max(1, cnt["all"])
                shares.append(share)
                ns.append(cnt["all"])
            ax.plot(RADII, shares, "o-", color=colors[si], label=name, linewidth=2, markersize=6)
        ax.set_title(cat, fontsize=11)
        ax.set_xlabel("near-robot radius (m)")
        ax.set_ylabel("% of cases" if cat == CATEGORIES[0] else "")
        ax.set_xticks(RADII)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Category share vs near-robot radius  (TIME_STRIDE={TIME_STRIDE}s)",
        fontsize=12, y=1.02
    )
    fig.tight_layout()

    # --- Second figure: aggregated across all scenes ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for r_idx, r in enumerate(RADII):
        agg = {c: 0 for c in CATEGORIES + ["all"]}
        for name in scene_names:
            for c in CATEGORIES + ["all"]:
                agg[c] += results[(name, r)][c]
        shares = [100.0 * agg[c] / max(1, agg["all"]) for c in CATEGORIES]
        x = np.arange(len(CATEGORIES))
        width = 0.18
        offset = (r_idx - 1.5) * width
        bars = ax2.bar(x + offset, shares, width, label=f"r={r}m  (N={agg['all']})")

    ax2.set_xticks(np.arange(len(CATEGORIES)))
    ax2.set_xticklabels(CATEGORIES, fontsize=10)
    ax2.set_ylabel("% of near-robot cases")
    ax2.set_title("All scenes pooled — category share by near-robot radius")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig2.tight_layout()

    out1 = Path("./docs/figures/analysis/radius_sweep_per_category.png")
    out2 = Path("./docs/figures/analysis/radius_sweep_pooled.png")
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    print(f"Saved: {out2}")

    # Print table
    print("\nPooled category share by radius:")
    print(f"  {'radius':>8} | " + " | ".join(f"{c:>18}" for c in CATEGORIES) + " | N")
    for r in RADII:
        agg = {c: 0 for c in CATEGORIES + ["all"]}
        for name in scene_names:
            for c in CATEGORIES + ["all"]:
                agg[c] += results[(name, r)][c]
        shares = " | ".join(f"{100*agg[c]/max(1,agg['all']):17.1f}%" for c in CATEGORIES)
        print(f"  {r:>7.1f}m | {shares} | {agg['all']}")


if __name__ == "__main__":
    main()
