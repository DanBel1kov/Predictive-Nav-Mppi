"""Classify candidate cases on benchmark vs manual recordings with identical rules.

Goal: produce one figure comparing interaction-category counts across scenes
(3 benchmark force-sweep scenes + 3 manual react_0p5 scenes).

Sampling is time-based (not frame-stride) because benchmark is recorded at ~70 Hz
and manual at ~15 Hz, which would otherwise inflate benchmark counts ~5x.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- Mirror benchmark_people_predictors._classify_case_tags ---
OBS_LEN = 8
OBS_DT = 0.4
INTERACTION_DIST = 1.5
NEIGHBOR_RADIUS = 2.0
TURN_THRESH_DEG = 45.0
STOP_SPEED_THRESH = 0.10
MOVING_SPEED_MIN = 0.25
STOP_GO_DELTA = 0.25
NEAR_ROBOT_RADIUS = 3.0   # match study "near_robot3"
NEAR_ROBOT_FOV_DEG = 360.0
TIME_STRIDE = 0.5          # seconds between sampled frames (time-based)


def heading_change(obs_xy: np.ndarray) -> float:
    if obs_xy.shape[0] < 3:
        return 0.0
    step = obs_xy[1:] - obs_xy[:-1]
    ang = np.arctan2(step[:, 1], step[:, 0])
    if ang.shape[0] < 2:
        return 0.0
    d = np.diff(ang)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


def speed_mags(obs_xy: np.ndarray, dt: float) -> np.ndarray:
    if obs_xy.shape[0] < 2:
        return np.zeros((0,))
    v = (obs_xy[1:] - obs_xy[:-1]) / max(1e-6, dt)
    return np.linalg.norm(v, axis=1)


def classify(obs_xy: np.ndarray, neigh_xy: List[np.ndarray]) -> set:
    tags = {"all"}
    hc = math.degrees(heading_change(obs_xy))
    if hc < 30.0:
        tags.add("linear")
    if heading_change(obs_xy) >= math.radians(TURN_THRESH_DEG):
        tags.add("turning")
    sp = speed_mags(obs_xy, OBS_DT)
    if sp.size > 0:
        s_min, s_max = float(sp.min()), float(sp.max())
        if s_max >= MOVING_SPEED_MIN and s_min <= STOP_SPEED_THRESH and (s_max - s_min) >= STOP_GO_DELTA:
            tags.add("stop_go")
    # neighbors
    if neigh_xy:
        p = obs_xy[-1]
        dmin = min(float(np.linalg.norm(p - n[-1])) for n in neigh_xy)
        n_neigh = len(neigh_xy)
        if 1 <= n_neigh <= 3 and dmin <= INTERACTION_DIST:
            tags.add("interaction")
        if n_neigh >= 4 and dmin <= INTERACTION_DIST:
            tags.add("dense_interaction")
    axes = sum(t in tags for t in ("turning", "stop_go", "interaction", "dense_interaction"))
    if axes >= 2:
        tags.add("complex")
    return tags


def build_tracks(frames: List[dict]) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
    by_id: Dict[int, List[Tuple[float, float, float]]] = {}
    rt: List[Tuple[float, float, float, float]] = []
    for fr in frames:
        t = float(fr["t"])
        for person in fr.get("people", []):
            pid = int(person["id"])
            by_id.setdefault(pid, []).append((t, float(person["x"]), float(person["y"])))
        robot = fr.get("robot")
        if robot:
            rt.append((t, float(robot["x"]), float(robot["y"]), float(robot.get("yaw", 0.0))))
    tracks = {pid: np.asarray(rows, dtype=np.float64) for pid, rows in by_id.items()}
    robot_arr = np.asarray(rt, dtype=np.float64) if rt else None
    return tracks, robot_arr


def interp_xy(track: np.ndarray, t: float) -> Optional[np.ndarray]:
    if track.shape[0] == 0 or t < track[0, 0] or t > track[-1, 0]:
        return None
    idx = np.searchsorted(track[:, 0], t)
    if idx == 0:
        return track[0, 1:3]
    if idx >= track.shape[0]:
        return track[-1, 1:3]
    t0, t1 = track[idx - 1, 0], track[idx, 0]
    if t1 == t0:
        return track[idx, 1:3]
    a = (t - t0) / (t1 - t0)
    return (1 - a) * track[idx - 1, 1:3] + a * track[idx, 1:3]


def sample_obs(track: np.ndarray, t_now: float) -> Optional[np.ndarray]:
    pts = []
    for i in range(OBS_LEN):
        t = t_now - (OBS_LEN - 1 - i) * OBS_DT
        p = interp_xy(track, t)
        if p is None:
            return None
        pts.append(p)
    return np.asarray(pts)


def interp_robot(robot: np.ndarray, t: float) -> Optional[np.ndarray]:
    if robot is None or robot.shape[0] == 0:
        return None
    if t < robot[0, 0] or t > robot[-1, 0]:
        return None
    idx = np.searchsorted(robot[:, 0], t)
    if idx == 0:
        return robot[0, 1:4]
    if idx >= robot.shape[0]:
        return robot[-1, 1:4]
    t0, t1 = robot[idx - 1, 0], robot[idx, 0]
    if t1 == t0:
        return robot[idx, 1:4]
    a = (t - t0) / (t1 - t0)
    r0, r1 = robot[idx - 1, 1:4], robot[idx, 1:4]
    out = (1 - a) * r0 + a * r1
    # yaw shortest path
    dy = (r1[2] - r0[2] + math.pi) % (2 * math.pi) - math.pi
    out[2] = r0[2] + a * dy
    return out


def analyze(path: Path, label: str) -> Dict[str, int]:
    data = json.loads(path.read_text())
    frames = data["frames"]
    tracks, robot = build_tracks(frames)
    t0 = float(frames[0]["t"])
    t_end = float(frames[-1]["t"])
    duration = t_end - t0
    counts = {"all": 0, "linear": 0, "turning": 0, "stop_go": 0,
              "interaction": 0, "dense_interaction": 0, "complex": 0}
    n_samples = 0
    n_near = 0
    t = t0 + OBS_DT * (OBS_LEN - 1)
    while t <= t_end:
        n_samples += 1
        rb = interp_robot(robot, t)
        # collect persons present near t
        candidates_pid = [pid for pid, tr in tracks.items()
                          if tr[0, 0] <= t <= tr[-1, 0]]
        # current positions
        cur = {}
        for pid in candidates_pid:
            p = interp_xy(tracks[pid], t)
            if p is not None:
                cur[pid] = p
        for pid, p_now in cur.items():
            # near-robot filter
            if rb is None:
                continue
            if math.hypot(p_now[0] - rb[0], p_now[1] - rb[1]) > NEAR_ROBOT_RADIUS:
                continue
            obs_xy = sample_obs(tracks[pid], t)
            if obs_xy is None:
                continue
            n_near += 1
            # neighbors
            neigh = []
            for oid, q_now in cur.items():
                if oid == pid:
                    continue
                if math.hypot(q_now[0] - p_now[0], q_now[1] - p_now[1]) > NEIGHBOR_RADIUS:
                    continue
                o_obs = sample_obs(tracks[oid], t)
                if o_obs is None:
                    continue
                neigh.append(o_obs)
            tags = classify(obs_xy, neigh)
            for tg in counts:
                if tg in tags:
                    counts[tg] += 1
        t += TIME_STRIDE
    counts["__duration_s"] = round(duration, 1)
    counts["__samples"] = n_samples
    counts["__near_cases"] = n_near
    print(f"[{label}] duration={duration:.1f}s samples={n_samples} near_cases={n_near} counts={ {k:v for k,v in counts.items() if not k.startswith('__')} }")
    return counts


def main() -> None:
    scenes = [
        ("bench f=0.00\nlong_corridor", "bench",
         "./benchmark_force_sweep/kalman_react_force_0_025_05_near_robot3/force_0p00/people_dataset_force_0p00.json"),
        ("bench f=0.25\nlong_corridor", "bench",
         "./benchmark_force_sweep/kalman_react_force_0_025_05_near_robot3/force_0p25/people_dataset_force_0p25.json"),
        ("bench f=0.50\nlong_corridor", "bench",
         "./benchmark_force_sweep/kalman_react_force_0_025_05_near_robot3/force_0p50/people_dataset_force_0p50.json"),
        ("manual f=0.5\nlong_corridor", "manual",
         "./datasets/raw_benchmark_force/people_dataset_long_corridor.json"),
        ("manual f=0.5\nlabyrinth_turns", "manual",
         "./datasets/raw_benchmark_force/people_dataset_labyrinth_turns.json"),
        ("manual f=0.5\nnonlinear_corridor", "manual",
         "./datasets/raw_benchmark_force/people_dataset_nonlinear_corridor.json"),
    ]
    results = []
    for label, source, path in scenes:
        r = analyze(Path(path), label)
        results.append((label, source, r))

    categories = ["linear", "interaction", "dense_interaction", "turning", "stop_go", "complex"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 14))

    # --- (1) per-scene absolute counts ---
    x = np.arange(len(scenes))
    width = 0.13
    for i, cat in enumerate(categories):
        vals = [r[cat] for _, _, r in results]
        axes[0].bar(x + (i - 2.5) * width, vals, width, label=cat)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lab for lab, _, _ in results], fontsize=9)
    axes[0].set_ylabel("# near-robot cases")
    axes[0].set_title(f"Per-scene category counts (near_robot ≤ {NEAR_ROBOT_RADIUS}m, time-stride {TIME_STRIDE}s)")
    axes[0].legend(ncol=6, fontsize=9, loc="upper right")
    axes[0].grid(True, axis="y", alpha=0.3)
    ymax = axes[0].get_ylim()[1]
    for xi, (_, _, r) in zip(x, results):
        axes[0].text(xi, ymax * 0.97, f"dur={r['__duration_s']:.0f}s\nN={r['__near_cases']}",
                     ha="center", va="top", fontsize=7, color="#333")

    # --- (2) per-scene normalized (% of near-robot cases) ---
    for i, cat in enumerate(categories):
        vals = [100.0 * r[cat] / max(1, r["all"]) for _, _, r in results]
        axes[1].bar(x + (i - 2.5) * width, vals, width, label=cat)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([lab for lab, _, _ in results], fontsize=9)
    axes[1].set_ylabel("% of near-robot cases")
    axes[1].set_title("Per-scene category share (normalized to scene total)")
    axes[1].grid(True, axis="y", alpha=0.3)

    # --- (3) aggregated summary: bench-total vs manual-total ---
    agg: Dict[str, Dict[str, int]] = {
        "bench (3 forces, long_corridor)": {c: 0 for c in categories + ["all"]},
        "manual (3 scenes, force=0.5)":   {c: 0 for c in categories + ["all"]},
    }
    for _, source, r in results:
        key = "bench (3 forces, long_corridor)" if source == "bench" else "manual (3 scenes, force=0.5)"
        for c in categories + ["all"]:
            agg[key][c] += r[c]
    agg_keys = list(agg.keys())
    xa = np.arange(len(categories))
    width2 = 0.4
    for k_idx, key in enumerate(agg_keys):
        vals = [100.0 * agg[key][c] / max(1, agg[key]["all"]) for c in categories]
        axes[2].bar(xa + (k_idx - 0.5) * width2, vals, width2, label=f"{key}  (N={agg[key]['all']})")
    axes[2].set_xticks(xa)
    axes[2].set_xticklabels(categories, fontsize=10)
    axes[2].set_ylabel("% of near-robot cases")
    axes[2].set_title("Aggregated category share: bench vs manual (all scenes pooled)")
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = Path("./docs/figures/analysis/compare_bench_vs_manual_categories.png")
    fig.savefig(out, dpi=130)
    print("Saved:", out)

    # --- print summary table ---
    print("\nPer-scene category counts:")
    hdr = ["scene"] + categories + ["all", "dur_s", "near_cases"]
    print("  " + " | ".join(f"{h:>20}" for h in hdr))
    for lab, _src, r in results:
        row = [lab.replace("\n", " ")] + [str(r[c]) for c in categories] + [str(r["all"]), f"{r['__duration_s']:.0f}", str(r['__near_cases'])]
        print("  " + " | ".join(f"{v:>20}" for v in row))

    print("\nAggregated (% share):")
    for key in agg_keys:
        shares = " | ".join(f"{c}={100.0*agg[key][c]/max(1,agg[key]['all']):5.1f}%" for c in categories)
        print(f"  {key} (N={agg[key]['all']}): {shares}")


if __name__ == "__main__":
    main()
