#!/usr/bin/env python3
"""Comprehensive comparison of all 9 raw datasets.

Reports per scene:
- frames, duration, # trajectories (people IDs)
- # samples that can be curated (full obs+gt window)
- per movement-tag breakdown (linear / turning / interaction / dense / stop_go / complex / very_complex)
- distribution of #neighbors in robot visibility (within 6m)
- distribution of distances target→neighbors (to argue k_neighbors=3 vs 5)
- distribution of distances robot→target_person
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Mirror curate logic
sys.path.insert(0, "./src/predictive_nav_mppi")
from predictive_nav_mppi.benchmark_people_predictors import (
    _classify_case_tags, _sample_obs, _sample_gt, Case
)

SOURCES = [
    ("react_v2/long_corridor", "./datasets/raw/long_corridor_react_better_rft.json"),
    ("react_v2/labyrinth",     "./datasets/raw/labyrinth_react_test_better_rft.json"),
    ("react_v2/nonlinear",     "./datasets/raw/nonlinear_corridor_react_better_rft.json"),
    ("v5loop/long_corridor",   "./datasets/raw_v5loop/long_corridor_v5loop.json"),
    ("v5loop/labyrinth",       "./datasets/raw_v5loop/labyrinth_v5loop.json"),
    ("v5loop/nonlinear",       "./datasets/raw_v5loop/nonlinear_v5loop.json"),
]

OBS_LEN, OBS_DT = 8, 0.4
PRED_LEN, PRED_DT = 26, 0.4
STRIDE = 5
NEIGHBOR_RADIUS = 2.0    # for tags
VISIBILITY_RADIUS = 6.0  # what robot "sees"


def _build_tracks(frames):
    tracks = defaultdict(list)
    for fr in frames:
        t = float(fr["t"])
        for p in fr.get("people", []):
            pid = int(p["id"])
            tracks[pid].append((t, float(p["x"]), float(p["y"])))
    return {pid: np.asarray(sorted(s), dtype=np.float64) for pid, s in tracks.items() if len(s) >= 2}


def _robot_track(frames):
    samples = []
    for fr in frames:
        r = fr.get("robot")
        if r is None or r.get("x") is None: continue
        samples.append((float(fr["t"]), float(r["x"]), float(r["y"])))
    return np.asarray(sorted(samples), dtype=np.float64) if len(samples) >= 2 else None


def analyze(label: str, path: Path):
    if not path.exists():
        return {"label": label, "missing": True}
    data = json.loads(path.read_text())
    frames = data["frames"]
    if not frames:
        return {"label": label, "frames": 0}
    ts = [fr["t"] for fr in frames]
    duration = ts[-1] - ts[0]
    tracks = _build_tracks(frames)
    robot_tr = _robot_track(frames)

    # Per-frame: # people, # in visibility, distances to robot
    n_per_frame, n_visible_per_frame = [], []
    dist_to_robot = []
    for fr in frames:
        people = fr.get("people", [])
        n_per_frame.append(len(people))
        if fr.get("robot") and fr["robot"].get("x") is not None:
            rx, ry = fr["robot"]["x"], fr["robot"]["y"]
            visible = 0
            for p in people:
                d = math.hypot(p["x"] - rx, p["y"] - ry)
                dist_to_robot.append(d)
                if d <= VISIBILITY_RADIUS:
                    visible += 1
            n_visible_per_frame.append(visible)

    # Build curated-like cases for tag analysis
    tag_counter = Counter()
    case_count_full = 0  # cases with complete obs+gt
    neighbors_per_case_2m, neighbors_per_case_6m = [], []
    dists_to_neighbors_in_case = []  # distance from target person to its neighbors
    for frame_index in range(0, len(frames), STRIDE):
        fr = frames[frame_index]
        t0 = float(fr["t"])
        by_id = {int(p["id"]): p for p in fr.get("people", [])}
        for pid in by_id:
            track = tracks.get(pid)
            if track is None: continue
            obs = _sample_obs(track, t0, OBS_LEN, OBS_DT)
            gt = _sample_gt(track, t0, PRED_LEN, PRED_DT)
            if obs is None or gt is None: continue
            case_count_full += 1

            # Collect neighbors at this frame
            px, py = float(by_id[pid]["x"]), float(by_id[pid]["y"])
            neigh_2m, neigh_6m, dists = 0, 0, []
            for oid, op in by_id.items():
                if oid == pid: continue
                d = math.hypot(op["x"] - px, op["y"] - py)
                dists.append(d)
                if d <= 2.0: neigh_2m += 1
                if d <= 6.0: neigh_6m += 1
            neighbors_per_case_2m.append(neigh_2m)
            neighbors_per_case_6m.append(neigh_6m)
            dists_to_neighbors_in_case.extend(dists[:10])  # top 10 to avoid blow up

            # Build Case for tag classification
            neigh_xy = []
            for oid in by_id:
                if oid == pid: continue
                otr = tracks.get(oid)
                if otr is None: continue
                ox, oy = float(by_id[oid]["x"]), float(by_id[oid]["y"])
                if math.hypot(ox - px, oy - py) > NEIGHBOR_RADIUS: continue
                oobs = _sample_obs(otr, t0, OBS_LEN, OBS_DT)
                if oobs is None: continue
                neigh_xy.append(oobs)
            case = Case(t=t0, person_id=pid, obs_xy=obs, neigh_xy=neigh_xy, gt_xy=gt, source_name=label)
            tags = _classify_case_tags(
                case=case, obs_dt=OBS_DT, interaction_dist=1.5, dense_neighbors_min=3,
                turn_threshold_deg=45.0, stop_speed_thresh=0.10, stop_go_delta=0.25, moving_speed_min=0.25,
            )
            # Add a "linear" tag manually: target moves in roughly straight line, no other tag fits
            if not (tags & {"turning", "stop_go", "complex", "very_complex"}):
                tags.add("linear")
            for tag in tags:
                tag_counter[tag] += 1

    return {
        "label": label,
        "frames": len(frames),
        "duration_s": duration,
        "trajectories": len(tracks),
        "people_per_frame_mean": np.mean(n_per_frame),
        "people_per_frame_max": max(n_per_frame),
        "visible_per_frame_mean": np.mean(n_visible_per_frame) if n_visible_per_frame else 0,
        "visible_per_frame_max": max(n_visible_per_frame) if n_visible_per_frame else 0,
        "dist_to_robot_p50": np.median(dist_to_robot) if dist_to_robot else 0,
        "dist_to_robot_p90": np.percentile(dist_to_robot, 90) if dist_to_robot else 0,
        "cases_full": case_count_full,
        "tags": dict(tag_counter),
        "neighbors_2m_dist": Counter(neighbors_per_case_2m),
        "neighbors_6m_dist": Counter(neighbors_per_case_6m),
        "neigh_dist_p50": np.median(dists_to_neighbors_in_case) if dists_to_neighbors_in_case else 0,
        "neigh_dist_p90": np.percentile(dists_to_neighbors_in_case, 90) if dists_to_neighbors_in_case else 0,
        "neigh_dist_p99": np.percentile(dists_to_neighbors_in_case, 99) if dists_to_neighbors_in_case else 0,
    }


def main():
    results = [analyze(lab, Path(p)) for lab, p in SOURCES]

    print("\n" + "="*120)
    print("OVERVIEW: frames / trajectories / cases")
    print("="*120)
    print(f"{'label':<25} {'frames':>7} {'dur,s':>7} {'tracks':>7} {'cases':>7} {'people/fr':>10} {'visible/fr':>11}")
    for r in results:
        if r.get("missing"):
            print(f"{r['label']:<25} MISSING")
            continue
        print(f"{r['label']:<25} {r['frames']:>7} {r['duration_s']:>7.1f} {r['trajectories']:>7} {r['cases_full']:>7} "
              f"{r['people_per_frame_mean']:>7.1f}/{r['people_per_frame_max']:>2} "
              f"{r['visible_per_frame_mean']:>8.1f}/{r['visible_per_frame_max']:>2}")

    print("\n" + "="*120)
    print("TAGS (% of cases). Note: each case can have multiple tags. 'linear' = target walks without significant turn/stop_go")
    print("="*120)
    TAGS = ["linear", "turning", "interaction", "dense_interaction", "stop_go", "complex", "very_complex"]
    header = f"{'label':<25} " + " ".join(f"{t:>10}" for t in TAGS) + "  total"
    print(header)
    for r in results:
        if r.get("missing"): continue
        total = r["cases_full"]
        row = f"{r['label']:<25} "
        for t in TAGS:
            n = r["tags"].get(t, 0)
            pct = 100.0 * n / total if total else 0.0
            row += f"{n:>5}({pct:>3.0f}%) "
        row += f"  {total}"
        print(row)

    print("\n" + "="*120)
    print("NEIGHBOR DISTRIBUTION (in radius 2m around target person)")
    print("how many cases have N neighbors within 2m → does k_neighbors=3 cover most cases?")
    print("="*120)
    print(f"{'label':<25}  {'0':>5}  {'1':>5}  {'2':>5}  {'3':>5}  {'4':>5}  {'5+':>5}  {'≥4 (%)':>8}")
    for r in results:
        if r.get("missing"): continue
        d = r["neighbors_2m_dist"]
        total = sum(d.values())
        n0 = d.get(0, 0); n1 = d.get(1, 0); n2 = d.get(2, 0); n3 = d.get(3, 0); n4 = d.get(4, 0)
        n5p = sum(c for n, c in d.items() if n >= 5)
        n_ge4 = sum(c for n, c in d.items() if n >= 4)
        pct_ge4 = 100.0 * n_ge4 / total if total else 0.0
        print(f"{r['label']:<25}  {n0:>5}  {n1:>5}  {n2:>5}  {n3:>5}  {n4:>5}  {n5p:>5}  {pct_ge4:>7.1f}%")

    print("\n" + "="*120)
    print("NEIGHBOR DISTANCES (target → neighbor, in meters)")
    print("="*120)
    print(f"{'label':<25}  {'p50':>6}  {'p90':>6}  {'p99':>6}")
    for r in results:
        if r.get("missing"): continue
        print(f"{r['label']:<25}  {r['neigh_dist_p50']:>6.2f}  {r['neigh_dist_p90']:>6.2f}  {r['neigh_dist_p99']:>6.2f}")

    print("\n" + "="*120)
    print("ROBOT-PERSON DISTANCE")
    print("="*120)
    print(f"{'label':<25}  {'p50':>6}  {'p90':>6}")
    for r in results:
        if r.get("missing"): continue
        print(f"{r['label']:<25}  {r['dist_to_robot_p50']:>6.2f}  {r['dist_to_robot_p90']:>6.2f}")


if __name__ == "__main__":
    main()
