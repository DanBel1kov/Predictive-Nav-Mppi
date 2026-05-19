#!/usr/bin/env python3
"""Convert our raw JSON datasets to SocialVAE format (frame_id  person_id  x  y, TSV).

Resamples to 0.4 s/frame (= obs_dt) so OB_HORIZON=8 covers 3.2 s past, PRED_HORIZON=26 covers 10.4 s future.

For each scene we create one train file and one test file using temporal split (last 25% → test).
Also injects the ROBOT as another agent with a fixed person_id (= 9000+i), so SocialVAE learns
human-robot interactions.

Output layout (matches SocialVAE conventions):
    /home/danbel1kov/SocialVAE/data/predictive_nav/
        train/long_corridor_react.txt
        train/labyrinth_react.txt
        ...
        test/long_corridor_react.txt
        ...
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SOURCES = [
    ("long_corridor_react", "/home/danbel1kov/predictive-nav-mppi/datasets/raw/long_corridor_react_better_rft.json"),
    ("labyrinth_react",     "/home/danbel1kov/predictive-nav-mppi/datasets/raw/labyrinth_react_test_better_rft.json"),
    ("nonlinear_react",     "/home/danbel1kov/predictive-nav-mppi/datasets/raw/nonlinear_corridor_react_better_rft.json"),
    ("long_corridor_v5loop", "/home/danbel1kov/predictive-nav-mppi/datasets/raw_v5loop/long_corridor_v5loop.json"),
    ("labyrinth_v5loop",     "/home/danbel1kov/predictive-nav-mppi/datasets/raw_v5loop/labyrinth_v5loop.json"),
    ("nonlinear_v5loop",     "/home/danbel1kov/predictive-nav-mppi/datasets/raw_v5loop/nonlinear_v5loop.json"),
]

OUT_ROOT = Path("/home/danbel1kov/SocialVAE/data/predictive_nav")
HOLDOUT_FRACTION = 0.25
TARGET_DT = 0.4
ROBOT_ID_OFFSET = 9000  # unique robot id per file


def resample_to_uniform(frames: List[dict], target_dt: float) -> List[dict]:
    """Pick frames closest to t0 + k*target_dt for k=0,1,2,..."""
    if not frames:
        return []
    t0 = frames[0]["t"]
    tN = frames[-1]["t"]
    n_steps = int((tN - t0) / target_dt) + 1
    ts = [fr["t"] for fr in frames]
    out = []
    j = 0
    for k in range(n_steps):
        target = t0 + k * target_dt
        while j + 1 < len(frames) and abs(ts[j + 1] - target) <= abs(ts[j] - target):
            j += 1
        out.append(frames[j])
    return out


def convert_one(label: str, path: Path, robot_pid: int) -> Tuple[List[str], List[str]]:
    """Returns (train_lines, test_lines)."""
    data = json.loads(path.read_text())
    frames = data["frames"]
    if not frames:
        return [], []
    resampled = resample_to_uniform(frames, TARGET_DT)
    n_split = int((1 - HOLDOUT_FRACTION) * len(resampled))
    train_frames = resampled[:n_split]
    test_frames = resampled[n_split:]

    def to_lines(fr_list: List[dict], frame_offset: int) -> List[str]:
        lines = []
        for k, fr in enumerate(fr_list):
            fid = float(frame_offset + k)
            for p in fr.get("people", []):
                pid = float(int(p["id"]))
                lines.append(f"{fid}\t{pid}\t{p['x']:.6f}\t{p['y']:.6f}")
            rb = fr.get("robot")
            if rb is not None and rb.get("x") is not None:
                lines.append(f"{fid}\t{float(robot_pid)}\t{rb['x']:.6f}\t{rb['y']:.6f}")
        return lines

    return to_lines(train_frames, 0), to_lines(test_frames, len(train_frames))


def main():
    (OUT_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "test").mkdir(parents=True, exist_ok=True)
    for i, (label, p) in enumerate(SOURCES):
        path = Path(p)
        if not path.exists():
            print(f"  skip (missing): {label}")
            continue
        tr, te = convert_one(label, path, ROBOT_ID_OFFSET + i)
        (OUT_ROOT / "train" / f"{label}.txt").write_text("\n".join(tr) + "\n")
        (OUT_ROOT / "test" / f"{label}.txt").write_text("\n".join(te) + "\n")
        print(f"  {label}: train={len(tr)} rows, test={len(te)} rows")
    print(f"\nWrote to {OUT_ROOT}")
    print("\nNext: train SocialVAE with --train data/predictive_nav/train --test data/predictive_nav/test --config config/predictive_nav.py")


if __name__ == "__main__":
    main()
