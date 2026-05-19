#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from predictive_nav_mppi.scene_context import default_scene_map_path, load_occupancy_scene_map


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_scene_name(case: Dict[str, Any]) -> str:
    source_name = str(case.get("source_name", "")).strip()
    if source_name:
        return source_name
    source_path = str(case.get("source_path", "")).strip()
    if source_path:
        stem = Path(source_path).stem
        for prefix in ("people_dataset_", "columns_dataset_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        return stem or Path(source_path).parent.name
    return "dataset"


def _guess_map_yaml(scene_name: str) -> Optional[Path]:
    try:
        return default_scene_map_path(scene_name)
    except Exception:
        pass

    maps_dir = _repo_root() / "maps"
    candidates = [
        maps_dir / f"{scene_name}_map.yaml",
        maps_dir / f"{scene_name}.yaml",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    lowered = scene_name.lower()
    matches = sorted(p for p in maps_dir.glob("*.yaml") if lowered in p.stem.lower())
    return matches[0] if matches else None


def _load_cases(curated_dir: Path, split: str) -> List[Dict[str, Any]]:
    if split == "train":
        path = curated_dir / "train_cases.json"
        return json.loads(path.read_text())["cases"]
    if split == "benchmark":
        path = curated_dir / "benchmark_cases.json"
        return json.loads(path.read_text())["cases"]
    if split == "both":
        cases: List[Dict[str, Any]] = []
        for name in ("train_cases.json", "benchmark_cases.json"):
            path = curated_dir / name
            if path.exists():
                cases.extend(json.loads(path.read_text())["cases"])
        return cases
    raise ValueError(f"Unsupported split={split!r}")


def _is_linear_case(case: Dict[str, Any], turn_threshold_deg: float) -> bool:
    metrics = case.get("metrics", {})
    heading_change_deg = float(metrics.get("heading_change_deg", 999.0))
    curvature_ratio = float(metrics.get("curvature_ratio", 999.0))
    return (
        heading_change_deg < max(12.0, 0.35 * turn_threshold_deg)
        and curvature_ratio < 1.05
    )


def _case_matches_category(case: Dict[str, Any], category: str, turn_threshold_deg: float) -> bool:
    if category == "linear":
        return _is_linear_case(case, turn_threshold_deg)
    return category in set(case.get("tags", []))


def _velocity_arrow(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] >= 2:
        return points_xy[-1] - points_xy[-2]
    return np.zeros((2,), dtype=np.float64)


_RAW_SOURCE_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def _load_raw_robot_track(source_path: str) -> Optional[Dict[str, np.ndarray]]:
    key = str(Path(source_path).expanduser().resolve())
    cached = _RAW_SOURCE_CACHE.get(key)
    if cached is not None:
        return cached
    path = Path(key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    frames = payload.get("frames", [])
    samples: List[Tuple[float, float, float]] = []
    for fr in frames:
        robot = fr.get("robot")
        if robot is None:
            continue
        t = float(robot.get("t", fr.get("t", 0.0)))
        samples.append((t, float(robot["x"]), float(robot["y"])))
    if len(samples) < 2:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    track = {"t": arr[:, 0], "x": arr[:, 1], "y": arr[:, 2]}
    _RAW_SOURCE_CACHE[key] = track
    return track


def _sample_track_window(
    track: Optional[Dict[str, np.ndarray]],
    t0: float,
    obs_len: int,
    pred_len: int,
    dt: float,
) -> Optional[np.ndarray]:
    if track is None:
        return None
    ts = np.asarray(track["t"], dtype=np.float64)
    xs = np.asarray(track["x"], dtype=np.float64)
    ys = np.asarray(track["y"], dtype=np.float64)
    if ts.size < 2:
        return None
    rel_times = np.concatenate([
        -np.arange(obs_len - 1, -1, -1, dtype=np.float64) * dt,
        np.arange(1, pred_len + 1, dtype=np.float64) * dt,
    ])
    sample_t = t0 + rel_times
    if sample_t[0] < ts[0] or sample_t[-1] > ts[-1]:
        return None
    x = np.interp(sample_t, ts, xs)
    y = np.interp(sample_t, ts, ys)
    return np.stack([x, y], axis=1)


def _path_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)))


def _draw_map(ax, scene_name: str) -> None:
    map_yaml = _guess_map_yaml(scene_name)
    if map_yaml is None or not map_yaml.exists():
        return
    scene_map = load_occupancy_scene_map(map_yaml)
    # Occupancy image is stored in top-left raster convention; flip for world-frame plotting.
    img = np.flipud(1.0 - scene_map.occupied_mask)
    origin_x, origin_y = scene_map.origin_xy
    extent = [
        origin_x,
        origin_x + scene_map.width * scene_map.resolution,
        origin_y,
        origin_y + scene_map.height * scene_map.resolution,
    ]
    ax.imshow(img, extent=extent, origin="lower", cmap="gray", alpha=0.35, vmin=0.0, vmax=1.0)


def _plot_case(
    ax,
    case: Dict[str, Any],
    zoom_radius: float,
    turn_threshold_deg: float,
    obs_len: int,
    pred_len: int,
    dt: float,
) -> None:
    scene_name = _normalize_scene_name(case)
    _draw_map(ax, scene_name)

    obs = np.asarray(case["obs_xy"], dtype=np.float64)
    gt = np.asarray(case["gt_xy"], dtype=np.float64)
    neigh = [np.asarray(arr, dtype=np.float64) for arr in case.get("neigh_xy", [])]
    robot_obs = np.asarray(case.get("robot_obs_xy", []), dtype=np.float64)
    robot_window = _sample_track_window(
        _load_raw_robot_track(str(case.get("source_path", ""))),
        t0=float(case.get("t", 0.0)),
        obs_len=int(obs_len),
        pred_len=int(pred_len),
        dt=float(dt),
    )

    target_now = obs[-1]
    robot_now = robot_window[obs_len - 1] if robot_window is not None else (robot_obs[-1] if robot_obs.size else target_now)
    center = robot_now

    # robot
    if robot_window is not None:
        robot_past = robot_window[:obs_len]
        robot_future = robot_window[obs_len - 1 :]
        ax.plot(robot_past[:, 0], robot_past[:, 1], color="black", linewidth=2.6, label="robot past")
        ax.plot(robot_future[:, 0], robot_future[:, 1], color="black", linewidth=2.2, linestyle="--", label="robot future")
        rv = _velocity_arrow(robot_past)
        ax.scatter([robot_now[0]], [robot_now[1]], color="black", s=35, marker="s", zorder=5)
        ax.arrow(robot_now[0], robot_now[1], rv[0], rv[1], color="black", width=0.01,
                 head_width=0.10, length_includes_head=True, alpha=0.9)
    elif robot_obs.size:
        ax.plot(robot_obs[:, 0], robot_obs[:, 1], color="black", linewidth=2.5, label="robot")
        rv = _velocity_arrow(robot_obs)
        ax.scatter([robot_now[0]], [robot_now[1]], color="black", s=35, marker="s", zorder=5)
        ax.arrow(robot_now[0], robot_now[1], rv[0], rv[1], color="black", width=0.01,
                 head_width=0.10, length_includes_head=True, alpha=0.9)

    # target human
    full_target = np.vstack([obs, gt]) if gt.size else obs
    ax.plot(obs[:, 0], obs[:, 1], color="#1f77b4", linewidth=2.0, label="target past")
    if gt.size:
        ax.plot(full_target[obs.shape[0] - 1 :, 0], full_target[obs.shape[0] - 1 :, 1],
                color="#1f77b4", linewidth=1.8, linestyle="--", label="target future")
    tv = _velocity_arrow(obs)
    ax.scatter([target_now[0]], [target_now[1]], color="#1f77b4", s=32, zorder=6)
    ax.arrow(target_now[0], target_now[1], tv[0], tv[1], color="#1f77b4", width=0.008,
             head_width=0.08, length_includes_head=True, alpha=0.9)

    # neighbors
    for idx, arr in enumerate(neigh):
        color = "#ff7f0e" if idx < 4 else "#2ca02c"
        ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=1.3, alpha=0.85)
        nv = _velocity_arrow(arr)
        now = arr[-1]
        ax.scatter([now[0]], [now[1]], color=color, s=20, alpha=0.9)
        ax.arrow(now[0], now[1], nv[0], nv[1], color=color, width=0.006,
                 head_width=0.06, length_includes_head=True, alpha=0.65)

    tags = sorted(set(case.get("tags", [])) - {"all"})
    if _is_linear_case(case, turn_threshold_deg):
        tags = ["linear"] + [t for t in tags if t != "linear"]

    ax.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
    ax.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_title(
        f"{scene_name}\n{case.get('case_id', 'case')} | {', '.join(tags[:3]) if tags else 'all'}",
        fontsize=9,
    )
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")


def _interesting_case(
    case: Dict[str, Any],
    obs_len: int,
    pred_len: int,
    dt: float,
    min_robot_path_len: float,
    min_target_path_len: float,
) -> bool:
    target_window = np.vstack([
        np.asarray(case["obs_xy"], dtype=np.float64),
        np.asarray(case["gt_xy"], dtype=np.float64),
    ])
    if _path_length(target_window) < float(min_target_path_len):
        return False

    robot_window = _sample_track_window(
        _load_raw_robot_track(str(case.get("source_path", ""))),
        t0=float(case.get("t", 0.0)),
        obs_len=int(obs_len),
        pred_len=int(pred_len),
        dt=float(dt),
    )
    if robot_window is None:
        return False
    return _path_length(robot_window) >= float(min_robot_path_len)


def _write_gallery(
    cases: Sequence[Dict[str, Any]],
    scene_name: str,
    category: str,
    out_path: Path,
    per_group: int,
    zoom_radius: float,
    turn_threshold_deg: float,
    obs_len: int,
    pred_len: int,
    dt: float,
    min_robot_path_len: float,
    min_target_path_len: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping {out_path}")
        return

    interesting = [
        c for c in cases
        if _interesting_case(
            c,
            obs_len=int(obs_len),
            pred_len=int(pred_len),
            dt=float(dt),
            min_robot_path_len=float(min_robot_path_len),
            min_target_path_len=float(min_target_path_len),
        )
    ]
    pool = interesting if interesting else list(cases)
    selected = sorted(pool, key=lambda c: (-float(c.get("score", 0.0)), str(c.get("case_id", ""))))[:per_group]
    if not selected:
        return

    cols = min(3, len(selected))
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.8 * rows), squeeze=False)
    axes_flat = list(axes.reshape(-1))

    for ax, case in zip(axes_flat, selected):
        _plot_case(
            ax,
            case=case,
            zoom_radius=zoom_radius,
            turn_threshold_deg=turn_threshold_deg,
            obs_len=int(obs_len),
            pred_len=int(pred_len),
            dt=float(dt),
        )

    for ax in axes_flat[len(selected):]:
        ax.axis("off")

    fig.suptitle(f"{scene_name} | {category} | top {len(selected)} cases", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


def _write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    (out_dir / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize curated human-robot interaction cases by scene and category.")
    parser.add_argument("--curated_dir", required=True, help="Directory containing train_cases.json / benchmark_cases.json.")
    parser.add_argument("--output_dir", default="", help="Where to save galleries. Default: <curated_dir>/visualizations")
    parser.add_argument("--split", choices=("train", "benchmark", "both"), default="both")
    parser.add_argument("--categories", default="linear,interaction,dense_interaction,turning,stop_go,complex,very_complex",
                        help="Comma-separated category list.")
    parser.add_argument("--examples_per_group", type=int, default=6)
    parser.add_argument("--zoom_radius", type=float, default=6.0, help="Half-width of local snapshot window in meters.")
    parser.add_argument("--turn_threshold_deg", type=float, default=45.0)
    parser.add_argument("--min_cases", type=int, default=1, help="Skip empty/small groups below this count.")
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=26)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--min_robot_path_len", type=float, default=1.2,
                        help="Minimum robot path length over the full 34-step window to keep a case interesting.")
    parser.add_argument("--min_target_path_len", type=float, default=1.0,
                        help="Minimum target-human path length over the full 34-step window to keep a case interesting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curated_dir = Path(args.curated_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else curated_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = [item.strip() for item in str(args.categories).split(",") if item.strip()]
    cases = _load_cases(curated_dir, args.split)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        scene_name = _normalize_scene_name(case)
        for category in categories:
            if _case_matches_category(case, category, float(args.turn_threshold_deg)):
                grouped[(scene_name, category)].append(case)

    manifest: Dict[str, Any] = {
        "curated_dir": str(curated_dir),
        "split": args.split,
        "categories": categories,
        "examples_per_group": int(args.examples_per_group),
        "zoom_radius": float(args.zoom_radius),
        "groups": [],
    }

    for (scene_name, category), group_cases in sorted(grouped.items()):
        if len(group_cases) < int(args.min_cases):
            continue
        scene_dir = out_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_path = scene_dir / f"{category}.png"
        _write_gallery(
            cases=group_cases,
            scene_name=scene_name,
            category=category,
            out_path=out_path,
            per_group=int(args.examples_per_group),
            zoom_radius=float(args.zoom_radius),
            turn_threshold_deg=float(args.turn_threshold_deg),
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            dt=float(args.dt),
            min_robot_path_len=float(args.min_robot_path_len),
            min_target_path_len=float(args.min_target_path_len),
        )
        manifest["groups"].append({
            "scene": scene_name,
            "category": category,
            "count": len(group_cases),
            "path": str(out_path),
        })

    _write_manifest(out_dir, manifest)
    print(f"visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
