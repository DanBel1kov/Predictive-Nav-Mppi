#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from predictive_nav_mppi.benchmark_people_predictors import (
    Case,
    _classify_case_tags,
    _heading_change,
    _kalman_predict,
    _min_neighbor_distance,
    _sample_gt,
    _sample_obs,
    _speed_magnitudes,
)


@dataclass
class Candidate:
    case_id: str
    source_name: str
    source_path: str
    frame_index: int
    t: float
    person_id: int
    obs_xy: np.ndarray
    gt_xy: np.ndarray
    neigh_xy: List[np.ndarray]
    tags: Set[str]
    metrics: Dict[str, float]
    score: float
    partition: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "frame_index": self.frame_index,
            "t": self.t,
            "person_id": self.person_id,
            "obs_xy": np.round(self.obs_xy, 6).tolist(),
            "gt_xy": np.round(self.gt_xy, 6).tolist(),
            "neigh_xy": [np.round(arr, 6).tolist() for arr in self.neigh_xy],
            "tags": sorted(self.tags),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "score": round(self.score, 6),
            "partition": self.partition,
        }


def _dataset_name(path: Path) -> str:
    name = path.stem
    name = name.replace("people_dataset_", "")
    name = name.replace("columns_dataset_", "")
    return name


def _build_track_map(frames: Sequence[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    track_map: Dict[int, List[Tuple[float, float, float]]] = {}
    for fr in frames:
        t = float(fr["t"])
        for person in fr["people"]:
            pid = int(person["id"])
            track_map.setdefault(pid, []).append((t, float(person["x"]), float(person["y"])))
    return {
        pid: np.asarray(sorted(samples, key=lambda item: item[0]), dtype=np.float64)
        for pid, samples in track_map.items()
        if len(samples) >= 2
    }


def _candidate_metrics(obs_xy: np.ndarray, neigh_xy: List[np.ndarray], obs_dt: float) -> Dict[str, float]:
    path_len = float(np.sum(np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1))) if obs_xy.shape[0] >= 2 else 0.0
    displacement = float(np.linalg.norm(obs_xy[-1] - obs_xy[0])) if obs_xy.shape[0] >= 2 else 0.0
    speeds = _speed_magnitudes(obs_xy, obs_dt)
    heading_change_deg = math.degrees(_heading_change(obs_xy))
    case = Case(t=0.0, person_id=0, obs_xy=obs_xy, neigh_xy=neigh_xy, gt_xy=np.zeros((0, 2), dtype=np.float64))
    min_neighbor_distance = _min_neighbor_distance(case)
    curvature_ratio = path_len / max(displacement, 1e-6)
    return {
        "path_len": path_len,
        "displacement": displacement,
        "curvature_ratio": curvature_ratio,
        "heading_change_deg": heading_change_deg,
        "neighbor_count": float(len(neigh_xy)),
        "min_neighbor_distance": float(min_neighbor_distance if math.isfinite(min_neighbor_distance) else 999.0),
        "mean_speed": float(np.mean(speeds)) if speeds.size else 0.0,
        "max_speed": float(np.max(speeds)) if speeds.size else 0.0,
    }


def _candidate_score(tags: Set[str], metrics: Dict[str, float]) -> float:
    score = 0.0
    score += 3.0 if "very_complex" in tags else 0.0
    score += 2.4 if "complex" in tags else 0.0
    score += 1.8 if "turning" in tags else 0.0
    score += 1.4 if "dense_interaction" in tags else 0.0
    score += 1.0 if "interaction" in tags else 0.0
    score += 0.6 if "stop_go" in tags else 0.0
    score += min(1.5, metrics["heading_change_deg"] / 75.0)
    score += min(1.2, max(0.0, metrics["curvature_ratio"] - 1.0) * 2.0)
    score += min(1.0, metrics["neighbor_count"] / 4.0)
    if metrics["min_neighbor_distance"] < 999.0:
        score += min(1.0, 1.5 / max(0.3, metrics["min_neighbor_distance"])) - 0.5
    score += min(0.8, metrics["path_len"] / 4.0)
    if tags == {"all"}:
        score -= 0.4
    return float(score)


def _is_linear_case(candidate: Candidate, turn_threshold_deg: float) -> bool:
    if candidate.metrics["neighbor_count"] > 0.0:
        return False
    if "interaction" in candidate.tags or "dense_interaction" in candidate.tags:
        return False
    if "turning" in candidate.tags or "complex" in candidate.tags or "very_complex" in candidate.tags:
        return False
    return (
        candidate.metrics["heading_change_deg"] < max(12.0, 0.35 * turn_threshold_deg)
        and candidate.metrics["curvature_ratio"] < 1.05
    )


def _build_candidates_for_dataset(
    dataset_path: Path,
    obs_len: int,
    obs_dt: float,
    pred_len: int,
    pred_dt: float,
    neighbor_radius: float,
    max_neighbors: int,
    stride: int,
    holdout_fraction: float,
    interaction_dist: float,
    dense_neighbors_min: int,
    turn_threshold_deg: float,
    stop_speed_thresh: float,
    moving_speed_min: float,
    stop_go_delta: float,
) -> List[Candidate]:
    payload = json.loads(dataset_path.read_text())
    frames = payload["frames"]
    track_map = _build_track_map(frames)
    dataset_name = _dataset_name(dataset_path)
    total_frames = len(frames)
    out: List[Candidate] = []

    for frame_index in range(0, total_frames, max(1, stride)):
        fr = frames[frame_index]
        t0 = float(fr["t"])
        present = fr["people"]
        by_id = {int(person["id"]): person for person in present}
        ids = list(by_id.keys())
        partition = "benchmark" if frame_index >= int((1.0 - holdout_fraction) * total_frames) else "train"
        for pid in ids:
            track = track_map.get(pid)
            if track is None:
                continue
            obs_xy = _sample_obs(track, t0, obs_len, obs_dt)
            gt_xy = _sample_gt(track, t0, pred_len, pred_dt)
            if obs_xy is None or gt_xy is None:
                continue

            px = float(by_id[pid]["x"])
            py = float(by_id[pid]["y"])
            neigh_xy: List[np.ndarray] = []
            for oid in ids:
                if oid == pid:
                    continue
                ox = float(by_id[oid]["x"])
                oy = float(by_id[oid]["y"])
                if math.hypot(ox - px, oy - py) > neighbor_radius:
                    continue
                other_track = track_map.get(oid)
                if other_track is None:
                    continue
                other_obs = _sample_obs(other_track, t0, obs_len, obs_dt)
                if other_obs is None:
                    continue
                neigh_xy.append(other_obs)
            if len(neigh_xy) > max_neighbors:
                neigh_xy = neigh_xy[:max_neighbors]

            case = Case(t=t0, person_id=pid, obs_xy=obs_xy, neigh_xy=neigh_xy, gt_xy=gt_xy)
            tags = _classify_case_tags(
                case=case,
                obs_dt=obs_dt,
                interaction_dist=interaction_dist,
                dense_neighbors_min=dense_neighbors_min,
                turn_threshold_deg=turn_threshold_deg,
                stop_speed_thresh=stop_speed_thresh,
                stop_go_delta=stop_go_delta,
                moving_speed_min=moving_speed_min,
            )
            metrics = _candidate_metrics(obs_xy, neigh_xy, obs_dt)
            score = _candidate_score(tags, metrics)
            out.append(
                Candidate(
                    case_id=f"{dataset_name}_f{frame_index:05d}_p{pid:03d}",
                    source_name=dataset_name,
                    source_path=str(dataset_path),
                    frame_index=frame_index,
                    t=t0,
                    person_id=pid,
                    obs_xy=obs_xy,
                    gt_xy=gt_xy,
                    neigh_xy=neigh_xy,
                    tags=tags,
                    metrics=metrics,
                    score=score,
                    partition=partition,
                )
            )
    return out


def _sorted_unique(candidates: Iterable[Candidate], bucket_tag: Optional[str] = None) -> List[Candidate]:
    filtered = [c for c in candidates if bucket_tag is None or bucket_tag in c.tags]
    return sorted(filtered, key=lambda c: (-c.score, c.source_name, c.frame_index, c.person_id))


def _can_take(candidate: Candidate, selected_by_track: Dict[Tuple[str, int], List[int]], min_gap_frames: int) -> bool:
    anchors = selected_by_track[(candidate.source_name, candidate.person_id)]
    return all(abs(candidate.frame_index - existing) >= min_gap_frames for existing in anchors)


def _take_candidates(
    pool: Sequence[Candidate],
    target_count: int,
    min_gap_frames: int,
    source_quota: Optional[Dict[str, int]] = None,
) -> List[Candidate]:
    selected: List[Candidate] = []
    selected_ids: Set[str] = set()
    selected_by_track: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    source_counts: Counter[str] = Counter()

    for cand in pool:
        if len(selected) >= target_count:
            break
        if cand.case_id in selected_ids:
            continue
        if source_quota is not None and source_counts[cand.source_name] >= source_quota.get(cand.source_name, target_count):
            continue
        if not _can_take(cand, selected_by_track, min_gap_frames):
            continue
        selected.append(cand)
        selected_ids.add(cand.case_id)
        selected_by_track[(cand.source_name, cand.person_id)].append(cand.frame_index)
        source_counts[cand.source_name] += 1
    return selected


def _compose_selection(
    candidates: Sequence[Candidate],
    target_count: int,
    min_gap_frames: int,
    bucket_weights: Sequence[Tuple[Optional[str], float]],
    max_linear_fraction: float,
    turn_threshold_deg: float,
) -> List[Candidate]:
    if not candidates:
        return []

    linear_cases = [c for c in candidates if _is_linear_case(c, turn_threshold_deg)]
    diverse_cases = [c for c in candidates if not _is_linear_case(c, turn_threshold_deg)]
    linear_target = len(linear_cases)
    if diverse_cases and max_linear_fraction < 1.0:
        linear_target = min(
            linear_target,
            int(math.floor((max_linear_fraction / max(1e-6, 1.0 - max_linear_fraction)) * len(diverse_cases))),
        )

    filtered_linear = _take_candidates(
        pool=_sorted_unique(linear_cases, None),
        target_count=max(0, linear_target),
        min_gap_frames=min_gap_frames,
    )
    filtered_pool = list(diverse_cases) + filtered_linear
    effective_target = len(filtered_pool) if target_count <= 0 else min(target_count, len(filtered_pool))

    selected: List[Candidate] = []
    selected_ids: Set[str] = set()
    selected_by_track: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    def add_from_pool(pool: Sequence[Candidate], count: int) -> None:
        for cand in pool:
            if len(selected) >= effective_target or count <= 0:
                return
            if cand.case_id in selected_ids:
                continue
            if not _can_take(cand, selected_by_track, min_gap_frames):
                continue
            selected.append(cand)
            selected_ids.add(cand.case_id)
            selected_by_track[(cand.source_name, cand.person_id)].append(cand.frame_index)
            count -= 1

    for bucket_tag, weight in bucket_weights:
        quota = max(1, int(round(effective_target * weight)))
        pool = _sorted_unique(filtered_pool, bucket_tag)
        add_from_pool(pool, quota)

    if len(selected) < effective_target:
        add_from_pool(_sorted_unique(filtered_pool, None), effective_target - len(selected))

    return selected


def _tag_summary(cases: Sequence[Candidate]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for case in cases:
        for tag in case.tags:
            counter[tag] += 1
    return dict(sorted(counter.items()))


def _source_summary(cases: Sequence[Candidate]) -> Dict[str, int]:
    counter: Counter[str] = Counter(case.source_name for case in cases)
    return dict(sorted(counter.items()))


def _avg_score(cases: Sequence[Candidate]) -> float:
    if not cases:
        return 0.0
    return float(np.mean([case.score for case in cases]))


def _summary_payload(
    all_candidates: Sequence[Candidate],
    train_cases: Sequence[Candidate],
    benchmark_cases: Sequence[Candidate],
    args: argparse.Namespace,
    dataset_paths: Sequence[Path],
) -> Dict[str, Any]:
    candidates_by_partition = {
        "train": [c for c in all_candidates if c.partition == "train"],
        "benchmark": [c for c in all_candidates if c.partition == "benchmark"],
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in dataset_paths],
        "config": {
            "obs_len": args.obs_len,
            "obs_dt": args.obs_dt,
            "pred_len": args.pred_len,
            "pred_dt": args.pred_dt,
            "stride": args.stride,
            "neighbor_radius": args.neighbor_radius,
            "max_neighbors": args.max_neighbors,
            "holdout_fraction": args.holdout_fraction,
            "train_target": args.train_target,
            "benchmark_target": args.benchmark_target,
            "min_gap_frames": args.min_gap_frames,
            "max_linear_fraction": args.max_linear_fraction,
        },
        "candidates": {
            "total": len(all_candidates),
            "by_partition": {name: len(items) for name, items in candidates_by_partition.items()},
            "tag_counts": {
                name: _tag_summary(items) for name, items in candidates_by_partition.items()
            },
            "source_counts": {
                name: _source_summary(items) for name, items in candidates_by_partition.items()
            },
        },
        "selected": {
            "train": {
                "count": len(train_cases),
                "avg_score": round(_avg_score(train_cases), 6),
                "tag_counts": _tag_summary(train_cases),
                "source_counts": _source_summary(train_cases),
                "linear_cases": sum(_is_linear_case(case, args.turn_threshold_deg) for case in train_cases),
            },
            "benchmark": {
                "count": len(benchmark_cases),
                "avg_score": round(_avg_score(benchmark_cases), 6),
                "tag_counts": _tag_summary(benchmark_cases),
                "source_counts": _source_summary(benchmark_cases),
                "linear_cases": sum(_is_linear_case(case, args.turn_threshold_deg) for case in benchmark_cases),
            },
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _make_residual_cases_payload(
    cases: Sequence[Candidate],
    split_name: str,
    obs_dt: float,
    pred_dt: float,
) -> Dict[str, Any]:
    exported_cases: List[Dict[str, Any]] = []
    for case in cases:
        kalman_pred_xy = _kalman_predict(case.obs_xy, case.gt_xy.shape[0], obs_dt, pred_dt)
        residual_xy = case.gt_xy - kalman_pred_xy
        state_now = {
            "x": float(case.obs_xy[-1, 0]),
            "y": float(case.obs_xy[-1, 1]),
            "vx": float((case.obs_xy[-1, 0] - case.obs_xy[-2, 0]) / max(1e-6, obs_dt)) if case.obs_xy.shape[0] >= 2 else 0.0,
            "vy": float((case.obs_xy[-1, 1] - case.obs_xy[-2, 1]) / max(1e-6, obs_dt)) if case.obs_xy.shape[0] >= 2 else 0.0,
        }
        exported = case.to_json()
        exported["state_now"] = state_now
        exported["kalman_pred_xy"] = np.round(kalman_pred_xy, 6).tolist()
        exported["residual_xy"] = np.round(residual_xy, 6).tolist()
        exported["kalman_metrics"] = {
            "ade_full": round(float(np.mean(np.linalg.norm(residual_xy, axis=1))), 6),
            "fde_full": round(float(np.linalg.norm(residual_xy[-1])), 6),
        }
        exported_cases.append(exported)

    return {
        "meta": {
            "split": split_name,
            "description": "Curated cases augmented with Kalman predictions and residual targets.",
            "predictor_type": "kalman_residual_targets",
            "residual_definition": "residual_xy = gt_xy - kalman_pred_xy",
            "obs_dt": obs_dt,
            "pred_dt": pred_dt,
            "cases": len(exported_cases),
        },
        "cases": exported_cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate diverse train/benchmark trajectory datasets from recorded people logs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Input JSON datasets from record_people_dataset.",
    )
    parser.add_argument("--output_dir", default="datasets/curated_people")
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--obs_dt", type=float, default=0.4)
    parser.add_argument("--pred_len", type=int, default=26)
    parser.add_argument("--pred_dt", type=float, default=0.4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--neighbor_radius", type=float, default=2.0)
    parser.add_argument("--max_neighbors", type=int, default=16)
    parser.add_argument("--holdout_fraction", type=float, default=0.25)
    parser.add_argument("--train_target", type=int, default=0, help="0 means keep all filtered train cases.")
    parser.add_argument("--benchmark_target", type=int, default=0, help="0 means keep all filtered benchmark cases.")
    parser.add_argument("--min_gap_frames", type=int, default=15)
    parser.add_argument(
        "--max_linear_fraction",
        type=float,
        default=0.35,
        help="Maximum fraction of simple linear cases after filtering.",
    )
    parser.add_argument("--interaction_dist", type=float, default=1.5)
    parser.add_argument("--dense_neighbors_min", type=int, default=3)
    parser.add_argument("--turn_threshold_deg", type=float, default=45.0)
    parser.add_argument("--stop_speed_thresh", type=float, default=0.10)
    parser.add_argument("--moving_speed_min", type=float, default=0.25)
    parser.add_argument("--stop_go_delta", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = [Path(item).expanduser().resolve() for item in args.datasets]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: List[Candidate] = []
    for dataset_path in dataset_paths:
        all_candidates.extend(
            _build_candidates_for_dataset(
                dataset_path=dataset_path,
                obs_len=args.obs_len,
                obs_dt=args.obs_dt,
                pred_len=args.pred_len,
                pred_dt=args.pred_dt,
                neighbor_radius=args.neighbor_radius,
                max_neighbors=args.max_neighbors,
                stride=args.stride,
                holdout_fraction=args.holdout_fraction,
                interaction_dist=args.interaction_dist,
                dense_neighbors_min=args.dense_neighbors_min,
                turn_threshold_deg=args.turn_threshold_deg,
                stop_speed_thresh=args.stop_speed_thresh,
                moving_speed_min=args.moving_speed_min,
                stop_go_delta=args.stop_go_delta,
            )
        )

    train_pool = [c for c in all_candidates if c.partition == "train"]
    benchmark_pool = [c for c in all_candidates if c.partition == "benchmark"]

    train_weights: Sequence[Tuple[Optional[str], float]] = (
        ("very_complex", 0.15),
        ("complex", 0.25),
        ("turning", 0.20),
        ("dense_interaction", 0.15),
        ("interaction", 0.10),
        (None, 0.15),
    )
    benchmark_weights: Sequence[Tuple[Optional[str], float]] = (
        ("very_complex", 0.20),
        ("complex", 0.25),
        ("turning", 0.20),
        ("dense_interaction", 0.15),
        ("interaction", 0.10),
        (None, 0.10),
    )

    train_cases = _compose_selection(
        candidates=train_pool,
        target_count=args.train_target,
        min_gap_frames=args.min_gap_frames,
        bucket_weights=train_weights,
        max_linear_fraction=args.max_linear_fraction,
        turn_threshold_deg=args.turn_threshold_deg,
    )
    benchmark_cases = _compose_selection(
        candidates=benchmark_pool,
        target_count=args.benchmark_target,
        min_gap_frames=args.min_gap_frames,
        bucket_weights=benchmark_weights,
        max_linear_fraction=args.max_linear_fraction,
        turn_threshold_deg=args.turn_threshold_deg,
    )

    train_payload = {
        "meta": {
            "split": "train",
            "description": "Curated diverse training windows from recorded people trajectories.",
        },
        "cases": [case.to_json() for case in train_cases],
    }
    benchmark_payload = {
        "meta": {
            "split": "benchmark",
            "description": "Curated holdout benchmark windows with emphasis on difficult nonlinear interactions.",
        },
        "cases": [case.to_json() for case in benchmark_cases],
    }
    summary_payload = _summary_payload(
        all_candidates=all_candidates,
        train_cases=train_cases,
        benchmark_cases=benchmark_cases,
        args=args,
        dataset_paths=dataset_paths,
    )
    train_residual_payload = _make_residual_cases_payload(
        cases=train_cases,
        split_name="train",
        obs_dt=args.obs_dt,
        pred_dt=args.pred_dt,
    )
    benchmark_residual_payload = _make_residual_cases_payload(
        cases=benchmark_cases,
        split_name="benchmark",
        obs_dt=args.obs_dt,
        pred_dt=args.pred_dt,
    )

    _write_json(out_dir / "train_cases.json", train_payload)
    _write_json(out_dir / "benchmark_cases.json", benchmark_payload)
    _write_json(out_dir / "train_residual_cases.json", train_residual_payload)
    _write_json(out_dir / "benchmark_residual_cases.json", benchmark_residual_payload)
    _write_json(out_dir / "summary.json", summary_payload)

    print(f"[curate] wrote train cases: {out_dir / 'train_cases.json'} ({len(train_cases)})")
    print(f"[curate] wrote benchmark cases: {out_dir / 'benchmark_cases.json'} ({len(benchmark_cases)})")
    print(f"[curate] wrote train residual cases: {out_dir / 'train_residual_cases.json'} ({len(train_cases)})")
    print(f"[curate] wrote benchmark residual cases: {out_dir / 'benchmark_residual_cases.json'} ({len(benchmark_cases)})")
    print(f"[curate] wrote summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
