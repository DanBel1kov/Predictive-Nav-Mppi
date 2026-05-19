#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _force_label(force: float) -> str:
    return f"force_{force:.2f}".replace(".", "p")


def _prepare_cfg(base_cfg: Dict[str, Any], episode_count: int, output_dir: Path) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    bench = cfg["benchmark"]
    goals = list(bench.get("goals", []))
    if not goals:
        raise SystemExit("benchmark.goals is empty")
    if len(goals) < episode_count:
        raise SystemExit(
            f"Requested {episode_count} unique episodes, but config only has {len(goals)} goals. "
            "Add more goals or lower --episode-count."
        )
    bench["goals"] = goals[:episode_count]
    bench["repeats"] = 1
    bench["repeat_offset"] = 0
    bench["output_dir"] = str(output_dir)
    return cfg


def _start_raw_recorder(
    *,
    raw_dataset_path: Path,
    force: float,
    map_name: str,
    visibility_radius: float,
    visibility_fov_deg: float,
) -> subprocess.Popen:
    cmd = [
        "ros2", "run", "predictive_nav_mppi", "record_people_dataset",
        "--ros-args",
        "-p", "input_topic:=/people",
        "-p", f"output_path:={raw_dataset_path}",
        "-p", f"map_name:={map_name}",
        "-p", f"robot_force_scale:={float(force)}",
        "-p", f"visibility_radius:={float(visibility_radius)}",
        "-p", f"visibility_fov_deg:={float(visibility_fov_deg)}",
        "-p", "require_robot_pose:=true",
    ]
    return subprocess.Popen(cmd, preexec_fn=os.setsid)


def _stop_recorder(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        proc.wait(timeout=10.0)


def _run_force(
    cfg_path: Path,
    force: float,
    predictor: str,
    *,
    raw_dataset_path: Path | None = None,
    map_name: str = "",
    visibility_radius: float = 100.0,
    visibility_fov_deg: float = 360.0,
) -> Path:
    recorder_proc = None
    if raw_dataset_path is not None:
        recorder_proc = _start_raw_recorder(
            raw_dataset_path=raw_dataset_path,
            force=force,
            map_name=map_name,
            visibility_radius=visibility_radius,
            visibility_fov_deg=visibility_fov_deg,
        )
    cmd = [
        "ros2", "run", "predictive_nav_mppi", "run_benchmark",
        "--config", str(cfg_path),
        "--predictor", predictor,
        "--robot-force-scale", str(float(force)),
    ]
    if abs(float(force)) <= 1e-9:
        cmd.append("--humans-ignore-robot")
    else:
        cmd.append("--humans-react-to-robot")
    try:
        subprocess.run(cmd, check=True)
    finally:
        if recorder_proc is not None:
            _stop_recorder(recorder_proc)
    output_root = Path(yaml.safe_load(cfg_path.read_text())["benchmark"]["output_dir"])
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No benchmark output folder created under {output_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _sum_tag_counts(*maps: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in maps:
        for k, v in m.items():
            out[k] = out.get(k, 0) + int(v)
    return dict(sorted(out.items()))


def _write_interaction_summary_from_curated(curated_dir: Path, run_dir: Path) -> Path:
    summary = json.loads((curated_dir / "summary.json").read_text())
    train_cases_payload = json.loads((curated_dir / "train_cases.json").read_text())
    benchmark_cases_payload = json.loads((curated_dir / "benchmark_cases.json").read_text())

    selected = summary.get("selected", {})
    train = selected.get("train", {})
    bench = selected.get("benchmark", {})
    counts = _sum_tag_counts(
        train.get("tag_counts", {}),
        bench.get("tag_counts", {}),
    )
    counts["linear"] = int(train.get("linear_cases", 0)) + int(bench.get("linear_cases", 0))
    counts = {k: int(v) for k, v in counts.items() if int(v) > 0}

    examples: List[Dict[str, Any]] = []
    for payload in (train_cases_payload, benchmark_cases_payload):
        for case in payload.get("cases", []):
            tags = [str(t) for t in case.get("tags", []) if str(t) != "all"]
            categories = sorted(set(tags))
            if not categories:
                metrics = case.get("metrics", {})
                heading_change_deg = float(metrics.get("heading_change_deg", 0.0))
                curvature_ratio = float(metrics.get("curvature_ratio", 999.0))
                neighbor_count = int(metrics.get("neighbor_count", 0))
                if neighbor_count == 0 and heading_change_deg < 12.0 and curvature_ratio < 1.05:
                    categories = ["linear"]
            if not categories:
                continue
            examples.append({
                "case_id": case.get("case_id", ""),
                "source_name": case.get("source_name", ""),
                "frame_index": int(case.get("frame_index", -1)),
                "person_id": int(case.get("person_id", -1)),
                "t": float(case.get("t", 0.0)),
                "categories": categories,
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
    return out_path


def _postprocess_force(
    *,
    raw_dataset_path: Path,
    run_dir: Path,
    near_robot_radius: float,
    near_robot_fov_deg: float,
) -> Path:
    curated_dir = run_dir / "curated_near_robot"
    cmd = [
        "ros2", "run", "predictive_nav_mppi", "curate_people_dataset",
        "--datasets", str(raw_dataset_path),
        "--output_dir", str(curated_dir),
        "--near_robot_radius", str(float(near_robot_radius)),
        "--near_robot_fov_deg", str(float(near_robot_fov_deg)),
    ]
    subprocess.run(cmd, check=True)
    _write_interaction_summary_from_curated(curated_dir, run_dir)
    return curated_dir


def _run_study_postprocess(module_name: str, study_dir: Path) -> None:
    cmd = [sys.executable, "-m", module_name, "--study-dir", str(study_dir)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run navigation benchmark sweep over robot_force_scale values.")
    parser.add_argument("--config", default="src/predictive_nav_mppi/config/benchmark_config.yaml")
    parser.add_argument("--forces", default="0.0,1.0",
                        help="Comma-separated robot_force_scale values, e.g. 0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--episode-count", type=int, default=10,
                        help="Number of unique goals/episodes to run with repeats forced to 1.")
    parser.add_argument("--predictor", default="kalman")
    parser.add_argument("--output-root", default="benchmark_force_sweep")
    parser.add_argument("--study-name", default="")
    parser.add_argument("--record-raw-dataset", action="store_true", default=True,
                        help="Record raw people trajectories during each force run.")
    parser.add_argument("--no-record-raw-dataset", dest="record_raw_dataset",
                        action="store_false",
                        help="Skip raw people trajectory recording and curation.")
    parser.add_argument("--raw-visibility-radius", type=float, default=100.0)
    parser.add_argument("--raw-visibility-fov-deg", type=float, default=360.0)
    parser.add_argument("--near-robot-radius", type=float, default=6.0)
    parser.add_argument("--near-robot-fov-deg", type=float, default=360.0)
    parser.add_argument("--auto-report", action="store_true", default=True,
                        help="Generate report_robot_force_interactions automatically after the sweep.")
    parser.add_argument("--no-auto-report", dest="auto_report", action="store_false",
                        help="Skip report_robot_force_interactions after the sweep.")
    parser.add_argument("--auto-benchmark-metrics", action="store_true", default=True,
                        help="Generate benchmark episode metrics boxplot/report automatically after the sweep.")
    parser.add_argument("--no-auto-benchmark-metrics", dest="auto_benchmark_metrics",
                        action="store_false",
                        help="Skip benchmark metrics boxplot/report after the sweep.")
    parser.add_argument("--auto-radius-analysis", action="store_true", default=True,
                        help="Generate category-vs-radius analysis automatically after the sweep.")
    parser.add_argument("--no-auto-radius-analysis", dest="auto_radius_analysis",
                        action="store_false",
                        help="Skip category-vs-radius analysis after the sweep.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open() as f:
        base_cfg = yaml.safe_load(f)
    scenario = str(base_cfg.get("benchmark", {}).get("scenario", ""))

    forces = [float(x.strip()) for x in str(args.forces).split(",") if x.strip()]
    if not forces:
        raise SystemExit("No forces requested")

    study_name = args.study_name.strip() or f"kalman_react_force_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_dir = Path(args.output_root).resolve() / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "study_name": study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictor": args.predictor,
        "episode_count": int(args.episode_count),
        "forces": [],
    }

    for force in forces:
        force_root = study_dir / _force_label(force)
        force_root.mkdir(parents=True, exist_ok=True)
        prepared_cfg = _prepare_cfg(base_cfg, args.episode_count, force_root)
        prepared_cfg["benchmark"]["predictor_type"] = str(args.predictor)
        prepared_cfg["benchmark"]["humans_ignore_robot"] = abs(float(force)) <= 1e-9
        prepared_cfg["benchmark"]["robot_force_scale"] = float(force)
        prepared_cfg["benchmark"]["predict_robot_as_agent"] = False
        temp_cfg = study_dir / f"{_force_label(force)}.yaml"
        temp_cfg.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False))
        raw_dataset_path = force_root / f"people_dataset_{_force_label(force)}.json"

        print(f"\n=== Running force={force:.2f} ===", flush=True)
        run_dir = _run_force(
            temp_cfg,
            force=force,
            predictor=args.predictor,
            raw_dataset_path=raw_dataset_path if args.record_raw_dataset else None,
            map_name=scenario,
            visibility_radius=args.raw_visibility_radius,
            visibility_fov_deg=args.raw_visibility_fov_deg,
        )
        curated_dir = None
        if args.record_raw_dataset and raw_dataset_path.is_file():
            curated_dir = _postprocess_force(
                raw_dataset_path=raw_dataset_path,
                run_dir=run_dir,
                near_robot_radius=args.near_robot_radius,
                near_robot_fov_deg=args.near_robot_fov_deg,
            )
        manifest["forces"].append({
            "force": float(force),
            "label": _force_label(force),
            "config_path": str(temp_cfg),
            "run_dir": str(run_dir),
            "raw_dataset_path": str(raw_dataset_path) if args.record_raw_dataset else "",
            "curated_dir": str(curated_dir) if curated_dir is not None else "",
        })

    manifest_path = study_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nStudy manifest saved to {manifest_path}")

    if args.auto_report and len(forces) >= 2:
        report_cmd = [
            "ros2", "run", "predictive_nav_mppi", "report_robot_force_interactions",
            "--study-dir", str(study_dir),
            "--forces", ",".join(str(float(force)) for force in forces),
        ]
        subprocess.run(report_cmd, check=True)

    if args.auto_benchmark_metrics:
        _run_study_postprocess("predictive_nav_mppi.plot_force_sweep_benchmark_metrics", study_dir)

    if args.auto_radius_analysis and len(forces) >= 2:
        _run_study_postprocess("predictive_nav_mppi.analyze_force_radius", study_dir)


if __name__ == "__main__":
    main()
