#!/usr/bin/env python3
"""Run and compare two benchmark variants on the same episode set.

The runner is intentionally conservative:
  * runs variants sequentially, never in parallel;
  * caches completed (goal_idx, repeat) episodes per variant;
  * when target episode count grows, runs only missing repeats;
  * compares paired episode results with permutation p-values and bootstrap CIs.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml


METRICS = [
    ("time_to_goal", "lower"),
    ("path_length", "lower"),
    ("min_dist", "higher"),
    ("collision_count", "lower"),
    ("viol_time", "lower"),
    ("avg_robot_influence", "lower"),
]


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    if len(vals) < 2:
        return 0.0 if vals else float("nan")
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _ci_percentile(vals, lo=2.5, hi=97.5):
    if not vals:
        return float("nan"), float("nan")
    ordered = sorted(vals)

    def pick(p):
        idx = (len(ordered) - 1) * (p / 100.0)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return ordered[lower]
        frac = idx - lower
        return ordered[lower] * (1.0 - frac) + ordered[upper] * frac

    return pick(lo), pick(hi)


def _bootstrap_ci(diffs, samples=10000, seed=17):
    if not diffs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(samples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    return _ci_percentile(means)


def _permutation_pvalue(diffs, samples=20000, seed=19):
    """Two-sided paired sign-flip permutation p-value for mean difference."""
    diffs = [d for d in diffs if _finite(d)]
    if not diffs:
        return float("nan")
    observed = abs(_mean(diffs))
    n = len(diffs)
    rng = random.Random(seed)

    # Exact enumeration is cheap up to 16 pairs.
    if n <= 16:
        total = 1 << n
        extreme = 0
        for mask in range(total):
            signed_sum = 0.0
            for i, d in enumerate(diffs):
                signed_sum += d if (mask & (1 << i)) else -d
            if abs(signed_sum / n) >= observed - 1e-12:
                extreme += 1
        return extreme / total

    extreme = 0
    for _ in range(samples):
        signed_sum = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(signed_sum / n) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def _normalize_mode(mode):
    mode = str(mode).strip().lower()
    if mode in ("standart", "std"):
        mode = "standard"
    if mode not in ("standard", "custom"):
        raise SystemExit(f"Unsupported MPPI mode: {mode}")
    return mode


def _normalize_predictor(predictor):
    predictor = str(predictor).strip().lower()
    if predictor == "socialvae":
        predictor = "social_vae"
    if predictor not in ("kalman", "residual", "model", "social_vae"):
        raise SystemExit(f"Unsupported predictor: {predictor}")
    return predictor


def _variant_id(mode, predictor, args):
    parts = [mode, predictor]
    if predictor == "residual":
        parts.extend([
            f"a{args.residual_alpha:g}",
            f"b{args.residual_beta:g}",
            f"c{args.residual_clip:g}",
        ])
    return "_".join(parts).replace(".", "p").replace("-", "m")


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _write_config(path, cfg):
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _goal_signature(goals):
    normalized = []
    for idx, goal in enumerate(goals):
        normalized.append({
            "goal_idx": int(goal.get("goal_idx", idx)),
            "start": goal.get("start", {}),
            "x": goal.get("x"),
            "y": goal.get("y"),
            "yaw": goal.get("yaw", 0.0),
        })
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _study_id(base_cfg, args):
    bench = base_cfg["benchmark"]
    if args.study_name:
        return args.study_name
    goals_sig = _goal_signature(bench.get("goals", []))
    parts = [
        str(bench.get("scenario", "scenario")),
        f"force{args.robot_force_scale:g}",
        f"predrobot{int(args.predict_robot_as_agent)}",
        f"goals{goals_sig}",
    ]
    return "_".join(parts).replace(".", "p").replace("-", "m")


def _result_key(result):
    return int(result["goal_idx"]), int(result["repeat"])


def _load_cached(path):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    return []


def _save_cached(path, results):
    by_key = {}
    for result in results:
        if "goal_idx" in result and "repeat" in result:
            by_key[_result_key(result)] = result
    ordered = [
        by_key[key]
        for key in sorted(by_key)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, indent=2, allow_nan=True))
    return ordered


def _desired_keys(goal_count, repeats):
    return {(goal_idx, repeat) for repeat in range(repeats) for goal_idx in range(goal_count)}


def _variant_config(base_cfg, args, mode, predictor, repeat, missing_goal_indices, output_dir):
    cfg = json.loads(json.dumps(base_cfg))
    bench = cfg["benchmark"]
    source_goals = list(bench.get("goals", []))
    goals = []
    for goal_idx in missing_goal_indices:
        goal = dict(source_goals[goal_idx])
        goal["goal_idx"] = goal_idx
        goals.append(goal)

    bench["mppi_mode"] = mode
    bench["predictor_type"] = predictor
    bench["robot_force_scale"] = float(args.robot_force_scale)
    bench["predict_robot_as_agent"] = bool(args.predict_robot_as_agent)
    bench["repeats"] = 1
    bench["repeat_offset"] = int(repeat)
    bench["goals"] = goals
    bench["output_dir"] = str(output_dir)
    bench["gui"] = False
    if args.sim_real_time_update_rate is not None:
        bench["sim_real_time_update_rate"] = float(args.sim_real_time_update_rate)
    if args.sim_max_step_size is not None:
        bench["sim_max_step_size"] = float(args.sim_max_step_size)
    if args.sim_speedup is not None:
        bench["sim_speedup"] = float(args.sim_speedup)

    if predictor == "residual":
        bench["residual_alpha"] = float(args.residual_alpha)
        bench["residual_smoothing_beta"] = float(args.residual_beta)
        bench["residual_clip_norm"] = float(args.residual_clip)

    return cfg


def _run_variant_batch(base_cfg, args, run_root, variant, repeat, missing_goal_indices):
    mode, predictor, variant_id = variant
    output_dir = run_root / variant_id / "runs"
    cfg = _variant_config(base_cfg, args, mode, predictor, repeat, missing_goal_indices, output_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = Path(f.name)

    cmd = [
        "ros2", "run", "predictive_nav_mppi", "run_benchmark",
        "--config", str(cfg_path),
    ]
    print(
        f"\n[paired] run {variant_id}: repeat={repeat}, "
        f"goals={missing_goal_indices}, config={cfg_path}",
        flush=True,
    )
    try:
        result = subprocess.run(cmd, check=False)
    finally:
        try:
            cfg_path.unlink()
        except OSError:
            pass

    if result.returncode != 0:
        raise SystemExit(f"Benchmark failed for {variant_id}, repeat={repeat}")

    summaries = sorted(output_dir.glob("*/summary.json"))
    if not summaries:
        raise SystemExit(f"No summary.json found for {variant_id} in {output_dir}")
    with summaries[-1].open() as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _collect_missing_runs(base_cfg, args, run_root, variants, target_repeats):
    goals = list(base_cfg["benchmark"].get("goals", []))
    goal_count = len(goals)
    wanted = _desired_keys(goal_count, target_repeats)
    cached_by_variant = {}

    for variant in variants:
        variant_id = variant[2]
        cache_path = run_root / variant_id / "cached_summary.json"
        cached = _save_cached(cache_path, _load_cached(cache_path))
        cached_by_variant[variant_id] = cached

    for repeat in range(target_repeats):
        for variant in variants:
            variant_id = variant[2]
            cache_path = run_root / variant_id / "cached_summary.json"
            cached = cached_by_variant[variant_id]
            present = {_result_key(r) for r in cached}
            missing_goal_indices = [
                goal_idx
                for goal_idx in range(goal_count)
                if (goal_idx, repeat) in wanted and (goal_idx, repeat) not in present
            ]
            if not missing_goal_indices:
                print(f"[paired] cache hit {variant_id}: repeat={repeat}")
                continue
            new_results = _run_variant_batch(
                base_cfg, args, run_root, variant, repeat, missing_goal_indices)
            cached = _save_cached(cache_path, cached + new_results)
            cached_by_variant[variant_id] = cached

    return cached_by_variant


def _metric_value(result, key, nav_timeout):
    if key == "time_to_goal" and result.get("status") != "SUCCEEDED":
        return nav_timeout
    value = _safe_float(result.get(key))
    return value


def _compare(run_root, variants, cached_by_variant, target_repeats, nav_timeout):
    left_id = variants[0][2]
    right_id = variants[1][2]
    left = {_result_key(r): r for r in cached_by_variant[left_id]}
    right = {_result_key(r): r for r in cached_by_variant[right_id]}
    paired_keys = sorted(set(left) & set(right))
    paired_keys = [key for key in paired_keys if key[1] < target_repeats]

    report = {
        "left": left_id,
        "right": right_id,
        "paired_episodes": len(paired_keys),
        "metrics": {},
        "success": {
            left_id: sum(1 for key in paired_keys if left[key].get("status") == "SUCCEEDED"),
            right_id: sum(1 for key in paired_keys if right[key].get("status") == "SUCCEEDED"),
        },
    }

    rows = []
    for key in paired_keys:
        row = {
            "goal_idx": key[0],
            "repeat": key[1],
            f"{left_id}_status": left[key].get("status"),
            f"{right_id}_status": right[key].get("status"),
        }
        for metric, _direction in METRICS:
            row[f"{left_id}_{metric}"] = left[key].get(metric)
            row[f"{right_id}_{metric}"] = right[key].get(metric)
        rows.append(row)

    for metric, direction in METRICS:
        left_vals = []
        right_vals = []
        diffs = []
        for key in paired_keys:
            lv = _metric_value(left[key], metric, nav_timeout)
            rv = _metric_value(right[key], metric, nav_timeout)
            if not (_finite(lv) and _finite(rv)):
                continue
            left_vals.append(lv)
            right_vals.append(rv)
            diffs.append(rv - lv)
        ci_low, ci_high = _bootstrap_ci(diffs)
        report["metrics"][metric] = {
            "direction": direction,
            "n": len(diffs),
            "left_mean": _mean(left_vals),
            "left_std": _std(left_vals),
            "right_mean": _mean(right_vals),
            "right_std": _std(right_vals),
            "mean_diff_right_minus_left": _mean(diffs),
            "diff_95ci": [ci_low, ci_high],
            "permutation_p_two_sided": _permutation_pvalue(diffs),
        }

    (run_root / "paired_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True))

    csv_path = run_root / "paired_rows.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        f"left={left_id}",
        f"right={right_id}",
        f"paired_episodes={len(paired_keys)}",
        f"success: {left_id}={report['success'][left_id]}/{len(paired_keys)}, "
        f"{right_id}={report['success'][right_id]}/{len(paired_keys)}",
        "",
        "metric,n,left_mean±std,right_mean±std,diff_right_minus_left,95%CI,p",
    ]
    for metric, _direction in METRICS:
        item = report["metrics"][metric]
        lines.append(
            f"{metric},{item['n']},"
            f"{item['left_mean']:.4g}±{item['left_std']:.4g},"
            f"{item['right_mean']:.4g}±{item['right_std']:.4g},"
            f"{item['mean_diff_right_minus_left']:.4g},"
            f"[{item['diff_95ci'][0]:.4g},{item['diff_95ci'][1]:.4g}],"
            f"{item['permutation_p_two_sided']:.4g}"
        )
    (run_root / "paired_report.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run cached paired MPPI benchmark comparison")
    parser.add_argument("--config", default="src/predictive_nav_mppi/config/benchmark_config.yaml")
    parser.add_argument("--episodes", type=int, required=True,
                        help="Total episodes per variant; must be divisible by number of goals")
    parser.add_argument("--study-name", default="", help="Stable cache directory name")
    parser.add_argument("--output-root", default="benchmark_paired")

    parser.add_argument("--left-mode", required=True, help="custom or standard")
    parser.add_argument("--left-predictor", required=True, help="kalman or residual")
    parser.add_argument("--right-mode", required=True, help="custom or standard")
    parser.add_argument("--right-predictor", required=True, help="kalman or residual")

    parser.add_argument("--robot-force-scale", type=float, required=True)
    parser.add_argument("--predict-robot-as-agent", action="store_true",
                        help="Also inject robot into the people predictor input")

    parser.add_argument("--residual-alpha", type=float, default=0.3)
    parser.add_argument("--residual-beta", type=float, default=0.8)
    parser.add_argument("--residual-clip", type=float, default=0.35)

    parser.add_argument("--sim-speedup", type=float)
    parser.add_argument("--sim-max-step-size", type=float)
    parser.add_argument(
        "--sim-real-time-update-rate",
        type=float,
        help=(
            "Override Gazebo real_time_update_rate. "
            "Use 0 only for stress/speed tests; it can change closed-loop behavior "
            "if Nav2/predictor cannot keep up with sim time."
        ),
    )
    args = parser.parse_args()

    base_cfg = _load_config(args.config)
    bench = base_cfg["benchmark"]
    goals = list(bench.get("goals", []))
    if not goals:
        raise SystemExit("benchmark.goals is empty")
    if args.episodes % len(goals) != 0:
        raise SystemExit(
            f"--episodes must be divisible by number of goals ({len(goals)})")
    target_repeats = args.episodes // len(goals)

    left = (
        _normalize_mode(args.left_mode),
        _normalize_predictor(args.left_predictor),
        _variant_id(_normalize_mode(args.left_mode), _normalize_predictor(args.left_predictor), args),
    )
    right = (
        _normalize_mode(args.right_mode),
        _normalize_predictor(args.right_predictor),
        _variant_id(_normalize_mode(args.right_mode), _normalize_predictor(args.right_predictor), args),
    )
    variants = [left, right]

    study_id = _study_id(base_cfg, args)
    run_root = Path(args.output_root) / study_id
    run_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_or_updated": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "episodes_per_variant": args.episodes,
        "repeats": target_repeats,
        "goals": len(goals),
        "robot_force_scale": args.robot_force_scale,
        "predict_robot_as_agent": args.predict_robot_as_agent,
        "variants": [left[2], right[2]],
    }
    (run_root / "study_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[paired] study: {run_root}")
    print(f"[paired] target: {args.episodes} episodes = {len(goals)} goals x {target_repeats} repeats")
    print(f"[paired] left: {left[2]}")
    print(f"[paired] right: {right[2]}")

    cached_by_variant = _collect_missing_runs(base_cfg, args, run_root, variants, target_repeats)
    _compare(
        run_root,
        variants,
        cached_by_variant,
        target_repeats,
        float(bench.get("nav_timeout_sec", 180.0)),
    )


if __name__ == "__main__":
    main()
