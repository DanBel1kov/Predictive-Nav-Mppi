#!/usr/bin/env python3
"""Parallel benchmark runner.

Splits goals across N simultaneous simulation instances to increase throughput.
Each worker is fully isolated via ROS_DOMAIN_ID and GAZEBO_MASTER_URI.
Goals are interleaved across workers (worker i gets goals[i::N]) so that every
worker sees structurally different scenario types.

Usage
-----
    ros2 run predictive_nav_mppi parallel_benchmark \\
        --config src/predictive_nav_mppi/config/benchmark_config.yaml

    # Explicitly specify worker count:
    ros2 run predictive_nav_mppi parallel_benchmark \\
        --config benchmark_config.yaml --workers 2
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

# Worker isolation constants.
# Using domain IDs 10+ avoids collision with interactive sessions on domain 0.
_BASE_ROS_DOMAIN_ID = 10
_BASE_GAZEBO_PORT = 11345


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _cleanup_orphans():
    targets = [
        "gzserver", "gzclient",
        "hunav_gazebo_world_generator", "hunav_loader", "hunav_agent_manager",
        "controller_server", "planner_server", "behavior_server",
        "bt_navigator", "waypoint_follower", "velocity_smoother",
        "smoother_server", "lifecycle_manager", "amcl", "map_server",
        "component_container_isolated", "component_container",
        "robot_state_publisher", "rviz2",
        "people_predictor", "people_kf_predictor", "compute_agents_proxy",
        "publish_initial_pose", "benchmark_episode", "benchmark_session",
        "spawn_entity",
    ]
    for name in targets:
        subprocess.run(
            ["pkill", "-9", "-f", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    # NOTE: intentionally NOT running `pkill -9 -f ros2` here — that would
    # kill the parent `ros2 run parallel_benchmark` process itself.
    # The specific kills above are sufficient to remove leftover sim processes.
    time.sleep(3)
    for action in ("stop", "start"):
        subprocess.run(
            ["ros2", "daemon", action],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        time.sleep(1)


def _stats(vals: list) -> dict:
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return {
        "mean": round(mean, 4),
        "std": round(math.sqrt(var), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": n,
    }


def _stream_worker_output(worker_id: int, proc: subprocess.Popen):
    """Forward worker stdout to console with a [wN] prefix."""
    try:
        for line in proc.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                print(f"  [w{worker_id}] {text}", flush=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Worker execution
# ─────────────────────────────────────────────────────────────────────

def _run_worker(
    worker_id: int,
    worker_cfg: dict,
    worker_output_base: Path,
) -> list:
    """
    Run one benchmark worker subprocess in an isolated ROS/Gazebo environment.

    Returns the list of episode result dicts from the worker's summary.json,
    or [] if the worker failed to produce results.
    """
    # Write a per-worker temporary config file.
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_worker{worker_id}.yaml", delete=False) as f:
        yaml.dump({"benchmark": worker_cfg}, f, allow_unicode=True)
        cfg_path = f.name

    worker_output_base.mkdir(parents=True, exist_ok=True)

    domain_id = _BASE_ROS_DOMAIN_ID + worker_id
    gazebo_port = _BASE_GAZEBO_PORT + worker_id

    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["GAZEBO_MASTER_URI"] = f"http://localhost:{gazebo_port}"
    # Prevent workers from doing the global pkill cleanup — they would kill
    # each other. The parent handles cleanup before start and after finish.
    env["PREDICTIVE_NAV_MPPI_DISABLE_GLOBAL_CLEANUP"] = "1"

    n_goals = len(worker_cfg.get("goals", []))
    repeats = int(worker_cfg.get("repeats", 1))
    print(
        f"  [w{worker_id}] Starting: "
        f"ROS_DOMAIN_ID={domain_id}  Gazebo port={gazebo_port}  "
        f"{n_goals} goals × {repeats} repeats = {n_goals * repeats} episodes",
        flush=True,
    )
    print(f"  [w{worker_id}] Output → {worker_output_base}", flush=True)

    cmd = [
        "ros2", "run", "predictive_nav_mppi", "run_benchmark",
        "--config", cfg_path,
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Stream output in a daemon thread so workers don't block each other.
    stream_thread = threading.Thread(
        target=_stream_worker_output, args=(worker_id, proc), daemon=True
    )
    stream_thread.start()
    proc.wait()
    stream_thread.join(timeout=5)

    try:
        os.unlink(cfg_path)
    except OSError:
        pass

    # Locate the summary written by run_benchmark (it creates its own
    # timestamp subdirectory inside output_dir).
    summaries = sorted(worker_output_base.glob("*/summary.json"))
    if not summaries:
        print(f"  [w{worker_id}] [warn] No summary.json found — worker may have failed.")
        return []

    with open(summaries[-1]) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark with N parallel simulation instances."
    )
    parser.add_argument("--config", required=True, help="Path to benchmark_config.yaml")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers. 0 = read from config (parallel_workers key), "
             "default 2 if not set."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["benchmark"]

    # Resolve worker count: CLI > config > default 2.
    if args.workers > 0:
        n_workers = args.workers
    else:
        n_workers = int(cfg.get("parallel_workers", 2))
    n_workers = max(1, n_workers)

    goals = cfg.get("goals", [])
    if not goals:
        print("[error] No goals defined in config.")
        sys.exit(1)

    # Cap workers to number of goals (can't split more finely than that).
    if n_workers > len(goals):
        print(
            f"  [warn] {n_workers} workers > {len(goals)} goals — "
            f"reducing to {len(goals)} workers."
        )
        n_workers = len(goals)

    if n_workers == 1:
        # Fall through to plain run_benchmark — no overhead needed.
        print("  Only 1 worker — delegating to run_benchmark directly.")
        subprocess.run(
            ["ros2", "run", "predictive_nav_mppi", "run_benchmark",
             "--config", args.config],
            check=False,
        )
        return

    # Split goals interleaved: worker i gets goals[i], goals[i+N], goals[i+2N], ...
    # This ensures every worker covers structurally different scenario types
    # rather than one worker getting only easy scenarios.
    worker_goals = [goals[i::n_workers] for i in range(n_workers)]

    predictor = str(cfg.get("predictor_type", "kalman"))
    mode = str(cfg.get("mppi_mode", "custom"))
    repeats = int(cfg.get("repeats", 1))
    total_episodes = sum(len(wg) * repeats for wg in worker_goals)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(cfg.get("output_dir", "benchmark_results"))

    print(f"\n{'=' * 64}")
    print(
        f"  PARALLEL BENCHMARK  |  {n_workers} workers  |  "
        f"{total_episodes} total episodes"
    )
    print(f"  mode={mode}  predictor={predictor}  |  {timestamp}")
    print(
        f"  Goals per worker: "
        + "  ".join(f"w{i}={len(worker_goals[i])}" for i in range(n_workers))
    )
    print(f"{'=' * 64}")

    print("\n  Cleaning up orphan processes before start …")
    _cleanup_orphans()

    # Build per-worker configs (deep-copy to avoid cross-contamination).
    worker_cfgs = []
    for i in range(n_workers):
        wcfg = deepcopy(cfg)
        wcfg["goals"] = worker_goals[i]
        wcfg["output_dir"] = str(base_output_dir / timestamp / f"worker_{i}")
        worker_cfgs.append(wcfg)

    # Launch all workers concurrently.
    results_per_worker: list = [[] for _ in range(n_workers)]

    def _thread_fn(i):
        worker_output_base = Path(worker_cfgs[i]["output_dir"])
        results_per_worker[i] = _run_worker(i, worker_cfgs[i], worker_output_base)

    threads = [threading.Thread(target=_thread_fn, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Merge all episode results.
    all_results = []
    for i, results in enumerate(results_per_worker):
        if results:
            all_results.extend(results)
        else:
            print(f"  [warn] Worker {i} produced no results.")

    # Final cleanup.
    print("\n  Final cleanup …")
    _cleanup_orphans()

    if not all_results:
        print("\n  [error] No results collected from any worker.")
        sys.exit(1)

    # Save merged summary.
    merged_dir = base_output_dir / timestamp
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_summary = merged_dir / "summary.json"
    with open(merged_summary, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print aggregated stats.
    print(f"\n{'=' * 64}")
    print(f"  RESULTS  |  mode={mode}  predictor={predictor}")
    print(f"{'=' * 64}")

    ok = [r for r in all_results if r.get("status") == "SUCCEEDED"]
    total = len(all_results)
    pct = int(100 * len(ok) / max(1, total))
    print(f"\n  {len(ok)}/{total} succeeded  ({pct}%)")

    if ok:
        for label, key in [
            ("Time-to-goal  (s)", "time_to_goal"),
            ("Path length   (m)", "path_length"),
            ("Min distance  (m)", "min_dist"),
            ("Collisions      ", "collision_count"),
            ("Violation time(s)", "viol_time"),
        ]:
            s = _stats([r.get(key) for r in ok])
            print(
                f"    {label}:  "
                f"{s['mean']:.3f} ± {s['std']:.3f}  "
                f"[{s['min']:.3f} … {s['max']:.3f}]  "
                f"n={s['n']}"
            )

    print(f"\n  Merged summary → {merged_summary}")
    print()


if __name__ == "__main__":
    main()
