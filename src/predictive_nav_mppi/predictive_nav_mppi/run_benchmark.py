#!/usr/bin/env python3
"""Benchmark orchestrator.

For every (mppi_mode × goal × repeat):
  1. Launch the full simulation (headless).
  2. Wait for it to boot.
  3. Run ``benchmark_episode`` (single NavigateToPose goal + metric collection).
  4. Kill the simulation.
  5. Record the results.

Usage
-----
    python3 -m predictive_nav_mppi.run_benchmark --config path/to/benchmark_config.yaml

Or after ``colcon build``:
    ros2 run predictive_nav_mppi run_benchmark --ros-args -p config:=path/to/config.yaml
"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _launch_sim(scenario: str, mppi_mode: str, gui: bool = False,
                extra_args: list | None = None) -> subprocess.Popen:
    cmd = [
        "ros2", "launch", "predictive_nav_mppi", "sim_nav2.launch.py",
        f"scenario:={scenario}",
        f"mppi_mode:={mppi_mode}",
        f"gui:={'True' if gui else 'False'}",
        "use_hunav:=True",
        "humans_ignore_robot:=True",
        "publish_initial_pose:=False",
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def _kill_sim(proc: subprocess.Popen, timeout: float = 15.0):
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=5)


def _cleanup_orphans():
    """Aggressively kill every process from the simulation stack.

    After ros2 launch is terminated, child processes often survive
    (especially component_container_isolated which hosts Nav2 nodes,
    robot_state_publisher, gzserver, etc.).  A leftover gzserver blocks
    the port for the next episode; a leftover map_server means AMCL
    never re-publishes the map→odom TF.
    """
    targets = [
        # Gazebo
        "gzserver", "gzclient",
        # HuNav
        "hunav_gazebo_world_generator", "hunav_loader", "hunav_agent_manager",
        # Nav2 (individual names, in case they run standalone)
        "controller_server", "planner_server", "behavior_server",
        "bt_navigator", "waypoint_follower", "velocity_smoother",
        "smoother_server", "lifecycle_manager", "amcl", "map_server",
        # Nav2 component container (hosts all Nav2 nodes as plugins)
        "component_container_isolated", "component_container",
        # Robot / visualization
        "robot_state_publisher", "rviz2",
        # Our nodes
        "people_kf_predictor", "compute_agents_proxy",
        "publish_initial_pose", "benchmark_episode",
        # spawn_entity helper
        "spawn_entity",
    ]
    for name in targets:
        subprocess.run(
            ["pkill", "-9", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )

    # Also kill any remaining ros2-launched python nodes
    subprocess.run(
        ["pkill", "-9", "-f", "ros2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=2,
    )

    time.sleep(3)

    # Restart DDS daemon to clear stale discovery state
    subprocess.run(
        ["ros2", "daemon", "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    time.sleep(1)
    subprocess.run(
        ["ros2", "daemon", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    time.sleep(1)


def _run_episode(goal: dict, cfg: dict, output_file: str,
                 start: dict) -> dict | None:
    """Spawn the benchmark_episode ROS2 node, wait for it to finish."""
    cmd = [
        "ros2", "run", "predictive_nav_mppi", "benchmark_episode",
        "--ros-args",
        "-p", f"goal_x:={goal['x']}",
        "-p", f"goal_y:={goal['y']}",
        "-p", f"goal_yaw:={goal.get('yaw', 0.0)}",
        "-p", f"start_x:={start['x']}",
        "-p", f"start_y:={start['y']}",
        "-p", f"start_yaw:={start.get('yaw', 0.0)}",
        "-p", f"robot_radius:={cfg['robot_radius']}",
        "-p", f"person_radius:={cfg['person_radius']}",
        "-p", f"personal_space:={cfg['personal_space']}",
        "-p", f"nav_timeout:={cfg['nav_timeout_sec']}",
        "-p", f"output_file:={output_file}",
        "-p", f"global_frame:={cfg['global_frame']}",
        "-p", f"robot_frame:={cfg['robot_frame']}",
        "-p", f"people_topic:={cfg['people_topic']}",
        "-p", f"sample_rate_hz:={cfg['sample_rate_hz']}",
        "-p", f"settle_time:={cfg.get('settle_time_sec', 5.0)}",
        "-p", "use_sim_time:=True",
    ]

    wall_timeout = (
        cfg["nav_timeout_sec"] * 3
        + cfg.get("settle_time_sec", 5.0)
        + 120
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        proc.wait(timeout=wall_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        return None

    if os.path.isfile(output_file):
        with open(output_file) as f:
            return json.load(f)
    return None


def _stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return {
        "mean": round(mean, 4),
        "std": round(math.sqrt(var), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MPPI benchmark episodes")
    parser.add_argument(
        "--config", required=True,
        help="Path to benchmark_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["benchmark"]

    output_dir = Path(cfg.get("output_dir", "benchmark_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    start = cfg.get("start", {"x": 0.0, "y": 0.0, "yaw": 0.0})
    modes = cfg.get("mppi_modes", ["custom"])
    goals = cfg.get("goals", [{"x": 12.0, "y": 0.0}])
    repeats = int(cfg.get("repeats", 3))
    gui = cfg.get("gui", False)
    startup_delay = float(cfg.get("startup_delay_sec", 30.0))
    cooldown = float(cfg.get("cooldown_sec", 5.0))

    total_episodes = len(modes) * len(goals) * repeats
    episode_num = 0

    print(f"\n{'=' * 64}")
    print(f"  BENCHMARK  |  {total_episodes} episodes  |  {timestamp}")
    print(f"{'=' * 64}")

    for mode in modes:
        for gi, goal in enumerate(goals):
            for rep in range(repeats):
                episode_num += 1
                eid = f"{mode}_g{gi}_r{rep}"
                print(f"\n[{episode_num}/{total_episodes}]  {eid}")
                print(f"  mode={mode}  goal=({goal['x']}, {goal['y']})  "
                      f"repeat={rep + 1}/{repeats}")

                sim = _launch_sim(cfg["scenario"], mode, gui=gui)
                print(f"  Launched sim (pid {sim.pid}).  "
                      f"Waiting {startup_delay:.0f}s …")
                time.sleep(startup_delay)

                ep_file = str(run_dir / f"{eid}.json")
                result = _run_episode(goal, cfg, ep_file, start)

                if result:
                    result["episode_id"] = eid
                    result["mppi_mode"] = mode
                    result["goal_idx"] = gi
                    result["repeat"] = rep
                    all_results.append(result)
                    print(f"  ✓ {result['status']}  "
                          f"t={result['time_to_goal']:.1f}s  "
                          f"path={result['path_length']:.2f}m  "
                          f"minD={result['min_dist']:.3f}m  "
                          f"coll={result['collision_count']}  "
                          f"viol={result['viol_time']:.1f}s")
                else:
                    print("  ✗ No results (crash / wall-clock timeout)")

                print("  Killing sim …")
                _kill_sim(sim)
                print("  Cleaning orphans (gzserver, Nav2, HuNav, …) …")
                _cleanup_orphans()
                time.sleep(cooldown)

    # ── save everything ─────────────────────────────────────────────
    summary_json = run_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = run_dir / "results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        for r in all_results:
            if "goal" in r and isinstance(r["goal"], dict):
                r["goal"] = f"({r['goal']['x']}, {r['goal']['y']})"
            if "start" in r and isinstance(r["start"], dict):
                r["start"] = f"({r['start']['x']}, {r['start']['y']})"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)

    # ── print summary ───────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  SUMMARY")
    print(f"{'=' * 64}")

    for mode in modes:
        mr = [r for r in all_results if r.get("mppi_mode") == mode]
        ok = [r for r in mr if r["status"] == "SUCCEEDED"]
        print(f"\n  ── {mode.upper()} MPPI  "
              f"({len(ok)}/{len(mr)} succeeded) ──")
        if not ok:
            print("    (no successful episodes)")
            continue
        for label, key in [
            ("Time-to-goal  (s)", "time_to_goal"),
            ("Path length   (m)", "path_length"),
            ("Min distance  (m)", "min_dist"),
            ("Collisions      ", "collision_count"),
            ("Violation time(s)", "viol_time"),
        ]:
            s = _stats([r[key] for r in ok])
            print(f"    {label}:  "
                  f"{s['mean']:.3f} ± {s['std']:.3f}  "
                  f"[{s['min']:.3f} … {s['max']:.3f}]")

    print(f"\nResults saved to {run_dir}/")
    print()


if __name__ == "__main__":
    main()
