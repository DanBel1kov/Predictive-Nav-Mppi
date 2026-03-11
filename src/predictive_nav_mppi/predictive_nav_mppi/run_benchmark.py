#!/usr/bin/env python3
"""Benchmark orchestrator (single-simulation mode).

Launches the simulation once, then runs all (goal × repeat) episodes
inside it via ``benchmark_session`` – which teleports the robot between
episodes instead of restarting Gazebo.

Usage
-----
    ros2 run predictive_nav_mppi run_benchmark \\
        --config src/predictive_nav_mppi/config/benchmark_config.yaml
"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────

def _launch_sim(scenario: str, mppi_mode: str, gui: bool = False) -> subprocess.Popen:
    cmd = [
        "ros2", "launch", "predictive_nav_mppi", "sim_nav2.launch.py",
        f"scenario:={scenario}",
        f"mppi_mode:={mppi_mode}",
        f"gui:={'True' if gui else 'False'}",
        "use_hunav:=True",
        "humans_ignore_robot:=True",
        "publish_initial_pose:=False",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def _kill_sim(proc: subprocess.Popen, timeout: float = 15.0):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
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
    targets = [
        "gzserver", "gzclient",
        "hunav_gazebo_world_generator", "hunav_loader", "hunav_agent_manager",
        "controller_server", "planner_server", "behavior_server",
        "bt_navigator", "waypoint_follower", "velocity_smoother",
        "smoother_server", "lifecycle_manager", "amcl", "map_server",
        "component_container_isolated", "component_container",
        "robot_state_publisher", "rviz2",
        "people_kf_predictor", "compute_agents_proxy",
        "publish_initial_pose", "benchmark_episode", "benchmark_session",
        "spawn_entity",
    ]
    for name in targets:
        subprocess.run(
            ["pkill", "-9", "-f", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    subprocess.run(
        ["pkill", "-9", "-f", "ros2"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)

    time.sleep(3)

    for action in ("stop", "start"):
        subprocess.run(
            ["ros2", "daemon", action],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        time.sleep(1)


# ─────────────────────────────────────────────────────────────────────
# Session runner
# ─────────────────────────────────────────────────────────────────────

def _run_session(episodes: list, cfg: dict, output_dir: str,
                 mppi_mode: str) -> list:
    """Write the episode list to a temp file and run benchmark_session."""
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as f:
        json.dump(episodes, f, indent=2)
        ep_file = f.name

    wall_timeout = (
        len(episodes)
        * (cfg["nav_timeout_sec"] + cfg.get("settle_time_sec", 5.0) + 30)
        + 120
    )

    cmd = [
        "ros2", "run", "predictive_nav_mppi", "benchmark_session",
        "--ros-args",
        "-p", f"episodes_file:={ep_file}",
        "-p", f"output_dir:={output_dir}",
        "-p", f"robot_radius:={cfg['robot_radius']}",
        "-p", f"person_radius:={cfg['person_radius']}",
        "-p", f"personal_space:={cfg['personal_space']}",
        "-p", f"nav_timeout:={cfg['nav_timeout_sec']}",
        "-p", f"global_frame:={cfg['global_frame']}",
        "-p", f"robot_frame:={cfg['robot_frame']}",
        "-p", f"people_topic:={cfg['people_topic']}",
        "-p", f"sample_rate_hz:={cfg['sample_rate_hz']}",
        "-p", f"settle_time:={cfg.get('settle_time_sec', 5.0)}",
        "-p", f"robot_model_name:={cfg.get('robot_model_name', 'waffle')}",
        "-p", "use_sim_time:=True",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        # Stream output so operator can watch progress
        for line in proc.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                print(f"  [session] {text}")
        proc.wait(timeout=wall_timeout)
    except subprocess.TimeoutExpired:
        print("  [!] Wall-clock timeout – killing session node")
        proc.kill()
        proc.wait(timeout=5)

    os.unlink(ep_file)

    # Collect results written by the session node
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────

def _stats(vals: list) -> dict:
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    mean = sum(vals) / n
    var  = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return {
        "mean": round(mean,         4),
        "std":  round(math.sqrt(var), 4),
        "min":  round(min(vals),    4),
        "max":  round(max(vals),    4),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MPPI benchmark (single-sim mode)")
    parser.add_argument("--config", required=True, help="Path to benchmark_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["benchmark"]

    # ── resolve single mppi_mode ─────────────────────────────────────
    if "mppi_mode" in cfg:
        mode = str(cfg["mppi_mode"])
    elif "mppi_modes" in cfg:
        modes_list = cfg["mppi_modes"]
        if isinstance(modes_list, list):
            mode = str(modes_list[0])
        else:
            mode = str(modes_list)
        print(f"[warn] 'mppi_modes' found in config – using first entry: {mode}")
    else:
        mode = "custom"

    goals   = cfg.get("goals",   [{"x": 12.0, "y": 0.0, "yaw": 0.0}])
    repeats = int(cfg.get("repeats", 3))
    default_start = cfg.get("start", {"x": 0.0, "y": 0.0, "yaw": 0.0})
    gui           = cfg.get("gui", False)
    startup_delay = float(cfg.get("startup_delay_sec", 30.0))
    cooldown      = float(cfg.get("cooldown_sec", 5.0))

    # ── map bounds по сценарию (для проверки start/goal) ──
    scenario = (cfg.get("scenario") or "long_corridor").strip().lower()
    if scenario == "long_corridor":
        # Карта коридора 20m x 6.6m, x∈[-10,10], y∈[-3.3,3.25]
        MAP_X_MIN, MAP_X_MAX = -9.9, 9.9
        MAP_Y_MIN, MAP_Y_MAX = -3.2, 3.2
    else:
        MAP_X_MIN, MAP_X_MAX = -9.9, 9.9
        MAP_Y_MIN, MAP_Y_MAX = -3.2, 3.2

    # ── build episode list ───────────────────────────────────────────
    episodes = []
    for gi, goal in enumerate(goals):
        # Per-goal start overrides global start
        start = goal.get("start", default_start)
        for pt, coords in [("start", start), ("goal", goal)]:
            gx, gy = float(coords.get("x", 0)), float(coords.get("y", 0))
            if not (MAP_X_MIN <= gx <= MAP_X_MAX and MAP_Y_MIN <= gy <= MAP_Y_MAX):
                print(f"  [warn] goal {gi} {pt}=({gx:.2f},{gy:.2f}) outside map "
                      f"[x:{MAP_X_MIN}..{MAP_X_MAX}, y:{MAP_Y_MIN}..{MAP_Y_MAX}]")
        for rep in range(repeats):
            eid = f"{mode}_g{gi}_r{rep}"
            episodes.append({
                "episode_id": eid,
                "mppi_mode":  mode,
                "goal_idx":   gi,
                "repeat":     rep,
                "start": {
                    "x":   float(start.get("x",   0.0)),
                    "y":   float(start.get("y",   0.0)),
                    "yaw": float(start.get("yaw", 0.0)),
                },
                "goal": {
                    "x":   float(goal.get("x",   0.0)),
                    "y":   float(goal.get("y",   0.0)),
                    "yaw": float(goal.get("yaw", 0.0)),
                },
            })

    total = len(episodes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output_dir", "benchmark_results")) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  BENCHMARK  |  mode={mode}  |  {total} episodes  |  {timestamp}")
    print(f"{'=' * 64}")
    print(f"  goals={len(goals)}  repeats={repeats}  startup={startup_delay:.0f}s")
    print(f"  output: {output_dir}")

    # ── launch simulation ────────────────────────────────────────────
    sim = _launch_sim(cfg["scenario"], mode, gui=gui)
    print(f"\n  Launched sim (pid {sim.pid}).  Waiting {startup_delay:.0f}s for Nav2 …")
    time.sleep(startup_delay)

    if sim.poll() is not None:
        print("  [!] Simulation exited early – aborting")
        _cleanup_orphans()
        sys.exit(1)

    # ── run all episodes in one session ─────────────────────────────
    print(f"\n  Running {total} episodes (no sim restart between them) …\n")
    all_results = _run_session(episodes, cfg, str(output_dir), mode)

    # ── kill simulation ──────────────────────────────────────────────
    print("\n  Killing simulation …")
    _kill_sim(sim)
    print("  Cleaning orphan processes …")
    _cleanup_orphans()
    time.sleep(cooldown)

    # ── write CSV ────────────────────────────────────────────────────
    if all_results:
        csv_path = output_dir / "results.csv"
        rows = []
        for r in all_results:
            row = dict(r)
            for key in ("goal", "start"):
                if isinstance(row.get(key), dict):
                    row[key] = f"({row[key]['x']}, {row[key]['y']})"
            rows.append(row)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── print summary ────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  SUMMARY  |  mode={mode}")
    print(f"{'=' * 64}")

    ok  = [r for r in all_results if r.get("status") == "SUCCEEDED"]
    all_r = all_results
    print(f"\n  {mode.upper()} MPPI  ({len(ok)}/{len(all_r)} succeeded)")

    if ok:
        for label, key in [
            ("Time-to-goal  (s)", "time_to_goal"),
            ("Path length   (m)", "path_length"),
            ("Min distance  (m)", "min_dist"),
            ("Collisions      ",  "collision_count"),
            ("Violation time(s)", "viol_time"),
        ]:
            s = _stats([r[key] for r in ok])
            print(f"    {label}:  "
                  f"{s['mean']:.3f} ± {s['std']:.3f}  "
                  f"[{s['min']:.3f} … {s['max']:.3f}]")

    print(f"\n  Results saved to {output_dir}/")
    print()


if __name__ == "__main__":
    main()
