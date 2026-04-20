#!/usr/bin/env python3
"""Benchmark orchestrator (single-simulation mode).

Launches the simulation once, then runs all episodes inside it. By default
the robot stays in the same Gazebo instance, while HuNav agents are reset
between episodes by teleporting them to their initial poses and restarting
the HuNav behavior-tree manager so goal queues start from the same state.

Usage
-----
    # From repo root, after: source install/setup.bash
    ros2 run predictive_nav_mppi run_benchmark \\
        --config src/predictive_nav_mppi/config/benchmark_config.yaml

With people (HuNav) and predictive model
---------------------------------------
    In benchmark_config.yaml set:
      predictor_type: model   # or "kalman" / "social_vae"
    Then run the same command. The launch starts HuNav (people), the
    people_predictor node (kalman / model / social_vae), and Nav2 with MPPI that
    uses /predicted_people_markers for dynamic obstacles.
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

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


# ─────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────

def _launch_sim(
    scenario: str,
    mppi_mode: str,
    gui: bool = False,
    sim_speedup: float = 1.0,
    sim_max_step_size: float = 0.0,
    sim_real_time_update_rate: float = -1.0,
    predictor_type: str = "kalman",
    residual_model_weights: str = "",
    residual_alpha: float = 0.3,
    residual_smoothing_beta: float = 0.8,
    residual_clip_norm: float = 0.35,
    residual_turn_gate_enable: bool = True,
    residual_turn_gate_tau: float = 0.1,
    residual_turn_gate_alpha: float = 30.0,
    social_vae_repo_path: str = "",
    social_vae_ckpt_path: str = "",
    social_vae_config_path: str = "",
    social_vae_pred_samples: int = 20,
    robot_force_scale: float = 0.0,
    predict_robot_as_agent: bool = False,
    sim_log_path: str = "",
) -> subprocess.Popen:
    cmd = [
        "ros2", "launch", "predictive_nav_mppi", "sim_nav2.launch.py",
        f"scenario:={scenario}",
        f"mppi_mode:={mppi_mode}",
        f"gui:={'True' if gui else 'False'}",
        f"sim_speedup:={float(sim_speedup)}",
        f"sim_max_step_size:={float(sim_max_step_size)}",
        f"sim_real_time_update_rate:={float(sim_real_time_update_rate)}",
        "use_hunav:=True",
        "humans_ignore_robot:=True",
        "publish_initial_pose:=False",
        f"predictor_type:={predictor_type}",
    ]
    if residual_model_weights:
        cmd.append(f"residual_model_weights:={residual_model_weights}")
        cmd.append(f"residual_alpha:={float(residual_alpha)}")
        cmd.append(f"residual_smoothing_beta:={float(residual_smoothing_beta)}")
        cmd.append(f"residual_clip_norm:={float(residual_clip_norm)}")
        cmd.append(f"residual_turn_gate_enable:={'True' if residual_turn_gate_enable else 'False'}")
        cmd.append(f"residual_turn_gate_tau:={float(residual_turn_gate_tau)}")
        cmd.append(f"residual_turn_gate_alpha:={float(residual_turn_gate_alpha)}")
    if social_vae_repo_path:
        cmd.append(f"social_vae_repo_path:={social_vae_repo_path}")
    if social_vae_ckpt_path:
        cmd.append(f"social_vae_ckpt_path:={social_vae_ckpt_path}")
    if social_vae_config_path:
        cmd.append(f"social_vae_config_path:={social_vae_config_path}")
    cmd.append(f"social_vae_pred_samples:={int(social_vae_pred_samples)}")
    cmd.append(f"robot_force_scale:={float(robot_force_scale)}")
    cmd.append(f"predict_robot_as_agent:={'True' if predict_robot_as_agent else 'False'}")
    if sim_log_path:
        log_fh = open(sim_log_path, "w")
        stdout_dest = log_fh
        stderr_dest = log_fh
    else:
        log_fh = None
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdout=stdout_dest,
        stderr=stderr_dest,
        preexec_fn=os.setsid,
    )
    proc._log_fh = log_fh  # keep reference so GC doesn't close it early
    return proc


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


def _kill_process_by_pattern(pattern: str, timeout: float = 5.0):
    subprocess.run(
        ["pkill", "-TERM", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )


def _cleanup_orphans():
    if os.environ.get("PREDICTIVE_NAV_MPPI_DISABLE_GLOBAL_CLEANUP", "").strip() == "1":
        print("  Global orphan cleanup disabled by environment")
        return
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
    # NOTE: intentionally NOT running `pkill -9 -f ros2` — that would
    # kill the parent `ros2 run run_benchmark` process itself.
    time.sleep(3)

    for action in ("stop", "start"):
        subprocess.run(
            ["ros2", "daemon", action],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        time.sleep(1)


def _reset_hunav_agents(cfg: dict) -> bool:
    scenario = str(cfg.get("scenario", "long_corridor")).strip().lower()
    if get_package_share_directory is not None:
        config_root = Path(get_package_share_directory("predictive_nav_mppi")) / "config"
    else:
        config_root = Path(__file__).resolve().parents[1] / "config"
    params_map = {
        "room_box": config_root / "hunav_agents_params.yaml",
        "long_corridor": config_root / "hunav_agents_corridor_params.yaml",
        "labyrinth_turns": config_root / "hunav_agents_labyrinth_params.yaml",
        "nonlinear_corridor": config_root / "hunav_agents_nonlinear_params.yaml",
    }
    hunav_params_file = params_map.get(scenario, params_map["long_corridor"])
    reset_cmd = [
        "ros2", "run", "predictive_nav_mppi", "reset_hunav_agents",
        "--ros-args",
        "-p", "use_sim_time:=True",
        "-p", f"robot_model_name:={cfg.get('robot_model_name', 'waffle')}",
        "-p", f"hunav_params_file:={hunav_params_file}",
    ]
    attempts = int(cfg.get("reset_hunav_attempts", 3))
    retry_sleep = float(cfg.get("reset_hunav_retry_sleep_sec", 2.0))
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            reset_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            if result.stdout.strip():
                print(result.stdout.rstrip())
            return True
        print(f"  [reset] attempt {attempt}/{attempts} failed (rc={result.returncode})")
        if result.stdout.strip():
            for line in result.stdout.rstrip().splitlines():
                print(f"  [reset] {line}")
        if attempt < attempts:
            time.sleep(retry_sleep)
    return False


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

    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────

def _stats(vals: list) -> dict:
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
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


def _normalize_mppi_mode(mode: str) -> str:
    """Normalize common spellings and fail fast on unsupported MPPI modes."""
    normalized = str(mode).strip().lower()
    aliases = {
        "standart": "standard",
        "std": "standard",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ("custom", "standard"):
        raise ValueError(
            f"Unsupported mppi_mode={mode!r}; expected 'custom' or 'standard'")
    return normalized


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MPPI benchmark (single-sim mode)")
    parser.add_argument("--config", required=True, help="Path to benchmark_config.yaml")
    parser.add_argument("--mppi-mode", type=str, help="Override mppi_mode (custom/standard)")
    parser.add_argument("--predictor", type=str, help="Override predictor_type (kalman/residual/model/social_vae)")
    parser.add_argument("--residual-alpha", type=float, help="Override residual_alpha")
    parser.add_argument("--residual-beta", type=float, help="Override residual_smoothing_beta")
    parser.add_argument("--residual-clip", type=float, help="Override residual_clip_norm")
    parser.add_argument("--repeats", type=int, help="Override repeats")
    parser.add_argument("--robot-force-scale", type=float,
                        help="Override robot_force_scale (0.0=invisible, 1.0=full reaction)")
    parser.add_argument("--predict-robot-as-agent", action="store_true", default=None,
                        help="Inject robot into people predictor as neighbour agent")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["benchmark"]

    # Override config with CLI arguments
    if args.mppi_mode:
        cfg["mppi_mode"] = args.mppi_mode
    if args.predictor:
        cfg["predictor_type"] = args.predictor
    if args.residual_alpha is not None:
        cfg["residual_alpha"] = args.residual_alpha
    if args.residual_beta is not None:
        cfg["residual_smoothing_beta"] = args.residual_beta
    if args.residual_clip is not None:
        cfg["residual_clip_norm"] = args.residual_clip
    if args.repeats is not None:
        cfg["repeats"] = args.repeats
    if args.robot_force_scale is not None:
        cfg["robot_force_scale"] = args.robot_force_scale
    if args.predict_robot_as_agent:
        cfg["predict_robot_as_agent"] = True

    predictor_type = str(cfg.get("predictor_type", "kalman")).strip().lower()
    if predictor_type not in ("kalman", "model", "social_vae", "socialvae", "social_gru", "residual"):
        predictor_type = "kalman"
    if predictor_type == "socialvae":
        predictor_type = "social_vae"
    if predictor_type == "social_gru":
        predictor_type = "model"
    residual_model_weights = str(cfg.get("residual_model_weights", "")).strip()
    residual_alpha = float(cfg.get("residual_alpha", 0.3))
    residual_smoothing_beta = float(cfg.get("residual_smoothing_beta", 0.8))
    residual_clip_norm = float(cfg.get("residual_clip_norm", 0.35))
    residual_turn_gate_enable = bool(cfg.get("residual_turn_gate_enable", True))
    residual_turn_gate_tau = float(cfg.get("residual_turn_gate_tau", 0.1))
    residual_turn_gate_alpha = float(cfg.get("residual_turn_gate_alpha", 30.0))

    social_vae_repo_path = str(cfg.get("social_vae_repo_path", "")).strip()
    social_vae_ckpt_path = str(cfg.get("social_vae_ckpt_path", "")).strip()
    social_vae_config_path = str(cfg.get("social_vae_config_path", "")).strip()
    social_vae_pred_samples = int(cfg.get("social_vae_pred_samples", 20))
    robot_force_scale = float(cfg.get("robot_force_scale", 0.0))
    predict_robot_as_agent = bool(cfg.get("predict_robot_as_agent", False))

    if "mppi_mode" in cfg:
        mode = _normalize_mppi_mode(cfg["mppi_mode"])
    elif "mppi_modes" in cfg:
        modes_list = cfg["mppi_modes"]
        if isinstance(modes_list, list):
            mode = _normalize_mppi_mode(modes_list[0])
        else:
            mode = _normalize_mppi_mode(modes_list)
        print(f"[warn] 'mppi_modes' found in config – using first entry: {mode}")
    else:
        mode = "custom"

    goals = cfg.get("goals", [{"x": 12.0, "y": 0.0, "yaw": 0.0}])
    repeats = int(cfg.get("repeats", 3))
    repeat_offset = int(cfg.get("repeat_offset", 0))
    default_start = cfg.get("start", {"x": 0.0, "y": 0.0, "yaw": 0.0})
    gui = cfg.get("gui", False)
    sim_speedup = float(cfg.get("sim_speedup", 1.0))
    sim_max_step_size = float(cfg.get("sim_max_step_size", 0.0))
    sim_real_time_update_rate = float(cfg.get("sim_real_time_update_rate", -1.0))
    startup_delay = float(cfg.get("startup_delay_sec", 30.0))
    cooldown = float(cfg.get("cooldown_sec", 5.0))
    reset_hunav_between_episodes = bool(cfg.get("reset_hunav_between_episodes", True))

    scenario = (cfg.get("scenario") or "long_corridor").strip().lower()
    if scenario == "long_corridor":
        map_x_min, map_x_max = -9.9, 9.9
        map_y_min, map_y_max = -3.2, 3.2
    else:
        map_x_min, map_x_max = -9.9, 9.9
        map_y_min, map_y_max = -3.2, 3.2

    episodes = []
    for gi, goal in enumerate(goals):
        source_goal_idx = int(goal.get("goal_idx", gi))
        start = goal.get("start", default_start)
        for pt, coords in [("start", start), ("goal", goal)]:
            gx, gy = float(coords.get("x", 0)), float(coords.get("y", 0))
            if not (map_x_min <= gx <= map_x_max and map_y_min <= gy <= map_y_max):
                print(f"  [warn] goal {gi} {pt}=({gx:.2f},{gy:.2f}) outside map "
                      f"[x:{map_x_min}..{map_x_max}, y:{map_y_min}..{map_y_max}]")
        for rep in range(repeats):
            source_repeat = repeat_offset + rep
            eid = f"{mode}_g{source_goal_idx}_r{source_repeat}"
            episodes.append({
                "episode_id": eid,
                "mppi_mode": mode,
                "goal_idx": source_goal_idx,
                "repeat": source_repeat,
                "start": {
                    "x": float(start.get("x", 0.0)),
                    "y": float(start.get("y", 0.0)),
                    "yaw": float(start.get("yaw", 0.0)),
                },
                "goal": {
                    "x": float(goal.get("x", 0.0)),
                    "y": float(goal.get("y", 0.0)),
                    "yaw": float(goal.get("yaw", 0.0)),
                },
            })

    total = len(episodes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output_dir", "benchmark_results")) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  BENCHMARK  |  mode={mode}  |  predictor={predictor_type}  |  {total} episodes  |  {timestamp}")
    print(f"{'=' * 64}")
    print(f"  goals={len(goals)}  repeats={repeats}  startup={startup_delay:.0f}s")
    print(f"  output: {output_dir}")

    print("\n  Cleaning up orphan processes before start …")
    _cleanup_orphans()

    sim_log_path = str(output_dir / "sim_launch.log")
    sim = _launch_sim(
        cfg["scenario"],
        mode,
        gui=gui,
        sim_speedup=sim_speedup,
        sim_max_step_size=sim_max_step_size,
        sim_real_time_update_rate=sim_real_time_update_rate,
        predictor_type=predictor_type,
        residual_model_weights=residual_model_weights,
        residual_alpha=residual_alpha,
        residual_smoothing_beta=residual_smoothing_beta,
        residual_clip_norm=residual_clip_norm,
        residual_turn_gate_enable=residual_turn_gate_enable,
        residual_turn_gate_tau=residual_turn_gate_tau,
        residual_turn_gate_alpha=residual_turn_gate_alpha,
        social_vae_repo_path=social_vae_repo_path,
        social_vae_ckpt_path=social_vae_ckpt_path,
        social_vae_config_path=social_vae_config_path,
        social_vae_pred_samples=social_vae_pred_samples,
        robot_force_scale=robot_force_scale,
        predict_robot_as_agent=predict_robot_as_agent,
        sim_log_path=sim_log_path,
    )
    print(f"\n  Launched sim (pid {sim.pid}).  Waiting {startup_delay:.0f}s for Nav2 …")
    print(f"  Sim log → {sim_log_path}")
    time.sleep(startup_delay)

    if sim.poll() is not None:
        print("  [!] Simulation exited early – aborting")
        print(f"  Last lines from sim log ({sim_log_path}):")
        try:
            with open(sim_log_path) as _lf:
                lines = _lf.readlines()
            for ln in lines[-40:]:
                print(f"  [sim] {ln.rstrip()}")
        except Exception:
            pass
        _cleanup_orphans()
        sys.exit(1)

    all_results = []
    try:
        if reset_hunav_between_episodes:
            print(f"\n  Running {total} episodes with HuNav reset between episodes …\n")
            for idx, episode in enumerate(episodes):
                if idx > 0:
                    print(f"  Resetting HuNav before episode {idx + 1}/{total} …")
                    if not _reset_hunav_agents(cfg):
                        print("  [!] HuNav reset failed – aborting benchmark")
                        break
                episode_results = _run_session([episode], cfg, str(output_dir), mode)
                all_results.extend(episode_results)
        else:
            print(f"\n  Running {total} episodes without HuNav reset …\n")
            all_results = _run_session(episodes, cfg, str(output_dir), mode)
    except KeyboardInterrupt:
        print("\n  [!] Interrupted by user – cleaning up …")
    finally:
        print("\n  Killing simulation …")
        _kill_sim(sim)
        if getattr(sim, "_log_fh", None):
            try:
                sim._log_fh.close()
            except Exception:
                pass
        print("  Cleaning orphan processes …")
        _cleanup_orphans()
        time.sleep(cooldown)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

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

    print(f"\n{'=' * 64}")
    print(f"  SUMMARY  |  mode={mode}")
    print(f"{'=' * 64}")

    ok = [r for r in all_results if r.get("status") == "SUCCEEDED"]
    missing_people = [r for r in ok if not math.isfinite(float(r.get("min_dist", float("nan"))))]
    print(f"\n  {mode.upper()} MPPI  ({len(ok)}/{len(all_results)} succeeded)")
    if missing_people:
        print(f"  [warn] {len(missing_people)} succeeded episodes had no people samples; min_dist excluded from stats")

    if ok:
        for label, key in [
            ("Time-to-goal  (s)", "time_to_goal"),
            ("Path length   (m)", "path_length"),
            ("Min distance  (m)", "min_dist"),
            ("Collisions      ", "collision_count"),
            ("Violation time(s)", "viol_time"),
            ("Robot influence ", "avg_robot_influence"),
        ]:
            s = _stats([r[key] for r in ok])
            print(f"    {label}:  "
                  f"{s['mean']:.3f} ± {s['std']:.3f}  "
                  f"[{s['min']:.3f} … {s['max']:.3f}]")

    print(f"\n  Results saved to {output_dir}/")
    print()


if __name__ == "__main__":
    main()
