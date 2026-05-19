"""Multi-episode benchmark session.

Runs all episodes inside a single running simulation:
  for each episode → teleport robot → publish AMCL pose → clear costmaps
                    → settle → navigate → collect metrics → save → repeat

Usage (from run_benchmark.py):
    ros2 run predictive_nav_mppi benchmark_session \\
        --ros-args \\
        -p episodes_file:=/tmp/bench_episodes.json \\
        -p output_dir:=/tmp/bench_out \\
        -p nav_timeout:=180.0 \\
        ...
"""

import json
import math
import os
import subprocess
import sys
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import rclpy
import numpy as np
from std_msgs.msg import Float32MultiArray
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Twist
from nav2_msgs.action import NavigateToPose
from people_msgs.msg import People
from std_srvs.srv import Empty
import tf2_ros

try:
    from gazebo_msgs.srv import SetEntityState
    from gazebo_msgs.msg import EntityState
    _HAVE_GAZEBO_MSGS = True
except ImportError:
    _HAVE_GAZEBO_MSGS = False


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _yaw_from_quat(z: float, w: float) -> float:
    return float(2.0 * math.atan2(z, w))


def _speed_magnitudes(obs_xy: np.ndarray, dt: float) -> np.ndarray:
    if obs_xy.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    v = (obs_xy[1:] - obs_xy[:-1]) / max(1e-6, dt)
    return np.linalg.norm(v, axis=1)


def _heading_change(obs_xy: np.ndarray) -> float:
    if obs_xy.shape[0] < 3:
        return 0.0
    step = obs_xy[1:] - obs_xy[:-1]
    ang = np.arctan2(step[:, 1], step[:, 0])
    if ang.shape[0] < 2:
        return 0.0
    d = np.diff(ang)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


def _min_neighbor_distance(obs_xy: np.ndarray, neigh_xy: List[np.ndarray]) -> float:
    if not neigh_xy:
        return float("inf")
    p = obs_xy[-1]
    dmin = float("inf")
    for nxy in neigh_xy:
        if nxy.shape[0] == 0:
            continue
        d = float(np.linalg.norm(p - nxy[-1]))
        if d < dmin:
            dmin = d
    return dmin


def _interaction_metrics(obs_xy: np.ndarray, neigh_xy: List[np.ndarray], obs_dt: float) -> Dict[str, float]:
    path_len = float(np.sum(np.linalg.norm(obs_xy[1:] - obs_xy[:-1], axis=1))) if obs_xy.shape[0] >= 2 else 0.0
    displacement = float(np.linalg.norm(obs_xy[-1] - obs_xy[0])) if obs_xy.shape[0] >= 2 else 0.0
    speeds = _speed_magnitudes(obs_xy, obs_dt)
    heading_change_deg = math.degrees(_heading_change(obs_xy))
    curvature_ratio = path_len / max(displacement, 1e-6)
    min_neighbor_distance = _min_neighbor_distance(obs_xy, neigh_xy)
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


def _classify_interaction_tags(
    obs_xy: np.ndarray,
    neigh_xy: List[np.ndarray],
    obs_dt: float,
    interaction_dist: float,
    dense_neighbors_min: int,
    turn_threshold_deg: float,
    stop_speed_thresh: float,
    stop_go_delta: float,
    moving_speed_min: float,
) -> Set[str]:
    tags: Set[str] = {"all"}
    n_neigh = len(neigh_xy)
    min_neigh_dist = _min_neighbor_distance(obs_xy, neigh_xy)
    if n_neigh > 0 and min_neigh_dist <= interaction_dist:
        tags.add("interaction")
    if n_neigh >= dense_neighbors_min:
        tags.add("dense_interaction")

    turn_rad = math.radians(max(0.0, turn_threshold_deg))
    if _heading_change(obs_xy) >= turn_rad:
        tags.add("turning")

    sp = _speed_magnitudes(obs_xy, obs_dt)
    if sp.size > 0:
        s_min = float(np.min(sp))
        s_max = float(np.max(sp))
        if s_max >= moving_speed_min and s_min <= stop_speed_thresh and (s_max - s_min) >= stop_go_delta:
            tags.add("stop_go")

    complexity_axes = (
        int("interaction" in tags)
        + int("dense_interaction" in tags)
        + int("turning" in tags)
        + int("stop_go" in tags)
    )
    if complexity_axes >= 2:
        tags.add("complex")
    if complexity_axes >= 3:
        tags.add("very_complex")
    return tags


def _is_linear_interaction(tags: Set[str], metrics: Dict[str, float], turn_threshold_deg: float) -> bool:
    if "interaction" in tags or "dense_interaction" in tags:
        return False
    if "turning" in tags or "complex" in tags or "very_complex" in tags:
        return False
    return (
        metrics["heading_change_deg"] < max(12.0, 0.35 * turn_threshold_deg)
        and metrics["curvature_ratio"] < 1.05
    )


def _yaw_quat(yaw: float):
    """Return (z, w) quaternion fields for pure-yaw rotation."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class BenchmarkSession(Node):
    """Runs a full benchmark session (multiple episodes) without restarting Gazebo."""

    def __init__(self):
        super().__init__("benchmark_session")

        p = self.declare_parameter

        # Metric params
        self.robot_radius   = float(p("robot_radius",   0.22).value)
        self.person_radius  = float(p("person_radius",  0.10).value)
        self.personal_space = float(p("personal_space", 0.60).value)
        self.nav_timeout    = float(p("nav_timeout",  180.0).value)
        self.sample_rate    = float(p("sample_rate_hz", 10.0).value)
        self.settle_time    = float(p("settle_time",    5.0).value)

        # I/O
        self.output_dir      = str(p("output_dir",      "/tmp/benchmark_session").value)
        self.episodes_file   = str(p("episodes_file",   "/tmp/bench_episodes.json").value)
        self.global_frame    = str(p("global_frame",    "map").value)
        self.robot_frame     = str(p("robot_frame",     "base_link").value)
        self.people_topic    = str(p("people_topic",    "/people").value)
        self.robot_model_name = str(p("robot_model_name", "waffle").value)
        self.interaction_range = float(p("interaction_range", 3.5).value)
        self.interaction_fov_deg = float(p("interaction_fov_deg", 180.0).value)
        self.interaction_near_robot_only = bool(p("interaction_near_robot_only", False).value)
        self.interaction_obs_len = int(p("interaction_obs_len", 8).value)
        self.interaction_obs_dt = float(p("interaction_obs_dt", 0.4).value)
        self.interaction_neighbor_radius = float(p("interaction_neighbor_radius", 4.0).value)
        self.interaction_min_gap_sec = float(p("interaction_min_gap_sec", 2.0).value)
        self.interaction_dist = float(p("interaction_dist", 1.5).value)
        self.dense_neighbors_min = int(p("dense_neighbors_min", 3).value)
        self.turn_threshold_deg = float(p("turn_threshold_deg", 20.0).value)
        self.stop_speed_thresh = float(p("stop_speed_thresh", 0.15).value)
        self.stop_go_delta = float(p("stop_go_delta", 0.25).value)
        self.moving_speed_min = float(p("moving_speed_min", 0.3).value)

        os.makedirs(self.output_dir, exist_ok=True)

        with open(self.episodes_file) as f:
            self._episodes: list = json.load(f)

        if not self._episodes:
            self.get_logger().error("Episode list is empty – nothing to do")
            raise SystemExit(1)

        self.get_logger().info(
            f"Loaded {len(self._episodes)} episodes from {self.episodes_file}")

        # ── ROS plumbing ────────────────────────────────────────────
        cbg = ReentrantCallbackGroup()
        self._cbg = cbg

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        self._people_sub = self.create_subscription(
            People, self.people_topic, self._on_people, 10, callback_group=cbg)
        self._robot_forces_sub = self.create_subscription(
            Float32MultiArray, 'human_robot_forces', self._on_robot_forces, 10,
            callback_group=cbg)

        self._initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)

        self._nav_cli = ActionClient(
            self, NavigateToPose, "navigate_to_pose", callback_group=cbg)

        # Gazebo teleportation
        self._gz_available = False
        self._gz_cli = None
        self._gz_cli_alt = None
        if _HAVE_GAZEBO_MSGS:
            self._gz_cli = self.create_client(
                SetEntityState, "/gazebo/set_entity_state", callback_group=cbg)
            self._gz_cli_alt = self.create_client(
                SetEntityState, "/set_entity_state", callback_group=cbg)

        # Costmap clear services
        self._clear_local  = self.create_client(
            Empty, "/local_costmap/clear_entirely_local_costmap",  callback_group=cbg)
        self._clear_global = self.create_client(
            Empty, "/global_costmap/clear_entirely_global_costmap", callback_group=cbg)

        # ── state ───────────────────────────────────────────────────
        self._episode_idx = 0
        self._all_results: list = []
        self._sample_timer  = None
        self._timeout_timer = None
        self._reset_episode_state()

        # Main tick (1 Hz) handles wait_nav2 and settle phases
        self._phase = "wait_nav2"
        self._phase_timer = self.create_timer(1.0, self._phase_tick, callback_group=cbg)
        self._last_reloc_method = "none"
        self._tf_warn_count = 0
        self._people_seen_samples = 0
        self._name_to_track_id: Dict[str, int] = {}
        self._next_track_id = 1
        self._all_interaction_examples: List[dict] = []
        self._all_interaction_counts: DefaultDict[str, int] = defaultdict(int)

    # ── episode-local state ─────────────────────────────────────────
    def _reset_episode_state(self):
        self._people:            list  = []
        self._people_latest:     Dict[int, Tuple[float, float]] = {}
        self._positions:         list  = []   # [(sim_t, rx, ry)]
        self._min_dists:         list  = []
        self._collision_events:  int   = 0
        self._in_collision:      bool  = False
        self._viol_time:         float = 0.0
        self._last_sample_time         = None
        self._episode_start            = None
        self._nav_result               = None
        self._goal_handle              = None
        self._tf_warn_count            = 0
        self._people_seen_samples      = 0
        self._current_robot_forces:    list  = []  # latest [|F_robot|, ...] per agent
        self._robot_force_samples:     list  = []  # mean |F_robot| per sample tick
        self._robot_force_nearest_samples: list = []  # |F_robot| on nearest person
        self._robot_force_peak_samples:    list = []  # max |F_robot| over nearby people
        self._robot_force_close_samples:   list = []  # mean |F_robot| inside personal_space
        self._robot_force_peak_auc:    float = 0.0
        self._interaction_examples:    List[dict] = []
        self._interaction_counts:      DefaultDict[str, int] = defaultdict(int)
        self._interaction_last_emit_sec: Dict[int, float] = {}
        self._people_tracks:           Dict[int, deque] = {}
        for attr in ("_sample_timer", "_timeout_timer"):
            t = getattr(self, attr, None)
            if t is not None:
                t.cancel()
            setattr(self, attr, None)

    # ── phase tick ──────────────────────────────────────────────────
    def _phase_tick(self):
        if self._phase == "wait_nav2":
            if self._nav_cli.wait_for_server(timeout_sec=1.0):
                # Discover which Gazebo relocation method works
                if _HAVE_GAZEBO_MSGS:
                    if self._gz_cli is not None and self._gz_cli.wait_for_service(timeout_sec=3.0):
                        self._gz_available = True
                        self.get_logger().info(
                            "/gazebo/set_entity_state ready – teleport enabled")
                    elif self._gz_cli_alt is not None and self._gz_cli_alt.wait_for_service(timeout_sec=3.0):
                        self._gz_cli = self._gz_cli_alt
                        self._gz_available = True
                        self.get_logger().info(
                            "/set_entity_state ready – teleport enabled")
                    else:
                        self._gz_available = False
                        self.get_logger().warn(
                            "set_entity_state unavailable; will use 'gz model' CLI fallback")
                self.get_logger().info("Nav2 ready – starting episode 1")
                self._start_episode()

        elif self._phase == "settle":
            elapsed = (self.get_clock().now() - self._settle_t0).nanoseconds * 1e-9
            if elapsed >= self.settle_time:
                self._phase = "navigating"
                self._send_goal()

    # ── episode orchestration ────────────────────────────────────────
    def _start_episode(self):
        if self._episode_idx >= len(self._episodes):
            self._finish_session()
            return

        ep = self._episodes[self._episode_idx]
        print(
            f"\n  ▶ Episode {self._episode_idx + 1}/{len(self._episodes)}: "
            f"[{ep.get('episode_id', '')}]  "
            f"start=({ep['start']['x']:.2f}, {ep['start']['y']:.2f}) → "
            f"goal=({ep['goal']['x']:.2f}, {ep['goal']['y']:.2f})",
            flush=True)
        self.get_logger().info(
            f"── Episode {self._episode_idx + 1}/{len(self._episodes)} "
            f"[{ep.get('episode_id', '')}] ──")

        self._reset_episode_state()
        self._phase = "teleporting"
        self._do_teleport(ep["start"])

    def _do_teleport(self, start: dict):
        """Teleport robot in Gazebo, then publish AMCL initial pose."""
        sx, sy = start["x"], start["y"]
        syaw = start.get("yaw", 0.0)

        if _HAVE_GAZEBO_MSGS and self._gz_cli is not None and self._gz_available:
            self._last_reloc_method = "set_entity_state"
            req = SetEntityState.Request()
            state = EntityState()
            state.name = self.robot_model_name
            state.pose = Pose()
            state.pose.position.x = sx
            state.pose.position.y = sy
            state.pose.position.z = 0.0
            qz, qw = _yaw_quat(syaw)
            state.pose.orientation.z = qz
            state.pose.orientation.w = qw
            state.twist = Twist()
            state.reference_frame = "world"
            req.state = state
            fut = self._gz_cli.call_async(req)
            fut.add_done_callback(self._on_teleport_done)
            return

        # Fallback: use `gz model` CLI (Gazebo internal transport, no ROS plugin needed).
        self._last_reloc_method = "gz_model_cli"
        self._gz_model_teleport(sx, sy, syaw)

    def _on_teleport_done(self, future):
        ep = self._episodes[self._episode_idx]
        start = ep["start"]
        try:
            res = future.result()
            if not res.success:
                self.get_logger().warn("Gazebo set_entity_state returned success=False")
        except Exception as e:
            self.get_logger().error(f"Teleport service call failed: {e}")
        self._after_teleport(start["x"], start["y"], start.get("yaw", 0.0))

    def _gz_model_teleport(self, x: float, y: float, yaw: float):
        """Move robot using `gz model` CLI (works with any running gzserver)."""
        cmd = [
            "gz", "model",
            "-m", self.robot_model_name,
            "-x", str(x),
            "-y", str(y),
            "-z", "0.05",
            "-R", "0", "-P", "0",
            "-Y", str(yaw),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5.0)
            if result.returncode != 0:
                self.get_logger().warn(
                    f"gz model returned code {result.returncode}: {result.stderr.strip()}")
            else:
                self.get_logger().info(f"gz model teleport OK → ({x:.2f}, {y:.2f})")
        except FileNotFoundError:
            self.get_logger().error(
                "'gz' CLI not found — cannot teleport. Install gazebo command-line tools.")
        except subprocess.TimeoutExpired:
            self.get_logger().error("gz model command timed out")
        except Exception as e:
            self.get_logger().error(f"gz model teleport error: {e}")

        self._after_teleport(x, y, yaw)

    def _after_teleport(self, x: float, y: float, yaw: float):
        # 2. Tell AMCL where we are (publish several times for reliability)
        self._publish_initial_pose(x, y, yaw)

        # 3. Wait for sensors/TF after relocation then clear costmaps twice.
        #    Respawn needs longer stabilization than set_entity_state.
        first_wait = 1.5 if self._last_reloc_method == "set_entity_state" else 3.0
        second_wait = 1.0
        _ref1 = [None]
        _ref2 = [None]

        def _clear_once():
            _ref1[0].cancel()
            self._publish_initial_pose(x, y, yaw)
            if self._clear_local.service_is_ready():
                self._clear_local.call_async(Empty.Request())
            if self._clear_global.service_is_ready():
                self._clear_global.call_async(Empty.Request())

            def _clear_twice():
                _ref2[0].cancel()
                self._publish_initial_pose(x, y, yaw)
                if self._clear_local.service_is_ready():
                    self._clear_local.call_async(Empty.Request())
                if self._clear_global.service_is_ready():
                    self._clear_global.call_async(Empty.Request())
                # 4. Settle
                self._settle_t0 = self.get_clock().now()
                self._phase = "settle"

            t2 = self.create_timer(second_wait, _clear_twice, callback_group=self._cbg)
            _ref2[0] = t2

        t1 = self.create_timer(first_wait, _clear_once, callback_group=self._cbg)
        _ref1[0] = t1

    def _publish_initial_pose(self, x: float, y: float, yaw: float):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        qz, qw = _yaw_quat(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        cov = [0.0] * 36
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = 0.0685
        msg.pose.covariance = cov
        for _ in range(3):
            self._initpose_pub.publish(msg)

    # ── navigation ──────────────────────────────────────────────────
    def _send_goal(self):
        ep = self._episodes[self._episode_idx]
        gx = ep["goal"]["x"]
        gy = ep["goal"]["y"]
        gyaw = ep["goal"].get("yaw", 0.0)

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        qz, qw = _yaw_quat(gyaw)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        self._episode_start    = self.get_clock().now()
        self._last_sample_time = self._episode_start
        self._positions.append((0.0, float(ep["start"]["x"]), float(ep["start"]["y"])))

        self._sample_timer = self.create_timer(
            1.0 / self.sample_rate, self._sample, callback_group=self._cbg)
        self._timeout_timer = self.create_timer(
            self.nav_timeout, self._on_timeout, callback_group=self._cbg)

        fut = self._nav_cli.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("Goal REJECTED by Nav2")
            self._nav_result = "REJECTED"
            self._end_episode()
            return
        self.get_logger().info("Goal accepted – navigating …")
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        self._nav_result = {
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_ABORTED:   "ABORTED",
            GoalStatus.STATUS_CANCELED:  "CANCELED",
        }.get(status, f"STATUS_{status}")
        self.get_logger().info(f"Navigation result: {self._nav_result}")
        self._end_episode()

    def _on_timeout(self):
        self.get_logger().warn("Navigation TIMEOUT")
        self._nav_result = "TIMEOUT"
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self._end_episode()

    # ── metric sampling ──────────────────────────────────────────────
    def _on_people(self, msg):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._people = []
        self._people_latest = {}
        for idx, person in enumerate(msg.people):
            pid = self._person_id(person, idx + 1)
            px = float(person.position.x)
            py = float(person.position.y)
            self._people.append((px, py))
            self._people_latest[pid] = (px, py)
            if pid not in self._people_tracks:
                self._people_tracks[pid] = deque(maxlen=128)
            self._people_tracks[pid].append((now_sec, px, py))
        if self._people:
            self._people_seen_samples += 1

    def _on_robot_forces(self, msg):
        # msg.data = [fx0, fy0, fx1, fy1, ...] per agent, same order as /people topic.
        self._current_robot_forces = [
            math.hypot(msg.data[i], msg.data[i + 1])
            for i in range(0, len(msg.data) - 1, 2)
        ]

    def _sample(self):
        if self._episode_start is None:
            return
        now = self.get_clock().now()
        dt = (now - self._last_sample_time).nanoseconds * 1e-9
        self._last_sample_time = now

        tf = self._lookup_robot_tf()
        if tf is None:
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        ryaw = _yaw_from_quat(tf.transform.rotation.z, tf.transform.rotation.w)
        now_abs_sec = now.nanoseconds * 1e-9
        sim_t = (now - self._episode_start).nanoseconds * 1e-9
        self._positions.append((sim_t, rx, ry))

        if self._people:
            min_d = min(math.hypot(rx - px, ry - py) for px, py in self._people)
        else:
            min_d = float("inf")
        self._min_dists.append(min_d)

        thresh = self.robot_radius + self.person_radius
        if min_d < thresh:
            if not self._in_collision:
                self._collision_events += 1
                self._in_collision = True
        else:
            self._in_collision = False

        if min_d < self.personal_space:
            self._viol_time += dt

        self._collect_interaction_examples(
            now_sec=now_abs_sec,
            sim_t=sim_t,
            rx=rx,
            ry=ry,
            ryaw=ryaw,
        )

        # Robot social-force metrics use /human_robot_forces from HuNav.  The
        # legacy metric averages all nearby people; nearest/peak are more
        # sensitive to the person who actually constrains the robot.
        # /human_robot_forces and /people are published in the same agent order.
        if self._current_robot_forces and self._people:
            sensor_range = 3.5  # TurtleBot3 lidar range, metres
            entries = [
                (math.hypot(rx - px, ry - py), self._current_robot_forces[i])
                for i, (px, py) in enumerate(self._people)
                if i < len(self._current_robot_forces)
            ]
            nearby = [force for dist, force in entries if dist < sensor_range]
            if nearby:
                self._robot_force_samples.append(sum(nearby) / len(nearby))
                peak_force = max(nearby)
                self._robot_force_peak_samples.append(peak_force)
                self._robot_force_peak_auc += peak_force * max(0.0, dt)

            if entries:
                _, nearest_force = min(entries, key=lambda item: item[0])
                self._robot_force_nearest_samples.append(nearest_force)

            close = [force for dist, force in entries if dist < self.personal_space]
            if close:
                self._robot_force_close_samples.append(sum(close) / len(close))

    def _person_id(self, person: Any, fallback_idx: int) -> int:
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))
        name = str(getattr(person, "name", ""))
        if name.startswith("robot_0"):
            return 0
        if not name:
            return int(fallback_idx)
        if name not in self._name_to_track_id:
            self._name_to_track_id[name] = self._next_track_id
            self._next_track_id += 1
        return self._name_to_track_id[name]

    def _sample_track_obs(self, history: deque, now_sec: float) -> Optional[np.ndarray]:
        if len(history) < 2:
            return None
        times = np.asarray([item[0] for item in history], dtype=np.float64)
        xy = np.asarray([[item[1], item[2]] for item in history], dtype=np.float64)
        target_t = np.asarray(
            [
                now_sec - (self.interaction_obs_len - 1 - i) * self.interaction_obs_dt
                for i in range(self.interaction_obs_len)
            ],
            dtype=np.float64,
        )
        if target_t[0] < times[0] or target_t[-1] > times[-1]:
            return None
        out = np.zeros((self.interaction_obs_len, 2), dtype=np.float64)
        for i, t in enumerate(target_t):
            idx = int(np.searchsorted(times, t, side="left"))
            if idx == 0:
                out[i] = xy[0]
            elif idx >= len(times):
                out[i] = xy[-1]
            else:
                t0 = times[idx - 1]
                t1 = times[idx]
                if t1 <= t0:
                    out[i] = xy[idx]
                else:
                    a = float((t - t0) / (t1 - t0))
                    out[i] = (1.0 - a) * xy[idx - 1] + a * xy[idx]
        return out

    def _collect_interaction_examples(self, now_sec: float, sim_t: float, rx: float, ry: float, ryaw: float):
        if not self._people_latest:
            return
        visible: List[Tuple[int, float, float, float, float]] = []
        half_fov = math.radians(max(1.0, self.interaction_fov_deg)) * 0.5
        for pid, (px, py) in self._people_latest.items():
            dx = px - rx
            dy = py - ry
            dist = math.hypot(dx, dy)
            bearing = _wrap_to_pi(math.atan2(dy, dx) - ryaw)
            if self.interaction_near_robot_only:
                if dist > self.interaction_range:
                    continue
                if abs(bearing) > half_fov:
                    continue
            visible.append((pid, px, py, dist, bearing))

        for pid, px, py, dist, bearing in visible:
            last_emit = self._interaction_last_emit_sec.get(pid, -1e9)
            if (now_sec - last_emit) < self.interaction_min_gap_sec:
                continue
            obs_xy = self._sample_track_obs(self._people_tracks.get(pid, deque()), now_sec)
            if obs_xy is None:
                continue
            neigh_xy: List[np.ndarray] = []
            for oid, opx, opy, _, _ in visible:
                if oid == pid:
                    continue
                if math.hypot(opx - px, opy - py) > self.interaction_neighbor_radius:
                    continue
                other_obs = self._sample_track_obs(self._people_tracks.get(oid, deque()), now_sec)
                if other_obs is not None:
                    neigh_xy.append(other_obs)

            tags = _classify_interaction_tags(
                obs_xy=obs_xy,
                neigh_xy=neigh_xy,
                obs_dt=self.interaction_obs_dt,
                interaction_dist=self.interaction_dist,
                dense_neighbors_min=self.dense_neighbors_min,
                turn_threshold_deg=self.turn_threshold_deg,
                stop_speed_thresh=self.stop_speed_thresh,
                stop_go_delta=self.stop_go_delta,
                moving_speed_min=self.moving_speed_min,
            )
            metrics = _interaction_metrics(obs_xy=obs_xy, neigh_xy=neigh_xy, obs_dt=self.interaction_obs_dt)
            categories = set(tags)
            categories.discard("all")
            if _is_linear_interaction(tags, metrics, self.turn_threshold_deg):
                categories.add("linear")
            if not categories:
                continue

            example = {
                "episode_id": self._episodes[self._episode_idx].get("episode_id", f"ep{self._episode_idx}"),
                "person_id": int(pid),
                "sample_time": round(float(sim_t), 3),
                "distance": round(float(dist), 4),
                "bearing_deg": round(math.degrees(float(bearing)), 2),
                "categories": sorted(categories),
                "metrics": {
                    "neighbor_count": int(metrics["neighbor_count"]),
                    "heading_change_deg": round(float(metrics["heading_change_deg"]), 3),
                    "curvature_ratio": round(float(metrics["curvature_ratio"]), 4),
                    "mean_speed": round(float(metrics["mean_speed"]), 4),
                    "min_neighbor_distance": round(float(metrics["min_neighbor_distance"]), 4),
                },
            }
            self._interaction_examples.append(example)
            for category in categories:
                self._interaction_counts[category] += 1
            self._interaction_last_emit_sec[pid] = now_sec

    # ── save & advance ───────────────────────────────────────────────
    def _end_episode(self):
        for attr in ("_sample_timer", "_timeout_timer"):
            t = getattr(self, attr)
            if t is not None:
                t.cancel()
            setattr(self, attr, None)

        ep = self._episodes[self._episode_idx]

        # Try one final TF sample at episode end so metrics do not collapse to
        # zero when intermediate timer callbacks were starved or TF was late.
        if self._episode_start is not None:
            tf = self._lookup_robot_tf()
            if tf is not None:
                now = self.get_clock().now()
                sim_t = (now - self._episode_start).nanoseconds * 1e-9
                rx = tf.transform.translation.x
                ry = tf.transform.translation.y
                if not self._positions or (rx, ry) != (self._positions[-1][1], self._positions[-1][2]):
                    self._positions.append((sim_t, rx, ry))
                if self._people:
                    min_d = min(math.hypot(rx - px, ry - py) for px, py in self._people)
                    self._min_dists.append(min_d)

        total_time  = 0.0
        path_length = 0.0
        if self._positions:
            total_time = self._positions[-1][0]
            for i in range(1, len(self._positions)):
                dx = self._positions[i][1] - self._positions[i - 1][1]
                dy = self._positions[i][2] - self._positions[i - 1][2]
                path_length += math.hypot(dx, dy)

        finite_min_dists = [d for d in self._min_dists if math.isfinite(d)]
        min_dist = min(finite_min_dists) if finite_min_dists else float("nan")
        avg_dist = (
            sum(finite_min_dists) / len(finite_min_dists)
            if finite_min_dists else float("nan")
        )

        result = {
            "episode_id":      ep.get("episode_id", f"ep{self._episode_idx}"),
            "mppi_mode":       ep.get("mppi_mode",  ""),
            "goal_idx":        ep.get("goal_idx",   self._episode_idx),
            "repeat":          ep.get("repeat",     0),
            "status":          self._nav_result,
            "time_to_goal":    round(total_time,    3),
            "path_length":     round(path_length,   4),
            "min_dist":        round(min_dist,      4),
            "avg_dist":        round(avg_dist,      4),
            "collision_count": self._collision_events,
            "viol_time":       round(self._viol_time, 3),
            "samples":         len(self._positions),
            "people_seen_samples": self._people_seen_samples,
            # Mean force on nearby agents only (|F| > threshold); nan if nobody was near.
            "avg_robot_influence": round(
                sum(self._robot_force_samples) / len(self._robot_force_samples), 4)
                if self._robot_force_samples else float("nan"),
            "nearest_robot_influence": round(
                sum(self._robot_force_nearest_samples) / len(self._robot_force_nearest_samples), 4)
                if self._robot_force_nearest_samples else float("nan"),
            "peak_robot_influence": round(
                sum(self._robot_force_peak_samples) / len(self._robot_force_peak_samples), 4)
                if self._robot_force_peak_samples else float("nan"),
            "close_robot_influence": round(
                sum(self._robot_force_close_samples) / len(self._robot_force_close_samples), 4)
                if self._robot_force_close_samples else float("nan"),
            "robot_influence_auc": round(self._robot_force_peak_auc, 4)
                if self._robot_force_peak_samples else float("nan"),
            "interaction_examples": len(self._interaction_examples),
            "interaction_counts": dict(sorted(self._interaction_counts.items())),
            "goal":            ep["goal"],
            "start":           ep["start"],
        }

        # Print results directly to stdout for immediate visibility
        robot_infl = result['avg_robot_influence']
        peak_robot_infl = result['peak_robot_influence']
        infl_str = f"{robot_infl:.3f}" if math.isfinite(robot_infl) else "n/a"
        peak_infl_str = f"{peak_robot_infl:.3f}" if math.isfinite(peak_robot_infl) else "n/a"
        result_line = (
            f"[Episode {self._episode_idx}/{len(self._episodes)}]  "
            f"✓ {result['status']}  "
            f"t={result['time_to_goal']:.1f}s  "
            f"path={result['path_length']:.2f}m  "
            f"minD={result['min_dist']:.3f}m  "
            f"avgD={result['avg_dist']:.3f}m  "
            f"coll={result['collision_count']}  "
            f"viol={result['viol_time']:.1f}s  "
            f"rob_infl={infl_str}  "
            f"rob_peak={peak_infl_str}"
        )
        print(f"  {result_line}", flush=True)

        self.get_logger().info(
            f"  ✓ {result['status']}  "
            f"t={result['time_to_goal']:.1f}s  "
            f"path={result['path_length']:.2f}m  "
            f"minD={result['min_dist']:.3f}m  "
            f"avgD={result['avg_dist']:.3f}m  "
            f"coll={result['collision_count']}  "
            f"viol={result['viol_time']:.1f}s  "
            f"people_seen={result['people_seen_samples']}  "
            f"rob_infl={infl_str}  "
            f"rob_peak={peak_infl_str}")

        ep_file = os.path.join(self.output_dir, f"{result['episode_id']}.json")
        with open(ep_file, "w") as f:
            json.dump(result, f, indent=2)

        interactions_file = os.path.join(self.output_dir, f"{result['episode_id']}_interactions.json")
        with open(interactions_file, "w") as f:
            json.dump(self._interaction_examples, f, indent=2)

        self._all_results.append(result)
        self._all_interaction_examples.extend(self._interaction_examples)
        for category, count in self._interaction_counts.items():
            self._all_interaction_counts[category] += int(count)
        self._episode_idx += 1

        # Brief pause then start next episode (one-shot timer)
        self._phase = "between_episodes"
        _timer_ref = [None]

        def _next():
            _timer_ref[0].cancel()
            self._start_episode()

        t = self.create_timer(1.5, _next, callback_group=self._cbg)
        _timer_ref[0] = t

    def _finish_session(self):
        self._phase_timer.cancel()

        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(self._all_results, f, indent=2)

        interaction_summary = {
            "counts": dict(sorted(self._all_interaction_counts.items())),
            "examples": self._all_interaction_examples,
        }
        interaction_summary_file = os.path.join(self.output_dir, "interaction_summary.json")
        with open(interaction_summary_file, "w") as f:
            json.dump(interaction_summary, f, indent=2)

        self.get_logger().info(
            f"Session complete – {len(self._all_results)} episodes saved to {self.output_dir}")
        raise SystemExit(0)

    def _lookup_robot_tf(self):
        frames = [self.robot_frame]
        if self.robot_frame != "base_footprint":
            frames.append("base_footprint")
        if self.robot_frame != "base_link":
            frames.append("base_link")

        last_exc = None
        for frame in frames:
            try:
                return self._tf_buf.lookup_transform(
                    self.global_frame, frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as exc:
                last_exc = exc

        self._tf_warn_count += 1
        if self._tf_warn_count <= 5:
            self.get_logger().warn(
                f"TF lookup failed for {frames}: {last_exc}")
        return None


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkSession()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
