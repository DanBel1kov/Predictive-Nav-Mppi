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

import rclpy
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

    # ── episode-local state ─────────────────────────────────────────
    def _reset_episode_state(self):
        self._people:            list  = []
        self._positions:         list  = []   # [(sim_t, rx, ry)]
        self._min_dists:         list  = []
        self._collision_events:  int   = 0
        self._in_collision:      bool  = False
        self._viol_time:         float = 0.0
        self._last_sample_time         = None
        self._episode_start            = None
        self._nav_result               = None
        self._goal_handle              = None
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
        self.get_logger().info(
            f"\n── Episode {self._episode_idx + 1}/{len(self._episodes)} "
            f"[{ep.get('episode_id', '')}] ──\n"
            f"  start=({ep['start']['x']:.2f}, {ep['start']['y']:.2f})  "
            f"goal=({ep['goal']['x']:.2f}, {ep['goal']['y']:.2f})")

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
        self._people = [(p.position.x, p.position.y) for p in msg.people]

    def _sample(self):
        if self._episode_start is None:
            return
        now = self.get_clock().now()
        dt = (now - self._last_sample_time).nanoseconds * 1e-9
        self._last_sample_time = now

        try:
            tf = self._tf_buf.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
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

    # ── save & advance ───────────────────────────────────────────────
    def _end_episode(self):
        for attr in ("_sample_timer", "_timeout_timer"):
            t = getattr(self, attr)
            if t is not None:
                t.cancel()
            setattr(self, attr, None)

        ep = self._episodes[self._episode_idx]

        total_time  = 0.0
        path_length = 0.0
        if self._positions:
            total_time = self._positions[-1][0]
            for i in range(1, len(self._positions)):
                dx = self._positions[i][1] - self._positions[i - 1][1]
                dy = self._positions[i][2] - self._positions[i - 1][2]
                path_length += math.hypot(dx, dy)

        min_dist = min(self._min_dists) if self._min_dists else float("inf")

        result = {
            "episode_id":      ep.get("episode_id", f"ep{self._episode_idx}"),
            "mppi_mode":       ep.get("mppi_mode",  ""),
            "goal_idx":        ep.get("goal_idx",   self._episode_idx),
            "repeat":          ep.get("repeat",     0),
            "status":          self._nav_result,
            "time_to_goal":    round(total_time,    3),
            "path_length":     round(path_length,   4),
            "min_dist":        round(min_dist,      4),
            "collision_count": self._collision_events,
            "viol_time":       round(self._viol_time, 3),
            "samples":         len(self._positions),
            "goal":            ep["goal"],
            "start":           ep["start"],
        }

        self.get_logger().info(
            f"  ✓ {result['status']}  "
            f"t={result['time_to_goal']:.1f}s  "
            f"path={result['path_length']:.2f}m  "
            f"minD={result['min_dist']:.3f}m  "
            f"coll={result['collision_count']}  "
            f"viol={result['viol_time']:.1f}s")

        ep_file = os.path.join(self.output_dir, f"{result['episode_id']}.json")
        with open(ep_file, "w") as f:
            json.dump(result, f, indent=2)

        self._all_results.append(result)
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

        self.get_logger().info(
            f"Session complete – {len(self._all_results)} episodes saved to {self.output_dir}")
        raise SystemExit(0)


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
