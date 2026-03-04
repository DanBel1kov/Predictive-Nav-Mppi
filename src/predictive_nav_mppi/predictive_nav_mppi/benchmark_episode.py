"""Single benchmark episode: send one navigation goal, collect metrics, save, exit."""

import json
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from people_msgs.msg import People
import tf2_ros


class BenchmarkEpisode(Node):
    def __init__(self):
        super().__init__("benchmark_episode")

        # ── parameters ──────────────────────────────────────────────
        p = self.declare_parameter
        self.goal_x = float(p("goal_x", 12.0).value)
        self.goal_y = float(p("goal_y", 0.0).value)
        self.goal_yaw = float(p("goal_yaw", 0.0).value)
        self.robot_radius = float(p("robot_radius", 0.22).value)
        self.person_radius = float(p("person_radius", 0.10).value)
        self.personal_space = float(p("personal_space", 0.6).value)
        self.nav_timeout = float(p("nav_timeout", 180.0).value)
        self.output_file = str(p("output_file", "/tmp/benchmark_episode.json").value)
        self.global_frame = str(p("global_frame", "map").value)
        self.robot_frame = str(p("robot_frame", "base_link").value)
        self.people_topic = str(p("people_topic", "/people").value)
        self.sample_rate = float(p("sample_rate_hz", 10.0).value)
        self.settle_time = float(p("settle_time", 5.0).value)
        self.start_x = float(p("start_x", 0.0).value)
        self.start_y = float(p("start_y", 0.0).value)
        self.start_yaw = float(p("start_yaw", 0.0).value)

        # ── runtime state ───────────────────────────────────────────
        self._people: list = []
        self._positions: list = []      # [(sim_t, x, y)]
        self._min_dists: list = []      # [float]
        self._collision_events = 0
        self._in_collision = False
        self._viol_time = 0.0
        self._last_sample_time = None
        self._episode_start = None
        self._done = False
        self._nav_result = None
        self._goal_handle = None

        # ── ROS plumbing ────────────────────────────────────────────
        cbg = ReentrantCallbackGroup()
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        self._people_sub = self.create_subscription(
            People, self.people_topic, self._on_people, 10,
            callback_group=cbg)

        self._initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)

        self._nav_cli = ActionClient(
            self, NavigateToPose, "navigate_to_pose",
            callback_group=cbg)

        self._sample_timer = None
        self._timeout_timer = None

        # kick off the state machine
        self._phase = "wait_nav2"
        self._phase_timer = self.create_timer(1.0, self._phase_tick, callback_group=cbg)

    # ── phase state machine ─────────────────────────────────────────
    def _phase_tick(self):
        if self._done:
            return

        if self._phase == "wait_nav2":
            self.get_logger().info("Waiting for navigate_to_pose action server …")
            if self._nav_cli.wait_for_server(timeout_sec=1.0):
                self.get_logger().info("Nav2 ready.  Publishing initial pose …")
                self._publish_initial_pose()
                self._phase = "settle"
                self._settle_t0 = self.get_clock().now()

        elif self._phase == "settle":
            elapsed = (self.get_clock().now() - self._settle_t0).nanoseconds * 1e-9
            if elapsed >= self.settle_time:
                self.get_logger().info("Sending navigation goal …")
                self._send_goal()
                self._phase = "navigating"

    # ── initial pose ────────────────────────────────────────────────
    def _publish_initial_pose(self):
        for _ in range(3):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = self.global_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = self.start_x
            msg.pose.pose.position.y = self.start_y
            msg.pose.pose.orientation.z = math.sin(self.start_yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(self.start_yaw / 2.0)
            cov = [0.0] * 36
            cov[0] = 0.25
            cov[7] = 0.25
            cov[35] = 0.0685
            msg.pose.covariance = cov
            self._initpose_pub.publish(msg)

    # ── navigation goal ─────────────────────────────────────────────
    def _send_goal(self):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.goal_x
        goal.pose.pose.position.y = self.goal_y
        goal.pose.pose.orientation.z = math.sin(self.goal_yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(self.goal_yaw / 2.0)

        self._episode_start = self.get_clock().now()
        self._last_sample_time = self._episode_start

        self._sample_timer = self.create_timer(
            1.0 / self.sample_rate, self._sample)
        self._timeout_timer = self.create_timer(
            self.nav_timeout, self._on_timeout)

        future = self._nav_cli.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("Goal REJECTED by Nav2")
            self._nav_result = "REJECTED"
            self._finish()
            return
        self.get_logger().info("Goal accepted – navigating …")
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        status_map = {
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
        }
        self._nav_result = status_map.get(status, f"STATUS_{status}")
        self.get_logger().info(f"Navigation finished: {self._nav_result}")
        self._finish()

    def _on_timeout(self):
        if not self._done:
            self.get_logger().warn("Navigation TIMEOUT")
            self._nav_result = "TIMEOUT"
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
            self._finish()

    # ── people callback ─────────────────────────────────────────────
    def _on_people(self, msg):
        self._people = [(p.position.x, p.position.y) for p in msg.people]

    # ── metric sampling (sim-time timer) ────────────────────────────
    def _sample(self):
        if self._done or self._episode_start is None:
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
            min_d = min(
                math.hypot(rx - px, ry - py) for px, py in self._people)
        else:
            min_d = float("inf")
        self._min_dists.append(min_d)

        collision_thresh = self.robot_radius + self.person_radius
        if min_d < collision_thresh:
            if not self._in_collision:
                self._collision_events += 1
                self._in_collision = True
        else:
            self._in_collision = False

        if min_d < self.personal_space:
            self._viol_time += dt

    # ── finish & save ───────────────────────────────────────────────
    def _finish(self):
        if self._done:
            return
        self._done = True
        if self._sample_timer:
            self._sample_timer.cancel()
        if self._timeout_timer:
            self._timeout_timer.cancel()
        self._phase_timer.cancel()

        total_time = 0.0
        path_length = 0.0
        if self._positions:
            total_time = self._positions[-1][0]
            for i in range(1, len(self._positions)):
                dx = self._positions[i][1] - self._positions[i - 1][1]
                dy = self._positions[i][2] - self._positions[i - 1][2]
                path_length += math.hypot(dx, dy)

        min_dist = min(self._min_dists) if self._min_dists else float("inf")

        results = {
            "status": self._nav_result,
            "time_to_goal": round(total_time, 3),
            "path_length": round(path_length, 4),
            "min_dist": round(min_dist, 4),
            "collision_count": self._collision_events,
            "viol_time": round(self._viol_time, 3),
            "samples": len(self._positions),
            "goal": {"x": self.goal_x, "y": self.goal_y, "yaw": self.goal_yaw},
            "start": {"x": self.start_x, "y": self.start_y, "yaw": self.start_yaw},
        }

        self.get_logger().info(
            f"Episode done — {json.dumps(results, indent=2)}")

        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=2)

        self.get_logger().info(f"Results saved to {self.output_file}")

        # Exit cleanly so the orchestrator knows we're done.
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkEpisode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
