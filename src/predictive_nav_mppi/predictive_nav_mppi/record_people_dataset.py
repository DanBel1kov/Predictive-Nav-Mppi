#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from people_msgs.msg import People
from rclpy.node import Node


def _quat_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(math.atan2(siny, cosy))


class PeopleDatasetRecorder(Node):
    def __init__(self) -> None:
        super().__init__("record_people_dataset")
        self.input_topic = str(self.declare_parameter("input_topic", "/people").value)
        self.output_path = str(self.declare_parameter("output_path", "people_dataset.json").value)
        self.max_people = int(self.declare_parameter("max_people", 200).value)

        self.robot_pose_topic = str(self.declare_parameter("robot_pose_topic", "/amcl_pose").value)
        self.robot_odom_topic = str(self.declare_parameter("robot_odom_topic", "/odom").value)
        self.visibility_radius = float(self.declare_parameter("visibility_radius", 15.0).value)
        self.visibility_fov_deg = float(self.declare_parameter("visibility_fov_deg", 360.0).value)
        self.robot_force_scale = float(self.declare_parameter("robot_force_scale", 1.0).value)
        self.map_name = str(self.declare_parameter("map_name", "").value)
        self.require_robot_pose = bool(self.declare_parameter("require_robot_pose", True).value)

        self.frames: List[Dict[str, Any]] = []
        self.robot_state: Optional[Dict[str, float]] = None
        self._prev_pose: Optional[Dict[str, float]] = None

        self.sub = self.create_subscription(People, self.input_topic, self._cb, 10)
        self.robot_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, self.robot_pose_topic, self._robot_pose_cb, 10
        )
        self.robot_odom_sub = self.create_subscription(
            Odometry, self.robot_odom_topic, self._robot_odom_cb, 20
        )

        self.get_logger().info(
            f"Recording people dataset from {self.input_topic} -> {self.output_path} | "
            f"robot_pose={self.robot_pose_topic} odom={self.robot_odom_topic} "
            f"visibility_radius={self.visibility_radius} fov_deg={self.visibility_fov_deg} "
            f"robot_force_scale={self.robot_force_scale} map={self.map_name}"
        )

    def _person_id(self, person: Any, fallback_idx: int) -> int:
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))
        return int(fallback_idx)

    def _robot_pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        pose = msg.pose.pose
        yaw = _quat_to_yaw(pose.orientation)
        x = float(pose.position.x)
        y = float(pose.position.y)
        if self._prev_pose is not None:
            dt = max(1e-3, now - self._prev_pose["t"])
            vx = (x - self._prev_pose["x"]) / dt
            vy = (y - self._prev_pose["y"]) / dt
        else:
            vx = vy = 0.0
        self._prev_pose = {"x": x, "y": y, "t": now}
        # Only set if we don't have odom-derived state (odom is more accurate for vel)
        if self.robot_state is None or self.robot_state.get("source") != "odom":
            self.robot_state = {
                "x": x, "y": y, "yaw": yaw, "vx": vx, "vy": vy, "t": now, "source": "amcl",
            }
        else:
            # update map-frame xy from amcl, keep velocity from odom
            self.robot_state["x"] = x
            self.robot_state["y"] = y
            self.robot_state["yaw"] = yaw
            self.robot_state["t"] = now

    def _robot_odom_cb(self, msg: Odometry) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        # /odom is in odom frame; we use it only to derive velocity in map frame
        # by combining body-frame twist with map-frame yaw if available.
        if self.robot_state is None:
            return
        yaw = float(self.robot_state["yaw"])
        v_body_x = float(msg.twist.twist.linear.x)
        v_body_y = float(msg.twist.twist.linear.y)
        vx = v_body_x * math.cos(yaw) - v_body_y * math.sin(yaw)
        vy = v_body_x * math.sin(yaw) + v_body_y * math.cos(yaw)
        self.robot_state["vx"] = vx
        self.robot_state["vy"] = vy
        self.robot_state["t"] = now
        self.robot_state["source"] = "odom"

    def _person_visible(self, px: float, py: float, rs: Dict[str, float]) -> bool:
        dx = px - rs["x"]
        dy = py - rs["y"]
        if dx * dx + dy * dy > self.visibility_radius * self.visibility_radius:
            return False
        if self.visibility_fov_deg >= 360.0:
            return True
        bearing = math.atan2(dy, dx) - rs["yaw"]
        bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
        return abs(bearing) <= math.radians(self.visibility_fov_deg / 2.0)

    def _cb(self, msg: People) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        rs = self.robot_state
        if rs is None and self.require_robot_pose:
            return
        people = []
        n = min(len(msg.people), max(0, self.max_people))
        for idx in range(n):
            p = msg.people[idx]
            px = float(p.position.x)
            py = float(p.position.y)
            if rs is not None and not self._person_visible(px, py, rs):
                continue
            people.append(
                {
                    "id": self._person_id(p, idx + 1),
                    "x": px,
                    "y": py,
                    "vx": float(getattr(getattr(p, "velocity", None), "x", 0.0)),
                    "vy": float(getattr(getattr(p, "velocity", None), "y", 0.0)),
                }
            )
        frame: Dict[str, Any] = {
            "t": now,
            "frame_id": msg.header.frame_id,
            "people": people,
        }
        if rs is not None:
            frame["robot"] = {
                "x": rs["x"], "y": rs["y"], "yaw": rs["yaw"],
                "vx": rs["vx"], "vy": rs["vy"], "t": rs["t"],
            }
        self.frames.append(frame)

    def save(self) -> Path:
        out = Path(self.output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "input_topic": self.input_topic,
                "frames": len(self.frames),
                "node": "record_people_dataset",
                "robot_pose_topic": self.robot_pose_topic,
                "robot_odom_topic": self.robot_odom_topic,
                "visibility_radius": self.visibility_radius,
                "visibility_fov_deg": self.visibility_fov_deg,
                "robot_force_scale": self.robot_force_scale,
                "map_name": self.map_name,
            },
            "frames": self.frames,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        self.get_logger().info(f"Saved dataset: {out}")
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PeopleDatasetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
