#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
from typing import Dict, List, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from people_msgs.msg import People
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from predictive_nav_mppi.kf_cv import (
    TrackState,
    clamp_dt,
    predict_state_cov,
    prune_stale_tracks,
    update_state_cov,
)


def _float_or_default(value: float, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    return float(value)


def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _cov2d_to_ellipse(
    cov_xx: float,
    cov_xy: float,
    cov_yy: float,
    n_sigma: float = 2.0,
) -> Tuple[float, float, float]:
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    tmp = max(0.0, trace * trace * 0.25 - det)
    root = math.sqrt(tmp)
    l1 = max(0.0, 0.5 * trace + root)
    l2 = max(0.0, 0.5 * trace - root)
    yaw = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    sx = max(1e-3, 2.0 * n_sigma * math.sqrt(l1))
    sy = max(1e-3, 2.0 * n_sigma * math.sqrt(l2))
    return sx, sy, yaw


class PeopleKFPredictor(Node):
    def __init__(self) -> None:
        super().__init__("people_kf_predictor")

        self.input_topic = self.declare_parameter("input_topic", "/people").value
        self.output_cloud_topic = self.declare_parameter(
            "output_cloud_topic",
            "/predicted_people_cloud",
        ).value
        self.output_markers_topic = self.declare_parameter(
            "output_markers_topic",
            "/predicted_people_markers",
        ).value
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 10.0).value)
        self.pred_dt = float(self.declare_parameter("pred_dt", 0.1).value)
        self.pred_steps = int(self.declare_parameter("pred_steps", 10).value)
        self.sigma_meas = float(self.declare_parameter("sigma_meas", 0.08).value)
        self.sigma_acc = float(self.declare_parameter("sigma_acc", 0.6).value)
        self.sigma_p0 = float(self.declare_parameter("sigma_p0", 0.2).value)
        self.sigma_v0 = float(self.declare_parameter("sigma_v0", 0.8).value)
        self.min_dt = float(self.declare_parameter("min_dt", 0.02).value)
        self.max_dt = float(self.declare_parameter("max_dt", 0.3).value)
        self.track_timeout = float(self.declare_parameter("track_timeout", 1.0).value)
        self.max_people = int(self.declare_parameter("max_people", 100).value)
        self.publish_markers = bool(self.declare_parameter("publish_markers", True).value)
        self.publish_ellipses = bool(self.declare_parameter("publish_ellipses", True).value)
        self.ellipse_steps = int(self.declare_parameter("ellipse_steps", 3).value)
        self.frame_id_override = str(self.declare_parameter("frame_id_override", "").value)

        self.tracks: Dict[int, TrackState] = {}
        self.name_to_track_id: Dict[str, int] = {}
        self.next_track_id: int = 1
        self.latest_frame_id: str = "map"

        self.sub_people = self.create_subscription(
            People,
            self.input_topic,
            self.people_callback,
            10,
        )
        self.pub_cloud = self.create_publisher(PointCloud2, self.output_cloud_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_markers_topic, 10)
        timer_period = max(0.01, 1.0 / max(1e-3, self.publish_rate_hz))
        self.timer = self.create_timer(timer_period, self.publish_prediction)

        self.get_logger().info(
            f"people_kf_predictor started: in={self.input_topic}, "
            f"cloud={self.output_cloud_topic}, markers={self.output_markers_topic}"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _init_track(self, now_sec: float, px: float, py: float, vx: float, vy: float) -> TrackState:
        sigma_p2 = self.sigma_p0 * self.sigma_p0
        sigma_v2 = self.sigma_v0 * self.sigma_v0
        sigma = [
            [sigma_p2, 0.0, 0.0, 0.0],
            [0.0, sigma_p2, 0.0, 0.0],
            [0.0, 0.0, sigma_v2, 0.0],
            [0.0, 0.0, 0.0, sigma_v2],
        ]
        return TrackState(mu=[px, py, vx, vy], sigma=sigma, last_update_sec=now_sec)

    def _person_track_id(self, person) -> int:
        # Keep compatibility with multiple message variants:
        # - custom variants with `id` or `person_id`
        # - people_msgs/Person from HuNav with string `name`
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))

        name = ""
        if hasattr(person, "name"):
            name = str(getattr(person, "name"))
        if not name:
            name = f"anon_{self.next_track_id}"

        if name not in self.name_to_track_id:
            self.name_to_track_id[name] = self.next_track_id
            self.next_track_id += 1
        return self.name_to_track_id[name]

    def people_callback(self, msg: People) -> None:
        now_sec = self._now_sec()
        if self.frame_id_override:
            self.latest_frame_id = self.frame_id_override
        elif msg.header.frame_id:
            self.latest_frame_id = msg.header.frame_id

        people_count = min(len(msg.people), max(0, self.max_people))
        for idx in range(people_count):
            person = msg.people[idx]
            person_id = self._person_track_id(person)
            px = _float_or_default(person.position.x, 0.0)
            py = _float_or_default(person.position.y, 0.0)
            vx = _float_or_default(person.velocity.x, 0.0)
            vy = _float_or_default(person.velocity.y, 0.0)

            if person_id not in self.tracks:
                self.tracks[person_id] = self._init_track(now_sec, px, py, vx, vy)
                continue

            track = self.tracks[person_id]
            dt = clamp_dt(now_sec - track.last_update_sec, self.min_dt, self.max_dt)
            mu_pred, sigma_pred = predict_state_cov(track.mu, track.sigma, dt, self.sigma_acc)
            mu_upd, sigma_upd = update_state_cov(mu_pred, sigma_pred, px, py, self.sigma_meas)
            track.mu = mu_upd
            track.sigma = sigma_upd
            track.last_update_sec = now_sec

        prune_stale_tracks(self.tracks, now_sec, self.track_timeout)

    def _predict_horizon(
        self,
        mu: List[float],
        sigma: List[List[float]],
    ) -> List[Tuple[List[float], List[List[float]]]]:
        out: List[Tuple[List[float], List[List[float]]]] = []
        mu_h = list(mu)
        sigma_h = [row[:] for row in sigma]
        for _ in range(self.pred_steps):
            mu_h, sigma_h = predict_state_cov(mu_h, sigma_h, self.pred_dt, self.sigma_acc)
            out.append((list(mu_h), [row[:] for row in sigma_h]))
        return out

    def _create_cloud_xyz32(
        self,
        header: Header,
        points_xyz: List[Tuple[float, float, float]],
    ) -> PointCloud2:
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points_xyz)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        if points_xyz:
            data = bytearray()
            for x, y, z in points_xyz:
                data.extend(struct.pack("<fff", float(x), float(y), float(z)))
            msg.data = bytes(data)
        else:
            msg.data = b""
        return msg

    def _build_markers(
        self,
        header: Header,
        horizon_by_id: Dict[int, List[Tuple[List[float], List[List[float]]]]],
    ) -> MarkerArray:
        out = MarkerArray()
        delete_all = Marker()
        delete_all.header = header
        delete_all.action = Marker.DELETEALL
        out.markers.append(delete_all)

        if not self.publish_markers:
            return out

        lifetime = Duration(sec=0, nanosec=int(2e8))  # 0.2s
        ellipse_max = max(0, min(self.ellipse_steps, self.pred_steps))

        for person_id, horizon in horizon_by_id.items():
            line = Marker()
            line.header = header
            line.ns = "predicted_people_path"
            line.id = int(person_id)
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.035
            line.color.r = 0.1
            line.color.g = 0.8
            line.color.b = 0.2
            line.color.a = 0.95
            line.lifetime = lifetime
            for mu_t, _ in horizon:
                p = Point()
                p.x = float(mu_t[0])
                p.y = float(mu_t[1])
                p.z = 0.05
                line.points.append(p)
            out.markers.append(line)

            if not self.publish_ellipses:
                continue

            for step_idx in range(ellipse_max):
                mu_t, sigma_t = horizon[step_idx]
                cov_xx = sigma_t[0][0]
                cov_xy = sigma_t[0][1]
                cov_yy = sigma_t[1][1]
                sx, sy, yaw = _cov2d_to_ellipse(cov_xx, cov_xy, cov_yy)
                qx, qy, qz, qw = _yaw_to_quat(yaw)

                ell = Marker()
                ell.header = header
                ell.ns = "predicted_people_cov"
                ell.id = int(person_id * 100 + step_idx)
                ell.type = Marker.CYLINDER
                ell.action = Marker.ADD
                ell.pose.position.x = float(mu_t[0])
                ell.pose.position.y = float(mu_t[1])
                ell.pose.position.z = 0.01
                ell.pose.orientation.x = qx
                ell.pose.orientation.y = qy
                ell.pose.orientation.z = qz
                ell.pose.orientation.w = qw
                ell.scale.x = float(sx)
                ell.scale.y = float(sy)
                ell.scale.z = 0.02
                ell.color.r = 0.2
                ell.color.g = 0.4
                ell.color.b = 1.0
                ell.color.a = 0.30
                ell.lifetime = lifetime
                out.markers.append(ell)

        return out

    def publish_prediction(self) -> None:
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        prune_stale_tracks(self.tracks, now_sec, self.track_timeout)

        horizon_by_id: Dict[int, List[Tuple[List[float], List[List[float]]]]] = {}
        points_xyz: List[Tuple[float, float, float]] = []

        for person_id, track in self.tracks.items():
            dt = now_sec - track.last_update_sec
            if dt > 0.0:
                dt = clamp_dt(dt, self.min_dt, self.max_dt)
                track.mu, track.sigma = predict_state_cov(track.mu, track.sigma, dt, self.sigma_acc)
                track.last_update_sec = now_sec

            horizon = self._predict_horizon(track.mu, track.sigma)
            horizon_by_id[person_id] = horizon
            for mu_t, _ in horizon:
                points_xyz.append((mu_t[0], mu_t[1], 0.0))

        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = self.frame_id_override if self.frame_id_override else self.latest_frame_id

        cloud_msg = self._create_cloud_xyz32(header, points_xyz)
        self.pub_cloud.publish(cloud_msg)
        markers_msg = self._build_markers(header, horizon_by_id)
        self.pub_markers.publish(markers_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PeopleKFPredictor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
