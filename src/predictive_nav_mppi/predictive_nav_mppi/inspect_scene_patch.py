from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
from typing import Optional
import zlib

import numpy as np
import rclpy
from people_msgs.msg import People
from rclpy.node import Node

from predictive_nav_mppi.scene_context import (
    ScenePatchConfig,
    extract_scene_patch,
    load_occupancy_scene_map,
)


def _save_patch_pgm(path: Path, patch: np.ndarray) -> None:
    data = np.asarray(patch, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    data = np.clip(data, 0.0, 1.0)
    img = (255.0 * (1.0 - data)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def _save_patch_png(path: Path, patch: np.ndarray) -> None:
    data = np.asarray(patch, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    data = np.clip(data, 0.0, 1.0)
    img = (255.0 * (1.0 - data)).astype(np.uint8)
    height, width = img.shape
    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _save_patch(path: Path, patch: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix == ".png":
        _save_patch_png(path, patch)
        return
    _save_patch_pgm(path, patch)


class ScenePatchInspector(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("inspect_scene_patch")
        self.args = args
        self._done = False
        self._scene_map = load_occupancy_scene_map(args.map_yaml)
        self._cfg = ScenePatchConfig(
            size_m=float(args.size_m),
            pixels=int(args.pixels),
            align_to_heading=bool(args.align_to_heading),
        )
        self._sub = self.create_subscription(People, args.people_topic, self._on_people, 10)
        self._timer = self.create_timer(float(args.timeout_sec), self._on_timeout)
        self.get_logger().info(
            f"Waiting for one message on {args.people_topic}; map={args.map_yaml}, "
            f"person_index={args.person_index}"
        )

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self.destroy_node()

    def _on_timeout(self) -> None:
        self.get_logger().error("Timed out waiting for /people")
        self._finish()

    def _on_people(self, msg: People) -> None:
        if self._done:
            return
        if not msg.people:
            self.get_logger().warn("Received /people, but list is empty")
            return
        idx = int(self.args.person_index)
        if idx < 0 or idx >= len(msg.people):
            self.get_logger().error(f"person_index={idx} out of range, msg has {len(msg.people)} people")
            self._finish()
            return

        person = msg.people[idx]
        px = float(person.position.x)
        py = float(person.position.y)
        vx = float(getattr(getattr(person, "velocity", None), "x", 0.0))
        vy = float(getattr(getattr(person, "velocity", None), "y", 0.0))
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx) if speed > 1e-4 else 0.0

        patch = extract_scene_patch(
            scene_map=self._scene_map,
            center_xy=np.asarray([px, py], dtype=np.float32),
            heading_rad=heading,
            cfg=self._cfg,
        )
        out = Path(self.args.output)
        _save_patch(out, patch)
        self.get_logger().info(
            f"Saved patch to {out} for person_index={idx} "
            f"at x={px:.3f}, y={py:.3f}, heading={heading:.3f} rad"
        )
        self._finish()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save the current scene patch used by residual predictor.")
    parser.add_argument("--map-yaml", required=True, help="Path to map yaml used by people_predictor.")
    parser.add_argument("--people-topic", default="/people")
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument("--size-m", type=float, default=6.0)
    parser.add_argument("--pixels", type=int, default=32)
    parser.add_argument("--align-to-heading", action="store_true", default=True)
    parser.add_argument("--disable-align-to-heading", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default=str(Path.cwd() / "debug" / "live_scene_patch_person0.png"),
        help="Output .png or .pgm path",
    )
    args = parser.parse_args()
    if args.disable_align_to_heading:
        args.align_to_heading = False
    return args


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node: Optional[ScenePatchInspector] = None
    try:
        node = ScenePatchInspector(args)
        while rclpy.ok() and node is not None and not node._done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        if node is not None and not node._done:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
