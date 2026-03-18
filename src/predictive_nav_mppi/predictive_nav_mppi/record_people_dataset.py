#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import rclpy
from people_msgs.msg import People
from rclpy.node import Node


class PeopleDatasetRecorder(Node):
    def __init__(self) -> None:
        super().__init__("record_people_dataset")
        self.input_topic = str(self.declare_parameter("input_topic", "/people").value)
        self.output_path = str(self.declare_parameter("output_path", "people_dataset.json").value)
        self.max_people = int(self.declare_parameter("max_people", 200).value)
        self.frames: List[Dict[str, Any]] = []
        self.sub = self.create_subscription(People, self.input_topic, self._cb, 10)
        self.get_logger().info(f"Recording people dataset from {self.input_topic} -> {self.output_path}")

    def _person_id(self, person: Any, fallback_idx: int) -> int:
        if hasattr(person, "id"):
            return int(getattr(person, "id"))
        if hasattr(person, "person_id"):
            return int(getattr(person, "person_id"))
        return int(fallback_idx)

    def _cb(self, msg: People) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        people = []
        n = min(len(msg.people), max(0, self.max_people))
        for idx in range(n):
            p = msg.people[idx]
            people.append(
                {
                    "id": self._person_id(p, idx + 1),
                    "x": float(p.position.x),
                    "y": float(p.position.y),
                    "vx": float(getattr(getattr(p, "velocity", None), "x", 0.0)),
                    "vy": float(getattr(getattr(p, "velocity", None), "y", 0.0)),
                }
            )
        self.frames.append({"t": now, "frame_id": msg.header.frame_id, "people": people})

    def save(self) -> Path:
        out = Path(self.output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "input_topic": self.input_topic,
                "frames": len(self.frames),
                "node": "record_people_dataset",
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
