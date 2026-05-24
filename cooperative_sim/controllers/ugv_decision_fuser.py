"""Fuse UGV perception inputs into one final controller-facing decision."""

from __future__ import annotations

import math

from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import String


class UgvDecisionFuser(Node):
    """Publish one final front-decision for the Husky controller."""

    def __init__(
        self,
        node_name: str,
        obstacle_action_topic: str,
        obstacle_clearance_topic: str,
        hazard_guidance_topic: str,
        depth_classification_topic: str,
        decision_topic: str,
        hard_block_distance: float = 0.8,
    ):
        super().__init__(node_name)
        self.hard_block_distance = float(hard_block_distance)
        self.obstacle_action = "clear"
        self.front_clearance = float("inf")
        self.hazard_guidance = "clear"
        self.depth_overall = "unknown"
        self.last_decision = None
        self.last_log_time = 0.0

        self.pub = self.create_publisher(String, decision_topic, 10)
        self.create_subscription(String, obstacle_action_topic, self.obstacle_action_cb, 10)
        self.create_subscription(Vector3, obstacle_clearance_topic, self.obstacle_clearance_cb, 10)
        self.create_subscription(String, hazard_guidance_topic, self.hazard_guidance_cb, 10)
        self.create_subscription(String, depth_classification_topic, self.depth_classification_cb, 10)

        self.timer = self.create_timer(0.1, self.step)
        self.get_logger().info(f"UGV decision fuser publishing on {decision_topic}")

    def obstacle_action_cb(self, msg: String):
        self.obstacle_action = msg.data.strip().lower() if msg.data else "clear"

    def obstacle_clearance_cb(self, msg: Vector3):
        self.front_clearance = float(msg.x)

    def hazard_guidance_cb(self, msg: String):
        self.hazard_guidance = msg.data.strip().lower() if msg.data else "clear"

    def depth_classification_cb(self, msg: String):
        text = msg.data.strip().lower() if msg.data else ""
        overall = "unknown"
        for token in text.split():
            if token.startswith("overall="):
                overall = token.split("=", 1)[1]
                break
        self.depth_overall = overall

    def _local_block_label(self) -> str | None:
        action = self.obstacle_action or "clear"
        if action == "clear":
            return None
        if action.endswith("left"):
            return "block_left"
        if action.endswith("right"):
            return "block_right"
        return "block_center"

    def _hazard_label(self) -> str | None:
        guidance = self.hazard_guidance or "clear"
        if guidance.startswith("terrain_sure_front"):
            return "terrain"
        # Scout/hazard input is treated only as terrain confirmation.
        # Block-side decisions must come from local UGV sensing and depth cues.
        return None

    def _depth_label(self) -> str | None:
        overall = self.depth_overall or "unknown"
        if overall == "terrain":
            return "terrain"
        if overall.startswith("block"):
            return overall
        return None

    def _choose_block_side(self, labels: list[str]) -> str:
        for preferred in ("block_left", "block_right", "block_center"):
            if labels.count(preferred) >= 2:
                return preferred
        for preferred in ("block_left", "block_right", "block_center"):
            if preferred in labels:
                return preferred
        return "block_center"

    def _final_decision(self) -> str:
        local_label = self._local_block_label()
        hazard_label = self._hazard_label()
        depth_label = self._depth_label()

        if local_label is not None and math.isfinite(self.front_clearance) and self.front_clearance <= self.hard_block_distance:
            return local_label

        if hazard_label == "terrain" and (self.hazard_guidance or "").startswith("terrain_sure_front"):
            return "terrain"

        terrain_votes = 0
        block_labels: list[str] = []

        if local_label is None:
            terrain_votes += 1
        else:
            block_labels.append(local_label)

        if hazard_label == "terrain":
            terrain_votes += 1
        elif hazard_label is not None:
            block_labels.append(hazard_label)

        if depth_label == "terrain":
            terrain_votes += 1
        elif depth_label is not None:
            block_labels.append(depth_label)

        block_votes = len(block_labels)
        if terrain_votes >= 2 and terrain_votes > block_votes:
            return "terrain"
        if block_votes >= 2 and block_votes > terrain_votes:
            return self._choose_block_side(block_labels)
        if local_label is None:
            return "clear"
        return local_label

    def step(self):
        decision = self._final_decision()
        msg = String()
        msg.data = decision
        self.pub.publish(msg)

        now = self.get_clock().now().nanoseconds / 1e9
        if decision != self.last_decision or (now - self.last_log_time) >= 2.0:
            self.get_logger().info(
                "UGV decision: "
                f"final={decision} local={self.obstacle_action} front={self.front_clearance:.2f} "
                f"hazard={self.hazard_guidance} depth={self.depth_overall}"
            )
            self.last_decision = decision
            self.last_log_time = now
