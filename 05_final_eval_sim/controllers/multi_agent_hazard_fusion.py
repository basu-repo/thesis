"""Fuse UGV, UAV1, and UAV2 hazard observations into one navigation hint.

The fusion node is intentionally independent from the AI model. The trained
trajectory model remains unchanged, while this node provides a safety/context
signal that can adjust the selected waypoint during live simulation.
"""

import json
import math
import time

from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import String


class MultiAgentHazardFusion(Node):
    """Combine local UGV obstacle detection with UAV hazard messages.

    Design:
    - UGV LiDAR has highest priority for near obstacles.
    - UAV messages are used as early-warning signals.
    - Stale UAV messages are ignored, so the UGV can continue autonomously
      during temporary communication loss.
    """

    def __init__(
        self,
        node_name: str = "multi_agent_hazard_fusion",
        ugv_action_topic: str = "/husky_local/obstacle_action",
        ugv_clearance_topic: str = "/husky_local/obstacle_clearance",
        uav1_topic: str = "/uav1/hazard_hint_raw",
        uav2_topic: str = "/uav2/hazard_hint_raw",
        output_topic: str = "/fused_hazard_hint",
        publish_period: float = 0.1,
        uav_timeout: float = 1.0,
        ugv_emergency_distance: float = 1.4,
        ugv_caution_distance: float = 2.5,
    ):
        super().__init__(node_name)

        self.uav_timeout = uav_timeout
        self.ugv_emergency_distance = ugv_emergency_distance
        self.ugv_caution_distance = ugv_caution_distance

        self.ugv_action = "clear"
        self.ugv_clearance = (float("inf"), float("inf"), float("inf"))

        self.uav_messages = {
            "uav1": None,
            "uav2": None,
        }
        self.uav_received_time = {
            "uav1": 0.0,
            "uav2": 0.0,
        }

        self.pub = self.create_publisher(String, output_topic, 10)

        self.create_subscription(String, ugv_action_topic, self.ugv_action_cb, 10)
        self.create_subscription(Vector3, ugv_clearance_topic, self.ugv_clearance_cb, 10)
        self.create_subscription(String, uav1_topic, self.make_uav_cb("uav1"), 10)
        self.create_subscription(String, uav2_topic, self.make_uav_cb("uav2"), 10)

        self.timer = self.create_timer(publish_period, self.publish_fused_hazard)

    def ugv_action_cb(self, msg: String):
        self.ugv_action = msg.data.strip().lower() if msg.data else "clear"

    def ugv_clearance_cb(self, msg: Vector3):
        # x = front, y = left, z = right
        self.ugv_clearance = (float(msg.x), float(msg.y), float(msg.z))

    def make_uav_cb(self, source: str):
        def cb(msg: String):
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            self.uav_messages[source] = payload
            self.uav_received_time[source] = time.monotonic()

        return cb

    def _fresh_uav_messages(self):
        now = time.monotonic()
        fresh = {}
        for source, payload in self.uav_messages.items():
            if payload is None:
                continue
            age = now - self.uav_received_time[source]
            if age <= self.uav_timeout:
                fresh[source] = payload
        return fresh
    

    def _uav_status_summary(self, source: str, fresh_uavs: dict) -> dict:
        """Return diagnostic status for one UAV hazard source.

        This makes the fused hazard output easier to debug and also shows
        whether a UAV is connected, stale, or currently reporting an obstacle.
        """
        payload = self.uav_messages.get(source)
        received_time = self.uav_received_time.get(source, 0.0)

        if payload is None or received_time <= 0.0:
            return {
                "connection": "never_received",
                "age": None,
                "last_blocked": None,
                "last_confidence": None,
                "last_distance_ahead": None,
                "accepted_points": None,
                "total_points_read": None,
            }

        age = time.monotonic() - received_time
        connection = "fresh" if source in fresh_uavs else "stale_or_disconnected"

        return {
            "connection": connection,
            "age": age,
            "last_blocked": bool(payload.get("blocked", False)),
            "last_confidence": payload.get("confidence"),
            "last_distance_ahead": payload.get("distance_ahead"),
            "accepted_points": payload.get("accepted_points"),
            "total_points_read": payload.get("total_points_read"),
        }
    


    def _ugv_turn_bias(self) -> float:
        front, left, right = self.ugv_clearance
        if left > right:
            return 1.0
        if right > left:
            return -1.0
        return 0.0

    def publish_fused_hazard(self):
        front, left, right = self.ugv_clearance
        fresh_uavs = self._fresh_uav_messages()

        sources = []
        weighted_bias = 0.0
        total_weight = 0.0
        confidence = 0.0
        distance_candidates = []

        # UGV local LiDAR always has priority for near obstacles.
        ugv_blocked = self.ugv_action != "clear" or front < self.ugv_caution_distance
        if ugv_blocked:
            sources.append("ugv")
            ugv_weight = 0.65 if front < self.ugv_emergency_distance else 0.45
            weighted_bias += ugv_weight * self._ugv_turn_bias()
            total_weight += ugv_weight
            confidence += ugv_weight
            distance_candidates.append(front)

        # UAVs are early-warning sources. They are ignored if stale.
        for source, payload in fresh_uavs.items():
            if not payload.get("blocked", False):
                continue

            sources.append(source)
            uav_conf = float(payload.get("confidence", 0.5))
            uav_weight = 0.25 * max(0.2, min(uav_conf, 1.0))
            weighted_bias += uav_weight * float(payload.get("turn_bias", 0.0))
            total_weight += uav_weight
            confidence += uav_weight

            dist = payload.get("distance_ahead")
            if dist is not None:
                try:
                    distance_candidates.append(float(dist))
                except (TypeError, ValueError):
                    pass

        if total_weight > 0:
            turn_bias = weighted_bias / total_weight
        else:
            turn_bias = 0.0

        if turn_bias > 0.15:
            turn_direction = "left"
            turn_bias_sign = 1.0
        elif turn_bias < -0.15:
            turn_direction = "right"
            turn_bias_sign = -1.0
        else:
            turn_direction = "unknown"
            turn_bias_sign = 0.0

        blocked = len(sources) > 0
        if "ugv" in sources and front < self.ugv_emergency_distance:
            mode = "emergency_local"
        elif len(sources) >= 2:
            mode = "confirmed_multi_agent"
        elif len(sources) == 1:
            mode = "caution_single_source"
        else:
            mode = "clear"

        uav_status = {
            "uav1": self._uav_status_summary("uav1", fresh_uavs),
            "uav2": self._uav_status_summary("uav2", fresh_uavs),
        }

        connection_state = {
            "uav1": uav_status["uav1"]["connection"],
            "uav2": uav_status["uav2"]["connection"],
        }

        fused = {
            "blocked": blocked,
            "mode": mode,
            "sources": sources,
            "confidence": min(confidence, 1.0),
            "turn_bias": turn_bias_sign,
            "turn_direction": turn_direction,
            "distance_ahead": min(distance_candidates) if distance_candidates else None,
            "connection_state": connection_state,
            "uav_status": uav_status,
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
        }

        msg = String()
        msg.data = json.dumps(fused)
        self.pub.publish(msg)


# UGV LiDAR has highest priority for near obstacles.
# UAV messages are early-warning signals.
# Stale UAV messages are ignored so the UGV can continue during communication loss.