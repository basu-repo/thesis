"""Scout readiness and preview fusion for UAV-to-UGV sharing."""

import math

from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import Bool, String


class ScoutCoordinatorNode(Node):
    """Aggregate scout readiness and preview upcoming obstacles for the Husky."""

    def __init__(
        self,
        node_name: str,
        scout_report_topics: list[str],
        scout_ready_topics: list[str],
        husky_obstacle_action_topic: str,
        husky_obstacle_clearance_topic: str,
        scouts_ready_topic: str,
        summary_topic: str,
        min_ready_count: int = 1,
    ):
        super().__init__(node_name)
        self.scout_names = [f"scout_{idx + 1}" for idx in range(len(scout_report_topics))]
        self.scout_reports = {
            name: {"distance": 999.0, "lateral": 0.0, "blocked": False}
            for name in self.scout_names
        }
        self.scout_ready = {name: False for name in self.scout_names}
        self.husky_obstacle_action = "clear"
        self.husky_front_clearance = float("inf")
        self.last_ready_state = None
        self.last_summary = None
        self.last_log_time = 0.0
        self.min_ready_count = max(1, int(min_ready_count))

        self.ready_pub = self.create_publisher(Bool, scouts_ready_topic, 10)
        self.summary_pub = self.create_publisher(String, summary_topic, 10)

        for idx, topic in enumerate(scout_report_topics):
            self.create_subscription(Vector3, topic, self._make_report_cb(self.scout_names[idx]), 10)
        for idx, topic in enumerate(scout_ready_topics):
            self.create_subscription(Bool, topic, self._make_ready_cb(self.scout_names[idx]), 10)
        self.create_subscription(String, husky_obstacle_action_topic, self.husky_obstacle_action_cb, 10)
        self.create_subscription(Vector3, husky_obstacle_clearance_topic, self.husky_obstacle_clearance_cb, 10)
        self.timer = self.create_timer(0.2, self.publish_updates)

        self.get_logger().info("Scout coordinator started.")

    def _make_report_cb(self, name: str):
        def cb(msg: Vector3):
            self.scout_reports[name] = {
                "distance": float(msg.x),
                "lateral": float(msg.y),
                "blocked": bool(msg.z >= 0.5 and math.isfinite(float(msg.x)) and float(msg.x) < 998.0),
            }

        return cb

    def _make_ready_cb(self, name: str):
        def cb(msg: Bool):
            self.scout_ready[name] = bool(msg.data)

        return cb

    def husky_obstacle_action_cb(self, msg: String):
        self.husky_obstacle_action = msg.data.strip().lower() if msg.data else "clear"

    def husky_obstacle_clearance_cb(self, msg: Vector3):
        self.husky_front_clearance = float(msg.x)

    def publish_updates(self):
        ready_count = sum(1 for ready in self.scout_ready.values() if ready)
        ready_state = ready_count >= self.min_ready_count if self.scout_ready else True
        ready_msg = Bool()
        ready_msg.data = ready_state
        self.ready_pub.publish(ready_msg)

        nearest_name = None
        nearest_report = None
        for name, report in self.scout_reports.items():
            if not report["blocked"]:
                continue
            if nearest_report is None or report["distance"] < nearest_report["distance"]:
                nearest_name = name
                nearest_report = report

        if nearest_report is None:
            summary = "clear"
        else:
            local_detected = (self.husky_obstacle_action or "clear") != "clear" or self.husky_front_clearance < 998.0
            state = "verified" if local_detected else "preview"
            summary = (
                f"{state} source={nearest_name} distance={nearest_report['distance']:.2f} "
                f"lateral={nearest_report['lateral']:.2f} local_front={self.husky_front_clearance:.2f}"
            )

        msg = String()
        msg.data = summary
        self.summary_pub.publish(msg)

        now = self.get_clock().now().nanoseconds / 1e9
        if (
            summary != self.last_summary
            or ready_state != self.last_ready_state
            or (now - self.last_log_time) >= 3.0
        ):
            self.get_logger().info(
                f"scout_coordinator ready={ready_state} ready_count={ready_count}/{len(self.scout_ready)} summary={summary}"
            )
            self.last_summary = summary
            self.last_ready_state = ready_state
            self.last_log_time = now
