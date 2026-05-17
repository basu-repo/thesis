"""Front-pointcloud obstacle detection for an independent UAV."""

import math

from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String


class UavObstacleDetectionNode(Node):
    """Classify nearby forward obstacles for simple aerial avoidance."""

    def __init__(
        self,
        node_name: str,
        pointcloud_topic: str,
        action_topic: str,
        clearance_topic: str,
        min_forward_x: float = 0.8,
        max_forward_x: float = 8.0,
        front_half_width_y: float = 1.2,
        side_width_y: float = 4.0,
        front_min_z: float = -0.25,
        front_max_z: float = 1.2,
        up_min_z: float = 1.0,
        up_max_z: float = 4.0,
        stop_distance: float = 2.0,
        caution_distance: float = 3.5,
        min_points: int = 5,
        direction_switch_margin: float = 0.5,
    ):
        super().__init__(node_name)
        self.pointcloud_topic = pointcloud_topic
        self.action_topic = action_topic
        self.clearance_topic = clearance_topic
        self.min_forward_x = min_forward_x
        self.max_forward_x = max_forward_x
        self.front_half_width_y = front_half_width_y
        self.side_width_y = side_width_y
        self.front_min_z = front_min_z
        self.front_max_z = front_max_z
        self.up_min_z = up_min_z
        self.up_max_z = up_max_z
        self.stop_distance = stop_distance
        self.caution_distance = caution_distance
        self.min_points = min_points
        self.direction_switch_margin = direction_switch_margin

        self.action_pub = self.create_publisher(String, self.action_topic, 10)
        self.clearance_pub = self.create_publisher(Vector3, self.clearance_topic, 10)
        self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_cb,
            qos_profile_sensor_data,
        )

        self.last_action = None
        self.last_log_time = 0.0
        self.avoid_direction = None

        self.get_logger().info(
            f"UAV obstacle detector listening on {self.pointcloud_topic}, publishing to {self.action_topic}"
        )

    def _choose_direction(self, left_min: float, right_min: float, up_min: float) -> str:
        candidates = {
            "left": left_min,
            "right": right_min,
            "up": up_min,
        }
        preferred = max(candidates, key=candidates.get)
        if self.avoid_direction is None:
            self.avoid_direction = preferred
            return preferred
        current = candidates.get(self.avoid_direction, float("-inf"))
        if candidates[preferred] > current + self.direction_switch_margin:
            self.avoid_direction = preferred
        return self.avoid_direction

    def pointcloud_cb(self, msg: PointCloud2):
        front_min = float("inf")
        left_min = float("inf")
        right_min = float("inf")
        up_min = float("inf")
        front_count = 0
        left_count = 0
        right_count = 0
        up_count = 0

        for x, y, z in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            x = float(x)
            y = float(y)
            z = float(z)
            if x < self.min_forward_x or x > self.max_forward_x:
                continue

            abs_y = abs(y)
            if abs_y <= self.front_half_width_y and self.front_min_z <= z <= self.front_max_z:
                front_min = min(front_min, x)
                front_count += 1

            if self.front_half_width_y < y <= self.side_width_y and self.front_min_z <= z <= self.front_max_z:
                left_min = min(left_min, x)
                left_count += 1
            elif -self.side_width_y <= y < -self.front_half_width_y and self.front_min_z <= z <= self.front_max_z:
                right_min = min(right_min, x)
                right_count += 1

            if self.up_min_z <= z <= self.up_max_z and abs_y <= self.side_width_y:
                up_min = min(up_min, x)
                up_count += 1

        if front_count < self.min_points:
            front_min = float("inf")
        if left_count < self.min_points:
            left_min = float("inf")
        if right_count < self.min_points:
            right_min = float("inf")
        if up_count < self.min_points:
            up_min = float("inf")

        clearances = Vector3()
        clearances.x = front_min if math.isfinite(front_min) else 999.0
        clearances.y = left_min if math.isfinite(left_min) else 999.0
        clearances.z = right_min if math.isfinite(right_min) else 999.0
        self.clearance_pub.publish(clearances)

        action = "clear"
        if math.isfinite(front_min) and front_min <= self.stop_distance:
            action = f"turn_{self._choose_direction(left_min, right_min, up_min)}"
        elif math.isfinite(front_min) and front_min <= self.caution_distance:
            action = f"caution_{self._choose_direction(left_min, right_min, up_min)}"
        else:
            self.avoid_direction = None

        msg_out = String()
        msg_out.data = action
        self.action_pub.publish(msg_out)

        now = self.get_clock().now().nanoseconds / 1e9
        if action != self.last_action or (now - self.last_log_time) >= 2.0:
            self.get_logger().info(
                "UAV obstacle status: "
                f"action={action} front={clearances.x:.2f} "
                f"left={clearances.y:.2f} right={clearances.z:.2f} "
                f"up={(up_min if math.isfinite(up_min) else 999.0):.2f} "
                f"direction={self.avoid_direction or 'none'}"
            )
            self.last_action = action
            self.last_log_time = now
