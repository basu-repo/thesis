"""Planar-lidar obstacle detection for the live Husky simulation."""

import math

from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ObstacleDetectionNode(Node):
    """Read a planar laser scan and publish simple avoidance decisions."""

    def __init__(
        self,
        node_name: str,
        scan_topic: str,
        action_topic: str,
        clearance_topic: str,
        pointcloud_topic: str | None = None,
        front_half_angle_deg: float = 30.0,
        side_angle_deg: float = 90.0,
        stop_distance: float = 1.8,
        caution_distance: float = 3.0,
        min_valid_range: float = 0.08,
        turn_commit_seconds: float = 1.2,
        direction_switch_margin: float = 0.4,
        pointcloud_timeout: float = 0.75,
        pointcloud_min_forward_x: float = 0.35,
        pointcloud_max_forward_x: float = 3.5,
        pointcloud_front_half_width_y: float = 0.7,
        pointcloud_side_width_y: float = 2.0,
        pointcloud_obstacle_min_z: float = -0.20,
        pointcloud_obstacle_max_z: float = 0.45,
        pointcloud_min_points: int = 6,
        terrain_bin_size_x: float = 0.45,
        terrain_bin_min_points: int = 4,
        terrain_min_profile_bins: int = 3,
        terrain_max_step_z: float = 0.65,
        terrain_max_slope_deg: float = 45.0,
    ):
        super().__init__(node_name)
        self.scan_topic = scan_topic
        self.action_topic = action_topic
        self.clearance_topic = clearance_topic
        self.pointcloud_topic = pointcloud_topic
        self.front_half_angle = math.radians(front_half_angle_deg)
        self.side_angle = math.radians(side_angle_deg)
        self.stop_distance = stop_distance
        self.caution_distance = caution_distance
        self.min_valid_range = min_valid_range
        self.turn_commit_seconds = turn_commit_seconds
        self.direction_switch_margin = direction_switch_margin
        self.pointcloud_timeout = pointcloud_timeout
        self.pointcloud_min_forward_x = pointcloud_min_forward_x
        self.pointcloud_max_forward_x = pointcloud_max_forward_x
        self.pointcloud_front_half_width_y = pointcloud_front_half_width_y
        self.pointcloud_side_width_y = pointcloud_side_width_y
        self.pointcloud_obstacle_min_z = pointcloud_obstacle_min_z
        self.pointcloud_obstacle_max_z = pointcloud_obstacle_max_z
        self.pointcloud_min_points = pointcloud_min_points
        self.terrain_bin_size_x = terrain_bin_size_x
        self.terrain_bin_min_points = terrain_bin_min_points
        self.terrain_min_profile_bins = terrain_min_profile_bins
        self.terrain_max_step_z = terrain_max_step_z
        self.terrain_max_slope = math.tan(math.radians(terrain_max_slope_deg))

        self.action_pub = self.create_publisher(String, self.action_topic, 10)
        self.clearance_pub = self.create_publisher(Vector3, self.clearance_topic, 10)
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_cb,
            qos_profile_sensor_data,
        )
        if self.pointcloud_topic is not None:
            self.create_subscription(
                PointCloud2,
                self.pointcloud_topic,
                self.pointcloud_cb,
                qos_profile_sensor_data,
            )

        self.last_action = None
        self.last_log_time = 0.0
        self.latest_pointcloud = None
        self.latest_pointcloud_time = None
        self.avoid_direction = None
        self.direction_locked_until = 0.0

        self.get_logger().info(
            f"Obstacle detector listening on {self.scan_topic}, publishing to {self.action_topic}"
        )
        if self.pointcloud_topic is not None:
            self.get_logger().info(
                f"Ground obstacle checks enabled from {self.pointcloud_topic}"
            )

    def _sector_min(self, msg: LaserScan, angle_min: float, angle_max: float) -> float:
        best = float("inf")
        angle = msg.angle_min
        for distance in msg.ranges:
            if angle_min <= wrap_angle(angle) <= angle_max:
                if math.isfinite(distance) and distance >= self.min_valid_range:
                    best = min(best, float(distance))
            angle += msg.angle_increment
        return best

    def pointcloud_cb(self, msg: PointCloud2):
        self.latest_pointcloud = msg
        self.latest_pointcloud_time = self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return float(ordered[mid])
        return 0.5 * float(ordered[mid - 1] + ordered[mid])

    def _classify_front_terrain(self, front_points: list[tuple[float, float]]) -> tuple[float, bool]:
        """Return (obstacle distance, climbable flag) for the front terrain profile."""

        if len(front_points) < self.pointcloud_min_points:
            return (float("inf"), False)

        bins: dict[int, list[tuple[float, float]]] = {}
        for x, z in front_points:
            bin_index = int((x - self.pointcloud_min_forward_x) / max(self.terrain_bin_size_x, 1e-3))
            bins.setdefault(bin_index, []).append((x, z))

        profile: list[tuple[float, float]] = []
        for _, samples in sorted(bins.items()):
            if len(samples) < self.terrain_bin_min_points:
                continue
            xs = [sample[0] for sample in samples]
            zs = [sample[1] for sample in samples]
            profile.append((self._median(xs), self._median(zs)))

        if len(profile) < self.terrain_min_profile_bins:
            return (min(x for x, _ in front_points), False)

        for idx in range(1, len(profile)):
            prev_x, prev_z = profile[idx - 1]
            curr_x, curr_z = profile[idx]
            dx = max(curr_x - prev_x, 1e-3)
            dz = curr_z - prev_z
            if dz <= 0.0:
                continue
            if dz > self.terrain_max_step_z:
                return (curr_x, False)
            if (dz / dx) > self.terrain_max_slope:
                return (curr_x, False)

        return (float("inf"), True)

    def _pointcloud_clearances(self, now: float) -> tuple[float, float, float, bool]:
        if self.latest_pointcloud is None or self.latest_pointcloud_time is None:
            return (float("inf"), float("inf"), float("inf"), False)
        if (now - self.latest_pointcloud_time) > self.pointcloud_timeout:
            return (float("inf"), float("inf"), float("inf"), False)

        front_points: list[tuple[float, float]] = []
        left_min = float("inf")
        right_min = float("inf")
        left_count = 0
        right_count = 0

        for x, y, z in point_cloud2.read_points(
            self.latest_pointcloud,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            x = float(x)
            y = float(y)
            z = float(z)
            if x < self.pointcloud_min_forward_x or x > self.pointcloud_max_forward_x:
                continue
            if z < self.pointcloud_obstacle_min_z or z > self.pointcloud_obstacle_max_z:
                continue

            abs_y = abs(y)
            if abs_y <= self.pointcloud_front_half_width_y:
                front_points.append((x, z))
            elif y > self.pointcloud_front_half_width_y and y <= self.pointcloud_side_width_y:
                left_min = min(left_min, x)
                left_count += 1
            elif y < -self.pointcloud_front_half_width_y and abs_y <= self.pointcloud_side_width_y:
                right_min = min(right_min, x)
                right_count += 1

        front_min, front_climbable = self._classify_front_terrain(front_points)
        if left_count < self.pointcloud_min_points:
            left_min = float("inf")
        if right_count < self.pointcloud_min_points:
            right_min = float("inf")
        return (front_min, left_min, right_min, front_climbable)

    def _choose_direction(self, left_min: float, right_min: float, now: float) -> str:
        preferred = "left" if left_min >= right_min else "right"
        if self.avoid_direction is None:
            self.avoid_direction = preferred
            self.direction_locked_until = now + self.turn_commit_seconds
            return self.avoid_direction

        if now < self.direction_locked_until:
            return self.avoid_direction

        if self.avoid_direction == "left":
            current_clearance = left_min
            alternate_clearance = right_min
            alternate_direction = "right"
        else:
            current_clearance = right_min
            alternate_clearance = left_min
            alternate_direction = "left"

        if alternate_clearance > current_clearance + self.direction_switch_margin:
            self.avoid_direction = alternate_direction
            self.direction_locked_until = now + self.turn_commit_seconds

        return self.avoid_direction

    def scan_cb(self, msg: LaserScan):
        now = self.get_clock().now().nanoseconds / 1e9
        front_min = self._sector_min(msg, -self.front_half_angle, self.front_half_angle)
        left_min = self._sector_min(msg, self.front_half_angle, self.side_angle)
        right_min = self._sector_min(msg, -self.side_angle, -self.front_half_angle)
        pc_front_min, pc_left_min, pc_right_min, front_climbable = self._pointcloud_clearances(now)

        # When the 3D profile looks like a smooth climbable rise, do not let the planar
        # scan alone force obstacle avoidance on that terrain patch.
        if front_climbable:
            front_min = pc_front_min
        else:
            front_min = min(front_min, pc_front_min)
        left_min = min(left_min, pc_left_min)
        right_min = min(right_min, pc_right_min)

        clearances = Vector3()
        clearances.x = front_min if math.isfinite(front_min) else 999.0
        clearances.y = left_min if math.isfinite(left_min) else 999.0
        clearances.z = right_min if math.isfinite(right_min) else 999.0
        self.clearance_pub.publish(clearances)

        action = "clear"
        if math.isfinite(front_min) and front_min <= self.stop_distance:
            action = f"turn_{self._choose_direction(left_min, right_min, now)}"
        elif math.isfinite(front_min) and front_min <= self.caution_distance:
            action = f"caution_{self._choose_direction(left_min, right_min, now)}"
        else:
            self.avoid_direction = None
            self.direction_locked_until = 0.0

        msg_out = String()
        msg_out.data = action
        self.action_pub.publish(msg_out)

        if action != self.last_action or (now - self.last_log_time) >= 2.0:
            self.get_logger().info(
                "Obstacle status: "
                f"action={action} front={clearances.x:.2f} "
                f"left={clearances.y:.2f} right={clearances.z:.2f} "
                f"direction={self.avoid_direction or 'none'}"
            )
            self.last_action = action
            self.last_log_time = now
