"""Planar-lidar obstacle detection for the live Husky simulation."""

import math
from collections import deque

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
        terrain_profile_topic: str | None = None,
        gap_profile_topic: str | None = None,
        pointcloud_topic: str | None = None,
        front_half_angle_deg: float = 45.0,
        side_angle_deg: float = 65.0,
        stop_distance: float = 1.8,
        caution_distance: float = 3.0,
        passable_gap_width_m: float = 1.25,
        passable_gap_margin_m: float = 0.20,
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
        terrain_bin_size_x: float = 0.60,
        terrain_bin_min_points: int = 4,
        terrain_min_profile_bins: int = 3,
        terrain_max_step_z: float = 0.80,
        terrain_long_min_extent_x: float = 1.5,
        terrain_long_high_extent_x: float = 3.0,
        terrain_block_max_segment_x: float = 0.9,
        terrain_block_min_step_z: float = 0.45,
        terrain_memory_size: int = 6,
        terrain_memory_min_votes: int = 4,
        clear_memory_size: int = 3,
        clear_memory_min_votes: int = 2,
    ):
        super().__init__(node_name)
        self.scan_topic = scan_topic
        self.action_topic = action_topic
        self.clearance_topic = clearance_topic
        self.terrain_profile_topic = terrain_profile_topic
        self.gap_profile_topic = gap_profile_topic
        self.pointcloud_topic = pointcloud_topic
        self.front_half_angle = math.radians(front_half_angle_deg)
        self.side_angle = math.radians(side_angle_deg)
        self.stop_distance = stop_distance
        self.caution_distance = caution_distance
        self.passable_gap_width_m = passable_gap_width_m
        self.passable_gap_margin_m = passable_gap_margin_m
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
        self.terrain_long_min_extent_x = terrain_long_min_extent_x
        self.terrain_long_high_extent_x = terrain_long_high_extent_x
        self.terrain_block_max_segment_x = terrain_block_max_segment_x
        self.terrain_block_min_step_z = terrain_block_min_step_z
        self.terrain_memory_size = max(3, int(terrain_memory_size))
        self.terrain_memory_min_votes = max(2, int(terrain_memory_min_votes))
        self.clear_memory_size = max(2, int(clear_memory_size))
        self.clear_memory_min_votes = max(1, int(clear_memory_min_votes))

        self.action_pub = self.create_publisher(String, self.action_topic, 10)
        self.clearance_pub = self.create_publisher(Vector3, self.clearance_topic, 10)
        self.terrain_profile_pub = (
            self.create_publisher(Vector3, self.terrain_profile_topic, 10)
            if self.terrain_profile_topic is not None
            else None
        )
        self.gap_profile_pub = (
            self.create_publisher(Vector3, self.gap_profile_topic, 10)
            if self.gap_profile_topic is not None
            else None
        )
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
        self.front_kind_history = deque(maxlen=self.terrain_memory_size)
        self.clear_history = deque(maxlen=self.clear_memory_size)

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

    def _classify_front_terrain(self, front_points: list[tuple[float, float]]) -> tuple[float, bool, float]:
        """Return (obstacle distance, long-terrain flag, terrain extent x)."""

        if len(front_points) < self.pointcloud_min_points:
            return (float("inf"), False, 0.0)

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

        if len(profile) < 2:
            return (min(x for x, _ in front_points), False, 0.0)

        profile_extent_x = max(0.0, profile[-1][0] - profile[0][0])
        max_abs_step_z = 0.0
        first_block_x = None
        smooth_segments = 0

        for idx in range(1, len(profile)):
            prev_x, prev_z = profile[idx - 1]
            curr_x, curr_z = profile[idx]
            dx = max(curr_x - prev_x, 1e-3)
            dz = curr_z - prev_z
            abs_dz = abs(dz)
            max_abs_step_z = max(max_abs_step_z, abs_dz)

            # A short, abrupt rise/drop should be treated like a block.
            if abs_dz >= self.terrain_block_min_step_z and dx <= self.terrain_block_max_segment_x:
                first_block_x = curr_x
                break

            # Smooth continuous terrain can rise or fall, but should not jump sharply.
            if abs_dz <= self.terrain_max_step_z:
                smooth_segments += 1

        if first_block_x is not None:
            return (first_block_x, False, profile_extent_x)

        required_smooth_segments = max(1, len(profile) - 1)
        is_long_terrain = (
            profile_extent_x >= self.terrain_long_min_extent_x
            and smooth_segments >= required_smooth_segments
            and max_abs_step_z <= self.terrain_max_step_z
        )
        if is_long_terrain:
            return (float("inf"), True, profile_extent_x)

        return (min(x for x, _ in front_points), False, profile_extent_x)

    def _pointcloud_clearances(self, now: float) -> tuple[float, float, float, bool, float]:
        if self.latest_pointcloud is None or self.latest_pointcloud_time is None:
            return (float("inf"), float("inf"), float("inf"), False, 0.0)
        if (now - self.latest_pointcloud_time) > self.pointcloud_timeout:
            return (float("inf"), float("inf"), float("inf"), False, 0.0)

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

        front_min, front_climbable, front_slope_deg = self._classify_front_terrain(front_points)
        if left_count < self.pointcloud_min_points:
            left_min = float("inf")
        if right_count < self.pointcloud_min_points:
            right_min = float("inf")
        return (front_min, left_min, right_min, front_climbable, front_slope_deg)

    def _smooth_front_classification(
        self,
        raw_front_min: float,
        front_climbable: bool,
        terrain_extent_x: float,
    ) -> tuple[float, bool, float]:
        front_kind = "terrain" if front_climbable else "block"
        self.front_kind_history.append(front_kind)

        terrain_votes = sum(1 for kind in self.front_kind_history if kind == "terrain")
        block_votes = sum(1 for kind in self.front_kind_history if kind == "block")

        if terrain_votes >= self.terrain_memory_min_votes and terrain_votes > block_votes:
            return (float("inf"), True, max(terrain_extent_x, self.terrain_long_min_extent_x))

        if block_votes >= self.terrain_memory_min_votes and block_votes >= terrain_votes:
            return (raw_front_min, False, terrain_extent_x)

        return (raw_front_min, front_climbable, terrain_extent_x)

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

    def _scan_gap_profile(self, msg: LaserScan) -> tuple[float, float, bool]:
        lookahead = min(self.caution_distance + 0.8, 6.0)
        left_edge_y = float("inf")
        right_edge_y = float("-inf")
        sample_count = 0

        angle = msg.angle_min
        for distance in msg.ranges:
            wrapped = wrap_angle(angle)
            if -self.front_half_angle <= wrapped <= self.front_half_angle:
                if math.isfinite(distance) and self.min_valid_range <= distance <= lookahead:
                    x = float(distance) * math.cos(wrapped)
                    y = float(distance) * math.sin(wrapped)
                    if x >= self.pointcloud_min_forward_x:
                        sample_count += 1
                        if y >= 0.0:
                            left_edge_y = min(left_edge_y, y)
                        else:
                            right_edge_y = max(right_edge_y, y)
            angle += msg.angle_increment

        if sample_count == 0:
            return (999.0, 0.0, True)

        half_required = 0.5 * (self.passable_gap_width_m + 2.0 * self.passable_gap_margin_m)
        if left_edge_y == float("inf") and right_edge_y == float("-inf"):
            return (999.0, 0.0, True)
        if left_edge_y == float("inf"):
            left_edge_y = half_required
        if right_edge_y == float("-inf"):
            right_edge_y = -half_required

        gap_width = max(0.0, left_edge_y - right_edge_y)
        gap_center_y = 0.5 * (left_edge_y + right_edge_y)
        passable = gap_width >= (self.passable_gap_width_m + 2.0 * self.passable_gap_margin_m)
        return (gap_width, gap_center_y, passable)

    def scan_cb(self, msg: LaserScan):
        now = self.get_clock().now().nanoseconds / 1e9
        front_min = self._sector_min(msg, -self.front_half_angle, self.front_half_angle)
        left_min = self._sector_min(msg, self.front_half_angle, self.side_angle)
        right_min = self._sector_min(msg, -self.side_angle, -self.front_half_angle)
        pc_front_min, pc_left_min, pc_right_min, front_climbable, front_terrain_extent_x = self._pointcloud_clearances(now)
        pc_front_min, front_climbable, front_terrain_extent_x = self._smooth_front_classification(
            pc_front_min,
            front_climbable,
            front_terrain_extent_x,
        )

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

        if self.terrain_profile_pub is not None:
            terrain = Vector3()
            terrain.x = float(front_terrain_extent_x)
            terrain.y = 1.0 if front_climbable else 0.0
            terrain.z = front_min if math.isfinite(front_min) else 999.0
            self.terrain_profile_pub.publish(terrain)

        gap_width = 0.0
        gap_center_y = 0.0
        gap_passable = False
        if self.gap_profile_pub is not None:
            gap_width, gap_center_y, gap_passable = self._scan_gap_profile(msg)
            gap = Vector3()
            gap.x = float(gap_width)
            gap.y = float(gap_center_y)
            gap.z = 1.0 if gap_passable else 0.0
            self.gap_profile_pub.publish(gap)

        raw_action = "clear"
        if math.isfinite(front_min) and front_min <= self.stop_distance:
            raw_action = f"turn_{self._choose_direction(left_min, right_min, now)}"
        elif math.isfinite(front_min) and front_min <= self.caution_distance:
            raw_action = f"caution_{self._choose_direction(left_min, right_min, now)}"

        self.clear_history.append("clear" if raw_action == "clear" else "blocked")
        blocked_votes = sum(1 for state in self.clear_history if state == "blocked")
        clear_votes = sum(1 for state in self.clear_history if state == "clear")

        action = raw_action
        if raw_action == "clear":
            if clear_votes < self.clear_memory_min_votes and self.last_action is not None:
                action = self.last_action
            else:
                self.avoid_direction = None
                self.direction_locked_until = 0.0
        else:
            if blocked_votes < self.clear_memory_min_votes and self.last_action == "clear":
                action = "clear"
                self.avoid_direction = None
                self.direction_locked_until = 0.0

        msg_out = String()
        msg_out.data = action
        self.action_pub.publish(msg_out)

        if action != self.last_action or (now - self.last_log_time) >= 2.0:
            if self.gap_profile_pub is not None:
                self.get_logger().info(
                    "Obstacle status: "
                    f"action={action} front={clearances.x:.2f} "
                    f"left={clearances.y:.2f} right={clearances.z:.2f} "
                    f"direction={self.avoid_direction or 'none'} "
                    f"gap_width={gap_width:.2f} gap_center={gap_center_y:.2f} "
                    f"gap_passable={'yes' if gap_passable else 'no'}"
                )
            else:
                self.get_logger().info(
                    "Obstacle status: "
                    f"action={action} front={clearances.x:.2f} "
                    f"left={clearances.y:.2f} right={clearances.z:.2f} "
                    f"direction={self.avoid_direction or 'none'}"
                )
            self.last_action = action
            self.last_log_time = now
