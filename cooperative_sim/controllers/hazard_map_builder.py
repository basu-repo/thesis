"""Shared local 2D hazard map builder for UAV/UGV fusion."""

import math

from geometry_msgs.msg import Vector3
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_model_transform(msg: TFMessage, model_name: str):
    selected_model = None
    selected_base_link = None
    for transform in msg.transforms:
        child = transform.child_frame_id or ""
        child_parts = [part for part in child.split("/") if part]
        if model_name not in child_parts:
            continue
        if child_parts and child_parts[-1] == "base_link":
            selected_base_link = transform
            continue
        if (
            child == model_name
            or child.endswith(f"/{model_name}")
            or (child_parts and child_parts[-1] == model_name)
        ):
            selected_model = transform
    return selected_base_link or selected_model


class HazardMapBuilderNode(Node):
    """Fuse UAV scout previews and Husky local cues into a small 2D hazard map."""

    def __init__(
        self,
        node_name: str,
        husky_name: str,
        world_pose_topic: str,
        scout_report_topics: list[str],
        husky_obstacle_action_topic: str,
        husky_obstacle_clearance_topic: str,
        map_topic: str,
        memory_map_topic: str,
        guidance_topic: str,
        map_resolution_m: float = 0.5,
        map_forward_m: float = 56.0,
        map_rear_m: float = 8.0,
        map_half_width_m: float = 32.0,
        publish_period: float = 0.25,
        decay_per_publish: float = 0.90,
        scout_hazard_radius_m: float = 1.5,
        local_hazard_radius_m: float = 1.1,
        terrain_band_depth_m: float = 3.0,
        terrain_band_padding_m: float = 1.0,
        terrain_sure_hold_seconds: float = 2.5,
        terrain_sure_break_distance_m: float = 4.5,
    ):
        super().__init__(node_name)
        self.husky_name = husky_name
        self.world_pose_topic = world_pose_topic
        self.map_resolution_m = float(map_resolution_m)
        self.map_forward_m = float(map_forward_m)
        self.map_rear_m = float(map_rear_m)
        self.map_half_width_m = float(map_half_width_m)
        self.decay_per_publish = float(decay_per_publish)
        self.scout_hazard_radius_m = float(scout_hazard_radius_m)
        self.local_hazard_radius_m = float(local_hazard_radius_m)
        self.terrain_band_depth_m = float(terrain_band_depth_m)
        self.terrain_band_padding_m = float(terrain_band_padding_m)
        self.terrain_sure_hold_seconds = float(terrain_sure_hold_seconds)
        self.terrain_sure_break_distance_m = float(terrain_sure_break_distance_m)

        self.grid_width = max(1, int(round((self.map_forward_m + self.map_rear_m) / self.map_resolution_m)))
        self.grid_height = max(1, int(round((2.0 * self.map_half_width_m) / self.map_resolution_m)))

        self.husky_world_state = None
        self.husky_obstacle_action = "clear"
        self.husky_front_clearance = float("inf")
        self.scout_names = [f"scout_{idx + 1}" for idx in range(len(scout_report_topics))]
        self.scout_reports = {
            name: {"distance": 999.0, "lateral": 0.0, "blocked": False}
            for name in self.scout_names
        }
        self.last_guidance = None
        self.last_log_time = 0.0
        self.terrain_sure_until = 0.0
        self.terrain_sure_distance_m = 999.0
        self.live_data = [0.0] * (self.grid_width * self.grid_height)
        self.memory_data = [0] * (self.grid_width * self.grid_height)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, map_topic, map_qos)
        self.memory_map_pub = self.create_publisher(OccupancyGrid, memory_map_topic, map_qos)
        self.guidance_pub = self.create_publisher(String, guidance_topic, 10)

        self.create_subscription(TFMessage, self.world_pose_topic, self.world_pose_cb, 10)
        self.create_subscription(String, husky_obstacle_action_topic, self.husky_obstacle_action_cb, 10)
        self.create_subscription(Vector3, husky_obstacle_clearance_topic, self.husky_obstacle_clearance_cb, 10)
        for idx, topic in enumerate(scout_report_topics):
            self.create_subscription(Vector3, topic, self._make_scout_report_cb(self.scout_names[idx]), 10)

        self.timer = self.create_timer(publish_period, self.publish_updates)
        self.get_logger().info("Hazard map builder started.")

    def world_pose_cb(self, msg: TFMessage):
        husky_tf = extract_model_transform(msg, self.husky_name)
        if husky_tf is None:
            return
        t = husky_tf.transform.translation
        r = husky_tf.transform.rotation
        self.husky_world_state = {
            "x": float(t.x),
            "y": float(t.y),
            "z": float(t.z),
            "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
        }

    def husky_obstacle_action_cb(self, msg: String):
        self.husky_obstacle_action = msg.data.strip().lower() if msg.data else "clear"

    def husky_obstacle_clearance_cb(self, msg: Vector3):
        self.husky_front_clearance = float(msg.x)

    def _make_scout_report_cb(self, name: str):
        def cb(msg: Vector3):
            distance = float(msg.x)
            self.scout_reports[name] = {
                "distance": distance,
                "lateral": float(msg.y),
                "blocked": bool(msg.z >= 0.5 and math.isfinite(distance) and distance < 998.0),
            }

        return cb

    def _grid_index(self, x_forward: float, y_lateral: float) -> int | None:
        grid_x = int((x_forward + self.map_rear_m) / self.map_resolution_m)
        grid_y = int((y_lateral + self.map_half_width_m) / self.map_resolution_m)
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return None
        return grid_y * self.grid_width + grid_x

    def _stamp_disc(self, data: list[float], x_forward: float, y_lateral: float, radius_m: float, value: float):
        radius_cells = max(1, int(round(radius_m / self.map_resolution_m)))
        center_idx = self._grid_index(x_forward, y_lateral)
        if center_idx is None:
            return
        center_x = center_idx % self.grid_width
        center_y = center_idx // self.grid_width
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx + dy * dy) > (radius_cells * radius_cells):
                    continue
                gx = center_x + dx
                gy = center_y + dy
                if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                    continue
                idx = gy * self.grid_width + gx
                data[idx] = max(data[idx], value)

    def _stamp_band(self, data: list[float], x_forward: float, y_min: float, y_max: float, depth_m: float, value: float):
        x_steps = max(1, int(round(depth_m / self.map_resolution_m)))
        x0 = x_forward
        y_lo = min(y_min, y_max)
        y_hi = max(y_min, y_max)
        y_step = self.map_resolution_m
        current_x = x0
        for _ in range(x_steps):
            current_y = y_lo
            while current_y <= y_hi + 1e-6:
                idx = self._grid_index(current_x, current_y)
                if idx is not None:
                    data[idx] = max(data[idx], value)
                current_y += y_step
            current_x += self.map_resolution_m

    def _compute_guidance(self) -> tuple[str, list[float]]:
        now = self.get_clock().now().nanoseconds / 1e9
        data = [max(0.0, value * self.decay_per_publish) for value in self.live_data]
        blocked_reports = [report for report in self.scout_reports.values() if report["blocked"]]
        guidance = "clear"

        for report in blocked_reports:
            self._stamp_disc(
                data,
                report["distance"],
                report["lateral"],
                radius_m=self.scout_hazard_radius_m,
                value=85.0,
            )

        local_blocked = (self.husky_obstacle_action or "clear") != "clear" and math.isfinite(self.husky_front_clearance)
        if local_blocked:
            self._stamp_disc(
                data,
                self.husky_front_clearance,
                0.0,
                radius_m=self.local_hazard_radius_m,
                value=100.0,
            )

        if len(blocked_reports) >= 2:
            sorted_reports = sorted(blocked_reports, key=lambda report: report["distance"])
            left = min(sorted_reports, key=lambda report: report["lateral"])
            right = max(sorted_reports, key=lambda report: report["lateral"])
            broad_front = (
                left["lateral"] < 0.0
                and right["lateral"] > 0.0
                and abs(left["distance"] - right["distance"]) <= 3.0
                and min(left["distance"], right["distance"]) <= 12.0
            )
            if broad_front:
                avg_distance = 0.5 * (left["distance"] + right["distance"])
                self._stamp_band(
                    data,
                    avg_distance,
                    left["lateral"] - self.terrain_band_padding_m,
                    right["lateral"] + self.terrain_band_padding_m,
                    depth_m=self.terrain_band_depth_m,
                    value=65.0,
                )
                self.terrain_sure_until = now + self.terrain_sure_hold_seconds
                self.terrain_sure_distance_m = avg_distance
                guidance = f"terrain_sure_front distance={avg_distance:.2f}"

        terrain_sure_latched = False
        if guidance == "clear" and now < self.terrain_sure_until:
            nearest_distance = min((report["distance"] for report in blocked_reports), default=999.0)
            center_like_block = any(abs(report["lateral"]) <= 1.0 for report in blocked_reports)
            strong_close_block = nearest_distance <= self.terrain_sure_break_distance_m
            if not center_like_block and not strong_close_block:
                terrain_sure_latched = True
                guidance = f"terrain_sure_front distance={self.terrain_sure_distance_m:.2f}"

        # UAV reports are used only to confirm broad highland / terrain traversability.
        # Object and center/side obstacle blocking remains the UGV's own responsibility.
        # Therefore scout-reported blocked regions may still be stamped into the hazard map
        # for visualization, but they do not produce obstacle guidance for the UGV.

        if terrain_sure_latched:
            self._stamp_band(
                data,
                self.terrain_sure_distance_m,
                -self.terrain_band_padding_m,
                self.terrain_band_padding_m,
                depth_m=self.terrain_band_depth_m,
                value=65.0,
            )

        return guidance, data

    def _update_memory_map(self, live_data: list[float]):
        for idx, value in enumerate(live_data):
            int_value = int(round(max(0.0, min(100.0, value))))
            if int_value > self.memory_data[idx]:
                self.memory_data[idx] = int_value

    def _build_grid_msg(self, data: list[int], frame_id: str) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = frame_id
        grid.info.resolution = float(self.map_resolution_m)
        grid.info.width = int(self.grid_width)
        grid.info.height = int(self.grid_height)
        grid.info.origin.position.x = -float(self.map_rear_m)
        grid.info.origin.position.y = -float(self.map_half_width_m)
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = data
        return grid

    def publish_updates(self):
        if self.husky_world_state is None:
            return

        guidance, live_data = self._compute_guidance()
        self.live_data = live_data
        self._update_memory_map(live_data)

        live_grid = self._build_grid_msg(
            [int(round(max(0.0, min(100.0, value)))) for value in live_data],
            "base_link",
        )
        memory_grid = self._build_grid_msg(
            list(self.memory_data),
            "base_link",
        )
        self.map_pub.publish(live_grid)
        self.memory_map_pub.publish(memory_grid)

        msg = String()
        msg.data = guidance
        self.guidance_pub.publish(msg)

        now = self.get_clock().now().nanoseconds / 1e9
        if guidance != self.last_guidance or (now - self.last_log_time) >= 3.0:
            self.get_logger().info(f"hazard_map guidance={guidance}")
            self.last_guidance = guidance
            self.last_log_time = now
