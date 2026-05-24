"""High-altitude UAV scout driver that follows a slot ahead of the Husky."""

import math
import time

from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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


class UavScoutDriver(Node):
    """Keep a UAV at high altitude in a scout slot ahead of the Husky."""

    def __init__(
        self,
        node_name: str,
        uav_name: str,
        husky_name: str,
        husky_state_topic: str | None,
        odom_topic: str,
        world_pose_topic: str,
        obstacle_action_topic: str,
        obstacle_clearance_topic: str,
        state_topic: str,
        ready_topic: str,
        report_topic: str,
        scout_altitude_z: float,
        slot_forward_m: float,
        slot_lateral_m: float,
        max_xy_speed: float = 5.0,
        max_z_speed: float = 2.0,
        max_yaw_rate: float = 0.9,
        xy_gain: float = 0.8,
        z_gain: float = 0.45,
        yaw_gain: float = 0.8,
        slot_ready_radius: float = 4.0,
        altitude_ready_tolerance: float = 1.0,
        preview_distance_max: float = 25.0,
        avoid_forward_speed: float = 0.6,
        avoid_lateral_speed: float = 1.6,
        avoid_climb_speed: float = 1.0,
        avoid_clear_hold_seconds: float = 1.0,
        control_period: float = 0.1,
        takeoff_follow_xy_scale: float = 0.45,
        takeoff_release_altitude_margin: float = 6.0,
        altitude_recovery_margin: float = 4.0,
        slot_catchup_radius: float = 20.0,
        slot_catchup_xy_scale: float = 1.0,
        slot_realign_radius: float = 25.0,
        slot_realign_max_xy_speed: float = 6.0,
        max_husky_xy_radius_m: float = 5.0,
        leash_reentry_radius_m: float = 4.0,
        landing_z_margin: float = 0.35,
        landing_z_tolerance: float = 0.35,
        landing_xy_tolerance: float = 2.0,
        landing_descend_trigger_radius: float = 5.0,
        landing_goal_xy: tuple[float, float] | None = None,
        publish_direct_topics: bool = True,
    ):
        super().__init__(node_name)
        self.uav_name = uav_name
        self.husky_name = husky_name
        self.scout_altitude_z = float(scout_altitude_z)
        self.slot_forward_m = float(slot_forward_m)
        self.slot_lateral_m = float(slot_lateral_m)
        self.max_xy_speed = float(max_xy_speed)
        self.max_z_speed = float(max_z_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.xy_gain = float(xy_gain)
        self.z_gain = float(z_gain)
        self.yaw_gain = float(yaw_gain)
        self.slot_ready_radius = float(slot_ready_radius)
        self.altitude_ready_tolerance = float(altitude_ready_tolerance)
        self.preview_distance_max = float(preview_distance_max)
        self.avoid_forward_speed = float(avoid_forward_speed)
        self.avoid_lateral_speed = float(avoid_lateral_speed)
        self.avoid_climb_speed = float(avoid_climb_speed)
        self.avoid_clear_hold_seconds = float(avoid_clear_hold_seconds)
        self.takeoff_follow_xy_scale = float(takeoff_follow_xy_scale)
        self.takeoff_release_altitude_margin = float(takeoff_release_altitude_margin)
        self.altitude_recovery_margin = float(altitude_recovery_margin)
        self.slot_catchup_radius = float(slot_catchup_radius)
        self.slot_catchup_xy_scale = float(slot_catchup_xy_scale)
        self.slot_realign_radius = float(slot_realign_radius)
        self.slot_realign_max_xy_speed = float(slot_realign_max_xy_speed)
        self.max_husky_xy_radius_m = float(max_husky_xy_radius_m)
        self.leash_reentry_radius_m = float(leash_reentry_radius_m)
        self.landing_z_margin = float(landing_z_margin)
        self.landing_z_tolerance = float(landing_z_tolerance)
        self.landing_xy_tolerance = float(landing_xy_tolerance)
        self.landing_descend_trigger_radius = float(landing_descend_trigger_radius)
        self.landing_goal_xy = (
            None
            if landing_goal_xy is None
            else (float(landing_goal_xy[0]), float(landing_goal_xy[1]))
        )
        self.publish_direct_topics = bool(publish_direct_topics)

        self.cmd_pub_model = self.create_publisher(Twist, f"/model/{self.uav_name}/command/twist", 10)
        self.enable_pub_model = self.create_publisher(Bool, f"/model/{self.uav_name}/enable", 10)
        self.cmd_pub_direct = (
            self.create_publisher(Twist, f"/{self.uav_name}/command/twist", 10)
            if self.publish_direct_topics
            else None
        )
        self.enable_pub_direct = (
            self.create_publisher(Bool, f"/{self.uav_name}/enable", 10)
            if self.publish_direct_topics
            else None
        )
        self.state_pub = self.create_publisher(String, state_topic, 10)
        self.ready_pub = self.create_publisher(Bool, ready_topic, 10)
        self.report_pub = self.create_publisher(Vector3, report_topic, 10)

        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(TFMessage, world_pose_topic, self.world_pose_cb, 10)
        self.create_subscription(String, obstacle_action_topic, self.obstacle_action_cb, 10)
        self.create_subscription(Vector3, obstacle_clearance_topic, self.obstacle_clearance_cb, 10)
        if husky_state_topic is not None:
            self.create_subscription(String, husky_state_topic, self.husky_state_cb, 10)

        self.timer = self.create_timer(control_period, self.step)
        self.enable_timer = self.create_timer(2.0, self.enable_controller)

        self.uav_pose = None
        self.uav_world_state = None
        self.husky_world_state = None
        self.obstacle_action = "clear"
        self.obstacle_clearance = (float("inf"), float("inf"), float("inf"))
        self.state = "takeoff"
        self.ready = False
        self.avoid_direction = None
        self.avoid_clear_since = None
        self.last_diag_log = 0.0
        self.last_ready_log = 0.0
        self.husky_arrived = False
        self.landing_target_xy = None

        self.enable_controller()
        self.get_logger().info(
            f"UAV scout driver started for {self.uav_name}: husky={self.husky_name} "
            f"altitude_z={self.scout_altitude_z:.2f} slot=({self.slot_forward_m:.1f}, {self.slot_lateral_m:.1f})"
        )

    def enable_controller(self):
        msg = Bool()
        msg.data = True
        self.enable_pub_model.publish(msg)
        if self.enable_pub_direct is not None:
            self.enable_pub_direct.publish(msg)

    def odom_cb(self, msg: Odometry):
        self.uav_pose = msg.pose.pose

    def world_pose_cb(self, msg: TFMessage):
        uav_tf = extract_model_transform(msg, self.uav_name)
        if uav_tf is not None:
            t = uav_tf.transform.translation
            r = uav_tf.transform.rotation
            self.uav_world_state = {
                "x": float(t.x),
                "y": float(t.y),
                "z": float(t.z),
                "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
            }

        husky_tf = extract_model_transform(msg, self.husky_name)
        if husky_tf is not None:
            t = husky_tf.transform.translation
            r = husky_tf.transform.rotation
            self.husky_world_state = {
                "x": float(t.x),
                "y": float(t.y),
                "z": float(t.z),
                "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
            }

    def obstacle_action_cb(self, msg: String):
        self.obstacle_action = msg.data.strip().lower() if msg.data else "clear"

    def obstacle_clearance_cb(self, msg: Vector3):
        self.obstacle_clearance = (float(msg.x), float(msg.y), float(msg.z))

    def husky_state_cb(self, msg: String):
        state = msg.data.strip().lower() if msg.data else ""
        self.husky_arrived = state == "reached"

    def _uav_state(self):
        return self.uav_world_state

    def _desired_slot(self):
        if self.husky_world_state is None:
            return None
        hx = self.husky_world_state["x"]
        hy = self.husky_world_state["y"]
        hyaw = self.husky_world_state["yaw"]
        offset_x = math.cos(hyaw) * self.slot_forward_m - math.sin(hyaw) * self.slot_lateral_m
        offset_y = math.sin(hyaw) * self.slot_forward_m + math.cos(hyaw) * self.slot_lateral_m
        offset_norm = math.hypot(offset_x, offset_y)
        if offset_norm > self.max_husky_xy_radius_m and offset_norm > 1e-6:
            scale = self.max_husky_xy_radius_m / offset_norm
            offset_x *= scale
            offset_y *= scale
        slot_x = hx + offset_x
        slot_y = hy + offset_y
        return {
            "x": slot_x,
            "y": slot_y,
            "z": self.scout_altitude_z,
            "yaw": hyaw,
        }

    def _husky_xy_distance(self, state) -> float:
        if self.husky_world_state is None:
            return 0.0
        return math.hypot(
            state["x"] - self.husky_world_state["x"],
            state["y"] - self.husky_world_state["y"],
        )

    def _leash_recovery_pose(self):
        if self.husky_world_state is None:
            return None
        return {
            "x": float(self.husky_world_state["x"]),
            "y": float(self.husky_world_state["y"]),
            "z": float(self.scout_altitude_z),
            "yaw": float(self.husky_world_state["yaw"]),
        }

    def _recovery_twist(self, state):
        target = self._leash_recovery_pose()
        if target is None:
            return Twist()

        dx = target["x"] - state["x"]
        dy = target["y"] - state["y"]
        distance = math.hypot(dx, dy)
        error_z = self.scout_altitude_z - state["z"]

        # When the UAV is far above / below the commanded scout altitude, recover
        # altitude first instead of continuing an aggressive XY chase.
        if abs(error_z) > self.altitude_recovery_margin:
            target_yaw = state["yaw"]
            if distance > 0.25:
                target_yaw = math.atan2(dy, dx)
            yaw_error = wrap_angle(target_yaw - state["yaw"])
            yaw_cmd = clamp(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.linear.z = float(clamp(1.5 * self.z_gain * error_z, -self.max_z_speed, self.max_z_speed))
            cmd.angular.z = float(yaw_cmd)
            return cmd

        desired_vx_world = clamp(1.4 * dx, -self.slot_realign_max_xy_speed, self.slot_realign_max_xy_speed)
        desired_vy_world = clamp(1.4 * dy, -self.slot_realign_max_xy_speed, self.slot_realign_max_xy_speed)
        speed = math.hypot(desired_vx_world, desired_vy_world)
        if speed > self.slot_realign_max_xy_speed and speed > 1e-6:
            scale = self.slot_realign_max_xy_speed / speed
            desired_vx_world *= scale
            desired_vy_world *= scale
        body_x, body_y = self._world_to_body(desired_vx_world, desired_vy_world, state["yaw"])

        linear_z = clamp(1.2 * self.z_gain * error_z, -self.max_z_speed, self.max_z_speed)

        target_yaw = state["yaw"]
        if distance > 0.25:
            target_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(target_yaw - state["yaw"])
        yaw_cmd = clamp(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)

        cmd = Twist()
        cmd.linear.x = float(body_x)
        cmd.linear.y = float(body_y)
        cmd.linear.z = float(linear_z)
        cmd.angular.z = float(yaw_cmd)
        return cmd

    def _desired_landing_pose(self, state):
        if self.husky_world_state is not None:
            ground_z = float(self.husky_world_state["z"]) + self.landing_z_margin
        else:
            ground_z = self.landing_z_margin
        if self.landing_target_xy is None:
            if self.landing_goal_xy is not None:
                self.landing_target_xy = self.landing_goal_xy
            else:
                self.landing_target_xy = (float(state["x"]), float(state["y"]))
        return {
            "x": float(self.landing_target_xy[0]),
            "y": float(self.landing_target_xy[1]),
            "z": max(self.landing_z_margin, ground_z),
            "yaw": float(state["yaw"]),
        }

    def _landing_xy_error(self, state) -> float:
        landing_pose = self._desired_landing_pose(state)
        return math.hypot(
            landing_pose["x"] - state["x"],
            landing_pose["y"] - state["y"],
        )

    def _go_to_land_twist(self, state):
        landing_pose = self._desired_landing_pose(state)
        cruise_target = {
            "x": landing_pose["x"],
            "y": landing_pose["y"],
            "z": max(state["z"], self.scout_altitude_z),
            "yaw": landing_pose["yaw"],
        }
        return self._slot_twist(state, cruise_target)

    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

    def _publish_ready(self):
        msg = Bool()
        msg.data = bool(self.ready)
        self.ready_pub.publish(msg)

    def _publish_report(self):
        report = Vector3()
        front = float(self.obstacle_clearance[0])
        if math.isfinite(front) and front <= self.preview_distance_max:
            report.x = float(self.slot_forward_m + front)
            report.y = float(self.slot_lateral_m)
            report.z = 1.0
        else:
            report.x = 999.0
            report.y = float(self.slot_lateral_m)
            report.z = 0.0
        self.report_pub.publish(report)

    def _obstacle_active(self) -> bool:
        return (self.obstacle_action or "clear") != "clear"

    def _world_to_body(self, vx_world: float, vy_world: float, yaw: float):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        body_x = cos_yaw * vx_world + sin_yaw * vy_world
        body_y = -sin_yaw * vx_world + cos_yaw * vy_world
        return body_x, body_y

    def _slot_twist(self, state, slot):
        return self._slot_twist_with_scale(state, slot, 1.0)

    def _slot_twist_with_scale(self, state, slot, xy_scale: float):
        dx = slot["x"] - state["x"]
        dy = slot["y"] - state["y"]
        slot_error = math.hypot(dx, dy)
        desired_vx_world = clamp(
            self.xy_gain * dx,
            -self.max_xy_speed * xy_scale,
            self.max_xy_speed * xy_scale,
        )
        desired_vy_world = clamp(
            self.xy_gain * dy,
            -self.max_xy_speed * xy_scale,
            self.max_xy_speed * xy_scale,
        )
        speed = math.hypot(desired_vx_world, desired_vy_world)
        scaled_max_xy_speed = max(0.0, self.max_xy_speed * xy_scale)
        if slot_error >= self.slot_realign_radius:
            scaled_max_xy_speed = min(scaled_max_xy_speed, self.slot_realign_max_xy_speed)
        if speed > scaled_max_xy_speed and speed > 1e-6:
            scale = scaled_max_xy_speed / speed
            desired_vx_world *= scale
            desired_vy_world *= scale
        body_x, body_y = self._world_to_body(desired_vx_world, desired_vy_world, state["yaw"])

        # Keep scouts pinned to the commanded altitude during tracking. Landing
        # logic is handled separately and is the only time we intentionally
        # descend below the scout altitude target.
        target_z = float(slot["z"])
        error_z = target_z - state["z"]
        linear_z = clamp(self.z_gain * error_z, -self.max_z_speed, self.max_z_speed)
        target_yaw = slot["yaw"]
        if slot_error >= self.slot_realign_radius and speed > 1e-6:
            target_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(target_yaw - state["yaw"])
        yaw_cmd = clamp(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)

        cmd = Twist()
        cmd.linear.x = float(body_x)
        cmd.linear.y = float(body_y)
        cmd.linear.z = float(linear_z)
        cmd.angular.z = float(yaw_cmd)
        return cmd

    def _avoid_twist(self, state):
        cmd = Twist()
        direction = self.avoid_direction or "up"
        cmd.linear.x = float(self.avoid_forward_speed)
        if direction == "left":
            cmd.linear.y = float(self.avoid_lateral_speed)
        elif direction == "right":
            cmd.linear.y = float(-self.avoid_lateral_speed)
        else:
            cmd.linear.y = 0.0
        altitude_error = self.scout_altitude_z - state["z"]
        cmd.linear.z = float(clamp(self.z_gain * altitude_error, -self.max_z_speed, self.max_z_speed))

        if self.husky_world_state is not None:
            yaw_error = wrap_angle(self.husky_world_state["yaw"] - state["yaw"])
            cmd.angular.z = float(clamp(0.5 * self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate))
        return cmd

    def _slot_error(self, state, slot):
        return math.hypot(slot["x"] - state["x"], slot["y"] - state["y"])

    def step(self):
        self._publish_state()
        self._publish_ready()
        self._publish_report()

        state = self._uav_state()
        slot = self._desired_slot()
        if state is None or slot is None:
            return

        now = time.monotonic()
        slot_error = self._slot_error(state, slot)
        altitude_error = abs(slot["z"] - state["z"])
        husky_xy_distance = self._husky_xy_distance(state)

        if self.husky_arrived and self.state != "landed":
            self.ready = False
            landing_error_xy = self._landing_xy_error(state)
            if landing_error_xy <= self.landing_descend_trigger_radius:
                self.state = "descend"
            else:
                self.state = "go_to_land"

        if self.state == "go_to_land":
            landing_error_xy = self._landing_xy_error(state)
            if landing_error_xy <= self.landing_descend_trigger_radius:
                self.state = "descend"
            else:
                self.publish_cmd(self._go_to_land_twist(state))
                if (now - self.last_diag_log) >= 2.0:
                    landing_pose = self._desired_landing_pose(state)
                    self.get_logger().info(
                        f"{self.uav_name} go_to_land pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                        f"target=({landing_pose['x']:.2f},{landing_pose['y']:.2f},{landing_pose['z']:.2f}) "
                        f"landing_xy_error={landing_error_xy:.2f}"
                    )
                    self.last_diag_log = now
                return

        if self.state == "descend":
            landing_pose = self._desired_landing_pose(state)
            landing_error_xy = math.hypot(
                landing_pose["x"] - state["x"],
                landing_pose["y"] - state["y"],
            )
            landing_error_z = abs(landing_pose["z"] - state["z"])
            if (
                landing_error_z <= self.landing_z_tolerance
                and landing_error_xy <= self.landing_xy_tolerance
            ):
                self.state = "landed"
                self.publish_cmd(Twist())
                return
            self.publish_cmd(self._slot_twist(state, landing_pose))
            if (now - self.last_diag_log) >= 2.0:
                self.get_logger().info(
                    f"{self.uav_name} descend pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                    f"target=({landing_pose['x']:.2f},{landing_pose['y']:.2f},{landing_pose['z']:.2f})"
                )
                self.last_diag_log = now
            return

        if self.state == "landed":
            self.ready = False
            self.publish_cmd(Twist())
            return

        if (
            not self.husky_arrived
            and self.state not in {"takeoff", "descend", "landed", "go_to_land"}
            and husky_xy_distance > self.max_husky_xy_radius_m
        ):
            self.state = (
                "altitude_recover"
                if abs(self.scout_altitude_z - state["z"]) > self.altitude_recovery_margin
                else "return_to_husky"
            )
            self.ready = False
            self.publish_cmd(self._recovery_twist(state))
            if (now - self.last_diag_log) >= 2.0:
                self.get_logger().info(
                    f"{self.uav_name} {self.state} pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                    f"husky=({self.husky_world_state['x']:.2f},{self.husky_world_state['y']:.2f}) "
                    f"husky_xy_distance={husky_xy_distance:.2f} target_z={self.scout_altitude_z:.2f}"
                )
                self.last_diag_log = now
            return

        if self.state == "return_to_husky" and husky_xy_distance > self.leash_reentry_radius_m:
            self.ready = False
            self.publish_cmd(self._recovery_twist(state))
            return

        if self.state == "altitude_recover":
            altitude_error = abs(self.scout_altitude_z - state["z"])
            if altitude_error <= self.altitude_recovery_margin:
                self.state = "return_to_husky" if husky_xy_distance > self.leash_reentry_radius_m else "go_to_slot"
            else:
                self.ready = False
                self.publish_cmd(self._recovery_twist(state))
                if (now - self.last_diag_log) >= 2.0:
                    self.get_logger().info(
                        f"{self.uav_name} altitude_recover pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                        f"husky=({self.husky_world_state['x']:.2f},{self.husky_world_state['y']:.2f}) "
                        f"husky_xy_distance={husky_xy_distance:.2f} target_z={self.scout_altitude_z:.2f}"
                    )
                    self.last_diag_log = now
                return

        if self.state == "takeoff":
            self.ready = False
            if state["z"] >= (self.scout_altitude_z - self.takeoff_release_altitude_margin):
                self.state = "go_to_slot"
            else:
                cmd = self._slot_twist_with_scale(state, slot, self.takeoff_follow_xy_scale)
                cmd.linear.z = max(float(cmd.linear.z), float(self.max_z_speed))
                self.publish_cmd(cmd)
                if (now - self.last_diag_log) >= 2.0:
                    self.get_logger().info(
                        f"{self.uav_name} takeoff pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                        f"target_z={self.scout_altitude_z:.2f} slot_error={slot_error:.2f}"
                    )
                    self.last_diag_log = now
                return

        if self.state == "return_to_husky":
            self.state = "go_to_slot"

        if self._obstacle_active():
            self.state = "avoid"
            self.ready = False
            self.avoid_direction = self.obstacle_action.split("_")[-1]
            self.avoid_clear_since = None

        if self.state == "avoid":
            if not self._obstacle_active():
                if self.avoid_clear_since is None:
                    self.avoid_clear_since = now
                elif (now - self.avoid_clear_since) >= self.avoid_clear_hold_seconds:
                    self.state = "go_to_slot"
                    self.avoid_direction = None
                    self.avoid_clear_since = None
            else:
                self.avoid_clear_since = None
                self.avoid_direction = self.obstacle_action.split("_")[-1]
            self.publish_cmd(self._avoid_twist(state))
            return

        self.state = "go_to_slot" if slot_error > self.slot_ready_radius else "scout_ready"
        self.ready = self.state == "scout_ready" and altitude_error <= self.altitude_ready_tolerance
        xy_scale = self.slot_catchup_xy_scale if slot_error >= self.slot_catchup_radius else 1.0
        self.publish_cmd(self._slot_twist_with_scale(state, slot, xy_scale))

        if self.ready and (now - self.last_ready_log) >= 5.0:
            self.get_logger().info(
                f"{self.uav_name} scout ready: slot_error={slot_error:.2f} alt={state['z']:.2f} "
                f"front_clearance={self.obstacle_clearance[0] if math.isfinite(self.obstacle_clearance[0]) else 999.0:.2f}"
            )
            self.last_ready_log = now

        if (now - self.last_diag_log) >= 2.0:
            front = self.obstacle_clearance[0] if math.isfinite(self.obstacle_clearance[0]) else 999.0
            self.get_logger().info(
                f"{self.uav_name} scout state={self.state} "
                f"pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                f"slot=({slot['x']:.2f},{slot['y']:.2f},{slot['z']:.2f}) "
                f"slot_error={slot_error:.2f} husky_xy_distance={husky_xy_distance:.2f} "
                f"front={front:.2f} obstacle={self.obstacle_action}"
            )
            self.last_diag_log = now

    def publish_cmd(self, cmd: Twist):
        self.cmd_pub_model.publish(cmd)
        if self.cmd_pub_direct is not None:
            self.cmd_pub_direct.publish(cmd)
