"""Independent UAV goal driver with takeoff, obstacle avoidance, and landing."""

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


def local_pose_to_world(pose, spawn_xyz, spawn_yaw):
    p = pose.position
    q = pose.orientation
    local_x = float(p.x)
    local_y = float(p.y)
    local_z = float(p.z)
    cos_yaw = math.cos(spawn_yaw)
    sin_yaw = math.sin(spawn_yaw)
    world_x = spawn_xyz[0] + cos_yaw * local_x - sin_yaw * local_y
    world_y = spawn_xyz[1] + sin_yaw * local_x + cos_yaw * local_y
    world_z = spawn_xyz[2] + local_z
    local_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
    return {
        "x": world_x,
        "y": world_y,
        "z": world_z,
        "yaw": wrap_angle(spawn_yaw + local_yaw),
    }


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


class UavGoalDriver(Node):
    """Fly a UAV independently to a goal with basic local avoidance and landing."""

    def __init__(
        self,
        node_name: str,
        uav_name: str,
        odom_topic: str,
        world_pose_topic: str | None,
        obstacle_action_topic: str,
        obstacle_clearance_topic: str,
        state_topic: str,
        world_goal_xyz: tuple[float, float, float],
        spawn_xyz: tuple[float, float, float] | None = None,
        spawn_yaw: float = 0.0,
        cruise_height_agl: float = 10.0,
        control_period: float = 0.1,
        goal_xy_tolerance: float = 1.5,
        landing_altitude_tolerance: float = 0.35,
        descend_radius: float = 6.0,
        max_xy_speed: float = 4.5,
        max_z_speed: float = 1.2,
        max_yaw_rate: float = 0.9,
        xy_gain: float = 1.0,
        z_gain: float = 0.45,
        yaw_gain: float = 0.8,
        avoid_forward_speed: float = 0.4,
        avoid_lateral_speed: float = 1.0,
        avoid_climb_speed: float = 0.9,
        avoid_clear_hold_seconds: float = 0.8,
    ):
        super().__init__(node_name)
        self.uav_name = uav_name
        self.spawn_xyz = spawn_xyz
        self.spawn_yaw = spawn_yaw
        self.world_goal_xyz = world_goal_xyz
        self.goal_xy_tolerance = goal_xy_tolerance
        self.landing_altitude_tolerance = landing_altitude_tolerance
        self.descend_radius = descend_radius
        self.max_xy_speed = max_xy_speed
        self.max_z_speed = max_z_speed
        self.max_yaw_rate = max_yaw_rate
        self.xy_gain = xy_gain
        self.z_gain = z_gain
        self.yaw_gain = yaw_gain
        self.avoid_forward_speed = avoid_forward_speed
        self.avoid_lateral_speed = avoid_lateral_speed
        self.avoid_climb_speed = avoid_climb_speed
        self.avoid_clear_hold_seconds = avoid_clear_hold_seconds
        self.cruise_altitude = (
            float(world_goal_xyz[2]) + cruise_height_agl
            if spawn_xyz is None
            else max(float(spawn_xyz[2]) + cruise_height_agl, float(world_goal_xyz[2]) + 4.0)
        )

        self.cmd_pub_model = self.create_publisher(Twist, f"/model/{self.uav_name}/command/twist", 10)
        self.cmd_pub_direct = self.create_publisher(Twist, f"/{self.uav_name}/command/twist", 10)
        self.enable_pub_model = self.create_publisher(Bool, f"/model/{self.uav_name}/enable", 10)
        self.enable_pub_direct = self.create_publisher(Bool, f"/{self.uav_name}/enable", 10)
        self.state_pub = self.create_publisher(String, state_topic, 10)

        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        if world_pose_topic is not None:
            self.create_subscription(TFMessage, world_pose_topic, self.world_pose_cb, 10)
        self.create_subscription(String, obstacle_action_topic, self.obstacle_action_cb, 10)
        self.create_subscription(Vector3, obstacle_clearance_topic, self.obstacle_clearance_cb, 10)

        self.timer = self.create_timer(control_period, self.step)
        self.enable_timer = self.create_timer(2.0, self.enable_controller)

        self.uav_pose = None
        self.uav_world_state = None
        self.obstacle_action = "clear"
        self.obstacle_clearance = (float("inf"), float("inf"), float("inf"))
        self.state = "takeoff"
        self.arrived = False
        self.avoid_direction = None
        self.avoid_clear_since = None
        self.last_diag_log = 0.0
        self.held_altitude_floor = None

        self.enable_controller()
        self.get_logger().info(
            f"UAV goal driver started for {self.uav_name}: goal=({self.world_goal_xyz[0]:.2f}, {self.world_goal_xyz[1]:.2f}, {self.world_goal_xyz[2]:.2f}) cruise_alt={self.cruise_altitude:.2f}"
        )

    def enable_controller(self):
        msg = Bool()
        msg.data = True
        self.enable_pub_model.publish(msg)
        self.enable_pub_direct.publish(msg)

    def publish_state(self):
        msg = String()
        msg.data = self.state if not self.arrived else "arrived"
        self.state_pub.publish(msg)

    def publish_cmd(self, cmd: Twist):
        self.cmd_pub_model.publish(cmd)
        self.cmd_pub_direct.publish(cmd)

    def odom_cb(self, msg: Odometry):
        self.uav_pose = msg.pose.pose

    def world_pose_cb(self, msg: TFMessage):
        tf = extract_model_transform(msg, self.uav_name)
        if tf is None:
            return
        t = tf.transform.translation
        r = tf.transform.rotation
        self.uav_world_state = {
            "x": float(t.x),
            "y": float(t.y),
            "z": float(t.z),
            "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
        }

    def obstacle_action_cb(self, msg: String):
        self.obstacle_action = msg.data.strip().lower() if msg.data else "clear"

    def obstacle_clearance_cb(self, msg: Vector3):
        self.obstacle_clearance = (float(msg.x), float(msg.y), float(msg.z))

    def _uav_state(self):
        if self.uav_world_state is not None:
            return self.uav_world_state
        if self.uav_pose is None:
            return None
        if self.spawn_xyz is not None:
            return local_pose_to_world(self.uav_pose, self.spawn_xyz, self.spawn_yaw)
        p = self.uav_pose.position
        q = self.uav_pose.orientation
        return {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
        }

    def _obstacle_active(self) -> bool:
        return (self.obstacle_action or "clear") != "clear"

    def _xy_goal_error(self, state):
        dx = float(self.world_goal_xyz[0]) - state["x"]
        dy = float(self.world_goal_xyz[1]) - state["y"]
        return dx, dy, math.hypot(dx, dy)

    def _world_to_body(self, vx_world: float, vy_world: float, yaw: float):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        body_x = cos_yaw * vx_world + sin_yaw * vy_world
        body_y = -sin_yaw * vx_world + cos_yaw * vy_world
        return body_x, body_y

    def _goal_twist(self, state, target_z: float) -> Twist:
        dx, dy, xy_dist = self._xy_goal_error(state)
        desired_vx_world = clamp(self.xy_gain * dx, -self.max_xy_speed, self.max_xy_speed)
        desired_vy_world = clamp(self.xy_gain * dy, -self.max_xy_speed, self.max_xy_speed)
        speed = math.hypot(desired_vx_world, desired_vy_world)
        if speed > self.max_xy_speed and speed > 1e-6:
            scale = self.max_xy_speed / speed
            desired_vx_world *= scale
            desired_vy_world *= scale
        body_x, body_y = self._world_to_body(desired_vx_world, desired_vy_world, state["yaw"])

        error_z = target_z - state["z"]
        linear_z = clamp(self.z_gain * error_z, -self.max_z_speed, self.max_z_speed)
        yaw_target = math.atan2(dy, dx) if xy_dist > 1e-6 else state["yaw"]
        yaw_error = wrap_angle(yaw_target - state["yaw"])
        yaw_cmd = clamp(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)

        cmd = Twist()
        cmd.linear.x = float(body_x)
        cmd.linear.y = float(body_y)
        cmd.linear.z = float(linear_z)
        cmd.angular.z = float(yaw_cmd)
        return cmd

    def _avoid_twist(self, state) -> Twist:
        cmd = Twist()
        direction = self.avoid_direction or "up"
        cmd.linear.x = float(self.avoid_forward_speed)
        if direction == "left":
            cmd.linear.y = float(self.avoid_lateral_speed)
            cmd.linear.z = float(0.25 * self.avoid_climb_speed)
        elif direction == "right":
            cmd.linear.y = float(-self.avoid_lateral_speed)
            cmd.linear.z = float(0.25 * self.avoid_climb_speed)
        else:
            cmd.linear.y = 0.0
            cmd.linear.z = float(self.avoid_climb_speed)

        dx, dy, xy_dist = self._xy_goal_error(state)
        yaw_target = math.atan2(dy, dx) if xy_dist > 1e-6 else state["yaw"]
        yaw_error = wrap_angle(yaw_target - state["yaw"])
        cmd.angular.z = float(clamp(0.5 * self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate))
        return cmd

    def step(self):
        self.publish_state()
        state = self._uav_state()
        if state is None:
            return

        if self.arrived:
            self.publish_cmd(Twist())
            return

        now = time.monotonic()
        dx, dy, xy_dist = self._xy_goal_error(state)
        alt_err_to_ground_goal = float(self.world_goal_xyz[2]) - state["z"]

        if self.state == "takeoff":
            if (now - self.last_diag_log) >= 2.0:
                self.get_logger().info(
                    "uav_goal "
                    f"state={self.state} "
                    f"pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                    f"goal=({self.world_goal_xyz[0]:.2f},{self.world_goal_xyz[1]:.2f},{self.world_goal_xyz[2]:.2f}) "
                    f"xy_dist={xy_dist:.2f} obstacle={self.obstacle_action}"
                )
                self.last_diag_log = now
            if state["z"] >= (self.cruise_altitude - 0.5):
                self.held_altitude_floor = float(state["z"])
                self.state = "go_to_goal"
            else:
                cmd = Twist()
                cmd.linear.z = float(self.max_z_speed)
                self.publish_cmd(cmd)
                return

        if self.state == "descend":
            self.held_altitude_floor = None
            if self._obstacle_active():
                self.state = "avoid"
                self.avoid_direction = self.obstacle_action.split("_")[-1]
                self.avoid_clear_since = None
            elif xy_dist <= self.goal_xy_tolerance and abs(alt_err_to_ground_goal) <= self.landing_altitude_tolerance:
                self.arrived = True
                self.get_logger().info(
                    "UAV arrival triggered: "
                    f"pose=({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f}) "
                    f"goal=({self.world_goal_xyz[0]:.3f}, {self.world_goal_xyz[1]:.3f}, {self.world_goal_xyz[2]:.3f}) "
                    f"remaining_xy={xy_dist:.3f}"
                )
                self.publish_cmd(Twist())
                return
            else:
                self.publish_cmd(self._goal_twist(state, float(self.world_goal_xyz[2]) + 0.1))
                return

        if self._obstacle_active() and self.state != "takeoff":
            self.state = "avoid"
            self.avoid_direction = self.obstacle_action.split("_")[-1]
            self.avoid_clear_since = None

        if self.state == "avoid":
            if self.held_altitude_floor is not None:
                self.held_altitude_floor = max(self.held_altitude_floor, float(state["z"]))
            if not self._obstacle_active():
                if self.avoid_clear_since is None:
                    self.avoid_clear_since = now
                elif (now - self.avoid_clear_since) >= self.avoid_clear_hold_seconds:
                    self.state = "descend" if xy_dist <= self.descend_radius else "go_to_goal"
                    self.avoid_direction = None
                    self.avoid_clear_since = None
                    self.publish_cmd(self._goal_twist(
                        state,
                        float(self.world_goal_xyz[2]) + 0.1 if self.state == "descend" else self.cruise_altitude,
                    ))
                    return
            else:
                self.avoid_clear_since = None
                self.avoid_direction = self.obstacle_action.split("_")[-1]
            self.publish_cmd(self._avoid_twist(state))
            return

        if xy_dist <= self.descend_radius:
            self.state = "descend"
            self.publish_cmd(self._goal_twist(state, float(self.world_goal_xyz[2]) + 0.1))
            return

        self.state = "go_to_goal"
        target_z = self.cruise_altitude
        if self.held_altitude_floor is not None:
            target_z = max(target_z, self.held_altitude_floor)
            self.held_altitude_floor = max(self.held_altitude_floor, float(state["z"]))
        self.publish_cmd(self._goal_twist(state, target_z))

        if (now - self.last_diag_log) >= 2.0:
            self.get_logger().info(
                "uav_goal "
                f"state={self.state} "
                f"pose=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                f"goal=({self.world_goal_xyz[0]:.2f},{self.world_goal_xyz[1]:.2f},{self.world_goal_xyz[2]:.2f}) "
                f"xy_dist={xy_dist:.2f} obstacle={self.obstacle_action}"
            )
            self.last_diag_log = now
