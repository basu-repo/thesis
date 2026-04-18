"""Simple UAV follower used during live simulation.

The UAV should stay above the Husky, face the Husky, and recover altitude
aggressively if it drops too low while chasing.
"""

import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def xy_distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def clamp_vector(x, y, max_mag):
    mag = math.hypot(x, y)
    if mag <= max_mag or mag <= 1e-9:
        return x, y
    scale = max_mag / mag
    return x * scale, y * scale


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


class UavFollower(Node):
    """Keep the UAV above the Husky and yawed toward it."""

    def __init__(
        self,
        node_name: str = "uav_follower",
        husky_odom_topic: str = "/model/husky_local/odometry",
        uav_odom_topic: str = "/model/uav1/odometry",
        world_pose_topic: str | None = None,
        husky_model_name: str = "husky_local",
        uav_model_name: str = "uav1",
        uav_name: str = "uav1",
        follow_distance: float = 0.0,
        follow_height: float = 18.0,
        update_period: float = 0.1,
        max_xy_speed: float = 7.0,
        max_z_speed: float = 1.2,
        max_yaw_rate: float = 0.9,
        xy_gain: float = 1.8,
        z_gain: float = 0.35,
        yaw_gain: float = 0.8,
        heading_align_gain: float = 0.0,
        min_forward_speed: float = 0.0,
        target_smoothing: float = 1.0,
        xy_deadband: float = 0.02,
        z_deadband: float = 0.15,
        yaw_deadband: float = 0.18,
        min_track_speed: float = 0.0,
        catchup_distance: float = 3.0,
        catchup_xy_gain: float = 0.8,
        catchup_max_xy_speed: float = 2.4,
        reenable_period: float = 2.0,
        takeoff_hold_seconds: float = 0.0,
        altitude_tolerance: float = 0.4,
        min_follow_altitude: float = 2.0,
        ready_topic: str = "/uav1/ready",
        husky_spawn_xyz: tuple[float, float, float] | None = None,
        husky_spawn_yaw: float = 0.0,
        uav_spawn_xyz: tuple[float, float, float] | None = None,
        uav_spawn_yaw: float = 0.0,
        mission_goal_xyz: tuple[float, float, float] | None = None,
        path_start_xyz: tuple[float, float, float] | None = None,
        path_goal_xyz: tuple[float, float, float] | None = None,
        goal_blend_start_progress: float = 0.7,
    ):
        super().__init__(node_name)
        self.uav_name = uav_name
        self.world_pose_topic = world_pose_topic
        self.husky_model_name = husky_model_name
        self.uav_model_name = uav_model_name
        self.follow_distance = follow_distance
        self.follow_height = follow_height
        self.max_xy_speed = max_xy_speed
        self.max_z_speed = max_z_speed
        self.max_yaw_rate = max_yaw_rate
        self.xy_gain = xy_gain
        self.z_gain = z_gain
        self.yaw_gain = yaw_gain
        self.xy_deadband = xy_deadband
        self.z_deadband = z_deadband
        self.yaw_deadband = yaw_deadband
        self.min_follow_altitude = min_follow_altitude
        self.ready_topic = ready_topic
        self.husky_spawn_xyz = husky_spawn_xyz
        self.husky_spawn_yaw = husky_spawn_yaw
        self.uav_spawn_xyz = uav_spawn_xyz
        self.uav_spawn_yaw = uav_spawn_yaw
        self.min_world_altitude = 4.0
        self.catchup_height_buffer = 6.0
        self.state_log_period = 2.0
        self.follow_log_period = 1.0

        self.husky_pose = None
        self.husky_twist = None
        self.uav_pose = None
        self.husky_world_state = None
        self.uav_world_state = None
        self.last_state_log_time = 0.0
        self.last_follow_log_time = 0.0
        self.ready_sent = False

        self.cmd_pub_model = self.create_publisher(Twist, f"/model/{self.uav_name}/command/twist", 10)
        self.cmd_pub_direct = self.create_publisher(Twist, f"/{self.uav_name}/command/twist", 10)
        self.enable_pub_model = self.create_publisher(Bool, f"/model/{self.uav_name}/enable", 10)
        self.enable_pub_direct = self.create_publisher(Bool, f"/{self.uav_name}/enable", 10)
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, 10)

        self.create_subscription(Odometry, husky_odom_topic, self.husky_odom_cb, 10)
        self.create_subscription(Odometry, uav_odom_topic, self.uav_odom_cb, 10)
        if self.world_pose_topic is not None:
            self.create_subscription(TFMessage, self.world_pose_topic, self.world_pose_cb, 10)
        self.create_timer(update_period, self.follow_husky)
        self.create_timer(reenable_period, self.enable_controller)
        self.enable_controller()
        self.get_logger().info(
            "UAV follower started: "
            f"follow_distance={self.follow_distance:.2f}m "
            f"follow_height={self.follow_height:.2f}m "
            f"cmd_topics=(/model/{self.uav_name}/command/twist, /{self.uav_name}/command/twist) "
            f"enable_topics=(/model/{self.uav_name}/enable, /{self.uav_name}/enable)"
        )

    def husky_odom_cb(self, msg):
        self.husky_pose = msg.pose.pose
        self.husky_twist = msg.twist.twist

    def uav_odom_cb(self, msg):
        self.uav_pose = msg.pose.pose

    def world_pose_cb(self, msg: TFMessage):
        husky_tf = extract_model_transform(msg, self.husky_model_name)
        uav_tf = extract_model_transform(msg, self.uav_model_name)

        if husky_tf is not None:
            t = husky_tf.transform.translation
            r = husky_tf.transform.rotation
            self.husky_world_state = {
                "x": float(t.x),
                "y": float(t.y),
                "z": float(t.z),
                "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
            }
        if uav_tf is not None:
            t = uav_tf.transform.translation
            r = uav_tf.transform.rotation
            self.uav_world_state = {
                "x": float(t.x),
                "y": float(t.y),
                "z": float(t.z),
                "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
            }

    def _husky_state(self):
        if self.husky_world_state is not None:
            return self.husky_world_state, "world_pose"
        if self.husky_pose is None:
            return None, "missing"
        if self.husky_spawn_xyz is not None:
            return local_pose_to_world(self.husky_pose, self.husky_spawn_xyz, self.husky_spawn_yaw), "spawn_corrected_odom"
        p = self.husky_pose.position
        q = self.husky_pose.orientation
        return {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
        }, "odom"

    def _uav_state(self):
        if self.uav_world_state is not None:
            return self.uav_world_state, "world_pose"
        if self.uav_pose is None:
            return None, "missing"
        if self.uav_spawn_xyz is not None:
            return local_pose_to_world(self.uav_pose, self.uav_spawn_xyz, self.uav_spawn_yaw), "spawn_corrected_odom"
        p = self.uav_pose.position
        q = self.uav_pose.orientation
        return {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
        }, "odom"

    def enable_controller(self):
        msg = Bool()
        msg.data = True
        self.enable_pub_model.publish(msg)
        self.enable_pub_direct.publish(msg)
        self.get_logger().info(
            f"UAV controller enable sent on /model/{self.uav_name}/enable and /{self.uav_name}/enable"
        )

    def follow_husky(self):
        husky_state, husky_source = self._husky_state()
        uav_state, uav_source = self._uav_state()
        if husky_state is None or uav_state is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_state_log_time >= self.state_log_period:
            self.get_logger().info(
                f"uav_state_source husky={husky_source} uav={uav_source} "
                f"world_topic={'on' if self.world_pose_topic is not None else 'off'}"
            )
            self.last_state_log_time = now

        husky_yaw = husky_state["yaw"]
        uav_yaw = uav_state["yaw"]

        target_x = husky_state["x"] + self.follow_distance * math.cos(husky_yaw)
        target_y = husky_state["y"] + self.follow_distance * math.sin(husky_yaw)

        full_target_z = max(husky_state["z"] + self.follow_height, self.min_world_altitude)
        catchup_target_z = max(
            full_target_z - 1.0,
            husky_state["z"] + self.min_follow_altitude + self.catchup_height_buffer,
            self.min_world_altitude,
        )

        error_x_world = target_x - uav_state["x"]
        error_y_world = target_y - uav_state["y"]
        xy_error = math.hypot(error_x_world, error_y_world)

        if xy_error > 8.0:
            alt_mode = "catchup"
            target_z = catchup_target_z
        else:
            alt_mode = "full"
            target_z = full_target_z

        error_z = target_z - uav_state["z"]
        altitude_recovery = uav_state["z"] < (self.min_world_altitude + 1.0) or error_z > 6.0

        husky_vx_world = 0.0
        husky_vy_world = 0.0
        if self.husky_twist is not None:
            husky_vx_body = self.husky_twist.linear.x
            husky_vy_body = self.husky_twist.linear.y
            husky_vx_world = math.cos(husky_yaw) * husky_vx_body - math.sin(husky_yaw) * husky_vy_body
            husky_vy_world = math.sin(husky_yaw) * husky_vx_body + math.cos(husky_yaw) * husky_vy_body

        desired_vx_world = husky_vx_world + self.xy_gain * error_x_world
        desired_vy_world = husky_vy_world + self.xy_gain * error_y_world
        desired_vx_world, desired_vy_world = clamp_vector(desired_vx_world, desired_vy_world, self.max_xy_speed)

        if altitude_recovery:
            desired_vx_world *= 0.10
            desired_vy_world *= 0.10
            alt_mode = "altitude_recovery"

        cos_yaw = math.cos(uav_yaw)
        sin_yaw = math.sin(uav_yaw)
        cmd_body_x = cos_yaw * desired_vx_world + sin_yaw * desired_vy_world
        cmd_body_y = -sin_yaw * desired_vx_world + cos_yaw * desired_vy_world
        cmd_body_x, cmd_body_y = clamp_vector(cmd_body_x, cmd_body_y, self.max_xy_speed)

        linear_z = clamp(self.z_gain * error_z, -self.max_z_speed, self.max_z_speed)
        if altitude_recovery and error_z > 0.0:
            linear_z = max(linear_z, 0.85 * self.max_z_speed)
        if uav_state["z"] <= self.min_world_altitude + 0.25:
            linear_z = max(linear_z, self.max_z_speed)
        elif uav_state["z"] <= self.min_world_altitude + 0.5:
            linear_z = max(0.0, linear_z)
        if abs(error_z) < self.z_deadband:
            linear_z = 0.0

        if xy_error > 1e-6:
            yaw_target = math.atan2(error_y_world, error_x_world)
        else:
            yaw_target = husky_yaw
        yaw_error = wrap_angle(yaw_target - uav_yaw)
        yaw_cmd = clamp(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)
        if abs(yaw_error) < self.yaw_deadband:
            yaw_cmd = 0.0

        cmd_msg = Twist()
        cmd_msg.linear.x = float(cmd_body_x)
        cmd_msg.linear.y = float(cmd_body_y)
        cmd_msg.linear.z = float(linear_z)
        cmd_msg.angular.z = float(yaw_cmd)
        self.cmd_pub_model.publish(cmd_msg)
        self.cmd_pub_direct.publish(cmd_msg)

        ready_now = (
            uav_state["z"] >= max(self.min_world_altitude, husky_state["z"] + self.min_follow_altitude)
            and abs(error_z) <= max(0.5, self.z_deadband)
        )
        ready_msg = Bool()
        ready_msg.data = ready_now
        self.ready_pub.publish(ready_msg)
        if ready_now and not self.ready_sent:
            self.get_logger().info(f"UAV ready on {self.ready_topic}")
            self.ready_sent = True
        elif not ready_now:
            self.ready_sent = False

        if now - self.last_follow_log_time >= self.follow_log_period:
            self.get_logger().info(
                "uav_follow "
                f"husky=({husky_state['x']:.2f},{husky_state['y']:.2f},{husky_state['z']:.2f}) "
                f"uav=({uav_state['x']:.2f},{uav_state['y']:.2f},{uav_state['z']:.2f}) "
                f"target=({target_x:.2f},{target_y:.2f},{target_z:.2f}) "
                f"err_xy={xy_error:.2f} err_z={error_z:.2f} "
                f"z_floor={self.min_world_altitude:.2f} alt_mode={alt_mode} "
                f"cmd_body=({cmd_body_x:.2f},{cmd_body_y:.2f},{linear_z:.2f}) "
                f"cmd_world=({desired_vx_world:.2f},{desired_vy_world:.2f}) "
                f"yaw_target={yaw_target:.2f} yaw_cmd={yaw_cmd:.2f} "
                f"alt_recovery={altitude_recovery} ready={ready_now}"
            )
            self.last_follow_log_time = now
