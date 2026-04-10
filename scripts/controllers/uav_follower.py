"""UAV follower controller used during live simulation.

The UAV tracks the Husky with a configurable offset and altitude so it can act
as an aerial observer while the ground vehicles execute their mission.
"""

import math
import subprocess

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
    selected_base_link = None
    for transform in msg.transforms:
        child = transform.child_frame_id or ""
        child_parts = [part for part in child.split("/") if part]
        if (
            child == model_name
            or child.endswith(f"/{model_name}")
            or model_name in child_parts
        ) and not child.endswith("/base_link"):
            return transform
        if (
            child.endswith("/base_link")
            and model_name in child_parts
        ):
            selected_base_link = transform
    return selected_base_link


class UavFollower(Node):
    """Track the Husky smoothly by publishing body-frame UAV velocity commands."""

    def __init__(
        self,
        node_name: str = "uav_follower",
        husky_odom_topic: str = "/model/husky_local/odometry",
        uav_odom_topic: str = "/model/uav1/odometry",
        world_pose_topic: str | None = None,
        husky_model_name: str = "husky_local",
        uav_model_name: str = "uav1",
        uav_name: str = "uav1",
        follow_distance: float = 2.0,
        follow_height: float = 10.0,
        update_period: float = 0.1,
        max_xy_speed: float = 1.6,
        max_z_speed: float = 0.6,
        max_yaw_rate: float = 0.6,
        xy_gain: float = 0.45,
        z_gain: float = 0.35,
        yaw_gain: float = 0.18,
        target_smoothing: float = 0.1,
        xy_deadband: float = 0.12,
        z_deadband: float = 0.15,
        yaw_deadband: float = 0.18,
        min_track_speed: float = 0.05,
        catchup_distance: float = 3.0,
        catchup_xy_gain: float = 0.8,
        catchup_max_xy_speed: float = 2.4,
        reenable_period: float = 2.0,
        takeoff_hold_seconds: float = 2.0,
        altitude_tolerance: float = 0.4,
        min_follow_altitude: float = 2.0,
        ready_topic: str = "/uav1/ready",
        husky_spawn_xyz: tuple[float, float, float] | None = None,
        husky_spawn_yaw: float = 0.0,
        uav_spawn_xyz: tuple[float, float, float] | None = None,
        uav_spawn_yaw: float = 0.0,
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
        self.target_smoothing = target_smoothing
        self.xy_deadband = xy_deadband
        self.z_deadband = z_deadband
        self.yaw_deadband = yaw_deadband
        self.min_track_speed = min_track_speed
        self.catchup_distance = catchup_distance
        self.catchup_xy_gain = catchup_xy_gain
        self.catchup_max_xy_speed = catchup_max_xy_speed
        self.takeoff_hold_seconds = takeoff_hold_seconds
        self.altitude_tolerance = altitude_tolerance
        self.min_follow_altitude = min_follow_altitude
        self.ready_topic = ready_topic
        self.husky_spawn_xyz = husky_spawn_xyz
        self.husky_spawn_yaw = husky_spawn_yaw
        self.uav_spawn_xyz = uav_spawn_xyz
        self.uav_spawn_yaw = uav_spawn_yaw

        self.husky_pose = None
        self.husky_twist = None
        self.uav_pose = None
        self.husky_world_state = None
        self.uav_world_state = None
        self.filtered_target = None
        self.takeoff_start_time = None
        self.ready_sent = False
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, 10)
        self.create_subscription(Odometry, husky_odom_topic, self.husky_odom_cb, 10)
        self.create_subscription(Odometry, uav_odom_topic, self.uav_odom_cb, 10)
        if self.world_pose_topic is not None:
            self.create_subscription(TFMessage, self.world_pose_topic, self.world_pose_cb, 10)
        self.create_timer(update_period, self.follow_husky)
        self.create_timer(reenable_period, self.enable_controller)
        self.enable_controller()

    def husky_odom_cb(self, msg):
        self.husky_pose = msg.pose.pose
        self.husky_twist = msg.twist.twist

    def uav_odom_cb(self, msg):
        self.uav_pose = msg.pose.pose
        if self.takeoff_start_time is None:
            self.takeoff_start_time = self.get_clock().now().nanoseconds / 1e9

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
            if self.takeoff_start_time is None:
                self.takeoff_start_time = self.get_clock().now().nanoseconds / 1e9

    def _husky_state(self):
        if self.husky_world_state is not None:
            return self.husky_world_state
        if self.husky_pose is None:
            return None
        if self.husky_spawn_xyz is not None:
            return local_pose_to_world(self.husky_pose, self.husky_spawn_xyz, self.husky_spawn_yaw)
        p = self.husky_pose.position
        q = self.husky_pose.orientation
        return {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
        }

    def _uav_state(self):
        if self.uav_world_state is not None:
            return self.uav_world_state
        if self.uav_pose is None:
            return None
        if self.uav_spawn_xyz is not None:
            return local_pose_to_world(self.uav_pose, self.uav_spawn_xyz, self.uav_spawn_yaw)
        p = self.uav_pose.position
        q = self.uav_pose.orientation
        return {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
        }

    def enable_controller(self):
        cmd = (
            "ign topic "
            f"-t /{self.uav_name}/enable "
            "-m ignition.msgs.Boolean "
            "-p 'data: true'"
        )
        subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def follow_husky(self):
        husky_state = self._husky_state()
        uav_state = self._uav_state()
        if husky_state is None or uav_state is None:
            return

        husky_twist = self.husky_twist
        husky_yaw = husky_state["yaw"]
        uav_yaw = uav_state["yaw"]

        raw_target_x = husky_state["x"] + self.follow_distance * math.cos(husky_yaw)
        raw_target_y = husky_state["y"] + self.follow_distance * math.sin(husky_yaw)
        raw_target_z = max(husky_state["z"] + self.follow_height, 1.5)

        now = self.get_clock().now().nanoseconds / 1e9
        altitude_error = raw_target_z - uav_state["z"]
        ready_now = False
        if self.takeoff_start_time is not None:
            min_safe_altitude = max(husky_state["z"] + self.min_follow_altitude, 1.5)
            ready_now = (
                (now - self.takeoff_start_time) >= self.takeoff_hold_seconds
                and uav_state["z"] >= min_safe_altitude
                and abs(altitude_error) <= self.altitude_tolerance
            )

        if self.filtered_target is None:
            self.filtered_target = {"x": raw_target_x, "y": raw_target_y, "z": raw_target_z}
        else:
            alpha = self.target_smoothing
            self.filtered_target["x"] += alpha * (raw_target_x - self.filtered_target["x"])
            self.filtered_target["y"] += alpha * (raw_target_y - self.filtered_target["y"])
            self.filtered_target["z"] += alpha * (raw_target_z - self.filtered_target["z"])

        error_x_world = self.filtered_target["x"] - uav_state["x"]
        error_y_world = self.filtered_target["y"] - uav_state["y"]
        error_z = self.filtered_target["z"] - uav_state["z"]
        xy_error = math.hypot(error_x_world, error_y_world)

        if husky_twist is None:
            husky_vx_body = 0.0
            husky_vy_body = 0.0
            husky_yaw_rate = 0.0
        else:
            husky_vx_body = husky_twist.linear.x
            husky_vy_body = husky_twist.linear.y
            husky_yaw_rate = husky_twist.angular.z

        husky_vx_world = math.cos(husky_yaw) * husky_vx_body - math.sin(husky_yaw) * husky_vy_body
        husky_vy_world = math.sin(husky_yaw) * husky_vx_body + math.cos(husky_yaw) * husky_vy_body

        # Pure tracking controller: match UGV world velocity, then add only the
        # correction needed to stay above the target point.
        desired_vx_world = husky_vx_world + self.xy_gain * error_x_world
        desired_vy_world = husky_vy_world + self.xy_gain * error_y_world

        cos_yaw = math.cos(uav_yaw)
        sin_yaw = math.sin(uav_yaw)
        linear_x = cos_yaw * desired_vx_world + sin_yaw * desired_vy_world
        linear_y = -sin_yaw * desired_vx_world + cos_yaw * desired_vy_world
        linear_x = clamp(linear_x, -self.max_xy_speed, self.max_xy_speed)
        linear_y = clamp(linear_y, -self.max_xy_speed, self.max_xy_speed)
        linear_z = clamp(self.z_gain * error_z, -self.max_z_speed, self.max_z_speed)
        yaw_error = wrap_angle(husky_yaw - uav_yaw)
        angular_z = clamp(husky_yaw_rate + self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)
        husky_speed = math.hypot(husky_vx_world, husky_vy_world)

        if xy_error < self.xy_deadband and husky_speed < self.min_track_speed:
            linear_x = 0.0
            linear_y = 0.0
        if abs(error_z) < self.z_deadband:
            linear_z = 0.0
        if abs(yaw_error) < self.yaw_deadband:
            angular_z = 0.0

        msg = (
            f"linear: {{x: {linear_x}, y: {linear_y}, z: {linear_z}}} "
            f"angular: {{x: 0.0, y: 0.0, z: {angular_z}}}"
        )
        cmd = f"ign topic -t /{self.uav_name}/command/twist -m ignition.msgs.Twist -p '{msg}'"
        subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ready_msg = Bool()
        ready_msg.data = ready_now
        self.ready_pub.publish(ready_msg)
        if ready_now and not self.ready_sent:
            self.get_logger().info(
                f"UAV ready: altitude reached and follower active on {self.ready_topic}"
            )
            self.ready_sent = True
