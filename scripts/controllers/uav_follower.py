"""UAV follower controller used during live simulation.

The UAV tracks the Husky with a configurable offset and altitude so it can act
as an aerial observer while the ground vehicles execute their mission.
"""

import math
import subprocess

from nav_msgs.msg import Odometry
from rclpy.node import Node


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class UavFollower(Node):
    """Track the Husky smoothly by publishing body-frame UAV velocity commands."""

    def __init__(
        self,
        node_name: str = "uav_follower",
        husky_odom_topic: str = "/model/husky_local/odometry",
        uav_odom_topic: str = "/model/uav1/odometry",
        uav_name: str = "uav1",
        follow_distance: float = 0.0,
        follow_height: float = 6.0,
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
    ):
        super().__init__(node_name)
        self.uav_name = uav_name
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

        self.husky_pose = None
        self.husky_twist = None
        self.uav_pose = None
        self.filtered_target = None
        self.create_subscription(Odometry, husky_odom_topic, self.husky_odom_cb, 10)
        self.create_subscription(Odometry, uav_odom_topic, self.uav_odom_cb, 10)
        self.create_timer(update_period, self.follow_husky)
        self.create_timer(reenable_period, self.enable_controller)
        self.enable_controller()

    def husky_odom_cb(self, msg):
        self.husky_pose = msg.pose.pose
        self.husky_twist = msg.twist.twist

    def uav_odom_cb(self, msg):
        self.uav_pose = msg.pose.pose

    def enable_controller(self):
        cmd = (
            "ign topic "
            f"-t /{self.uav_name}/enable "
            "-m ignition.msgs.Boolean "
            "-p 'data: true'"
        )
        subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def follow_husky(self):
        if self.husky_pose is None or self.uav_pose is None:
            return

        husky_position = self.husky_pose.position
        husky_orientation = self.husky_pose.orientation
        husky_twist = self.husky_twist
        uav_position = self.uav_pose.position
        uav_orientation = self.uav_pose.orientation

        husky_yaw = quaternion_to_yaw(
            husky_orientation.x,
            husky_orientation.y,
            husky_orientation.z,
            husky_orientation.w,
        )
        uav_yaw = quaternion_to_yaw(
            uav_orientation.x,
            uav_orientation.y,
            uav_orientation.z,
            uav_orientation.w,
        )

        raw_target_x = husky_position.x - self.follow_distance * math.cos(husky_yaw)
        raw_target_y = husky_position.y - self.follow_distance * math.sin(husky_yaw)
        raw_target_z = max(husky_position.z + self.follow_height, 1.5)

        if self.filtered_target is None:
            self.filtered_target = {"x": raw_target_x, "y": raw_target_y, "z": raw_target_z}
        else:
            alpha = self.target_smoothing
            self.filtered_target["x"] += alpha * (raw_target_x - self.filtered_target["x"])
            self.filtered_target["y"] += alpha * (raw_target_y - self.filtered_target["y"])
            self.filtered_target["z"] += alpha * (raw_target_z - self.filtered_target["z"])

        error_x_world = self.filtered_target["x"] - uav_position.x
        error_y_world = self.filtered_target["y"] - uav_position.y
        error_z = self.filtered_target["z"] - uav_position.z
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

        xy_gain = self.xy_gain
        max_xy_speed = self.max_xy_speed
        if xy_error > self.catchup_distance:
            xy_gain = self.catchup_xy_gain
            max_xy_speed = self.catchup_max_xy_speed

        desired_vx_world = husky_vx_world + xy_gain * error_x_world
        desired_vy_world = husky_vy_world + xy_gain * error_y_world

        cos_yaw = math.cos(uav_yaw)
        sin_yaw = math.sin(uav_yaw)
        linear_x = cos_yaw * desired_vx_world + sin_yaw * desired_vy_world
        linear_y = -sin_yaw * desired_vx_world + cos_yaw * desired_vy_world
        linear_x = clamp(linear_x, -max_xy_speed, max_xy_speed)
        linear_y = clamp(linear_y, -max_xy_speed, max_xy_speed)
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
