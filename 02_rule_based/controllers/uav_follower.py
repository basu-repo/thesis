# # """Simple UAV follower used during live simulation.

# # The UAV follows the ego Husky from above and provides aerial/contextual data
# # for the hybrid trajectory-prediction dataset.

# # Role in the dataset:
# # - keep UAV1 close enough to the ego Husky to observe the surrounding/front area,
# # - publish UAV odometry through Gazebo/ROS bridge,
# # - keep UAV forward pointcloud topics active,
# # - publish a simple /uav1/ready status.

# # This node is intentionally not a learned controller. It is only a stable
# # rule-based support behavior for data collection.
# # """

# # import math

# # from geometry_msgs.msg import Twist
# # from nav_msgs.msg import Odometry
# # from rclpy.node import Node
# # from std_msgs.msg import Bool
# # from tf2_msgs.msg import TFMessage


# # def quaternion_to_yaw(x, y, z, w):
# #     siny_cosp = 2.0 * (w * z + x * y)
# #     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
# #     return math.atan2(siny_cosp, cosy_cosp)


# # def clamp(value, min_value, max_value):
# #     return max(min(value, max_value), min_value)


# # def wrap_angle(angle):
# #     return math.atan2(math.sin(angle), math.cos(angle))


# # def clamp_vector(x, y, max_mag):
# #     mag = math.hypot(x, y)

# #     if mag <= max_mag or mag <= 1e-9:
# #         return x, y

# #     scale = max_mag / mag
# #     return x * scale, y * scale


# # def local_pose_to_world(pose, spawn_xyz, spawn_yaw):
# #     """Convert spawn-relative odometry pose into Gazebo world-like coordinates."""
# #     p = pose.position
# #     q = pose.orientation

# #     local_x = float(p.x)
# #     local_y = float(p.y)
# #     local_z = float(p.z)

# #     cos_yaw = math.cos(spawn_yaw)
# #     sin_yaw = math.sin(spawn_yaw)

# #     world_x = spawn_xyz[0] + cos_yaw * local_x - sin_yaw * local_y
# #     world_y = spawn_xyz[1] + sin_yaw * local_x + cos_yaw * local_y
# #     world_z = spawn_xyz[2] + local_z

# #     local_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

# #     return {
# #         "x": world_x,
# #         "y": world_y,
# #         "z": world_z,
# #         "yaw": wrap_angle(spawn_yaw + local_yaw),
# #     }


# # def extract_model_transform(msg: TFMessage, model_name: str):
# #     """Extract the transform corresponding to a Gazebo model.

# #     Prefer the model's base_link transform when available, otherwise fall back
# #     to the model-level transform.
# #     """
# #     selected_model = None
# #     selected_base_link = None

# #     for transform in msg.transforms:
# #         child = transform.child_frame_id or ""
# #         child_parts = [part for part in child.split("/") if part]

# #         if model_name not in child_parts:
# #             continue

# #         if child_parts and child_parts[-1] == "base_link":
# #             selected_base_link = transform
# #             continue

# #         if (
# #             child == model_name
# #             or child.endswith(f"/{model_name}")
# #             or (child_parts and child_parts[-1] == model_name)
# #         ):
# #             selected_model = transform

# #     return selected_base_link or selected_model


# # class UavFollower(Node):
# #     """Keep the UAV above/near the Husky and yawed toward its tracking target."""

# #     def __init__(
# #         self,
# #         node_name: str = "uav_follower",
# #         husky_odom_topic: str = "/model/husky_local/odometry",
# #         uav_odom_topic: str = "/model/uav1/odometry",
# #         world_pose_topic: str | None = None,
# #         husky_model_name: str = "husky_local",
# #         uav_model_name: str = "uav1",
# #         uav_name: str = "uav1",
# #         follow_distance: float = 0.0,
# #         follow_height: float = 12.0,
# #         update_period: float = 0.1,
# #         max_xy_speed: float = 7.0,
# #         max_z_speed: float = 1.2,
# #         max_yaw_rate: float = 0.9,
# #         xy_gain: float = 1.8,
# #         z_gain: float = 0.35,
# #         yaw_gain: float = 0.8,
# #         heading_align_gain: float = 0.0,
# #         min_forward_speed: float = 0.0,
# #         target_smoothing: float = 1.0,
# #         xy_deadband: float = 0.02,
# #         z_deadband: float = 0.15,
# #         yaw_deadband: float = 0.18,
# #         min_track_speed: float = 0.0,
# #         catchup_distance: float = 8.0,
# #         catchup_height_buffer: float = 6.0,
# #         reenable_period: float = 2.0,
# #         takeoff_hold_seconds: float = 0.0,
# #         altitude_tolerance: float = 0.4,
# #         min_follow_altitude: float = 2.0,
# #         ready_topic: str = "/uav1/ready",
# #         husky_spawn_xyz: tuple[float, float, float] | None = None,
# #         husky_spawn_yaw: float = 0.0,
# #         uav_spawn_xyz: tuple[float, float, float] | None = None,
# #         uav_spawn_yaw: float = 0.0,
# #         mission_goal_xyz: tuple[float, float, float] | None = None,
# #         path_start_xyz: tuple[float, float, float] | None = None,
# #         path_goal_xyz: tuple[float, float, float] | None = None,
# #         goal_blend_start_progress: float = 0.7,
# #     ):
# #         super().__init__(node_name)

# #         self.uav_name = uav_name

# #         self.world_pose_topic = world_pose_topic
# #         self.husky_model_name = husky_model_name
# #         self.uav_model_name = uav_model_name

# #         self.follow_distance = float(follow_distance)
# #         self.follow_height = float(follow_height)

# #         self.max_xy_speed = float(max_xy_speed)
# #         self.max_z_speed = float(max_z_speed)
# #         self.max_yaw_rate = float(max_yaw_rate)

# #         self.xy_gain = float(xy_gain)
# #         self.z_gain = float(z_gain)
# #         self.yaw_gain = float(yaw_gain)

# #         # Kept for compatibility with existing runner arguments.
# #         self.heading_align_gain = float(heading_align_gain)
# #         self.min_forward_speed = float(min_forward_speed)
# #         self.target_smoothing = float(target_smoothing)
# #         self.min_track_speed = float(min_track_speed)
# #         self.takeoff_hold_seconds = float(takeoff_hold_seconds)
# #         self.altitude_tolerance = float(altitude_tolerance)

# #         self.xy_deadband = float(xy_deadband)
# #         self.z_deadband = float(z_deadband)
# #         self.yaw_deadband = float(yaw_deadband)

# #         self.catchup_distance = float(catchup_distance)
# #         self.catchup_height_buffer = float(catchup_height_buffer)

# #         self.min_follow_altitude = float(min_follow_altitude)
# #         self.ready_topic = ready_topic

# #         self.husky_spawn_xyz = husky_spawn_xyz
# #         self.husky_spawn_yaw = float(husky_spawn_yaw)

# #         self.uav_spawn_xyz = uav_spawn_xyz
# #         self.uav_spawn_yaw = float(uav_spawn_yaw)

# #         self.mission_goal_xyz = mission_goal_xyz
# #         self.path_start_xyz = path_start_xyz
# #         self.path_goal_xyz = path_goal_xyz
# #         self.goal_blend_start_progress = float(goal_blend_start_progress)

# #         # Safety floor. Keeps UAV from trying to fly too low in uneven terrain.
# #         self.min_world_altitude = 4.0

# #         # Logging rates.
# #         self.state_log_period = 2.0
# #         self.follow_log_period = 1.0

# #         self.husky_pose = None
# #         self.husky_twist = None
# #         self.uav_pose = None

# #         self.husky_world_state = None
# #         self.uav_world_state = None

# #         self.last_state_log_time = 0.0
# #         self.last_follow_log_time = 0.0

# #         self.ready_sent = False

# #         self.smoothed_target_xy = None

# #         self.cmd_pub_model = self.create_publisher(
# #             Twist,
# #             f"/model/{self.uav_name}/command/twist",
# #             10,
# #         )
# #         self.cmd_pub_direct = self.create_publisher(
# #             Twist,
# #             f"/{self.uav_name}/command/twist",
# #             10,
# #         )

# #         self.enable_pub_model = self.create_publisher(
# #             Bool,
# #             f"/model/{self.uav_name}/enable",
# #             10,
# #         )
# #         self.enable_pub_direct = self.create_publisher(
# #             Bool,
# #             f"/{self.uav_name}/enable",
# #             10,
# #         )

# #         self.ready_pub = self.create_publisher(
# #             Bool,
# #             self.ready_topic,
# #             10,
# #         )

# #         self.create_subscription(
# #             Odometry,
# #             husky_odom_topic,
# #             self.husky_odom_cb,
# #             10,
# #         )
# #         self.create_subscription(
# #             Odometry,
# #             uav_odom_topic,
# #             self.uav_odom_cb,
# #             10,
# #         )

# #         if self.world_pose_topic is not None:
# #             self.create_subscription(
# #                 TFMessage,
# #                 self.world_pose_topic,
# #                 self.world_pose_cb,
# #                 10,
# #             )

# #         self.create_timer(update_period, self.follow_husky)
# #         self.create_timer(reenable_period, self.enable_controller)

# #         self.enable_controller()

# #         self.get_logger().info(
# #             "UAV follower started: "
# #             f"uav={self.uav_name} "
# #             f"follow_distance={self.follow_distance:.2f}m "
# #             f"follow_height={self.follow_height:.2f}m "
# #             f"cmd_topics=(/model/{self.uav_name}/command/twist, /{self.uav_name}/command/twist) "
# #             f"enable_topics=(/model/{self.uav_name}/enable, /{self.uav_name}/enable)"
# #         )

# #     def husky_odom_cb(self, msg):
# #         self.husky_pose = msg.pose.pose
# #         self.husky_twist = msg.twist.twist

# #     def uav_odom_cb(self, msg):
# #         self.uav_pose = msg.pose.pose

# #     def world_pose_cb(self, msg: TFMessage):
# #         husky_tf = extract_model_transform(msg, self.husky_model_name)
# #         uav_tf = extract_model_transform(msg, self.uav_model_name)

# #         if husky_tf is not None:
# #             t = husky_tf.transform.translation
# #             r = husky_tf.transform.rotation

# #             self.husky_world_state = {
# #                 "x": float(t.x),
# #                 "y": float(t.y),
# #                 "z": float(t.z),
# #                 "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
# #             }

# #         if uav_tf is not None:
# #             t = uav_tf.transform.translation
# #             r = uav_tf.transform.rotation

# #             self.uav_world_state = {
# #                 "x": float(t.x),
# #                 "y": float(t.y),
# #                 "z": float(t.z),
# #                 "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
# #             }

# #     def _husky_state(self):
# #         if self.husky_world_state is not None:
# #             return self.husky_world_state, "world_pose"

# #         if self.husky_pose is None:
# #             return None, "missing"

# #         if self.husky_spawn_xyz is not None:
# #             return (
# #                 local_pose_to_world(
# #                     self.husky_pose,
# #                     self.husky_spawn_xyz,
# #                     self.husky_spawn_yaw,
# #                 ),
# #                 "spawn_corrected_odom",
# #             )

# #         p = self.husky_pose.position
# #         q = self.husky_pose.orientation

# #         return (
# #             {
# #                 "x": float(p.x),
# #                 "y": float(p.y),
# #                 "z": float(p.z),
# #                 "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
# #             },
# #             "odom",
# #         )

# #     def _uav_state(self):
# #         if self.uav_world_state is not None:
# #             return self.uav_world_state, "world_pose"

# #         if self.uav_pose is None:
# #             return None, "missing"

# #         if self.uav_spawn_xyz is not None:
# #             return (
# #                 local_pose_to_world(
# #                     self.uav_pose,
# #                     self.uav_spawn_xyz,
# #                     self.uav_spawn_yaw,
# #                 ),
# #                 "spawn_corrected_odom",
# #             )

# #         p = self.uav_pose.position
# #         q = self.uav_pose.orientation

# #         return (
# #             {
# #                 "x": float(p.x),
# #                 "y": float(p.y),
# #                 "z": float(p.z),
# #                 "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
# #             },
# #             "odom",
# #         )

# #     def enable_controller(self):
# #         msg = Bool()
# #         msg.data = True

# #         self.enable_pub_model.publish(msg)
# #         self.enable_pub_direct.publish(msg)

# #         self.get_logger().info(
# #             f"UAV controller enable sent on /model/{self.uav_name}/enable and /{self.uav_name}/enable"
# #         )

# #     def _husky_velocity_world(self, husky_yaw: float):
# #         if self.husky_twist is None:
# #             return 0.0, 0.0

# #         husky_vx_body = float(self.husky_twist.linear.x)
# #         husky_vy_body = float(self.husky_twist.linear.y)

# #         husky_vx_world = (
# #             math.cos(husky_yaw) * husky_vx_body
# #             - math.sin(husky_yaw) * husky_vy_body
# #         )
# #         husky_vy_world = (
# #             math.sin(husky_yaw) * husky_vx_body
# #             + math.cos(husky_yaw) * husky_vy_body
# #         )

# #         return husky_vx_world, husky_vy_world

# #     def _smooth_target(self, target_x: float, target_y: float):
# #         """Smooth target location to reduce UAV jitter.

# #         target_smoothing = 1.0 means no smoothing.
# #         target_smoothing = 0.0 means hold previous target.
# #         """
# #         alpha = clamp(self.target_smoothing, 0.0, 1.0)

# #         if self.smoothed_target_xy is None or alpha >= 0.999:
# #             self.smoothed_target_xy = (target_x, target_y)
# #             return target_x, target_y

# #         old_x, old_y = self.smoothed_target_xy
# #         new_x = alpha * target_x + (1.0 - alpha) * old_x
# #         new_y = alpha * target_y + (1.0 - alpha) * old_y

# #         self.smoothed_target_xy = (new_x, new_y)
# #         return new_x, new_y

# #     def follow_husky(self):
# #         husky_state, husky_source = self._husky_state()
# #         uav_state, uav_source = self._uav_state()

# #         if husky_state is None or uav_state is None:
# #             return

# #         now = self.get_clock().now().nanoseconds / 1e9

# #         if now - self.last_state_log_time >= self.state_log_period:
# #             self.get_logger().info(
# #                 f"uav_state_source husky={husky_source} uav={uav_source} "
# #                 f"world_topic={'on' if self.world_pose_topic is not None else 'off'}"
# #             )
# #             self.last_state_log_time = now

# #         husky_yaw = float(husky_state["yaw"])
# #         uav_yaw = float(uav_state["yaw"])

# #         # Follow target is in front/above the Husky depending on follow_distance.
# #         # follow_distance = 0.0 means directly above Husky.
# #         # follow_distance > 0.0 means slightly ahead of Husky.
# #         # follow_distance < 0.0 means behind Husky.
# #         raw_target_x = husky_state["x"] + self.follow_distance * math.cos(husky_yaw)
# #         raw_target_y = husky_state["y"] + self.follow_distance * math.sin(husky_yaw)

# #         target_x, target_y = self._smooth_target(raw_target_x, raw_target_y)

# #         full_target_z = max(
# #             float(husky_state["z"]) + self.follow_height,
# #             self.min_world_altitude,
# #         )

# #         catchup_target_z = max(
# #             full_target_z - 1.0,
# #             float(husky_state["z"]) + self.min_follow_altitude + self.catchup_height_buffer,
# #             self.min_world_altitude,
# #         )

# #         error_x_world = target_x - float(uav_state["x"])
# #         error_y_world = target_y - float(uav_state["y"])
# #         xy_error = math.hypot(error_x_world, error_y_world)

# #         if xy_error > self.catchup_distance:
# #             alt_mode = "catchup"
# #             target_z = catchup_target_z
# #         else:
# #             alt_mode = "full"
# #             target_z = full_target_z

# #         error_z = target_z - float(uav_state["z"])

# #         altitude_recovery = (
# #             float(uav_state["z"]) < (self.min_world_altitude + 1.0)
# #             or error_z > 6.0
# #         )

# #         husky_vx_world, husky_vy_world = self._husky_velocity_world(husky_yaw)

# #         desired_vx_world = husky_vx_world + self.xy_gain * error_x_world
# #         desired_vy_world = husky_vy_world + self.xy_gain * error_y_world

# #         desired_vx_world, desired_vy_world = clamp_vector(
# #             desired_vx_world,
# #             desired_vy_world,
# #             self.max_xy_speed,
# #         )

# #         # If UAV is dangerously low, prioritize vertical recovery over horizontal chase.
# #         if altitude_recovery:
# #             desired_vx_world *= 0.10
# #             desired_vy_world *= 0.10
# #             alt_mode = "altitude_recovery"

# #         # Small XY deadband to avoid constant micro commands.
# #         if xy_error < self.xy_deadband:
# #             desired_vx_world = 0.0
# #             desired_vy_world = 0.0

# #         # Convert desired world-frame velocity to UAV body frame.
# #         cos_yaw = math.cos(uav_yaw)
# #         sin_yaw = math.sin(uav_yaw)

# #         cmd_body_x = cos_yaw * desired_vx_world + sin_yaw * desired_vy_world
# #         cmd_body_y = -sin_yaw * desired_vx_world + cos_yaw * desired_vy_world

# #         cmd_body_x, cmd_body_y = clamp_vector(
# #             cmd_body_x,
# #             cmd_body_y,
# #             self.max_xy_speed,
# #         )

# #         linear_z = clamp(
# #             self.z_gain * error_z,
# #             -self.max_z_speed,
# #             self.max_z_speed,
# #         )

# #         if altitude_recovery and error_z > 0.0:
# #             linear_z = max(linear_z, 0.85 * self.max_z_speed)

# #         if float(uav_state["z"]) <= self.min_world_altitude + 0.25:
# #             linear_z = max(linear_z, self.max_z_speed)

# #         elif float(uav_state["z"]) <= self.min_world_altitude + 0.5:
# #             linear_z = max(0.0, linear_z)

# #         if abs(error_z) < self.z_deadband:
# #             linear_z = 0.0

# #         # Yaw toward the tracking target.
# #         if xy_error > 1e-6:
# #             yaw_target = math.atan2(error_y_world, error_x_world)
# #         else:
# #             yaw_target = husky_yaw

# #         yaw_error = wrap_angle(yaw_target - uav_yaw)
# #         yaw_cmd = clamp(
# #             self.yaw_gain * yaw_error,
# #             -self.max_yaw_rate,
# #             self.max_yaw_rate,
# #         )

# #         if abs(yaw_error) < self.yaw_deadband:
# #             yaw_cmd = 0.0

# #         cmd_msg = Twist()
# #         cmd_msg.linear.x = float(cmd_body_x)
# #         cmd_msg.linear.y = float(cmd_body_y)
# #         cmd_msg.linear.z = float(linear_z)
# #         cmd_msg.angular.z = float(yaw_cmd)

# #         self.cmd_pub_model.publish(cmd_msg)
# #         self.cmd_pub_direct.publish(cmd_msg)

# #         ready_now = (
# #             float(uav_state["z"])
# #             >= max(
# #                 self.min_world_altitude,
# #                 float(husky_state["z"]) + self.min_follow_altitude,
# #             )
# #             and abs(error_z) <= max(0.5, self.z_deadband)
# #         )

# #         ready_msg = Bool()
# #         ready_msg.data = bool(ready_now)
# #         self.ready_pub.publish(ready_msg)

# #         if ready_now and not self.ready_sent:
# #             self.get_logger().info(f"UAV ready on {self.ready_topic}")
# #             self.ready_sent = True

# #         elif not ready_now:
# #             self.ready_sent = False

# #         if now - self.last_follow_log_time >= self.follow_log_period:
# #             self.get_logger().info(
# #                 "uav_follow "
# #                 f"husky=({husky_state['x']:.2f},{husky_state['y']:.2f},{husky_state['z']:.2f}) "
# #                 f"uav=({uav_state['x']:.2f},{uav_state['y']:.2f},{uav_state['z']:.2f}) "
# #                 f"target=({target_x:.2f},{target_y:.2f},{target_z:.2f}) "
# #                 f"err_xy={xy_error:.2f} err_z={error_z:.2f} "
# #                 f"z_floor={self.min_world_altitude:.2f} alt_mode={alt_mode} "
# #                 f"cmd_body=({cmd_body_x:.2f},{cmd_body_y:.2f},{linear_z:.2f}) "
# #                 f"cmd_world=({desired_vx_world:.2f},{desired_vy_world:.2f}) "
# #                 f"yaw_target={yaw_target:.2f} yaw_cmd={yaw_cmd:.2f} "
# #                 f"alt_recovery={altitude_recovery} ready={ready_now}"
# #             )
# #             self.last_follow_log_time = now


# """Simple UAV follower used during live simulation.

# The UAV follows the ego Husky from above/side and provides aerial/contextual data
# for the hybrid trajectory-prediction dataset.

# Role in the dataset:
# - keep UAVs close enough to the ego Husky to observe the surrounding/front area,
# - support left/right side formation around the Husky,
# - publish UAV odometry through Gazebo/ROS bridge,
# - keep UAV forward pointcloud topics active,
# - publish a simple ready status, for example /uav1/ready or /uav2/ready.

# This node is intentionally not a learned controller. It is only a stable
# rule-based support behavior for data collection.
# """

# import math

# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# from rclpy.node import Node
# from std_msgs.msg import Bool
# from tf2_msgs.msg import TFMessage


# def quaternion_to_yaw(x, y, z, w):
#     siny_cosp = 2.0 * (w * z + x * y)
#     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
#     return math.atan2(siny_cosp, cosy_cosp)


# def clamp(value, min_value, max_value):
#     return max(min(value, max_value), min_value)


# def wrap_angle(angle):
#     return math.atan2(math.sin(angle), math.cos(angle))


# def clamp_vector(x, y, max_mag):
#     mag = math.hypot(x, y)

#     if mag <= max_mag or mag <= 1e-9:
#         return x, y

#     scale = max_mag / mag
#     return x * scale, y * scale


# def local_pose_to_world(pose, spawn_xyz, spawn_yaw):
#     """Convert spawn-relative odometry pose into Gazebo world-like coordinates."""
#     p = pose.position
#     q = pose.orientation

#     local_x = float(p.x)
#     local_y = float(p.y)
#     local_z = float(p.z)

#     cos_yaw = math.cos(spawn_yaw)
#     sin_yaw = math.sin(spawn_yaw)

#     world_x = spawn_xyz[0] + cos_yaw * local_x - sin_yaw * local_y
#     world_y = spawn_xyz[1] + sin_yaw * local_x + cos_yaw * local_y
#     world_z = spawn_xyz[2] + local_z

#     local_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

#     return {
#         "x": world_x,
#         "y": world_y,
#         "z": world_z,
#         "yaw": wrap_angle(spawn_yaw + local_yaw),
#     }


# def extract_model_transform(msg: TFMessage, model_name: str):
#     """Extract the transform corresponding to a Gazebo model.

#     Prefer the model's base_link transform when available, otherwise fall back
#     to the model-level transform.
#     """
#     selected_model = None
#     selected_base_link = None

#     for transform in msg.transforms:
#         child = transform.child_frame_id or ""
#         child_parts = [part for part in child.split("/") if part]

#         if model_name not in child_parts:
#             continue

#         if child_parts and child_parts[-1] == "base_link":
#             selected_base_link = transform
#             continue

#         if (
#             child == model_name
#             or child.endswith(f"/{model_name}")
#             or (child_parts and child_parts[-1] == model_name)
#         ):
#             selected_model = transform

#     return selected_base_link or selected_model


# class UavFollower(Node):
#     """Keep the UAV above/near the Husky and yawed toward its tracking target.

#     Formation convention:
#     - follow_distance > 0  means in front of Husky
#     - follow_distance = 0  means aligned with Husky, neither ahead nor behind
#     - follow_distance < 0  means behind Husky

#     - follow_lateral_offset > 0 means left side of Husky
#     - follow_lateral_offset < 0 means right side of Husky
#     - follow_lateral_offset = 0 means directly above/centered
#     """

#     def __init__(
#         self,
#         node_name: str = "uav_follower",
#         husky_odom_topic: str = "/model/husky_local/odometry",
#         uav_odom_topic: str = "/model/uav1/odometry",
#         world_pose_topic: str | None = None,
#         husky_model_name: str = "husky_local",
#         uav_model_name: str = "uav1",
#         uav_name: str = "uav1",
#         follow_distance: float = 0.0,
#         follow_lateral_offset: float = 0.0,
#         follow_height: float = 12.0,
#         update_period: float = 0.1,
#         max_xy_speed: float = 7.0,
#         max_z_speed: float = 1.2,
#         max_yaw_rate: float = 0.9,
#         xy_gain: float = 1.8,
#         z_gain: float = 0.35,
#         yaw_gain: float = 0.8,
#         heading_align_gain: float = 0.0,
#         min_forward_speed: float = 0.0,
#         target_smoothing: float = 1.0,
#         xy_deadband: float = 0.02,
#         z_deadband: float = 0.15,
#         yaw_deadband: float = 0.18,
#         min_track_speed: float = 0.0,
#         catchup_distance: float = 8.0,
#         catchup_height_buffer: float = 6.0,
#         reenable_period: float = 2.0,
#         takeoff_hold_seconds: float = 0.0,
#         altitude_tolerance: float = 0.4,
#         min_follow_altitude: float = 2.0,
#         ready_topic: str = "/uav1/ready",
#         husky_spawn_xyz: tuple[float, float, float] | None = None,
#         husky_spawn_yaw: float = 0.0,
#         uav_spawn_xyz: tuple[float, float, float] | None = None,
#         uav_spawn_yaw: float = 0.0,
#         mission_goal_xyz: tuple[float, float, float] | None = None,
#         path_start_xyz: tuple[float, float, float] | None = None,
#         path_goal_xyz: tuple[float, float, float] | None = None,
#         goal_blend_start_progress: float = 0.7,
#     ):
#         super().__init__(node_name)

#         self.uav_name = uav_name

#         self.world_pose_topic = world_pose_topic
#         self.husky_model_name = husky_model_name
#         self.uav_model_name = uav_model_name

#         self.follow_distance = float(follow_distance)
#         self.follow_lateral_offset = float(follow_lateral_offset)
#         self.follow_height = float(follow_height)

#         self.max_xy_speed = float(max_xy_speed)
#         self.max_z_speed = float(max_z_speed)
#         self.max_yaw_rate = float(max_yaw_rate)

#         self.xy_gain = float(xy_gain)
#         self.z_gain = float(z_gain)
#         self.yaw_gain = float(yaw_gain)

#         # Kept for compatibility with existing runner arguments.
#         self.heading_align_gain = float(heading_align_gain)
#         self.min_forward_speed = float(min_forward_speed)
#         self.target_smoothing = float(target_smoothing)
#         self.min_track_speed = float(min_track_speed)
#         self.takeoff_hold_seconds = float(takeoff_hold_seconds)
#         self.altitude_tolerance = float(altitude_tolerance)

#         self.xy_deadband = float(xy_deadband)
#         self.z_deadband = float(z_deadband)
#         self.yaw_deadband = float(yaw_deadband)

#         self.catchup_distance = float(catchup_distance)
#         self.catchup_height_buffer = float(catchup_height_buffer)

#         self.min_follow_altitude = float(min_follow_altitude)
#         self.ready_topic = ready_topic

#         self.husky_spawn_xyz = husky_spawn_xyz
#         self.husky_spawn_yaw = float(husky_spawn_yaw)

#         self.uav_spawn_xyz = uav_spawn_xyz
#         self.uav_spawn_yaw = float(uav_spawn_yaw)

#         self.mission_goal_xyz = mission_goal_xyz
#         self.path_start_xyz = path_start_xyz
#         self.path_goal_xyz = path_goal_xyz
#         self.goal_blend_start_progress = float(goal_blend_start_progress)

#         # Safety floor. Keeps UAV from trying to fly too low in uneven terrain.
#         self.min_world_altitude = 4.0

#         # Logging rates.
#         self.state_log_period = 2.0
#         self.follow_log_period = 1.0

#         self.husky_pose = None
#         self.husky_twist = None
#         self.uav_pose = None

#         self.husky_world_state = None
#         self.uav_world_state = None

#         self.last_state_log_time = 0.0
#         self.last_follow_log_time = 0.0

#         self.ready_sent = False
#         self.smoothed_target_xy = None

#         self.cmd_pub_model = self.create_publisher(
#             Twist,
#             f"/model/{self.uav_name}/command/twist",
#             10,
#         )
#         self.cmd_pub_direct = self.create_publisher(
#             Twist,
#             f"/{self.uav_name}/command/twist",
#             10,
#         )

#         self.enable_pub_model = self.create_publisher(
#             Bool,
#             f"/model/{self.uav_name}/enable",
#             10,
#         )
#         self.enable_pub_direct = self.create_publisher(
#             Bool,
#             f"/{self.uav_name}/enable",
#             10,
#         )

#         self.ready_pub = self.create_publisher(
#             Bool,
#             self.ready_topic,
#             10,
#         )

#         self.create_subscription(
#             Odometry,
#             husky_odom_topic,
#             self.husky_odom_cb,
#             10,
#         )
#         self.create_subscription(
#             Odometry,
#             uav_odom_topic,
#             self.uav_odom_cb,
#             10,
#         )

#         if self.world_pose_topic is not None:
#             self.create_subscription(
#                 TFMessage,
#                 self.world_pose_topic,
#                 self.world_pose_cb,
#                 10,
#             )

#         self.create_timer(update_period, self.follow_husky)
#         self.create_timer(reenable_period, self.enable_controller)

#         self.enable_controller()

#         side_name = "center"
#         if self.follow_lateral_offset > 0.0:
#             side_name = "left"
#         elif self.follow_lateral_offset < 0.0:
#             side_name = "right"

#         self.get_logger().info(
#             "UAV follower started: "
#             f"uav={self.uav_name} "
#             f"formation_side={side_name} "
#             f"follow_distance={self.follow_distance:.2f}m "
#             f"follow_lateral_offset={self.follow_lateral_offset:.2f}m "
#             f"follow_height={self.follow_height:.2f}m "
#             f"cmd_topics=(/model/{self.uav_name}/command/twist, /{self.uav_name}/command/twist) "
#             f"enable_topics=(/model/{self.uav_name}/enable, /{self.uav_name}/enable)"
#         )

#     def husky_odom_cb(self, msg):
#         self.husky_pose = msg.pose.pose
#         self.husky_twist = msg.twist.twist

#     def uav_odom_cb(self, msg):
#         self.uav_pose = msg.pose.pose

#     def world_pose_cb(self, msg: TFMessage):
#         husky_tf = extract_model_transform(msg, self.husky_model_name)
#         uav_tf = extract_model_transform(msg, self.uav_model_name)

#         if husky_tf is not None:
#             t = husky_tf.transform.translation
#             r = husky_tf.transform.rotation

#             self.husky_world_state = {
#                 "x": float(t.x),
#                 "y": float(t.y),
#                 "z": float(t.z),
#                 "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
#             }

#         if uav_tf is not None:
#             t = uav_tf.transform.translation
#             r = uav_tf.transform.rotation

#             self.uav_world_state = {
#                 "x": float(t.x),
#                 "y": float(t.y),
#                 "z": float(t.z),
#                 "yaw": quaternion_to_yaw(r.x, r.y, r.z, r.w),
#             }

#     def _husky_state(self):
#         if self.husky_world_state is not None:
#             return self.husky_world_state, "world_pose"

#         if self.husky_pose is None:
#             return None, "missing"

#         if self.husky_spawn_xyz is not None:
#             return (
#                 local_pose_to_world(
#                     self.husky_pose,
#                     self.husky_spawn_xyz,
#                     self.husky_spawn_yaw,
#                 ),
#                 "spawn_corrected_odom",
#             )

#         p = self.husky_pose.position
#         q = self.husky_pose.orientation

#         return (
#             {
#                 "x": float(p.x),
#                 "y": float(p.y),
#                 "z": float(p.z),
#                 "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
#             },
#             "odom",
#         )

#     def _uav_state(self):
#         if self.uav_world_state is not None:
#             return self.uav_world_state, "world_pose"

#         if self.uav_pose is None:
#             return None, "missing"

#         if self.uav_spawn_xyz is not None:
#             return (
#                 local_pose_to_world(
#                     self.uav_pose,
#                     self.uav_spawn_xyz,
#                     self.uav_spawn_yaw,
#                 ),
#                 "spawn_corrected_odom",
#             )

#         p = self.uav_pose.position
#         q = self.uav_pose.orientation

#         return (
#             {
#                 "x": float(p.x),
#                 "y": float(p.y),
#                 "z": float(p.z),
#                 "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
#             },
#             "odom",
#         )

#     def enable_controller(self):
#         msg = Bool()
#         msg.data = True

#         self.enable_pub_model.publish(msg)
#         self.enable_pub_direct.publish(msg)

#         self.get_logger().info(
#             f"UAV controller enable sent on /model/{self.uav_name}/enable and /{self.uav_name}/enable"
#         )

#     def _husky_velocity_world(self, husky_yaw: float):
#         if self.husky_twist is None:
#             return 0.0, 0.0

#         husky_vx_body = float(self.husky_twist.linear.x)
#         husky_vy_body = float(self.husky_twist.linear.y)

#         husky_vx_world = (
#             math.cos(husky_yaw) * husky_vx_body
#             - math.sin(husky_yaw) * husky_vy_body
#         )
#         husky_vy_world = (
#             math.sin(husky_yaw) * husky_vx_body
#             + math.cos(husky_yaw) * husky_vy_body
#         )

#         return husky_vx_world, husky_vy_world

#     def _formation_target_xy(self, husky_state: dict):
#         """Compute the desired UAV formation point around the Husky.

#         Body-frame formula:
#             x_target = husky_x + forward*cos(yaw) - lateral*sin(yaw)
#             y_target = husky_y + forward*sin(yaw) + lateral*cos(yaw)
#         """
#         husky_yaw = float(husky_state["yaw"])

#         target_x = (
#             float(husky_state["x"])
#             + self.follow_distance * math.cos(husky_yaw)
#             - self.follow_lateral_offset * math.sin(husky_yaw)
#         )
#         target_y = (
#             float(husky_state["y"])
#             + self.follow_distance * math.sin(husky_yaw)
#             + self.follow_lateral_offset * math.cos(husky_yaw)
#         )

#         return target_x, target_y

#     def _smooth_target(self, target_x: float, target_y: float):
#         """Smooth target location to reduce UAV jitter.

#         target_smoothing = 1.0 means no smoothing.
#         target_smoothing = 0.0 means hold previous target.
#         """
#         alpha = clamp(self.target_smoothing, 0.0, 1.0)

#         if self.smoothed_target_xy is None or alpha >= 0.999:
#             self.smoothed_target_xy = (target_x, target_y)
#             return target_x, target_y

#         old_x, old_y = self.smoothed_target_xy
#         new_x = alpha * target_x + (1.0 - alpha) * old_x
#         new_y = alpha * target_y + (1.0 - alpha) * old_y

#         self.smoothed_target_xy = (new_x, new_y)
#         return new_x, new_y

#     def follow_husky(self):
#         husky_state, husky_source = self._husky_state()
#         uav_state, uav_source = self._uav_state()

#         if husky_state is None or uav_state is None:
#             return

#         now = self.get_clock().now().nanoseconds / 1e9

#         if now - self.last_state_log_time >= self.state_log_period:
#             self.get_logger().info(
#                 f"uav_state_source husky={husky_source} uav={uav_source} "
#                 f"world_topic={'on' if self.world_pose_topic is not None else 'off'}"
#             )
#             self.last_state_log_time = now

#         husky_yaw = float(husky_state["yaw"])
#         uav_yaw = float(uav_state["yaw"])

#         raw_target_x, raw_target_y = self._formation_target_xy(husky_state)
#         target_x, target_y = self._smooth_target(raw_target_x, raw_target_y)

#         full_target_z = max(
#             float(husky_state["z"]) + self.follow_height,
#             self.min_world_altitude,
#         )

#         catchup_target_z = max(
#             full_target_z - 1.0,
#             float(husky_state["z"]) + self.min_follow_altitude + self.catchup_height_buffer,
#             self.min_world_altitude,
#         )

#         error_x_world = target_x - float(uav_state["x"])
#         error_y_world = target_y - float(uav_state["y"])
#         xy_error = math.hypot(error_x_world, error_y_world)

#         if xy_error > self.catchup_distance:
#             alt_mode = "catchup"
#             target_z = catchup_target_z
#         else:
#             alt_mode = "full"
#             target_z = full_target_z

#         error_z = target_z - float(uav_state["z"])

#         altitude_recovery = (
#             float(uav_state["z"]) < (self.min_world_altitude + 1.0)
#             or error_z > 6.0
#         )

#         husky_vx_world, husky_vy_world = self._husky_velocity_world(husky_yaw)

#         desired_vx_world = husky_vx_world + self.xy_gain * error_x_world
#         desired_vy_world = husky_vy_world + self.xy_gain * error_y_world

#         desired_vx_world, desired_vy_world = clamp_vector(
#             desired_vx_world,
#             desired_vy_world,
#             self.max_xy_speed,
#         )

#         # If UAV is dangerously low, prioritize vertical recovery over horizontal chase.
#         if altitude_recovery:
#             desired_vx_world *= 0.10
#             desired_vy_world *= 0.10
#             alt_mode = "altitude_recovery"

#         # Small XY deadband to avoid constant micro commands.
#         if xy_error < self.xy_deadband:
#             desired_vx_world = 0.0
#             desired_vy_world = 0.0

#         # Convert desired world-frame velocity to UAV body frame.
#         cos_yaw = math.cos(uav_yaw)
#         sin_yaw = math.sin(uav_yaw)

#         cmd_body_x = cos_yaw * desired_vx_world + sin_yaw * desired_vy_world
#         cmd_body_y = -sin_yaw * desired_vx_world + cos_yaw * desired_vy_world

#         cmd_body_x, cmd_body_y = clamp_vector(
#             cmd_body_x,
#             cmd_body_y,
#             self.max_xy_speed,
#         )

#         linear_z = clamp(
#             self.z_gain * error_z,
#             -self.max_z_speed,
#             self.max_z_speed,
#         )

#         if altitude_recovery and error_z > 0.0:
#             linear_z = max(linear_z, 0.85 * self.max_z_speed)

#         if float(uav_state["z"]) <= self.min_world_altitude + 0.25:
#             linear_z = max(linear_z, self.max_z_speed)

#         elif float(uav_state["z"]) <= self.min_world_altitude + 0.5:
#             linear_z = max(0.0, linear_z)

#         if abs(error_z) < self.z_deadband:
#             linear_z = 0.0

#         # Yaw toward the Husky, not only toward the side target.
#         # This keeps the UAV sensor generally facing the ego UGV/route area.
#         yaw_target = math.atan2(
#             float(husky_state["y"]) - float(uav_state["y"]),
#             float(husky_state["x"]) - float(uav_state["x"]),
#         )

#         yaw_error = wrap_angle(yaw_target - uav_yaw)
#         yaw_cmd = clamp(
#             self.yaw_gain * yaw_error,
#             -self.max_yaw_rate,
#             self.max_yaw_rate,
#         )

#         if abs(yaw_error) < self.yaw_deadband:
#             yaw_cmd = 0.0

#         cmd_msg = Twist()
#         cmd_msg.linear.x = float(cmd_body_x)
#         cmd_msg.linear.y = float(cmd_body_y)
#         cmd_msg.linear.z = float(linear_z)
#         cmd_msg.angular.z = float(yaw_cmd)

#         self.cmd_pub_model.publish(cmd_msg)
#         self.cmd_pub_direct.publish(cmd_msg)

#         ready_now = (
#             float(uav_state["z"])
#             >= max(
#                 self.min_world_altitude,
#                 float(husky_state["z"]) + self.min_follow_altitude,
#             )
#             and abs(error_z) <= max(0.8, self.z_deadband)
#         )

#         ready_msg = Bool()
#         ready_msg.data = bool(ready_now)
#         self.ready_pub.publish(ready_msg)

#         if ready_now and not self.ready_sent:
#             self.get_logger().info(f"UAV ready on {self.ready_topic}")
#             self.ready_sent = True

#         elif not ready_now:
#             self.ready_sent = False

#         if now - self.last_follow_log_time >= self.follow_log_period:
#             self.get_logger().info(
#                 "uav_follow "
#                 f"uav={self.uav_name} "
#                 f"follow_distance={self.follow_distance:.2f} "
#                 f"follow_lateral_offset={self.follow_lateral_offset:.2f} "
#                 f"husky=({husky_state['x']:.2f},{husky_state['y']:.2f},{husky_state['z']:.2f}) "
#                 f"uav=({uav_state['x']:.2f},{uav_state['y']:.2f},{uav_state['z']:.2f}) "
#                 f"target=({target_x:.2f},{target_y:.2f},{target_z:.2f}) "
#                 f"err_xy={xy_error:.2f} err_z={error_z:.2f} "
#                 f"z_floor={self.min_world_altitude:.2f} alt_mode={alt_mode} "
#                 f"cmd_body=({cmd_body_x:.2f},{cmd_body_y:.2f},{linear_z:.2f}) "
#                 f"cmd_world=({desired_vx_world:.2f},{desired_vy_world:.2f}) "
#                 f"yaw_target={yaw_target:.2f} yaw_cmd={yaw_cmd:.2f} "
#                 f"alt_recovery={altitude_recovery} ready={ready_now}"
#             )
#             self.last_follow_log_time = now



"""Stable side-formation UAV follower for hybrid UGV/UAV data collection.

The UAV acts as a support sensor platform for the ego Husky. It can fly on the
left or right side of the Husky, stay approximately aligned with the Husky
without moving ahead/behind, and keep its forward sensor facing the same route
direction as the Husky.

Main purpose:
- keep UAVs close to the ego Husky,
- support two-UAV side formation: left UAV and right UAV,
- keep UAVs high enough to avoid tree/canopy collisions,
- keep UAV forward pointcloud sensors looking toward the Husky route direction,
- publish UAV odometry and ready status for dataset export.

This is not a learned controller. It is a deterministic support controller for
generating clean training data.
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


def clamp_vector(x, y, max_mag):
    mag = math.hypot(x, y)

    if mag <= max_mag or mag <= 1e-9:
        return x, y

    scale = max_mag / mag
    return x * scale, y * scale


def local_pose_to_world(pose, spawn_xyz, spawn_yaw):
    """Convert spawn-relative odometry pose into Gazebo world-like coordinates."""
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
    """Extract the transform corresponding to a Gazebo model.

    Prefer the model's base_link transform when available, otherwise fall back
    to the model-level transform.
    """
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
    """Keep one UAV in a side formation around the ego Husky.

    Formation convention:
    - follow_distance > 0.0  means in front of Husky
    - follow_distance = 0.0  means aligned with Husky
    - follow_distance < 0.0  means behind Husky

    - follow_lateral_offset > 0.0 means left side of Husky
    - follow_lateral_offset < 0.0 means right side of Husky
    - follow_lateral_offset = 0.0 means directly above/centered

    For this thesis dataset, the recommended setup is:
    - UAV1: follow_distance=0.0, follow_lateral_offset=+6.0
    - UAV2: follow_distance=0.0, follow_lateral_offset=-6.0
    - follow_height around 22.0 to avoid trees
    """

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
        follow_lateral_offset: float = 0.0,
        follow_height: float = 22.0,
        update_period: float = 0.1,
        max_xy_speed: float = 7.0,
        max_z_speed: float = 2.0,
        max_yaw_rate: float = 1.2,
        xy_gain: float = 1.8,
        z_gain: float = 0.45,
        yaw_gain: float = 1.0,
        heading_align_gain: float = 0.0,
        min_forward_speed: float = 0.0,
        target_smoothing: float = 0.85,
        xy_deadband: float = 0.05,
        z_deadband: float = 0.35,
        yaw_deadband: float = 0.08,
        min_track_speed: float = 0.0,
        catchup_distance: float = 10.0,
        catchup_height_buffer: float = 6.0,
        reenable_period: float = 2.0,
        takeoff_hold_seconds: float = 0.0,
        altitude_tolerance: float = 0.6,
        min_follow_altitude: float = 18.0,
        min_world_altitude: float = 18.0,
        ready_topic: str = "/uav1/ready",
        husky_spawn_xyz: tuple[float, float, float] | None = None,
        husky_spawn_yaw: float = 0.0,
        uav_spawn_xyz: tuple[float, float, float] | None = None,
        uav_spawn_yaw: float = 0.0,
        mission_goal_xyz: tuple[float, float, float] | None = None,
        path_start_xyz: tuple[float, float, float] | None = None,
        path_goal_xyz: tuple[float, float, float] | None = None,
        goal_blend_start_progress: float = 0.7,
        yaw_mode: str = "husky_heading",
    ):
        super().__init__(node_name)

        self.uav_name = uav_name

        self.world_pose_topic = world_pose_topic
        self.husky_model_name = husky_model_name
        self.uav_model_name = uav_model_name

        self.follow_distance = float(follow_distance)
        self.follow_lateral_offset = float(follow_lateral_offset)
        self.follow_height = float(follow_height)

        self.max_xy_speed = float(max_xy_speed)
        self.max_z_speed = float(max_z_speed)
        self.max_yaw_rate = float(max_yaw_rate)

        self.xy_gain = float(xy_gain)
        self.z_gain = float(z_gain)
        self.yaw_gain = float(yaw_gain)

        # Kept for compatibility with existing runner arguments.
        self.heading_align_gain = float(heading_align_gain)
        self.min_forward_speed = float(min_forward_speed)
        self.target_smoothing = float(target_smoothing)
        self.min_track_speed = float(min_track_speed)
        self.takeoff_hold_seconds = float(takeoff_hold_seconds)
        self.altitude_tolerance = float(altitude_tolerance)

        self.xy_deadband = float(xy_deadband)
        self.z_deadband = float(z_deadband)
        self.yaw_deadband = float(yaw_deadband)

        self.catchup_distance = float(catchup_distance)
        self.catchup_height_buffer = float(catchup_height_buffer)

        self.min_follow_altitude = float(min_follow_altitude)
        self.min_world_altitude = float(min_world_altitude)

        self.ready_topic = ready_topic

        self.husky_spawn_xyz = husky_spawn_xyz
        self.husky_spawn_yaw = float(husky_spawn_yaw)

        self.uav_spawn_xyz = uav_spawn_xyz
        self.uav_spawn_yaw = float(uav_spawn_yaw)

        self.mission_goal_xyz = mission_goal_xyz
        self.path_start_xyz = path_start_xyz
        self.path_goal_xyz = path_goal_xyz
        self.goal_blend_start_progress = float(goal_blend_start_progress)

        # yaw_mode:
        # - "husky_heading": UAV faces the same route direction as the Husky.
        # - "goal_heading": UAV faces from Husky toward mission goal/path goal.
        # - "look_at_husky": UAV faces inward toward Husky. Not recommended for front sensing.
        self.yaw_mode = str(yaw_mode)

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
        self.smoothed_target_xy = None

        self.cmd_pub_model = self.create_publisher(
            Twist,
            f"/model/{self.uav_name}/command/twist",
            10,
        )
        self.cmd_pub_direct = self.create_publisher(
            Twist,
            f"/{self.uav_name}/command/twist",
            10,
        )

        self.enable_pub_model = self.create_publisher(
            Bool,
            f"/model/{self.uav_name}/enable",
            10,
        )
        self.enable_pub_direct = self.create_publisher(
            Bool,
            f"/{self.uav_name}/enable",
            10,
        )

        self.ready_pub = self.create_publisher(
            Bool,
            self.ready_topic,
            10,
        )

        self.create_subscription(
            Odometry,
            husky_odom_topic,
            self.husky_odom_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            uav_odom_topic,
            self.uav_odom_cb,
            10,
        )

        if self.world_pose_topic is not None:
            self.create_subscription(
                TFMessage,
                self.world_pose_topic,
                self.world_pose_cb,
                10,
            )

        self.create_timer(update_period, self.follow_husky)
        self.create_timer(reenable_period, self.enable_controller)

        self.enable_controller()

        side_name = "center"
        if self.follow_lateral_offset > 0.0:
            side_name = "left"
        elif self.follow_lateral_offset < 0.0:
            side_name = "right"

        self.get_logger().info(
            "UAV side-formation follower started: "
            f"uav={self.uav_name} "
            f"formation_side={side_name} "
            f"follow_distance={self.follow_distance:.2f}m "
            f"follow_lateral_offset={self.follow_lateral_offset:.2f}m "
            f"follow_height={self.follow_height:.2f}m "
            f"min_world_altitude={self.min_world_altitude:.2f}m "
            f"yaw_mode={self.yaw_mode} "
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
            return (
                local_pose_to_world(
                    self.husky_pose,
                    self.husky_spawn_xyz,
                    self.husky_spawn_yaw,
                ),
                "spawn_corrected_odom",
            )

        p = self.husky_pose.position
        q = self.husky_pose.orientation

        return (
            {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
                "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
            },
            "odom",
        )

    def _uav_state(self):
        if self.uav_world_state is not None:
            return self.uav_world_state, "world_pose"

        if self.uav_pose is None:
            return None, "missing"

        if self.uav_spawn_xyz is not None:
            return (
                local_pose_to_world(
                    self.uav_pose,
                    self.uav_spawn_xyz,
                    self.uav_spawn_yaw,
                ),
                "spawn_corrected_odom",
            )

        p = self.uav_pose.position
        q = self.uav_pose.orientation

        return (
            {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
                "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
            },
            "odom",
        )

    def enable_controller(self):
        msg = Bool()
        msg.data = True

        self.enable_pub_model.publish(msg)
        self.enable_pub_direct.publish(msg)

        self.get_logger().info(
            f"UAV controller enable sent on /model/{self.uav_name}/enable and /{self.uav_name}/enable"
        )

    def _husky_velocity_world(self, husky_yaw: float):
        if self.husky_twist is None:
            return 0.0, 0.0

        husky_vx_body = float(self.husky_twist.linear.x)
        husky_vy_body = float(self.husky_twist.linear.y)

        husky_vx_world = (
            math.cos(husky_yaw) * husky_vx_body
            - math.sin(husky_yaw) * husky_vy_body
        )
        husky_vy_world = (
            math.sin(husky_yaw) * husky_vx_body
            + math.cos(husky_yaw) * husky_vy_body
        )

        return husky_vx_world, husky_vy_world

    def _formation_target_xy(self, husky_state: dict):
        """Compute the desired UAV formation point around the Husky.

        Body-frame formula:
            target = Husky
                     + forward_offset along Husky heading
                     + lateral_offset perpendicular to Husky heading

        x_target = husky_x + forward*cos(yaw) - lateral*sin(yaw)
        y_target = husky_y + forward*sin(yaw) + lateral*cos(yaw)
        """
        husky_yaw = float(husky_state["yaw"])

        target_x = (
            float(husky_state["x"])
            + self.follow_distance * math.cos(husky_yaw)
            - self.follow_lateral_offset * math.sin(husky_yaw)
        )
        target_y = (
            float(husky_state["y"])
            + self.follow_distance * math.sin(husky_yaw)
            + self.follow_lateral_offset * math.cos(husky_yaw)
        )

        return target_x, target_y

    def _smooth_target(self, target_x: float, target_y: float):
        """Smooth target location to reduce UAV jitter."""
        alpha = clamp(self.target_smoothing, 0.0, 1.0)

        if self.smoothed_target_xy is None or alpha >= 0.999:
            self.smoothed_target_xy = (target_x, target_y)
            return target_x, target_y

        old_x, old_y = self.smoothed_target_xy

        new_x = alpha * target_x + (1.0 - alpha) * old_x
        new_y = alpha * target_y + (1.0 - alpha) * old_y

        self.smoothed_target_xy = (new_x, new_y)
        return new_x, new_y

    def _mission_goal_heading(self, husky_state: dict) -> float | None:
        goal = self.mission_goal_xyz or self.path_goal_xyz

        if goal is None:
            return None

        dx = float(goal[0]) - float(husky_state["x"])
        dy = float(goal[1]) - float(husky_state["y"])

        if math.hypot(dx, dy) < 1e-6:
            return None

        return math.atan2(dy, dx)

    def _yaw_target(self, husky_state: dict, uav_state: dict) -> float:
        """Choose where the UAV should face.

        For this project, the default is husky_heading. This keeps the UAV's
        forward sensor looking along the same route direction as the Husky,
        instead of looking inward toward the Husky.
        """
        husky_yaw = float(husky_state["yaw"])

        if self.yaw_mode == "goal_heading":
            goal_heading = self._mission_goal_heading(husky_state)
            if goal_heading is not None:
                return goal_heading
            return husky_yaw

        if self.yaw_mode == "look_at_husky":
            return math.atan2(
                float(husky_state["y"]) - float(uav_state["y"]),
                float(husky_state["x"]) - float(uav_state["x"]),
            )

        return husky_yaw

    def _formation_error_components(
        self,
        husky_state: dict,
        uav_state: dict,
    ) -> tuple[float, float]:
        """Return UAV error in Husky body frame.

        forward_error:
            positive means UAV is ahead of desired formation point.

        lateral_error:
            positive means UAV is more to the left than desired.
        """
        husky_yaw = float(husky_state["yaw"])

        dx = float(uav_state["x"]) - float(husky_state["x"])
        dy = float(uav_state["y"]) - float(husky_state["y"])

        forward_actual = math.cos(husky_yaw) * dx + math.sin(husky_yaw) * dy
        lateral_actual = -math.sin(husky_yaw) * dx + math.cos(husky_yaw) * dy

        forward_error = forward_actual - self.follow_distance
        lateral_error = lateral_actual - self.follow_lateral_offset

        return forward_error, lateral_error

    def follow_husky(self):
        husky_state, husky_source = self._husky_state()
        uav_state, uav_source = self._uav_state()

        if husky_state is None or uav_state is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        if now - self.last_state_log_time >= self.state_log_period:
            self.get_logger().info(
                f"uav_state_source uav={self.uav_name} "
                f"husky={husky_source} uav_source={uav_source} "
                f"world_topic={'on' if self.world_pose_topic is not None else 'off'}"
            )
            self.last_state_log_time = now

        husky_yaw = float(husky_state["yaw"])
        uav_yaw = float(uav_state["yaw"])

        raw_target_x, raw_target_y = self._formation_target_xy(husky_state)
        target_x, target_y = self._smooth_target(raw_target_x, raw_target_y)

        target_z = max(
            float(husky_state["z"]) + self.follow_height,
            self.min_world_altitude,
        )

        error_x_world = target_x - float(uav_state["x"])
        error_y_world = target_y - float(uav_state["y"])
        error_z = target_z - float(uav_state["z"])

        xy_error = math.hypot(error_x_world, error_y_world)

        altitude_recovery = (
            float(uav_state["z"]) < (self.min_world_altitude + 1.0)
            or error_z > 6.0
        )

        husky_vx_world, husky_vy_world = self._husky_velocity_world(husky_yaw)

        desired_vx_world = husky_vx_world + self.xy_gain * error_x_world
        desired_vy_world = husky_vy_world + self.xy_gain * error_y_world

        desired_vx_world, desired_vy_world = clamp_vector(
            desired_vx_world,
            desired_vy_world,
            self.max_xy_speed,
        )

        alt_mode = "formation"

        if xy_error > self.catchup_distance:
            alt_mode = "catchup"

        if altitude_recovery:
            desired_vx_world *= 0.15
            desired_vy_world *= 0.15
            alt_mode = "altitude_recovery"

        if xy_error < self.xy_deadband:
            desired_vx_world = 0.0
            desired_vy_world = 0.0

        # Convert world-frame desired velocity to UAV body frame.
        cos_yaw = math.cos(uav_yaw)
        sin_yaw = math.sin(uav_yaw)

        cmd_body_x = cos_yaw * desired_vx_world + sin_yaw * desired_vy_world
        cmd_body_y = -sin_yaw * desired_vx_world + cos_yaw * desired_vy_world

        cmd_body_x, cmd_body_y = clamp_vector(
            cmd_body_x,
            cmd_body_y,
            self.max_xy_speed,
        )

        linear_z = clamp(
            self.z_gain * error_z,
            -self.max_z_speed,
            self.max_z_speed,
        )

        if altitude_recovery and error_z > 0.0:
            linear_z = max(linear_z, 0.85 * self.max_z_speed)

        if float(uav_state["z"]) <= self.min_world_altitude + 0.25:
            linear_z = max(linear_z, self.max_z_speed)

        elif float(uav_state["z"]) <= self.min_world_altitude + 0.5:
            linear_z = max(0.0, linear_z)

        if abs(error_z) < self.z_deadband:
            linear_z = 0.0

        yaw_target = self._yaw_target(husky_state, uav_state)
        yaw_error = wrap_angle(yaw_target - uav_yaw)

        yaw_cmd = clamp(
            self.yaw_gain * yaw_error,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        )

        if abs(yaw_error) < self.yaw_deadband:
            yaw_cmd = 0.0

        cmd_msg = Twist()
        cmd_msg.linear.x = float(cmd_body_x)
        cmd_msg.linear.y = float(cmd_body_y)
        cmd_msg.linear.z = float(linear_z)
        cmd_msg.angular.z = float(yaw_cmd)

        self.cmd_pub_model.publish(cmd_msg)
        self.cmd_pub_direct.publish(cmd_msg)

        forward_error, lateral_error = self._formation_error_components(
            husky_state,
            uav_state,
        )

        formation_ok = (
            abs(forward_error) <= 4.0
            and abs(lateral_error) <= 4.0
            and abs(error_z) <= max(3.0, self.altitude_tolerance)
        )

        ready_now = (
            float(uav_state["z"])
            >= max(
                self.min_world_altitude,
                float(husky_state["z"]) + self.min_follow_altitude,
            )
            and formation_ok
        )

        ready_msg = Bool()
        ready_msg.data = bool(ready_now)
        self.ready_pub.publish(ready_msg)

        if ready_now and not self.ready_sent:
            self.get_logger().info(f"UAV ready on {self.ready_topic}")
            self.ready_sent = True

        elif not ready_now:
            self.ready_sent = False

        if now - self.last_follow_log_time >= self.follow_log_period:
            self.get_logger().info(
                "uav_side_follow "
                f"uav={self.uav_name} "
                f"follow_distance={self.follow_distance:.2f} "
                f"follow_lateral_offset={self.follow_lateral_offset:.2f} "
                f"husky=({husky_state['x']:.2f},{husky_state['y']:.2f},{husky_state['z']:.2f}) "
                f"uav=({uav_state['x']:.2f},{uav_state['y']:.2f},{uav_state['z']:.2f}) "
                f"target=({target_x:.2f},{target_y:.2f},{target_z:.2f}) "
                f"err_xy={xy_error:.2f} err_z={error_z:.2f} "
                f"forward_error={forward_error:.2f} lateral_error={lateral_error:.2f} "
                f"yaw_mode={self.yaw_mode} yaw_target={yaw_target:.2f} yaw_cmd={yaw_cmd:.2f} "
                f"alt_mode={alt_mode} "
                f"cmd_body=({cmd_body_x:.2f},{cmd_body_y:.2f},{linear_z:.2f}) "
                f"cmd_world=({desired_vx_world:.2f},{desired_vy_world:.2f}) "
                f"alt_recovery={altitude_recovery} formation_ok={formation_ok} ready={ready_now}"
            )
            self.last_follow_log_time = now