"""Run the manual Husky/UAV simulation used for baseline checks.

The Husky follows a scripted motion plan while the UAV tracks it overhead.
This gives a predictable reference run before switching to learned controllers.
"""

import subprocess
import time
import os
import datetime
import signal
import math
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

# ---------------- CONFIG ----------------

WORLD = "/home/basudeo/Documents/Thesis/worlds/sim_world.sdf"
WORLD_NAME = "sim_world"
MODEL_PATH = "/home/basudeo/Documents/Thesis/models"

SPAWN_X, SPAWN_Y, SPAWN_Z = 0.0, 0.0, 0.35
UAV_X, UAV_Y, UAV_Z = 0.0, 2.0, 1.0
UAV_FOLLOW_DISTANCE = 0.0
UAV_FOLLOW_HEIGHT = 2.2
UAV_UPDATE_PERIOD = 0.1
UAV_MAX_XY_SPEED = 1.6
UAV_MAX_Z_SPEED = 0.6
UAV_MAX_YAW_RATE = 0.6
UAV_XY_GAIN = 0.45
UAV_Z_GAIN = 0.35
UAV_YAW_GAIN = 0.18
UAV_TARGET_SMOOTHING = 0.1
UAV_XY_DEADBAND = 0.12
UAV_Z_DEADBAND = 0.15
UAV_YAW_DEADBAND = 0.18
UAV_MIN_TRACK_SPEED = 0.05

ROAD_X, ROAD_Y, ROAD_Z = 0.0, 0.0, 0.12
ROAD_LEN, ROAD_WID, ROAD_THICK = 30.0, 12.0, 0.2

def run_bg(cmd):
    return subprocess.Popen(["bash", "-c", cmd])


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class UavFollower(Node):
    """Simple in-file UAV follower used by the manual simulation runner."""

    def __init__(self):
        super().__init__("uav_follower")
        self.husky_pose = None
        self.husky_twist = None
        self.uav_pose = None
        self.filtered_target = None
        self.create_subscription(
            Odometry,
            "/model/husky_local/odometry",
            self.husky_odom_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            "/model/uav1/odometry",
            self.uav_odom_cb,
            10,
        )
        self.create_timer(UAV_UPDATE_PERIOD, self.follow_husky)
        self.enable_controller()

    def husky_odom_cb(self, msg):
        self.husky_pose = msg.pose.pose
        self.husky_twist = msg.twist.twist

    def uav_odom_cb(self, msg):
        self.uav_pose = msg.pose.pose

    def enable_controller(self):
        cmd = (
            "ign topic "
            "-t /uav1/enable "
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

        raw_target_x = husky_position.x - UAV_FOLLOW_DISTANCE * math.cos(husky_yaw)
        raw_target_y = husky_position.y - UAV_FOLLOW_DISTANCE * math.sin(husky_yaw)
        raw_target_z = max(husky_position.z + UAV_FOLLOW_HEIGHT, 1.5)

        if self.filtered_target is None:
            self.filtered_target = {
                "x": raw_target_x,
                "y": raw_target_y,
                "z": raw_target_z,
            }
        else:
            alpha = UAV_TARGET_SMOOTHING
            self.filtered_target["x"] += alpha * (raw_target_x - self.filtered_target["x"])
            self.filtered_target["y"] += alpha * (raw_target_y - self.filtered_target["y"])
            self.filtered_target["z"] += alpha * (raw_target_z - self.filtered_target["z"])

        target_x = self.filtered_target["x"]
        target_y = self.filtered_target["y"]
        target_z = self.filtered_target["z"]

        error_x_world = target_x - uav_position.x
        error_y_world = target_y - uav_position.y
        error_z = target_z - uav_position.z

        if husky_twist is None:
            husky_vx_body = 0.0
            husky_vy_body = 0.0
            husky_yaw_rate = 0.0
        else:
            husky_vx_body = husky_twist.linear.x
            husky_vy_body = husky_twist.linear.y
            husky_yaw_rate = husky_twist.angular.z

        husky_vx_world = (
            math.cos(husky_yaw) * husky_vx_body
            - math.sin(husky_yaw) * husky_vy_body
        )
        husky_vy_world = (
            math.sin(husky_yaw) * husky_vx_body
            + math.cos(husky_yaw) * husky_vy_body
        )

        desired_vx_world = husky_vx_world + UAV_XY_GAIN * error_x_world
        desired_vy_world = husky_vy_world + UAV_XY_GAIN * error_y_world

        cos_yaw = math.cos(uav_yaw)
        sin_yaw = math.sin(uav_yaw)
        linear_x = cos_yaw * desired_vx_world + sin_yaw * desired_vy_world
        linear_y = -sin_yaw * desired_vx_world + cos_yaw * desired_vy_world
        yaw_error = wrap_angle(husky_yaw - uav_yaw)

        xy_error = math.hypot(error_x_world, error_y_world)

        linear_x = clamp(linear_x, -UAV_MAX_XY_SPEED, UAV_MAX_XY_SPEED)
        linear_y = clamp(linear_y, -UAV_MAX_XY_SPEED, UAV_MAX_XY_SPEED)
        linear_z = clamp(UAV_Z_GAIN * error_z, -UAV_MAX_Z_SPEED, UAV_MAX_Z_SPEED)
        angular_z = clamp(
            husky_yaw_rate + UAV_YAW_GAIN * yaw_error,
            -UAV_MAX_YAW_RATE,
            UAV_MAX_YAW_RATE,
        )
        husky_speed = math.hypot(husky_vx_world, husky_vy_world)

        if xy_error < UAV_XY_DEADBAND and husky_speed < UAV_MIN_TRACK_SPEED:
            linear_x = 0.0
            linear_y = 0.0
        if abs(error_z) < UAV_Z_DEADBAND:
            linear_z = 0.0
        if abs(yaw_error) < UAV_YAW_DEADBAND:
            angular_z = 0.0

        msg = (
            f"linear: {{x: {linear_x}, y: {linear_y}, z: {linear_z}}} "
            f"angular: {{x: 0.0, y: 0.0, z: {angular_z}}}"
        )
        cmd = (
            "ign topic "
            "-t /uav1/command/twist "
            "-m ignition.msgs.Twist "
            f"-p '{msg}'"
        )
        subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def destroy_node(self):
        super().destroy_node()

# ---------------- CLEANUP OLD PROCESSES ----------------

os.environ["IGN_GAZEBO_RESOURCE_PATH"] = (
    MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
)
os.environ["GZ_SIM_RESOURCE_PATH"] = (
    MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
)

subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
subprocess.run(["bash", "-c", "pkill -f ign || true"])

print("Starting Gazebo...")
gz = run_bg(f"ign gazebo {WORLD}")
time.sleep(5)

print("Spawning testworld terrain...")
spawn_testworld = f"""
ign service -s /world/{WORLD_NAME}/create \
--reqtype ignition.msgs.EntityFactory \
--reptype ignition.msgs.Boolean \
--timeout 5000 \
--req 'sdf_filename: "model://testworld/model.sdf", name: "testworld"'
"""
subprocess.run(["bash", "-c", spawn_testworld])
time.sleep(5)

# print("Spawning road patch...")
# road_patch_sdf = f"""<sdf version='1.7'>
#   <model name='road_patch'>
#     <static>true</static>
#     <pose>{ROAD_X} {ROAD_Y} {ROAD_Z} 0 0 0</pose>
#     <link name='link'>
#       <collision name='collision'>
#         <geometry>
#           <box>
#             <size>{ROAD_LEN} {ROAD_WID} {ROAD_THICK}</size>
#           </box>
#         </geometry>
#       </collision>
#       <visual name='visual'>
#         <geometry>
#           <box>
#             <size>{ROAD_LEN} {ROAD_WID} {ROAD_THICK}</size>
#           </box>
#         </geometry>
#         <material>
#           <ambient>0.2 0.2 0.2 1</ambient>
#           <diffuse>0.2 0.2 0.2 1</diffuse>
#         </material>
#       </visual>
#     </link>
#   </model>
# </sdf>"""

# road_patch_sdf_one_line = road_patch_sdf.replace("\n", " ").replace('"', '\\"')

# spawn_patch = (
#     f"ign service -s /world/{WORLD_NAME}/create "
#     f"--reqtype ignition.msgs.EntityFactory "
#     f"--reptype ignition.msgs.Boolean "
#     f"--timeout 5000 "
#     f'--req \'sdf: "{road_patch_sdf_one_line}"\''
# )



# subprocess.run(["bash", "-c", spawn_patch])
# time.sleep(2)

print("Spawning Husky...")
spawn_husky = f"""
ign service -s /world/{WORLD_NAME}/create \
--reqtype ignition.msgs.EntityFactory \
--reptype ignition.msgs.Boolean \
--timeout 5000 \
--req 'sdf_filename: "/home/basudeo/Documents/Thesis/models/husky/model.sdf", name: "husky_local",
pose: {{position: {{x: {SPAWN_X}, y: {SPAWN_Y}, z: {SPAWN_Z}}}, orientation: {{w: 1.0}}}}'
"""
subprocess.run(["bash", "-c", spawn_husky])
time.sleep(5)

print("Spawning UAV...")
spawn_uav = f"""
ign service -s /world/{WORLD_NAME}/create \
--reqtype ignition.msgs.EntityFactory \
--reptype ignition.msgs.Boolean \
--timeout 5000 \
--req 'sdf_filename: "model://m100/model.sdf", name: "uav1",
pose: {{position: {{x: {UAV_X}, y: {UAV_Y}, z: {UAV_Z}}}, orientation: {{w: 1.0}}}}'
"""
subprocess.run(["bash", "-c", spawn_uav])
time.sleep(5)

print("Starting bridge...")
bridge_cmd = (
    "source /opt/ros/humble/setup.bash && "
    "ros2 run ros_gz_bridge parameter_bridge "
    "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist "
    "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry "
    "/model/uav1/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan"
    "@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points"
    "@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/imu_sensor/imu"
    "@sensor_msgs/msg/Imu[ignition.msgs.IMU "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points"
    "@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/imu_sensor/imu"
    "@sensor_msgs/msg/Imu[ignition.msgs.IMU "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/air_pressure/air_pressure"
    "@sensor_msgs/msg/FluidPressure[ignition.msgs.FluidPressure "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/magnetometer/magnetometer"
    "@sensor_msgs/msg/MagneticField[ignition.msgs.Magnetometer "
)

bridge = run_bg(bridge_cmd)
time.sleep(2)

# ---------------- RECORD BAG ----------------

BAG_DIR = os.path.expanduser("~/Documents/Thesis/bags")
os.makedirs(BAG_DIR, exist_ok=True)

run_name = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bag_path = f"{BAG_DIR}/{run_name}"

print("Recording bag:", bag_path)

record_cmd = (
    "source /opt/ros/humble/setup.bash && "
    f"ros2 bag record -o {bag_path} "
    "/cmd_vel "
    "/model/husky_local/odometry "
    "/model/uav1/odometry "
    "/clock "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/imu_sensor/imu "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/imu_sensor/imu "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/air_pressure/air_pressure "
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/magnetometer/magnetometer "
)

recorder = run_bg(record_cmd)

print("\n==============================")
print(" MANUAL MODE ENABLED ")
print("==============================")
print("Open a new terminal and run:")
print("source /opt/ros/humble/setup.bash")
print("ros2 run teleop_twist_keyboard teleop_twist_keyboard")
print("\nPress Play in Gazebo, then drive. Press Ctrl+C here when done.\n")

# ---------------- WAIT FOR CTRL+C ----------------

rclpy.init()
follower = UavFollower()

try:
    rclpy.spin(follower)

except KeyboardInterrupt:
    print("\nStopping recorder...")

finally:
    follower.destroy_node()
    rclpy.shutdown()
    recorder.send_signal(signal.SIGINT)
    time.sleep(2)

    print("Stopping bridge and Gazebo...")
    bridge.send_signal(signal.SIGINT)
    gz.send_signal(signal.SIGINT)
    time.sleep(2)

    print("All processes stopped cleanly.")
