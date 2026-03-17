import subprocess
import time
import os
import datetime
import math
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
import signal
import numpy as np
from sensor_msgs.msg import LaserScan

# ---------------- CONFIG ----------------

WORLD = "/home/basudeo/Documents/Thesis/worlds/sim_world.sdf"
WORLD_NAME = "sim_world"

SPAWN_X, SPAWN_Y, SPAWN_Z = -60.0, -60.0, 0.3
UAV_X, UAV_Y, UAV_Z = -45.0, -64.0, 0.45
SPAWN_SETTLE_TIME = 5.0

def run_bg(cmd):
    return subprocess.Popen(["bash", "-c", cmd])

# ---------------- DRIVER ----------------

class ManualStructuredDriver(Node):
    def __init__(self):
        super().__init__('manual_structured_driver')

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Optional: subscribe for logging (not control)
        self.create_subscription(
            Odometry,
            "/model/husky_local/odometry",
            self.odom_cb,
            10)

        self.create_subscription(
            LaserScan,
            f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
            self.scan_cb,
            10)

        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.timer = self.create_timer(0.1, self.step)

        self.x = 0.0
        self.y = 0.0
        self.scan = None

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

    def scan_cb(self, msg):
        self.scan = msg  # optional logging

    def step(self):
        now = self.get_clock().now().seconds_nanoseconds()[0]
        t = now - self.start_time

        msg = Twist()

        # ---------- Structured Motion Plan ----------
        # 0–30s   → straight
        # 30–60s  → left curve
        # 60–90s  → straight
        # 90–120s → right curve
        # 120–150 → sinusoidal steering
        # >150s   → straight cruise

        if t < 30:
            msg.linear.x = 1.0
            msg.angular.z = 0.0

        elif t < 60:
            msg.linear.x = 0.8
            msg.angular.z = 0.4   # left

        elif t < 90:
            msg.linear.x = 1.0
            msg.angular.z = 0.0

        elif t < 120:
            msg.linear.x = 0.8
            msg.angular.z = -0.4  # right

        elif t < 150:
            msg.linear.x = 1.0
            msg.angular.z = 0.5 * math.sin(0.5 * t)

        else:
            msg.linear.x = 1.2
            msg.angular.z = 0.0

        self.pub.publish(msg)


# ---------------- CLEANUP ----------------

subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
subprocess.run(["bash", "-c", "pkill -f ign || true"])
subprocess.run(["bash", "-c", "pkill -f auto_driver || true"])

print("Starting Gazebo (empty world)...")
gz = run_bg(f"ign gazebo {WORLD}")
time.sleep(5)

print("Spawning Baylands terrain...")
spawn_baylands = f"""
ign service -s /world/{WORLD_NAME}/create \
--reqtype ignition.msgs.EntityFactory \
--reptype ignition.msgs.Boolean \
--timeout 5000 \
--req 'sdf_filename: "model://baylands/model.sdf", name: "baylands"'
"""
subprocess.run(["bash", "-c", spawn_baylands])

# print("Spawning Rubicon terrain...")
# spawn_rubicon = f"""
# ign service -s /world/{WORLD_NAME}/create \
# --reqtype ignition.msgs.EntityFactory \
# --reptype ignition.msgs.Boolean \
# --timeout 5000 \
# --req 'sdf_filename: "model://rubicon/model.sdf", name: "rubicon"'
# """
# subprocess.run(["bash", "-c", spawn_rubicon])

print("Waiting for terrain to fully load...")
time.sleep(40)   # <-- IMPORTANT

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

print("Spawning M100...")
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

print("Unpausing world...")

unpause_cmd = f"""
ign service -s /world/{WORLD_NAME}/control \
--reqtype ignition.msgs.WorldControl \
--reptype ignition.msgs.Boolean \
--timeout 2000 \
--req 'pause: false'
"""
subprocess.run(["bash", "-c", unpause_cmd])

bridge = run_bg(bridge_cmd)
time.sleep(2)

# ---------------- RECORD ----------------

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

# ---------------- START DRIVER ----------------

print("Starting AUTO driver...")

rclpy.init()
node = ManualStructuredDriver()

try:
    rclpy.spin(node)

except KeyboardInterrupt:
    print("\nCtrl+C detected. Shutting down cleanly...")

finally:
    print("Stopping recorder...")
    recorder.send_signal(signal.SIGINT)
    time.sleep(2)

    print("Stopping ROS node...")
    node.destroy_node()
    rclpy.shutdown()

    print("Shutting down bridge and Gazebo...")
    bridge.send_signal(signal.SIGINT)
    gz.send_signal(signal.SIGINT)
    time.sleep(2)

    print("All processes stopped cleanly.")
