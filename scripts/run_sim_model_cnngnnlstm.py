"""Launch the live GNN-LSTM simulation pipeline.

This runner mirrors the CNN live setup but swaps in the graph-based predictor.
Its job is to expose the same simulation and control loop while feeding the
Husky drivers with multi-agent graph context.
"""

import datetime
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

from controllers.episode_metadata import EpisodeMetadataPublisher
from controllers.husky_gnn_model_driver import GNNModelHuskyDriver
from controllers.uav_follower import UavFollower


WORLD = "/home/basudeo/Documents/Thesis/worlds/sim_world.sdf"
WORLD_NAME = "sim_world"
MODEL_PATH = "/home/basudeo/Documents/Thesis/models"
GNN_SUMMARY_PATH = Path("/home/basudeo/Documents/Thesis/models/summary_gnn_graph_done.json")

SPAWN_X, SPAWN_Y, SPAWN_Z = 0.0, 0.0, 0.35
HUSKY2_X, HUSKY2_Y, HUSKY2_Z = 0.05, -2.5, 0.35
UAV_X, UAV_Y, UAV_Z = 0.0, 2.0, 1.0
HUSKY1_SPAWN_YAW = math.pi / 2.0
HUSKY2_SPAWN_YAW = math.pi / 2.0
UAV_SPAWN_YAW = 0.0
WORLD_SHARED_GOAL = (34.0, 24.0, 0.35)
WORLD_UAV_GOAL = (34.0, 24.0, 3.0)
GROUND_MARKER_Z = 0.005

BOOTSTRAP_SECONDS = 3.0
BOOTSTRAP_LINEAR_SPEED = 0.8

TARGET_INDEX = 4
CONTROL_PERIOD = 0.1
CMD_LINEAR_GAIN = 1.45
CMD_ANGULAR_GAIN = 1.15
MIN_LINEAR_SPEED = 1.0
MAX_LINEAR_SPEED = 1.45
MAX_ANGULAR_SPEED = 0.85
HEADING_DEADBAND = 0.12
WAYPOINT_REACHED_DIST = 0.3
CRUISE_SPEED = 1.1
GOAL_TOLERANCE = 1.5
GOAL_BLEND = 1.0
OBSTACLE_SCAN_DISTANCE = 2.2
OBSTACLE_CLEAR_DISTANCE = 2.8
TURN_IN_PLACE_SPEED = 0.85
SECOND_HUSKY_TARGET_BIAS_X = 0.0
SECOND_HUSKY_TARGET_BIAS_Y = 0.0

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

ENABLE_BAG_RECORDING = False


# Gazebo process and model-spawn helpers -------------------------------------

def run_bg(cmd):
    return subprocess.Popen(["bash", "-c", cmd])


def world_to_local_goal(
    world_goal: tuple[float, float, float],
    spawn_xyz: tuple[float, float, float],
    spawn_yaw: float,
) -> tuple[float, float, float]:
    dx = float(world_goal[0]) - float(spawn_xyz[0])
    dy = float(world_goal[1]) - float(spawn_xyz[1])
    # The odometry used by the live controllers is origin-shifted, but in practice
    # it behaves aligned with the world axes rather than rotated by spawn yaw.
    local_x = dx
    local_y = dy
    local_z = float(world_goal[2]) - float(spawn_xyz[2])
    return (local_x, local_y, local_z)


HUSKY1_GOAL = world_to_local_goal(
    WORLD_SHARED_GOAL,
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    HUSKY1_SPAWN_YAW,
)
HUSKY2_GOAL = world_to_local_goal(
    WORLD_SHARED_GOAL,
    (HUSKY2_X, HUSKY2_Y, HUSKY2_Z),
    HUSKY2_SPAWN_YAW,
)
UAV_GOAL = world_to_local_goal(
    WORLD_UAV_GOAL,
    (UAV_X, UAV_Y, UAV_Z),
    UAV_SPAWN_YAW,
)


def load_husky_sdf_with_topic(topic_name: str) -> str:
    husky_sdf = Path("/home/basudeo/Documents/Thesis/models/husky/model.sdf").read_text()
    husky_sdf = husky_sdf.replace("<topic>/cmd_vel</topic>", f"<topic>{topic_name}</topic>", 1)
    return husky_sdf


def add_husky_marker(sdf_text: str, marker_name: str, rgba: tuple[float, float, float, float]) -> str:
    marker = f"""
    <link name="{marker_name}">
      <pose>0 0 0.45 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.03</radius>
            <length>1.0</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>1.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
          <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          <emissive>{rgba[0] * 0.4} {rgba[1] * 0.4} {rgba[2] * 0.4} {rgba[3]}</emissive>
        </material>
      </visual>
    </link>
    <joint name="{marker_name}_joint" type="fixed">
      <parent>base_link</parent>
      <child>{marker_name}</child>
    </joint>
"""
    return sdf_text.replace("</model>", marker + "\n  </model>", 1)


def write_husky_variant(output_path: Path, topic_name: str, marker_name: str, rgba: tuple[float, float, float, float]) -> Path:
    sdf_text = load_husky_sdf_with_topic(topic_name)
    sdf_text = add_husky_marker(sdf_text, marker_name, rgba)
    output_path.write_text(sdf_text)
    return output_path


def spawn_goal_marker(world_name: str, name: str, xyz: tuple[float, float, float], rgba: tuple[float, float, float, float]):
    marker_sdf = f"""<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.52</radius>
            <length>0.01</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
          <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          <emissive>{rgba[0] * 0.35} {rgba[1] * 0.35} {rgba[2] * 0.35} {rgba[3]}</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
    one_line = marker_sdf.replace("\n", " ").replace('"', '\\"')
    cmd = (
        f"ign service -s /world/{world_name}/create "
        f"--reqtype ignition.msgs.EntityFactory "
        f"--reptype ignition.msgs.Boolean "
        f"--timeout 5000 "
        f'--req \'sdf: "{one_line}"\''
    )
    subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if not GNN_SUMMARY_PATH.exists():
    raise FileNotFoundError(
        f"GNN summary not found: {GNN_SUMMARY_PATH}. "
        "Train the graph model first with scripts/train_gnn_lstm.py."
    )


os.environ["IGN_GAZEBO_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
subprocess.run(["bash", "-c", "pkill -f ign || true"])

print("Starting Gazebo...")
gz = run_bg(f"ign gazebo {WORLD} -r")
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

print("Waiting for terrain to fully load...")
time.sleep(40)

print("Spawning Husky...")
husky1_sdf_path = write_husky_variant(
    Path("/home/basudeo/Documents/Thesis/models/husky/model_red_tag.sdf"),
    "/cmd_vel",
    "id_marker_red",
    (0.95, 0.12, 0.12, 1.0),
)
spawn_husky = (
    f"ign service -s /world/{WORLD_NAME}/create "
    f"--reqtype ignition.msgs.EntityFactory "
    f"--reptype ignition.msgs.Boolean "
    f"--timeout 5000 "
    f'--req \'sdf_filename: "{husky1_sdf_path}", name: "husky_local", '
    f'pose: {{position: {{x: {SPAWN_X}, y: {SPAWN_Y}, z: {SPAWN_Z}}}, '
    f'orientation: {{z: 0.70710678, w: 0.70710678}}}}\''
)
subprocess.run(["bash", "-c", spawn_husky])
time.sleep(5)

print("Spawning Husky 2...")
husky2_sdf_path = write_husky_variant(
    Path("/home/basudeo/Documents/Thesis/models/husky/model_blue_tag.sdf"),
    "/cmd_vel_husky2",
    "id_marker_blue",
    (0.12, 0.36, 0.95, 1.0),
)
spawn_husky2 = (
    f"ign service -s /world/{WORLD_NAME}/create "
    f"--reqtype ignition.msgs.EntityFactory "
    f"--reptype ignition.msgs.Boolean "
    f"--timeout 5000 "
    f'--req \'sdf_filename: "{husky2_sdf_path}", name: "husky_2", '
    f'pose: {{position: {{x: {HUSKY2_X}, y: {HUSKY2_Y}, z: {HUSKY2_Z}}}, '
    f'orientation: {{z: 0.70710678, w: 0.70710678}}}}\''
)
subprocess.run(["bash", "-c", spawn_husky2])
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

print("Spawning goal markers...")
spawn_goal_marker(WORLD_NAME, "start_husky_local", (SPAWN_X, SPAWN_Y, 0.0387), (0.65, 0.15, 0.15, 1.0))
spawn_goal_marker(WORLD_NAME, "goal_husky_local", (WORLD_SHARED_GOAL[0], WORLD_SHARED_GOAL[1], GROUND_MARKER_Z), (0.95, 0.12, 0.12, 1.0))
spawn_goal_marker(WORLD_NAME, "start_husky_2", (0.1008, -2.4100, 0.0387), (0.65, 0.25, 0.70, 1.0))
spawn_goal_marker(WORLD_NAME, "goal_husky_2", (WORLD_SHARED_GOAL[0], WORLD_SHARED_GOAL[1], GROUND_MARKER_Z), (0.12, 0.36, 0.95, 1.0))
spawn_goal_marker(WORLD_NAME, "start_uav1", (UAV_X, UAV_Y, 0.0387), (0.65, 0.55, 0.12, 1.0))
spawn_goal_marker(WORLD_NAME, "goal_uav1", (WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1], GROUND_MARKER_Z), (0.95, 0.85, 0.12, 1.0))
time.sleep(1)

print("Starting bridge...")
bridge_cmd = (
    "source /opt/ros/humble/setup.bash && "
    "ros2 run ros_gz_bridge parameter_bridge "
    "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist "
    "/cmd_vel_husky2@geometry_msgs/msg/Twist@ignition.msgs.Twist "
    "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry "
    "/model/husky_2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry "
    "/model/uav1/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points"
    "@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked "
    f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points"
    "@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked "
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan"
    "@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan "
    f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan"
    "@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan "
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

BAG_DIR = os.path.expanduser("~/Documents/Thesis/bags")
os.makedirs(BAG_DIR, exist_ok=True)
run_name = "run_model_gnn_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bag_path = f"{BAG_DIR}/{run_name}"
recorder = None

if ENABLE_BAG_RECORDING:
    print("Recording bag:", bag_path)
    record_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"ros2 bag record -o {bag_path} "
        "/cmd_vel "
        "/cmd_vel_husky2 "
        "/episode/husky_local/start "
        "/episode/husky_local/goal "
        "/episode/husky_2/start "
        "/episode/husky_2/goal "
        "/episode/uav1/start "
        "/episode/uav1/goal "
        "/model/husky_local/odometry "
        "/model/husky_2/odometry "
        "/model/uav1/odometry "
        "/clock "
        f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points "
        f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points "
        f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan "
        f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan "
        f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/imu_sensor/imu "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/imu_sensor/imu "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/air_pressure/air_pressure "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/magnetometer/magnetometer "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points "
    )
    recorder = run_bg(record_cmd)
else:
    print("Bag recording disabled. Set ENABLE_BAG_RECORDING = True to record a run.")

print("\n==============================")
print(" GNN-LSTM MODEL MODE ENABLED ")
print("==============================")
print("Press Play in Gazebo if needed. Huskies will bootstrap briefly, then switch to graph-based motion.")
print("Press Ctrl+C here when done.\n")

rclpy.init()
# ROS 2 nodes for metadata, graph-based Husky control, and UAV following.
episode_metadata = EpisodeMetadataPublisher(
    world_name=WORLD_NAME,
    start_goals={
        "husky_local": {"start": (SPAWN_X, SPAWN_Y, SPAWN_Z), "goal": WORLD_SHARED_GOAL},
        "husky_2": {"start": (HUSKY2_X, HUSKY2_Y, HUSKY2_Z), "goal": WORLD_SHARED_GOAL},
        "uav1": {"start": (UAV_X, UAV_Y, UAV_Z), "goal": WORLD_UAV_GOAL},
    },
)
odom_topics = {
    "husky_local": "/model/husky_local/odometry",
    "husky_2": "/model/husky_2/odometry",
    "uav1": "/model/uav1/odometry",
}
command_topics = {
    "husky_local": "/cmd_vel",
    "husky_2": "/cmd_vel_husky2",
}
goals = {
    "husky_local": HUSKY1_GOAL,
    "husky_2": HUSKY2_GOAL,
    "uav1": UAV_GOAL,
}

driver = GNNModelHuskyDriver(
    node_name="gnn_husky_driver_1",
    ego_node="husky_local",
    cmd_topic="/cmd_vel",
    odom_topics=odom_topics,
    command_topics=command_topics,
    summary_path=GNN_SUMMARY_PATH,
    scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
    hazard_topic=None,
    spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
    goals=goals,
    bootstrap_seconds=BOOTSTRAP_SECONDS,
    bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
    target_index=TARGET_INDEX,
    control_period=CONTROL_PERIOD,
    cmd_linear_gain=CMD_LINEAR_GAIN,
    cmd_angular_gain=CMD_ANGULAR_GAIN,
    min_linear_speed=MIN_LINEAR_SPEED,
    max_linear_speed=MAX_LINEAR_SPEED,
    max_angular_speed=MAX_ANGULAR_SPEED,
    heading_deadband=HEADING_DEADBAND,
    waypoint_reached_dist=WAYPOINT_REACHED_DIST,
    cruise_speed=CRUISE_SPEED,
    goal_tolerance=GOAL_TOLERANCE,
    goal_blend=GOAL_BLEND,
    obstacle_scan_distance=OBSTACLE_SCAN_DISTANCE,
    obstacle_clear_distance=OBSTACLE_CLEAR_DISTANCE,
    turn_in_place_speed=TURN_IN_PLACE_SPEED,
)
driver2 = GNNModelHuskyDriver(
    node_name="gnn_husky_driver_2",
    ego_node="husky_2",
    cmd_topic="/cmd_vel_husky2",
    odom_topics=odom_topics,
    command_topics=command_topics,
    summary_path=GNN_SUMMARY_PATH,
    scan_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
    hazard_topic=None,
    spawn_xyz=(HUSKY2_X, HUSKY2_Y, HUSKY2_Z),
    goals=goals,
    target_bias_x=SECOND_HUSKY_TARGET_BIAS_X,
    target_bias_y=SECOND_HUSKY_TARGET_BIAS_Y,
    bootstrap_seconds=BOOTSTRAP_SECONDS,
    bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
    target_index=TARGET_INDEX,
    control_period=CONTROL_PERIOD,
    cmd_linear_gain=CMD_LINEAR_GAIN,
    cmd_angular_gain=CMD_ANGULAR_GAIN,
    min_linear_speed=MIN_LINEAR_SPEED,
    max_linear_speed=MAX_LINEAR_SPEED,
    max_angular_speed=MAX_ANGULAR_SPEED,
    heading_deadband=HEADING_DEADBAND,
    waypoint_reached_dist=WAYPOINT_REACHED_DIST,
    cruise_speed=CRUISE_SPEED,
    goal_tolerance=GOAL_TOLERANCE,
    goal_blend=GOAL_BLEND,
    obstacle_scan_distance=OBSTACLE_SCAN_DISTANCE,
    obstacle_clear_distance=OBSTACLE_CLEAR_DISTANCE,
    turn_in_place_speed=TURN_IN_PLACE_SPEED,
)
follower = UavFollower(
    husky_odom_topic="/model/husky_local/odometry",
    uav_odom_topic="/model/uav1/odometry",
    uav_name="uav1",
    follow_distance=UAV_FOLLOW_DISTANCE,
    follow_height=UAV_FOLLOW_HEIGHT,
    update_period=UAV_UPDATE_PERIOD,
    max_xy_speed=UAV_MAX_XY_SPEED,
    max_z_speed=UAV_MAX_Z_SPEED,
    max_yaw_rate=UAV_MAX_YAW_RATE,
    xy_gain=UAV_XY_GAIN,
    z_gain=UAV_Z_GAIN,
    yaw_gain=UAV_YAW_GAIN,
    target_smoothing=UAV_TARGET_SMOOTHING,
    xy_deadband=UAV_XY_DEADBAND,
    z_deadband=UAV_Z_DEADBAND,
    yaw_deadband=UAV_YAW_DEADBAND,
    min_track_speed=UAV_MIN_TRACK_SPEED,
)

executor = MultiThreadedExecutor()
executor.add_node(episode_metadata)
executor.add_node(driver)
executor.add_node(driver2)
executor.add_node(follower)

try:
    executor.spin()
except KeyboardInterrupt:
    print("\nStopping GNN model run...")
finally:
    executor.shutdown()
    episode_metadata.destroy_node()
    driver.destroy_node()
    driver2.destroy_node()
    follower.destroy_node()
    rclpy.shutdown()
    if recorder is not None:
        recorder.send_signal(signal.SIGINT)
        time.sleep(2)
    print("Stopping bridge and Gazebo...")
    bridge.send_signal(signal.SIGINT)
    gz.send_signal(signal.SIGINT)
    time.sleep(2)
    print("All processes stopped cleanly.")
