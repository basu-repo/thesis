"""Launch the live CNN-LSTM simulation pipeline.

This runner starts Gazebo, spawns the Husky/UAV agents, publishes episode
metadata, optionally records a bag, and connects the working CNN-based Husky
controller to the live odometry stream.
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
from controllers.husky_model_driver import ModelHuskyDriver
from controllers.omnet_hazard_bridge import OmnetHazardBridge
from controllers.uav_hazard_estimator import UavHazardEstimator
from controllers.uav_follower import UavFollower


WORLD = "/home/basudeo/Documents/Thesis/worlds/sim_world.sdf"
WORLD_NAME = "sim_world"
MODEL_PATH = "/home/basudeo/Documents/Thesis/models"
SUMMARY_PATH = Path("/home/basudeo/Documents/Thesis/models/summary_75pct_done.json")
OMNET_DIR = Path("/home/basudeo/Documents/Thesis/onmetpp")
OMNET_BIN = OMNET_DIR / "onmetpp"
OMNET_CONFIG = "WifiRelay"

SPAWN_X, SPAWN_Y, SPAWN_Z = 0.0, 0.0, 0.35
HUSKY2_X, HUSKY2_Y, HUSKY2_Z = -2.5, 0.05, 0.35
UAV_X, UAV_Y, UAV_Z = 0.0, 2.0, 0.36
HUSKY1_SPAWN_YAW = math.pi
HUSKY2_SPAWN_YAW = 0.0
UAV_SPAWN_YAW = 0.0
HUSKY1_SPAWN_QZ = math.sin(HUSKY1_SPAWN_YAW / 2.0)
HUSKY1_SPAWN_QW = math.cos(HUSKY1_SPAWN_YAW / 2.0)
UAV_SPAWN_QZ = math.sin(UAV_SPAWN_YAW / 2.0)
UAV_SPAWN_QW = math.cos(UAV_SPAWN_YAW / 2.0)

def world_to_local_goal(
    world_goal: tuple[float, float, float],
    spawn_xyz: tuple[float, float, float],
    spawn_yaw: float,
) -> tuple[float, float, float]:
    """Convert a world goal into the spawn-aligned odom frame used by the driver.

    Gazebo model odometry is origin-shifted at spawn and aligned to the model's
    initial heading, so we must translate by spawn position and inverse-rotate
    by the configured spawn yaw to express the goal in the controller frame.
    """
    dx = float(world_goal[0]) - float(spawn_xyz[0])
    dy = float(world_goal[1]) - float(spawn_xyz[1])
    c = math.cos(spawn_yaw)
    s = math.sin(spawn_yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    local_z = float(world_goal[2]) - float(spawn_xyz[2])
    return (local_x, local_y, local_z)


WORLD_HUSKY1_GOAL = (-34.0, -24.0, 0.35)
WORLD_HUSKY2_GOAL = WORLD_HUSKY1_GOAL
WORLD_UAV_GOAL = WORLD_HUSKY1_GOAL
GROUND_MARKER_Z = -0.7984

BOOTSTRAP_SECONDS = 3.0
BOOTSTRAP_LINEAR_SPEED = 0.8
ENABLE_SECOND_HUSKY = False
ENABLE_UAV = False

TARGET_INDEX = 2
CONTROL_PERIOD = 0.1
CMD_LINEAR_GAIN = 1.45
CMD_ANGULAR_GAIN = 1.15
MIN_LINEAR_SPEED = 1.0
MAX_LINEAR_SPEED = 1.45
MAX_ANGULAR_SPEED = 0.85
HEADING_DEADBAND = 0.12
WAYPOINT_REACHED_DIST = 0.3
GOAL_TOLERANCE = 0.05
GOAL_BLEND = 0.9
SECOND_HUSKY_TARGET_BIAS_X = 0.0
SECOND_HUSKY_TARGET_BIAS_Y = 0.0

UAV_FOLLOW_DISTANCE = 0.0
UAV_FOLLOW_HEIGHT = 2.2
UAV_UPDATE_PERIOD = 0.1
UAV_MAX_XY_SPEED = 3.0
UAV_MAX_Z_SPEED = 0.6
UAV_MAX_YAW_RATE = 0.0
UAV_XY_GAIN = 1.2
UAV_Z_GAIN = 0.35
UAV_YAW_GAIN = 0.0
UAV_TARGET_SMOOTHING = 1.0
UAV_XY_DEADBAND = 0.02
UAV_Z_DEADBAND = 0.15
UAV_YAW_DEADBAND = 0.18
UAV_MIN_TRACK_SPEED = 0.0
ENABLE_BAG_RECORDING = False


# Gazebo process and model-spawn helpers -------------------------------------

def run_bg(cmd):
    return subprocess.Popen(["bash", "-c", cmd])


def load_husky_sdf_with_topic(topic_name: str) -> str:
    husky_sdf = Path("/home/basudeo/Documents/Thesis/models/husky/model.sdf").read_text()
    husky_sdf = husky_sdf.replace("<topic>/cmd_vel</topic>", f"<topic>{topic_name}</topic>", 1)
    return husky_sdf


def add_pose_publisher(sdf_text: str) -> str:
    """Inject a Gazebo pose publisher so ROS can see world-truth Husky poses."""

    if "ignition-gazebo-pose-publisher-system" in sdf_text:
        return sdf_text
    plugin = """
    <plugin
      filename="ignition-gazebo-pose-publisher-system"
      name="ignition::gazebo::systems::PosePublisher">
      <publish_link_pose>true</publish_link_pose>
      <use_pose_vector_msg>true</use_pose_vector_msg>
    </plugin>
"""
    return sdf_text.replace("</model>", plugin + "\n  </model>", 1)


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
    sdf_text = add_pose_publisher(sdf_text)
    sdf_text = add_husky_marker(sdf_text, marker_name, rgba)
    output_path.write_text(sdf_text)
    return output_path


def spawn_goal_marker(world_name: str, name: str, xyz: tuple[float, float, float], rgba: tuple[float, float, float, float]):
    marker_sdf = f"""<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>
    <link name="marker_link">
      <visual name="marker_visual">
        <pose>0 0 1.25 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>2.5</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
          <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          <emissive>{rgba[0] * 0.5} {rgba[1] * 0.5} {rgba[2] * 0.5} {rgba[3]}</emissive>
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


os.environ["IGN_GAZEBO_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
subprocess.run(["bash", "-c", "pkill -f ign || true"])
subprocess.run(["bash", "-c", "pkill -f /home/basudeo/Documents/Thesis/onmetpp/onmetpp || true"])

print("Starting Gazebo...")
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

print("Waiting for terrain to fully load...")
time.sleep(40)

print("Spawning Husky...")
husky1_sdf_path = write_husky_variant(
    Path("/home/basudeo/Documents/Thesis/models/husky/model_red_tag.sdf"),
    "/cmd_vel",
    "flag_marker_red",
    (0.95, 0.12, 0.12, 1.0),
)
spawn_husky = (
    "ign service -s /world/{world_name}/create "
    "--reqtype ignition.msgs.EntityFactory "
    "--reptype ignition.msgs.Boolean "
    "--timeout 5000 "
    "--req 'sdf_filename: \"{sdf_path}\", name: \"husky_local\", "
    "pose: {{position: {{x: {spawn_x}, y: {spawn_y}, z: {spawn_z}}}, "
    "orientation: {{z: {spawn_qz}, w: {spawn_qw}}}}}'"
).format(
    world_name=WORLD_NAME,
    sdf_path=husky1_sdf_path,
    spawn_x=SPAWN_X,
    spawn_y=SPAWN_Y,
    spawn_z=SPAWN_Z,
    spawn_qz=HUSKY1_SPAWN_QZ,
    spawn_qw=HUSKY1_SPAWN_QW,
)
subprocess.run(["bash", "-c", spawn_husky])
time.sleep(5)

if ENABLE_SECOND_HUSKY:
    print("Spawning Husky 2...")
    husky2_sdf_path = write_husky_variant(
        Path("/home/basudeo/Documents/Thesis/models/husky/model_blue_tag.sdf"),
        "/cmd_vel_husky2",
        "flag_marker_blue",
        (0.12, 0.36, 0.95, 1.0),
    )
    spawn_husky2 = (
        f"ign service -s /world/{WORLD_NAME}/create "
        f"--reqtype ignition.msgs.EntityFactory "
        f"--reptype ignition.msgs.Boolean "
        f"--timeout 5000 "
        f'--req \'sdf_filename: "{husky2_sdf_path}", name: "husky_2", '
        f'pose: {{position: {{x: {HUSKY2_X}, y: {HUSKY2_Y}, z: {HUSKY2_Z}}}, '
        f'orientation: {{w: 1.0}}}}\''
    )
    subprocess.run(["bash", "-c", spawn_husky2])
    time.sleep(5)

if ENABLE_UAV:
    print("Spawning UAV...")
    spawn_uav = """
ign service -s /world/{world_name}/create \
--reqtype ignition.msgs.EntityFactory \
--reptype ignition.msgs.Boolean \
--timeout 5000 \
--req 'sdf_filename: "model://m100/model.sdf", name: "uav1",
pose: {{position: {{x: {uav_x}, y: {uav_y}, z: {uav_z}}}, orientation: {{z: {uav_qz}, w: {uav_qw}}}}}'
""".format(
        world_name=WORLD_NAME,
        uav_x=UAV_X,
        uav_y=UAV_Y,
        uav_z=UAV_Z,
        uav_qz=UAV_SPAWN_QZ,
        uav_qw=UAV_SPAWN_QW,
    )
    subprocess.run(["bash", "-c", spawn_uav])
    time.sleep(5)

print("Spawning goal markers...")
# spawn_goal_marker(WORLD_NAME, "start_husky_local", (SPAWN_X, SPAWN_Y, 0.0387), (0.65, 0.15, 0.15, 1.0))
spawn_goal_marker(WORLD_NAME, "goal_husky_local", (WORLD_HUSKY1_GOAL[0], WORLD_HUSKY1_GOAL[1], GROUND_MARKER_Z), (0.95, 0.12, 0.12, 1.0))
if ENABLE_SECOND_HUSKY:
    spawn_goal_marker(WORLD_NAME, "start_husky_2", (HUSKY2_X, HUSKY2_Y, 0.0387), (0.65, 0.25, 0.70, 1.0))
    spawn_goal_marker(WORLD_NAME, "goal_husky_2", (WORLD_HUSKY2_GOAL[0], WORLD_HUSKY2_GOAL[1], GROUND_MARKER_Z), (0.12, 0.36, 0.95, 1.0))
if ENABLE_UAV:
    spawn_goal_marker(WORLD_NAME, "goal_uav1", (WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1], GROUND_MARKER_Z), (0.95, 0.85, 0.12, 1.0))
time.sleep(1)

print("Starting bridge...")
bridge_topics = [
    "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
    "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
    f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
]
if ENABLE_SECOND_HUSKY:
    bridge_topics.extend(
        [
            "/cmd_vel_husky2@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/model/husky_2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
        ]
    )
if ENABLE_UAV:
    bridge_topics.extend(
        [
            "/model/uav1/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/air_pressure/air_pressure@sensor_msgs/msg/FluidPressure[ignition.msgs.FluidPressure",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/magnetometer/magnetometer@sensor_msgs/msg/MagneticField[ignition.msgs.Magnetometer",
        ]
    )
bridge_cmd = (
    "source /opt/ros/humble/setup.bash && "
    "ros2 run ros_gz_bridge parameter_bridge "
    + " ".join(bridge_topics)
)
bridge = run_bg(bridge_cmd)
time.sleep(2)

omnet = None
if ENABLE_UAV:
    print("Starting OMNeT++ relay...")
    omnet_cmd = f"cd {OMNET_DIR} && ./onmetpp -u Cmdenv -f omnetpp.ini -c {OMNET_CONFIG}"
    omnet = run_bg(omnet_cmd)
    time.sleep(2)

BAG_DIR = os.path.expanduser("~/Documents/Thesis/bags")
os.makedirs(BAG_DIR, exist_ok=True)
run_name = "run_model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        "/uav1/hazard_hint_raw "
        "/uav1/hazard_hint_net "
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
print(" MODEL MODE ENABLED ")
print("==============================")
print("Press Play in Gazebo. Husky will bootstrap briefly, then switch to model-driven motion.")
print("Press Ctrl+C here when done.\n")

rclpy.init()
# Drive from Gazebo world pose so the controller and the visible goal marker use
# one consistent coordinate frame.
HUSKY1_GOAL = WORLD_HUSKY1_GOAL
HUSKY2_GOAL = WORLD_HUSKY2_GOAL
UAV_GOAL = WORLD_UAV_GOAL
print(
    "Controller goals (world frame): "
    f"husky_local=({HUSKY1_GOAL[0]:.3f}, {HUSKY1_GOAL[1]:.3f}), "
    f"husky_2=({HUSKY2_GOAL[0]:.3f}, {HUSKY2_GOAL[1]:.3f}), "
    f"uav=({UAV_GOAL[0]:.3f}, {UAV_GOAL[1]:.3f})"
)
# ROS 2 nodes for metadata, learned control, UAV support, and network effects.
start_goals = {
    "husky_local": {"start": (SPAWN_X, SPAWN_Y, SPAWN_Z), "goal": WORLD_HUSKY1_GOAL},
}
if ENABLE_SECOND_HUSKY:
    start_goals["husky_2"] = {"start": (HUSKY2_X, HUSKY2_Y, HUSKY2_Z), "goal": WORLD_HUSKY2_GOAL}
if ENABLE_UAV:
    start_goals["uav1"] = {"start": (UAV_X, UAV_Y, UAV_Z), "goal": WORLD_UAV_GOAL}

episode_metadata = EpisodeMetadataPublisher(
    world_name=WORLD_NAME,
    start_goals=start_goals,
)
driver = ModelHuskyDriver(
    node_name="model_husky_driver_1",
    cmd_topic="/cmd_vel",
    odom_topic="/model/husky_local/odometry",
    world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
    scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
    pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
    hazard_topic="/uav1/hazard_hint_net" if ENABLE_UAV else None,
    summary_path=SUMMARY_PATH,
    goal_xyz=HUSKY1_GOAL,
    world_goal_xyz=WORLD_HUSKY1_GOAL,
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
    goal_tolerance=GOAL_TOLERANCE,
    goal_blend=GOAL_BLEND,
)
driver2 = None
if ENABLE_SECOND_HUSKY:
    driver2 = ModelHuskyDriver(
        node_name="model_husky_driver_2",
        cmd_topic="/cmd_vel_husky2",
        odom_topic="/model/husky_2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        scan_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
        pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
        hazard_topic="/uav1/hazard_hint_net" if ENABLE_UAV else None,
        summary_path=SUMMARY_PATH,
        goal_xyz=HUSKY2_GOAL,
        world_goal_xyz=WORLD_HUSKY2_GOAL,
        target_bias_y=SECOND_HUSKY_TARGET_BIAS_Y,
        bootstrap_seconds=BOOTSTRAP_SECONDS,
        bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
        bootstrap_angular_speed=0.0,
        target_index=TARGET_INDEX,
        control_period=CONTROL_PERIOD,
        cmd_linear_gain=CMD_LINEAR_GAIN,
        cmd_angular_gain=CMD_ANGULAR_GAIN,
        min_linear_speed=MIN_LINEAR_SPEED,
        max_linear_speed=MAX_LINEAR_SPEED,
        max_angular_speed=MAX_ANGULAR_SPEED,
        heading_deadband=HEADING_DEADBAND,
        waypoint_reached_dist=WAYPOINT_REACHED_DIST,
        goal_tolerance=GOAL_TOLERANCE,
        goal_blend=GOAL_BLEND,
    )
follower = None
hazard_estimator = None
hazard_bridge = None
if ENABLE_UAV:
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
    hazard_estimator = UavHazardEstimator(
        husky_odom_topic="/model/husky_local/odometry",
        uav_odom_topic="/model/uav1/odometry",
        uav_pointcloud_topic=f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points",
        output_topic="/uav1/hazard_hint_raw",
    )
    hazard_bridge = OmnetHazardBridge(
        input_topic="/uav1/hazard_hint_raw",
        output_topic="/uav1/hazard_hint_net",
    )
executor = MultiThreadedExecutor()
executor.add_node(episode_metadata)
executor.add_node(driver)
if driver2 is not None:
    executor.add_node(driver2)
if follower is not None:
    executor.add_node(follower)
if hazard_estimator is not None:
    executor.add_node(hazard_estimator)
if hazard_bridge is not None:
    executor.add_node(hazard_bridge)

try:
    executor.spin()
except KeyboardInterrupt:
    print("\nStopping model run...")
finally:
    executor.shutdown()
    episode_metadata.destroy_node()
    driver.destroy_node()
    if driver2 is not None:
        driver2.destroy_node()
    if follower is not None:
        follower.destroy_node()
    if hazard_estimator is not None:
        hazard_estimator.destroy_node()
    if hazard_bridge is not None:
        hazard_bridge.destroy_node()
    rclpy.shutdown()
    if recorder is not None:
        recorder.send_signal(signal.SIGINT)
        time.sleep(2)
    print("Stopping bridge, OMNeT++, and Gazebo...")
    bridge.send_signal(signal.SIGINT)
    if omnet is not None:
        omnet.send_signal(signal.SIGINT)
    gz.send_signal(signal.SIGINT)
    time.sleep(2)
    print("All processes stopped cleanly.")
