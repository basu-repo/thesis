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
from controllers.obstacle_detection import ObstacleDetectionNode
from controllers.uav_follower import UavFollower
from project_paths import MODELS_DIR, OMNET_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH


WORLD = str(WORLD_SDF_PATH)
WORLD_NAME = "sim_world"
MODEL_PATH = str(MODELS_DIR)
OMNET_BIN = OMNET_DIR / "onmetpp"
OMNET_CONFIG = "WifiRelay"

SPAWN_X, SPAWN_Y, SPAWN_Z = -311.3400, -121.7800, -0.2387
HUSKY2_X, HUSKY2_Y, HUSKY2_Z = SPAWN_X + 3.0, SPAWN_Y + 2.0, SPAWN_Z
UAV_X, UAV_Y, UAV_Z = SPAWN_X + 6.0, SPAWN_Y - 4.0, SPAWN_Z + 4.0
HUSKY1_SPAWN_YAW = math.pi
HUSKY2_SPAWN_YAW = HUSKY1_SPAWN_YAW
UAV_SPAWN_YAW = 0.0
HUSKY1_SPAWN_QZ = math.sin(HUSKY1_SPAWN_YAW / 2.0)
HUSKY1_SPAWN_QW = math.cos(HUSKY1_SPAWN_YAW / 2.0)
HUSKY2_SPAWN_QZ = math.sin(HUSKY2_SPAWN_YAW / 2.0)
HUSKY2_SPAWN_QW = math.cos(HUSKY2_SPAWN_YAW / 2.0)
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


def offset_goal_along_path(
    world_goal: tuple[float, float, float],
    start_xyz: tuple[float, float, float],
    offset_distance: float,
) -> tuple[float, float, float]:
    """Shift the stopping target along the start->goal line by a fixed distance."""

    dx = float(world_goal[0]) - float(start_xyz[0])
    dy = float(world_goal[1]) - float(start_xyz[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6 or abs(offset_distance) < 1e-9:
        return world_goal
    ux = dx / norm
    uy = dy / norm
    return (
        float(world_goal[0]) + offset_distance * ux,
        float(world_goal[1]) + offset_distance * uy,
        float(world_goal[2]),
    )


WORLD_SHARED_GOAL = (-248.1530, -82.4012, -1,3000)
WORLD_HUSKY1_GOAL = WORLD_SHARED_GOAL
WORLD_HUSKY2_GOAL = WORLD_SHARED_GOAL
WORLD_UAV_GOAL = WORLD_SHARED_GOAL
GROUND_MARKER_Z = 0.2025
GOAL_STOP_OFFSET = -0.5

BOOTSTRAP_SECONDS = 3.0
BOOTSTRAP_LINEAR_SPEED = 0.8
ENABLE_SECOND_HUSKY = True
ENABLE_UAV = True

CONTROL_PERIOD = 0.1
CMD_LINEAR_GAIN = 1.45
CMD_ANGULAR_GAIN = 1.15
MIN_LINEAR_SPEED = 1.5
MAX_LINEAR_SPEED = 2.0
MAX_ANGULAR_SPEED = 0.85
HEADING_DEADBAND = 0.12
GOAL_TOLERANCE = 1.5
STUCK_TIMEOUT_SECONDS = 3.0
STUCK_PROGRESS_DISTANCE = 0.15
STUCK_REVERSE_SPEED = -0.8
STUCK_REVERSE_SECONDS = 2.0
STUCK_BOOTSTRAP_SECONDS = 2.0
OBSTACLE_FRONT_HALF_ANGLE_DEG = 30.0
OBSTACLE_SIDE_ANGLE_DEG = 90.0
OBSTACLE_STOP_DISTANCE = 1.8
OBSTACLE_CAUTION_DISTANCE = 3.2

UAV_FOLLOW_DISTANCE = 0.0
UAV_FOLLOW_HEIGHT = 12.0
UAV_UPDATE_PERIOD = 0.1
UAV_MAX_XY_SPEED = 9.0
UAV_MAX_Z_SPEED = 1.2
UAV_MAX_YAW_RATE = 0.9
UAV_XY_GAIN = 2.0
UAV_Z_GAIN = 0.35
UAV_YAW_GAIN = 0.8
UAV_HEADING_ALIGN_GAIN = 0.9
UAV_MIN_FORWARD_SPEED = 0.25
UAV_TARGET_SMOOTHING = 1.0
UAV_XY_DEADBAND = 0.02
UAV_Z_DEADBAND = 0.15
UAV_YAW_DEADBAND = 0.18
UAV_MIN_TRACK_SPEED = 0.0
ENABLE_BAG_RECORDING = True
# ENABLE_RVIZ = True
# ENABLE_CAMERA_VIEW = True
ENABLE_RVIZ = False
ENABLE_CAMERA_VIEW = False


# Gazebo process and model-spawn helpers -------------------------------------

def run_bg(cmd):
    return subprocess.Popen(["bash", "-c", cmd])


def load_husky_sdf_with_topic(topic_name: str) -> str:
    husky_sdf = (MODELS_DIR / "husky" / "model.sdf").read_text()
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
      <pose>0 0 0.32 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.015</radius>
            <length>0.25</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.2625</length>
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
      <collision name="marker_collision">
        <pose>0 0 1.25 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>2.5</length>
          </cylinder>
        </geometry>
      </collision>
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


def rgbd_camera_bridge_topics(world_name: str, model_name: str, link_name: str, sensor_name: str) -> list[str]:
    prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"
    return [
        f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
        f"{prefix}/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
        f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
        f"{prefix}/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
    ]


def camera_bridge_topics(world_name: str, model_name: str, link_name: str, sensor_name: str) -> list[str]:
    prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"
    return [
        f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
        f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
    ]


def husky_sensor_bridge_topics(world_name: str, model_name: str) -> list[str]:
    base_prefix = f"/world/{world_name}/model/{model_name}/link/base_link/sensor"
    topics = [
        f"{base_prefix}/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
        f"{base_prefix}/planar_laser/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
        f"{base_prefix}/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
    ]
    topics.extend(rgbd_camera_bridge_topics(world_name, model_name, "base_link", "camera_front"))
    topics.extend(rgbd_camera_bridge_topics(world_name, model_name, "base_link", "camera_down"))
    topics.extend(rgbd_camera_bridge_topics(world_name, model_name, "tilt_gimbal_link", "camera_pan_tilt"))
    return topics


os.environ["IGN_GAZEBO_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
subprocess.run(["bash", "-c", "pkill -f ign || true"])
subprocess.run(["bash", "-c", f"pkill -f {OMNET_BIN} || true"])

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
    MODELS_DIR / "husky" / "model_red_tag.sdf",
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
        MODELS_DIR / "husky" / "model_red_tag.sdf",
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
        f'orientation: {{z: {HUSKY2_SPAWN_QZ}, w: {HUSKY2_SPAWN_QW}}}}}\''
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
    spawn_goal_marker(WORLD_NAME, "goal_husky_2", (WORLD_HUSKY2_GOAL[0], WORLD_HUSKY2_GOAL[1], GROUND_MARKER_Z), (0.12, 0.36, 0.95, 1.0))
if ENABLE_UAV:
    spawn_goal_marker(WORLD_NAME, "goal_uav1", (WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1], GROUND_MARKER_Z), (0.95, 0.85, 0.12, 1.0))
time.sleep(1)

print("Starting bridge...")
bridge_topics = [
    "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
    "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
    f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
]
bridge_topics.extend(husky_sensor_bridge_topics(WORLD_NAME, "husky_local"))
if ENABLE_SECOND_HUSKY:
    bridge_topics.extend(
        [
            "/cmd_vel_husky2@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/model/husky_2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        ]
    )
    bridge_topics.extend(husky_sensor_bridge_topics(WORLD_NAME, "husky_2"))
if ENABLE_UAV:
    bridge_topics.extend(
        [
            "/uav1/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/uav1/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
            "/model/uav1/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/model/uav1/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
            "/model/uav1/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/air_pressure/air_pressure@sensor_msgs/msg/FluidPressure[ignition.msgs.FluidPressure",
            f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/magnetometer/magnetometer@sensor_msgs/msg/MagneticField[ignition.msgs.Magnetometer",
        ]
    )
    bridge_topics.extend(camera_bridge_topics(WORLD_NAME, "uav1", "base_link", "camera_front"))
bridge_cmd = (
    "source /opt/ros/humble/setup.bash && "
    "ros2 run ros_gz_bridge parameter_bridge "
    + " ".join(bridge_topics)
)
bridge = run_bg(bridge_cmd)
time.sleep(2)

rviz = None
camera_view = None

if ENABLE_RVIZ:
    print(f"Starting RViz with config: {RVIZ_CONFIG_PATH}")
    rviz_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"rviz2 -d {RVIZ_CONFIG_PATH}"
    )
    rviz = run_bg(rviz_cmd)
    time.sleep(2)

if ENABLE_CAMERA_VIEW:
    print("Starting camera viewer...")
    camera_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "ros2 run rqt_image_view rqt_image_view"
    )
    camera_view = run_bg(camera_cmd)
    time.sleep(2)

omnet = None
if ENABLE_UAV:
    print("OMNeT++ relay disabled for this run.")

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
        "/husky_local/controller_state "
        "/husky_2/controller_state "
        "/husky_local/obstacle_action "
        "/husky_2/obstacle_action "
        "/husky_local/obstacle_clearance "
        "/husky_2/obstacle_clearance "
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
        f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/camera_front/image "
        f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/camera_front/image "
        f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/imu_sensor/imu "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/imu_sensor/imu "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/air_pressure/air_pressure "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/magnetometer/magnetometer "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points "
        f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/camera_front/image "
    )
    recorder = run_bg(record_cmd)
else:
    print("Bag recording disabled. Set ENABLE_BAG_RECORDING = True to record a run.")

print("\n==============================")
print(" MODEL MODE ENABLED ")
print("==============================")
print("Press Play in Gazebo")
print("Press Ctrl+C here when done.\n")

rclpy.init()
# Drive from Gazebo world pose so the controller and the visible goal marker use
# one consistent coordinate frame.
HUSKY1_GOAL = offset_goal_along_path(
    WORLD_HUSKY1_GOAL,
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    GOAL_STOP_OFFSET,
)
HUSKY2_GOAL = offset_goal_along_path(
    WORLD_HUSKY2_GOAL,
    (HUSKY2_X, HUSKY2_Y, HUSKY2_Z),
    GOAL_STOP_OFFSET,
)
UAV_GOAL = offset_goal_along_path(
    WORLD_UAV_GOAL,
    (UAV_X, UAV_Y, UAV_Z),
    GOAL_STOP_OFFSET,
)
print(
    "Controller goals (world frame, stop offset applied): "
    f"husky_local=({HUSKY1_GOAL[0]:.3f}, {HUSKY1_GOAL[1]:.3f}), "
    f"husky_2=({HUSKY2_GOAL[0]:.3f}, {HUSKY2_GOAL[1]:.3f}), "
    f"uav=({UAV_GOAL[0]:.3f}, {UAV_GOAL[1]:.3f})"
)
print(
    f"Visible goal marker remains at ({WORLD_HUSKY1_GOAL[0]:.3f}, {WORLD_HUSKY1_GOAL[1]:.3f}); "
    f"controller stop tolerance is {GOAL_TOLERANCE:.2f} m."
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
husky1_obstacle_action_topic = "/husky_local/obstacle_action"
husky1_obstacle_clearance_topic = "/husky_local/obstacle_clearance"
husky1_controller_state_topic = "/husky_local/controller_state"
uav_ready_topic = "/uav1/ready"
obstacle_detector = ObstacleDetectionNode(
    node_name="husky_local_obstacle_detector",
    scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
    action_topic=husky1_obstacle_action_topic,
    clearance_topic=husky1_obstacle_clearance_topic,
    pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
    front_half_angle_deg=OBSTACLE_FRONT_HALF_ANGLE_DEG,
    side_angle_deg=OBSTACLE_SIDE_ANGLE_DEG,
    stop_distance=OBSTACLE_STOP_DISTANCE,
    caution_distance=OBSTACLE_CAUTION_DISTANCE,
)
driver = ModelHuskyDriver(
    node_name="model_husky_driver_1",
    cmd_topic="/cmd_vel",
    odom_topic="/model/husky_local/odometry",
    world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
    uav_ready_topic=None,
    require_uav_ready=False,
    obstacle_action_topic=husky1_obstacle_action_topic,
    obstacle_clearance_topic=husky1_obstacle_clearance_topic,
    state_topic=husky1_controller_state_topic,
    goal_xyz=HUSKY1_GOAL,
    world_goal_xyz=HUSKY1_GOAL,
    bootstrap_seconds=BOOTSTRAP_SECONDS,
    bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
    control_period=CONTROL_PERIOD,
    cmd_linear_gain=CMD_LINEAR_GAIN,
    cmd_angular_gain=CMD_ANGULAR_GAIN,
    min_linear_speed=MIN_LINEAR_SPEED,
    max_linear_speed=MAX_LINEAR_SPEED,
    max_angular_speed=MAX_ANGULAR_SPEED,
    heading_deadband=HEADING_DEADBAND,
    goal_tolerance=GOAL_TOLERANCE,
    stuck_timeout_seconds=STUCK_TIMEOUT_SECONDS,
    stuck_progress_distance=STUCK_PROGRESS_DISTANCE,
    stuck_reverse_speed=STUCK_REVERSE_SPEED,
    stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
    stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
)
driver2 = None
obstacle_detector2 = None
if ENABLE_SECOND_HUSKY:
    husky2_obstacle_action_topic = "/husky_2/obstacle_action"
    husky2_obstacle_clearance_topic = "/husky_2/obstacle_clearance"
    husky2_controller_state_topic = "/husky_2/controller_state"
    obstacle_detector2 = ObstacleDetectionNode(
        node_name="husky_2_obstacle_detector",
        scan_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
        action_topic=husky2_obstacle_action_topic,
        clearance_topic=husky2_obstacle_clearance_topic,
        pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
        front_half_angle_deg=OBSTACLE_FRONT_HALF_ANGLE_DEG,
        side_angle_deg=OBSTACLE_SIDE_ANGLE_DEG,
        stop_distance=OBSTACLE_STOP_DISTANCE,
        caution_distance=OBSTACLE_CAUTION_DISTANCE,
    )
    driver2 = ModelHuskyDriver(
        node_name="model_husky_driver_2",
        cmd_topic="/cmd_vel_husky2",
        odom_topic="/model/husky_2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        uav_ready_topic=None,
        require_uav_ready=False,
        obstacle_action_topic=husky2_obstacle_action_topic,
        obstacle_clearance_topic=husky2_obstacle_clearance_topic,
        state_topic=husky2_controller_state_topic,
        goal_xyz=HUSKY2_GOAL,
        world_goal_xyz=HUSKY2_GOAL,
        bootstrap_seconds=BOOTSTRAP_SECONDS,
        bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
        control_period=CONTROL_PERIOD,
        cmd_linear_gain=CMD_LINEAR_GAIN,
        cmd_angular_gain=CMD_ANGULAR_GAIN,
        min_linear_speed=MIN_LINEAR_SPEED,
        max_linear_speed=MAX_LINEAR_SPEED,
        max_angular_speed=MAX_ANGULAR_SPEED,
        heading_deadband=HEADING_DEADBAND,
        goal_tolerance=GOAL_TOLERANCE,
        stuck_timeout_seconds=STUCK_TIMEOUT_SECONDS,
        stuck_progress_distance=STUCK_PROGRESS_DISTANCE,
        stuck_reverse_speed=STUCK_REVERSE_SPEED,
        stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
        stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
    )
follower = None
if ENABLE_UAV:
    follower = UavFollower(
        husky_odom_topic="/model/husky_local/odometry",
        uav_odom_topic="/model/uav1/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        husky_model_name="husky_local",
        uav_model_name="uav1",
        uav_name="uav1",
        follow_distance=UAV_FOLLOW_DISTANCE,
        follow_height=UAV_FOLLOW_HEIGHT,
        ready_topic=uav_ready_topic,
        update_period=UAV_UPDATE_PERIOD,
        max_xy_speed=UAV_MAX_XY_SPEED,
        max_z_speed=UAV_MAX_Z_SPEED,
        max_yaw_rate=UAV_MAX_YAW_RATE,
        xy_gain=UAV_XY_GAIN,
        z_gain=UAV_Z_GAIN,
        yaw_gain=UAV_YAW_GAIN,
        heading_align_gain=UAV_HEADING_ALIGN_GAIN,
        min_forward_speed=UAV_MIN_FORWARD_SPEED,
        target_smoothing=UAV_TARGET_SMOOTHING,
        xy_deadband=UAV_XY_DEADBAND,
        z_deadband=UAV_Z_DEADBAND,
        yaw_deadband=UAV_YAW_DEADBAND,
        min_track_speed=UAV_MIN_TRACK_SPEED,
        husky_spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        husky_spawn_yaw=HUSKY1_SPAWN_YAW,
        uav_spawn_xyz=(UAV_X, UAV_Y, UAV_Z),
        uav_spawn_yaw=UAV_SPAWN_YAW,
    )
executor = MultiThreadedExecutor()
executor.add_node(episode_metadata)
executor.add_node(obstacle_detector)
executor.add_node(driver)
if obstacle_detector2 is not None:
    executor.add_node(obstacle_detector2)
if driver2 is not None:
    executor.add_node(driver2)
if follower is not None:
    executor.add_node(follower)

try:
    executor.spin()
except KeyboardInterrupt:
    print("\nStopping model run...")
finally:
    executor.shutdown()
    episode_metadata.destroy_node()
    obstacle_detector.destroy_node()
    driver.destroy_node()
    if obstacle_detector2 is not None:
        obstacle_detector2.destroy_node()
    if driver2 is not None:
        driver2.destroy_node()
    if follower is not None:
        follower.destroy_node()
    rclpy.shutdown()
    if recorder is not None:
        recorder.send_signal(signal.SIGINT)
        time.sleep(2)
    print("Stopping bridge, OMNeT++, and Gazebo...")
    bridge.send_signal(signal.SIGINT)
    if rviz is not None:
        rviz.send_signal(signal.SIGINT)
    if camera_view is not None:
        camera_view.send_signal(signal.SIGINT)
    if omnet is not None:
        omnet.send_signal(signal.SIGINT)
    gz.send_signal(signal.SIGINT)
    time.sleep(2)
    print("All processes stopped cleanly.")
