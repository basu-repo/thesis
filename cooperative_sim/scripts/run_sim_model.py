"""Launch the live CNN-LSTM simulation pipeline.

This runner starts Gazebo, spawns the Husky/UAV agents, publishes episode
metadata, optionally records a bag, and connects the working CNN-based Husky
controller to the live odometry stream.
"""

import datetime
import argparse
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RULE_BASED_ROOT = SCRIPT_DIR.parent
GUI_CONFIG = RULE_BASED_ROOT.parent / "simulation" / "gui" / "baylands_gui.config"
if str(RULE_BASED_ROOT) not in sys.path:
    sys.path.insert(0, str(RULE_BASED_ROOT))

import rclpy
from rclpy.executors import MultiThreadedExecutor

from controllers.episode_metadata import EpisodeMetadataPublisher
from controllers.depth_image_classification import RuleBasedDepthClassifier
from controllers.hazard_map_builder import HazardMapBuilderNode
from controllers.obstacle_detection import ObstacleDetectionNode
from controllers.resource_usage_monitor import RunResourceMonitor
from controllers.husky_model_driver import ModelHuskyDriver
from controllers.scout_coordinator import ScoutCoordinatorNode
from controllers.ugv_decision_fuser import UgvDecisionFuser
from controllers.uav_obstacle_detection import UavObstacleDetectionNode
from controllers.uav_scout_driver import UavScoutDriver
from project_paths import MODELS_DIR, OMNET_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH


WORLD = str(WORLD_SDF_PATH)
WORLD_NAME = "baylands"
MODEL_PATH = str(MODELS_DIR)
OMNET_BIN = OMNET_DIR / "onmetpp"
OMNET_CONFIG = "WifiRelay"
RUN_START_DT = datetime.datetime.now()
LOG_DIR = Path.home() / "Documents/Thesis/dataset/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = LOG_DIR / f"rule_based_run_{RUN_START_DT.strftime('%Y%m%d_%H%M%S')}.log"
TEE_PROCESS = None
ORIGINAL_STDOUT_FD = None
ORIGINAL_STDERR_FD = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run Gazebo server-only with headless rendering.")
    parser.add_argument("--no-rviz", action="store_true", help="Skip RViz.")
    parser.add_argument("--no-camera", action="store_true", help="Skip the image viewer.")
    parser.add_argument("--no-bag", action="store_true", help="Disable ROS bag recording for the run.")
    parser.add_argument("--no-depth", action="store_true", help="Disable UGV depth-image classification.")
    parser.add_argument("--no-hazard-map", action="store_true", help="Disable hazard-map fusion.")
    parser.add_argument("--enable-decision-fuser", action="store_true", help="Enable the UGV decision-fuser node.")
    parser.add_argument("--disable-uavs", action="store_true", help="Disable both UAV scout agents.")
    parser.add_argument("--disable-second-uav", action="store_true", help="Disable only the second UAV scout agent.")
    parser.add_argument("--enable-lidar-path-planning", action="store_true", help="Enable lidar path-planning mode.")
    parser.add_argument("--disable-lidar-straight-approach", action="store_true", help="Disable lidar straight-approach mode.")
    parser.add_argument("--debug-isolate-husky-local", action="store_true", help="Spawn only husky_local for debugging.")
    return parser.parse_args()


def setup_terminal_tee(log_path: Path):
    global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD
    if TEE_PROCESS is not None:
        return

    ORIGINAL_STDOUT_FD = os.dup(sys.__stdout__.fileno())
    ORIGINAL_STDERR_FD = os.dup(sys.__stderr__.fileno())
    TEE_PROCESS = subprocess.Popen(
        ["tee", "-a", str(log_path)],
        stdin=subprocess.PIPE,
        stdout=ORIGINAL_STDOUT_FD,
        stderr=ORIGINAL_STDERR_FD,
        bufsize=0,
    )
    os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stdout__.fileno())
    os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stderr__.fileno())


def close_terminal_tee():
    global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD
    if TEE_PROCESS is None:
        return

    with suppress(Exception):
        sys.stdout.flush()
    with suppress(Exception):
        sys.stderr.flush()
    if ORIGINAL_STDOUT_FD is not None:
        with suppress(Exception):
            os.dup2(ORIGINAL_STDOUT_FD, sys.__stdout__.fileno())
    if ORIGINAL_STDERR_FD is not None:
        with suppress(Exception):
            os.dup2(ORIGINAL_STDERR_FD, sys.__stderr__.fileno())
    with suppress(Exception):
        if TEE_PROCESS.stdin is not None:
            TEE_PROCESS.stdin.close()
    with suppress(Exception):
        TEE_PROCESS.wait(timeout=2.0)
    if ORIGINAL_STDOUT_FD is not None:
        with suppress(Exception):
            os.close(ORIGINAL_STDOUT_FD)
    if ORIGINAL_STDERR_FD is not None:
        with suppress(Exception):
            os.close(ORIGINAL_STDERR_FD)
    TEE_PROCESS = None
    ORIGINAL_STDOUT_FD = None
    ORIGINAL_STDERR_FD = None


def log_event(message: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with suppress(Exception):
        sys.stdout.flush()


ARGS = parse_args()

SPAWN_X, SPAWN_Y, SPAWN_Z = -216.8780, -166.3550, -1.37494
HUSKY2_X, HUSKY2_Y, HUSKY2_Z = 108.00, -275.00, 0.68
UAV_X, UAV_Y, UAV_Z = 106.035, -271.38, 11.00
UAV2_X, UAV2_Y, UAV2_Z = 107.369, -280.957, 11.00

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


def yaw_toward_goal(
    start_xyz: tuple[float, float, float],
    world_goal: tuple[float, float, float],
) -> float:
    dx = float(world_goal[0]) - float(start_xyz[0])
    dy = float(world_goal[1]) - float(start_xyz[1])
    return math.atan2(dy, dx)


def yaw_to_quat(yaw: float) -> tuple[float, float]:
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# Match the 07 rule-based Baylands scenario directly for cross-run comparison.
RAW_WORLD_SHARED_GOAL = (-35, -290.30, 0.35)
GOAL_WORLD_PULLBACK = 0.0
WORLD_SHARED_GOAL = offset_goal_along_path(
    RAW_WORLD_SHARED_GOAL,
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    -GOAL_WORLD_PULLBACK,
)
WORLD_HUSKY2_GOAL = WORLD_SHARED_GOAL
WORLD_HUSKY1_GOAL = WORLD_SHARED_GOAL
WORLD_UAV_GOAL = WORLD_SHARED_GOAL
GROUND_MARKER_Z = 0.832744
GOAL_STOP_OFFSET = -0.5

HUSKY1_SPAWN_YAW = yaw_toward_goal((SPAWN_X, SPAWN_Y, SPAWN_Z), WORLD_HUSKY1_GOAL)
HUSKY2_SPAWN_YAW = yaw_toward_goal((HUSKY2_X, HUSKY2_Y, HUSKY2_Z), WORLD_HUSKY2_GOAL)
UAV_SPAWN_YAW = -3.036460
UAV2_SPAWN_YAW = -3.033500
HUSKY1_SPAWN_QZ, HUSKY1_SPAWN_QW = yaw_to_quat(HUSKY1_SPAWN_YAW)
HUSKY2_SPAWN_QZ, HUSKY2_SPAWN_QW = yaw_to_quat(HUSKY2_SPAWN_YAW)
UAV_SPAWN_QZ, UAV_SPAWN_QW = yaw_to_quat(UAV_SPAWN_YAW)
UAV2_SPAWN_QZ, UAV2_SPAWN_QW = yaw_to_quat(UAV2_SPAWN_YAW)

BOOTSTRAP_SECONDS = 3.0
BOOTSTRAP_LINEAR_SPEED = 0.8
ENABLE_PRIMARY_HUSKY = False
ENABLE_SECOND_HUSKY = True
ENABLE_UAV = True
ENABLE_SECOND_UAV = True
DEBUG_ISOLATE_HUSKY_LOCAL = False
ENABLE_HEADLESS = False

CONTROL_PERIOD = 0.1
CMD_LINEAR_GAIN = 1.45
CMD_ANGULAR_GAIN = 1.15
MIN_LINEAR_SPEED = 1.5
MAX_LINEAR_SPEED = 2.0
MAX_ANGULAR_SPEED = 0.85
HEADING_DEADBAND = 0.12
GOAL_TOLERANCE = 1.5
STUCK_TIMEOUT_SECONDS = 3.0
STUCK_PROGRESS_DISTANCE = 0.01
STUCK_REVERSE_SPEED = -2.0
STUCK_REVERSE_SECONDS = 3.0
STUCK_BOOTSTRAP_SECONDS = 2.0
OBSTACLE_FRONT_HALF_ANGLE_DEG = 45.0
OBSTACLE_SIDE_ANGLE_DEG = 65.0
OBSTACLE_STOP_DISTANCE = 1.8
OBSTACLE_CAUTION_DISTANCE = 3.2
# Gap-based passability is disabled for now.
# UGV_PASSABLE_GAP_WIDTH_M = 1.25
# UGV_PASSABLE_GAP_MARGIN_M = 0.20

UAV_CRUISE_HEIGHT = 10.0
UAV_SCOUT_ALTITUDE_Z = 35.97
UAV_SCOUT_SLOT_FORWARD = 2.5
UAV_SCOUT_SLOT_LATERAL = 1.5
UAV_UPDATE_PERIOD = 0.1
UAV_MAX_XY_SPEED = 14.0
UAV_MAX_Z_SPEED = 5.0
UAV_MAX_YAW_RATE = 0.9
UAV_XY_GAIN = 4.8
UAV_Z_GAIN = 0.9
UAV_YAW_GAIN = 0.8
UAV_HEADING_ALIGN_GAIN = 0.9
UAV_MIN_FORWARD_SPEED = 0.25
UAV_TARGET_SMOOTHING = 1.0
UAV_XY_DEADBAND = 0.02
UAV_Z_DEADBAND = 0.15
UAV_YAW_DEADBAND = 0.18
UAV_MIN_TRACK_SPEED = 0.0
ENABLE_BAG_RECORDING = True
ENABLE_RVIZ = True
ENABLE_CAMERA_VIEW = True
ENABLE_LIDAR_STRAIGHT_APPROACH = True
ENABLE_LIDAR_PATH_PLANNING = False
ENABLE_DEPTH_IMAGE_CLASSIFICATION = True
ENABLE_HAZARD_MAP = True
ENABLE_UGV_DECISION_FUSER = False

ENABLE_HEADLESS = bool(ARGS.headless)
ENABLE_BAG_RECORDING = not bool(ARGS.no_bag)
ENABLE_RVIZ = not bool(ARGS.no_rviz)
ENABLE_CAMERA_VIEW = not bool(ARGS.no_camera)
ENABLE_LIDAR_STRAIGHT_APPROACH = not bool(ARGS.disable_lidar_straight_approach)
ENABLE_LIDAR_PATH_PLANNING = bool(ARGS.enable_lidar_path_planning)
ENABLE_DEPTH_IMAGE_CLASSIFICATION = not bool(ARGS.no_depth)
ENABLE_HAZARD_MAP = not bool(ARGS.no_hazard_map)
ENABLE_UGV_DECISION_FUSER = bool(ARGS.enable_decision_fuser)
DEBUG_ISOLATE_HUSKY_LOCAL = bool(ARGS.debug_isolate_husky_local)

if ARGS.disable_uavs:
    ENABLE_UAV = False
    ENABLE_SECOND_UAV = False
elif ARGS.disable_second_uav:
    ENABLE_SECOND_UAV = False

if DEBUG_ISOLATE_HUSKY_LOCAL:
    ENABLE_SECOND_HUSKY = False
    ENABLE_UAV = False
    ENABLE_SECOND_UAV = False
    ENABLE_BAG_RECORDING = False


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
      <visual name="marker_visual">
        <pose>0 0 1.0 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>5.0</length>
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


def build_bag_topics(
    world_name: str,
    *,
    include_primary_husky: bool,
    include_second_husky: bool,
    include_uav: bool,
    include_second_uav: bool,
) -> list[str]:
    """Return the smallest useful topic set for trajectory-focused dataset collection."""

    topics = [
        "/clock",
        f"/world/{world_name}/dynamic_pose/info",
    ]

    if include_primary_husky:
        topics.extend(
            [
                "/cmd_vel",
                "/husky_local/controller_state",
                "/husky_local/obstacle_action",
                "/husky_local/obstacle_clearance",
                "/episode/husky_local/start",
                "/episode/husky_local/goal",
                "/model/husky_local/odometry",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/imu_sensor/imu",
            ]
        )

    if include_second_husky:
        topics.extend(
            [
                "/cmd_vel_husky2",
                "/husky_2/controller_state",
                "/husky_2/obstacle_action",
                "/husky_2/obstacle_clearance",
                "/episode/husky_2/start",
                "/episode/husky_2/goal",
                "/model/husky_2/odometry",
                f"/world/{world_name}/model/husky_2/link/base_link/sensor/planar_laser/scan",
                f"/world/{world_name}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
                f"/world/{world_name}/model/husky_2/link/base_link/sensor/imu_sensor/imu",
            ]
        )

    if include_uav:
        topics.extend(
            [
                "/episode/uav1/start",
                "/episode/uav1/goal",
                "/uav1/controller_state",
                "/uav1/obstacle_action",
                "/uav1/obstacle_clearance",
                "/model/uav1/odometry",
                "/uav1/command/twist",
                "/model/uav1/command/twist",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/front_laser/scan/points",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/imu_sensor/imu",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/air_pressure/air_pressure",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/magnetometer/magnetometer",
            ]
        )

    if include_second_uav:
        topics.extend(
            [
                "/episode/uav2/start",
                "/episode/uav2/goal",
                "/uav2/controller_state",
                "/uav2/obstacle_action",
                "/uav2/obstacle_clearance",
                "/model/uav2/odometry",
                "/uav2/command/twist",
                "/model/uav2/command/twist",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/front_laser/scan/points",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/imu_sensor/imu",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/air_pressure/air_pressure",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/magnetometer/magnetometer",
            ]
        )

    return topics


os.environ["IGN_GAZEBO_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

setup_terminal_tee(RUN_LOG_PATH)
subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
subprocess.run(["bash", "-c", "pkill -f ign || true"])
subprocess.run(["bash", "-c", f"pkill -f {OMNET_BIN} || true"])

log_event(f"START timestamp: {RUN_START_DT.isoformat(timespec='seconds')}")
log_event(f"Run log file: {RUN_LOG_PATH}")
log_event(
    "Feature flags: "
    f"headless={ENABLE_HEADLESS}, "
    f"rviz={ENABLE_RVIZ}, "
    f"camera={ENABLE_CAMERA_VIEW}, "
    f"bag={ENABLE_BAG_RECORDING}, "
    f"depth={ENABLE_DEPTH_IMAGE_CLASSIFICATION}, "
    f"hazard_map={ENABLE_HAZARD_MAP}, "
    f"decision_fuser={ENABLE_UGV_DECISION_FUSER}, "
    f"uav1={ENABLE_UAV}, "
    f"uav2={ENABLE_SECOND_UAV}"
)
log_event("Starting Gazebo...")
gazebo_cmd = f"ign gazebo --gui-config {GUI_CONFIG} {WORLD}"
if ENABLE_HEADLESS:
    gazebo_cmd = f"ign gazebo -s -r --headless-rendering {WORLD}"
gz = run_bg(gazebo_cmd)
time.sleep(5)

log_event("Waiting for Baylands world to fully load...")
time.sleep(20)

if ENABLE_PRIMARY_HUSKY:
    log_event("Spawning Husky...")
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
    log_event("Spawning Husky 2...")
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
    log_event("Spawning UAV...")
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

if ENABLE_SECOND_UAV:
    log_event("Spawning UAV 2...")
    spawn_uav2 = """
ign service -s /world/{world_name}/create \
--reqtype ignition.msgs.EntityFactory \
--reptype ignition.msgs.Boolean \
--timeout 5000 \
--req 'sdf_filename: "model://m100/model.sdf", name: "uav2",
pose: {{position: {{x: {uav_x}, y: {uav_y}, z: {uav_z}}}, orientation: {{z: {uav_qz}, w: {uav_qw}}}}}'
""".format(
        world_name=WORLD_NAME,
        uav_x=UAV2_X,
        uav_y=UAV2_Y,
        uav_z=UAV2_Z,
        uav_qz=UAV2_SPAWN_QZ,
        uav_qw=UAV2_SPAWN_QW,
    )
    subprocess.run(["bash", "-c", spawn_uav2])
    time.sleep(5)

log_event("Spawning goal markers...")
# spawn_goal_marker(WORLD_NAME, "start_husky_local", (SPAWN_X, SPAWN_Y, 0.0387), (0.65, 0.15, 0.15, 1.0))
if ENABLE_PRIMARY_HUSKY:
    spawn_goal_marker(WORLD_NAME, "goal_husky_local", (WORLD_HUSKY1_GOAL[0], WORLD_HUSKY1_GOAL[1], GROUND_MARKER_Z), (0.95, 0.12, 0.12, 1.0))
if ENABLE_SECOND_HUSKY:
    spawn_goal_marker(WORLD_NAME, "goal_husky_2", (WORLD_HUSKY2_GOAL[0], WORLD_HUSKY2_GOAL[1], GROUND_MARKER_Z), (0.12, 0.36, 0.95, 1.0))
if ENABLE_UAV:
    spawn_goal_marker(WORLD_NAME, "goal_uav1", (WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1], GROUND_MARKER_Z), (0.95, 0.85, 0.12, 1.0))
if ENABLE_SECOND_UAV:
    spawn_goal_marker(WORLD_NAME, "goal_uav2", (WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1], GROUND_MARKER_Z), (0.25, 0.95, 0.65, 1.0))
time.sleep(1)

log_event("Starting bridge...")
bridge_topics = [
    f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
]
if ENABLE_PRIMARY_HUSKY:
    bridge_topics.extend(
        [
            "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        ]
    )
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
if ENABLE_SECOND_UAV:
    bridge_topics.extend(
        [
            "/uav2/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/uav2/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
            "/model/uav2/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/model/uav2/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
            "/model/uav2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            f"/world/{WORLD_NAME}/model/uav2/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            f"/world/{WORLD_NAME}/model/uav2/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            f"/world/{WORLD_NAME}/model/uav2/link/base_link/sensor/air_pressure/air_pressure@sensor_msgs/msg/FluidPressure[ignition.msgs.FluidPressure",
            f"/world/{WORLD_NAME}/model/uav2/link/base_link/sensor/magnetometer/magnetometer@sensor_msgs/msg/MagneticField[ignition.msgs.Magnetometer",
        ]
    )
    bridge_topics.extend(camera_bridge_topics(WORLD_NAME, "uav2", "base_link", "camera_front"))
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
    log_event(f"Starting RViz with config: {RVIZ_CONFIG_PATH}")
    rviz_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"rviz2 -d {RVIZ_CONFIG_PATH}"
    )
    rviz = run_bg(rviz_cmd)
    time.sleep(2)

omnet = None
if ENABLE_UAV:
    log_event("OMNeT++ relay disabled for this run.")

BAG_DIR = os.path.expanduser("~/Documents/Thesis/dataset/bags")
os.makedirs(BAG_DIR, exist_ok=True)
run_name = "run_model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bag_path = f"{BAG_DIR}/{run_name}"
recorder = None
bag_topics = []
if ENABLE_BAG_RECORDING:
    bag_topics = build_bag_topics(
        WORLD_NAME,
        include_primary_husky=ENABLE_PRIMARY_HUSKY,
        include_second_husky=ENABLE_SECOND_HUSKY,
        include_uav=ENABLE_UAV,
        include_second_uav=ENABLE_SECOND_UAV,
    )
else:
    log_event("Bag recording disabled. Set ENABLE_BAG_RECORDING = True to record a run.")

if DEBUG_ISOLATE_HUSKY_LOCAL:
    log_event("DEBUG isolate mode: only husky_local is spawned; husky_2, uav1, RViz, and camera viewer are disabled. Bag recording remains enabled.")

log_event("==============================")
log_event("MODEL MODE ENABLED")
log_event("==============================")
log_event("Press Play in Gazebo")
log_event("Press Ctrl+C here when done.")

if ENABLE_BAG_RECORDING:
    log_event(f"Recording bag: {bag_path}")
    log_event(
        "Bag topic set for trajectory collection: "
        + ", ".join(bag_topics)
    )
    record_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"ros2 bag record -o {bag_path} "
        + " ".join(bag_topics)
    )
    recorder = run_bg(record_cmd)

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
UAV2_GOAL = offset_goal_along_path(
    WORLD_UAV_GOAL,
    (UAV2_X, UAV2_Y, UAV2_Z),
    GOAL_STOP_OFFSET,
)
goal_parts = []
if ENABLE_PRIMARY_HUSKY:
    goal_parts.append(f"husky_local=({HUSKY1_GOAL[0]:.3f}, {HUSKY1_GOAL[1]:.3f})")
if ENABLE_SECOND_HUSKY:
    goal_parts.append(f"husky_2=({HUSKY2_GOAL[0]:.3f}, {HUSKY2_GOAL[1]:.3f})")
if ENABLE_UAV:
    goal_parts.append(f"uav=({UAV_GOAL[0]:.3f}, {UAV_GOAL[1]:.3f})")
if ENABLE_SECOND_UAV:
    goal_parts.append(f"uav2=({UAV2_GOAL[0]:.3f}, {UAV2_GOAL[1]:.3f})")
log_event("Controller goals (world frame, stop offset applied): " + ", ".join(goal_parts))
if ENABLE_UAV or ENABLE_SECOND_UAV:
    log_event(
        f"Scout altitude target: z={UAV_SCOUT_ALTITUDE_Z:.2f}, "
        f"slot_forward={UAV_SCOUT_SLOT_FORWARD:.1f} m, slot_lateral={UAV_SCOUT_SLOT_LATERAL:.1f} m"
    )
if ENABLE_PRIMARY_HUSKY:
    log_event(
        f"Visible goal marker remains at ({WORLD_HUSKY1_GOAL[0]:.3f}, {WORLD_HUSKY1_GOAL[1]:.3f}); "
        f"controller stop tolerance is {GOAL_TOLERANCE:.2f} m."
    )
elif ENABLE_SECOND_HUSKY:
    log_event(
        f"Visible goal marker remains at ({WORLD_HUSKY2_GOAL[0]:.3f}, {WORLD_HUSKY2_GOAL[1]:.3f}); "
        f"controller stop tolerance is {GOAL_TOLERANCE:.2f} m."
    )
# ROS 2 nodes for metadata, learned control, UAV support, and network effects.
start_goals = {}
if ENABLE_PRIMARY_HUSKY:
    start_goals["husky_local"] = {"start": (SPAWN_X, SPAWN_Y, SPAWN_Z), "goal": WORLD_HUSKY1_GOAL}
if ENABLE_SECOND_HUSKY:
    start_goals["husky_2"] = {"start": (HUSKY2_X, HUSKY2_Y, HUSKY2_Z), "goal": WORLD_HUSKY2_GOAL}
if ENABLE_UAV:
    start_goals["uav1"] = {"start": (UAV_X, UAV_Y, UAV_Z), "goal": WORLD_UAV_GOAL}
if ENABLE_SECOND_UAV:
    start_goals["uav2"] = {"start": (UAV2_X, UAV2_Y, UAV2_Z), "goal": WORLD_UAV_GOAL}

episode_metadata = EpisodeMetadataPublisher(
    world_name=WORLD_NAME,
    start_goals=start_goals,
)
resource_summary_path = RUN_LOG_PATH.with_name(RUN_LOG_PATH.stem + "_resource_summary.json")
resource_samples_path = RUN_LOG_PATH.with_name(RUN_LOG_PATH.stem + "_resource_samples.csv")
resource_monitor = RunResourceMonitor(
    node_name="run_resource_monitor",
    controller_state_topic="/husky_2/controller_state" if ENABLE_SECOND_HUSKY else "/husky_local/controller_state",
    clock_topic="/clock",
    summary_path=resource_summary_path,
    samples_path=resource_samples_path,
    log_fn=log_event,
)
husky1_obstacle_action_topic = "/husky_local/obstacle_action"
husky1_obstacle_clearance_topic = "/husky_local/obstacle_clearance"
husky1_terrain_profile_topic = "/husky_local/terrain_profile"
husky1_controller_state_topic = "/husky_local/controller_state"
obstacle_detector = None
driver = None
if ENABLE_PRIMARY_HUSKY:
    obstacle_detector = ObstacleDetectionNode(
        node_name="husky_local_obstacle_detector",
        scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
        action_topic=husky1_obstacle_action_topic,
        clearance_topic=husky1_obstacle_clearance_topic,
        terrain_profile_topic=husky1_terrain_profile_topic,
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
        terrain_profile_topic=husky1_terrain_profile_topic,
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
        obstacle_caution_distance=OBSTACLE_CAUTION_DISTANCE,
        stuck_reverse_speed=STUCK_REVERSE_SPEED,
        stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
        stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
    )
driver2 = None
obstacle_detector2 = None
scout_coordinator = None
hazard_map_builder = None
husky2_depth_classifier = None
ugv_decision_fuser = None
if ENABLE_SECOND_HUSKY:
    husky2_obstacle_action_topic = "/husky_2/obstacle_action"
    husky2_obstacle_clearance_topic = "/husky_2/obstacle_clearance"
    husky2_terrain_profile_topic = "/husky_2/terrain_profile"
    husky2_gap_profile_topic = None
    husky2_controller_state_topic = "/husky_2/controller_state"
    obstacle_detector2 = ObstacleDetectionNode(
        node_name="husky_2_obstacle_detector",
        scan_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
        action_topic=husky2_obstacle_action_topic,
        clearance_topic=husky2_obstacle_clearance_topic,
        terrain_profile_topic=husky2_terrain_profile_topic,
        gap_profile_topic=husky2_gap_profile_topic,
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
        uav_ready_topic="/scouts/ready" if (ENABLE_UAV or ENABLE_SECOND_UAV) else None,
        require_uav_ready=False,
        obstacle_action_topic=husky2_obstacle_action_topic,
        obstacle_clearance_topic=husky2_obstacle_clearance_topic,
        terrain_profile_topic=husky2_terrain_profile_topic,
        gap_profile_topic=husky2_gap_profile_topic,
        hazard_guidance_topic="/husky_2/hazard_guidance" if ENABLE_HAZARD_MAP else None,
        depth_classification_topic="/husky_2/depth_classification" if ENABLE_DEPTH_IMAGE_CLASSIFICATION else None,
        final_decision_topic="/husky_2/final_decision" if ENABLE_UGV_DECISION_FUSER else None,
        state_topic=husky2_controller_state_topic,
        use_lidar_straight_approach=ENABLE_LIDAR_STRAIGHT_APPROACH,
        use_lidar_path_planning=ENABLE_LIDAR_PATH_PLANNING,
        use_depth_classification=ENABLE_DEPTH_IMAGE_CLASSIFICATION,
        use_hazard_map=ENABLE_HAZARD_MAP,
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
    if ENABLE_DEPTH_IMAGE_CLASSIFICATION:
        husky2_depth_classifier = RuleBasedDepthClassifier(
            node_name="husky_2_depth_classifier",
            image_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/camera_front/depth_image",
            classification_topic="/husky_2/depth_classification",
            center_metrics_topic="/husky_2/depth_classification_metrics",
        )
    if ENABLE_UGV_DECISION_FUSER:
        ugv_decision_fuser = UgvDecisionFuser(
            node_name="husky_2_decision_fuser",
            obstacle_action_topic=husky2_obstacle_action_topic,
            obstacle_clearance_topic=husky2_obstacle_clearance_topic,
            hazard_guidance_topic="/husky_2/hazard_guidance",
            depth_classification_topic="/husky_2/depth_classification",
            decision_topic="/husky_2/final_decision",
        )
follower = None
if ENABLE_UAV:
    uav_obstacle_action_topic = "/uav1/obstacle_action"
    uav_obstacle_clearance_topic = "/uav1/obstacle_clearance"
    uav_controller_state_topic = "/uav1/controller_state"
    uav_ready_topic = "/uav1/scout_ready"
    uav_report_topic = "/uav1/scout_report"
    obstacle_detector_uav = UavObstacleDetectionNode(
        node_name="uav1_obstacle_detector",
        pointcloud_topic=f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points",
        action_topic=uav_obstacle_action_topic,
        clearance_topic=uav_obstacle_clearance_topic,
        max_forward_x=25.0,
        front_half_width_y=3.0,
        side_width_y=8.0,
        front_min_z=-6.0,
        front_max_z=6.0,
        up_min_z=1.5,
        up_max_z=10.0,
        stop_distance=5.0,
        caution_distance=10.0,
        min_points=8,
    )
    follower = UavScoutDriver(
        node_name="uav1_scout_driver",
        uav_name="uav1",
        husky_name="husky_2",
        husky_state_topic=husky2_controller_state_topic if ENABLE_SECOND_HUSKY else None,
        odom_topic="/model/uav1/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        obstacle_action_topic=uav_obstacle_action_topic,
        obstacle_clearance_topic=uav_obstacle_clearance_topic,
        state_topic=uav_controller_state_topic,
        ready_topic=uav_ready_topic,
        report_topic=uav_report_topic,
        scout_altitude_z=UAV_SCOUT_ALTITUDE_Z,
        slot_forward_m=UAV_SCOUT_SLOT_FORWARD,
        slot_lateral_m=-UAV_SCOUT_SLOT_LATERAL,
        control_period=UAV_UPDATE_PERIOD,
        max_xy_speed=UAV_MAX_XY_SPEED,
        max_z_speed=UAV_MAX_Z_SPEED,
        max_yaw_rate=UAV_MAX_YAW_RATE,
        xy_gain=UAV_XY_GAIN,
        z_gain=UAV_Z_GAIN,
        yaw_gain=UAV_YAW_GAIN,
        avoid_forward_speed=0.9,
        avoid_lateral_speed=1.8,
        avoid_climb_speed=1.4,
        takeoff_follow_xy_scale=0.80,
        takeoff_release_altitude_margin=8.0,
        slot_ready_radius=2.5,
        slot_catchup_xy_scale=1.15,
        slot_realign_radius=25.0,
        slot_realign_max_xy_speed=10.0,
        max_husky_xy_radius_m=3.0,
        leash_reentry_radius_m=2.0,
        landing_goal_xy=(WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1]),
    )
else:
    obstacle_detector_uav = None
obstacle_detector_uav2 = None
driver_uav2 = None
if ENABLE_SECOND_UAV:
    uav2_obstacle_action_topic = "/uav2/obstacle_action"
    uav2_obstacle_clearance_topic = "/uav2/obstacle_clearance"
    uav2_controller_state_topic = "/uav2/controller_state"
    uav2_ready_topic = "/uav2/scout_ready"
    uav2_report_topic = "/uav2/scout_report"
    obstacle_detector_uav2 = UavObstacleDetectionNode(
        node_name="uav2_obstacle_detector",
        pointcloud_topic=f"/world/{WORLD_NAME}/model/uav2/link/base_link/sensor/front_laser/scan/points",
        action_topic=uav2_obstacle_action_topic,
        clearance_topic=uav2_obstacle_clearance_topic,
        max_forward_x=25.0,
        front_half_width_y=3.0,
        side_width_y=8.0,
        front_min_z=-6.0,
        front_max_z=6.0,
        up_min_z=1.5,
        up_max_z=10.0,
        stop_distance=5.0,
        caution_distance=10.0,
        min_points=8,
    )
    driver_uav2 = UavScoutDriver(
        node_name="uav2_scout_driver",
        uav_name="uav2",
        husky_name="husky_2",
        husky_state_topic=husky2_controller_state_topic if ENABLE_SECOND_HUSKY else None,
        odom_topic="/model/uav2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        obstacle_action_topic=uav2_obstacle_action_topic,
        obstacle_clearance_topic=uav2_obstacle_clearance_topic,
        state_topic=uav2_controller_state_topic,
        ready_topic=uav2_ready_topic,
        report_topic=uav2_report_topic,
        scout_altitude_z=UAV_SCOUT_ALTITUDE_Z,
        slot_forward_m=UAV_SCOUT_SLOT_FORWARD,
        slot_lateral_m=UAV_SCOUT_SLOT_LATERAL,
        control_period=UAV_UPDATE_PERIOD,
        max_xy_speed=UAV_MAX_XY_SPEED,
        max_z_speed=UAV_MAX_Z_SPEED,
        max_yaw_rate=UAV_MAX_YAW_RATE,
        xy_gain=UAV_XY_GAIN,
        z_gain=UAV_Z_GAIN,
        yaw_gain=UAV_YAW_GAIN,
        avoid_forward_speed=0.9,
        avoid_lateral_speed=1.8,
        avoid_climb_speed=1.4,
        takeoff_follow_xy_scale=0.80,
        takeoff_release_altitude_margin=8.0,
        slot_ready_radius=2.5,
        slot_catchup_xy_scale=1.15,
        slot_realign_radius=25.0,
        slot_realign_max_xy_speed=10.0,
        max_husky_xy_radius_m=3.0,
        leash_reentry_radius_m=2.0,
        landing_goal_xy=(WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1]),
    )
if ENABLE_SECOND_HUSKY and (ENABLE_UAV or ENABLE_SECOND_UAV):
    scout_report_topics = []
    scout_ready_topics = []
    if ENABLE_UAV:
        scout_report_topics.append("/uav1/scout_report")
        scout_ready_topics.append("/uav1/scout_ready")
    if ENABLE_SECOND_UAV:
        scout_report_topics.append("/uav2/scout_report")
        scout_ready_topics.append("/uav2/scout_ready")
    scout_coordinator = ScoutCoordinatorNode(
        node_name="scout_coordinator",
        scout_report_topics=scout_report_topics,
        scout_ready_topics=scout_ready_topics,
        husky_obstacle_action_topic=husky2_obstacle_action_topic,
        husky_obstacle_clearance_topic=husky2_obstacle_clearance_topic,
        scouts_ready_topic="/scouts/ready",
        summary_topic="/husky_2/scout_summary",
        min_ready_count=1,
    )
    if ENABLE_HAZARD_MAP:
        hazard_map_builder = HazardMapBuilderNode(
            node_name="hazard_map_builder",
            husky_name="husky_2",
            world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
            scout_report_topics=scout_report_topics,
            husky_obstacle_action_topic=husky2_obstacle_action_topic,
            husky_obstacle_clearance_topic=husky2_obstacle_clearance_topic,
            map_topic="/husky_2/hazard_map",
            memory_map_topic="/husky_2/hazard_map_memory",
            guidance_topic="/husky_2/hazard_guidance",
        )
executor = MultiThreadedExecutor()
executor.add_node(episode_metadata)
executor.add_node(resource_monitor)
if obstacle_detector is not None:
    executor.add_node(obstacle_detector)
if driver is not None:
    executor.add_node(driver)
if obstacle_detector2 is not None:
    executor.add_node(obstacle_detector2)
if driver2 is not None:
    executor.add_node(driver2)
if follower is not None:
    executor.add_node(follower)
if obstacle_detector_uav is not None:
    executor.add_node(obstacle_detector_uav)
if obstacle_detector_uav2 is not None:
    executor.add_node(obstacle_detector_uav2)
if driver_uav2 is not None:
    executor.add_node(driver_uav2)
if scout_coordinator is not None:
    executor.add_node(scout_coordinator)
if hazard_map_builder is not None:
    executor.add_node(hazard_map_builder)
if husky2_depth_classifier is not None:
    executor.add_node(husky2_depth_classifier)
if ugv_decision_fuser is not None:
    executor.add_node(ugv_decision_fuser)

if ENABLE_CAMERA_VIEW:
    log_event("Waiting briefly before starting camera viewer so image topics are available...")
    time.sleep(3)
    log_event("Starting camera viewer...")
    camera_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "ros2 run rqt_image_view rqt_image_view"
    )
    camera_view = run_bg(camera_cmd)
    time.sleep(2)

resource_monitor.set_tracked_processes([gz, bridge, rviz, camera_view, omnet, recorder])

try:
    executor.spin()
except KeyboardInterrupt:
    log_event("Stopping model run...")
finally:
    managed_nodes = [episode_metadata, resource_monitor]
    if obstacle_detector is not None:
        managed_nodes.append(obstacle_detector)
    if driver is not None:
        managed_nodes.append(driver)
    if obstacle_detector2 is not None:
        managed_nodes.append(obstacle_detector2)
    if driver2 is not None:
        managed_nodes.append(driver2)
    if follower is not None:
        managed_nodes.append(follower)
    if obstacle_detector_uav is not None:
        managed_nodes.append(obstacle_detector_uav)
    if obstacle_detector_uav2 is not None:
        managed_nodes.append(obstacle_detector_uav2)
    if driver_uav2 is not None:
        managed_nodes.append(driver_uav2)
    if scout_coordinator is not None:
        managed_nodes.append(scout_coordinator)
    if hazard_map_builder is not None:
        managed_nodes.append(hazard_map_builder)
    if husky2_depth_classifier is not None:
        managed_nodes.append(husky2_depth_classifier)

    for node in managed_nodes:
        with suppress(Exception):
            executor.remove_node(node)
    executor.shutdown(timeout_sec=2.0)
    time.sleep(0.25)

    for node in managed_nodes:
        with suppress(Exception):
            node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()

    if recorder is not None:
        recorder.send_signal(signal.SIGINT)
        time.sleep(2)
    log_event(f"STOP timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log_event(f"Resource summary file: {resource_summary_path}")
    log_event(f"Resource samples file: {resource_samples_path}")
    log_event("Stopping bridge, OMNeT++, and Gazebo...")
    managed_processes = [bridge, rviz, camera_view, omnet, gz]
    for proc in managed_processes:
        if proc is None:
            continue
        with suppress(Exception):
            proc.send_signal(signal.SIGINT)
    time.sleep(2)
    for proc in managed_processes:
        if proc is None:
            continue
        with suppress(Exception):
            if proc.poll() is None:
                proc.terminate()
    log_event("All processes stopped cleanly.")
    close_terminal_tee()
