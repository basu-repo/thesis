"""Launch Gazebo with the best saved AI Husky checkpoint controlling the ego robot."""

from __future__ import annotations

import argparse
import datetime
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parent
RULE_BASED_ROOT = THESIS_ROOT / "02_rule_based"
RULE_BASED_SCRIPTS_ROOT = RULE_BASED_ROOT / "scripts"
CONTROLLERS_ROOT = SCRIPT_DIR / "controllers"
COMMUNICATION_ROS_BRIDGES_ROOT = THESIS_ROOT / "06_Communication" / "basic" / "ros_bridges"

for extra_path in (
    RULE_BASED_ROOT,
    RULE_BASED_SCRIPTS_ROOT,
    CONTROLLERS_ROOT,
    COMMUNICATION_ROS_BRIDGES_ROOT,
):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

import rclpy
from rclpy.executors import MultiThreadedExecutor

from controllers.husky_model_driver import ModelHuskyDriver
from controllers.obstacle_detection import ObstacleDetectionNode
from controllers.episode_metadata import EpisodeMetadataPublisher
from controllers.uav_follower import UavFollower
from husky_ai_model_driver import HuskyAIModelDriver
from project_paths import MODELS_DIR, OMNET_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH
from select_best_live_model import select_best_live_model
from uav_hazard_estimator import UavHazardEstimator
from multi_agent_hazard_fusion import MultiAgentHazardFusion

WORLD = str(WORLD_SDF_PATH)
WORLD_NAME = "baylands"
MODEL_PATH = str(MODELS_DIR)
OMNET_BIN = OMNET_DIR / "onmetpp"

LOG_DIR = THESIS_ROOT / "03_dataset" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SPAWN_X, SPAWN_Y, SPAWN_Z = -273.6910, -103.7170, -0.4349
HUSKY2_OFFSET_X = 4.0
HUSKY2_OFFSET_Y = 3.0
HUSKY2_X, HUSKY2_Y, HUSKY2_Z = SPAWN_X + HUSKY2_OFFSET_X, SPAWN_Y + HUSKY2_OFFSET_Y, SPAWN_Z
# UAV1 starts on one side of the ego UGV.
UAV_X, UAV_Y, UAV_Z = SPAWN_X + 6.0, SPAWN_Y - 4.0, SPAWN_Z + 4.0

# UAV2 starts on the opposite side of the ego UGV.
# The two UAVs later keep left/right formation offsets around the UGV.
UAV2_X, UAV2_Y, UAV2_Z = SPAWN_X + 6.0, SPAWN_Y + 4.0, SPAWN_Z + 4.0

HUSKY1_SPAWN_YAW = math.pi
HUSKY2_SPAWN_YAW = HUSKY1_SPAWN_YAW
UAV_SPAWN_YAW = 0.0
UAV2_SPAWN_YAW = 0.0
HUSKY1_SPAWN_QZ = math.sin(HUSKY1_SPAWN_YAW / 2.0)
HUSKY1_SPAWN_QW = math.cos(HUSKY1_SPAWN_YAW / 2.0)
HUSKY2_SPAWN_QZ = math.sin(HUSKY2_SPAWN_YAW / 2.0)
HUSKY2_SPAWN_QW = math.cos(HUSKY2_SPAWN_YAW / 2.0)
UAV_SPAWN_QZ = math.sin(UAV_SPAWN_YAW / 2.0)
UAV_SPAWN_QW = math.cos(UAV_SPAWN_YAW / 2.0)
UAV2_SPAWN_QZ = math.sin(UAV2_SPAWN_YAW / 2.0)
UAV2_SPAWN_QW = math.cos(UAV2_SPAWN_YAW / 2.0)

RAW_WORLD_SHARED_GOAL = (-324.5690, -31.8468, -1.5615)
GOAL_WORLD_PULLBACK = 13.0
GROUND_MARKER_Z = 0.2025
GOAL_STOP_OFFSET = -0.5

BOOTSTRAP_SECONDS = 3.0
BOOTSTRAP_LINEAR_SPEED = 0.8
CONTROL_PERIOD = 0.1
CMD_LINEAR_GAIN = 1.45
CMD_ANGULAR_GAIN = 1.15
MIN_LINEAR_SPEED = 1.5
MAX_LINEAR_SPEED = 2.0
MAX_ANGULAR_SPEED = 0.85
HEADING_DEADBAND = 0.12
GOAL_TOLERANCE = 1.5
STUCK_TIMEOUT_SECONDS = 5.0
STUCK_PROGRESS_DISTANCE = 0.08
STUCK_REVERSE_SPEED = -0.8
STUCK_REVERSE_SECONDS = 2.0
STUCK_BOOTSTRAP_SECONDS = 2.0
OBSTACLE_FRONT_HALF_ANGLE_DEG = 45.0
OBSTACLE_SIDE_ANGLE_DEG = 65.0
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

TEE_PROCESS = None
ORIGINAL_STDOUT_FD = None
ORIGINAL_STDERR_FD = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live AI Husky evaluation.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a saved AI checkpoint (.pt). If omitted, the best live-compatible model is selected automatically.",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run only husky_local and use synthetic placeholders for husky_2 and uav1.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Gazebo in server-only headless mode.")
    parser.add_argument("--no-rviz", action="store_true", help="Skip launching RViz.")
    parser.add_argument("--no-camera", action="store_true", help="Skip launching the image viewer.")
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
    print(f"[{timestamp}] {message}")
    with suppress(Exception):
        sys.stdout.flush()


def run_bg(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", cmd])


def offset_goal_along_path(
    world_goal: tuple[float, float, float],
    start_xyz: tuple[float, float, float],
    offset_distance: float,
) -> tuple[float, float, float]:
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


WORLD_SHARED_GOAL = offset_goal_along_path(
    RAW_WORLD_SHARED_GOAL,
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    -GOAL_WORLD_PULLBACK,
)
WORLD_HUSKY1_GOAL = WORLD_SHARED_GOAL
WORLD_HUSKY2_GOAL = WORLD_SHARED_GOAL
WORLD_UAV_GOAL = WORLD_SHARED_GOAL


def load_husky_sdf_with_topic(topic_name: str) -> str:
    husky_sdf = (MODELS_DIR / "husky" / "model.sdf").read_text()
    husky_sdf = husky_sdf.replace("<topic>/cmd_vel</topic>", f"<topic>{topic_name}</topic>", 1)
    return husky_sdf


def add_pose_publisher(sdf_text: str) -> str:
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


def main() -> int:
    args = parse_args()
    debug_isolate_husky_local = args.isolate
    if debug_isolate_husky_local:
        args.no_rviz = True
        args.no_camera = True
    selected_info = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        selected_info = select_best_live_model()
        checkpoint_path = Path(selected_info["checkpoint_path"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    run_start_dt = datetime.datetime.now()
    run_log_path = LOG_DIR / f"ai_live_run_{run_start_dt.strftime('%Y%m%d_%H%M%S')}.log"

    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

    setup_terminal_tee(run_log_path)
    subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
    subprocess.run(["bash", "-c", "pkill -f ign || true"])
    subprocess.run(["bash", "-c", f"pkill -f {OMNET_BIN} || true"])

    log_event(f"START timestamp: {run_start_dt.isoformat(timespec='seconds')}")
    log_event(f"Run log file: {run_log_path}")
    log_event(f"Checkpoint: {checkpoint_path}")
    if selected_info is not None:
        if selected_info.get("selection_task") == "trajectory":
            log_event(
                "Auto-selected live trajectory model: "
                f"{selected_info['selected_model']} "
                f"(ADE={selected_info['selected_ADE']:.4f}, "
                f"FDE={selected_info['selected_FDE']:.4f}, "
                f"RMSE={selected_info['selected_RMSE']:.4f})"
            )
        else:
            log_event(
                "Auto-selected live model: "
                f"{selected_info['selected_model']} "
                f"(macro_f1={selected_info['selected_macro_f1']:.4f}, "
                f"accuracy={selected_info['selected_accuracy']:.4f})"
            )
        if selected_info["used_fallback"]:
            log_event(
                "Overall best summary model is "
                f"{selected_info['overall_best_model']}, "
                "but the live runner is currently wired for the supported live checkpoint path."
            )
    log_event("Starting Gazebo...")
    gazebo_cmd = f"ign gazebo {WORLD}"
    if args.headless:
        gazebo_cmd = f"ign gazebo -s -r --headless-rendering {WORLD}"
    gz = run_bg(gazebo_cmd)
    time.sleep(5)

    log_event("Waiting for Baylands world to fully load...")
    time.sleep(40)

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

    if not debug_isolate_husky_local:
        log_event("Spawning Husky 2...")
        husky2_sdf_path = write_husky_variant(
            MODELS_DIR / "husky" / "model_blue_tag.sdf",
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
    spawn_goal_marker(WORLD_NAME, "goal_husky_local", (WORLD_HUSKY1_GOAL[0], WORLD_HUSKY1_GOAL[1], GROUND_MARKER_Z), (0.95, 0.12, 0.12, 1.0))
    if not debug_isolate_husky_local:
        spawn_goal_marker(WORLD_NAME, "goal_husky_2", (WORLD_HUSKY2_GOAL[0], WORLD_HUSKY2_GOAL[1], GROUND_MARKER_Z), (0.12, 0.36, 0.95, 1.0))
        spawn_goal_marker(WORLD_NAME, "goal_uav1", (WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1], GROUND_MARKER_Z), (0.95, 0.85, 0.12, 1.0))
        spawn_goal_marker(WORLD_NAME, "goal_uav2", (WORLD_UAV_GOAL[0], WORLD_UAV_GOAL[1], GROUND_MARKER_Z), (0.95, 0.55, 0.12, 1.0))
    time.sleep(1)

    log_event("Starting bridge...")
    bridge_topics = [
        "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
    ]
    bridge_topics.extend(husky_sensor_bridge_topics(WORLD_NAME, "husky_local"))
    if not debug_isolate_husky_local:
        bridge_topics.extend(husky_sensor_bridge_topics(WORLD_NAME, "husky_2"))
        bridge_topics.extend(
            [
                "/cmd_vel_husky2@geometry_msgs/msg/Twist@ignition.msgs.Twist",
                "/model/husky_2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
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
    if not args.no_rviz:
        log_event(f"Starting RViz with config: {RVIZ_CONFIG_PATH}")
        rviz = run_bg(f"source /opt/ros/humble/setup.bash && rviz2 -d {RVIZ_CONFIG_PATH}")
        time.sleep(2)

    camera_view = None
    if not args.no_camera:
        log_event("Starting camera viewer...")
        camera_view = run_bg("source /opt/ros/humble/setup.bash && ros2 run rqt_image_view rqt_image_view")
        time.sleep(2)

    log_event("OMNeT++ relay disabled for this run.")
    log_event("Bag recording disabled for live model testing.")
    if debug_isolate_husky_local:
        log_event("DEBUG isolate mode: only husky_local is spawned; husky_2, uav1, uav2, RViz, and camera viewer are disabled.")
    log_event("==============================")
    log_event("LIVE AI MODEL MODE ENABLED")
    log_event("==============================")
    log_event("Press Play in Gazebo")
    log_event("Press Ctrl+C here when done.")

    rclpy.init()
    husky1_goal = offset_goal_along_path(WORLD_HUSKY1_GOAL, (SPAWN_X, SPAWN_Y, SPAWN_Z), GOAL_STOP_OFFSET)
    husky2_goal = offset_goal_along_path(WORLD_HUSKY2_GOAL, (HUSKY2_X, HUSKY2_Y, HUSKY2_Z), GOAL_STOP_OFFSET)
    uav_goal = offset_goal_along_path(WORLD_UAV_GOAL, (UAV_X, UAV_Y, UAV_Z), GOAL_STOP_OFFSET)
    uav2_goal = offset_goal_along_path(WORLD_UAV_GOAL, (UAV2_X, UAV2_Y, UAV2_Z), GOAL_STOP_OFFSET)
    
    if debug_isolate_husky_local:
        log_event(
            "Controller goals (world frame, stop offset applied): "
            f"husky_local=({husky1_goal[0]:.3f}, {husky1_goal[1]:.3f})"
        )
    else:
        log_event(
            "Controller goals (world frame, stop offset applied): "
            f"husky_local=({husky1_goal[0]:.3f}, {husky1_goal[1]:.3f}), "
            f"husky_2=({husky2_goal[0]:.3f}, {husky2_goal[1]:.3f}), "
            f"uav1=({uav_goal[0]:.3f}, {uav_goal[1]:.3f}), "
            f"uav2=({uav2_goal[0]:.3f}, {uav2_goal[1]:.3f})"
        )

    start_goals = {
        "husky_local": {"start": (SPAWN_X, SPAWN_Y, SPAWN_Z), "goal": WORLD_HUSKY1_GOAL},
    }
    
    if not debug_isolate_husky_local:
        start_goals["husky_2"] = {"start": (HUSKY2_X, HUSKY2_Y, HUSKY2_Z), "goal": WORLD_HUSKY2_GOAL}
        start_goals["uav1"] = {"start": (UAV_X, UAV_Y, UAV_Z), "goal": WORLD_UAV_GOAL}
        start_goals["uav2"] = {"start": (UAV2_X, UAV2_Y, UAV2_Z), "goal": WORLD_UAV_GOAL}
    episode_metadata = EpisodeMetadataPublisher(world_name=WORLD_NAME, start_goals=start_goals)

    husky1_obstacle_action_topic = "/husky_local/obstacle_action"
    husky1_obstacle_clearance_topic = "/husky_local/obstacle_clearance"
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
    ai_driver = HuskyAIModelDriver(
        node_name="ai_model_husky_driver_1",
        ego_node="husky_local",
        cmd_topic="/cmd_vel",
        odom_topics={
            "husky_local": "/model/husky_local/odometry",
            "husky_2": "/model/husky_2/odometry",
            "uav1": "/model/uav1/odometry",
        },
        command_topics={
            "husky_local": "/cmd_vel",
            "husky_2": "/cmd_vel_husky2",
            "uav1": "/uav1/command/twist",
        },
        checkpoint_path=checkpoint_path,
        summary_path=None,
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        obstacle_action_topic=husky1_obstacle_action_topic,
        obstacle_clearance_topic=husky1_obstacle_clearance_topic,
        scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
        hazard_topic="/fused_hazard_hint",
        spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        goals={
            "husky_local": husky1_goal,
            "husky_2": husky2_goal,
            "uav1": uav_goal,
        },
        active_nodes=["husky_local"] if debug_isolate_husky_local else ["husky_local", "husky_2", "uav1"],
        bootstrap_seconds=BOOTSTRAP_SECONDS,
        bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
        control_period=CONTROL_PERIOD,
        cmd_linear_gain=CMD_LINEAR_GAIN,
        cmd_angular_gain=CMD_ANGULAR_GAIN,
        min_linear_speed=MIN_LINEAR_SPEED,
        max_linear_speed=MAX_LINEAR_SPEED,
        max_angular_speed=MAX_ANGULAR_SPEED,
        heading_deadband=HEADING_DEADBAND,
        goal_align_heading_threshold=0.5,
        goal_align_linear_speed=0.28,
        waypoint_reached_dist=0.2,
        goal_tolerance=GOAL_TOLERANCE,
        goal_blend=0.35,
        obstacle_stop_distance=OBSTACLE_STOP_DISTANCE,
        obstacle_clear_distance=OBSTACLE_CAUTION_DISTANCE,
        obstacle_turn_speed=1.0,
        obstacle_turn_speed_close=1.4,
        stuck_timeout_seconds=STUCK_TIMEOUT_SECONDS,
        stuck_progress_distance=STUCK_PROGRESS_DISTANCE,
        stuck_min_command_speed=0.2,
        stuck_reverse_speed=STUCK_REVERSE_SPEED,
        stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
        stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
        stuck_cooldown_seconds=4.0,
        strict_reverse_distance=0.8,
        strict_reverse_cycles=4,
        post_avoid_forward_speed=0.18,
        post_avoid_forward_seconds=0.8,
    )
    executor = MultiThreadedExecutor()
    managed_nodes = [
        episode_metadata,
        obstacle_detector,
        ai_driver,
    ]
    if not debug_isolate_husky_local:
        husky2_obstacle_action_topic = "/husky_2/obstacle_action"
        husky2_obstacle_clearance_topic = "/husky_2/obstacle_clearance"
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
            state_topic="/husky_2/controller_state",
            goal_xyz=husky2_goal,
            world_goal_xyz=husky2_goal,
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
        # follower = UavFollower(
        #     husky_odom_topic="/model/husky_local/odometry",
        #     uav_odom_topic="/model/uav1/odometry",
        #     world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        #     husky_model_name="husky_local",
        #     uav_model_name="uav1",
        #     uav_name="uav1",
        #     follow_distance=UAV_FOLLOW_DISTANCE,
        #     follow_height=UAV_FOLLOW_HEIGHT,
        #     ready_topic="/uav1/ready",
        #     update_period=UAV_UPDATE_PERIOD,
        #     max_xy_speed=UAV_MAX_XY_SPEED,
        #     max_z_speed=UAV_MAX_Z_SPEED,
        #     max_yaw_rate=UAV_MAX_YAW_RATE,
        #     xy_gain=UAV_XY_GAIN,
        #     z_gain=UAV_Z_GAIN,
        #     yaw_gain=UAV_YAW_GAIN,
        #     heading_align_gain=UAV_HEADING_ALIGN_GAIN,
        #     min_forward_speed=UAV_MIN_FORWARD_SPEED,
        #     target_smoothing=UAV_TARGET_SMOOTHING,
        #     xy_deadband=UAV_XY_DEADBAND,
        #     z_deadband=UAV_Z_DEADBAND,
        #     yaw_deadband=UAV_YAW_DEADBAND,
        #     min_track_speed=UAV_MIN_TRACK_SPEED,
        #     husky_spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        #     husky_spawn_yaw=HUSKY1_SPAWN_YAW,
        #     uav_spawn_xyz=(UAV_X, UAV_Y, UAV_Z),
        #     uav_spawn_yaw=UAV_SPAWN_YAW,
        # )

        follower_left = UavFollower(
            node_name="uav1_left_follower",
            husky_odom_topic="/model/husky_local/odometry",
            uav_odom_topic="/model/uav1/odometry",
            world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
            husky_model_name="husky_local",
            uav_model_name="uav1",
            uav_name="uav1",
            follow_distance=UAV_FOLLOW_DISTANCE,
            follow_height=UAV_FOLLOW_HEIGHT,
            ready_topic="/uav1/ready",
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

            # UAV1 stays ahead-left of the ego UGV.
            formation_forward_offset=6.0,
            formation_lateral_offset=4.0,
        )

        follower_right = UavFollower(
            node_name="uav2_right_follower",
            husky_odom_topic="/model/husky_local/odometry",
            uav_odom_topic="/model/uav2/odometry",
            world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
            husky_model_name="husky_local",
            uav_model_name="uav2",
            uav_name="uav2",
            follow_distance=UAV_FOLLOW_DISTANCE,
            follow_height=UAV_FOLLOW_HEIGHT,
            ready_topic="/uav2/ready",
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
            uav_spawn_xyz=(UAV2_X, UAV2_Y, UAV2_Z),
            uav_spawn_yaw=UAV2_SPAWN_YAW,

            # UAV2 stays ahead-right of the ego UGV.
            formation_forward_offset=6.0,
            formation_lateral_offset=-4.0,
        )

        

        uav1_hazard = UavHazardEstimator(
            node_name="uav1_hazard_estimator",
            husky_odom_topic="/model/husky_local/odometry",
            uav_odom_topic="/model/uav1/odometry",
            uav_pointcloud_topic=f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points",
            output_topic="/uav1/hazard_hint_raw",
            source_name="uav1",
            # More permissive UAV hazard detection for initial testing.
            lookahead_min_x=0.5,
            lookahead_max_x=15.0,
            lane_half_width=6.0,
            obstacle_min_z=-2.0,
            obstacle_max_z=5.0,
            min_points_blocked=3,
        )

        uav2_hazard = UavHazardEstimator(
            node_name="uav2_hazard_estimator",
            husky_odom_topic="/model/husky_local/odometry",
            uav_odom_topic="/model/uav2/odometry",
            uav_pointcloud_topic=f"/world/{WORLD_NAME}/model/uav2/link/base_link/sensor/front_laser/scan/points",
            output_topic="/uav2/hazard_hint_raw",
            source_name="uav2",
            # More permissive UAV hazard detection for initial testing.
            lookahead_min_x=0.5,
            lookahead_max_x=15.0,
            lane_half_width=6.0,
            obstacle_min_z=-2.0,
            obstacle_max_z=5.0,
            min_points_blocked=3,
        )

        hazard_fusion = MultiAgentHazardFusion(
            node_name="multi_agent_hazard_fusion",
            ugv_action_topic=husky1_obstacle_action_topic,
            ugv_clearance_topic=husky1_obstacle_clearance_topic,
            uav1_topic="/uav1/hazard_hint_raw",
            uav2_topic="/uav2/hazard_hint_raw",
            output_topic="/fused_hazard_hint",
            uav_timeout=1.0,
        )
        managed_nodes.extend([
            obstacle_detector2,
            driver2,
            follower_left,
            follower_right,
            uav1_hazard,
            uav2_hazard,
            hazard_fusion,
        ])
    for node in managed_nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        log_event("Stopping live AI model run...")
    finally:
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

        log_event(f"STOP timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}")
        log_event("Stopping bridge and Gazebo...")
        managed_processes = [bridge, rviz, camera_view, gz]
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
