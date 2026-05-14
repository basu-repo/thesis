#!/usr/bin/env python3
"""Live simulation evaluator for 08 trajectory-model weights using a 05-style flow."""

from __future__ import annotations

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parent
CONTROLLERS_ROOT = SCRIPT_DIR / "controllers"
BRIDGES_ROOT = SCRIPT_DIR / "bridges"
PIPELINE_ROOT = THESIS_ROOT / "08_model_training_pipeline"
TRAINING_ROOT = PIPELINE_ROOT / "training"
SIM_SCRIPTS_ROOT = THESIS_ROOT / "07_multi_uav_hazard_map" / "scripts"
SIM_CONTROLLERS_ROOT = THESIS_ROOT / "07_multi_uav_hazard_map" / "controllers"

for extra_path in (
    SIM_CONTROLLERS_ROOT,
    SIM_SCRIPTS_ROOT,
    TRAINING_ROOT,
    PIPELINE_ROOT,
    BRIDGES_ROOT,
    CONTROLLERS_ROOT,
    SCRIPT_DIR,
):
    extra_str = str(extra_path)
    if extra_str in sys.path:
        sys.path.remove(extra_str)
    sys.path.insert(0, extra_str)

import rclpy
from rclpy.executors import MultiThreadedExecutor

from episode_metadata import EpisodeMetadataPublisher
from evaluation_exports import export_live_run_bundle
from gazebo_pose_tcp_bridge import GazeboPoseTcpBridge
from husky_trajectory_model_driver import HuskyTrajectoryModelDriver
from obstacle_detection import ObstacleDetectionNode
from omnet_metrics_bridge import OmnetMetricsBridge
from project_paths import GUI_CONFIG_PATH, MODELS_DIR, OMNET_EXTERNAL_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH
from select_best_live_model import select_best_live_model
from uav_obstacle_detection import UavObstacleDetectionNode
from uav_scout_driver import UavScoutDriver


WORLD = str(WORLD_SDF_PATH)
WORLD_NAME = "baylands"
MODEL_PATH = str(MODELS_DIR)

LOG_DIR = THESIS_ROOT / "03_dataset" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HUSKY2_X, HUSKY2_Y, HUSKY2_Z = 108.00, -275.00, 0.68
UAV1_X, UAV1_Y, UAV1_Z = 110.00, -275.00, 11.00
UAV2_X, UAV2_Y, UAV2_Z = 106.00, -275.00, 11.00
WORLD_SHARED_GOAL = (-35.0, -290.30, 0.35)
GROUND_MARKER_Z = 0.2025

CONTROL_PERIOD = 0.1
OBSTACLE_FRONT_HALF_ANGLE_DEG = 45.0
OBSTACLE_SIDE_ANGLE_DEG = 65.0
OBSTACLE_STOP_DISTANCE = 1.8
OBSTACLE_CAUTION_DISTANCE = 3.2
UAV_SCOUT_ALTITUDE_Z = 35.97
UAV_SCOUT_SLOT_FORWARD = 2.5
UAV_SCOUT_SLOT_LATERAL = 2.0
UAV_UPDATE_PERIOD = 0.1
UAV_MAX_XY_SPEED = 6.0
UAV_MAX_Z_SPEED = 2.0
UAV_MAX_YAW_RATE = 0.9
UAV_XY_GAIN = 2.0
UAV_Z_GAIN = 0.45
UAV_YAW_GAIN = 0.8
UAV_HEADING_ALIGN_GAIN = 0.9
UAV_MIN_FORWARD_SPEED = 0.25
UAV_TARGET_SMOOTHING = 1.0
UAV_XY_DEADBAND = 0.02
UAV_Z_DEADBAND = 0.15
UAV_YAW_DEADBAND = 0.18
UAV_MIN_TRACK_SPEED = 0.0
UAV_SLOT_READY_RADIUS = 2.5
UAV_SLOT_CATCHUP_XY_SCALE = 1.15
UAV_SLOT_REALIGN_RADIUS = 25.0
UAV_SLOT_REALIGN_MAX_XY_SPEED = 10.0
UAV_MAX_HUSKY_XY_RADIUS = 4.5
UAV_LEASH_REENTRY_RADIUS = 3.0

TEE_PROCESS = None
ORIGINAL_STDOUT_FD = None
ORIGINAL_STDERR_FD = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="best", help="Model slug to test, or 'best'.")
    parser.add_argument("--checkpoint", default=None, help="Explicit path to .pt weight file.")
    parser.add_argument("--headless", action="store_true", help="Run Gazebo server-only.")
    parser.add_argument("--no-rviz", action="store_true", help="Skip RViz.")
    parser.add_argument("--no-camera", action="store_true", help="Skip the image viewer.")
    parser.add_argument("--target-index", type=int, default=4, help="Future waypoint index to follow.")
    parser.add_argument("--enable-omnet", action="store_true", help="Enable external OMNeT communication co-simulation.")
    parser.add_argument(
        "--omnet-config",
        default="Communication-GazeboBridge-WiFi",
        help="External OMNeT configuration name, e.g. Communication-GazeboBridge-WiFi/5G/LoRa.",
    )
    return parser.parse_args()


def run_bg(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", cmd])


def build_omnet_command(config_name: str) -> str:
    ned_path = "src:../inet4.5/src"
    if "lora" in config_name.lower():
        ned_path += ":../flora/src"
    return (
        f"cd {OMNET_EXTERNAL_DIR} && "
        f"./UAV_UGV -u Cmdenv -f omnetpp.ini -c {config_name} -n {ned_path}"
    )


def log_event(message: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with suppress(Exception):
        sys.stdout.flush()


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


def resolve_checkpoint(model_arg: str, checkpoint_arg: str | None) -> tuple[str, Path, dict | None]:
    if checkpoint_arg:
        checkpoint = Path(checkpoint_arg).expanduser().resolve()
        return checkpoint.parent.name, checkpoint, None
    selected_info = select_best_live_model()
    if model_arg == "best":
        checkpoint = Path(selected_info["checkpoint_path"]).resolve()
        return str(selected_info["selected_model"]), checkpoint, selected_info
    checkpoint = PIPELINE_ROOT / "model_weights" / model_arg / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return model_arg, checkpoint, selected_info


def load_husky_sdf_with_topic(topic_name: str) -> str:
    husky_sdf = (MODELS_DIR / "husky" / "model.sdf").read_text()
    return husky_sdf.replace("<topic>/cmd_vel</topic>", f"<topic>{topic_name}</topic>", 1)


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


def write_husky_variant(output_path: Path, topic_name: str) -> Path:
    sdf_text = load_husky_sdf_with_topic(topic_name)
    sdf_text = add_husky_marker(sdf_text, "flag_marker_blue", (0.12, 0.36, 0.95, 1.0))
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


def main() -> int:
    args = parse_args()
    model_slug, checkpoint_path, selected_info = resolve_checkpoint(args.model, args.checkpoint)
    run_start_dt = datetime.datetime.now()
    run_log_path = LOG_DIR / f"trajectory_model_eval_09_{run_start_dt.strftime('%Y%m%d_%H%M%S')}.log"

    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

    setup_terminal_tee(run_log_path)
    log_event(f"START timestamp: {run_start_dt.isoformat(timespec='seconds')}")
    log_event(f"Run log file: {run_log_path}")
    log_event(f"Model slug: {model_slug}")
    log_event(f"Checkpoint: {checkpoint_path}")
    if selected_info is not None:
        log_event(f"Selected info: {selected_info}")
    log_event(f"OMNeT enabled: {args.enable_omnet}")
    if args.enable_omnet:
        log_event(f"OMNeT config: {args.omnet_config}")

    gazebo_cmd = f"ign gazebo --gui-config {GUI_CONFIG_PATH} {WORLD}"
    if args.headless:
        gazebo_cmd = f"ign gazebo -s -r --headless-rendering {WORLD}"
    log_event("Starting Gazebo...")
    gz = run_bg(gazebo_cmd)
    time.sleep(5)

    log_event("Waiting for Baylands world to load...")
    time.sleep(20)

    husky_sdf_path = write_husky_variant(MODELS_DIR / "husky" / "model_09_eval.sdf", "/cmd_vel_husky2")
    spawn_husky = (
        f"ign service -s /world/{WORLD_NAME}/create "
        f"--reqtype ignition.msgs.EntityFactory "
        f"--reptype ignition.msgs.Boolean "
        f"--timeout 5000 "
        f'--req \'sdf_filename: "{husky_sdf_path}", name: "husky_2", '
        f'pose: {{position: {{x: {HUSKY2_X}, y: {HUSKY2_Y}, z: {HUSKY2_Z}}}}}\''
    )
    log_event("Spawning Husky 2...")
    subprocess.run(["bash", "-c", spawn_husky])
    time.sleep(4)

    for name, x, y, z in (("uav1", UAV1_X, UAV1_Y, UAV1_Z), ("uav2", UAV2_X, UAV2_Y, UAV2_Z)):
        log_event(f"Spawning {name}...")
        spawn_uav = (
            f"ign service -s /world/{WORLD_NAME}/create "
            f"--reqtype ignition.msgs.EntityFactory "
            f"--reptype ignition.msgs.Boolean "
            f"--timeout 5000 "
            f'--req \'sdf_filename: "model://m100/model.sdf", name: "{name}", '
            f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}\''
        )
        subprocess.run(["bash", "-c", spawn_uav])
        time.sleep(2)

    log_event("Spawning goal marker...")
    spawn_goal_marker(WORLD_NAME, "goal_husky_2", (WORLD_SHARED_GOAL[0], WORLD_SHARED_GOAL[1], GROUND_MARKER_Z), (0.12, 0.36, 0.95, 1.0))

    log_event("Starting bridge...")
    bridge_topics = [
        f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        "/cmd_vel_husky2@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        "/model/husky_2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
        f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
        "/uav1/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        "/uav1/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
        "/model/uav1/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        "/model/uav1/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
        "/model/uav1/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        "/uav2/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        "/uav2/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
        "/model/uav2/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        "/model/uav2/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
        "/model/uav2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
    ]
    bridge = run_bg(
        "source /opt/ros/humble/setup.bash && "
        "ros2 run ros_gz_bridge parameter_bridge " + " ".join(bridge_topics)
    )
    time.sleep(4)

    omnet = None
    if args.enable_omnet:
        log_event("Starting external OMNeT communication co-simulation...")
        omnet = run_bg(build_omnet_command(args.omnet_config))
        time.sleep(3)

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

    log_event("Bag recording disabled for this run.")
    log_event("==============================")
    log_event("09 LIVE TRAJECTORY MODEL MODE ENABLED")
    log_event("==============================")
    log_event("Press Play in Gazebo")
    log_event("Press Ctrl+C here when done.")
    log_event(
        f"Scout altitude target: z={UAV_SCOUT_ALTITUDE_Z:.2f}, "
        f"slot_forward={UAV_SCOUT_SLOT_FORWARD:.1f} m, slot_lateral={UAV_SCOUT_SLOT_LATERAL:.1f} m"
    )

    rclpy.init()
    start_goals = {
        "husky_2": {"start": (HUSKY2_X, HUSKY2_Y, HUSKY2_Z), "goal": WORLD_SHARED_GOAL},
        "uav1": {"start": (UAV1_X, UAV1_Y, UAV1_Z), "goal": WORLD_SHARED_GOAL},
        "uav2": {"start": (UAV2_X, UAV2_Y, UAV2_Z), "goal": WORLD_SHARED_GOAL},
    }
    episode_metadata = EpisodeMetadataPublisher(world_name=WORLD_NAME, start_goals=start_goals)
    gazebo_pose_bridge = None
    omnet_metrics_bridge = None
    if args.enable_omnet:
        gazebo_pose_bridge = GazeboPoseTcpBridge(
            node_name="gazebo_pose_tcp_bridge",
            world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
            tracked_models=["husky_2", "uav1"],
            host="127.0.0.1",
            port=5555,
        )
        omnet_metrics_bridge = OmnetMetricsBridge(
            node_name="omnet_metrics_bridge",
            host="127.0.0.1",
            port=5556,
            topic_prefix="/omnet",
        )

    husky_obstacle_action_topic = "/husky_2/obstacle_action"
    husky_obstacle_clearance_topic = "/husky_2/obstacle_clearance"
    husky_controller_state_topic = "/husky_2/controller_state"
    obstacle_detector = ObstacleDetectionNode(
        node_name="husky_2_obstacle_detector",
        scan_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
        action_topic=husky_obstacle_action_topic,
        clearance_topic=husky_obstacle_clearance_topic,
        pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
        front_half_angle_deg=OBSTACLE_FRONT_HALF_ANGLE_DEG,
        side_angle_deg=OBSTACLE_SIDE_ANGLE_DEG,
        stop_distance=OBSTACLE_STOP_DISTANCE,
        caution_distance=OBSTACLE_CAUTION_DISTANCE,
    )
    model_driver = HuskyTrajectoryModelDriver(
        node_name="trajectory_model_husky_driver_2",
        model_slug=model_slug,
        checkpoint_path=checkpoint_path,
        cmd_topic="/cmd_vel_husky2",
        husky_odom_topic="/model/husky_2/odometry",
        uav1_odom_topic="/model/uav1/odometry",
        uav2_odom_topic="/model/uav2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        obstacle_clearance_topic=husky_obstacle_clearance_topic,
        state_topic=husky_controller_state_topic,
        omnet_rssi_topic="/omnet/rssi_dbm" if args.enable_omnet else None,
        omnet_snir_topic="/omnet/snir_db" if args.enable_omnet else None,
        omnet_per_topic="/omnet/packet_error_rate" if args.enable_omnet else None,
        omnet_link_distance_topic="/omnet/link_distance" if args.enable_omnet else None,
        goal_xyz=WORLD_SHARED_GOAL,
        control_period=CONTROL_PERIOD,
        target_index=args.target_index,
        goal_tolerance=1.5,
        max_linear_speed=0.65,
        max_angular_speed=0.45,
    )
    uav1_obstacle_action_topic = "/uav1/obstacle_action"
    uav1_obstacle_clearance_topic = "/uav1/obstacle_clearance"
    uav1_controller_state_topic = "/uav1/controller_state"
    uav1_ready_topic = "/uav1/scout_ready"
    uav1_report_topic = "/uav1/scout_report"
    obstacle_detector_uav1 = UavObstacleDetectionNode(
        node_name="uav1_obstacle_detector",
        pointcloud_topic=f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points",
        action_topic=uav1_obstacle_action_topic,
        clearance_topic=uav1_obstacle_clearance_topic,
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
    shared_scout_kwargs = dict(
        husky_name="husky_2",
        husky_state_topic=husky_controller_state_topic,
        scout_altitude_z=UAV_SCOUT_ALTITUDE_Z,
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
        slot_forward_m=UAV_SCOUT_SLOT_FORWARD,
        slot_ready_radius=UAV_SLOT_READY_RADIUS,
        slot_catchup_xy_scale=UAV_SLOT_CATCHUP_XY_SCALE,
        slot_realign_radius=UAV_SLOT_REALIGN_RADIUS,
        slot_realign_max_xy_speed=UAV_SLOT_REALIGN_MAX_XY_SPEED,
        max_husky_xy_radius_m=UAV_MAX_HUSKY_XY_RADIUS,
        leash_reentry_radius_m=UAV_LEASH_REENTRY_RADIUS,
        landing_goal_xy=(WORLD_SHARED_GOAL[0], WORLD_SHARED_GOAL[1]),
    )

    scout_driver_uav1 = UavScoutDriver(
        node_name="uav1_scout_driver",
        uav_name="uav1",
        odom_topic="/model/uav1/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        obstacle_action_topic=uav1_obstacle_action_topic,
        obstacle_clearance_topic=uav1_obstacle_clearance_topic,
        state_topic=uav1_controller_state_topic,
        ready_topic=uav1_ready_topic,
        report_topic=uav1_report_topic,
        slot_lateral_m=-UAV_SCOUT_SLOT_LATERAL,
        **shared_scout_kwargs,
    )
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
    scout_driver_uav2 = UavScoutDriver(
        node_name="uav2_scout_driver",
        uav_name="uav2",
        odom_topic="/model/uav2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        obstacle_action_topic=uav2_obstacle_action_topic,
        obstacle_clearance_topic=uav2_obstacle_clearance_topic,
        state_topic=uav2_controller_state_topic,
        ready_topic=uav2_ready_topic,
        report_topic=uav2_report_topic,
        slot_lateral_m=UAV_SCOUT_SLOT_LATERAL,
        **shared_scout_kwargs,
    )

    executor = MultiThreadedExecutor()
    managed_nodes = [
        episode_metadata,
        obstacle_detector,
        model_driver,
        obstacle_detector_uav1,
        scout_driver_uav1,
        obstacle_detector_uav2,
        scout_driver_uav2,
    ]
    if gazebo_pose_bridge is not None:
        managed_nodes.append(gazebo_pose_bridge)
    if omnet_metrics_bridge is not None:
        managed_nodes.append(omnet_metrics_bridge)
    for node in managed_nodes:
        executor.add_node(node)

    export_result = None
    try:
        executor.spin()
    except KeyboardInterrupt:
        log_event("Stopping 09 live trajectory model run...")
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

        run_end_dt = datetime.datetime.now()
        log_event(f"STOP timestamp: {run_end_dt.isoformat(timespec='seconds')}")
        log_event("Stopping bridge, OMNeT, and Gazebo...")
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

        with suppress(Exception):
            export_result = export_live_run_bundle(
                run_log_path=run_log_path,
                checkpoint_path=checkpoint_path,
                selected_info=selected_info,
                run_start_dt=run_start_dt,
                run_end_dt=run_end_dt,
                world_name=WORLD_NAME,
                runner_args=vars(args),
                start_goals=start_goals,
                model_slug=model_slug,
            )

    if export_result is not None:
        print(f"Saved 09 live-eval metrics: {export_result['metrics_path']}")
        print(f"Saved 09 live-eval plot: {export_result['trajectory_plot_path']}")
        print(f"Saved 09 live-eval summary JSON: {export_result['summary_json_path']}")
        print(f"Saved 09 live-eval summary CSV: {export_result['summary_csv_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
