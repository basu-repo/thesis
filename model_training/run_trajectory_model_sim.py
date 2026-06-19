#!/usr/bin/env python3
"""Separate live simulation runner for testing 08 trajectory model weights."""

from __future__ import annotations

import argparse
import datetime
import json
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
PIPELINE_ROOT = SCRIPT_DIR
CONTROLLERS_ROOT = PIPELINE_ROOT / "controllers"
TRAINING_ROOT = PIPELINE_ROOT / "training"
SIM_SCRIPTS_ROOT = THESIS_ROOT / "cooperative_sim" / "scripts"
SIM_CONTROLLERS_ROOT = THESIS_ROOT / "cooperative_sim" / "controllers"

for extra_path in (PIPELINE_ROOT, CONTROLLERS_ROOT, TRAINING_ROOT, SIM_SCRIPTS_ROOT, SIM_CONTROLLERS_ROOT):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

import pandas as pd
import rclpy
from rclpy.executors import MultiThreadedExecutor

from husky_trajectory_model_driver import HuskyTrajectoryModelDriver
from obstacle_detection import ObstacleDetectionNode
from project_paths import MODELS_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH
from uav_follower import UavFollower


WORLD = str(WORLD_SDF_PATH)
WORLD_NAME = "baylands"
MODEL_PATH = str(MODELS_DIR)
GUI_CONFIG = THESIS_ROOT / "simulation" / "gui" / "baylands_gui.config"

LOG_DIR = THESIS_ROOT / "dataset" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HUSKY2_X, HUSKY2_Y, HUSKY2_Z = -191.1260, -159.2520, 1.3347
UAV1_X, UAV1_Y, UAV1_Z = -197.8070, -162.5520, 9.8197
UAV2_X, UAV2_Y, UAV2_Z = -192.9630, -151.6350, 9.9604
WORLD_SHARED_GOAL = (-35.0, -290.30, 0.35)
GROUND_MARKER_Z = 0.2025

CONTROL_PERIOD = 0.1
OBSTACLE_STOP_DISTANCE = 1.8
OBSTACLE_CAUTION_DISTANCE = 3.2
UAV_FOLLOW_HEIGHT = 12.0

TEE_PROCESS = None
ORIGINAL_STDOUT_FD = None
ORIGINAL_STDERR_FD = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="best", help="Model slug to test, or 'best'.")
    parser.add_argument("--checkpoint", default=None, help="Explicit path to .pt weight file.")
    parser.add_argument("--headless", action="store_true", help="Run Gazebo server-only.")
    parser.add_argument("--no-rviz", action="store_true", help="Skip RViz.")
    parser.add_argument("--target-index", type=int, default=4, help="Future waypoint index to follow.")
    return parser.parse_args()


def run_bg(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", cmd])


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


def select_checkpoint(model_arg: str, checkpoint_arg: str | None) -> tuple[str, Path]:
    if checkpoint_arg:
        checkpoint = Path(checkpoint_arg).expanduser().resolve()
        return checkpoint.parent.name, checkpoint

    summary_csv = PIPELINE_ROOT / "comparison_exports" / "trajectory_model_summary_latest.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing comparison summary: {summary_csv}")
    summary = pd.read_csv(summary_csv)
    if "model_slug" not in summary.columns and "model" in summary.columns:
        summary = summary.rename(columns={"model": "model_slug"})
    if model_arg == "best":
        row = summary.sort_values("ADE", ascending=True).iloc[0]
    else:
        matches = summary[summary["model_slug"] == model_arg]
        if matches.empty:
            raise RuntimeError(f"Model slug {model_arg!r} not found in {summary_csv}")
        row = matches.iloc[0]
    model_slug = str(row["model_slug"])
    checkpoint = PIPELINE_ROOT / "model_weights" / model_slug / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return model_slug, checkpoint


def load_husky_sdf_with_topic(topic_name: str) -> str:
    husky_sdf = (MODELS_DIR / "husky" / "model.sdf").read_text()
    return husky_sdf.replace("<topic>/cmd_vel</topic>", f"<topic>{topic_name}</topic>", 1)


def write_husky_variant(output_path: Path, topic_name: str) -> Path:
    output_path.write_text(load_husky_sdf_with_topic(topic_name))
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
        f"gz service -s /world/{world_name}/create "
        f"--reqtype gz.msgs.EntityFactory "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 5000 "
        f'--req \'sdf: "{one_line}"\''
    )
    subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    args = parse_args()
    model_slug, checkpoint_path = select_checkpoint(args.model, args.checkpoint)
    run_start = datetime.datetime.now()
    log_path = LOG_DIR / f"trajectory_model_live_{run_start.strftime('%Y%m%d_%H%M%S')}.log"

    os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    os.environ["GZ_SIM_RESOURCE_PATH"] = MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

    setup_terminal_tee(log_path)
    log_event(f"START timestamp: {run_start.isoformat(timespec='seconds')}")
    log_event(f"Run log file: {log_path}")
    log_event(f"Model slug: {model_slug}")
    log_event(f"Checkpoint: {checkpoint_path}")

    gazebo_cmd = f"gz sim --gui-config {GUI_CONFIG} {WORLD}"
    if args.headless:
        gazebo_cmd = f"gz sim -s -r --headless-rendering {WORLD}"
    gz = run_bg(gazebo_cmd)
    time.sleep(5)

    log_event("Waiting for Baylands world to load...")
    time.sleep(20)

    husky_sdf_path = write_husky_variant(MODELS_DIR / "husky" / "model_blue_tag.sdf", "/cmd_vel_husky2")
    spawn_husky = (
        f"gz service -s /world/{WORLD_NAME}/create "
        f"--reqtype gz.msgs.EntityFactory "
        f"--reptype gz.msgs.Boolean "
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
            f"gz service -s /world/{WORLD_NAME}/create "
            f"--reqtype gz.msgs.EntityFactory "
            f"--reptype gz.msgs.Boolean "
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
        "/cmd_vel_husky2@geometry_msgs/msg/Twist@gz.msgs.Twist",
        "/model/husky_2/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        "/uav1/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist",
        "/uav1/enable@std_msgs/msg/Bool@gz.msgs.Boolean",
        "/model/uav1/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist",
        "/model/uav1/enable@std_msgs/msg/Bool@gz.msgs.Boolean",
        "/model/uav1/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/uav2/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist",
        "/uav2/enable@std_msgs/msg/Bool@gz.msgs.Boolean",
        "/model/uav2/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist",
        "/model/uav2/enable@std_msgs/msg/Bool@gz.msgs.Boolean",
        "/model/uav2/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
    ]
    bridge_cmd = "source /opt/ros/jazzy/setup.bash && ros2 run ros_gz_bridge parameter_bridge " + " ".join(bridge_topics)
    bridge = run_bg(bridge_cmd)
    time.sleep(2)

    rviz = None
    if not args.no_rviz:
        log_event(f"Starting RViz with config: {RVIZ_CONFIG_PATH}")
        rviz = run_bg(f"source /opt/ros/jazzy/setup.bash && rviz2 -d {RVIZ_CONFIG_PATH}")

    rclpy.init(args=None)
    executor = MultiThreadedExecutor()

    obstacle_node = ObstacleDetectionNode(
        node_name="husky_2_obstacle_detector",
        scan_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
        action_topic="/husky_2/obstacle_action",
        clearance_topic="/husky_2/obstacle_clearance",
        pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
        stop_distance=OBSTACLE_STOP_DISTANCE,
        caution_distance=OBSTACLE_CAUTION_DISTANCE,
        gap_profile_topic=None,
    )
    model_driver = HuskyTrajectoryModelDriver(
        node_name="husky_2_trajectory_model_driver",
        model_slug=model_slug,
        checkpoint_path=checkpoint_path,
        cmd_topic="/cmd_vel_husky2",
        husky_odom_topic="/model/husky_2/odometry",
        uav1_odom_topic="/model/uav1/odometry",
        uav2_odom_topic="/model/uav2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        obstacle_clearance_topic="/husky_2/obstacle_clearance",
        goal_xyz=WORLD_SHARED_GOAL,
        target_index=args.target_index,
        control_period=CONTROL_PERIOD,
    )
    uav1_follower = UavFollower(
        node_name="uav1_follower",
        husky_odom_topic="/model/husky_2/odometry",
        uav_odom_topic="/model/uav1/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        husky_model_name="husky_2",
        uav_model_name="uav1",
        uav_name="uav1",
        follow_distance=0.0,
        follow_height=UAV_FOLLOW_HEIGHT,
        husky_spawn_xyz=(HUSKY2_X, HUSKY2_Y, HUSKY2_Z),
        uav_spawn_xyz=(UAV1_X, UAV1_Y, UAV1_Z),
    )
    uav2_follower = UavFollower(
        node_name="uav2_follower",
        husky_odom_topic="/model/husky_2/odometry",
        uav_odom_topic="/model/uav2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        husky_model_name="husky_2",
        uav_model_name="uav2",
        uav_name="uav2",
        follow_distance=0.0,
        follow_height=UAV_FOLLOW_HEIGHT,
        husky_spawn_xyz=(HUSKY2_X, HUSKY2_Y, HUSKY2_Z),
        uav_spawn_xyz=(UAV2_X, UAV2_Y, UAV2_Z),
        ready_topic="/uav2/ready",
    )

    for node in (obstacle_node, model_driver, uav1_follower, uav2_follower):
        executor.add_node(node)

    processes = [gz, bridge]
    if rviz is not None:
        processes.append(rviz)

    try:
        log_event("Model-weight simulation test ready. Press Play in Gazebo, then Ctrl+C here when done.")
        executor.spin()
    except KeyboardInterrupt:
        log_event("Stopping model-weight simulation test...")
    finally:
        for node in (obstacle_node, model_driver, uav1_follower, uav2_follower):
            with suppress(Exception):
                executor.remove_node(node)
            with suppress(Exception):
                node.destroy_node()
        with suppress(Exception):
            rclpy.shutdown()
        for proc in processes:
            with suppress(Exception):
                proc.send_signal(signal.SIGINT)
        time.sleep(1.0)
        for proc in processes:
            with suppress(Exception):
                proc.terminate()
        close_terminal_tee()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
