#!/usr/bin/env python3
"""Simple one-bag exporter: ROS2 bag -> episode_frames."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import re
from pathlib import Path

import yaml
from rosbags.typesys import Stores, get_typestore


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
THESIS_ROOT = Path.home() / "Documents/Thesis"
OUT_ROOT = THESIS_ROOT / "model_training" / "results" / "episode_frames"
WORLD_TOPIC_RE = re.compile(r"^/world/([^/]+)/")


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_from_tf_transform(transform) -> dict[str, float]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return {
        "x": float(translation.x),
        "y": float(translation.y),
        "z": float(translation.z),
        "yaw": float(
            quaternion_to_yaw(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )
        ),
    }


def update_world_poses(msg, world_poses: dict[str, dict | None]) -> None:
    for transform in msg.transforms:
        child = transform.child_frame_id or ""
        parts = [part for part in child.split("/") if part]
        if "husky_2" in parts:
            world_poses["husky_2"] = pose_from_tf_transform(transform)
        elif "uav1" in parts:
            world_poses["uav1"] = pose_from_tf_transform(transform)
        elif "uav2" in parts:
            world_poses["uav2"] = pose_from_tf_transform(transform)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path, help="Path to one run_* bag directory")
    args = parser.parse_args()

    bag_dir = args.bag.resolve()
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing metadata.yaml in {bag_dir}")

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        raise RuntimeError(f"No .db3 file found in {bag_dir}")
    db3_path = db3_files[0]

    metadata = yaml.safe_load(metadata_path.read_text())
    topic_types = {}
    for entry in metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
        topic_types[entry["topic_metadata"]["name"]] = entry["topic_metadata"]["type"]
    world_name = None
    for topic_name in topic_types:
        match = WORLD_TOPIC_RE.match(topic_name)
        if match:
            world_name = match.group(1)
            break
    if world_name is None:
        raise RuntimeError("Could not infer Gazebo world name from bag topics.")

    episode_id = bag_dir.name
    out_dir = OUT_ROOT / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"
    manifest_path = out_dir / "manifest.json"

    dynamic_pose_topic = f"/world/{world_name}/dynamic_pose/info"
    husky_odom_topic = "/model/husky_2/odometry"
    goal_topic = "/episode/husky_2/goal"
    start_topic = "/episode/husky_2/start"
    cmd_topic = "/cmd_vel_husky2"
    state_topic = "/husky_2/controller_state"
    obstacle_action_topic = "/husky_2/obstacle_action"
    obstacle_clearance_topic = "/husky_2/obstacle_clearance"
    uav1_odom_topic = "/model/uav1/odometry"
    uav2_odom_topic = "/model/uav2/odometry"

    if husky_odom_topic not in topic_types:
        raise RuntimeError(f"Required topic missing: {husky_odom_topic}")
    if goal_topic not in topic_types:
        raise RuntimeError(f"Required topic missing: {goal_topic}")

    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()
    topic_rows = cur.execute("SELECT id, name, type FROM topics").fetchall()
    topic_info = {name: {"id": topic_id, "type": ros_type} for topic_id, name, ros_type in topic_rows}

    goal_world = None
    start_world = None
    latest_cmd = {"linear_x": 0.0, "angular_z": 0.0}
    latest_state = None
    latest_obstacle_action = "clear"
    latest_obstacle_clearance = {"front": 999.0, "left": 999.0, "right": 999.0}
    latest_uav1_odom = None
    latest_uav2_odom = None
    world_poses = {
        "husky_2": None,
        "uav1": None,
        "uav2": None,
    }

    frame_count = 0
    start_time_ns = None
    end_time_ns = None

    query = (
        "SELECT topics.name, messages.timestamp, messages.data "
        "FROM messages JOIN topics ON messages.topic_id = topics.id "
        "ORDER BY messages.timestamp"
    )

    with frames_path.open("w", encoding="utf-8") as fp:
        for topic_name, timestamp_ns, blob in cur.execute(query):
            if start_time_ns is None:
                start_time_ns = int(timestamp_ns)
            end_time_ns = int(timestamp_ns)

            ros_type = topic_info[topic_name]["type"]
            msg = TYPESTORE.deserialize_cdr(blob, ros_type)

            if topic_name == dynamic_pose_topic:
                update_world_poses(msg, world_poses)
                continue

            if topic_name == goal_topic:
                goal_world = {
                    "x": float(msg.pose.position.x),
                    "y": float(msg.pose.position.y),
                    "z": float(msg.pose.position.z),
                }
                continue

            if topic_name == start_topic:
                start_world = {
                    "x": float(msg.pose.position.x),
                    "y": float(msg.pose.position.y),
                    "z": float(msg.pose.position.z),
                }
                continue

            if topic_name == cmd_topic:
                latest_cmd = {
                    "linear_x": float(msg.linear.x),
                    "angular_z": float(msg.angular.z),
                }
                continue

            if topic_name == state_topic:
                latest_state = str(msg.data)
                continue

            if topic_name == obstacle_action_topic:
                latest_obstacle_action = str(msg.data)
                continue

            if topic_name == obstacle_clearance_topic:
                latest_obstacle_clearance = {
                    "front": float(msg.x),
                    "left": float(msg.y),
                    "right": float(msg.z),
                }
                continue

            if topic_name == uav1_odom_topic:
                pose = msg.pose.pose
                twist = msg.twist.twist
                world_pose = world_poses["uav1"]
                latest_uav1_odom = {
                    "x": float(world_pose["x"]) if world_pose else float(pose.position.x),
                    "y": float(world_pose["y"]) if world_pose else float(pose.position.y),
                    "z": float(world_pose["z"]) if world_pose else float(pose.position.z),
                    "yaw": float(world_pose["yaw"]) if world_pose else float(
                        quaternion_to_yaw(
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        )
                    ),
                    "vx": float(twist.linear.x),
                    "wz": float(twist.angular.z),
                }
                continue

            if topic_name == uav2_odom_topic:
                pose = msg.pose.pose
                twist = msg.twist.twist
                world_pose = world_poses["uav2"]
                latest_uav2_odom = {
                    "x": float(world_pose["x"]) if world_pose else float(pose.position.x),
                    "y": float(world_pose["y"]) if world_pose else float(pose.position.y),
                    "z": float(world_pose["z"]) if world_pose else float(pose.position.z),
                    "yaw": float(world_pose["yaw"]) if world_pose else float(
                        quaternion_to_yaw(
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        )
                    ),
                    "vx": float(twist.linear.x),
                    "wz": float(twist.angular.z),
                }
                continue

            if topic_name != husky_odom_topic:
                continue

            if goal_world is None or world_poses["husky_2"] is None:
                continue

            pose = msg.pose.pose
            twist = msg.twist.twist
            world_pose = world_poses["husky_2"]
            yaw = float(world_pose["yaw"])
            x = float(world_pose["x"])
            y = float(world_pose["y"])
            z = float(world_pose["z"])
            dx = float(goal_world["x"]) - x
            dy = float(goal_world["y"]) - y
            c = math.cos(-yaw)
            s = math.sin(-yaw)
            rel_goal_x = c * dx - s * dy
            rel_goal_y = s * dx + c * dy

            frame = {
                "episode_id": episode_id,
                "timestamp_ns": int(timestamp_ns),
                "ego": {
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": float(yaw),
                    "vx": float(twist.linear.x),
                    "vy": float(twist.linear.y),
                    "vz": float(twist.linear.z),
                    "wz": float(twist.angular.z),
                },
                "goal": {
                    "goal_x_world": float(goal_world["x"]),
                    "goal_y_world": float(goal_world["y"]),
                    "rel_goal_x_ego": float(rel_goal_x),
                    "rel_goal_y_ego": float(rel_goal_y),
                    "goal_distance": float(math.hypot(dx, dy)),
                    "goal_heading_error": float(wrap_angle(math.atan2(dy, dx) - yaw)),
                },
                "teacher_cmd": latest_cmd,
                "teacher_state": latest_state,
                "obstacle_action": latest_obstacle_action,
                "obstacle_clearance": latest_obstacle_clearance,
                "start_world": start_world,
                "goal_world": goal_world,
                "agents": {
                    "uav1": latest_uav1_odom,
                    "uav2": latest_uav2_odom,
                },
            }
            fp.write(json.dumps(frame) + "\n")
            frame_count += 1

    conn.close()

    manifest = {
        "episode_id": episode_id,
        "world_name": world_name,
        "bag_dir": str(bag_dir),
        "db3_path": str(db3_path),
        "output_dir": str(out_dir),
        "frames_path": str(frames_path),
        "frame_count": int(frame_count),
        "start_time_ns": int(start_time_ns or 0),
        "end_time_ns": int(end_time_ns or 0),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
