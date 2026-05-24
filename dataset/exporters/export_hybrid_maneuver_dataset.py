#!/usr/bin/env python3
"""Export a control-focused Husky dataset from a recorded rosbag.

The exported frames are anchored on each Husky planar lidar scan and keep the
signals needed for imitation learning or trajectory supervision:
- timestamp
- lidar observation
- robot state
- goal state and relative goal features
- teacher command and controller state
- optional second-Husky context
"""

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore


THESIS_ROOT = Path.home() / "Documents/Thesis"
DATASET_ROOT = THESIS_ROOT / "dataset"
BAGS_DIR = DATASET_ROOT / "bags"
OUT_ROOT = DATASET_ROOT / "husky_control_dataset"

AGENT_KEYS = ["husky_local", "husky_2"]
WORLD_TOPIC_RE = re.compile(r"^/world/([^/]+)/")


def latest_bag(path: Path) -> Path:
    bags = sorted(path.glob("run_*"))
    if not bags:
        raise RuntimeError(f"No bag found in {path}")
    return bags[-1].resolve()


def bag_db3_path(bag_dir: Path) -> Path:
    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        raise RuntimeError(f"No .db3 file found in {bag_dir}")
    return db3_files[0]


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def infer_world_name(topic_names: list[str]) -> str:
    for topic in topic_names:
        match = WORLD_TOPIC_RE.match(topic)
        if match:
            return match.group(1)
    raise RuntimeError("Could not infer Gazebo world name from bag topics.")


def build_topics(world_name: str) -> dict[str, str]:
    return {
        "dynamic_pose": f"/world/{world_name}/dynamic_pose/info",
        "husky_local_odom": "/model/husky_local/odometry",
        "husky_2_odom": "/model/husky_2/odometry",
        "cmd_husky_local": "/cmd_vel",
        "cmd_husky_2": "/cmd_vel_husky2",
        "state_husky_local": "/husky_local/controller_state",
        "state_husky_2": "/husky_2/controller_state",
        "obstacle_action_husky_local": "/husky_local/obstacle_action",
        "obstacle_action_husky_2": "/husky_2/obstacle_action",
        "obstacle_clearance_husky_local": "/husky_local/obstacle_clearance",
        "obstacle_clearance_husky_2": "/husky_2/obstacle_clearance",
        "husky_local_planar_scan": f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan",
        "husky_2_planar_scan": f"/world/{world_name}/model/husky_2/link/base_link/sensor/planar_laser/scan",
        "husky_local_front_points": f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
        "husky_2_front_points": f"/world/{world_name}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
        "husky_local_start": "/episode/husky_local/start",
        "husky_local_goal": "/episode/husky_local/goal",
        "husky_2_start": "/episode/husky_2/start",
        "husky_2_goal": "/episode/husky_2/goal",
    }


def pose_from_odom(msg):
    pose = msg.pose.pose
    twist = msg.twist.twist
    yaw = quaternion_to_yaw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
        "qx": float(pose.orientation.x),
        "qy": float(pose.orientation.y),
        "qz": float(pose.orientation.z),
        "qw": float(pose.orientation.w),
        "yaw": float(yaw),
        "vx": float(twist.linear.x),
        "vy": float(twist.linear.y),
        "vz": float(twist.linear.z),
        "wz": float(twist.angular.z),
    }


def pose_from_tf_transform(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    yaw = quaternion_to_yaw(rotation.x, rotation.y, rotation.z, rotation.w)
    return {
        "x": float(translation.x),
        "y": float(translation.y),
        "z": float(translation.z),
        "qx": float(rotation.x),
        "qy": float(rotation.y),
        "qz": float(rotation.z),
        "qw": float(rotation.w),
        "yaw": float(yaw),
    }


def update_agents_from_dynamic_pose(msg, agents: dict):
    for transform in msg.transforms:
        child = transform.child_frame_id or ""
        parts = [part for part in child.split("/") if part]
        matched_agent = None
        for agent_name in AGENT_KEYS:
            if agent_name in parts:
                matched_agent = agent_name
                break
        if matched_agent is None:
            continue
        state = pose_from_tf_transform(transform)
        current = agents[matched_agent].get("state")
        if current is not None:
            state["vx"] = float(current.get("vx", 0.0))
            state["vy"] = float(current.get("vy", 0.0))
            state["vz"] = float(current.get("vz", 0.0))
            state["wz"] = float(current.get("wz", 0.0))
        else:
            state["vx"] = 0.0
            state["vy"] = 0.0
            state["vz"] = 0.0
            state["wz"] = 0.0
        agents[matched_agent]["state"] = state
        agents[matched_agent]["available"] = True


def pose_from_pose_stamped(msg):
    pose = msg.pose
    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
    }


def cmd_from_twist(msg):
    return {
        "linear_x": float(msg.linear.x),
        "angular_z": float(msg.angular.z),
    }


def clearance_from_vector3(msg):
    return {
        "front": float(msg.x),
        "left": float(msg.y),
        "right": float(msg.z),
    }


def decode_pointcloud2_to_xyz_i(msg):
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("intensity", np.float32),
        ]
    )
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    point_step = int(msg.point_step)
    count = int(len(raw) / max(point_step, 1))
    if count == 0:
        return np.zeros((0, 4), dtype=np.float32)

    field_offsets = {field.name: int(field.offset) for field in msg.fields}
    points = np.zeros((count, 4), dtype=np.float32)
    for i in range(count):
        base = i * point_step
        for col, name in enumerate(("x", "y", "z", "intensity")):
            offset = field_offsets.get(name)
            if offset is None:
                continue
            points[i, col] = np.frombuffer(raw[base + offset : base + offset + 4], dtype=np.float32)[0]
    return points


def save_laserscan(msg, out_path: Path):
    intensities = np.asarray(msg.intensities, dtype=np.float32)
    if intensities.size == 0:
        intensities = np.zeros(len(msg.ranges), dtype=np.float32)
    scan = np.stack(
        [
            np.asarray(msg.ranges, dtype=np.float32),
            intensities,
        ],
        axis=1,
    )
    np.save(out_path, scan)
    return {
        "path": str(out_path),
        "modality": "planar_scan",
        "shape": list(scan.shape),
        "dtype": str(scan.dtype),
    }


def save_pointcloud(msg, out_path: Path):
    points = decode_pointcloud2_to_xyz_i(msg)
    np.save(out_path, points)
    return {
        "path": str(out_path),
        "modality": "pointcloud_xyz_i",
        "shape": list(points.shape),
        "dtype": str(points.dtype),
    }


def make_asset_dir(root: Path, group: str) -> Path:
    path = root / "assets" / group
    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_ref(latest_assets: dict, key: str):
    ref = latest_assets.get(key)
    if ref is None:
        return None
    return {
        "path": ref["path"],
        "timestamp_ns": ref["timestamp_ns"],
        "modality": ref["modality"],
        "shape": ref["shape"],
        "dtype": ref["dtype"],
    }


def relative_goal_features(state: dict | None, goal: dict | None) -> dict | None:
    if state is None or goal is None:
        return None
    dx = float(goal["x"] - state["x"])
    dy = float(goal["y"] - state["y"])
    dz = float(goal["z"] - state["z"])
    distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
    goal_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(goal_heading - float(state.get("yaw", 0.0)))
    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "distance_to_goal": distance,
        "goal_heading": float(goal_heading),
        "heading_error": float(heading_error),
    }


def build_frame(
    episode_id: str,
    timestamp_ns: int,
    world_name: str,
    ego_id: str,
    agents: dict,
    latest_assets: dict,
):
    ego = agents[ego_id]
    other_id = "husky_2" if ego_id == "husky_local" else "husky_local"
    other = agents[other_id]
    return {
        "episode_id": episode_id,
        "timestamp_ns": int(timestamp_ns),
        "world_name": world_name,
        "ego_id": ego_id,
        "observation": {
            "ego_planar_scan": asset_ref(latest_assets, f"{ego_id}_planar_scan"),
            "ego_front_pointcloud": asset_ref(latest_assets, f"{ego_id}_front_points"),
        },
        "state": ego.get("state"),
        "goal": ego.get("goal"),
        "goal_features": relative_goal_features(ego.get("state"), ego.get("goal")),
        "teacher": {
            "command": ego.get("command"),
            "controller_state": ego.get("controller_state"),
            "obstacle_action": ego.get("obstacle_action"),
            "obstacle_clearance": ego.get("obstacle_clearance"),
        },
        "other_husky": {
            "id": other_id,
            "available": bool(other.get("available", False)),
            "state": other.get("state"),
            "goal": other.get("goal"),
            "goal_features": relative_goal_features(other.get("state"), other.get("goal")),
            "teacher_command": other.get("command"),
        },
        "readiness": {
            "has_scan": asset_ref(latest_assets, f"{ego_id}_planar_scan") is not None,
            "has_state": ego.get("state") is not None,
            "has_goal": ego.get("goal") is not None,
            "has_teacher_command": ego.get("command") is not None,
        },
    }


def schema():
    return {
        "description": "Per-frame Husky control dataset for imitation learning and short-horizon trajectory supervision.",
        "frame_anchor": "Each frame is anchored on one Husky planar lidar timestamp and tagged with ego_id.",
        "primary_inputs": [
            "observation.ego_planar_scan",
            "state",
            "goal",
            "goal_features",
        ],
        "primary_targets": [
            "teacher.command.linear_x",
            "teacher.command.angular_z",
        ],
        "optional_context": [
            "observation.ego_front_pointcloud",
            "other_husky",
            "teacher.controller_state",
            "teacher.obstacle_action",
            "teacher.obstacle_clearance",
        ],
        "state_fields": ["x", "y", "z", "qx", "qy", "qz", "qw", "yaw", "vx", "vy", "vz", "wz"],
        "goal_feature_fields": ["dx", "dy", "dz", "distance_to_goal", "goal_heading", "heading_error"],
    }


def available_bags(root: Path) -> list[Path]:
    return sorted([p.resolve() for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def export_one_bag(bag_path: Path, out_root: Path):
    db3_path = bag_db3_path(bag_path)
    out_dir = out_root / bag_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"
    schema_path = out_dir / "schema.json"
    if frames_path.exists():
        frames_path.unlink()

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()
    topic_rows = list(cur.execute("SELECT id, name, type FROM topics ORDER BY id"))
    topic_map = {topic_id: (name, msgtype) for topic_id, name, msgtype in topic_rows}
    world_name = infer_world_name([name for _, name, _ in topic_rows])
    topics = build_topics(world_name)

    agents = {
        "husky_local": {
            "state": None,
            "start": None,
            "goal": None,
            "command": {"linear_x": 0.0, "angular_z": 0.0},
            "controller_state": None,
            "obstacle_action": "clear",
            "obstacle_clearance": None,
            "available": False,
        },
        "husky_2": {
            "state": None,
            "start": None,
            "goal": None,
            "command": {"linear_x": 0.0, "angular_z": 0.0},
            "controller_state": None,
            "obstacle_action": "clear",
            "obstacle_clearance": None,
            "available": False,
        },
    }
    latest_assets = {}

    asset_dirs = {
        "husky_local_planar_scan": make_asset_dir(out_dir, "husky_local/planar_scan"),
        "husky_2_planar_scan": make_asset_dir(out_dir, "husky_2/planar_scan"),
        "husky_local_front_points": make_asset_dir(out_dir, "husky_local/front_points"),
        "husky_2_front_points": make_asset_dir(out_dir, "husky_2/front_points"),
    }

    topic_to_asset_key = {
        topics["husky_local_planar_scan"]: "husky_local_planar_scan",
        topics["husky_2_planar_scan"]: "husky_2_planar_scan",
        topics["husky_local_front_points"]: "husky_local_front_points",
        topics["husky_2_front_points"]: "husky_2_front_points",
    }

    anchor_topics = {
        topics["husky_local_planar_scan"]: "husky_local",
        topics["husky_2_planar_scan"]: "husky_2",
    }

    frame_count = 0
    for topic_id, timestamp, rawdata in cur.execute(
        "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
    ):
        topic, msgtype = topic_map[topic_id]
        msg = typestore.deserialize_cdr(rawdata, msgtype)
        ts = int(timestamp)

        if topic == topics["husky_local_odom"]:
            agents["husky_local"]["state"] = pose_from_odom(msg)
            agents["husky_local"]["available"] = True
        elif topic == topics["husky_2_odom"]:
            agents["husky_2"]["state"] = pose_from_odom(msg)
            agents["husky_2"]["available"] = True
        elif topic == topics["dynamic_pose"]:
            update_agents_from_dynamic_pose(msg, agents)
        elif topic == topics["cmd_husky_local"]:
            agents["husky_local"]["command"] = cmd_from_twist(msg)
        elif topic == topics["cmd_husky_2"]:
            agents["husky_2"]["command"] = cmd_from_twist(msg)
        elif topic == topics["state_husky_local"]:
            agents["husky_local"]["controller_state"] = str(msg.data)
        elif topic == topics["state_husky_2"]:
            agents["husky_2"]["controller_state"] = str(msg.data)
        elif topic == topics["obstacle_action_husky_local"]:
            agents["husky_local"]["obstacle_action"] = str(msg.data)
        elif topic == topics["obstacle_action_husky_2"]:
            agents["husky_2"]["obstacle_action"] = str(msg.data)
        elif topic == topics["obstacle_clearance_husky_local"]:
            agents["husky_local"]["obstacle_clearance"] = clearance_from_vector3(msg)
        elif topic == topics["obstacle_clearance_husky_2"]:
            agents["husky_2"]["obstacle_clearance"] = clearance_from_vector3(msg)
        elif topic == topics["husky_local_start"]:
            agents["husky_local"]["start"] = pose_from_pose_stamped(msg)
        elif topic == topics["husky_local_goal"]:
            agents["husky_local"]["goal"] = pose_from_pose_stamped(msg)
        elif topic == topics["husky_2_start"]:
            agents["husky_2"]["start"] = pose_from_pose_stamped(msg)
        elif topic == topics["husky_2_goal"]:
            agents["husky_2"]["goal"] = pose_from_pose_stamped(msg)

        if topic in topic_to_asset_key:
            asset_key = topic_to_asset_key[topic]
            asset_dir = asset_dirs[asset_key]
            out_path = asset_dir / f"{ts}.npy"
            if asset_key.endswith("planar_scan"):
                meta = save_laserscan(msg, out_path)
            else:
                meta = save_pointcloud(msg, out_path)
            meta["timestamp_ns"] = ts
            latest_assets[asset_key] = meta

        if topic in anchor_topics:
            ego_id = anchor_topics[topic]
            if agents[ego_id]["state"] is None or agents[ego_id]["goal"] is None:
                continue
            frame = build_frame(
                episode_id=bag_path.name,
                timestamp_ns=ts,
                world_name=world_name,
                ego_id=ego_id,
                agents=agents,
                latest_assets=latest_assets,
            )
            with frames_path.open("a") as f:
                f.write(json.dumps(frame) + "\n")
            frame_count += 1

    conn.close()
    schema_path.write_text(json.dumps(schema(), indent=2))
    print("Saved Husky control dataset to:", out_dir)
    print("World:", world_name)
    print("Frames:", frame_count)
    print("Schema:", schema_path)
    return out_dir, frame_count


def main():
    parser = argparse.ArgumentParser(description="Export a Husky control JSONL dataset from recorded bag(s).")
    parser.add_argument("--bag", default="", help="Optional path to a specific run_* bag directory.")
    parser.add_argument("--out-root", default=str(OUT_ROOT), help="Dataset output root.")
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    if args.bag:
        bag_paths = [Path(args.bag).expanduser().resolve()]
    else:
        bag_paths = available_bags(BAGS_DIR)
        if not bag_paths:
            raise FileNotFoundError(f"No run_* bag directories found in {BAGS_DIR}")

    print("Export root:", out_root)
    print("Bag count:", len(bag_paths))
    total_frames = 0
    for bag_path in bag_paths:
        print("\n=== Exporting", bag_path.name, "===")
        _out_dir, frame_count = export_one_bag(bag_path, out_root)
        total_frames += int(frame_count)

    print("\nCompleted bag export.")
    print("Total bags:", len(bag_paths))
    print("Total frames:", total_frames)


if __name__ == "__main__":
    main()
