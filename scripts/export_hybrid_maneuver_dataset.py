#!/usr/bin/env python3
"""Export a JSONL hybrid maneuver dataset from a recorded rosbag.

The output is aimed at a future CNN-GNN-LSTM + lightweight transformer stack:
- sensor assets are saved to disk and referenced from JSONL frames
- each frame includes multi-agent graph context
- each Husky frame includes a teacher maneuver label from the rule-based controller
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore


WORLD_NAME = "sim_world"
BAGS_DIR = Path.home() / "Documents/Thesis/bags"
OUT_ROOT = Path.home() / "Documents/Thesis/hybrid_maneuver_dataset"

TOPICS = {
    "husky_local_odom": "/model/husky_local/odometry",
    "husky_2_odom": "/model/husky_2/odometry",
    "uav1_odom": "/model/uav1/odometry",
    "cmd_husky_local": "/cmd_vel",
    "cmd_husky_2": "/cmd_vel_husky2",
    "state_husky_local": "/husky_local/controller_state",
    "state_husky_2": "/husky_2/controller_state",
    "obstacle_action_husky_local": "/husky_local/obstacle_action",
    "obstacle_action_husky_2": "/husky_2/obstacle_action",
    "obstacle_clearance_husky_local": "/husky_local/obstacle_clearance",
    "obstacle_clearance_husky_2": "/husky_2/obstacle_clearance",
    "husky_local_planar_scan": f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
    "husky_2_planar_scan": f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
    "husky_local_front_points": f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
    "husky_2_front_points": f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
    "uav1_front_points": f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points",
    "husky_local_front_image": f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/camera_front/image",
    "husky_2_front_image": f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/camera_front/image",
    "uav1_front_image": f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/camera_front/image",
    "husky_local_start": "/episode/husky_local/start",
    "husky_local_goal": "/episode/husky_local/goal",
    "husky_2_start": "/episode/husky_2/start",
    "husky_2_goal": "/episode/husky_2/goal",
    "uav1_start": "/episode/uav1/start",
    "uav1_goal": "/episode/uav1/goal",
}

AGENT_KEYS = ["husky_local", "husky_2", "uav1"]
PLATFORM_TYPES = {"husky_local": "UGV", "husky_2": "UGV", "uav1": "UAV"}


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


def pose_from_odom(msg):
    pose = msg.pose.pose
    twist = msg.twist.twist
    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
        "qx": float(pose.orientation.x),
        "qy": float(pose.orientation.y),
        "qz": float(pose.orientation.z),
        "qw": float(pose.orientation.w),
        "vx": float(twist.linear.x),
        "vy": float(twist.linear.y),
        "vz": float(twist.linear.z),
        "wz": float(twist.angular.z),
    }


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
    """Convert PointCloud2 -> (N, 4) float32 [x, y, z, intensity]."""
    n_points = msg.width * msg.height
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("pad1", np.float32),
            ("intensity", np.float32),
            ("pad2", np.float32),
            ("ring", np.uint16),
            ("pad3", np.uint16),
        ]
    )
    arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)
    pts = np.zeros((n_points, 4), dtype=np.float32)
    pts[:, 0] = arr["x"]
    pts[:, 1] = arr["y"]
    pts[:, 2] = arr["z"]
    pts[:, 3] = arr["intensity"]
    return pts


def decode_image(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    encoding = getattr(msg, "encoding", "") or ""
    width = int(msg.width)
    height = int(msg.height)
    if encoding in {"rgb8", "bgr8"}:
        return data.reshape(height, width, 3)
    if encoding in {"rgba8", "bgra8"}:
        return data.reshape(height, width, 4)
    if encoding == "mono8":
        return data.reshape(height, width)
    if encoding == "mono16":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
    return data.reshape(height, int(msg.step)) if height > 0 and getattr(msg, "step", 0) else data


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


def save_image(msg, out_path: Path):
    image = decode_image(msg)
    np.save(out_path, image)
    return {
        "path": str(out_path),
        "modality": "image",
        "encoding": getattr(msg, "encoding", "") or "",
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }


def edge(src: str, dst: str, agents: dict):
    src_state = agents[src]["state"]
    dst_state = agents[dst]["state"]
    dx = float(dst_state["x"] - src_state["x"])
    dy = float(dst_state["y"] - src_state["y"])
    dz = float(dst_state["z"] - src_state["z"])
    distance = float((dx * dx + dy * dy + dz * dz) ** 0.5)
    inv_distance = float(1.0 / max(distance, 1e-3))
    bearing = float(np.arctan2(dy, dx))
    return {
        "source": src,
        "target": dst,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "distance": distance,
        "inv_distance": inv_distance,
        "bearing_sin": float(np.sin(bearing)),
        "bearing_cos": float(np.cos(bearing)),
        "same_platform": float(PLATFORM_TYPES[src] == PLATFORM_TYPES[dst]),
    }


def heuristic_maneuver_label(agent: dict) -> str:
    state = agent.get("controller_state")
    if state:
        return str(state)
    command = agent.get("command") or {"linear_x": 0.0, "angular_z": 0.0}
    obstacle_action = agent.get("obstacle_action") or "clear"
    linear_x = float(command["linear_x"])
    angular_z = float(command["angular_z"])
    if linear_x < -0.1:
        return "reverse"
    if obstacle_action.endswith("left") and angular_z > 0.1:
        return "avoid_left"
    if obstacle_action.endswith("right") and angular_z < -0.1:
        return "avoid_right"
    if abs(linear_x) < 0.05 and abs(angular_z) < 0.05:
        return "stop"
    return "go_to_goal"


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
        **({"encoding": ref["encoding"]} if "encoding" in ref else {}),
    }


def build_frame(
    episode_id: str,
    timestamp_ns: int,
    ego_id: str,
    agents: dict,
    latest_assets: dict,
):
    other_husky = "husky_2" if ego_id == "husky_local" else "husky_local"
    edges = [edge(src, dst, agents) for src in AGENT_KEYS for dst in AGENT_KEYS if src != dst]
    teacher_label = heuristic_maneuver_label(agents[ego_id])
    return {
        "episode_id": episode_id,
        "timestamp_ns": int(timestamp_ns),
        "world_name": WORLD_NAME,
        "ego_id": ego_id,
        "teacher": {
            "label": teacher_label,
            "label_source": "controller_state" if agents[ego_id].get("controller_state") else "heuristic_fallback",
            "controller_state": agents[ego_id].get("controller_state"),
            "command": agents[ego_id].get("command"),
            "obstacle_action": agents[ego_id].get("obstacle_action"),
            "obstacle_clearance": agents[ego_id].get("obstacle_clearance"),
        },
        "agents": agents,
        "edges": edges,
        "modalities": {
            "ego_planar_scan": asset_ref(latest_assets, f"{ego_id}_planar_scan"),
            "ego_front_pointcloud": asset_ref(latest_assets, f"{ego_id}_front_points"),
            "ego_front_camera": asset_ref(latest_assets, f"{ego_id}_front_image"),
            "other_husky_planar_scan": asset_ref(latest_assets, f"{other_husky}_planar_scan"),
            "other_husky_front_pointcloud": asset_ref(latest_assets, f"{other_husky}_front_points"),
            "other_husky_front_camera": asset_ref(latest_assets, f"{other_husky}_front_image"),
            "uav_front_pointcloud": asset_ref(latest_assets, "uav1_front_points"),
            "uav_front_camera": asset_ref(latest_assets, "uav1_front_image"),
        },
        "readiness": {
            "has_teacher_label": teacher_label is not None,
            "has_graph_context": all(agents[agent]["state"] is not None for agent in AGENT_KEYS),
            "has_ego_scan": asset_ref(latest_assets, f"{ego_id}_planar_scan") is not None,
            "has_uav_context": asset_ref(latest_assets, "uav1_front_points") is not None,
        },
    }


def schema():
    return {
        "description": "Per-frame JSONL hybrid maneuver dataset for CNN-GNN-LSTM + lightweight transformer training.",
        "frame_anchor": "Each frame is anchored on one Husky planar laser timestamp and tagged with ego_id.",
        "required_for_maneuver_training": [
            "teacher.label",
            "agents",
            "edges",
            "modalities.ego_planar_scan",
        ],
        "recommended_for_hybrid_model": [
            "modalities.ego_front_pointcloud",
            "modalities.ego_front_camera",
            "modalities.uav_front_pointcloud",
            "modalities.uav_front_camera",
        ],
        "teacher_labels": [
            "bootstrap",
            "go_to_goal",
            "avoid_left",
            "avoid_right",
            "commit_forward",
            "reverse",
            "recover",
            "reassess",
            "arrived",
            "stop",
        ],
        "agent_state_fields": ["x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wz"],
        "edge_fields": ["dx", "dy", "dz", "distance", "inv_distance", "bearing_sin", "bearing_cos", "same_platform"],
    }


def main():
    parser = argparse.ArgumentParser(description="Export a hybrid maneuver JSONL dataset from the latest bag.")
    parser.add_argument("--bag", default="", help="Optional path to a specific run_* bag directory.")
    parser.add_argument("--out-root", default=str(OUT_ROOT), help="Dataset output root.")
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve() if args.bag else latest_bag(BAGS_DIR)
    db3_path = bag_db3_path(bag_path)
    out_root = Path(args.out_root).expanduser().resolve()
    out_dir = out_root / bag_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"
    schema_path = out_dir / "schema.json"
    if frames_path.exists():
        frames_path.unlink()

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()
    topic_map = {
        topic_id: (name, msgtype)
        for topic_id, name, msgtype in cur.execute("SELECT id, name, type FROM topics ORDER BY id")
    }

    agents = {
        "husky_local": {
            "platform_type": "UGV",
            "state": None,
            "start": None,
            "goal": None,
            "command": {"linear_x": 0.0, "angular_z": 0.0},
            "controller_state": None,
            "obstacle_action": "clear",
            "obstacle_clearance": None,
        },
        "husky_2": {
            "platform_type": "UGV",
            "state": None,
            "start": None,
            "goal": None,
            "command": {"linear_x": 0.0, "angular_z": 0.0},
            "controller_state": None,
            "obstacle_action": "clear",
            "obstacle_clearance": None,
        },
        "uav1": {
            "platform_type": "UAV",
            "state": None,
            "start": None,
            "goal": None,
            "command": None,
            "controller_state": None,
            "obstacle_action": None,
            "obstacle_clearance": None,
        },
    }
    latest_assets = {}

    asset_dirs = {
        "husky_local_planar_scan": make_asset_dir(out_dir, "husky_local/planar_scan"),
        "husky_2_planar_scan": make_asset_dir(out_dir, "husky_2/planar_scan"),
        "husky_local_front_points": make_asset_dir(out_dir, "husky_local/front_points"),
        "husky_2_front_points": make_asset_dir(out_dir, "husky_2/front_points"),
        "uav1_front_points": make_asset_dir(out_dir, "uav1/front_points"),
        "husky_local_front_image": make_asset_dir(out_dir, "husky_local/front_image"),
        "husky_2_front_image": make_asset_dir(out_dir, "husky_2/front_image"),
        "uav1_front_image": make_asset_dir(out_dir, "uav1/front_image"),
    }

    topic_to_asset_key = {
        TOPICS["husky_local_planar_scan"]: "husky_local_planar_scan",
        TOPICS["husky_2_planar_scan"]: "husky_2_planar_scan",
        TOPICS["husky_local_front_points"]: "husky_local_front_points",
        TOPICS["husky_2_front_points"]: "husky_2_front_points",
        TOPICS["uav1_front_points"]: "uav1_front_points",
        TOPICS["husky_local_front_image"]: "husky_local_front_image",
        TOPICS["husky_2_front_image"]: "husky_2_front_image",
        TOPICS["uav1_front_image"]: "uav1_front_image",
    }

    anchor_topics = {
        TOPICS["husky_local_planar_scan"]: "husky_local",
        TOPICS["husky_2_planar_scan"]: "husky_2",
    }

    frame_count = 0
    for topic_id, timestamp, rawdata in cur.execute(
        "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
    ):
        topic, msgtype = topic_map[topic_id]
        msg = typestore.deserialize_cdr(rawdata, msgtype)
        ts = int(timestamp)

        if topic == TOPICS["husky_local_odom"]:
            agents["husky_local"]["state"] = pose_from_odom(msg)
        elif topic == TOPICS["husky_2_odom"]:
            agents["husky_2"]["state"] = pose_from_odom(msg)
        elif topic == TOPICS["uav1_odom"]:
            agents["uav1"]["state"] = pose_from_odom(msg)
        elif topic == TOPICS["cmd_husky_local"]:
            agents["husky_local"]["command"] = cmd_from_twist(msg)
        elif topic == TOPICS["cmd_husky_2"]:
            agents["husky_2"]["command"] = cmd_from_twist(msg)
        elif topic == TOPICS["state_husky_local"]:
            agents["husky_local"]["controller_state"] = str(msg.data)
        elif topic == TOPICS["state_husky_2"]:
            agents["husky_2"]["controller_state"] = str(msg.data)
        elif topic == TOPICS["obstacle_action_husky_local"]:
            agents["husky_local"]["obstacle_action"] = str(msg.data)
        elif topic == TOPICS["obstacle_action_husky_2"]:
            agents["husky_2"]["obstacle_action"] = str(msg.data)
        elif topic == TOPICS["obstacle_clearance_husky_local"]:
            agents["husky_local"]["obstacle_clearance"] = clearance_from_vector3(msg)
        elif topic == TOPICS["obstacle_clearance_husky_2"]:
            agents["husky_2"]["obstacle_clearance"] = clearance_from_vector3(msg)
        elif topic == TOPICS["husky_local_start"]:
            agents["husky_local"]["start"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["husky_local_goal"]:
            agents["husky_local"]["goal"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["husky_2_start"]:
            agents["husky_2"]["start"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["husky_2_goal"]:
            agents["husky_2"]["goal"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["uav1_start"]:
            agents["uav1"]["start"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["uav1_goal"]:
            agents["uav1"]["goal"] = pose_from_pose_stamped(msg)

        if topic in topic_to_asset_key:
            asset_key = topic_to_asset_key[topic]
            asset_dir = asset_dirs[asset_key]
            if asset_key.endswith("planar_scan"):
                out_path = asset_dir / f"{ts}.npy"
                meta = save_laserscan(msg, out_path)
            elif asset_key.endswith("front_points"):
                out_path = asset_dir / f"{ts}.npy"
                meta = save_pointcloud(msg, out_path)
            else:
                out_path = asset_dir / f"{ts}.npy"
                meta = save_image(msg, out_path)
            meta["timestamp_ns"] = ts
            latest_assets[asset_key] = meta

        if topic in anchor_topics:
            ego_id = anchor_topics[topic]
            if not all(agents[agent]["state"] is not None for agent in AGENT_KEYS):
                continue
            frame = build_frame(
                episode_id=bag_path.name,
                timestamp_ns=ts,
                ego_id=ego_id,
                agents=agents,
                latest_assets=latest_assets,
            )
            with frames_path.open("a") as f:
                f.write(json.dumps(frame) + "\n")
            frame_count += 1

    conn.close()
    schema_path.write_text(json.dumps(schema(), indent=2))
    print("Saved hybrid dataset to:", out_dir)
    print("Frames:", frame_count)
    print("Schema:", schema_path)


if __name__ == "__main__":
    main()
