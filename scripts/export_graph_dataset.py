#!/usr/bin/env python3

import json
import sqlite3
from pathlib import Path

from rosbags.typesys import Stores, get_typestore


BAGS_DIR = Path.home() / "Documents/Thesis/bags"
OUT_ROOT = Path.home() / "Documents/Thesis/graph_dataset"

TOPICS = {
    "husky_local_odom": "/model/husky_local/odometry",
    "husky_2_odom": "/model/husky_2/odometry",
    "uav1_odom": "/model/uav1/odometry",
    "cmd_husky_local": "/cmd_vel",
    "cmd_husky_2": "/cmd_vel_husky2",
    "husky_local_start": "/episode/husky_local/start",
    "husky_local_goal": "/episode/husky_local/goal",
    "husky_2_start": "/episode/husky_2/start",
    "husky_2_goal": "/episode/husky_2/goal",
    "uav1_start": "/episode/uav1/start",
    "uav1_goal": "/episode/uav1/goal",
}


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


def edge(src: str, dst: str, nodes: dict):
    dx = nodes[dst]["state"]["x"] - nodes[src]["state"]["x"]
    dy = nodes[dst]["state"]["y"] - nodes[src]["state"]["y"]
    dz = nodes[dst]["state"]["z"] - nodes[src]["state"]["z"]
    return {
        "source": src,
        "target": dst,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "distance": (dx * dx + dy * dy + dz * dz) ** 0.5,
    }


def main():
    bag_path = latest_bag(BAGS_DIR)
    db3_path = bag_db3_path(bag_path)
    print("Using bag:", bag_path)

    out_dir = OUT_ROOT / bag_path.name
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

    state = {
        "husky_local": None,
        "husky_2": None,
        "uav1": None,
    }
    cmd = {
        "husky_local": {"linear_x": 0.0, "angular_z": 0.0},
        "husky_2": {"linear_x": 0.0, "angular_z": 0.0},
    }
    goals = {
        "husky_local": {"start": None, "goal": None},
        "husky_2": {"start": None, "goal": None},
        "uav1": {"start": None, "goal": None},
    }

    odom_frames = []
    for topic_id, timestamp, rawdata in cur.execute(
        "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
    ):
        topic, msgtype = topic_map[topic_id]
        msg = typestore.deserialize_cdr(rawdata, msgtype)

        if topic == TOPICS["husky_local_odom"]:
            state["husky_local"] = pose_from_odom(msg)
            odom_frames.append(int(timestamp))
        elif topic == TOPICS["husky_2_odom"]:
            state["husky_2"] = pose_from_odom(msg)
        elif topic == TOPICS["uav1_odom"]:
            state["uav1"] = pose_from_odom(msg)
        elif topic == TOPICS["cmd_husky_local"]:
            cmd["husky_local"] = cmd_from_twist(msg)
        elif topic == TOPICS["cmd_husky_2"]:
            cmd["husky_2"] = cmd_from_twist(msg)
        elif topic == TOPICS["husky_local_start"]:
            goals["husky_local"]["start"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["husky_local_goal"]:
            goals["husky_local"]["goal"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["husky_2_start"]:
            goals["husky_2"]["start"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["husky_2_goal"]:
            goals["husky_2"]["goal"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["uav1_start"]:
            goals["uav1"]["start"] = pose_from_pose_stamped(msg)
        elif topic == TOPICS["uav1_goal"]:
            goals["uav1"]["goal"] = pose_from_pose_stamped(msg)

        if topic == TOPICS["husky_local_odom"] and all(state.values()):
            nodes = {
                "husky_local": {
                    "platform_type": "UGV",
                    "state": state["husky_local"],
                    "goal": goals["husky_local"]["goal"],
                    "command": cmd["husky_local"],
                },
                "husky_2": {
                    "platform_type": "UGV",
                    "state": state["husky_2"],
                    "goal": goals["husky_2"]["goal"],
                    "command": cmd["husky_2"],
                },
                "uav1": {
                    "platform_type": "UAV",
                    "state": state["uav1"],
                    "goal": goals["uav1"]["goal"],
                    "command": None,
                },
            }
            edges = [
                edge("husky_local", "husky_2", nodes),
                edge("husky_local", "uav1", nodes),
                edge("husky_2", "husky_local", nodes),
                edge("husky_2", "uav1", nodes),
                edge("uav1", "husky_local", nodes),
                edge("uav1", "husky_2", nodes),
            ]
            with frames_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp_ns": int(timestamp),
                            "nodes": nodes,
                            "edges": edges,
                        }
                    )
                    + "\n"
                )

    conn.close()

    schema = {
        "frame_key": "timestamp_ns",
        "nodes": {
            "id": "entity name",
            "platform_type": "UGV or UAV",
            "state": ["x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wz"],
            "goal": ["x", "y", "z"],
            "command": ["linear_x", "angular_z"] if True else None,
        },
        "edges": {
            "source": "node id",
            "target": "node id",
            "dx": "target.x - source.x",
            "dy": "target.y - source.y",
            "dz": "target.z - source.z",
            "distance": "euclidean distance",
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2))

    print("Saved graph dataset to:", out_dir)
    print("Frames:", sum(1 for _ in frames_path.open()))


if __name__ == "__main__":
    main()
