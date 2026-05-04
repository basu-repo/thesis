# # #!/usr/bin/env python3
# # """Export a hybrid UAV-UGV maneuver dataset from recorded ROS 2 bags.

# # This exporter converts rosbag2 SQLite bags into a JSONL + NPY dataset suitable
# # for the thesis hybrid models:

# # - CV baseline
# # - CNN-LSTM / temporal CNN-LSTM baseline
# # - GNN-LSTM
# # - CNN-GNN-LSTM

# # Each JSONL row is one synchronized frame anchored on a Husky planar lidar scan.

# # Important dataset idea:
# # The model observes past frames and learns to predict the ego Husky's future
# # 20-step trajectory. The frame contains:
# # - ego Husky state,
# # - ego local lidar/pointcloud references,
# # - ego goal and goal-relative features,
# # - teacher command/state labels from the rule-based controller,
# # - other Husky context,
# # - UAV context,
# # - UAV pointcloud reference,
# # - graph nodes and graph edges.

# # The exporter does NOT train the model. It only prepares the dataset.
# # """

# # from __future__ import annotations

# # import argparse
# # import json
# # import math
# # import re
# # import sqlite3
# # import sys
# # from pathlib import Path
# # from typing import Any

# # import numpy as np


# # try:
# #     from rosbags.typesys import Stores, get_typestore
# # except ModuleNotFoundError:
# #     print(
# #         "\nERROR: Python package 'rosbags' is not installed in this conda environment.\n\n"
# #         "Run this first:\n"
# #         "  conda activate maneuver\n"
# #         "  pip install rosbags\n\n"
# #         "Then run the exporter again:\n"
# #         "  python 03_dataset/exporters/export_hybrid_maneuver_dataset.py\n",
# #         file=sys.stderr,
# #     )
# #     raise


# # THESIS_ROOT = Path.home() / "Documents/Thesis"
# # DATASET_ROOT = THESIS_ROOT / "03_dataset"
# # BAGS_DIR = DATASET_ROOT / "bags"
# # OUT_ROOT = DATASET_ROOT / "husky_control_dataset"

# # WORLD_TOPIC_RE = re.compile(r"^/world/([^/]+)/")

# # AGENT_ORDER = ["husky_local", "husky_2", "uav1"]

# # PLATFORM_TYPE = {
# #     "husky_local": "UGV",
# #     "husky_2": "UGV",
# #     "uav1": "UAV",
# # }

# # PLATFORM_ONEHOT = {
# #     "UGV": [1.0, 0.0],
# #     "UAV": [0.0, 1.0],
# # }

# # DEFAULT_COMMAND = {
# #     "linear_x": 0.0,
# #     "angular_z": 0.0,
# # }

# # DEFAULT_NETWORK_STATE = {
# #     "latency_s": 0.0,
# #     "jitter_s": 0.0,
# #     "packet_loss": 0.0,
# #     "link_quality": 1.0,
# #     "connected": True,
# #     "source": "default_no_omnet",
# # }


# # def bag_db3_path(bag_dir: Path) -> Path:
# #     db3_files = sorted(bag_dir.glob("*.db3"))

# #     if not db3_files:
# #         raise RuntimeError(f"No .db3 file found in {bag_dir}")

# #     return db3_files[0]


# # def available_bags(root: Path) -> list[Path]:
# #     if not root.exists():
# #         return []

# #     prefixes = ("run_", "run_model_", "open_hazard_teacher_")

# #     return sorted(
# #         [
# #             p.resolve()
# #             for p in root.iterdir()
# #             if p.is_dir() and p.name.startswith(prefixes)
# #         ]
# #     )


# # def infer_world_name(topic_names: list[str]) -> str:
# #     for topic in topic_names:
# #         match = WORLD_TOPIC_RE.match(topic)

# #         if match:
# #             return match.group(1)

# #     raise RuntimeError("Could not infer Gazebo world name from bag topics.")


# # def build_topics(world_name: str) -> dict[str, str]:
# #     return {
# #         "dynamic_pose": f"/world/{world_name}/dynamic_pose/info",

# #         "husky_local_odom": "/model/husky_local/odometry",
# #         "husky_2_odom": "/model/husky_2/odometry",
# #         "uav1_odom": "/model/uav1/odometry",

# #         "cmd_husky_local": "/cmd_vel",
# #         "cmd_husky_2": "/cmd_vel_husky2",

# #         "state_husky_local": "/husky_local/controller_state",
# #         "state_husky_2": "/husky_2/controller_state",

# #         "obstacle_action_husky_local": "/husky_local/obstacle_action",
# #         "obstacle_action_husky_2": "/husky_2/obstacle_action",

# #         "obstacle_clearance_husky_local": "/husky_local/obstacle_clearance",
# #         "obstacle_clearance_husky_2": "/husky_2/obstacle_clearance",

# #         "husky_local_start": "/episode/husky_local/start",
# #         "husky_local_goal": "/episode/husky_local/goal",

# #         "husky_2_start": "/episode/husky_2/start",
# #         "husky_2_goal": "/episode/husky_2/goal",

# #         "uav1_start": "/episode/uav1/start",
# #         "uav1_goal": "/episode/uav1/goal",

# #         "husky_local_planar_scan": (
# #             f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan"
# #         ),
# #         "husky_2_planar_scan": (
# #             f"/world/{world_name}/model/husky_2/link/base_link/sensor/planar_laser/scan"
# #         ),

# #         "husky_local_front_points": (
# #             f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points"
# #         ),
# #         "husky_2_front_points": (
# #             f"/world/{world_name}/model/husky_2/link/base_link/sensor/front_laser/scan/points"
# #         ),
# #         "uav1_front_points": (
# #             f"/world/{world_name}/model/uav1/link/base_link/sensor/front_laser/scan/points"
# #         ),

# #         "husky_local_imu": (
# #             f"/world/{world_name}/model/husky_local/link/base_link/sensor/imu_sensor/imu"
# #         ),
# #         "husky_2_imu": (
# #             f"/world/{world_name}/model/husky_2/link/base_link/sensor/imu_sensor/imu"
# #         ),
# #         "uav1_imu": (
# #             f"/world/{world_name}/model/uav1/link/base_link/sensor/imu_sensor/imu"
# #         ),
# #     }


# # def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
# #     siny_cosp = 2.0 * (w * z + x * y)
# #     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

# #     return math.atan2(siny_cosp, cosy_cosp)


# # def wrap_angle(angle: float) -> float:
# #     return math.atan2(math.sin(angle), math.cos(angle))


# # def safe_float(value: Any, default: float = 0.0) -> float:
# #     try:
# #         return float(value)
# #     except Exception:
# #         return float(default)


# # def pose_from_odom(msg) -> dict[str, float]:
# #     pose = msg.pose.pose
# #     twist = msg.twist.twist

# #     yaw = quaternion_to_yaw(
# #         pose.orientation.x,
# #         pose.orientation.y,
# #         pose.orientation.z,
# #         pose.orientation.w,
# #     )

# #     return {
# #         "x": float(pose.position.x),
# #         "y": float(pose.position.y),
# #         "z": float(pose.position.z),

# #         "qx": float(pose.orientation.x),
# #         "qy": float(pose.orientation.y),
# #         "qz": float(pose.orientation.z),
# #         "qw": float(pose.orientation.w),

# #         "yaw": float(yaw),

# #         "vx": float(twist.linear.x),
# #         "vy": float(twist.linear.y),
# #         "vz": float(twist.linear.z),
# #         "wz": float(twist.angular.z),
# #     }


# # def pose_from_tf_transform(transform) -> dict[str, float]:
# #     translation = transform.transform.translation
# #     rotation = transform.transform.rotation

# #     yaw = quaternion_to_yaw(
# #         rotation.x,
# #         rotation.y,
# #         rotation.z,
# #         rotation.w,
# #     )

# #     return {
# #         "x": float(translation.x),
# #         "y": float(translation.y),
# #         "z": float(translation.z),

# #         "qx": float(rotation.x),
# #         "qy": float(rotation.y),
# #         "qz": float(rotation.z),
# #         "qw": float(rotation.w),

# #         "yaw": float(yaw),

# #         "vx": 0.0,
# #         "vy": 0.0,
# #         "vz": 0.0,
# #         "wz": 0.0,
# #     }


# # def pose_from_pose_stamped(msg) -> dict[str, float]:
# #     pose = msg.pose

# #     return {
# #         "x": float(pose.position.x),
# #         "y": float(pose.position.y),
# #         "z": float(pose.position.z),
# #     }


# # def cmd_from_twist(msg) -> dict[str, float]:
# #     return {
# #         "linear_x": float(msg.linear.x),
# #         "angular_z": float(msg.angular.z),
# #     }


# # def clearance_from_vector3(msg) -> dict[str, float]:
# #     return {
# #         "front": float(msg.x),
# #         "left": float(msg.y),
# #         "right": float(msg.z),
# #     }


# # def imu_from_msg(msg) -> dict[str, float]:
# #     return {
# #         "orientation_x": float(msg.orientation.x),
# #         "orientation_y": float(msg.orientation.y),
# #         "orientation_z": float(msg.orientation.z),
# #         "orientation_w": float(msg.orientation.w),

# #         "angular_velocity_x": float(msg.angular_velocity.x),
# #         "angular_velocity_y": float(msg.angular_velocity.y),
# #         "angular_velocity_z": float(msg.angular_velocity.z),

# #         "linear_acceleration_x": float(msg.linear_acceleration.x),
# #         "linear_acceleration_y": float(msg.linear_acceleration.y),
# #         "linear_acceleration_z": float(msg.linear_acceleration.z),
# #     }


# # def update_agents_from_dynamic_pose(msg, agents: dict[str, dict[str, Any]]) -> None:
# #     for transform in msg.transforms:
# #         child = transform.child_frame_id or ""
# #         parts = [part for part in child.split("/") if part]

# #         matched_agent = None

# #         for agent_name in AGENT_ORDER:
# #             if agent_name in parts:
# #                 matched_agent = agent_name
# #                 break

# #         if matched_agent is None:
# #             continue

# #         new_state = pose_from_tf_transform(transform)
# #         old_state = agents[matched_agent].get("state")

# #         if old_state is not None:
# #             new_state["vx"] = safe_float(old_state.get("vx", 0.0))
# #             new_state["vy"] = safe_float(old_state.get("vy", 0.0))
# #             new_state["vz"] = safe_float(old_state.get("vz", 0.0))
# #             new_state["wz"] = safe_float(old_state.get("wz", 0.0))

# #         agents[matched_agent]["state"] = new_state
# #         agents[matched_agent]["available"] = True


# # def decode_pointcloud2_to_xyz_i(msg) -> np.ndarray:
# #     """Decode PointCloud2 into an Nx4 [x,y,z,intensity] float32 array.

# #     This is intentionally minimal and works for common Gazebo pointcloud fields.
# #     Missing intensity is filled with zero.
# #     """
# #     raw = np.frombuffer(msg.data, dtype=np.uint8)
# #     point_step = int(msg.point_step)

# #     if point_step <= 0:
# #         return np.zeros((0, 4), dtype=np.float32)

# #     count = int(len(raw) / point_step)

# #     if count == 0:
# #         return np.zeros((0, 4), dtype=np.float32)

# #     field_offsets = {field.name: int(field.offset) for field in msg.fields}

# #     points = np.zeros((count, 4), dtype=np.float32)

# #     for i in range(count):
# #         base = i * point_step

# #         for col, name in enumerate(("x", "y", "z", "intensity")):
# #             offset = field_offsets.get(name)

# #             if offset is None:
# #                 continue

# #             if base + offset + 4 > len(raw):
# #                 continue

# #             points[i, col] = np.frombuffer(
# #                 raw[base + offset : base + offset + 4],
# #                 dtype=np.float32,
# #             )[0]

# #     points = np.nan_to_num(
# #         points,
# #         nan=0.0,
# #         posinf=0.0,
# #         neginf=0.0,
# #     ).astype(np.float32)

# #     return points


# # def save_laserscan(msg, out_path: Path) -> dict[str, Any]:
# #     ranges = np.asarray(msg.ranges, dtype=np.float32)

# #     if len(msg.intensities) > 0:
# #         intensities = np.asarray(msg.intensities, dtype=np.float32)
# #     else:
# #         intensities = np.zeros(len(ranges), dtype=np.float32)

# #     ranges = np.nan_to_num(
# #         ranges,
# #         nan=float(msg.range_max) if msg.range_max > 0 else 999.0,
# #         posinf=float(msg.range_max) if msg.range_max > 0 else 999.0,
# #         neginf=0.0,
# #     )

# #     intensities = np.nan_to_num(
# #         intensities,
# #         nan=0.0,
# #         posinf=0.0,
# #         neginf=0.0,
# #     )

# #     scan = np.stack([ranges, intensities], axis=1).astype(np.float32)

# #     out_path.parent.mkdir(parents=True, exist_ok=True)
# #     np.save(out_path, scan)

# #     return {
# #         "path": str(out_path),
# #         "timestamp_ns": None,
# #         "modality": "planar_scan",
# #         "shape": list(scan.shape),
# #         "dtype": str(scan.dtype),
# #         "angle_min": safe_float(msg.angle_min),
# #         "angle_max": safe_float(msg.angle_max),
# #         "angle_increment": safe_float(msg.angle_increment),
# #         "range_min": safe_float(msg.range_min),
# #         "range_max": safe_float(msg.range_max),
# #     }


# # def save_pointcloud(msg, out_path: Path) -> dict[str, Any]:
# #     points = decode_pointcloud2_to_xyz_i(msg)

# #     out_path.parent.mkdir(parents=True, exist_ok=True)
# #     np.save(out_path, points)

# #     return {
# #         "path": str(out_path),
# #         "timestamp_ns": None,
# #         "modality": "pointcloud_xyz_i",
# #         "shape": list(points.shape),
# #         "dtype": str(points.dtype),
# #     }


# # def make_asset_dir(root: Path, group: str) -> Path:
# #     path = root / "assets" / group
# #     path.mkdir(parents=True, exist_ok=True)

# #     return path


# # def asset_ref(latest_assets: dict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
# #     ref = latest_assets.get(key)

# #     if ref is None:
# #         return None

# #     return dict(ref)


# # def relative_goal_features(
# #     state: dict[str, Any] | None,
# #     goal: dict[str, Any] | None,
# # ) -> dict[str, float] | None:
# #     if state is None or goal is None:
# #         return None

# #     dx = float(goal["x"] - state["x"])
# #     dy = float(goal["y"] - state["y"])
# #     dz = float(goal["z"] - state["z"])

# #     distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
# #     goal_heading = math.atan2(dy, dx)
# #     heading_error = wrap_angle(goal_heading - float(state.get("yaw", 0.0)))

# #     return {
# #         "dx": dx,
# #         "dy": dy,
# #         "dz": dz,
# #         "distance_to_goal": distance,
# #         "goal_heading": float(goal_heading),
# #         "heading_error": float(heading_error),
# #     }


# # def platform_onehot(agent_id: str) -> list[float]:
# #     return PLATFORM_ONEHOT.get(PLATFORM_TYPE.get(agent_id, "UGV"), [0.0, 0.0])


# # def build_agent_node(
# #     agent_id: str,
# #     agent: dict[str, Any],
# #     ego_state: dict[str, Any],
# # ) -> dict[str, Any]:
# #     state = agent.get("state")
# #     goal = agent.get("goal")
# #     command = agent.get("command") or DEFAULT_COMMAND

# #     available = bool(agent.get("available", False)) and state is not None

# #     if state is None:
# #         state = {
# #             "x": ego_state["x"],
# #             "y": ego_state["y"],
# #             "z": ego_state["z"],
# #             "yaw": 0.0,
# #             "vx": 0.0,
# #             "vy": 0.0,
# #             "vz": 0.0,
# #             "wz": 0.0,
# #             "qx": 0.0,
# #             "qy": 0.0,
# #             "qz": 0.0,
# #             "qw": 1.0,
# #         }

# #     if goal is None:
# #         goal = {
# #             "x": state["x"],
# #             "y": state["y"],
# #             "z": state["z"],
# #         }

# #     p_onehot = platform_onehot(agent_id)

# #     features = [
# #         float(state["x"] - ego_state["x"]),
# #         float(state["y"] - ego_state["y"]),
# #         float(state["z"] - ego_state["z"]),

# #         float(state.get("vx", 0.0)),
# #         float(state.get("vy", 0.0)),
# #         float(state.get("vz", 0.0)),
# #         float(state.get("wz", 0.0)),

# #         float(goal["x"] - state["x"]),
# #         float(goal["y"] - state["y"]),
# #         float(goal["z"] - state["z"]),

# #         float(command.get("linear_x", 0.0)),
# #         float(command.get("angular_z", 0.0)),

# #         float(p_onehot[0]),
# #         float(p_onehot[1]),
# #     ]

# #     return {
# #         "id": agent_id,
# #         "platform_type": PLATFORM_TYPE.get(agent_id, "unknown"),
# #         "available": available,
# #         "state": state,
# #         "goal": goal,
# #         "goal_features": relative_goal_features(state, goal),
# #         "command": command,
# #         "imu": agent.get("imu"),
# #         "feature": features,
# #     }


# # def build_graph_edges(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
# #     edges = []

# #     for src_id, src in nodes.items():
# #         for dst_id, dst in nodes.items():
# #             if src_id == dst_id:
# #                 continue

# #             src_state = src["state"]
# #             dst_state = dst["state"]

# #             dx = float(dst_state["x"] - src_state["x"])
# #             dy = float(dst_state["y"] - src_state["y"])
# #             dz = float(dst_state["z"] - src_state["z"])

# #             distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
# #             inv_distance = 1.0 / max(distance, 1e-3)
# #             bearing = math.atan2(dy, dx)

# #             same_platform = (
# #                 1.0
# #                 if src.get("platform_type") == dst.get("platform_type")
# #                 else 0.0
# #             )

# #             edges.append(
# #                 {
# #                     "source": src_id,
# #                     "target": dst_id,

# #                     "dx": dx,
# #                     "dy": dy,
# #                     "dz": dz,
# #                     "distance": distance,
# #                     "inv_distance": float(inv_distance),
# #                     "bearing_sin": float(math.sin(bearing)),
# #                     "bearing_cos": float(math.cos(bearing)),
# #                     "same_platform": same_platform,

# #                     # Extra edge context for communication-aware experiments.
# #                     "network": dict(DEFAULT_NETWORK_STATE),
# #                 }
# #             )

# #     return edges


# # def summarize_forward_pointcloud(
# #     points_path: str | None,
# #     *,
# #     x_min: float = 0.0,
# #     x_max: float = 25.0,
# #     center_half_width: float = 2.0,
# #     side_width: float = 8.0,
# #     z_min: float = -5.0,
# #     z_max: float = 5.0,
# # ) -> dict[str, Any]:
# #     """Create a compact left/center/right hazard summary from a pointcloud asset.

# #     This is used as simple UAV-derived contextual information. It does not
# #     replace the raw pointcloud .npy file; it only gives the models an explicit
# #     compact feature as well.
# #     """
# #     if not points_path:
# #         return {
# #             "available": False,
# #             "left_count": 0,
# #             "center_count": 0,
# #             "right_count": 0,
# #             "total_count": 0,
# #             "nearest_x": 999.0,
# #         }

# #     path = Path(points_path)

# #     if not path.exists():
# #         return {
# #             "available": False,
# #             "left_count": 0,
# #             "center_count": 0,
# #             "right_count": 0,
# #             "total_count": 0,
# #             "nearest_x": 999.0,
# #         }

# #     try:
# #         points = np.load(path)
# #     except Exception:
# #         return {
# #             "available": False,
# #             "left_count": 0,
# #             "center_count": 0,
# #             "right_count": 0,
# #             "total_count": 0,
# #             "nearest_x": 999.0,
# #         }

# #     if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
# #         return {
# #             "available": True,
# #             "left_count": 0,
# #             "center_count": 0,
# #             "right_count": 0,
# #             "total_count": 0,
# #             "nearest_x": 999.0,
# #         }

# #     x = points[:, 0]
# #     y = points[:, 1]
# #     z = points[:, 2]

# #     valid = (
# #         np.isfinite(x)
# #         & np.isfinite(y)
# #         & np.isfinite(z)
# #         & (x >= x_min)
# #         & (x <= x_max)
# #         & (z >= z_min)
# #         & (z <= z_max)
# #         & (np.abs(y) <= side_width)
# #     )

# #     if not np.any(valid):
# #         return {
# #             "available": True,
# #             "left_count": 0,
# #             "center_count": 0,
# #             "right_count": 0,
# #             "total_count": 0,
# #             "nearest_x": 999.0,
# #         }

# #     xv = x[valid]
# #     yv = y[valid]

# #     center = np.abs(yv) <= center_half_width
# #     left = yv > center_half_width
# #     right = yv < -center_half_width

# #     return {
# #         "available": True,
# #         "left_count": int(np.sum(left)),
# #         "center_count": int(np.sum(center)),
# #         "right_count": int(np.sum(right)),
# #         "total_count": int(len(xv)),
# #         "nearest_x": float(np.min(xv)) if len(xv) else 999.0,
# #     }


# # def teacher_label(agent: dict[str, Any]) -> str:
# #     state = agent.get("controller_state")

# #     if state:
# #         return str(state)

# #     action = str(agent.get("obstacle_action") or "clear").lower()

# #     if action.endswith("left"):
# #         return "avoid_left"

# #     if action.endswith("right"):
# #         return "avoid_right"

# #     return "go_to_goal"


# # def build_frame(
# #     *,
# #     episode_id: str,
# #     timestamp_ns: int,
# #     world_name: str,
# #     ego_id: str,
# #     agents: dict[str, dict[str, Any]],
# #     latest_assets: dict[str, dict[str, Any]],
# # ) -> dict[str, Any] | None:
# #     ego = agents[ego_id]

# #     if ego.get("state") is None or ego.get("goal") is None:
# #         return None

# #     ego_state = ego["state"]

# #     nodes = {
# #         agent_id: build_agent_node(agent_id, agents[agent_id], ego_state)
# #         for agent_id in AGENT_ORDER
# #     }

# #     edges = build_graph_edges(nodes)

# #     ego_planar_scan = asset_ref(latest_assets, f"{ego_id}_planar_scan")
# #     ego_front_pointcloud = asset_ref(latest_assets, f"{ego_id}_front_points")
# #     uav1_front_pointcloud = asset_ref(latest_assets, "uav1_front_points")

# #     uav_hazard_summary = summarize_forward_pointcloud(
# #         uav1_front_pointcloud["path"] if uav1_front_pointcloud else None
# #     )

# #     label = teacher_label(ego)

# #     frame = {
# #         "episode_id": episode_id,
# #         "timestamp_ns": int(timestamp_ns),
# #         "world_name": world_name,
# #         "ego_id": ego_id,

# #         # Backward-compatible single-agent fields.
# #         "state": ego_state,
# #         "goal": ego.get("goal"),
# #         "goal_features": relative_goal_features(ego_state, ego.get("goal")),

# #         # Sensor references.
# #         "observation": {
# #             "ego_planar_scan": ego_planar_scan,
# #             "ego_front_pointcloud": ego_front_pointcloud,
# #             "uav1_front_pointcloud": uav1_front_pointcloud,
# #             "uav1_hazard_summary": uav_hazard_summary,
# #         },

# #         # Same content under 'modalities' because some notebooks use this key.
# #         "modalities": {
# #             "ego_planar_scan": ego_planar_scan,
# #             "ego_front_pointcloud": ego_front_pointcloud,
# #             "uav1_front_pointcloud": uav1_front_pointcloud,
# #             "uav1_hazard_summary": uav_hazard_summary,
# #         },

# #         # Teacher / label information.
# #         "teacher": {
# #             "label": label,
# #             "command": ego.get("command") or DEFAULT_COMMAND,
# #             "controller_state": ego.get("controller_state"),
# #             "obstacle_action": ego.get("obstacle_action"),
# #             "obstacle_clearance": ego.get("obstacle_clearance"),
# #         },

# #         # Graph representation.
# #         "agents": nodes,
# #         "edges": edges,

# #         # Convenience compatibility block.
# #         "other_husky": {
# #             "id": "husky_2" if ego_id == "husky_local" else "husky_local",
# #             "available": bool(nodes["husky_2" if ego_id == "husky_local" else "husky_local"]["available"]),
# #             "state": nodes["husky_2" if ego_id == "husky_local" else "husky_local"]["state"],
# #             "goal": nodes["husky_2" if ego_id == "husky_local" else "husky_local"]["goal"],
# #             "goal_features": nodes["husky_2" if ego_id == "husky_local" else "husky_local"]["goal_features"],
# #             "teacher_command": agents["husky_2" if ego_id == "husky_local" else "husky_local"].get("command"),
# #         },

# #         "uav_context": {
# #             "id": "uav1",
# #             "available": bool(nodes["uav1"]["available"]),
# #             "state": nodes["uav1"]["state"],
# #             "goal": nodes["uav1"]["goal"],
# #             "goal_features": nodes["uav1"]["goal_features"],
# #             "hazard_summary": uav_hazard_summary,
# #         },

# #         # Communication placeholder. If OMNeT++ features are later recorded,
# #         # this can be replaced with real per-frame values.
# #         "network_state": dict(DEFAULT_NETWORK_STATE),

# #         "readiness": {
# #             "has_scan": ego_planar_scan is not None,
# #             "has_state": ego.get("state") is not None,
# #             "has_goal": ego.get("goal") is not None,
# #             "has_teacher_command": ego.get("command") is not None,

# #             "has_husky_2_state": agents["husky_2"].get("state") is not None,
# #             "has_uav1_state": agents["uav1"].get("state") is not None,
# #             "has_uav1_pointcloud": uav1_front_pointcloud is not None,

# #             "has_graph_nodes": True,
# #             "has_graph_edges": len(edges) > 0,
# #         },
# #     }

# #     return frame


# # def schema() -> dict[str, Any]:
# #     return {
# #         "description": (
# #             "Hybrid UAV-UGV trajectory prediction dataset. Each frame is anchored "
# #             "on a Husky planar lidar scan and includes ego state, goal features, "
# #             "local perception, UAV context, graph nodes/edges, and teacher labels."
# #         ),
# #         "frame_anchor": "Husky planar lidar scan timestamp.",
# #         "prediction_task": {
# #             "input": "past 10 frames",
# #             "target": "future 20-step ego trajectory",
# #             "target_units": "relative future (x, y) positions in meters from anchor frame",
# #         },
# #         "models_supported": [
# #             "CV",
# #             "CNN-LSTM",
# #             "GNN-LSTM",
# #             "CNN-GNN-LSTM",
# #         ],
# #         "primary_inputs": [
# #             "state",
# #             "goal_features",
# #             "observation.ego_planar_scan",
# #             "observation.ego_front_pointcloud",
# #             "observation.uav1_front_pointcloud",
# #             "observation.uav1_hazard_summary",
# #             "agents",
# #             "edges",
# #             "network_state",
# #         ],
# #         "primary_targets": [
# #             "future ego trajectory generated by sliding windows",
# #             "teacher.command.linear_x",
# #             "teacher.command.angular_z",
# #             "teacher.label",
# #         ],
# #         "node_feature_dim": 14,
# #         "node_feature_fields": [
# #             "rel_x_to_ego",
# #             "rel_y_to_ego",
# #             "rel_z_to_ego",
# #             "vx",
# #             "vy",
# #             "vz",
# #             "wz",
# #             "goal_dx",
# #             "goal_dy",
# #             "goal_dz",
# #             "command_linear_x",
# #             "command_angular_z",
# #             "is_ugv",
# #             "is_uav",
# #         ],
# #         "edge_feature_dim": 8,
# #         "edge_feature_fields": [
# #             "dx",
# #             "dy",
# #             "dz",
# #             "distance",
# #             "inv_distance",
# #             "bearing_sin",
# #             "bearing_cos",
# #             "same_platform",
# #         ],
# #         "agents": AGENT_ORDER,
# #         "platform_types": PLATFORM_TYPE,
# #     }


# # def init_agents() -> dict[str, dict[str, Any]]:
# #     return {
# #         agent_id: {
# #             "state": None,
# #             "start": None,
# #             "goal": None,
# #             "command": dict(DEFAULT_COMMAND),
# #             "controller_state": None,
# #             "obstacle_action": "clear",
# #             "obstacle_clearance": None,
# #             "imu": None,
# #             "available": False,
# #         }
# #         for agent_id in AGENT_ORDER
# #     }


# # def export_one_bag(bag_path: Path, out_root: Path) -> tuple[Path, int]:
# #     db3_path = bag_db3_path(bag_path)

# #     out_dir = out_root / bag_path.name
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     frames_path = out_dir / "frames.jsonl"
# #     schema_path = out_dir / "schema.json"
# #     manifest_path = out_dir / "manifest.json"

# #     if frames_path.exists():
# #         frames_path.unlink()

# #     typestore = get_typestore(Stores.ROS2_HUMBLE)

# #     conn = sqlite3.connect(str(db3_path))
# #     cur = conn.cursor()

# #     topic_rows = list(cur.execute("SELECT id, name, type FROM topics ORDER BY id"))
# #     topic_map = {
# #         topic_id: (name, msgtype)
# #         for topic_id, name, msgtype in topic_rows
# #     }

# #     topic_names = [name for _, name, _ in topic_rows]
# #     world_name = infer_world_name(topic_names)
# #     topics = build_topics(world_name)

# #     topic_name_set = set(topic_names)

# #     agents = init_agents()
# #     latest_assets: dict[str, dict[str, Any]] = {}

# #     asset_dirs = {
# #         "husky_local_planar_scan": make_asset_dir(out_dir, "husky_local/planar_scan"),
# #         "husky_2_planar_scan": make_asset_dir(out_dir, "husky_2/planar_scan"),

# #         "husky_local_front_points": make_asset_dir(out_dir, "husky_local/front_points"),
# #         "husky_2_front_points": make_asset_dir(out_dir, "husky_2/front_points"),
# #         "uav1_front_points": make_asset_dir(out_dir, "uav1/front_points"),
# #     }

# #     topic_to_asset_key = {
# #         topics["husky_local_planar_scan"]: "husky_local_planar_scan",
# #         topics["husky_2_planar_scan"]: "husky_2_planar_scan",

# #         topics["husky_local_front_points"]: "husky_local_front_points",
# #         topics["husky_2_front_points"]: "husky_2_front_points",
# #         topics["uav1_front_points"]: "uav1_front_points",
# #     }

# #     anchor_topics = {
# #         topics["husky_local_planar_scan"]: "husky_local",
# #         topics["husky_2_planar_scan"]: "husky_2",
# #     }

# #     frame_count = 0
# #     topic_message_counts: dict[str, int] = {}

# #     query = "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"

# #     for topic_id, timestamp, rawdata in cur.execute(query):
# #         topic, msgtype = topic_map[topic_id]
# #         topic_message_counts[topic] = topic_message_counts.get(topic, 0) + 1

# #         try:
# #             msg = typestore.deserialize_cdr(rawdata, msgtype)
# #         except Exception as exc:
# #             print(f"Warning: failed to deserialize topic={topic} type={msgtype}: {exc}")
# #             continue

# #         ts = int(timestamp)

# #         if topic == topics["dynamic_pose"]:
# #             update_agents_from_dynamic_pose(msg, agents)

# #         elif topic == topics["husky_local_odom"]:
# #             agents["husky_local"]["state"] = pose_from_odom(msg)
# #             agents["husky_local"]["available"] = True

# #         elif topic == topics["husky_2_odom"]:
# #             agents["husky_2"]["state"] = pose_from_odom(msg)
# #             agents["husky_2"]["available"] = True

# #         elif topic == topics["uav1_odom"]:
# #             agents["uav1"]["state"] = pose_from_odom(msg)
# #             agents["uav1"]["available"] = True

# #         elif topic == topics["cmd_husky_local"]:
# #             agents["husky_local"]["command"] = cmd_from_twist(msg)

# #         elif topic == topics["cmd_husky_2"]:
# #             agents["husky_2"]["command"] = cmd_from_twist(msg)

# #         elif topic == topics["state_husky_local"]:
# #             agents["husky_local"]["controller_state"] = str(msg.data)

# #         elif topic == topics["state_husky_2"]:
# #             agents["husky_2"]["controller_state"] = str(msg.data)

# #         elif topic == topics["obstacle_action_husky_local"]:
# #             agents["husky_local"]["obstacle_action"] = str(msg.data)

# #         elif topic == topics["obstacle_action_husky_2"]:
# #             agents["husky_2"]["obstacle_action"] = str(msg.data)

# #         elif topic == topics["obstacle_clearance_husky_local"]:
# #             agents["husky_local"]["obstacle_clearance"] = clearance_from_vector3(msg)

# #         elif topic == topics["obstacle_clearance_husky_2"]:
# #             agents["husky_2"]["obstacle_clearance"] = clearance_from_vector3(msg)

# #         elif topic == topics["husky_local_start"]:
# #             agents["husky_local"]["start"] = pose_from_pose_stamped(msg)

# #         elif topic == topics["husky_local_goal"]:
# #             agents["husky_local"]["goal"] = pose_from_pose_stamped(msg)

# #         elif topic == topics["husky_2_start"]:
# #             agents["husky_2"]["start"] = pose_from_pose_stamped(msg)

# #         elif topic == topics["husky_2_goal"]:
# #             agents["husky_2"]["goal"] = pose_from_pose_stamped(msg)

# #         elif topic == topics["uav1_start"]:
# #             agents["uav1"]["start"] = pose_from_pose_stamped(msg)

# #         elif topic == topics["uav1_goal"]:
# #             agents["uav1"]["goal"] = pose_from_pose_stamped(msg)

# #         elif topic == topics["husky_local_imu"]:
# #             agents["husky_local"]["imu"] = imu_from_msg(msg)

# #         elif topic == topics["husky_2_imu"]:
# #             agents["husky_2"]["imu"] = imu_from_msg(msg)

# #         elif topic == topics["uav1_imu"]:
# #             agents["uav1"]["imu"] = imu_from_msg(msg)

# #         # Save heavy sensor assets as .npy and reference them from JSONL.
# #         if topic in topic_to_asset_key:
# #             asset_key = topic_to_asset_key[topic]
# #             asset_dir = asset_dirs[asset_key]
# #             out_path = asset_dir / f"{ts}.npy"

# #             if asset_key.endswith("planar_scan"):
# #                 meta = save_laserscan(msg, out_path)
# #             else:
# #                 meta = save_pointcloud(msg, out_path)

# #             meta["timestamp_ns"] = ts
# #             latest_assets[asset_key] = meta

# #         # Anchor frame after the current message has updated latest assets.
# #         if topic in anchor_topics:
# #             ego_id = anchor_topics[topic]

# #             frame = build_frame(
# #                 episode_id=bag_path.name,
# #                 timestamp_ns=ts,
# #                 world_name=world_name,
# #                 ego_id=ego_id,
# #                 agents=agents,
# #                 latest_assets=latest_assets,
# #             )

# #             if frame is None:
# #                 continue

# #             with frames_path.open("a", encoding="utf-8") as f:
# #                 f.write(json.dumps(frame) + "\n")

# #             frame_count += 1

# #     conn.close()

# #     schema_path.write_text(json.dumps(schema(), indent=2), encoding="utf-8")

# #     manifest = {
# #         "bag_path": str(bag_path),
# #         "db3_path": str(db3_path),
# #         "out_dir": str(out_dir),
# #         "world_name": world_name,
# #         "frame_count": frame_count,
# #         "topics_present": sorted(topic_name_set),
# #         "expected_topics": topics,
# #         "missing_expected_topics": sorted(
# #             [
# #                 topic
# #                 for topic in topics.values()
# #                 if topic not in topic_name_set
# #             ]
# #         ),
# #         "topic_message_counts": topic_message_counts,
# #         "notes": [
# #             "Frames are anchored on Husky planar scan topics.",
# #             "Future trajectory targets are built later by sliding windows in dataset_helper/notebooks.",
# #             "UAV/context data is included when the source bag recorded those topics.",
# #         ],
# #     }

# #     manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# #     print("Saved hybrid dataset to:", out_dir)
# #     print("World:", world_name)
# #     print("Frames:", frame_count)
# #     print("Schema:", schema_path)
# #     print("Manifest:", manifest_path)

# #     if frame_count == 0:
# #         print(
# #             "WARNING: Exported 0 frames. This usually means the bag does not contain "
# #             "Husky planar scan topics, odometry, or episode goal metadata."
# #         )

# #     return out_dir, frame_count


# # def main() -> None:
# #     parser = argparse.ArgumentParser(
# #         description="Export hybrid UAV-UGV JSONL dataset from recorded ROS 2 bag(s)."
# #     )

# #     parser.add_argument(
# #         "--bag",
# #         default="",
# #         help="Optional path to a specific bag directory. If omitted, all run_* bags are exported.",
# #     )

# #     parser.add_argument(
# #         "--out-root",
# #         default=str(OUT_ROOT),
# #         help="Dataset output root.",
# #     )

# #     args = parser.parse_args()

# #     out_root = Path(args.out_root).expanduser().resolve()

# #     if args.bag:
# #         bag_paths = [Path(args.bag).expanduser().resolve()]
# #     else:
# #         bag_paths = available_bags(BAGS_DIR)

# #         if not bag_paths:
# #             raise FileNotFoundError(
# #                 f"No bag directories found in {BAGS_DIR}. "
# #                 "First run the simulation and record at least one bag."
# #             )

# #     print("Export root:", out_root)
# #     print("Bag count:", len(bag_paths))

# #     total_frames = 0

# #     for bag_path in bag_paths:
# #         print("\n=== Exporting", bag_path.name, "===")
# #         _out_dir, frame_count = export_one_bag(bag_path, out_root)
# #         total_frames += int(frame_count)

# #     print("\nCompleted hybrid dataset export.")
# #     print("Total bags:", len(bag_paths))
# #     print("Total frames:", total_frames)
# #     print("Output root:", out_root)


# # if __name__ == "__main__":
# #     main()



# #!/usr/bin/env python3
# """Export a hybrid two-UAV / one-UGV maneuver dataset from recorded ROS 2 bags.

# This exporter converts rosbag2 SQLite bags into a JSONL + NPY dataset suitable
# for the thesis hybrid models:

# - CV baseline
# - CNN-LSTM / temporal CNN-LSTM baseline
# - GNN-LSTM
# - CNN-GNN-LSTM

# Each JSONL row is one synchronized frame anchored on the ego Husky planar lidar scan.

# Current intended agent structure:
# - husky_local: ego UGV whose future trajectory is predicted
# - uav1: left-side UAV context / aerial perception
# - uav2: right-side UAV context / aerial perception

# Important dataset idea:
# The model observes past frames and learns to predict the ego Husky's future
# 20-step trajectory. Each exported frame contains:
# - ego Husky state,
# - ego local lidar/pointcloud references,
# - ego goal and goal-relative features,
# - teacher command/state labels from the rule-based controller,
# - UAV1 and UAV2 context,
# - UAV1 and UAV2 pointcloud references,
# - compact UAV hazard summaries,
# - graph nodes and graph edges,
# - placeholder communication/network features.

# The exporter does NOT train the model. It only prepares the dataset.
# """

# from __future__ import annotations

# import argparse
# import json
# import math
# import re
# import sqlite3
# import sys
# from pathlib import Path
# from typing import Any

# import numpy as np


# try:
#     from rosbags.typesys import Stores, get_typestore
# except ModuleNotFoundError:
#     print(
#         "\nERROR: Python package 'rosbags' is not installed in this conda environment.\n\n"
#         "Run this first:\n"
#         "  conda activate maneuver\n"
#         "  pip install rosbags\n\n"
#         "Then run the exporter again:\n"
#         "  python 03_dataset/exporters/export_hybrid_maneuver_dataset.py\n",
#         file=sys.stderr,
#     )
#     raise


# THESIS_ROOT = Path.home() / "Documents/Thesis"
# DATASET_ROOT = THESIS_ROOT / "03_dataset"
# BAGS_DIR = DATASET_ROOT / "bags"
# OUT_ROOT = DATASET_ROOT / "husky_control_dataset"

# WORLD_TOPIC_RE = re.compile(r"^/world/([^/]+)/")

# # New project structure: one ego Husky + two UAV context agents.
# AGENT_ORDER = ["husky_local", "uav1", "uav2"]

# PLATFORM_TYPE = {
#     "husky_local": "UGV",
#     "uav1": "UAV",
#     "uav2": "UAV",
# }

# PLATFORM_ONEHOT = {
#     "UGV": [1.0, 0.0],
#     "UAV": [0.0, 1.0],
# }

# DEFAULT_COMMAND = {
#     "linear_x": 0.0,
#     "angular_z": 0.0,
# }

# DEFAULT_NETWORK_STATE = {
#     "latency_s": 0.0,
#     "jitter_s": 0.0,
#     "packet_loss": 0.0,
#     "link_quality": 1.0,
#     "connected": True,
#     "source": "default_no_omnet",
# }


# def bag_db3_path(bag_dir: Path) -> Path:
#     db3_files = sorted(bag_dir.glob("*.db3"))

#     if not db3_files:
#         raise RuntimeError(f"No .db3 file found in {bag_dir}")

#     return db3_files[0]


# def available_bags(root: Path) -> list[Path]:
#     if not root.exists():
#         return []

#     prefixes = (
#         "run_",
#         "run_model_",
#         "run_dataset_",
#         "open_hazard_teacher_",
#     )

#     return sorted(
#         [
#             p.resolve()
#             for p in root.iterdir()
#             if p.is_dir() and p.name.startswith(prefixes)
#         ]
#     )


# def infer_world_name(topic_names: list[str]) -> str:
#     for topic in topic_names:
#         match = WORLD_TOPIC_RE.match(topic)

#         if match:
#             return match.group(1)

#     raise RuntimeError("Could not infer Gazebo world name from bag topics.")


# def build_topics(world_name: str) -> dict[str, str]:
#     """Build all expected topic names for the one-Husky/two-UAV dataset."""
#     return {
#         "dynamic_pose": f"/world/{world_name}/dynamic_pose/info",

#         # Ego Husky.
#         "husky_local_odom": "/model/husky_local/odometry",
#         "cmd_husky_local": "/cmd_vel",
#         "state_husky_local": "/husky_local/controller_state",
#         "obstacle_action_husky_local": "/husky_local/obstacle_action",
#         "obstacle_clearance_husky_local": "/husky_local/obstacle_clearance",
#         "husky_local_start": "/episode/husky_local/start",
#         "husky_local_goal": "/episode/husky_local/goal",

#         # UAV 1.
#         "uav1_odom": "/model/uav1/odometry",
#         "uav1_ready": "/uav1/ready",
#         "uav1_start": "/episode/uav1/start",
#         "uav1_goal": "/episode/uav1/goal",
#         "cmd_uav1_model": "/model/uav1/command/twist",
#         "cmd_uav1_direct": "/uav1/command/twist",

#         # UAV 2.
#         "uav2_odom": "/model/uav2/odometry",
#         "uav2_ready": "/uav2/ready",
#         "uav2_start": "/episode/uav2/start",
#         "uav2_goal": "/episode/uav2/goal",
#         "cmd_uav2_model": "/model/uav2/command/twist",
#         "cmd_uav2_direct": "/uav2/command/twist",

#         # Ego Husky sensors.
#         "husky_local_planar_scan": (
#             f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan"
#         ),
#         "husky_local_front_points": (
#             f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points"
#         ),
#         "husky_local_imu": (
#             f"/world/{world_name}/model/husky_local/link/base_link/sensor/imu_sensor/imu"
#         ),

#         # UAV sensors.
#         "uav1_front_points": (
#             f"/world/{world_name}/model/uav1/link/base_link/sensor/front_laser/scan/points"
#         ),
#         "uav1_imu": (
#             f"/world/{world_name}/model/uav1/link/base_link/sensor/imu_sensor/imu"
#         ),

#         "uav2_front_points": (
#             f"/world/{world_name}/model/uav2/link/base_link/sensor/front_laser/scan/points"
#         ),
#         "uav2_imu": (
#             f"/world/{world_name}/model/uav2/link/base_link/sensor/imu_sensor/imu"
#         ),
#     }


# def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
#     siny_cosp = 2.0 * (w * z + x * y)
#     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

#     return math.atan2(siny_cosp, cosy_cosp)


# def wrap_angle(angle: float) -> float:
#     return math.atan2(math.sin(angle), math.cos(angle))


# def safe_float(value: Any, default: float = 0.0) -> float:
#     try:
#         return float(value)
#     except Exception:
#         return float(default)


# def pose_from_odom(msg) -> dict[str, float]:
#     pose = msg.pose.pose
#     twist = msg.twist.twist

#     yaw = quaternion_to_yaw(
#         pose.orientation.x,
#         pose.orientation.y,
#         pose.orientation.z,
#         pose.orientation.w,
#     )

#     return {
#         "x": float(pose.position.x),
#         "y": float(pose.position.y),
#         "z": float(pose.position.z),

#         "qx": float(pose.orientation.x),
#         "qy": float(pose.orientation.y),
#         "qz": float(pose.orientation.z),
#         "qw": float(pose.orientation.w),

#         "yaw": float(yaw),

#         "vx": float(twist.linear.x),
#         "vy": float(twist.linear.y),
#         "vz": float(twist.linear.z),
#         "wz": float(twist.angular.z),
#     }


# def pose_from_tf_transform(transform) -> dict[str, float]:
#     translation = transform.transform.translation
#     rotation = transform.transform.rotation

#     yaw = quaternion_to_yaw(
#         rotation.x,
#         rotation.y,
#         rotation.z,
#         rotation.w,
#     )

#     return {
#         "x": float(translation.x),
#         "y": float(translation.y),
#         "z": float(translation.z),

#         "qx": float(rotation.x),
#         "qy": float(rotation.y),
#         "qz": float(rotation.z),
#         "qw": float(rotation.w),

#         "yaw": float(yaw),

#         "vx": 0.0,
#         "vy": 0.0,
#         "vz": 0.0,
#         "wz": 0.0,
#     }


# def pose_from_pose_stamped(msg) -> dict[str, float]:
#     pose = msg.pose

#     return {
#         "x": float(pose.position.x),
#         "y": float(pose.position.y),
#         "z": float(pose.position.z),
#     }


# def cmd_from_twist(msg) -> dict[str, float]:
#     return {
#         "linear_x": float(msg.linear.x),
#         "linear_y": float(msg.linear.y),
#         "linear_z": float(msg.linear.z),
#         "angular_z": float(msg.angular.z),
#     }


# def clearance_from_vector3(msg) -> dict[str, float]:
#     return {
#         "front": float(msg.x),
#         "left": float(msg.y),
#         "right": float(msg.z),
#     }


# def imu_from_msg(msg) -> dict[str, float]:
#     return {
#         "orientation_x": float(msg.orientation.x),
#         "orientation_y": float(msg.orientation.y),
#         "orientation_z": float(msg.orientation.z),
#         "orientation_w": float(msg.orientation.w),

#         "angular_velocity_x": float(msg.angular_velocity.x),
#         "angular_velocity_y": float(msg.angular_velocity.y),
#         "angular_velocity_z": float(msg.angular_velocity.z),

#         "linear_acceleration_x": float(msg.linear_acceleration.x),
#         "linear_acceleration_y": float(msg.linear_acceleration.y),
#         "linear_acceleration_z": float(msg.linear_acceleration.z),
#     }


# def update_agents_from_dynamic_pose(
#     msg,
#     agents: dict[str, dict[str, Any]],
# ) -> None:
#     for transform in msg.transforms:
#         child = transform.child_frame_id or ""
#         parts = [part for part in child.split("/") if part]

#         matched_agent = None

#         for agent_name in AGENT_ORDER:
#             if agent_name in parts:
#                 matched_agent = agent_name
#                 break

#         if matched_agent is None:
#             continue

#         new_state = pose_from_tf_transform(transform)
#         old_state = agents[matched_agent].get("state")

#         if old_state is not None:
#             new_state["vx"] = safe_float(old_state.get("vx", 0.0))
#             new_state["vy"] = safe_float(old_state.get("vy", 0.0))
#             new_state["vz"] = safe_float(old_state.get("vz", 0.0))
#             new_state["wz"] = safe_float(old_state.get("wz", 0.0))

#         agents[matched_agent]["state"] = new_state
#         agents[matched_agent]["available"] = True


# def decode_pointcloud2_to_xyz_i(msg) -> np.ndarray:
#     """Decode PointCloud2 into an Nx4 [x,y,z,intensity] float32 array.

#     This is intentionally minimal and works for common Gazebo pointcloud fields.
#     Missing intensity is filled with zero.
#     """
#     raw = np.frombuffer(msg.data, dtype=np.uint8)
#     point_step = int(msg.point_step)

#     if point_step <= 0:
#         return np.zeros((0, 4), dtype=np.float32)

#     count = int(len(raw) / point_step)

#     if count == 0:
#         return np.zeros((0, 4), dtype=np.float32)

#     field_offsets = {field.name: int(field.offset) for field in msg.fields}

#     points = np.zeros((count, 4), dtype=np.float32)

#     for i in range(count):
#         base = i * point_step

#         for col, name in enumerate(("x", "y", "z", "intensity")):
#             offset = field_offsets.get(name)

#             if offset is None:
#                 continue

#             if base + offset + 4 > len(raw):
#                 continue

#             points[i, col] = np.frombuffer(
#                 raw[base + offset : base + offset + 4],
#                 dtype=np.float32,
#             )[0]

#     points = np.nan_to_num(
#         points,
#         nan=0.0,
#         posinf=0.0,
#         neginf=0.0,
#     ).astype(np.float32)

#     return points


# def save_laserscan(msg, out_path: Path) -> dict[str, Any]:
#     ranges = np.asarray(msg.ranges, dtype=np.float32)

#     if len(msg.intensities) > 0:
#         intensities = np.asarray(msg.intensities, dtype=np.float32)
#     else:
#         intensities = np.zeros(len(ranges), dtype=np.float32)

#     ranges = np.nan_to_num(
#         ranges,
#         nan=float(msg.range_max) if msg.range_max > 0 else 999.0,
#         posinf=float(msg.range_max) if msg.range_max > 0 else 999.0,
#         neginf=0.0,
#     )

#     intensities = np.nan_to_num(
#         intensities,
#         nan=0.0,
#         posinf=0.0,
#         neginf=0.0,
#     )

#     scan = np.stack([ranges, intensities], axis=1).astype(np.float32)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.save(out_path, scan)

#     return {
#         "path": str(out_path),
#         "timestamp_ns": None,
#         "modality": "planar_scan",
#         "shape": list(scan.shape),
#         "dtype": str(scan.dtype),
#         "angle_min": safe_float(msg.angle_min),
#         "angle_max": safe_float(msg.angle_max),
#         "angle_increment": safe_float(msg.angle_increment),
#         "range_min": safe_float(msg.range_min),
#         "range_max": safe_float(msg.range_max),
#     }


# def save_pointcloud(msg, out_path: Path) -> dict[str, Any]:
#     points = decode_pointcloud2_to_xyz_i(msg)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.save(out_path, points)

#     return {
#         "path": str(out_path),
#         "timestamp_ns": None,
#         "modality": "pointcloud_xyz_i",
#         "shape": list(points.shape),
#         "dtype": str(points.dtype),
#     }


# def make_asset_dir(root: Path, group: str) -> Path:
#     path = root / "assets" / group
#     path.mkdir(parents=True, exist_ok=True)

#     return path


# def asset_ref(
#     latest_assets: dict[str, dict[str, Any]],
#     key: str,
# ) -> dict[str, Any] | None:
#     ref = latest_assets.get(key)

#     if ref is None:
#         return None

#     return dict(ref)


# def relative_goal_features(
#     state: dict[str, Any] | None,
#     goal: dict[str, Any] | None,
# ) -> dict[str, float] | None:
#     if state is None or goal is None:
#         return None

#     dx = float(goal["x"] - state["x"])
#     dy = float(goal["y"] - state["y"])
#     dz = float(goal["z"] - state["z"])

#     distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
#     goal_heading = math.atan2(dy, dx)
#     heading_error = wrap_angle(goal_heading - float(state.get("yaw", 0.0)))

#     return {
#         "dx": dx,
#         "dy": dy,
#         "dz": dz,
#         "distance_to_goal": distance,
#         "goal_heading": float(goal_heading),
#         "heading_error": float(heading_error),
#     }


# def platform_onehot(agent_id: str) -> list[float]:
#     return PLATFORM_ONEHOT.get(
#         PLATFORM_TYPE.get(agent_id, "UGV"),
#         [0.0, 0.0],
#     )


# def build_agent_node(
#     agent_id: str,
#     agent: dict[str, Any],
#     ego_state: dict[str, Any],
# ) -> dict[str, Any]:
#     state = agent.get("state")
#     goal = agent.get("goal")
#     command = agent.get("command") or DEFAULT_COMMAND

#     available = bool(agent.get("available", False)) and state is not None

#     if state is None:
#         state = {
#             "x": ego_state["x"],
#             "y": ego_state["y"],
#             "z": ego_state["z"],
#             "yaw": 0.0,
#             "vx": 0.0,
#             "vy": 0.0,
#             "vz": 0.0,
#             "wz": 0.0,
#             "qx": 0.0,
#             "qy": 0.0,
#             "qz": 0.0,
#             "qw": 1.0,
#         }

#     if goal is None:
#         goal = {
#             "x": state["x"],
#             "y": state["y"],
#             "z": state["z"],
#         }

#     p_onehot = platform_onehot(agent_id)

#     features = [
#         float(state["x"] - ego_state["x"]),
#         float(state["y"] - ego_state["y"]),
#         float(state["z"] - ego_state["z"]),

#         float(state.get("vx", 0.0)),
#         float(state.get("vy", 0.0)),
#         float(state.get("vz", 0.0)),
#         float(state.get("wz", 0.0)),

#         float(goal["x"] - state["x"]),
#         float(goal["y"] - state["y"]),
#         float(goal["z"] - state["z"]),

#         float(command.get("linear_x", 0.0)),
#         float(command.get("angular_z", 0.0)),

#         float(p_onehot[0]),
#         float(p_onehot[1]),
#     ]

#     return {
#         "id": agent_id,
#         "platform_type": PLATFORM_TYPE.get(agent_id, "unknown"),
#         "available": available,
#         "ready": agent.get("ready"),
#         "state": state,
#         "start": agent.get("start"),
#         "goal": goal,
#         "goal_features": relative_goal_features(state, goal),
#         "command": command,
#         "imu": agent.get("imu"),
#         "feature": features,
#     }


# def build_graph_edges(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
#     edges = []

#     for src_id, src in nodes.items():
#         for dst_id, dst in nodes.items():
#             if src_id == dst_id:
#                 continue

#             src_state = src["state"]
#             dst_state = dst["state"]

#             dx = float(dst_state["x"] - src_state["x"])
#             dy = float(dst_state["y"] - src_state["y"])
#             dz = float(dst_state["z"] - src_state["z"])

#             distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
#             inv_distance = 1.0 / max(distance, 1e-3)
#             bearing = math.atan2(dy, dx)

#             same_platform = (
#                 1.0
#                 if src.get("platform_type") == dst.get("platform_type")
#                 else 0.0
#             )

#             network = dict(DEFAULT_NETWORK_STATE)

#             edges.append(
#                 {
#                     "source": src_id,
#                     "target": dst_id,

#                     "dx": dx,
#                     "dy": dy,
#                     "dz": dz,
#                     "distance": distance,
#                     "inv_distance": float(inv_distance),
#                     "bearing_sin": float(math.sin(bearing)),
#                     "bearing_cos": float(math.cos(bearing)),
#                     "same_platform": same_platform,

#                     # Flat fields used by dataset_helper.
#                     "latency_s": float(network["latency_s"]),
#                     "jitter_s": float(network["jitter_s"]),
#                     "packet_loss": float(network["packet_loss"]),
#                     "link_quality": float(network["link_quality"]),
#                     "connected": bool(network["connected"]),

#                     # Nested block kept for clarity/manifest compatibility.
#                     "network": network,
#                 }
#             )

#     return edges


# def summarize_forward_pointcloud(
#     points_path: str | None,
#     *,
#     x_min: float = 0.0,
#     x_max: float = 25.0,
#     center_half_width: float = 2.0,
#     side_width: float = 8.0,
#     z_min: float = -5.0,
#     z_max: float = 5.0,
# ) -> dict[str, Any]:
#     """Create a compact left/center/right hazard summary from a pointcloud asset.

#     This is used as simple UAV-derived contextual information. It does not
#     replace the raw pointcloud .npy file; it only gives the models an explicit
#     compact feature as well.
#     """
#     empty = {
#         "available": False,
#         "left_count": 0,
#         "center_count": 0,
#         "right_count": 0,
#         "total_count": 0,
#         "nearest_x": 999.0,
#         "log_left": 0.0,
#         "log_center": 0.0,
#         "log_right": 0.0,
#     }

#     if not points_path:
#         return dict(empty)

#     path = Path(points_path)

#     if not path.exists():
#         return dict(empty)

#     try:
#         points = np.load(path)
#     except Exception:
#         return dict(empty)

#     if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
#         result = dict(empty)
#         result["available"] = True
#         return result

#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]

#     valid = (
#         np.isfinite(x)
#         & np.isfinite(y)
#         & np.isfinite(z)
#         & (x >= x_min)
#         & (x <= x_max)
#         & (z >= z_min)
#         & (z <= z_max)
#         & (np.abs(y) <= side_width)
#     )

#     if not np.any(valid):
#         result = dict(empty)
#         result["available"] = True
#         return result

#     xv = x[valid]
#     yv = y[valid]

#     center = np.abs(yv) <= center_half_width
#     left = yv > center_half_width
#     right = yv < -center_half_width

#     left_count = int(np.sum(left))
#     center_count = int(np.sum(center))
#     right_count = int(np.sum(right))

#     return {
#         "available": True,
#         "left_count": left_count,
#         "center_count": center_count,
#         "right_count": right_count,
#         "total_count": int(len(xv)),
#         "nearest_x": float(np.min(xv)) if len(xv) else 999.0,
#         "log_left": float(np.log1p(left_count)),
#         "log_center": float(np.log1p(center_count)),
#         "log_right": float(np.log1p(right_count)),
#     }


# def teacher_label(agent: dict[str, Any]) -> str:
#     state = agent.get("controller_state")

#     if state:
#         return str(state)

#     action = str(agent.get("obstacle_action") or "clear").lower()

#     if action.endswith("left"):
#         return "avoid_left"

#     if action.endswith("right"):
#         return "avoid_right"

#     return "go_to_goal"


# def build_frame(
#     *,
#     episode_id: str,
#     timestamp_ns: int,
#     world_name: str,
#     ego_id: str,
#     agents: dict[str, dict[str, Any]],
#     latest_assets: dict[str, dict[str, Any]],
# ) -> dict[str, Any] | None:
#     ego = agents[ego_id]

#     if ego.get("state") is None or ego.get("goal") is None:
#         return None

#     ego_state = ego["state"]

#     nodes = {
#         agent_id: build_agent_node(agent_id, agents[agent_id], ego_state)
#         for agent_id in AGENT_ORDER
#     }

#     edges = build_graph_edges(nodes)

#     ego_planar_scan = asset_ref(latest_assets, f"{ego_id}_planar_scan")
#     ego_front_pointcloud = asset_ref(latest_assets, f"{ego_id}_front_points")

#     uav1_front_pointcloud = asset_ref(latest_assets, "uav1_front_points")
#     uav2_front_pointcloud = asset_ref(latest_assets, "uav2_front_points")

#     uav1_hazard_summary = summarize_forward_pointcloud(
#         uav1_front_pointcloud["path"] if uav1_front_pointcloud else None
#     )
#     uav2_hazard_summary = summarize_forward_pointcloud(
#         uav2_front_pointcloud["path"] if uav2_front_pointcloud else None
#     )

#     label = teacher_label(ego)

#     frame = {
#         "episode_id": episode_id,
#         "timestamp_ns": int(timestamp_ns),
#         "world_name": world_name,
#         "ego_id": ego_id,

#         # Backward-compatible single-agent fields.
#         "state": ego_state,
#         "goal": ego.get("goal"),
#         "goal_features": relative_goal_features(ego_state, ego.get("goal")),

#         # Sensor references.
#         "observation": {
#             "ego_planar_scan": ego_planar_scan,
#             "ego_front_pointcloud": ego_front_pointcloud,

#             "uav1_front_pointcloud": uav1_front_pointcloud,
#             "uav1_hazard_summary": uav1_hazard_summary,

#             "uav2_front_pointcloud": uav2_front_pointcloud,
#             "uav2_hazard_summary": uav2_hazard_summary,
#         },

#         # Same content under 'modalities' because notebooks may use this key.
#         "modalities": {
#             "ego_planar_scan": ego_planar_scan,
#             "ego_front_pointcloud": ego_front_pointcloud,

#             "uav1_front_pointcloud": uav1_front_pointcloud,
#             "uav1_hazard_summary": uav1_hazard_summary,

#             "uav2_front_pointcloud": uav2_front_pointcloud,
#             "uav2_hazard_summary": uav2_hazard_summary,
#         },

#         # Teacher / label information.
#         "teacher": {
#             "label": label,
#             "command": ego.get("command") or DEFAULT_COMMAND,
#             "controller_state": ego.get("controller_state"),
#             "obstacle_action": ego.get("obstacle_action"),
#             "obstacle_clearance": ego.get("obstacle_clearance"),
#         },

#         # Graph representation.
#         "agents": nodes,
#         "edges": edges,

#         "uav_context": {
#             "uav1": {
#                 "id": "uav1",
#                 "available": bool(nodes["uav1"]["available"]),
#                 "ready": agents["uav1"].get("ready"),
#                 "state": nodes["uav1"]["state"],
#                 "goal": nodes["uav1"]["goal"],
#                 "goal_features": nodes["uav1"]["goal_features"],
#                 "hazard_summary": uav1_hazard_summary,
#                 "front_pointcloud": uav1_front_pointcloud,
#             },
#             "uav2": {
#                 "id": "uav2",
#                 "available": bool(nodes["uav2"]["available"]),
#                 "ready": agents["uav2"].get("ready"),
#                 "state": nodes["uav2"]["state"],
#                 "goal": nodes["uav2"]["goal"],
#                 "goal_features": nodes["uav2"]["goal_features"],
#                 "hazard_summary": uav2_hazard_summary,
#                 "front_pointcloud": uav2_front_pointcloud,
#             },
#         },

#         # Compatibility block. Old notebooks expected "other_husky".
#         # In the new structure there is no second Husky, so this is intentionally None.
#         "other_husky": None,

#         # Communication placeholder. If OMNeT++ features are later recorded,
#         # this can be replaced with real per-frame values.
#         "network_state": dict(DEFAULT_NETWORK_STATE),

#         "readiness": {
#             "has_scan": ego_planar_scan is not None,
#             "has_state": ego.get("state") is not None,
#             "has_goal": ego.get("goal") is not None,
#             "has_teacher_command": ego.get("command") is not None,

#             "has_uav1_state": agents["uav1"].get("state") is not None,
#             "has_uav2_state": agents["uav2"].get("state") is not None,

#             "has_uav1_pointcloud": uav1_front_pointcloud is not None,
#             "has_uav2_pointcloud": uav2_front_pointcloud is not None,

#             "uav1_ready": bool(agents["uav1"].get("ready", False)),
#             "uav2_ready": bool(agents["uav2"].get("ready", False)),

#             "has_graph_nodes": True,
#             "has_graph_edges": len(edges) > 0,
#         },
#     }

#     return frame


# def schema() -> dict[str, Any]:
#     return {
#         "description": (
#             "Hybrid two-UAV/one-UGV trajectory prediction dataset. Each frame is "
#             "anchored on the ego Husky planar lidar scan and includes ego state, "
#             "goal features, local perception, UAV1/UAV2 context, graph nodes/edges, "
#             "and teacher labels."
#         ),
#         "frame_anchor": "Ego Husky planar lidar scan timestamp.",
#         "prediction_task": {
#             "input": "past 10 frames",
#             "target": "future 20-step ego Husky trajectory",
#             "target_units": "relative future (x, y) positions in meters from anchor frame",
#         },
#         "models_supported": [
#             "CV",
#             "CNN-LSTM",
#             "GNN-LSTM",
#             "CNN-GNN-LSTM",
#         ],
#         "primary_inputs": [
#             "state",
#             "goal_features",
#             "observation.ego_planar_scan",
#             "observation.ego_front_pointcloud",
#             "observation.uav1_front_pointcloud",
#             "observation.uav1_hazard_summary",
#             "observation.uav2_front_pointcloud",
#             "observation.uav2_hazard_summary",
#             "agents",
#             "edges",
#             "network_state",
#         ],
#         "primary_targets": [
#             "future ego trajectory generated by sliding windows",
#             "teacher.command.linear_x",
#             "teacher.command.angular_z",
#             "teacher.label",
#         ],
#         "node_feature_dim": 14,
#         "node_feature_fields": [
#             "rel_x_to_ego",
#             "rel_y_to_ego",
#             "rel_z_to_ego",
#             "vx",
#             "vy",
#             "vz",
#             "wz",
#             "goal_dx",
#             "goal_dy",
#             "goal_dz",
#             "command_linear_x",
#             "command_angular_z",
#             "is_ugv",
#             "is_uav",
#         ],
#         "edge_feature_dim": 11,
#         "edge_feature_fields": [
#             "dx",
#             "dy",
#             "dz",
#             "distance",
#             "inv_distance",
#             "bearing_sin",
#             "bearing_cos",
#             "same_platform",
#             "latency_s",
#             "packet_loss",
#             "link_quality",
#         ],
#         "agents": AGENT_ORDER,
#         "platform_types": PLATFORM_TYPE,
#     }


# def init_agents() -> dict[str, dict[str, Any]]:
#     return {
#         agent_id: {
#             "state": None,
#             "start": None,
#             "goal": None,
#             "command": dict(DEFAULT_COMMAND),
#             "controller_state": None,
#             "obstacle_action": "clear",
#             "obstacle_clearance": None,
#             "imu": None,
#             "ready": False,
#             "available": False,
#         }
#         for agent_id in AGENT_ORDER
#     }


# def export_one_bag(bag_path: Path, out_root: Path) -> tuple[Path, int]:
#     db3_path = bag_db3_path(bag_path)

#     out_dir = out_root / bag_path.name
#     out_dir.mkdir(parents=True, exist_ok=True)

#     frames_path = out_dir / "frames.jsonl"
#     schema_path = out_dir / "schema.json"
#     manifest_path = out_dir / "manifest.json"

#     if frames_path.exists():
#         frames_path.unlink()

#     typestore = get_typestore(Stores.ROS2_HUMBLE)

#     conn = sqlite3.connect(str(db3_path))
#     cur = conn.cursor()

#     topic_rows = list(cur.execute("SELECT id, name, type FROM topics ORDER BY id"))
#     topic_map = {
#         topic_id: (name, msgtype)
#         for topic_id, name, msgtype in topic_rows
#     }

#     topic_names = [name for _, name, _ in topic_rows]
#     world_name = infer_world_name(topic_names)
#     topics = build_topics(world_name)

#     topic_name_set = set(topic_names)

#     agents = init_agents()
#     latest_assets: dict[str, dict[str, Any]] = {}

#     asset_dirs = {
#         "husky_local_planar_scan": make_asset_dir(out_dir, "husky_local/planar_scan"),
#         "husky_local_front_points": make_asset_dir(out_dir, "husky_local/front_points"),

#         "uav1_front_points": make_asset_dir(out_dir, "uav1/front_points"),
#         "uav2_front_points": make_asset_dir(out_dir, "uav2/front_points"),
#     }

#     topic_to_asset_key = {
#         topics["husky_local_planar_scan"]: "husky_local_planar_scan",
#         topics["husky_local_front_points"]: "husky_local_front_points",
#         topics["uav1_front_points"]: "uav1_front_points",
#         topics["uav2_front_points"]: "uav2_front_points",
#     }

#     # New structure: frame is anchored only on ego Husky scan.
#     anchor_topics = {
#         topics["husky_local_planar_scan"]: "husky_local",
#     }

#     frame_count = 0
#     topic_message_counts: dict[str, int] = {}

#     query = "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"

#     for topic_id, timestamp, rawdata in cur.execute(query):
#         topic, msgtype = topic_map[topic_id]
#         topic_message_counts[topic] = topic_message_counts.get(topic, 0) + 1

#         try:
#             msg = typestore.deserialize_cdr(rawdata, msgtype)
#         except Exception as exc:
#             print(f"Warning: failed to deserialize topic={topic} type={msgtype}: {exc}")
#             continue

#         ts = int(timestamp)

#         if topic == topics["dynamic_pose"]:
#             update_agents_from_dynamic_pose(msg, agents)

#         elif topic == topics["husky_local_odom"]:
#             agents["husky_local"]["state"] = pose_from_odom(msg)
#             agents["husky_local"]["available"] = True

#         elif topic == topics["uav1_odom"]:
#             agents["uav1"]["state"] = pose_from_odom(msg)
#             agents["uav1"]["available"] = True

#         elif topic == topics["uav2_odom"]:
#             agents["uav2"]["state"] = pose_from_odom(msg)
#             agents["uav2"]["available"] = True

#         elif topic == topics["cmd_husky_local"]:
#             agents["husky_local"]["command"] = cmd_from_twist(msg)

#         elif topic in (topics["cmd_uav1_model"], topics["cmd_uav1_direct"]):
#             agents["uav1"]["command"] = cmd_from_twist(msg)

#         elif topic in (topics["cmd_uav2_model"], topics["cmd_uav2_direct"]):
#             agents["uav2"]["command"] = cmd_from_twist(msg)

#         elif topic == topics["state_husky_local"]:
#             agents["husky_local"]["controller_state"] = str(msg.data)

#         elif topic == topics["obstacle_action_husky_local"]:
#             agents["husky_local"]["obstacle_action"] = str(msg.data)

#         elif topic == topics["obstacle_clearance_husky_local"]:
#             agents["husky_local"]["obstacle_clearance"] = clearance_from_vector3(msg)

#         elif topic == topics["husky_local_start"]:
#             agents["husky_local"]["start"] = pose_from_pose_stamped(msg)

#         elif topic == topics["husky_local_goal"]:
#             agents["husky_local"]["goal"] = pose_from_pose_stamped(msg)

#         elif topic == topics["uav1_start"]:
#             agents["uav1"]["start"] = pose_from_pose_stamped(msg)

#         elif topic == topics["uav1_goal"]:
#             agents["uav1"]["goal"] = pose_from_pose_stamped(msg)

#         elif topic == topics["uav2_start"]:
#             agents["uav2"]["start"] = pose_from_pose_stamped(msg)

#         elif topic == topics["uav2_goal"]:
#             agents["uav2"]["goal"] = pose_from_pose_stamped(msg)

#         elif topic == topics["uav1_ready"]:
#             agents["uav1"]["ready"] = bool(msg.data)

#         elif topic == topics["uav2_ready"]:
#             agents["uav2"]["ready"] = bool(msg.data)

#         elif topic == topics["husky_local_imu"]:
#             agents["husky_local"]["imu"] = imu_from_msg(msg)

#         elif topic == topics["uav1_imu"]:
#             agents["uav1"]["imu"] = imu_from_msg(msg)

#         elif topic == topics["uav2_imu"]:
#             agents["uav2"]["imu"] = imu_from_msg(msg)

#         # Save heavy sensor assets as .npy and reference them from JSONL.
#         if topic in topic_to_asset_key:
#             asset_key = topic_to_asset_key[topic]
#             asset_dir = asset_dirs[asset_key]
#             out_path = asset_dir / f"{ts}.npy"

#             if asset_key.endswith("planar_scan"):
#                 meta = save_laserscan(msg, out_path)
#             else:
#                 meta = save_pointcloud(msg, out_path)

#             meta["timestamp_ns"] = ts
#             latest_assets[asset_key] = meta

#         # Anchor frame after the current message has updated latest assets.
#         if topic in anchor_topics:
#             ego_id = anchor_topics[topic]

#             frame = build_frame(
#                 episode_id=bag_path.name,
#                 timestamp_ns=ts,
#                 world_name=world_name,
#                 ego_id=ego_id,
#                 agents=agents,
#                 latest_assets=latest_assets,
#             )

#             if frame is None:
#                 continue

#             with frames_path.open("a", encoding="utf-8") as f:
#                 f.write(json.dumps(frame) + "\n")

#             frame_count += 1

#     conn.close()

#     schema_path.write_text(json.dumps(schema(), indent=2), encoding="utf-8")

#     manifest = {
#         "bag_path": str(bag_path),
#         "db3_path": str(db3_path),
#         "out_dir": str(out_dir),
#         "world_name": world_name,
#         "frame_count": frame_count,
#         "agents": AGENT_ORDER,
#         "topics_present": sorted(topic_name_set),
#         "expected_topics": topics,
#         "missing_expected_topics": sorted(
#             [
#                 topic
#                 for topic in topics.values()
#                 if topic not in topic_name_set
#             ]
#         ),
#         "topic_message_counts": topic_message_counts,
#         "notes": [
#             "Frames are anchored only on the ego Husky planar scan topic.",
#             "Future trajectory targets are built later by sliding windows in dataset_helper/notebooks.",
#             "UAV1 and UAV2 context data is included when the source bag recorded those topics.",
#             "The new intended graph is husky_local + uav1 + uav2.",
#             "No second Husky is required in this dataset structure.",
#         ],
#     }

#     manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

#     print("Saved hybrid two-UAV dataset to:", out_dir)
#     print("World:", world_name)
#     print("Frames:", frame_count)
#     print("Schema:", schema_path)
#     print("Manifest:", manifest_path)

#     if frame_count == 0:
#         print(
#             "WARNING: Exported 0 frames. This usually means the bag does not contain "
#             "the ego Husky planar scan topic, odometry, or episode goal metadata."
#         )

#     return out_dir, frame_count


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Export hybrid two-UAV/one-UGV JSONL dataset from recorded ROS 2 bag(s)."
#     )

#     parser.add_argument(
#         "--bag",
#         default="",
#         help="Optional path to a specific bag directory. If omitted, all run_* bags are exported.",
#     )

#     parser.add_argument(
#         "--out-root",
#         default=str(OUT_ROOT),
#         help="Dataset output root.",
#     )

#     args = parser.parse_args()

#     out_root = Path(args.out_root).expanduser().resolve()

#     if args.bag:
#         bag_paths = [Path(args.bag).expanduser().resolve()]
#     else:
#         bag_paths = available_bags(BAGS_DIR)

#         if not bag_paths:
#             raise FileNotFoundError(
#                 f"No bag directories found in {BAGS_DIR}. "
#                 "First run the simulation and record at least one bag."
#             )

#     print("Export root:", out_root)
#     print("Bag count:", len(bag_paths))

#     total_frames = 0

#     for bag_path in bag_paths:
#         print("\n=== Exporting", bag_path.name, "===")
#         _out_dir, frame_count = export_one_bag(bag_path, out_root)
#         total_frames += int(frame_count)

#     print("\nCompleted hybrid two-UAV dataset export.")
#     print("Total bags:", len(bag_paths))
#     print("Total frames:", total_frames)
#     print("Output root:", out_root)


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""Export a hybrid two-UAV / one-UGV maneuver dataset from recorded ROS 2 bags.

This exporter converts rosbag2 SQLite bags into a JSONL + NPY dataset suitable
for the thesis hybrid models:

- CV baseline
- CNN-LSTM / temporal CNN-LSTM baseline
- GNN-LSTM
- CNN-GNN-LSTM

Each JSONL row is one synchronized frame anchored on the ego Husky planar lidar scan.

Current intended agent structure:
- husky_local: ego UGV whose future trajectory is predicted
- uav1: left-side UAV context / aerial perception
- uav2: right-side UAV context / aerial perception

Important dataset idea:
The model observes past frames and learns to predict the ego Husky's future
20-step trajectory. Each exported frame contains:
- ego Husky state,
- ego local lidar/pointcloud references,
- ego goal and goal-relative features,
- teacher command/state labels from the rule-based controller,
- UAV1 and UAV2 context,
- UAV1 and UAV2 pointcloud references,
- compact UAV hazard summaries,
- graph nodes and graph edges,
- placeholder communication/network features.

The exporter does NOT train the model. It only prepares the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np


try:
    from rosbags.typesys import Stores, get_typestore
except ModuleNotFoundError:
    print(
        "\nERROR: Python package 'rosbags' is not installed in this conda environment.\n\n"
        "Run this first:\n"
        "  conda activate maneuver\n"
        "  pip install rosbags\n\n"
        "Then run the exporter again:\n"
        "  python 03_dataset/exporters/export_hybrid_maneuver_dataset.py\n",
        file=sys.stderr,
    )
    raise


THESIS_ROOT = Path.home() / "Documents/Thesis"
DATASET_ROOT = THESIS_ROOT / "03_dataset"
BAGS_DIR = DATASET_ROOT / "bags"
OUT_ROOT = DATASET_ROOT / "husky_control_dataset"

WORLD_TOPIC_RE = re.compile(r"^/world/([^/]+)/")

AGENT_ORDER = ["husky_local", "uav1", "uav2"]

PLATFORM_TYPE = {
    "husky_local": "UGV",
    "uav1": "UAV",
    "uav2": "UAV",
}

PLATFORM_ONEHOT = {
    "UGV": [1.0, 0.0],
    "UAV": [0.0, 1.0],
}

DEFAULT_COMMAND = {
    "linear_x": 0.0,
    "linear_y": 0.0,
    "linear_z": 0.0,
    "angular_z": 0.0,
}

DEFAULT_NETWORK_STATE = {
    "latency_s": 0.0,
    "jitter_s": 0.0,
    "packet_loss": 0.0,
    "link_quality": 1.0,
    "connected": True,
    "source": "default_no_omnet",
}


def bag_db3_path(bag_dir: Path) -> Path:
    db3_files = sorted(bag_dir.glob("*.db3"))

    if not db3_files:
        raise RuntimeError(f"No .db3 file found in {bag_dir}")

    return db3_files[0]


def available_bags(root: Path) -> list[Path]:
    if not root.exists():
        return []

    prefixes = (
        "run_",
        "run_model_",
        "run_dataset_",
        "open_hazard_teacher_",
    )

    return sorted(
        [
            p.resolve()
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith(prefixes)
        ]
    )


def infer_world_name(topic_names: list[str]) -> str:
    for topic in topic_names:
        match = WORLD_TOPIC_RE.match(topic)

        if match:
            return match.group(1)

    raise RuntimeError("Could not infer Gazebo world name from bag topics.")


def build_topics(world_name: str) -> dict[str, str]:
    """Build all expected topic names for the one-Husky/two-UAV dataset."""
    return {
        "dynamic_pose": f"/world/{world_name}/dynamic_pose/info",

        # Ego Husky.
        "husky_local_odom": "/model/husky_local/odometry",
        "cmd_husky_local": "/cmd_vel",
        "state_husky_local": "/husky_local/controller_state",
        "obstacle_action_husky_local": "/husky_local/obstacle_action",
        "obstacle_clearance_husky_local": "/husky_local/obstacle_clearance",
        "husky_local_start": "/episode/husky_local/start",
        "husky_local_goal": "/episode/husky_local/goal",

        # UAV 1.
        "uav1_odom": "/model/uav1/odometry",
        "uav1_ready": "/uav1/ready",
        "uav1_start": "/episode/uav1/start",
        "uav1_goal": "/episode/uav1/goal",
        "cmd_uav1_model": "/model/uav1/command/twist",
        "cmd_uav1_direct": "/uav1/command/twist",

        # UAV 2.
        "uav2_odom": "/model/uav2/odometry",
        "uav2_ready": "/uav2/ready",
        "uav2_start": "/episode/uav2/start",
        "uav2_goal": "/episode/uav2/goal",
        "cmd_uav2_model": "/model/uav2/command/twist",
        "cmd_uav2_direct": "/uav2/command/twist",

        # Ego Husky sensors.
        "husky_local_planar_scan": (
            f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan"
        ),
        "husky_local_front_points": (
            f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points"
        ),
        "husky_local_imu": (
            f"/world/{world_name}/model/husky_local/link/base_link/sensor/imu_sensor/imu"
        ),

        # UAV sensors.
        "uav1_front_points": (
            f"/world/{world_name}/model/uav1/link/base_link/sensor/front_laser/scan/points"
        ),
        "uav1_imu": (
            f"/world/{world_name}/model/uav1/link/base_link/sensor/imu_sensor/imu"
        ),
        "uav2_front_points": (
            f"/world/{world_name}/model/uav2/link/base_link/sensor/front_laser/scan/points"
        ),
        "uav2_imu": (
            f"/world/{world_name}/model/uav2/link/base_link/sensor/imu_sensor/imu"
        ),
    }


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def pose_from_odom(msg) -> dict[str, float]:
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


def pose_from_tf_transform(transform) -> dict[str, float]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation

    yaw = quaternion_to_yaw(
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    )

    return {
        "x": float(translation.x),
        "y": float(translation.y),
        "z": float(translation.z),

        "qx": float(rotation.x),
        "qy": float(rotation.y),
        "qz": float(rotation.z),
        "qw": float(rotation.w),

        "yaw": float(yaw),

        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "wz": 0.0,
    }


def pose_from_pose_stamped(msg) -> dict[str, float]:
    pose = msg.pose

    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
    }


def cmd_from_twist(msg) -> dict[str, float]:
    return {
        "linear_x": float(msg.linear.x),
        "linear_y": float(msg.linear.y),
        "linear_z": float(msg.linear.z),
        "angular_z": float(msg.angular.z),
    }


def clearance_from_vector3(msg) -> dict[str, float]:
    return {
        "front": float(msg.x),
        "left": float(msg.y),
        "right": float(msg.z),
    }


def imu_from_msg(msg) -> dict[str, float]:
    return {
        "orientation_x": float(msg.orientation.x),
        "orientation_y": float(msg.orientation.y),
        "orientation_z": float(msg.orientation.z),
        "orientation_w": float(msg.orientation.w),

        "angular_velocity_x": float(msg.angular_velocity.x),
        "angular_velocity_y": float(msg.angular_velocity.y),
        "angular_velocity_z": float(msg.angular_velocity.z),

        "linear_acceleration_x": float(msg.linear_acceleration.x),
        "linear_acceleration_y": float(msg.linear_acceleration.y),
        "linear_acceleration_z": float(msg.linear_acceleration.z),
    }


def update_agents_from_dynamic_pose(
    msg,
    agents: dict[str, dict[str, Any]],
) -> None:
    """Update agent states from Gazebo dynamic_pose/info.

    This gives world-coordinate positions for husky_local, uav1, and uav2.
    If odometry has already supplied velocities, those velocities are kept.
    """
    for transform in msg.transforms:
        child = transform.child_frame_id or ""
        parts = [part for part in child.split("/") if part]

        matched_agent = None

        for agent_name in AGENT_ORDER:
            if agent_name in parts:
                matched_agent = agent_name
                break

        if matched_agent is None:
            continue

        new_state = pose_from_tf_transform(transform)
        old_state = agents[matched_agent].get("state")

        if old_state is not None:
            new_state["vx"] = safe_float(old_state.get("vx", 0.0))
            new_state["vy"] = safe_float(old_state.get("vy", 0.0))
            new_state["vz"] = safe_float(old_state.get("vz", 0.0))
            new_state["wz"] = safe_float(old_state.get("wz", 0.0))

        agents[matched_agent]["state"] = new_state
        agents[matched_agent]["available"] = True


def decode_pointcloud2_to_xyz_i(msg) -> np.ndarray:
    """Decode PointCloud2 into an Nx4 [x, y, z, intensity] float32 array.

    This is intentionally minimal and works for common Gazebo pointcloud fields.
    Missing intensity is filled with zero.
    """
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    point_step = int(msg.point_step)

    if point_step <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    count = int(len(raw) / point_step)

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

            if base + offset + 4 > len(raw):
                continue

            points[i, col] = np.frombuffer(
                raw[base + offset : base + offset + 4],
                dtype=np.float32,
            )[0]

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    return points


def save_laserscan(msg, out_path: Path) -> dict[str, Any]:
    ranges = np.asarray(msg.ranges, dtype=np.float32)

    if len(msg.intensities) > 0:
        intensities = np.asarray(msg.intensities, dtype=np.float32)
    else:
        intensities = np.zeros(len(ranges), dtype=np.float32)

    ranges = np.nan_to_num(
        ranges,
        nan=float(msg.range_max) if msg.range_max > 0 else 999.0,
        posinf=float(msg.range_max) if msg.range_max > 0 else 999.0,
        neginf=0.0,
    )

    intensities = np.nan_to_num(
        intensities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scan = np.stack([ranges, intensities], axis=1).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, scan)

    return {
        "path": str(out_path),
        "timestamp_ns": None,
        "modality": "planar_scan",
        "shape": list(scan.shape),
        "dtype": str(scan.dtype),
        "angle_min": safe_float(msg.angle_min),
        "angle_max": safe_float(msg.angle_max),
        "angle_increment": safe_float(msg.angle_increment),
        "range_min": safe_float(msg.range_min),
        "range_max": safe_float(msg.range_max),
    }


def save_pointcloud(msg, out_path: Path) -> dict[str, Any]:
    points = decode_pointcloud2_to_xyz_i(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, points)

    return {
        "path": str(out_path),
        "timestamp_ns": None,
        "modality": "pointcloud_xyz_i",
        "shape": list(points.shape),
        "dtype": str(points.dtype),
    }


def make_asset_dir(root: Path, group: str) -> Path:
    path = root / "assets" / group
    path.mkdir(parents=True, exist_ok=True)

    return path


def asset_ref(
    latest_assets: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    ref = latest_assets.get(key)

    if ref is None:
        return None

    return dict(ref)


def relative_goal_features(
    state: dict[str, Any] | None,
    goal: dict[str, Any] | None,
) -> dict[str, float] | None:
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


def platform_onehot(agent_id: str) -> list[float]:
    return PLATFORM_ONEHOT.get(
        PLATFORM_TYPE.get(agent_id, "UGV"),
        [0.0, 0.0],
    )


def build_agent_node(
    agent_id: str,
    agent: dict[str, Any],
    ego_state: dict[str, Any],
) -> dict[str, Any]:
    state = agent.get("state")
    goal = agent.get("goal")
    command = agent.get("command") or dict(DEFAULT_COMMAND)

    available = bool(agent.get("available", False)) and state is not None

    if state is None:
        state = {
            "x": ego_state["x"],
            "y": ego_state["y"],
            "z": ego_state["z"],
            "yaw": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wz": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }

    if goal is None:
        goal = {
            "x": state["x"],
            "y": state["y"],
            "z": state["z"],
        }

    p_onehot = platform_onehot(agent_id)

    features = [
        float(state["x"] - ego_state["x"]),
        float(state["y"] - ego_state["y"]),
        float(state["z"] - ego_state["z"]),

        float(state.get("vx", 0.0)),
        float(state.get("vy", 0.0)),
        float(state.get("vz", 0.0)),
        float(state.get("wz", 0.0)),

        float(goal["x"] - state["x"]),
        float(goal["y"] - state["y"]),
        float(goal["z"] - state["z"]),

        float(command.get("linear_x", 0.0)),
        float(command.get("angular_z", 0.0)),

        float(p_onehot[0]),
        float(p_onehot[1]),
    ]

    return {
        "id": agent_id,
        "platform_type": PLATFORM_TYPE.get(agent_id, "unknown"),
        "available": available,
        "ready": agent.get("ready"),
        "state": state,
        "start": agent.get("start"),
        "goal": goal,
        "goal_features": relative_goal_features(state, goal),
        "command": command,
        "imu": agent.get("imu"),
        "feature": features,
    }


def build_graph_edges(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []

    for src_id, src in nodes.items():
        for dst_id, dst in nodes.items():
            if src_id == dst_id:
                continue

            src_state = src["state"]
            dst_state = dst["state"]

            dx = float(dst_state["x"] - src_state["x"])
            dy = float(dst_state["y"] - src_state["y"])
            dz = float(dst_state["z"] - src_state["z"])

            distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
            inv_distance = 1.0 / max(distance, 1e-3)
            bearing = math.atan2(dy, dx)

            same_platform = (
                1.0
                if src.get("platform_type") == dst.get("platform_type")
                else 0.0
            )

            network = dict(DEFAULT_NETWORK_STATE)

            edges.append(
                {
                    "source": src_id,
                    "target": dst_id,

                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "distance": distance,
                    "inv_distance": float(inv_distance),
                    "bearing_sin": float(math.sin(bearing)),
                    "bearing_cos": float(math.cos(bearing)),
                    "same_platform": same_platform,

                    # Flat fields used by dataset_helper.
                    "latency_s": float(network["latency_s"]),
                    "jitter_s": float(network["jitter_s"]),
                    "packet_loss": float(network["packet_loss"]),
                    "link_quality": float(network["link_quality"]),
                    "connected": bool(network["connected"]),

                    # Nested block kept for clarity and future OMNeT++ integration.
                    "network": network,
                }
            )

    return edges


def summarize_forward_pointcloud(
    points_path: str | None,
    *,
    x_min: float = 0.0,
    x_max: float = 25.0,
    center_half_width: float = 2.0,
    side_width: float = 8.0,
    z_min: float = -5.0,
    z_max: float = 5.0,
) -> dict[str, Any]:
    """Create a compact left/center/right hazard summary from a pointcloud asset.

    The raw pointcloud is still saved as .npy. This summary is only an extra
    compact feature for the models.
    """
    empty = {
        "available": False,
        "left_count": 0,
        "center_count": 0,
        "right_count": 0,
        "total_count": 0,
        "nearest_x": 999.0,
        "log_left": 0.0,
        "log_center": 0.0,
        "log_right": 0.0,
    }

    if not points_path:
        return dict(empty)

    path = Path(points_path)

    if not path.exists():
        return dict(empty)

    try:
        points = np.load(path)
    except Exception:
        return dict(empty)

    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        result = dict(empty)
        result["available"] = True
        return result

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x >= x_min)
        & (x <= x_max)
        & (z >= z_min)
        & (z <= z_max)
        & (np.abs(y) <= side_width)
    )

    if not np.any(valid):
        result = dict(empty)
        result["available"] = True
        return result

    xv = x[valid]
    yv = y[valid]

    center = np.abs(yv) <= center_half_width
    left = yv > center_half_width
    right = yv < -center_half_width

    left_count = int(np.sum(left))
    center_count = int(np.sum(center))
    right_count = int(np.sum(right))

    return {
        "available": True,
        "left_count": left_count,
        "center_count": center_count,
        "right_count": right_count,
        "total_count": int(len(xv)),
        "nearest_x": float(np.min(xv)) if len(xv) else 999.0,
        "log_left": float(np.log1p(left_count)),
        "log_center": float(np.log1p(center_count)),
        "log_right": float(np.log1p(right_count)),
    }


def teacher_label(agent: dict[str, Any]) -> str:
    state = agent.get("controller_state")

    if state:
        return str(state)

    action = str(agent.get("obstacle_action") or "clear").lower()

    if action.endswith("left"):
        return "avoid_left"

    if action.endswith("right"):
        return "avoid_right"

    return "go_to_goal"


def build_frame(
    *,
    episode_id: str,
    timestamp_ns: int,
    world_name: str,
    ego_id: str,
    agents: dict[str, dict[str, Any]],
    latest_assets: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    ego = agents[ego_id]

    if ego.get("state") is None or ego.get("goal") is None:
        return None

    ego_state = ego["state"]

    nodes = {
        agent_id: build_agent_node(agent_id, agents[agent_id], ego_state)
        for agent_id in AGENT_ORDER
    }

    edges = build_graph_edges(nodes)

    ego_planar_scan = asset_ref(latest_assets, f"{ego_id}_planar_scan")
    ego_front_pointcloud = asset_ref(latest_assets, f"{ego_id}_front_points")

    uav1_front_pointcloud = asset_ref(latest_assets, "uav1_front_points")
    uav2_front_pointcloud = asset_ref(latest_assets, "uav2_front_points")

    uav1_hazard_summary = summarize_forward_pointcloud(
        uav1_front_pointcloud["path"] if uav1_front_pointcloud else None
    )
    uav2_hazard_summary = summarize_forward_pointcloud(
        uav2_front_pointcloud["path"] if uav2_front_pointcloud else None
    )

    label = teacher_label(ego)

    frame = {
        "episode_id": episode_id,
        "timestamp_ns": int(timestamp_ns),
        "world_name": world_name,
        "ego_id": ego_id,

        # Backward-compatible single-agent fields.
        "state": ego_state,
        "goal": ego.get("goal"),
        "goal_features": relative_goal_features(ego_state, ego.get("goal")),

        # Sensor references.
        "observation": {
            "ego_planar_scan": ego_planar_scan,
            "ego_front_pointcloud": ego_front_pointcloud,

            "uav1_front_pointcloud": uav1_front_pointcloud,
            "uav1_hazard_summary": uav1_hazard_summary,

            "uav2_front_pointcloud": uav2_front_pointcloud,
            "uav2_hazard_summary": uav2_hazard_summary,
        },

        # Same content under 'modalities' because notebooks may use this key.
        "modalities": {
            "ego_planar_scan": ego_planar_scan,
            "ego_front_pointcloud": ego_front_pointcloud,

            "uav1_front_pointcloud": uav1_front_pointcloud,
            "uav1_hazard_summary": uav1_hazard_summary,

            "uav2_front_pointcloud": uav2_front_pointcloud,
            "uav2_hazard_summary": uav2_hazard_summary,
        },

        # Teacher / label information.
        "teacher": {
            "label": label,
            "command": ego.get("command") or dict(DEFAULT_COMMAND),
            "controller_state": ego.get("controller_state"),
            "obstacle_action": ego.get("obstacle_action"),
            "obstacle_clearance": ego.get("obstacle_clearance"),
        },

        # Graph representation.
        "agents": nodes,
        "edges": edges,

        "uav_context": {
            "uav1": {
                "id": "uav1",
                "available": bool(nodes["uav1"]["available"]),
                "ready": agents["uav1"].get("ready"),
                "state": nodes["uav1"]["state"],
                "goal": nodes["uav1"]["goal"],
                "goal_features": nodes["uav1"]["goal_features"],
                "hazard_summary": uav1_hazard_summary,
                "front_pointcloud": uav1_front_pointcloud,
            },
            "uav2": {
                "id": "uav2",
                "available": bool(nodes["uav2"]["available"]),
                "ready": agents["uav2"].get("ready"),
                "state": nodes["uav2"]["state"],
                "goal": nodes["uav2"]["goal"],
                "goal_features": nodes["uav2"]["goal_features"],
                "hazard_summary": uav2_hazard_summary,
                "front_pointcloud": uav2_front_pointcloud,
            },
        },

        # Compatibility block. Old notebooks expected "other_husky".
        # In the new structure there is no second Husky.
        "other_husky": None,

        # Communication placeholder. If OMNeT++ features are later recorded,
        # this can be replaced with real per-frame values.
        "network_state": dict(DEFAULT_NETWORK_STATE),

        "readiness": {
            "has_scan": ego_planar_scan is not None,
            "has_state": ego.get("state") is not None,
            "has_goal": ego.get("goal") is not None,
            "has_teacher_command": ego.get("command") is not None,

            "has_uav1_state": agents["uav1"].get("state") is not None,
            "has_uav2_state": agents["uav2"].get("state") is not None,

            "has_uav1_pointcloud": uav1_front_pointcloud is not None,
            "has_uav2_pointcloud": uav2_front_pointcloud is not None,

            "uav1_ready": bool(agents["uav1"].get("ready", False)),
            "uav2_ready": bool(agents["uav2"].get("ready", False)),

            "has_graph_nodes": True,
            "has_graph_edges": len(edges) > 0,
        },
    }

    return frame


def schema() -> dict[str, Any]:
    return {
        "description": (
            "Hybrid two-UAV/one-UGV trajectory prediction dataset. Each frame is "
            "anchored on the ego Husky planar lidar scan and includes ego state, "
            "goal features, local perception, UAV1/UAV2 context, graph nodes/edges, "
            "and teacher labels."
        ),
        "frame_anchor": "Ego Husky planar lidar scan timestamp.",
        "prediction_task": {
            "input": "past 10 frames",
            "target": "future 20-step ego Husky trajectory",
            "target_units": "relative future (x, y) positions in meters from anchor frame",
        },
        "models_supported": [
            "CV",
            "CNN-LSTM",
            "GNN-LSTM",
            "CNN-GNN-LSTM",
        ],
        "primary_inputs": [
            "state",
            "goal_features",
            "observation.ego_planar_scan",
            "observation.ego_front_pointcloud",
            "observation.uav1_front_pointcloud",
            "observation.uav1_hazard_summary",
            "observation.uav2_front_pointcloud",
            "observation.uav2_hazard_summary",
            "agents",
            "edges",
            "network_state",
        ],
        "primary_targets": [
            "future ego trajectory generated by sliding windows",
            "teacher.command.linear_x",
            "teacher.command.angular_z",
            "teacher.label",
        ],
        "node_feature_dim": 14,
        "node_feature_fields": [
            "rel_x_to_ego",
            "rel_y_to_ego",
            "rel_z_to_ego",
            "vx",
            "vy",
            "vz",
            "wz",
            "goal_dx",
            "goal_dy",
            "goal_dz",
            "command_linear_x",
            "command_angular_z",
            "is_ugv",
            "is_uav",
        ],
        "edge_feature_dim": 11,
        "edge_feature_fields": [
            "dx",
            "dy",
            "dz",
            "distance",
            "inv_distance",
            "bearing_sin",
            "bearing_cos",
            "same_platform",
            "latency_s",
            "packet_loss",
            "link_quality",
        ],
        "agents": AGENT_ORDER,
        "platform_types": PLATFORM_TYPE,
    }


def init_agents() -> dict[str, dict[str, Any]]:
    return {
        agent_id: {
            "state": None,
            "start": None,
            "goal": None,
            "command": dict(DEFAULT_COMMAND),
            "controller_state": None,
            "obstacle_action": "clear",
            "obstacle_clearance": None,
            "imu": None,
            "ready": False,
            "available": False,
        }
        for agent_id in AGENT_ORDER
    }


def export_one_bag(bag_path: Path, out_root: Path) -> tuple[Path, int]:
    db3_path = bag_db3_path(bag_path)

    out_dir = out_root / bag_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_path = out_dir / "frames.jsonl"
    schema_path = out_dir / "schema.json"
    manifest_path = out_dir / "manifest.json"

    if frames_path.exists():
        frames_path.unlink()

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()

    topic_rows = list(cur.execute("SELECT id, name, type FROM topics ORDER BY id"))
    topic_map = {
        topic_id: (name, msgtype)
        for topic_id, name, msgtype in topic_rows
    }

    topic_names = [name for _, name, _ in topic_rows]
    world_name = infer_world_name(topic_names)
    topics = build_topics(world_name)

    topic_name_set = set(topic_names)

    agents = init_agents()
    latest_assets: dict[str, dict[str, Any]] = {}

    asset_dirs = {
        "husky_local_planar_scan": make_asset_dir(out_dir, "husky_local/planar_scan"),
        "husky_local_front_points": make_asset_dir(out_dir, "husky_local/front_points"),
        "uav1_front_points": make_asset_dir(out_dir, "uav1/front_points"),
        "uav2_front_points": make_asset_dir(out_dir, "uav2/front_points"),
    }

    topic_to_asset_key = {
        topics["husky_local_planar_scan"]: "husky_local_planar_scan",
        topics["husky_local_front_points"]: "husky_local_front_points",
        topics["uav1_front_points"]: "uav1_front_points",
        topics["uav2_front_points"]: "uav2_front_points",
    }

    # New structure: frames are anchored only on the ego Husky planar scan.
    anchor_topics = {
        topics["husky_local_planar_scan"]: "husky_local",
    }

    frame_count = 0
    topic_message_counts: dict[str, int] = {}

    query = "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"

    for topic_id, timestamp, rawdata in cur.execute(query):
        topic, msgtype = topic_map[topic_id]
        topic_message_counts[topic] = topic_message_counts.get(topic, 0) + 1

        try:
            msg = typestore.deserialize_cdr(rawdata, msgtype)
        except Exception as exc:
            print(f"Warning: failed to deserialize topic={topic} type={msgtype}: {exc}")
            continue

        ts = int(timestamp)

        if topic == topics["dynamic_pose"]:
            update_agents_from_dynamic_pose(msg, agents)

        elif topic == topics["husky_local_odom"]:
            agents["husky_local"]["state"] = pose_from_odom(msg)
            agents["husky_local"]["available"] = True

        elif topic == topics["uav1_odom"]:
            agents["uav1"]["state"] = pose_from_odom(msg)
            agents["uav1"]["available"] = True

        elif topic == topics["uav2_odom"]:
            agents["uav2"]["state"] = pose_from_odom(msg)
            agents["uav2"]["available"] = True

        elif topic == topics["cmd_husky_local"]:
            agents["husky_local"]["command"] = cmd_from_twist(msg)

        elif topic in (topics["cmd_uav1_model"], topics["cmd_uav1_direct"]):
            agents["uav1"]["command"] = cmd_from_twist(msg)

        elif topic in (topics["cmd_uav2_model"], topics["cmd_uav2_direct"]):
            agents["uav2"]["command"] = cmd_from_twist(msg)

        elif topic == topics["state_husky_local"]:
            agents["husky_local"]["controller_state"] = str(msg.data)

        elif topic == topics["obstacle_action_husky_local"]:
            agents["husky_local"]["obstacle_action"] = str(msg.data)

        elif topic == topics["obstacle_clearance_husky_local"]:
            agents["husky_local"]["obstacle_clearance"] = clearance_from_vector3(msg)

        elif topic == topics["husky_local_start"]:
            agents["husky_local"]["start"] = pose_from_pose_stamped(msg)

        elif topic == topics["husky_local_goal"]:
            agents["husky_local"]["goal"] = pose_from_pose_stamped(msg)

        elif topic == topics["uav1_start"]:
            agents["uav1"]["start"] = pose_from_pose_stamped(msg)

        elif topic == topics["uav1_goal"]:
            agents["uav1"]["goal"] = pose_from_pose_stamped(msg)

        elif topic == topics["uav2_start"]:
            agents["uav2"]["start"] = pose_from_pose_stamped(msg)

        elif topic == topics["uav2_goal"]:
            agents["uav2"]["goal"] = pose_from_pose_stamped(msg)

        elif topic == topics["uav1_ready"]:
            agents["uav1"]["ready"] = bool(msg.data)

        elif topic == topics["uav2_ready"]:
            agents["uav2"]["ready"] = bool(msg.data)

        elif topic == topics["husky_local_imu"]:
            agents["husky_local"]["imu"] = imu_from_msg(msg)

        elif topic == topics["uav1_imu"]:
            agents["uav1"]["imu"] = imu_from_msg(msg)

        elif topic == topics["uav2_imu"]:
            agents["uav2"]["imu"] = imu_from_msg(msg)

        # Save heavy sensor assets as .npy and reference them from JSONL.
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

        # Anchor frame after the current message has updated latest assets.
        if topic in anchor_topics:
            ego_id = anchor_topics[topic]

            frame = build_frame(
                episode_id=bag_path.name,
                timestamp_ns=ts,
                world_name=world_name,
                ego_id=ego_id,
                agents=agents,
                latest_assets=latest_assets,
            )

            if frame is None:
                continue

            with frames_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(frame) + "\n")

            frame_count += 1

    conn.close()

    schema_path.write_text(json.dumps(schema(), indent=2), encoding="utf-8")

    manifest = {
        "bag_path": str(bag_path),
        "db3_path": str(db3_path),
        "out_dir": str(out_dir),
        "world_name": world_name,
        "frame_count": frame_count,
        "agents": AGENT_ORDER,
        "topics_present": sorted(topic_name_set),
        "expected_topics": topics,
        "missing_expected_topics": sorted(
            [
                topic
                for topic in topics.values()
                if topic not in topic_name_set
            ]
        ),
        "topic_message_counts": topic_message_counts,
        "notes": [
            "Frames are anchored only on the ego Husky planar scan topic.",
            "Future trajectory targets are built later by sliding windows in dataset_helper/notebooks.",
            "UAV1 and UAV2 context data is included when the source bag recorded those topics.",
            "The intended graph is husky_local + uav1 + uav2.",
            "No second Husky is required in this dataset structure.",
        ],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Saved hybrid two-UAV dataset to:", out_dir)
    print("World:", world_name)
    print("Frames:", frame_count)
    print("Schema:", schema_path)
    print("Manifest:", manifest_path)

    if frame_count == 0:
        print(
            "WARNING: Exported 0 frames. This usually means the bag does not contain "
            "the ego Husky planar scan topic, odometry, or episode goal metadata."
        )

    return out_dir, frame_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export hybrid two-UAV/one-UGV JSONL dataset from recorded ROS 2 bag(s)."
    )

    parser.add_argument(
        "--bag",
        default="",
        help="Optional path to a specific bag directory. If omitted, all run_* bags are exported.",
    )

    parser.add_argument(
        "--out-root",
        default=str(OUT_ROOT),
        help="Dataset output root.",
    )

    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()

    if args.bag:
        bag_paths = [Path(args.bag).expanduser().resolve()]
    else:
        bag_paths = available_bags(BAGS_DIR)

        if not bag_paths:
            raise FileNotFoundError(
                f"No bag directories found in {BAGS_DIR}. "
                "First run the simulation and record at least one bag."
            )

    print("Export root:", out_root)
    print("Bag count:", len(bag_paths))

    total_frames = 0

    for bag_path in bag_paths:
        print("\n=== Exporting", bag_path.name, "===")
        _out_dir, frame_count = export_one_bag(bag_path, out_root)
        total_frames += int(frame_count)

    print("\nCompleted hybrid two-UAV dataset export.")
    print("Total bags:", len(bag_paths))
    print("Total frames:", total_frames)
    print("Output root:", out_root)


if __name__ == "__main__":
    main()