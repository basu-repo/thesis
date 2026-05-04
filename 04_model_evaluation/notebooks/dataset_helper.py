# # """Shared data and evaluation helpers for the thesis notebooks.

# # This module keeps the repetitive dataset-loading, path-remapping, metric, and
# # result-saving utilities in one place so the notebooks can stay focused on
# # model-specific logic.

# # This version supports the hybrid UGV/UAV dataset exported by:

# #     03_dataset/exporters/export_hybrid_maneuver_dataset.py

# # Expected dataset structure:

# #     03_dataset/husky_control_dataset/
# #         run_xxx/
# #             frames.jsonl
# #             schema.json
# #             assets/
# #                 husky_local/planar_scan/*.npy
# #                 husky_local/front_points/*.npy
# #                 husky_2/planar_scan/*.npy
# #                 husky_2/front_points/*.npy
# #                 uav1/front_points/*.npy

# # Each JSONL frame may include:

# #     agents
# #     edges
# #     modalities
# #     observation
# #     state
# #     goal
# #     goal_features
# #     teacher
# #     readiness

# # The main supervised trajectory target is built by sliding windows:
# #     past_len observed frames  ->  future_len future ego (x, y) trajectory
# # """

# # from __future__ import annotations

# # import json
# # import random
# # from collections import Counter
# # from datetime import datetime
# # from functools import lru_cache
# # from pathlib import Path

# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd
# # import torch

# # try:
# #     from sklearn.metrics import (
# #         auc,
# #         average_precision_score,
# #         precision_recall_curve,
# #         roc_curve,
# #     )
# #     from sklearn.preprocessing import label_binarize

# #     SKLEARN_AVAILABLE = True
# # except Exception:
# #     SKLEARN_AVAILABLE = False


# # # ---------------------------------------------------------------------------
# # # Labels
# # # ---------------------------------------------------------------------------

# # DEFAULT_LABELS = [
# #     "bootstrap",
# #     "go_to_goal",
# #     "avoid_left",
# #     "avoid_right",
# #     "commit_forward",
# #     "reverse",
# #     "recover",
# #     "reassess",
# #     "arrived",
# #     "stop",
# # ]

# # REDUCED_LABELS = [
# #     "go_to_goal",
# #     "avoid_left",
# #     "avoid_right",
# #     "commit_forward",
# #     "arrived",
# # ]

# # PLATFORM_ONEHOT = {
# #     "UGV": [1.0, 0.0],
# #     "UAV": [0.0, 1.0],
# # }

# # DEFAULT_EXTERNAL_DATASET_ROOT = (
# #     Path.home() / "Documents/Thesis/03_dataset/husky_control_dataset"
# # )

# # DATASET_ROOT: Path | None = None
# # ORIGINAL_DATASET_ROOT: Path | None = None
# # RESULTS_ROOT: Path | None = None
# # WEIGHTS_ROOT: Path | None = None


# # # ---------------------------------------------------------------------------
# # # Configuration
# # # ---------------------------------------------------------------------------

# # def configure_helper(
# #     *,
# #     dataset_root: Path,
# #     original_dataset_root: Path | None = None,
# #     results_root: Path | None = None,
# #     weights_root: Path | None = None,
# # ) -> None:
# #     """Set notebook-specific roots once so helper functions stay simple."""
# #     global DATASET_ROOT, ORIGINAL_DATASET_ROOT, RESULTS_ROOT, WEIGHTS_ROOT

# #     DATASET_ROOT = Path(dataset_root).expanduser().resolve()
# #     ORIGINAL_DATASET_ROOT = (
# #         Path(original_dataset_root).expanduser().resolve()
# #         if original_dataset_root is not None
# #         else DATASET_ROOT
# #     )
# #     RESULTS_ROOT = (
# #         Path(results_root).expanduser().resolve()
# #         if results_root is not None
# #         else None
# #     )
# #     WEIGHTS_ROOT = (
# #         Path(weights_root).expanduser().resolve()
# #         if weights_root is not None
# #         else None
# #     )

# #     load_npy_cached.cache_clear()


# # def _require_roots() -> tuple[Path, Path]:
# #     if DATASET_ROOT is None or ORIGINAL_DATASET_ROOT is None:
# #         raise RuntimeError(
# #             "dataset_helper is not configured. "
# #             "Call configure_helper(...) in the notebook setup cell first."
# #         )

# #     return DATASET_ROOT, ORIGINAL_DATASET_ROOT


# # def set_seed(seed: int) -> None:
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)

# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)


# # # ---------------------------------------------------------------------------
# # # Label mapping
# # # ---------------------------------------------------------------------------

# # def build_label_mapping(label_mode: str):
# #     """Return labels and mapping.

# #     label_mode='full':
# #         keeps all controller states.

# #     label_mode='reduced':
# #         removes transitional/recovery states and keeps the most useful maneuver
# #         classes for classification-style auxiliary training.
# #     """
# #     if label_mode == "full":
# #         labels = list(DEFAULT_LABELS)
# #         mapping = {label: label for label in DEFAULT_LABELS}
# #         return labels, mapping

# #     labels = list(REDUCED_LABELS)
# #     mapping = {
# #         "bootstrap": None,
# #         "go_to_goal": "go_to_goal",
# #         "avoid_left": "avoid_left",
# #         "avoid_right": "avoid_right",
# #         "commit_forward": "commit_forward",
# #         "reverse": None,
# #         "recover": None,
# #         "reassess": None,
# #         "arrived": "arrived",
# #         "stop": None,
# #     }

# #     return labels, mapping


# # # ---------------------------------------------------------------------------
# # # File discovery and path remapping
# # # ---------------------------------------------------------------------------

# # def _frame_files_under(root: Path) -> list[Path]:
# #     root = Path(root)

# #     if (root / "frames.jsonl").exists():
# #         return [root / "frames.jsonl"]

# #     return sorted(root.glob("*/frames.jsonl"))


# # def discover_frame_files(dataset_root: Path) -> list[Path]:
# #     """Find extracted frame files.

# #     The function tries several sensible roots to avoid notebook breakage when
# #     paths change after moving the dataset or restarting the kernel.
# #     """
# #     candidate_roots: list[Path] = []
# #     seen: set[Path] = set()

# #     def add_candidate(path: Path | None) -> None:
# #         if path is None:
# #             return

# #         path = Path(path).expanduser()

# #         try:
# #             path = path.resolve()
# #         except Exception:
# #             pass

# #         if path in seen:
# #             return

# #         seen.add(path)
# #         candidate_roots.append(path)

# #     add_candidate(Path(dataset_root))
# #     add_candidate(DATASET_ROOT)
# #     add_candidate(ORIGINAL_DATASET_ROOT)
# #     add_candidate(DEFAULT_EXTERNAL_DATASET_ROOT)

# #     for root in candidate_roots:
# #         frame_files = _frame_files_under(root)
# #         if frame_files:
# #             return frame_files

# #     return []


# # def remap_dataset_path(path_str: str) -> Path:
# #     """Map stored asset paths to the current local dataset root."""
# #     dataset_root, original_dataset_root = _require_roots()

# #     path = Path(path_str).expanduser()

# #     if path.exists():
# #         return path

# #     try:
# #         rel = path.relative_to(original_dataset_root)
# #     except ValueError:
# #         parts = path.parts

# #         for dataset_marker in (
# #             "hybrid_maneuvers_dataset",
# #             "husky_control_dataset",
# #         ):
# #             if dataset_marker in parts:
# #                 marker = parts.index(dataset_marker)
# #                 rel = Path(*parts[marker + 1 :])

# #                 candidate = dataset_root / rel
# #                 if candidate.exists():
# #                     return candidate

# #                 if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
# #                     fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
# #                     if fallback_candidate.exists():
# #                         return fallback_candidate

# #         return path

# #     candidate = dataset_root / rel

# #     if candidate.exists():
# #         return candidate

# #     if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
# #         fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
# #         if fallback_candidate.exists():
# #             return fallback_candidate

# #     return candidate


# # @lru_cache(maxsize=32768)
# # def load_npy_cached(path: str):
# #     return np.load(remap_dataset_path(path))


# # # ---------------------------------------------------------------------------
# # # Sensor preprocessing
# # # ---------------------------------------------------------------------------

# # def resample_scan(
# #     scan: np.ndarray,
# #     num_beams: int,
# #     range_clip: float,
# # ) -> np.ndarray:
# #     """Convert saved LaserScan array into a normalized 2 x num_beams tensor.

# #     Input saved by exporter:
# #         shape = [N, 2]
# #         column 0 = range
# #         column 1 = intensity
# #     """
# #     ranges = np.asarray(scan[:, 0], dtype=np.float32)
# #     intensities = np.asarray(scan[:, 1], dtype=np.float32)

# #     ranges = np.nan_to_num(
# #         ranges,
# #         nan=range_clip,
# #         posinf=range_clip,
# #         neginf=0.0,
# #     )
# #     ranges = np.clip(ranges, 0.0, range_clip)

# #     intensities = np.nan_to_num(
# #         intensities,
# #         nan=0.0,
# #         posinf=255.0,
# #         neginf=0.0,
# #     )
# #     intensities = np.clip(intensities, 0.0, 255.0)

# #     if ranges.shape[0] != num_beams:
# #         src_x = np.linspace(0.0, 1.0, ranges.shape[0], dtype=np.float32)
# #         dst_x = np.linspace(0.0, 1.0, num_beams, dtype=np.float32)

# #         ranges = np.interp(dst_x, src_x, ranges).astype(np.float32)
# #         intensities = np.interp(dst_x, src_x, intensities).astype(np.float32)

# #     return np.stack(
# #         [
# #             ranges / max(range_clip, 1e-6),
# #             intensities / 255.0,
# #         ],
# #         axis=0,
# #     ).astype(np.float32)


# # def summarize_pointcloud_corridor(
# #     points: np.ndarray,
# #     *,
# #     max_points: int = 512,
# #     x_min: float = 0.0,
# #     x_max: float = 25.0,
# #     y_abs_max: float = 12.0,
# #     z_min: float = -5.0,
# #     z_max: float = 25.0,
# # ) -> np.ndarray:
# #     """Filter and downsample a point cloud into a fixed-size Nx4 array.

# #     This helper is useful if a notebook wants to use UAV or Husky pointclouds.
# #     It keeps a forward corridor and returns max_points rows.

# #     Output:
# #         shape = [max_points, 4]
# #         columns = x, y, z, intensity
# #     """
# #     if points is None or points.size == 0:
# #         return np.zeros((max_points, 4), dtype=np.float32)

# #     points = np.asarray(points, dtype=np.float32)

# #     if points.ndim != 2 or points.shape[1] < 3:
# #         return np.zeros((max_points, 4), dtype=np.float32)

# #     if points.shape[1] == 3:
# #         zeros = np.zeros((points.shape[0], 1), dtype=np.float32)
# #         points = np.concatenate([points, zeros], axis=1)

# #     x = points[:, 0]
# #     y = points[:, 1]
# #     z = points[:, 2]

# #     mask = (
# #         (x >= x_min)
# #         & (x <= x_max)
# #         & (np.abs(y) <= y_abs_max)
# #         & (z >= z_min)
# #         & (z <= z_max)
# #     )

# #     filtered = points[mask, :4]

# #     if filtered.shape[0] == 0:
# #         return np.zeros((max_points, 4), dtype=np.float32)

# #     if filtered.shape[0] >= max_points:
# #         idx = np.linspace(0, filtered.shape[0] - 1, max_points).astype(np.int64)
# #         filtered = filtered[idx]
# #     else:
# #         pad = np.zeros((max_points - filtered.shape[0], 4), dtype=np.float32)
# #         filtered = np.concatenate([filtered, pad], axis=0)

# #     return filtered.astype(np.float32)


# # def hazard_summary_from_pointcloud(
# #     points: np.ndarray,
# #     *,
# #     x_min: float = 0.0,
# #     x_max: float = 25.0,
# #     center_half_width: float = 2.0,
# #     side_width: float = 6.0,
# #     z_min: float = -2.0,
# #     z_max: float = 5.0,
# # ) -> np.ndarray:
# #     """Convert a point cloud into simple [left, center, right] hazard counts.

# #     This is useful for adding UAV-derived forward context without feeding the
# #     full point cloud into every model.
# #     """
# #     if points is None or points.size == 0:
# #         return np.zeros(3, dtype=np.float32)

# #     points = np.asarray(points, dtype=np.float32)

# #     if points.ndim != 2 or points.shape[1] < 3:
# #         return np.zeros(3, dtype=np.float32)

# #     x = points[:, 0]
# #     y = points[:, 1]
# #     z = points[:, 2]

# #     valid = (
# #         (x >= x_min)
# #         & (x <= x_max)
# #         & (z >= z_min)
# #         & (z <= z_max)
# #         & (np.abs(y) <= side_width)
# #     )

# #     yv = y[valid]

# #     left = np.sum((yv > center_half_width) & (yv <= side_width))
# #     center = np.sum(np.abs(yv) <= center_half_width)
# #     right = np.sum((yv < -center_half_width) & (yv >= -side_width))

# #     counts = np.asarray([left, center, right], dtype=np.float32)
# #     return np.log1p(counts)


# # def load_asset_ref(ref: dict | None):
# #     if ref is None:
# #         return None

# #     path = ref.get("path")
# #     if not path:
# #         return None

# #     try:
# #         return load_npy_cached(str(path))
# #     except Exception:
# #         return None


# # # ---------------------------------------------------------------------------
# # # Frame accessors supporting old and new schema
# # # ---------------------------------------------------------------------------

# # def canonical_agent_order(ego_id: str) -> list[str]:
# #     other_husky = "husky_2" if ego_id == "husky_local" else "husky_local"
# #     return [ego_id, other_husky, "uav1"]


# # def _zero_state() -> dict:
# #     return {
# #         "x": 0.0,
# #         "y": 0.0,
# #         "z": 0.0,
# #         "qx": 0.0,
# #         "qy": 0.0,
# #         "qz": 0.0,
# #         "qw": 1.0,
# #         "yaw": 0.0,
# #         "vx": 0.0,
# #         "vy": 0.0,
# #         "vz": 0.0,
# #         "wz": 0.0,
# #     }


# # def _default_agent_node(agent_id: str) -> dict:
# #     platform = "UAV" if agent_id.startswith("uav") else "UGV"

# #     return {
# #         "id": agent_id,
# #         "available": False,
# #         "platform_type": platform,
# #         "state": _zero_state(),
# #         "start": None,
# #         "goal": None,
# #         "goal_features": None,
# #         "command": {"linear_x": 0.0, "angular_z": 0.0},
# #         "controller_state": None,
# #         "obstacle_action": None,
# #         "obstacle_clearance": None,
# #         "ready": None,
# #     }


# # def frame_agents(frame: dict) -> dict:
# #     """Return normalized agent dictionary for both old and new exports."""
# #     if "agents" in frame and isinstance(frame["agents"], dict):
# #         agents = dict(frame["agents"])
# #     else:
# #         ego_id = frame.get("ego_id", "husky_local")
# #         other = frame.get("other_husky") or {}
# #         other_id = other.get("id", "husky_2" if ego_id == "husky_local" else "husky_local")

# #         agents = {
# #             ego_id: {
# #                 "id": ego_id,
# #                 "available": frame.get("state") is not None,
# #                 "platform_type": "UGV",
# #                 "state": frame.get("state"),
# #                 "start": None,
# #                 "goal": frame.get("goal"),
# #                 "goal_features": frame.get("goal_features"),
# #                 "command": frame.get("teacher", {}).get("command")
# #                 or {"linear_x": 0.0, "angular_z": 0.0},
# #                 "controller_state": frame.get("teacher", {}).get("controller_state"),
# #                 "obstacle_action": frame.get("teacher", {}).get("obstacle_action"),
# #                 "obstacle_clearance": frame.get("teacher", {}).get("obstacle_clearance"),
# #                 "ready": None,
# #             },
# #             other_id: {
# #                 "id": other_id,
# #                 "available": bool(other.get("available", False)),
# #                 "platform_type": "UGV",
# #                 "state": other.get("state"),
# #                 "start": None,
# #                 "goal": other.get("goal"),
# #                 "goal_features": other.get("goal_features"),
# #                 "command": other.get("teacher_command")
# #                 or {"linear_x": 0.0, "angular_z": 0.0},
# #                 "controller_state": None,
# #                 "obstacle_action": None,
# #                 "obstacle_clearance": None,
# #                 "ready": None,
# #             },
# #         }

# #     for agent_id in canonical_agent_order(frame.get("ego_id", "husky_local")):
# #         if agent_id not in agents:
# #             agents[agent_id] = _default_agent_node(agent_id)
# #             continue

# #         node = dict(_default_agent_node(agent_id)) | dict(agents[agent_id])

# #         if node.get("state") is None:
# #             node["state"] = _zero_state()
# #             node["available"] = False

# #         if node.get("command") is None:
# #             node["command"] = {"linear_x": 0.0, "angular_z": 0.0}

# #         if node.get("platform_type") is None:
# #             node["platform_type"] = "UAV" if agent_id.startswith("uav") else "UGV"

# #         agents[agent_id] = node

# #     return agents


# # def frame_state(frame: dict) -> dict:
# #     if "agents" in frame:
# #         ego_id = frame["ego_id"]
# #         state = frame["agents"][ego_id].get("state")
# #         return state if state is not None else _zero_state()

# #     state = frame.get("state")
# #     return state if state is not None else _zero_state()


# # def frame_scan_ref(frame: dict):
# #     if "modalities" in frame:
# #         ref = frame["modalities"].get("ego_planar_scan")
# #         if ref is not None:
# #             return ref

# #     if "observation" in frame:
# #         return frame["observation"].get("ego_planar_scan")

# #     return None


# # def frame_ego_pointcloud_ref(frame: dict):
# #     if "modalities" in frame:
# #         ref = frame["modalities"].get("ego_front_pointcloud")
# #         if ref is not None:
# #             return ref

# #     if "observation" in frame:
# #         return frame["observation"].get("ego_front_pointcloud")

# #     return None


# # def frame_uav_pointcloud_ref(frame: dict):
# #     if "modalities" in frame:
# #         ref = frame["modalities"].get("uav1_front_pointcloud")
# #         if ref is not None:
# #             return ref

# #     if "observation" in frame:
# #         return frame["observation"].get("uav1_front_pointcloud")

# #     return None


# # def frame_teacher_label(frame: dict) -> str:
# #     teacher = frame.get("teacher", {})

# #     label = teacher.get("label")
# #     if label is not None:
# #         return str(label)

# #     controller_state = teacher.get("controller_state")
# #     if controller_state:
# #         return str(controller_state)

# #     obstacle_action = teacher.get("obstacle_action")
# #     if obstacle_action:
# #         obstacle_action = str(obstacle_action).strip().lower()
# #         if obstacle_action.endswith("left"):
# #             return "avoid_left"
# #         if obstacle_action.endswith("right"):
# #             return "avoid_right"

# #     return "go_to_goal"


# # # ---------------------------------------------------------------------------
# # # Graph features
# # # ---------------------------------------------------------------------------

# # def node_feature(node: dict, ego_state: dict) -> list[float]:
# #     """Build a fixed 14D node feature vector.

# #     This matches the implementation-level description:
# #         node feature dimension = 14

# #     Features:
# #         relative position to ego: dx, dy, dz
# #         velocity: vx, vy, vz, wz
# #         goal offset: gx, gy, gz
# #         command: linear_x, angular_z
# #         platform one-hot: UGV, UAV
# #     """
# #     state = node.get("state") or _zero_state()
# #     goal = node.get("goal") or {
# #         "x": state["x"],
# #         "y": state["y"],
# #         "z": state["z"],
# #     }
# #     command = node.get("command") or {
# #         "linear_x": 0.0,
# #         "angular_z": 0.0,
# #     }
# #     platform = PLATFORM_ONEHOT.get(
# #         node.get("platform_type", "UGV"),
# #         [0.0, 0.0],
# #     )

# #     return [
# #         float(state.get("x", 0.0) - ego_state.get("x", 0.0)),
# #         float(state.get("y", 0.0) - ego_state.get("y", 0.0)),
# #         float(state.get("z", 0.0) - ego_state.get("z", 0.0)),
# #         float(state.get("vx", 0.0)),
# #         float(state.get("vy", 0.0)),
# #         float(state.get("vz", 0.0)),
# #         float(state.get("wz", 0.0)),
# #         float(goal.get("x", state.get("x", 0.0)) - state.get("x", 0.0)),
# #         float(goal.get("y", state.get("y", 0.0)) - state.get("y", 0.0)),
# #         float(goal.get("z", state.get("z", 0.0)) - state.get("z", 0.0)),
# #         float(command.get("linear_x", 0.0)),
# #         float(command.get("angular_z", 0.0)),
# #         float(platform[0]),
# #         float(platform[1]),
# #     ]


# # def graph_node_features_for_frame(frame: dict, order: list[str] | None = None) -> np.ndarray:
# #     """Return graph node feature matrix [num_nodes, 14]."""
# #     agents = frame_agents(frame)
# #     ego_id = frame.get("ego_id", "husky_local")
# #     ego_state = agents[ego_id].get("state") or _zero_state()
# #     order = order or canonical_agent_order(ego_id)

# #     features = []
# #     for agent_id in order:
# #         node = agents.get(agent_id, _default_agent_node(agent_id))
# #         features.append(node_feature(node, ego_state))

# #     return np.asarray(features, dtype=np.float32)


# # def build_edge_lookup(frame: dict) -> dict[tuple[str, str], dict]:
# #     edges = frame.get("edges") or []
# #     return {
# #         (edge["source"], edge["target"]): edge
# #         for edge in edges
# #         if "source" in edge and "target" in edge
# #     }


# # def _fallback_edge(src_node: dict, dst_node: dict) -> dict:
# #     src_state = src_node.get("state") or _zero_state()
# #     dst_state = dst_node.get("state") or _zero_state()

# #     dx = float(dst_state.get("x", 0.0) - src_state.get("x", 0.0))
# #     dy = float(dst_state.get("y", 0.0) - src_state.get("y", 0.0))
# #     dz = float(dst_state.get("z", 0.0) - src_state.get("z", 0.0))

# #     distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
# #     bearing = float(np.arctan2(dy, dx))

# #     return {
# #         "dx": dx,
# #         "dy": dy,
# #         "dz": dz,
# #         "distance": distance,
# #         "inv_distance": float(1.0 / max(distance, 1e-6)),
# #         "bearing_sin": float(np.sin(bearing)),
# #         "bearing_cos": float(np.cos(bearing)),
# #         "same_platform": float(
# #             src_node.get("platform_type") == dst_node.get("platform_type")
# #         ),
# #         "latency_s": 0.0,
# #         "packet_loss": 0.0,
# #         "link_quality": 1.0,
# #     }


# # def edge_features_for_order(frame: dict, order: list[str]) -> list[list[list[float]]]:
# #     """Return edge features [num_nodes, num_nodes, 11].

# #     This supports both:
# #     - old exports with 8 edge fields
# #     - new exports with 11 edge fields including placeholder network features
# #     """
# #     agents = frame_agents(frame)
# #     edge_map = build_edge_lookup(frame)

# #     src_edges = []

# #     for src in order:
# #         row = []

# #         for dst in order:
# #             edge = edge_map.get((src, dst))

# #             if edge is None:
# #                 edge = _fallback_edge(
# #                     agents.get(src, _default_agent_node(src)),
# #                     agents.get(dst, _default_agent_node(dst)),
# #                 )

# #             row.append(
# #                 [
# #                     float(edge.get("dx", 0.0)),
# #                     float(edge.get("dy", 0.0)),
# #                     float(edge.get("dz", 0.0)),
# #                     float(edge.get("distance", 0.0)),
# #                     float(edge.get("inv_distance", 0.0)),
# #                     float(edge.get("bearing_sin", 0.0)),
# #                     float(edge.get("bearing_cos", 1.0)),
# #                     float(edge.get("same_platform", 0.0)),
# #                     float(edge.get("latency_s", 0.0)),
# #                     float(edge.get("packet_loss", 0.0)),
# #                     float(edge.get("link_quality", 1.0)),
# #                 ]
# #             )

# #         src_edges.append(row)

# #     return src_edges


# # def graph_edge_features_for_frame(frame: dict, order: list[str] | None = None) -> np.ndarray:
# #     ego_id = frame.get("ego_id", "husky_local")
# #     order = order or canonical_agent_order(ego_id)
# #     return np.asarray(edge_features_for_order(frame, order), dtype=np.float32)


# # # ---------------------------------------------------------------------------
# # # Dataset grouping and sample construction
# # # ---------------------------------------------------------------------------

# # def group_streams(
# #     dataset_root: Path,
# #     allowed_labels: set[str] | None = None,
# #     label_mapping: dict | None = None,
# #     require_uav: bool = False,
# # ):
# #     """Group frames by episode_id and ego_id.

# #     The output is a list of sorted streams. Each stream is one continuous
# #     ego trajectory from one bag.
# #     """
# #     streams = []
# #     frame_files = discover_frame_files(dataset_root)

# #     allowed_labels = set(allowed_labels) if allowed_labels is not None else None
# #     label_mapping = label_mapping or {}

# #     for frames_path in frame_files:
# #         with frames_path.open() as f:
# #             rows = [json.loads(line) for line in f if line.strip()]

# #         buckets = {}

# #         for row in rows:
# #             raw_label = frame_teacher_label(row)
# #             mapped_label = label_mapping.get(raw_label, raw_label)

# #             if allowed_labels is not None and (
# #                 mapped_label is None or mapped_label not in allowed_labels
# #             ):
# #                 continue

# #             if frame_scan_ref(row) is None:
# #                 continue

# #             if frame_state(row) is None:
# #                 continue

# #             if require_uav:
# #                 readiness = row.get("readiness", {})
# #                 if not readiness.get("has_uav1_state", False):
# #                     continue

# #             row = dict(row)
# #             row["teacher"] = dict(row.get("teacher", {}))
# #             row["teacher"]["raw_label"] = raw_label
# #             row["teacher"]["label"] = mapped_label

# #             key = f"{row['episode_id']}::{row['ego_id']}"
# #             buckets.setdefault(key, []).append(row)

# #         for key in sorted(buckets):
# #             stream = sorted(
# #                 buckets[key],
# #                 key=lambda item: int(item["timestamp_ns"]),
# #             )

# #             if stream:
# #                 streams.append(stream)

# #     if not streams:
# #         raise RuntimeError(
# #             f"No usable frame streams found under {dataset_root}. "
# #             f"Discovered {len(frame_files)} frame file(s); "
# #             f"check label filtering, dataset paths, and whether export completed."
# #         )

# #     return streams


# # def build_sample_table(
# #     streams: list[list[dict]],
# #     past_len: int,
# #     future_len: int,
# # ):
# #     """Build sliding-window sample metadata.

# #     The target is future ego trajectory relative to the anchor frame.
# #     """
# #     sample_table = []

# #     for stream_idx, stream in enumerate(streams):
# #         usable = len(stream) - past_len - future_len + 1

# #         for start in range(max(0, usable)):
# #             anchor = stream[start + past_len - 1]
# #             future_frames = stream[start + past_len : start + past_len + future_len]

# #             anchor_state = frame_state(anchor)
# #             anchor_ts = int(anchor["timestamp_ns"])

# #             future_xy = []
# #             future_dt = []

# #             valid = True

# #             for future_frame in future_frames:
# #                 state = frame_state(future_frame)

# #                 if state is None:
# #                     valid = False
# #                     break

# #                 future_xy.append(
# #                     [
# #                         float(state["x"] - anchor_state["x"]),
# #                         float(state["y"] - anchor_state["y"]),
# #                     ]
# #                 )
# #                 future_dt.append(
# #                     (int(future_frame["timestamp_ns"]) - anchor_ts) * 1e-9
# #                 )

# #             if not valid:
# #                 continue

# #             sample_table.append(
# #                 {
# #                     "sample_id": f"stream{stream_idx:03d}_start{start:05d}",
# #                     "stream_index": stream_idx,
# #                     "stream_idx": stream_idx,
# #                     "start_index": start,
# #                     "start": start,
# #                     "anchor_index": start + past_len - 1,
# #                     "ego_id": anchor["ego_id"],
# #                     "label": anchor["teacher"].get("label"),
# #                     "raw_label": anchor["teacher"].get("raw_label"),
# #                     "future_xy": future_xy,
# #                     "future_dt": future_dt,
# #                 }
# #             )

# #     return sample_table


# # def save_or_load_fixed_split(
# #     sample_table,
# #     split_path: Path,
# #     seed: int,
# #     train_ratio: float,
# #     val_ratio: float,
# #     past_len: int,
# #     future_len: int,
# # ):
# #     """Create or reuse a deterministic train/val/test split."""
# #     split_path = Path(split_path)

# #     if split_path.exists():
# #         with split_path.open() as f:
# #             split_info = json.load(f)

# #         current_sample_ids = [row["sample_id"] for row in sample_table]

# #         if (
# #             split_info.get("sample_count") == len(sample_table)
# #             and split_info.get("past_len") == past_len
# #             and split_info.get("future_len") == future_len
# #             and split_info.get("sample_ids") == current_sample_ids
# #         ):
# #             return split_info

# #     rng = random.Random(seed)
# #     indices = list(range(len(sample_table)))
# #     rng.shuffle(indices)

# #     if len(indices) < 3:
# #         raise RuntimeError(
# #             f"Need at least 3 samples to split train/val/test, got {len(indices)}."
# #         )

# #     train_len = max(1, int(len(indices) * train_ratio))
# #     val_len = max(1, int(len(indices) * val_ratio))
# #     test_len = len(indices) - train_len - val_len

# #     if test_len < 1:
# #         test_len = 1
# #         if train_len > val_len:
# #             train_len -= 1
# #         else:
# #             val_len -= 1

# #     split_info = {
# #         "seed": seed,
# #         "sample_count": len(sample_table),
# #         "past_len": past_len,
# #         "future_len": future_len,
# #         "train_indices": indices[:train_len],
# #         "val_indices": indices[train_len : train_len + val_len],
# #         "test_indices": indices[train_len + val_len :],
# #         "sample_ids": [row["sample_id"] for row in sample_table],
# #     }

# #     split_path.parent.mkdir(parents=True, exist_ok=True)
# #     split_path.write_text(json.dumps(split_info, indent=2))

# #     return split_info


# # # ---------------------------------------------------------------------------
# # # Optional ready-to-use feature builders for notebooks
# # # ---------------------------------------------------------------------------

# # def build_past_ego_xy(
# #     stream: list[dict],
# #     start: int,
# #     past_len: int,
# # ) -> np.ndarray:
# #     """Return past ego xy relative to the anchor frame.

# #     Output:
# #         shape = [past_len, 2]
# #     """
# #     past_frames = stream[start : start + past_len]
# #     anchor_state = frame_state(past_frames[-1])

# #     xy = []

# #     for frame in past_frames:
# #         state = frame_state(frame)
# #         xy.append(
# #             [
# #                 float(state["x"] - anchor_state["x"]),
# #                 float(state["y"] - anchor_state["y"]),
# #             ]
# #         )

# #     return np.asarray(xy, dtype=np.float32)


# # def build_past_graph_sequence(
# #     stream: list[dict],
# #     start: int,
# #     past_len: int,
# #     order: list[str] | None = None,
# # ) -> tuple[np.ndarray, np.ndarray]:
# #     """Return past graph node and edge sequences.

# #     Outputs:
# #         node_seq: [past_len, num_nodes, 14]
# #         edge_seq: [past_len, num_nodes, num_nodes, 11]
# #     """
# #     past_frames = stream[start : start + past_len]

# #     if order is None:
# #         order = canonical_agent_order(past_frames[-1].get("ego_id", "husky_local"))

# #     node_seq = []
# #     edge_seq = []

# #     for frame in past_frames:
# #         node_seq.append(graph_node_features_for_frame(frame, order))
# #         edge_seq.append(graph_edge_features_for_frame(frame, order))

# #     return (
# #         np.asarray(node_seq, dtype=np.float32),
# #         np.asarray(edge_seq, dtype=np.float32),
# #     )


# # def build_past_scan_sequence(
# #     stream: list[dict],
# #     start: int,
# #     past_len: int,
# #     *,
# #     num_beams: int = 256,
# #     range_clip: float = 30.0,
# # ) -> np.ndarray:
# #     """Return past ego lidar sequence.

# #     Output:
# #         shape = [past_len, 2, num_beams]
# #     """
# #     past_frames = stream[start : start + past_len]
# #     scans = []

# #     for frame in past_frames:
# #         ref = frame_scan_ref(frame)
# #         scan = load_asset_ref(ref)

# #         if scan is None:
# #             scans.append(np.zeros((2, num_beams), dtype=np.float32))
# #         else:
# #             scans.append(resample_scan(scan, num_beams, range_clip))

# #     return np.asarray(scans, dtype=np.float32)


# # def build_past_uav_hazard_sequence(
# #     stream: list[dict],
# #     start: int,
# #     past_len: int,
# # ) -> np.ndarray:
# #     """Return past UAV forward hazard summaries.

# #     Output:
# #         shape = [past_len, 3]
# #         columns = left_count_log, center_count_log, right_count_log
# #     """
# #     past_frames = stream[start : start + past_len]
# #     summaries = []

# #     for frame in past_frames:
# #         ref = frame_uav_pointcloud_ref(frame)
# #         points = load_asset_ref(ref)

# #         if points is None:
# #             summaries.append(np.zeros(3, dtype=np.float32))
# #         else:
# #             summaries.append(hazard_summary_from_pointcloud(points))

# #     return np.asarray(summaries, dtype=np.float32)


# # # ---------------------------------------------------------------------------
# # # Metrics and class helpers
# # # ---------------------------------------------------------------------------

# # def build_class_weights(
# #     label_indices: list[int],
# #     num_classes: int,
# # ):
# #     counts = Counter(label_indices)
# #     total = sum(counts.values())

# #     weights = []

# #     for idx in range(num_classes):
# #         count = counts.get(idx, 0)
# #         weights.append(0.0 if count == 0 else total / (num_classes * count))

# #     return torch.tensor(weights, dtype=torch.float32)


# # def compute_trajectory_metrics(
# #     pred_future_xy: np.ndarray,
# #     true_future_xy: np.ndarray,
# # ):
# #     diff = pred_future_xy - true_future_xy
# #     dist = np.linalg.norm(diff, axis=-1)

# #     return {
# #         "ADE": float(dist.mean()),
# #         "FDE": float(dist[:, -1].mean()),
# #         "RMSE": float(np.sqrt(np.mean(np.sum(diff**2, axis=-1)))),
# #     }


# # def compute_classification_metrics_from_probs(
# #     probabilities: np.ndarray,
# #     targets: np.ndarray,
# #     labels: list[str],
# # ):
# #     preds = probabilities.argmax(axis=1)
# #     num_classes = len(labels)

# #     confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

# #     for truth, guess in zip(targets, preds):
# #         confusion[int(truth), int(guess)] += 1

# #     precisions, recalls, f1s = [], [], []

# #     for idx in range(num_classes):
# #         tp = float(confusion[idx, idx])
# #         fn = float(confusion[idx, :].sum() - tp)
# #         fp = float(confusion[:, idx].sum() - tp)

# #         precision = tp / max(tp + fp, 1.0)
# #         recall = tp / max(tp + fn, 1.0)
# #         f1 = (
# #             0.0
# #             if (precision + recall) == 0.0
# #             else (2.0 * precision * recall / (precision + recall))
# #         )

# #         precisions.append(precision)
# #         recalls.append(recall)
# #         f1s.append(f1)

# #     metrics = {
# #         "accuracy": float((preds == targets).mean()),
# #         "macro_precision": float(np.mean(precisions)),
# #         "macro_recall": float(np.mean(recalls)),
# #         "macro_f1": float(np.mean(f1s)),
# #         "confusion_matrix": confusion.tolist(),
# #         "ADE": None,
# #         "FDE": None,
# #         "RMSE": None,
# #     }

# #     return metrics, preds, confusion


# # # ---------------------------------------------------------------------------
# # # Saving outputs
# # # ---------------------------------------------------------------------------

# # def save_training_history(
# #     history: dict,
# #     out_path: Path,
# # ):
# #     pd.DataFrame(history).to_csv(out_path, index=False)


# # def save_confusion_matrix(
# #     confusion: np.ndarray,
# #     labels: list[str],
# #     csv_path: Path,
# #     png_path: Path,
# #     title: str,
# # ):
# #     df = pd.DataFrame(confusion, index=labels, columns=labels)
# #     df.to_csv(csv_path)

# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     im = ax.imshow(confusion, cmap="Blues")

# #     ax.set_xticks(range(len(labels)))
# #     ax.set_yticks(range(len(labels)))
# #     ax.set_xticklabels(labels, rotation=45, ha="right")
# #     ax.set_yticklabels(labels)

# #     ax.set_xlabel("Predicted")
# #     ax.set_ylabel("True")
# #     ax.set_title(title)

# #     for i in range(len(labels)):
# #         for j in range(len(labels)):
# #             ax.text(
# #                 j,
# #                 i,
# #                 str(confusion[i, j]),
# #                 ha="center",
# #                 va="center",
# #                 color="black",
# #                 fontsize=8,
# #             )

# #     fig.colorbar(im, ax=ax)
# #     plt.tight_layout()
# #     plt.savefig(png_path, dpi=180, bbox_inches="tight")
# #     plt.close(fig)


# # def save_roc_pr_curves(
# #     probabilities: np.ndarray,
# #     targets: np.ndarray,
# #     labels: list[str],
# #     out_dir: Path,
# # ):
# #     summary = {
# #         "roc_auc_macro": None,
# #         "pr_auc_macro": None,
# #         "status": "skipped",
# #     }

# #     if not SKLEARN_AVAILABLE:
# #         return summary

# #     y_true = label_binarize(targets, classes=list(range(len(labels))))

# #     roc_aucs = []
# #     pr_aucs = []

# #     fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
# #     fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

# #     for idx, label in enumerate(labels):
# #         try:
# #             fpr, tpr, _ = roc_curve(y_true[:, idx], probabilities[:, idx])
# #             roc_auc_value = auc(fpr, tpr)

# #             precision, recall, _ = precision_recall_curve(
# #                 y_true[:, idx],
# #                 probabilities[:, idx],
# #             )
# #             pr_auc_value = average_precision_score(
# #                 y_true[:, idx],
# #                 probabilities[:, idx],
# #             )

# #             roc_aucs.append(roc_auc_value)
# #             pr_aucs.append(pr_auc_value)

# #             ax_roc.plot(
# #                 fpr,
# #                 tpr,
# #                 label=f"{label} (AUC={roc_auc_value:.3f})",
# #             )
# #             ax_pr.plot(
# #                 recall,
# #                 precision,
# #                 label=f"{label} (AP={pr_auc_value:.3f})",
# #             )
# #         except Exception:
# #             continue

# #     ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
# #     ax_roc.set_title("One-vs-Rest ROC Curves")
# #     ax_roc.set_xlabel("False Positive Rate")
# #     ax_roc.set_ylabel("True Positive Rate")
# #     ax_roc.legend(fontsize=8)

# #     plt.tight_layout()
# #     fig_roc.savefig(out_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
# #     plt.close(fig_roc)

# #     ax_pr.set_title("One-vs-Rest Precision-Recall Curves")
# #     ax_pr.set_xlabel("Recall")
# #     ax_pr.set_ylabel("Precision")
# #     ax_pr.legend(fontsize=8)

# #     plt.tight_layout()
# #     fig_pr.savefig(out_dir / "pr_curves.png", dpi=180, bbox_inches="tight")
# #     plt.close(fig_pr)

# #     if roc_aucs:
# #         summary["roc_auc_macro"] = float(np.mean(roc_aucs))

# #     if pr_aucs:
# #         summary["pr_auc_macro"] = float(np.mean(pr_aucs))

# #     summary["status"] = "saved"

# #     return summary


# # def save_predictions_csv(
# #     sample_ids,
# #     targets,
# #     preds,
# #     probabilities,
# #     labels,
# #     out_path: Path,
# # ):
# #     rows = []

# #     for sid, truth, pred, probs in zip(
# #         sample_ids,
# #         targets,
# #         preds,
# #         probabilities,
# #     ):
# #         row = {
# #             "sample_id": sid,
# #             "true_label": labels[int(truth)],
# #             "pred_label": labels[int(pred)],
# #         }

# #         for idx, label in enumerate(labels):
# #             row[f"prob_{label}"] = float(probs[idx])

# #         rows.append(row)

# #     pd.DataFrame(rows).to_csv(out_path, index=False)


# # def save_history_plot(
# #     history: dict,
# #     out_path: Path,
# #     title_prefix: str,
# # ):
# #     if not history or len(history.get("epoch", [])) == 0:
# #         return

# #     fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# #     axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
# #     axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
# #     axes[0].set_title(f"{title_prefix}: Loss")
# #     axes[0].legend()

# #     axes[1].plot(
# #         history["epoch"],
# #         history["val_accuracy"],
# #         label="val_accuracy",
# #     )
# #     axes[1].set_title(f"{title_prefix}: Validation Accuracy")
# #     axes[1].legend()

# #     axes[2].plot(
# #         history["epoch"],
# #         history["val_macro_f1"],
# #         label="val_macro_f1",
# #     )
# #     axes[2].set_title(f"{title_prefix}: Validation Macro-F1")
# #     axes[2].legend()

# #     plt.tight_layout()
# #     plt.savefig(out_path, dpi=180, bbox_inches="tight")
# #     plt.close(fig)


# # def save_trajectory_overlay_plots(
# #     pred_future_xy: np.ndarray,
# #     true_future_xy: np.ndarray,
# #     targets: np.ndarray,
# #     labels: list[str],
# #     output_dir: Path,
# #     prefix: str,
# #     max_plots: int = 8,
# # ):
# #     output_dir.mkdir(parents=True, exist_ok=True)

# #     saved = []
# #     total = min(max_plots, pred_future_xy.shape[0])

# #     for idx in range(total):
# #         fig, ax = plt.subplots(figsize=(5, 5))

# #         ax.plot([0.0], [0.0], "ko", label="anchor")
# #         ax.plot(
# #             true_future_xy[idx, :, 0],
# #             true_future_xy[idx, :, 1],
# #             "-o",
# #             label="ground truth",
# #         )
# #         ax.plot(
# #             pred_future_xy[idx, :, 0],
# #             pred_future_xy[idx, :, 1],
# #             "--o",
# #             label="prediction",
# #         )

# #         ax.set_title(f"{prefix} sample {idx} ({labels[int(targets[idx])]})")
# #         ax.set_xlabel("Relative x (m)")
# #         ax.set_ylabel("Relative y (m)")
# #         ax.axis("equal")
# #         ax.grid(True, linestyle="--", alpha=0.4)
# #         ax.legend()

# #         path = output_dir / f"{prefix}_trajectory_overlay_{idx:02d}.png"

# #         plt.tight_layout()
# #         plt.savefig(path, dpi=180, bbox_inches="tight")
# #         plt.close(fig)

# #         saved.append(str(path))

# #     return saved


# # def save_mean_step_error_plot(
# #     pred_future_xy: np.ndarray,
# #     true_future_xy: np.ndarray,
# #     output_path: Path,
# #     title: str,
# # ):
# #     diff = pred_future_xy - true_future_xy
# #     step_error = np.linalg.norm(diff, axis=-1).mean(axis=0)

# #     fig, ax = plt.subplots(figsize=(7, 4))

# #     ax.plot(
# #         np.arange(1, len(step_error) + 1),
# #         step_error,
# #         marker="o",
# #     )

# #     ax.set_title(title)
# #     ax.set_xlabel("Future step")
# #     ax.set_ylabel("Mean displacement error (m)")
# #     ax.grid(True, linestyle="--", alpha=0.4)

# #     plt.tight_layout()
# #     plt.savefig(output_path, dpi=180, bbox_inches="tight")
# #     plt.close(fig)

# #     return str(output_path)


# # # ---------------------------------------------------------------------------
# # # Result directories and manifests
# # # ---------------------------------------------------------------------------

# # def prepare_result_dirs(model_slug: str):
# #     if RESULTS_ROOT is None or WEIGHTS_ROOT is None:
# #         raise RuntimeError(
# #             "dataset_helper output roots are not configured. "
# #             "Pass results_root and weights_root to configure_helper(...)."
# #         )

# #     result_dir = RESULTS_ROOT / model_slug
# #     weight_dir = WEIGHTS_ROOT / model_slug
# #     plot_dir = result_dir / "plots"

# #     for path in [result_dir, weight_dir, plot_dir]:
# #         path.mkdir(parents=True, exist_ok=True)

# #     return result_dir, weight_dir, plot_dir


# # def timestamp_tag():
# #     return datetime.now().strftime("%Y%m%d_%H%M%S")


# # def build_run_manifest(
# #     model_slug: str,
# #     timestamp: str,
# #     labels: list[str],
# #     split_path: Path,
# #     extra: dict | None = None,
# # ):
# #     manifest = {
# #         "model_slug": model_slug,
# #         "timestamp": timestamp,
# #         "labels": labels,
# #         "split_path": str(split_path),
# #     }

# #     if extra:
# #         manifest.update(extra)

# #     return manifest


# # def save_run_manifest(
# #     result_dir: Path,
# #     manifest: dict,
# #     timestamp: str,
# # ):
# #     latest_path = result_dir / "latest_run_manifest.json"
# #     dated_path = result_dir / f"{timestamp}_run_manifest.json"

# #     latest_path.write_text(json.dumps(manifest, indent=2))
# #     dated_path.write_text(json.dumps(manifest, indent=2))


# """Shared data and evaluation helpers for the thesis notebooks.

# This module keeps the repetitive dataset-loading, path-remapping, metric, and
# result-saving utilities in one place so the notebooks can stay focused on
# model-specific logic.

# This version supports the hybrid two-UAV / one-UGV dataset exported by:

#     03_dataset/exporters/export_hybrid_maneuver_dataset.py

# Expected dataset structure:

#     03_dataset/husky_control_dataset/
#         run_xxx/
#             frames.jsonl
#             schema.json
#             manifest.json
#             assets/
#                 husky_local/planar_scan/*.npy
#                 husky_local/front_points/*.npy
#                 uav1/front_points/*.npy
#                 uav2/front_points/*.npy

# Each JSONL frame may include:

#     agents
#     edges
#     modalities
#     observation
#     state
#     goal
#     goal_features
#     teacher
#     readiness
#     uav_context
#     network_state

# The main supervised trajectory target is built by sliding windows:

#     past_len observed frames  ->  future_len future ego (x, y) trajectory

# Current intended agent order:

#     [husky_local, uav1, uav2]

# The ego agent is always the Husky. The two UAVs provide left/right aerial context
# and forward hazard information.
# """

# from __future__ import annotations

# import json
# import random
# from collections import Counter
# from datetime import datetime
# from functools import lru_cache
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch

# try:
#     from sklearn.metrics import (
#         auc,
#         average_precision_score,
#         precision_recall_curve,
#         roc_curve,
#     )
#     from sklearn.preprocessing import label_binarize

#     SKLEARN_AVAILABLE = True
# except Exception:
#     SKLEARN_AVAILABLE = False


# # ---------------------------------------------------------------------------
# # Labels
# # ---------------------------------------------------------------------------

# DEFAULT_LABELS = [
#     "bootstrap",
#     "go_to_goal",
#     "avoid_left",
#     "avoid_right",
#     "commit_forward",
#     "reverse",
#     "recover",
#     "reassess",
#     "arrived",
#     "stop",
# ]

# REDUCED_LABELS = [
#     "go_to_goal",
#     "avoid_left",
#     "avoid_right",
#     "commit_forward",
#     "arrived",
# ]

# PLATFORM_ONEHOT = {
#     "UGV": [1.0, 0.0],
#     "UAV": [0.0, 1.0],
# }

# DEFAULT_AGENT_ORDER = [
#     "husky_local",
#     "uav1",
#     "uav2",
# ]

# DEFAULT_EXTERNAL_DATASET_ROOT = (
#     Path.home() / "Documents/Thesis/03_dataset/husky_control_dataset"
# )

# DATASET_ROOT: Path | None = None
# ORIGINAL_DATASET_ROOT: Path | None = None
# RESULTS_ROOT: Path | None = None
# WEIGHTS_ROOT: Path | None = None


# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------

# def configure_helper(
#     *,
#     dataset_root: Path,
#     original_dataset_root: Path | None = None,
#     results_root: Path | None = None,
#     weights_root: Path | None = None,
# ) -> None:
#     """Set notebook-specific roots once so helper functions stay simple."""
#     global DATASET_ROOT, ORIGINAL_DATASET_ROOT, RESULTS_ROOT, WEIGHTS_ROOT

#     DATASET_ROOT = Path(dataset_root).expanduser().resolve()
#     ORIGINAL_DATASET_ROOT = (
#         Path(original_dataset_root).expanduser().resolve()
#         if original_dataset_root is not None
#         else DATASET_ROOT
#     )
#     RESULTS_ROOT = (
#         Path(results_root).expanduser().resolve()
#         if results_root is not None
#         else None
#     )
#     WEIGHTS_ROOT = (
#         Path(weights_root).expanduser().resolve()
#         if weights_root is not None
#         else None
#     )

#     load_npy_cached.cache_clear()


# def _require_roots() -> tuple[Path, Path]:
#     if DATASET_ROOT is None or ORIGINAL_DATASET_ROOT is None:
#         raise RuntimeError(
#             "dataset_helper is not configured. "
#             "Call configure_helper(...) in the notebook setup cell first."
#         )

#     return DATASET_ROOT, ORIGINAL_DATASET_ROOT


# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# # ---------------------------------------------------------------------------
# # Label mapping
# # ---------------------------------------------------------------------------

# def build_label_mapping(label_mode: str):
#     """Return labels and mapping.

#     label_mode='full':
#         keeps all controller states.

#     label_mode='reduced':
#         removes transitional/recovery states and keeps the most useful maneuver
#         classes for classification-style auxiliary training.
#     """
#     if label_mode == "full":
#         labels = list(DEFAULT_LABELS)
#         mapping = {label: label for label in DEFAULT_LABELS}
#         return labels, mapping

#     labels = list(REDUCED_LABELS)
#     mapping = {
#         "bootstrap": None,
#         "go_to_goal": "go_to_goal",
#         "avoid_left": "avoid_left",
#         "avoid_right": "avoid_right",
#         "commit_forward": "commit_forward",
#         "reverse": None,
#         "recover": None,
#         "reassess": None,
#         "arrived": "arrived",
#         "stop": None,
#     }

#     return labels, mapping


# # ---------------------------------------------------------------------------
# # File discovery and path remapping
# # ---------------------------------------------------------------------------

# def _frame_files_under(root: Path) -> list[Path]:
#     root = Path(root)

#     if (root / "frames.jsonl").exists():
#         return [root / "frames.jsonl"]

#     return sorted(root.glob("*/frames.jsonl"))


# def discover_frame_files(dataset_root: Path) -> list[Path]:
#     """Find extracted frame files.

#     The function tries several sensible roots to avoid notebook breakage when
#     paths change after moving the dataset or restarting the kernel.
#     """
#     candidate_roots: list[Path] = []
#     seen: set[Path] = set()

#     def add_candidate(path: Path | None) -> None:
#         if path is None:
#             return

#         path = Path(path).expanduser()

#         try:
#             path = path.resolve()
#         except Exception:
#             pass

#         if path in seen:
#             return

#         seen.add(path)
#         candidate_roots.append(path)

#     add_candidate(Path(dataset_root))
#     add_candidate(DATASET_ROOT)
#     add_candidate(ORIGINAL_DATASET_ROOT)
#     add_candidate(DEFAULT_EXTERNAL_DATASET_ROOT)

#     for root in candidate_roots:
#         frame_files = _frame_files_under(root)
#         if frame_files:
#             return frame_files

#     return []


# def remap_dataset_path(path_str: str) -> Path:
#     """Map stored asset paths to the current local dataset root."""
#     dataset_root, original_dataset_root = _require_roots()

#     path = Path(path_str).expanduser()

#     if path.exists():
#         return path

#     try:
#         rel = path.relative_to(original_dataset_root)
#     except ValueError:
#         parts = path.parts

#         for dataset_marker in (
#             "hybrid_maneuvers_dataset",
#             "husky_control_dataset",
#         ):
#             if dataset_marker in parts:
#                 marker = parts.index(dataset_marker)
#                 rel = Path(*parts[marker + 1 :])

#                 candidate = dataset_root / rel
#                 if candidate.exists():
#                     return candidate

#                 if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
#                     fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
#                     if fallback_candidate.exists():
#                         return fallback_candidate

#         return path

#     candidate = dataset_root / rel

#     if candidate.exists():
#         return candidate

#     if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
#         fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
#         if fallback_candidate.exists():
#             return fallback_candidate

#     return candidate


# @lru_cache(maxsize=32768)
# def load_npy_cached(path: str):
#     return np.load(remap_dataset_path(path))


# def load_asset_ref(ref: dict | None):
#     """Load an asset reference from JSONL.

#     Returns None when the reference is missing or the file cannot be loaded.
#     """
#     if ref is None:
#         return None

#     path = ref.get("path")
#     if not path:
#         return None

#     try:
#         return load_npy_cached(str(path))
#     except Exception:
#         return None


# # ---------------------------------------------------------------------------
# # Sensor preprocessing
# # ---------------------------------------------------------------------------

# def resample_scan(
#     scan: np.ndarray,
#     num_beams: int,
#     range_clip: float,
# ) -> np.ndarray:
#     """Convert saved LaserScan array into a normalized 2 x num_beams tensor.

#     Input saved by exporter:
#         shape = [N, 2]
#         column 0 = range
#         column 1 = intensity
#     """
#     ranges = np.asarray(scan[:, 0], dtype=np.float32)
#     intensities = np.asarray(scan[:, 1], dtype=np.float32)

#     ranges = np.nan_to_num(
#         ranges,
#         nan=range_clip,
#         posinf=range_clip,
#         neginf=0.0,
#     )
#     ranges = np.clip(ranges, 0.0, range_clip)

#     intensities = np.nan_to_num(
#         intensities,
#         nan=0.0,
#         posinf=255.0,
#         neginf=0.0,
#     )
#     intensities = np.clip(intensities, 0.0, 255.0)

#     if ranges.shape[0] != num_beams:
#         src_x = np.linspace(0.0, 1.0, ranges.shape[0], dtype=np.float32)
#         dst_x = np.linspace(0.0, 1.0, num_beams, dtype=np.float32)

#         ranges = np.interp(dst_x, src_x, ranges).astype(np.float32)
#         intensities = np.interp(dst_x, src_x, intensities).astype(np.float32)

#     return np.stack(
#         [
#             ranges / max(range_clip, 1e-6),
#             intensities / 255.0,
#         ],
#         axis=0,
#     ).astype(np.float32)


# def summarize_pointcloud_corridor(
#     points: np.ndarray,
#     *,
#     max_points: int = 512,
#     x_min: float = 0.0,
#     x_max: float = 25.0,
#     y_abs_max: float = 12.0,
#     z_min: float = -5.0,
#     z_max: float = 25.0,
# ) -> np.ndarray:
#     """Filter and downsample a point cloud into a fixed-size Nx4 array.

#     This helper is useful if a notebook wants to use UAV or Husky pointclouds.
#     It keeps a forward corridor and returns max_points rows.

#     Output:
#         shape = [max_points, 4]
#         columns = x, y, z, intensity
#     """
#     if points is None or points.size == 0:
#         return np.zeros((max_points, 4), dtype=np.float32)

#     points = np.asarray(points, dtype=np.float32)

#     if points.ndim != 2 or points.shape[1] < 3:
#         return np.zeros((max_points, 4), dtype=np.float32)

#     if points.shape[1] == 3:
#         zeros = np.zeros((points.shape[0], 1), dtype=np.float32)
#         points = np.concatenate([points, zeros], axis=1)

#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]

#     mask = (
#         (x >= x_min)
#         & (x <= x_max)
#         & (np.abs(y) <= y_abs_max)
#         & (z >= z_min)
#         & (z <= z_max)
#     )

#     filtered = points[mask, :4]

#     if filtered.shape[0] == 0:
#         return np.zeros((max_points, 4), dtype=np.float32)

#     if filtered.shape[0] >= max_points:
#         idx = np.linspace(0, filtered.shape[0] - 1, max_points).astype(np.int64)
#         filtered = filtered[idx]
#     else:
#         pad = np.zeros((max_points - filtered.shape[0], 4), dtype=np.float32)
#         filtered = np.concatenate([filtered, pad], axis=0)

#     return filtered.astype(np.float32)


# def hazard_summary_from_pointcloud(
#     points: np.ndarray,
#     *,
#     x_min: float = 0.0,
#     x_max: float = 25.0,
#     center_half_width: float = 2.0,
#     side_width: float = 6.0,
#     z_min: float = -2.0,
#     z_max: float = 5.0,
# ) -> np.ndarray:
#     """Convert a point cloud into simple [left, center, right] hazard counts.

#     Output:
#         np.ndarray with shape [3]
#         columns = log_left_count, log_center_count, log_right_count
#     """
#     if points is None or points.size == 0:
#         return np.zeros(3, dtype=np.float32)

#     points = np.asarray(points, dtype=np.float32)

#     if points.ndim != 2 or points.shape[1] < 3:
#         return np.zeros(3, dtype=np.float32)

#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]

#     valid = (
#         (x >= x_min)
#         & (x <= x_max)
#         & (z >= z_min)
#         & (z <= z_max)
#         & (np.abs(y) <= side_width)
#     )

#     yv = y[valid]

#     left = np.sum((yv > center_half_width) & (yv <= side_width))
#     center = np.sum(np.abs(yv) <= center_half_width)
#     right = np.sum((yv < -center_half_width) & (yv >= -side_width))

#     counts = np.asarray([left, center, right], dtype=np.float32)
#     return np.log1p(counts)


# def hazard_summary_from_dict(summary: dict | None) -> np.ndarray:
#     """Read the compact hazard summary stored directly in JSONL.

#     Supports summaries generated by the updated exporter:
#         log_left
#         log_center
#         log_right

#     If the dict is missing, returns zeros.
#     """
#     if not isinstance(summary, dict):
#         return np.zeros(3, dtype=np.float32)

#     if "log_left" in summary or "log_center" in summary or "log_right" in summary:
#         return np.asarray(
#             [
#                 float(summary.get("log_left", 0.0)),
#                 float(summary.get("log_center", 0.0)),
#                 float(summary.get("log_right", 0.0)),
#             ],
#             dtype=np.float32,
#         )

#     return np.asarray(
#         [
#             np.log1p(float(summary.get("left_count", 0.0))),
#             np.log1p(float(summary.get("center_count", 0.0))),
#             np.log1p(float(summary.get("right_count", 0.0))),
#         ],
#         dtype=np.float32,
#     )


# # ---------------------------------------------------------------------------
# # Frame accessors supporting old and new schema
# # ---------------------------------------------------------------------------

# def canonical_agent_order(ego_id: str = "husky_local") -> list[str]:
#     """Return the model graph agent order.

#     New structure:
#         husky_local + uav1 + uav2

#     The ego should normally be husky_local. If an old exported frame uses a
#     different ego_id, this still keeps that ego first, then adds uav1/uav2.
#     """
#     if ego_id == "husky_local":
#         return ["husky_local", "uav1", "uav2"]

#     order = [ego_id]
#     for agent_id in ["husky_local", "uav1", "uav2"]:
#         if agent_id not in order:
#             order.append(agent_id)
#     return order


# def _zero_state() -> dict:
#     return {
#         "x": 0.0,
#         "y": 0.0,
#         "z": 0.0,
#         "qx": 0.0,
#         "qy": 0.0,
#         "qz": 0.0,
#         "qw": 1.0,
#         "yaw": 0.0,
#         "vx": 0.0,
#         "vy": 0.0,
#         "vz": 0.0,
#         "wz": 0.0,
#     }


# def _default_agent_node(agent_id: str) -> dict:
#     platform = "UAV" if agent_id.startswith("uav") else "UGV"

#     return {
#         "id": agent_id,
#         "available": False,
#         "platform_type": platform,
#         "state": _zero_state(),
#         "start": None,
#         "goal": None,
#         "goal_features": None,
#         "command": {
#             "linear_x": 0.0,
#             "linear_y": 0.0,
#             "linear_z": 0.0,
#             "angular_z": 0.0,
#         },
#         "controller_state": None,
#         "obstacle_action": None,
#         "obstacle_clearance": None,
#         "ready": None,
#     }


# def frame_agents(frame: dict) -> dict:
#     """Return normalized agent dictionary for both old and new exports."""
#     ego_id = frame.get("ego_id", "husky_local")

#     if "agents" in frame and isinstance(frame["agents"], dict):
#         agents = dict(frame["agents"])
#     else:
#         agents = {
#             ego_id: {
#                 "id": ego_id,
#                 "available": frame.get("state") is not None,
#                 "platform_type": "UGV",
#                 "state": frame.get("state"),
#                 "start": None,
#                 "goal": frame.get("goal"),
#                 "goal_features": frame.get("goal_features"),
#                 "command": frame.get("teacher", {}).get("command")
#                 or {
#                     "linear_x": 0.0,
#                     "linear_y": 0.0,
#                     "linear_z": 0.0,
#                     "angular_z": 0.0,
#                 },
#                 "controller_state": frame.get("teacher", {}).get("controller_state"),
#                 "obstacle_action": frame.get("teacher", {}).get("obstacle_action"),
#                 "obstacle_clearance": frame.get("teacher", {}).get("obstacle_clearance"),
#                 "ready": None,
#             }
#         }

#         uav_context = frame.get("uav_context", {})
#         for uav_id in ["uav1", "uav2"]:
#             ctx = uav_context.get(uav_id, {}) if isinstance(uav_context, dict) else {}
#             agents[uav_id] = {
#                 "id": uav_id,
#                 "available": bool(ctx.get("available", False)),
#                 "platform_type": "UAV",
#                 "state": ctx.get("state"),
#                 "start": None,
#                 "goal": ctx.get("goal"),
#                 "goal_features": ctx.get("goal_features"),
#                 "command": {
#                     "linear_x": 0.0,
#                     "linear_y": 0.0,
#                     "linear_z": 0.0,
#                     "angular_z": 0.0,
#                 },
#                 "controller_state": None,
#                 "obstacle_action": None,
#                 "obstacle_clearance": None,
#                 "ready": ctx.get("ready"),
#             }

#         # Backward compatibility with older one-UAV + second-Husky export.
#         other = frame.get("other_husky")
#         if isinstance(other, dict) and "husky_2" not in agents:
#             agents["husky_2"] = {
#                 "id": "husky_2",
#                 "available": bool(other.get("available", False)),
#                 "platform_type": "UGV",
#                 "state": other.get("state"),
#                 "start": None,
#                 "goal": other.get("goal"),
#                 "goal_features": other.get("goal_features"),
#                 "command": other.get("teacher_command")
#                 or {
#                     "linear_x": 0.0,
#                     "linear_y": 0.0,
#                     "linear_z": 0.0,
#                     "angular_z": 0.0,
#                 },
#                 "controller_state": None,
#                 "obstacle_action": None,
#                 "obstacle_clearance": None,
#                 "ready": None,
#             }

#     for agent_id in canonical_agent_order(ego_id):
#         if agent_id not in agents:
#             agents[agent_id] = _default_agent_node(agent_id)
#             continue

#         node = dict(_default_agent_node(agent_id)) | dict(agents[agent_id])

#         if node.get("state") is None:
#             node["state"] = _zero_state()
#             node["available"] = False

#         if node.get("command") is None:
#             node["command"] = _default_agent_node(agent_id)["command"]

#         if node.get("platform_type") is None:
#             node["platform_type"] = "UAV" if agent_id.startswith("uav") else "UGV"

#         agents[agent_id] = node

#     return agents


# def frame_state(frame: dict) -> dict:
#     if "agents" in frame:
#         ego_id = frame.get("ego_id", "husky_local")
#         state = frame["agents"].get(ego_id, {}).get("state")
#         return state if state is not None else _zero_state()

#     state = frame.get("state")
#     return state if state is not None else _zero_state()


# def frame_scan_ref(frame: dict):
#     if "modalities" in frame:
#         ref = frame["modalities"].get("ego_planar_scan")
#         if ref is not None:
#             return ref

#     if "observation" in frame:
#         return frame["observation"].get("ego_planar_scan")

#     return None


# def frame_ego_pointcloud_ref(frame: dict):
#     if "modalities" in frame:
#         ref = frame["modalities"].get("ego_front_pointcloud")
#         if ref is not None:
#             return ref

#     if "observation" in frame:
#         return frame["observation"].get("ego_front_pointcloud")

#     return None


# def frame_uav_pointcloud_ref(frame: dict, uav_id: str = "uav1"):
#     key = f"{uav_id}_front_pointcloud"

#     if "modalities" in frame:
#         ref = frame["modalities"].get(key)
#         if ref is not None:
#             return ref

#     if "observation" in frame:
#         return frame["observation"].get(key)

#     return None


# def frame_uav_hazard_summary(frame: dict, uav_id: str = "uav1") -> np.ndarray:
#     key = f"{uav_id}_hazard_summary"

#     if "modalities" in frame and key in frame["modalities"]:
#         return hazard_summary_from_dict(frame["modalities"].get(key))

#     if "observation" in frame and key in frame["observation"]:
#         return hazard_summary_from_dict(frame["observation"].get(key))

#     uav_context = frame.get("uav_context", {})
#     if isinstance(uav_context, dict):
#         ctx = uav_context.get(uav_id, {})
#         if isinstance(ctx, dict):
#             return hazard_summary_from_dict(ctx.get("hazard_summary"))

#     return np.zeros(3, dtype=np.float32)


# def frame_teacher_label(frame: dict) -> str:
#     teacher = frame.get("teacher", {})

#     label = teacher.get("label")
#     if label is not None:
#         return str(label)

#     controller_state = teacher.get("controller_state")
#     if controller_state:
#         return str(controller_state)

#     obstacle_action = teacher.get("obstacle_action")
#     if obstacle_action:
#         obstacle_action = str(obstacle_action).strip().lower()
#         if obstacle_action.endswith("left"):
#             return "avoid_left"
#         if obstacle_action.endswith("right"):
#             return "avoid_right"

#     return "go_to_goal"


# # ---------------------------------------------------------------------------
# # Graph features
# # ---------------------------------------------------------------------------

# def node_feature(node: dict, ego_state: dict) -> list[float]:
#     """Build a fixed 14D node feature vector.

#     Features:
#         relative position to ego: dx, dy, dz
#         velocity: vx, vy, vz, wz
#         goal offset: gx, gy, gz
#         command: linear_x, angular_z
#         platform one-hot: UGV, UAV
#     """
#     state = node.get("state") or _zero_state()
#     goal = node.get("goal") or {
#         "x": state["x"],
#         "y": state["y"],
#         "z": state["z"],
#     }
#     command = node.get("command") or {
#         "linear_x": 0.0,
#         "angular_z": 0.0,
#     }
#     platform = PLATFORM_ONEHOT.get(
#         node.get("platform_type", "UGV"),
#         [0.0, 0.0],
#     )

#     return [
#         float(state.get("x", 0.0) - ego_state.get("x", 0.0)),
#         float(state.get("y", 0.0) - ego_state.get("y", 0.0)),
#         float(state.get("z", 0.0) - ego_state.get("z", 0.0)),
#         float(state.get("vx", 0.0)),
#         float(state.get("vy", 0.0)),
#         float(state.get("vz", 0.0)),
#         float(state.get("wz", 0.0)),
#         float(goal.get("x", state.get("x", 0.0)) - state.get("x", 0.0)),
#         float(goal.get("y", state.get("y", 0.0)) - state.get("y", 0.0)),
#         float(goal.get("z", state.get("z", 0.0)) - state.get("z", 0.0)),
#         float(command.get("linear_x", 0.0)),
#         float(command.get("angular_z", 0.0)),
#         float(platform[0]),
#         float(platform[1]),
#     ]


# def graph_node_features_for_frame(
#     frame: dict,
#     order: list[str] | None = None,
# ) -> np.ndarray:
#     """Return graph node feature matrix [num_nodes, 14]."""
#     agents = frame_agents(frame)
#     ego_id = frame.get("ego_id", "husky_local")
#     ego_state = agents[ego_id].get("state") or _zero_state()
#     order = order or canonical_agent_order(ego_id)

#     features = []

#     for agent_id in order:
#         node = agents.get(agent_id, _default_agent_node(agent_id))
#         features.append(node_feature(node, ego_state))

#     return np.asarray(features, dtype=np.float32)


# def build_edge_lookup(frame: dict) -> dict[tuple[str, str], dict]:
#     edges = frame.get("edges") or []
#     return {
#         (edge["source"], edge["target"]): edge
#         for edge in edges
#         if "source" in edge and "target" in edge
#     }


# def _fallback_edge(src_node: dict, dst_node: dict) -> dict:
#     src_state = src_node.get("state") or _zero_state()
#     dst_state = dst_node.get("state") or _zero_state()

#     dx = float(dst_state.get("x", 0.0) - src_state.get("x", 0.0))
#     dy = float(dst_state.get("y", 0.0) - src_state.get("y", 0.0))
#     dz = float(dst_state.get("z", 0.0) - src_state.get("z", 0.0))

#     distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
#     bearing = float(np.arctan2(dy, dx))

#     return {
#         "dx": dx,
#         "dy": dy,
#         "dz": dz,
#         "distance": distance,
#         "inv_distance": float(1.0 / max(distance, 1e-6)),
#         "bearing_sin": float(np.sin(bearing)),
#         "bearing_cos": float(np.cos(bearing)),
#         "same_platform": float(
#             src_node.get("platform_type") == dst_node.get("platform_type")
#         ),
#         "latency_s": 0.0,
#         "packet_loss": 0.0,
#         "link_quality": 1.0,
#     }


# def _edge_network_value(edge: dict, name: str, default: float) -> float:
#     if name in edge:
#         return float(edge.get(name, default))

#     network = edge.get("network", {})
#     if isinstance(network, dict) and name in network:
#         return float(network.get(name, default))

#     return float(default)


# def edge_features_for_order(frame: dict, order: list[str]) -> list[list[list[float]]]:
#     """Return edge features [num_nodes, num_nodes, 11].

#     Edge feature order:
#         dx, dy, dz,
#         distance, inv_distance,
#         bearing_sin, bearing_cos,
#         same_platform,
#         latency_s, packet_loss, link_quality
#     """
#     agents = frame_agents(frame)
#     edge_map = build_edge_lookup(frame)

#     src_edges = []

#     for src in order:
#         row = []

#         for dst in order:
#             if src == dst:
#                 row.append([0.0] * 11)
#                 continue

#             edge = edge_map.get((src, dst))

#             if edge is None:
#                 edge = _fallback_edge(
#                     agents.get(src, _default_agent_node(src)),
#                     agents.get(dst, _default_agent_node(dst)),
#                 )

#             row.append(
#                 [
#                     float(edge.get("dx", 0.0)),
#                     float(edge.get("dy", 0.0)),
#                     float(edge.get("dz", 0.0)),
#                     float(edge.get("distance", 0.0)),
#                     float(edge.get("inv_distance", 0.0)),
#                     float(edge.get("bearing_sin", 0.0)),
#                     float(edge.get("bearing_cos", 1.0)),
#                     float(edge.get("same_platform", 0.0)),
#                     _edge_network_value(edge, "latency_s", 0.0),
#                     _edge_network_value(edge, "packet_loss", 0.0),
#                     _edge_network_value(edge, "link_quality", 1.0),
#                 ]
#             )

#         src_edges.append(row)

#     return src_edges


# def graph_edge_features_for_frame(
#     frame: dict,
#     order: list[str] | None = None,
# ) -> np.ndarray:
#     ego_id = frame.get("ego_id", "husky_local")
#     order = order or canonical_agent_order(ego_id)

#     return np.asarray(edge_features_for_order(frame, order), dtype=np.float32)


# # ---------------------------------------------------------------------------
# # Dataset grouping and sample construction
# # ---------------------------------------------------------------------------

# def group_streams(
#     dataset_root: Path,
#     allowed_labels: set[str] | None = None,
#     label_mapping: dict | None = None,
#     require_uav: bool = False,
#     require_both_uavs: bool = False,
# ):
#     """Group frames by episode_id and ego_id.

#     The output is a list of sorted streams. Each stream is one continuous
#     ego trajectory from one bag.

#     require_uav=True:
#         require at least uav1 state.

#     require_both_uavs=True:
#         require both uav1 and uav2 states.
#     """
#     streams = []
#     frame_files = discover_frame_files(dataset_root)

#     allowed_labels = set(allowed_labels) if allowed_labels is not None else None
#     label_mapping = label_mapping or {}

#     for frames_path in frame_files:
#         with frames_path.open() as f:
#             rows = [json.loads(line) for line in f if line.strip()]

#         buckets = {}

#         for row in rows:
#             raw_label = frame_teacher_label(row)
#             mapped_label = label_mapping.get(raw_label, raw_label)

#             if allowed_labels is not None and (
#                 mapped_label is None or mapped_label not in allowed_labels
#             ):
#                 continue

#             if frame_scan_ref(row) is None:
#                 continue

#             if frame_state(row) is None:
#                 continue

#             readiness = row.get("readiness", {})

#             if require_uav:
#                 if not readiness.get("has_uav1_state", False):
#                     continue

#             if require_both_uavs:
#                 if not readiness.get("has_uav1_state", False):
#                     continue
#                 if not readiness.get("has_uav2_state", False):
#                     continue

#             row = dict(row)
#             row["teacher"] = dict(row.get("teacher", {}))
#             row["teacher"]["raw_label"] = raw_label
#             row["teacher"]["label"] = mapped_label

#             key = f"{row['episode_id']}::{row['ego_id']}"
#             buckets.setdefault(key, []).append(row)

#         for key in sorted(buckets):
#             stream = sorted(
#                 buckets[key],
#                 key=lambda item: int(item["timestamp_ns"]),
#             )

#             if stream:
#                 streams.append(stream)

#     if not streams:
#         raise RuntimeError(
#             f"No usable frame streams found under {dataset_root}. "
#             f"Discovered {len(frame_files)} frame file(s); "
#             f"check label filtering, dataset paths, and whether export completed."
#         )

#     return streams


# def build_sample_table(
#     streams: list[list[dict]],
#     past_len: int,
#     future_len: int,
# ):
#     """Build sliding-window sample metadata.

#     The target is future ego trajectory relative to the anchor frame.
#     """
#     sample_table = []

#     for stream_idx, stream in enumerate(streams):
#         usable = len(stream) - past_len - future_len + 1

#         for start in range(max(0, usable)):
#             anchor = stream[start + past_len - 1]
#             future_frames = stream[start + past_len : start + past_len + future_len]

#             anchor_state = frame_state(anchor)
#             anchor_ts = int(anchor["timestamp_ns"])

#             future_xy = []
#             future_dt = []

#             valid = True

#             for future_frame in future_frames:
#                 state = frame_state(future_frame)

#                 if state is None:
#                     valid = False
#                     break

#                 future_xy.append(
#                     [
#                         float(state["x"] - anchor_state["x"]),
#                         float(state["y"] - anchor_state["y"]),
#                     ]
#                 )
#                 future_dt.append(
#                     (int(future_frame["timestamp_ns"]) - anchor_ts) * 1e-9
#                 )

#             if not valid:
#                 continue

#             sample_table.append(
#                 {
#                     "sample_id": f"stream{stream_idx:03d}_start{start:05d}",
#                     "stream_index": stream_idx,
#                     "stream_idx": stream_idx,
#                     "start_index": start,
#                     "start": start,
#                     "anchor_index": start + past_len - 1,
#                     "ego_id": anchor["ego_id"],
#                     "label": anchor["teacher"].get("label"),
#                     "raw_label": anchor["teacher"].get("raw_label"),
#                     "future_xy": future_xy,
#                     "future_dt": future_dt,
#                 }
#             )

#     return sample_table


# def save_or_load_fixed_split(
#     sample_table,
#     split_path: Path,
#     seed: int,
#     train_ratio: float,
#     val_ratio: float,
#     past_len: int,
#     future_len: int,
# ):
#     """Create or reuse a deterministic train/val/test split."""
#     split_path = Path(split_path)

#     if split_path.exists():
#         with split_path.open() as f:
#             split_info = json.load(f)

#         current_sample_ids = [row["sample_id"] for row in sample_table]

#         if (
#             split_info.get("sample_count") == len(sample_table)
#             and split_info.get("past_len") == past_len
#             and split_info.get("future_len") == future_len
#             and split_info.get("sample_ids") == current_sample_ids
#         ):
#             return split_info

#     rng = random.Random(seed)
#     indices = list(range(len(sample_table)))
#     rng.shuffle(indices)

#     if len(indices) < 3:
#         raise RuntimeError(
#             f"Need at least 3 samples to split train/val/test, got {len(indices)}."
#         )

#     train_len = max(1, int(len(indices) * train_ratio))
#     val_len = max(1, int(len(indices) * val_ratio))
#     test_len = len(indices) - train_len - val_len

#     if test_len < 1:
#         test_len = 1
#         if train_len > val_len:
#             train_len -= 1
#         else:
#             val_len -= 1

#     split_info = {
#         "seed": seed,
#         "sample_count": len(sample_table),
#         "past_len": past_len,
#         "future_len": future_len,
#         "train_indices": indices[:train_len],
#         "val_indices": indices[train_len : train_len + val_len],
#         "test_indices": indices[train_len + val_len :],
#         "sample_ids": [row["sample_id"] for row in sample_table],
#     }

#     split_path.parent.mkdir(parents=True, exist_ok=True)
#     split_path.write_text(json.dumps(split_info, indent=2))

#     return split_info


# # ---------------------------------------------------------------------------
# # Optional ready-to-use feature builders for notebooks
# # ---------------------------------------------------------------------------

# def build_past_ego_xy(
#     stream: list[dict],
#     start: int,
#     past_len: int,
# ) -> np.ndarray:
#     """Return past ego xy relative to the anchor frame.

#     Output:
#         shape = [past_len, 2]
#     """
#     past_frames = stream[start : start + past_len]
#     anchor_state = frame_state(past_frames[-1])

#     xy = []

#     for frame in past_frames:
#         state = frame_state(frame)
#         xy.append(
#             [
#                 float(state["x"] - anchor_state["x"]),
#                 float(state["y"] - anchor_state["y"]),
#             ]
#         )

#     return np.asarray(xy, dtype=np.float32)


# def build_past_graph_sequence(
#     stream: list[dict],
#     start: int,
#     past_len: int,
#     order: list[str] | None = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Return past graph node and edge sequences.

#     Outputs:
#         node_seq: [past_len, num_nodes, 14]
#         edge_seq: [past_len, num_nodes, num_nodes, 11]

#     With the new dataset, num_nodes is normally 3:
#         husky_local, uav1, uav2
#     """
#     past_frames = stream[start : start + past_len]

#     if order is None:
#         order = canonical_agent_order(past_frames[-1].get("ego_id", "husky_local"))

#     node_seq = []
#     edge_seq = []

#     for frame in past_frames:
#         node_seq.append(graph_node_features_for_frame(frame, order))
#         edge_seq.append(graph_edge_features_for_frame(frame, order))

#     return (
#         np.asarray(node_seq, dtype=np.float32),
#         np.asarray(edge_seq, dtype=np.float32),
#     )


# def build_past_scan_sequence(
#     stream: list[dict],
#     start: int,
#     past_len: int,
#     *,
#     num_beams: int = 256,
#     range_clip: float = 30.0,
# ) -> np.ndarray:
#     """Return past ego lidar sequence.

#     Output:
#         shape = [past_len, 2, num_beams]
#     """
#     past_frames = stream[start : start + past_len]
#     scans = []

#     for frame in past_frames:
#         ref = frame_scan_ref(frame)
#         scan = load_asset_ref(ref)

#         if scan is None:
#             scans.append(np.zeros((2, num_beams), dtype=np.float32))
#         else:
#             scans.append(resample_scan(scan, num_beams, range_clip))

#     return np.asarray(scans, dtype=np.float32)


# def build_past_uav_hazard_sequence(
#     stream: list[dict],
#     start: int,
#     past_len: int,
#     *,
#     include_uav1: bool = True,
#     include_uav2: bool = True,
# ) -> np.ndarray:
#     """Return past UAV hazard summaries.

#     New output by default:
#         shape = [past_len, 6]

#     Columns:
#         uav1_left, uav1_center, uav1_right,
#         uav2_left, uav2_center, uav2_right

#     If only one UAV is requested, output shape becomes [past_len, 3].
#     """
#     past_frames = stream[start : start + past_len]
#     summaries = []

#     for frame in past_frames:
#         parts = []

#         if include_uav1:
#             summary1 = frame_uav_hazard_summary(frame, "uav1")
#             if np.allclose(summary1, 0.0):
#                 ref1 = frame_uav_pointcloud_ref(frame, "uav1")
#                 points1 = load_asset_ref(ref1)
#                 if points1 is not None:
#                     summary1 = hazard_summary_from_pointcloud(points1)
#             parts.append(summary1)

#         if include_uav2:
#             summary2 = frame_uav_hazard_summary(frame, "uav2")
#             if np.allclose(summary2, 0.0):
#                 ref2 = frame_uav_pointcloud_ref(frame, "uav2")
#                 points2 = load_asset_ref(ref2)
#                 if points2 is not None:
#                     summary2 = hazard_summary_from_pointcloud(points2)
#             parts.append(summary2)

#         if parts:
#             summaries.append(np.concatenate(parts).astype(np.float32))
#         else:
#             summaries.append(np.zeros(0, dtype=np.float32))

#     return np.asarray(summaries, dtype=np.float32)


# def build_past_uav_pointcloud_sequence(
#     stream: list[dict],
#     start: int,
#     past_len: int,
#     *,
#     uav_id: str,
#     max_points: int = 512,
# ) -> np.ndarray:
#     """Return a fixed-size pointcloud sequence for one UAV.

#     Output:
#         shape = [past_len, max_points, 4]
#     """
#     past_frames = stream[start : start + past_len]
#     clouds = []

#     for frame in past_frames:
#         ref = frame_uav_pointcloud_ref(frame, uav_id)
#         points = load_asset_ref(ref)

#         if points is None:
#             clouds.append(np.zeros((max_points, 4), dtype=np.float32))
#         else:
#             clouds.append(
#                 summarize_pointcloud_corridor(
#                     points,
#                     max_points=max_points,
#                 )
#             )

#     return np.asarray(clouds, dtype=np.float32)


# # ---------------------------------------------------------------------------
# # Metrics and class helpers
# # ---------------------------------------------------------------------------

# def build_class_weights(
#     label_indices: list[int],
#     num_classes: int,
# ):
#     counts = Counter(label_indices)
#     total = sum(counts.values())

#     weights = []

#     for idx in range(num_classes):
#         count = counts.get(idx, 0)
#         weights.append(0.0 if count == 0 else total / (num_classes * count))

#     return torch.tensor(weights, dtype=torch.float32)


# def compute_trajectory_metrics(
#     pred_future_xy: np.ndarray,
#     true_future_xy: np.ndarray,
# ):
#     diff = pred_future_xy - true_future_xy
#     dist = np.linalg.norm(diff, axis=-1)

#     return {
#         "ADE": float(dist.mean()),
#         "FDE": float(dist[:, -1].mean()),
#         "RMSE": float(np.sqrt(np.mean(np.sum(diff**2, axis=-1)))),
#     }


# def compute_classification_metrics_from_probs(
#     probabilities: np.ndarray,
#     targets: np.ndarray,
#     labels: list[str],
# ):
#     preds = probabilities.argmax(axis=1)
#     num_classes = len(labels)

#     confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

#     for truth, guess in zip(targets, preds):
#         confusion[int(truth), int(guess)] += 1

#     precisions, recalls, f1s = [], [], []

#     for idx in range(num_classes):
#         tp = float(confusion[idx, idx])
#         fn = float(confusion[idx, :].sum() - tp)
#         fp = float(confusion[:, idx].sum() - tp)

#         precision = tp / max(tp + fp, 1.0)
#         recall = tp / max(tp + fn, 1.0)
#         f1 = (
#             0.0
#             if (precision + recall) == 0.0
#             else (2.0 * precision * recall / (precision + recall))
#         )

#         precisions.append(precision)
#         recalls.append(recall)
#         f1s.append(f1)

#     metrics = {
#         "accuracy": float((preds == targets).mean()),
#         "macro_precision": float(np.mean(precisions)),
#         "macro_recall": float(np.mean(recalls)),
#         "macro_f1": float(np.mean(f1s)),
#         "confusion_matrix": confusion.tolist(),
#         "ADE": None,
#         "FDE": None,
#         "RMSE": None,
#     }

#     return metrics, preds, confusion


# # ---------------------------------------------------------------------------
# # Saving outputs
# # ---------------------------------------------------------------------------

# def save_training_history(
#     history: dict,
#     out_path: Path,
# ):
#     pd.DataFrame(history).to_csv(out_path, index=False)


# def save_confusion_matrix(
#     confusion: np.ndarray,
#     labels: list[str],
#     csv_path: Path,
#     png_path: Path,
#     title: str,
# ):
#     df = pd.DataFrame(confusion, index=labels, columns=labels)
#     df.to_csv(csv_path)

#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(confusion, cmap="Blues")

#     ax.set_xticks(range(len(labels)))
#     ax.set_yticks(range(len(labels)))
#     ax.set_xticklabels(labels, rotation=45, ha="right")
#     ax.set_yticklabels(labels)

#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     ax.set_title(title)

#     for i in range(len(labels)):
#         for j in range(len(labels)):
#             ax.text(
#                 j,
#                 i,
#                 str(confusion[i, j]),
#                 ha="center",
#                 va="center",
#                 color="black",
#                 fontsize=8,
#             )

#     fig.colorbar(im, ax=ax)
#     plt.tight_layout()
#     plt.savefig(png_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)


# def save_roc_pr_curves(
#     probabilities: np.ndarray,
#     targets: np.ndarray,
#     labels: list[str],
#     out_dir: Path,
# ):
#     summary = {
#         "roc_auc_macro": None,
#         "pr_auc_macro": None,
#         "status": "skipped",
#     }

#     if not SKLEARN_AVAILABLE:
#         return summary

#     y_true = label_binarize(targets, classes=list(range(len(labels))))

#     roc_aucs = []
#     pr_aucs = []

#     fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
#     fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

#     for idx, label in enumerate(labels):
#         try:
#             fpr, tpr, _ = roc_curve(y_true[:, idx], probabilities[:, idx])
#             roc_auc_value = auc(fpr, tpr)

#             precision, recall, _ = precision_recall_curve(
#                 y_true[:, idx],
#                 probabilities[:, idx],
#             )
#             pr_auc_value = average_precision_score(
#                 y_true[:, idx],
#                 probabilities[:, idx],
#             )

#             roc_aucs.append(roc_auc_value)
#             pr_aucs.append(pr_auc_value)

#             ax_roc.plot(
#                 fpr,
#                 tpr,
#                 label=f"{label} (AUC={roc_auc_value:.3f})",
#             )
#             ax_pr.plot(
#                 recall,
#                 precision,
#                 label=f"{label} (AP={pr_auc_value:.3f})",
#             )
#         except Exception:
#             continue

#     ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     ax_roc.set_title("One-vs-Rest ROC Curves")
#     ax_roc.set_xlabel("False Positive Rate")
#     ax_roc.set_ylabel("True Positive Rate")
#     ax_roc.legend(fontsize=8)

#     plt.tight_layout()
#     fig_roc.savefig(out_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
#     plt.close(fig_roc)

#     ax_pr.set_title("One-vs-Rest Precision-Recall Curves")
#     ax_pr.set_xlabel("Recall")
#     ax_pr.set_ylabel("Precision")
#     ax_pr.legend(fontsize=8)

#     plt.tight_layout()
#     fig_pr.savefig(out_dir / "pr_curves.png", dpi=180, bbox_inches="tight")
#     plt.close(fig_pr)

#     if roc_aucs:
#         summary["roc_auc_macro"] = float(np.mean(roc_aucs))

#     if pr_aucs:
#         summary["pr_auc_macro"] = float(np.mean(pr_aucs))

#     summary["status"] = "saved"

#     return summary


# def save_predictions_csv(
#     sample_ids,
#     targets,
#     preds,
#     probabilities,
#     labels,
#     out_path: Path,
# ):
#     rows = []

#     for sid, truth, pred, probs in zip(
#         sample_ids,
#         targets,
#         preds,
#         probabilities,
#     ):
#         row = {
#             "sample_id": sid,
#             "true_label": labels[int(truth)],
#             "pred_label": labels[int(pred)],
#         }

#         for idx, label in enumerate(labels):
#             row[f"prob_{label}"] = float(probs[idx])

#         rows.append(row)

#     pd.DataFrame(rows).to_csv(out_path, index=False)


# def save_history_plot(
#     history: dict,
#     out_path: Path,
#     title_prefix: str,
# ):
#     if not history or len(history.get("epoch", [])) == 0:
#         return

#     fig, axes = plt.subplots(1, 3, figsize=(18, 4))

#     axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
#     axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
#     axes[0].set_title(f"{title_prefix}: Loss")
#     axes[0].legend()

#     axes[1].plot(
#         history["epoch"],
#         history["val_accuracy"],
#         label="val_accuracy",
#     )
#     axes[1].set_title(f"{title_prefix}: Validation Accuracy")
#     axes[1].legend()

#     axes[2].plot(
#         history["epoch"],
#         history["val_macro_f1"],
#         label="val_macro_f1",
#     )
#     axes[2].set_title(f"{title_prefix}: Validation Macro-F1")
#     axes[2].legend()

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)


# def save_trajectory_overlay_plots(
#     pred_future_xy: np.ndarray,
#     true_future_xy: np.ndarray,
#     targets: np.ndarray,
#     labels: list[str],
#     output_dir: Path,
#     prefix: str,
#     max_plots: int = 8,
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)

#     saved = []
#     total = min(max_plots, pred_future_xy.shape[0])

#     for idx in range(total):
#         fig, ax = plt.subplots(figsize=(5, 5))

#         ax.plot([0.0], [0.0], "ko", label="anchor")
#         ax.plot(
#             true_future_xy[idx, :, 0],
#             true_future_xy[idx, :, 1],
#             "-o",
#             label="ground truth",
#         )
#         ax.plot(
#             pred_future_xy[idx, :, 0],
#             pred_future_xy[idx, :, 1],
#             "--o",
#             label="prediction",
#         )

#         ax.set_title(f"{prefix} sample {idx} ({labels[int(targets[idx])]})")
#         ax.set_xlabel("Relative x (m)")
#         ax.set_ylabel("Relative y (m)")
#         ax.axis("equal")
#         ax.grid(True, linestyle="--", alpha=0.4)
#         ax.legend()

#         path = output_dir / f"{prefix}_trajectory_overlay_{idx:02d}.png"

#         plt.tight_layout()
#         plt.savefig(path, dpi=180, bbox_inches="tight")
#         plt.close(fig)

#         saved.append(str(path))

#     return saved


# def save_mean_step_error_plot(
#     pred_future_xy: np.ndarray,
#     true_future_xy: np.ndarray,
#     output_path: Path,
#     title: str,
# ):
#     diff = pred_future_xy - true_future_xy
#     step_error = np.linalg.norm(diff, axis=-1).mean(axis=0)

#     fig, ax = plt.subplots(figsize=(7, 4))

#     ax.plot(
#         np.arange(1, len(step_error) + 1),
#         step_error,
#         marker="o",
#     )

#     ax.set_title(title)
#     ax.set_xlabel("Future step")
#     ax.set_ylabel("Mean displacement error (m)")
#     ax.grid(True, linestyle="--", alpha=0.4)

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)

#     return str(output_path)


# # ---------------------------------------------------------------------------
# # Result directories and manifests
# # ---------------------------------------------------------------------------

# def prepare_result_dirs(model_slug: str):
#     if RESULTS_ROOT is None or WEIGHTS_ROOT is None:
#         raise RuntimeError(
#             "dataset_helper output roots are not configured. "
#             "Pass results_root and weights_root to configure_helper(...)."
#         )

#     result_dir = RESULTS_ROOT / model_slug
#     weight_dir = WEIGHTS_ROOT / model_slug
#     plot_dir = result_dir / "plots"

#     for path in [result_dir, weight_dir, plot_dir]:
#         path.mkdir(parents=True, exist_ok=True)

#     return result_dir, weight_dir, plot_dir


# def timestamp_tag():
#     return datetime.now().strftime("%Y%m%d_%H%M%S")


# def build_run_manifest(
#     model_slug: str,
#     timestamp: str,
#     labels: list[str],
#     split_path: Path,
#     extra: dict | None = None,
# ):
#     manifest = {
#         "model_slug": model_slug,
#         "timestamp": timestamp,
#         "labels": labels,
#         "split_path": str(split_path),
#     }

#     if extra:
#         manifest.update(extra)

#     return manifest


# def save_run_manifest(
#     result_dir: Path,
#     manifest: dict,
#     timestamp: str,
# ):
#     latest_path = result_dir / "latest_run_manifest.json"
#     dated_path = result_dir / f"{timestamp}_run_manifest.json"

#     latest_path.write_text(json.dumps(manifest, indent=2))
#     dated_path.write_text(json.dumps(manifest, indent=2))




"""Shared data and evaluation helpers for the thesis notebooks.

This module keeps the repetitive dataset-loading, path-remapping, metric, and
result-saving utilities in one place so the notebooks can stay focused on
model-specific logic.

This version supports the hybrid two-UAV / one-UGV dataset exported by:

    03_dataset/exporters/export_hybrid_maneuver_dataset.py

Expected dataset structure:

    03_dataset/husky_control_dataset/
        run_xxx/
            frames.jsonl
            schema.json
            manifest.json
            assets/
                husky_local/planar_scan/*.npy
                husky_local/front_points/*.npy
                uav1/front_points/*.npy
                uav2/front_points/*.npy

Each JSONL frame may include:

    agents
    edges
    modalities
    observation
    state
    goal
    goal_features
    teacher
    readiness
    uav_context
    network_state

The main supervised trajectory target is built by sliding windows:

    past_len observed frames  ->  future_len future ego (x, y) trajectory

Current intended agent order:

    [husky_local, uav1, uav2]

The ego agent is always the Husky. The two UAVs provide left/right aerial context
and forward hazard information.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )
    from sklearn.preprocessing import label_binarize

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

DEFAULT_LABELS = [
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
]

REDUCED_LABELS = [
    "go_to_goal",
    "avoid_left",
    "avoid_right",
    "commit_forward",
    "arrived",
]

PLATFORM_ONEHOT = {
    "UGV": [1.0, 0.0],
    "UAV": [0.0, 1.0],
}

DEFAULT_AGENT_ORDER = [
    "husky_local",
    "uav1",
    "uav2",
]

DEFAULT_COMMAND = {
    "linear_x": 0.0,
    "linear_y": 0.0,
    "linear_z": 0.0,
    "angular_z": 0.0,
}

DEFAULT_EXTERNAL_DATASET_ROOT = (
    Path.home() / "Documents/Thesis/03_dataset/husky_control_dataset"
)

DATASET_ROOT: Path | None = None
ORIGINAL_DATASET_ROOT: Path | None = None
RESULTS_ROOT: Path | None = None
WEIGHTS_ROOT: Path | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def configure_helper(
    *,
    dataset_root: Path,
    original_dataset_root: Path | None = None,
    results_root: Path | None = None,
    weights_root: Path | None = None,
) -> None:
    """Set notebook-specific roots once so helper functions stay simple."""
    global DATASET_ROOT, ORIGINAL_DATASET_ROOT, RESULTS_ROOT, WEIGHTS_ROOT

    DATASET_ROOT = Path(dataset_root).expanduser().resolve()
    ORIGINAL_DATASET_ROOT = (
        Path(original_dataset_root).expanduser().resolve()
        if original_dataset_root is not None
        else DATASET_ROOT
    )
    RESULTS_ROOT = (
        Path(results_root).expanduser().resolve()
        if results_root is not None
        else None
    )
    WEIGHTS_ROOT = (
        Path(weights_root).expanduser().resolve()
        if weights_root is not None
        else None
    )

    load_npy_cached.cache_clear()


def _require_roots() -> tuple[Path, Path]:
    if DATASET_ROOT is None or ORIGINAL_DATASET_ROOT is None:
        raise RuntimeError(
            "dataset_helper is not configured. "
            "Call configure_helper(...) in the notebook setup cell first."
        )

    return DATASET_ROOT, ORIGINAL_DATASET_ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def build_label_mapping(label_mode: str):
    """Return labels and mapping.

    label_mode='full':
        keeps all controller states.

    label_mode='reduced':
        removes transitional/recovery states and keeps the most useful maneuver
        classes for classification-style auxiliary training.
    """
    if label_mode == "full":
        labels = list(DEFAULT_LABELS)
        mapping = {label: label for label in DEFAULT_LABELS}
        return labels, mapping

    labels = list(REDUCED_LABELS)
    mapping = {
        "bootstrap": None,
        "go_to_goal": "go_to_goal",
        "avoid_left": "avoid_left",
        "avoid_right": "avoid_right",
        "commit_forward": "commit_forward",
        "reverse": None,
        "recover": None,
        "reassess": None,
        "arrived": "arrived",
        "stop": None,
    }

    return labels, mapping


# ---------------------------------------------------------------------------
# File discovery and path remapping
# ---------------------------------------------------------------------------

def _frame_files_under(root: Path) -> list[Path]:
    root = Path(root)

    if (root / "frames.jsonl").exists():
        return [root / "frames.jsonl"]

    return sorted(root.glob("*/frames.jsonl"))


def discover_frame_files(dataset_root: Path) -> list[Path]:
    """Find extracted frame files.

    The function tries several sensible roots to avoid notebook breakage when
    paths change after moving the dataset or restarting the kernel.
    """
    candidate_roots: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path | None) -> None:
        if path is None:
            return

        path = Path(path).expanduser()

        try:
            path = path.resolve()
        except Exception:
            pass

        if path in seen:
            return

        seen.add(path)
        candidate_roots.append(path)

    add_candidate(Path(dataset_root))
    add_candidate(DATASET_ROOT)
    add_candidate(ORIGINAL_DATASET_ROOT)
    add_candidate(DEFAULT_EXTERNAL_DATASET_ROOT)

    for root in candidate_roots:
        frame_files = _frame_files_under(root)
        if frame_files:
            return frame_files

    return []


def remap_dataset_path(path_str: str) -> Path:
    """Map stored asset paths to the current local dataset root."""
    dataset_root, original_dataset_root = _require_roots()

    path = Path(path_str).expanduser()

    if path.exists():
        return path

    try:
        rel = path.relative_to(original_dataset_root)
    except ValueError:
        parts = path.parts

        for dataset_marker in (
            "hybrid_maneuvers_dataset",
            "husky_control_dataset",
        ):
            if dataset_marker in parts:
                marker = parts.index(dataset_marker)
                rel = Path(*parts[marker + 1 :])

                candidate = dataset_root / rel
                if candidate.exists():
                    return candidate

                if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
                    fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
                    if fallback_candidate.exists():
                        return fallback_candidate

        return path

    candidate = dataset_root / rel

    if candidate.exists():
        return candidate

    if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
        fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
        if fallback_candidate.exists():
            return fallback_candidate

    return candidate


@lru_cache(maxsize=32768)
def load_npy_cached(path: str):
    return np.load(remap_dataset_path(path))


def load_asset_ref(ref: dict | None):
    """Load an asset reference from JSONL.

    Returns None when the reference is missing or the file cannot be loaded.
    """
    if ref is None:
        return None

    path = ref.get("path")
    if not path:
        return None

    try:
        return load_npy_cached(str(path))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sensor preprocessing
# ---------------------------------------------------------------------------

def resample_scan(
    scan: np.ndarray,
    num_beams: int,
    range_clip: float,
) -> np.ndarray:
    """Convert saved LaserScan array into a normalized 2 x num_beams tensor.

    Input saved by exporter:
        shape = [N, 2]
        column 0 = range
        column 1 = intensity
    """
    ranges = np.asarray(scan[:, 0], dtype=np.float32)
    intensities = np.asarray(scan[:, 1], dtype=np.float32)

    ranges = np.nan_to_num(
        ranges,
        nan=range_clip,
        posinf=range_clip,
        neginf=0.0,
    )
    ranges = np.clip(ranges, 0.0, range_clip)

    intensities = np.nan_to_num(
        intensities,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )
    intensities = np.clip(intensities, 0.0, 255.0)

    if ranges.shape[0] != num_beams:
        src_x = np.linspace(0.0, 1.0, ranges.shape[0], dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, num_beams, dtype=np.float32)

        ranges = np.interp(dst_x, src_x, ranges).astype(np.float32)
        intensities = np.interp(dst_x, src_x, intensities).astype(np.float32)

    return np.stack(
        [
            ranges / max(range_clip, 1e-6),
            intensities / 255.0,
        ],
        axis=0,
    ).astype(np.float32)


def summarize_pointcloud_corridor(
    points: np.ndarray,
    *,
    max_points: int = 512,
    x_min: float = 0.0,
    x_max: float = 25.0,
    y_abs_max: float = 12.0,
    z_min: float = -5.0,
    z_max: float = 25.0,
) -> np.ndarray:
    """Filter and downsample a point cloud into a fixed-size Nx4 array.

    Output:
        shape = [max_points, 4]
        columns = x, y, z, intensity
    """
    if points is None or points.size == 0:
        return np.zeros((max_points, 4), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((max_points, 4), dtype=np.float32)

    if points.shape[1] == 3:
        zeros = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, zeros], axis=1)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= x_min)
        & (x <= x_max)
        & (np.abs(y) <= y_abs_max)
        & (z >= z_min)
        & (z <= z_max)
    )

    filtered = points[mask, :4]

    if filtered.shape[0] == 0:
        return np.zeros((max_points, 4), dtype=np.float32)

    if filtered.shape[0] >= max_points:
        idx = np.linspace(0, filtered.shape[0] - 1, max_points).astype(np.int64)
        filtered = filtered[idx]
    else:
        pad = np.zeros((max_points - filtered.shape[0], 4), dtype=np.float32)
        filtered = np.concatenate([filtered, pad], axis=0)

    return filtered.astype(np.float32)


def hazard_summary_from_pointcloud(
    points: np.ndarray,
    *,
    x_min: float = 0.0,
    x_max: float = 25.0,
    center_half_width: float = 2.0,
    side_width: float = 6.0,
    z_min: float = -2.0,
    z_max: float = 5.0,
) -> np.ndarray:
    """Convert a point cloud into simple [left, center, right] hazard counts.

    Output:
        np.ndarray with shape [3]
        columns = log_left_count, log_center_count, log_right_count
    """
    if points is None or points.size == 0:
        return np.zeros(3, dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros(3, dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    valid = (
        (x >= x_min)
        & (x <= x_max)
        & (z >= z_min)
        & (z <= z_max)
        & (np.abs(y) <= side_width)
    )

    yv = y[valid]

    left = np.sum((yv > center_half_width) & (yv <= side_width))
    center = np.sum(np.abs(yv) <= center_half_width)
    right = np.sum((yv < -center_half_width) & (yv >= -side_width))

    counts = np.asarray([left, center, right], dtype=np.float32)
    return np.log1p(counts)


def hazard_summary_from_dict(summary: dict | None) -> np.ndarray:
    """Read the compact hazard summary stored directly in JSONL.

    Supports summaries generated by the updated exporter:
        log_left
        log_center
        log_right
    """
    if not isinstance(summary, dict):
        return np.zeros(3, dtype=np.float32)

    if "log_left" in summary or "log_center" in summary or "log_right" in summary:
        return np.asarray(
            [
                float(summary.get("log_left", 0.0)),
                float(summary.get("log_center", 0.0)),
                float(summary.get("log_right", 0.0)),
            ],
            dtype=np.float32,
        )

    return np.asarray(
        [
            np.log1p(float(summary.get("left_count", 0.0))),
            np.log1p(float(summary.get("center_count", 0.0))),
            np.log1p(float(summary.get("right_count", 0.0))),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Frame accessors supporting old and new schema
# ---------------------------------------------------------------------------

def canonical_agent_order(ego_id: str = "husky_local") -> list[str]:
    """Return the model graph agent order.

    New structure:
        husky_local + uav1 + uav2
    """
    if ego_id == "husky_local":
        return ["husky_local", "uav1", "uav2"]

    order = [ego_id]
    for agent_id in DEFAULT_AGENT_ORDER:
        if agent_id not in order:
            order.append(agent_id)

    return order


def _zero_state() -> dict:
    return {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "yaw": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "wz": 0.0,
    }


def _default_agent_node(agent_id: str) -> dict:
    platform = "UAV" if agent_id.startswith("uav") else "UGV"

    return {
        "id": agent_id,
        "available": False,
        "platform_type": platform,
        "state": _zero_state(),
        "start": None,
        "goal": None,
        "goal_features": None,
        "command": dict(DEFAULT_COMMAND),
        "controller_state": None,
        "obstacle_action": None,
        "obstacle_clearance": None,
        "ready": None,
    }


def frame_agents(frame: dict) -> dict:
    """Return normalized agent dictionary for both old and new exports."""
    ego_id = frame.get("ego_id", "husky_local")

    if "agents" in frame and isinstance(frame["agents"], dict):
        agents = dict(frame["agents"])
    else:
        agents = {
            ego_id: {
                "id": ego_id,
                "available": frame.get("state") is not None,
                "platform_type": "UGV",
                "state": frame.get("state"),
                "start": None,
                "goal": frame.get("goal"),
                "goal_features": frame.get("goal_features"),
                "command": frame.get("teacher", {}).get("command")
                or dict(DEFAULT_COMMAND),
                "controller_state": frame.get("teacher", {}).get("controller_state"),
                "obstacle_action": frame.get("teacher", {}).get("obstacle_action"),
                "obstacle_clearance": frame.get("teacher", {}).get("obstacle_clearance"),
                "ready": None,
            }
        }

        uav_context = frame.get("uav_context", {})
        for uav_id in ["uav1", "uav2"]:
            ctx = uav_context.get(uav_id, {}) if isinstance(uav_context, dict) else {}
            agents[uav_id] = {
                "id": uav_id,
                "available": bool(ctx.get("available", False)),
                "platform_type": "UAV",
                "state": ctx.get("state"),
                "start": None,
                "goal": ctx.get("goal"),
                "goal_features": ctx.get("goal_features"),
                "command": dict(DEFAULT_COMMAND),
                "controller_state": None,
                "obstacle_action": None,
                "obstacle_clearance": None,
                "ready": ctx.get("ready"),
            }

        # Backward compatibility with older one-UAV + second-Husky export.
        other = frame.get("other_husky")
        if isinstance(other, dict) and "husky_2" not in agents:
            agents["husky_2"] = {
                "id": "husky_2",
                "available": bool(other.get("available", False)),
                "platform_type": "UGV",
                "state": other.get("state"),
                "start": None,
                "goal": other.get("goal"),
                "goal_features": other.get("goal_features"),
                "command": other.get("teacher_command") or dict(DEFAULT_COMMAND),
                "controller_state": None,
                "obstacle_action": None,
                "obstacle_clearance": None,
                "ready": None,
            }

    for agent_id in canonical_agent_order(ego_id):
        if agent_id not in agents:
            agents[agent_id] = _default_agent_node(agent_id)
            continue

        node = dict(_default_agent_node(agent_id)) | dict(agents[agent_id])

        if node.get("state") is None:
            node["state"] = _zero_state()
            node["available"] = False

        if node.get("command") is None:
            node["command"] = dict(DEFAULT_COMMAND)

        if node.get("platform_type") is None:
            node["platform_type"] = "UAV" if agent_id.startswith("uav") else "UGV"

        agents[agent_id] = node

    return agents


def frame_state(frame: dict) -> dict:
    if "agents" in frame:
        ego_id = frame.get("ego_id", "husky_local")
        state = frame["agents"].get(ego_id, {}).get("state")
        return state if state is not None else _zero_state()

    state = frame.get("state")
    return state if state is not None else _zero_state()


def frame_scan_ref(frame: dict):
    if "modalities" in frame:
        ref = frame["modalities"].get("ego_planar_scan")
        if ref is not None:
            return ref

    if "observation" in frame:
        return frame["observation"].get("ego_planar_scan")

    return None


def frame_ego_pointcloud_ref(frame: dict):
    if "modalities" in frame:
        ref = frame["modalities"].get("ego_front_pointcloud")
        if ref is not None:
            return ref

    if "observation" in frame:
        return frame["observation"].get("ego_front_pointcloud")

    return None


def frame_uav_pointcloud_ref(frame: dict, uav_id: str = "uav1"):
    key = f"{uav_id}_front_pointcloud"

    if "modalities" in frame:
        ref = frame["modalities"].get(key)
        if ref is not None:
            return ref

    if "observation" in frame:
        return frame["observation"].get(key)

    return None


def frame_uav_hazard_summary(frame: dict, uav_id: str = "uav1") -> np.ndarray:
    key = f"{uav_id}_hazard_summary"

    if "modalities" in frame and key in frame["modalities"]:
        return hazard_summary_from_dict(frame["modalities"].get(key))

    if "observation" in frame and key in frame["observation"]:
        return hazard_summary_from_dict(frame["observation"].get(key))

    uav_context = frame.get("uav_context", {})
    if isinstance(uav_context, dict):
        ctx = uav_context.get(uav_id, {})
        if isinstance(ctx, dict):
            return hazard_summary_from_dict(ctx.get("hazard_summary"))

    return np.zeros(3, dtype=np.float32)


def frame_teacher_label(frame: dict) -> str:
    teacher = frame.get("teacher", {})

    label = teacher.get("label")
    if label is not None:
        return str(label)

    controller_state = teacher.get("controller_state")
    if controller_state:
        return str(controller_state)

    obstacle_action = teacher.get("obstacle_action")
    if obstacle_action:
        obstacle_action = str(obstacle_action).strip().lower()
        if obstacle_action.endswith("left"):
            return "avoid_left"
        if obstacle_action.endswith("right"):
            return "avoid_right"

    return "go_to_goal"


# ---------------------------------------------------------------------------
# Graph features
# ---------------------------------------------------------------------------

def node_feature(node: dict, ego_state: dict) -> list[float]:
    """Build a fixed 14D node feature vector."""
    state = node.get("state") or _zero_state()
    goal = node.get("goal") or {
        "x": state["x"],
        "y": state["y"],
        "z": state["z"],
    }
    command = node.get("command") or dict(DEFAULT_COMMAND)
    platform = PLATFORM_ONEHOT.get(
        node.get("platform_type", "UGV"),
        [0.0, 0.0],
    )

    return [
        float(state.get("x", 0.0) - ego_state.get("x", 0.0)),
        float(state.get("y", 0.0) - ego_state.get("y", 0.0)),
        float(state.get("z", 0.0) - ego_state.get("z", 0.0)),
        float(state.get("vx", 0.0)),
        float(state.get("vy", 0.0)),
        float(state.get("vz", 0.0)),
        float(state.get("wz", 0.0)),
        float(goal.get("x", state.get("x", 0.0)) - state.get("x", 0.0)),
        float(goal.get("y", state.get("y", 0.0)) - state.get("y", 0.0)),
        float(goal.get("z", state.get("z", 0.0)) - state.get("z", 0.0)),
        float(command.get("linear_x", 0.0)),
        float(command.get("angular_z", 0.0)),
        float(platform[0]),
        float(platform[1]),
    ]


def graph_node_features_for_frame(
    frame: dict,
    order: list[str] | None = None,
) -> np.ndarray:
    """Return graph node feature matrix [num_nodes, 14]."""
    agents = frame_agents(frame)
    ego_id = frame.get("ego_id", "husky_local")
    ego_state = agents[ego_id].get("state") or _zero_state()
    order = order or canonical_agent_order(ego_id)

    features = []

    for agent_id in order:
        node = agents.get(agent_id, _default_agent_node(agent_id))
        features.append(node_feature(node, ego_state))

    return np.asarray(features, dtype=np.float32)


def build_edge_lookup(frame: dict) -> dict[tuple[str, str], dict]:
    edges = frame.get("edges") or []
    return {
        (edge["source"], edge["target"]): edge
        for edge in edges
        if "source" in edge and "target" in edge
    }


def _fallback_edge(src_node: dict, dst_node: dict) -> dict:
    src_state = src_node.get("state") or _zero_state()
    dst_state = dst_node.get("state") or _zero_state()

    dx = float(dst_state.get("x", 0.0) - src_state.get("x", 0.0))
    dy = float(dst_state.get("y", 0.0) - src_state.get("y", 0.0))
    dz = float(dst_state.get("z", 0.0) - src_state.get("z", 0.0))

    distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    bearing = float(np.arctan2(dy, dx))

    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "distance": distance,
        "inv_distance": float(1.0 / max(distance, 1e-6)),
        "bearing_sin": float(np.sin(bearing)),
        "bearing_cos": float(np.cos(bearing)),
        "same_platform": float(
            src_node.get("platform_type") == dst_node.get("platform_type")
        ),
        "latency_s": 0.0,
        "packet_loss": 0.0,
        "link_quality": 1.0,
    }


def _edge_network_value(edge: dict, name: str, default: float) -> float:
    if name in edge:
        return float(edge.get(name, default))

    network = edge.get("network", {})
    if isinstance(network, dict) and name in network:
        return float(network.get(name, default))

    return float(default)


def edge_features_for_order(frame: dict, order: list[str]) -> list[list[list[float]]]:
    """Return edge features [num_nodes, num_nodes, 11].

    Edge feature order:
        dx, dy, dz,
        distance, inv_distance,
        bearing_sin, bearing_cos,
        same_platform,
        latency_s, packet_loss, link_quality
    """
    agents = frame_agents(frame)
    edge_map = build_edge_lookup(frame)

    src_edges = []

    for src in order:
        row = []

        for dst in order:
            if src == dst:
                row.append([0.0] * 11)
                continue

            edge = edge_map.get((src, dst))

            if edge is None:
                edge = _fallback_edge(
                    agents.get(src, _default_agent_node(src)),
                    agents.get(dst, _default_agent_node(dst)),
                )

            row.append(
                [
                    float(edge.get("dx", 0.0)),
                    float(edge.get("dy", 0.0)),
                    float(edge.get("dz", 0.0)),
                    float(edge.get("distance", 0.0)),
                    float(edge.get("inv_distance", 0.0)),
                    float(edge.get("bearing_sin", 0.0)),
                    float(edge.get("bearing_cos", 1.0)),
                    float(edge.get("same_platform", 0.0)),
                    _edge_network_value(edge, "latency_s", 0.0),
                    _edge_network_value(edge, "packet_loss", 0.0),
                    _edge_network_value(edge, "link_quality", 1.0),
                ]
            )

        src_edges.append(row)

    return src_edges


def graph_edge_features_for_frame(
    frame: dict,
    order: list[str] | None = None,
) -> np.ndarray:
    ego_id = frame.get("ego_id", "husky_local")
    order = order or canonical_agent_order(ego_id)

    return np.asarray(edge_features_for_order(frame, order), dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset grouping and sample construction
# ---------------------------------------------------------------------------

def group_streams(
    dataset_root: Path,
    allowed_labels: set[str] | None = None,
    label_mapping: dict | None = None,
    require_uav: bool = False,
    require_both_uavs: bool = False,
    require_uav_pointclouds: bool = False,
):
    """Group frames by episode_id and ego_id.

    require_uav=True:
        require at least uav1 state.

    require_both_uavs=True:
        require both uav1 and uav2 states.

    require_uav_pointclouds=True:
        require both uav1 and uav2 pointcloud references.
    """
    streams = []
    frame_files = discover_frame_files(dataset_root)

    allowed_labels = set(allowed_labels) if allowed_labels is not None else None
    label_mapping = label_mapping or {}

    for frames_path in frame_files:
        with frames_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]

        buckets = {}

        for row in rows:
            raw_label = frame_teacher_label(row)
            mapped_label = label_mapping.get(raw_label, raw_label)

            if allowed_labels is not None and (
                mapped_label is None or mapped_label not in allowed_labels
            ):
                continue

            if frame_scan_ref(row) is None:
                continue

            if frame_state(row) is None:
                continue

            readiness = row.get("readiness", {})

            if require_uav:
                if not readiness.get("has_uav1_state", False):
                    continue

            if require_both_uavs:
                if not readiness.get("has_uav1_state", False):
                    continue
                if not readiness.get("has_uav2_state", False):
                    continue

            if require_uav_pointclouds:
                if not readiness.get("has_uav1_pointcloud", False):
                    continue
                if not readiness.get("has_uav2_pointcloud", False):
                    continue

            row = dict(row)
            row["teacher"] = dict(row.get("teacher", {}))
            row["teacher"]["raw_label"] = raw_label
            row["teacher"]["label"] = mapped_label

            key = f"{row['episode_id']}::{row['ego_id']}"
            buckets.setdefault(key, []).append(row)

        for key in sorted(buckets):
            stream = sorted(
                buckets[key],
                key=lambda item: int(item["timestamp_ns"]),
            )

            if stream:
                streams.append(stream)

    if not streams:
        raise RuntimeError(
            f"No usable frame streams found under {dataset_root}. "
            f"Discovered {len(frame_files)} frame file(s); "
            f"check label filtering, dataset paths, and whether export completed."
        )

    return streams


def build_sample_table(
    streams: list[list[dict]],
    past_len: int,
    future_len: int,
):
    """Build sliding-window sample metadata.

    The target is future ego trajectory relative to the anchor frame.
    """
    sample_table = []

    for stream_idx, stream in enumerate(streams):
        usable = len(stream) - past_len - future_len + 1

        for start in range(max(0, usable)):
            anchor = stream[start + past_len - 1]
            future_frames = stream[start + past_len : start + past_len + future_len]

            anchor_state = frame_state(anchor)
            anchor_ts = int(anchor["timestamp_ns"])

            future_xy = []
            future_dt = []

            valid = True

            for future_frame in future_frames:
                state = frame_state(future_frame)

                if state is None:
                    valid = False
                    break

                future_xy.append(
                    [
                        float(state["x"] - anchor_state["x"]),
                        float(state["y"] - anchor_state["y"]),
                    ]
                )
                future_dt.append(
                    (int(future_frame["timestamp_ns"]) - anchor_ts) * 1e-9
                )

            if not valid:
                continue

            sample_table.append(
                {
                    "sample_id": f"stream{stream_idx:03d}_start{start:05d}",
                    "stream_index": stream_idx,
                    "stream_idx": stream_idx,
                    "start_index": start,
                    "start": start,
                    "anchor_index": start + past_len - 1,
                    "ego_id": anchor["ego_id"],
                    "label": anchor["teacher"].get("label"),
                    "raw_label": anchor["teacher"].get("raw_label"),
                    "future_xy": future_xy,
                    "future_dt": future_dt,
                }
            )

    return sample_table


def save_or_load_fixed_split(
    sample_table,
    split_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    past_len: int,
    future_len: int,
):
    """Create or reuse a deterministic train/val/test split."""
    split_path = Path(split_path)

    if split_path.exists():
        with split_path.open() as f:
            split_info = json.load(f)

        current_sample_ids = [row["sample_id"] for row in sample_table]

        if (
            split_info.get("sample_count") == len(sample_table)
            and split_info.get("past_len") == past_len
            and split_info.get("future_len") == future_len
            and split_info.get("sample_ids") == current_sample_ids
        ):
            return split_info

    rng = random.Random(seed)
    indices = list(range(len(sample_table)))
    rng.shuffle(indices)

    if len(indices) < 3:
        raise RuntimeError(
            f"Need at least 3 samples to split train/val/test, got {len(indices)}."
        )

    train_len = max(1, int(len(indices) * train_ratio))
    val_len = max(1, int(len(indices) * val_ratio))
    test_len = len(indices) - train_len - val_len

    if test_len < 1:
        test_len = 1
        if train_len > val_len:
            train_len -= 1
        else:
            val_len -= 1

    split_info = {
        "seed": seed,
        "sample_count": len(sample_table),
        "past_len": past_len,
        "future_len": future_len,
        "train_indices": indices[:train_len],
        "val_indices": indices[train_len : train_len + val_len],
        "test_indices": indices[train_len + val_len :],
        "sample_ids": [row["sample_id"] for row in sample_table],
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_info, indent=2))

    return split_info


# ---------------------------------------------------------------------------
# Optional ready-to-use feature builders for notebooks
# ---------------------------------------------------------------------------

def build_past_ego_xy(
    stream: list[dict],
    start: int,
    past_len: int,
) -> np.ndarray:
    """Return past ego xy relative to the anchor frame.

    Output:
        shape = [past_len, 2]
    """
    past_frames = stream[start : start + past_len]
    anchor_state = frame_state(past_frames[-1])

    xy = []

    for frame in past_frames:
        state = frame_state(frame)
        xy.append(
            [
                float(state["x"] - anchor_state["x"]),
                float(state["y"] - anchor_state["y"]),
            ]
        )

    return np.asarray(xy, dtype=np.float32)


def build_past_graph_sequence(
    stream: list[dict],
    start: int,
    past_len: int,
    order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return past graph node and edge sequences.

    Outputs:
        node_seq: [past_len, num_nodes, 14]
        edge_seq: [past_len, num_nodes, num_nodes, 11]

    With the new dataset, num_nodes is normally 3:
        husky_local, uav1, uav2
    """
    past_frames = stream[start : start + past_len]

    if order is None:
        order = canonical_agent_order(past_frames[-1].get("ego_id", "husky_local"))

    node_seq = []
    edge_seq = []

    for frame in past_frames:
        node_seq.append(graph_node_features_for_frame(frame, order))
        edge_seq.append(graph_edge_features_for_frame(frame, order))

    return (
        np.asarray(node_seq, dtype=np.float32),
        np.asarray(edge_seq, dtype=np.float32),
    )


def build_past_scan_sequence(
    stream: list[dict],
    start: int,
    past_len: int,
    *,
    num_beams: int = 256,
    range_clip: float = 30.0,
) -> np.ndarray:
    """Return past ego lidar sequence.

    Output:
        shape = [past_len, 2, num_beams]
    """
    past_frames = stream[start : start + past_len]
    scans = []

    for frame in past_frames:
        ref = frame_scan_ref(frame)
        scan = load_asset_ref(ref)

        if scan is None:
            scans.append(np.zeros((2, num_beams), dtype=np.float32))
        else:
            scans.append(resample_scan(scan, num_beams, range_clip))

    return np.asarray(scans, dtype=np.float32)


def build_past_uav_hazard_sequence(
    stream: list[dict],
    start: int,
    past_len: int,
    *,
    include_uav1: bool = True,
    include_uav2: bool = True,
) -> np.ndarray:
    """Return past UAV hazard summaries.

    New output by default:
        shape = [past_len, 6]

    Columns:
        uav1_left, uav1_center, uav1_right,
        uav2_left, uav2_center, uav2_right

    If only one UAV is requested, output shape becomes [past_len, 3].
    """
    past_frames = stream[start : start + past_len]
    summaries = []

    for frame in past_frames:
        parts = []

        if include_uav1:
            summary1 = frame_uav_hazard_summary(frame, "uav1")
            if np.allclose(summary1, 0.0):
                ref1 = frame_uav_pointcloud_ref(frame, "uav1")
                points1 = load_asset_ref(ref1)
                if points1 is not None:
                    summary1 = hazard_summary_from_pointcloud(points1)
            parts.append(summary1)

        if include_uav2:
            summary2 = frame_uav_hazard_summary(frame, "uav2")
            if np.allclose(summary2, 0.0):
                ref2 = frame_uav_pointcloud_ref(frame, "uav2")
                points2 = load_asset_ref(ref2)
                if points2 is not None:
                    summary2 = hazard_summary_from_pointcloud(points2)
            parts.append(summary2)

        if parts:
            summaries.append(np.concatenate(parts).astype(np.float32))
        else:
            summaries.append(np.zeros(0, dtype=np.float32))

    return np.asarray(summaries, dtype=np.float32)


def build_past_uav_pointcloud_sequence(
    stream: list[dict],
    start: int,
    past_len: int,
    *,
    uav_id: str,
    max_points: int = 512,
) -> np.ndarray:
    """Return a fixed-size pointcloud sequence for one UAV.

    Output:
        shape = [past_len, max_points, 4]
    """
    past_frames = stream[start : start + past_len]
    clouds = []

    for frame in past_frames:
        ref = frame_uav_pointcloud_ref(frame, uav_id)
        points = load_asset_ref(ref)

        if points is None:
            clouds.append(np.zeros((max_points, 4), dtype=np.float32))
        else:
            clouds.append(
                summarize_pointcloud_corridor(
                    points,
                    max_points=max_points,
                )
            )

    return np.asarray(clouds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics and class helpers
# ---------------------------------------------------------------------------

def build_class_weights(
    label_indices: list[int],
    num_classes: int,
):
    counts = Counter(label_indices)
    total = sum(counts.values())

    weights = []

    for idx in range(num_classes):
        count = counts.get(idx, 0)
        weights.append(0.0 if count == 0 else total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


def compute_trajectory_metrics(
    pred_future_xy: np.ndarray,
    true_future_xy: np.ndarray,
):
    diff = pred_future_xy - true_future_xy
    dist = np.linalg.norm(diff, axis=-1)

    return {
        "ADE": float(dist.mean()),
        "FDE": float(dist[:, -1].mean()),
        "RMSE": float(np.sqrt(np.mean(np.sum(diff**2, axis=-1)))),
    }


def compute_classification_metrics_from_probs(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
):
    preds = probabilities.argmax(axis=1)
    num_classes = len(labels)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for truth, guess in zip(targets, preds):
        confusion[int(truth), int(guess)] += 1

    precisions, recalls, f1s = [], [], []

    for idx in range(num_classes):
        tp = float(confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - tp)
        fp = float(confusion[:, idx].sum() - tp)

        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = (
            0.0
            if (precision + recall) == 0.0
            else (2.0 * precision * recall / (precision + recall))
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = {
        "accuracy": float((preds == targets).mean()),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion.tolist(),
        "ADE": None,
        "FDE": None,
        "RMSE": None,
    }

    return metrics, preds, confusion


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------

def save_training_history(
    history: dict,
    out_path: Path,
):
    pd.DataFrame(history).to_csv(out_path, index=False)


def save_confusion_matrix(
    confusion: np.ndarray,
    labels: list[str],
    csv_path: Path,
    png_path: Path,
    title: str,
):
    df = pd.DataFrame(confusion, index=labels, columns=labels)
    df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_pr_curves(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
    out_dir: Path,
):
    summary = {
        "roc_auc_macro": None,
        "pr_auc_macro": None,
        "status": "skipped",
    }

    if not SKLEARN_AVAILABLE:
        return summary

    y_true = label_binarize(targets, classes=list(range(len(labels))))

    roc_aucs = []
    pr_aucs = []

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for idx, label in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, idx], probabilities[:, idx])
            roc_auc_value = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(
                y_true[:, idx],
                probabilities[:, idx],
            )
            pr_auc_value = average_precision_score(
                y_true[:, idx],
                probabilities[:, idx],
            )

            roc_aucs.append(roc_auc_value)
            pr_aucs.append(pr_auc_value)

            ax_roc.plot(
                fpr,
                tpr,
                label=f"{label} (AUC={roc_auc_value:.3f})",
            )
            ax_pr.plot(
                recall,
                precision,
                label=f"{label} (AP={pr_auc_value:.3f})",
            )
        except Exception:
            continue

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_title("One-vs-Rest ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8)

    plt.tight_layout()
    fig_roc.savefig(out_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_roc)

    ax_pr.set_title("One-vs-Rest Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)

    plt.tight_layout()
    fig_pr.savefig(out_dir / "pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_pr)

    if roc_aucs:
        summary["roc_auc_macro"] = float(np.mean(roc_aucs))

    if pr_aucs:
        summary["pr_auc_macro"] = float(np.mean(pr_aucs))

    summary["status"] = "saved"

    return summary


def save_predictions_csv(
    sample_ids,
    targets,
    preds,
    probabilities,
    labels,
    out_path: Path,
):
    rows = []

    for sid, truth, pred, probs in zip(
        sample_ids,
        targets,
        preds,
        probabilities,
    ):
        row = {
            "sample_id": sid,
            "true_label": labels[int(truth)],
            "pred_label": labels[int(pred)],
        }

        for idx, label in enumerate(labels):
            row[f"prob_{label}"] = float(probs[idx])

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_history_plot(
    history: dict,
    out_path: Path,
    title_prefix: str,
):
    if not history or len(history.get("epoch", [])) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_title(f"{title_prefix}: Loss")
    axes[0].legend()

    axes[1].plot(
        history["epoch"],
        history["val_accuracy"],
        label="val_accuracy",
    )
    axes[1].set_title(f"{title_prefix}: Validation Accuracy")
    axes[1].legend()

    axes[2].plot(
        history["epoch"],
        history["val_macro_f1"],
        label="val_macro_f1",
    )
    axes[2].set_title(f"{title_prefix}: Validation Macro-F1")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_overlay_plots(
    pred_future_xy: np.ndarray,
    true_future_xy: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
    output_dir: Path,
    prefix: str,
    max_plots: int = 8,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    total = min(max_plots, pred_future_xy.shape[0])

    for idx in range(total):
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.plot([0.0], [0.0], "ko", label="anchor")
        ax.plot(
            true_future_xy[idx, :, 0],
            true_future_xy[idx, :, 1],
            "-o",
            label="ground truth",
        )
        ax.plot(
            pred_future_xy[idx, :, 0],
            pred_future_xy[idx, :, 1],
            "--o",
            label="prediction",
        )

        ax.set_title(f"{prefix} sample {idx} ({labels[int(targets[idx])]})")
        ax.set_xlabel("Relative x (m)")
        ax.set_ylabel("Relative y (m)")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        path = output_dir / f"{prefix}_trajectory_overlay_{idx:02d}.png"

        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        saved.append(str(path))

    return saved


def save_mean_step_error_plot(
    pred_future_xy: np.ndarray,
    true_future_xy: np.ndarray,
    output_path: Path,
    title: str,
):
    diff = pred_future_xy - true_future_xy
    step_error = np.linalg.norm(diff, axis=-1).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        np.arange(1, len(step_error) + 1),
        step_error,
        marker="o",
    )

    ax.set_title(title)
    ax.set_xlabel("Future step")
    ax.set_ylabel("Mean displacement error (m)")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


# ---------------------------------------------------------------------------
# Result directories and manifests
# ---------------------------------------------------------------------------

def prepare_result_dirs(model_slug: str):
    if RESULTS_ROOT is None or WEIGHTS_ROOT is None:
        raise RuntimeError(
            "dataset_helper output roots are not configured. "
            "Pass results_root and weights_root to configure_helper(...)."
        )

    result_dir = RESULTS_ROOT / model_slug
    weight_dir = WEIGHTS_ROOT / model_slug
    plot_dir = result_dir / "plots"

    for path in [result_dir, weight_dir, plot_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return result_dir, weight_dir, plot_dir


def timestamp_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_manifest(
    model_slug: str,
    timestamp: str,
    labels: list[str],
    split_path: Path,
    extra: dict | None = None,
):
    manifest = {
        "model_slug": model_slug,
        "timestamp": timestamp,
        "labels": labels,
        "split_path": str(split_path),
    }

    if extra:
        manifest.update(extra)

    return manifest


def save_run_manifest(
    result_dir: Path,
    manifest: dict,
    timestamp: str,
):
    latest_path = result_dir / "latest_run_manifest.json"
    dated_path = result_dir / f"{timestamp}_run_manifest.json"

    latest_path.write_text(json.dumps(manifest, indent=2))
    dated_path.write_text(json.dumps(manifest, indent=2))