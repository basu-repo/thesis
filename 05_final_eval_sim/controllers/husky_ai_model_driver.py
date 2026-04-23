"""Live Husky AI model controller.

This node builds a graph snapshot from all active agents, predicts a
short-horizon ego trajectory with a saved checkpoint, and converts that
prediction into goal-aware ``cmd_vel`` commands for the Husky.
"""

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from ai_model_predictor import (
    NODE_ORDER,
    architecture_requires_scan,
    build_runtime_model,
    infer_runtime_architecture,
)


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


PLATFORM_TYPES = {
    "husky_local": "UGV",
    "husky_2": "UGV",
    "uav1": "UAV",
}


class HuskyAIModelDriver(Node):
    """Use graph-based multi-agent context to predict and follow an ego path."""

    def __init__(
        self,
        node_name: str,
        ego_node: str,
        cmd_topic: str,
        odom_topics: dict[str, str],
        command_topics: dict[str, str],
        checkpoint_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        world_pose_topic: str | None = None,
        obstacle_action_topic: str | None = None,
        obstacle_clearance_topic: str | None = None,
        scan_topic: str | None = None,
        hazard_topic: str | None = None,
        spawn_xyz: tuple[float, float, float] | None = None,
        goals: dict[str, tuple[float, float, float]] | None = None,
        target_bias_x: float = 0.0,
        target_bias_y: float = 0.0,
        bootstrap_seconds: float = 3.0,
        bootstrap_linear_speed: float = 0.45,
        bootstrap_angular_speed: float = 0.0,
        bootstrap_turn_gain: float = 1.0,
        target_index: int = 4,
        control_period: float = 0.1,
        cmd_linear_gain: float = 0.9,
        cmd_angular_gain: float = 1.6,
        min_linear_speed: float = 0.0,
        max_linear_speed: float = 0.9,
        max_angular_speed: float = 1.2,
        heading_deadband: float = 0.08,
        goal_align_heading_threshold: float = 0.5,
        goal_align_linear_speed: float = 0.28,
        waypoint_reached_dist: float = 0.2,
        cruise_speed: float = 0.9,
        goal_tolerance: float = 0.8,
        goal_blend: float = 0.35,
        obstacle_stop_distance: float = 1.8,
        obstacle_turn_speed: float = 1.0,
        obstacle_turn_speed_close: float = 1.4,
        obstacle_scan_distance: float = 1.6,
        obstacle_clear_distance: float = 2.1,
        turn_in_place_speed: float = 0.85,
        hazard_timeout: float = 0.8,
        hazard_turn_speed: float = 0.7,
        history_size: int = 200,
        stuck_timeout_seconds: float = 3.0,
        stuck_progress_distance: float = 0.3,
        stuck_min_command_speed: float = 0.2,
        stuck_reverse_speed: float = -0.35,
        stuck_reverse_seconds: float = 1.5,
        stuck_bootstrap_seconds: float = 2.0,
        stuck_cooldown_seconds: float = 4.0,
        strict_reverse_distance: float = 0.8,
        strict_reverse_cycles: int = 4,
        post_avoid_forward_speed: float = 0.18,
        post_avoid_forward_seconds: float = 0.8,
        active_nodes: list[str] | None = None,
    ):
        super().__init__(node_name)
        if ego_node not in NODE_ORDER:
            raise ValueError(f"Unknown ego node: {ego_node}")

        checkpoint = self._resolve_checkpoint_path(checkpoint_path=checkpoint_path, summary_path=summary_path)
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model_slug = self._extract_model_slug(ckpt)
        self.runtime_architecture = infer_runtime_architecture(ckpt)
        raw_cfg = self._extract_cfg(ckpt)
        self.past_len = raw_cfg["past_len"]
        self.future_len = raw_cfg["future_len"]
        self.ego_node = ego_node
        self.scan_beams = int(raw_cfg.get("scan_beams", 512))
        self.range_clip = float(raw_cfg.get("range_clip", 30.0))
        self.uses_scan = architecture_requires_scan(self.runtime_architecture)
        self.scan_history = deque(maxlen=self.past_len)
        self.active_nodes = set(active_nodes or NODE_ORDER)
        self.debug_decision_interval = 1.0
        self.model = build_runtime_model(
            self.runtime_architecture,
            raw_cfg,
            ego_idx=NODE_ORDER.index(ego_node),
        )
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()

        self.cmd_topic = cmd_topic
        self.odom_topics = odom_topics
        self.command_topics = command_topics
        self.world_pose_topic = world_pose_topic
        self.obstacle_action_topic = obstacle_action_topic
        self.obstacle_clearance_topic = obstacle_clearance_topic
        self.scan_topic = scan_topic
        self.hazard_topic = hazard_topic
        self.spawn_xyz = spawn_xyz
        self.goals = goals or {}
        self.target_bias_x = target_bias_x
        self.target_bias_y = target_bias_y
        self.bootstrap_seconds = bootstrap_seconds
        self.bootstrap_linear_speed = bootstrap_linear_speed
        self.bootstrap_angular_speed = bootstrap_angular_speed
        self.bootstrap_turn_gain = bootstrap_turn_gain
        self.target_index = target_index
        self.control_period = control_period
        self.cmd_linear_gain = cmd_linear_gain
        self.cmd_angular_gain = cmd_angular_gain
        self.min_linear_speed = min_linear_speed
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.heading_deadband = heading_deadband
        self.goal_align_heading_threshold = goal_align_heading_threshold
        self.goal_align_linear_speed = goal_align_linear_speed
        self.waypoint_reached_dist = waypoint_reached_dist
        self.cruise_speed = cruise_speed
        self.goal_tolerance = goal_tolerance
        self.goal_blend = goal_blend
        self.obstacle_stop_distance = obstacle_stop_distance
        self.obstacle_turn_speed = obstacle_turn_speed
        self.obstacle_turn_speed_close = obstacle_turn_speed_close
        self.obstacle_scan_distance = obstacle_scan_distance
        self.obstacle_clear_distance = obstacle_clear_distance
        self.turn_in_place_speed = turn_in_place_speed
        self.hazard_timeout = hazard_timeout
        self.hazard_turn_speed = hazard_turn_speed
        self.stuck_timeout_seconds = stuck_timeout_seconds
        self.stuck_progress_distance = stuck_progress_distance
        self.stuck_min_command_speed = stuck_min_command_speed
        self.stuck_reverse_speed = stuck_reverse_speed
        self.stuck_reverse_seconds = stuck_reverse_seconds
        self.stuck_bootstrap_seconds = stuck_bootstrap_seconds
        self.stuck_cooldown_seconds = stuck_cooldown_seconds
        self.strict_reverse_distance = strict_reverse_distance
        self.strict_reverse_cycles = max(1, int(strict_reverse_cycles))
        self.post_avoid_forward_speed = post_avoid_forward_speed
        self.post_avoid_forward_seconds = post_avoid_forward_seconds

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.states: dict[str, dict | None] = {name: None for name in NODE_ORDER}
        self.model_frame_ids = {
            name: self._frame_id_from_odom_topic(topic)
            for name, topic in odom_topics.items()
        }
        self.world_positions: dict[str, tuple[float, float, float] | None] = {
            name: None for name in NODE_ORDER
        }
        self.world_yaws: dict[str, float | None] = {name: None for name in NODE_ORDER}
        self.world_pose_stamps: dict[str, float] = {name: 0.0 for name in NODE_ORDER}
        self.world_pose_anchor: dict[str, dict[str, float] | None] = {name: None for name in NODE_ORDER}
        self.commands: dict[str, dict] = {
            "husky_local": {"linear_x": 0.0, "angular_z": 0.0},
            "husky_2": {"linear_x": 0.0, "angular_z": 0.0},
            "uav1": {"linear_x": 0.0, "angular_z": 0.0},
        }
        self.current_pose = None
        self.current_yaw = 0.0
        self.predicted_path = None
        self.arrived = False
        self.obstacle_action = "clear"
        self.obstacle_clearance = (float("inf"), float("inf"), float("inf"))
        self.last_obstacle_log = 0.0
        self.graph_history = deque(maxlen=self.past_len)
        self.path_history = deque(maxlen=history_size)
        self.progress_history = deque(maxlen=history_size)
        self.start_time = time.monotonic()
        self.last_snapshot_time = 0.0
        self.last_diag_log = 0.0
        self.last_decision_log = 0.0
        self.last_command_linear_x = 0.0
        self.last_command_angular_z = 0.0
        self.stuck_recovery_until = 0.0
        self.stuck_bootstrap_until = 0.0
        self.stuck_cooldown_until = 0.0
        self.avoid_direction: str | None = None
        self.avoid_start_heading: float | None = None
        self.clear_path_since: float | None = None
        self.strict_blocked_cycles = 0
        self.world_pose_timeout = 0.5
        self.last_pose_source_log = 0.0

        self.create_subscription(Odometry, odom_topics["husky_local"], self._make_odom_cb("husky_local"), 10)
        self.create_subscription(Odometry, odom_topics["husky_2"], self._make_odom_cb("husky_2"), 10)
        self.create_subscription(Odometry, odom_topics["uav1"], self._make_odom_cb("uav1"), 10)
        if self.world_pose_topic is not None:
            self.create_subscription(TFMessage, self.world_pose_topic, self.world_pose_cb, 10)
        if self.obstacle_action_topic is not None:
            self.create_subscription(String, self.obstacle_action_topic, self.obstacle_action_cb, 10)
        if self.obstacle_clearance_topic is not None:
            self.create_subscription(Vector3, self.obstacle_clearance_topic, self.obstacle_clearance_cb, 10)
        if self.uses_scan:
            if self.scan_topic is None:
                raise ValueError(f"{self.model_slug} requires scan_topic for live inference.")
            self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(Twist, command_topics["husky_local"], self._make_cmd_cb("husky_local"), 10)
        self.create_subscription(Twist, command_topics["husky_2"], self._make_cmd_cb("husky_2"), 10)

        self.timer = self.create_timer(self.control_period, self.step)
        self.get_logger().info(
            f"Loaded AI model ({self.model_slug}, arch={self.runtime_architecture}) "
            f"for {self.ego_node} on {self.cmd_topic} from {checkpoint}"
        )

    @staticmethod
    def _frame_id_from_odom_topic(topic: str) -> str | None:
        parts = [part for part in topic.split("/") if part]
        return parts[1] if len(parts) >= 2 else None

    @staticmethod
    def _resolve_checkpoint_path(
        *,
        checkpoint_path: str | Path | None,
        summary_path: str | Path | None,
    ) -> Path:
        if checkpoint_path is not None:
            return Path(checkpoint_path).expanduser().resolve()
        if summary_path is None:
            raise ValueError("Either checkpoint_path or summary_path must be provided.")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        model_path = summary.get("model_path")
        if not model_path:
            raise ValueError(f"Summary file {summary_path} does not contain model_path.")
        return Path(model_path).expanduser().resolve()

    @staticmethod
    def _extract_cfg(ckpt: dict[str, Any]) -> dict[str, Any]:
        run_manifest = ckpt.get("run_manifest") or {}
        if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
            cfg = dict(ckpt["cfg"])
            cfg.setdefault("scan_beams", 512)
            cfg.setdefault("range_clip", 30.0)
            return cfg

        model_slug = str(run_manifest.get("model_slug") or ckpt.get("model_slug") or "").lower()
        shared_scan_graph_cfg = {
            "node_dim": 14,
            "edge_dim": 8,
            "cnn_hidden": int(run_manifest.get("cnn_hidden", 96)),
            "graph_hidden": int(run_manifest.get("graph_hidden", 96)),
            "fusion_hidden": int(run_manifest.get("fusion_hidden", 128)),
            "future_len": int(run_manifest.get("future_len", 5)),
            "past_len": 10,
            "msg_passes": int(run_manifest.get("msg_passes", 2)),
            "dropout": float(run_manifest.get("dropout", 0.1)),
            "scan_beams": 512,
            "range_clip": 30.0,
        }

        if model_slug == "cnn_gnn_lstm":
            return {
                **shared_scan_graph_cfg,
                "lstm_hidden": int(run_manifest.get("lstm_hidden", 128)),
                "lstm_layers": int(run_manifest.get("lstm_layers", 1)),
            }

        if model_slug == "cnn_gnn_transformer":
            return {
                **shared_scan_graph_cfg,
                "transformer_heads": int(run_manifest.get("transformer_heads", 4)),
                "transformer_ff": int(run_manifest.get("transformer_ff", 256)),
                "transformer_layers": int(run_manifest.get("transformer_layers", 2)),
            }

        if model_slug == "cnn_gnn_lstm_transformer":
            return {
                **shared_scan_graph_cfg,
                "lstm_hidden": int(run_manifest.get("lstm_hidden", 128)),
                "lstm_layers": int(run_manifest.get("lstm_layers", 1)),
                "transformer_heads": int(run_manifest.get("transformer_heads", 4)),
                "transformer_ff": int(run_manifest.get("transformer_ff", 256)),
                "transformer_layers": int(run_manifest.get("transformer_layers", 2)),
            }

        return {
            "node_dim": 14,
            "edge_dim": 8,
            "hidden_dim": int(run_manifest.get("graph_hidden", 96)),
            "lstm_hidden": int(run_manifest.get("lstm_hidden", 128)),
            "lstm_layers": int(run_manifest.get("lstm_layers", 1)),
            "future_len": int(run_manifest.get("future_len", 5)),
            "past_len": 10,
            "msg_passes": int(run_manifest.get("msg_passes", 2)),
            "dropout": float(run_manifest.get("dropout", 0.1)),
            "scan_beams": 512,
            "range_clip": 30.0,
        }

    @staticmethod
    def _extract_model_slug(ckpt: dict[str, Any]) -> str:
        run_manifest = ckpt.get("run_manifest") or {}
        model_slug = run_manifest.get("model_slug") or ckpt.get("model_slug")
        if not model_slug:
            return "gnn_lstm"
        return str(model_slug)

    def _resample_scan(self, msg: LaserScan) -> np.ndarray:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        intensities = np.asarray(msg.intensities, dtype=np.float32)
        if intensities.size == 0:
            intensities = np.zeros_like(ranges, dtype=np.float32)
        elif intensities.shape[0] != ranges.shape[0]:
            src_x = np.linspace(0.0, 1.0, intensities.shape[0], dtype=np.float32)
            dst_x = np.linspace(0.0, 1.0, ranges.shape[0], dtype=np.float32)
            intensities = np.interp(dst_x, src_x, intensities).astype(np.float32)

        ranges = np.nan_to_num(ranges, nan=self.range_clip, posinf=self.range_clip, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.range_clip)
        intensities = np.nan_to_num(intensities, nan=0.0, posinf=255.0, neginf=0.0)
        intensities = np.clip(intensities, 0.0, 255.0)

        if ranges.shape[0] != self.scan_beams:
            src_x = np.linspace(0.0, 1.0, ranges.shape[0], dtype=np.float32)
            dst_x = np.linspace(0.0, 1.0, self.scan_beams, dtype=np.float32)
            ranges = np.interp(dst_x, src_x, ranges).astype(np.float32)
            intensities = np.interp(dst_x, src_x, intensities).astype(np.float32)

        return np.stack(
            [ranges / max(self.range_clip, 1e-6), intensities / 255.0],
            axis=0,
        ).astype(np.float32)

    def world_pose_cb(self, msg: TFMessage):
        selected: dict[str, Any] = {}
        for node_name in NODE_ORDER:
            if node_name not in self.active_nodes and node_name != self.ego_node:
                continue
            frame_id = self.model_frame_ids.get(node_name)
            fallback = None
            for transform in msg.transforms:
                child = transform.child_frame_id or ""
                child_parts = [part for part in child.split("/") if part]
                if frame_id is not None and (
                    child == frame_id
                    or child.endswith(f"/{frame_id}")
                    or frame_id in child_parts
                ):
                    selected[node_name] = transform
                    break
                if node_name in child_parts and ("base_link" in child_parts or child.endswith("/base_link")):
                    fallback = transform
                elif node_name == self.ego_node and (child == "base_link" or child.endswith("/base_link")):
                    fallback = transform
            if node_name not in selected and fallback is not None:
                selected[node_name] = fallback

        for matched_name, transform in selected.items():
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            self.world_positions[matched_name] = (
                float(translation.x),
                float(translation.y),
                float(translation.z),
            )
            self.world_yaws[matched_name] = quaternion_to_yaw(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )
            self.world_pose_stamps[matched_name] = time.monotonic()
            state = self.states.get(matched_name)
            if state is not None:
                odom_yaw = self._state_yaw(state)
                world_yaw = self.world_yaws[matched_name]
                self.world_pose_anchor[matched_name] = {
                    "world_x": self.world_positions[matched_name][0],
                    "world_y": self.world_positions[matched_name][1],
                    "world_z": self.world_positions[matched_name][2],
                    "world_yaw": world_yaw if world_yaw is not None else odom_yaw,
                    "odom_x": float(state["x"]),
                    "odom_y": float(state["y"]),
                    "odom_z": float(state["z"]),
                    "odom_yaw": odom_yaw,
                }

    def obstacle_action_cb(self, msg: String):
        self.obstacle_action = msg.data.strip().lower() if msg.data else "clear"

    def obstacle_clearance_cb(self, msg: Vector3):
        self.obstacle_clearance = (float(msg.x), float(msg.y), float(msg.z))

    def scan_cb(self, msg: LaserScan):
        self.scan_history.append(self._resample_scan(msg))

    def _effective_state(self, node_name: str) -> dict | None:
        state = self.states[node_name]
        if state is None:
            if node_name not in self.active_nodes:
                return self._synthetic_state(node_name)
            return None
        world_pos = self.world_positions.get(node_name)
        if world_pos is None:
            return state
        merged = dict(state)
        merged["x"] = world_pos[0]
        merged["y"] = world_pos[1]
        merged["z"] = world_pos[2]
        return merged

    @staticmethod
    def _state_yaw(state: dict) -> float:
        return quaternion_to_yaw(
            float(state["qx"]),
            float(state["qy"]),
            float(state["qz"]),
            float(state["qw"]),
        )

    def _estimated_world_state(self, node_name: str, state: dict) -> dict | None:
        anchor = self.world_pose_anchor.get(node_name)
        if anchor is None:
            return None

        odom_x = float(state["x"])
        odom_y = float(state["y"])
        odom_z = float(state["z"])
        odom_yaw = self._state_yaw(state)

        dx = odom_x - anchor["odom_x"]
        dy = odom_y - anchor["odom_y"]
        yaw_offset = anchor["world_yaw"] - anchor["odom_yaw"]
        cos_off = math.cos(yaw_offset)
        sin_off = math.sin(yaw_offset)

        world_dx = cos_off * dx - sin_off * dy
        world_dy = sin_off * dx + cos_off * dy

        merged = dict(state)
        merged["x"] = anchor["world_x"] + world_dx
        merged["y"] = anchor["world_y"] + world_dy
        merged["z"] = anchor["world_z"] + (odom_z - anchor["odom_z"])
        return merged

    def _synthetic_state(self, node_name: str) -> dict | None:
        """Provide stable placeholder context for agents not spawned in debug runs."""

        # Keep synthetic agents in the same frame as the ego state used by the
        # model. In debug isolate mode the ego node usually has a fresh world
        # pose, so building placeholders from raw odom coordinates makes the
        # fake agents appear hundreds of meters away.
        ego_state = self._effective_state(self.ego_node)
        if ego_state is None:
            return None

        if node_name == "husky_2":
            offset_x, offset_y, offset_z = 8.0, 6.0, 0.0
        elif node_name == "uav1":
            offset_x, offset_y, offset_z = 0.0, 0.0, 12.0
        else:
            offset_x, offset_y, offset_z = 0.0, 0.0, 0.0

        return {
            "x": float(ego_state["x"]) + offset_x,
            "y": float(ego_state["y"]) + offset_y,
            "z": float(ego_state["z"]) + offset_z,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wz": 0.0,
        }

    def _ego_xy(self) -> tuple[float, float] | None:
        state = self._effective_state(self.ego_node)
        if state is None:
            return None
        return (float(state["x"]), float(state["y"]))

    def _ego_z(self) -> float | None:
        state = self._effective_state(self.ego_node)
        if state is None:
            return None
        return float(state["z"])

    def _ego_yaw(self) -> float:
        world_yaw = self.world_yaws.get(self.ego_node)
        if world_yaw is not None:
            return world_yaw
        return self.current_yaw

    def _use_fresh_world_pose(self, node_name: str) -> bool:
        stamp = self.world_pose_stamps.get(node_name, 0.0)
        if stamp <= 0.0:
            return False
        return (time.monotonic() - stamp) <= self.world_pose_timeout

    def _pose_source_label(self, node_name: str) -> str:
        if self.world_positions.get(node_name) is not None:
            return "world"
        return "odom"

    def _front_clearance(self) -> float:
        return float(self.obstacle_clearance[0])

    def _left_clearance(self) -> float:
        return float(self.obstacle_clearance[1])

    def _right_clearance(self) -> float:
        return float(self.obstacle_clearance[2])

    def _obstacle_active(self) -> bool:
        return (self.obstacle_action or "clear") != "clear"

    def _obstacle_turn_direction(self) -> float:
        if self.obstacle_action.endswith("left"):
            return 1.0
        if self.obstacle_action.endswith("right"):
            return -1.0
        return 1.0 if self._left_clearance() >= self._right_clearance() else -1.0

    def _choose_avoid_direction(self) -> str:
        if self.obstacle_action.endswith("left"):
            return "left"
        if self.obstacle_action.endswith("right"):
            return "right"
        if self._left_clearance() >= self._right_clearance():
            return "left"
        return "right"

    def _goal_speed(self, remaining: float, heading_error: float) -> float:
        linear = clamp(
            self.cmd_linear_gain * remaining,
            self.min_linear_speed,
            self.max_linear_speed,
        )
        abs_error = abs(heading_error)
        if abs_error > 1.2:
            linear *= 0.20
        elif abs_error > 0.9:
            linear *= 0.35
        elif abs_error > 0.6:
            linear *= 0.55
        elif abs_error > 0.35:
            linear *= 0.80

        if remaining < 2.0:
            linear = min(linear, 0.35)
        if remaining < 1.0:
            linear = min(linear, 0.20)
        return max(0.0, linear)

    def _tracking_speed(self, target_distance: float, heading_error: float, remaining: float | None) -> float:
        linear = clamp(
            self.cmd_linear_gain * target_distance,
            self.min_linear_speed,
            self.max_linear_speed,
        )
        abs_error = abs(heading_error)
        if abs_error > 1.2:
            linear *= 0.20
        elif abs_error > 0.9:
            linear *= 0.35
        elif abs_error > 0.6:
            linear *= 0.55
        elif abs_error > 0.35:
            linear *= 0.80

        if remaining is not None:
            if remaining < 2.0:
                linear = min(linear, 0.35)
            if remaining < 1.0:
                linear = min(linear, 0.20)
        return max(0.0, linear)

    def _should_strict_reverse(self, remaining: float | None) -> bool:
        if remaining is None:
            self.strict_blocked_cycles = 0
            return False
        if self._stuck_recovery_active(time.monotonic()) or time.monotonic() < self.stuck_cooldown_until:
            self.strict_blocked_cycles = 0
            return False

        front = self._front_clearance()
        hard_blocked = self._obstacle_active() and front <= self.strict_reverse_distance
        if not hard_blocked:
            self.strict_blocked_cycles = 0
            return False

        self.strict_blocked_cycles += 1
        if self.strict_blocked_cycles < self.strict_reverse_cycles:
            return False
        self.strict_blocked_cycles = 0
        return True

    def _obstacle_override_command(self, now: float, remaining: float | None) -> tuple[float, float] | None:
        if not self._obstacle_active() and self.avoid_direction is None:
            self.clear_path_since = None
            return None
        front = self._front_clearance()
        if (not self._obstacle_active()) or front > self.obstacle_clear_distance:
            if self.clear_path_since is None:
                self.clear_path_since = now
                return (self.post_avoid_forward_speed, 0.0)
            if (now - self.clear_path_since) < self.post_avoid_forward_seconds:
                return (self.post_avoid_forward_speed, 0.0)
            self.avoid_direction = None
            self.avoid_start_heading = None
            self.clear_path_since = None
            return None

        current_heading = self._ego_yaw()
        if self.avoid_direction is None:
            self.avoid_direction = self._choose_avoid_direction()
        if self.avoid_start_heading is None:
            self.avoid_start_heading = current_heading

        turned = abs(wrap_angle(current_heading - self.avoid_start_heading))
        sign = 1.0 if self.avoid_direction == "left" else -1.0
        close_to_obstacle = front <= (self.obstacle_stop_distance + 0.8)
        angular_speed = (
            self.obstacle_turn_speed_close
            if close_to_obstacle
            else self.obstacle_turn_speed
        )
        angular_z = sign * angular_speed
        if turned >= math.radians(110.0):
            if not self._obstacle_active():
                self.avoid_direction = None
                self.avoid_start_heading = None
                self.clear_path_since = None
                return None
            if remaining is not None:
                self._begin_stuck_recovery(now, remaining)
            return (self.stuck_reverse_speed, 0.0)

        self.clear_path_since = None
        if front <= self.obstacle_stop_distance:
            linear_x = 0.0
            state = "avoid_turn"
        elif front <= (self.obstacle_stop_distance + 0.8):
            linear_x = 0.10
            state = "avoid_caution"
        else:
            linear_x = 0.22
            state = "avoid_escape"

        if (now - self.last_obstacle_log) >= 1.5:
            direction = self.avoid_direction
            self.get_logger().info(
                f"Avoiding obstacle: direction={direction} front={front:.2f} state={state}"
            )
            self.last_obstacle_log = now
        return (linear_x, angular_z)

    def _log_decision(
        self,
        now: float,
        *,
        stage: str,
        target_xy: tuple[float, float] | None = None,
        goal_heading_error: float | None = None,
        distance: float | None = None,
        linear_x: float | None = None,
        angular_z: float | None = None,
    ):
        if (now - self.last_decision_log) < self.debug_decision_interval:
            return
        ego_xy = self._ego_xy()
        if ego_xy is None:
            return
        target_text = "none"
        if target_xy is not None:
            target_text = f"({target_xy[0]:.3f}, {target_xy[1]:.3f})"
        heading_text = "none" if goal_heading_error is None else f"{goal_heading_error:.3f}"
        distance_text = "none" if distance is None else f"{distance:.3f}"
        linear_text = "none" if linear_x is None else f"{linear_x:.3f}"
        angular_text = "none" if angular_z is None else f"{angular_z:.3f}"
        front = self._front_clearance()
        self.get_logger().info(
            "AI decision: "
            f"stage={stage} "
            f"ego=({ego_xy[0]:.3f}, {ego_xy[1]:.3f}) "
            f"target={target_text} "
            f"target_distance={distance_text} "
            f"heading_error={heading_text} "
            f"cmd=({linear_text}, {angular_text}) "
            f"front_clearance={front:.3f} "
            f"scan_frames={len(self.scan_history)} "
            f"graph_frames={len(self.graph_history)} "
            f"pose_source={self._pose_source_label(self.ego_node)} "
            f"{self._motion_response_summary()} "
            f"{self._pose_debug_summary(self.ego_node)} "
            f"{self._predicted_path_summary()}"
        )
        self.last_decision_log = now

    def _motion_response_summary(self) -> str:
        ego_state = self._effective_state(self.ego_node)
        if ego_state is None:
            return "motion=unknown"
        measured_linear = abs(float(ego_state["vx"]))
        measured_angular = abs(float(ego_state["wz"]))
        commanded_linear = abs(self.last_command_linear_x)
        commanded_angular = abs(self.last_command_angular_z)
        return (
            f"odom_v=({ego_state['vx']:.3f},{ego_state['wz']:.3f}) "
            f"cmd_v=({self.last_command_linear_x:.3f},{self.last_command_angular_z:.3f}) "
            f"response={'no_linear' if commanded_linear > 0.2 and measured_linear < 0.02 else 'no_angular' if commanded_angular > 0.2 and measured_angular < 0.05 else 'ok'}"
        )

    def _pose_debug_summary(self, node_name: str) -> str:
        raw_state = self.states.get(node_name)
        anchor = self.world_pose_anchor.get(node_name)
        if raw_state is None:
            return "raw_odom=none anchor=none odom_delta=none"

        raw_yaw = self._state_yaw(raw_state)
        raw_summary = f"raw_odom=({float(raw_state['x']):.3f},{float(raw_state['y']):.3f},{raw_yaw:.3f}) "
        if anchor is None:
            return raw_summary + "anchor=none odom_delta=none"

        dx = float(raw_state["x"]) - anchor["odom_x"]
        dy = float(raw_state["y"]) - anchor["odom_y"]
        dyaw = wrap_angle(raw_yaw - anchor["odom_yaw"])
        return (
            raw_summary
            + f"anchor_odom=({anchor['odom_x']:.3f},{anchor['odom_y']:.3f},{anchor['odom_yaw']:.3f}) "
            + f"anchor_world=({anchor['world_x']:.3f},{anchor['world_y']:.3f},{anchor['world_yaw']:.3f}) "
            + f"odom_delta=({dx:.3f},{dy:.3f},{dyaw:.3f})"
        )

    def _predicted_path_summary(self) -> str:
        if self.predicted_path is None:
            return "pred_path=none"
        points = []
        for point in np.asarray(self.predicted_path):
            points.append(f"({float(point[0]):.2f},{float(point[1]):.2f})")
        return "pred_path=[" + ",".join(points) + "]"

    def _ai_state_label(self, now: float, remaining: float | None) -> str:
        if self.arrived:
            return "arrived"
        if self._stuck_recovery_active(now):
            return "reverse"
        if now < self.stuck_bootstrap_until:
            return "recover"
        if self._obstacle_active():
            if self.avoid_direction == "left":
                return "avoid_left"
            if self.avoid_direction == "right":
                return "avoid_right"
            return "avoid"
        if (now - self.start_time) < self.bootstrap_seconds or len(self.graph_history) < self.past_len:
            return "bootstrap"
        goal_heading = self._goal_heading()
        if goal_heading is not None:
            goal_heading_error = wrap_angle(goal_heading - self._ego_yaw())
            if abs(goal_heading_error) > self.goal_align_heading_threshold:
                return "align_goal"
        if remaining is not None and remaining <= self.goal_tolerance:
            return "arrived"
        return "follow_model"

    def _log_tracking_status(self, now: float, remaining: float | None):
        if (now - self.last_diag_log) < 2.0:
            return
        ego_xy = self._ego_xy()
        ego_z = self._ego_z()
        goal = self._current_goal()
        if ego_xy is None or ego_z is None or goal is None or remaining is None:
            return
        self.get_logger().info(
            "Tracking status: "
            f"pose=({ego_xy[0]:.3f}, {ego_xy[1]:.3f}) "
            f"z={ego_z:.3f} "
            f"goal=({goal[0]:.3f}, {goal[1]:.3f}) "
            f"remaining={remaining:.3f} "
            f"state={self._ai_state_label(now, remaining)} "
            f"pose_source={self._pose_source_label(self.ego_node)} "
            f"{self._motion_response_summary()} "
            f"{self._pose_debug_summary(self.ego_node)}"
        )
        self.last_diag_log = now

    def _make_odom_cb(self, node_name: str):
        def cb(msg: Odometry):
            pose = msg.pose.pose
            twist = msg.twist.twist
            self.states[node_name] = {
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
            if node_name == self.ego_node:
                self.current_pose = pose
                self.current_yaw = quaternion_to_yaw(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
                effective = self._effective_state(node_name)
                if effective is not None:
                    px, py, pz = float(effective["x"]), float(effective["y"]), float(effective["z"])
                else:
                    px, py, pz = float(pose.position.x), float(pose.position.y), float(pose.position.z)
                self.path_history.append((time.monotonic(), px, py, pz))

        return cb

    def _make_cmd_cb(self, node_name: str):
        def cb(msg: Twist):
            self.commands[node_name] = {
                "linear_x": float(msg.linear.x),
                "angular_z": float(msg.angular.z),
            }

        return cb

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)
        self.last_command_linear_x = float(linear_x)
        self.last_command_angular_z = float(angular_z)
        self.commands[self.ego_node] = {
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
        }

    def _stuck_recovery_active(self, now: float) -> bool:
        return now < self.stuck_recovery_until

    def _should_trigger_stuck_recovery(self, now: float, remaining: float | None) -> bool:
        if self.stuck_timeout_seconds <= 0.0 or self.stuck_progress_distance <= 0.0:
            return False
        if self._stuck_recovery_active(now) or now < self.stuck_cooldown_until:
            return False
        if self.last_command_linear_x < self.stuck_min_command_speed:
            return False
        if remaining is not None and remaining <= max(
            self.goal_tolerance * 1.5,
            self.stuck_progress_distance,
        ):
            return False
        if len(self.path_history) < 2:
            return False

        reference = None
        for sample in self.path_history:
            if (now - sample[0]) >= self.stuck_timeout_seconds:
                reference = sample
                break
        if reference is None:
            return False

        moved = math.hypot(
            float(self.current_pose.position.x) - reference[1],
            float(self.current_pose.position.y) - reference[2],
        )
        return moved < self.stuck_progress_distance

    def _begin_stuck_recovery(self, now: float, remaining: float | None):
        reference = None
        moved = 0.0
        for sample in self.path_history:
            if (now - sample[0]) >= self.stuck_timeout_seconds:
                reference = sample
                break
        if reference is not None:
            moved = math.hypot(
                float(self.current_pose.position.x) - reference[1],
                float(self.current_pose.position.y) - reference[2],
            )

        self.stuck_recovery_until = now + self.stuck_reverse_seconds
        self.stuck_bootstrap_until = self.stuck_recovery_until + self.stuck_bootstrap_seconds
        self.stuck_cooldown_until = self.stuck_bootstrap_until + self.stuck_cooldown_seconds
        self.path_history.clear()
        self.path_history.append(
            (
                now,
                float(self.current_pose.position.x),
                float(self.current_pose.position.y),
                float(self.current_pose.position.z),
            )
        )
        remaining_text = "unknown" if remaining is None else f"{remaining:.3f}"
        self.get_logger().warn(
            "Stuck detected, backing up before retrying: "
            f"pose=({self.current_pose.position.x:.3f}, {self.current_pose.position.y:.3f}) "
            f"moved={moved:.3f}m over {self.stuck_timeout_seconds:.1f}s "
            f"remaining={remaining_text}"
        )

    def bootstrap_drive(self) -> tuple[float, float]:
        """Collect enough motion history before the model is trusted."""

        goal_heading = self._goal_heading()
        current_heading = self._ego_yaw()
        if goal_heading is None:
            return (self.bootstrap_linear_speed, self.bootstrap_angular_speed)

        heading_error = wrap_angle(goal_heading - current_heading)
        angular_z = clamp(
            self.bootstrap_turn_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        linear_x = min(self.bootstrap_linear_speed, self.goal_align_linear_speed)
        if abs(heading_error) > 1.0:
            linear_x *= 0.35
        elif abs(heading_error) > 0.6:
            linear_x *= 0.60
        return (linear_x, angular_z)

    def _snapshot_ready(self) -> bool:
        return self.states[self.ego_node] is not None

    def _node_feature(self, node_name: str, ego_state: dict) -> list[float]:
        state = self._effective_state(node_name)
        goal = self.goals[node_name]
        command = self.commands.get(node_name, {"linear_x": 0.0, "angular_z": 0.0})
        platform = [1.0, 0.0] if PLATFORM_TYPES[node_name] == "UGV" else [0.0, 1.0]
        return [
            state["x"] - ego_state["x"],
            state["y"] - ego_state["y"],
            state["z"] - ego_state["z"],
            state["vx"],
            state["vy"],
            state["vz"],
            state["wz"],
            goal[0] - state["x"],
            goal[1] - state["y"],
            goal[2] - state["z"],
            command["linear_x"],
            command["angular_z"],
            platform[0],
            platform[1],
        ]

    def _edge_feature(self, src: str, dst: str) -> list[float]:
        src_state = self._effective_state(src)
        dst_state = self._effective_state(dst)
        dx = dst_state["x"] - src_state["x"]
        dy = dst_state["y"] - src_state["y"]
        dz = dst_state["z"] - src_state["z"]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        inv_distance = 0.0 if distance < 1e-6 else 1.0 / distance
        bearing = math.atan2(dy, dx)
        same_platform = 1.0 if PLATFORM_TYPES[src] == PLATFORM_TYPES[dst] else 0.0
        return [
            dx,
            dy,
            dz,
            distance,
            inv_distance,
            math.sin(bearing),
            math.cos(bearing),
            same_platform,
        ]

    def _append_snapshot(self):
        """Capture one graph frame from the latest live multi-agent state."""

        now = time.monotonic()
        if now - self.last_snapshot_time < self.control_period * 0.8:
            return
        ego_state = self._effective_state(self.ego_node)
        node_feats = []
        edge_feats = []
        for src in NODE_ORDER:
            node_feats.append(self._node_feature(src, ego_state))
        for src in NODE_ORDER:
            src_edges = []
            for dst in NODE_ORDER:
                if src == dst:
                    src_edges.append([0.0] * 8)
                else:
                    src_edges.append(self._edge_feature(src, dst))
            edge_feats.append(src_edges)
        self.graph_history.append(
            {
                "node_feats": node_feats,
                "edge_feats": edge_feats,
                "origin": [ego_state["x"], ego_state["y"]],
            }
        )
        self.last_snapshot_time = now

    def predict_path(self):
        """Run the model on recent graph history to predict future waypoints."""

        node_seq = torch.tensor(
            [frame["node_feats"] for frame in self.graph_history], dtype=torch.float32
        ).unsqueeze(0)
        edge_seq = torch.tensor(
            [frame["edge_feats"] for frame in self.graph_history], dtype=torch.float32
        ).unsqueeze(0)
        origin = torch.tensor(self.graph_history[-1]["origin"], dtype=torch.float32)
        with torch.no_grad():
            if self.uses_scan:
                scan_seq = torch.tensor(np.asarray(self.scan_history, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
                model_out = self.model(scan_seq, node_seq, edge_seq)
            else:
                model_out = self.model(node_seq, edge_seq)
        pred_rel = model_out[-1] if isinstance(model_out, tuple) else model_out
        pred_rel = pred_rel.squeeze(0)
        pred_abs = pred_rel + origin.view(1, 2)
        self.predicted_path = pred_abs.numpy()
        return self.predicted_path

    def _current_goal(self):
        ego_xy = self._ego_xy()
        if ego_xy is None:
            return None
        goal_xyz = self.goals[self.ego_node]
        return (float(goal_xyz[0]) + self.target_bias_x, float(goal_xyz[1]) + self.target_bias_y)

    def _distance_to_goal(self):
        goal = self._current_goal()
        ego_xy = self._ego_xy()
        if goal is None or ego_xy is None:
            return None
        dx = goal[0] - ego_xy[0]
        dy = goal[1] - ego_xy[1]
        return math.hypot(dx, dy)

    def _goal_heading(self):
        goal = self._current_goal()
        ego_xy = self._ego_xy()
        if goal is None or ego_xy is None:
            return None
        return math.atan2(
            goal[1] - ego_xy[1],
            goal[0] - ego_xy[0],
        )

    def step(self):
        """Main control loop: stop at goal, align to goal, then follow predictions."""

        ego_xy = self._ego_xy()
        if not self._snapshot_ready() or ego_xy is None:
            return

        if self.arrived:
            self.publish_cmd(0.0, 0.0)
            return

        self._append_snapshot()
        remaining = self._distance_to_goal()
        self._log_tracking_status(time.monotonic(), remaining)
        if remaining is not None and remaining <= self.goal_tolerance:
            self.arrived = True
            self.get_logger().info(f"Arrived at goal: remaining={remaining:.3f}")
            self.publish_cmd(0.0, 0.0)
            return

        now = time.monotonic()
        if self._stuck_recovery_active(now):
            self._log_decision(
                now,
                stage="stuck_recovery",
                linear_x=self.stuck_reverse_speed,
                angular_z=0.0,
            )
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        if self._should_strict_reverse(remaining):
            self._begin_stuck_recovery(now, remaining)
            self._log_decision(
                now,
                stage="strict_reverse",
                linear_x=self.stuck_reverse_speed,
                angular_z=0.0,
            )
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        obstacle_override = self._obstacle_override_command(now, remaining)
        if obstacle_override is not None:
            self._log_decision(
                now,
                stage="obstacle_override",
                linear_x=obstacle_override[0],
                angular_z=obstacle_override[1],
            )
            self.publish_cmd(*obstacle_override)
            return

        if self._should_trigger_stuck_recovery(now, remaining):
            self._begin_stuck_recovery(now, remaining)
            self._log_decision(
                now,
                stage="begin_recovery",
                linear_x=self.stuck_reverse_speed,
                angular_z=0.0,
            )
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        if (
            (now - self.start_time) < self.bootstrap_seconds
            or now < self.stuck_bootstrap_until
            or len(self.graph_history) < self.past_len
            or (self.uses_scan and len(self.scan_history) < self.past_len)
        ):
            bootstrap_cmd = self.bootstrap_drive()
            self._log_decision(
                now,
                stage="bootstrap",
                linear_x=bootstrap_cmd[0],
                angular_z=bootstrap_cmd[1],
            )
            self.publish_cmd(*bootstrap_cmd)
            return

        goal = self._current_goal()
        goal_heading = self._goal_heading()
        if goal_heading is not None:
            goal_heading_error = wrap_angle(goal_heading - self._ego_yaw())
            if abs(goal_heading_error) > self.goal_align_heading_threshold:
                if abs(goal_heading_error) < self.heading_deadband:
                    goal_heading_error = 0.0
                angular_z = clamp(
                    self.cmd_angular_gain * goal_heading_error,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )
                linear_x = min(
                    self.goal_align_linear_speed,
                    self._goal_speed(remaining if remaining is not None else float("inf"), goal_heading_error),
                )
                self._log_decision(
                    now,
                    stage="align_goal",
                    goal_heading_error=goal_heading_error,
                    linear_x=linear_x,
                    angular_z=angular_z,
                )
                self.publish_cmd(linear_x, angular_z)
                return

        pred_abs = self.predict_path()
        target_idx = min(self.target_index, len(pred_abs) - 1)
        target_x, target_y = pred_abs[target_idx]
        target_x += self.target_bias_x
        target_y += self.target_bias_y
        if goal is not None:
            target_x = (1.0 - self.goal_blend) * target_x + self.goal_blend * goal[0]
            target_y = (1.0 - self.goal_blend) * target_y + self.goal_blend * goal[1]

        dx = target_x - ego_xy[0]
        dy = target_y - ego_xy[1]
        distance = math.hypot(dx, dy)

        if distance < self.waypoint_reached_dist:
            target_x, target_y = pred_abs[-1]
            target_x += self.target_bias_x
            target_y += self.target_bias_y
            if goal is not None:
                target_x = (1.0 - self.goal_blend) * target_x + self.goal_blend * goal[0]
                target_y = (1.0 - self.goal_blend) * target_y + self.goal_blend * goal[1]
            dx = target_x - ego_xy[0]
            dy = target_y - ego_xy[1]
            distance = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(target_heading - self._ego_yaw())
        if abs(heading_error) < self.heading_deadband:
            heading_error = 0.0
        linear_x = self._tracking_speed(distance, heading_error, remaining)

        angular_z = clamp(
            self.cmd_angular_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        self._log_decision(
            now,
            stage="follow_prediction",
            target_xy=(float(target_x), float(target_y)),
            goal_heading_error=heading_error,
            distance=distance,
            linear_x=linear_x,
            angular_z=angular_z,
        )
        self.publish_cmd(linear_x, angular_z)
