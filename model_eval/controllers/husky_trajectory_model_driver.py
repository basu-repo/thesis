"""Live trajectory-model controller for 09 evaluation of 08 weights."""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Float32, String
from tf2_msgs.msg import TFMessage

from training.notebook_workflow import (
    CNNGNNLSTMTrajectoryPredictor,
    CNNGNNLSTMTransformerTrajectoryPredictor,
    CNNGNNTransformerTrajectoryPredictor,
    CNNLSTMTrajectoryPredictor,
)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


def extract_model_transform(msg: TFMessage, model_name: str):
    selected_model = None
    selected_base_link = None
    for transform in msg.transforms:
        child = transform.child_frame_id or ""
        child_parts = [part for part in child.split("/") if part]
        if model_name not in child_parts:
            continue
        if child_parts and child_parts[-1] == "base_link":
            selected_base_link = transform
            continue
        if child == model_name or child.endswith(f"/{model_name}") or (child_parts and child_parts[-1] == model_name):
            selected_model = transform
    return selected_base_link or selected_model


class HuskyTrajectoryModelDriver(Node):
    """Follow a learned future trajectory predicted from live 08-style features."""

    GOAL_DIM = 13
    NODE_DIM = 12
    EDGE_DIM = 8
    HIDDEN_DIM = 128
    GRAPH_HIDDEN = 96
    DROPOUT = 0.10
    MSG_PASSES = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_FF = 256
    POSITION_SCALE = 250.0
    ALTITUDE_SCALE = 100.0
    VELOCITY_SCALE = 10.0
    GOAL_DISTANCE_SCALE = 250.0
    RANGE_CLIP = 30.0

    def __init__(
        self,
        *,
        node_name: str,
        model_slug: str,
        checkpoint_path: str | Path,
        cmd_topic: str,
        husky_odom_topic: str,
        uav1_odom_topic: str,
        uav2_odom_topic: str,
        world_pose_topic: str,
        obstacle_clearance_topic: str,
        state_topic: str,
        omnet_rssi_topic: str | None = None,
        omnet_snir_topic: str | None = None,
        omnet_per_topic: str | None = None,
        omnet_link_distance_topic: str | None = None,
        goal_xyz: tuple[float, float, float],
        past_len: int = 10,
        future_len: int = 5,
        target_index: int = 4,
        control_period: float = 0.1,
        goal_tolerance: float = 1.5,
        max_linear_speed: float = 0.65,
        max_angular_speed: float = 0.45,
        cmd_linear_gain: float = 0.8,
        cmd_angular_gain: float = 1.0,
        heading_slowdown_threshold: float = 0.45,
        heading_deadband: float = 0.10,
        emergency_stop_distance: float = 1.1,
        caution_distance: float = 1.8,
        goal_blend: float = 0.25,
        progress_window: int = 12,
        min_progress_delta: float = 0.20,
        progress_fallback_seconds: float = 2.0,
        angular_smoothing: float = 0.55,
        linear_smoothing: float = 0.45,
        stuck_reverse_seconds: float = 1.2,
        stuck_reverse_speed: float = -0.40,
        reverse_pause_seconds: float = 0.6,
        reverse_min_distance: float = 0.25,
        escape_turn_radians: float = 1.0,
        escape_turn_timeout_seconds: float = 1.6,
        escape_drive_seconds: float = 1.0,
        escape_drive_speed: float = 0.45,
        post_recover_commit_cooldown_seconds: float = 1.5,
        hard_avoid_cycles: int = 12,
    ):
        super().__init__(node_name)
        self.model_slug = str(model_slug)
        self.checkpoint_path = Path(checkpoint_path)
        self.cmd_topic = cmd_topic
        self.goal_xyz = goal_xyz
        self.past_len = int(past_len)
        self.future_len = int(future_len)
        self.target_index = int(max(0, min(target_index, future_len - 1)))
        self.control_period = float(control_period)
        self.goal_tolerance = float(goal_tolerance)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.cmd_linear_gain = float(cmd_linear_gain)
        self.cmd_angular_gain = float(cmd_angular_gain)
        self.heading_slowdown_threshold = float(heading_slowdown_threshold)
        self.heading_deadband = float(heading_deadband)
        self.emergency_stop_distance = float(emergency_stop_distance)
        self.caution_distance = float(caution_distance)
        self.goal_blend = float(goal_blend)
        self.progress_window = max(4, int(progress_window))
        self.min_progress_delta = float(min_progress_delta)
        self.progress_fallback_seconds = float(progress_fallback_seconds)
        self.angular_smoothing = float(max(0.0, min(0.95, angular_smoothing)))
        self.linear_smoothing = float(max(0.0, min(0.95, linear_smoothing)))
        self.stuck_reverse_seconds = float(stuck_reverse_seconds)
        self.stuck_reverse_speed = float(stuck_reverse_speed)
        self.reverse_pause_seconds = float(reverse_pause_seconds)
        self.reverse_min_distance = float(reverse_min_distance)
        self.escape_turn_radians = float(escape_turn_radians)
        self.escape_turn_timeout_seconds = float(escape_turn_timeout_seconds)
        self.escape_drive_seconds = float(escape_drive_seconds)
        self.escape_drive_speed = float(escape_drive_speed)
        self.post_recover_commit_cooldown_seconds = float(post_recover_commit_cooldown_seconds)
        self.hard_avoid_cycles = max(3, int(hard_avoid_cycles))

        self.io_group = ReentrantCallbackGroup()
        self.step_group = ReentrantCallbackGroup()
        self.publisher = self.create_publisher(Twist, cmd_topic, 10)
        self.state_pub = self.create_publisher(String, state_topic, 10)
        self.create_subscription(Odometry, husky_odom_topic, self.husky_odom_cb, 10, callback_group=self.io_group)
        self.create_subscription(Odometry, uav1_odom_topic, self.uav1_odom_cb, 10, callback_group=self.io_group)
        self.create_subscription(Odometry, uav2_odom_topic, self.uav2_odom_cb, 10, callback_group=self.io_group)
        self.create_subscription(TFMessage, world_pose_topic, self.world_pose_cb, 10, callback_group=self.io_group)
        self.create_subscription(Vector3, obstacle_clearance_topic, self.clearance_cb, 10, callback_group=self.io_group)
        if omnet_rssi_topic:
            self.create_subscription(Float32, omnet_rssi_topic, self.omnet_rssi_cb, 10, callback_group=self.io_group)
        if omnet_snir_topic:
            self.create_subscription(Float32, omnet_snir_topic, self.omnet_snir_cb, 10, callback_group=self.io_group)
        if omnet_per_topic:
            self.create_subscription(Float32, omnet_per_topic, self.omnet_per_cb, 10, callback_group=self.io_group)
        if omnet_link_distance_topic:
            self.create_subscription(Float32, omnet_link_distance_topic, self.omnet_link_distance_cb, 10, callback_group=self.io_group)
        self.timer = self.create_timer(control_period, self.step, callback_group=self.step_group)

        self.current_odom = None
        self.husky_world_state = None
        self.uav1_world_state = None
        self.uav2_world_state = None
        self.uav1_odom = None
        self.uav2_odom = None
        self.last_cmd = {"linear_x": 0.0, "angular_z": 0.0}
        self.obstacle_clearance = {"front": self.RANGE_CLIP, "left": self.RANGE_CLIP, "right": self.RANGE_CLIP}
        self.goal_history = deque(maxlen=self.past_len)
        self.scan_history = deque(maxlen=self.past_len)
        self.node_history = deque(maxlen=self.past_len)
        self.edge_history = deque(maxlen=self.past_len)
        self.last_diag_log = 0.0
        self.arrived = False
        self.remaining_history = deque(maxlen=self.progress_window)
        self.force_goal_until = 0.0
        self.hard_avoid_count = 0
        self.recovery_state = ""
        self.recovery_until = 0.0
        self.reverse_start_xy = None
        self.reverse_failed_escape = False
        self.escape_turn_direction = "left"
        self.escape_turn_start_heading = None
        self.last_pred_local = (0.0, 0.0)
        self.last_blend = self.goal_blend
        self.last_state = "boot"
        self.last_husky_odom_update = 0.0
        self.last_world_pose_update = 0.0
        self.last_clearance_update = 0.0
        self.world_pose_updates = 0
        self.clearance_updates = 0
        self.odom_updates = 0
        self.omnet_rssi_dbm = None
        self.omnet_snir_db = None
        self.omnet_packet_error_rate = None
        self.omnet_link_distance = None

        self.model = self._load_model()
        self.get_logger().info(
            f"Loaded 09 trajectory model {self.model_slug} from {self.checkpoint_path} "
            f"for cmd topic {self.cmd_topic}"
        )

    def _load_model(self):
        if self.model_slug == "cnn_lstm":
            model = CNNLSTMTrajectoryPredictor(
                goal_dim=self.GOAL_DIM,
                hidden_dim=self.HIDDEN_DIM,
                cnn_hidden=self.HIDDEN_DIM,
                future_len=self.future_len,
                dropout=self.DROPOUT,
            )
            self.uses_scan = True
            self.uses_graph = False
        elif self.model_slug == "cnn_gnn_lstm":
            model = CNNGNNLSTMTrajectoryPredictor(
                goal_dim=self.GOAL_DIM,
                node_dim=self.NODE_DIM,
                edge_dim=self.EDGE_DIM,
                hidden_dim=self.HIDDEN_DIM,
                graph_hidden=self.GRAPH_HIDDEN,
                future_len=self.future_len,
                dropout=self.DROPOUT,
                msg_passes=self.MSG_PASSES,
            )
            self.uses_scan = True
            self.uses_graph = True
        elif self.model_slug == "cnn_gnn_transformer":
            model = CNNGNNTransformerTrajectoryPredictor(
                goal_dim=self.GOAL_DIM,
                node_dim=self.NODE_DIM,
                edge_dim=self.EDGE_DIM,
                hidden_dim=self.HIDDEN_DIM,
                graph_hidden=self.GRAPH_HIDDEN,
                future_len=self.future_len,
                dropout=self.DROPOUT,
                msg_passes=self.MSG_PASSES,
                num_heads=self.TRANSFORMER_HEADS,
                num_layers=self.TRANSFORMER_LAYERS,
                ff_dim=self.TRANSFORMER_FF,
            )
            self.uses_scan = True
            self.uses_graph = True
        elif self.model_slug == "cnn_gnn_lstm_transformer":
            model = CNNGNNLSTMTransformerTrajectoryPredictor(
                goal_dim=self.GOAL_DIM,
                node_dim=self.NODE_DIM,
                edge_dim=self.EDGE_DIM,
                hidden_dim=self.HIDDEN_DIM,
                graph_hidden=self.GRAPH_HIDDEN,
                future_len=self.future_len,
                dropout=self.DROPOUT,
                msg_passes=self.MSG_PASSES,
                num_heads=self.TRANSFORMER_HEADS,
                num_layers=self.TRANSFORMER_LAYERS,
                ff_dim=self.TRANSFORMER_FF,
            )
            self.uses_scan = True
            self.uses_graph = True
        else:
            raise ValueError(f"Unsupported 09 live model slug: {self.model_slug}")

        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def husky_odom_cb(self, msg: Odometry):
        self.current_odom = msg
        self.last_husky_odom_update = time.monotonic()
        self.odom_updates += 1

    def uav1_odom_cb(self, msg: Odometry):
        self.uav1_odom = msg

    def uav2_odom_cb(self, msg: Odometry):
        self.uav2_odom = msg

    def world_pose_cb(self, msg: TFMessage):
        for name in ("husky_2", "uav1", "uav2"):
            transform = extract_model_transform(msg, name)
            if transform is None:
                continue
            t = transform.transform.translation
            r = transform.transform.rotation
            state = {
                "x": float(t.x),
                "y": float(t.y),
                "z": float(t.z),
                "yaw": float(quaternion_to_yaw(r.x, r.y, r.z, r.w)),
            }
            if name == "husky_2":
                self.husky_world_state = state
                self.last_world_pose_update = time.monotonic()
                self.world_pose_updates += 1
            elif name == "uav1":
                self.uav1_world_state = state
            else:
                self.uav2_world_state = state

    def clearance_cb(self, msg: Vector3):
        def fix(value: float) -> float:
            if value >= 900.0:
                return self.RANGE_CLIP
            return float(max(0.0, min(value, self.RANGE_CLIP)))

        self.obstacle_clearance = {
            "front": fix(float(msg.x)),
            "left": fix(float(msg.y)),
            "right": fix(float(msg.z)),
        }
        self.last_clearance_update = time.monotonic()
        self.clearance_updates += 1

    def omnet_rssi_cb(self, msg: Float32):
        self.omnet_rssi_dbm = float(msg.data)

    def omnet_snir_cb(self, msg: Float32):
        self.omnet_snir_db = float(msg.data)

    def omnet_per_cb(self, msg: Float32):
        self.omnet_packet_error_rate = float(msg.data)

    def omnet_link_distance_cb(self, msg: Float32):
        self.omnet_link_distance = float(msg.data)

    def _communication_quality_scale(self) -> float:
        if self.omnet_packet_error_rate is None and self.omnet_snir_db is None and self.omnet_rssi_dbm is None:
            return 1.0
        scale = 1.0
        if self.omnet_packet_error_rate is not None:
            per = max(0.0, min(self.omnet_packet_error_rate, 1.0))
            scale *= max(0.10, 1.0 - per)
        if self.omnet_snir_db is not None:
            if self.omnet_snir_db <= 0.0:
                scale *= 0.25
            elif self.omnet_snir_db < 5.0:
                scale *= 0.55
            elif self.omnet_snir_db < 10.0:
                scale *= 0.75
        elif self.omnet_rssi_dbm is not None:
            if self.omnet_rssi_dbm < -95.0:
                scale *= 0.35
            elif self.omnet_rssi_dbm < -85.0:
                scale *= 0.65
        return float(max(0.10, min(scale, 1.0)))

    def _uav_state(self, which: str) -> dict:
        world_state = self.uav1_world_state if which == "uav1" else self.uav2_world_state
        odom_msg = self.uav1_odom if which == "uav1" else self.uav2_odom
        if world_state is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 0.0, "wz": 0.0}
        vx = float(odom_msg.twist.twist.linear.x) if odom_msg is not None else 0.0
        wz = float(odom_msg.twist.twist.angular.z) if odom_msg is not None else 0.0
        return {
            "x": world_state["x"],
            "y": world_state["y"],
            "z": world_state["z"],
            "yaw": world_state["yaw"],
            "vx": vx,
            "wz": wz,
        }

    def _goal_feature_vector(self) -> list[float]:
        ego = self.husky_world_state
        goal_x, goal_y, _goal_z = self.goal_xyz
        dx = goal_x - ego["x"]
        dy = goal_y - ego["y"]
        c = math.cos(-ego["yaw"])
        s = math.sin(-ego["yaw"])
        rel_x = c * dx - s * dy
        rel_y = s * dx + c * dy
        goal_distance = math.hypot(dx, dy)
        goal_heading_error = wrap_angle(math.atan2(dy, dx) - ego["yaw"])
        odom_twist = self.current_odom.twist.twist if self.current_odom is not None else None
        vx = float(odom_twist.linear.x) if odom_twist else 0.0
        vy = float(odom_twist.linear.y) if odom_twist else 0.0
        vz = float(odom_twist.linear.z) if odom_twist else 0.0
        wz = float(odom_twist.angular.z) if odom_twist else 0.0
        return [
            vx / self.VELOCITY_SCALE,
            vy / self.VELOCITY_SCALE,
            vz / self.VELOCITY_SCALE,
            wz / math.pi,
            rel_x / self.GOAL_DISTANCE_SCALE,
            rel_y / self.GOAL_DISTANCE_SCALE,
            goal_distance / self.GOAL_DISTANCE_SCALE,
            goal_heading_error / math.pi,
            self.last_cmd["linear_x"] / self.VELOCITY_SCALE,
            self.last_cmd["angular_z"] / math.pi,
            self.obstacle_clearance["front"] / self.RANGE_CLIP,
            self.obstacle_clearance["left"] / self.RANGE_CLIP,
            self.obstacle_clearance["right"] / self.RANGE_CLIP,
        ]

    def _scan_feature_vector(self) -> list[float]:
        return [
            self.obstacle_clearance["front"] / self.RANGE_CLIP,
            self.obstacle_clearance["left"] / self.RANGE_CLIP,
            self.obstacle_clearance["right"] / self.RANGE_CLIP,
        ]

    def _node_feature_vector(self, role: str) -> list[float]:
        ego = self.husky_world_state
        goal_x, goal_y, _goal_z = self.goal_xyz
        if role == "ego":
            odom_twist = self.current_odom.twist.twist if self.current_odom is not None else None
            vx = float(odom_twist.linear.x) if odom_twist else 0.0
            vy = float(odom_twist.linear.y) if odom_twist else 0.0
            vz = float(odom_twist.linear.z) if odom_twist else 0.0
            wz = float(odom_twist.angular.z) if odom_twist else 0.0
            dx = goal_x - ego["x"]
            dy = goal_y - ego["y"]
            c = math.cos(-ego["yaw"])
            s = math.sin(-ego["yaw"])
            rel_x = c * dx - s * dy
            rel_y = s * dx + c * dy
            return [
                0.0,
                0.0,
                0.0,
                vx / self.VELOCITY_SCALE,
                vy / self.VELOCITY_SCALE,
                vz / self.VELOCITY_SCALE,
                wz / math.pi,
                rel_x / self.GOAL_DISTANCE_SCALE,
                rel_y / self.GOAL_DISTANCE_SCALE,
                math.hypot(dx, dy) / self.GOAL_DISTANCE_SCALE,
                wrap_angle(math.atan2(dy, dx) - ego["yaw"]) / math.pi,
                1.0,
            ]

        state = self._uav_state(role)
        dx = state["x"] - ego["x"]
        dy = state["y"] - ego["y"]
        dz = state["z"] - ego["z"]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        bearing = wrap_angle(math.atan2(dy, dx) - ego["yaw"])
        comm_scale = self._communication_quality_scale()
        return [
            comm_scale * dx / self.POSITION_SCALE,
            comm_scale * dy / self.POSITION_SCALE,
            comm_scale * dz / self.ALTITUDE_SCALE,
            comm_scale * state["vx"] / self.VELOCITY_SCALE,
            0.0,
            0.0,
            comm_scale * state["wz"] / math.pi,
            comm_scale * distance / self.POSITION_SCALE,
            comm_scale * math.sin(bearing),
            comm_scale * math.cos(bearing),
            0.0,
            1.0,
        ]

    def _edge_features(self, nodes: np.ndarray) -> np.ndarray:
        edge_rows = []
        for src_idx in range(nodes.shape[0]):
            src_xyz = nodes[src_idx, :3]
            row = []
            for dst_idx in range(nodes.shape[0]):
                dst_xyz = nodes[dst_idx, :3]
                dx, dy, dz = dst_xyz - src_xyz
                distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                inv_distance = 0.0 if distance <= 1e-6 else 1.0 / distance
                bearing = math.atan2(float(dy), float(dx)) if distance > 1e-6 else 0.0
                row.append(
                    [
                        float(dx),
                        float(dy),
                        float(dz),
                        float(distance),
                        float(inv_distance),
                        float(math.sin(bearing)),
                        float(math.cos(bearing)),
                        1.0 if src_idx == dst_idx else 0.0,
                    ]
                )
            edge_rows.append(row)
        return np.asarray(edge_rows, dtype=np.float32)

    def _append_history(self):
        goal_vec = np.asarray(self._goal_feature_vector(), dtype=np.float32)
        self.goal_history.append(goal_vec)
        self.scan_history.append(np.asarray(self._scan_feature_vector(), dtype=np.float32))
        nodes = np.asarray(
            [
                self._node_feature_vector("ego"),
                self._node_feature_vector("uav1"),
                self._node_feature_vector("uav2"),
            ],
            dtype=np.float32,
        )
        self.node_history.append(nodes)
        self.edge_history.append(self._edge_features(nodes))

    def _publish_cmd(self, linear_x: float, angular_z: float, *, smooth: bool = True):
        if smooth:
            linear_x = ((1.0 - self.linear_smoothing) * float(linear_x)) + (self.linear_smoothing * self.last_cmd["linear_x"])
            angular_z = ((1.0 - self.angular_smoothing) * float(angular_z)) + (self.angular_smoothing * self.last_cmd["angular_z"])
        else:
            linear_x = float(linear_x)
            angular_z = float(angular_z)
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.publisher.publish(cmd)
        self.last_cmd = {"linear_x": float(linear_x), "angular_z": float(angular_z)}

    def _publish_state(self):
        msg = String()
        msg.data = "reached" if self.last_state == "arrived" else self.last_state
        self.state_pub.publish(msg)

    def _goal_local(self) -> tuple[float, float, float]:
        ego = self.husky_world_state
        dx = self.goal_xyz[0] - ego["x"]
        dy = self.goal_xyz[1] - ego["y"]
        c = math.cos(-ego["yaw"])
        s = math.sin(-ego["yaw"])
        return (c * dx - s * dy, s * dx + c * dy, math.hypot(dx, dy))

    def _current_xy(self) -> tuple[float, float] | None:
        if self.husky_world_state is None:
            return None
        return (float(self.husky_world_state["x"]), float(self.husky_world_state["y"]))

    def _current_heading(self) -> float | None:
        if self.husky_world_state is None:
            return None
        return float(self.husky_world_state["yaw"])

    def _progress_stalled(self, remaining: float) -> bool:
        self.remaining_history.append(float(remaining))
        if len(self.remaining_history) < self.progress_window:
            return False
        progress_delta = self.remaining_history[0] - self.remaining_history[-1]
        return progress_delta < self.min_progress_delta

    def _goal_speed(self, distance: float, heading_error: float, remaining: float) -> float:
        linear = clamp(self.cmd_linear_gain * max(distance, 0.0), 0.0, self.max_linear_speed)
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

    def _reverse_distance_traveled(self) -> float:
        current_xy = self._current_xy()
        if self.reverse_start_xy is None or current_xy is None:
            return 0.0
        return math.hypot(current_xy[0] - self.reverse_start_xy[0], current_xy[1] - self.reverse_start_xy[1])

    def _enter_reverse(self, now: float, remaining: float, reason: str):
        self.recovery_state = "reverse"
        self.recovery_until = now + self.stuck_reverse_seconds
        self.reverse_start_xy = self._current_xy()
        self.reverse_failed_escape = False
        self.escape_turn_start_heading = None
        self.hard_avoid_count = 0
        self.last_state = "reverse"
        self.get_logger().warn(
            "Stuck recovery: "
            f"reason={reason} remaining={remaining:.3f} "
            f"front={self.obstacle_clearance['front']:.2f} "
            f"reverse_speed={self.stuck_reverse_speed:.2f} "
            f"duration={self.stuck_reverse_seconds:.2f}"
        )

    def _enter_reverse_pause(self, now: float, remaining: float, reverse_failed_escape: bool):
        self.recovery_state = "reverse_pause"
        self.recovery_until = now + self.reverse_pause_seconds
        self.reverse_failed_escape = bool(reverse_failed_escape)
        self.last_state = "reverse_pause"
        self.get_logger().info(
            "Reverse pause: "
            f"remaining={remaining:.3f} escape_next={self.reverse_failed_escape}"
        )

    def _enter_recover(self, now: float):
        self.recovery_state = "recover"
        self.recovery_until = now + self.post_recover_commit_cooldown_seconds
        self.last_state = "recover"

    def _enter_escape_turn(self, now: float):
        self.recovery_state = "escape_turn"
        self.recovery_until = now + self.escape_turn_timeout_seconds
        self.escape_turn_direction = "left" if self.obstacle_clearance["left"] >= self.obstacle_clearance["right"] else "right"
        self.escape_turn_start_heading = self._current_heading()
        self.last_state = "escape_turn"
        self.get_logger().warn(
            "Escape turn: "
            f"direction={self.escape_turn_direction} "
            f"left={self.obstacle_clearance['left']:.2f} "
            f"right={self.obstacle_clearance['right']:.2f}"
        )

    def _enter_escape_drive(self, now: float):
        self.recovery_state = "escape_drive"
        self.recovery_until = now + self.escape_drive_seconds
        self.last_state = "escape_drive"
        self.get_logger().warn(
            "Escape drive: "
            f"speed={self.escape_drive_speed:.2f} duration={self.escape_drive_seconds:.2f}"
        )

    def _run_recovery(self, now: float, remaining: float) -> bool:
        if self.recovery_state == "":
            return False
        if self.recovery_state == "reverse":
            if now < self.recovery_until:
                self._publish_cmd(self.stuck_reverse_speed, 0.0, smooth=False)
                self.last_state = "reverse"
                return True
            reverse_failed = self._reverse_distance_traveled() <= self.reverse_min_distance
            self._enter_reverse_pause(now, remaining, reverse_failed)
            self._publish_cmd(0.0, 0.0, smooth=False)
            return True
        if self.recovery_state == "reverse_pause":
            if now < self.recovery_until:
                self._publish_cmd(0.0, 0.0, smooth=False)
                self.last_state = "reverse_pause"
                return True
            if self.reverse_failed_escape:
                self._enter_escape_turn(now)
            else:
                self._enter_recover(now)
            self._publish_cmd(0.0, 0.0, smooth=False)
            return True
        if self.recovery_state == "recover":
            if now < self.recovery_until:
                self._publish_cmd(0.0, 0.0, smooth=False)
                self.last_state = "recover"
                return True
            self.recovery_state = ""
            self.force_goal_until = max(self.force_goal_until, now + self.post_recover_commit_cooldown_seconds)
            return False
        if self.recovery_state == "escape_turn":
            heading = self._current_heading()
            if heading is None:
                self._publish_cmd(0.0, 0.0, smooth=False)
                return True
            if self.escape_turn_start_heading is None:
                self.escape_turn_start_heading = heading
            turned = abs(wrap_angle(heading - self.escape_turn_start_heading))
            sign = 1.0 if self.escape_turn_direction == "left" else -1.0
            if turned >= self.escape_turn_radians or now >= self.recovery_until:
                self._enter_escape_drive(now)
                self._publish_cmd(self.escape_drive_speed, 0.0, smooth=False)
                return True
            self._publish_cmd(0.0, sign * self.max_angular_speed, smooth=False)
            self.last_state = "escape_turn"
            return True
        if self.recovery_state == "escape_drive":
            if now < self.recovery_until:
                self._publish_cmd(self.escape_drive_speed, 0.0, smooth=False)
                self.last_state = "escape_drive"
                return True
            self._enter_recover(now)
            self._publish_cmd(0.0, 0.0, smooth=False)
            return True
        self.recovery_state = ""
        return False

    def _state_label(self, *, bootstrap: bool, safety: bool, force_goal: bool, remaining: float) -> str:
        if remaining <= self.goal_tolerance:
            return "arrived"
        if self.recovery_state:
            return self.recovery_state
        if safety and self.obstacle_clearance["front"] <= self.emergency_stop_distance:
            return "avoid"
        if safety:
            return "caution"
        if bootstrap:
            return "bootstrap"
        if force_goal:
            return "force_goal"
        return "follow_model"

    def _safety_override(self, now: float) -> tuple[bool, str]:
        front = self.obstacle_clearance["front"]
        if front > self.caution_distance:
            self.hard_avoid_count = 0
            return False, "clear"
        left = self.obstacle_clearance["left"]
        right = self.obstacle_clearance["right"]
        turn_sign = 1.0 if left >= right else -1.0
        if front <= self.emergency_stop_distance:
            self.hard_avoid_count += 1
            if self.hard_avoid_count >= self.hard_avoid_cycles:
                return True, "front_blocked_hard"
            turn = 0.9 * turn_sign
            self._publish_cmd(0.06, turn)
            return True, "hard_stop_turn"
        self.hard_avoid_count = 0
        linear = 0.16 if front <= (self.emergency_stop_distance + 0.8) else 0.28
        turn = (0.9 if front <= (self.emergency_stop_distance + 0.8) else 0.55) * turn_sign
        self._publish_cmd(linear, turn)
        return True, "caution_turn"

    def step(self):
        if self.husky_world_state is None:
            return

        if self.arrived:
            self.last_state = "arrived"
            self._publish_cmd(0.0, 0.0, smooth=False)
            self._publish_state()
            return

        goal_x_local, goal_y_local, remaining = self._goal_local()
        if remaining <= self.goal_tolerance:
            if not self.arrived:
                self.get_logger().info(
                    f"Arrival triggered: remaining={remaining:.3f} model={self.model_slug}"
                )
                self.arrived = True
            self.last_state = "arrived"
            self._publish_cmd(0.0, 0.0, smooth=False)
            self._publish_state()
            return

        self._append_history()

        now = time.monotonic()
        if self._run_recovery(now, remaining):
            self._publish_state()
            return

        safety_active, safety_state = self._safety_override(now)
        if safety_active:
            if safety_state == "front_blocked_hard":
                self._enter_reverse(now, remaining, "front_blocked_hard")
                self._publish_cmd(self.stuck_reverse_speed, 0.0, smooth=False)
                self._publish_state()
                return
            if self.recovery_state:
                self.last_state = self.recovery_state
            else:
                self.last_state = "avoid" if safety_state == "hard_stop_turn" else "caution"

        stalled = self._progress_stalled(remaining)
        if stalled:
            self.force_goal_until = max(self.force_goal_until, now + self.progress_fallback_seconds)

        bootstrap = len(self.goal_history) < self.past_len
        if bootstrap:
            heading = math.atan2(goal_y_local, goal_x_local)
            if abs(heading) < self.heading_deadband:
                heading = 0.0
            linear = clamp(0.7 * max(goal_x_local, 0.0), 0.0, self.max_linear_speed)
            if abs(heading) > self.heading_slowdown_threshold:
                linear *= 0.4
            angular = clamp(self.cmd_angular_gain * heading, -self.max_angular_speed, self.max_angular_speed)
            if not safety_active:
                self._publish_cmd(linear, angular)
            self.last_pred_local = (goal_x_local, goal_y_local)
            self.last_blend = 1.0
            self.last_state = self._state_label(
                bootstrap=True,
                safety=safety_active,
                force_goal=False,
                remaining=remaining,
            )
        elif not safety_active:
            goal_seq = torch.tensor(np.asarray(self.goal_history)[None, ...], dtype=torch.float32)
            scan_seq = torch.tensor(np.asarray(self.scan_history)[None, ...], dtype=torch.float32) if self.uses_scan else None
            node_seq = torch.tensor(np.asarray(self.node_history)[None, ...], dtype=torch.float32) if self.uses_graph else None
            edge_seq = torch.tensor(np.asarray(self.edge_history)[None, ...], dtype=torch.float32) if self.uses_graph else None

            with torch.no_grad():
                pred = self.model(goal_seq, scan_seq, node_seq, edge_seq)
            pred_xy = pred[0, self.target_index].cpu().numpy()
            pred_x = float(pred_xy[0])
            pred_y = float(pred_xy[1])
            self.last_pred_local = (pred_x, pred_y)

            sideways_ratio = abs(pred_y) / max(abs(pred_x), 0.10)
            # Short-horizon models often predict a small but valid forward step.
            # Treat only backward/near-zero-forward or strongly lateral outputs as sideways.
            pred_is_sideways = pred_x <= 0.0 or sideways_ratio > 1.35
            force_goal_mode = now < self.force_goal_until

            if force_goal_mode:
                heading = math.atan2(goal_y_local, goal_x_local)
                if abs(heading) < self.heading_deadband:
                    heading = 0.0
                linear = self._goal_speed(remaining, heading, remaining)
                angular = clamp(self.cmd_angular_gain * heading, -self.max_angular_speed, self.max_angular_speed)
                self.last_blend = 1.0
            else:
                blend = self.goal_blend
                if pred_is_sideways:
                    blend = max(blend, 0.35)
                self.last_blend = blend

                target_x = float((1.0 - blend) * pred_x + blend * goal_x_local)
                target_y = float((1.0 - blend) * pred_y + blend * goal_y_local)
                lookahead_x = max(target_x, 0.35)
                heading = math.atan2(target_y, lookahead_x)
                if abs(heading) < self.heading_deadband:
                    heading = 0.0
                target_distance = math.hypot(target_x, target_y)
                linear = self._goal_speed(target_distance, heading, remaining)
                angular = clamp(self.cmd_angular_gain * heading, -self.max_angular_speed, self.max_angular_speed)
            self._publish_cmd(linear, angular)
            self.last_state = self._state_label(
                bootstrap=False,
                safety=False,
                force_goal=force_goal_mode,
                remaining=remaining,
            )

        if now - self.last_diag_log >= 1.0:
            self.last_diag_log = now
            world_age = -1.0 if self.last_world_pose_update <= 0.0 else now - self.last_world_pose_update
            clearance_age = -1.0 if self.last_clearance_update <= 0.0 else now - self.last_clearance_update
            odom_age = -1.0 if self.last_husky_odom_update <= 0.0 else now - self.last_husky_odom_update
            self.get_logger().info(
                "Tracking status: "
                f"pose=({self.husky_world_state['x']:.3f}, {self.husky_world_state['y']:.3f}) "
                f"goal=({self.goal_xyz[0]:.3f}, {self.goal_xyz[1]:.3f}) "
                f"remaining={remaining:.3f} "
                f"state={self.last_state} "
                f"front={self.obstacle_clearance['front']:.2f} "
                f"pred_local=({self.last_pred_local[0]:.3f}, {self.last_pred_local[1]:.3f}) "
                f"blend={self.last_blend:.2f} "
                f"cmd=({self.last_cmd['linear_x']:.3f}, {self.last_cmd['angular_z']:.3f}) "
                f"comm_scale={self._communication_quality_scale():.2f} "
                f"omnet=(rssi={self.omnet_rssi_dbm if self.omnet_rssi_dbm is not None else 'na'}, "
                f"snir={self.omnet_snir_db if self.omnet_snir_db is not None else 'na'}, "
                f"per={self.omnet_packet_error_rate if self.omnet_packet_error_rate is not None else 'na'}, "
                f"dist={self.omnet_link_distance if self.omnet_link_distance is not None else 'na'}) "
                f"ages=(world={world_age:.2f}s, clearance={clearance_age:.2f}s, odom={odom_age:.2f}s) "
                f"updates=(world={self.world_pose_updates}, clearance={self.clearance_updates}, odom={self.odom_updates}) "
                f"model={self.model_slug}"
            )
        self._publish_state()
