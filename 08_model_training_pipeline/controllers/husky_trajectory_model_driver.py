"""Live trajectory-model controller for testing 08 model weights in simulation."""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
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
        goal_xyz: tuple[float, float, float],
        past_len: int = 10,
        future_len: int = 5,
        target_index: int = 4,
        control_period: float = 0.1,
        goal_tolerance: float = 1.5,
        max_linear_speed: float = 1.8,
        max_angular_speed: float = 1.1,
        cmd_linear_gain: float = 1.2,
        cmd_angular_gain: float = 1.8,
        heading_slowdown_threshold: float = 0.45,
        emergency_stop_distance: float = 1.1,
        caution_distance: float = 1.8,
        goal_blend: float = 0.25,
        progress_window: int = 12,
        min_progress_delta: float = 0.20,
        progress_fallback_seconds: float = 2.0,
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
        self.emergency_stop_distance = float(emergency_stop_distance)
        self.caution_distance = float(caution_distance)
        self.goal_blend = float(goal_blend)
        self.progress_window = max(4, int(progress_window))
        self.min_progress_delta = float(min_progress_delta)
        self.progress_fallback_seconds = float(progress_fallback_seconds)

        self.publisher = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, husky_odom_topic, self.husky_odom_cb, 10)
        self.create_subscription(Odometry, uav1_odom_topic, self.uav1_odom_cb, 10)
        self.create_subscription(Odometry, uav2_odom_topic, self.uav2_odom_cb, 10)
        self.create_subscription(TFMessage, world_pose_topic, self.world_pose_cb, 10)
        self.create_subscription(Vector3, obstacle_clearance_topic, self.clearance_cb, 10)
        self.timer = self.create_timer(control_period, self.step)

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

        self.model = self._load_model()
        self.get_logger().info(
            f"Loaded 08 trajectory model {self.model_slug} from {self.checkpoint_path} "
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
            raise ValueError(f"Unsupported live 08 model slug: {self.model_slug}")

        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def husky_odom_cb(self, msg: Odometry):
        self.current_odom = msg

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
        return [
            dx / self.POSITION_SCALE,
            dy / self.POSITION_SCALE,
            dz / self.ALTITUDE_SCALE,
            state["vx"] / self.VELOCITY_SCALE,
            0.0,
            0.0,
            state["wz"] / math.pi,
            distance / self.POSITION_SCALE,
            math.sin(bearing),
            math.cos(bearing),
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

    def _publish_cmd(self, linear_x: float, angular_z: float):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.publisher.publish(cmd)
        self.last_cmd = {"linear_x": float(linear_x), "angular_z": float(angular_z)}

    def _goal_local(self) -> tuple[float, float, float]:
        ego = self.husky_world_state
        dx = self.goal_xyz[0] - ego["x"]
        dy = self.goal_xyz[1] - ego["y"]
        c = math.cos(-ego["yaw"])
        s = math.sin(-ego["yaw"])
        return (c * dx - s * dy, s * dx + c * dy, math.hypot(dx, dy))

    def _safety_override(self) -> bool:
        front = self.obstacle_clearance["front"]
        if front > self.caution_distance:
            return False
        left = self.obstacle_clearance["left"]
        right = self.obstacle_clearance["right"]
        if front <= self.emergency_stop_distance:
            turn = 0.9 if left >= right else -0.9
            self._publish_cmd(0.0, turn)
            return True
        linear = 0.18
        turn = 0.55 if left >= right else -0.55
        self._publish_cmd(linear, turn)
        return True

    def _progress_stalled(self, remaining: float) -> bool:
        self.remaining_history.append(float(remaining))
        if len(self.remaining_history) < self.progress_window:
            return False
        progress_delta = self.remaining_history[0] - self.remaining_history[-1]
        return progress_delta < self.min_progress_delta

    def step(self):
        if self.husky_world_state is None:
            return

        goal_x_local, goal_y_local, remaining = self._goal_local()
        if remaining <= self.goal_tolerance:
            if not self.arrived:
                self.get_logger().info(
                    f"Arrival triggered: remaining={remaining:.3f} model={self.model_slug}"
                )
                self.arrived = True
            self._publish_cmd(0.0, 0.0)
            return
        self.arrived = False

        self._append_history()
        if self._safety_override():
            return

        now = time.monotonic()
        stalled = self._progress_stalled(remaining)
        if stalled:
            self.force_goal_until = max(self.force_goal_until, now + self.progress_fallback_seconds)

        if len(self.goal_history) < self.past_len:
            heading = math.atan2(goal_y_local, goal_x_local)
            linear = clamp(0.7 * max(goal_x_local, 0.0), 0.0, self.max_linear_speed)
            if abs(heading) > self.heading_slowdown_threshold:
                linear *= 0.4
            angular = clamp(self.cmd_angular_gain * heading, -self.max_angular_speed, self.max_angular_speed)
            self._publish_cmd(linear, angular)
            return

        goal_seq = torch.tensor(np.asarray(self.goal_history)[None, ...], dtype=torch.float32)
        scan_seq = torch.tensor(np.asarray(self.scan_history)[None, ...], dtype=torch.float32) if self.uses_scan else None
        node_seq = torch.tensor(np.asarray(self.node_history)[None, ...], dtype=torch.float32) if self.uses_graph else None
        edge_seq = torch.tensor(np.asarray(self.edge_history)[None, ...], dtype=torch.float32) if self.uses_graph else None

        with torch.no_grad():
            pred = self.model(goal_seq, scan_seq, node_seq, edge_seq)
        pred_xy = pred[0, self.target_index].cpu().numpy()
        pred_x = float(pred_xy[0])
        pred_y = float(pred_xy[1])

        sideways_ratio = abs(pred_y) / max(abs(pred_x), 0.15)
        pred_is_sideways = pred_x < 0.15 or sideways_ratio > 1.35
        force_goal_mode = now < self.force_goal_until

        blend = self.goal_blend
        if pred_is_sideways:
            blend = max(blend, 0.60)
        if force_goal_mode:
            blend = 1.0

        target_x = float((1.0 - blend) * pred_x + blend * goal_x_local)
        target_y = float((1.0 - blend) * pred_y + blend * goal_y_local)
        heading = math.atan2(target_y, max(target_x, 1e-3))
        linear = clamp(self.cmd_linear_gain * max(target_x, 0.0), 0.0, self.max_linear_speed)
        if abs(heading) > self.heading_slowdown_threshold:
            linear *= 0.45
        angular = clamp(self.cmd_angular_gain * heading, -self.max_angular_speed, self.max_angular_speed)
        self._publish_cmd(linear, angular)

        if now - self.last_diag_log >= 1.0:
            self.last_diag_log = now
            self.get_logger().info(
                "Tracking status: "
                f"pose=({self.husky_world_state['x']:.3f}, {self.husky_world_state['y']:.3f}) "
                f"goal=({self.goal_xyz[0]:.3f}, {self.goal_xyz[1]:.3f}) "
                f"remaining={remaining:.3f} "
                f"front={self.obstacle_clearance['front']:.2f} "
                f"pred_local=({pred_x:.3f}, {pred_y:.3f}) "
                f"blend={blend:.2f} "
                f"force_goal={force_goal_mode} "
                f"cmd=({linear:.3f}, {angular:.3f}) "
                f"model={self.model_slug}"
            )
