"""Live GNN-LSTM Husky controller.

This node builds a graph snapshot from all active agents, predicts a
short-horizon ego trajectory with the GNN-LSTM model, and converts that
prediction into goal-aware ``cmd_vel`` commands for the Husky.
"""

import json
import math
import time
from collections import deque
from pathlib import Path

import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from graph_predictor import GNNLSTM, NODE_ORDER


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


class GNNModelHuskyDriver(Node):
    """Use graph-based multi-agent context to predict and follow an ego path."""

    def __init__(
        self,
        node_name: str,
        ego_node: str,
        cmd_topic: str,
        odom_topics: dict[str, str],
        command_topics: dict[str, str],
        summary_path: str | Path,
        scan_topic: str | None,
        hazard_topic: str | None,
        spawn_xyz: tuple[float, float, float] | None,
        goals: dict[str, tuple[float, float, float]],
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
        waypoint_reached_dist: float = 0.2,
        cruise_speed: float = 0.9,
        goal_tolerance: float = 0.8,
        goal_blend: float = 0.35,
        obstacle_scan_distance: float = 1.6,
        obstacle_clear_distance: float = 2.1,
        turn_in_place_speed: float = 0.85,
        hazard_timeout: float = 0.8,
        hazard_turn_speed: float = 0.7,
        history_size: int = 200,
    ):
        super().__init__(node_name)
        if ego_node not in NODE_ORDER:
            raise ValueError(f"Unknown ego node: {ego_node}")

        with open(summary_path, "r") as f:
            summary = json.load(f)
        ckpt = torch.load(summary["model_path"], map_location="cpu")
        raw_cfg = ckpt["cfg"]
        model_cfg = {
            "node_dim": raw_cfg["node_dim"],
            "edge_dim": raw_cfg["edge_dim"],
            "hidden_dim": raw_cfg["hidden_dim"],
            "lstm_hidden": raw_cfg["lstm_hidden"],
            "lstm_layers": raw_cfg["lstm_layers"],
            "future_len": raw_cfg["future_len"],
            "ego_idx": NODE_ORDER.index(ego_node),
            "msg_passes": raw_cfg.get("msg_passes", 2),
            "dropout": raw_cfg.get("dropout", 0.1),
        }
        self.past_len = raw_cfg["past_len"]
        self.future_len = raw_cfg["future_len"]
        self.ego_node = ego_node
        self.model = GNNLSTM(**model_cfg)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()

        self.cmd_topic = cmd_topic
        self.odom_topics = odom_topics
        self.command_topics = command_topics
        self.scan_topic = scan_topic
        self.hazard_topic = hazard_topic
        self.spawn_xyz = spawn_xyz
        self.goals = goals
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
        self.waypoint_reached_dist = waypoint_reached_dist
        self.cruise_speed = cruise_speed
        self.goal_tolerance = goal_tolerance
        self.goal_blend = goal_blend
        self.obstacle_scan_distance = obstacle_scan_distance
        self.obstacle_clear_distance = obstacle_clear_distance
        self.turn_in_place_speed = turn_in_place_speed
        self.hazard_timeout = hazard_timeout
        self.hazard_turn_speed = hazard_turn_speed

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.states: dict[str, dict | None] = {name: None for name in NODE_ORDER}
        self.commands: dict[str, dict] = {
            "husky_local": {"linear_x": 0.0, "angular_z": 0.0},
            "husky_2": {"linear_x": 0.0, "angular_z": 0.0},
            "uav1": {"linear_x": 0.0, "angular_z": 0.0},
        }
        self.current_pose = None
        self.current_yaw = 0.0
        self.predicted_path = None
        self.arrived = False
        self.graph_history = deque(maxlen=self.past_len)
        self.path_history = deque(maxlen=history_size)
        self.progress_history = deque(maxlen=history_size)
        self.start_time = time.monotonic()
        self.last_snapshot_time = 0.0

        self.create_subscription(Odometry, odom_topics["husky_local"], self._make_odom_cb("husky_local"), 10)
        self.create_subscription(Odometry, odom_topics["husky_2"], self._make_odom_cb("husky_2"), 10)
        self.create_subscription(Odometry, odom_topics["uav1"], self._make_odom_cb("uav1"), 10)
        self.create_subscription(Twist, command_topics["husky_local"], self._make_cmd_cb("husky_local"), 10)
        self.create_subscription(Twist, command_topics["husky_2"], self._make_cmd_cb("husky_2"), 10)

        self.timer = self.create_timer(self.control_period, self.step)
        self.get_logger().info(
            f"Loaded GNN-LSTM model for {self.ego_node} on {self.cmd_topic}"
        )

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
                self.path_history.append(
                    (time.monotonic(), float(pose.position.x), float(pose.position.y), float(pose.position.z))
                )

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
        self.commands[self.ego_node] = {
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
        }

    def bootstrap_drive(self):
        """Collect enough motion history before the model is trusted, turning toward goal."""

        if self.current_pose is None or self.ego_node not in self.goals:
            # Fallback to original behavior
            self.publish_cmd(self.bootstrap_linear_speed, self.bootstrap_angular_speed)
            return

        goal = self.goals[self.ego_node]
        if goal is None:
            self.publish_cmd(self.bootstrap_linear_speed, self.bootstrap_angular_speed)
            return

        # Compute heading to goal
        dx = goal[0] - self.current_pose.position.x
        dy = goal[1] - self.current_pose.position.y
        goal_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(goal_heading - self.current_yaw)

        # Proportional control for turning
        angular_z = clamp(heading_error * self.bootstrap_turn_gain, -self.max_angular_speed, self.max_angular_speed)

        # Move forward slowly while turning
        linear_x = self.bootstrap_linear_speed * 0.5

        self.publish_cmd(linear_x, angular_z)

    def _snapshot_ready(self) -> bool:
        return all(self.states[name] is not None for name in NODE_ORDER)

    def _node_feature(self, node_name: str, ego_state: dict) -> list[float]:
        state = self.states[node_name]
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
        src_state = self.states[src]
        dst_state = self.states[dst]
        dx = dst_state["x"] - src_state["x"]
        dy = dst_state["y"] - src_state["y"]
        dz = dst_state["z"] - src_state["z"]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return [dx, dy, dz, distance]

    def _append_snapshot(self):
        """Capture one graph frame from the latest live multi-agent state."""

        now = time.monotonic()
        if now - self.last_snapshot_time < self.control_period * 0.8:
            return
        ego_state = self.states[self.ego_node]
        node_feats = []
        edge_feats = []
        for src in NODE_ORDER:
            node_feats.append(self._node_feature(src, ego_state))
        for src in NODE_ORDER:
            src_edges = []
            for dst in NODE_ORDER:
                if src == dst:
                    src_edges.append([0.0, 0.0, 0.0, 0.0])
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
        """Run the GNN-LSTM on recent graph history to predict future waypoints."""

        node_seq = torch.tensor(
            [frame["node_feats"] for frame in self.graph_history], dtype=torch.float32
        ).unsqueeze(0)
        edge_seq = torch.tensor(
            [frame["edge_feats"] for frame in self.graph_history], dtype=torch.float32
        ).unsqueeze(0)
        origin = torch.tensor(self.graph_history[-1]["origin"], dtype=torch.float32)
        with torch.no_grad():
            pred_rel = self.model(node_seq, edge_seq).squeeze(0)
        pred_abs = pred_rel + origin.view(1, 2)
        self.predicted_path = pred_abs.numpy()
        return self.predicted_path

    def _current_goal(self):
        if self.current_pose is None:
            return None
        goal_xyz = self.goals[self.ego_node]
        return (float(goal_xyz[0]) + self.target_bias_x, float(goal_xyz[1]) + self.target_bias_y)

    def _distance_to_goal(self):
        goal = self._current_goal()
        if goal is None or self.current_pose is None:
            return None
        dx = goal[0] - self.current_pose.position.x
        dy = goal[1] - self.current_pose.position.y
        return math.hypot(dx, dy)

    def _goal_heading(self):
        goal = self._current_goal()
        if goal is None or self.current_pose is None:
            return None
        return math.atan2(
            goal[1] - self.current_pose.position.y,
            goal[0] - self.current_pose.position.x,
        )

    def step(self):
        """Main control loop: stop at goal, align to goal, then follow predictions."""

        if not self._snapshot_ready() or self.current_pose is None:
            return

        if self.arrived:
            self.publish_cmd(0.0, 0.0)
            return

        self._append_snapshot()
        remaining = self._distance_to_goal()
        if remaining is not None and remaining <= self.goal_tolerance:
            self.arrived = True
            self.publish_cmd(0.0, 0.0)
            return

        if time.monotonic() - self.start_time < self.bootstrap_seconds or len(self.graph_history) < self.past_len:
            self.bootstrap_drive()
            return

        goal = self._current_goal()
        goal_heading = self._goal_heading()
        if goal_heading is not None:
            goal_heading_error = wrap_angle(goal_heading - self.current_yaw)
            if abs(goal_heading_error) > 0.5:
                angular_z = clamp(
                    self.cmd_angular_gain * goal_heading_error,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )
                self.publish_cmd(0.0, angular_z)
                return

        pred_abs = self.predict_path()
        target_idx = min(self.target_index, len(pred_abs) - 1)
        target_x, target_y = pred_abs[target_idx]
        target_x += self.target_bias_x
        target_y += self.target_bias_y
        if goal is not None:
            target_x = (1.0 - self.goal_blend) * target_x + self.goal_blend * goal[0]
            target_y = (1.0 - self.goal_blend) * target_y + self.goal_blend * goal[1]

        dx = target_x - self.current_pose.position.x
        dy = target_y - self.current_pose.position.y
        distance = math.hypot(dx, dy)

        if distance < self.waypoint_reached_dist:
            target_x, target_y = pred_abs[-1]
            target_x += self.target_bias_x
            target_y += self.target_bias_y
            if goal is not None:
                target_x = (1.0 - self.goal_blend) * target_x + self.goal_blend * goal[0]
                target_y = (1.0 - self.goal_blend) * target_y + self.goal_blend * goal[1]
            dx = target_x - self.current_pose.position.x
            dy = target_y - self.current_pose.position.y
            distance = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(target_heading - self.current_yaw)

        linear_x = clamp(self.cmd_linear_gain * distance, 0.0, self.max_linear_speed)
        if distance > self.waypoint_reached_dist and abs(heading_error) < 0.35:
            linear_x = max(linear_x, self.min_linear_speed)
        if abs(heading_error) > 0.6:
            linear_x *= 0.5
        if abs(heading_error) < self.heading_deadband:
            heading_error = 0.0

        angular_z = clamp(
            self.cmd_angular_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        self.publish_cmd(linear_x, angular_z)
