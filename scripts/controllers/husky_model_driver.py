import json
import math
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CNNLSTM(nn.Module):
    def __init__(
        self,
        past_len=10,
        future_len=20,
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.future_len = future_len
        self.cnn = nn.Sequential(
            nn.Conv1d(2, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, future_len * 2),
        )

    def forward(self, past_xy):
        x = past_xy.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        out = self.head(h[-1])
        return out.view(-1, self.future_len, 2)


class ModelHuskyDriver(Node):
    def __init__(
        self,
        node_name: str,
        cmd_topic: str,
        odom_topic: str,
        scan_topic: str | None,
        pointcloud_topic: str | None,
        hazard_topic: str | None,
        summary_path: str | Path,
        target_bias_x: float = 0.0,
        target_bias_y: float = 0.0,
        bootstrap_seconds: float = 3.0,
        bootstrap_linear_speed: float = 0.45,
        bootstrap_angular_speed: float = 0.0,
        target_index: int = 4,
        control_period: float = 0.1,
        cmd_linear_gain: float = 0.9,
        cmd_angular_gain: float = 1.6,
        min_linear_speed: float = 0.0,
        max_linear_speed: float = 0.9,
        max_angular_speed: float = 1.2,
        heading_deadband: float = 0.08,
        waypoint_reached_dist: float = 0.2,
        obstacle_stop_distance: float = 1.8,
        obstacle_turn_speed: float = 1.35,
        obstacle_turn_speed_close: float = 1.8,
        obstacle_reverse_speed: float = -0.35,
        obstacle_stop_seconds: float = 0.5,
        obstacle_recovery_seconds: float = 2.0,
        cliff_forward_min_x: float = 0.8,
        cliff_forward_max_x: float = 3.0,
        cliff_half_width_y: float = 0.8,
        cliff_ground_min_z: float = -1.2,
        cliff_ground_max_z: float = -0.03,
        cliff_min_ground_points: int = 12,
        hazard_timeout: float = 0.8,
        hazard_caution_speed: float = 0.18,
        hazard_turn_speed: float = 0.7,
    ):
        super().__init__(node_name)

        with open(summary_path, "r") as f:
            summary = json.load(f)
        ckpt = torch.load(summary["model_path"], map_location="cpu")
        cfg = ckpt["cfg"]
        self.past_len = cfg["past_len"]
        self.future_len = cfg["future_len"]
        self.model = CNNLSTM(**cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.scan_topic = scan_topic
        self.pointcloud_topic = pointcloud_topic
        self.hazard_topic = hazard_topic
        self.target_bias_x = target_bias_x
        self.target_bias_y = target_bias_y
        self.bootstrap_seconds = bootstrap_seconds
        self.bootstrap_linear_speed = bootstrap_linear_speed
        self.bootstrap_angular_speed = bootstrap_angular_speed
        self.target_index = target_index
        self.control_period = control_period
        self.cmd_linear_gain = cmd_linear_gain
        self.cmd_angular_gain = cmd_angular_gain
        self.min_linear_speed = min_linear_speed
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.heading_deadband = heading_deadband
        self.waypoint_reached_dist = waypoint_reached_dist
        self.obstacle_stop_distance = obstacle_stop_distance
        self.obstacle_turn_speed = obstacle_turn_speed
        self.obstacle_turn_speed_close = obstacle_turn_speed_close
        self.obstacle_reverse_speed = obstacle_reverse_speed
        self.obstacle_stop_seconds = obstacle_stop_seconds
        self.obstacle_recovery_seconds = obstacle_recovery_seconds
        self.cliff_forward_min_x = cliff_forward_min_x
        self.cliff_forward_max_x = cliff_forward_max_x
        self.cliff_half_width_y = cliff_half_width_y
        self.cliff_ground_min_z = cliff_ground_min_z
        self.cliff_ground_max_z = cliff_ground_max_z
        self.cliff_min_ground_points = cliff_min_ground_points
        self.hazard_timeout = hazard_timeout
        self.hazard_caution_speed = hazard_caution_speed
        self.hazard_turn_speed = hazard_turn_speed

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)

        self.positions = deque(maxlen=self.past_len)
        self.current_pose = None
        self.current_yaw = 0.0
        self.predicted_path = None
        self.start_time = time.monotonic()
        self.timer = self.create_timer(self.control_period, self.step)

        self.get_logger().info(
            f"Loaded KITTI model on {self.cmd_topic} using {self.odom_topic}"
        )

    def odom_cb(self, msg):
        pose = msg.pose.pose
        self.current_pose = pose
        self.current_yaw = quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.positions.append((pose.position.x, pose.position.y))

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def bootstrap_drive(self):
        self.publish_cmd(self.bootstrap_linear_speed, self.bootstrap_angular_speed)

    def predict_path(self):
        xy = torch.tensor(list(self.positions), dtype=torch.float32).unsqueeze(0)
        origin = xy[:, -1:, :].clone()
        xy_rel = xy - origin
        with torch.no_grad():
            pred_rel = self.model(xy_rel).squeeze(0).cpu().numpy()
        pred_abs = pred_rel + origin.squeeze(0).cpu().numpy()
        self.predicted_path = pred_abs
        return pred_abs

    def step(self):
        if self.current_pose is None:
            return

        if time.monotonic() - self.start_time < self.bootstrap_seconds or len(self.positions) < self.past_len:
            self.bootstrap_drive()
            return

        pred_abs = self.predict_path()
        target_idx = min(self.target_index, len(pred_abs) - 1)
        target_x, target_y = pred_abs[target_idx]
        target_x += self.target_bias_x
        target_y += self.target_bias_y

        dx = target_x - self.current_pose.position.x
        dy = target_y - self.current_pose.position.y
        distance = math.hypot(dx, dy)

        if distance < self.waypoint_reached_dist:
            target_x, target_y = pred_abs[-1]
            target_x += self.target_bias_x
            target_y += self.target_bias_y
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
