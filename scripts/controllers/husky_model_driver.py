"""Live CNN-LSTM Husky controller.

The CNN-LSTM predicts short-horizon future waypoints from recent ego motion.
This node blends those predicted waypoints with the final goal direction and
publishes ``cmd_vel`` commands to drive the robot toward that goal.
"""

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
from tf2_msgs.msg import TFMessage


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CNNLSTM(nn.Module):
    """Convolutional-temporal predictor used for the live CNN baseline."""

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
    """Convert ego-motion predictions into live Husky velocity commands."""

    def __init__(
        self,
        node_name: str,
        cmd_topic: str,
        odom_topic: str,
        world_pose_topic: str | None,
        scan_topic: str | None,
        pointcloud_topic: str | None,
        hazard_topic: str | None,
        summary_path: str | Path,
        goal_xyz: tuple[float, float, float] | None = None,
        world_goal_xyz: tuple[float, float, float] | None = None,
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
        goal_tolerance: float = 1.5,
        goal_blend: float = 0.75,
        final_approach_distance: float = 6.0,
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
        self.world_pose_topic = world_pose_topic
        parts = [part for part in self.odom_topic.split("/") if part]
        self.model_frame_id = parts[1] if len(parts) >= 2 else None
        self.scan_topic = scan_topic
        self.pointcloud_topic = pointcloud_topic
        self.hazard_topic = hazard_topic
        self.goal_xyz = goal_xyz
        self.world_goal_xyz = world_goal_xyz
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
        self.goal_tolerance = goal_tolerance
        self.goal_blend = goal_blend
        self.final_approach_distance = final_approach_distance
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
        if self.world_pose_topic is not None:
            self.create_subscription(TFMessage, self.world_pose_topic, self.world_pose_cb, 10)

        self.positions = deque(maxlen=self.past_len)
        self.history_source = "odom"
        self.current_pose = None
        self.current_yaw = 0.0
        self.current_world_pose = None
        self.current_world_yaw = None
        self.initial_odom_pose = None
        self.initial_world_pose = None
        self.world_to_odom_yaw_offset = None
        self.fixed_goal_xy = None
        self.fixed_goal_logged = False
        self.predicted_path = None
        self.arrived = False
        self.start_time = time.monotonic()
        self.last_diag_log = 0.0
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
        if self.history_source != "world":
            self.positions.append((pose.position.x, pose.position.y))
        if self.initial_odom_pose is None:
            self.initial_odom_pose = (
                float(pose.position.x),
                float(pose.position.y),
            )
        self._update_fixed_goal_if_possible()

    def world_pose_cb(self, msg: TFMessage):
        model_transform = None
        base_link_transform = None
        for transform in msg.transforms:
            child = transform.child_frame_id or ""
            child_parts = [part for part in child.split("/") if part]
            if self.model_frame_id is not None and (
                child == self.model_frame_id
                or child.endswith(f"/{self.model_frame_id}")
                or self.model_frame_id in child_parts
            ) and not child.endswith("/base_link"):
                model_transform = transform
            elif child == "base_link" or child.endswith("/base_link"):
                base_link_transform = transform

        selected = model_transform if model_transform is not None else base_link_transform
        if selected is None:
            return

        translation = selected.transform.translation
        rotation = selected.transform.rotation
        self.current_world_pose = (
            float(translation.x),
            float(translation.y),
        )
        if self.initial_world_pose is None:
            self.initial_world_pose = self.current_world_pose
        self.current_world_yaw = quaternion_to_yaw(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
        if self.history_source != "world":
            self.positions.clear()
            self.history_source = "world"
        self.positions.append(self.current_world_pose)

    def _update_fixed_goal_if_possible(self):
        if self.world_pose_topic is not None:
            return
        goal = self._current_world_goal()
        if (
            goal is None
            or self.fixed_goal_xy is not None
            or self.initial_odom_pose is None
            or self.initial_world_pose is None
            or self.world_to_odom_yaw_offset is None
        ):
            return

        world_dx = goal[0] - self.initial_world_pose[0]
        world_dy = goal[1] - self.initial_world_pose[1]
        c = math.cos(self.world_to_odom_yaw_offset)
        s = math.sin(self.world_to_odom_yaw_offset)
        odom_dx = c * world_dx + s * world_dy
        odom_dy = -s * world_dx + c * world_dy
        self.fixed_goal_xy = (
            self.initial_odom_pose[0] + odom_dx,
            self.initial_odom_pose[1] + odom_dy,
        )
        if not self.fixed_goal_logged:
            self.get_logger().info(
                "Fixed odom goal computed from world goal: "
                f"world_goal=({goal[0]:.3f}, {goal[1]:.3f}) "
                f"odom_goal=({self.fixed_goal_xy[0]:.3f}, {self.fixed_goal_xy[1]:.3f})"
            )
            self.fixed_goal_logged = True

    def _use_world_control(self):
        return self.current_world_pose is not None and self.current_world_yaw is not None

    def _current_xy(self):
        if self._use_world_control():
            return self.current_world_pose
        if self.current_pose is None:
            return None
        return (self.current_pose.position.x, self.current_pose.position.y)

    def _current_heading(self):
        if self._use_world_control():
            return self.current_world_yaw
        if self.current_pose is None:
            return None
        return self.current_yaw

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def bootstrap_drive(self):
        """Collect enough motion history before the model is trusted, turning toward goal."""

        current_xy = self._current_xy()
        current_yaw = self._current_heading()
        if current_xy is None or current_yaw is None:
            # Fallback to original behavior
            self.publish_cmd(self.bootstrap_linear_speed, self.bootstrap_angular_speed)
            return

        goal = self._current_goal()
        if goal is None:
            self.publish_cmd(self.bootstrap_linear_speed, self.bootstrap_angular_speed)
            return

        # Compute heading to goal
        dx = goal[0] - current_xy[0]
        dy = goal[1] - current_xy[1]
        goal_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(goal_heading - current_yaw)

        # Proportional control for turning
        angular_z = clamp(heading_error * self.bootstrap_turn_gain, -self.max_angular_speed, self.max_angular_speed)

        # If heading error is large, rotate first; otherwise creep forward.
        if abs(heading_error) > 0.6:
            linear_x = 0.0
        else:
            linear_x = self.bootstrap_linear_speed * 0.5

        self.publish_cmd(linear_x, angular_z)

    def predict_path(self):
        """Predict future planar waypoints in the current odometry frame."""

        xy = torch.tensor(list(self.positions), dtype=torch.float32).unsqueeze(0)
        origin = xy[:, -1:, :].clone()
        xy_rel = xy - origin
        with torch.no_grad():
            pred_rel = self.model(xy_rel).squeeze(0).cpu().numpy()
        pred_abs = pred_rel + origin.squeeze(0).cpu().numpy()
        self.predicted_path = pred_abs
        return pred_abs

    def _current_goal(self):
        if self._use_world_control():
            goal = self._current_world_goal()
            if goal is not None:
                return goal
        if self.fixed_goal_xy is not None:
            return self.fixed_goal_xy
        if self.goal_xyz is None:
            return None
        return (float(self.goal_xyz[0]), float(self.goal_xyz[1]))

    def _current_world_goal(self):
        if self.world_goal_xyz is None:
            return None
        return (float(self.world_goal_xyz[0]), float(self.world_goal_xyz[1]))

    def _current_goal_from_world(self):
        return self._current_goal()

    def _distance_to_goal_world(self):
        goal = self._current_world_goal()
        if goal is None or self.current_world_pose is None:
            return None
        dx = goal[0] - self.current_world_pose[0]
        dy = goal[1] - self.current_world_pose[1]
        return math.hypot(dx, dy)

    def _distance_to_goal(self):
        goal = self._current_goal()
        current_xy = self._current_xy()
        if goal is None or current_xy is None:
            return None
        dx = goal[0] - current_xy[0]
        dy = goal[1] - current_xy[1]
        return math.hypot(dx, dy)

    def _goal_heading(self):
        goal = self._current_goal()
        current_xy = self._current_xy()
        if goal is None or current_xy is None:
            return None
        return math.atan2(
            goal[1] - current_xy[1],
            goal[0] - current_xy[0],
        )

    def step(self):
        if self.world_pose_topic is not None and not self._use_world_control():
            return

        current_xy = self._current_xy()
        current_yaw = self._current_heading()
        if current_xy is None or current_yaw is None:
            return

        if self.arrived:
            self.publish_cmd(0.0, 0.0)
            return

        remaining = self._distance_to_goal()
        if remaining is not None and remaining <= self.goal_tolerance:
            goal = self._current_goal()
            if goal is not None:
                self.get_logger().info(
                    "Arrival triggered: "
                    f"pose=({current_xy[0]:.3f}, {current_xy[1]:.3f}) "
                    f"goal=({goal[0]:.3f}, {goal[1]:.3f}) "
                    f"remaining={remaining:.3f} tol={self.goal_tolerance:.3f}"
                )
            self.arrived = True
            self.publish_cmd(0.0, 0.0)
            return

        now = time.monotonic()
        if remaining is not None and (now - self.last_diag_log) >= 2.0:
            goal = self._current_goal()
            if goal is not None:
                self.get_logger().info(
                    "Tracking status: "
                    f"pose=({current_xy[0]:.3f}, {current_xy[1]:.3f}) "
                    f"goal=({goal[0]:.3f}, {goal[1]:.3f}) "
                    f"remaining={remaining:.3f}"
                )
            self.last_diag_log = now

        if time.monotonic() - self.start_time < self.bootstrap_seconds or len(self.positions) < self.past_len:
            self.bootstrap_drive()
            return

        # Enforce coarse goal alignment before trusting predicted waypoints.
        goal_heading = self._goal_heading()
        if goal_heading is not None:
            goal_heading_error = wrap_angle(goal_heading - current_yaw)
            if abs(goal_heading_error) > 0.5:
                angular_z = clamp(
                    self.cmd_angular_gain * goal_heading_error,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )
                self.publish_cmd(0.0, angular_z)
                return

        goal = self._current_goal()
        world_goal = self._current_goal_from_world()
        pred_abs = None
        if goal is not None and remaining is not None and remaining <= self.final_approach_distance:
            if world_goal is not None:
                target_x, target_y = world_goal
            else:
                target_x, target_y = goal
        else:
            pred_abs = self.predict_path()
            target_idx = min(self.target_index, len(pred_abs) - 1)
            target_x, target_y = pred_abs[target_idx]
            target_x += self.target_bias_x
            target_y += self.target_bias_y
            if goal is not None:
                target_x = (1.0 - self.goal_blend) * target_x + self.goal_blend * goal[0]
                target_y = (1.0 - self.goal_blend) * target_y + self.goal_blend * goal[1]

        dx = target_x - current_xy[0]
        dy = target_y - current_xy[1]
        distance = math.hypot(dx, dy)

        if pred_abs is not None and distance < self.waypoint_reached_dist:
            target_x, target_y = pred_abs[-1]
            target_x += self.target_bias_x
            target_y += self.target_bias_y
            if goal is not None:
                target_x = (1.0 - self.goal_blend) * target_x + self.goal_blend * goal[0]
                target_y = (1.0 - self.goal_blend) * target_y + self.goal_blend * goal[1]
            dx = target_x - current_xy[0]
            dy = target_y - current_xy[1]
            distance = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(target_heading - current_yaw)

        linear_x = clamp(self.cmd_linear_gain * distance, 0.0, self.max_linear_speed)
        if distance > self.waypoint_reached_dist and abs(heading_error) < 0.35:
            linear_x = max(linear_x, self.min_linear_speed)
        if abs(heading_error) > 0.6:
            linear_x *= 0.5
        # If we are strongly misaligned, prioritize heading correction.
        if abs(heading_error) > 0.9:
            linear_x = 0.0
        if abs(heading_error) < self.heading_deadband:
            heading_error = 0.0

        angular_z = clamp(
            self.cmd_angular_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        self.publish_cmd(linear_x, angular_z)
