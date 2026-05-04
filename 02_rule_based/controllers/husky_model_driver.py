"""Test Husky goal-seeking controller with stronger obstacle-avoidance turns."""

import math
import time
from collections import deque

from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ModelHuskyDriver(Node):
    """Drive the Husky toward a goal while avoiding obstacles with a small state machine."""

    def __init__(
        self,
        node_name: str,
        cmd_topic: str,
        odom_topic: str,
        world_pose_topic: str | None,
        uav_ready_topic: str | None = None,
        require_uav_ready: bool = False,
        obstacle_action_topic: str | None = None,
        obstacle_clearance_topic: str | None = None,
        state_topic: str | None = None,
        goal_xyz: tuple[float, float, float] | None = None,
        world_goal_xyz: tuple[float, float, float] | None = None,
        bootstrap_seconds: float = 3.0,
        bootstrap_linear_speed: float = 0.30,
        bootstrap_turn_gain: float = 1.0,
        control_period: float = 0.1,
        cmd_linear_gain: float = 0.65,
        cmd_angular_gain: float = 1.6,
        min_linear_speed: float = 0.0,
        max_linear_speed: float = 0.55,
        max_angular_speed: float = 1.2,
        heading_deadband: float = 0.08,
        goal_tolerance: float = 1.5,
        goal_align_heading_threshold: float = 0.5,
        goal_align_linear_speed: float = 0.28,
        obstacle_stop_distance: float = 1.8,
        obstacle_turn_speed: float = 1.0,
        obstacle_turn_speed_close: float = 1.4,
        stuck_timeout_seconds: float = 3.0,
        stuck_progress_distance: float = 0.15,
        stuck_min_command_speed: float = 0.2,
        stuck_reverse_speed: float = -0.35,
        stuck_reverse_seconds: float = 1.5,
        stuck_bootstrap_seconds: float = 2.0,
        stuck_cooldown_seconds: float = 4.0,
        strict_reverse_distance: float = 0.80,
        strict_reverse_cycles: int = 4,
        commit_clearance_distance: float = 4.8,
        commit_min_distance: float = 1.8,
        commit_linear_speed: float = 0.45,
        commit_max_angular_speed: float = 0.25,
        commit_clear_hold_seconds: float = 1.5,
        post_avoid_forward_speed: float = 0.18,
        post_avoid_forward_seconds: float = 0.8,
        post_recover_commit_cooldown_seconds: float = 2.0,
        reverse_loop_window_seconds: float = 12.0,
        reverse_loop_limit: int = 2,
        loop_reassess_pause_seconds: float = 2.5,
        reassess_timeout_seconds: float = 5.0,
        reassess_pause_seconds: float = 1.5,
        reassess_min_goal_progress: float = 0.12,
        reassess_cooldown_seconds: float = 1.5,
    ):
        super().__init__(node_name)

        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.world_pose_topic = world_pose_topic
        self.uav_ready_topic = uav_ready_topic
        self.require_uav_ready = require_uav_ready
        self.obstacle_action_topic = obstacle_action_topic
        self.obstacle_clearance_topic = obstacle_clearance_topic
        self.state_topic = state_topic
        self.goal_xyz = goal_xyz
        self.world_goal_xyz = world_goal_xyz

        self.bootstrap_seconds = bootstrap_seconds
        self.bootstrap_linear_speed = bootstrap_linear_speed
        self.bootstrap_turn_gain = bootstrap_turn_gain
        self.control_period = control_period
        self.cmd_linear_gain = cmd_linear_gain
        self.cmd_angular_gain = cmd_angular_gain
        self.min_linear_speed = min_linear_speed
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.heading_deadband = heading_deadband
        self.goal_tolerance = goal_tolerance
        self.goal_align_heading_threshold = goal_align_heading_threshold
        self.goal_align_linear_speed = goal_align_linear_speed

        self.obstacle_stop_distance = obstacle_stop_distance
        self.obstacle_turn_speed = obstacle_turn_speed
        self.obstacle_turn_speed_close = obstacle_turn_speed_close

        self.stuck_timeout_seconds = stuck_timeout_seconds
        self.stuck_progress_distance = stuck_progress_distance
        self.stuck_min_command_speed = stuck_min_command_speed
        self.stuck_reverse_speed = stuck_reverse_speed
        self.stuck_reverse_seconds = stuck_reverse_seconds
        self.stuck_bootstrap_seconds = stuck_bootstrap_seconds
        self.stuck_cooldown_seconds = stuck_cooldown_seconds
        self.strict_reverse_distance = strict_reverse_distance
        self.strict_reverse_cycles = max(1, int(strict_reverse_cycles))

        self.commit_clearance_distance = commit_clearance_distance
        self.commit_min_distance = commit_min_distance
        self.commit_linear_speed = commit_linear_speed
        self.commit_max_angular_speed = commit_max_angular_speed
        self.commit_clear_hold_seconds = commit_clear_hold_seconds

        self.post_avoid_forward_speed = post_avoid_forward_speed
        self.post_avoid_forward_seconds = post_avoid_forward_seconds
        self.post_recover_commit_cooldown_seconds = post_recover_commit_cooldown_seconds

        self.reverse_loop_window_seconds = reverse_loop_window_seconds
        self.reverse_loop_limit = max(1, int(reverse_loop_limit))
        self.loop_reassess_pause_seconds = loop_reassess_pause_seconds
        self.reassess_timeout_seconds = reassess_timeout_seconds
        self.reassess_pause_seconds = reassess_pause_seconds
        self.reassess_min_goal_progress = reassess_min_goal_progress
        self.reassess_cooldown_seconds = reassess_cooldown_seconds

        # Hold avoid turns long enough to generate clear avoid_left/avoid_right labels.
        self.turn_limit_radians = math.radians(110.0)

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.state_pub = (
            self.create_publisher(String, self.state_topic, 10)
            if self.state_topic is not None
            else None
        )

        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)

        if self.world_pose_topic is not None:
            self.create_subscription(TFMessage, self.world_pose_topic, self.world_pose_cb, 10)

        if self.uav_ready_topic is not None:
            self.create_subscription(Bool, self.uav_ready_topic, self.uav_ready_cb, 10)

        if self.obstacle_action_topic is not None:
            self.create_subscription(String, self.obstacle_action_topic, self.obstacle_action_cb, 10)

        if self.obstacle_clearance_topic is not None:
            self.create_subscription(Vector3, self.obstacle_clearance_topic, self.obstacle_clearance_cb, 10)

        parts = [part for part in self.odom_topic.split("/") if part]
        self.model_frame_id = parts[1] if len(parts) >= 2 else None

        self.current_pose = None
        self.current_yaw = None
        self.current_world_pose = None
        self.current_world_yaw = None

        self.uav_ready = not self.require_uav_ready

        self.obstacle_action = "clear"
        self.obstacle_clearance = (float("inf"), float("inf"), float("inf"))

        self.arrived = False

        self.state = "bootstrap"
        self.state_until = 0.0

        self.avoid_direction = None
        self.avoid_start_heading = None
        self.commit_start_xy = None

        self.remaining_history = deque(maxlen=120)
        self.reverse_events = deque(maxlen=8)

        self.last_diag_log = 0.0
        self.last_command_linear_x = 0.0
        self.last_command_angular_z = 0.0
        self.last_uav_wait_log = 0.0

        self.stuck_cooldown_until = 0.0
        self.strict_blocked_cycles = 0
        self.clear_path_since = None
        self.post_recover_until = 0.0

        self.start_time = time.monotonic()

        self.timer = self.create_timer(self.control_period, self.step)

        self.get_logger().info(
            f"Loaded clean state-machine controller on {self.cmd_topic} using {self.odom_topic}"
        )

    def publish_cmd(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.pub.publish(msg)

        self.last_command_linear_x = float(linear_x)
        self.last_command_angular_z = float(angular_z)

    def _state_label(self) -> str:
        if self.arrived:
            return "arrived"

        if self.state == "avoid":
            if self.avoid_direction == "left":
                return "avoid_left"
            if self.avoid_direction == "right":
                return "avoid_right"

        return self.state

    def publish_state(self):
        if self.state_pub is None:
            return

        msg = String()
        msg.data = self._state_label()
        self.state_pub.publish(msg)

    def odom_cb(self, msg: Odometry):
        pose = msg.pose.pose
        self.current_pose = pose
        self.current_yaw = quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

    def world_pose_cb(self, msg: TFMessage):
        selected = None

        for transform in msg.transforms:
            child = transform.child_frame_id or ""
            child_parts = [part for part in child.split("/") if part]

            if self.model_frame_id is not None and (
                child == self.model_frame_id
                or child.endswith(f"/{self.model_frame_id}")
                or self.model_frame_id in child_parts
            ):
                selected = transform
                break

            if child == "base_link" or child.endswith("/base_link"):
                selected = transform

        if selected is None:
            return

        translation = selected.transform.translation
        rotation = selected.transform.rotation

        self.current_world_pose = (
            float(translation.x),
            float(translation.y),
            float(translation.z),
        )
        self.current_world_yaw = quaternion_to_yaw(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

    def obstacle_action_cb(self, msg: String):
        self.obstacle_action = msg.data.strip().lower() if msg.data else "clear"

    def obstacle_clearance_cb(self, msg: Vector3):
        self.obstacle_clearance = (
            float(msg.x),
            float(msg.y),
            float(msg.z),
        )

    def uav_ready_cb(self, msg: Bool):
        self.uav_ready = bool(msg.data)

    def _use_world_control(self) -> bool:
        return self.current_world_pose is not None and self.current_world_yaw is not None

    def _current_xy(self):
        if self._use_world_control():
            return (
                self.current_world_pose[0],
                self.current_world_pose[1],
            )

        if self.current_pose is None:
            return None

        return (
            float(self.current_pose.position.x),
            float(self.current_pose.position.y),
        )

    def _current_altitude(self):
        if self._use_world_control():
            return float(self.current_world_pose[2])

        if self.current_pose is None:
            return None

        return float(self.current_pose.position.z)

    def _current_heading(self):
        if self._use_world_control():
            return self.current_world_yaw

        return self.current_yaw

    def _current_goal(self):
        if self._use_world_control() and self.world_goal_xyz is not None:
            return (
                float(self.world_goal_xyz[0]),
                float(self.world_goal_xyz[1]),
            )

        if self.goal_xyz is not None:
            return (
                float(self.goal_xyz[0]),
                float(self.goal_xyz[1]),
            )

        return None

    def _distance_to_goal(self):
        goal = self._current_goal()
        current_xy = self._current_xy()

        if goal is None or current_xy is None:
            return None

        return math.hypot(goal[0] - current_xy[0], goal[1] - current_xy[1])

    def _goal_heading(self):
        goal = self._current_goal()
        current_xy = self._current_xy()

        if goal is None or current_xy is None:
            return None

        return math.atan2(goal[1] - current_xy[1], goal[0] - current_xy[0])

    def _front_clearance(self) -> float:
        return float(self.obstacle_clearance[0])

    def _left_clearance(self) -> float:
        return float(self.obstacle_clearance[1])

    def _right_clearance(self) -> float:
        return float(self.obstacle_clearance[2])

    def _obstacle_active(self) -> bool:
        return (self.obstacle_action or "clear") != "clear"

    def _record_remaining(self, now: float, remaining: float):
        self.remaining_history.append((now, float(remaining)))

        while self.remaining_history and (now - self.remaining_history[0][0]) > 8.0:
            self.remaining_history.popleft()

    def _choose_avoid_direction(self) -> str:
        if self.obstacle_action.endswith("left"):
            return "left"

        if self.obstacle_action.endswith("right"):
            return "right"

        if self._left_clearance() >= self._right_clearance():
            return "left"

        return "right"

    def _enter_state(self, state: str, until: float | None = None):
        self.state = state
        self.state_until = 0.0 if until is None else float(until)

    def _enter_avoid(self, now: float):
        heading = self._current_heading()

        self.avoid_direction = self._choose_avoid_direction()
        self.avoid_start_heading = heading

        self._enter_state("avoid")

        self.get_logger().info(
            f"Avoiding obstacle: direction={self.avoid_direction} front={self._front_clearance():.2f}"
        )

    def _enter_reassess(self, now: float, remaining: float, reason: str):
        pause_seconds = (
            self.loop_reassess_pause_seconds
            if reason == "loop_guard"
            else self.reassess_pause_seconds
        )

        self._enter_state("reassess", now + pause_seconds)

        self.remaining_history.clear()
        self.remaining_history.append((now, float(remaining)))

        self.clear_path_since = None

        self.get_logger().info(
            f"Reassessing navigation: reason={reason} remaining={remaining:.3f}"
        )

    def _enter_commit_forward(self, now: float, remaining: float):
        current_xy = self._current_xy()

        self.commit_start_xy = (
            None
            if current_xy is None
            else (float(current_xy[0]), float(current_xy[1]))
        )

        self.avoid_direction = None
        self.avoid_start_heading = None
        self.clear_path_since = None

        self._enter_state("commit_forward")

        self.remaining_history.clear()
        self.remaining_history.append((now, float(remaining)))

        self.get_logger().info(
            "Commit forward: "
            f"front={self._front_clearance():.2f} "
            f"clear_target={self.commit_clearance_distance:.2f} "
            f"travel_target={self.commit_min_distance:.2f}"
        )

    def _enter_reverse(self, now: float, remaining: float, reason: str):
        self._enter_state("reverse", now + self.stuck_reverse_seconds)

        self.stuck_cooldown_until = (
            now + self.stuck_reverse_seconds + self.stuck_cooldown_seconds
        )

        self.clear_path_since = None
        self.reverse_events.append(float(now))

        self.remaining_history.clear()
        self.remaining_history.append((now, float(remaining)))

        self.get_logger().info(
            "Stuck recovery: "
            f"reason={reason} remaining={remaining:.3f} "
            f"reverse_speed={self.stuck_reverse_speed:.2f} "
            f"duration={self.stuck_reverse_seconds:.2f}"
        )

    def _should_reassess(self, now: float, remaining: float | None) -> bool:
        if remaining is None or self._obstacle_active():
            return False

        if remaining <= max(self.goal_tolerance + 1.0, 2.5):
            return False

        if self.state == "reassess" or now < self.state_until:
            return False

        reference = None

        for sample in self.remaining_history:
            if (now - sample[0]) >= self.reassess_timeout_seconds:
                reference = sample
                break

        if reference is None:
            return False

        goal_progress = reference[1] - remaining

        return goal_progress < self.reassess_min_goal_progress

    def _should_reverse(self, now: float, remaining: float | None) -> bool:
        if remaining is None:
            return False

        if self.state in {"reverse", "recover", "reassess", "commit_forward"} or now < self.state_until:
            return False

        if now < self.stuck_cooldown_until:
            return False

        if self.last_command_linear_x < self.stuck_min_command_speed:
            return False

        reference = None

        for sample in self.remaining_history:
            if (now - sample[0]) >= self.stuck_timeout_seconds:
                reference = sample
                break

        if reference is None:
            return False

        goal_progress = reference[1] - remaining

        if goal_progress >= self.stuck_progress_distance:
            return False

        front = self._front_clearance()

        if not self._obstacle_active():
            return False

        return front <= self.obstacle_stop_distance

    def _should_strict_reverse(self, now: float, remaining: float | None) -> bool:
        if remaining is None:
            self.strict_blocked_cycles = 0
            return False

        if self.state in {"reverse", "recover"} or now < self.stuck_cooldown_until:
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

    def _reverse_loop_detected(self, now: float) -> bool:
        while self.reverse_events and (now - self.reverse_events[0]) > self.reverse_loop_window_seconds:
            self.reverse_events.popleft()

        return len(self.reverse_events) >= self.reverse_loop_limit

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

    def _commit_distance_traveled(self) -> float:
        current_xy = self._current_xy()

        if self.commit_start_xy is None or current_xy is None:
            return 0.0

        return math.hypot(
            current_xy[0] - self.commit_start_xy[0],
            current_xy[1] - self.commit_start_xy[1],
        )

    def _commit_command(self):
        goal_heading = self._goal_heading()
        current_heading = self._current_heading()

        if goal_heading is None or current_heading is None:
            return (self.commit_linear_speed, 0.0)

        heading_error = wrap_angle(goal_heading - current_heading)

        angular_z = clamp(
            0.45 * self.cmd_angular_gain * heading_error,
            -self.commit_max_angular_speed,
            self.commit_max_angular_speed,
        )

        return (self.commit_linear_speed, angular_z)

    def _goal_command(self):
        goal_heading = self._goal_heading()
        remaining = self._distance_to_goal()
        current_heading = self._current_heading()

        if goal_heading is None or remaining is None or current_heading is None:
            return (0.0, 0.0)

        heading_error = wrap_angle(goal_heading - current_heading)

        if abs(heading_error) < self.heading_deadband:
            heading_error = 0.0

        angular_z = clamp(
            self.cmd_angular_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        linear_x = self._goal_speed(remaining, heading_error)

        return (linear_x, angular_z)

    def _bootstrap_command(self):
        goal_heading = self._goal_heading()
        current_heading = self._current_heading()

        if goal_heading is None or current_heading is None:
            return (0.0, 0.0)

        heading_error = wrap_angle(goal_heading - current_heading)

        angular_z = clamp(
            self.bootstrap_turn_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        linear_x = min(
            self.bootstrap_linear_speed,
            self.goal_align_linear_speed,
        )

        if abs(heading_error) > 1.0:
            linear_x *= 0.35
        elif abs(heading_error) > 0.6:
            linear_x *= 0.60

        return (linear_x, angular_z)

    def _avoid_command(self, now: float, remaining: float):
        current_heading = self._current_heading()

        if current_heading is None:
            return (0.0, 0.0)

        if self.avoid_direction is None:
            self.avoid_direction = self._choose_avoid_direction()

        if self.avoid_start_heading is None:
            self.avoid_start_heading = current_heading

        turned = abs(wrap_angle(current_heading - self.avoid_start_heading))
        front = self._front_clearance()

        sign = 1.0 if self.avoid_direction == "left" else -1.0

        close_to_obstacle = front <= (self.obstacle_stop_distance + 0.8)

        angular_speed = (
            self.obstacle_turn_speed_close
            if close_to_obstacle
            else self.obstacle_turn_speed
        )

        angular_z = sign * angular_speed

        # When front is clear, move a little forward before returning to goal tracking.
        # This prevents immediate turning back into the obstacle.
        if not self._obstacle_active() and front > (self.obstacle_stop_distance + 0.5):
            if self.clear_path_since is None:
                self.clear_path_since = now
            elif (now - self.clear_path_since) >= self.post_avoid_forward_seconds:
                self.avoid_direction = None
                self.avoid_start_heading = None
                self.clear_path_since = None
                self._enter_state("go_to_goal")
                return self._goal_command()

            return (self.post_avoid_forward_speed, 0.0)

        self.clear_path_since = None

        if turned >= self.turn_limit_radians:
            if not self._obstacle_active():
                self.avoid_direction = None
                self.avoid_start_heading = None
                self.clear_path_since = None
                self._enter_state("go_to_goal")
                return self._goal_command()

            self._enter_reverse(now, remaining, "avoid_turn_limit")
            return (self.stuck_reverse_speed, 0.0)

        if front <= self.obstacle_stop_distance:
            linear_x = 0.0
        elif front <= (self.obstacle_stop_distance + 0.8):
            linear_x = 0.10
        else:
            linear_x = 0.22

        return (linear_x, angular_z)

    def step(self):
        self.publish_state()

        current_xy = self._current_xy()
        current_heading = self._current_heading()

        if current_xy is None or current_heading is None:
            return

        if self.arrived:
            self.publish_cmd(0.0, 0.0)
            return

        if self.require_uav_ready and not self.uav_ready:
            now = time.monotonic()

            if (now - self.last_uav_wait_log) >= 2.0:
                self.get_logger().info("Waiting for UAV readiness before starting UGV motion")
                self.last_uav_wait_log = now

            self.publish_cmd(0.0, 0.0)
            return

        remaining = self._distance_to_goal()

        if remaining is None:
            self.publish_cmd(0.0, 0.0)
            return

        now = time.monotonic()
        self._record_remaining(now, remaining)

        if remaining <= self.goal_tolerance:
            goal = self._current_goal()
            altitude = self._current_altitude()

            if goal is not None:
                self.get_logger().info(
                    "Arrival triggered: "
                    f"pose=({current_xy[0]:.3f}, {current_xy[1]:.3f}) "
                    f"z={altitude:.3f} "
                    f"goal=({goal[0]:.3f}, {goal[1]:.3f}) "
                    f"remaining={remaining:.3f} "
                    f"tol={self.goal_tolerance:.3f}"
                )

            self.arrived = True
            self.publish_cmd(0.0, 0.0)
            return

        if (now - self.last_diag_log) >= 2.0:
            goal = self._current_goal()
            altitude = self._current_altitude()

            if goal is not None:
                self.get_logger().info(
                    "Tracking status: "
                    f"pose=({current_xy[0]:.3f}, {current_xy[1]:.3f}) "
                    f"z={altitude:.3f} "
                    f"goal=({goal[0]:.3f}, {goal[1]:.3f}) "
                    f"remaining={remaining:.3f} "
                    f"state={self.state}"
                )

            self.last_diag_log = now

        if self.state == "reassess":
            if now < self.state_until:
                self.publish_cmd(0.0, 0.0)
                return

            self._enter_state("go_to_goal")

        if self.state == "reverse":
            if now < self.state_until:
                self.publish_cmd(self.stuck_reverse_speed, 0.0)
                return

            self.remaining_history.clear()
            self.remaining_history.append((now, float(remaining)))
            self._enter_state("recover", now + self.stuck_bootstrap_seconds)
            self.publish_cmd(0.0, 0.0)
            return

        if self.state == "recover":
            if now < self.state_until:
                self.publish_cmd(0.0, 0.0)
                return

            self.remaining_history.clear()
            self.remaining_history.append((now, float(remaining)))
            self.post_recover_until = now + self.post_recover_commit_cooldown_seconds
            self.clear_path_since = None
            self._enter_state("go_to_goal")

        if self._reverse_loop_detected(now):
            self._enter_reassess(now, remaining, "loop_guard")
            self.publish_cmd(0.0, 0.0)
            return

        if self.state == "commit_forward":
            traveled = self._commit_distance_traveled()
            front = self._front_clearance()

            if self._obstacle_active() and front <= self.strict_reverse_distance:
                self.commit_start_xy = None
                self._enter_reverse(now, remaining, "commit_front_blocked_hard")
                self.publish_cmd(self.stuck_reverse_speed, 0.0)
                return

            if traveled >= self.commit_min_distance and front >= self.commit_clearance_distance:
                self.commit_start_xy = None
                self.remaining_history.clear()
                self.remaining_history.append((now, float(remaining)))
                self._enter_state("go_to_goal")
            else:
                self.publish_cmd(*self._commit_command())
                return

        if self.state == "bootstrap":
            if (now - self.start_time) < self.bootstrap_seconds:
                self.publish_cmd(*self._bootstrap_command())
                return

            self._enter_state("go_to_goal")

        if self._should_strict_reverse(now, remaining):
            self._enter_reverse(now, remaining, "front_blocked_hard")
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        if self._obstacle_active() and self.state != "avoid":
            self._enter_avoid(now)

        if self.state == "avoid":
            self.publish_cmd(*self._avoid_command(now, remaining))
            return

        if self._should_reverse(now, remaining):
            self._enter_reverse(now, remaining, "goal_progress_stalled")
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        if self._should_reassess(now, remaining):
            self._enter_reassess(now, remaining, "goal_progress_stalled")
            self.publish_cmd(0.0, 0.0)
            return

        self.publish_cmd(*self._goal_command())