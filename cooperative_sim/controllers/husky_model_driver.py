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
        terrain_profile_topic: str | None = None,
        gap_profile_topic: str | None = None,
        hazard_guidance_topic: str | None = None,
        depth_classification_topic: str | None = None,
        final_decision_topic: str | None = None,
        state_topic: str | None = None,
        use_lidar_straight_approach: bool = False,
        use_lidar_path_planning: bool = True,
        use_depth_classification: bool = False,
        use_hazard_map: bool = False,
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
        obstacle_caution_distance: float = 3.2,
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
        post_recover_commit_cooldown_seconds: float = 2.0,
        post_avoid_min_progress_distance: float = 0.01,
        post_avoid_progress_timeout_seconds: float = 1.2,
        reverse_loop_window_seconds: float = 12.0,
        reverse_loop_limit: int = 2,
        loop_reassess_pause_seconds: float = 2.5,
        reassess_timeout_seconds: float = 5.0,
        reassess_pause_seconds: float = 1.5,
        reassess_min_goal_progress: float = 0.01,
        reassess_cooldown_seconds: float = 1.5,
        terrain_speedup_extent_mid_m: float = 1.5,
        terrain_speedup_extent_high_m: float = 3.0,
        terrain_speedup_mid_multiplier: float = 2.0,
        terrain_speedup_high_multiplier: float = 3.0,
        terrain_commit_progress_timeout_seconds: float = 2.0,
        terrain_commit_min_progress_distance: float = 0.01,
        terrain_commit_linear_speed: float = 0.55,
        terrain_commit_max_angular_speed: float = 0.20,
        zero_motion_reverse_timeout_seconds: float = 6.0,
        zero_motion_reverse_distance: float = 0.02,
        circling_reverse_timeout_seconds: float = 3.0,
        circling_reverse_distance: float = 0.02,
        circling_heading_change_radians: float = math.radians(270.0),
        self_stuck_confirm_cycles: int = 2,
        reverse_pause_seconds: float = 2.0,
        escape_turn_radians: float = math.pi / 2.0,
        escape_turn_timeout_seconds: float = 3.0,
        escape_drive_seconds: float = 3.0,
        near_goal_commit_radius: float = 8.0,
    ):
        super().__init__(node_name)

        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.world_pose_topic = world_pose_topic
        self.uav_ready_topic = uav_ready_topic
        self.require_uav_ready = require_uav_ready
        self.obstacle_action_topic = obstacle_action_topic
        self.obstacle_clearance_topic = obstacle_clearance_topic
        self.terrain_profile_topic = terrain_profile_topic
        self.gap_profile_topic = gap_profile_topic
        self.hazard_guidance_topic = hazard_guidance_topic
        self.depth_classification_topic = depth_classification_topic
        self.final_decision_topic = final_decision_topic
        self.state_topic = state_topic
        self.use_lidar_straight_approach = bool(use_lidar_straight_approach)
        self.use_lidar_path_planning = bool(use_lidar_path_planning)
        self.use_depth_classification = bool(use_depth_classification)
        self.use_hazard_map = bool(use_hazard_map)
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
        self.obstacle_caution_distance = obstacle_caution_distance
        self.obstacle_turn_speed = obstacle_turn_speed
        self.obstacle_turn_speed_close = obstacle_turn_speed_close
        self.stuck_timeout_seconds = stuck_timeout_seconds
        self.stuck_progress_distance = stuck_progress_distance
        self.stuck_min_command_speed = stuck_min_command_speed
        self.stuck_reverse_speed = -max(abs(float(stuck_reverse_speed)), abs(float(self.max_linear_speed)))
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
        self.post_recover_commit_cooldown_seconds = post_recover_commit_cooldown_seconds
        self.post_avoid_min_progress_distance = post_avoid_min_progress_distance
        self.post_avoid_progress_timeout_seconds = post_avoid_progress_timeout_seconds
        self.reverse_loop_window_seconds = reverse_loop_window_seconds
        self.reverse_loop_limit = max(1, int(reverse_loop_limit))
        self.loop_reassess_pause_seconds = loop_reassess_pause_seconds
        self.reassess_timeout_seconds = reassess_timeout_seconds
        self.reassess_pause_seconds = reassess_pause_seconds
        self.reassess_min_goal_progress = reassess_min_goal_progress
        self.reassess_cooldown_seconds = reassess_cooldown_seconds
        self.terrain_speedup_extent_mid_m = terrain_speedup_extent_mid_m
        self.terrain_speedup_extent_high_m = terrain_speedup_extent_high_m
        self.terrain_speedup_mid_multiplier = terrain_speedup_mid_multiplier
        self.terrain_speedup_high_multiplier = terrain_speedup_high_multiplier
        self.terrain_commit_progress_timeout_seconds = terrain_commit_progress_timeout_seconds
        self.terrain_commit_min_progress_distance = terrain_commit_min_progress_distance
        self.terrain_commit_linear_speed = terrain_commit_linear_speed
        self.terrain_commit_max_angular_speed = terrain_commit_max_angular_speed
        self.zero_motion_reverse_timeout_seconds = zero_motion_reverse_timeout_seconds
        self.zero_motion_reverse_distance = zero_motion_reverse_distance
        self.circling_reverse_timeout_seconds = circling_reverse_timeout_seconds
        self.circling_reverse_distance = circling_reverse_distance
        self.circling_heading_change_radians = circling_heading_change_radians
        self.self_stuck_confirm_cycles = max(1, int(self_stuck_confirm_cycles))
        self.reverse_pause_seconds = reverse_pause_seconds
        self.escape_turn_radians = escape_turn_radians
        self.escape_turn_timeout_seconds = escape_turn_timeout_seconds
        self.escape_drive_seconds = escape_drive_seconds
        self.near_goal_commit_radius = near_goal_commit_radius
        # Let the test driver hold avoid turns longer so left/right labels are
        # clearer and more visually distinct during scenario tuning.
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
        if self.terrain_profile_topic is not None:
            self.create_subscription(Vector3, self.terrain_profile_topic, self.terrain_profile_cb, 10)
        if self.gap_profile_topic is not None:
            self.create_subscription(Vector3, self.gap_profile_topic, self.gap_profile_cb, 10)
        if self.hazard_guidance_topic is not None:
            self.create_subscription(String, self.hazard_guidance_topic, self.hazard_guidance_cb, 10)
        if self.depth_classification_topic is not None:
            self.create_subscription(String, self.depth_classification_topic, self.depth_classification_cb, 10)
        if self.final_decision_topic is not None:
            self.create_subscription(String, self.final_decision_topic, self.final_decision_cb, 10)

        parts = [part for part in self.odom_topic.split("/") if part]
        self.model_frame_id = parts[1] if len(parts) >= 2 else None

        self.current_pose = None
        self.current_yaw = None
        self.current_world_pose = None
        self.current_world_yaw = None
        self.uav_ready = not self.require_uav_ready
        self.obstacle_action = "clear"
        self.obstacle_clearance = (float("inf"), float("inf"), float("inf"))
        self.terrain_extent_x = 0.0
        self.terrain_climbable = False
        self.gap_width = 0.0
        self.gap_center_y = 0.0
        self.gap_passable = False
        self.hazard_guidance = "clear"
        self.hazard_terrain_hint = False
        self.final_decision = "clear"
        self.depth_overall = "unknown"
        self.depth_left_class = "unknown"
        self.depth_center_class = "unknown"
        self.depth_right_class = "unknown"
        self.depth_left_m = 999.0
        self.depth_center_m = 999.0
        self.depth_right_m = 999.0
        self.depth_terrain_votes = 0
        self.depth_block_votes = 0
        self.depth_broad_block_votes = 0
        self.last_depth_override_log = 0.0
        self.last_lidar_priority_mode = None
        self.arrived = False

        self.state = "bootstrap"
        self.state_until = 0.0
        self.avoid_direction = None
        self.avoid_start_heading = None
        self.avoid_start_time = 0.0
        self.avoid_start_remaining = None
        self.commit_start_xy = None

        self.remaining_history = deque(maxlen=120)
        self.motion_history = deque(maxlen=240)
        self.last_diag_log = 0.0
        self.last_command_linear_x = 0.0
        self.last_command_angular_z = 0.0
        self.last_uav_wait_log = 0.0
        self.stuck_cooldown_until = 0.0
        self.strict_blocked_cycles = 0
        self.clear_path_since = None
        self.clear_path_start_remaining = None
        self.clear_path_best_remaining = None
        self.clear_path_last_progress_time = None
        self.terrain_commit_start_remaining = None
        self.terrain_commit_best_remaining = None
        self.terrain_commit_last_progress_time = None
        self.forward_clear_since = None
        self.forward_clear_start_remaining = None
        self.initial_remaining = None
        self.post_recover_until = 0.0
        self.reverse_events = deque(maxlen=8)
        self.reverse_start_xy = None
        self.reverse_failed_escape = False
        self.escape_turn_direction = None
        self.escape_turn_start_heading = None
        self.path_plan_direction = None
        self.path_plan_phase = None
        self.path_plan_heading_target = None
        self.path_plan_start_xy = None
        self.path_plan_distance_target = 0.0
        self.zero_motion_stuck_votes = 0
        self.circling_stuck_votes = 0
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
            return "reached"
        if self.state == "avoid":
            if self.avoid_direction == "left":
                return "avoidLeft"
            if self.avoid_direction == "right":
                return "avoidRight"
        mapping = {
            "bootstrap": "bootStrap",
            "go_to_goal": "toGoal",
            "stall": "stall",
            "commit_forward": "goForward",
            "terrain_commit": "terrain",
            "planned_path": "goForward",
            "reverse_pause": "reversePause",
            "escape_turn": "escapeTurn",
            "escape_drive": "escapeDrive",
            "recover": "recover",
            "reassess": "recheck",
            "reverse": "reverse",
        }
        return mapping.get(self.state, self.state)

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
        self.obstacle_clearance = (float(msg.x), float(msg.y), float(msg.z))

    def terrain_profile_cb(self, msg: Vector3):
        self.terrain_extent_x = max(0.0, float(msg.x))
        self.terrain_climbable = bool(msg.y >= 0.5)

    def gap_profile_cb(self, msg: Vector3):
        self.gap_width = max(0.0, float(msg.x))
        self.gap_center_y = float(msg.y)
        self.gap_passable = bool(msg.z >= 0.5)

    def hazard_guidance_cb(self, msg: String):
        self.hazard_guidance = msg.data.strip().lower() if msg.data else "clear"
        self.hazard_terrain_hint = self.hazard_guidance.startswith("terrain_sure_front")

    def final_decision_cb(self, msg: String):
        self.final_decision = msg.data.strip().lower() if msg.data else "clear"

    def depth_classification_cb(self, msg: String):
        text = msg.data.strip().lower() if msg.data else ""
        parsed: dict[str, str] = {}
        for token in text.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key] = value

        def parse_region(name: str) -> tuple[str, float]:
            raw = parsed.get(name, "unknown:999.0")
            if ":" not in raw:
                return (raw, 999.0)
            cls, depth_text = raw.split(":", 1)
            try:
                depth_value = float(depth_text)
            except ValueError:
                depth_value = 999.0
            return (cls, depth_value)

        self.depth_left_class, self.depth_left_m = parse_region("left")
        self.depth_center_class, self.depth_center_m = parse_region("center")
        self.depth_right_class, self.depth_right_m = parse_region("right")
        self.depth_overall = parsed.get("overall", "unknown")

        if self.depth_overall == "terrain":
            self.depth_terrain_votes = min(self.depth_terrain_votes + 1, 20)
            self.depth_block_votes = 0
            self.depth_broad_block_votes = 0
        elif self.depth_overall.startswith("block"):
            self.depth_block_votes = min(self.depth_block_votes + 1, 20)
            self.depth_terrain_votes = 0
            if self.depth_overall == "block_broad":
                self.depth_broad_block_votes = min(self.depth_broad_block_votes + 1, 20)
            else:
                self.depth_broad_block_votes = 0
        else:
            self.depth_terrain_votes = 0
            self.depth_block_votes = 0
            self.depth_broad_block_votes = 0

    def uav_ready_cb(self, msg: Bool):
        self.uav_ready = bool(msg.data)

    def _use_world_control(self) -> bool:
        return self.current_world_pose is not None and self.current_world_yaw is not None

    def _current_xy(self):
        if self._use_world_control():
            return (self.current_world_pose[0], self.current_world_pose[1])
        if self.current_pose is None:
            return None
        return (float(self.current_pose.position.x), float(self.current_pose.position.y))

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
            return (float(self.world_goal_xyz[0]), float(self.world_goal_xyz[1]))
        if self.goal_xyz is not None:
            return (float(self.goal_xyz[0]), float(self.goal_xyz[1]))
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
        if self.final_decision_topic is not None:
            return self.final_decision.startswith("block")
        return (self.obstacle_action or "clear") != "clear"

    def _forward_clear_consensus(self) -> bool:
        if self.final_decision_topic is not None:
            return self.final_decision == "clear"
        return (not self._obstacle_active()) and self.hazard_guidance == "clear"

    def _depth_supports_terrain(self) -> bool:
        return self.depth_terrain_votes >= 3 and self.depth_overall == "terrain"

    def _depth_supports_block(self) -> bool:
        return self.depth_block_votes >= 2 and self.depth_overall.startswith("block")

    def _depth_terrain_override_active(self) -> bool:
        if not self._obstacle_active():
            return False
        if self._front_clearance() <= self.strict_reverse_distance:
            return False
        if not self._depth_supports_terrain():
            return False
        return self.hazard_terrain_hint or self.terrain_climbable

    def _uav_terrain_sure_active(self) -> bool:
        if self._front_clearance() <= self.strict_reverse_distance:
            return False
        return (self.hazard_guidance or "").startswith("terrain_sure_front")

    def _depth_block_matches_local(self) -> bool:
        if not self._depth_supports_block():
            return False
        local = self.obstacle_action or ""
        overall = self.depth_overall
        if overall == "block_broad":
            return True
        if overall == "block_left":
            return local.endswith("left")
        if overall == "block_right":
            return local.endswith("right")
        if overall == "block_center":
            return local in {"turn_left", "turn_right", "caution_left", "caution_right"}
        return False

    def _hazard_supports_block(self) -> bool:
        # UAV-derived hazard guidance is intentionally terrain-only.
        # Local object/block obstacles must come from the UGV's own sensing.
        return False

    def _near_goal_broad_block_active(self, remaining: float | None) -> bool:
        if remaining is None:
            return False
        if remaining > self.near_goal_commit_radius:
            return False
        return self.depth_broad_block_votes >= 2 and self._hazard_supports_block()

    def _perception_priority_mode(self, remaining: float | None) -> str:
        if self._uav_terrain_sure_active():
            return "terrain"
        if self._near_goal_broad_block_active(remaining):
            return "block"
        if self._depth_supports_block():
            return "block"
        if self._depth_supports_terrain():
            return "terrain"
        return "local"

    def _majority_decision(self) -> tuple[str | None, int, int]:
        terrain_votes = 0
        block_votes = 0

        # Local detector always contributes one vote, which lets the other two
        # sources outvote it in terrain-heavy scenes without ignoring it fully.
        if self._obstacle_active():
            block_votes += 1
        else:
            terrain_votes += 1

        if self.hazard_terrain_hint:
            terrain_votes += 1
        elif self._hazard_supports_block():
            block_votes += 1

        if self._depth_supports_terrain():
            terrain_votes += 1
        elif self._depth_supports_block():
            block_votes += 1

        if terrain_votes >= 2 and terrain_votes > block_votes:
            return ("terrain", terrain_votes, block_votes)
        if block_votes >= 2 and block_votes > terrain_votes:
            return ("block", terrain_votes, block_votes)
        return (None, terrain_votes, block_votes)

    def _terrain_speed_multiplier(self) -> float:
        if not self.terrain_climbable:
            return 1.0
        if self.terrain_extent_x >= self.terrain_speedup_extent_high_m:
            return self.terrain_speedup_high_multiplier
        if self.terrain_extent_x >= self.terrain_speedup_extent_mid_m:
            return self.terrain_speedup_mid_multiplier
        return 1.0

    def _record_remaining(self, now: float, remaining: float):
        self.remaining_history.append((now, float(remaining)))
        while self.remaining_history and (now - self.remaining_history[0][0]) > 8.0:
            self.remaining_history.popleft()

    def _record_motion(self, now: float, current_xy: tuple[float, float], heading: float):
        self.motion_history.append((now, float(current_xy[0]), float(current_xy[1]), float(heading)))
        while self.motion_history and (now - self.motion_history[0][0]) > 8.0:
            self.motion_history.popleft()

    def _displacement_since(self, now: float, window_seconds: float) -> float | None:
        if len(self.motion_history) < 2:
            return None
        reference = None
        latest = self.motion_history[-1]
        for sample in self.motion_history:
            if (now - sample[0]) >= window_seconds:
                reference = sample
                break
        if reference is None:
            return None
        return math.hypot(latest[1] - reference[1], latest[2] - reference[2])

    def _heading_change_since(self, now: float, window_seconds: float) -> float | None:
        if len(self.motion_history) < 2:
            return None
        samples = [sample for sample in self.motion_history if (now - sample[0]) <= window_seconds]
        if len(samples) < 2:
            return None
        total = 0.0
        for prev, curr in zip(samples[:-1], samples[1:]):
            total += abs(wrap_angle(curr[3] - prev[3]))
        return total

    def _choose_avoid_direction(self) -> str:
        if self.final_decision_topic is not None:
            if self.final_decision.endswith("left"):
                return "left"
            if self.final_decision.endswith("right"):
                return "right"
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
        if state != "planned_path":
            self.path_plan_direction = None
            self.path_plan_phase = None
            self.path_plan_heading_target = None
            self.path_plan_start_xy = None
            self.path_plan_distance_target = 0.0

    def _lidar_only_mode(self) -> bool:
        return (
            self.final_decision_topic is None
            and not self.use_depth_classification
            and not self.use_hazard_map
        )

    def _in_obstacle_escape_zone(self) -> bool:
        front = self._front_clearance()
        left = self._left_clearance()
        right = self._right_clearance()
        if front <= (self.obstacle_stop_distance + 0.25):
            return True
        side_squeeze_limit = max(0.75, self.strict_reverse_distance)
        return (
            front <= (self.obstacle_caution_distance - 0.4)
            and left <= side_squeeze_limit
            and right <= side_squeeze_limit
        )

    def _gap_guidance_available(self) -> bool:
        # Gap-following is intentionally disabled for now. Keep receiving and
        # logging the gap profile for diagnostics, but do not let it drive the
        # UGV decision path.
        return False

    def _lidar_priority_mode(self) -> str:
        if self._in_obstacle_escape_zone():
            return "escape"
        if self._gap_guidance_available():
            return "pass_gap"
        if self.use_lidar_path_planning:
            return "path_plan"
        return "avoid"

    def _log_lidar_priority_mode(self, mode: str):
        if mode == self.last_lidar_priority_mode:
            return
        self.get_logger().info(
            "Lidar decision: "
            f"mode={mode} front={self._front_clearance():.2f} "
            f"left={self._left_clearance():.2f} right={self._right_clearance():.2f} "
            f"gap_width={self.gap_width:.2f} gap_center={self.gap_center_y:.2f} "
            f"gap_passable={'yes' if self.gap_passable else 'no'}"
        )
        self.last_lidar_priority_mode = mode

    def _enter_path_plan(self, now: float, remaining: float, reason: str):
        direction = self._choose_avoid_direction()
        sign = 1.0 if direction == "left" else -1.0
        front = self._front_clearance()
        side_clearance = self._left_clearance() if direction == "left" else self._right_clearance()
        current_heading = self._current_heading()
        current_xy = self._current_xy()
        if current_heading is None or current_xy is None:
            self._enter_state("go_to_goal")
            return

        if front <= (self.obstacle_stop_distance + 0.2):
            turn_angle = math.radians(55.0)
        elif front <= (self.obstacle_stop_distance + 0.9):
            turn_angle = math.radians(40.0)
        else:
            turn_angle = math.radians(30.0)

        self.path_plan_phase = "turn"
        self.path_plan_direction = direction
        self.path_plan_heading_target = wrap_angle(current_heading + sign * turn_angle)
        self.path_plan_start_xy = (float(current_xy[0]), float(current_xy[1]))
        self.path_plan_distance_target = clamp(side_clearance - 1.2, 1.2, 3.0)
        self._enter_state("planned_path")
        self.get_logger().info(
            "Lidar path plan: "
            f"reason={reason} direction={direction} front={front:.2f} "
            f"left={self._left_clearance():.2f} right={self._right_clearance():.2f} "
            f"distance_target={self.path_plan_distance_target:.2f}"
        )

    def _path_plan_command(self, now: float, remaining: float) -> tuple[float, float]:
        current_heading = self._current_heading()
        current_xy = self._current_xy()
        if current_heading is None or current_xy is None or self.path_plan_direction is None:
            self._enter_state("go_to_goal")
            return self._goal_command()

        front = self._front_clearance()
        sign = 1.0 if self.path_plan_direction == "left" else -1.0

        if self.path_plan_phase == "turn":
            heading_error = wrap_angle(self.path_plan_heading_target - current_heading)
            if abs(heading_error) <= math.radians(8.0):
                self.path_plan_phase = "drive"
                self.path_plan_start_xy = (float(current_xy[0]), float(current_xy[1]))
            else:
                angular = clamp(1.1 * heading_error, -self.max_angular_speed, self.max_angular_speed)
                linear = 0.05 if front <= self.obstacle_stop_distance else 0.12
                return (linear, angular)

        traveled = 0.0
        if self.path_plan_start_xy is not None:
            traveled = math.hypot(
                current_xy[0] - self.path_plan_start_xy[0],
                current_xy[1] - self.path_plan_start_xy[1],
            )

        if self._obstacle_active() and front <= self.obstacle_stop_distance:
            self.get_logger().info(
                "Lidar path reassess: "
                f"phase=drive front={front:.2f} obstacle={self.obstacle_action} traveled={traveled:.2f}"
            )
            self._enter_reassess(now, remaining, "path_blocked")
            return (0.0, 0.0)

        if traveled >= self.path_plan_distance_target:
            self._enter_state("go_to_goal")
            return self._goal_command()

        goal_heading = self._goal_heading()
        if goal_heading is None:
            return (0.30, sign * 0.08)
        heading_error = wrap_angle(goal_heading - current_heading)
        angular = clamp(0.45 * heading_error, -0.35, 0.35)
        return (0.42, angular)

    def _gap_follow_command(self) -> tuple[float, float]:
        goal_heading = self._goal_heading()
        current_heading = self._current_heading()
        remaining = self._distance_to_goal()
        if goal_heading is None or current_heading is None or remaining is None:
            return (0.0, 0.0)

        lookahead_x = max(self._front_clearance(), self.obstacle_caution_distance, 1.0)
        gap_heading_bias = math.atan2(self.gap_center_y, lookahead_x)
        target_heading = wrap_angle(goal_heading + clamp(gap_heading_bias, -0.45, 0.45))
        heading_error = wrap_angle(target_heading - current_heading)
        if abs(heading_error) < self.heading_deadband:
            heading_error = 0.0

        angular_z = clamp(
            0.85 * self.cmd_angular_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        linear_x = max(0.32, min(self._goal_speed(remaining, heading_error), self.max_linear_speed))
        return (linear_x, angular_z)

    def _enter_avoid(self, now: float):
        heading = self._current_heading()
        self.avoid_direction = self._choose_avoid_direction()
        self.avoid_start_heading = heading
        self.avoid_start_time = float(now)
        remaining = self._distance_to_goal()
        self.avoid_start_remaining = None if remaining is None else float(remaining)
        self._enter_state("avoid")
        self.get_logger().info(
            f"Avoiding obstacle: direction={self.avoid_direction} front={self._front_clearance():.2f}"
        )

    def _enter_reassess(self, now: float, remaining: float, reason: str):
        pause_seconds = self.loop_reassess_pause_seconds if reason == "loop_guard" else self.reassess_pause_seconds
        self._enter_state("reassess", now + pause_seconds)
        self.remaining_history.clear()
        self.remaining_history.append((now, float(remaining)))
        self.clear_path_since = None
        self.clear_path_start_remaining = None
        self.clear_path_best_remaining = None
        self.clear_path_last_progress_time = None
        self.terrain_commit_start_remaining = None
        self.terrain_commit_best_remaining = None
        self.terrain_commit_last_progress_time = None
        self.forward_clear_since = None
        self.forward_clear_start_remaining = None
        self.get_logger().info(
            f"Reassessing navigation: reason={reason} remaining={remaining:.3f}"
        )

    def _enter_commit_forward(self, now: float, remaining: float):
        current_xy = self._current_xy()
        self.commit_start_xy = None if current_xy is None else (float(current_xy[0]), float(current_xy[1]))
        self.avoid_direction = None
        self.avoid_start_heading = None
        self.clear_path_since = None
        self.clear_path_start_remaining = None
        self.clear_path_best_remaining = None
        self.clear_path_last_progress_time = None
        self.terrain_commit_start_remaining = None
        self.terrain_commit_best_remaining = None
        self.terrain_commit_last_progress_time = None
        self.forward_clear_since = None
        self.forward_clear_start_remaining = None
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
        self.stuck_cooldown_until = now + self.stuck_reverse_seconds + self.stuck_cooldown_seconds
        self._reset_self_stuck_votes()
        self.clear_path_since = None
        self.clear_path_start_remaining = None
        self.clear_path_best_remaining = None
        self.clear_path_last_progress_time = None
        self.terrain_commit_start_remaining = None
        self.terrain_commit_best_remaining = None
        self.terrain_commit_last_progress_time = None
        self.forward_clear_since = None
        self.forward_clear_start_remaining = None
        current_xy = self._current_xy()
        self.reverse_start_xy = None if current_xy is None else (float(current_xy[0]), float(current_xy[1]))
        self.reverse_failed_escape = False
        self.escape_turn_direction = None
        self.escape_turn_start_heading = None
        self.reverse_events.append(float(now))
        self.remaining_history.clear()
        self.remaining_history.append((now, float(remaining)))
        self.get_logger().info(
            "Stuck recovery: "
            f"reason={reason} remaining={remaining:.3f} "
            f"reverse_speed={self.stuck_reverse_speed:.2f} duration={self.stuck_reverse_seconds:.2f}"
        )

    def _enter_reverse_pause(self, now: float, remaining: float, reverse_failed_escape: bool):
        self._enter_state("reverse_pause", now + self.reverse_pause_seconds)
        self._reset_self_stuck_votes()
        self.reverse_failed_escape = bool(reverse_failed_escape)
        self.remaining_history.clear()
        self.remaining_history.append((now, float(remaining)))
        self.get_logger().info(
            "Reverse pause: "
            f"remaining={remaining:.3f} escape_next={self.reverse_failed_escape}"
        )

    def _enter_escape_turn(self, now: float):
        self._reset_self_stuck_votes()
        self.escape_turn_direction = "left" if self._left_clearance() >= self._right_clearance() else "right"
        self.escape_turn_start_heading = self._current_heading()
        self._enter_state("escape_turn", now + self.escape_turn_timeout_seconds)
        self.get_logger().info(
            f"Escape turn: direction={self.escape_turn_direction} left={self._left_clearance():.2f} right={self._right_clearance():.2f}"
        )

    def _enter_escape_drive(self, now: float):
        self._enter_state("escape_drive", now + self.escape_drive_seconds)
        self._reset_self_stuck_votes()
        self.get_logger().info(
            f"Escape drive: speed={self.max_linear_speed:.2f} duration={self.escape_drive_seconds:.2f}"
        )

    def _reverse_distance_traveled(self) -> float:
        current_xy = self._current_xy()
        if self.reverse_start_xy is None or current_xy is None:
            return 0.0
        return math.hypot(current_xy[0] - self.reverse_start_xy[0], current_xy[1] - self.reverse_start_xy[1])

    def _reset_self_stuck_votes(self):
        self.zero_motion_stuck_votes = 0
        self.circling_stuck_votes = 0

    def _enter_stall(self):
        if self.state not in {"reverse", "reverse_pause", "escape_turn", "escape_drive", "recover"}:
            self._enter_state("stall")

    def _confirm_self_stuck(self, kind: str, details: str) -> bool:
        if kind == "zero_motion":
            self.zero_motion_stuck_votes += 1
            self.circling_stuck_votes = 0
            votes = self.zero_motion_stuck_votes
        else:
            self.circling_stuck_votes += 1
            self.zero_motion_stuck_votes = 0
            votes = self.circling_stuck_votes

        self.get_logger().info(
            f"UGV self-stuck check: kind={kind} votes={votes}/{self.self_stuck_confirm_cycles} {details}"
        )
        if votes < self.self_stuck_confirm_cycles:
            self._enter_stall()
        return votes >= self.self_stuck_confirm_cycles

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
        if self.state in {"reverse", "reverse_pause", "recover", "reassess", "commit_forward", "escape_turn", "escape_drive", "planned_path"} or now < self.state_until:
            return False
        if now < self.stuck_cooldown_until:
            return False
        if self.last_command_linear_x < self.stuck_min_command_speed:
            return False
        if remaining <= self.near_goal_commit_radius:
            return False
        displacement = self._displacement_since(now, self.zero_motion_reverse_timeout_seconds)
        if displacement is None:
            return False
        if displacement > self.zero_motion_reverse_distance:
            self.zero_motion_stuck_votes = 0
            return False
        return self._confirm_self_stuck(
            "zero_motion",
            f"displacement={displacement:.3f} window={self.zero_motion_reverse_timeout_seconds:.1f}s remaining={remaining:.3f}",
        )

    def _should_circle_reverse(self, now: float, remaining: float | None) -> bool:
        if remaining is None:
            return False
        if self.state in {"reverse", "reverse_pause", "recover", "reassess", "commit_forward", "escape_turn", "escape_drive", "planned_path"} or now < self.state_until:
            return False
        if now < self.stuck_cooldown_until:
            return False
        if self.last_command_linear_x < self.stuck_min_command_speed:
            return False
        if remaining <= self.near_goal_commit_radius:
            return False
        displacement = self._displacement_since(now, self.circling_reverse_timeout_seconds)
        heading_change = self._heading_change_since(now, self.circling_reverse_timeout_seconds)
        if displacement is None or heading_change is None:
            return False
        if not (
            displacement <= self.circling_reverse_distance
            and heading_change >= self.circling_heading_change_radians
        ):
            self.circling_stuck_votes = 0
            return False
        return self._confirm_self_stuck(
            "circling",
            f"displacement={displacement:.3f} heading_change={heading_change:.3f} remaining={remaining:.3f}",
        )

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

        # Count repeated control cycles with a near-contact obstacle ahead.
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
        linear = clamp(self.cmd_linear_gain * remaining, self.min_linear_speed, self.max_linear_speed)
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

        speed_multiplier = self._terrain_speed_multiplier()
        if remaining > max(self.goal_tolerance + 1.0, 3.0) and speed_multiplier > 1.0:
            linear = min(linear * speed_multiplier, self.max_linear_speed * speed_multiplier)
        return max(0.0, linear)

    def _goal_arc_bias(self) -> float:
        """Return a smooth steering bias toward the more open lidar side.

        Positive means bias left, negative means bias right. The bias grows as
        the forward clearance shrinks or the side-clearance imbalance grows,
        which makes the Husky prefer rounder obstacle bypasses instead of
        insisting on a straight line to the goal.
        """
        front = self._front_clearance()
        left = self._left_clearance()
        right = self._right_clearance()

        if not math.isfinite(front) or not math.isfinite(left) or not math.isfinite(right):
            return 0.0

        side_gap = clamp(left - right, -6.0, 6.0)
        if abs(side_gap) < 0.15 and front > 6.0:
            return 0.0

        # Stronger bias when the front is getting tighter.
        front_factor = 1.0 - clamp((front - 1.0) / 5.0, 0.0, 1.0)
        side_factor = clamp(abs(side_gap) / 3.0, 0.0, 1.0)
        strength = max(front_factor, 0.65 * side_factor)
        bias = clamp((side_gap / 3.0) * (0.55 * strength), -0.55, 0.55)
        return bias

    def _terrain_commit_active(self, remaining: float) -> bool:
        if self.final_decision_topic is not None:
            return self.final_decision == "terrain" and remaining > self.goal_tolerance
        has_terrain_signal = self.terrain_climbable or self.hazard_terrain_hint or self._depth_supports_terrain()
        if not has_terrain_signal:
            return False
        if (
            not self.hazard_terrain_hint
            and not self._depth_supports_terrain()
            and self.terrain_extent_x < self.terrain_speedup_extent_mid_m
        ):
            return False
        return remaining > self.goal_tolerance

    def _enter_terrain_commit(self, now: float, remaining: float):
        self._enter_state("terrain_commit")
        self.terrain_commit_start_remaining = float(remaining)
        self.terrain_commit_best_remaining = float(remaining)
        self.terrain_commit_last_progress_time = now
        self.avoid_direction = None
        self.avoid_start_heading = None
        self.get_logger().info(
            "Terrain commit: "
            f"front={self._front_clearance():.2f} terrain_extent={self.terrain_extent_x:.2f}"
        )

    def _terrain_commit_command(self):
        goal_heading = self._goal_heading()
        current_heading = self._current_heading()
        if goal_heading is None or current_heading is None:
            return (self.terrain_commit_linear_speed, 0.0)
        heading_error = wrap_angle(goal_heading - current_heading)
        angular_z = clamp(
            0.35 * self.cmd_angular_gain * heading_error,
            -self.terrain_commit_max_angular_speed,
            self.terrain_commit_max_angular_speed,
        )
        linear_x = max(self.terrain_commit_linear_speed, self._goal_speed(max(self._distance_to_goal() or 0.0, 0.0), heading_error))
        return (linear_x, angular_z)

    def _commit_distance_traveled(self) -> float:
        current_xy = self._current_xy()
        if self.commit_start_xy is None or current_xy is None:
            return 0.0
        return math.hypot(current_xy[0] - self.commit_start_xy[0], current_xy[1] - self.commit_start_xy[1])

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

        goal_bias = 0.0 if self.use_lidar_straight_approach else self._goal_arc_bias()
        biased_goal_heading = wrap_angle(goal_heading + goal_bias)
        heading_error = wrap_angle(biased_goal_heading - current_heading)
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
        linear_x = min(self.bootstrap_linear_speed, self.goal_align_linear_speed)
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

        avoid_elapsed = max(0.0, now - self.avoid_start_time)
        avoid_progress = 0.0
        if self.avoid_start_remaining is not None:
            avoid_progress = self.avoid_start_remaining - float(remaining)

        if (
            not self._gap_guidance_available()
            and front <= self.obstacle_caution_distance
            and avoid_elapsed >= 1.8
            and avoid_progress < 0.20
        ):
            self._enter_reverse(now, remaining, "avoid_stuck_no_gap")
            return (self.stuck_reverse_speed, 0.0)

        # Once the front clears, keep a short straight escape move before
        # resuming goal-seeking. Keep that follow-through only while the path
        # stays clear enough and the robot is actually improving its goal distance.
        if not self._obstacle_active() and front > (self.obstacle_stop_distance + 0.5):
            if self.clear_path_since is None:
                self.clear_path_since = now
                self.clear_path_start_remaining = float(remaining)
                self.clear_path_best_remaining = float(remaining)
                self.clear_path_last_progress_time = now
            else:
                if (
                    self.clear_path_best_remaining is None
                    or remaining < (self.clear_path_best_remaining - 0.02)
                ):
                    self.clear_path_best_remaining = float(remaining)
                    self.clear_path_last_progress_time = now

                progress = 0.0
                if self.clear_path_start_remaining is not None:
                    progress = self.clear_path_start_remaining - remaining

                if front <= self.obstacle_stop_distance:
                    self.clear_path_since = None
                    self.clear_path_start_remaining = None
                    self.clear_path_best_remaining = None
                    self.clear_path_last_progress_time = None
                elif progress >= self.post_avoid_min_progress_distance:
                    self.avoid_direction = None
                    self.avoid_start_heading = None
                    self.avoid_start_remaining = None
                    self.clear_path_since = None
                    self.clear_path_start_remaining = None
                    self.clear_path_best_remaining = None
                    self.clear_path_last_progress_time = None
                    self._enter_state("go_to_goal")
                    return self._goal_command()
                elif (
                    self.clear_path_last_progress_time is not None
                    and (now - self.clear_path_last_progress_time) >= self.post_avoid_progress_timeout_seconds
                ):
                    self.avoid_direction = None
                    self.avoid_start_heading = None
                    self.avoid_start_remaining = None
                    self.clear_path_since = None
                    self.clear_path_start_remaining = None
                    self.clear_path_best_remaining = None
                    self.clear_path_last_progress_time = None
                    self._enter_state("go_to_goal")
                    return self._goal_command()
            return (self.post_avoid_forward_speed, 0.0)
        self.clear_path_since = None
        self.clear_path_start_remaining = None
        self.clear_path_best_remaining = None
        self.clear_path_last_progress_time = None

        if turned >= self.turn_limit_radians:
            if not self._obstacle_active():
                self.avoid_direction = None
                self.avoid_start_heading = None
                self.avoid_start_remaining = None
                self.clear_path_since = None
                self.clear_path_start_remaining = None
                self.clear_path_best_remaining = None
                self.clear_path_last_progress_time = None
                self._enter_state("go_to_goal")
                return self._goal_command()
            self._enter_reverse(now, remaining, "avoid_turn_limit")
            return (self.stuck_reverse_speed, 0.0)

        self.clear_path_since = None
        self.clear_path_start_remaining = None
        self.clear_path_best_remaining = None
        self.clear_path_last_progress_time = None

        if front <= self.obstacle_stop_distance:
            linear_x = 0.06
        elif front <= (self.obstacle_stop_distance + 0.8):
            linear_x = 0.16
        else:
            linear_x = 0.28
        return (linear_x, angular_z)

    def step(self):
        self.publish_state()
        current_xy = self._current_xy()
        current_heading = self._current_heading()
        if current_xy is None or current_heading is None:
            return
        now = time.monotonic()
        self._record_motion(now, current_xy, current_heading)

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

        self._record_remaining(now, remaining)
        if self.initial_remaining is None:
            self.initial_remaining = float(remaining)

        if remaining <= self.goal_tolerance:
            goal = self._current_goal()
            altitude = self._current_altitude()
            if goal is not None:
                self.get_logger().info(
                    "Arrival triggered: "
                    f"pose=({current_xy[0]:.3f}, {current_xy[1]:.3f}) "
                    f"z={altitude:.3f} goal=({goal[0]:.3f}, {goal[1]:.3f}) "
                    f"remaining={remaining:.3f} tol={self.goal_tolerance:.3f}"
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
                    f"z={altitude:.3f} goal=({goal[0]:.3f}, {goal[1]:.3f}) "
                    f"remaining={remaining:.3f} state={self._state_label()}"
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
            reverse_failed = self._reverse_distance_traveled() <= self.zero_motion_reverse_distance
            self.remaining_history.clear()
            self.remaining_history.append((now, float(remaining)))
            self._enter_reverse_pause(now, remaining, reverse_failed)
            self.publish_cmd(0.0, 0.0)
            return

        if self.state == "reverse_pause":
            if now < self.state_until:
                self.publish_cmd(0.0, 0.0)
                return
            if self.reverse_failed_escape:
                self._enter_escape_turn(now)
            else:
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
            self.clear_path_start_remaining = None
            self.clear_path_best_remaining = None
            self.clear_path_last_progress_time = None
            self._enter_state("go_to_goal")

        if self.state == "escape_turn":
            sign = 1.0 if self.escape_turn_direction == "left" else -1.0
            if self.escape_turn_start_heading is None:
                self.escape_turn_start_heading = current_heading
            turned = abs(wrap_angle(current_heading - self.escape_turn_start_heading))
            if turned >= self.escape_turn_radians or now >= self.state_until:
                self._enter_escape_drive(now)
                self.publish_cmd(self.max_linear_speed, 0.0)
                return
            self.publish_cmd(0.0, sign * self.max_angular_speed)
            return

        if self.state == "escape_drive":
            if now < self.state_until:
                self.publish_cmd(self.max_linear_speed, 0.0)
                return
            self.remaining_history.clear()
            self.remaining_history.append((now, float(remaining)))
            self.post_recover_until = now + self.post_recover_commit_cooldown_seconds
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

        if self.state == "terrain_commit":
            front = self._front_clearance()
            if self._obstacle_active() and front <= self.strict_reverse_distance and not self.terrain_climbable:
                self._enter_reverse(now, remaining, "terrain_commit_front_blocked_hard")
                self.publish_cmd(self.stuck_reverse_speed, 0.0)
                return

            if not self._terrain_commit_active(remaining):
                self.terrain_commit_start_remaining = None
                self.terrain_commit_best_remaining = None
                self.terrain_commit_last_progress_time = None
                self.remaining_history.clear()
                self.remaining_history.append((now, float(remaining)))
                self._enter_state("go_to_goal")
            else:
                if (
                    self.terrain_commit_best_remaining is None
                    or remaining < (self.terrain_commit_best_remaining - 0.02)
                ):
                    self.terrain_commit_best_remaining = float(remaining)
                    self.terrain_commit_last_progress_time = now

                progress = 0.0
                if self.terrain_commit_start_remaining is not None:
                    progress = self.terrain_commit_start_remaining - remaining

                if (
                    progress >= self.terrain_commit_min_progress_distance
                    and self.terrain_commit_last_progress_time is not None
                ):
                    self.terrain_commit_start_remaining = float(remaining)
                    self.terrain_commit_last_progress_time = now
                elif (
                    self.terrain_commit_last_progress_time is not None
                    and (now - self.terrain_commit_last_progress_time) >= self.terrain_commit_progress_timeout_seconds
                ):
                    self.get_logger().info(
                        "Terrain commit ended: progress stalled "
                        f"terrain_extent={self.terrain_extent_x:.2f} remaining={remaining:.3f}"
                    )
                    self.terrain_commit_start_remaining = None
                    self.terrain_commit_best_remaining = None
                    self.terrain_commit_last_progress_time = None
                    self.remaining_history.clear()
                    self.remaining_history.append((now, float(remaining)))
                    self._enter_state("go_to_goal")
                else:
                    self.publish_cmd(*self._terrain_commit_command())
                    return

        if self.state == "planned_path":
            self.publish_cmd(*self._path_plan_command(now, remaining))
            return

        hard_front_block = self._obstacle_active() and self._front_clearance() <= self.obstacle_stop_distance
        near_goal_commit = remaining <= self.near_goal_commit_radius
        near_goal_broad_block = self._near_goal_broad_block_active(remaining)
        depth_terrain_override = self._depth_terrain_override_active()
        uav_terrain_sure = self._uav_terrain_sure_active()
        perception_mode = self._perception_priority_mode(remaining)
        terrain_preferred = perception_mode == "terrain"
        block_preferred = perception_mode == "block"

        if (
            (uav_terrain_sure or terrain_preferred or depth_terrain_override)
            and self.state == "avoid"
            and not hard_front_block
            and not near_goal_broad_block
        ):
            self.avoid_direction = None
            self.avoid_start_heading = None
            self.clear_path_since = None
            self.clear_path_start_remaining = None
            self.clear_path_best_remaining = None
            self.clear_path_last_progress_time = None
            self._enter_state("go_to_goal")

        if uav_terrain_sure and (now - self.last_depth_override_log) >= 2.0:
            self.get_logger().info(
                "UAV terrain sure override: "
                f"local={self.obstacle_action} front={self._front_clearance():.2f} "
                f"hazard={self.hazard_guidance}"
            )
            self.last_depth_override_log = now

        if depth_terrain_override and (now - self.last_depth_override_log) >= 2.0:
            self.get_logger().info(
                "Depth terrain override: "
                f"local={self.obstacle_action} front={self._front_clearance():.2f} "
                f"hazard={self.hazard_guidance} depth={self.depth_overall}"
            )
            self.last_depth_override_log = now

        if (
            self.state not in {"avoid", "reverse", "recover", "reassess", "commit_forward", "terrain_commit"}
            and not near_goal_broad_block
            and (uav_terrain_sure or terrain_preferred or self._terrain_commit_active(remaining))
        ):
            self._enter_terrain_commit(now, remaining)
            self.publish_cmd(*self._terrain_commit_command())
            return

        if self._should_strict_reverse(now, remaining):
            self._enter_reverse(now, remaining, "front_blocked_hard")
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        if self._should_circle_reverse(now, remaining):
            self._enter_reverse(now, remaining, "circling_stalled")
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        if (
            self.state == "terrain_commit"
            and not near_goal_broad_block
            and (uav_terrain_sure or terrain_preferred)
            and not hard_front_block
        ):
            self.publish_cmd(*self._terrain_commit_command())
            return

        if self._obstacle_active() and self.state != "avoid":
            if self._lidar_only_mode():
                mode = self._lidar_priority_mode()
                self._log_lidar_priority_mode(mode)
                if mode == "escape":
                    self._enter_avoid(now)
                    self.publish_cmd(*self._avoid_command(now, remaining))
                    return
                if mode == "pass_gap":
                    self.publish_cmd(*self._gap_follow_command())
                    return
                if mode == "path_plan":
                    self._enter_path_plan(now, remaining, "local_obstacle")
                    self.publish_cmd(*self._path_plan_command(now, remaining))
                    return
                self._enter_avoid(now)
                self.publish_cmd(*self._avoid_command(now, remaining))
                return
            if ((uav_terrain_sure or terrain_preferred or depth_terrain_override) and not hard_front_block and not near_goal_broad_block):
                self.publish_cmd(*self._terrain_commit_command())
                return
            if (
                near_goal_broad_block
                or block_preferred
                or self._depth_block_matches_local()
            ):
                if near_goal_commit and not hard_front_block:
                    self.publish_cmd(*self._goal_command())
                    return
                self._enter_avoid(now)
                self.publish_cmd(*self._avoid_command(now, remaining))
                return
            if near_goal_commit and not hard_front_block:
                self.publish_cmd(*self._goal_command())
                return
            self._enter_avoid(now)

        if self.state == "avoid":
            if self._lidar_only_mode():
                mode = self._lidar_priority_mode()
                self._log_lidar_priority_mode(mode)
                if mode == "pass_gap":
                    self.avoid_direction = None
                    self.avoid_start_heading = None
                    self.avoid_start_remaining = None
                    self._enter_state("go_to_goal")
                    self.publish_cmd(*self._gap_follow_command())
                    return
                if mode == "path_plan" and self._front_clearance() > self.obstacle_stop_distance:
                    self.avoid_direction = None
                    self.avoid_start_heading = None
                    self.avoid_start_remaining = None
                    self._enter_path_plan(now, remaining, "avoid_exit_to_path")
                    self.publish_cmd(*self._path_plan_command(now, remaining))
                    return
            self.publish_cmd(*self._avoid_command(now, remaining))
            return

        if self._should_reverse(now, remaining):
            self._enter_reverse(now, remaining, "zero_motion_stalled")
            self.publish_cmd(self.stuck_reverse_speed, 0.0)
            return

        if self._should_reassess(now, remaining):
            if near_goal_commit and not hard_front_block:
                self.publish_cmd(*self._goal_command())
                return
            if self._forward_clear_consensus():
                self.publish_cmd(*self._goal_command())
                return
            self._enter_reassess(now, remaining, "goal_progress_stalled")
            self.publish_cmd(0.0, 0.0)
            return

        self.publish_cmd(*self._goal_command())
