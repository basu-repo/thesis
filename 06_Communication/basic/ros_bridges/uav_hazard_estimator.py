"""Estimate hazard hints from the UAV point cloud.

The estimator projects aerial point-cloud obstacles into the Husky's forward
corridor and publishes a compact left/center/right blocking summary that can be
consumed directly or passed through OMNeT++.
"""

import json
import math

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class UavHazardEstimator(Node):
    """Summarize aerial obstacle observations into a simple navigation hint."""

    def __init__(
        self,
        node_name: str = "uav_hazard_estimator",
        husky_odom_topic: str = "/model/husky_local/odometry",
        uav_odom_topic: str = "/model/uav1/odometry",
        uav_pointcloud_topic: str = "/world/sim_world/model/uav1/link/base_link/sensor/front_laser/scan/points",
        output_topic: str = "/uav1/hazard_hint_raw",
        update_period: float = 0.2,
        lookahead_min_x: float = 1.5,
        lookahead_max_x: float = 8.0,
        lane_half_width: float = 3.0,
        obstacle_min_z: float = -0.2,
        obstacle_max_z: float = 3.0,
        min_points_blocked: int = 12,
        source_name: str = "uav1",
    ):
        super().__init__(node_name)
        self.lookahead_min_x = lookahead_min_x
        self.lookahead_max_x = lookahead_max_x
        self.lane_half_width = lane_half_width
        self.obstacle_min_z = obstacle_min_z
        self.obstacle_max_z = obstacle_max_z
        self.min_points_blocked = min_points_blocked
        self.source_name = source_name

        self.husky_pose = None
        self.uav_pose = None
        self.latest_pointcloud = None

        self.pub = self.create_publisher(String, output_topic, 10)
        self.create_subscription(Odometry, husky_odom_topic, self.husky_odom_cb, 10)
        self.create_subscription(Odometry, uav_odom_topic, self.uav_odom_cb, 10)
        self.create_subscription(
            PointCloud2,
            uav_pointcloud_topic,
            self.pointcloud_cb,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(update_period, self.publish_hazard_hint)

    def husky_odom_cb(self, msg):
        self.husky_pose = msg.pose.pose

    def uav_odom_cb(self, msg):
        self.uav_pose = msg.pose.pose

    def pointcloud_cb(self, msg):
        self.latest_pointcloud = msg

    def publish_hazard_hint(self):
        if self.husky_pose is None or self.uav_pose is None or self.latest_pointcloud is None:
            return

        husky_pos = self.husky_pose.position
        husky_yaw = quaternion_to_yaw(
            self.husky_pose.orientation.x,
            self.husky_pose.orientation.y,
            self.husky_pose.orientation.z,
            self.husky_pose.orientation.w,
        )
        uav_pos = self.uav_pose.position
        uav_yaw = quaternion_to_yaw(
            self.uav_pose.orientation.x,
            self.uav_pose.orientation.y,
            self.uav_pose.orientation.z,
            self.uav_pose.orientation.w,
        )

        cos_uav = math.cos(uav_yaw)
        sin_uav = math.sin(uav_yaw)
        cos_h = math.cos(husky_yaw)
        sin_h = math.sin(husky_yaw)

        left_count = 0
        center_count = 0
        right_count = 0
        min_distance = self.lookahead_max_x
        obstacle_points_world = []

        # Diagnostic counters for debugging UAV point-cloud filtering.
        # These values help us understand why a UAV reports blocked=false:
        # - total_points_read: how many point-cloud points were received
        # - z_filtered_points: how many points were rejected by height filtering
        # - corridor_x_filtered_points: how many points were outside the forward lookahead range
        # - corridor_y_filtered_points: how many points were outside the lane width
        # - accepted_points: how many points remained inside the UGV forward corridor
        total_points_read = 0
        z_filtered_points = 0
        corridor_x_filtered_points = 0
        corridor_y_filtered_points = 0
        accepted_points = 0

        for x_uav, y_uav, z_uav in point_cloud2.read_points(
            self.latest_pointcloud,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            total_points_read += 1

            if z_uav < self.obstacle_min_z or z_uav > self.obstacle_max_z:
                z_filtered_points += 1
                continue

            x_world = uav_pos.x + cos_uav * x_uav - sin_uav * y_uav
            y_world = uav_pos.y + sin_uav * x_uav + cos_uav * y_uav

            dx_world = x_world - husky_pos.x
            dy_world = y_world - husky_pos.y

            x_h = cos_h * dx_world + sin_h * dy_world
            y_h = -sin_h * dx_world + cos_h * dy_world

            if x_h < self.lookahead_min_x or x_h > self.lookahead_max_x:
                corridor_x_filtered_points += 1
                continue

            if abs(y_h) > self.lane_half_width:
                corridor_y_filtered_points += 1
                continue
            # Store accepted point-cloud points in world coordinates.
            # These points are inside the UGV forward corridor and are used
            # to estimate the approximate obstacle position.
            accepted_points += 1
            obstacle_points_world.append((x_world, y_world, z_uav))

            min_distance = min(min_distance, x_h)
            if y_h < -0.8:
                right_count += 1
            elif y_h > 0.8:
                left_count += 1
            else:
                center_count += 1

        blocked = center_count >= self.min_points_blocked or (left_count + center_count + right_count) >= self.min_points_blocked * 2
        turn_bias = 0.0
        if blocked:
            turn_bias = 1.0 if left_count <= right_count else -1.0


        if obstacle_points_world:
            obstacle_world_x = sum(p[0] for p in obstacle_points_world) / len(obstacle_points_world)
            obstacle_world_y = sum(p[1] for p in obstacle_points_world) / len(obstacle_points_world)
            obstacle_world_z = sum(p[2] for p in obstacle_points_world) / len(obstacle_points_world)
        else:
            obstacle_world_x = None
            obstacle_world_y = None
            obstacle_world_z = None

        total_points = left_count + center_count + right_count
        confidence = min(1.0, total_points / max(float(self.min_points_blocked * 3), 1.0))


        # payload = {
        #     "blocked": blocked,
        #     "turn_bias": turn_bias,
        #     "left_count": left_count,
        #     "center_count": center_count,
        #     "right_count": right_count,
        #     "distance_ahead": min_distance,
        # }

        # Estimate an approximate obstacle coordinate by averaging the blocked
        # point-cloud points inside the UGV forward corridor. This coordinate is
        # later used by the fusion node to decide whether the predicted UGV path
        # overlaps with a hazard region.
        payload = {
            "source": self.source_name,
            "blocked": blocked,
            "turn_bias": turn_bias,
            "left_count": left_count,
            "center_count": center_count,
            "right_count": right_count,
            "distance_ahead": min_distance,
            "obstacle_world_x": obstacle_world_x,
            "obstacle_world_y": obstacle_world_y,
            "obstacle_world_z": obstacle_world_z,
            "confidence": confidence,
            "total_points_read": total_points_read,
            "z_filtered_points": z_filtered_points,
            "corridor_x_filtered_points": corridor_x_filtered_points,
            "corridor_y_filtered_points": corridor_y_filtered_points,
            "accepted_points": accepted_points,
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)
