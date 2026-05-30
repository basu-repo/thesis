"""Rule-based depth-image scene classification for optional hazard hints.

This module is intentionally standalone so it can be launched only when needed.
It does not change the current 07 run path unless we wire it in later.
"""

from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _clamp_int(value, lower, upper):
    return max(lower, min(value, upper))


class RuleBasedDepthClassifier(Node):
    """Classify left/center/right depth regions into open/terrain/block."""

    def __init__(
        self,
        # Constructor parameters configure ROS topics and rule thresholds.
        # Topic names connect the node to the simulation, while threshold values
        # decide when depth-image regions are treated as open, terrain, or block.
        node_name: str,  # ROS node name.
        image_topic: str,  # Input depth-image topic.
        classification_topic: str,  # Output topic for terrain/block/open label.
        center_metrics_topic: str,  # Output topic for compact center-depth metrics.
        near_depth_m: float = 4.0,  # Depth below this is treated as near.
        far_clear_depth_m: float = 10.0,  # Depth above this is treated as clear/far.
        valid_min_ratio: float = 0.20,  # Minimum valid-pixel ratio needed to trust a region.
        abrupt_edge_ratio_block: float = 0.18,  # Sudden depth-change ratio used for block detection.
        smooth_gradient_ratio_terrain: float = 0.08,  # Smooth depth-change ratio used for terrain detection.
        terrain_min_depth_m: float = 0.90,  # Minimum depth before a region can be considered terrain.
        broad_close_block_depth_m: float = 0.75,  # Very close broad center depth means blocking obstacle.
        center_anomaly_threshold_m: float = 0.12,  # Center-depth difference threshold for anomaly detection.
        region_top_ratio: float = 0.35,  # Top crop boundary as a fraction of image height.
        region_bottom_ratio: float = 0.95,  # Bottom crop boundary as a fraction of image height.
    ):
        super().__init__(node_name)
        self.image_topic = image_topic
        self.classification_topic = classification_topic
        self.center_metrics_topic = center_metrics_topic
        self.near_depth_m = float(near_depth_m)
        self.far_clear_depth_m = float(far_clear_depth_m)
        self.valid_min_ratio = float(valid_min_ratio)
        self.abrupt_edge_ratio_block = float(abrupt_edge_ratio_block)
        self.smooth_gradient_ratio_terrain = float(smooth_gradient_ratio_terrain)
        self.terrain_min_depth_m = float(terrain_min_depth_m)
        self.broad_close_block_depth_m = float(broad_close_block_depth_m)
        self.center_anomaly_threshold_m = float(center_anomaly_threshold_m)
        self.region_top_ratio = float(region_top_ratio)
        self.region_bottom_ratio = float(region_bottom_ratio)

        self.class_pub = self.create_publisher(String, self.classification_topic, 10)
        self.metrics_pub = self.create_publisher(Vector3, self.center_metrics_topic, 10)
        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)

        self.last_summary = None
        self.last_log_time = 0.0

        self.get_logger().info(
            f"Depth classifier listening on {self.image_topic}, publishing to {self.classification_topic}"
        )

    def _image_to_depth(self, msg: Image) -> np.ndarray | None:
        if msg.height <= 0 or msg.width <= 0:
            return None

        if msg.encoding == "32FC1":
            arr = np.frombuffer(msg.data, dtype=np.float32) # already in meters
        elif msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) * 0.001 # convert mm to m
        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            return None

        expected = int(msg.height) * int(msg.width)
        if arr.size < expected:
            return None
        return arr[:expected].reshape((int(msg.height), int(msg.width)))

    def _classify_region(self, region: np.ndarray) -> tuple[str, float, float, float]:
        valid = np.isfinite(region) & (region > 0.05)
        valid_ratio = float(np.mean(valid))
        if valid_ratio < self.valid_min_ratio:
            return ("unknown", 999.0, 0.0, 0.0)

        depths = region[valid]
        median_depth = float(np.median(depths))
        near_ratio = float(np.mean(depths <= self.near_depth_m))

        # Horizontal discontinuities capture short abrupt faces.
        diff_x = np.abs(np.diff(region, axis=1))
        valid_x = valid[:, 1:] & valid[:, :-1]
        abrupt_ratio = 0.0
        if np.any(valid_x):
            abrupt_ratio = float(np.mean((diff_x[valid_x]) >= 1.5))

        # Smooth depth trend toward the lower image often indicates terrain/ground.
        rows = region.shape[0]
        upper = region[: max(1, rows // 2), :]
        lower = region[max(1, rows // 2) :, :]
        upper_valid = np.isfinite(upper) & (upper > 0.05)
        lower_valid = np.isfinite(lower) & (lower > 0.05)
        upper_med = float(np.median(upper[upper_valid])) if np.any(upper_valid) else median_depth
        lower_med = float(np.median(lower[lower_valid])) if np.any(lower_valid) else median_depth
        smooth_gradient = max(0.0, lower_med - upper_med)

        if abrupt_ratio >= self.abrupt_edge_ratio_block and near_ratio >= 0.10:
            return ("block", median_depth, abrupt_ratio, smooth_gradient)
        if median_depth <= self.broad_close_block_depth_m and near_ratio >= 0.30:
            return ("block", median_depth, abrupt_ratio, smooth_gradient)
        if smooth_gradient >= self.smooth_gradient_ratio_terrain:
            return ("terrain", median_depth, abrupt_ratio, smooth_gradient)
        if median_depth >= self.far_clear_depth_m and near_ratio <= 0.05:
            return ("open", median_depth, abrupt_ratio, smooth_gradient)
        if near_ratio >= 0.20:
            return ("block", median_depth, abrupt_ratio, smooth_gradient)
        return ("terrain", median_depth, abrupt_ratio, smooth_gradient)

    def _region_consensus_override(
        self,
        left_cls: str,
        left_depth: float,
        center_cls: str,
        center_depth: float,
        center_edge: float,
        center_grad: float,
        right_cls: str,
        right_depth: float,
    ) -> tuple[str, str, str, str]:
        classes = [left_cls, center_cls, right_cls]
        depths = [left_depth, center_depth, right_depth]
        finite_depths = [depth for depth in depths if math.isfinite(depth) and depth < 900.0]
        if len(finite_depths) < 3:
            overall = center_cls
            return (left_cls, center_cls, right_cls, overall)

        depth_spread = max(finite_depths) - min(finite_depths)
        side_mean = 0.5 * (left_depth + right_depth)
        center_anomaly = abs(center_depth - side_mean)
        monotonic_side_gradient = (
            (left_depth <= center_depth <= right_depth)
            or (right_depth <= center_depth <= left_depth)
        )
        all_block = all(cls == "block" for cls in classes)
        close_uniform_band = (
            max(finite_depths) <= max(self.near_depth_m, 2.8)
            and depth_spread <= 0.45
            and center_edge < self.abrupt_edge_ratio_block
        )
        broad_close_front = max(finite_depths) <= self.broad_close_block_depth_m

        # Broad, similar near-depth across the full width looks more like terrain
        # than a localized discrete obstacle.
        if all_block and broad_close_front:
            return ("block", "block", "block", "block_broad")

        if (
            all_block
            and close_uniform_band
            and monotonic_side_gradient
            and center_anomaly <= self.center_anomaly_threshold_m
            and min(finite_depths) >= self.terrain_min_depth_m
        ):
            return ("terrain", "terrain", "terrain", "terrain")

        # A real localized block should be meaningfully closer in one region.
        left_delta = left_depth - min(center_depth, right_depth)
        center_delta = center_depth - min(left_depth, right_depth)
        right_delta = right_depth - min(left_depth, center_depth)
        localized_spike = (
            min(left_delta, center_delta, right_delta) <= -0.60
            or max(abs(left_depth - center_depth), abs(center_depth - right_depth), abs(left_depth - right_depth)) >= 0.90
        )

        if (
            all_block
            and not localized_spike
            and monotonic_side_gradient
            and center_anomaly <= self.center_anomaly_threshold_m
            and center_grad >= self.smooth_gradient_ratio_terrain * 0.5
            and min(finite_depths) >= self.terrain_min_depth_m
        ):
            return ("terrain", "terrain", "terrain", "terrain")

        overall = center_cls
        if center_cls == "unknown":
            votes = [left_cls, center_cls, right_cls]
            for candidate in ("block", "terrain", "open"):
                if votes.count(candidate) >= 2:
                    overall = candidate
                    break
        return (left_cls, center_cls, right_cls, overall)

    def _block_side_label(
        self,
        left_cls: str,
        left_depth: float,
        center_cls: str,
        center_depth: float,
        right_cls: str,
        right_depth: float,
        overall: str,
    ) -> str:
        if overall != "block":
            return overall

        block_depths = []
        if left_cls == "block" and math.isfinite(left_depth):
            block_depths.append(("left", left_depth))
        if center_cls == "block" and math.isfinite(center_depth):
            block_depths.append(("center", center_depth))
        if right_cls == "block" and math.isfinite(right_depth):
            block_depths.append(("right", right_depth))

        if not block_depths:
            return overall

        block_depths.sort(key=lambda item: item[1])
        dominant_side, dominant_depth = block_depths[0]

        # If one side is clearly nearer than the others, expose that as the block side.
        if len(block_depths) >= 2:
            second_depth = block_depths[1][1]
            if (second_depth - dominant_depth) >= 0.20:
                return f"block_{dominant_side}"

        # If all three are close but still unresolved, keep the generic block label.
        if len(block_depths) == 3:
            depth_spread = max(depth for _, depth in block_depths) - min(depth for _, depth in block_depths)
            if depth_spread <= 0.20:
                return "block_broad"

        return f"block_{dominant_side}"

    def image_cb(self, msg: Image):
        depth = self._image_to_depth(msg)
        if depth is None:
            return

        h, w = depth.shape
        top = _clamp_int(int(h * self.region_top_ratio), 0, h - 1)
        bottom = _clamp_int(int(h * self.region_bottom_ratio), top + 1, h)
        roi = depth[top:bottom, :]
        third = max(1, w // 3)

        left_region = roi[:, :third]
        center_region = roi[:, third : min(w, 2 * third)]
        right_region = roi[:, min(w, 2 * third) :]

        left_cls, left_depth, _, _ = self._classify_region(left_region)
        center_cls, center_depth, center_edge, center_grad = self._classify_region(center_region)
        right_cls, right_depth, _, _ = self._classify_region(right_region)
        left_cls, center_cls, right_cls, overall = self._region_consensus_override(
            left_cls,
            left_depth,
            center_cls,
            center_depth,
            center_edge,
            center_grad,
            right_cls,
            right_depth,
        )
        overall = self._block_side_label(
            left_cls,
            left_depth,
            center_cls,
            center_depth,
            right_cls,
            right_depth,
            overall,
        )

        summary = (
            f"left={left_cls}:{left_depth:.2f} "
            f"center={center_cls}:{center_depth:.2f} "
            f"right={right_cls}:{right_depth:.2f} "
            f"overall={overall}"
        )

        class_msg = String()
        class_msg.data = summary
        self.class_pub.publish(class_msg)

        metrics = Vector3()
        metrics.x = center_depth if math.isfinite(center_depth) else 999.0
        metrics.y = float(center_edge)
        metrics.z = float(center_grad)
        self.metrics_pub.publish(metrics)

        now = self.get_clock().now().nanoseconds / 1e9
        if summary != self.last_summary or (now - self.last_log_time) >= 2.0:
            self.get_logger().info(f"Depth classification: {summary}")
            self.last_summary = summary
            self.last_log_time = now
