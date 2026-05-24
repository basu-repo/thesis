"""Apply OMNeT-style network impairments to UAV state before Husky consumes it.

This bridge is a practical in-repo stand-in for the missing external OMNeT++
project currently referenced by 09. It delays and drops UAV state updates,
re-publishes impaired UAV odometry/world-pose topics for the learned Husky
driver, and emits OMNeT-style link metrics for logging and feature scaling.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
import time
from dataclasses import dataclass

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32
from tf2_msgs.msg import TFMessage


def extract_model_transform(msg: TFMessage, model_name: str) -> TransformStamped | None:
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
    return copy.deepcopy(selected_base_link or selected_model)


@dataclass(frozen=True)
class OmnetProfile:
    name: str
    fixed_delay_s: float
    jitter_s: float
    drop_probability: float
    base_rssi_dbm: float
    base_snir_db: float


PROFILES: dict[str, OmnetProfile] = {
    "wifi": OmnetProfile(
        name="WifiRelay",
        fixed_delay_s=0.04,
        jitter_s=0.01,
        drop_probability=0.01,
        base_rssi_dbm=-70.0,
        base_snir_db=18.0,
    ),
    "bluetooth": OmnetProfile(
        name="BluetoothRelay",
        fixed_delay_s=0.12,
        jitter_s=0.03,
        drop_probability=0.03,
        base_rssi_dbm=-85.0,
        base_snir_db=8.0,
    ),
}


class OmnetUavStateBridge(Node):
    """Delay/drop UAV state updates and publish OMNeT-style metrics."""

    def __init__(
        self,
        *,
        node_name: str = "omnet_uav_state_bridge",
        world_pose_input_topic: str,
        world_pose_output_topic: str,
        husky_model_name: str = "husky_2",
        uav1_input_odom_topic: str,
        uav2_input_odom_topic: str,
        uav1_output_odom_topic: str,
        uav2_output_odom_topic: str,
        topic_prefix: str = "/omnet",
        fixed_delay_s: float,
        jitter_s: float,
        drop_probability: float,
        base_rssi_dbm: float,
        base_snir_db: float,
        process_period_s: float = 0.01,
        publish_period_s: float = 0.10,
        metrics_period_s: float = 0.25,
    ):
        super().__init__(node_name)
        self.topic_prefix = topic_prefix.rstrip("/")
        self.husky_model_name = str(husky_model_name)
        self.fixed_delay_s = max(0.0, float(fixed_delay_s))
        self.jitter_s = max(0.0, float(jitter_s))
        self.drop_probability = max(0.0, min(float(drop_probability), 1.0))
        self.base_rssi_dbm = float(base_rssi_dbm)
        self.base_snir_db = float(base_snir_db)
        self.start_wall = time.monotonic()
        self.current_sim_time_s = 0.0

        self.latest_husky_transform: TransformStamped | None = None
        self.latest_raw_uav_transform: dict[str, TransformStamped | None] = {"uav1": None, "uav2": None}
        self.latest_delivered_uav_transform: dict[str, TransformStamped | None] = {"uav1": None, "uav2": None}
        self.latest_delivered_odom: dict[str, Odometry | None] = {"uav1": None, "uav2": None}

        self._queue: list[tuple[float, int, str, str, object]] = []
        self._seq = 0

        self.pub_world_pose = self.create_publisher(TFMessage, world_pose_output_topic, 10)
        self.pub_uav1_odom = self.create_publisher(Odometry, uav1_output_odom_topic, 10)
        self.pub_uav2_odom = self.create_publisher(Odometry, uav2_output_odom_topic, 10)
        self.pub_sim_time = self.create_publisher(Float32, f"{self.topic_prefix}/sim_time", 10)
        self.pub_link_distance = self.create_publisher(Float32, f"{self.topic_prefix}/link_distance", 10)
        self.pub_rssi = self.create_publisher(Float32, f"{self.topic_prefix}/rssi_dbm", 10)
        self.pub_snir = self.create_publisher(Float32, f"{self.topic_prefix}/snir_db", 10)
        self.pub_per = self.create_publisher(Float32, f"{self.topic_prefix}/packet_error_rate", 10)
        self.pub_radio_distance = self.create_publisher(Float32, f"{self.topic_prefix}/radio_distance", 10)

        self.create_subscription(TFMessage, world_pose_input_topic, self.world_pose_cb, 10)
        self.create_subscription(Odometry, uav1_input_odom_topic, self.uav1_odom_cb, 10)
        self.create_subscription(Odometry, uav2_input_odom_topic, self.uav2_odom_cb, 10)
        self.create_subscription(Clock, "/clock", self.clock_cb, 10)

        self.process_timer = self.create_timer(process_period_s, self.process_queue)
        self.publish_timer = self.create_timer(publish_period_s, self.publish_impaired_state)
        self.metrics_timer = self.create_timer(metrics_period_s, self.publish_metrics)

        self.get_logger().info(
            "OMNeT-style UAV state bridge enabled: "
            f"delay={self.fixed_delay_s:.3f}s jitter={self.jitter_s:.3f}s drop={self.drop_probability:.3f}"
        )

    def clock_cb(self, msg: Clock):
        self.current_sim_time_s = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9

    def _delay_seconds(self) -> float:
        if self.jitter_s <= 0.0:
            return self.fixed_delay_s
        delay = self.fixed_delay_s + random.uniform(-self.jitter_s, self.jitter_s)
        return max(0.0, delay)

    def _schedule(self, kind: str, which: str, payload: object):
        if random.uniform(0.0, 1.0) < self.drop_probability:
            return
        self._seq += 1
        deliver_at = time.monotonic() + self._delay_seconds()
        heapq.heappush(self._queue, (deliver_at, self._seq, kind, which, copy.deepcopy(payload)))

    def uav1_odom_cb(self, msg: Odometry):
        self._schedule("odom", "uav1", msg)

    def uav2_odom_cb(self, msg: Odometry):
        self._schedule("odom", "uav2", msg)

    def world_pose_cb(self, msg: TFMessage):
        husky_tf = extract_model_transform(msg, self.husky_model_name)
        if husky_tf is not None:
            self.latest_husky_transform = husky_tf
        for name in ("uav1", "uav2"):
            transform = extract_model_transform(msg, name)
            if transform is None:
                continue
            self.latest_raw_uav_transform[name] = transform
            self._schedule("transform", name, transform)

    def process_queue(self):
        now = time.monotonic()
        while self._queue and self._queue[0][0] <= now:
            _deliver_at, _seq, kind, which, payload = heapq.heappop(self._queue)
            if kind == "odom":
                self.latest_delivered_odom[which] = payload
                if which == "uav1":
                    self.pub_uav1_odom.publish(payload)
                else:
                    self.pub_uav2_odom.publish(payload)
            elif kind == "transform":
                self.latest_delivered_uav_transform[which] = payload

    def publish_impaired_state(self):
        if self.latest_husky_transform is None:
            return
        transforms = [copy.deepcopy(self.latest_husky_transform)]
        for name in ("uav1", "uav2"):
            transform = self.latest_delivered_uav_transform[name]
            if transform is not None:
                transforms.append(copy.deepcopy(transform))
        msg = TFMessage()
        msg.transforms = transforms
        self.pub_world_pose.publish(msg)

    def _publish_float(self, publisher, value: float):
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)

    def _current_link_distance(self) -> float:
        if self.latest_husky_transform is None:
            return 0.0
        hx = float(self.latest_husky_transform.transform.translation.x)
        hy = float(self.latest_husky_transform.transform.translation.y)
        hz = float(self.latest_husky_transform.transform.translation.z)
        distances = []
        for name in ("uav1", "uav2"):
            transform = self.latest_raw_uav_transform[name]
            if transform is None:
                continue
            ux = float(transform.transform.translation.x)
            uy = float(transform.transform.translation.y)
            uz = float(transform.transform.translation.z)
            distances.append(math.sqrt((ux - hx) ** 2 + (uy - hy) ** 2 + (uz - hz) ** 2))
        if not distances:
            return 0.0
        return float(sum(distances) / len(distances))

    def publish_metrics(self):
        distance = self._current_link_distance()
        rssi = self.base_rssi_dbm - (0.04 * distance)
        snir = max(-5.0, self.base_snir_db - (0.03 * distance))
        radio_distance = distance
        sim_time = self.current_sim_time_s if self.current_sim_time_s > 0.0 else (time.monotonic() - self.start_wall)

        self._publish_float(self.pub_sim_time, sim_time)
        self._publish_float(self.pub_link_distance, distance)
        self._publish_float(self.pub_rssi, rssi)
        self._publish_float(self.pub_snir, snir)
        self._publish_float(self.pub_per, self.drop_probability)
        self._publish_float(self.pub_radio_distance, radio_distance)
