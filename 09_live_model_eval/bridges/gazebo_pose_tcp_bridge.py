"""Serve live Gazebo model poses over TCP for the external OMNeT bridge."""

from __future__ import annotations

import socket
import threading
from typing import Dict
import math

from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


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


class GazeboPoseTcpBridge(Node):
    """Expose tracked model world poses as a simple TCP snapshot service."""

    def __init__(
        self,
        *,
        node_name: str = "gazebo_pose_tcp_bridge",
        world_pose_topic: str,
        tracked_models: list[str],
        host: str = "127.0.0.1",
        port: int = 5555,
    ):
        super().__init__(node_name)
        self.tracked_models = list(tracked_models)
        self.host = str(host)
        self.port = int(port)
        self.poses: Dict[str, tuple[float, float, float, float]] = {}
        self._lock = threading.Lock()
        self._running = True

        self.create_subscription(TFMessage, world_pose_topic, self.world_pose_cb, 10)

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(4)
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Serving Gazebo poses for {self.tracked_models} on tcp://{self.host}:{self.port}"
        )

    def world_pose_cb(self, msg: TFMessage):
        updated = {}
        for name in self.tracked_models:
            transform = extract_model_transform(msg, name)
            if transform is None:
                continue
            t = transform.transform.translation
            r = transform.transform.rotation
            updated[name] = (
                float(t.x),
                float(t.y),
                float(t.z),
                quaternion_to_yaw(r.x, r.y, r.z, r.w),
            )
        if updated:
            with self._lock:
                self.poses.update(updated)

    def _snapshot_line(self) -> bytes:
        with self._lock:
            items = [(name, self.poses[name]) for name in self.tracked_models if name in self.poses]
        parts = [str(len(items))]
        for name, (x, y, z, yaw) in items:
            parts.extend([name, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{yaw:.6f}"])
        return (" ".join(parts) + "\n").encode("utf-8")

    def _serve_loop(self):
        while self._running:
            try:
                conn, _addr = self._server.accept()
            except OSError:
                break
            with conn:
                conn.settimeout(0.2)
                while self._running:
                    try:
                        data = conn.recv(1024)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not data:
                        break
                    try:
                        conn.sendall(self._snapshot_line())
                    except OSError:
                        break

    def destroy_node(self):
        self._running = False
        try:
            self._server.close()
        except OSError:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        return super().destroy_node()
