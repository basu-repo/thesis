"""Bridge live OMNeT metrics from TCP into ROS 2 topics."""

from __future__ import annotations

import socket
import threading
import time

from rclpy.node import Node
from std_msgs.msg import Float32


class OmnetMetricsBridge(Node):
    """Connect to the OMNeT metrics server and republish its live values."""

    def __init__(
        self,
        *,
        node_name: str = "omnet_metrics_bridge",
        host: str = "127.0.0.1",
        port: int = 5556,
        reconnect_seconds: float = 1.0,
        topic_prefix: str = "/omnet",
    ):
        super().__init__(node_name)
        self.host = str(host)
        self.port = int(port)
        self.reconnect_seconds = float(reconnect_seconds)
        self.topic_prefix = topic_prefix.rstrip("/")
        self._running = True

        self.pub_sim_time = self.create_publisher(Float32, f"{self.topic_prefix}/sim_time", 10)
        self.pub_link_distance = self.create_publisher(Float32, f"{self.topic_prefix}/link_distance", 10)
        self.pub_rssi = self.create_publisher(Float32, f"{self.topic_prefix}/rssi_dbm", 10)
        self.pub_snir = self.create_publisher(Float32, f"{self.topic_prefix}/snir_db", 10)
        self.pub_per = self.create_publisher(Float32, f"{self.topic_prefix}/packet_error_rate", 10)
        self.pub_radio_distance = self.create_publisher(Float32, f"{self.topic_prefix}/radio_distance", 10)

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(f"Connecting to OMNeT metrics at tcp://{self.host}:{self.port}")

    def _publish_value(self, pub, value: float):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def _handle_line(self, line: str):
        parts = line.strip().split()
        if len(parts) < 6:
            return
        try:
            sim_time, distance, rssi, snir, per, radio_distance = map(float, parts[:6])
        except ValueError:
            return
        self._publish_value(self.pub_sim_time, sim_time)
        self._publish_value(self.pub_link_distance, distance)
        self._publish_value(self.pub_rssi, rssi)
        self._publish_value(self.pub_snir, snir)
        self._publish_value(self.pub_per, per)
        self._publish_value(self.pub_radio_distance, radio_distance)

    def _recv_loop(self):
        while self._running:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            try:
                sock.connect((self.host, self.port))
                sock.settimeout(0.5)
                buffer = b""
                while self._running:
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not data:
                        break
                    buffer += data
                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)
                        self._handle_line(raw.decode("utf-8", errors="ignore"))
            except OSError:
                pass
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
            if self._running:
                time.sleep(self.reconnect_seconds)

    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        return super().destroy_node()
