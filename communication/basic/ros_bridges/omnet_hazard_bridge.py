"""Bridge UAV hazard hints through OMNeT++ using UDP.

The bridge forwards locally estimated hazard messages into the network
simulator and republishes the delayed/degraded result back into ROS 2.
"""

import socket
import threading

from rclpy.node import Node
from std_msgs.msg import String


class OmnetHazardBridge(Node):
    """ROS 2 <-> UDP bridge for hazard messages passed through OMNeT++."""

    def __init__(
        self,
        node_name: str = "omnet_hazard_bridge",
        input_topic: str = "/uav1/hazard_hint_raw",
        output_topic: str = "/uav1/hazard_hint_net",
        send_host: str = "127.0.0.1",
        send_port: int = 5001,
        listen_host: str = "127.0.0.1",
        listen_port: int = 5002,
    ):
        super().__init__(node_name)
        self.send_addr = (send_host, send_port)
        self.pub = self.create_publisher(String, output_topic, 10)
        self.create_subscription(String, input_topic, self.input_cb, 10)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind((listen_host, listen_port))
        self.recv_sock.settimeout(0.2)

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def input_cb(self, msg: String):
        try:
            self.send_sock.sendto(msg.data.encode("utf-8"), self.send_addr)
        except OSError:
            pass

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self.recv_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            msg = String()
            msg.data = data.decode("utf-8", errors="ignore")
            self.pub.publish(msg)

    def destroy_node(self):
        self._running = False
        try:
            self.recv_sock.close()
        except OSError:
            pass
        try:
            self.send_sock.close()
        except OSError:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        return super().destroy_node()
