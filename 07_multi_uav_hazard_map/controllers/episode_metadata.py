"""Publish episode start and goal poses for every simulated agent.

These topics make each run self-describing in recorded bags and let the
exporters reconstruct where every agent started and what target it was given.
"""

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class EpisodeMetadataPublisher(Node):
    """Continuously publish start/goal poses so late subscribers still receive them."""

    def __init__(self, world_name: str, start_goals: dict[str, dict[str, tuple[float, ...]]]):
        super().__init__("episode_metadata_publisher")
        self.world_name = world_name
        self.start_goals = start_goals
        self.pose_publishers = {}

        for entity_name in start_goals:
            self.pose_publishers[(entity_name, "start")] = self.create_publisher(
                PoseStamped, f"/episode/{entity_name}/start", 10
            )
            self.pose_publishers[(entity_name, "goal")] = self.create_publisher(
                PoseStamped, f"/episode/{entity_name}/goal", 10
            )

        self.timer = self.create_timer(1.0, self.publish_all)
        self.publish_all()

    def make_pose(self, xyz: tuple[float, ...]) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.world_name
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2]) if len(xyz) > 2 else 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def publish_all(self):
        now = self.get_clock().now().to_msg()
        for entity_name, data in self.start_goals.items():
            start_msg = self.make_pose(data["start"])
            goal_msg = self.make_pose(data["goal"])
            start_msg.header.stamp = now
            goal_msg.header.stamp = now
            self.pose_publishers[(entity_name, "start")].publish(start_msg)
            self.pose_publishers[(entity_name, "goal")].publish(goal_msg)
