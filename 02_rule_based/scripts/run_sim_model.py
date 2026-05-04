# # """Launch the rule-based data-collection simulation pipeline.

# # This runner starts Gazebo, spawns the Husky/UAV agents, publishes episode
# # metadata, records a rosbag, and runs the rule-based Husky/UAV controllers.

# # Purpose of this runner:
# # - Generate training/evaluation data for trajectory prediction models.
# # - Record synchronized UGV, UAV, lidar, pointcloud, goal, command, and controller-state topics.
# # - Keep the environment and topic structure compatible with the hybrid dataset exporter.

# # Dataset target:
# # The generated bags are later exported into frames.jsonl and .npy assets by:

# #     03_dataset/exporters/export_hybrid_maneuver_dataset.py

# # Main recorded modalities:
# # - past Husky trajectory and command history
# # - Husky local lidar / front pointcloud
# # - UAV odometry and UAV forward pointcloud
# # - second Husky context
# # - inter-agent relations through exported graph features
# # - goal/start metadata
# # - controller labels such as go_to_goal, avoid_left, avoid_right, arrived
# # """

# # import datetime
# # import math
# # import os
# # import signal
# # import subprocess
# # import sys
# # import time
# # from contextlib import suppress
# # from pathlib import Path

# # SCRIPT_DIR = Path(__file__).resolve().parent
# # RULE_BASED_ROOT = SCRIPT_DIR.parent
# # if str(RULE_BASED_ROOT) not in sys.path:
# #     sys.path.insert(0, str(RULE_BASED_ROOT))

# # import rclpy
# # from rclpy.executors import MultiThreadedExecutor

# # from controllers.episode_metadata import EpisodeMetadataPublisher
# # from controllers.husky_model_driver import ModelHuskyDriver
# # from controllers.obstacle_detection import ObstacleDetectionNode
# # from controllers.uav_follower import UavFollower
# # from project_paths import MODELS_DIR, OMNET_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH


# # # ---------------------------------------------------------------------------
# # # Global paths and runtime logging
# # # ---------------------------------------------------------------------------

# # WORLD = str(WORLD_SDF_PATH)
# # WORLD_NAME = "baylands"

# # MODEL_PATH = str(MODELS_DIR)
# # OMNET_BIN = OMNET_DIR / "onmetpp"
# # OMNET_CONFIG = "WifiRelay"

# # RUN_START_DT = datetime.datetime.now()

# # LOG_DIR = Path.home() / "Documents/Thesis/03_dataset/logs"
# # LOG_DIR.mkdir(parents=True, exist_ok=True)

# # RUN_LOG_PATH = LOG_DIR / f"rule_based_dataset_{RUN_START_DT.strftime('%Y%m%d_%H%M%S')}.log"

# # TEE_PROCESS = None
# # ORIGINAL_STDOUT_FD = None
# # ORIGINAL_STDERR_FD = None


# # def setup_terminal_tee(log_path: Path):
# #     global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD

# #     if TEE_PROCESS is not None:
# #         return

# #     ORIGINAL_STDOUT_FD = os.dup(sys.__stdout__.fileno())
# #     ORIGINAL_STDERR_FD = os.dup(sys.__stderr__.fileno())

# #     TEE_PROCESS = subprocess.Popen(
# #         ["tee", "-a", str(log_path)],
# #         stdin=subprocess.PIPE,
# #         stdout=ORIGINAL_STDOUT_FD,
# #         stderr=ORIGINAL_STDERR_FD,
# #         bufsize=0,
# #     )

# #     os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stdout__.fileno())
# #     os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stderr__.fileno())


# # def close_terminal_tee():
# #     global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD

# #     if TEE_PROCESS is None:
# #         return

# #     with suppress(Exception):
# #         sys.stdout.flush()
# #     with suppress(Exception):
# #         sys.stderr.flush()

# #     if ORIGINAL_STDOUT_FD is not None:
# #         with suppress(Exception):
# #             os.dup2(ORIGINAL_STDOUT_FD, sys.__stdout__.fileno())

# #     if ORIGINAL_STDERR_FD is not None:
# #         with suppress(Exception):
# #             os.dup2(ORIGINAL_STDERR_FD, sys.__stderr__.fileno())

# #     with suppress(Exception):
# #         if TEE_PROCESS.stdin is not None:
# #             TEE_PROCESS.stdin.close()

# #     with suppress(Exception):
# #         TEE_PROCESS.wait(timeout=2.0)

# #     if ORIGINAL_STDOUT_FD is not None:
# #         with suppress(Exception):
# #             os.close(ORIGINAL_STDOUT_FD)

# #     if ORIGINAL_STDERR_FD is not None:
# #         with suppress(Exception):
# #             os.close(ORIGINAL_STDERR_FD)

# #     TEE_PROCESS = None
# #     ORIGINAL_STDOUT_FD = None
# #     ORIGINAL_STDERR_FD = None


# # def log_event(message: str):
# #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# #     line = f"[{timestamp}] {message}"
# #     print(line)

# #     with suppress(Exception):
# #         sys.stdout.flush()


# # # ---------------------------------------------------------------------------
# # # Scenario configuration
# # # ---------------------------------------------------------------------------

# # # Baylands route start positions.
# # SPAWN_X, SPAWN_Y, SPAWN_Z = 84.6951, 24.1579, 0.6490
# # HUSKY2_X, HUSKY2_Y, HUSKY2_Z = 90.6728, 15.5194, 0.6490

# # # UAV starts close to the ego Husky and then follows above it.
# # UAV_X, UAV_Y, UAV_Z = SPAWN_X + 6.0, SPAWN_Y - 4.0, SPAWN_Z + 4.0

# # HUSKY1_SPAWN_YAW = math.pi
# # HUSKY2_SPAWN_YAW = HUSKY1_SPAWN_YAW
# # UAV_SPAWN_YAW = 0.0

# # HUSKY1_SPAWN_QZ = math.sin(HUSKY1_SPAWN_YAW / 2.0)
# # HUSKY1_SPAWN_QW = math.cos(HUSKY1_SPAWN_YAW / 2.0)

# # HUSKY2_SPAWN_QZ = math.sin(HUSKY2_SPAWN_YAW / 2.0)
# # HUSKY2_SPAWN_QW = math.cos(HUSKY2_SPAWN_YAW / 2.0)

# # UAV_SPAWN_QZ = math.sin(UAV_SPAWN_YAW / 2.0)
# # UAV_SPAWN_QW = math.cos(UAV_SPAWN_YAW / 2.0)


# # def offset_goal_along_path(
# #     world_goal: tuple[float, float, float],
# #     start_xyz: tuple[float, float, float],
# #     offset_distance: float,
# # ) -> tuple[float, float, float]:
# #     """Shift the stopping target along the start->goal line by a fixed distance."""
# #     dx = float(world_goal[0]) - float(start_xyz[0])
# #     dy = float(world_goal[1]) - float(start_xyz[1])
# #     norm = math.hypot(dx, dy)

# #     if norm < 1e-6 or abs(offset_distance) < 1e-9:
# #         return world_goal

# #     ux = dx / norm
# #     uy = dy / norm

# #     return (
# #         float(world_goal[0]) + offset_distance * ux,
# #         float(world_goal[1]) + offset_distance * uy,
# #         float(world_goal[2]),
# #     )


# # # Base goal from the longer Baylands route.
# # # Pulling it slightly back keeps data collection shorter but preserves the same route direction.
# # RAW_WORLD_SHARED_GOAL = (-54.4904, -96.9701, -0.2737)
# # GOAL_WORLD_PULLBACK = 13.0

# # WORLD_SHARED_GOAL = offset_goal_along_path(
# #     RAW_WORLD_SHARED_GOAL,
# #     (SPAWN_X, SPAWN_Y, SPAWN_Z),
# #     -GOAL_WORLD_PULLBACK,
# # )

# # WORLD_HUSKY1_GOAL = WORLD_SHARED_GOAL
# # WORLD_HUSKY2_GOAL = WORLD_SHARED_GOAL
# # WORLD_UAV_GOAL = WORLD_SHARED_GOAL

# # GROUND_MARKER_Z = 0.2025
# # GOAL_STOP_OFFSET = -0.5


# # # ---------------------------------------------------------------------------
# # # Data-collection switches
# # # ---------------------------------------------------------------------------

# # ENABLE_SECOND_HUSKY = True
# # ENABLE_UAV = True

# # ENABLE_BAG_RECORDING = True
# # ENABLE_RVIZ = False
# # ENABLE_CAMERA_VIEW = False

# # # Camera topics are useful but can make bags very large.
# # # Keep this False for normal trajectory/graph experiments.
# # RECORD_CAMERA_TOPICS = False

# # # Useful when debugging only the main Husky.
# # DEBUG_ISOLATE_HUSKY_LOCAL = False


# # # ---------------------------------------------------------------------------
# # # Controller tuning for stable teacher behavior
# # # ---------------------------------------------------------------------------

# # BOOTSTRAP_SECONDS = 3.0
# # BOOTSTRAP_LINEAR_SPEED = 0.8

# # CONTROL_PERIOD = 0.1

# # CMD_LINEAR_GAIN = 1.45
# # CMD_ANGULAR_GAIN = 1.15

# # MIN_LINEAR_SPEED = 1.5
# # MAX_LINEAR_SPEED = 2.0
# # MAX_ANGULAR_SPEED = 0.85

# # HEADING_DEADBAND = 0.12
# # GOAL_TOLERANCE = 1.5

# # STUCK_TIMEOUT_SECONDS = 3.0
# # STUCK_PROGRESS_DISTANCE = 0.15
# # STUCK_REVERSE_SPEED = -0.8
# # STUCK_REVERSE_SECONDS = 2.0
# # STUCK_BOOTSTRAP_SECONDS = 2.0

# # OBSTACLE_FRONT_HALF_ANGLE_DEG = 45.0
# # OBSTACLE_SIDE_ANGLE_DEG = 65.0
# # OBSTACLE_STOP_DISTANCE = 1.8
# # OBSTACLE_CAUTION_DISTANCE = 3.2


# # # ---------------------------------------------------------------------------
# # # UAV follower tuning
# # # ---------------------------------------------------------------------------

# # UAV_FOLLOW_DISTANCE = 0.0
# # UAV_FOLLOW_HEIGHT = 12.0

# # UAV_UPDATE_PERIOD = 0.1
# # UAV_MAX_XY_SPEED = 9.0
# # UAV_MAX_Z_SPEED = 1.2
# # UAV_MAX_YAW_RATE = 0.9

# # UAV_XY_GAIN = 2.0
# # UAV_Z_GAIN = 0.35
# # UAV_YAW_GAIN = 0.8
# # UAV_HEADING_ALIGN_GAIN = 0.9

# # UAV_MIN_FORWARD_SPEED = 0.25
# # UAV_TARGET_SMOOTHING = 1.0
# # UAV_XY_DEADBAND = 0.02
# # UAV_Z_DEADBAND = 0.15
# # UAV_YAW_DEADBAND = 0.18
# # UAV_MIN_TRACK_SPEED = 0.0


# # if DEBUG_ISOLATE_HUSKY_LOCAL:
# #     ENABLE_SECOND_HUSKY = False
# #     ENABLE_UAV = False
# #     ENABLE_BAG_RECORDING = True
# #     RECORD_CAMERA_TOPICS = False


# # # ---------------------------------------------------------------------------
# # # Gazebo process and model-spawn helpers
# # # ---------------------------------------------------------------------------

# # def run_bg(cmd: str):
# #     return subprocess.Popen(["bash", "-c", cmd])


# # def load_husky_sdf_with_topic(topic_name: str) -> str:
# #     husky_sdf = (MODELS_DIR / "husky" / "model.sdf").read_text()
# #     husky_sdf = husky_sdf.replace(
# #         "<topic>/cmd_vel</topic>",
# #         f"<topic>{topic_name}</topic>",
# #         1,
# #     )
# #     return husky_sdf


# # def add_pose_publisher(sdf_text: str) -> str:
# #     """Inject a Gazebo pose publisher so ROS can see world-truth Husky poses."""
# #     if "ignition-gazebo-pose-publisher-system" in sdf_text:
# #         return sdf_text

# #     plugin = """
# #     <plugin
# #       filename="ignition-gazebo-pose-publisher-system"
# #       name="ignition::gazebo::systems::PosePublisher">
# #       <publish_link_pose>true</publish_link_pose>
# #       <use_pose_vector_msg>true</use_pose_vector_msg>
# #     </plugin>
# # """

# #     return sdf_text.replace("</model>", plugin + "\n  </model>", 1)


# # def add_husky_marker(
# #     sdf_text: str,
# #     marker_name: str,
# #     rgba: tuple[float, float, float, float],
# # ) -> str:
# #     marker = f"""
# #     <link name="{marker_name}">
# #       <pose>0 0 0.32 0 0 0</pose>
# #       <collision name="collision">
# #         <geometry>
# #           <cylinder>
# #             <radius>0.015</radius>
# #             <length>0.25</length>
# #           </cylinder>
# #         </geometry>
# #       </collision>
# #       <visual name="visual">
# #         <geometry>
# #           <cylinder>
# #             <radius>0.02</radius>
# #             <length>0.2625</length>
# #           </cylinder>
# #         </geometry>
# #         <material>
# #           <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
# #           <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
# #           <emissive>{rgba[0] * 0.4} {rgba[1] * 0.4} {rgba[2] * 0.4} {rgba[3]}</emissive>
# #         </material>
# #       </visual>
# #     </link>
# #     <joint name="{marker_name}_joint" type="fixed">
# #       <parent>base_link</parent>
# #       <child>{marker_name}</child>
# #     </joint>
# # """

# #     return sdf_text.replace("</model>", marker + "\n  </model>", 1)


# # def write_husky_variant(
# #     output_path: Path,
# #     topic_name: str,
# #     marker_name: str,
# #     rgba: tuple[float, float, float, float],
# # ) -> Path:
# #     sdf_text = load_husky_sdf_with_topic(topic_name)
# #     sdf_text = add_pose_publisher(sdf_text)
# #     sdf_text = add_husky_marker(sdf_text, marker_name, rgba)
# #     output_path.write_text(sdf_text)
# #     return output_path


# # def spawn_goal_marker(
# #     world_name: str,
# #     name: str,
# #     xyz: tuple[float, float, float],
# #     rgba: tuple[float, float, float, float],
# # ):
# #     marker_sdf = f"""<sdf version="1.7">
# #   <model name="{name}">
# #     <static>true</static>
# #     <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>
# #     <link name="marker_link">
# #       <visual name="marker_visual">
# #         <pose>0 0 1.0 0 0 0</pose>
# #         <geometry>
# #           <cylinder>
# #             <radius>0.08</radius>
# #             <length>5.0</length>
# #           </cylinder>
# #         </geometry>
# #         <material>
# #           <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
# #           <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
# #           <emissive>{rgba[0] * 0.5} {rgba[1] * 0.5} {rgba[2] * 0.5} {rgba[3]}</emissive>
# #         </material>
# #       </visual>
# #     </link>
# #   </model>
# # </sdf>"""

# #     one_line = marker_sdf.replace("\n", " ").replace('"', '\\"')

# #     cmd = (
# #         f"ign service -s /world/{world_name}/create "
# #         f"--reqtype ignition.msgs.EntityFactory "
# #         f"--reptype ignition.msgs.Boolean "
# #         f"--timeout 5000 "
# #         f'--req \'sdf: "{one_line}"\''
# #     )

# #     subprocess.run(
# #         ["bash", "-c", cmd],
# #         stdout=subprocess.DEVNULL,
# #         stderr=subprocess.DEVNULL,
# #     )


# # def rgbd_camera_bridge_topics(
# #     world_name: str,
# #     model_name: str,
# #     link_name: str,
# #     sensor_name: str,
# # ) -> list[str]:
# #     prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"

# #     return [
# #         f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
# #         f"{prefix}/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
# #         f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
# #         f"{prefix}/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
# #     ]


# # def camera_bridge_topics(
# #     world_name: str,
# #     model_name: str,
# #     link_name: str,
# #     sensor_name: str,
# # ) -> list[str]:
# #     prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"

# #     return [
# #         f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
# #         f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
# #     ]


# # def husky_sensor_bridge_topics(world_name: str, model_name: str) -> list[str]:
# #     base_prefix = f"/world/{world_name}/model/{model_name}/link/base_link/sensor"

# #     topics = [
# #         f"{base_prefix}/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
# #         f"{base_prefix}/planar_laser/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
# #         f"{base_prefix}/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
# #     ]

# #     topics.extend(
# #         rgbd_camera_bridge_topics(
# #             world_name,
# #             model_name,
# #             "base_link",
# #             "camera_front",
# #         )
# #     )
# #     topics.extend(
# #         rgbd_camera_bridge_topics(
# #             world_name,
# #             model_name,
# #             "base_link",
# #             "camera_down",
# #         )
# #     )
# #     topics.extend(
# #         rgbd_camera_bridge_topics(
# #             world_name,
# #             model_name,
# #             "tilt_gimbal_link",
# #             "camera_pan_tilt",
# #         )
# #     )

# #     return topics


# # def uav_bridge_topics(world_name: str, uav_name: str) -> list[str]:
# #     topics = [
# #         f"/{uav_name}/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
# #         f"/{uav_name}/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
# #         f"/model/{uav_name}/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
# #         f"/model/{uav_name}/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
# #         f"/model/{uav_name}/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
# #         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
# #         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
# #         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/air_pressure/air_pressure@sensor_msgs/msg/FluidPressure[ignition.msgs.FluidPressure",
# #         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/magnetometer/magnetometer@sensor_msgs/msg/MagneticField[ignition.msgs.Magnetometer",
# #     ]

# #     topics.extend(
# #         camera_bridge_topics(
# #             world_name,
# #             uav_name,
# #             "base_link",
# #             "camera_front",
# #         )
# #     )

# #     return topics


# # def build_bag_topics(
# #     world_name: str,
# #     *,
# #     include_second_husky: bool,
# #     include_uav: bool,
# #     include_camera_topics: bool,
# # ) -> list[str]:
# #     """Return the useful topic set for hybrid trajectory dataset collection."""

# #     topics = [
# #         "/clock",
# #         f"/world/{world_name}/dynamic_pose/info",

# #         # Ego Husky control/state.
# #         "/cmd_vel",
# #         "/husky_local/controller_state",
# #         "/husky_local/obstacle_action",
# #         "/husky_local/obstacle_clearance",
# #         "/episode/husky_local/start",
# #         "/episode/husky_local/goal",
# #         "/model/husky_local/odometry",

# #         # Ego Husky local perception.
# #         f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan",
# #         f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
# #         f"/world/{world_name}/model/husky_local/link/base_link/sensor/imu_sensor/imu",
# #     ]

# #     if include_camera_topics:
# #         topics.extend(
# #             [
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/image",
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/depth_image",
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/camera_info",
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/points",
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/image",
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/depth_image",
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/camera_info",
# #                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/points",
# #             ]
# #         )

# #     if include_second_husky:
# #         topics.extend(
# #             [
# #                 "/cmd_vel_husky2",
# #                 "/husky_2/controller_state",
# #                 "/husky_2/obstacle_action",
# #                 "/husky_2/obstacle_clearance",
# #                 "/episode/husky_2/start",
# #                 "/episode/husky_2/goal",
# #                 "/model/husky_2/odometry",
# #                 f"/world/{world_name}/model/husky_2/link/base_link/sensor/planar_laser/scan",
# #                 f"/world/{world_name}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
# #                 f"/world/{world_name}/model/husky_2/link/base_link/sensor/imu_sensor/imu",
# #             ]
# #         )

# #         if include_camera_topics:
# #             topics.extend(
# #                 [
# #                     f"/world/{world_name}/model/husky_2/link/base_link/sensor/camera_front/image",
# #                     f"/world/{world_name}/model/husky_2/link/base_link/sensor/camera_front/depth_image",
# #                     f"/world/{world_name}/model/husky_2/link/base_link/sensor/camera_front/camera_info",
# #                     f"/world/{world_name}/model/husky_2/link/base_link/sensor/camera_front/points",
# #                 ]
# #             )

# #     if include_uav:
# #         topics.extend(
# #             [
# #                 "/episode/uav1/start",
# #                 "/episode/uav1/goal",
# #                 "/model/uav1/odometry",
# #                 "/uav1/ready",

# #                 # UAV forward perception.
# #                 f"/world/{world_name}/model/uav1/link/base_link/sensor/front_laser/scan/points",
# #                 f"/world/{world_name}/model/uav1/link/base_link/sensor/imu_sensor/imu",
# #                 f"/world/{world_name}/model/uav1/link/base_link/sensor/air_pressure/air_pressure",
# #                 f"/world/{world_name}/model/uav1/link/base_link/sensor/magnetometer/magnetometer",
# #             ]
# #         )

# #         if include_camera_topics:
# #             topics.extend(
# #                 [
# #                     f"/world/{world_name}/model/uav1/link/base_link/sensor/camera_front/image",
# #                     f"/world/{world_name}/model/uav1/link/base_link/sensor/camera_front/camera_info",
# #                 ]
# #             )

# #     return topics


# # # ---------------------------------------------------------------------------
# # # Start Gazebo and spawn entities
# # # ---------------------------------------------------------------------------

# # os.environ["IGN_GAZEBO_RESOURCE_PATH"] = (
# #     MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
# # )
# # os.environ["GZ_SIM_RESOURCE_PATH"] = (
# #     MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
# # )

# # setup_terminal_tee(RUN_LOG_PATH)

# # subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
# # subprocess.run(["bash", "-c", "pkill -f ign || true"])
# # subprocess.run(["bash", "-c", f"pkill -f {OMNET_BIN} || true"])

# # log_event(f"START timestamp: {RUN_START_DT.isoformat(timespec='seconds')}")
# # log_event(f"Run log file: {RUN_LOG_PATH}")
# # log_event(f"World path: {WORLD}")
# # log_event("Starting Gazebo...")

# # gz = run_bg(f"ign gazebo {WORLD}")

# # time.sleep(5)

# # log_event("Waiting for Baylands world to fully load...")
# # time.sleep(40)

# # log_event("Spawning Husky ego: husky_local")

# # husky1_sdf_path = write_husky_variant(
# #     MODELS_DIR / "husky" / "model_red_tag.sdf",
# #     "/cmd_vel",
# #     "flag_marker_red",
# #     (0.95, 0.12, 0.12, 1.0),
# # )

# # spawn_husky = (
# #     "ign service -s /world/{world_name}/create "
# #     "--reqtype ignition.msgs.EntityFactory "
# #     "--reptype ignition.msgs.Boolean "
# #     "--timeout 5000 "
# #     "--req 'sdf_filename: \"{sdf_path}\", name: \"husky_local\", "
# #     "pose: {{position: {{x: {spawn_x}, y: {spawn_y}, z: {spawn_z}}}, "
# #     "orientation: {{z: {spawn_qz}, w: {spawn_qw}}}}}'"
# # ).format(
# #     world_name=WORLD_NAME,
# #     sdf_path=husky1_sdf_path,
# #     spawn_x=SPAWN_X,
# #     spawn_y=SPAWN_Y,
# #     spawn_z=SPAWN_Z,
# #     spawn_qz=HUSKY1_SPAWN_QZ,
# #     spawn_qw=HUSKY1_SPAWN_QW,
# # )

# # subprocess.run(["bash", "-c", spawn_husky])
# # time.sleep(5)


# # if ENABLE_SECOND_HUSKY:
# #     log_event("Spawning second Husky context agent: husky_2")

# #     husky2_sdf_path = write_husky_variant(
# #         MODELS_DIR / "husky" / "model_blue_tag.sdf",
# #         "/cmd_vel_husky2",
# #         "flag_marker_blue",
# #         (0.12, 0.36, 0.95, 1.0),
# #     )

# #     spawn_husky2 = (
# #         f"ign service -s /world/{WORLD_NAME}/create "
# #         f"--reqtype ignition.msgs.EntityFactory "
# #         f"--reptype ignition.msgs.Boolean "
# #         f"--timeout 5000 "
# #         f'--req \'sdf_filename: "{husky2_sdf_path}", name: "husky_2", '
# #         f'pose: {{position: {{x: {HUSKY2_X}, y: {HUSKY2_Y}, z: {HUSKY2_Z}}}, '
# #         f'orientation: {{z: {HUSKY2_SPAWN_QZ}, w: {HUSKY2_SPAWN_QW}}}}}\''
# #     )

# #     subprocess.run(["bash", "-c", spawn_husky2])
# #     time.sleep(5)


# # if ENABLE_UAV:
# #     log_event("Spawning UAV forward/context agent: uav1")

# #     spawn_uav = """
# # ign service -s /world/{world_name}/create \
# # --reqtype ignition.msgs.EntityFactory \
# # --reptype ignition.msgs.Boolean \
# # --timeout 5000 \
# # --req 'sdf_filename: "model://m100/model.sdf", name: "uav1",
# # pose: {{position: {{x: {uav_x}, y: {uav_y}, z: {uav_z}}}, orientation: {{z: {uav_qz}, w: {uav_qw}}}}}'
# # """.format(
# #         world_name=WORLD_NAME,
# #         uav_x=UAV_X,
# #         uav_y=UAV_Y,
# #         uav_z=UAV_Z,
# #         uav_qz=UAV_SPAWN_QZ,
# #         uav_qw=UAV_SPAWN_QW,
# #     )

# #     subprocess.run(["bash", "-c", spawn_uav])
# #     time.sleep(5)


# # log_event("Spawning visible goal markers...")

# # spawn_goal_marker(
# #     WORLD_NAME,
# #     "goal_husky_local",
# #     (
# #         WORLD_HUSKY1_GOAL[0],
# #         WORLD_HUSKY1_GOAL[1],
# #         GROUND_MARKER_Z,
# #     ),
# #     (0.95, 0.12, 0.12, 1.0),
# # )

# # if ENABLE_SECOND_HUSKY:
# #     spawn_goal_marker(
# #         WORLD_NAME,
# #         "goal_husky_2",
# #         (
# #             WORLD_HUSKY2_GOAL[0],
# #             WORLD_HUSKY2_GOAL[1],
# #             GROUND_MARKER_Z,
# #         ),
# #         (0.12, 0.36, 0.95, 1.0),
# #     )

# # if ENABLE_UAV:
# #     spawn_goal_marker(
# #         WORLD_NAME,
# #         "goal_uav1",
# #         (
# #             WORLD_UAV_GOAL[0],
# #             WORLD_UAV_GOAL[1],
# #             GROUND_MARKER_Z,
# #         ),
# #         (0.95, 0.85, 0.12, 1.0),
# #     )

# # time.sleep(1)


# # # ---------------------------------------------------------------------------
# # # Start ROS-Gazebo bridge
# # # ---------------------------------------------------------------------------

# # log_event("Starting ROS-Gazebo bridge...")

# # bridge_topics = [
# #     "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
# #     "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
# #     f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
# # ]

# # bridge_topics.extend(
# #     husky_sensor_bridge_topics(
# #         WORLD_NAME,
# #         "husky_local",
# #     )
# # )

# # if ENABLE_SECOND_HUSKY:
# #     bridge_topics.extend(
# #         [
# #             "/cmd_vel_husky2@geometry_msgs/msg/Twist@ignition.msgs.Twist",
# #             "/model/husky_2/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
# #             f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
# #         ]
# #     )

# #     bridge_topics.extend(
# #         husky_sensor_bridge_topics(
# #             WORLD_NAME,
# #             "husky_2",
# #         )
# #     )

# # if ENABLE_UAV:
# #     bridge_topics.extend(
# #         uav_bridge_topics(
# #             WORLD_NAME,
# #             "uav1",
# #         )
# #     )

# # bridge_cmd = (
# #     "source /opt/ros/humble/setup.bash && "
# #     "ros2 run ros_gz_bridge parameter_bridge "
# #     + " ".join(bridge_topics)
# # )

# # bridge = run_bg(bridge_cmd)
# # time.sleep(2)


# # # ---------------------------------------------------------------------------
# # # Optional visual tools
# # # ---------------------------------------------------------------------------

# # rviz = None
# # camera_view = None

# # if ENABLE_RVIZ:
# #     log_event(f"Starting RViz with config: {RVIZ_CONFIG_PATH}")
# #     rviz_cmd = (
# #         "source /opt/ros/humble/setup.bash && "
# #         f"rviz2 -d {RVIZ_CONFIG_PATH}"
# #     )
# #     rviz = run_bg(rviz_cmd)
# #     time.sleep(2)

# # if ENABLE_CAMERA_VIEW:
# #     log_event("Starting camera viewer...")
# #     camera_cmd = (
# #         "source /opt/ros/humble/setup.bash && "
# #         "ros2 run rqt_image_view rqt_image_view"
# #     )
# #     camera_view = run_bg(camera_cmd)
# #     time.sleep(2)


# # # ---------------------------------------------------------------------------
# # # OMNeT++
# # # ---------------------------------------------------------------------------

# # omnet = None

# # if ENABLE_UAV:
# #     log_event("OMNeT++ relay disabled in this data-collection runner.")
# #     log_event("Network features will be exported as placeholder edge fields unless OMNeT++ topics are added later.")


# # # ---------------------------------------------------------------------------
# # # Start rosbag recording
# # # ---------------------------------------------------------------------------

# # BAG_DIR = Path.home() / "Documents/Thesis/03_dataset/bags"
# # BAG_DIR.mkdir(parents=True, exist_ok=True)

# # run_name = "run_dataset_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# # bag_path = BAG_DIR / run_name

# # recorder = None

# # if ENABLE_BAG_RECORDING:
# #     bag_topics = build_bag_topics(
# #         WORLD_NAME,
# #         include_second_husky=ENABLE_SECOND_HUSKY,
# #         include_uav=ENABLE_UAV,
# #         include_camera_topics=RECORD_CAMERA_TOPICS,
# #     )

# #     log_event(f"Recording bag: {bag_path}")
# #     log_event("Bag topic set for hybrid trajectory collection:")
# #     for topic in bag_topics:
# #         log_event(f"  - {topic}")

# #     record_cmd = (
# #         "source /opt/ros/humble/setup.bash && "
# #         f"ros2 bag record -o {bag_path} "
# #         + " ".join(bag_topics)
# #     )

# #     recorder = run_bg(record_cmd)
# # else:
# #     log_event("Bag recording disabled. Set ENABLE_BAG_RECORDING = True to record a run.")


# # if DEBUG_ISOLATE_HUSKY_LOCAL:
# #     log_event(
# #         "DEBUG isolate mode: only husky_local is spawned. "
# #         "Bag recording remains enabled."
# #     )


# # # ---------------------------------------------------------------------------
# # # ROS 2 nodes
# # # ---------------------------------------------------------------------------

# # log_event("==============================")
# # log_event("RULE-BASED DATASET COLLECTION MODE")
# # log_event("==============================")
# # log_event("Press Play in Gazebo.")
# # log_event("Let the simulation run until the route is completed or enough data is collected.")
# # log_event("Press Ctrl+C in this terminal to stop and close the bag cleanly.")

# # rclpy.init()

# # HUSKY1_GOAL = offset_goal_along_path(
# #     WORLD_HUSKY1_GOAL,
# #     (SPAWN_X, SPAWN_Y, SPAWN_Z),
# #     GOAL_STOP_OFFSET,
# # )

# # HUSKY2_GOAL = offset_goal_along_path(
# #     WORLD_HUSKY2_GOAL,
# #     (HUSKY2_X, HUSKY2_Y, HUSKY2_Z),
# #     GOAL_STOP_OFFSET,
# # )

# # UAV_GOAL = offset_goal_along_path(
# #     WORLD_UAV_GOAL,
# #     (UAV_X, UAV_Y, UAV_Z),
# #     GOAL_STOP_OFFSET,
# # )

# # log_event(
# #     "Controller goals with stop offset: "
# #     f"husky_local=({HUSKY1_GOAL[0]:.3f}, {HUSKY1_GOAL[1]:.3f}), "
# #     f"husky_2=({HUSKY2_GOAL[0]:.3f}, {HUSKY2_GOAL[1]:.3f}), "
# #     f"uav1=({UAV_GOAL[0]:.3f}, {UAV_GOAL[1]:.3f})"
# # )

# # log_event(
# #     f"Visible goal marker remains at "
# #     f"({WORLD_HUSKY1_GOAL[0]:.3f}, {WORLD_HUSKY1_GOAL[1]:.3f}); "
# #     f"controller stop tolerance is {GOAL_TOLERANCE:.2f} m."
# # )


# # start_goals = {
# #     "husky_local": {
# #         "start": (SPAWN_X, SPAWN_Y, SPAWN_Z),
# #         "goal": WORLD_HUSKY1_GOAL,
# #     },
# # }

# # if ENABLE_SECOND_HUSKY:
# #     start_goals["husky_2"] = {
# #         "start": (HUSKY2_X, HUSKY2_Y, HUSKY2_Z),
# #         "goal": WORLD_HUSKY2_GOAL,
# #     }

# # if ENABLE_UAV:
# #     start_goals["uav1"] = {
# #         "start": (UAV_X, UAV_Y, UAV_Z),
# #         "goal": WORLD_UAV_GOAL,
# #     }

# # episode_metadata = EpisodeMetadataPublisher(
# #     world_name=WORLD_NAME,
# #     start_goals=start_goals,
# # )


# # # Ego Husky nodes.
# # husky1_obstacle_action_topic = "/husky_local/obstacle_action"
# # husky1_obstacle_clearance_topic = "/husky_local/obstacle_clearance"
# # husky1_controller_state_topic = "/husky_local/controller_state"

# # obstacle_detector = ObstacleDetectionNode(
# #     node_name="husky_local_obstacle_detector",
# #     scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
# #     action_topic=husky1_obstacle_action_topic,
# #     clearance_topic=husky1_obstacle_clearance_topic,
# #     pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
# #     front_half_angle_deg=OBSTACLE_FRONT_HALF_ANGLE_DEG,
# #     side_angle_deg=OBSTACLE_SIDE_ANGLE_DEG,
# #     stop_distance=OBSTACLE_STOP_DISTANCE,
# #     caution_distance=OBSTACLE_CAUTION_DISTANCE,
# # )

# # driver = ModelHuskyDriver(
# #     node_name="model_husky_driver_1",
# #     cmd_topic="/cmd_vel",
# #     odom_topic="/model/husky_local/odometry",
# #     world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
# #     uav_ready_topic=None,
# #     require_uav_ready=False,
# #     obstacle_action_topic=husky1_obstacle_action_topic,
# #     obstacle_clearance_topic=husky1_obstacle_clearance_topic,
# #     state_topic=husky1_controller_state_topic,
# #     goal_xyz=HUSKY1_GOAL,
# #     world_goal_xyz=HUSKY1_GOAL,
# #     bootstrap_seconds=BOOTSTRAP_SECONDS,
# #     bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
# #     control_period=CONTROL_PERIOD,
# #     cmd_linear_gain=CMD_LINEAR_GAIN,
# #     cmd_angular_gain=CMD_ANGULAR_GAIN,
# #     min_linear_speed=MIN_LINEAR_SPEED,
# #     max_linear_speed=MAX_LINEAR_SPEED,
# #     max_angular_speed=MAX_ANGULAR_SPEED,
# #     heading_deadband=HEADING_DEADBAND,
# #     goal_tolerance=GOAL_TOLERANCE,
# #     stuck_timeout_seconds=STUCK_TIMEOUT_SECONDS,
# #     stuck_progress_distance=STUCK_PROGRESS_DISTANCE,
# #     stuck_reverse_speed=STUCK_REVERSE_SPEED,
# #     stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
# #     stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
# # )


# # # Second Husky nodes.
# # driver2 = None
# # obstacle_detector2 = None

# # if ENABLE_SECOND_HUSKY:
# #     husky2_obstacle_action_topic = "/husky_2/obstacle_action"
# #     husky2_obstacle_clearance_topic = "/husky_2/obstacle_clearance"
# #     husky2_controller_state_topic = "/husky_2/controller_state"

# #     obstacle_detector2 = ObstacleDetectionNode(
# #         node_name="husky_2_obstacle_detector",
# #         scan_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/planar_laser/scan",
# #         action_topic=husky2_obstacle_action_topic,
# #         clearance_topic=husky2_obstacle_clearance_topic,
# #         pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_2/link/base_link/sensor/front_laser/scan/points",
# #         front_half_angle_deg=OBSTACLE_FRONT_HALF_ANGLE_DEG,
# #         side_angle_deg=OBSTACLE_SIDE_ANGLE_DEG,
# #         stop_distance=OBSTACLE_STOP_DISTANCE,
# #         caution_distance=OBSTACLE_CAUTION_DISTANCE,
# #     )

# #     driver2 = ModelHuskyDriver(
# #         node_name="model_husky_driver_2",
# #         cmd_topic="/cmd_vel_husky2",
# #         odom_topic="/model/husky_2/odometry",
# #         world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
# #         uav_ready_topic=None,
# #         require_uav_ready=False,
# #         obstacle_action_topic=husky2_obstacle_action_topic,
# #         obstacle_clearance_topic=husky2_obstacle_clearance_topic,
# #         state_topic=husky2_controller_state_topic,
# #         goal_xyz=HUSKY2_GOAL,
# #         world_goal_xyz=HUSKY2_GOAL,
# #         bootstrap_seconds=BOOTSTRAP_SECONDS,
# #         bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
# #         control_period=CONTROL_PERIOD,
# #         cmd_linear_gain=CMD_LINEAR_GAIN,
# #         cmd_angular_gain=CMD_ANGULAR_GAIN,
# #         min_linear_speed=MIN_LINEAR_SPEED,
# #         max_linear_speed=MAX_LINEAR_SPEED,
# #         max_angular_speed=MAX_ANGULAR_SPEED,
# #         heading_deadband=HEADING_DEADBAND,
# #         goal_tolerance=GOAL_TOLERANCE,
# #         stuck_timeout_seconds=STUCK_TIMEOUT_SECONDS,
# #         stuck_progress_distance=STUCK_PROGRESS_DISTANCE,
# #         stuck_reverse_speed=STUCK_REVERSE_SPEED,
# #         stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
# #         stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
# #     )


# # # UAV follower node.
# # follower = None
# # uav_ready_topic = "/uav1/ready"

# # if ENABLE_UAV:
# #     follower = UavFollower(
# #         node_name="uav1_follower",
# #         husky_odom_topic="/model/husky_local/odometry",
# #         uav_odom_topic="/model/uav1/odometry",
# #         world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
# #         husky_model_name="husky_local",
# #         uav_model_name="uav1",
# #         uav_name="uav1",
# #         follow_distance=UAV_FOLLOW_DISTANCE,
# #         follow_height=UAV_FOLLOW_HEIGHT,
# #         ready_topic=uav_ready_topic,
# #         update_period=UAV_UPDATE_PERIOD,
# #         max_xy_speed=UAV_MAX_XY_SPEED,
# #         max_z_speed=UAV_MAX_Z_SPEED,
# #         max_yaw_rate=UAV_MAX_YAW_RATE,
# #         xy_gain=UAV_XY_GAIN,
# #         z_gain=UAV_Z_GAIN,
# #         yaw_gain=UAV_YAW_GAIN,
# #         heading_align_gain=UAV_HEADING_ALIGN_GAIN,
# #         min_forward_speed=UAV_MIN_FORWARD_SPEED,
# #         target_smoothing=UAV_TARGET_SMOOTHING,
# #         xy_deadband=UAV_XY_DEADBAND,
# #         z_deadband=UAV_Z_DEADBAND,
# #         yaw_deadband=UAV_YAW_DEADBAND,
# #         min_track_speed=UAV_MIN_TRACK_SPEED,
# #         husky_spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
# #         husky_spawn_yaw=HUSKY1_SPAWN_YAW,
# #         uav_spawn_xyz=(UAV_X, UAV_Y, UAV_Z),
# #         uav_spawn_yaw=UAV_SPAWN_YAW,
# #     )


# # executor = MultiThreadedExecutor()

# # executor.add_node(episode_metadata)
# # executor.add_node(obstacle_detector)
# # executor.add_node(driver)

# # if obstacle_detector2 is not None:
# #     executor.add_node(obstacle_detector2)

# # if driver2 is not None:
# #     executor.add_node(driver2)

# # if follower is not None:
# #     executor.add_node(follower)


# # try:
# #     executor.spin()

# # except KeyboardInterrupt:
# #     log_event("Stopping dataset collection run...")

# # finally:
# #     managed_nodes = [
# #         episode_metadata,
# #         obstacle_detector,
# #         driver,
# #     ]

# #     if obstacle_detector2 is not None:
# #         managed_nodes.append(obstacle_detector2)

# #     if driver2 is not None:
# #         managed_nodes.append(driver2)

# #     if follower is not None:
# #         managed_nodes.append(follower)

# #     for node in managed_nodes:
# #         with suppress(Exception):
# #             executor.remove_node(node)

# #     executor.shutdown(timeout_sec=2.0)
# #     time.sleep(0.25)

# #     for node in managed_nodes:
# #         with suppress(Exception):
# #             node.destroy_node()

# #     if rclpy.ok():
# #         rclpy.shutdown()

# #     if recorder is not None:
# #         log_event("Stopping rosbag recorder cleanly...")
# #         with suppress(Exception):
# #             recorder.send_signal(signal.SIGINT)
# #         time.sleep(2)

# #     log_event(f"STOP timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}")
# #     log_event("Stopping bridge, OMNeT++, RViz/camera viewer, and Gazebo...")

# #     managed_processes = [
# #         bridge,
# #         rviz,
# #         camera_view,
# #         omnet,
# #         gz,
# #     ]

# #     for proc in managed_processes:
# #         if proc is None:
# #             continue
# #         with suppress(Exception):
# #             proc.send_signal(signal.SIGINT)

# #     time.sleep(2)

# #     for proc in managed_processes:
# #         if proc is None:
# #             continue
# #         with suppress(Exception):
# #             if proc.poll() is None:
# #                 proc.terminate()

# #     log_event("All processes stopped cleanly.")
# #     log_event(f"Bag output folder should be under: {BAG_DIR}")
# #     log_event(f"Latest bag for this run: {bag_path}")

# #     close_terminal_tee()



# """Launch the rule-based data-collection simulation pipeline.

# This runner starts Gazebo, spawns one Husky and two UAV support agents,
# publishes episode metadata, records a rosbag, and runs the rule-based
# Husky/UAV controllers.

# Purpose of this runner:
# - Generate training/evaluation data for trajectory prediction models.
# - Record synchronized UGV, UAV, lidar, pointcloud, goal, command, and controller-state topics.
# - Keep the environment and topic structure compatible with the hybrid dataset exporter.

# New agent structure:
# - husky_local: ego UGV whose future trajectory is predicted.
# - uav1: left-side aerial support agent.
# - uav2: right-side aerial support agent.

# Dataset target:
# The generated bags are later exported into frames.jsonl and .npy assets by:

#     03_dataset/exporters/export_hybrid_maneuver_dataset.py

# Main recorded modalities:
# - past Husky trajectory and command history
# - Husky local lidar / front pointcloud
# - UAV1 and UAV2 odometry
# - UAV1 and UAV2 forward pointclouds
# - inter-agent relations through exported graph features
# - goal/start metadata
# - controller labels such as go_to_goal, avoid_left, avoid_right, arrived
# """

# import datetime
# import math
# import os
# import signal
# import subprocess
# import sys
# import time
# from contextlib import suppress
# from pathlib import Path

# SCRIPT_DIR = Path(__file__).resolve().parent
# RULE_BASED_ROOT = SCRIPT_DIR.parent
# if str(RULE_BASED_ROOT) not in sys.path:
#     sys.path.insert(0, str(RULE_BASED_ROOT))

# import rclpy
# from rclpy.executors import MultiThreadedExecutor

# from controllers.episode_metadata import EpisodeMetadataPublisher
# from controllers.husky_model_driver import ModelHuskyDriver
# from controllers.obstacle_detection import ObstacleDetectionNode
# from controllers.uav_follower import UavFollower
# from project_paths import MODELS_DIR, OMNET_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH


# # ---------------------------------------------------------------------------
# # Global paths and runtime logging
# # ---------------------------------------------------------------------------

# WORLD = str(WORLD_SDF_PATH)
# WORLD_NAME = "baylands"

# MODEL_PATH = str(MODELS_DIR)
# OMNET_BIN = OMNET_DIR / "onmetpp"
# OMNET_CONFIG = "WifiRelay"

# RUN_START_DT = datetime.datetime.now()

# LOG_DIR = Path.home() / "Documents/Thesis/03_dataset/logs"
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# RUN_LOG_PATH = LOG_DIR / f"rule_based_dataset_{RUN_START_DT.strftime('%Y%m%d_%H%M%S')}.log"

# TEE_PROCESS = None
# ORIGINAL_STDOUT_FD = None
# ORIGINAL_STDERR_FD = None


# def setup_terminal_tee(log_path: Path):
#     global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD

#     if TEE_PROCESS is not None:
#         return

#     ORIGINAL_STDOUT_FD = os.dup(sys.__stdout__.fileno())
#     ORIGINAL_STDERR_FD = os.dup(sys.__stderr__.fileno())

#     TEE_PROCESS = subprocess.Popen(
#         ["tee", "-a", str(log_path)],
#         stdin=subprocess.PIPE,
#         stdout=ORIGINAL_STDOUT_FD,
#         stderr=ORIGINAL_STDERR_FD,
#         bufsize=0,
#     )

#     os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stdout__.fileno())
#     os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stderr__.fileno())


# def close_terminal_tee():
#     global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD

#     if TEE_PROCESS is None:
#         return

#     with suppress(Exception):
#         sys.stdout.flush()
#     with suppress(Exception):
#         sys.stderr.flush()

#     if ORIGINAL_STDOUT_FD is not None:
#         with suppress(Exception):
#             os.dup2(ORIGINAL_STDOUT_FD, sys.__stdout__.fileno())

#     if ORIGINAL_STDERR_FD is not None:
#         with suppress(Exception):
#             os.dup2(ORIGINAL_STDERR_FD, sys.__stderr__.fileno())

#     with suppress(Exception):
#         if TEE_PROCESS.stdin is not None:
#             TEE_PROCESS.stdin.close()

#     with suppress(Exception):
#         TEE_PROCESS.wait(timeout=2.0)

#     if ORIGINAL_STDOUT_FD is not None:
#         with suppress(Exception):
#             os.close(ORIGINAL_STDOUT_FD)

#     if ORIGINAL_STDERR_FD is not None:
#         with suppress(Exception):
#             os.close(ORIGINAL_STDERR_FD)

#     TEE_PROCESS = None
#     ORIGINAL_STDOUT_FD = None
#     ORIGINAL_STDERR_FD = None


# def log_event(message: str):
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     line = f"[{timestamp}] {message}"
#     print(line)

#     with suppress(Exception):
#         sys.stdout.flush()


# # ---------------------------------------------------------------------------
# # Scenario configuration
# # ---------------------------------------------------------------------------

# # Baylands route start position for the ego Husky.
# SPAWN_X, SPAWN_Y, SPAWN_Z = 84.6951, 24.1579, 0.6490

# HUSKY1_SPAWN_YAW = math.pi
# HUSKY1_SPAWN_QZ = math.sin(HUSKY1_SPAWN_YAW / 2.0)
# HUSKY1_SPAWN_QW = math.cos(HUSKY1_SPAWN_YAW / 2.0)

# # Two UAVs are placed on the left/right side of the Husky.
# # They are not placed ahead or behind the Husky.
# UAV_INITIAL_FORWARD_OFFSET = 0.0
# UAV_SIDE_OFFSET = 4.0
# UAV_INITIAL_HEIGHT = 12.0

# UAV_SPAWN_YAW = 0.0
# UAV2_SPAWN_YAW = 0.0

# UAV_SPAWN_QZ = math.sin(UAV_SPAWN_YAW / 2.0)
# UAV_SPAWN_QW = math.cos(UAV_SPAWN_YAW / 2.0)

# UAV2_SPAWN_QZ = math.sin(UAV2_SPAWN_YAW / 2.0)
# UAV2_SPAWN_QW = math.cos(UAV2_SPAWN_YAW / 2.0)


# def body_offset_to_world(
#     base_xyz: tuple[float, float, float],
#     base_yaw: float,
#     forward_offset: float,
#     lateral_offset: float,
#     height_offset: float,
# ) -> tuple[float, float, float]:
#     """Convert a body-frame offset around the Husky into world coordinates.

#     forward_offset:
#         positive means in front of the Husky.
#         negative means behind the Husky.

#     lateral_offset:
#         positive means left side of the Husky.
#         negative means right side of the Husky.
#     """
#     x = (
#         float(base_xyz[0])
#         + forward_offset * math.cos(base_yaw)
#         - lateral_offset * math.sin(base_yaw)
#     )
#     y = (
#         float(base_xyz[1])
#         + forward_offset * math.sin(base_yaw)
#         + lateral_offset * math.cos(base_yaw)
#     )
#     z = float(base_xyz[2]) + height_offset
#     return (x, y, z)


# UAV1_X, UAV1_Y, UAV1_Z = body_offset_to_world(
#     (SPAWN_X, SPAWN_Y, SPAWN_Z),
#     HUSKY1_SPAWN_YAW,
#     UAV_INITIAL_FORWARD_OFFSET,
#     UAV_SIDE_OFFSET,
#     UAV_INITIAL_HEIGHT,
# )

# UAV2_X, UAV2_Y, UAV2_Z = body_offset_to_world(
#     (SPAWN_X, SPAWN_Y, SPAWN_Z),
#     HUSKY1_SPAWN_YAW,
#     UAV_INITIAL_FORWARD_OFFSET,
#     -UAV_SIDE_OFFSET,
#     UAV_INITIAL_HEIGHT,
# )


# def offset_goal_along_path(
#     world_goal: tuple[float, float, float],
#     start_xyz: tuple[float, float, float],
#     offset_distance: float,
# ) -> tuple[float, float, float]:
#     """Shift the stopping target along the start->goal line by a fixed distance."""
#     dx = float(world_goal[0]) - float(start_xyz[0])
#     dy = float(world_goal[1]) - float(start_xyz[1])
#     norm = math.hypot(dx, dy)

#     if norm < 1e-6 or abs(offset_distance) < 1e-9:
#         return world_goal

#     ux = dx / norm
#     uy = dy / norm

#     return (
#         float(world_goal[0]) + offset_distance * ux,
#         float(world_goal[1]) + offset_distance * uy,
#         float(world_goal[2]),
#     )


# # Base goal from the longer Baylands route.
# # Pulling it slightly back keeps data collection shorter but preserves the same route direction.
# RAW_WORLD_SHARED_GOAL = (-54.4904, -96.9701, -0.2737)
# GOAL_WORLD_PULLBACK = 13.0

# WORLD_SHARED_GOAL = offset_goal_along_path(
#     RAW_WORLD_SHARED_GOAL,
#     (SPAWN_X, SPAWN_Y, SPAWN_Z),
#     -GOAL_WORLD_PULLBACK,
# )

# WORLD_HUSKY1_GOAL = WORLD_SHARED_GOAL
# WORLD_UAV1_GOAL = WORLD_SHARED_GOAL
# WORLD_UAV2_GOAL = WORLD_SHARED_GOAL

# GROUND_MARKER_Z = 0.2025
# GOAL_STOP_OFFSET = -0.5


# # ---------------------------------------------------------------------------
# # Data-collection switches
# # ---------------------------------------------------------------------------

# ENABLE_UAV1 = True
# ENABLE_UAV2 = True

# ENABLE_BAG_RECORDING = True
# ENABLE_RVIZ = False
# ENABLE_CAMERA_VIEW = False

# # Camera topics are useful but can make bags very large.
# # Keep this False for normal trajectory/graph experiments.
# RECORD_CAMERA_TOPICS = False

# # Useful when debugging only the main Husky.
# DEBUG_ISOLATE_HUSKY_LOCAL = False


# # ---------------------------------------------------------------------------
# # Controller tuning for stable teacher behavior
# # ---------------------------------------------------------------------------

# BOOTSTRAP_SECONDS = 3.0
# BOOTSTRAP_LINEAR_SPEED = 0.8

# CONTROL_PERIOD = 0.1

# CMD_LINEAR_GAIN = 1.45
# CMD_ANGULAR_GAIN = 1.15

# MIN_LINEAR_SPEED = 1.5
# MAX_LINEAR_SPEED = 2.0
# MAX_ANGULAR_SPEED = 0.85

# HEADING_DEADBAND = 0.12
# GOAL_TOLERANCE = 1.5

# STUCK_TIMEOUT_SECONDS = 3.0
# STUCK_PROGRESS_DISTANCE = 0.15
# STUCK_REVERSE_SPEED = -0.8
# STUCK_REVERSE_SECONDS = 2.0
# STUCK_BOOTSTRAP_SECONDS = 2.0

# OBSTACLE_FRONT_HALF_ANGLE_DEG = 45.0
# OBSTACLE_SIDE_ANGLE_DEG = 65.0
# OBSTACLE_STOP_DISTANCE = 1.8
# OBSTACLE_CAUTION_DISTANCE = 3.2


# # ---------------------------------------------------------------------------
# # UAV follower tuning
# # ---------------------------------------------------------------------------

# # Important:
# # follow_distance = 0 means the UAV should stay aligned with the Husky,
# # not ahead and not behind.
# UAV_FOLLOW_DISTANCE = 0.0
# UAV_FOLLOW_HEIGHT = 12.0

# # uav1 stays left, uav2 stays right.
# UAV1_FOLLOW_LATERAL_OFFSET = UAV_SIDE_OFFSET
# UAV2_FOLLOW_LATERAL_OFFSET = -UAV_SIDE_OFFSET

# UAV_UPDATE_PERIOD = 0.1
# UAV_MAX_XY_SPEED = 9.0
# UAV_MAX_Z_SPEED = 1.2
# UAV_MAX_YAW_RATE = 0.9

# UAV_XY_GAIN = 2.0
# UAV_Z_GAIN = 0.35
# UAV_YAW_GAIN = 0.8
# UAV_HEADING_ALIGN_GAIN = 0.9

# UAV_MIN_FORWARD_SPEED = 0.0
# UAV_TARGET_SMOOTHING = 1.0
# UAV_XY_DEADBAND = 0.02
# UAV_Z_DEADBAND = 0.15
# UAV_YAW_DEADBAND = 0.18
# UAV_MIN_TRACK_SPEED = 0.0


# if DEBUG_ISOLATE_HUSKY_LOCAL:
#     ENABLE_UAV1 = False
#     ENABLE_UAV2 = False
#     ENABLE_BAG_RECORDING = True
#     RECORD_CAMERA_TOPICS = False


# # ---------------------------------------------------------------------------
# # Gazebo process and model-spawn helpers
# # ---------------------------------------------------------------------------

# def run_bg(cmd: str):
#     return subprocess.Popen(["bash", "-c", cmd])


# def load_husky_sdf_with_topic(topic_name: str) -> str:
#     husky_sdf = (MODELS_DIR / "husky" / "model.sdf").read_text()
#     husky_sdf = husky_sdf.replace(
#         "<topic>/cmd_vel</topic>",
#         f"<topic>{topic_name}</topic>",
#         1,
#     )
#     return husky_sdf


# def add_pose_publisher(sdf_text: str) -> str:
#     """Inject a Gazebo pose publisher so ROS can see world-truth Husky poses."""
#     if "ignition-gazebo-pose-publisher-system" in sdf_text:
#         return sdf_text

#     plugin = """
#     <plugin
#       filename="ignition-gazebo-pose-publisher-system"
#       name="ignition::gazebo::systems::PosePublisher">
#       <publish_link_pose>true</publish_link_pose>
#       <use_pose_vector_msg>true</use_pose_vector_msg>
#     </plugin>
# """

#     return sdf_text.replace("</model>", plugin + "\n  </model>", 1)


# def add_husky_marker(
#     sdf_text: str,
#     marker_name: str,
#     rgba: tuple[float, float, float, float],
# ) -> str:
#     marker = f"""
#     <link name="{marker_name}">
#       <pose>0 0 0.32 0 0 0</pose>
#       <collision name="collision">
#         <geometry>
#           <cylinder>
#             <radius>0.015</radius>
#             <length>0.25</length>
#           </cylinder>
#         </geometry>
#       </collision>
#       <visual name="visual">
#         <geometry>
#           <cylinder>
#             <radius>0.02</radius>
#             <length>0.2625</length>
#           </cylinder>
#         </geometry>
#         <material>
#           <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
#           <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
#           <emissive>{rgba[0] * 0.4} {rgba[1] * 0.4} {rgba[2] * 0.4} {rgba[3]}</emissive>
#         </material>
#       </visual>
#     </link>
#     <joint name="{marker_name}_joint" type="fixed">
#       <parent>base_link</parent>
#       <child>{marker_name}</child>
#     </joint>
# """

#     return sdf_text.replace("</model>", marker + "\n  </model>", 1)


# def write_husky_variant(
#     output_path: Path,
#     topic_name: str,
#     marker_name: str,
#     rgba: tuple[float, float, float, float],
# ) -> Path:
#     sdf_text = load_husky_sdf_with_topic(topic_name)
#     sdf_text = add_pose_publisher(sdf_text)
#     sdf_text = add_husky_marker(sdf_text, marker_name, rgba)
#     output_path.write_text(sdf_text)
#     return output_path


# def spawn_goal_marker(
#     world_name: str,
#     name: str,
#     xyz: tuple[float, float, float],
#     rgba: tuple[float, float, float, float],
# ):
#     marker_sdf = f"""<sdf version="1.7">
#   <model name="{name}">
#     <static>true</static>
#     <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>
#     <link name="marker_link">
#       <visual name="marker_visual">
#         <pose>0 0 1.0 0 0 0</pose>
#         <geometry>
#           <cylinder>
#             <radius>0.08</radius>
#             <length>5.0</length>
#           </cylinder>
#         </geometry>
#         <material>
#           <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
#           <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
#           <emissive>{rgba[0] * 0.5} {rgba[1] * 0.5} {rgba[2] * 0.5} {rgba[3]}</emissive>
#         </material>
#       </visual>
#     </link>
#   </model>
# </sdf>"""

#     one_line = marker_sdf.replace("\n", " ").replace('"', '\\"')

#     cmd = (
#         f"ign service -s /world/{world_name}/create "
#         f"--reqtype ignition.msgs.EntityFactory "
#         f"--reptype ignition.msgs.Boolean "
#         f"--timeout 5000 "
#         f'--req \'sdf: "{one_line}"\''
#     )

#     subprocess.run(
#         ["bash", "-c", cmd],
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )


# def spawn_uav(
#     *,
#     world_name: str,
#     name: str,
#     x: float,
#     y: float,
#     z: float,
#     qz: float,
#     qw: float,
# ):
#     spawn_cmd = """
# ign service -s /world/{world_name}/create \
# --reqtype ignition.msgs.EntityFactory \
# --reptype ignition.msgs.Boolean \
# --timeout 5000 \
# --req 'sdf_filename: "model://m100/model.sdf", name: "{name}",
# pose: {{position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{z: {qz}, w: {qw}}}}}'
# """.format(
#         world_name=world_name,
#         name=name,
#         x=x,
#         y=y,
#         z=z,
#         qz=qz,
#         qw=qw,
#     )

#     subprocess.run(["bash", "-c", spawn_cmd])


# def rgbd_camera_bridge_topics(
#     world_name: str,
#     model_name: str,
#     link_name: str,
#     sensor_name: str,
# ) -> list[str]:
#     prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"

#     return [
#         f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
#         f"{prefix}/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
#         f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
#         f"{prefix}/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
#     ]


# def camera_bridge_topics(
#     world_name: str,
#     model_name: str,
#     link_name: str,
#     sensor_name: str,
# ) -> list[str]:
#     prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"

#     return [
#         f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
#         f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
#     ]


# def husky_sensor_bridge_topics(world_name: str, model_name: str) -> list[str]:
#     base_prefix = f"/world/{world_name}/model/{model_name}/link/base_link/sensor"

#     topics = [
#         f"{base_prefix}/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
#         f"{base_prefix}/planar_laser/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
#         f"{base_prefix}/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
#     ]

#     topics.extend(
#         rgbd_camera_bridge_topics(
#             world_name,
#             model_name,
#             "base_link",
#             "camera_front",
#         )
#     )
#     topics.extend(
#         rgbd_camera_bridge_topics(
#             world_name,
#             model_name,
#             "base_link",
#             "camera_down",
#         )
#     )
#     topics.extend(
#         rgbd_camera_bridge_topics(
#             world_name,
#             model_name,
#             "tilt_gimbal_link",
#             "camera_pan_tilt",
#         )
#     )

#     return topics


# def uav_bridge_topics(world_name: str, uav_name: str) -> list[str]:
#     topics = [
#         f"/{uav_name}/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
#         f"/{uav_name}/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
#         f"/model/{uav_name}/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
#         f"/model/{uav_name}/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
#         f"/model/{uav_name}/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
#         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
#         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
#         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/air_pressure/air_pressure@sensor_msgs/msg/FluidPressure[ignition.msgs.FluidPressure",
#         f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/magnetometer/magnetometer@sensor_msgs/msg/MagneticField[ignition.msgs.Magnetometer",
#     ]

#     topics.extend(
#         camera_bridge_topics(
#             world_name,
#             uav_name,
#             "base_link",
#             "camera_front",
#         )
#     )

#     return topics


# def build_bag_topics(
#     world_name: str,
#     *,
#     include_uav1: bool,
#     include_uav2: bool,
#     include_camera_topics: bool,
# ) -> list[str]:
#     """Return the useful topic set for hybrid trajectory dataset collection."""

#     topics = [
#         "/clock",
#         f"/world/{world_name}/dynamic_pose/info",

#         # Ego Husky control/state.
#         "/cmd_vel",
#         "/husky_local/controller_state",
#         "/husky_local/obstacle_action",
#         "/husky_local/obstacle_clearance",
#         "/episode/husky_local/start",
#         "/episode/husky_local/goal",
#         "/model/husky_local/odometry",

#         # Ego Husky local perception.
#         f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan",
#         f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
#         f"/world/{world_name}/model/husky_local/link/base_link/sensor/imu_sensor/imu",
#     ]

#     if include_camera_topics:
#         topics.extend(
#             [
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/image",
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/depth_image",
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/camera_info",
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/points",
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/image",
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/depth_image",
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/camera_info",
#                 f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/points",
#             ]
#         )

#     if include_uav1:
#         topics.extend(
#             [
#                 "/episode/uav1/start",
#                 "/episode/uav1/goal",
#                 "/model/uav1/odometry",
#                 "/uav1/ready",
#                 f"/world/{world_name}/model/uav1/link/base_link/sensor/front_laser/scan/points",
#                 f"/world/{world_name}/model/uav1/link/base_link/sensor/imu_sensor/imu",
#                 f"/world/{world_name}/model/uav1/link/base_link/sensor/air_pressure/air_pressure",
#                 f"/world/{world_name}/model/uav1/link/base_link/sensor/magnetometer/magnetometer",
#             ]
#         )

#         if include_camera_topics:
#             topics.extend(
#                 [
#                     f"/world/{world_name}/model/uav1/link/base_link/sensor/camera_front/image",
#                     f"/world/{world_name}/model/uav1/link/base_link/sensor/camera_front/camera_info",
#                 ]
#             )

#     if include_uav2:
#         topics.extend(
#             [
#                 "/episode/uav2/start",
#                 "/episode/uav2/goal",
#                 "/model/uav2/odometry",
#                 "/uav2/ready",
#                 f"/world/{world_name}/model/uav2/link/base_link/sensor/front_laser/scan/points",
#                 f"/world/{world_name}/model/uav2/link/base_link/sensor/imu_sensor/imu",
#                 f"/world/{world_name}/model/uav2/link/base_link/sensor/air_pressure/air_pressure",
#                 f"/world/{world_name}/model/uav2/link/base_link/sensor/magnetometer/magnetometer",
#             ]
#         )

#         if include_camera_topics:
#             topics.extend(
#                 [
#                     f"/world/{world_name}/model/uav2/link/base_link/sensor/camera_front/image",
#                     f"/world/{world_name}/model/uav2/link/base_link/sensor/camera_front/camera_info",
#                 ]
#             )

#     return topics


# # ---------------------------------------------------------------------------
# # Start Gazebo and spawn entities
# # ---------------------------------------------------------------------------

# os.environ["IGN_GAZEBO_RESOURCE_PATH"] = (
#     MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
# )
# os.environ["GZ_SIM_RESOURCE_PATH"] = (
#     MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
# )

# setup_terminal_tee(RUN_LOG_PATH)

# subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
# subprocess.run(["bash", "-c", "pkill -f ign || true"])
# subprocess.run(["bash", "-c", f"pkill -f {OMNET_BIN} || true"])

# log_event(f"START timestamp: {RUN_START_DT.isoformat(timespec='seconds')}")
# log_event(f"Run log file: {RUN_LOG_PATH}")
# log_event(f"World path: {WORLD}")
# log_event("Starting Gazebo...")

# gz = run_bg(f"ign gazebo {WORLD}")

# time.sleep(5)

# log_event("Waiting for Baylands world to fully load...")
# time.sleep(40)

# log_event("Spawning Husky ego: husky_local")

# husky1_sdf_path = write_husky_variant(
#     MODELS_DIR / "husky" / "model_red_tag.sdf",
#     "/cmd_vel",
#     "flag_marker_red",
#     (0.95, 0.12, 0.12, 1.0),
# )

# spawn_husky = (
#     "ign service -s /world/{world_name}/create "
#     "--reqtype ignition.msgs.EntityFactory "
#     "--reptype ignition.msgs.Boolean "
#     "--timeout 5000 "
#     "--req 'sdf_filename: \"{sdf_path}\", name: \"husky_local\", "
#     "pose: {{position: {{x: {spawn_x}, y: {spawn_y}, z: {spawn_z}}}, "
#     "orientation: {{z: {spawn_qz}, w: {spawn_qw}}}}}'"
# ).format(
#     world_name=WORLD_NAME,
#     sdf_path=husky1_sdf_path,
#     spawn_x=SPAWN_X,
#     spawn_y=SPAWN_Y,
#     spawn_z=SPAWN_Z,
#     spawn_qz=HUSKY1_SPAWN_QZ,
#     spawn_qw=HUSKY1_SPAWN_QW,
# )

# subprocess.run(["bash", "-c", spawn_husky])
# time.sleep(5)


# if ENABLE_UAV1:
#     log_event(
#         f"Spawning UAV left support: uav1 at "
#         f"({UAV1_X:.3f}, {UAV1_Y:.3f}, {UAV1_Z:.3f})"
#     )

#     spawn_uav(
#         world_name=WORLD_NAME,
#         name="uav1",
#         x=UAV1_X,
#         y=UAV1_Y,
#         z=UAV1_Z,
#         qz=UAV_SPAWN_QZ,
#         qw=UAV_SPAWN_QW,
#     )

#     time.sleep(5)


# if ENABLE_UAV2:
#     log_event(
#         f"Spawning UAV right support: uav2 at "
#         f"({UAV2_X:.3f}, {UAV2_Y:.3f}, {UAV2_Z:.3f})"
#     )

#     spawn_uav(
#         world_name=WORLD_NAME,
#         name="uav2",
#         x=UAV2_X,
#         y=UAV2_Y,
#         z=UAV2_Z,
#         qz=UAV2_SPAWN_QZ,
#         qw=UAV2_SPAWN_QW,
#     )

#     time.sleep(5)


# log_event("Spawning visible goal markers...")

# spawn_goal_marker(
#     WORLD_NAME,
#     "goal_husky_local",
#     (
#         WORLD_HUSKY1_GOAL[0],
#         WORLD_HUSKY1_GOAL[1],
#         GROUND_MARKER_Z,
#     ),
#     (0.95, 0.12, 0.12, 1.0),
# )

# if ENABLE_UAV1:
#     spawn_goal_marker(
#         WORLD_NAME,
#         "goal_uav1",
#         (
#             WORLD_UAV1_GOAL[0],
#             WORLD_UAV1_GOAL[1],
#             GROUND_MARKER_Z,
#         ),
#         (0.95, 0.85, 0.12, 1.0),
#     )

# if ENABLE_UAV2:
#     spawn_goal_marker(
#         WORLD_NAME,
#         "goal_uav2",
#         (
#             WORLD_UAV2_GOAL[0],
#             WORLD_UAV2_GOAL[1],
#             GROUND_MARKER_Z,
#         ),
#         (0.20, 0.90, 0.95, 1.0),
#     )

# time.sleep(1)


# # ---------------------------------------------------------------------------
# # Start ROS-Gazebo bridge
# # ---------------------------------------------------------------------------

# log_event("Starting ROS-Gazebo bridge...")

# bridge_topics = [
#     "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
#     "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
#     f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
# ]

# bridge_topics.extend(
#     husky_sensor_bridge_topics(
#         WORLD_NAME,
#         "husky_local",
#     )
# )

# if ENABLE_UAV1:
#     bridge_topics.extend(
#         uav_bridge_topics(
#             WORLD_NAME,
#             "uav1",
#         )
#     )

# if ENABLE_UAV2:
#     bridge_topics.extend(
#         uav_bridge_topics(
#             WORLD_NAME,
#             "uav2",
#         )
#     )

# bridge_cmd = (
#     "source /opt/ros/humble/setup.bash && "
#     "ros2 run ros_gz_bridge parameter_bridge "
#     + " ".join(bridge_topics)
# )

# bridge = run_bg(bridge_cmd)
# time.sleep(2)


# # ---------------------------------------------------------------------------
# # Optional visual tools
# # ---------------------------------------------------------------------------

# rviz = None
# camera_view = None

# if ENABLE_RVIZ:
#     log_event(f"Starting RViz with config: {RVIZ_CONFIG_PATH}")
#     rviz_cmd = (
#         "source /opt/ros/humble/setup.bash && "
#         f"rviz2 -d {RVIZ_CONFIG_PATH}"
#     )
#     rviz = run_bg(rviz_cmd)
#     time.sleep(2)

# if ENABLE_CAMERA_VIEW:
#     log_event("Starting camera viewer...")
#     camera_cmd = (
#         "source /opt/ros/humble/setup.bash && "
#         "ros2 run rqt_image_view rqt_image_view"
#     )
#     camera_view = run_bg(camera_cmd)
#     time.sleep(2)


# # ---------------------------------------------------------------------------
# # OMNeT++
# # ---------------------------------------------------------------------------

# omnet = None

# if ENABLE_UAV1 or ENABLE_UAV2:
#     log_event("OMNeT++ relay disabled in this data-collection runner.")
#     log_event("Network features will be exported as placeholder edge fields unless OMNeT++ topics are added later.")


# # ---------------------------------------------------------------------------
# # Start rosbag recording
# # ---------------------------------------------------------------------------

# BAG_DIR = Path.home() / "Documents/Thesis/03_dataset/bags"
# BAG_DIR.mkdir(parents=True, exist_ok=True)

# run_name = "run_dataset_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# bag_path = BAG_DIR / run_name

# recorder = None

# if ENABLE_BAG_RECORDING:
#     bag_topics = build_bag_topics(
#         WORLD_NAME,
#         include_uav1=ENABLE_UAV1,
#         include_uav2=ENABLE_UAV2,
#         include_camera_topics=RECORD_CAMERA_TOPICS,
#     )

#     log_event(f"Recording bag: {bag_path}")
#     log_event("Bag topic set for hybrid trajectory collection:")
#     for topic in bag_topics:
#         log_event(f"  - {topic}")

#     record_cmd = (
#         "source /opt/ros/humble/setup.bash && "
#         f"ros2 bag record -o {bag_path} "
#         + " ".join(bag_topics)
#     )

#     recorder = run_bg(record_cmd)
# else:
#     log_event("Bag recording disabled. Set ENABLE_BAG_RECORDING = True to record a run.")


# if DEBUG_ISOLATE_HUSKY_LOCAL:
#     log_event(
#         "DEBUG isolate mode: only husky_local is spawned. "
#         "Bag recording remains enabled."
#     )


# # ---------------------------------------------------------------------------
# # ROS 2 nodes
# # ---------------------------------------------------------------------------

# log_event("==============================")
# log_event("RULE-BASED DATASET COLLECTION MODE")
# log_event("AGENTS: husky_local + uav1(left) + uav2(right)")
# log_event("==============================")
# log_event("Press Play in Gazebo.")
# log_event("Let the simulation run until the route is completed or enough data is collected.")
# log_event("Press Ctrl+C in this terminal to stop and close the bag cleanly.")

# rclpy.init()

# HUSKY1_GOAL = offset_goal_along_path(
#     WORLD_HUSKY1_GOAL,
#     (SPAWN_X, SPAWN_Y, SPAWN_Z),
#     GOAL_STOP_OFFSET,
# )

# UAV1_GOAL = offset_goal_along_path(
#     WORLD_UAV1_GOAL,
#     (UAV1_X, UAV1_Y, UAV1_Z),
#     GOAL_STOP_OFFSET,
# )

# UAV2_GOAL = offset_goal_along_path(
#     WORLD_UAV2_GOAL,
#     (UAV2_X, UAV2_Y, UAV2_Z),
#     GOAL_STOP_OFFSET,
# )

# log_event(
#     "Controller goals with stop offset: "
#     f"husky_local=({HUSKY1_GOAL[0]:.3f}, {HUSKY1_GOAL[1]:.3f}), "
#     f"uav1=({UAV1_GOAL[0]:.3f}, {UAV1_GOAL[1]:.3f}), "
#     f"uav2=({UAV2_GOAL[0]:.3f}, {UAV2_GOAL[1]:.3f})"
# )

# log_event(
#     f"Visible goal marker remains at "
#     f"({WORLD_HUSKY1_GOAL[0]:.3f}, {WORLD_HUSKY1_GOAL[1]:.3f}); "
#     f"controller stop tolerance is {GOAL_TOLERANCE:.2f} m."
# )


# start_goals = {
#     "husky_local": {
#         "start": (SPAWN_X, SPAWN_Y, SPAWN_Z),
#         "goal": WORLD_HUSKY1_GOAL,
#     },
# }

# if ENABLE_UAV1:
#     start_goals["uav1"] = {
#         "start": (UAV1_X, UAV1_Y, UAV1_Z),
#         "goal": WORLD_UAV1_GOAL,
#     }

# if ENABLE_UAV2:
#     start_goals["uav2"] = {
#         "start": (UAV2_X, UAV2_Y, UAV2_Z),
#         "goal": WORLD_UAV2_GOAL,
#     }

# episode_metadata = EpisodeMetadataPublisher(
#     world_name=WORLD_NAME,
#     start_goals=start_goals,
# )


# # Ego Husky nodes.
# husky1_obstacle_action_topic = "/husky_local/obstacle_action"
# husky1_obstacle_clearance_topic = "/husky_local/obstacle_clearance"
# husky1_controller_state_topic = "/husky_local/controller_state"

# obstacle_detector = ObstacleDetectionNode(
#     node_name="husky_local_obstacle_detector",
#     scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
#     action_topic=husky1_obstacle_action_topic,
#     clearance_topic=husky1_obstacle_clearance_topic,
#     pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
#     front_half_angle_deg=OBSTACLE_FRONT_HALF_ANGLE_DEG,
#     side_angle_deg=OBSTACLE_SIDE_ANGLE_DEG,
#     stop_distance=OBSTACLE_STOP_DISTANCE,
#     caution_distance=OBSTACLE_CAUTION_DISTANCE,
# )

# driver = ModelHuskyDriver(
#     node_name="model_husky_driver_1",
#     cmd_topic="/cmd_vel",
#     odom_topic="/model/husky_local/odometry",
#     world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
#     uav_ready_topic=None,
#     require_uav_ready=False,
#     obstacle_action_topic=husky1_obstacle_action_topic,
#     obstacle_clearance_topic=husky1_obstacle_clearance_topic,
#     state_topic=husky1_controller_state_topic,
#     goal_xyz=HUSKY1_GOAL,
#     world_goal_xyz=HUSKY1_GOAL,
#     bootstrap_seconds=BOOTSTRAP_SECONDS,
#     bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
#     control_period=CONTROL_PERIOD,
#     cmd_linear_gain=CMD_LINEAR_GAIN,
#     cmd_angular_gain=CMD_ANGULAR_GAIN,
#     min_linear_speed=MIN_LINEAR_SPEED,
#     max_linear_speed=MAX_LINEAR_SPEED,
#     max_angular_speed=MAX_ANGULAR_SPEED,
#     heading_deadband=HEADING_DEADBAND,
#     goal_tolerance=GOAL_TOLERANCE,
#     stuck_timeout_seconds=STUCK_TIMEOUT_SECONDS,
#     stuck_progress_distance=STUCK_PROGRESS_DISTANCE,
#     stuck_reverse_speed=STUCK_REVERSE_SPEED,
#     stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
#     stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
# )


# # UAV follower nodes.
# follower1 = None
# follower2 = None

# if ENABLE_UAV1:
#     follower1 = UavFollower(
#         node_name="uav1_follower",
#         husky_odom_topic="/model/husky_local/odometry",
#         uav_odom_topic="/model/uav1/odometry",
#         world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
#         husky_model_name="husky_local",
#         uav_model_name="uav1",
#         uav_name="uav1",
#         follow_distance=UAV_FOLLOW_DISTANCE,
#         follow_lateral_offset=UAV1_FOLLOW_LATERAL_OFFSET,
#         follow_height=UAV_FOLLOW_HEIGHT,
#         ready_topic="/uav1/ready",
#         update_period=UAV_UPDATE_PERIOD,
#         max_xy_speed=UAV_MAX_XY_SPEED,
#         max_z_speed=UAV_MAX_Z_SPEED,
#         max_yaw_rate=UAV_MAX_YAW_RATE,
#         xy_gain=UAV_XY_GAIN,
#         z_gain=UAV_Z_GAIN,
#         yaw_gain=UAV_YAW_GAIN,
#         heading_align_gain=UAV_HEADING_ALIGN_GAIN,
#         min_forward_speed=UAV_MIN_FORWARD_SPEED,
#         target_smoothing=UAV_TARGET_SMOOTHING,
#         xy_deadband=UAV_XY_DEADBAND,
#         z_deadband=UAV_Z_DEADBAND,
#         yaw_deadband=UAV_YAW_DEADBAND,
#         min_track_speed=UAV_MIN_TRACK_SPEED,
#         husky_spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
#         husky_spawn_yaw=HUSKY1_SPAWN_YAW,
#         uav_spawn_xyz=(UAV1_X, UAV1_Y, UAV1_Z),
#         uav_spawn_yaw=UAV_SPAWN_YAW,
#     )

# if ENABLE_UAV2:
#     follower2 = UavFollower(
#         node_name="uav2_follower",
#         husky_odom_topic="/model/husky_local/odometry",
#         uav_odom_topic="/model/uav2/odometry",
#         world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
#         husky_model_name="husky_local",
#         uav_model_name="uav2",
#         uav_name="uav2",
#         follow_distance=UAV_FOLLOW_DISTANCE,
#         follow_lateral_offset=UAV2_FOLLOW_LATERAL_OFFSET,
#         follow_height=UAV_FOLLOW_HEIGHT,
#         ready_topic="/uav2/ready",
#         update_period=UAV_UPDATE_PERIOD,
#         max_xy_speed=UAV_MAX_XY_SPEED,
#         max_z_speed=UAV_MAX_Z_SPEED,
#         max_yaw_rate=UAV_MAX_YAW_RATE,
#         xy_gain=UAV_XY_GAIN,
#         z_gain=UAV_Z_GAIN,
#         yaw_gain=UAV_YAW_GAIN,
#         heading_align_gain=UAV_HEADING_ALIGN_GAIN,
#         min_forward_speed=UAV_MIN_FORWARD_SPEED,
#         target_smoothing=UAV_TARGET_SMOOTHING,
#         xy_deadband=UAV_XY_DEADBAND,
#         z_deadband=UAV_Z_DEADBAND,
#         yaw_deadband=UAV_YAW_DEADBAND,
#         min_track_speed=UAV_MIN_TRACK_SPEED,
#         husky_spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
#         husky_spawn_yaw=HUSKY1_SPAWN_YAW,
#         uav_spawn_xyz=(UAV2_X, UAV2_Y, UAV2_Z),
#         uav_spawn_yaw=UAV2_SPAWN_YAW,
#     )


# executor = MultiThreadedExecutor()

# executor.add_node(episode_metadata)
# executor.add_node(obstacle_detector)
# executor.add_node(driver)

# if follower1 is not None:
#     executor.add_node(follower1)

# if follower2 is not None:
#     executor.add_node(follower2)


# try:
#     executor.spin()

# except KeyboardInterrupt:
#     log_event("Stopping dataset collection run...")

# finally:
#     managed_nodes = [
#         episode_metadata,
#         obstacle_detector,
#         driver,
#     ]

#     if follower1 is not None:
#         managed_nodes.append(follower1)

#     if follower2 is not None:
#         managed_nodes.append(follower2)

#     for node in managed_nodes:
#         with suppress(Exception):
#             executor.remove_node(node)

#     executor.shutdown(timeout_sec=2.0)
#     time.sleep(0.25)

#     for node in managed_nodes:
#         with suppress(Exception):
#             node.destroy_node()

#     if rclpy.ok():
#         rclpy.shutdown()

#     if recorder is not None:
#         log_event("Stopping rosbag recorder cleanly...")
#         with suppress(Exception):
#             recorder.send_signal(signal.SIGINT)
#         time.sleep(2)

#     log_event(f"STOP timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}")
#     log_event("Stopping bridge, OMNeT++, RViz/camera viewer, and Gazebo...")

#     managed_processes = [
#         bridge,
#         rviz,
#         camera_view,
#         omnet,
#         gz,
#     ]

#     for proc in managed_processes:
#         if proc is None:
#             continue
#         with suppress(Exception):
#             proc.send_signal(signal.SIGINT)

#     time.sleep(2)

#     for proc in managed_processes:
#         if proc is None:
#             continue
#         with suppress(Exception):
#             if proc.poll() is None:
#                 proc.terminate()

#     log_event("All processes stopped cleanly.")
#     log_event(f"Bag output folder should be under: {BAG_DIR}")
#     log_event(f"Latest bag for this run: {bag_path}")

#     close_terminal_tee()



"""Launch the rule-based two-UAV data-collection simulation pipeline.

This runner starts Gazebo, spawns one ego Husky and two UAV support agents,
publishes episode metadata, records a rosbag, and runs the rule-based Husky/UAV
controllers.

Purpose of this runner:
- Generate training/evaluation data for trajectory prediction models.
- Record synchronized UGV, UAV, lidar, pointcloud, goal, command, and controller-state topics.
- Keep the environment and topic structure compatible with the hybrid two-UAV dataset exporter.

Formation design:
- husky_local is the ego UGV.
- uav1 stays on the left side of the Husky.
- uav2 stays on the right side of the Husky.
- Both UAVs stay aligned with the Husky, not ahead and not behind.
- Both UAVs face the Husky route direction so their forward sensors look toward the path area.

Dataset target:
The generated bags are later exported into frames.jsonl and .npy assets by:

    03_dataset/exporters/export_hybrid_maneuver_dataset.py

Main recorded modalities:
- past Husky trajectory and command history
- Husky local lidar / front pointcloud
- UAV1 and UAV2 odometry
- UAV1 and UAV2 forward pointclouds
- inter-agent relations through exported graph features
- goal/start metadata
- controller labels such as go_to_goal, avoid_left, avoid_right, arrived
"""

import datetime
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RULE_BASED_ROOT = SCRIPT_DIR.parent
if str(RULE_BASED_ROOT) not in sys.path:
    sys.path.insert(0, str(RULE_BASED_ROOT))

import rclpy
from rclpy.executors import MultiThreadedExecutor

from controllers.episode_metadata import EpisodeMetadataPublisher
from controllers.husky_model_driver import ModelHuskyDriver
from controllers.obstacle_detection import ObstacleDetectionNode
from controllers.uav_follower import UavFollower
from project_paths import MODELS_DIR, OMNET_DIR, RVIZ_CONFIG_PATH, WORLD_SDF_PATH


# ---------------------------------------------------------------------------
# Global paths and runtime logging
# ---------------------------------------------------------------------------

WORLD = str(WORLD_SDF_PATH)
WORLD_NAME = "baylands"

MODEL_PATH = str(MODELS_DIR)
OMNET_BIN = OMNET_DIR / "onmetpp"
OMNET_CONFIG = "WifiRelay"

RUN_START_DT = datetime.datetime.now()

LOG_DIR = Path.home() / "Documents/Thesis/03_dataset/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG_PATH = LOG_DIR / f"rule_based_dataset_{RUN_START_DT.strftime('%Y%m%d_%H%M%S')}.log"

TEE_PROCESS = None
ORIGINAL_STDOUT_FD = None
ORIGINAL_STDERR_FD = None


def setup_terminal_tee(log_path: Path):
    global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD

    if TEE_PROCESS is not None:
        return

    ORIGINAL_STDOUT_FD = os.dup(sys.__stdout__.fileno())
    ORIGINAL_STDERR_FD = os.dup(sys.__stderr__.fileno())

    TEE_PROCESS = subprocess.Popen(
        ["tee", "-a", str(log_path)],
        stdin=subprocess.PIPE,
        stdout=ORIGINAL_STDOUT_FD,
        stderr=ORIGINAL_STDERR_FD,
        bufsize=0,
    )

    os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stdout__.fileno())
    os.dup2(TEE_PROCESS.stdin.fileno(), sys.__stderr__.fileno())


def close_terminal_tee():
    global TEE_PROCESS, ORIGINAL_STDOUT_FD, ORIGINAL_STDERR_FD

    if TEE_PROCESS is None:
        return

    with suppress(Exception):
        sys.stdout.flush()
    with suppress(Exception):
        sys.stderr.flush()

    if ORIGINAL_STDOUT_FD is not None:
        with suppress(Exception):
            os.dup2(ORIGINAL_STDOUT_FD, sys.__stdout__.fileno())

    if ORIGINAL_STDERR_FD is not None:
        with suppress(Exception):
            os.dup2(ORIGINAL_STDERR_FD, sys.__stderr__.fileno())

    with suppress(Exception):
        if TEE_PROCESS.stdin is not None:
            TEE_PROCESS.stdin.close()

    with suppress(Exception):
        TEE_PROCESS.wait(timeout=2.0)

    if ORIGINAL_STDOUT_FD is not None:
        with suppress(Exception):
            os.close(ORIGINAL_STDOUT_FD)

    if ORIGINAL_STDERR_FD is not None:
        with suppress(Exception):
            os.close(ORIGINAL_STDERR_FD)

    TEE_PROCESS = None
    ORIGINAL_STDOUT_FD = None
    ORIGINAL_STDERR_FD = None


def log_event(message: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    with suppress(Exception):
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

# Baylands route start position for the ego Husky.
SPAWN_X, SPAWN_Y, SPAWN_Z = 84.6951, 24.1579, 0.6490

HUSKY1_SPAWN_YAW = math.pi
HUSKY1_SPAWN_QZ = math.sin(HUSKY1_SPAWN_YAW / 2.0)
HUSKY1_SPAWN_QW = math.cos(HUSKY1_SPAWN_YAW / 2.0)

# Two UAVs are placed on the left/right side of the Husky.
# They are not placed ahead or behind the Husky.
UAV_INITIAL_FORWARD_OFFSET = 0.0
UAV_SIDE_OFFSET = 6.0

# Important:
# Use a high initial altitude so UAVs do not collide with trees/canopies
# during the first seconds of simulation.
UAV_INITIAL_HEIGHT = 22.0

# The UAV controller will later rotate the UAVs to the Husky heading.
# This spawn yaw only affects the initial pose.
UAV1_SPAWN_YAW = HUSKY1_SPAWN_YAW
UAV2_SPAWN_YAW = HUSKY1_SPAWN_YAW

UAV1_SPAWN_QZ = math.sin(UAV1_SPAWN_YAW / 2.0)
UAV1_SPAWN_QW = math.cos(UAV1_SPAWN_YAW / 2.0)

UAV2_SPAWN_QZ = math.sin(UAV2_SPAWN_YAW / 2.0)
UAV2_SPAWN_QW = math.cos(UAV2_SPAWN_YAW / 2.0)


def body_offset_to_world(
    base_xyz: tuple[float, float, float],
    base_yaw: float,
    forward_offset: float,
    lateral_offset: float,
    height_offset: float,
) -> tuple[float, float, float]:
    """Convert a body-frame offset around the Husky into world coordinates.

    forward_offset:
        positive means in front of the Husky.
        negative means behind the Husky.

    lateral_offset:
        positive means left side of the Husky.
        negative means right side of the Husky.
    """
    x = (
        float(base_xyz[0])
        + forward_offset * math.cos(base_yaw)
        - lateral_offset * math.sin(base_yaw)
    )
    y = (
        float(base_xyz[1])
        + forward_offset * math.sin(base_yaw)
        + lateral_offset * math.cos(base_yaw)
    )
    z = float(base_xyz[2]) + height_offset

    return (x, y, z)


UAV1_X, UAV1_Y, UAV1_Z = body_offset_to_world(
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    HUSKY1_SPAWN_YAW,
    UAV_INITIAL_FORWARD_OFFSET,
    UAV_SIDE_OFFSET,
    UAV_INITIAL_HEIGHT,
)

UAV2_X, UAV2_Y, UAV2_Z = body_offset_to_world(
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    HUSKY1_SPAWN_YAW,
    UAV_INITIAL_FORWARD_OFFSET,
    -UAV_SIDE_OFFSET,
    UAV_INITIAL_HEIGHT,
)


def offset_goal_along_path(
    world_goal: tuple[float, float, float],
    start_xyz: tuple[float, float, float],
    offset_distance: float,
) -> tuple[float, float, float]:
    """Shift the stopping target along the start->goal line by a fixed distance."""
    dx = float(world_goal[0]) - float(start_xyz[0])
    dy = float(world_goal[1]) - float(start_xyz[1])
    norm = math.hypot(dx, dy)

    if norm < 1e-6 or abs(offset_distance) < 1e-9:
        return world_goal

    ux = dx / norm
    uy = dy / norm

    return (
        float(world_goal[0]) + offset_distance * ux,
        float(world_goal[1]) + offset_distance * uy,
        float(world_goal[2]),
    )


# Base goal from the longer Baylands route.
# Pulling it slightly back keeps data collection shorter but preserves the same route direction.
RAW_WORLD_SHARED_GOAL = (-54.4904, -96.9701, -0.2737)
GOAL_WORLD_PULLBACK = 13.0

WORLD_SHARED_GOAL = offset_goal_along_path(
    RAW_WORLD_SHARED_GOAL,
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    -GOAL_WORLD_PULLBACK,
)

WORLD_HUSKY1_GOAL = WORLD_SHARED_GOAL
WORLD_UAV1_GOAL = WORLD_SHARED_GOAL
WORLD_UAV2_GOAL = WORLD_SHARED_GOAL

GROUND_MARKER_Z = 0.2025
GOAL_STOP_OFFSET = -0.5


# ---------------------------------------------------------------------------
# Data-collection switches
# ---------------------------------------------------------------------------

ENABLE_UAV1 = True
ENABLE_UAV2 = True

ENABLE_BAG_RECORDING = True
ENABLE_RVIZ = False
ENABLE_CAMERA_VIEW = False

# Camera topics are useful but can make bags very large.
# Keep this False for normal trajectory/graph experiments.
RECORD_CAMERA_TOPICS = False

# Useful when debugging only the main Husky.
DEBUG_ISOLATE_HUSKY_LOCAL = False


# ---------------------------------------------------------------------------
# Controller tuning for stable teacher behavior
# ---------------------------------------------------------------------------

BOOTSTRAP_SECONDS = 3.0
BOOTSTRAP_LINEAR_SPEED = 0.8

CONTROL_PERIOD = 0.1

CMD_LINEAR_GAIN = 1.45
CMD_ANGULAR_GAIN = 1.15

MIN_LINEAR_SPEED = 1.5
MAX_LINEAR_SPEED = 2.0
MAX_ANGULAR_SPEED = 0.85

HEADING_DEADBAND = 0.12
GOAL_TOLERANCE = 1.5

STUCK_TIMEOUT_SECONDS = 3.0
STUCK_PROGRESS_DISTANCE = 0.15
STUCK_REVERSE_SPEED = -0.8
STUCK_REVERSE_SECONDS = 2.0
STUCK_BOOTSTRAP_SECONDS = 2.0

OBSTACLE_FRONT_HALF_ANGLE_DEG = 45.0
OBSTACLE_SIDE_ANGLE_DEG = 65.0
OBSTACLE_STOP_DISTANCE = 1.8
OBSTACLE_CAUTION_DISTANCE = 3.2


# ---------------------------------------------------------------------------
# UAV follower tuning
# ---------------------------------------------------------------------------

# Important:
# follow_distance = 0 means the UAV should stay aligned with the Husky,
# not ahead and not behind.
UAV_FOLLOW_DISTANCE = 0.0

# Keep UAVs high enough for the tree-heavy part of Baylands.
UAV_FOLLOW_HEIGHT = 22.0
UAV_MIN_WORLD_ALTITUDE = 18.0
UAV_MIN_FOLLOW_ALTITUDE = 18.0

# uav1 stays left, uav2 stays right.
UAV1_FOLLOW_LATERAL_OFFSET = UAV_SIDE_OFFSET
UAV2_FOLLOW_LATERAL_OFFSET = -UAV_SIDE_OFFSET

UAV_UPDATE_PERIOD = 0.1
UAV_MAX_XY_SPEED = 9.0
UAV_MAX_Z_SPEED = 2.0
UAV_MAX_YAW_RATE = 1.2

UAV_XY_GAIN = 2.0
UAV_Z_GAIN = 0.45
UAV_YAW_GAIN = 1.0
UAV_HEADING_ALIGN_GAIN = 0.9

UAV_MIN_FORWARD_SPEED = 0.0
UAV_TARGET_SMOOTHING = 0.85
UAV_XY_DEADBAND = 0.05
UAV_Z_DEADBAND = 0.35
UAV_YAW_DEADBAND = 0.08
UAV_MIN_TRACK_SPEED = 0.0

# Important for the project:
# The UAVs should face the route direction, not look inward toward the Husky.
UAV_YAW_MODE = "husky_heading"


if DEBUG_ISOLATE_HUSKY_LOCAL:
    ENABLE_UAV1 = False
    ENABLE_UAV2 = False
    ENABLE_BAG_RECORDING = True
    RECORD_CAMERA_TOPICS = False


# ---------------------------------------------------------------------------
# Gazebo process and model-spawn helpers
# ---------------------------------------------------------------------------

def run_bg(cmd: str):
    return subprocess.Popen(["bash", "-c", cmd])


def load_husky_sdf_with_topic(topic_name: str) -> str:
    husky_sdf = (MODELS_DIR / "husky" / "model.sdf").read_text()
    husky_sdf = husky_sdf.replace(
        "<topic>/cmd_vel</topic>",
        f"<topic>{topic_name}</topic>",
        1,
    )
    return husky_sdf


def add_pose_publisher(sdf_text: str) -> str:
    """Inject a Gazebo pose publisher so ROS can see world-truth Husky poses."""
    if "ignition-gazebo-pose-publisher-system" in sdf_text:
        return sdf_text

    plugin = """
    <plugin
      filename="ignition-gazebo-pose-publisher-system"
      name="ignition::gazebo::systems::PosePublisher">
      <publish_link_pose>true</publish_link_pose>
      <use_pose_vector_msg>true</use_pose_vector_msg>
    </plugin>
"""

    return sdf_text.replace("</model>", plugin + "\n  </model>", 1)


def add_husky_marker(
    sdf_text: str,
    marker_name: str,
    rgba: tuple[float, float, float, float],
) -> str:
    marker = f"""
    <link name="{marker_name}">
      <pose>0 0 0.32 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.015</radius>
            <length>0.25</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.2625</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
          <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          <emissive>{rgba[0] * 0.4} {rgba[1] * 0.4} {rgba[2] * 0.4} {rgba[3]}</emissive>
        </material>
      </visual>
    </link>
    <joint name="{marker_name}_joint" type="fixed">
      <parent>base_link</parent>
      <child>{marker_name}</child>
    </joint>
"""

    return sdf_text.replace("</model>", marker + "\n  </model>", 1)


def write_husky_variant(
    output_path: Path,
    topic_name: str,
    marker_name: str,
    rgba: tuple[float, float, float, float],
) -> Path:
    sdf_text = load_husky_sdf_with_topic(topic_name)
    sdf_text = add_pose_publisher(sdf_text)
    sdf_text = add_husky_marker(sdf_text, marker_name, rgba)
    output_path.write_text(sdf_text)
    return output_path


def spawn_goal_marker(
    world_name: str,
    name: str,
    xyz: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
):
    marker_sdf = f"""<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>
    <link name="marker_link">
      <visual name="marker_visual">
        <pose>0 0 1.0 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>5.0</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
          <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          <emissive>{rgba[0] * 0.5} {rgba[1] * 0.5} {rgba[2] * 0.5} {rgba[3]}</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    one_line = marker_sdf.replace("\n", " ").replace('"', '\\"')

    cmd = (
        f"ign service -s /world/{world_name}/create "
        f"--reqtype ignition.msgs.EntityFactory "
        f"--reptype ignition.msgs.Boolean "
        f"--timeout 5000 "
        f'--req \'sdf: "{one_line}"\''
    )

    subprocess.run(
        ["bash", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def spawn_uav(
    *,
    world_name: str,
    name: str,
    x: float,
    y: float,
    z: float,
    qz: float,
    qw: float,
):
    spawn_cmd = """
ign service -s /world/{world_name}/create \
--reqtype ignition.msgs.EntityFactory \
--reptype ignition.msgs.Boolean \
--timeout 5000 \
--req 'sdf_filename: "model://m100/model.sdf", name: "{name}",
pose: {{position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{z: {qz}, w: {qw}}}}}'
""".format(
        world_name=world_name,
        name=name,
        x=x,
        y=y,
        z=z,
        qz=qz,
        qw=qw,
    )

    subprocess.run(["bash", "-c", spawn_cmd])


def rgbd_camera_bridge_topics(
    world_name: str,
    model_name: str,
    link_name: str,
    sensor_name: str,
) -> list[str]:
    prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"

    return [
        f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
        f"{prefix}/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
        f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
        f"{prefix}/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
    ]


def camera_bridge_topics(
    world_name: str,
    model_name: str,
    link_name: str,
    sensor_name: str,
) -> list[str]:
    prefix = f"/world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}"

    return [
        f"{prefix}/image@sensor_msgs/msg/Image[ignition.msgs.Image",
        f"{prefix}/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
    ]


def husky_sensor_bridge_topics(world_name: str, model_name: str) -> list[str]:
    base_prefix = f"/world/{world_name}/model/{model_name}/link/base_link/sensor"

    topics = [
        f"{base_prefix}/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
        f"{base_prefix}/planar_laser/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
        f"{base_prefix}/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
    ]

    topics.extend(
        rgbd_camera_bridge_topics(
            world_name,
            model_name,
            "base_link",
            "camera_front",
        )
    )
    topics.extend(
        rgbd_camera_bridge_topics(
            world_name,
            model_name,
            "base_link",
            "camera_down",
        )
    )
    topics.extend(
        rgbd_camera_bridge_topics(
            world_name,
            model_name,
            "tilt_gimbal_link",
            "camera_pan_tilt",
        )
    )

    return topics


def uav_bridge_topics(world_name: str, uav_name: str) -> list[str]:
    topics = [
        f"/{uav_name}/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        f"/{uav_name}/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
        f"/model/{uav_name}/command/twist@geometry_msgs/msg/Twist@ignition.msgs.Twist",
        f"/model/{uav_name}/enable@std_msgs/msg/Bool@ignition.msgs.Boolean",
        f"/model/{uav_name}/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/front_laser/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
        f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
        f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/air_pressure/air_pressure@sensor_msgs/msg/FluidPressure[ignition.msgs.FluidPressure",
        f"/world/{world_name}/model/{uav_name}/link/base_link/sensor/magnetometer/magnetometer@sensor_msgs/msg/MagneticField[ignition.msgs.Magnetometer",
    ]

    topics.extend(
        camera_bridge_topics(
            world_name,
            uav_name,
            "base_link",
            "camera_front",
        )
    )

    return topics


def build_bag_topics(
    world_name: str,
    *,
    include_uav1: bool,
    include_uav2: bool,
    include_camera_topics: bool,
) -> list[str]:
    """Return the useful topic set for hybrid trajectory dataset collection."""

    topics = [
        "/clock",
        f"/world/{world_name}/dynamic_pose/info",

        # Ego Husky control/state.
        "/cmd_vel",
        "/husky_local/controller_state",
        "/husky_local/obstacle_action",
        "/husky_local/obstacle_clearance",
        "/episode/husky_local/start",
        "/episode/husky_local/goal",
        "/model/husky_local/odometry",

        # Ego Husky local perception.
        f"/world/{world_name}/model/husky_local/link/base_link/sensor/planar_laser/scan",
        f"/world/{world_name}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
        f"/world/{world_name}/model/husky_local/link/base_link/sensor/imu_sensor/imu",
    ]

    if include_camera_topics:
        topics.extend(
            [
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/image",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/depth_image",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/camera_info",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_front/points",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/image",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/depth_image",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/camera_info",
                f"/world/{world_name}/model/husky_local/link/base_link/sensor/camera_down/points",
            ]
        )

    if include_uav1:
        topics.extend(
            [
                "/episode/uav1/start",
                "/episode/uav1/goal",
                "/model/uav1/odometry",
                "/uav1/ready",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/front_laser/scan/points",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/imu_sensor/imu",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/air_pressure/air_pressure",
                f"/world/{world_name}/model/uav1/link/base_link/sensor/magnetometer/magnetometer",
            ]
        )

        if include_camera_topics:
            topics.extend(
                [
                    f"/world/{world_name}/model/uav1/link/base_link/sensor/camera_front/image",
                    f"/world/{world_name}/model/uav1/link/base_link/sensor/camera_front/camera_info",
                ]
            )

    if include_uav2:
        topics.extend(
            [
                "/episode/uav2/start",
                "/episode/uav2/goal",
                "/model/uav2/odometry",
                "/uav2/ready",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/front_laser/scan/points",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/imu_sensor/imu",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/air_pressure/air_pressure",
                f"/world/{world_name}/model/uav2/link/base_link/sensor/magnetometer/magnetometer",
            ]
        )

        if include_camera_topics:
            topics.extend(
                [
                    f"/world/{world_name}/model/uav2/link/base_link/sensor/camera_front/image",
                    f"/world/{world_name}/model/uav2/link/base_link/sensor/camera_front/camera_info",
                ]
            )

    return topics


# ---------------------------------------------------------------------------
# Start Gazebo and spawn entities
# ---------------------------------------------------------------------------

os.environ["IGN_GAZEBO_RESOURCE_PATH"] = (
    MODEL_PATH + ":" + os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
)
os.environ["GZ_SIM_RESOURCE_PATH"] = (
    MODEL_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
)

setup_terminal_tee(RUN_LOG_PATH)

subprocess.run(["bash", "-c", "pkill -f ros_gz_bridge || true"])
subprocess.run(["bash", "-c", "pkill -f ign || true"])
subprocess.run(["bash", "-c", f"pkill -f {OMNET_BIN} || true"])

log_event(f"START timestamp: {RUN_START_DT.isoformat(timespec='seconds')}")
log_event(f"Run log file: {RUN_LOG_PATH}")
log_event(f"World path: {WORLD}")
log_event("Starting Gazebo...")

gz = run_bg(f"ign gazebo {WORLD}")

time.sleep(5)

log_event("Waiting for Baylands world to fully load...")
time.sleep(40)

log_event("Spawning Husky ego: husky_local")

husky1_sdf_path = write_husky_variant(
    MODELS_DIR / "husky" / "model_red_tag.sdf",
    "/cmd_vel",
    "flag_marker_red",
    (0.95, 0.12, 0.12, 1.0),
)

spawn_husky = (
    "ign service -s /world/{world_name}/create "
    "--reqtype ignition.msgs.EntityFactory "
    "--reptype ignition.msgs.Boolean "
    "--timeout 5000 "
    "--req 'sdf_filename: \"{sdf_path}\", name: \"husky_local\", "
    "pose: {{position: {{x: {spawn_x}, y: {spawn_y}, z: {spawn_z}}}, "
    "orientation: {{z: {spawn_qz}, w: {spawn_qw}}}}}'"
).format(
    world_name=WORLD_NAME,
    sdf_path=husky1_sdf_path,
    spawn_x=SPAWN_X,
    spawn_y=SPAWN_Y,
    spawn_z=SPAWN_Z,
    spawn_qz=HUSKY1_SPAWN_QZ,
    spawn_qw=HUSKY1_SPAWN_QW,
)

subprocess.run(["bash", "-c", spawn_husky])
time.sleep(5)


if ENABLE_UAV1:
    log_event(
        f"Spawning UAV left support: uav1 at "
        f"({UAV1_X:.3f}, {UAV1_Y:.3f}, {UAV1_Z:.3f})"
    )

    spawn_uav(
        world_name=WORLD_NAME,
        name="uav1",
        x=UAV1_X,
        y=UAV1_Y,
        z=UAV1_Z,
        qz=UAV1_SPAWN_QZ,
        qw=UAV1_SPAWN_QW,
    )

    time.sleep(5)


if ENABLE_UAV2:
    log_event(
        f"Spawning UAV right support: uav2 at "
        f"({UAV2_X:.3f}, {UAV2_Y:.3f}, {UAV2_Z:.3f})"
    )

    spawn_uav(
        world_name=WORLD_NAME,
        name="uav2",
        x=UAV2_X,
        y=UAV2_Y,
        z=UAV2_Z,
        qz=UAV2_SPAWN_QZ,
        qw=UAV2_SPAWN_QW,
    )

    time.sleep(5)


log_event("Spawning visible goal markers...")

spawn_goal_marker(
    WORLD_NAME,
    "goal_husky_local",
    (
        WORLD_HUSKY1_GOAL[0],
        WORLD_HUSKY1_GOAL[1],
        GROUND_MARKER_Z,
    ),
    (0.95, 0.12, 0.12, 1.0),
)

if ENABLE_UAV1:
    spawn_goal_marker(
        WORLD_NAME,
        "goal_uav1",
        (
            WORLD_UAV1_GOAL[0],
            WORLD_UAV1_GOAL[1],
            GROUND_MARKER_Z,
        ),
        (0.95, 0.85, 0.12, 1.0),
    )

if ENABLE_UAV2:
    spawn_goal_marker(
        WORLD_NAME,
        "goal_uav2",
        (
            WORLD_UAV2_GOAL[0],
            WORLD_UAV2_GOAL[1],
            GROUND_MARKER_Z,
        ),
        (0.20, 0.90, 0.95, 1.0),
    )

time.sleep(1)


# ---------------------------------------------------------------------------
# Start ROS-Gazebo bridge
# ---------------------------------------------------------------------------

log_event("Starting ROS-Gazebo bridge...")

bridge_topics = [
    "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
    "/model/husky_local/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
    f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
]

bridge_topics.extend(
    husky_sensor_bridge_topics(
        WORLD_NAME,
        "husky_local",
    )
)

if ENABLE_UAV1:
    bridge_topics.extend(
        uav_bridge_topics(
            WORLD_NAME,
            "uav1",
        )
    )

if ENABLE_UAV2:
    bridge_topics.extend(
        uav_bridge_topics(
            WORLD_NAME,
            "uav2",
        )
    )

bridge_cmd = (
    "source /opt/ros/humble/setup.bash && "
    "ros2 run ros_gz_bridge parameter_bridge "
    + " ".join(bridge_topics)
)

bridge = run_bg(bridge_cmd)
time.sleep(2)


# ---------------------------------------------------------------------------
# Optional visual tools
# ---------------------------------------------------------------------------

rviz = None
camera_view = None

if ENABLE_RVIZ:
    log_event(f"Starting RViz with config: {RVIZ_CONFIG_PATH}")
    rviz_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"rviz2 -d {RVIZ_CONFIG_PATH}"
    )
    rviz = run_bg(rviz_cmd)
    time.sleep(2)

if ENABLE_CAMERA_VIEW:
    log_event("Starting camera viewer...")
    camera_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "ros2 run rqt_image_view rqt_image_view"
    )
    camera_view = run_bg(camera_cmd)
    time.sleep(2)


# ---------------------------------------------------------------------------
# OMNeT++
# ---------------------------------------------------------------------------

omnet = None

if ENABLE_UAV1 or ENABLE_UAV2:
    log_event("OMNeT++ relay disabled in this data-collection runner.")
    log_event("Network features will be exported as placeholder edge fields unless OMNeT++ topics are added later.")


# ---------------------------------------------------------------------------
# Start rosbag recording
# ---------------------------------------------------------------------------

BAG_DIR = Path.home() / "Documents/Thesis/03_dataset/bags"
BAG_DIR.mkdir(parents=True, exist_ok=True)

run_name = "run_dataset_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bag_path = BAG_DIR / run_name

recorder = None

if ENABLE_BAG_RECORDING:
    bag_topics = build_bag_topics(
        WORLD_NAME,
        include_uav1=ENABLE_UAV1,
        include_uav2=ENABLE_UAV2,
        include_camera_topics=RECORD_CAMERA_TOPICS,
    )

    log_event(f"Recording bag: {bag_path}")
    log_event("Bag topic set for hybrid two-UAV trajectory collection:")
    for topic in bag_topics:
        log_event(f"  - {topic}")

    record_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"ros2 bag record -o {bag_path} "
        + " ".join(bag_topics)
    )

    recorder = run_bg(record_cmd)
else:
    log_event("Bag recording disabled. Set ENABLE_BAG_RECORDING = True to record a run.")


if DEBUG_ISOLATE_HUSKY_LOCAL:
    log_event(
        "DEBUG isolate mode: only husky_local is spawned. "
        "Bag recording remains enabled."
    )


# ---------------------------------------------------------------------------
# ROS 2 nodes
# ---------------------------------------------------------------------------

log_event("==============================")
log_event("RULE-BASED DATASET COLLECTION MODE")
log_event("AGENTS: husky_local + uav1(left) + uav2(right)")
log_event("UAV formation: side-by-side, not ahead/behind, high altitude, route-facing yaw")
log_event("==============================")
log_event("Press Play in Gazebo.")
log_event("Let the simulation run until the route is completed or enough data is collected.")
log_event("Press Ctrl+C in this terminal to stop and close the bag cleanly.")

rclpy.init()

HUSKY1_GOAL = offset_goal_along_path(
    WORLD_HUSKY1_GOAL,
    (SPAWN_X, SPAWN_Y, SPAWN_Z),
    GOAL_STOP_OFFSET,
)

UAV1_GOAL = offset_goal_along_path(
    WORLD_UAV1_GOAL,
    (UAV1_X, UAV1_Y, UAV1_Z),
    GOAL_STOP_OFFSET,
)

UAV2_GOAL = offset_goal_along_path(
    WORLD_UAV2_GOAL,
    (UAV2_X, UAV2_Y, UAV2_Z),
    GOAL_STOP_OFFSET,
)

log_event(
    "Controller goals with stop offset: "
    f"husky_local=({HUSKY1_GOAL[0]:.3f}, {HUSKY1_GOAL[1]:.3f}), "
    f"uav1=({UAV1_GOAL[0]:.3f}, {UAV1_GOAL[1]:.3f}), "
    f"uav2=({UAV2_GOAL[0]:.3f}, {UAV2_GOAL[1]:.3f})"
)

log_event(
    f"Visible goal marker remains at "
    f"({WORLD_HUSKY1_GOAL[0]:.3f}, {WORLD_HUSKY1_GOAL[1]:.3f}); "
    f"controller stop tolerance is {GOAL_TOLERANCE:.2f} m."
)


start_goals = {
    "husky_local": {
        "start": (SPAWN_X, SPAWN_Y, SPAWN_Z),
        "goal": WORLD_HUSKY1_GOAL,
    },
}

if ENABLE_UAV1:
    start_goals["uav1"] = {
        "start": (UAV1_X, UAV1_Y, UAV1_Z),
        "goal": WORLD_UAV1_GOAL,
    }

if ENABLE_UAV2:
    start_goals["uav2"] = {
        "start": (UAV2_X, UAV2_Y, UAV2_Z),
        "goal": WORLD_UAV2_GOAL,
    }

episode_metadata = EpisodeMetadataPublisher(
    world_name=WORLD_NAME,
    start_goals=start_goals,
)


# Ego Husky nodes.
husky1_obstacle_action_topic = "/husky_local/obstacle_action"
husky1_obstacle_clearance_topic = "/husky_local/obstacle_clearance"
husky1_controller_state_topic = "/husky_local/controller_state"

obstacle_detector = ObstacleDetectionNode(
    node_name="husky_local_obstacle_detector",
    scan_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan",
    action_topic=husky1_obstacle_action_topic,
    clearance_topic=husky1_obstacle_clearance_topic,
    pointcloud_topic=f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/front_laser/scan/points",
    front_half_angle_deg=OBSTACLE_FRONT_HALF_ANGLE_DEG,
    side_angle_deg=OBSTACLE_SIDE_ANGLE_DEG,
    stop_distance=OBSTACLE_STOP_DISTANCE,
    caution_distance=OBSTACLE_CAUTION_DISTANCE,
)

driver = ModelHuskyDriver(
    node_name="model_husky_driver_1",
    cmd_topic="/cmd_vel",
    odom_topic="/model/husky_local/odometry",
    world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
    uav_ready_topic=None,
    require_uav_ready=False,
    obstacle_action_topic=husky1_obstacle_action_topic,
    obstacle_clearance_topic=husky1_obstacle_clearance_topic,
    state_topic=husky1_controller_state_topic,
    goal_xyz=HUSKY1_GOAL,
    world_goal_xyz=HUSKY1_GOAL,
    bootstrap_seconds=BOOTSTRAP_SECONDS,
    bootstrap_linear_speed=BOOTSTRAP_LINEAR_SPEED,
    control_period=CONTROL_PERIOD,
    cmd_linear_gain=CMD_LINEAR_GAIN,
    cmd_angular_gain=CMD_ANGULAR_GAIN,
    min_linear_speed=MIN_LINEAR_SPEED,
    max_linear_speed=MAX_LINEAR_SPEED,
    max_angular_speed=MAX_ANGULAR_SPEED,
    heading_deadband=HEADING_DEADBAND,
    goal_tolerance=GOAL_TOLERANCE,
    stuck_timeout_seconds=STUCK_TIMEOUT_SECONDS,
    stuck_progress_distance=STUCK_PROGRESS_DISTANCE,
    stuck_reverse_speed=STUCK_REVERSE_SPEED,
    stuck_reverse_seconds=STUCK_REVERSE_SECONDS,
    stuck_bootstrap_seconds=STUCK_BOOTSTRAP_SECONDS,
)


# UAV follower nodes.
follower1 = None
follower2 = None

if ENABLE_UAV1:
    follower1 = UavFollower(
        node_name="uav1_follower",
        husky_odom_topic="/model/husky_local/odometry",
        uav_odom_topic="/model/uav1/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        husky_model_name="husky_local",
        uav_model_name="uav1",
        uav_name="uav1",

        # Formation.
        follow_distance=UAV_FOLLOW_DISTANCE,
        follow_lateral_offset=UAV1_FOLLOW_LATERAL_OFFSET,
        follow_height=UAV_FOLLOW_HEIGHT,

        # Status.
        ready_topic="/uav1/ready",

        # Control.
        update_period=UAV_UPDATE_PERIOD,
        max_xy_speed=UAV_MAX_XY_SPEED,
        max_z_speed=UAV_MAX_Z_SPEED,
        max_yaw_rate=UAV_MAX_YAW_RATE,
        xy_gain=UAV_XY_GAIN,
        z_gain=UAV_Z_GAIN,
        yaw_gain=UAV_YAW_GAIN,
        heading_align_gain=UAV_HEADING_ALIGN_GAIN,
        min_forward_speed=UAV_MIN_FORWARD_SPEED,
        target_smoothing=UAV_TARGET_SMOOTHING,
        xy_deadband=UAV_XY_DEADBAND,
        z_deadband=UAV_Z_DEADBAND,
        yaw_deadband=UAV_YAW_DEADBAND,
        min_track_speed=UAV_MIN_TRACK_SPEED,
        min_follow_altitude=UAV_MIN_FOLLOW_ALTITUDE,
        min_world_altitude=UAV_MIN_WORLD_ALTITUDE,

        # Coordinate correction.
        husky_spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        husky_spawn_yaw=HUSKY1_SPAWN_YAW,
        uav_spawn_xyz=(UAV1_X, UAV1_Y, UAV1_Z),
        uav_spawn_yaw=UAV1_SPAWN_YAW,

        # Yaw direction.
        mission_goal_xyz=WORLD_HUSKY1_GOAL,
        path_goal_xyz=WORLD_HUSKY1_GOAL,
        yaw_mode=UAV_YAW_MODE,
    )

if ENABLE_UAV2:
    follower2 = UavFollower(
        node_name="uav2_follower",
        husky_odom_topic="/model/husky_local/odometry",
        uav_odom_topic="/model/uav2/odometry",
        world_pose_topic=f"/world/{WORLD_NAME}/dynamic_pose/info",
        husky_model_name="husky_local",
        uav_model_name="uav2",
        uav_name="uav2",

        # Formation.
        follow_distance=UAV_FOLLOW_DISTANCE,
        follow_lateral_offset=UAV2_FOLLOW_LATERAL_OFFSET,
        follow_height=UAV_FOLLOW_HEIGHT,

        # Status.
        ready_topic="/uav2/ready",

        # Control.
        update_period=UAV_UPDATE_PERIOD,
        max_xy_speed=UAV_MAX_XY_SPEED,
        max_z_speed=UAV_MAX_Z_SPEED,
        max_yaw_rate=UAV_MAX_YAW_RATE,
        xy_gain=UAV_XY_GAIN,
        z_gain=UAV_Z_GAIN,
        yaw_gain=UAV_YAW_GAIN,
        heading_align_gain=UAV_HEADING_ALIGN_GAIN,
        min_forward_speed=UAV_MIN_FORWARD_SPEED,
        target_smoothing=UAV_TARGET_SMOOTHING,
        xy_deadband=UAV_XY_DEADBAND,
        z_deadband=UAV_Z_DEADBAND,
        yaw_deadband=UAV_YAW_DEADBAND,
        min_track_speed=UAV_MIN_TRACK_SPEED,
        min_follow_altitude=UAV_MIN_FOLLOW_ALTITUDE,
        min_world_altitude=UAV_MIN_WORLD_ALTITUDE,

        # Coordinate correction.
        husky_spawn_xyz=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        husky_spawn_yaw=HUSKY1_SPAWN_YAW,
        uav_spawn_xyz=(UAV2_X, UAV2_Y, UAV2_Z),
        uav_spawn_yaw=UAV2_SPAWN_YAW,

        # Yaw direction.
        mission_goal_xyz=WORLD_HUSKY1_GOAL,
        path_goal_xyz=WORLD_HUSKY1_GOAL,
        yaw_mode=UAV_YAW_MODE,
    )


executor = MultiThreadedExecutor()

executor.add_node(episode_metadata)
executor.add_node(obstacle_detector)
executor.add_node(driver)

if follower1 is not None:
    executor.add_node(follower1)

if follower2 is not None:
    executor.add_node(follower2)


try:
    executor.spin()

except KeyboardInterrupt:
    log_event("Stopping dataset collection run...")

finally:
    managed_nodes = [
        episode_metadata,
        obstacle_detector,
        driver,
    ]

    if follower1 is not None:
        managed_nodes.append(follower1)

    if follower2 is not None:
        managed_nodes.append(follower2)

    for node in managed_nodes:
        with suppress(Exception):
            executor.remove_node(node)

    executor.shutdown(timeout_sec=2.0)
    time.sleep(0.25)

    for node in managed_nodes:
        with suppress(Exception):
            node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()

    if recorder is not None:
        log_event("Stopping rosbag recorder cleanly...")
        with suppress(Exception):
            recorder.send_signal(signal.SIGINT)
        time.sleep(2)

    log_event(f"STOP timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log_event("Stopping bridge, OMNeT++, RViz/camera viewer, and Gazebo...")

    managed_processes = [
        bridge,
        rviz,
        camera_view,
        omnet,
        gz,
    ]

    for proc in managed_processes:
        if proc is None:
            continue
        with suppress(Exception):
            proc.send_signal(signal.SIGINT)

    time.sleep(2)

    for proc in managed_processes:
        if proc is None:
            continue
        with suppress(Exception):
            if proc.poll() is None:
                proc.terminate()

    log_event("All processes stopped cleanly.")
    log_event(f"Bag output folder should be under: {BAG_DIR}")
    log_event(f"Latest bag for this run: {bag_path}")

    close_terminal_tee()