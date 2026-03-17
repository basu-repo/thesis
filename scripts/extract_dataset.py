#!/usr/bin/env python3
# Extracts a Husky trajectory dataset from the latest rosbag.
# Uses Husky LaserScan timestamps as frame anchors and aligns odometry/cmd_vel.
# Optionally saves UAV lidar point clouds if they exist in the bag.

import csv
import sqlite3
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore


WORLD_NAME = "sim_world"

HUSKY_SCAN_TOPIC = (
    f"/world/{WORLD_NAME}/model/husky_local/link/base_link/sensor/planar_laser/scan"
)
UAV_POINTCLOUD_TOPIC = (
    f"/world/{WORLD_NAME}/model/uav1/link/base_link/sensor/front_laser/scan/points"
)
ODOM_TOPIC = "/model/husky_local/odometry"
CMD_TOPIC = "/cmd_vel"
HUSKY_START_TOPIC = "/episode/husky_local/start"
HUSKY_GOAL_TOPIC = "/episode/husky_local/goal"
HUSKY2_START_TOPIC = "/episode/husky_2/start"
HUSKY2_GOAL_TOPIC = "/episode/husky_2/goal"
UAV_START_TOPIC = "/episode/uav1/start"
UAV_GOAL_TOPIC = "/episode/uav1/goal"

BAGS_DIR = Path.home() / "Documents/Thesis/bags"
OUT_ROOT = Path.home() / "Documents/Thesis/rellis_like_dataset"

SEQ_NAME = "00000"


def latest_bag(path: Path) -> Path:
    bags = sorted(path.glob("run_*"))
    if not bags:
        raise RuntimeError(f"No bag found in {path}")
    return bags[-1].resolve()


def bag_db3_path(bag_dir: Path) -> Path:
    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        raise RuntimeError(f"No .db3 file found in {bag_dir}")
    return db3_files[0]


def decode_pointcloud2_to_xyz_i(msg):
    """Convert PointCloud2 -> (N,4) float32 [x, y, z, intensity]."""
    n_points = msg.width * msg.height
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("pad1", np.float32),
            ("intensity", np.float32),
            ("pad2", np.float32),
            ("ring", np.uint16),
            ("pad3", np.uint16),
        ]
    )
    arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)
    pts = np.zeros((n_points, 4), dtype=np.float32)
    pts[:, 0] = arr["x"]
    pts[:, 1] = arr["y"]
    pts[:, 2] = arr["z"]
    pts[:, 3] = arr["intensity"]
    return pts


def save_laserscan(msg, out_path: Path):
    intensities = np.asarray(msg.intensities, dtype=np.float32)
    if intensities.size == 0:
        intensities = np.zeros(len(msg.ranges), dtype=np.float32)
    scan = np.stack(
        [
            np.asarray(msg.ranges, dtype=np.float32),
            intensities,
        ],
        axis=1,
    )
    np.save(out_path, scan)


def main():
    bag_path = latest_bag(BAGS_DIR)
    print("Using bag:", bag_path)
    db3_path = bag_db3_path(bag_path)
    print("Using db3:", db3_path)

    scan_dir = OUT_ROOT / "husky_planar_laser" / SEQ_NAME
    uav_pc_dir = OUT_ROOT / "uav_front_laser_kitti_bin" / SEQ_NAME
    scan_dir.mkdir(parents=True, exist_ok=True)
    uav_pc_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = OUT_ROOT / "maneuver_targets.csv"

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    odom = []
    cmd = []
    husky_scans = []
    uav_pointclouds = []
    episode_pose = {}

    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()
    topic_map = {
        topic_id: (name, msgtype)
        for topic_id, name, msgtype in cur.execute(
            "SELECT id, name, type FROM topics ORDER BY id"
        )
    }

    for topic_id, timestamp, rawdata in cur.execute(
        "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
    ):
        topic, msgtype = topic_map[topic_id]
        ts = int(timestamp)

        if topic == ODOM_TOPIC:
            msg = typestore.deserialize_cdr(rawdata, msgtype)
            odom.append(
                (
                    ts,
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y),
                    float(msg.twist.twist.linear.x),
                    float(msg.twist.twist.angular.z),
                )
            )
        elif topic == CMD_TOPIC:
            msg = typestore.deserialize_cdr(rawdata, msgtype)
            cmd.append((ts, float(msg.linear.x), float(msg.angular.z)))
        elif topic == HUSKY_SCAN_TOPIC:
            husky_scans.append((ts, rawdata, msgtype))
        elif topic == UAV_POINTCLOUD_TOPIC:
            uav_pointclouds.append((ts, rawdata, msgtype))
        elif topic in {
            HUSKY_START_TOPIC,
            HUSKY_GOAL_TOPIC,
            HUSKY2_START_TOPIC,
            HUSKY2_GOAL_TOPIC,
            UAV_START_TOPIC,
            UAV_GOAL_TOPIC,
        }:
            msg = typestore.deserialize_cdr(rawdata, msgtype)
            episode_pose[topic] = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            )

    conn.close()

    odom.sort(key=lambda x: x[0])
    cmd.sort(key=lambda x: x[0])
    husky_scans.sort(key=lambda x: x[0])
    uav_pointclouds.sort(key=lambda x: x[0])

    if not husky_scans:
        raise RuntimeError(
            f"No Husky LaserScan frames found on topic {HUSKY_SCAN_TOPIC}"
        )

    print("Husky scan frames:", len(husky_scans))
    print("UAV pointcloud frames:", len(uav_pointclouds))

    husky_start = episode_pose.get(HUSKY_START_TOPIC, (0.0, 0.0, 0.0))
    husky_goal = episode_pose.get(HUSKY_GOAL_TOPIC, (0.0, 0.0, 0.0))
    husky2_start = episode_pose.get(HUSKY2_START_TOPIC, (0.0, 0.0, 0.0))
    husky2_goal = episode_pose.get(HUSKY2_GOAL_TOPIC, (0.0, 0.0, 0.0))
    uav_start = episode_pose.get(UAV_START_TOPIC, (0.0, 0.0, 0.0))
    uav_goal = episode_pose.get(UAV_GOAL_TOPIC, (0.0, 0.0, 0.0))

    odom_i = 0
    cmd_i = 0
    uav_i = 0
    cur_odom = (0.0, 0.0, 0.0, 0.0)
    cur_cmd = (0.0, 0.0)

    with open(dataset_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_id",
                "timestamp_ns",
                "x",
                "y",
                "linear_vel_odom",
                "angular_vel_odom",
                "cmd_v",
                "cmd_w",
                "husky_start_x",
                "husky_start_y",
                "husky_goal_x",
                "husky_goal_y",
                "husky2_start_x",
                "husky2_start_y",
                "husky2_goal_x",
                "husky2_goal_y",
                "uav_start_x",
                "uav_start_y",
                "uav_start_z",
                "uav_goal_x",
                "uav_goal_y",
                "uav_goal_z",
                "husky_scan_path",
                "uav_lidar_path",
            ]
        )

        for frame_id, (t, rawdata, msgtype) in enumerate(husky_scans):
            while odom_i < len(odom) and odom[odom_i][0] <= t:
                cur_odom = odom[odom_i][1:]
                odom_i += 1

            while cmd_i < len(cmd) and cmd[cmd_i][0] <= t:
                cur_cmd = cmd[cmd_i][1:]
                cmd_i += 1

            scan_msg = typestore.deserialize_cdr(rawdata, msgtype)
            scan_rel_path = Path("husky_planar_laser") / SEQ_NAME / f"{frame_id:06d}.npy"
            save_laserscan(scan_msg, OUT_ROOT / scan_rel_path)

            uav_rel_path = ""
            while uav_i + 1 < len(uav_pointclouds) and uav_pointclouds[uav_i + 1][0] <= t:
                uav_i += 1
            if uav_pointclouds:
                uav_ts, uav_rawdata, uav_msgtype = uav_pointclouds[uav_i]
                if abs(uav_ts - t) < 200_000_000:
                    uav_msg = typestore.deserialize_cdr(uav_rawdata, uav_msgtype)
                    pts = decode_pointcloud2_to_xyz_i(uav_msg)
                    uav_rel_path = (
                        Path("uav_front_laser_kitti_bin") / SEQ_NAME / f"{frame_id:06d}.bin"
                    )
                    pts.astype(np.float32).tofile(OUT_ROOT / uav_rel_path)

            writer.writerow(
                [
                    frame_id,
                    t,
                    cur_odom[0],
                    cur_odom[1],
                    cur_odom[2],
                    cur_odom[3],
                    cur_cmd[0],
                    cur_cmd[1],
                    husky_start[0],
                    husky_start[1],
                    husky_goal[0],
                    husky_goal[1],
                    husky2_start[0],
                    husky2_start[1],
                    husky2_goal[0],
                    husky2_goal[1],
                    uav_start[0],
                    uav_start[1],
                    uav_start[2],
                    uav_goal[0],
                    uav_goal[1],
                    uav_goal[2],
                    str(scan_rel_path),
                    str(uav_rel_path),
                ]
            )

            if (frame_id + 1) % 200 == 0:
                print("Saved", frame_id + 1, "frames")

    print("\nDone.")
    print("Saved to:", OUT_ROOT)
    print("Outputs:")
    print("  husky_planar_laser/00000/*.npy")
    print("  uav_front_laser_kitti_bin/00000/*.bin   (if available)")
    print("  maneuver_targets.csv")


if __name__ == "__main__":
    main()
