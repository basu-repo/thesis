"""Build trajectory sample tables and shared fixed splits for the 08 pipeline."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


def load_episode_streams(episode_frames_root: Path) -> list[list[dict]]:
    streams: list[list[dict]] = []
    frame_files = sorted(episode_frames_root.glob("*/frames.jsonl"))
    for frames_path in frame_files:
        with frames_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows.sort(key=lambda row: int(row["timestamp_ns"]))
        if rows:
            streams.append(rows)
    if not streams:
        raise RuntimeError(f"No episode frame files found under {episode_frames_root}")
    return streams


def _future_xy_local(anchor_ego: dict, future_ego: dict) -> list[float]:
    dx = float(future_ego["x"]) - float(anchor_ego["x"])
    dy = float(future_ego["y"]) - float(anchor_ego["y"])
    yaw = float(anchor_ego["yaw"])
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return [
        float(c * dx - s * dy),
        float(s * dx + c * dy),
    ]


def build_sample_table(streams: list[list[dict]], past_len: int, future_len: int) -> list[dict]:
    sample_table: list[dict] = []
    for stream_idx, stream in enumerate(streams):
        usable = len(stream) - past_len - future_len + 1
        if usable <= 0:
            continue
        for start in range(usable):
            anchor = stream[start + past_len - 1]
            future_frames = stream[start + past_len : start + past_len + future_len]
            anchor_ego = anchor["ego"]
            anchor_ts = int(anchor["timestamp_ns"])
            episode_id = anchor["episode_id"]

            future_xy_local = []
            future_dt = []
            for future_frame in future_frames:
                future_ego = future_frame["ego"]
                future_xy_local.append(_future_xy_local(anchor_ego, future_ego))
                future_dt.append((int(future_frame["timestamp_ns"]) - anchor_ts) * 1e-9)

            sample_table.append(
                {
                    "sample_id": f"{episode_id}_stream{stream_idx:03d}_start{start:05d}",
                    "episode_id": episode_id,
                    "stream_index": stream_idx,
                    "start_index": start,
                    "anchor_index": start + past_len - 1,
                    "past_len": past_len,
                    "future_len": future_len,
                    "anchor_timestamp_ns": anchor_ts,
                    "teacher_state": anchor.get("teacher_state"),
                    "future_xy_local": future_xy_local,
                    "future_dt": future_dt,
                }
            )
    return sample_table


def save_sample_table(sample_table: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample_table, indent=2), encoding="utf-8")


def save_or_load_fixed_split(
    sample_table: list[dict],
    split_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    past_len: int,
    future_len: int,
) -> dict:
    if split_path.exists():
        split_info = json.loads(split_path.read_text(encoding="utf-8"))
        current_sample_ids = [row["sample_id"] for row in sample_table]
        if (
            split_info.get("sample_count") == len(sample_table)
            and split_info.get("past_len") == past_len
            and split_info.get("future_len") == future_len
            and split_info.get("split_strategy") == "episode"
            and split_info.get("sample_ids") == current_sample_ids
        ):
            return split_info

    rng = random.Random(seed)
    episode_ids = sorted({row["episode_id"] for row in sample_table})
    if not episode_ids:
        raise RuntimeError("Cannot build a split from an empty sample table.")
    rng.shuffle(episode_ids)

    episode_count = len(episode_ids)
    if episode_count == 1:
        train_episode_ids = episode_ids[:]
        val_episode_ids = []
        test_episode_ids = []
    elif episode_count == 2:
        train_episode_ids = episode_ids[:1]
        val_episode_ids = []
        test_episode_ids = episode_ids[1:]
    else:
        train_episode_count = max(1, int(episode_count * train_ratio))
        val_episode_count = max(1, int(episode_count * val_ratio))
        if train_episode_count + val_episode_count >= episode_count:
            if train_episode_count >= val_episode_count:
                train_episode_count -= 1
            else:
                val_episode_count -= 1
        test_episode_count = episode_count - train_episode_count - val_episode_count
        if test_episode_count < 1:
            test_episode_count = 1
            if train_episode_count > val_episode_count and train_episode_count > 1:
                train_episode_count -= 1
            elif val_episode_count > 1:
                val_episode_count -= 1

        train_episode_ids = episode_ids[:train_episode_count]
        val_episode_ids = episode_ids[train_episode_count : train_episode_count + val_episode_count]
        test_episode_ids = episode_ids[train_episode_count + val_episode_count :]

    train_episode_set = set(train_episode_ids)
    val_episode_set = set(val_episode_ids)
    test_episode_set = set(test_episode_ids)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for idx, row in enumerate(sample_table):
        episode_id = row["episode_id"]
        if episode_id in train_episode_set:
            train_indices.append(idx)
        elif episode_id in val_episode_set:
            val_indices.append(idx)
        elif episode_id in test_episode_set:
            test_indices.append(idx)
        else:
            raise RuntimeError(f"Sample {row['sample_id']} has no split assignment for episode {episode_id}.")

    split_info = {
        "seed": seed,
        "split_strategy": "episode",
        "sample_count": len(sample_table),
        "past_len": past_len,
        "future_len": future_len,
        "episode_count": episode_count,
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "test_episode_ids": test_episode_ids,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "sample_ids": [row["sample_id"] for row in sample_table],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    return split_info
