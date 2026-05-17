"""Restart-safe notebook helpers for the 08 trajectory training pipeline."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.paths import COMPARISON_EXPORTS_ROOT, EPISODE_FRAMES_ROOT, MODEL_WEIGHTS_ROOT, NORMALIZATION_ROOT, RESULTS_ROOT, SPLITS_ROOT, TRAIN_READY_ROOT
from datasets.sample_table import build_sample_table, load_episode_streams, save_or_load_fixed_split, save_sample_table


RANGE_CLIP = 30.0
GOAL_DISTANCE_SCALE = 250.0
POSITION_SCALE = 250.0
ALTITUDE_SCALE = 100.0
VELOCITY_SCALE = 10.0
ANGULAR_SCALE = math.pi
EPS = 1e-6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def device_from_flag(use_cpu: bool = False) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")


def _clip_clearance(value: float) -> float:
    if value is None:
        return RANGE_CLIP
    if value >= 900.0:
        return RANGE_CLIP
    return float(np.clip(value, 0.0, RANGE_CLIP))


def _goal_feature_vector(frame: dict) -> list[float]:
    goal = frame["goal"]
    ego = frame["ego"]
    cmd = frame.get("teacher_cmd") or {"linear_x": 0.0, "angular_z": 0.0}
    clearance = frame.get("obstacle_clearance") or {"front": RANGE_CLIP, "left": RANGE_CLIP, "right": RANGE_CLIP}
    return [
        float(ego.get("vx", 0.0) / VELOCITY_SCALE),
        float(ego.get("vy", 0.0) / VELOCITY_SCALE),
        float(ego.get("vz", 0.0) / VELOCITY_SCALE),
        float(ego.get("wz", 0.0) / ANGULAR_SCALE),
        float(goal.get("rel_goal_x_ego", 0.0) / GOAL_DISTANCE_SCALE),
        float(goal.get("rel_goal_y_ego", 0.0) / GOAL_DISTANCE_SCALE),
        float(goal.get("goal_distance", 0.0) / GOAL_DISTANCE_SCALE),
        float(goal.get("goal_heading_error", 0.0) / math.pi),
        float(cmd.get("linear_x", 0.0) / VELOCITY_SCALE),
        float(cmd.get("angular_z", 0.0) / ANGULAR_SCALE),
        float(_clip_clearance(clearance.get("front", RANGE_CLIP)) / RANGE_CLIP),
        float(_clip_clearance(clearance.get("left", RANGE_CLIP)) / RANGE_CLIP),
        float(_clip_clearance(clearance.get("right", RANGE_CLIP)) / RANGE_CLIP),
    ]


def _scan_feature_vector(frame: dict) -> list[float]:
    clearance = frame.get("obstacle_clearance") or {"front": RANGE_CLIP, "left": RANGE_CLIP, "right": RANGE_CLIP}
    return [
        float(_clip_clearance(clearance.get("front", RANGE_CLIP)) / RANGE_CLIP),
        float(_clip_clearance(clearance.get("left", RANGE_CLIP)) / RANGE_CLIP),
        float(_clip_clearance(clearance.get("right", RANGE_CLIP)) / RANGE_CLIP),
    ]


def _node_feature_vector(frame: dict, role: str) -> list[float]:
    ego = frame["ego"]
    goal = frame["goal"]
    agent = frame.get("agents", {}).get(role)
    if role == "ego":
        return [
            0.0,
            0.0,
            0.0,
            float(ego.get("vx", 0.0) / VELOCITY_SCALE),
            float(ego.get("vy", 0.0) / VELOCITY_SCALE),
            float(ego.get("vz", 0.0) / VELOCITY_SCALE),
            float(ego.get("wz", 0.0) / ANGULAR_SCALE),
            float(goal.get("rel_goal_x_ego", 0.0) / GOAL_DISTANCE_SCALE),
            float(goal.get("rel_goal_y_ego", 0.0) / GOAL_DISTANCE_SCALE),
            float(goal.get("goal_distance", 0.0) / GOAL_DISTANCE_SCALE),
            float(goal.get("goal_heading_error", 0.0) / math.pi),
            1.0,
        ]
    if not agent:
        return [0.0] * 12

    dx = float(agent.get("x", ego["x"]) - ego["x"])
    dy = float(agent.get("y", ego["y"]) - ego["y"])
    dz = float(agent.get("z", ego["z"]) - ego["z"])
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    bearing = math.atan2(dy, dx) - float(ego.get("yaw", 0.0))
    return [
        float(dx / POSITION_SCALE),
        float(dy / POSITION_SCALE),
        float(dz / ALTITUDE_SCALE),
        float(agent.get("vx", 0.0) / VELOCITY_SCALE),
        0.0,
        0.0,
        float(agent.get("wz", 0.0) / ANGULAR_SCALE),
        float(distance / POSITION_SCALE),
        float(math.sin(bearing)),
        float(math.cos(bearing)),
        0.0,
        1.0,
    ]


def _edge_features(nodes: np.ndarray) -> np.ndarray:
    edge_rows = []
    for src_idx in range(nodes.shape[0]):
        src_xyz = nodes[src_idx, :3]
        row = []
        for dst_idx in range(nodes.shape[0]):
            dst_xyz = nodes[dst_idx, :3]
            dx, dy, dz = dst_xyz - src_xyz
            distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            inv_distance = 0.0 if distance <= EPS else 1.0 / distance
            bearing = math.atan2(float(dy), float(dx)) if distance > EPS else 0.0
            row.append(
                [
                    float(dx),
                    float(dy),
                    float(dz),
                    float(distance),
                    float(inv_distance),
                    float(math.sin(bearing)),
                    float(math.cos(bearing)),
                    1.0 if src_idx == dst_idx else 0.0,
                ]
            )
        edge_rows.append(row)
    return np.asarray(edge_rows, dtype=np.float32)


def _future_xy_local(anchor_ego: dict, future_ego: dict) -> list[float]:
    dx = float(future_ego["x"]) - float(anchor_ego["x"])
    dy = float(future_ego["y"]) - float(anchor_ego["y"])
    yaw = float(anchor_ego["yaw"])
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return [float(c * dx - s * dy), float(s * dx + c * dy)]


def load_or_build_shared_artifacts(
    *,
    past_len: int,
    future_len: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[list[dict]], list[dict], dict, Path, Path]:
    streams = load_episode_streams(EPISODE_FRAMES_ROOT)
    sample_table = build_sample_table(streams, past_len=past_len, future_len=future_len)
    sample_table_path = TRAIN_READY_ROOT / f"sample_table_seed{seed}_past{past_len}_future{future_len}.json"
    split_path = SPLITS_ROOT / f"trajectory_split_seed{seed}_past{past_len}_future{future_len}.json"
    save_sample_table(sample_table, sample_table_path)
    split_info = save_or_load_fixed_split(
        sample_table=sample_table,
        split_path=split_path,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        past_len=past_len,
        future_len=future_len,
    )
    return streams, sample_table, split_info, sample_table_path, split_path


@dataclass
class TrajectoryBatch:
    goal_seq: torch.Tensor
    future_xy: torch.Tensor
    future_dt: torch.Tensor
    sample_ids: list[str]
    scan_seq: torch.Tensor | None = None
    node_seq: torch.Tensor | None = None
    edge_seq: torch.Tensor | None = None


class GoalOnlyTrajectoryDataset(Dataset):
    def __init__(self, streams: list[list[dict]], sample_table: list[dict], past_len: int):
        self.streams = streams
        self.sample_table = sample_table
        self.past_len = past_len

    def __len__(self) -> int:
        return len(self.sample_table)

    def __getitem__(self, index: int) -> dict:
        meta = self.sample_table[index]
        stream = self.streams[meta["stream_index"]]
        start = meta["start_index"]
        anchor_index = meta["anchor_index"]
        past_frames = stream[start : anchor_index + 1]
        goal_seq = np.asarray([_goal_feature_vector(frame) for frame in past_frames], dtype=np.float32)
        future_xy = np.asarray(meta["future_xy_local"], dtype=np.float32)
        future_dt = np.asarray(meta["future_dt"], dtype=np.float32)
        return {
            "goal_seq": goal_seq,
            "future_xy": future_xy,
            "future_dt": future_dt,
            "sample_id": meta["sample_id"],
        }


class ScanGoalTrajectoryDataset(GoalOnlyTrajectoryDataset):
    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        meta = self.sample_table[index]
        stream = self.streams[meta["stream_index"]]
        start = meta["start_index"]
        anchor_index = meta["anchor_index"]
        past_frames = stream[start : anchor_index + 1]
        scan_seq = np.asarray([_scan_feature_vector(frame) for frame in past_frames], dtype=np.float32)
        item["scan_seq"] = scan_seq
        return item


class ScanGraphTrajectoryDataset(ScanGoalTrajectoryDataset):
    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        meta = self.sample_table[index]
        stream = self.streams[meta["stream_index"]]
        start = meta["start_index"]
        anchor_index = meta["anchor_index"]
        past_frames = stream[start : anchor_index + 1]

        node_seq = []
        edge_seq = []
        for frame in past_frames:
            nodes = np.asarray(
                [
                    _node_feature_vector(frame, "ego"),
                    _node_feature_vector(frame, "uav1"),
                    _node_feature_vector(frame, "uav2"),
                ],
                dtype=np.float32,
            )
            node_seq.append(nodes)
            edge_seq.append(_edge_features(nodes))

        item["node_seq"] = np.asarray(node_seq, dtype=np.float32)
        item["edge_seq"] = np.asarray(edge_seq, dtype=np.float32)
        return item


def collate_goal_only(batch: list[dict]) -> TrajectoryBatch:
    return TrajectoryBatch(
        goal_seq=torch.tensor(np.asarray([item["goal_seq"] for item in batch]), dtype=torch.float32),
        future_xy=torch.tensor(np.asarray([item["future_xy"] for item in batch]), dtype=torch.float32),
        future_dt=torch.tensor(np.asarray([item["future_dt"] for item in batch]), dtype=torch.float32),
        sample_ids=[item["sample_id"] for item in batch],
    )


def collate_scan(batch: list[dict]) -> TrajectoryBatch:
    packed = collate_goal_only(batch)
    packed.scan_seq = torch.tensor(np.asarray([item["scan_seq"] for item in batch]), dtype=torch.float32)
    return packed


def collate_scan_graph(batch: list[dict]) -> TrajectoryBatch:
    packed = collate_scan(batch)
    packed.node_seq = torch.tensor(np.asarray([item["node_seq"] for item in batch]), dtype=torch.float32)
    packed.edge_seq = torch.tensor(np.asarray([item["edge_seq"] for item in batch]), dtype=torch.float32)
    return packed


class LSTMGoalTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, future_len: int, dropout: float):
        super().__init__()
        self.future_len = future_len
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, goal_seq: torch.Tensor, scan_seq=None, node_seq=None, edge_seq=None) -> torch.Tensor:
        _, (hidden, _) = self.lstm(goal_seq)
        pred = self.head(hidden[-1])
        return pred.view(goal_seq.size(0), self.future_len, 2)


class ClearanceCNNEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, scan_seq: torch.Tensor) -> torch.Tensor:
        x = scan_seq.transpose(1, 2)
        return self.net(x).squeeze(-1)


class CNNLSTMTrajectoryPredictor(nn.Module):
    def __init__(self, goal_dim: int, hidden_dim: int, cnn_hidden: int, future_len: int, dropout: float):
        super().__init__()
        self.future_len = future_len
        self.goal_lstm = nn.LSTM(goal_dim, hidden_dim, batch_first=True)
        self.scan_encoder = ClearanceCNNEncoder(cnn_hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + cnn_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, goal_seq: torch.Tensor, scan_seq: torch.Tensor, node_seq=None, edge_seq=None) -> torch.Tensor:
        _, (hidden, _) = self.goal_lstm(goal_seq)
        scan_feat = self.scan_encoder(scan_seq)
        fused = torch.cat([hidden[-1], scan_feat], dim=-1)
        pred = self.head(fused)
        return pred.view(goal_seq.size(0), self.future_len, 2)


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, msg_passes: int):
        super().__init__()
        self.msg_passes = msg_passes
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, node_count, _ = node_seq.shape
        outputs = []
        for step in range(time_steps):
            h = self.node_proj(node_seq[:, step])
            edges = edge_seq[:, step]
            for _ in range(self.msg_passes):
                src = h.unsqueeze(2).expand(-1, -1, node_count, -1)
                dst = h.unsqueeze(1).expand(-1, node_count, -1, -1)
                messages = self.edge_mlp(torch.cat([src, dst, edges], dim=-1))
                agg = messages.sum(dim=1)
                h = self.node_update(torch.cat([h, agg], dim=-1))
            outputs.append(h.mean(dim=1))
        return torch.stack(outputs, dim=1)


class CNNGNNLSTMTrajectoryPredictor(nn.Module):
    def __init__(self, goal_dim: int, node_dim: int, edge_dim: int, hidden_dim: int, graph_hidden: int, future_len: int, dropout: float, msg_passes: int):
        super().__init__()
        self.future_len = future_len
        self.scan_encoder = ClearanceCNNEncoder(hidden_dim)
        self.graph_encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=graph_hidden, msg_passes=msg_passes)
        self.goal_lstm = nn.LSTM(goal_dim + graph_hidden, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, goal_seq: torch.Tensor, scan_seq: torch.Tensor, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> torch.Tensor:
        scan_feat = self.scan_encoder(scan_seq)
        graph_seq = self.graph_encoder(node_seq, edge_seq)
        temporal_input = torch.cat([goal_seq, graph_seq], dim=-1)
        _, (hidden, _) = self.goal_lstm(temporal_input)
        fused = torch.cat([hidden[-1], scan_feat], dim=-1)
        pred = self.head(fused)
        return pred.view(goal_seq.size(0), self.future_len, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, : x.size(1)]


class CNNGNNTransformerTrajectoryPredictor(nn.Module):
    def __init__(self, goal_dim: int, node_dim: int, edge_dim: int, hidden_dim: int, graph_hidden: int, future_len: int, dropout: float, msg_passes: int, num_heads: int, num_layers: int, ff_dim: int):
        super().__init__()
        self.future_len = future_len
        self.scan_encoder = ClearanceCNNEncoder(hidden_dim)
        self.graph_encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=graph_hidden, msg_passes=msg_passes)
        self.input_proj = nn.Linear(goal_dim + graph_hidden, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, goal_seq: torch.Tensor, scan_seq: torch.Tensor, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> torch.Tensor:
        scan_feat = self.scan_encoder(scan_seq)
        graph_seq = self.graph_encoder(node_seq, edge_seq)
        fused_seq = self.input_proj(torch.cat([goal_seq, graph_seq], dim=-1))
        fused_seq = self.pos(fused_seq)
        transformed = self.transformer(fused_seq)
        pooled = transformed.mean(dim=1)
        pred = self.head(torch.cat([pooled, scan_feat], dim=-1))
        return pred.view(goal_seq.size(0), self.future_len, 2)


class CNNGNNLSTMTransformerTrajectoryPredictor(nn.Module):
    def __init__(self, goal_dim: int, node_dim: int, edge_dim: int, hidden_dim: int, graph_hidden: int, future_len: int, dropout: float, msg_passes: int, num_heads: int, num_layers: int, ff_dim: int):
        super().__init__()
        self.future_len = future_len
        self.scan_encoder = ClearanceCNNEncoder(hidden_dim)
        self.graph_encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=graph_hidden, msg_passes=msg_passes)
        self.temporal_input_proj = nn.Linear(goal_dim + graph_hidden, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.pos = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, goal_seq: torch.Tensor, scan_seq: torch.Tensor, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> torch.Tensor:
        scan_feat = self.scan_encoder(scan_seq)
        graph_seq = self.graph_encoder(node_seq, edge_seq)
        seq = self.temporal_input_proj(torch.cat([goal_seq, graph_seq], dim=-1))
        _, (hidden, _) = self.lstm(seq)
        transformed = self.transformer(self.pos(seq)).mean(dim=1)
        fused = torch.cat([scan_feat, hidden[-1], transformed], dim=-1)
        pred = self.head(fused)
        return pred.view(goal_seq.size(0), self.future_len, 2)


def make_dataloaders(dataset: Dataset, split_info: dict, batch_size: int, collate_fn, max_samples: int | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]
    test_indices = split_info["test_indices"]
    if max_samples is not None:
        train_indices = train_indices[: max_samples]
        val_indices = val_indices[: max(1, max_samples // 4)]
        test_indices = test_indices[: max(1, max_samples // 4)]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )


def prepare_result_dirs(model_slug: str) -> tuple[Path, Path, Path]:
    result_dir = RESULTS_ROOT / model_slug
    weight_dir = MODEL_WEIGHTS_ROOT / model_slug
    plot_dir = result_dir / "plots"
    for path in [result_dir, weight_dir, plot_dir, COMPARISON_EXPORTS_ROOT, NORMALIZATION_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    return result_dir, weight_dir, plot_dir


def build_run_manifest(model_slug: str, timestamp: str, split_path: Path, sample_table_path: Path, extra: dict | None = None) -> dict:
    manifest = {
        "model_slug": model_slug,
        "timestamp": timestamp,
        "split_path": str(split_path),
        "sample_table_path": str(sample_table_path),
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_run_manifest(result_dir: Path, manifest: dict, timestamp: str) -> None:
    (result_dir / "latest_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (result_dir / f"{timestamp}_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_training_history(history: dict, out_path: Path) -> None:
    pd.DataFrame(history).to_csv(out_path, index=False)


def save_trajectory_history_plot(history: dict, out_path: Path, title_prefix: str) -> None:
    if not history or not history.get("epoch"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_title(f"{title_prefix}: Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_ade"], label="val_ADE")
    axes[1].plot(history["epoch"], history["val_fde"], label="val_FDE")
    axes[1].plot(history["epoch"], history["val_rmse"], label="val_RMSE")
    axes[1].set_title(f"{title_prefix}: Validation Metrics")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_trajectory_metrics(pred_future_xy: np.ndarray, true_future_xy: np.ndarray) -> dict:
    diff = pred_future_xy - true_future_xy
    dist = np.linalg.norm(diff, axis=-1)
    return {
        "ADE": float(dist.mean()),
        "FDE": float(dist[:, -1].mean()),
        "RMSE": float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))),
    }


def save_trajectory_overlay_plots(pred_future_xy: np.ndarray, true_future_xy: np.ndarray, sample_ids: list[str], output_dir: Path, prefix: str, max_plots: int = 8) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    total = min(max_plots, pred_future_xy.shape[0])
    for idx in range(total):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0.0], [0.0], "ko", label="anchor")
        ax.plot(true_future_xy[idx, :, 0], true_future_xy[idx, :, 1], "-o", label="ground truth")
        ax.plot(pred_future_xy[idx, :, 0], pred_future_xy[idx, :, 1], "--o", label="prediction")
        ax.set_title(sample_ids[idx])
        ax.set_xlabel("Relative x (m)")
        ax.set_ylabel("Relative y (m)")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        path = output_dir / f"{prefix}_trajectory_overlay_{idx:02d}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def save_mean_step_error_plot(pred_future_xy: np.ndarray, true_future_xy: np.ndarray, output_path: Path, title: str) -> str:
    diff = pred_future_xy - true_future_xy
    step_error = np.linalg.norm(diff, axis=-1).mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(step_error) + 1), step_error, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Future step")
    ax.set_ylabel("Mean displacement error (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def evaluate_trajectory_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    total_count = 0
    pred_batches = []
    true_batches = []
    future_dt_batches = []
    sample_ids = []
    with torch.no_grad():
        for batch in loader:
            goal_seq = batch.goal_seq.to(device)
            future_xy = batch.future_xy.to(device)
            scan_seq = batch.scan_seq.to(device) if batch.scan_seq is not None else None
            node_seq = batch.node_seq.to(device) if batch.node_seq is not None else None
            edge_seq = batch.edge_seq.to(device) if batch.edge_seq is not None else None

            pred = model(goal_seq, scan_seq, node_seq, edge_seq)
            loss = criterion(pred, future_xy)
            total_loss += float(loss.item()) * future_xy.size(0)
            total_count += int(future_xy.size(0))

            pred_batches.append(pred.cpu().numpy())
            true_batches.append(future_xy.cpu().numpy())
            future_dt_batches.append(batch.future_dt.cpu().numpy())
            sample_ids.extend(batch.sample_ids)

    pred_future_xy = np.concatenate(pred_batches, axis=0)
    true_future_xy = np.concatenate(true_batches, axis=0)
    future_dt = np.concatenate(future_dt_batches, axis=0)
    metrics = compute_trajectory_metrics(pred_future_xy, true_future_xy)
    metrics["loss"] = total_loss / max(total_count, 1)
    return {
        "metrics": metrics,
        "pred_future_xy": pred_future_xy,
        "true_future_xy": true_future_xy,
        "future_dt": future_dt,
        "sample_ids": sample_ids,
    }


def train_trajectory_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_slug: str,
    timestamp: str,
    split_path: Path,
    sample_table_path: Path,
    result_dir: Path,
    weight_dir: Path,
    plot_dir: Path,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    extra_manifest: dict | None = None,
    verbose: bool = True,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_ade": [], "val_fde": [], "val_rmse": []}
    best_state = None
    best_metrics = None
    best_epoch = -1
    best_val_ade = float("inf")
    stale_epochs = 0

    manifest = build_run_manifest(
        model_slug=model_slug,
        timestamp=timestamp,
        split_path=split_path,
        sample_table_path=sample_table_path,
        extra=extra_manifest or {},
    )
    save_run_manifest(result_dir, manifest, timestamp)
    if verbose:
        print(
            f"[{model_slug}] start: epochs={epochs} patience={patience} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
        )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            goal_seq = batch.goal_seq.to(device)
            future_xy = batch.future_xy.to(device)
            scan_seq = batch.scan_seq.to(device) if batch.scan_seq is not None else None
            node_seq = batch.node_seq.to(device) if batch.node_seq is not None else None
            edge_seq = batch.edge_seq.to(device) if batch.edge_seq is not None else None

            optimizer.zero_grad(set_to_none=True)
            pred = model(goal_seq, scan_seq, node_seq, edge_seq)
            loss = criterion(pred, future_xy)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * future_xy.size(0)
            train_count += int(future_xy.size(0))

        val_eval = evaluate_trajectory_model(model, val_loader, device)
        val_metrics = val_eval["metrics"]
        train_loss = train_loss_sum / max(train_count, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_ade"].append(val_metrics["ADE"])
        history["val_fde"].append(val_metrics["FDE"])
        history["val_rmse"].append(val_metrics["RMSE"])

        improved = val_metrics["ADE"] < best_val_ade
        if val_metrics["ADE"] < best_val_ade:
            best_val_ade = val_metrics["ADE"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
            torch.save(best_state, weight_dir / f"{timestamp}_best.pt")
            torch.save(best_state, weight_dir / "latest.pt")
        else:
            stale_epochs += 1

        if verbose:
            status = "improved" if improved else f"no_improve({stale_epochs}/{patience})"
            print(
                f"[{model_slug}] epoch {epoch:02d}/{epochs} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_ADE={val_metrics['ADE']:.6f} "
                f"val_FDE={val_metrics['FDE']:.6f} "
                f"val_RMSE={val_metrics['RMSE']:.6f} "
                f"best_ADE={best_val_ade:.6f} "
                f"status={status}"
            )

        if stale_epochs >= patience:
            if verbose:
                print(
                    f"[{model_slug}] early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_val_ADE={best_val_ade:.6f})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(
            f"[{model_slug}] done: best_epoch={best_epoch} "
            f"best_val_ADE={best_val_ade:.6f}"
        )

    history_csv = result_dir / f"{timestamp}_training_history.csv"
    save_training_history(history, history_csv)
    save_trajectory_history_plot(history, plot_dir / f"{timestamp}_training_history.png", model_slug)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics or {},
        "early_stop_epoch": history["epoch"][-1] if history["epoch"] else None,
        "history_csv": str(history_csv),
        "best_weight_path": str(weight_dir / f"{timestamp}_best.pt"),
        "latest_weight_path": str(weight_dir / "latest.pt"),
    }


def save_final_trajectory_evaluation(
    *,
    model_slug: str,
    timestamp: str,
    train_out: dict,
    test_eval: dict,
    split_path: Path,
    sample_table_path: Path,
    result_dir: Path,
    plot_dir: Path,
) -> dict:
    metrics = dict(test_eval["metrics"])
    metrics.update(
        {
            "model_slug": model_slug,
            "timestamp": timestamp,
            "split_path": str(split_path),
            "sample_table_path": str(sample_table_path),
            "best_epoch": train_out["best_epoch"],
            "best_val_metrics": train_out["best_val_metrics"],
            "history_csv": train_out["history_csv"],
            "best_weight_path": train_out["best_weight_path"],
            "latest_weight_path": train_out["latest_weight_path"],
        }
    )

    overlay_paths = save_trajectory_overlay_plots(
        test_eval["pred_future_xy"],
        test_eval["true_future_xy"],
        test_eval["sample_ids"],
        plot_dir,
        prefix=timestamp,
    )
    step_error_path = save_mean_step_error_plot(
        test_eval["pred_future_xy"],
        test_eval["true_future_xy"],
        plot_dir / f"{timestamp}_mean_step_error.png",
        title=f"{model_slug} Mean Future-Step Error",
    )

    prediction_path = result_dir / f"{timestamp}_trajectory_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=np.asarray(test_eval["sample_ids"], dtype=str),
        pred_future_xy=np.asarray(test_eval["pred_future_xy"], dtype=np.float32),
        true_future_xy=np.asarray(test_eval["true_future_xy"], dtype=np.float32),
        future_dt=np.asarray(test_eval["future_dt"], dtype=np.float32),
    )

    metrics["prediction_export_path"] = str(prediction_path)
    metrics["overlay_paths"] = overlay_paths
    metrics["step_error_plot"] = step_error_path

    metrics_path = result_dir / f"{timestamp}_metrics.json"
    latest_path = result_dir / "latest_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_path = COMPARISON_EXPORTS_ROOT / "trajectory_model_summary_latest.csv"
    row = pd.DataFrame(
        [
            {
                "model_slug": model_slug,
                "timestamp": timestamp,
                "ADE": metrics["ADE"],
                "FDE": metrics["FDE"],
                "RMSE": metrics["RMSE"],
                "loss": metrics["loss"],
                "split_path": str(split_path),
                "prediction_export_path": str(prediction_path),
            }
        ]
    )
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        if "model_slug" not in previous.columns and "model" in previous.columns:
            previous = previous.rename(columns={"model": "model_slug"})
        if "model_slug" in previous.columns:
            previous = previous[previous["model_slug"] != model_slug]
        pd.concat([previous, row], ignore_index=True).to_csv(summary_path, index=False)
    else:
        row.to_csv(summary_path, index=False)

    return metrics


def run_constant_velocity_baseline(
    *,
    streams: list[list[dict]],
    sample_table: list[dict],
    split_info: dict,
    model_slug: str,
    timestamp: str,
    split_path: Path,
    sample_table_path: Path,
    result_dir: Path,
    plot_dir: Path,
) -> dict:
    test_indices = split_info["test_indices"]
    pred_future_xy = []
    true_future_xy = []
    future_dt = []
    sample_ids = []
    for sample_idx in test_indices:
        meta = sample_table[sample_idx]
        stream = streams[meta["stream_index"]]
        anchor_idx = meta["anchor_index"]
        prev_idx = max(meta["start_index"], anchor_idx - 1)
        anchor = stream[anchor_idx]
        previous = stream[prev_idx]
        dt = max((int(anchor["timestamp_ns"]) - int(previous["timestamp_ns"])) * 1e-9, 1e-3)
        local_step = np.asarray(_future_xy_local(previous["ego"], anchor["ego"]), dtype=np.float32)
        local_velocity = local_step / dt
        sample_future_dt = np.asarray(meta["future_dt"], dtype=np.float32)
        pred_future_xy.append(np.outer(sample_future_dt, local_velocity))
        true_future_xy.append(np.asarray(meta["future_xy_local"], dtype=np.float32))
        future_dt.append(sample_future_dt)
        sample_ids.append(meta["sample_id"])

    pred_future_xy_arr = np.asarray(pred_future_xy, dtype=np.float32)
    true_future_xy_arr = np.asarray(true_future_xy, dtype=np.float32)
    future_dt_arr = np.asarray(future_dt, dtype=np.float32)
    metrics = compute_trajectory_metrics(pred_future_xy_arr, true_future_xy_arr)
    metrics["loss"] = metrics["ADE"]
    metrics["model_slug"] = model_slug
    metrics["timestamp"] = timestamp
    metrics["split_path"] = str(split_path)
    metrics["sample_table_path"] = str(sample_table_path)

    overlay_paths = save_trajectory_overlay_plots(
        pred_future_xy_arr,
        true_future_xy_arr,
        sample_ids,
        plot_dir,
        prefix=timestamp,
    )
    step_error_path = save_mean_step_error_plot(
        pred_future_xy_arr,
        true_future_xy_arr,
        plot_dir / f"{timestamp}_mean_step_error.png",
        title=f"{model_slug} Mean Future-Step Error",
    )
    prediction_path = result_dir / f"{timestamp}_trajectory_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=np.asarray(sample_ids, dtype=str),
        pred_future_xy=pred_future_xy_arr,
        true_future_xy=true_future_xy_arr,
        future_dt=future_dt_arr,
    )
    metrics["prediction_export_path"] = str(prediction_path)
    metrics["overlay_paths"] = overlay_paths
    metrics["step_error_plot"] = step_error_path

    metrics_path = result_dir / f"{timestamp}_metrics.json"
    latest_path = result_dir / "latest_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = build_run_manifest(
        model_slug=model_slug,
        timestamp=timestamp,
        split_path=split_path,
        sample_table_path=sample_table_path,
        extra={"baseline_type": "constant_velocity_local"},
    )
    save_run_manifest(result_dir, manifest, timestamp)
    return metrics
