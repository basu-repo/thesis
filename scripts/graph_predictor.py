"""Build graph-sequence samples and define the GNN-LSTM predictor.

This module is the offline graph-learning core of the project:
- discovers exported graph runs from ``frames.jsonl``
- converts each run into fixed past/future training windows
- defines the message-passing encoder and temporal predictor
- provides baseline and metric helpers used during training
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset


NODE_ORDER = ["husky_local", "husky_2", "uav1"]
PLATFORM_ONEHOT = {
    "UGV": [1.0, 0.0],
    "UAV": [0.0, 1.0],
}


# Dataset discovery and feature extraction -----------------------------------

def discover_frame_files(graph_root: str | Path) -> list[Path]:
    graph_root = Path(graph_root).expanduser()
    if graph_root.is_file():
        return [graph_root]
    return sorted(graph_root.glob("*/frames.jsonl"))


def load_runs(graph_root: str | Path) -> list[list[dict]]:
    runs = []
    for frames_path in discover_frame_files(graph_root):
        with frames_path.open() as f:
            frames = [json.loads(line) for line in f if line.strip()]
        if frames:
            runs.append(frames)
    if not runs:
        raise RuntimeError(f"No graph frames found under {graph_root}")
    return runs


def _node_feature(node: dict, ego_state: dict) -> list[float]:
    state = node["state"]
    goal = node.get("goal") or {"x": state["x"], "y": state["y"], "z": state["z"]}
    command = node.get("command") or {"linear_x": 0.0, "angular_z": 0.0}
    platform = PLATFORM_ONEHOT.get(node["platform_type"], [0.0, 0.0])
    return [
        state["x"] - ego_state["x"],
        state["y"] - ego_state["y"],
        state["z"] - ego_state["z"],
        state["vx"],
        state["vy"],
        state["vz"],
        state["wz"],
        goal["x"] - state["x"],
        goal["y"] - state["y"],
        goal["z"] - state["z"],
        command["linear_x"],
        command["angular_z"],
        platform[0],
        platform[1],
    ]


def _edge_lookup(frame: dict) -> dict[tuple[str, str], dict]:
    return {(edge["source"], edge["target"]): edge for edge in frame["edges"]}


@dataclass
class GraphSample:
    """One supervised training sample built from a graph history window."""

    node_seq: torch.Tensor
    edge_seq: torch.Tensor
    target_future: torch.Tensor
    past_xy: torch.Tensor


class GraphSequenceDataset(Dataset):
    """Turn graph runs into sliding windows for graph-based trajectory learning."""

    def __init__(
        self,
        runs: list[list[dict]],
        past_len: int = 10,
        future_len: int = 20,
        ego_node: str = "husky_local",
    ):
        self.runs = runs
        self.past_len = past_len
        self.future_len = future_len
        self.ego_node = ego_node
        self.ego_idx = NODE_ORDER.index(ego_node)
        self.samples: list[tuple[int, int]] = []
        for run_idx, frames in enumerate(runs):
            usable = len(frames) - past_len - future_len + 1
            for start in range(max(0, usable)):
                self.samples.append((run_idx, start))

        if not self.samples:
            raise RuntimeError(
                f"Not enough graph frames for past_len={past_len}, future_len={future_len}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        run_idx, start = self.samples[idx]
        frames = self.runs[run_idx]
        node_seq = []
        edge_seq = []
        past_xy = []

        for t in range(start, start + self.past_len):
            frame = frames[t]
            nodes = frame["nodes"]
            ego_state = nodes[self.ego_node]["state"]
            past_xy.append([ego_state["x"], ego_state["y"]])
            node_feats = []
            edge_feats = []
            edge_map = _edge_lookup(frame)
            for src in NODE_ORDER:
                node_feats.append(_node_feature(nodes[src], ego_state))
            for src in NODE_ORDER:
                src_edges = []
                for dst in NODE_ORDER:
                    edge = edge_map.get((src, dst))
                    if edge is None:
                        src_edges.append([0.0, 0.0, 0.0, 0.0])
                    else:
                        src_edges.append(
                            [
                                edge["dx"],
                                edge["dy"],
                                edge["dz"],
                                edge["distance"],
                            ]
                        )
                edge_feats.append(src_edges)
            node_seq.append(node_feats)
            edge_seq.append(edge_feats)

        origin_frame = frames[start + self.past_len - 1]
        origin_state = origin_frame["nodes"][self.ego_node]["state"]
        target_future = []
        for t in range(start + self.past_len, start + self.past_len + self.future_len):
            future_state = frames[t]["nodes"][self.ego_node]["state"]
            target_future.append(
                [
                    future_state["x"] - origin_state["x"],
                    future_state["y"] - origin_state["y"],
                ]
            )

        return GraphSample(
            node_seq=torch.tensor(node_seq, dtype=torch.float32),
            edge_seq=torch.tensor(edge_seq, dtype=torch.float32),
            target_future=torch.tensor(target_future, dtype=torch.float32),
            past_xy=torch.tensor(past_xy, dtype=torch.float32),
        )


class GraphEncoder(nn.Module):
    """Encode one multi-agent graph snapshot with message passing."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, msg_passes: int = 2):
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

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(node_feats)
        for _ in range(self.msg_passes):
            src = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
            dst = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
            messages = self.edge_mlp(torch.cat([src, dst, edge_feats], dim=-1))
            agg = messages.sum(dim=1)
            h = self.node_update(torch.cat([h, agg], dim=-1))
        return h


class GNNLSTM(nn.Module):
    """Predict future ego motion from graph snapshots over time."""

    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 4,
        hidden_dim: int = 96,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        future_len: int = 20,
        ego_idx: int = 0,
        msg_passes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_len = future_len
        self.ego_idx = ego_idx
        self.encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, msg_passes=msg_passes)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, future_len * 2),
        )

    def forward(self, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> torch.Tensor:
        encoded_steps = []
        for t in range(node_seq.size(1)):
            node_emb = self.encoder(node_seq[:, t], edge_seq[:, t])
            encoded_steps.append(node_emb[:, self.ego_idx])
        seq = torch.stack(encoded_steps, dim=1)
        _, (h, _) = self.lstm(seq)
        out = self.head(h[-1])
        return out.view(-1, self.future_len, 2)


def trajectory_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute standard trajectory prediction metrics."""

    dists = torch.linalg.norm(pred - target, dim=2)
    ade = dists.mean().item()
    fde = dists[:, -1].mean().item()
    rmse = torch.sqrt(((pred - target) ** 2).mean()).item()
    return {"ADE": ade, "FDE": fde, "RMSE": rmse}


def constant_velocity_predict(past_xy: torch.Tensor, future_len: int) -> torch.Tensor:
    """Simple baseline that extrapolates the last observed planar velocity."""

    last = past_xy[:, -1]
    prev = past_xy[:, -2]
    velocity = last - prev
    steps = (
        torch.arange(1, future_len + 1, device=past_xy.device, dtype=past_xy.dtype)
        .view(1, future_len, 1)
    )
    future = last.unsqueeze(1) + steps * velocity.unsqueeze(1)
    return future - last.unsqueeze(1)
