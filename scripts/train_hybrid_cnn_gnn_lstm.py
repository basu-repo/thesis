#!/usr/bin/env python3
"""Train a maneuver classifier on the hybrid JSONL dataset.

This trainer uses:
- a real 1D CNN over the ego planar lidar scan
- a GNN over the multi-agent graph context
- an LSTM over a temporal window of fused CNN+GNN embeddings

The target is the teacher maneuver label from the rule-based Husky controller.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from graph_predictor import GraphEncoder


DEFAULT_LABELS = [
    "bootstrap",
    "go_to_goal",
    "avoid_left",
    "avoid_right",
    "commit_forward",
    "reverse",
    "recover",
    "reassess",
    "arrived",
    "stop",
]

PLATFORM_ONEHOT = {
    "UGV": [1.0, 0.0],
    "UAV": [0.0, 1.0],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_frame_files(dataset_root: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root).expanduser()
    if dataset_root.is_file():
        return [dataset_root]
    if (dataset_root / "frames.jsonl").exists():
        return [dataset_root / "frames.jsonl"]
    return sorted(dataset_root.glob("*/frames.jsonl"))


def load_schema_labels(dataset_root: str | Path) -> list[str]:
    dataset_root = Path(dataset_root).expanduser()
    schema_candidates: list[Path] = []
    if dataset_root.is_file():
        schema_candidates = [dataset_root.with_name("schema.json")]
    else:
        schema_candidates = sorted(dataset_root.glob("*/schema.json"))

    for schema_path in schema_candidates:
        if schema_path.exists():
            with schema_path.open() as f:
                data = json.load(f)
            labels = data.get("teacher_labels")
            if labels:
                return [str(label) for label in labels]
    return list(DEFAULT_LABELS)


def group_streams(dataset_root: str | Path, allowed_labels: set[str]) -> list[list[dict]]:
    streams: list[list[dict]] = []
    for frames_path in discover_frame_files(dataset_root):
        with frames_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]

        buckets: dict[str, list[dict]] = {}
        for row in rows:
            label = str(row["teacher"]["label"])
            if label not in allowed_labels:
                continue
            if row["modalities"].get("ego_planar_scan") is None:
                continue
            key = f"{row['episode_id']}::{row['ego_id']}"
            buckets.setdefault(key, []).append(row)

        for key in sorted(buckets):
            stream = sorted(buckets[key], key=lambda item: int(item["timestamp_ns"]))
            if stream:
                streams.append(stream)

    if not streams:
        raise RuntimeError(f"No usable frame streams found under {dataset_root}")
    return streams


def _edge_lookup(frame: dict) -> dict[tuple[str, str], dict]:
    return {(edge["source"], edge["target"]): edge for edge in frame["edges"]}


def canonical_agent_order(ego_id: str) -> list[str]:
    other_husky = "husky_2" if ego_id == "husky_local" else "husky_local"
    return [ego_id, other_husky, "uav1"]


def node_feature(node: dict, ego_state: dict) -> list[float]:
    state = node["state"]
    goal = node.get("goal") or {"x": state["x"], "y": state["y"], "z": state["z"]}
    command = node.get("command") or {"linear_x": 0.0, "angular_z": 0.0}
    platform = PLATFORM_ONEHOT.get(node["platform_type"], [0.0, 0.0])
    return [
        float(state["x"] - ego_state["x"]),
        float(state["y"] - ego_state["y"]),
        float(state["z"] - ego_state["z"]),
        float(state["vx"]),
        float(state["vy"]),
        float(state["vz"]),
        float(state["wz"]),
        float(goal["x"] - state["x"]),
        float(goal["y"] - state["y"]),
        float(goal["z"] - state["z"]),
        float(command["linear_x"]),
        float(command["angular_z"]),
        float(platform[0]),
        float(platform[1]),
    ]


@lru_cache(maxsize=16384)
def load_npy_cached(path: str) -> np.ndarray:
    return np.load(path)


def resample_scan(scan: np.ndarray, num_beams: int, range_clip: float) -> np.ndarray:
    """Convert an Nx2 scan array to a fixed [2, num_beams] tensor."""
    if scan.ndim != 2 or scan.shape[1] < 2:
        raise ValueError(f"Expected planar scan shape [N,2], got {scan.shape}")

    ranges = np.asarray(scan[:, 0], dtype=np.float32)
    intensities = np.asarray(scan[:, 1], dtype=np.float32)

    ranges = np.nan_to_num(ranges, nan=range_clip, posinf=range_clip, neginf=0.0)
    ranges = np.clip(ranges, 0.0, range_clip)
    intensities = np.nan_to_num(intensities, nan=0.0, posinf=255.0, neginf=0.0)
    intensities = np.clip(intensities, 0.0, 255.0)

    if ranges.shape[0] != num_beams:
        src_x = np.linspace(0.0, 1.0, ranges.shape[0], dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, num_beams, dtype=np.float32)
        ranges = np.interp(dst_x, src_x, ranges).astype(np.float32)
        intensities = np.interp(dst_x, src_x, intensities).astype(np.float32)

    ranges = ranges / max(range_clip, 1e-6)
    intensities = intensities / 255.0
    return np.stack([ranges, intensities], axis=0).astype(np.float32)


@dataclass
class ManeuverSample:
    scan_seq: torch.Tensor
    node_seq: torch.Tensor
    edge_seq: torch.Tensor
    label: torch.Tensor


class HybridManeuverSequenceDataset(Dataset):
    """Build temporal windows from per-ego JSONL frame streams."""

    def __init__(
        self,
        streams: list[list[dict]],
        label_to_idx: dict[str, int],
        past_len: int = 10,
        scan_beams: int = 512,
        range_clip: float = 30.0,
        max_samples: int | None = None,
    ):
        self.streams = streams
        self.label_to_idx = label_to_idx
        self.past_len = past_len
        self.scan_beams = scan_beams
        self.range_clip = range_clip
        self.samples: list[tuple[int, int]] = []

        for stream_idx, stream in enumerate(streams):
            usable = len(stream) - past_len + 1
            for start in range(max(0, usable)):
                self.samples.append((stream_idx, start))

        if max_samples is not None:
            self.samples = self.samples[: max(0, int(max_samples))]

        if not self.samples:
            raise RuntimeError(f"Not enough frames for past_len={past_len}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ManeuverSample:
        stream_idx, start = self.samples[idx]
        stream = self.streams[stream_idx]
        window = stream[start : start + self.past_len]
        target_frame = window[-1]
        ego_id = str(target_frame["ego_id"])
        order = canonical_agent_order(ego_id)

        scan_seq: list[np.ndarray] = []
        node_seq: list[list[list[float]]] = []
        edge_seq: list[list[list[list[float]]]] = []

        for frame in window:
            scan_ref = frame["modalities"]["ego_planar_scan"]
            scan = load_npy_cached(str(scan_ref["path"]))
            scan_seq.append(resample_scan(scan, self.scan_beams, self.range_clip))

            agents = frame["agents"]
            ego_state = agents[ego_id]["state"]
            node_feats = [node_feature(agents[agent_id], ego_state) for agent_id in order]
            node_seq.append(node_feats)

            edge_map = _edge_lookup(frame)
            src_edges = []
            for src in order:
                row_feats = []
                for dst in order:
                    edge = edge_map.get((src, dst))
                    if edge is None:
                        row_feats.append([0.0] * 8)
                    else:
                        row_feats.append(
                            [
                                float(edge["dx"]),
                                float(edge["dy"]),
                                float(edge["dz"]),
                                float(edge["distance"]),
                                float(edge["inv_distance"]),
                                float(edge["bearing_sin"]),
                                float(edge["bearing_cos"]),
                                float(edge["same_platform"]),
                            ]
                        )
                src_edges.append(row_feats)
            edge_seq.append(src_edges)

        label_name = str(target_frame["teacher"]["label"])
        label_idx = self.label_to_idx[label_name]
        return ManeuverSample(
            scan_seq=torch.tensor(np.stack(scan_seq, axis=0), dtype=torch.float32),
            node_seq=torch.tensor(node_seq, dtype=torch.float32),
            edge_seq=torch.tensor(edge_seq, dtype=torch.float32),
            label=torch.tensor(label_idx, dtype=torch.long),
        )


def collate_maneuver_samples(batch: list[ManeuverSample]):
    return (
        torch.stack([sample.scan_seq for sample in batch], dim=0),
        torch.stack([sample.node_seq for sample in batch], dim=0),
        torch.stack([sample.edge_seq for sample in batch], dim=0),
        torch.stack([sample.label for sample in batch], dim=0),
    )


class LidarCNNEncoder(nn.Module):
    """Encode a fixed-size planar scan into a compact embedding."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)


class HybridCNNGNNLSTM(nn.Module):
    """CNN-GNN-LSTM classifier for Husky maneuver prediction."""

    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 8,
        cnn_hidden: int = 96,
        graph_hidden: int = 96,
        fusion_hidden: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        num_classes: int = 10,
        msg_passes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scan_encoder = LidarCNNEncoder(in_channels=2, hidden_dim=cnn_hidden)
        self.graph_encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=graph_hidden,
            msg_passes=msg_passes,
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_hidden + graph_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=fusion_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(
        self,
        scan_seq: torch.Tensor,
        node_seq: torch.Tensor,
        edge_seq: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, steps, _channels, _beams = scan_seq.shape
        temporal_embeddings = []
        for t in range(steps):
            scan_emb = self.scan_encoder(scan_seq[:, t])
            graph_emb = self.graph_encoder(node_seq[:, t], edge_seq[:, t])[:, 0]
            fused = self.fusion(torch.cat([scan_emb, graph_emb], dim=-1))
            temporal_embeddings.append(fused)

        seq = torch.stack(temporal_embeddings, dim=1)
        _, (h, _c) = self.lstm(seq)
        return self.classifier(h[-1])


def build_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    weights = []
    total = sum(counts.values())
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def classification_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    pred_labels = pred.argmax(dim=1)
    accuracy = float((pred_labels == target).float().mean().item())

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for truth, guess in zip(target.cpu(), pred_labels.cpu()):
        confusion[int(truth), int(guess)] += 1

    recalls = []
    precisions = []
    f1s = []
    for idx in range(num_classes):
        tp = float(confusion[idx, idx].item())
        fn = float(confusion[idx, :].sum().item() - tp)
        fp = float(confusion[:, idx].sum().item() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(sum(precisions) / num_classes),
        "macro_recall": float(sum(recalls) / num_classes),
        "macro_f1": float(sum(f1s) / num_classes),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_model(model, loader, device, num_classes: int):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for scan_seq, node_seq, edge_seq, labels in loader:
            scan_seq = scan_seq.to(device)
            node_seq = node_seq.to(device)
            edge_seq = edge_seq.to(device)
            labels = labels.to(device)

            logits = model(scan_seq, node_seq, edge_seq)
            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_count += int(labels.size(0))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = classification_metrics(logits, labels, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    labels = load_schema_labels(args.dataset_root)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    streams = group_streams(args.dataset_root, allowed_labels=set(labels))
    dataset = HybridManeuverSequenceDataset(
        streams=streams,
        label_to_idx=label_to_idx,
        past_len=args.past_len,
        scan_beams=args.scan_beams,
        range_clip=args.range_clip,
        max_samples=args.max_samples,
    )

    total = len(dataset)
    train_len = max(1, int(total * args.train_ratio))
    val_len = max(1, int(total * args.val_ratio))
    test_len = total - train_len - val_len
    if test_len < 1:
        test_len = 1
        if train_len > val_len:
            train_len -= 1
        else:
            val_len -= 1

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_maneuver_samples,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_maneuver_samples,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_maneuver_samples,
    )

    model = HybridCNNGNNLSTM(
        node_dim=14,
        edge_dim=8,
        cnn_hidden=args.cnn_hidden,
        graph_hidden=args.graph_hidden,
        fusion_hidden=args.fusion_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        num_classes=len(labels),
        msg_passes=args.msg_passes,
        dropout=args.dropout,
    ).to(device)

    train_label_indices = [int(dataset[idx].label.item()) for idx in train_set.indices]
    class_weights = build_class_weights(train_label_indices, num_classes=len(labels)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val_f1 = -math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for scan_seq, node_seq, edge_seq, labels_batch in train_loader:
            scan_seq = scan_seq.to(device)
            node_seq = node_seq.to(device)
            edge_seq = edge_seq.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(scan_seq, node_seq, edge_seq)
            loss = criterion(logits, labels_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = labels_batch.size(0)
            running_loss += float(loss.item()) * batch_size
            running_count += int(batch_size)

        train_loss = running_loss / max(running_count, 1)
        val_metrics = evaluate_model(model, val_loader, device, num_classes=len(labels))

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {
                "model_state": model.state_dict(),
                "labels": labels,
                "cfg": {
                    "past_len": args.past_len,
                    "scan_beams": args.scan_beams,
                    "range_clip": args.range_clip,
                    "node_dim": 14,
                    "edge_dim": 8,
                    "cnn_hidden": args.cnn_hidden,
                    "graph_hidden": args.graph_hidden,
                    "fusion_hidden": args.fusion_hidden,
                    "lstm_hidden": args.lstm_hidden,
                    "lstm_layers": args.lstm_layers,
                    "num_classes": len(labels),
                    "msg_passes": args.msg_passes,
                    "dropout": args.dropout,
                },
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.model_name
    summary_path = output_dir / args.summary_name

    torch.save(best_state, model_path)
    model.load_state_dict(best_state["model_state"])
    test_metrics = evaluate_model(model, test_loader, device, num_classes=len(labels))

    summary = {
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "total_samples": total,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "labels": labels,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "class_weights": class_weights.detach().cpu().tolist(),
        "args": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved model to {model_path}")
    print(f"Saved summary to {summary_path}")
    print(
        f"Test metrics: "
        f"acc={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"macro_recall={test_metrics['macro_recall']:.4f}"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="hybrid_maneuver_dataset",
        help="Path to the exported hybrid maneuver dataset root.",
    )
    parser.add_argument("--past-len", type=int, default=10)
    parser.add_argument("--scan-beams", type=int, default=512)
    parser.add_argument("--range-clip", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-hidden", type=int, default=96)
    parser.add_argument("--graph-hidden", type=int, default=96)
    parser.add_argument("--fusion-hidden", type=int, default=128)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--msg-passes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="hybrid_cnn_gnn_lstm_maneuver.pt")
    parser.add_argument("--summary-name", type=str, default="hybrid_cnn_gnn_lstm_maneuver_summary.json")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
