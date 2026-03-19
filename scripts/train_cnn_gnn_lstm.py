#!/usr/bin/env python3
"""Train a CNN-GNN-LSTM trajectory predictor on exported graph frames.

This keeps the existing GNN-LSTM pipeline untouched and adds a second model
variant for comparison. The new model uses:
- a temporal CNN to encode per-agent history
- a GNN to model interactions at each time step
- an LSTM head to predict the future ego trajectory
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from graph_predictor import (
    GraphEncoder,
    GraphSequenceDataset,
    constant_velocity_predict,
    load_runs,
    trajectory_metrics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_graph_samples(batch):
    node_seq = torch.stack([sample.node_seq for sample in batch], dim=0)
    edge_seq = torch.stack([sample.edge_seq for sample in batch], dim=0)
    target_future = torch.stack([sample.target_future for sample in batch], dim=0)
    past_xy = torch.stack([sample.past_xy for sample in batch], dim=0)
    return node_seq, edge_seq, target_future, past_xy


class CNNGNNLSTM(nn.Module):
    """Encode node histories with a temporal CNN before graph reasoning."""

    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 4,
        cnn_hidden: int = 64,
        graph_hidden: int = 96,
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
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(node_dim, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden, graph_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.graph_encoder = GraphEncoder(
            node_dim=graph_hidden,
            edge_dim=edge_dim,
            hidden_dim=graph_hidden,
            msg_passes=msg_passes,
        )
        self.lstm = nn.LSTM(
            input_size=graph_hidden,
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
        batch_size, steps, num_nodes, node_dim = node_seq.shape

        # Encode each node's temporal history before reasoning over the graph.
        temporal_in = node_seq.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, node_dim, steps)
        temporal_out = self.temporal_cnn(temporal_in)
        temporal_out = temporal_out.permute(0, 2, 1).reshape(batch_size, num_nodes, steps, -1)
        temporal_out = temporal_out.permute(0, 2, 1, 3)

        encoded_steps = []
        for t in range(steps):
            node_emb = self.graph_encoder(temporal_out[:, t], edge_seq[:, t])
            encoded_steps.append(node_emb[:, self.ego_idx])

        ego_seq = torch.stack(encoded_steps, dim=1)
        _, (h, _) = self.lstm(ego_seq)
        out = self.head(h[-1])
        return out.view(-1, self.future_len, 2)


def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for node_seq, edge_seq, target_future, _past_xy in loader:
            node_seq = node_seq.to(device)
            edge_seq = edge_seq.to(device)
            target_future = target_future.to(device)
            pred = model(node_seq, edge_seq)
            preds.append(pred.cpu())
            targets.append(target_future.cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return trajectory_metrics(pred, target)


def evaluate_cv(loader):
    preds = []
    targets = []
    with torch.no_grad():
        for _node_seq, _edge_seq, target_future, past_xy in loader:
            pred = constant_velocity_predict(past_xy, target_future.size(1))
            preds.append(pred)
            targets.append(target_future)
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return trajectory_metrics(pred, target)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    runs = load_runs(args.graph_root)
    dataset = GraphSequenceDataset(
        runs=runs,
        past_len=args.past_len,
        future_len=args.future_len,
        ego_node=args.ego_node,
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
        collate_fn=collate_graph_samples,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_samples,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_samples,
    )

    cfg = {
        "node_dim": 14,
        "edge_dim": 4,
        "cnn_hidden": args.cnn_hidden,
        "graph_hidden": args.graph_hidden,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "future_len": args.future_len,
        "ego_idx": 0,
        "msg_passes": args.msg_passes,
        "dropout": args.dropout,
    }
    model = CNNGNNLSTM(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for node_seq, edge_seq, target_future, _past_xy in train_loader:
            node_seq = node_seq.to(device)
            edge_seq = edge_seq.to(device)
            target_future = target_future.to(device)

            optimizer.zero_grad()
            pred = model(node_seq, edge_seq)
            loss = criterion(pred, target_future)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * node_seq.size(0)

        train_loss = running_loss / len(train_set)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for node_seq, edge_seq, target_future, _past_xy in val_loader:
                node_seq = node_seq.to(device)
                edge_seq = edge_seq.to(device)
                target_future = target_future.to(device)
                pred = model(node_seq, edge_seq)
                loss = criterion(pred, target_future)
                val_running += loss.item() * node_seq.size(0)
        val_loss = val_running / len(val_set)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "cfg": {
                    **cfg,
                    "past_len": args.past_len,
                    "ego_node": args.ego_node,
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
    test_metrics = evaluate_model(model, test_loader, device)
    cv_metrics = evaluate_cv(test_loader)
    comparison = {
        "constant_velocity": cv_metrics,
        "cnn_gnn_lstm": test_metrics,
    }

    summary = {
        "model": "cnn_gnn_lstm",
        "graph_root": str(Path(args.graph_root).expanduser()),
        "runs": len(runs),
        "samples": total,
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "comparison": comparison,
        "ADE": test_metrics["ADE"],
        "FDE": test_metrics["FDE"],
        "RMSE": test_metrics["RMSE"],
        "model_path": str(model_path.resolve()),
        "past_len": args.past_len,
        "future_len": args.future_len,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nComparison:")
    print(json.dumps(comparison, indent=2))
    print("\nSaved model to:", model_path)
    print("Saved summary to:", summary_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a CNN-GNN-LSTM predictor on exported graph_dataset frames."
    )
    parser.add_argument(
        "--graph-root",
        default=str(Path.home() / "Documents/Thesis/graph_dataset"),
        help="Directory containing run subfolders with frames.jsonl files.",
    )
    parser.add_argument("--ego-node", default="husky_local")
    parser.add_argument("--past-len", type=int, default=10)
    parser.add_argument("--future-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cnn-hidden", type=int, default=64)
    parser.add_argument("--graph-hidden", type=int, default=96)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--msg-passes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Documents/Thesis/models"),
    )
    parser.add_argument("--model-name", default="cnn_gnn_lstm_graph_done.pt")
    parser.add_argument("--summary-name", default="summary_cnn_gnn_graph_done.json")
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
