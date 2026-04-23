"""Live AI model predictors used for final evaluation."""

from __future__ import annotations

import torch
from torch import nn


NODE_ORDER = ["husky_local", "husky_2", "uav1"]
ARCH_GRAPH_ONLY_LSTM = "graph_only_lstm"
ARCH_SCAN_ONLY_LSTM = "scan_only_lstm"
ARCH_SCAN_GRAPH_LSTM = "scan_graph_lstm"
ARCH_SCAN_GRAPH_TRANSFORMER = "scan_graph_transformer"
ARCH_SCAN_GRAPH_LSTM_TRANSFORMER = "scan_graph_lstm_transformer"


class GraphEncoder(nn.Module):
    """Small message-passing encoder aligned with the training notebook."""

    def __init__(self, node_dim: int = 14, edge_dim: int = 8, hidden_dim: int = 96, msg_passes: int = 2):
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


class AIModelPredictor(nn.Module):
    """Graph-only temporal predictor that outputs future ego waypoints."""

    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 8,
        hidden_dim: int = 96,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        future_len: int = 5,
        ego_idx: int = 0,
        msg_passes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_len = future_len
        self.ego_idx = ego_idx
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            msg_passes=msg_passes,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.traj_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, future_len * 2),
        )

    def forward(self, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> torch.Tensor:
        encoded_steps = []
        for t in range(node_seq.size(1)):
            node_emb = self.encoder(node_seq[:, t], edge_seq[:, t])
            encoded_steps.append(node_emb[:, self.ego_idx])
        seq = torch.stack(encoded_steps, dim=1)
        _, (h, _c) = self.lstm(seq)
        hidden = h[-1]
        future_xy = self.traj_head(hidden).view(hidden.size(0), self.future_len, 2)
        return future_xy


class LidarCNNEncoder(nn.Module):
    """1D CNN encoder for the planar lidar scan."""

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
        return self.net(x).squeeze(-1)


class CNNGNNLSTMPredictor(nn.Module):
    """Hybrid lidar+graph temporal predictor for future ego waypoints."""

    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 8,
        cnn_hidden: int = 96,
        graph_hidden: int = 96,
        fusion_hidden: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        future_len: int = 5,
        ego_idx: int = 0,
        msg_passes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_len = future_len
        self.ego_idx = ego_idx
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
            nn.Linear(lstm_hidden, 5),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, future_len * 2),
        )

    def forward(
        self,
        scan_seq: torch.Tensor,
        node_seq: torch.Tensor,
        edge_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_steps = []
        for t in range(scan_seq.size(1)):
            scan_emb = self.scan_encoder(scan_seq[:, t])
            graph_emb = self.graph_encoder(node_seq[:, t], edge_seq[:, t])[:, self.ego_idx]
            fused_steps.append(self.fusion(torch.cat([scan_emb, graph_emb], dim=-1)))
        seq = torch.stack(fused_steps, dim=1)
        _, (h, _c) = self.lstm(seq)
        hidden = h[-1]
        logits = self.classifier(hidden)
        future_xy = self.traj_head(hidden).view(hidden.size(0), self.future_len, 2)
        return logits, future_xy


class LearnablePositionalEncoding(nn.Module):
    """A small learnable positional encoding for short temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, : x.size(1)]


class CNNGNNTransformerPredictor(nn.Module):
    """Transformer over fused CNN+GNN timestep embeddings for trajectory prediction."""

    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 8,
        cnn_hidden: int = 96,
        graph_hidden: int = 96,
        fusion_hidden: int = 128,
        future_len: int = 5,
        ego_idx: int = 0,
        msg_passes: int = 2,
        dropout: float = 0.1,
        transformer_heads: int = 4,
        transformer_ff: int = 256,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.future_len = future_len
        self.ego_idx = ego_idx
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
        self.pos = LearnablePositionalEncoding(fusion_hidden, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_hidden,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.traj_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, future_len * 2),
        )

    def forward(
        self,
        scan_seq: torch.Tensor,
        node_seq: torch.Tensor,
        edge_seq: torch.Tensor,
    ) -> torch.Tensor:
        fused_steps = []
        for t in range(scan_seq.size(1)):
            scan_emb = self.scan_encoder(scan_seq[:, t])
            graph_emb = self.graph_encoder(node_seq[:, t], edge_seq[:, t])[:, self.ego_idx]
            fused_steps.append(self.fusion(torch.cat([scan_emb, graph_emb], dim=-1)))
        seq = torch.stack(fused_steps, dim=1)
        encoded = self.transformer(self.pos(seq))
        pooled = encoded.mean(dim=1)
        future_xy = self.traj_head(pooled).view(pooled.size(0), self.future_len, 2)
        return future_xy


class CNNGNNLSTMTransformerPredictor(nn.Module):
    """LSTM first, transformer second, on fused CNN+GNN embeddings for trajectory prediction."""

    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 8,
        cnn_hidden: int = 96,
        graph_hidden: int = 96,
        fusion_hidden: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        future_len: int = 5,
        ego_idx: int = 0,
        msg_passes: int = 2,
        dropout: float = 0.1,
        transformer_heads: int = 4,
        transformer_ff: int = 256,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.future_len = future_len
        self.ego_idx = ego_idx
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
        self.pos = LearnablePositionalEncoding(lstm_hidden, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.traj_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, future_len * 2),
        )

    def forward(
        self,
        scan_seq: torch.Tensor,
        node_seq: torch.Tensor,
        edge_seq: torch.Tensor,
    ) -> torch.Tensor:
        fused_steps = []
        for t in range(scan_seq.size(1)):
            scan_emb = self.scan_encoder(scan_seq[:, t])
            graph_emb = self.graph_encoder(node_seq[:, t], edge_seq[:, t])[:, self.ego_idx]
            fused_steps.append(self.fusion(torch.cat([scan_emb, graph_emb], dim=-1)))
        seq = torch.stack(fused_steps, dim=1)
        lstm_out, _ = self.lstm(seq)
        encoded = self.transformer(self.pos(lstm_out))
        pooled = encoded.mean(dim=1)
        future_xy = self.traj_head(pooled).view(pooled.size(0), self.future_len, 2)
        return future_xy


def infer_runtime_architecture(checkpoint: dict) -> str:
    """Infer the live runtime architecture from checkpoint contents."""

    state = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or {}
    keys = set(state.keys())
    run_manifest = checkpoint.get("run_manifest") or {}
    model_slug = str(run_manifest.get("model_slug") or checkpoint.get("model_slug") or "").lower()

    has_scan_encoder = any(key.startswith("scan_encoder.") for key in keys)
    has_graph_encoder = any(key.startswith("graph_encoder.") for key in keys)
    has_transformer = any(key.startswith("transformer.") for key in keys)
    has_lstm = any(key.startswith("lstm.") for key in keys)
    has_encoder = any(key.startswith("encoder.") for key in keys)
    has_pos = any(key.startswith("pos.") for key in keys)

    if has_scan_encoder and has_graph_encoder and has_transformer and has_lstm:
        return ARCH_SCAN_GRAPH_LSTM_TRANSFORMER
    if has_scan_encoder and has_graph_encoder and (has_transformer or has_pos):
        return ARCH_SCAN_GRAPH_TRANSFORMER
    if any(key.startswith("scan_encoder.") for key in keys):
        return ARCH_SCAN_GRAPH_LSTM
    if any(key.startswith("fusion.") for key in keys) and any(key.startswith("graph_encoder.") for key in keys):
        return ARCH_SCAN_GRAPH_LSTM
    if has_encoder or model_slug == "cnn_lstm":
        return ARCH_SCAN_ONLY_LSTM
    if model_slug == "cnn_gnn_lstm_transformer":
        return ARCH_SCAN_GRAPH_LSTM_TRANSFORMER
    if model_slug == "cnn_gnn_transformer":
        return ARCH_SCAN_GRAPH_TRANSFORMER
    if model_slug.startswith("cnn_"):
        return ARCH_SCAN_GRAPH_LSTM
    return ARCH_GRAPH_ONLY_LSTM


def architecture_requires_scan(architecture: str) -> bool:
    return architecture in {
        ARCH_SCAN_ONLY_LSTM,
        ARCH_SCAN_GRAPH_LSTM,
        ARCH_SCAN_GRAPH_TRANSFORMER,
        ARCH_SCAN_GRAPH_LSTM_TRANSFORMER,
    }


def build_runtime_model(
    architecture: str,
    cfg: dict,
    *,
    ego_idx: int,
) -> nn.Module:
    """Instantiate the correct live predictor from an inferred architecture."""

    if architecture == ARCH_SCAN_GRAPH_LSTM:
        return CNNGNNLSTMPredictor(
            node_dim=cfg["node_dim"],
            edge_dim=cfg["edge_dim"],
            cnn_hidden=cfg.get("cnn_hidden", 96),
            graph_hidden=cfg.get("graph_hidden", 96),
            fusion_hidden=cfg.get("fusion_hidden", 128),
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            future_len=cfg["future_len"],
            ego_idx=ego_idx,
            msg_passes=cfg.get("msg_passes", 2),
            dropout=cfg.get("dropout", 0.1),
        )

    if architecture == ARCH_SCAN_GRAPH_TRANSFORMER:
        return CNNGNNTransformerPredictor(
            node_dim=cfg["node_dim"],
            edge_dim=cfg["edge_dim"],
            cnn_hidden=cfg.get("cnn_hidden", 96),
            graph_hidden=cfg.get("graph_hidden", 96),
            fusion_hidden=cfg.get("fusion_hidden", 128),
            future_len=cfg["future_len"],
            ego_idx=ego_idx,
            msg_passes=cfg.get("msg_passes", 2),
            dropout=cfg.get("dropout", 0.1),
            transformer_heads=cfg.get("transformer_heads", 4),
            transformer_ff=cfg.get("transformer_ff", 256),
            transformer_layers=cfg.get("transformer_layers", 2),
        )

    if architecture == ARCH_SCAN_GRAPH_LSTM_TRANSFORMER:
        return CNNGNNLSTMTransformerPredictor(
            node_dim=cfg["node_dim"],
            edge_dim=cfg["edge_dim"],
            cnn_hidden=cfg.get("cnn_hidden", 96),
            graph_hidden=cfg.get("graph_hidden", 96),
            fusion_hidden=cfg.get("fusion_hidden", 128),
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            future_len=cfg["future_len"],
            ego_idx=ego_idx,
            msg_passes=cfg.get("msg_passes", 2),
            dropout=cfg.get("dropout", 0.1),
            transformer_heads=cfg.get("transformer_heads", 4),
            transformer_ff=cfg.get("transformer_ff", 256),
            transformer_layers=cfg.get("transformer_layers", 2),
        )

    if architecture == ARCH_GRAPH_ONLY_LSTM:
        return AIModelPredictor(
            node_dim=cfg["node_dim"],
            edge_dim=cfg["edge_dim"],
            hidden_dim=cfg["hidden_dim"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            future_len=cfg["future_len"],
            ego_idx=ego_idx,
            msg_passes=cfg.get("msg_passes", 2),
            dropout=cfg.get("dropout", 0.1),
        )

    raise ValueError(f"Unsupported runtime architecture: {architecture}")
