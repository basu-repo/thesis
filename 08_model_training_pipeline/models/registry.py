"""Central model registry for the new training pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    family: str
    uses_scan: bool
    uses_graph: bool
    predicts_trajectory: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "cnn_lstm": ModelSpec(
        slug="cnn_lstm",
        family="scan_temporal",
        uses_scan=True,
        uses_graph=False,
    ),
    "cnn_gnn_lstm": ModelSpec(
        slug="cnn_gnn_lstm",
        family="scan_graph_temporal",
        uses_scan=True,
        uses_graph=True,
    ),
    "cnn_gnn_transformer": ModelSpec(
        slug="cnn_gnn_transformer",
        family="scan_graph_transformer",
        uses_scan=True,
        uses_graph=True,
    ),
    "cnn_gnn_lstm_transformer": ModelSpec(
        slug="cnn_gnn_lstm_transformer",
        family="scan_graph_hybrid_transformer",
        uses_scan=True,
        uses_graph=True,
    ),
}


def registered_model_slugs() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())
