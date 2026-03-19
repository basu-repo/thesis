#!/usr/bin/env python3
"""Compare saved trajectory-prediction summaries side by side.

This script reads the model summary JSON files from ``models/`` and prints a
compact comparison table for:
- Constant Velocity
- GNN-LSTM
- CNN-GNN-LSTM
"""

import json
from pathlib import Path


MODELS_DIR = Path.home() / "Documents/Thesis/models"
GNN_SUMMARY = MODELS_DIR / "summary_gnn_graph_done.json"
CNN_GNN_SUMMARY = MODELS_DIR / "summary_cnn_gnn_graph_done.json"


def load_summary(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_row(name: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"{name:<18} {'missing':>10} {'missing':>10} {'missing':>10}"
    return (
        f"{name:<18} "
        f"{metrics.get('ADE', float('nan')):>10.6f} "
        f"{metrics.get('FDE', float('nan')):>10.6f} "
        f"{metrics.get('RMSE', float('nan')):>10.6f}"
    )


def main():
    gnn = load_summary(GNN_SUMMARY)
    cnn_gnn = load_summary(CNN_GNN_SUMMARY)

    cv_metrics = None
    gnn_metrics = None
    cnn_gnn_metrics = None

    if gnn is not None:
        cv_metrics = gnn.get("comparison", {}).get("constant_velocity")
        gnn_metrics = gnn.get("comparison", {}).get("gnn_lstm")

    if cnn_gnn is not None:
        if cv_metrics is None:
            cv_metrics = cnn_gnn.get("comparison", {}).get("constant_velocity")
        cnn_gnn_metrics = cnn_gnn.get("comparison", {}).get("cnn_gnn_lstm")

    print("\nGraph Model Comparison")
    print("-" * 54)
    print(f"{'Model':<18} {'ADE':>10} {'FDE':>10} {'RMSE':>10}")
    print("-" * 54)
    print(metric_row("Constant Velocity", cv_metrics))
    print(metric_row("GNN-LSTM", gnn_metrics))
    print(metric_row("CNN-GNN-LSTM", cnn_gnn_metrics))
    print("-" * 54)

    if gnn is None:
        print(f"Missing summary: {GNN_SUMMARY}")
    if cnn_gnn is None:
        print(f"Missing summary: {CNN_GNN_SUMMARY}")


if __name__ == "__main__":
    main()
