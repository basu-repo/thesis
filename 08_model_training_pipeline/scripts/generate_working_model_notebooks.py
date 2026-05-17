#!/usr/bin/env python3
"""Generate working 08 training notebooks that save outputs to disk."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_ROOT = Path.home() / "Documents/Thesis" / "08_model_training_pipeline" / "notebooks"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").splitlines()],
    }


COMMON_IMPORTS = """
import gc
import json
import sys
from pathlib import Path

import torch

gc.collect()
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception:
    pass

PROJECT_ROOT = Path.home() / "Documents/Thesis"
PIPELINE_ROOT = PROJECT_ROOT / "08_model_training_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from training.notebook_workflow import (
    CNNGNNLSTMTrajectoryPredictor,
    CNNGNNLSTMTransformerTrajectoryPredictor,
    CNNGNNTransformerTrajectoryPredictor,
    CNNLSTMTrajectoryPredictor,
    ScanGoalTrajectoryDataset,
    ScanGraphTrajectoryDataset,
    collate_scan,
    collate_scan_graph,
    device_from_flag,
    evaluate_trajectory_model,
    load_or_build_shared_artifacts,
    make_dataloaders,
    prepare_result_dirs,
    run_constant_velocity_baseline,
    save_final_trajectory_evaluation,
    set_seed,
    timestamp_tag,
    train_trajectory_model,
)
"""


COMMON_CONFIG = """
SEED = 42
PAST_LEN = 10
FUTURE_LEN = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
MAX_SAMPLES = None
USE_CPU = False

BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

GOAL_DIM = 13
NODE_DIM = 12
EDGE_DIM = 8
HIDDEN_DIM = 128
GRAPH_HIDDEN = 96
DROPOUT = 0.10
MSG_PASSES = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF = 256

device = device_from_flag(USE_CPU)
print("Device:", device)
"""


SHARED_LOAD = """
set_seed(SEED)
streams, sample_table, split_info, sample_table_path, split_path = load_or_build_shared_artifacts(
    past_len=PAST_LEN,
    future_len=FUTURE_LEN,
    seed=SEED,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
)
print("Sample table:", sample_table_path)
print("Split path:", split_path)
print("Split strategy:", split_info["split_strategy"])
print("Episode count:", split_info["episode_count"])
print("Train / Val / Test samples:", len(split_info["train_indices"]), len(split_info["val_indices"]), len(split_info["test_indices"]))
print("Train episodes:", split_info["train_episode_ids"])
print("Val episodes:", split_info["val_episode_ids"])
print("Test episodes:", split_info["test_episode_ids"])
"""


def notebook_payload(title: str, cells: list[dict]) -> dict:
    return {
        "cells": [md_cell(title)] + cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(name: str, payload: dict) -> None:
    path = NOTEBOOK_ROOT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)

    write_notebook(
        "10_cv_baseline.ipynb",
        notebook_payload(
            "# 10 CV Baseline\n\nRun this notebook after `00_data_loading_and_split.ipynb`. It evaluates a restart-safe constant-velocity trajectory baseline and saves metrics, plots, and predictions to disk.",
            [
                code_cell(COMMON_IMPORTS),
                code_cell(COMMON_CONFIG),
                code_cell(SHARED_LOAD),
                code_cell(
                    """
MODEL_SLUG = "cv_baseline"
TIMESTAMP = timestamp_tag()
result_dir, weight_dir, plot_dir = prepare_result_dirs(MODEL_SLUG)
final_metrics = run_constant_velocity_baseline(
    streams=streams,
    sample_table=sample_table,
    split_info=split_info,
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    plot_dir=plot_dir,
)
print(json.dumps(final_metrics, indent=2))
"""
                ),
            ],
        ),
    )

    write_notebook(
        "20_cnn_lstm.ipynb",
        notebook_payload(
            "# 20 CNN LSTM\n\nUses past obstacle-clearance strips plus ego/goal sequence features. All outputs are saved to disk for restart-safe experiments.",
            [
                code_cell(COMMON_IMPORTS),
                code_cell(COMMON_CONFIG),
                code_cell(SHARED_LOAD),
                code_cell(
                    """
MODEL_SLUG = "cnn_lstm"
TIMESTAMP = timestamp_tag()
result_dir, weight_dir, plot_dir = prepare_result_dirs(MODEL_SLUG)

dataset = ScanGoalTrajectoryDataset(streams, sample_table, PAST_LEN)
train_loader, val_loader, test_loader = make_dataloaders(
    dataset,
    split_info,
    batch_size=BATCH_SIZE,
    collate_fn=collate_scan,
    max_samples=MAX_SAMPLES,
)

model = CNNLSTMTrajectoryPredictor(
    goal_dim=GOAL_DIM,
    hidden_dim=HIDDEN_DIM,
    cnn_hidden=HIDDEN_DIM,
    future_len=FUTURE_LEN,
    dropout=DROPOUT,
).to(device)

train_out = train_trajectory_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    weight_dir=weight_dir,
    plot_dir=plot_dir,
    epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    extra_manifest={"family": "scan_goal_cnn_lstm", "future_len": FUTURE_LEN},
)
test_eval = evaluate_trajectory_model(model, test_loader, device)
final_metrics = save_final_trajectory_evaluation(
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    train_out=train_out,
    test_eval=test_eval,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    plot_dir=plot_dir,
)
print(json.dumps(final_metrics, indent=2))
"""
                ),
            ],
        ),
    )

    write_notebook(
        "40_cnn_gnn_lstm.ipynb",
        notebook_payload(
            "# 40 CNN GNN LSTM\n\nUses clearance strips plus a small ego-uav graph sequence. Saves all artifacts to disk so each run survives a kernel restart.",
            [
                code_cell(COMMON_IMPORTS),
                code_cell(COMMON_CONFIG),
                code_cell(SHARED_LOAD),
                code_cell(
                    """
MODEL_SLUG = "cnn_gnn_lstm"
TIMESTAMP = timestamp_tag()
result_dir, weight_dir, plot_dir = prepare_result_dirs(MODEL_SLUG)

dataset = ScanGraphTrajectoryDataset(streams, sample_table, PAST_LEN)
train_loader, val_loader, test_loader = make_dataloaders(
    dataset,
    split_info,
    batch_size=BATCH_SIZE,
    collate_fn=collate_scan_graph,
    max_samples=MAX_SAMPLES,
)

model = CNNGNNLSTMTrajectoryPredictor(
    goal_dim=GOAL_DIM,
    node_dim=NODE_DIM,
    edge_dim=EDGE_DIM,
    hidden_dim=HIDDEN_DIM,
    graph_hidden=GRAPH_HIDDEN,
    future_len=FUTURE_LEN,
    dropout=DROPOUT,
    msg_passes=MSG_PASSES,
).to(device)

train_out = train_trajectory_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    weight_dir=weight_dir,
    plot_dir=plot_dir,
    epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    extra_manifest={"family": "scan_graph_lstm", "future_len": FUTURE_LEN, "msg_passes": MSG_PASSES},
)
test_eval = evaluate_trajectory_model(model, test_loader, device)
final_metrics = save_final_trajectory_evaluation(
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    train_out=train_out,
    test_eval=test_eval,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    plot_dir=plot_dir,
)
print(json.dumps(final_metrics, indent=2))
"""
                ),
            ],
        ),
    )

    write_notebook(
        "50_cnn_gnn_transformer.ipynb",
        notebook_payload(
            "# 50 CNN GNN Transformer\n\nTransformer-based trajectory model using the new `08` dataset fields. Saves metrics, weights, plots, and predictions to disk.",
            [
                code_cell(COMMON_IMPORTS),
                code_cell(COMMON_CONFIG),
                code_cell(SHARED_LOAD),
                code_cell(
                    """
MODEL_SLUG = "cnn_gnn_transformer"
TIMESTAMP = timestamp_tag()
result_dir, weight_dir, plot_dir = prepare_result_dirs(MODEL_SLUG)

dataset = ScanGraphTrajectoryDataset(streams, sample_table, PAST_LEN)
train_loader, val_loader, test_loader = make_dataloaders(
    dataset,
    split_info,
    batch_size=BATCH_SIZE,
    collate_fn=collate_scan_graph,
    max_samples=MAX_SAMPLES,
)

model = CNNGNNTransformerTrajectoryPredictor(
    goal_dim=GOAL_DIM,
    node_dim=NODE_DIM,
    edge_dim=EDGE_DIM,
    hidden_dim=HIDDEN_DIM,
    graph_hidden=GRAPH_HIDDEN,
    future_len=FUTURE_LEN,
    dropout=DROPOUT,
    msg_passes=MSG_PASSES,
    num_heads=TRANSFORMER_HEADS,
    num_layers=TRANSFORMER_LAYERS,
    ff_dim=TRANSFORMER_FF,
).to(device)

train_out = train_trajectory_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    weight_dir=weight_dir,
    plot_dir=plot_dir,
    epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    extra_manifest={"family": "scan_graph_transformer", "future_len": FUTURE_LEN, "msg_passes": MSG_PASSES},
)
test_eval = evaluate_trajectory_model(model, test_loader, device)
final_metrics = save_final_trajectory_evaluation(
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    train_out=train_out,
    test_eval=test_eval,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    plot_dir=plot_dir,
)
print(json.dumps(final_metrics, indent=2))
"""
                ),
            ],
        ),
    )

    write_notebook(
        "60_cnn_gnn_lstm_transformer.ipynb",
        notebook_payload(
            "# 60 CNN GNN LSTM Transformer\n\nHybrid LSTM/Transformer trajectory model using the same saved split as every other notebook.",
            [
                code_cell(COMMON_IMPORTS),
                code_cell(COMMON_CONFIG),
                code_cell(SHARED_LOAD),
                code_cell(
                    """
MODEL_SLUG = "cnn_gnn_lstm_transformer"
TIMESTAMP = timestamp_tag()
result_dir, weight_dir, plot_dir = prepare_result_dirs(MODEL_SLUG)

dataset = ScanGraphTrajectoryDataset(streams, sample_table, PAST_LEN)
train_loader, val_loader, test_loader = make_dataloaders(
    dataset,
    split_info,
    batch_size=BATCH_SIZE,
    collate_fn=collate_scan_graph,
    max_samples=MAX_SAMPLES,
)

model = CNNGNNLSTMTransformerTrajectoryPredictor(
    goal_dim=GOAL_DIM,
    node_dim=NODE_DIM,
    edge_dim=EDGE_DIM,
    hidden_dim=HIDDEN_DIM,
    graph_hidden=GRAPH_HIDDEN,
    future_len=FUTURE_LEN,
    dropout=DROPOUT,
    msg_passes=MSG_PASSES,
    num_heads=TRANSFORMER_HEADS,
    num_layers=TRANSFORMER_LAYERS,
    ff_dim=TRANSFORMER_FF,
).to(device)

train_out = train_trajectory_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    weight_dir=weight_dir,
    plot_dir=plot_dir,
    epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    extra_manifest={"family": "scan_graph_lstm_transformer", "future_len": FUTURE_LEN, "msg_passes": MSG_PASSES},
)
test_eval = evaluate_trajectory_model(model, test_loader, device)
final_metrics = save_final_trajectory_evaluation(
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    train_out=train_out,
    test_eval=test_eval,
    split_path=split_path,
    sample_table_path=sample_table_path,
    result_dir=result_dir,
    plot_dir=plot_dir,
)
print(json.dumps(final_metrics, indent=2))
"""
                ),
            ],
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
