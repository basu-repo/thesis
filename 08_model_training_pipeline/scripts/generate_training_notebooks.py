#!/usr/bin/env python3
"""Generate simple per-model training notebooks for the 08 pipeline."""

from __future__ import annotations

import json
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_ROOT = PIPELINE_ROOT / "notebooks"


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def common_setup_code() -> str:
    return """
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path.home() / "Documents/Thesis"
PIPELINE_ROOT = PROJECT_ROOT / "08_model_training_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

RESULTS_ROOT = PIPELINE_ROOT / "results"
EPISODE_FRAMES_ROOT = RESULTS_ROOT / "episode_frames"
TRAIN_READY_ROOT = RESULTS_ROOT / "train_ready"
SPLITS_ROOT = RESULTS_ROOT / "splits"
NORMALIZATION_ROOT = RESULTS_ROOT / "normalization"
MODEL_WEIGHTS_ROOT = PIPELINE_ROOT / "model_weights"

SEED = 42
PAST_LEN = 10
FUTURE_LEN = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

print("Pipeline root:", PIPELINE_ROOT)
print("Episode frames root:", EPISODE_FRAMES_ROOT)
"""


def data_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # 00 Data Loading And Split

            Use this notebook first.
            It is the place to:

            - inspect exported `episode_frames`
            - build the train-ready sample table
            - save one shared train/val/test split file
            - reuse that same split for every model notebook
            """
        ),
        code_cell(common_setup_code()),
        code_cell(
            """
sample_files = sorted(EPISODE_FRAMES_ROOT.glob("*/frames.jsonl"))
print("Episode files:", len(sample_files))
for path in sample_files[:10]:
    print(path)
"""
        ),
        code_cell(
            """
# Next step to implement in 08:
# 1. load frames.jsonl
# 2. build sliding-window trajectory samples
# 3. save a sample table under TRAIN_READY_ROOT
# 4. save one shared split JSON under SPLITS_ROOT
"""
        ),
    ]
    return notebook(cells)


def model_notebook(model_slug: str, title: str) -> dict:
    cells = [
        markdown_cell(
            f"""
            # {title}

            Run this notebook independently for the `{model_slug}` model.

            Intended workflow:
            - restart kernel
            - run setup cells
            - load the shared sample table
            - load the shared split file
            - train only this model
            - save weights and results for this model
            """
        ),
        code_cell(common_setup_code()),
        code_cell(
            f"""
MODEL_SLUG = "{model_slug}"
RUNS_ROOT = PIPELINE_ROOT / "results" / MODEL_SLUG
WEIGHTS_ROOT = MODEL_WEIGHTS_ROOT / MODEL_SLUG

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

print("Model:", MODEL_SLUG)
print("Result dir:", RUNS_ROOT)
print("Weight dir:", WEIGHTS_ROOT)
"""
        ),
        code_cell(
            """
# Expected shared artifacts:
# - TRAIN_READY_ROOT / ...
# - SPLITS_ROOT / ...
#
# Load them here once the sample builder is ready.
"""
        ),
        code_cell(
            """
# Training cell placeholder.
# Put model definition, dataloaders, optimizer, and training loop here.
"""
        ),
        code_cell(
            """
# Evaluation / export cell placeholder.
# Save:
# - metrics.json
# - prediction arrays
# - plots
# - latest checkpoint
"""
        ),
    ]
    return notebook(cells)


def comparison_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # 90 Model Comparison

            Use this notebook after individual model notebooks finish.

            Intended use:
            - load saved metrics from each model
            - compare val/test performance
            - produce summary tables and plots
            """
        ),
        code_cell(common_setup_code()),
        code_cell(
            """
model_dirs = sorted((PIPELINE_ROOT / "results").glob("*"))
for path in model_dirs:
    if path.is_dir():
        print(path.name)
"""
        ),
    ]
    return notebook(cells)


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    notebooks = {
        "00_data_loading_and_split.ipynb": data_notebook(),
        "10_lstm_goal.ipynb": model_notebook("lstm_goal", "10 LSTM Goal"),
        "20_cnn_lstm.ipynb": model_notebook("cnn_lstm", "20 CNN LSTM"),
        "30_gnn_lstm.ipynb": model_notebook("gnn_lstm", "30 GNN LSTM"),
        "40_cnn_gnn_lstm.ipynb": model_notebook("cnn_gnn_lstm", "40 CNN GNN LSTM"),
        "50_cnn_gnn_transformer.ipynb": model_notebook("cnn_gnn_transformer", "50 CNN GNN Transformer"),
        "60_cnn_gnn_lstm_transformer.ipynb": model_notebook(
            "cnn_gnn_lstm_transformer",
            "60 CNN GNN LSTM Transformer",
        ),
        "90_model_comparison.ipynb": comparison_notebook(),
    }

    for name, payload in notebooks.items():
        write_notebook(NOTEBOOKS_ROOT / name, payload)
        print("Wrote", NOTEBOOKS_ROOT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
