"""Choose the best saved 08 trajectory model checkpoint for live evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


THESIS_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = THESIS_ROOT / "model_training"
SUMMARY_CSV_PATH = PIPELINE_ROOT / "comparison_exports" / "trajectory_model_summary_latest.csv"
SUMMARY_JSON_PATH = PIPELINE_ROOT / "comparison_exports" / "trajectory_model_summary_latest.json"
WEIGHTS_ROOT = PIPELINE_ROOT / "model_weights"

SUPPORTED_MODELS = {
    "cnn_lstm",
    "cnn_gnn_lstm",
    "cnn_gnn_transformer",
    "cnn_gnn_lstm_transformer",
}


def _load_summary_dataframe(summary_csv_path: Path = SUMMARY_CSV_PATH) -> pd.DataFrame:
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Comparison summary not found: {summary_csv_path}")
    summary = pd.read_csv(summary_csv_path)
    if "model_slug" not in summary.columns and "model" in summary.columns:
        summary = summary.rename(columns={"model": "model_slug"})
    if "model_slug" not in summary.columns:
        raise ValueError(f"model_slug column missing in {summary_csv_path}")
    return summary


def _candidate_checkpoint(model_slug: str) -> Path:
    return WEIGHTS_ROOT / model_slug / "latest.pt"


def select_best_live_model(summary_csv_path: Path = SUMMARY_CSV_PATH) -> dict:
    summary = _load_summary_dataframe(summary_csv_path)
    summary = summary[summary["model_slug"].isin(SUPPORTED_MODELS)].copy()
    if summary.empty:
        raise RuntimeError("No supported 08 trajectory models found in the comparison summary.")

    compatible_rows = []
    for _, row in summary.iterrows():
        checkpoint_path = _candidate_checkpoint(str(row["model_slug"]))
        if checkpoint_path.exists():
            compatible_rows.append((row, checkpoint_path))

    if not compatible_rows:
        raise RuntimeError("No supported 08 trajectory checkpoints were found on disk.")

    best_row, checkpoint_path = min(
        compatible_rows,
        key=lambda item: (
            float(item[0].get("ADE", float("inf"))),
            float(item[0].get("FDE", float("inf"))),
            float(item[0].get("RMSE", float("inf"))),
        ),
    )

    result = {
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(SUMMARY_JSON_PATH),
        "selection_task": "trajectory",
        "selected_model": str(best_row["model_slug"]),
        "selected_architecture": str(best_row.get("model_slug")),
        "checkpoint_path": str(checkpoint_path),
        "selected_ADE": float(best_row.get("ADE", float("nan"))),
        "selected_FDE": float(best_row.get("FDE", float("nan"))),
        "selected_RMSE": float(best_row.get("RMSE", float("nan"))),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(select_best_live_model(), indent=2))
