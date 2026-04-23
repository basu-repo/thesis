"""Choose the best saved model checkpoint for live final evaluation.

The live runner supports a small set of architectures that can be driven with
the current simulation inputs. This selector ranks saved models by macro F1,
then accuracy, and returns the best compatible checkpoint that actually exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


THESIS_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = THESIS_ROOT / "04_model_evaluation" / "comparison_exports" / "model_summary_latest.json"
WEIGHTS_ROOT = THESIS_ROOT / "04_model_evaluation" / "model_weights"
CONTROLLERS_ROOT = Path(__file__).resolve().parent / "controllers"
if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_ROOT))

from ai_model_predictor import (
    ARCH_GRAPH_ONLY_LSTM,
    ARCH_SCAN_GRAPH_LSTM,
    ARCH_SCAN_GRAPH_LSTM_TRANSFORMER,
    ARCH_SCAN_GRAPH_TRANSFORMER,
    infer_runtime_architecture,
)


SUPPORTED_RUNTIME_ARCHITECTURES = {
    ARCH_GRAPH_ONLY_LSTM,
    ARCH_SCAN_GRAPH_LSTM,
    ARCH_SCAN_GRAPH_TRANSFORMER,
    ARCH_SCAN_GRAPH_LSTM_TRANSFORMER,
}


def _load_summary(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Comparison summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected summary format in {summary_path}")
    return rows


def _candidate_checkpoint(model_slug: str) -> Path:
    return WEIGHTS_ROOT / model_slug / "latest.pt"


def _infer_checkpoint_architecture(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return infer_runtime_architecture(checkpoint)


def _is_trajectory_row(row: dict) -> bool:
    return any(row.get(metric) is not None for metric in ("ADE", "FDE", "RMSE", "best_val_ADE"))


def _trajectory_rank_key(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row.get("ADE") if row.get("ADE") is not None else float("inf")),
        float(row.get("FDE") if row.get("FDE") is not None else float("inf")),
        float(row.get("RMSE") if row.get("RMSE") is not None else float("inf")),
        float(row.get("best_val_ADE") if row.get("best_val_ADE") is not None else float("inf")),
    )


def _classification_rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("macro_f1") or float("-inf")),
        float(row.get("accuracy") or float("-inf")),
        float(row.get("pr_auc_macro") or float("-inf")),
    )


def select_best_live_model(summary_path: Path = SUMMARY_PATH) -> dict:
    rows = _load_summary(summary_path)
    saved_rows = [row for row in rows if row.get("status") == "saved" and row.get("model")]
    if not saved_rows:
        raise RuntimeError(f"No saved models found in {summary_path}")

    trajectory_rows = [row for row in saved_rows if _is_trajectory_row(row)]
    if trajectory_rows:
        task_rows = trajectory_rows
        overall_best = min(task_rows, key=_trajectory_rank_key)
        selection_task = "trajectory"
    else:
        task_rows = saved_rows
        overall_best = max(task_rows, key=_classification_rank_key)
        selection_task = "classification"

    compatible_rows = []
    architecture_by_model: dict[str, str] = {}
    for row in task_rows:
        model_slug = str(row.get("model"))
        checkpoint_path = _candidate_checkpoint(model_slug)
        if not checkpoint_path.exists():
            continue
        architecture = _infer_checkpoint_architecture(checkpoint_path)
        if architecture not in SUPPORTED_RUNTIME_ARCHITECTURES:
            continue
        architecture_by_model[model_slug] = architecture
        compatible_rows.append(row)

    if not compatible_rows:
        raise RuntimeError(
            "No live-compatible models were found. "
            f"Supported architectures: {sorted(SUPPORTED_RUNTIME_ARCHITECTURES)}"
        )

    if selection_task == "trajectory":
        compatible_best = min(compatible_rows, key=_trajectory_rank_key)
    else:
        compatible_best = max(compatible_rows, key=_classification_rank_key)

    checkpoint_path = _candidate_checkpoint(str(compatible_best["model"]))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint_path}")

    result = {
        "summary_path": str(summary_path),
        "selection_task": selection_task,
        "overall_best_model": str(overall_best["model"]),
        "selected_model": str(compatible_best["model"]),
        "selected_architecture": architecture_by_model[str(compatible_best["model"])],
        "checkpoint_path": str(checkpoint_path),
        "used_fallback": str(overall_best["model"]) != str(compatible_best["model"]),
    }
    if selection_task == "trajectory":
        result.update(
            {
                "overall_best_ADE": float(overall_best.get("ADE") or float("nan")),
                "overall_best_FDE": float(overall_best.get("FDE") or float("nan")),
                "selected_ADE": float(compatible_best.get("ADE") or float("nan")),
                "selected_FDE": float(compatible_best.get("FDE") or float("nan")),
                "selected_RMSE": float(compatible_best.get("RMSE") or float("nan")),
            }
        )
    else:
        result.update(
            {
                "overall_best_macro_f1": float(overall_best.get("macro_f1") or float("nan")),
                "selected_macro_f1": float(compatible_best.get("macro_f1") or float("nan")),
                "selected_accuracy": float(compatible_best.get("accuracy") or float("nan")),
            }
        )
    return result


if __name__ == "__main__":
    choice = select_best_live_model()
    print(json.dumps(choice, indent=2))
