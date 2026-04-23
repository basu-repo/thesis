"""Shared data and evaluation helpers for the thesis notebooks.

This module keeps the repetitive dataset-loading, path-remapping, metric, and
result-saving utilities in one place so the notebooks can stay focused on
model-specific logic.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
    from sklearn.preprocessing import label_binarize

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


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

REDUCED_LABELS = [
    "go_to_goal",
    "avoid_left",
    "avoid_right",
    "commit_forward",
    "arrived",
]

PLATFORM_ONEHOT = {
    "UGV": [1.0, 0.0],
    "UAV": [0.0, 1.0],
}

DEFAULT_EXTERNAL_DATASET_ROOT = Path.home() / "Documents/Thesis/03_dataset/husky_control_dataset"

DATASET_ROOT: Path | None = None
ORIGINAL_DATASET_ROOT: Path | None = None
RESULTS_ROOT: Path | None = None
WEIGHTS_ROOT: Path | None = None


def configure_helper(
    *,
    dataset_root: Path,
    original_dataset_root: Path | None = None,
    results_root: Path | None = None,
    weights_root: Path | None = None,
) -> None:
    """Set notebook-specific roots once so helper functions stay simple."""
    global DATASET_ROOT, ORIGINAL_DATASET_ROOT, RESULTS_ROOT, WEIGHTS_ROOT
    DATASET_ROOT = Path(dataset_root)
    ORIGINAL_DATASET_ROOT = Path(original_dataset_root) if original_dataset_root is not None else DATASET_ROOT
    RESULTS_ROOT = Path(results_root) if results_root is not None else None
    WEIGHTS_ROOT = Path(weights_root) if weights_root is not None else None
    load_npy_cached.cache_clear()


def _require_roots() -> tuple[Path, Path]:
    if DATASET_ROOT is None or ORIGINAL_DATASET_ROOT is None:
        raise RuntimeError(
            "dataset_helper is not configured. Call configure_helper(...) in the notebook setup cell first."
        )
    return DATASET_ROOT, ORIGINAL_DATASET_ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_mapping(label_mode: str):
    if label_mode == "full":
        labels = list(DEFAULT_LABELS)
        mapping = {label: label for label in DEFAULT_LABELS}
        return labels, mapping

    labels = list(REDUCED_LABELS)
    mapping = {
        "bootstrap": None,
        "go_to_goal": "go_to_goal",
        "avoid_left": "avoid_left",
        "avoid_right": "avoid_right",
        "commit_forward": "commit_forward",
        "reverse": None,
        "recover": None,
        "reassess": None,
        "arrived": "arrived",
        "stop": None,
    }
    return labels, mapping


def _frame_files_under(root: Path) -> list[Path]:
    if (root / "frames.jsonl").exists():
        return [root / "frames.jsonl"]
    return sorted(root.glob("*/frames.jsonl"))


def discover_frame_files(dataset_root: Path):
    """Find extracted frame files, preferring the requested root but falling back safely.

    Some notebooks may still carry a stale local `DATASET_ROOT` even though the actual
    extracted data lives on the external drive. To keep the workflow resilient after
    kernel restarts or old notebook outputs, we try a few sensible roots in order.
    """
    candidate_roots: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path | None) -> None:
        if path is None:
            return
        path = Path(path)
        if path in seen:
            return
        seen.add(path)
        candidate_roots.append(path)

    add_candidate(Path(dataset_root))
    add_candidate(DATASET_ROOT)
    add_candidate(ORIGINAL_DATASET_ROOT)
    add_candidate(DEFAULT_EXTERNAL_DATASET_ROOT)

    for root in candidate_roots:
        frame_files = _frame_files_under(root)
        if frame_files:
            return frame_files
    return []


def remap_dataset_path(path_str: str) -> Path:
    dataset_root, original_dataset_root = _require_roots()
    path = Path(path_str)
    if path.exists():
        return path
    try:
        rel = path.relative_to(original_dataset_root)
    except ValueError:
        parts = path.parts
        for dataset_marker in ("hybrid_maneuvers_dataset", "husky_control_dataset"):
            if dataset_marker in parts:
                marker = parts.index(dataset_marker)
                rel = Path(*parts[marker + 1 :])
                candidate = dataset_root / rel
                if candidate.exists():
                    return candidate
                if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
                    fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
                    if fallback_candidate.exists():
                        return fallback_candidate
        return path
    candidate = dataset_root / rel
    if candidate.exists():
        return candidate
    if DEFAULT_EXTERNAL_DATASET_ROOT != dataset_root:
        fallback_candidate = DEFAULT_EXTERNAL_DATASET_ROOT / rel
        if fallback_candidate.exists():
            return fallback_candidate
    return candidate


@lru_cache(maxsize=32768)
def load_npy_cached(path: str):
    return np.load(remap_dataset_path(path))


def resample_scan(scan: np.ndarray, num_beams: int, range_clip: float):
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

    return np.stack(
        [ranges / max(range_clip, 1e-6), intensities / 255.0],
        axis=0,
    ).astype(np.float32)


def canonical_agent_order(ego_id: str):
    other_husky = "husky_2" if ego_id == "husky_local" else "husky_local"
    return [ego_id, other_husky, "uav1"]


def node_feature(node: dict, ego_state: dict):
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


def build_edge_lookup(frame: dict):
    return {(edge["source"], edge["target"]): edge for edge in frame["edges"]}


def frame_scan_ref(frame: dict):
    if "modalities" in frame:
        return frame["modalities"].get("ego_planar_scan")
    if "observation" in frame:
        return frame["observation"].get("ego_planar_scan")
    return None


def frame_state(frame: dict):
    if "agents" in frame:
        return frame["agents"][frame["ego_id"]]["state"]
    return frame.get("state")


def frame_teacher_label(frame: dict):
    teacher = frame.get("teacher", {})
    label = teacher.get("label")
    if label is not None:
        return str(label)
    controller_state = teacher.get("controller_state")
    if controller_state:
        return str(controller_state)
    return "go_to_goal"


def edge_features_for_order(frame: dict, order: list[str]):
    edge_map = build_edge_lookup(frame)
    src_edges = []
    for src in order:
        row = []
        for dst in order:
            edge = edge_map.get((src, dst))
            if edge is None:
                row.append([0.0] * 8)
            else:
                row.append(
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
        src_edges.append(row)
    return src_edges


def group_streams(dataset_root: Path, allowed_labels: set[str] | None = None, label_mapping: dict | None = None):
    streams = []
    frame_files = discover_frame_files(dataset_root)
    allowed_labels = set(allowed_labels) if allowed_labels is not None else None
    label_mapping = label_mapping or {}
    for frames_path in frame_files:
        with frames_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]

        buckets = {}
        for row in rows:
            raw_label = frame_teacher_label(row)
            mapped_label = label_mapping.get(raw_label, raw_label)
            if allowed_labels is not None and (mapped_label is None or mapped_label not in allowed_labels):
                continue
            if frame_scan_ref(row) is None:
                continue
            row = dict(row)
            row["teacher"] = dict(row["teacher"])
            row["teacher"]["raw_label"] = raw_label
            row["teacher"]["label"] = mapped_label
            key = f"{row['episode_id']}::{row['ego_id']}"
            buckets.setdefault(key, []).append(row)

        for key in sorted(buckets):
            stream = sorted(buckets[key], key=lambda item: int(item["timestamp_ns"]))
            if stream:
                streams.append(stream)

    if not streams:
        raise RuntimeError(
            f"No usable frame streams found under {dataset_root}. "
            f"Discovered {len(frame_files)} frame file(s); check label filtering and dataset paths."
        )
    return streams


def build_sample_table(streams: list[list[dict]], past_len: int, future_len: int):
    sample_table = []
    for stream_idx, stream in enumerate(streams):
        usable = len(stream) - past_len - future_len + 1
        for start in range(max(0, usable)):
            anchor = stream[start + past_len - 1]
            future_frames = stream[start + past_len : start + past_len + future_len]
            anchor_state = frame_state(anchor)
            anchor_ts = int(anchor["timestamp_ns"])

            future_xy = []
            future_dt = []
            for future_frame in future_frames:
                state = frame_state(future_frame)
                future_xy.append(
                    [
                        float(state["x"] - anchor_state["x"]),
                        float(state["y"] - anchor_state["y"]),
                    ]
                )
                future_dt.append((int(future_frame["timestamp_ns"]) - anchor_ts) * 1e-9)

            sample_table.append(
                {
                    "sample_id": f"stream{stream_idx:03d}_start{start:05d}",
                    "stream_index": stream_idx,
                    "stream_idx": stream_idx,
                    "start_index": start,
                    "start": start,
                    "anchor_index": start + past_len - 1,
                    "ego_id": anchor["ego_id"],
                    "label": anchor["teacher"].get("label"),
                    "raw_label": anchor["teacher"]["raw_label"],
                    "future_xy": future_xy,
                    "future_dt": future_dt,
                }
            )
    return sample_table


def save_or_load_fixed_split(
    sample_table,
    split_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    past_len: int,
    future_len: int,
):
    if split_path.exists():
        with split_path.open() as f:
            split_info = json.load(f)
        current_sample_ids = [row["sample_id"] for row in sample_table]
        if (
            split_info.get("sample_count") == len(sample_table)
            and split_info.get("past_len") == past_len
            and split_info.get("future_len") == future_len
            and split_info.get("sample_ids") == current_sample_ids
        ):
            return split_info

    rng = random.Random(seed)
    indices = list(range(len(sample_table)))
    rng.shuffle(indices)

    train_len = max(1, int(len(indices) * train_ratio))
    val_len = max(1, int(len(indices) * val_ratio))
    test_len = len(indices) - train_len - val_len
    if test_len < 1:
        test_len = 1
        if train_len > val_len:
            train_len -= 1
        else:
            val_len -= 1

    split_info = {
        "seed": seed,
        "sample_count": len(sample_table),
        "past_len": past_len,
        "future_len": future_len,
        "train_indices": indices[:train_len],
        "val_indices": indices[train_len : train_len + val_len],
        "test_indices": indices[train_len + val_len :],
        "sample_ids": [row["sample_id"] for row in sample_table],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_info, indent=2))
    return split_info


def build_class_weights(label_indices: list[int], num_classes: int):
    counts = Counter(label_indices)
    total = sum(counts.values())
    weights = []
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        weights.append(0.0 if count == 0 else total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def compute_trajectory_metrics(pred_future_xy: np.ndarray, true_future_xy: np.ndarray):
    diff = pred_future_xy - true_future_xy
    dist = np.linalg.norm(diff, axis=-1)
    return {
        "ADE": float(dist.mean()),
        "FDE": float(dist[:, -1].mean()),
        "RMSE": float(np.sqrt(np.mean(np.sum(diff**2, axis=-1)))),
    }


def compute_classification_metrics_from_probs(probabilities: np.ndarray, targets: np.ndarray, labels: list[str]):
    preds = probabilities.argmax(axis=1)
    num_classes = len(labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, guess in zip(targets, preds):
        confusion[int(truth), int(guess)] += 1

    precisions, recalls, f1s = [], [], []
    for idx in range(num_classes):
        tp = float(confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - tp)
        fp = float(confusion[:, idx].sum() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall / (precision + recall))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = {
        "accuracy": float((preds == targets).mean()),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion.tolist(),
        "ADE": None,
        "FDE": None,
        "RMSE": None,
    }
    return metrics, preds, confusion


def save_training_history(history: dict, out_path: Path):
    pd.DataFrame(history).to_csv(out_path, index=False)


def save_confusion_matrix(confusion: np.ndarray, labels: list[str], csv_path: Path, png_path: Path, title: str):
    df = pd.DataFrame(confusion, index=labels, columns=labels)
    df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_pr_curves(probabilities: np.ndarray, targets: np.ndarray, labels: list[str], out_dir: Path):
    summary = {"roc_auc_macro": None, "pr_auc_macro": None, "status": "skipped"}
    if not SKLEARN_AVAILABLE:
        return summary

    y_true = label_binarize(targets, classes=list(range(len(labels))))
    roc_aucs = []
    pr_aucs = []

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, idx], probabilities[:, idx])
            roc_auc_value = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_true[:, idx], probabilities[:, idx])
            pr_auc_value = average_precision_score(y_true[:, idx], probabilities[:, idx])
            roc_aucs.append(roc_auc_value)
            pr_aucs.append(pr_auc_value)
            ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_value:.3f})")
            ax_pr.plot(recall, precision, label=f"{label} (AP={pr_auc_value:.3f})")
        except Exception:
            continue

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_title("One-vs-Rest ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8)
    plt.tight_layout()
    fig_roc.savefig(out_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_roc)

    ax_pr.set_title("One-vs-Rest Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)
    plt.tight_layout()
    fig_pr.savefig(out_dir / "pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_pr)

    if roc_aucs:
        summary["roc_auc_macro"] = float(np.mean(roc_aucs))
    if pr_aucs:
        summary["pr_auc_macro"] = float(np.mean(pr_aucs))
    summary["status"] = "saved"
    return summary


def save_predictions_csv(sample_ids, targets, preds, probabilities, labels, out_path: Path):
    rows = []
    for sid, truth, pred, probs in zip(sample_ids, targets, preds, probabilities):
        row = {
            "sample_id": sid,
            "true_label": labels[int(truth)],
            "pred_label": labels[int(pred)],
        }
        for idx, label in enumerate(labels):
            row[f"prob_{label}"] = float(probs[idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_history_plot(history: dict, out_path: Path, title_prefix: str):
    if not history or len(history.get("epoch", [])) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_title(f"{title_prefix}: Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_accuracy"], label="val_accuracy", color="tab:green")
    axes[1].set_title(f"{title_prefix}: Validation Accuracy")
    axes[1].legend()

    axes[2].plot(history["epoch"], history["val_macro_f1"], label="val_macro_f1", color="tab:orange")
    axes[2].set_title(f"{title_prefix}: Validation Macro-F1")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_overlay_plots(
    pred_future_xy: np.ndarray,
    true_future_xy: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
    output_dir: Path,
    prefix: str,
    max_plots: int = 8,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    total = min(max_plots, pred_future_xy.shape[0])
    for idx in range(total):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0.0], [0.0], "ko", label="anchor")
        ax.plot(true_future_xy[idx, :, 0], true_future_xy[idx, :, 1], "-o", label="ground truth")
        ax.plot(pred_future_xy[idx, :, 0], pred_future_xy[idx, :, 1], "--o", label="prediction")
        ax.set_title(f"{prefix} sample {idx} ({labels[int(targets[idx])]})")
        ax.set_xlabel("Relative x (m)")
        ax.set_ylabel("Relative y (m)")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        path = output_dir / f"{prefix}_trajectory_overlay_{idx:02d}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def save_mean_step_error_plot(pred_future_xy: np.ndarray, true_future_xy: np.ndarray, output_path: Path, title: str):
    diff = pred_future_xy - true_future_xy
    step_error = np.linalg.norm(diff, axis=-1).mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(step_error) + 1), step_error, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Future step")
    ax.set_ylabel("Mean displacement error (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def prepare_result_dirs(model_slug: str):
    if RESULTS_ROOT is None or WEIGHTS_ROOT is None:
        raise RuntimeError(
            "dataset_helper output roots are not configured. Pass results_root and weights_root to configure_helper(...)."
        )
    result_dir = RESULTS_ROOT / model_slug
    weight_dir = WEIGHTS_ROOT / model_slug
    plot_dir = result_dir / "plots"
    for path in [result_dir, weight_dir, plot_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return result_dir, weight_dir, plot_dir


def timestamp_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_manifest(model_slug: str, timestamp: str, labels: list[str], split_path: Path, extra: dict | None = None):
    manifest = {
        "model_slug": model_slug,
        "timestamp": timestamp,
        "labels": labels,
        "split_path": str(split_path),
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_run_manifest(result_dir: Path, manifest: dict, timestamp: str):
    latest_path = result_dir / "latest_run_manifest.json"
    dated_path = result_dir / f"{timestamp}_run_manifest.json"
    latest_path.write_text(json.dumps(manifest, indent=2))
    dated_path.write_text(json.dumps(manifest, indent=2))
