"""Shared filesystem locations for the new training pipeline."""

from __future__ import annotations

from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = PIPELINE_ROOT.parent

RAW_BAGS_ROOT = THESIS_ROOT / "dataset" / "bags"
RAW_EXPORT_ROOT = THESIS_ROOT / "dataset" / "husky_control_dataset"

CONFIGS_ROOT = PIPELINE_ROOT / "configs"
EXPORTERS_ROOT = PIPELINE_ROOT / "exporters"
DATASETS_ROOT = PIPELINE_ROOT / "datasets"
MODELS_ROOT = PIPELINE_ROOT / "models"
TRAINING_ROOT = PIPELINE_ROOT / "training"
SCRIPTS_ROOT = PIPELINE_ROOT / "scripts"

RESULTS_ROOT = PIPELINE_ROOT / "results"
MODEL_WEIGHTS_ROOT = PIPELINE_ROOT / "model_weights"
COMPARISON_EXPORTS_ROOT = PIPELINE_ROOT / "comparison_exports"

EPISODE_FRAMES_ROOT = PIPELINE_ROOT / "results" / "episode_frames"
TRAIN_READY_ROOT = PIPELINE_ROOT / "results" / "train_ready"
SPLITS_ROOT = PIPELINE_ROOT / "results" / "splits"
NORMALIZATION_ROOT = PIPELINE_ROOT / "results" / "normalization"
