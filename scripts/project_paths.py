"""Shared absolute path configuration for local Thesis scripts."""

from pathlib import Path


DOCUMENTS_ROOT = Path("/home/basudeo/Documents")
THESIS_ROOT = DOCUMENTS_ROOT / "Thesis"

WORLDS_DIR = THESIS_ROOT / "worlds"
MODELS_DIR = THESIS_ROOT / "models"
SCRIPTS_DIR = THESIS_ROOT / "scripts"

WORLD_SDF_PATH = WORLDS_DIR / "sim_world.sdf"
RVIZ_CONFIG_PATH = THESIS_ROOT / "rviz_config_thesis.rviz"
OMNET_DIR = THESIS_ROOT / "onmetpp"

