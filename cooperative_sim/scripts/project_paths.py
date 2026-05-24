"""Shared absolute path configuration for local Thesis scripts."""

from pathlib import Path


DOCUMENTS_ROOT = Path("/home/basudeo/Documents")
THESIS_ROOT = DOCUMENTS_ROOT / "Thesis"

SIMULATION_WORLD_ROOT = THESIS_ROOT / "simulation"
COMMUNICATION_ROOT = THESIS_ROOT / "communication"

WORLDS_DIR = SIMULATION_WORLD_ROOT / "worlds"
MODELS_DIR = SIMULATION_WORLD_ROOT / "models"

WORLD_SDF_PATH = WORLDS_DIR / "baylands.sdf"
RVIZ_CONFIG_PATH = SIMULATION_WORLD_ROOT / "rviz_config_thesis.rviz"
OMNET_DIR = COMMUNICATION_ROOT / "basic"
