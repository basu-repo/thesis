"""Shared absolute path configuration for local Thesis scripts."""

from pathlib import Path


DOCUMENTS_ROOT = Path("/home/basudeo/Documents")
THESIS_ROOT = DOCUMENTS_ROOT / "Thesis"

SIMULATION_WORLD_ROOT = THESIS_ROOT / "01_simulation_world"
RULE_BASED_ROOT = THESIS_ROOT / "02_rule_based"
COMMUNICATION_ROOT = THESIS_ROOT / "06_Communication"

WORLDS_DIR = SIMULATION_WORLD_ROOT / "worlds"
MODELS_DIR = SIMULATION_WORLD_ROOT / "models"
SCRIPTS_DIR = RULE_BASED_ROOT / "scripts"

WORLD_SDF_PATH = WORLDS_DIR / "sim_world.sdf"
RVIZ_CONFIG_PATH = SIMULATION_WORLD_ROOT / "rviz_config_thesis.rviz"
OMNET_DIR = COMMUNICATION_ROOT / "basic"
