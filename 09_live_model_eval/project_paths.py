"""Shared absolute path configuration for 09 live model evaluation."""

from pathlib import Path


DOCUMENTS_ROOT = Path("/home/basudeo/Documents")
THESIS_ROOT = DOCUMENTS_ROOT / "Thesis"

SIMULATION_WORLD_ROOT = THESIS_ROOT / "01_simulation_world"
COMMUNICATION_ROOT = THESIS_ROOT / "06_Communication"

WORLDS_DIR = SIMULATION_WORLD_ROOT / "worlds"
MODELS_DIR = SIMULATION_WORLD_ROOT / "models"

WORLD_SDF_PATH = WORLDS_DIR / "baylands.sdf"
RVIZ_CONFIG_PATH = SIMULATION_WORLD_ROOT / "rviz_config_thesis.rviz"
GUI_CONFIG_PATH = SIMULATION_WORLD_ROOT / "gui" / "baylands_gui.config"
OMNET_DIR = COMMUNICATION_ROOT / "basic"
OMNET_EXTERNAL_DIR = COMMUNICATION_ROOT / "external_team" / "UAV_UGV_main_external"
