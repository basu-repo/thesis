# Thesis Repository Guide

This repository is organized as a clean pipeline for:
- simulation world setup
- rule-based data collection
- dataset export
- model evaluation
- final learned-model simulation
- optional communication-aware experiments

The structure is intentionally separated so that simulation, training, and communication work do not get mixed together.

## Folder Structure

- `01_simulation_world`
  - Gazebo world files, robot models, and RViz config
- `02_rule_based`
  - active rule-based runner and controllers for Husky and UAV
- `03_dataset`
  - rosbag recordings, dataset exporters, extracted dataset storage
- `04_model_evaluation`
  - notebooks, splits, results, plots, comparison exports, model weights
- `05_final_eval_sim`
  - final learned-model simulation code
- `06_Communication`
  - optional communication modules
  - `basic/` for your modular communication layer
  - `external_team/` for external OMNeT++ work kept separate

## Step-By-Step Installation

This section explains the minimum setup needed to run the full pipeline.

### 1. Operating System

Recommended:
- Ubuntu Linux

This repository is currently organized for a Linux environment and uses ROS 2, Gazebo, Python, and local file paths in Linux style.

### 2. Install ROS 2

Minimum:
- ROS 2 Humble

You need ROS 2 to run:
- the rule-based simulation
- ROS nodes for Husky and UAV
- rosbag recording
- controller communication

After installation, make sure you can source ROS:

```bash
source /opt/ros/humble/setup.bash
```

### 3. Install Gazebo and ROS-Gazebo bridge support

You need Gazebo plus the ROS-Gazebo bridge for:
- loading the world
- spawning robots
- reading sensors
- publishing simulation topics into ROS

Minimum:
- Gazebo
- `ros_gz`
- RViz

### 4. Install Python 3.10 and create an environment

Recommended:
- Python 3.10
- one dedicated environment for dataset export and model evaluation

Example:

```bash
python3 -m venv thesis_env
source thesis_env/bin/activate
```

If you use Conda instead, that is also fine.

### 5. Install required Python packages

Minimum packages for this repository:

```bash
pip install numpy pandas matplotlib torch scikit-learn jupyter rosbags
```

What these are used for:
- `numpy`: arrays, saved `.npy` assets
- `pandas`: comparison tables and result summaries
- `matplotlib`: plots
- `torch`: model training and evaluation
- `scikit-learn`: ROC, PR, AUC, and classification utilities
- `jupyter`: notebooks
- `rosbags`: rosbag export to extracted dataset

### 6. Optional communication installation

Only needed if you want communication-aware experiments.

Minimum:
- OMNeT++

If you are only doing:
- rule-based simulation
- dataset export
- notebook training/evaluation

then OMNeT++ is not required at first.

## What To Check First After Cloning Or Moving The Repository

Before running anything, verify these items in order.

### 1. Check the repository root location

The project currently assumes the repository root is:

```text
/home/basudeo/Documents/Thesis
```

If your repository is somewhere else, the first file to update is:

- `02_rule_based/scripts/project_paths.py`

Main variables there:
- `DOCUMENTS_ROOT`
- `THESIS_ROOT`

These control:
- world path
- models path
- RViz config path
- communication base path

### 2. Check the extracted dataset location

The notebooks and exporter are currently set up to prefer the external extracted dataset path:

```text
/media/basudeo/1044063744061FD8/hybrid_maneuver_dataset
```

This is the preferred path used by the evaluation notebooks.

The local fallback path is:

```text
03_dataset/hybrid_maneuver_dataset
```

### 3. If the external drive path changes

You should update:

- `03_dataset/exporters/export_hybrid_maneuver_dataset.py`
  - variable:
    - `PREFERRED_EXTERNAL_OUT_ROOT`

- all notebooks in:
  - `04_model_evaluation/notebooks/`
  - variables in the setup cell:
    - `PREFERRED_DATASET_ROOT`
    - `ORIGINAL_DATASET_ROOT`

### 4. Check final model path before deployment

When using the selected best model in final simulation, confirm the final controller points to the intended weight inside:

- `04_model_evaluation/model_weights/`

## Important Files To Edit

### A. World and simulation assets

Folder:
- `01_simulation_world`

Important items:
- `01_simulation_world/worlds/sim_world.sdf`
- `01_simulation_world/models/`
- `01_simulation_world/rviz_config_thesis.rviz`

Edit these when you want to change:
- the world
- model assets
- visualization setup

### B. Rule-based simulation

Folder:
- `02_rule_based`

Main files:
- `02_rule_based/scripts/run_sim_model.py`
- `02_rule_based/scripts/project_paths.py`
- `02_rule_based/controllers/husky_model_driver.py`
- `02_rule_based/controllers/uav_follower.py`
- `02_rule_based/controllers/obstacle_detection.py`
- `02_rule_based/controllers/episode_metadata.py`

Edit these when you want to change:
- spawn locations
- goal positions
- rule-based Husky behavior
- rule-based UAV following behavior
- obstacle handling
- bag-recording behavior

### C. Dataset export

Folder:
- `03_dataset`

Main files:
- `03_dataset/exporters/export_hybrid_maneuver_dataset.py`
- `03_dataset/bags/`

Edit this exporter when you want to change:
- bag source location
- extracted dataset output location
- what sensors or labels are exported

### D. Model evaluation

Folder:
- `04_model_evaluation`

Main notebook files:
- `04_model_evaluation/notebooks/00_data_loading_and_split.ipynb`
- `04_model_evaluation/notebooks/10_cv_baseline.ipynb`
- `04_model_evaluation/notebooks/20_cnn_lstm.ipynb`
- `04_model_evaluation/notebooks/30_gnn_lstm.ipynb`
- `04_model_evaluation/notebooks/40_cnn_gnn_lstm.ipynb`
- `04_model_evaluation/notebooks/50_cnn_gnn_transformer.ipynb`
- `04_model_evaluation/notebooks/60_cnn_gnn_lstm_transformer.ipynb`
- `04_model_evaluation/notebooks/90_model_comparison.ipynb`
- `04_model_evaluation/notebooks/dataset_helper.py`

This stage is used for:
- loading the extracted dataset
- training models
- evaluating classification and trajectory metrics
- saving weights
- saving plots
- generating comparison summaries

### E. Final evaluation simulation

Folder:
- `05_final_eval_sim`

Current main file:
- `05_final_eval_sim/controllers/husky_gnn_model_driver.py`

This stage is used after a best model has been selected.

### F. Communication

Folder:
- `06_Communication`

Main files:
- `06_Communication/basic/onmetpp/omnetpp.ini`
- `06_Communication/basic/ros_bridges/omnet_hazard_bridge.py`
- `06_Communication/basic/ros_bridges/uav_hazard_estimator.py`

Purpose:
- optional communication-aware testing
- modular integration without mixing external work into the main learning pipeline

## Pipeline Explanation

This repository is intended to be used in the following order.

### Stage 1. Prepare the simulation world

Folder:
- `01_simulation_world`

What happens here:
- Gazebo world and model assets are prepared
- the simulation scene is defined
- Husky and UAV model resources are available

### Stage 2. Run the rule-based pipeline

Folder:
- `02_rule_based`

What happens here:
- Husky and UAV run using rule-based controllers
- rosbag data is recorded
- maneuver-rich behavior is generated for later training

This is the stage you use when you want more data.

### Stage 3. Export the dataset

Folder:
- `03_dataset`

What happens here:
- recorded bags are converted into extracted dataset folders
- `frames.jsonl` and `.npy` assets are generated
- the hybrid maneuver dataset becomes ready for notebooks

Preferred extracted output:
- `/media/basudeo/1044063744061FD8/hybrid_maneuver_dataset`

Fallback extracted output:
- `03_dataset/hybrid_maneuver_dataset`

### Stage 4. Evaluate and compare models

Folder:
- `04_model_evaluation`

What happens here:
- the extracted dataset is loaded
- the shared split is built or reused
- model notebooks are run one at a time
- results are saved
- weights are saved
- plots and comparison tables are generated

Saved outputs appear under:
- `04_model_evaluation/results`
- `04_model_evaluation/model_weights`
- `04_model_evaluation/plots`
- `04_model_evaluation/comparison_exports`
- `04_model_evaluation/splits`

### Stage 5. Select the best model and run final learned simulation

Folder:
- `05_final_eval_sim`

What happens here:
- one best model is selected from `04_model_evaluation/model_weights`
- that model is used back inside the simulation loop
- final learned behavior can be tested

### Stage 6. Add communication if needed

Folder:
- `06_Communication`

What happens here:
- communication-aware experiments can be enabled
- OMNeT++ can be used as an optional layer
- external team work remains isolated under `external_team`

This stage is optional and should not be required to run the base pipeline.

## Practical Run Order

If you are starting fresh, use this order:

1. Verify ROS 2, Gazebo, and Python environment
2. Check `02_rule_based/scripts/project_paths.py`
3. Check external dataset path availability
4. Run rule-based simulation
5. Record bag data
6. Export the dataset
7. Open `04_model_evaluation/notebooks`
8. Restart kernel before each notebook
9. Run notebooks one by one
10. Compare models
11. Select the best model
12. Use the selected model in `05_final_eval_sim`
13. Add communication support only if required

## Notes

- The notebooks are already configured to prefer the external extracted dataset path.
- The local extracted dataset folder is kept as fallback only.
- Some historical notebook outputs may still show older absolute paths until notebooks are rerun.
- `__pycache__` folders can be removed safely.
- `06_Communication/external_team` should be treated as external work, not your core thesis implementation.
