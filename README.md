# Thesis Repository Guide

This repository is organized as the active thesis pipeline for:
- simulation setup
- cooperative rule-based data collection
- dataset export and preprocessing
- model training and comparison
- live learned-model evaluation
- communication-aware experiments
- dashboard-assisted run control

## Current Folder Structure

- `simulation`
  - Gazebo world files, robot models, RViz config, and Baylands assets
- `dataset`
  - rosbag recordings, run logs, exported summaries, and dataset helpers
- `communication`
  - communication-aware relay configuration and OMNeT-style support files
- `cooperative_sim`
  - active cooperative rule-based UAV--UGV simulation stack
- `model_training`
  - episode-frame export, sample-table generation, notebooks, model weights, and comparison outputs
- `model_eval`
  - live learned-model controller evaluation in Gazebo
- `simulation_dashboard`
  - local web dashboard for launching and monitoring runs

Local-only folders such as `thesis_template_hh`, `thesis_refrences`, and `.codex`
may exist in the working directory, but they are intentionally ignored and are
not part of the tracked Git repo.

## Main Pipeline

1. Run the cooperative rule-based simulator in `cooperative_sim`.
2. Record bags and logs into `dataset`.
3. Export bags to episode frames with the `model_training` scripts.
4. Build the shared sample table and train or compare models in the `model_training` notebooks.
5. Evaluate the selected model live in `model_eval`.
6. Optionally run communication-aware live evaluation with the local relay profiles in `communication`.

## Important Path Files

- `cooperative_sim/scripts/project_paths.py`
- `model_eval/project_paths.py`
- `model_training/datasets/paths.py`

These control the absolute repository-root paths used by the active scripts.

## Important Runtime Entry Points

### Rule-based data generation

- `cooperative_sim/scripts/run_sim_model.py`

### Episode-frame export and dataset preparation

- `model_training/scripts/export_one_bag_to_episode_frames.py`
- `model_training/scripts/export_bags_from_list.py`
- `model_training/scripts/build_sample_table_and_split.py`

### Training and comparison

- `model_training/notebooks/00_data_loading_and_split.ipynb`
- `model_training/notebooks/20_cnn_lstm.ipynb`
- `model_training/notebooks/40_cnn_gnn_lstm.ipynb`
- `model_training/notebooks/50_cnn_gnn_transformer.ipynb`
- `model_training/notebooks/60_cnn_gnn_lstm_transformer.ipynb`
- `model_training/notebooks/90_model_comparison.ipynb`
- `model_training/notebooks/91_model_comparison_no_cv.ipynb`

### Live learned-model evaluation

- `model_eval/trajectory_model_eval_sim.py`
- `model_eval/controllers/husky_trajectory_model_driver.py`

### Dashboard

- `simulation_dashboard/start_dashboard.sh`

## Communication Profiles

The active local relay profiles are:
- `WifiRelay`
- `BluetoothRelay`

These are defined under:
- `communication/basic/onmetpp/omnetpp.ini`

For local communication-aware `model_eval` runs, the UGV receives delayed and degraded UAV context through the relay profile while keeping its own local sensing direct.

## Quick Checks After Moving The Repository

1. Verify the repository root is still `/home/basudeo/Documents/Thesis`, or update the path helper files above.
2. Confirm `simulation/worlds/baylands.sdf` and `simulation/models/` are present.
3. Confirm `dataset/bags/` and `dataset/logs/` are writable.
4. Confirm the Python environment used for `model_training` includes `numpy`, `pandas`, `torch`, `matplotlib`, `scikit-learn`, `jupyter`, and `rosbags`.
5. Source ROS 2 before launching `cooperative_sim` or `model_eval`:

```bash
source /opt/ros/humble/setup.bash
```

## Notes

- `cooperative_sim` is the active rule-based stack.
- `model_training` is the active training and comparison stack.
- `model_eval` is the active learned-controller evaluation stack.
- Thesis-writing assets and reference PDFs are intentionally kept out of Git.
- Older superseded folders have already been removed to keep this repository focused on the final thesis workflow.
