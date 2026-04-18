# Thesis_Repo

This workspace is the thesis-facing evaluation area for the project.

It is designed to answer one question clearly:

- given the same extracted dataset
- given the same train / validation / test split
- how do different model families compare?

The notebooks in this folder are intentionally standalone and readable.
They do not import from `scripts/train_hybrid_cnn_gnn_lstm.py`.
That makes them easier to explain, defend, and extend during thesis work.

## What This Workspace Is For

Use `Thesis_Repo` for:

- fair model evaluation
- repeatable train / validation / test splitting
- saved model weights
- saved metrics and confusion matrices
- saved prediction tables
- thesis-ready comparison plots

Do **not** treat this folder as the main simulation launcher area.
Your live simulation code still remains in the project root under:

- `scripts/`
- `models/`
- `worlds/`

`Thesis_Repo` is the clean evaluation layer built on top of the extracted datasets.

## Folder Structure

### `notebooks/`

This contains the standalone evaluation notebooks.

Files:

- `00_data_loading_and_split.ipynb`
  - inspect extracted datasets
  - inspect label distribution
  - create or load the fixed split used by all models

- `10_cv_baseline.ipynb`
  - constant-velocity style heuristic baseline
  - no learning
  - useful as the simplest reference point

- `20_cnn_lstm.ipynb`
  - lidar-only sequence model
  - uses ego planar scan over time

- `30_gnn_lstm.ipynb`
  - graph-only sequence model
  - uses multi-agent graph over time

- `40_cnn_gnn_lstm.ipynb`
  - main hybrid baseline
  - uses lidar CNN + graph GNN + temporal LSTM

- `50_cnn_gnn_transformer.ipynb`
  - hybrid model with transformer temporal modeling
  - no LSTM in the temporal stage

- `60_cnn_gnn_lstm_transformer.ipynb`
  - hybrid model with both LSTM and transformer
  - most complex classification model in the current evaluation suite

- `90_model_comparison.ipynb`
  - reads saved outputs from all model folders
  - builds side-by-side comparison tables and plots

### `model_weights/`

This contains saved model checkpoints from trainable notebooks.

Structure:

- `model_weights/<model_name>/latest.pt`
- `model_weights/<model_name>/<model_name>_<timestamp>.pt`

Examples:

- `model_weights/cnn_lstm/latest.pt`
- `model_weights/cnn_gnn_lstm/latest.pt`

Purpose:

- use the latest best model for later simulation / trajectory work
- preserve timestamped versions for reproducibility

### `results/`

This contains saved outputs for each model run.

Structure:

- `results/<model_name>/latest_metrics.json`
- `results/<model_name>/metrics_<timestamp>.json`
- `results/<model_name>/predictions_<timestamp>.csv`
- `results/<model_name>/confusion_<timestamp>.csv`
- `results/<model_name>/history_<timestamp>.csv`
- `results/<model_name>/plots/...`

What gets stored here:

- test metrics
- confusion matrix values
- per-sample predictions
- training history
- confusion matrix images
- ROC / PR plots when available

### `splits/`

This contains the fixed train / validation / test split definition.

Structure:

- `splits/classification_split_<label_mode>_seed<seed>_past<past_len>.json`

Purpose:

- every model notebook reads the same split
- comparisons stay fair
- no accidental resplitting between models

### `plots/`

This folder is used for higher-level exported plots.

Typical content:

- comparison bar charts
- macro-F1 comparison plots
- thesis-ready figure exports

### `comparison_exports/`

This contains summary tables saved by the comparison notebook.

Typical content:

- `model_summary_latest.csv`
- `model_summary_latest.json`

This is useful when you want:

- one compact thesis table
- one CSV to load into another tool

### `docs/`

Reserved for thesis notes, experiment writeups, or workflow notes.

## Dataset Assumption

All notebooks assume the extracted hybrid maneuver dataset already exists under:

- `../hybrid_maneuver_dataset/`

That means the workflow before using `Thesis_Repo` is:

1. record a bag from simulation
2. export the bag into extracted dataset form
3. then run the notebooks in `Thesis_Repo`

The notebooks do **not** read directly from rosbag files.
They read from the extracted JSONL + `.npy` dataset.

## Recommended Run Order

Run the notebooks in this order.

### Step 1: Build the Shared Split

Run:

- `notebooks/00_data_loading_and_split.ipynb`

This should always be the first notebook.

What it does:

- finds all extracted dataset folders
- loads the frame streams
- applies the label mapping
- shows raw and mapped label counts
- creates or loads the canonical train / validation / test split

Output:

- a split JSON file under `splits/`

### Step 2: Run the Baseline

Run:

- `notebooks/10_cv_baseline.ipynb`

This gives you the simplest comparison point before any learning.

Output:

- metrics under `results/cv_baseline/`
- confusion matrix
- prediction CSV
- ROC / PR plots if available

### Step 3: Run the Learned Models

Run these one by one:

- `notebooks/20_cnn_lstm.ipynb`
- `notebooks/30_gnn_lstm.ipynb`
- `notebooks/40_cnn_gnn_lstm.ipynb`
- `notebooks/50_cnn_gnn_transformer.ipynb`
- `notebooks/60_cnn_gnn_lstm_transformer.ipynb`

Each of these notebooks:

- uses the same split file
- trains on the same data partition
- saves a best checkpoint
- saves final metrics
- saves predictions
- saves confusion matrix
- saves training curves

Output:

- weights under `model_weights/<model_name>/`
- metrics and plots under `results/<model_name>/`

### Step 4: Compare Models

Run:

- `notebooks/90_model_comparison.ipynb`

This notebook collects all saved `latest_metrics.json` files and builds:

- summary tables
- comparison bar charts
- macro-F1 vs accuracy plots
- confusion matrix review

Output:

- summary CSV / JSON under `comparison_exports/`
- comparison plots under `plots/`

## What Gets Saved Per Model

For every trainable model notebook, the intended saved outputs are:

- best checkpoint weights
- latest checkpoint alias
- latest metrics JSON
- timestamped metrics JSON
- timestamped prediction CSV
- timestamped confusion CSV
- timestamped confusion PNG
- timestamped training history CSV
- timestamped training history PNG
- ROC / PR plots when sklearn is available

This is important because the thesis workflow should not depend on notebook memory only.
You should be able to:

- rerun comparison later
- reuse model weights in simulation
- cite exact saved results

## About Metrics

### Classification Metrics Currently Active

The current model suite is focused on maneuver classification.

So the main active metrics are:

- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix
- ROC curves
- precision-recall curves

### ADE / FDE / RMSE

These metrics are trajectory metrics.

They are included in the saved metric structure as placeholders, but for the current
classification-only notebooks they will be:

- `ADE = null`
- `FDE = null`
- `RMSE = null`

They will become meaningful once we add a trajectory prediction head that outputs:

- future point
or
- future trajectory

Then we can compare:

- actual future positions from the dataset
- predicted future positions from the model

## Model Weights and Later Simulation Use

The reason model weights are saved here is that notebook-only evaluation is not enough.

Once we identify the best hybrid model, its saved weight file can be used for:

- later simulation integration
- trajectory prediction experiments
- live testing in the UGV/UAV environment

So the intended path is:

1. evaluate many models here
2. identify the best one
3. use the saved weight from `model_weights/`
4. integrate that model into testing / simulation code later

## Reproducibility Notes

To keep comparisons fair:

- keep `LABEL_MODE` the same across models
- keep `SEED` the same across models
- keep `PAST_LEN` the same across models unless intentionally doing an ablation
- always reuse the saved split file

If you intentionally change one of those:

- create a new split file
- note the change clearly in your thesis notes

## Practical Workflow Summary

If you want the shortest operational version:

1. Export data into `hybrid_maneuver_dataset/`
2. Run `00_data_loading_and_split.ipynb`
3. Run `10_cv_baseline.ipynb`
4. Run each learned model notebook
5. Run `90_model_comparison.ipynb`
6. Use `model_weights/<best_model>/latest.pt` for later deployment/testing

## Current Scope

Right now this workspace is built for:

- readable evaluation
- fair classification comparison
- saved artifacts for the thesis

The next extension, when needed, is:

- trajectory prediction outputs
- meaningful `ADE`, `FDE`, and `RMSE`
- deployment of the best saved weight back into simulation
