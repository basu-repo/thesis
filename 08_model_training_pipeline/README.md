# 08 Model Training Pipeline

This module is the new, standalone deep-learning workflow for the thesis.

It does not depend on `04_model_evaluation` or `05_final_eval_sim` for its
runtime behavior. Those directories may still be used as references, but all
new data preparation, model definition, training, evaluation, and result saving
should live here.

## Goal

Train models that predict a short-horizon ego trajectory for the UGV so it can
reach its goal.

The core supervised target is:

- future ego trajectory in the ego-local frame
- shape: `future_len x 2`
- values: `(dx, dy)` waypoint offsets

## Planned flow

1. `bags -> episode frames`
2. `episode frames -> train-ready samples`
3. `samples + split files + normalization stats`
4. `model training`
5. `saved checkpoints + predictions + metrics + plots + summaries`

## Directory layout

- `configs/`
  - dataset, model, and training configs
- `exporters/`
  - raw bag to standardized episode-frame export
- `datasets/`
  - canonical schemas, sample builders, split loading, normalization
- `models/`
  - all model definitions and a registry
- `training/`
  - train/eval loops, checkpoints, run manifests
- `scripts/`
  - entrypoints for export, build-samples, train, evaluate, compare
- `results/`
  - per-run metrics, plots, predictions
- `model_weights/`
  - saved checkpoints
- `comparison_exports/`
  - cross-run CSV/JSON summaries

## Canonical sample contents

Every model should read from one shared sample format, even if some fields are
unused by a simpler baseline.

Required sample groups:

- `ego_past`
- `goal_features`
- `target_future_xy`

Optional sample groups:

- `scan_features`
- `graph_features`
- `labels`
- `metadata`

