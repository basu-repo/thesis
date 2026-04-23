import json
from pathlib import Path


NOTEBOOK = Path("/home/basudeo/Documents/Thesis/04_model_evaluation/notebooks/10_cv_baseline.ipynb")


CELL5 = """# Shared data and evaluation utilities live in one helper module so that
# all notebooks reuse the same dataset, split, path-remapping, and trajectory metrics.
from dataset_helper import (
    build_label_mapping,
    build_run_manifest,
    build_sample_table,
    compute_trajectory_metrics,
    configure_helper,
    group_streams,
    prepare_result_dirs,
    save_mean_step_error_plot,
    save_or_load_fixed_split,
    save_run_manifest,
    save_trajectory_overlay_plots,
    set_seed,
    timestamp_tag,
)

configure_helper(
    dataset_root=DATASET_ROOT,
    original_dataset_root=ORIGINAL_DATASET_ROOT,
    results_root=RESULTS_ROOT,
    weights_root=WEIGHTS_ROOT,
)
"""


CELL6 = """# Resolve the project structure relative to this notebook.
PROJECT_ROOT = Path.cwd().resolve().parent
EVAL_ROOT = PROJECT_ROOT
RESULTS_ROOT = EVAL_ROOT / 'results'
WEIGHTS_ROOT = EVAL_ROOT / 'model_weights'
SPLITS_ROOT = EVAL_ROOT / 'splits'

ORIGINAL_DATASET_ROOT = PROJECT_ROOT.parent / '03_dataset' / 'husky_control_dataset'
PREFERRED_DATASET_ROOT = ORIGINAL_DATASET_ROOT
DATASET_ROOT = PREFERRED_DATASET_ROOT

print('Original dataset root:', ORIGINAL_DATASET_ROOT)
print('Preferred dataset root:', PREFERRED_DATASET_ROOT)
print('Dataset root:', DATASET_ROOT)
"""


CELL7 = """set_seed(SEED)
labels, label_mapping = build_label_mapping(LABEL_MODE)
streams = group_streams(DATASET_ROOT, allowed_labels=None, label_mapping=None)
sample_table = build_sample_table(streams, past_len=PAST_LEN, future_len=FUTURE_LEN)
split_path = SPLITS_ROOT / f'husky_control_split_{LABEL_MODE}_seed{SEED}_past{PAST_LEN}_future{FUTURE_LEN}.json'
split_info = save_or_load_fixed_split(sample_table, split_path, SEED, TRAIN_RATIO, VAL_RATIO, PAST_LEN, FUTURE_LEN)
assert split_info['sample_count'] == len(sample_table), (
    f"Split file {split_path} does not match the current dataset: "
    f"split has {split_info['sample_count']} samples, current table has {len(sample_table)}"
)

print('Controller-state metadata labels retained for plot titles only:', labels)
print('Split path:', split_path)
print('Total samples in canonical table:', len(sample_table))
print('Train / Val / Test:', len(split_info['train_indices']), len(split_info['val_indices']), len(split_info['test_indices']))
print('Future horizon:', split_info['future_len'])
"""


CELL8 = """MODEL_SLUG = 'cv_baseline'
TIMESTAMP = timestamp_tag()
result_dir, weight_dir, plot_dir = prepare_result_dirs(MODEL_SLUG)

# Constant-velocity baseline:
# use the final observed planar velocity and extrapolate the future trajectory
# in the world frame over the saved future timestamps.
test_indices = split_info['test_indices']
targets = np.asarray([labels.index(row['raw_label']) if row['raw_label'] in labels else 0 for row in (sample_table[idx] for idx in test_indices)], dtype=np.int64)

all_pred_future_xy = []
all_true_future_xy = []
for sample_idx in test_indices:
    meta = sample_table[sample_idx]
    stream = streams[meta['stream_idx']]
    anchor = stream[meta['anchor_index']]
    state = anchor['state']
    vx = float(state.get('vx', 0.0))
    vy = float(state.get('vy', 0.0))
    future_dt = np.asarray(meta['future_dt'], dtype=np.float32)
    pred_future_xy = np.stack([vx * future_dt, vy * future_dt], axis=-1)
    all_pred_future_xy.append(pred_future_xy)
    all_true_future_xy.append(np.asarray(meta['future_xy'], dtype=np.float32))

pred_future_xy = np.stack(all_pred_future_xy, axis=0)
true_future_xy = np.stack(all_true_future_xy, axis=0)
metrics = compute_trajectory_metrics(pred_future_xy, true_future_xy)
metrics['split_path'] = str(split_path)

run_manifest = build_run_manifest(
    model_slug=MODEL_SLUG,
    timestamp=TIMESTAMP,
    labels=labels,
    split_path=split_path,
    extra={
        'task': 'trajectory_only',
        'baseline_type': 'constant_velocity',
        'future_len': FUTURE_LEN,
        'trajectory_baseline': 'last_observed_velocity_world_frame',
    },
)
save_run_manifest(result_dir, run_manifest, TIMESTAMP)
(result_dir / 'latest_run_manifest.json').write_text(json.dumps(run_manifest, indent=2))

overlay_paths = save_trajectory_overlay_plots(
    pred_future_xy,
    true_future_xy,
    targets,
    labels,
    plot_dir,
    prefix=TIMESTAMP,
    max_plots=8,
)
step_error_path = save_mean_step_error_plot(
    pred_future_xy,
    true_future_xy,
    plot_dir / f'mean_step_error_{TIMESTAMP}.png',
    title=f'{MODEL_SLUG} Mean Future-Step Error',
)
metrics['trajectory_overlay_plots'] = overlay_paths
metrics['mean_step_error_plot'] = step_error_path
metrics['status'] = 'saved'

metrics_path = result_dir / f'metrics_{TIMESTAMP}.json'
metrics_path.write_text(json.dumps(metrics, indent=2))
(result_dir / 'latest_metrics.json').write_text(json.dumps(metrics, indent=2))

print('CV trajectory baseline metrics:')
print(json.dumps(metrics, indent=2))
"""


def set_cell_source(nb, idx, text):
    nb["cells"][idx]["source"] = [line + "\n" for line in text.strip("\n").splitlines()]
    if nb["cells"][idx].get("cell_type") == "code":
        nb["cells"][idx]["outputs"] = []
        nb["cells"][idx]["execution_count"] = None


def main():
    nb = json.loads(NOTEBOOK.read_text())
    set_cell_source(nb, 5, CELL5)
    set_cell_source(nb, 6, CELL6)
    set_cell_source(nb, 7, CELL7)
    set_cell_source(nb, 8, CELL8)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print(f"updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
