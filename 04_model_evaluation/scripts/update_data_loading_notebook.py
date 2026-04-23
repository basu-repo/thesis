import json
from pathlib import Path


NOTEBOOK = Path("/home/basudeo/Documents/Thesis/04_model_evaluation/notebooks/00_data_loading_and_split.ipynb")


CELL4 = """# Shared experiment configuration.
# Keep this block explicit and simple so the data pipeline stays easy to debug.
LABEL_MODE = 'reduced'      # Keep the 5-state mapping for teacher controller states.
SEED = 42
PAST_LEN = 10
FUTURE_LEN = 5             # Future steps used for ADE/FDE/RMSE trajectory targets.
SCAN_BEAMS = 512
RANGE_CLIP = 30.0
BATCH_SIZE = 32
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
MAX_SAMPLES = None
USE_CPU = False

CNN_HIDDEN = 96
GRAPH_HIDDEN = 96
FUSION_HIDDEN = 128
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
MSG_PASSES = 2
DROPOUT = 0.10
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF = 256

device = torch.device('cuda' if torch.cuda.is_available() and not USE_CPU else 'cpu')
print('Device:', device)
"""


CELL5 = """# Shared data and evaluation utilities live in one helper module so that
# all notebooks reuse the same dataset, split, path-remapping, and metric logic.
from dataset_helper import (
    build_label_mapping,
    build_sample_table,
    compute_trajectory_metrics,
    configure_helper,
    discover_frame_files,
    group_streams,
    load_npy_cached,
    remap_dataset_path,
    resample_scan,
    save_mean_step_error_plot,
    save_or_load_fixed_split,
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


CELL7 = """labels, label_mapping = build_label_mapping(LABEL_MODE)
print('Controller-state metadata labels retained for analysis only:', labels)
print('Label mapping:', label_mapping)

frame_files = discover_frame_files(DATASET_ROOT)
assert frame_files, f'No frames.jsonl files found under {DATASET_ROOT}'
print('Found extracted datasets:')
for path in frame_files:
    print(' -', path)
"""


CELL8 = """streams = group_streams(DATASET_ROOT, allowed_labels=None, label_mapping=None)
print('Number of ego streams:', len(streams))

raw_label_counts = Counter()
for frames_path in frame_files:
    with frames_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_label = str(row['teacher'].get('controller_state') or row['teacher'].get('label') or 'go_to_goal')
            raw_label_counts[raw_label] += 1

print('Raw controller-state counts:')
display(pd.DataFrame({'raw_label': list(raw_label_counts.keys()), 'count': list(raw_label_counts.values())}).sort_values('count', ascending=False))
"""


CELL9 = """# Peek at one raw frame and one raw scan so the new control dataset layout stays intuitive.
raw_frame = None
for frames_path in frame_files:
    with frames_path.open() as f:
        for line in f:
            if line.strip():
                raw_frame = json.loads(line)
                break
    if raw_frame is not None:
        break

print('Example frame keys:', sorted(raw_frame.keys()))
print('Example teacher block:')
print(json.dumps(raw_frame['teacher'], indent=2))
print('Example observation block:')
print(json.dumps(raw_frame['observation'], indent=2))
print('Example state block:')
print(json.dumps(raw_frame['state'], indent=2))
print('Example goal features:')
print(json.dumps(raw_frame['goal_features'], indent=2))

scan_ref = raw_frame['observation']['ego_planar_scan']
scan_arr = load_npy_cached(str(scan_ref['path']))
print('Raw planar scan shape:', scan_arr.shape)
print('First five rows of the raw scan array:')
print(scan_arr[:5])
print('Processed scan shape:', resample_scan(scan_arr, SCAN_BEAMS, RANGE_CLIP).shape)
"""


CELL11 = """sample_table = build_sample_table(streams, past_len=PAST_LEN, future_len=FUTURE_LEN)
split_path = SPLITS_ROOT / f'husky_control_split_{LABEL_MODE}_seed{SEED}_past{PAST_LEN}_future{FUTURE_LEN}.json'
split_info = save_or_load_fixed_split(
    sample_table=sample_table,
    split_path=split_path,
    seed=SEED,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    past_len=PAST_LEN,
    future_len=FUTURE_LEN,
)

assert split_info['sample_count'] == len(sample_table), (
    f"Split file {split_path} does not match the current dataset: "
    f"split has {split_info['sample_count']} samples, current table has {len(sample_table)}"
)
print('Split file:', split_path)
print('Total windowed samples:', split_info['sample_count'])
print('Future horizon:', split_info['future_len'])
print('Train samples:', len(split_info['train_indices']))
print('Validation samples:', len(split_info['val_indices']))
print('Test samples:', len(split_info['test_indices']))
"""


def set_cell_source(nb, idx, text):
    nb["cells"][idx]["source"] = [line + "\n" for line in text.strip("\n").splitlines()]
    if nb["cells"][idx].get("cell_type") == "code":
        nb["cells"][idx]["outputs"] = []
        nb["cells"][idx]["execution_count"] = None


def main():
    nb = json.loads(NOTEBOOK.read_text())
    set_cell_source(nb, 4, CELL4)
    set_cell_source(nb, 5, CELL5)
    set_cell_source(nb, 6, CELL6)
    set_cell_source(nb, 7, CELL7)
    set_cell_source(nb, 8, CELL8)
    set_cell_source(nb, 9, CELL9)
    set_cell_source(nb, 11, CELL11)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print(f"updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
