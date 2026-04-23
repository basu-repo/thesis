import json
from pathlib import Path


ROOT = Path("/home/basudeo/Documents/Thesis/04_model_evaluation/notebooks")


COMMON_IMPORTS = """# Shared data and evaluation utilities live in one helper module so that
# all notebooks reuse the same dataset, split, path-remapping, and trajectory metrics.
from dataset_helper import (
    build_run_manifest,
    build_sample_table,
    build_label_mapping,
    compute_trajectory_metrics,
    configure_helper,
    group_streams,
    load_npy_cached,
    prepare_result_dirs,
    resample_scan,
    save_mean_step_error_plot,
    save_or_load_fixed_split,
    save_run_manifest,
    save_training_history,
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


COMMON_PATHS = """# Resolve the project structure relative to this notebook.
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


COMMON_SPLIT_20 = """set_seed(SEED)
labels, label_mapping = build_label_mapping(LABEL_MODE)
label_to_idx = {label: idx for idx, label in enumerate(labels)}
streams = group_streams(DATASET_ROOT, allowed_labels=None, label_mapping=None)
sample_table = build_sample_table(streams, past_len=PAST_LEN, future_len=FUTURE_LEN)
split_path = SPLITS_ROOT / f'husky_control_split_{LABEL_MODE}_seed{SEED}_past{PAST_LEN}_future{FUTURE_LEN}.json'
split_info = save_or_load_fixed_split(sample_table, split_path, SEED, TRAIN_RATIO, VAL_RATIO, PAST_LEN, FUTURE_LEN)
assert split_info['sample_count'] == len(sample_table), (
    f"Split file {split_path} does not match the current dataset: "
    f"split has {split_info['sample_count']} samples, current table has {len(sample_table)}"
)

print('Controller-state metadata labels retained for plots only:', labels)
print('Split path:', split_path)
print('Total samples in canonical table:', len(sample_table))
print('Train / Val / Test:', len(split_info['train_indices']), len(split_info['val_indices']), len(split_info['test_indices']))
print('Future horizon:', split_info['future_len'])
"""


DATASET_20 = """@dataclass
class TrajectorySample:
    scan_seq: torch.Tensor | None
    node_seq: torch.Tensor | None
    edge_seq: torch.Tensor | None
    label: torch.Tensor
    future_xy: torch.Tensor
    future_dt: torch.Tensor
    sample_id: str


class BaseTrajectoryDataset(Dataset):
    def __init__(self, streams, sample_table, label_to_idx, past_len, scan_beams, range_clip):
        self.streams = streams
        self.sample_table = sample_table
        self.label_to_idx = label_to_idx
        self.past_len = past_len
        self.scan_beams = scan_beams
        self.range_clip = range_clip

    def __len__(self):
        return len(self.sample_table)

    def _window(self, idx):
        meta = self.sample_table[idx]
        stream = self.streams[meta['stream_idx']]
        return stream[meta['start']: meta['start'] + self.past_len], meta


class ScanOnlyDataset(BaseTrajectoryDataset):
    def __getitem__(self, idx):
        window, meta = self._window(idx)
        scan_seq = []
        for frame in window:
            scan_ref = frame['observation']['ego_planar_scan']
            scan = load_npy_cached(str(scan_ref['path']))
            scan_seq.append(resample_scan(scan, self.scan_beams, self.range_clip))
        label_name = meta.get('raw_label') or 'go_to_goal'
        label_idx = self.label_to_idx.get(label_name, 0)
        return TrajectorySample(
            scan_seq=torch.tensor(np.stack(scan_seq, axis=0), dtype=torch.float32),
            node_seq=None,
            edge_seq=None,
            label=torch.tensor(label_idx, dtype=torch.long),
            future_xy=torch.tensor(meta['future_xy'], dtype=torch.float32),
            future_dt=torch.tensor(meta['future_dt'], dtype=torch.float32),
            sample_id=meta['sample_id'],
        )


def collate_samples(batch):
    scan_seq = None if batch[0].scan_seq is None else torch.stack([sample.scan_seq for sample in batch], dim=0)
    node_seq = None if batch[0].node_seq is None else torch.stack([sample.node_seq for sample in batch], dim=0)
    edge_seq = None if batch[0].edge_seq is None else torch.stack([sample.edge_seq for sample in batch], dim=0)
    labels = torch.stack([sample.label for sample in batch], dim=0)
    future_xy = torch.stack([sample.future_xy for sample in batch], dim=0)
    future_dt = torch.stack([sample.future_dt for sample in batch], dim=0)
    sample_ids = [sample.sample_id for sample in batch]
    return scan_seq, node_seq, edge_seq, labels, future_xy, future_dt, sample_ids
"""


DATASET_40 = """@dataclass
class TrajectorySample:
    scan_seq: torch.Tensor | None
    node_seq: torch.Tensor | None
    edge_seq: torch.Tensor | None
    label: torch.Tensor
    future_xy: torch.Tensor
    future_dt: torch.Tensor
    sample_id: str


def _node_feature_from_frame(frame: dict, role: str):
    if role == 'ego':
        state = frame['state']
        goal = frame['goal_features']
        command = frame['teacher']['command'] or {'linear_x': 0.0, 'angular_z': 0.0}
        available = 1.0
    else:
        other = frame['other_husky']
        state = other['state'] or {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'wz': 0.0}
        goal = other['goal_features'] or {'dx': 0.0, 'dy': 0.0, 'dz': 0.0, 'distance_to_goal': 0.0, 'heading_error': 0.0}
        command = other['teacher_command'] or {'linear_x': 0.0, 'angular_z': 0.0}
        available = 1.0 if other.get('available', False) else 0.0
    return [
        float(state.get('x', 0.0)),
        float(state.get('y', 0.0)),
        float(state.get('z', 0.0)),
        float(state.get('yaw', 0.0)),
        float(state.get('vx', 0.0)),
        float(state.get('vy', 0.0)),
        float(state.get('vz', 0.0)),
        float(state.get('wz', 0.0)),
        float(goal.get('dx', 0.0)),
        float(goal.get('dy', 0.0)),
        float(goal.get('distance_to_goal', 0.0)),
        float(goal.get('heading_error', 0.0)),
        float(command.get('linear_x', 0.0)),
        available,
    ]


def _edge_features(frame: dict):
    ego = frame['state']
    other = frame['other_husky']['state']
    if other is None:
        dx = dy = dz = distance = 0.0
    else:
        dx = float(other['x'] - ego['x'])
        dy = float(other['y'] - ego['y'])
        dz = float(other['z'] - ego['z'])
        distance = float((dx * dx + dy * dy + dz * dz) ** 0.5)
    inv_distance = 0.0 if distance <= 1e-6 else float(1.0 / distance)
    bearing = float(np.arctan2(dy, dx)) if distance > 1e-6 else 0.0
    ego_to_other = [dx, dy, dz, distance, inv_distance, float(np.sin(bearing)), float(np.cos(bearing)), 1.0]
    other_to_ego = [-dx, -dy, -dz, distance, inv_distance, float(np.sin(-bearing)), float(np.cos(-bearing)), 1.0]
    self_edge = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    return [
        [self_edge, ego_to_other],
        [other_to_ego, self_edge],
    ]


class BaseTrajectoryDataset(Dataset):
    def __init__(self, streams, sample_table, label_to_idx, past_len, scan_beams, range_clip):
        self.streams = streams
        self.sample_table = sample_table
        self.label_to_idx = label_to_idx
        self.past_len = past_len
        self.scan_beams = scan_beams
        self.range_clip = range_clip

    def __len__(self):
        return len(self.sample_table)

    def _window(self, idx):
        meta = self.sample_table[idx]
        stream = self.streams[meta['stream_idx']]
        return stream[meta['start']: meta['start'] + self.past_len], meta


class ScanGraphDataset(BaseTrajectoryDataset):
    def __getitem__(self, idx):
        window, meta = self._window(idx)
        scan_seq, node_seq, edge_seq = [], [], []
        for frame in window:
            scan_ref = frame['observation']['ego_planar_scan']
            scan = load_npy_cached(str(scan_ref['path']))
            scan_seq.append(resample_scan(scan, self.scan_beams, self.range_clip))
            node_seq.append([
                _node_feature_from_frame(frame, 'ego'),
                _node_feature_from_frame(frame, 'other'),
            ])
            edge_seq.append(_edge_features(frame))
        label_name = meta.get('raw_label') or 'go_to_goal'
        label_idx = self.label_to_idx.get(label_name, 0)
        return TrajectorySample(
            scan_seq=torch.tensor(np.stack(scan_seq, axis=0), dtype=torch.float32),
            node_seq=torch.tensor(node_seq, dtype=torch.float32),
            edge_seq=torch.tensor(edge_seq, dtype=torch.float32),
            label=torch.tensor(label_idx, dtype=torch.long),
            future_xy=torch.tensor(meta['future_xy'], dtype=torch.float32),
            future_dt=torch.tensor(meta['future_dt'], dtype=torch.float32),
            sample_id=meta['sample_id'],
        )


def collate_samples(batch):
    scan_seq = None if batch[0].scan_seq is None else torch.stack([sample.scan_seq for sample in batch], dim=0)
    node_seq = None if batch[0].node_seq is None else torch.stack([sample.node_seq for sample in batch], dim=0)
    edge_seq = None if batch[0].edge_seq is None else torch.stack([sample.edge_seq for sample in batch], dim=0)
    labels = torch.stack([sample.label for sample in batch], dim=0)
    future_xy = torch.stack([sample.future_xy for sample in batch], dim=0)
    future_dt = torch.stack([sample.future_dt for sample in batch], dim=0)
    sample_ids = [sample.sample_id for sample in batch]
    return scan_seq, node_seq, edge_seq, labels, future_xy, future_dt, sample_ids
"""


def set_cell(nb, idx, text):
    nb["cells"][idx]["source"] = [line + "\n" for line in text.strip("\n").splitlines()]
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None


def update_notebook(name: str, dataset_cell_idx: int, split_cell_idx: int):
    path = ROOT / name
    nb = json.loads(path.read_text())
    set_cell(nb, 5, COMMON_IMPORTS)
    set_cell(nb, 6 if name == "20_cnn_lstm.ipynb" else 6, DATASET_20 if name == "20_cnn_lstm.ipynb" else DATASET_40)
    # Replace old path/config cell if still present in outputs/source
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "ORIGINAL_DATASET_ROOT =" in src and "husky_control_dataset" not in src:
                cell["source"] = [line + "\n" for line in COMMON_PATHS.strip("\n").splitlines()]
                cell["outputs"] = []
                cell["execution_count"] = None
                break
    set_cell(nb, split_cell_idx, COMMON_SPLIT_20)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1))
    print(f"updated {path}")


def main():
    update_notebook("20_cnn_lstm.ipynb", 6, 9)
    update_notebook("40_cnn_gnn_lstm.ipynb", 6, 10)


if __name__ == "__main__":
    main()
