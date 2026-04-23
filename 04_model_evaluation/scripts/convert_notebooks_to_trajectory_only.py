import json
from pathlib import Path


NOTEBOOKS_ROOT = Path("/home/basudeo/Documents/Thesis/04_model_evaluation/notebooks")


IMPORT_CELL = """# Shared data and evaluation utilities live in one helper module so that
# all notebooks reuse the same dataset, split, path-remapping, and trajectory metrics.
from dataset_helper import (
    build_run_manifest,
    build_sample_table,
    build_label_mapping,
    canonical_agent_order,
    compute_trajectory_metrics,
    configure_helper,
    edge_features_for_order,
    group_streams,
    load_npy_cached,
    node_feature,
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


DATASET_CELL = """@dataclass
class TrajectorySample:
    scan_seq: torch.Tensor | None
    node_seq: torch.Tensor | None
    edge_seq: torch.Tensor | None
    label: torch.Tensor
    future_xy: torch.Tensor
    future_dt: torch.Tensor
    sample_id: str


class BaseTrajectoryDataset(Dataset):
    \"\"\"Base dataset that turns JSONL streams into time windows.

    Subclasses decide which modalities they need from each window.
    \"\"\"
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
            scan_ref = frame['modalities']['ego_planar_scan']
            scan = load_npy_cached(str(scan_ref['path']))
            scan_seq.append(resample_scan(scan, self.scan_beams, self.range_clip))
        label_idx = self.label_to_idx[meta['label']]
        return TrajectorySample(
            scan_seq=torch.tensor(np.stack(scan_seq, axis=0), dtype=torch.float32),
            node_seq=None,
            edge_seq=None,
            label=torch.tensor(label_idx, dtype=torch.long),
            future_xy=torch.tensor(meta['future_xy'], dtype=torch.float32),
            future_dt=torch.tensor(meta['future_dt'], dtype=torch.float32),
            sample_id=meta['sample_id'],
        )


class GraphOnlyDataset(BaseTrajectoryDataset):
    def __getitem__(self, idx):
        window, meta = self._window(idx)
        node_seq, edge_seq = [], []
        ego_id = meta['ego_id']
        order = canonical_agent_order(ego_id)
        for frame in window:
            agents = frame['agents']
            ego_state = agents[ego_id]['state']
            node_seq.append([node_feature(agents[agent_id], ego_state) for agent_id in order])
            edge_seq.append(edge_features_for_order(frame, order))
        label_idx = self.label_to_idx[meta['label']]
        return TrajectorySample(
            scan_seq=None,
            node_seq=torch.tensor(node_seq, dtype=torch.float32),
            edge_seq=torch.tensor(edge_seq, dtype=torch.float32),
            label=torch.tensor(label_idx, dtype=torch.long),
            future_xy=torch.tensor(meta['future_xy'], dtype=torch.float32),
            future_dt=torch.tensor(meta['future_dt'], dtype=torch.float32),
            sample_id=meta['sample_id'],
        )


class ScanGraphDataset(BaseTrajectoryDataset):
    def __getitem__(self, idx):
        window, meta = self._window(idx)
        scan_seq, node_seq, edge_seq = [], [], []
        ego_id = meta['ego_id']
        order = canonical_agent_order(ego_id)
        for frame in window:
            scan_ref = frame['modalities']['ego_planar_scan']
            scan = load_npy_cached(str(scan_ref['path']))
            scan_seq.append(resample_scan(scan, self.scan_beams, self.range_clip))
            agents = frame['agents']
            ego_state = agents[ego_id]['state']
            node_seq.append([node_feature(agents[agent_id], ego_state) for agent_id in order])
            edge_seq.append(edge_features_for_order(frame, order))
        label_idx = self.label_to_idx[meta['label']]
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


SPLIT_CELL = """set_seed(SEED)
labels, label_mapping = build_label_mapping(LABEL_MODE)
label_to_idx = {label: idx for idx, label in enumerate(labels)}
streams = group_streams(DATASET_ROOT, allowed_labels=set(labels), label_mapping=label_mapping)
sample_table = build_sample_table(streams, past_len=PAST_LEN, future_len=FUTURE_LEN)
split_path = SPLITS_ROOT / f'hybrid_maneuvers_split_{LABEL_MODE}_seed{SEED}_past{PAST_LEN}_future{FUTURE_LEN}.json'
split_info = save_or_load_fixed_split(sample_table, split_path, SEED, TRAIN_RATIO, VAL_RATIO, PAST_LEN, FUTURE_LEN)
assert split_info['sample_count'] == len(sample_table), (
    f"Split file {split_path} does not match the current dataset: "
    f"split has {split_info['sample_count']} samples, current table has {len(sample_table)}"
)

print('Trajectory labels kept for dataset filtering and plot titles:', labels)
print('Split path:', split_path)
print('Total samples in canonical table:', len(sample_table))
print('Train / Val / Test:', len(split_info['train_indices']), len(split_info['val_indices']), len(split_info['test_indices']))
print('Future horizon:', split_info['future_len'])
"""


TRAJ_UTILS_CELL = """def save_trajectory_history_plot(history: dict, out_path: Path, title_prefix: str):
    if not history or len(history.get('epoch', [])) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history['epoch'], history['train_loss'], label='train_loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='val_loss')
    axes[0].set_title(f'{title_prefix}: Trajectory Loss')
    axes[0].legend()

    axes[1].plot(history['epoch'], history['val_ade'], label='val_ADE', color='tab:green')
    axes[1].plot(history['epoch'], history['val_fde'], label='val_FDE', color='tab:orange')
    axes[1].plot(history['epoch'], history['val_rmse'], label='val_RMSE', color='tab:red')
    axes[1].set_title(f'{title_prefix}: Validation Trajectory Metrics')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def evaluate_trajectory_model(model, loader, device, labels):
    model.eval()
    all_targets, all_sample_ids = [], []
    all_pred_future_xy, all_true_future_xy = [], []
    total_loss = 0.0
    total_count = 0
    criterion_traj = nn.MSELoss()
    with torch.no_grad():
        for scan_seq, node_seq, edge_seq, labels_batch, future_xy_batch, _future_dt_batch, sample_ids in loader:
            if scan_seq is not None:
                scan_seq = scan_seq.to(device)
            if node_seq is not None:
                node_seq = node_seq.to(device)
            if edge_seq is not None:
                edge_seq = edge_seq.to(device)
            future_xy_batch = future_xy_batch.to(device)

            pred_future_xy = model(scan_seq, node_seq, edge_seq)
            loss = criterion_traj(pred_future_xy, future_xy_batch)

            total_loss += float(loss.item()) * future_xy_batch.size(0)
            total_count += int(future_xy_batch.size(0))
            all_targets.append(labels_batch.cpu().numpy())
            all_pred_future_xy.append(pred_future_xy.cpu().numpy())
            all_true_future_xy.append(future_xy_batch.cpu().numpy())
            all_sample_ids.extend(sample_ids)

    targets = np.concatenate(all_targets, axis=0)
    pred_future_xy = np.concatenate(all_pred_future_xy, axis=0)
    true_future_xy = np.concatenate(all_true_future_xy, axis=0)

    metrics = compute_trajectory_metrics(pred_future_xy, true_future_xy)
    metrics['loss'] = total_loss / max(total_count, 1)
    return {
        'metrics': metrics,
        'targets': targets,
        'sample_ids': all_sample_ids,
        'pred_future_xy': pred_future_xy,
        'true_future_xy': true_future_xy,
    }


def train_trajectory_model(model, train_loader, val_loader, labels, model_slug, timestamp, split_path, extra_manifest=None, save_weights=True):
    result_dir, weight_dir, plot_dir = prepare_result_dirs(model_slug)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion_traj = nn.MSELoss()

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_ade': [],
        'val_fde': [],
        'val_rmse': [],
    }
    best_val_ade = math.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for scan_seq, node_seq, edge_seq, _labels_batch, future_xy_batch, _future_dt_batch, _sample_ids in train_loader:
            if scan_seq is not None:
                scan_seq = scan_seq.to(device)
            if node_seq is not None:
                node_seq = node_seq.to(device)
            if edge_seq is not None:
                edge_seq = edge_seq.to(device)
            future_xy_batch = future_xy_batch.to(device)

            optimizer.zero_grad()
            pred_future_xy = model(scan_seq, node_seq, edge_seq)
            loss = criterion_traj(pred_future_xy, future_xy_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * future_xy_batch.size(0)
            running_count += int(future_xy_batch.size(0))

        train_loss = running_loss / max(running_count, 1)
        val_eval = evaluate_trajectory_model(model, val_loader, device, labels)
        val_metrics = val_eval['metrics']

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_ade'].append(val_metrics['ADE'])
        history['val_fde'].append(val_metrics['FDE'])
        history['val_rmse'].append(val_metrics['RMSE'])

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_ADE={val_metrics['ADE']:.4f} "
            f"val_FDE={val_metrics['FDE']:.4f} "
            f"val_RMSE={val_metrics['RMSE']:.4f}"
        )

        if val_metrics['ADE'] < best_val_ade:
            best_val_ade = val_metrics['ADE']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping at epoch {epoch:02d}')
                break

    if best_state is None:
        raise RuntimeError('Training ended without a valid best checkpoint.')

    model.load_state_dict(best_state)
    run_manifest = build_run_manifest(
        model_slug=model_slug,
        timestamp=timestamp,
        labels=labels,
        split_path=split_path,
        extra={**(extra_manifest or {}), 'task': 'trajectory_only', 'selection_metric': 'ADE'},
    )
    manifest_path = save_run_manifest(result_dir, run_manifest, timestamp)

    weight_payload = {
        'model_state': best_state,
        'labels': labels,
        'timestamp': timestamp,
        'model_slug': model_slug,
        'run_manifest': run_manifest,
    }
    if save_weights:
        torch.save(weight_payload, weight_dir / f'{model_slug}_{timestamp}.pt')
        torch.save(weight_payload, weight_dir / 'latest.pt')

    save_training_history(history, result_dir / f'history_{timestamp}.csv')
    save_trajectory_history_plot(history, plot_dir / f'history_{timestamp}.png', model_slug)

    return {
        'history': history,
        'best_val_ade': float(best_val_ade),
        'result_dir': result_dir,
        'plot_dir': plot_dir,
        'weight_dir': weight_dir,
        'run_manifest_path': manifest_path,
        'run_manifest': run_manifest,
    }


def save_final_trajectory_evaluation(model_slug, timestamp, train_out, test_eval, labels, extra_summary=None, split_path=None):
    result_dir = train_out['result_dir'] if train_out is not None else prepare_result_dirs(model_slug)[0]
    plot_dir = train_out['plot_dir'] if train_out is not None else prepare_result_dirs(model_slug)[2]

    metrics = dict(test_eval['metrics'])
    if split_path is not None:
        metrics['split_path'] = str(split_path)
    if train_out is not None and 'best_val_ade' in train_out:
        metrics['best_val_ADE'] = float(train_out['best_val_ade'])
    if extra_summary:
        metrics.update(extra_summary)

    if train_out is not None and 'run_manifest' in train_out:
        (result_dir / 'latest_run_manifest.json').write_text(json.dumps(train_out['run_manifest'], indent=2))

    overlay_paths = save_trajectory_overlay_plots(
        test_eval['pred_future_xy'],
        test_eval['true_future_xy'],
        test_eval['targets'],
        labels,
        plot_dir,
        prefix=timestamp,
        max_plots=8,
    )
    step_error_path = save_mean_step_error_plot(
        test_eval['pred_future_xy'],
        test_eval['true_future_xy'],
        plot_dir / f'mean_step_error_{timestamp}.png',
        title=f'{model_slug} Mean Future-Step Error',
    )
    metrics['trajectory_overlay_plots'] = overlay_paths
    metrics['mean_step_error_plot'] = step_error_path
    metrics['status'] = 'saved'

    metrics_path = result_dir / f'metrics_{timestamp}.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))
    (result_dir / 'latest_metrics.json').write_text(json.dumps(metrics, indent=2))
    return metrics
"""


BASELINE_RUN_CELL = """MODEL_SLUG = 'cv_baseline'
TIMESTAMP = timestamp_tag()
result_dir, weight_dir, plot_dir = prepare_result_dirs(MODEL_SLUG)

# This baseline extrapolates the final observed ego velocity across the saved
# future timestamps. It is intentionally simple and gives us a cheap trajectory
# reference point to beat with the learned models.
dataset = ScanGraphDataset(
    streams=streams,
    sample_table=sample_table,
    label_to_idx=label_to_idx,
    past_len=PAST_LEN,
    scan_beams=SCAN_BEAMS,
    range_clip=RANGE_CLIP,
)

test_subset = Subset(dataset, split_info['test_indices'])
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_samples)

all_targets, all_sample_ids = [], []
all_pred_future_xy, all_true_future_xy = [], []
for _scan_seq, node_seq, _edge_seq, labels_batch, future_xy_batch, future_dt_batch, sample_ids in test_loader:
    ego_last = node_seq[:, -1, 0]
    vx_last = ego_last[:, 3].unsqueeze(1)
    vy_last = ego_last[:, 4].unsqueeze(1)
    pred_future_xy = torch.stack([vx_last * future_dt_batch, vy_last * future_dt_batch], dim=-1)

    all_targets.append(labels_batch.numpy())
    all_pred_future_xy.append(pred_future_xy.numpy())
    all_true_future_xy.append(future_xy_batch.numpy())
    all_sample_ids.extend(sample_ids)

targets = np.concatenate(all_targets, axis=0)
pred_future_xy = np.concatenate(all_pred_future_xy, axis=0)
true_future_xy = np.concatenate(all_true_future_xy, axis=0)
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


TRAJECTORY_NOTEBOOKS = {
    "10_cv_baseline.ipynb": {
        "import_idx": 5,
        "dataset_idx": 6,
        "split_idx": 7,
        "run_idx": 8,
    },
    "20_cnn_lstm.ipynb": {
        "import_idx": 5,
        "dataset_idx": 6,
        "utils_idx": 8,
        "split_idx": 9,
    },
    "40_cnn_gnn_lstm.ipynb": {
        "import_idx": 5,
        "dataset_idx": 6,
        "utils_idx": 9,
        "split_idx": 10,
    },
    "50_cnn_gnn_transformer.ipynb": {
        "import_idx": 5,
        "dataset_idx": 6,
        "utils_idx": 10,
        "split_idx": 11,
    },
    "60_cnn_gnn_lstm_transformer.ipynb": {
        "import_idx": 5,
        "dataset_idx": 6,
        "utils_idx": 10,
        "split_idx": 11,
    },
}


def set_source(cell, text):
    cell["source"] = [line + "\n" for line in text.strip("\n").splitlines()]
    cell["outputs"] = []
    cell["execution_count"] = None


def clear_all_outputs(nb):
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def main():
    for name, cfg in TRAJECTORY_NOTEBOOKS.items():
        path = NOTEBOOKS_ROOT / name
        nb = json.loads(path.read_text())
        clear_all_outputs(nb)
        set_source(nb["cells"][cfg["import_idx"]], IMPORT_CELL)
        set_source(nb["cells"][cfg["dataset_idx"]], DATASET_CELL)
        set_source(nb["cells"][cfg["split_idx"]], SPLIT_CELL)
        if "utils_idx" in cfg:
            set_source(nb["cells"][cfg["utils_idx"]], TRAJ_UTILS_CELL)
        if "run_idx" in cfg:
            set_source(nb["cells"][cfg["run_idx"]], BASELINE_RUN_CELL)
        path.write_text(json.dumps(nb, indent=1))
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
