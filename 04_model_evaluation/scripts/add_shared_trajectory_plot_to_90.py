import json
from pathlib import Path


nb_path = Path("/home/basudeo/Documents/Thesis/04_model_evaluation/notebooks/90_model_comparison.ipynb")
nb = json.loads(nb_path.read_text())

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in """
# Shared-sample trajectory line comparison across models.
trajectory_exports = {}
for slug in MODEL_SLUGS:
    npz_path = RESULTS_ROOT / slug / 'latest_trajectory_predictions.npz'
    if npz_path.exists():
        payload = np.load(npz_path, allow_pickle=False)
        trajectory_exports[slug] = {
            'sample_ids': payload['sample_ids'].astype(str),
            'pred_future_xy': payload['pred_future_xy'],
            'true_future_xy': payload['true_future_xy'],
        }

if len(trajectory_exports) < 2:
    print('Not enough trajectory export files found yet. Re-run the final evaluation cells of the model notebooks first.')
else:
    common_ids = None
    for slug, payload in trajectory_exports.items():
        ids = set(payload['sample_ids'].tolist())
        common_ids = ids if common_ids is None else common_ids.intersection(ids)

    if not common_ids:
        print('No shared sample_id found across the saved trajectory exports.')
    else:
        common_ids = sorted(common_ids)
        best_slug = 'cnn_gnn_lstm_transformer' if 'cnn_gnn_lstm_transformer' in trajectory_exports else next(iter(trajectory_exports))
        cv_slug = 'cv_baseline' if 'cv_baseline' in trajectory_exports else next(iter(trajectory_exports))

        best_map = {sid: idx for idx, sid in enumerate(trajectory_exports[best_slug]['sample_ids'])}
        cv_map = {sid: idx for idx, sid in enumerate(trajectory_exports[cv_slug]['sample_ids'])}

        def final_error(payload, idx):
            pred = payload['pred_future_xy'][idx, -1]
            true = payload['true_future_xy'][idx, -1]
            return float(np.linalg.norm(pred - true))

        scored_ids = []
        for sid in common_ids:
            if sid in best_map and sid in cv_map:
                cv_err = final_error(trajectory_exports[cv_slug], cv_map[sid])
                best_err = final_error(trajectory_exports[best_slug], best_map[sid])
                scored_ids.append((cv_err - best_err, sid))

        sample_id = max(scored_ids)[1] if scored_ids else common_ids[0]

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

        reference_payload = trajectory_exports[next(iter(trajectory_exports))]
        ref_idx = {sid: idx for idx, sid in enumerate(reference_payload['sample_ids'])}[sample_id]
        true_xy = reference_payload['true_future_xy'][ref_idx]
        ax.plot(true_xy[:, 0], true_xy[:, 1], '-o', color='black', linewidth=2.4, label='Ground Truth')

        for color, slug in zip(color_cycle, MODEL_SLUGS):
            if slug not in trajectory_exports:
                continue
            idx_map = {sid: idx for idx, sid in enumerate(trajectory_exports[slug]['sample_ids'])}
            if sample_id not in idx_map:
                continue
            pred_xy = trajectory_exports[slug]['pred_future_xy'][idx_map[sample_id]]
            ax.plot(pred_xy[:, 0], pred_xy[:, 1], '--o', color=color, linewidth=1.8, label=DISPLAY_NAMES.get(slug, slug))

        ax.set_title(f'Shared-Sample Trajectory Comparison\\nSample ID: {sample_id}')
        ax.set_xlabel('Relative X')
        ax.set_ylabel('Relative Y')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.axis('equal')
        fig.tight_layout()

        out_path = PLOTS_ROOT / 'trajectory_shared_sample_comparison.png'
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.show()
        print('Saved:', out_path)
""".strip("\n").splitlines()]
}

# insert before the final overlay panel cell
insert_at = len(nb["cells"])
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "trajectory_overlay_panel.png" in src:
            insert_at = i
            break

nb["cells"].insert(insert_at, new_cell)
nb_path.write_text(json.dumps(nb, indent=1))
print(f"updated {nb_path}")
