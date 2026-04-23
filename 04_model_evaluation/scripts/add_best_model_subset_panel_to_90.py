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
# Random 60% subset analysis for the best model using saved ground truth.
best_model_slug = 'cnn_gnn_lstm_transformer'
best_npz = RESULTS_ROOT / best_model_slug / 'latest_trajectory_predictions.npz'

if not best_npz.exists():
    print(f'Missing trajectory export for {best_model_slug}:', best_npz)
else:
    payload = np.load(best_npz, allow_pickle=False)
    sample_ids = payload['sample_ids'].astype(str)
    pred_future_xy = payload['pred_future_xy']
    true_future_xy = payload['true_future_xy']

    rng = np.random.default_rng(42)
    total = len(sample_ids)
    subset_size = max(1, int(round(0.60 * total)))
    subset_idx = np.sort(rng.choice(total, size=subset_size, replace=False))

    subset_pred = pred_future_xy[subset_idx]
    subset_true = true_future_xy[subset_idx]
    subset_ids = sample_ids[subset_idx]

    diff = subset_pred - subset_true
    subset_ade = float(np.linalg.norm(diff, axis=-1).mean())
    subset_fde = float(np.linalg.norm(diff[:, -1, :], axis=-1).mean())
    subset_rmse = float(np.sqrt((diff ** 2).sum(axis=-1).mean()))

    print('Best model:', best_model_slug)
    print('Subset size:', subset_size, 'out of', total)
    print('Subset ADE:', round(subset_ade, 4))
    print('Subset FDE:', round(subset_fde, 4))
    print('Subset RMSE:', round(subset_rmse, 4))

    # pick a diverse set of examples from the subset
    final_errors = np.linalg.norm(diff[:, -1, :], axis=-1)
    order = np.argsort(final_errors)
    picks = []
    if len(order) > 0:
        candidates = [order[0], order[len(order)//4], order[len(order)//2], order[(3*len(order))//4], order[-1]]
        seen = set()
        for idx in candidates:
            idx = int(idx)
            if idx not in seen and 0 <= idx < len(order):
                picks.append(idx)
                seen.add(idx)
        while len(picks) < min(6, len(order)):
            idx = int(order[len(picks)-1])
            if idx not in seen:
                picks.append(idx)
                seen.add(idx)

    ncols = 2
    nrows = int(np.ceil(max(1, len(picks)) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax in axes:
        ax.axis('off')

    for ax, idx in zip(axes, picks):
        ax.axis('on')
        gt = subset_true[idx]
        pred = subset_pred[idx]
        ax.plot(gt[:, 0], gt[:, 1], '-o', color='black', linewidth=2, label='Ground Truth')
        ax.plot(pred[:, 0], pred[:, 1], '--o', color='tab:red', linewidth=1.8, label='Prediction')
        ax.set_title(f'{subset_ids[idx]}\\nFDE={final_errors[idx]:.2f}', fontsize=9)
        ax.set_xlabel('Relative X')
        ax.set_ylabel('Relative Y')
        ax.grid(True, alpha=0.25)
        ax.axis('equal')

    if len(picks) > 0:
        axes[0].legend(fontsize=8)

    fig.suptitle('CNN-GNN-LSTM-Transformer: Random 60% Subset Trajectory Examples', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = PLOTS_ROOT / 'best_model_random60_trajectory_panel.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.show()
    print('Saved:', out_path)
""".strip("\n").splitlines()]
}

nb["cells"].append(new_cell)
nb_path.write_text(json.dumps(nb, indent=1))
print(f"updated {nb_path}")
