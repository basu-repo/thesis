import json
from pathlib import Path


def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


nb_path = Path("/home/basudeo/Documents/Thesis/04_model_evaluation/notebooks/90_model_comparison.ipynb")
nb = json.loads(nb_path.read_text())

nb["cells"] = [
    markdown_cell(
        """
# Trajectory Model Comparison

This notebook summarizes the saved trajectory-prediction results and generates paper-friendly plots.
"""
    ),
    markdown_cell(
        """
## Run Safety

Restart the kernel before running this notebook so the exported tables and plots come from a clean session.
"""
    ),
    code_cell(
        """
# Memory cleanup before starting this notebook.
import gc

gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception:
    pass

print('Kernel memory cleanup complete. Start the notebook from here after a restart.')
"""
    ),
    code_cell(
        """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve().parent
EVAL_ROOT = PROJECT_ROOT
RESULTS_ROOT = EVAL_ROOT / 'results'
PLOTS_ROOT = EVAL_ROOT / 'plots'
EXPORT_ROOT = EVAL_ROOT / 'comparison_exports'
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_SLUGS = [
    'cv_baseline',
    'cnn_lstm',
    'cnn_gnn_lstm',
    'cnn_gnn_transformer',
    'cnn_gnn_lstm_transformer',
]

DISPLAY_NAMES = {
    'cv_baseline': 'CV Baseline',
    'cnn_lstm': 'CNN-LSTM',
    'cnn_gnn_lstm': 'CNN-GNN-LSTM',
    'cnn_gnn_transformer': 'CNN-GNN-Transformer',
    'cnn_gnn_lstm_transformer': 'CNN-GNN-LSTM-Transformer',
}

summary_rows = []
for slug in MODEL_SLUGS:
    latest_metrics_path = RESULTS_ROOT / slug / 'latest_metrics.json'
    if not latest_metrics_path.exists():
        continue
    payload = json.loads(latest_metrics_path.read_text())
    payload['model'] = slug
    payload['display_name'] = DISPLAY_NAMES.get(slug, slug)
    summary_rows.append(payload)

summary_df = pd.DataFrame(summary_rows)
if summary_df.empty:
    raise RuntimeError('No latest_metrics.json files were found. Run the model notebooks first.')

summary_df = summary_df[['model', 'display_name', 'ADE', 'FDE', 'RMSE', 'loss', 'best_val_ADE', 'split_path', 'mean_step_error_plot', 'trajectory_overlay_plots']]
summary_df = summary_df.sort_values(['RMSE', 'FDE', 'ADE'], ascending=True).reset_index(drop=True)

display(summary_df[['display_name', 'ADE', 'FDE', 'RMSE']])
print(summary_df[['display_name', 'ADE', 'FDE', 'RMSE']].to_string(index=False))

comparison_csv = EXPORT_ROOT / 'trajectory_model_summary_latest.csv'
comparison_json = EXPORT_ROOT / 'trajectory_model_summary_latest.json'
summary_df.to_csv(comparison_csv, index=False)
comparison_json.write_text(summary_df.to_json(orient='records', indent=2))
print('Saved comparison CSV:', comparison_csv)
print('Saved comparison JSON:', comparison_json)
"""
    ),
    code_cell(
        """
# Paper table preview in the same order as the saved ranking.
table_df = summary_df[['display_name', 'ADE', 'FDE', 'RMSE']].copy()
table_df['ADE'] = table_df['ADE'].map(lambda x: f'{x:.2f}')
table_df['FDE'] = table_df['FDE'].map(lambda x: f'{x:.2f}')
table_df['RMSE'] = table_df['RMSE'].map(lambda x: f'{x:.2f}')
display(table_df)
"""
    ),
    code_cell(
        """
# Grouped bar chart for ADE / FDE / RMSE.
plot_df = summary_df[['display_name', 'ADE', 'FDE', 'RMSE']].copy()
x = np.arange(len(plot_df))
width = 0.24

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.bar(x - width, plot_df['ADE'], width=width, label='ADE')
ax.bar(x,         plot_df['FDE'], width=width, label='FDE')
ax.bar(x + width, plot_df['RMSE'], width=width, label='RMSE')

ax.set_xticks(x)
ax.set_xticklabels(plot_df['display_name'], rotation=20, ha='right')
ax.set_ylabel('Error')
ax.set_title('Trajectory Prediction Metrics by Model')
ax.legend()
ax.grid(True, axis='y', alpha=0.25)
fig.tight_layout()

out_path = PLOTS_ROOT / 'trajectory_metric_comparison_bar.png'
fig.savefig(out_path, dpi=220, bbox_inches='tight')
plt.show()
print('Saved:', out_path)
"""
    ),
    code_cell(
        """
# Relative improvement over the CV baseline. Positive is better.
baseline = summary_df.loc[summary_df['model'] == 'cv_baseline', ['ADE', 'FDE', 'RMSE']]
if baseline.empty:
    print('CV baseline not found; skipping improvement plot.')
else:
    baseline = baseline.iloc[0]
    improved = summary_df[summary_df['model'] != 'cv_baseline'][['display_name', 'ADE', 'FDE', 'RMSE']].copy()
    for metric in ['ADE', 'FDE', 'RMSE']:
        improved[metric] = 100.0 * (baseline[metric] - improved[metric]) / baseline[metric]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(improved))
    width = 0.24
    ax.bar(x - width, improved['ADE'], width=width, label='ADE improvement')
    ax.bar(x,         improved['FDE'], width=width, label='FDE improvement')
    ax.bar(x + width, improved['RMSE'], width=width, label='RMSE improvement')
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(improved['display_name'], rotation=20, ha='right')
    ax.set_ylabel('Improvement over CV baseline (%)')
    ax.set_title('Relative Improvement over the CV Baseline')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()

    out_path = PLOTS_ROOT / 'trajectory_improvement_over_cv.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.show()
    print('Saved:', out_path)
"""
    ),
    code_cell(
        """
# Combined panel of the saved mean-step-error plots.
panel_models = ['cv_baseline', 'cnn_lstm', 'cnn_gnn_lstm', 'cnn_gnn_transformer', 'cnn_gnn_lstm_transformer']
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
axes = axes.flatten()

for ax in axes:
    ax.axis('off')

for idx, slug in enumerate(panel_models):
    row = summary_df.loc[summary_df['model'] == slug]
    if row.empty:
        continue
    plot_path = row.iloc[0]['mean_step_error_plot']
    if not plot_path or not Path(plot_path).exists():
        continue
    img = mpimg.imread(plot_path)
    axes[idx].imshow(img)
    axes[idx].set_title(DISPLAY_NAMES.get(slug, slug), fontsize=10)
    axes[idx].axis('off')

fig.tight_layout()
out_path = PLOTS_ROOT / 'mean_step_error_panel.png'
fig.savefig(out_path, dpi=220, bbox_inches='tight')
plt.show()
print('Saved:', out_path)
"""
    ),
    code_cell(
        """
# Combined panel of one trajectory overlay example per model.
panel_models = ['cv_baseline', 'cnn_lstm', 'cnn_gnn_lstm', 'cnn_gnn_transformer', 'cnn_gnn_lstm_transformer']
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
axes = axes.flatten()

for ax in axes:
    ax.axis('off')

for idx, slug in enumerate(panel_models):
    row = summary_df.loc[summary_df['model'] == slug]
    if row.empty:
        continue
    overlays = row.iloc[0]['trajectory_overlay_plots']
    if not overlays:
        continue
    first_overlay = overlays[0]
    if not first_overlay or not Path(first_overlay).exists():
        continue
    img = mpimg.imread(first_overlay)
    axes[idx].imshow(img)
    axes[idx].set_title(DISPLAY_NAMES.get(slug, slug), fontsize=10)
    axes[idx].axis('off')

fig.tight_layout()
out_path = PLOTS_ROOT / 'trajectory_overlay_panel.png'
fig.savefig(out_path, dpi=220, bbox_inches='tight')
plt.show()
print('Saved:', out_path)
"""
    ),
]

nb_path.write_text(json.dumps(nb, indent=1))
print(f"updated {nb_path}")
