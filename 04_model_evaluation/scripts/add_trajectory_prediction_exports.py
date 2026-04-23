import json
from pathlib import Path


NOTEBOOK_DIR = Path("/home/basudeo/Documents/Thesis/04_model_evaluation/notebooks")


def patch_text(text: str, old: str, new: str) -> str:
    if old in text:
        return text.replace(old, new)
    return text


def update_notebook(path: Path):
    nb = json.loads(path.read_text())
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        old_eval_return = """    return {\n        'targets': targets,\n        'pred_future_xy': pred_future_xy,\n        'true_future_xy': true_future_xy,\n        'sample_ids': all_sample_ids,\n        'loss': total_loss / max(total_count, 1),\n        **metrics,\n    }\n"""
        new_eval_return = """    return {\n        'targets': targets,\n        'pred_future_xy': pred_future_xy,\n        'true_future_xy': true_future_xy,\n        'sample_ids': list(all_sample_ids),\n        'loss': total_loss / max(total_count, 1),\n        **metrics,\n    }\n"""
        if old_eval_return in src:
            src = src.replace(old_eval_return, new_eval_return)
            changed = True

        old_save_block = """    metrics['trajectory_overlay_plots'] = overlay_paths\n    metrics['mean_step_error_plot'] = step_error_path\n    metrics['status'] = 'saved'\n\n    metrics_path = result_dir / f'metrics_{timestamp}.json'\n"""
        new_save_block = """    prediction_export_path = result_dir / f'trajectory_predictions_{timestamp}.npz'\n    np.savez_compressed(\n        prediction_export_path,\n        sample_ids=np.asarray(test_eval['sample_ids'], dtype=str),\n        pred_future_xy=test_eval['pred_future_xy'],\n        true_future_xy=test_eval['true_future_xy'],\n    )\n    latest_prediction_export_path = result_dir / 'latest_trajectory_predictions.npz'\n    np.savez_compressed(\n        latest_prediction_export_path,\n        sample_ids=np.asarray(test_eval['sample_ids'], dtype=str),\n        pred_future_xy=test_eval['pred_future_xy'],\n        true_future_xy=test_eval['true_future_xy'],\n    )\n\n    metrics['trajectory_overlay_plots'] = overlay_paths\n    metrics['mean_step_error_plot'] = step_error_path\n    metrics['trajectory_prediction_file'] = str(latest_prediction_export_path)\n    metrics['status'] = 'saved'\n\n    metrics_path = result_dir / f'metrics_{timestamp}.json'\n"""
        if old_save_block in src:
            src = src.replace(old_save_block, new_save_block)
            changed = True

        old_cv_block = """metrics['trajectory_overlay_plots'] = overlay_paths\nmetrics['mean_step_error_plot'] = step_error_path\nmetrics['status'] = 'saved'\n\nmetrics_path = result_dir / f'metrics_{TIMESTAMP}.json'\n"""
        new_cv_block = """prediction_export_path = result_dir / f'trajectory_predictions_{TIMESTAMP}.npz'\nnp.savez_compressed(\n    prediction_export_path,\n    sample_ids=np.asarray([sample_table[idx]['sample_id'] for idx in test_indices], dtype=str),\n    pred_future_xy=pred_future_xy,\n    true_future_xy=true_future_xy,\n)\nlatest_prediction_export_path = result_dir / 'latest_trajectory_predictions.npz'\nnp.savez_compressed(\n    latest_prediction_export_path,\n    sample_ids=np.asarray([sample_table[idx]['sample_id'] for idx in test_indices], dtype=str),\n    pred_future_xy=pred_future_xy,\n    true_future_xy=true_future_xy,\n)\n\nmetrics['trajectory_overlay_plots'] = overlay_paths\nmetrics['mean_step_error_plot'] = step_error_path\nmetrics['trajectory_prediction_file'] = str(latest_prediction_export_path)\nmetrics['status'] = 'saved'\n\nmetrics_path = result_dir / f'metrics_{TIMESTAMP}.json'\n"""
        if old_cv_block in src:
            src = src.replace(old_cv_block, new_cv_block)
            changed = True

        cell["source"] = [line + "\n" for line in src.splitlines()]

    if changed:
        path.write_text(json.dumps(nb, indent=1))
        print(f"updated {path}")
    else:
        print(f"no change {path}")


for name in [
    "10_cv_baseline.ipynb",
    "20_cnn_lstm.ipynb",
    "40_cnn_gnn_lstm.ipynb",
    "50_cnn_gnn_transformer.ipynb",
    "60_cnn_gnn_lstm_transformer.ipynb",
]:
    update_notebook(NOTEBOOK_DIR / name)
