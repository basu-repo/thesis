"""Persist 09 live trajectory evaluation runs in a restart-safe format."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


THESIS_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EVAL_ROOT / "results"
WEIGHTS_ROOT = EVAL_ROOT / "model_weights"
COMPARISON_ROOT = EVAL_ROOT / "comparison_exports"

TRACKING_PREFIX = "Tracking status:"
ARRIVAL_PREFIX = "Arrival triggered:"
MODEL_NODE_LABELS = {
    "trajectory_model_husky_driver_2": "husky_2",
}


@dataclass
class TrackingPoint:
    ros_time: str
    x: float
    y: float
    goal_x: float
    goal_y: float
    remaining: float
    state: str
    front: float


def _extract_between(text: str, start: str, end: str) -> str:
    start_idx = text.index(start) + len(start)
    end_idx = text.index(end, start_idx)
    return text[start_idx:end_idx]


def _parse_float_pair(text: str) -> tuple[float, float]:
    left, right = [part.strip() for part in text.split(",", 1)]
    return float(left), float(right)


def _parse_tracking_line(line: str) -> tuple[str, TrackingPoint] | None:
    if TRACKING_PREFIX not in line:
        return None
    try:
        ros_time = _extract_between(line, "[INFO] [", "]")
        node_name = _extract_between(line, "] [", "]:")
        label = MODEL_NODE_LABELS.get(node_name)
        if label is None:
            return None
        pose_text = _extract_between(line, "pose=(", ") goal=(")
        goal_text = _extract_between(line, "goal=(", ") remaining=")
        remaining_text = _extract_between(line, "remaining=", " state=")
        state_text = _extract_between(line, " state=", " front=")
        front_text = _extract_between(line, " front=", " pred_local=")
        x, y = _parse_float_pair(pose_text)
        goal_x, goal_y = _parse_float_pair(goal_text)
        return label, TrackingPoint(
            ros_time=ros_time,
            x=x,
            y=y,
            goal_x=goal_x,
            goal_y=goal_y,
            remaining=float(remaining_text),
            state=state_text.strip(),
            front=float(front_text),
        )
    except Exception:
        return None


def _compute_path_length(points: list[TrackingPoint]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(points)):
        total += math.hypot(points[idx].x - points[idx - 1].x, points[idx].y - points[idx - 1].y)
    return total


def _write_trajectory_csv(path: Path, points: list[TrackingPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "ros_time", "x", "y", "goal_x", "goal_y", "remaining", "state", "front"])
        for idx, point in enumerate(points):
            writer.writerow(
                [
                    idx,
                    point.ros_time,
                    f"{point.x:.6f}",
                    f"{point.y:.6f}",
                    f"{point.goal_x:.6f}",
                    f"{point.goal_y:.6f}",
                    f"{point.remaining:.6f}",
                    point.state,
                    f"{point.front:.6f}",
                ]
            )


def _save_trajectory_plot(path: Path, points: list[TrackingPoint]) -> None:
    if not points:
        raise RuntimeError("No trajectory points found for plotting.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    ax.plot(xs, ys, color="tab:blue", linewidth=2.2, label="UGV trajectory")
    ax.scatter(xs[0], ys[0], color="tab:blue", marker="o", s=45, label="UGV start")
    ax.scatter(xs[-1], ys[-1], color="tab:blue", marker="s", s=45, label="UGV end")
    ax.scatter(points[-1].goal_x, points[-1].goal_y, color="tab:red", marker="*", s=160, label="Goal")
    ax.set_title("09 Live Trajectory Model Run")
    ax.set_xlabel("World X Coordinate (m)")
    ax.set_ylabel("World Y Coordinate (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _update_summary_files(row: dict[str, Any]) -> tuple[Path, Path]:
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = COMPARISON_ROOT / "live_model_summary_latest.json"
    csv_path = COMPARISON_ROOT / "live_model_summary_latest.csv"

    rows: list[dict[str, Any]] = []
    if json_path.exists():
        try:
            rows = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            rows = []

    rows = [existing for existing in rows if existing.get("run_id") != row.get("run_id")]
    rows.append(row)
    rows.sort(key=lambda item: str(item.get("run_id", "")))
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames: list[str] = []
    for existing in rows:
        for key in existing.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing in rows:
            writer.writerow(existing)
    return json_path, csv_path


def export_live_run_bundle(
    *,
    run_log_path: Path,
    checkpoint_path: Path,
    selected_info: dict[str, Any] | None,
    run_start_dt: dt.datetime,
    run_end_dt: dt.datetime,
    world_name: str,
    runner_args: dict[str, Any],
    start_goals: dict[str, dict[str, tuple[float, float, float]]],
    model_slug: str,
) -> dict[str, Any]:
    run_id = run_start_dt.strftime("%Y%m%d_%H%M%S")

    result_root = RESULTS_ROOT / model_slug
    plots_dir = result_root / "plots"
    trajectories_dir = result_root / "trajectories"
    logs_dir = result_root / "logs"
    weights_dir = WEIGHTS_ROOT / model_slug
    for directory in (plots_dir, trajectories_dir, logs_dir, weights_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied_log_path = logs_dir / f"{run_id}.log"
    latest_log_path = logs_dir / "latest.log"
    shutil.copy2(run_log_path, copied_log_path)
    shutil.copy2(run_log_path, latest_log_path)

    saved_checkpoint_path = weights_dir / f"{model_slug}_{run_id}.pt"
    latest_checkpoint_path = weights_dir / "latest.pt"
    shutil.copy2(checkpoint_path, saved_checkpoint_path)
    shutil.copy2(checkpoint_path, latest_checkpoint_path)

    points: list[TrackingPoint] = []
    arrival_events = 0
    for line in copied_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_tracking_line(line)
        if parsed is not None:
            _label, point = parsed
            points.append(point)
        if ARRIVAL_PREFIX in line:
            arrival_events += 1

    if not points:
        raise RuntimeError(f"No trajectory tracking samples found in {copied_log_path}")

    trajectory_csv_path = trajectories_dir / f"{run_id}_trajectory_points.csv"
    latest_trajectory_csv_path = trajectories_dir / "latest_trajectory_points.csv"
    _write_trajectory_csv(trajectory_csv_path, points)
    shutil.copy2(trajectory_csv_path, latest_trajectory_csv_path)

    trajectory_plot_path = plots_dir / f"{run_id}_trajectory_plot.png"
    latest_trajectory_plot_path = plots_dir / "latest_trajectory_plot.png"
    _save_trajectory_plot(trajectory_plot_path, points)
    shutil.copy2(trajectory_plot_path, latest_trajectory_plot_path)

    initial_remaining = points[0].remaining
    final_remaining = points[-1].remaining
    best_remaining = min(point.remaining for point in points)
    reached_goal = arrival_events > 0 or final_remaining <= 1.5
    path_length = _compute_path_length(points)

    row = {
        "run_id": run_id,
        "model_slug": model_slug,
        "status": "saved",
        "selection_task": None if selected_info is None else selected_info.get("selection_task"),
        "selected_model": None if selected_info is None else selected_info.get("selected_model"),
        "source_checkpoint": str(checkpoint_path),
        "saved_checkpoint": str(saved_checkpoint_path),
        "run_log_path": str(copied_log_path),
        "trajectory_plot": str(trajectory_plot_path),
        "trajectory_csv": str(trajectory_csv_path),
        "start_time": run_start_dt.isoformat(timespec="seconds"),
        "stop_time": run_end_dt.isoformat(timespec="seconds"),
        "run_duration_seconds": round((run_end_dt - run_start_dt).total_seconds(), 3),
        "world_name": world_name,
        "headless": bool(runner_args.get("headless")),
        "no_rviz": bool(runner_args.get("no_rviz")),
        "no_camera": bool(runner_args.get("no_camera")),
        "trajectory_points": len(points),
        "initial_remaining": round(initial_remaining, 6),
        "final_remaining": round(final_remaining, 6),
        "best_remaining": round(best_remaining, 6),
        "progress_delta": round(initial_remaining - final_remaining, 6),
        "path_length": round(path_length, 6),
        "reached_goal": reached_goal,
        "final_state": points[-1].state,
        "final_x": round(points[-1].x, 6),
        "final_y": round(points[-1].y, 6),
        "goal_x": round(points[-1].goal_x, 6),
        "goal_y": round(points[-1].goal_y, 6),
        "arrival_events": arrival_events,
        "start_goals": json.dumps(start_goals),
    }

    metrics_payload = {
        **row,
        "selected_info": selected_info,
        "runner_args": runner_args,
    }
    metrics_path = result_root / f"metrics_{run_id}.json"
    latest_metrics_path = result_root / "latest_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    latest_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    summary_json_path, summary_csv_path = _update_summary_files(row)
    return {
        "run_id": run_id,
        "model_slug": model_slug,
        "metrics_path": metrics_path,
        "trajectory_plot_path": trajectory_plot_path,
        "trajectory_csv_path": trajectory_csv_path,
        "saved_checkpoint_path": saved_checkpoint_path,
        "summary_json_path": summary_json_path,
        "summary_csv_path": summary_csv_path,
        "copied_log_path": copied_log_path,
    }
