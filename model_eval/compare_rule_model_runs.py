"""Compare rule-based and learned-model UGV trajectories from run logs.

The script reads the Tracking status lines written by cooperative_sim and
model_eval, then writes a shared trajectory plot and timing summary.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


THESIS_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = THESIS_ROOT / "dataset" / "logs"
OUTPUT_ROOT = THESIS_ROOT / "model_eval" / "comparison_exports" / "rule_vs_model"

TRACKING_RE = re.compile(
    r"\[INFO\] \[(?P<ros_time>[^\]]+)\] \[(?P<node>[^\]]+)\]: "
    r"Tracking status: pose=\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)"
    r"(?:\s+z=(?P<z>-?\d+(?:\.\d+)?))?\s+"
    r"goal=\((?P<goal_x>-?\d+(?:\.\d+)?),\s*(?P<goal_y>-?\d+(?:\.\d+)?)\)\s+"
    r"remaining=(?P<remaining>-?\d+(?:\.\d+)?)\s+state=(?P<state>[^\s]+)"
)
ARRIVAL_RE = re.compile(
    r"\[INFO\] \[(?P<ros_time>[^\]]+)\] \[(?P<node>[^\]]+)\]: "
    r"Arrival triggered: "
    r"(?:pose=\((?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\)\s+"
    r"(?:z=(?P<z>-?\d+(?:\.\d+)?)\s+)?"
    r"goal=\((?P<goal_x>-?\d+(?:\.\d+)?),\s*(?P<goal_y>-?\d+(?:\.\d+)?)\)\s+)?"
    r"remaining=(?P<remaining>-?\d+(?:\.\d+)?)"
)


@dataclass
class TrajectoryPoint:
    sample_idx: int
    ros_time: str
    x: float
    y: float
    goal_x: float
    goal_y: float
    remaining: float
    state: str


def latest_log(pattern: str) -> Path:
    candidates = sorted(LOG_ROOT.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No logs found for pattern {LOG_ROOT / pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resource_summary_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_resource_summary.json")


def load_resource_summary(log_path: Path) -> dict[str, Any]:
    summary_path = resource_summary_path(log_path)
    if not summary_path.exists():
        return {
            "finish_reason": "unknown",
            "wall_duration_s": None,
            "sim_duration_s": None,
            "summary_path": str(summary_path),
        }
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "finish_reason": f"unreadable: {exc}",
            "wall_duration_s": None,
            "sim_duration_s": None,
            "summary_path": str(summary_path),
        }
    data["summary_path"] = str(summary_path)
    return data


def parse_trajectory(log_path: Path) -> list[TrajectoryPoint]:
    points: list[TrajectoryPoint] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRACKING_RE.search(line)
        if match is not None:
            points.append(
                TrajectoryPoint(
                    sample_idx=len(points),
                    ros_time=match.group("ros_time"),
                    x=float(match.group("x")),
                    y=float(match.group("y")),
                    goal_x=float(match.group("goal_x")),
                    goal_y=float(match.group("goal_y")),
                    remaining=float(match.group("remaining")),
                    state=match.group("state"),
                )
            )
            continue
        arrival_match = ARRIVAL_RE.search(line)
        if arrival_match is not None and arrival_match.group("x") is not None:
            points.append(
                TrajectoryPoint(
                    sample_idx=len(points),
                    ros_time=arrival_match.group("ros_time"),
                    x=float(arrival_match.group("x")),
                    y=float(arrival_match.group("y")),
                    goal_x=float(arrival_match.group("goal_x")),
                    goal_y=float(arrival_match.group("goal_y")),
                    remaining=float(arrival_match.group("remaining")),
                    state="reached",
                )
            )
    if not points:
        raise RuntimeError(f"No Tracking status trajectory points found in {log_path}")
    return points


def path_length(points: list[TrajectoryPoint]) -> float:
    return sum(
        math.hypot(points[idx].x - points[idx - 1].x, points[idx].y - points[idx - 1].y)
        for idx in range(1, len(points))
    )


def run_summary(label: str, log_path: Path, points: list[TrajectoryPoint]) -> dict[str, Any]:
    resource = load_resource_summary(log_path)
    wall_duration_s = resource.get("wall_duration_s")
    sim_duration_s = resource.get("sim_duration_s")
    real_time_factor = None
    if wall_duration_s not in (None, 0) and sim_duration_s is not None:
        real_time_factor = float(sim_duration_s) / float(wall_duration_s)
    return {
        "label": label,
        "log_path": str(log_path),
        "resource_summary_path": resource.get("summary_path"),
        "finish_reason": resource.get("finish_reason"),
        "wall_duration_s": wall_duration_s,
        "sim_duration_s": sim_duration_s,
        "real_time_factor": None if real_time_factor is None else round(real_time_factor, 6),
        "trajectory_samples": len(points),
        "start_x": points[0].x,
        "start_y": points[0].y,
        "end_x": points[-1].x,
        "end_y": points[-1].y,
        "goal_x": points[-1].goal_x,
        "goal_y": points[-1].goal_y,
        "initial_remaining": points[0].remaining,
        "final_remaining": points[-1].remaining,
        "best_remaining": min(point.remaining for point in points),
        "path_length_m": path_length(points),
        "final_state": points[-1].state,
    }


def write_points_csv(path: Path, label: str, points: list[TrajectoryPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["run_type", "sample_idx", "ros_time", "x", "y", "goal_x", "goal_y", "remaining", "state"])
        for point in points:
            writer.writerow(
                [
                    label,
                    point.sample_idx,
                    point.ros_time,
                    f"{point.x:.6f}",
                    f"{point.y:.6f}",
                    f"{point.goal_x:.6f}",
                    f"{point.goal_y:.6f}",
                    f"{point.remaining:.6f}",
                    point.state,
                ]
            )


def fmt_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "n/a"


def total_seconds_label(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f} s"
    except Exception:
        return "n/a"


def save_plot(path: Path, rule_points: list[TrajectoryPoint], model_points: list[TrajectoryPoint], summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    series = [
        ("Rule-based", rule_points, "#1f77b4"),
        ("Learned model", model_points, "#d62728"),
    ]
    for label, points, color in series:
        ax.plot([p.x for p in points], [p.y for p in points], color=color, linewidth=2.0, label=label)
        ax.scatter(points[0].x, points[0].y, color=color, marker="o", s=50)
        ax.scatter(points[-1].x, points[-1].y, color=color, marker="s", s=50)

    goal_x = model_points[-1].goal_x
    goal_y = model_points[-1].goal_y
    ax.scatter(goal_x, goal_y, color="black", marker="*", s=170, label="Goal")

    rule_summary = next(item for item in summaries if item["label"] == "rule_based")
    model_summary = next(item for item in summaries if item["label"] == "model_based")
    subtitle = (
        f"Rule-based total time: {total_seconds_label(rule_summary['wall_duration_s'])} | "
        f"Model-based total time: {total_seconds_label(model_summary['wall_duration_s'])}"
    )
    ax.set_title("UGV Trajectory: Rule-Based vs Learned Model\n" + subtitle, fontsize=11)
    timing_text = (
        "Total time\n"
        f"Rule-based: {total_seconds_label(rule_summary['wall_duration_s'])}\n"
        f"Model-based: {total_seconds_label(model_summary['wall_duration_s'])}"
    )
    ax.text(
        0.02,
        0.02,
        timing_text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    ax.set_xlabel("World X Coordinate (m)")
    ax.set_ylabel("World Y Coordinate (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def elapsed_wall_axis(points: list[TrajectoryPoint], total_wall_s: Any) -> list[float]:
    if len(points) <= 1:
        return [0.0 for _ in points]
    try:
        total = float(total_wall_s)
    except Exception:
        total = float(len(points) - 1)
    return [idx * total / float(len(points) - 1) for idx in range(len(points))]


def save_remaining_distance_plot(
    path: Path,
    rule_points: list[TrajectoryPoint],
    model_points: list[TrajectoryPoint],
    summaries: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rule_summary = next(item for item in summaries if item["label"] == "rule_based")
    model_summary = next(item for item in summaries if item["label"] == "model_based")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    rule_t = elapsed_wall_axis(rule_points, rule_summary["wall_duration_s"])
    model_t = elapsed_wall_axis(model_points, model_summary["wall_duration_s"])

    ax.plot(rule_t, [p.remaining for p in rule_points], color="#1f77b4", linewidth=2.0, label="Rule-based")
    ax.plot(model_t, [p.remaining for p in model_points], color="#d62728", linewidth=2.0, label="Learned model")
    ax.scatter(rule_t[-1], rule_points[-1].remaining, color="#1f77b4", marker="s", s=50)
    ax.scatter(model_t[-1], model_points[-1].remaining, color="#d62728", marker="s", s=50)

    ax.set_title("Remaining Distance to Goal")
    ax.set_xlabel("Elapsed Wall Time (s)")
    ax.set_ylabel("Remaining Distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_remaining_by_step_plot(
    path: Path,
    rule_points: list[TrajectoryPoint],
    model_points: list[TrajectoryPoint],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot([p.sample_idx for p in rule_points], [p.remaining for p in rule_points], color="#1f77b4", linewidth=2.0, label="Rule-based")
    ax.plot([p.sample_idx for p in model_points], [p.remaining for p in model_points], color="#d62728", linewidth=2.0, label="Learned model")
    ax.scatter(rule_points[-1].sample_idx, rule_points[-1].remaining, color="#1f77b4", marker="s", s=50)
    ax.scatter(model_points[-1].sample_idx, model_points[-1].remaining, color="#d62728", marker="s", s=50)
    ax.set_title("Remaining Distance by Logged Step")
    ax.set_xlabel("Logged Trajectory Sample Index")
    ax.set_ylabel("Remaining Distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_summary_bar_plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Rule-based", "Learned model"]
    rule_summary = next(item for item in summaries if item["label"] == "rule_based")
    model_summary = next(item for item in summaries if item["label"] == "model_based")
    rows = [rule_summary, model_summary]
    colors = ["#1f77b4", "#d62728"]

    metrics = [
        ("Total time (s)", "wall_duration_s"),
        ("Logged samples", "trajectory_samples"),
        ("Path length (m)", "path_length_m"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2))
    for ax, (title, key) in zip(axes, metrics):
        values = [float(row[key]) for row in rows]
        bars = ax.bar(labels, values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            if key == "trajectory_samples":
                label = f"{int(value)}"
            else:
                label = f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Rule-Based vs Learned-Model Run Summary", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_path_length_bar_plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rule_summary = next(item for item in summaries if item["label"] == "rule_based")
    model_summary = next(item for item in summaries if item["label"] == "model_based")
    labels = ["Rule-based", "Learned model"]
    values = [float(rule_summary["path_length_m"]), float(model_summary["path_length_m"])]
    colors = ["#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_title("Total UGV Path Length")
    ax.set_ylabel("Path Length (m)")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f} m", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-log", type=Path, default=None, help="Rule-based run log. Defaults to latest rule_based_run_*.log.")
    parser.add_argument("--model-log", type=Path, default=None, help="Model-eval run log. Defaults to latest trajectory_model_eval_09_*.log.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="Directory for comparison plot and summary files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rule_log = args.rule_log or latest_log("rule_based_run_*.log")
    model_log = args.model_log or latest_log("trajectory_model_eval_09_*.log")

    rule_points = parse_trajectory(rule_log)
    model_points = parse_trajectory(model_log)
    summaries = [
        run_summary("rule_based", rule_log, rule_points),
        run_summary("model_based", model_log, model_points),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = args.output_dir / "rule_vs_model_trajectory.png"
    timestamped_plot_path = args.output_dir / f"rule_vs_model_trajectory_{output_id}.png"
    remaining_plot_path = args.output_dir / "rule_vs_model_remaining_distance.png"
    timestamped_remaining_plot_path = args.output_dir / f"rule_vs_model_remaining_distance_{output_id}.png"
    remaining_step_plot_path = args.output_dir / "rule_vs_model_remaining_by_step.png"
    timestamped_remaining_step_plot_path = args.output_dir / f"rule_vs_model_remaining_by_step_{output_id}.png"
    summary_bar_plot_path = args.output_dir / "rule_vs_model_summary_bars.png"
    timestamped_summary_bar_plot_path = args.output_dir / f"rule_vs_model_summary_bars_{output_id}.png"
    path_length_plot_path = args.output_dir / "rule_vs_model_path_length.png"
    timestamped_path_length_plot_path = args.output_dir / f"rule_vs_model_path_length_{output_id}.png"
    summary_json = args.output_dir / "rule_vs_model_summary.json"
    timestamped_summary_json = args.output_dir / f"rule_vs_model_summary_{output_id}.json"
    summary_csv = args.output_dir / "rule_vs_model_summary.csv"
    points_csv = args.output_dir / "rule_vs_model_trajectory_points.csv"

    if points_csv.exists():
        points_csv.unlink()
    write_points_csv(points_csv, "rule_based", rule_points)
    write_points_csv(points_csv, "model_based", model_points)
    save_plot(plot_path, rule_points, model_points, summaries)
    save_plot(timestamped_plot_path, rule_points, model_points, summaries)
    save_remaining_distance_plot(remaining_plot_path, rule_points, model_points, summaries)
    save_remaining_distance_plot(timestamped_remaining_plot_path, rule_points, model_points, summaries)
    save_remaining_by_step_plot(remaining_step_plot_path, rule_points, model_points)
    save_remaining_by_step_plot(timestamped_remaining_step_plot_path, rule_points, model_points)
    save_summary_bar_plot(summary_bar_plot_path, summaries)
    save_summary_bar_plot(timestamped_summary_bar_plot_path, summaries)
    save_path_length_bar_plot(path_length_plot_path, summaries)
    save_path_length_bar_plot(timestamped_path_length_plot_path, summaries)
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    timestamped_summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv, summaries)

    print(f"Rule log: {rule_log}")
    print(f"Model log: {model_log}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved timestamped plot: {timestamped_plot_path}")
    print(f"Saved remaining-distance plot: {remaining_plot_path}")
    print(f"Saved timestamped remaining-distance plot: {timestamped_remaining_plot_path}")
    print(f"Saved remaining-by-step plot: {remaining_step_plot_path}")
    print(f"Saved summary bar plot: {summary_bar_plot_path}")
    print(f"Saved path-length plot: {path_length_plot_path}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved timestamped summary JSON: {timestamped_summary_json}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved trajectory CSV: {points_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
