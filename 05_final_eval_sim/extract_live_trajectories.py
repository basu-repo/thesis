"""Extract and plot live Husky trajectories from a saved AI run log."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


THESIS_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = THESIS_ROOT / "03_dataset" / "logs"

TRACKING_RE = re.compile(
    r"^\[INFO\]\s+\[(?P<ros_time>[^\]]+)\]\s+\[(?P<node>[^\]]+)\]: "
    r"Tracking status: pose=\((?P<x>-?\d+(?:\.\d+)?), (?P<y>-?\d+(?:\.\d+)?)\)"
)
GOALS_RE = re.compile(
    r"Controller goals \(world frame, stop offset applied\):\s+"
    r"husky_local=\((?P<hx>-?\d+(?:\.\d+)?), (?P<hy>-?\d+(?:\.\d+)?)\)"
    r"(?:,\s+husky_2=\((?P<h2x>-?\d+(?:\.\d+)?), (?P<h2y>-?\d+(?:\.\d+)?)\))?"
)

NODE_LABELS = {
    "ai_model_husky_driver_1": "husky_local",
    "model_husky_driver_2": "husky_2",
}
NODE_COLORS = {
    "husky_local": "tab:red",
    "husky_2": "tab:blue",
}
DISPLAY_LABELS = {
    "husky_local": "Target robot",
    "husky_2": "Secondary robot",
}


@dataclass
class TrajectoryPoint:
    wall_time: str
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and plot Husky trajectories from a saved live-run log.")
    parser.add_argument(
        "--log",
        default=None,
        help="Path to a specific ai_live_run_*.log file. Defaults to the newest multi-agent log, else newest log.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output PNG path. Defaults next to the log as <log_stem>_trajectory_plot.png.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path. Defaults next to the log as <log_stem>_trajectory_points.csv.",
    )
    return parser.parse_args()


def choose_default_log() -> Path:
    candidates = sorted(LOG_ROOT.glob("ai_live_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No ai_live_run_*.log files found in {LOG_ROOT}")

    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "model_husky_driver_2" in text:
            return path
    return candidates[0]


def parse_log(path: Path) -> tuple[dict[str, list[TrajectoryPoint]], dict[str, tuple[float, float]]]:
    trajectories: dict[str, list[TrajectoryPoint]] = {"husky_local": [], "husky_2": []}
    goals: dict[str, tuple[float, float]] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        goal_match = GOALS_RE.search(line)
        if goal_match:
            goals["husky_local"] = (float(goal_match.group("hx")), float(goal_match.group("hy")))
            if goal_match.group("h2x") and goal_match.group("h2y"):
                goals["husky_2"] = (float(goal_match.group("h2x")), float(goal_match.group("h2y")))

        match = TRACKING_RE.search(line)
        if not match:
            continue
        node_name = match.group("node")
        label = NODE_LABELS.get(node_name)
        if label is None:
            continue
        trajectories[label].append(
            TrajectoryPoint(
                wall_time=match.group("ros_time"),
                x=float(match.group("x")),
                y=float(match.group("y")),
            )
        )

    return trajectories, goals


def write_csv(path: Path, trajectories: dict[str, list[TrajectoryPoint]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "sample_idx", "wall_time", "x", "y"])
        for agent, points in trajectories.items():
            for idx, point in enumerate(points):
                writer.writerow([agent, idx, point.wall_time, f"{point.x:.6f}", f"{point.y:.6f}"])


def save_plot(path: Path, log_path: Path, trajectories: dict[str, list[TrajectoryPoint]], goals: dict[str, tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    plotted_any = False
    for agent in ("husky_local", "husky_2"):
        points = trajectories.get(agent) or []
        if not points:
            continue
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        color = NODE_COLORS[agent]
        label = DISPLAY_LABELS.get(agent, agent)
        ax.plot(xs, ys, color=color, linewidth=2.2, label=f"{label} trajectory")
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=45, label=f"{label} start")
        ax.scatter(xs[-1], ys[-1], color=color, marker="s", s=45, label=f"{label} end")
        plotted_any = True

        goal_xy = goals.get(agent)
        if goal_xy is not None:
            ax.scatter(goal_xy[0], goal_xy[1], color=color, marker="*", s=150, alpha=0.9, label=f"{label} goal")

    if not plotted_any:
        raise RuntimeError(f"No husky trajectories found in {log_path}")

    ax.set_title("Recorded Multi-Robot Trajectories in the Simulated Environment")
    ax.set_xlabel("World X Coordinate (m)")
    ax.set_ylabel("World Y Coordinate (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve() if args.log else choose_default_log()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    trajectories, goals = parse_log(log_path)
    out_png = Path(args.out).expanduser().resolve() if args.out else log_path.with_name(f"{log_path.stem}_trajectory_plot.png")
    out_csv = Path(args.csv).expanduser().resolve() if args.csv else log_path.with_name(f"{log_path.stem}_trajectory_points.csv")

    write_csv(out_csv, trajectories)
    save_plot(out_png, log_path, trajectories, goals)

    print(f"log: {log_path}")
    for agent in ("husky_local", "husky_2"):
        print(f"{agent}_points: {len(trajectories.get(agent, []))}")
    print(f"csv: {out_csv}")
    print(f"plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
