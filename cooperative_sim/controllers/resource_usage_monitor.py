"""Record runtime/resource usage between first sim advance and UGV goal reach."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
from pathlib import Path

import psutil
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


class RunResourceMonitor(Node):
    """Track wall/sim time and sampled resource usage for the active mission window."""

    def __init__(
        self,
        *,
        node_name: str,
        controller_state_topic: str,
        clock_topic: str,
        summary_path: str | Path,
        samples_path: str | Path,
        log_fn=None,
        sample_period: float = 1.0,
    ):
        super().__init__(node_name)
        self.summary_path = Path(summary_path)
        self.samples_path = Path(samples_path)
        self.log_fn = log_fn
        self.sample_period = float(sample_period)

        self.active = False
        self.completed = False
        self.finish_reason = "not_started"
        self.initial_clock_ns = None
        self.last_clock_ns = None
        self.sim_start_ns = None
        self.sim_end_ns = None
        self.wall_start_mono = None
        self.wall_end_mono = None
        self.wall_start_iso = None
        self.wall_end_iso = None
        self.samples = []
        self.tracked_processes = []

        self.create_subscription(String, controller_state_topic, self.state_cb, 10)
        self.create_subscription(Clock, clock_topic, self.clock_cb, 10)
        self.timer = self.create_timer(self.sample_period, self.sample)

        psutil.cpu_percent(interval=None)

    def set_tracked_processes(self, processes):
        tracked = [psutil.Process(os.getpid())]
        for proc in processes:
            if proc is None:
                continue
            pid = getattr(proc, "pid", None)
            if pid is None:
                continue
            try:
                tracked.append(psutil.Process(int(pid)))
            except (psutil.Error, ValueError):
                continue
        unique = {}
        for proc in tracked:
            unique[proc.pid] = proc
        self.tracked_processes = list(unique.values())
        for proc in self.tracked_processes:
            try:
                proc.cpu_percent(interval=None)
            except psutil.Error:
                continue

    def _log(self, message: str):
        if self.log_fn is not None:
            self.log_fn(message)
        else:
            self.get_logger().info(message)

    def clock_cb(self, msg: Clock):
        clock_ns = (int(msg.clock.sec) * 1_000_000_000) + int(msg.clock.nanosec)
        if self.initial_clock_ns is None:
            self.initial_clock_ns = clock_ns
            self.last_clock_ns = clock_ns
            return
        if not self.active and clock_ns > self.initial_clock_ns:
            now = time.monotonic()
            self.active = True
            self.finish_reason = "running"
            self.wall_start_mono = now
            self.wall_start_iso = dt.datetime.now().isoformat(timespec="seconds")
            self.sim_start_ns = self.initial_clock_ns
            self.last_clock_ns = clock_ns
            self._log(
                f"RESOURCE MONITOR start: wall={self.wall_start_iso} sim_time_s={clock_ns / 1e9:.3f}"
            )
            return
        if self.active and clock_ns >= (self.last_clock_ns or clock_ns):
            self.last_clock_ns = clock_ns

    def state_cb(self, msg: String):
        state = (msg.data or "").strip().lower()
        if state == "reached":
            self.finish("goal_reached")

    def _collect_sample(self) -> dict:
        cpu_percent = float(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()

        tracked_cpu = 0.0
        tracked_rss_mb = 0.0
        alive = []
        for proc in self.tracked_processes:
            try:
                tracked_cpu += float(proc.cpu_percent(interval=None))
                tracked_rss_mb += float(proc.memory_info().rss) / (1024.0 * 1024.0)
                alive.append(proc)
            except psutil.Error:
                continue
        self.tracked_processes = alive

        wall_elapsed_s = 0.0 if self.wall_start_mono is None else max(0.0, time.monotonic() - self.wall_start_mono)
        sim_elapsed_s = 0.0
        if self.sim_start_ns is not None and self.last_clock_ns is not None:
            sim_elapsed_s = max(0.0, (self.last_clock_ns - self.sim_start_ns) / 1e9)

        return {
            "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
            "wall_elapsed_s": round(wall_elapsed_s, 3),
            "sim_elapsed_s": round(sim_elapsed_s, 3),
            "system_cpu_percent": round(cpu_percent, 3),
            "system_memory_percent": round(float(mem.percent), 3),
            "tracked_cpu_percent": round(tracked_cpu, 3),
            "tracked_rss_mb": round(tracked_rss_mb, 3),
            "tracked_process_count": len(self.tracked_processes),
        }

    def sample(self):
        if not self.active or self.completed:
            return
        self.samples.append(self._collect_sample())

    def finish(self, reason: str):
        if self.completed:
            return
        self.completed = True
        self.finish_reason = reason
        self.wall_end_mono = time.monotonic()
        self.wall_end_iso = dt.datetime.now().isoformat(timespec="seconds")
        if self.last_clock_ns is not None:
            self.sim_end_ns = self.last_clock_ns
        if self.active:
            self.samples.append(self._collect_sample())
        summary = self._build_summary()
        self._write_outputs(summary)
        self._log(
            "RESOURCE MONITOR done: "
            f"reason={summary['finish_reason']} "
            f"wall_duration_s={summary['wall_duration_s']:.3f} "
            f"sim_duration_s={summary['sim_duration_s']:.3f} "
            f"peak_system_cpu={summary['peak_system_cpu_percent']:.2f}% "
            f"peak_tracked_rss_mb={summary['peak_tracked_rss_mb']:.2f}"
        )

    def _build_summary(self) -> dict:
        def avg(key: str) -> float:
            if not self.samples:
                return 0.0
            return float(sum(sample[key] for sample in self.samples) / len(self.samples))

        def peak(key: str) -> float:
            if not self.samples:
                return 0.0
            return float(max(sample[key] for sample in self.samples))

        wall_duration_s = 0.0
        if self.wall_start_mono is not None and self.wall_end_mono is not None:
            wall_duration_s = max(0.0, self.wall_end_mono - self.wall_start_mono)
        sim_duration_s = 0.0
        if self.sim_start_ns is not None and self.sim_end_ns is not None:
            sim_duration_s = max(0.0, (self.sim_end_ns - self.sim_start_ns) / 1e9)

        return {
            "finish_reason": self.finish_reason,
            "wall_start_iso": self.wall_start_iso,
            "wall_end_iso": self.wall_end_iso,
            "wall_duration_s": round(wall_duration_s, 3),
            "sim_duration_s": round(sim_duration_s, 3),
            "sample_count": len(self.samples),
            "avg_system_cpu_percent": round(avg("system_cpu_percent"), 3),
            "peak_system_cpu_percent": round(peak("system_cpu_percent"), 3),
            "avg_system_memory_percent": round(avg("system_memory_percent"), 3),
            "peak_system_memory_percent": round(peak("system_memory_percent"), 3),
            "avg_tracked_cpu_percent": round(avg("tracked_cpu_percent"), 3),
            "peak_tracked_cpu_percent": round(peak("tracked_cpu_percent"), 3),
            "avg_tracked_rss_mb": round(avg("tracked_rss_mb"), 3),
            "peak_tracked_rss_mb": round(peak("tracked_rss_mb"), 3),
            "tracked_process_count_last": self.samples[-1]["tracked_process_count"] if self.samples else 0,
            "summary_path": str(self.summary_path),
            "samples_path": str(self.samples_path),
        }

    def _write_outputs(self, summary: dict):
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2))
        with self.samples_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp_iso",
                    "wall_elapsed_s",
                    "sim_elapsed_s",
                    "system_cpu_percent",
                    "system_memory_percent",
                    "tracked_cpu_percent",
                    "tracked_rss_mb",
                    "tracked_process_count",
                ],
            )
            writer.writeheader()
            writer.writerows(self.samples)

    def destroy_node(self):
        if not self.completed:
            self.finish("stopped_before_goal" if self.active else "never_started")
        return super().destroy_node()
