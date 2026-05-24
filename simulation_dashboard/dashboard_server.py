#!/usr/bin/env python3
"""Lightweight web dashboard for launching Thesis simulation stacks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import threading
import time
import contextlib
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import psutil
except ImportError:  # pragma: no cover - fallback when psutil is unavailable
    psutil = None


ROOT = Path(__file__).resolve().parent
THESIS_ROOT = ROOT.parent
STATIC_ROOT = ROOT / "static"
RULE_BASED_SCRIPT = THESIS_ROOT / "cooperative_sim" / "scripts" / "run_sim_model.py"
LIVE_MODEL_SCRIPT = THESIS_ROOT / "model_eval" / "trajectory_model_eval_sim.py"
DEFAULT_PYTHON = Path("/home/basudeo/miniconda3/envs/tct/bin/python")
if not DEFAULT_PYTHON.exists():
    DEFAULT_PYTHON = Path(sys.executable)

DEFAULT_GZWEB_URL = "http://127.0.0.1:8080"
GZWEB_DOC_URL = "https://github.com/Intelligent-Quads/iq_tutorials/blob/master/docs/gzweb_install.md"
AVAILABLE_MODELS = [
    "best",
    "cnn_lstm",
    "cnn_gnn_lstm",
    "cnn_gnn_transformer",
    "cnn_gnn_lstm_transformer",
]


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class RunSession:
    mode: str | None = None
    status: str = "idle"
    config: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    pid: int | None = None
    start_iso: str | None = None
    stop_iso: str | None = None
    start_monotonic: float | None = None
    stop_monotonic: float | None = None
    return_code: int | None = None
    log_items: list[dict[str, Any]] = field(default_factory=list)
    next_log_id: int = 1
    run_log_path: str | None = None
    resource_summary_path: str | None = None
    resource_samples_path: str | None = None
    log_doc_url: str = GZWEB_DOC_URL
    resource_samples: list[dict[str, Any]] = field(default_factory=list)


class SimulationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._session = RunSession()

    def metadata(self) -> dict[str, Any]:
        return {
            "default_python": str(DEFAULT_PYTHON),
            "default_gzweb_url": DEFAULT_GZWEB_URL,
            "gzweb_doc_url": GZWEB_DOC_URL,
            "available_models": AVAILABLE_MODELS,
            "modes": [
                {"id": "07", "label": "07 Rule-Based Baseline"},
                {"id": "09", "label": "09 Learned-Model Evaluation"},
            ],
        }

    def _append_log_unlocked(self, text: str) -> None:
        line = text.rstrip("\n")
        item = {"id": self._session.next_log_id, "text": line}
        self._session.log_items.append(item)
        self._session.next_log_id += 1

        if "Run log file:" in line:
            self._session.run_log_path = line.split("Run log file:", 1)[1].strip()
        elif "Resource summary file:" in line:
            self._session.resource_summary_path = line.split("Resource summary file:", 1)[1].strip()
        elif "Resource samples file:" in line:
            self._session.resource_samples_path = line.split("Resource samples file:", 1)[1].strip()

    def _append_log(self, text: str) -> None:
        with self._lock:
            self._append_log_unlocked(text)

    def _reader_loop(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append_log(line)
        return_code = process.wait()
        with self._lock:
            if self._session.process is process:
                self._session.return_code = int(return_code)
                self._session.stop_iso = now_iso()
                self._session.stop_monotonic = time.monotonic()
                self._session.status = "finished" if return_code == 0 else "failed"
                self._append_log_unlocked(f"[dashboard] Process exited with code {return_code}.")

    def _collect_resource_sample(self, pid: int, elapsed_s: float) -> dict[str, Any] | None:
        if psutil is None:
            return None
        with contextlib.suppress(Exception):
            root = psutil.Process(pid)
            processes = [root] + root.children(recursive=True)
            cpu_percent = 0.0
            rss_bytes = 0
            child_count = 0
            for proc in processes:
                try:
                    cpu_percent += float(proc.cpu_percent(interval=None))
                    rss_bytes += int(proc.memory_info().rss)
                    child_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return {
                "timestamp_iso": now_iso(),
                "elapsed_s": round(elapsed_s, 1),
                "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
                "system_memory_percent": float(psutil.virtual_memory().percent),
                "process_tree_cpu_percent": round(cpu_percent, 3),
                "process_tree_rss_mb": round(rss_bytes / (1024.0 * 1024.0), 3),
                "tracked_process_count": child_count,
            }
        return None

    def _resource_loop(self, process: subprocess.Popen[str]) -> None:
        if psutil is None:
            return
        with contextlib.suppress(Exception):
            psutil.cpu_percent(interval=None)
        with contextlib.suppress(Exception):
            root = psutil.Process(process.pid)
            for proc in [root] + root.children(recursive=True):
                with contextlib.suppress(Exception):
                    proc.cpu_percent(interval=None)

        while True:
            if process.poll() is not None:
                break
            with self._lock:
                if self._session.process is not process:
                    break
                elapsed_s = max(0.0, time.monotonic() - float(self._session.start_monotonic or time.monotonic()))
            sample = self._collect_resource_sample(process.pid, elapsed_s)
            if sample is not None:
                with self._lock:
                    if self._session.process is process:
                        self._session.resource_samples.append(sample)
            time.sleep(1.0)

    def _force_stop_after_grace(self, process: subprocess.Popen[str], grace_seconds: float = 8.0) -> None:
        deadline = time.time() + grace_seconds
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.5)
        if process.poll() is None:
            with self._lock:
                self._append_log_unlocked("[dashboard] Grace period expired. Terminating process.")
            self._signal_process_tree(process, signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                with self._lock:
                    self._append_log_unlocked("[dashboard] Process still active. Killing now.")
                self._signal_process_tree(process, signal.SIGKILL)

    def _signal_process_tree(self, process: subprocess.Popen[str], sig: int) -> None:
        with contextlib.suppress(Exception):
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, sig)
            return
        with contextlib.suppress(Exception):
            process.send_signal(sig)

    def _build_command(self, config: dict[str, Any]) -> list[str]:
        python_bin = str(DEFAULT_PYTHON)
        mode = str(config.get("mode") or "09")
        shared_headless = bool(config.get("headless"))
        shared_rviz = bool(config.get("rviz", True))
        shared_camera = bool(config.get("camera", True))

        if mode == "09":
            cmd = [python_bin, str(LIVE_MODEL_SCRIPT)]
            checkpoint = str(config.get("checkpoint") or "").strip()
            model = str(config.get("model") or "best").strip() or "best"
            if checkpoint:
                cmd.extend(["--checkpoint", checkpoint])
            else:
                cmd.extend(["--model", model])
            if shared_headless:
                cmd.append("--headless")
            if not shared_rviz:
                cmd.append("--no-rviz")
            if not shared_camera:
                cmd.append("--no-camera")
            target_index = int(config.get("target_index") or 4)
            cmd.extend(["--target-index", str(target_index)])
            if bool(config.get("enable_omnet")):
                cmd.append("--enable-omnet")
                omnet_config = str(config.get("omnet_config") or "Communication-GazeboBridge-WiFi").strip()
                cmd.extend(["--omnet-config", omnet_config])
            return cmd

        cmd = [python_bin, str(RULE_BASED_SCRIPT)]
        if shared_headless:
            cmd.append("--headless")
        if not shared_rviz:
            cmd.append("--no-rviz")
        if not shared_camera:
            cmd.append("--no-camera")
        if not bool(config.get("bag_recording", True)):
            cmd.append("--no-bag")
        if not bool(config.get("depth_classification", True)):
            cmd.append("--no-depth")
        if not bool(config.get("hazard_map", True)):
            cmd.append("--no-hazard-map")
        if bool(config.get("decision_fuser", False)):
            cmd.append("--enable-decision-fuser")
        if not bool(config.get("uav_enabled", True)):
            cmd.append("--disable-uavs")
        elif not bool(config.get("second_uav", True)):
            cmd.append("--disable-second-uav")
        if bool(config.get("lidar_path_planning", False)):
            cmd.append("--enable-lidar-path-planning")
        if not bool(config.get("lidar_straight_approach", True)):
            cmd.append("--disable-lidar-straight-approach")
        if bool(config.get("debug_isolate_husky_local", False)):
            cmd.append("--debug-isolate-husky-local")
        return cmd

    def _resource_summary_unlocked(self, elapsed_s: float) -> dict[str, Any]:
        samples = self._session.resource_samples
        latest = samples[-1] if samples else None

        def _avg(key: str) -> float:
            return round(sum(float(item[key]) for item in samples) / len(samples), 3) if samples else 0.0

        def _peak(key: str) -> float:
            return round(max(float(item[key]) for item in samples), 3) if samples else 0.0

        return {
            "available": psutil is not None,
            "sample_count": len(samples),
            "current": latest,
            "summary": {
                "elapsed_s": round(elapsed_s, 1),
                "avg_system_cpu_percent": _avg("system_cpu_percent"),
                "peak_system_cpu_percent": _peak("system_cpu_percent"),
                "avg_system_memory_percent": _avg("system_memory_percent"),
                "peak_system_memory_percent": _peak("system_memory_percent"),
                "avg_process_tree_cpu_percent": _avg("process_tree_cpu_percent"),
                "peak_process_tree_cpu_percent": _peak("process_tree_cpu_percent"),
                "avg_process_tree_rss_mb": _avg("process_tree_rss_mb"),
                "peak_process_tree_rss_mb": _peak("process_tree_rss_mb"),
                "peak_tracked_process_count": _peak("tracked_process_count"),
            },
        }

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._session.process is not None and self._session.process.poll() is None:
                raise RuntimeError("A simulation is already running.")

            command = self._build_command(config)
            session = RunSession(
                mode=str(config.get("mode") or "09"),
                status="running",
                config=dict(config),
                command=command,
                start_iso=now_iso(),
                start_monotonic=time.monotonic(),
            )
            session.log_doc_url = GZWEB_DOC_URL
            self._session = session
            self._append_log_unlocked(f"[dashboard] Launching {' '.join(command)}")

            process = subprocess.Popen(
                command,
                cwd=str(THESIS_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._session.process = process
            self._session.pid = process.pid
            self._append_log_unlocked(f"[dashboard] Process started with PID {process.pid} at {self._session.start_iso}.")

        thread = threading.Thread(target=self._reader_loop, args=(process,), daemon=True)
        thread.start()
        resource_thread = threading.Thread(target=self._resource_loop, args=(process,), daemon=True)
        resource_thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._session.process
            if process is None or process.poll() is not None:
                return self.status()
            self._session.status = "stopping"
            self._append_log_unlocked("[dashboard] Stop requested. Sending SIGINT to simulation process.")

        self._signal_process_tree(process, signal.SIGINT)
        thread = threading.Thread(target=self._force_stop_after_grace, args=(process,), daemon=True)
        thread.start()
        return self.status()

    def hard_stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._session.process
            if process is None or process.poll() is not None:
                return self.status()
            self._session.status = "stopping"
            self._append_log_unlocked("[dashboard] Hard stop requested. Killing simulation process tree.")
        self._signal_process_tree(process, signal.SIGKILL)
        with contextlib.suppress(Exception):
            process.wait(timeout=2.0)
        return self.status()

    def shutdown(self) -> None:
        with self._lock:
            process = self._session.process
            running = process is not None and process.poll() is None
            if not running:
                return
            self._session.status = "stopping"
            self._append_log_unlocked("[dashboard] Dashboard shutdown requested. Sending SIGINT to simulation process.")
        self._signal_process_tree(process, signal.SIGINT)
        self._force_stop_after_grace(process, grace_seconds=5.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            running = session.process is not None and session.process.poll() is None
            elapsed_s = 0.0
            if session.start_monotonic is not None:
                end = time.monotonic() if running else (session.stop_monotonic or time.monotonic())
                elapsed_s = max(0.0, end - session.start_monotonic)
            return {
                "mode": session.mode,
                "status": session.status,
                "running": running,
                "pid": session.pid,
                "command": session.command,
                "config": session.config,
                "start_iso": session.start_iso,
                "stop_iso": session.stop_iso,
                "elapsed_s": round(elapsed_s, 1),
                "return_code": session.return_code,
                "run_log_path": session.run_log_path,
                "log_count": len(session.log_items),
                "resource_tracker": self._resource_summary_unlocked(elapsed_s),
            }

    def logs_after(self, after_id: int) -> dict[str, Any]:
        with self._lock:
            items = [item for item in self._session.log_items if int(item["id"]) > after_id]
            next_id = self._session.log_items[-1]["id"] if self._session.log_items else 0
            return {"items": items, "next_id": next_id}


MANAGER = SimulationManager()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ThesisSimDashboard/0.1"

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_text_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
        if parsed.path == "/styles.css":
            return self._send_text_file(STATIC_ROOT / "styles.css", "text/css; charset=utf-8")
        if parsed.path == "/app.js":
            return self._send_text_file(STATIC_ROOT / "app.js", "application/javascript; charset=utf-8")
        if parsed.path == "/api/meta":
            return self._send_json(MANAGER.metadata())
        if parsed.path == "/api/status":
            return self._send_json(MANAGER.status())
        if parsed.path == "/api/logs":
            qs = parse_qs(parsed.query)
            after_id = int((qs.get("after") or ["0"])[0])
            return self._send_json(MANAGER.logs_after(after_id))
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            try:
                payload = self._read_json_body()
                return self._send_json(MANAGER.start(payload), status=HTTPStatus.ACCEPTED)
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/stop":
            try:
                return self._send_json(MANAGER.stop(), status=HTTPStatus.ACCEPTED)
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/hard-stop":
            try:
                return self._send_json(MANAGER.hard_stop(), status=HTTPStatus.ACCEPTED)
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Bind port. Default: 8765")
    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] Thesis simulation dashboard running at http://{args.host}:{args.port}")
    print(f"[dashboard] Child simulations will use Python: {DEFAULT_PYTHON}")
    print(f"[dashboard] GZweb install guide: {GZWEB_DOC_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Shutting down.")
    finally:
        MANAGER.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
