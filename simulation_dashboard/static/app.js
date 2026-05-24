const state = {
  lastLogId: 0,
  pollTimer: null,
  meta: null,
};

const els = {
  mode: document.getElementById("mode"),
  rviz: document.getElementById("rviz"),
  camera: document.getElementById("camera"),
  headless: document.getElementById("headless"),
  bagRecording: document.getElementById("bagRecording"),
  depthClassification: document.getElementById("depthClassification"),
  hazardMap: document.getElementById("hazardMap"),
  decisionFuser: document.getElementById("decisionFuser"),
  uavEnabled: document.getElementById("uavEnabled"),
  secondUav: document.getElementById("secondUav"),
  lidarStraightApproach: document.getElementById("lidarStraightApproach"),
  lidarPathPlanning: document.getElementById("lidarPathPlanning"),
  debugIsolateHuskyLocal: document.getElementById("debugIsolateHuskyLocal"),
  model: document.getElementById("model"),
  checkpoint: document.getElementById("checkpoint"),
  targetIndex: document.getElementById("targetIndex"),
  enableOmnet: document.getElementById("enableOmnet"),
  omnetConfig: document.getElementById("omnetConfig"),
  launchForm: document.getElementById("launchForm"),
  startButton: document.getElementById("startButton"),
  stopButton: document.getElementById("stopButton"),
  hardStopButton: document.getElementById("hardStopButton"),
  refreshButton: document.getElementById("refreshButton"),
  clearLogsButton: document.getElementById("clearLogsButton"),
  mode07Options: document.getElementById("mode07Options"),
  mode09Options: document.getElementById("mode09Options"),
  statusValue: document.getElementById("statusValue"),
  pidValue: document.getElementById("pidValue"),
  startValue: document.getElementById("startValue"),
  elapsedValue: document.getElementById("elapsedValue"),
  commandValue: document.getElementById("commandValue"),
  runLogValue: document.getElementById("runLogValue"),
  resourceSampleCount: document.getElementById("resourceSampleCount"),
  resourceCurrentSystemCpu: document.getElementById("resourceCurrentSystemCpu"),
  resourceAvgSystemCpu: document.getElementById("resourceAvgSystemCpu"),
  resourcePeakSystemCpu: document.getElementById("resourcePeakSystemCpu"),
  resourceCurrentSystemMem: document.getElementById("resourceCurrentSystemMem"),
  resourceAvgSystemMem: document.getElementById("resourceAvgSystemMem"),
  resourcePeakSystemMem: document.getElementById("resourcePeakSystemMem"),
  resourceCurrentProcessCpu: document.getElementById("resourceCurrentProcessCpu"),
  resourceAvgProcessCpu: document.getElementById("resourceAvgProcessCpu"),
  resourcePeakProcessCpu: document.getElementById("resourcePeakProcessCpu"),
  resourceCurrentProcessMem: document.getElementById("resourceCurrentProcessMem"),
  resourceAvgProcessMem: document.getElementById("resourceAvgProcessMem"),
  resourcePeakProcessMem: document.getElementById("resourcePeakProcessMem"),
  resourceCurrentProcessCount: document.getElementById("resourceCurrentProcessCount"),
  resourcePeakProcessCount: document.getElementById("resourcePeakProcessCount"),
  logOutput: document.getElementById("logOutput"),
  logCountValue: document.getElementById("logCountValue"),
  gzwebUrl: document.getElementById("gzwebUrl"),
  loadViewerButton: document.getElementById("loadViewerButton"),
  openViewerLink: document.getElementById("openViewerLink"),
  gzwebFrame: document.getElementById("gzwebFrame"),
  gzwebDocsLink: document.getElementById("gzwebDocsLink"),
};

function updateModeVisibility() {
  const mode = els.mode.value;
  els.mode07Options.classList.toggle("hidden", mode !== "07");
  els.mode09Options.classList.toggle("hidden", mode !== "09");
}

function collectConfig() {
  const mode = els.mode.value;
  const config = {
    mode,
    headless: els.headless.checked,
    rviz: els.rviz.checked,
    camera: els.camera.checked,
  };

  if (mode === "09") {
    config.model = els.model.value;
    config.checkpoint = els.checkpoint.value.trim();
    config.target_index = Number.parseInt(els.targetIndex.value || "4", 10);
    config.enable_omnet = els.enableOmnet.checked;
    config.omnet_config = els.omnetConfig.value;
    return config;
  }

  config.bag_recording = els.bagRecording.checked;
  config.depth_classification = els.depthClassification.checked;
  config.hazard_map = els.hazardMap.checked;
  config.decision_fuser = els.decisionFuser.checked;
  config.uav_enabled = els.uavEnabled.checked;
  config.second_uav = els.secondUav.checked;
  config.lidar_straight_approach = els.lidarStraightApproach.checked;
  config.lidar_path_planning = els.lidarPathPlanning.checked;
  config.debug_isolate_husky_local = els.debugIsolateHuskyLocal.checked;
  return config;
}

function applyConfig(config) {
  const mode = config.mode || "09";
  els.mode.value = mode;
  els.headless.checked = !!config.headless;
  els.rviz.checked = config.rviz !== false;
  els.camera.checked = config.camera !== false;

  if (mode === "09") {
    els.model.value = config.model || "best";
    els.checkpoint.value = config.checkpoint || "";
    els.targetIndex.value = config.target_index ?? 4;
    els.enableOmnet.checked = !!config.enable_omnet;
    els.omnetConfig.value = config.omnet_config || "Communication-GazeboBridge-WiFi";
  } else {
    els.bagRecording.checked = config.bag_recording !== false;
    els.depthClassification.checked = config.depth_classification !== false;
    els.hazardMap.checked = config.hazard_map !== false;
    els.decisionFuser.checked = !!config.decision_fuser;
    els.uavEnabled.checked = config.uav_enabled !== false;
    els.secondUav.checked = config.second_uav !== false;
    els.lidarStraightApproach.checked = config.lidar_straight_approach !== false;
    els.lidarPathPlanning.checked = !!config.lidar_path_planning;
    els.debugIsolateHuskyLocal.checked = !!config.debug_isolate_husky_local;
  }

  updateModeVisibility();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function renderStatus(status) {
  els.statusValue.textContent = status.status || "idle";
  els.pidValue.textContent = status.pid || "-";
  els.startValue.textContent = status.start_iso || "-";
  els.elapsedValue.textContent = `${(status.elapsed_s || 0).toFixed(1)} s`;
  els.commandValue.textContent = status.command && status.command.length ? status.command.join(" ") : "-";
  els.runLogValue.textContent = status.run_log_path || "-";
  els.startButton.disabled = !!status.running;
  els.stopButton.disabled = !status.running && status.status !== "stopping";
  els.hardStopButton.disabled = !status.running && status.status !== "stopping";
  renderResourceTracker(status.resource_tracker || null);
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function renderResourceTracker(tracker) {
  if (!tracker || !tracker.available) {
    els.resourceSampleCount.textContent = "tracker unavailable";
    els.resourceCurrentSystemCpu.textContent = "-";
    els.resourceAvgSystemCpu.textContent = "-";
    els.resourcePeakSystemCpu.textContent = "-";
    els.resourceCurrentSystemMem.textContent = "-";
    els.resourceAvgSystemMem.textContent = "-";
    els.resourcePeakSystemMem.textContent = "-";
    els.resourceCurrentProcessCpu.textContent = "-";
    els.resourceAvgProcessCpu.textContent = "-";
    els.resourcePeakProcessCpu.textContent = "-";
    els.resourceCurrentProcessMem.textContent = "-";
    els.resourceAvgProcessMem.textContent = "-";
    els.resourcePeakProcessMem.textContent = "-";
    els.resourceCurrentProcessCount.textContent = "-";
    els.resourcePeakProcessCount.textContent = "-";
    return;
  }

  const current = tracker.current || {};
  const summary = tracker.summary || {};
  els.resourceSampleCount.textContent = `${tracker.sample_count || 0} samples`;
  els.resourceCurrentSystemCpu.textContent = fmt(current.system_cpu_percent, 1);
  els.resourceAvgSystemCpu.textContent = fmt(summary.avg_system_cpu_percent, 1);
  els.resourcePeakSystemCpu.textContent = fmt(summary.peak_system_cpu_percent, 1);
  els.resourceCurrentSystemMem.textContent = fmt(current.system_memory_percent, 1);
  els.resourceAvgSystemMem.textContent = fmt(summary.avg_system_memory_percent, 1);
  els.resourcePeakSystemMem.textContent = fmt(summary.peak_system_memory_percent, 1);
  els.resourceCurrentProcessCpu.textContent = fmt(current.process_tree_cpu_percent, 1);
  els.resourceAvgProcessCpu.textContent = fmt(summary.avg_process_tree_cpu_percent, 1);
  els.resourcePeakProcessCpu.textContent = fmt(summary.peak_process_tree_cpu_percent, 1);
  els.resourceCurrentProcessMem.textContent = fmt(current.process_tree_rss_mb, 1);
  els.resourceAvgProcessMem.textContent = fmt(summary.avg_process_tree_rss_mb, 1);
  els.resourcePeakProcessMem.textContent = fmt(summary.peak_process_tree_rss_mb, 1);
  els.resourceCurrentProcessCount.textContent = current.tracked_process_count ?? "-";
  els.resourcePeakProcessCount.textContent = summary.peak_tracked_process_count ?? "-";
}

async function refreshStatus() {
  try {
    const status = await fetchJson("/api/status");
    renderStatus(status);
  } catch (error) {
    appendLocalLog(`[dashboard-ui] ${error.message}`);
  }
}

function appendLocalLog(text) {
  els.logOutput.textContent += `${text}\n`;
  els.logOutput.scrollTop = els.logOutput.scrollHeight;
}

async function refreshLogs() {
  try {
    const payload = await fetchJson(`/api/logs?after=${state.lastLogId}`);
    for (const item of payload.items) {
      els.logOutput.textContent += `${item.text}\n`;
      state.lastLogId = item.id;
    }
    els.logCountValue.textContent = `${state.lastLogId} lines`;
    els.logOutput.scrollTop = els.logOutput.scrollHeight;
  } catch (error) {
    appendLocalLog(`[dashboard-ui] ${error.message}`);
  }
}

async function restoreSessionView() {
  state.lastLogId = 0;
  els.logOutput.textContent = "";
  const status = await fetchJson("/api/status");
  renderStatus(status);
  if (status.config && Object.keys(status.config).length > 0) {
    applyConfig(status.config);
  }
  const payload = await fetchJson("/api/logs?after=0");
  for (const item of payload.items) {
    els.logOutput.textContent += `${item.text}\n`;
    state.lastLogId = item.id;
  }
  els.logCountValue.textContent = `${status.log_count || state.lastLogId} lines`;
  els.logOutput.scrollTop = els.logOutput.scrollHeight;
}

async function poll() {
  await Promise.all([refreshStatus(), refreshLogs()]);
}

function loadViewer() {
  const url = els.gzwebUrl.value.trim();
  if (!url) return;
  els.openViewerLink.href = url;
  els.gzwebFrame.src = url;
}

async function boot() {
  state.meta = await fetchJson("/api/meta");
  for (const model of state.meta.available_models) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    if (model === "best") option.selected = true;
    els.model.appendChild(option);
  }

  els.gzwebUrl.value = state.meta.default_gzweb_url;
  els.openViewerLink.href = state.meta.default_gzweb_url;
  els.gzwebDocsLink.href = state.meta.gzweb_doc_url;
  updateModeVisibility();
  await restoreSessionView();
  state.pollTimer = window.setInterval(poll, 1000);
}

els.mode.addEventListener("change", updateModeVisibility);
els.loadViewerButton.addEventListener("click", loadViewer);

els.launchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    els.logOutput.textContent = "";
    state.lastLogId = 0;
    const payload = collectConfig();
    await fetchJson("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await poll();
  } catch (error) {
    appendLocalLog(`[dashboard-ui] ${error.message}`);
  }
});

els.stopButton.addEventListener("click", async () => {
  try {
    await fetchJson("/api/stop", { method: "POST" });
    await poll();
  } catch (error) {
    appendLocalLog(`[dashboard-ui] ${error.message}`);
  }
});

els.hardStopButton.addEventListener("click", async () => {
  try {
    await fetchJson("/api/hard-stop", { method: "POST" });
    await poll();
  } catch (error) {
    appendLocalLog(`[dashboard-ui] ${error.message}`);
  }
});

els.refreshButton.addEventListener("click", async () => {
  try {
    await restoreSessionView();
  } catch (error) {
    appendLocalLog(`[dashboard-ui] ${error.message}`);
  }
});

els.clearLogsButton.addEventListener("click", () => {
  els.logOutput.textContent = "";
  els.logCountValue.textContent = "0 visible lines";
});

boot().catch((error) => {
  appendLocalLog(`[dashboard-ui] ${error.message}`);
});
