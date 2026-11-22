// Settings & resources tab: readiness, env creation, model downloads, service control.
import * as api from "./api.js";
import { state, emit, on } from "./state.js";
import { $, show, fmtBytes, toast, escapeHtml } from "./utils.js";
import { t, onLanguageChange } from "./i18n.js";

const FLOW_MODEL_KEYS = {
  "FlowHMR 主模型": "flowModel.main",
  "主模型配置": "flowModel.config",
  "YOLOX 检测器": "flowModel.yolox",
  "SAM 3D Body": "flowModel.sam",
  "SMPL-H / MANO 体型": "flowModel.body",
  "VGGT-Omega 相机": "flowModel.vggt",
};

let initialized = false;
let setupTimer = 0;
let setupOffset = 0;
let watchedTask = null;

const CHECK = { ok: "✓", bad: "!", warn: "•" };

function shorten(path, limit = 58) {
  if (!path) return "";
  if (path.length <= limit) return path;
  return "…" + path.slice(-(limit - 1));
}

function statusChip(element, ok, okText = t("settings.ready"), badText = t("settings.notReady")) {
  element.textContent = ok ? okText : badText;
  element.className = `chip ${ok ? "green" : "red"}`;
}

function checkRow(label, ok, detail = "", warn = false) {
  const mark = ok ? "ok" : warn ? "warn" : "bad";
  return `<div class="status-row" title="${escapeHtml(detail)}">
    <span class="status-mark ${mark}">${CHECK[mark]}</span>
    <span class="status-label">${escapeHtml(label)}</span>
    <span class="status-detail">${escapeHtml(detail)}</span>
  </div>`;
}

export function initSettings() {
  if (initialized) return;
  initialized = true;

  $("#instantSaveBtn").addEventListener("click", () => saveInstant(true));
  $("#instantEnvBtn").addEventListener("click", () => startSetup("instanthmr", "env"));
  $("#instantCameraBtn").addEventListener("click", toggleInstantCamera);
  $("#instantOpenModels").addEventListener("click", () => openEngineModels("instanthmr"));
  $("#copyInstantModels").addEventListener("click", () => copyPath(
    state.engines?.instanthmr?.models_dir || $("#instantModelsDir").value.trim()));
  $("#instantModels").addEventListener("click", (event) => {
    const button = event.target.closest("[data-download]");
    if (!button) return;
    const tier = button.dataset.download;
    downloadInstantModel(tier, button);
  });

  $("#flowSaveBtn").addEventListener("click", () => saveFlow(true));
  $("#flowEnvBtn").addEventListener("click", () => startSetup("flowhmr", "env"));
  $("#flowDownloadBtn").addEventListener("click", () => startSetup("flowhmr", "models"));
  $("#flowServiceBtn").addEventListener("click", toggleFlowService);
  $("#flowOpenModels").addEventListener("click", () => openEngineModels("flowhmr"));
  $("#copyFlowModels").addEventListener("click", () => copyPath(state.engines?.flowhmr?.models_dir || ""));
  $("#flowOpenRoot").addEventListener("click", () => openFolder("flowhmr", "root"));
  $("#flowModels").addEventListener("click", (event) => {
    const button = event.target.closest("[data-open-path]");
    if (button) openModelPath(button.dataset.openPath);
  });

  $("#saveDefaultsBtn").addEventListener("click", saveDefaults);
  $("#stopSetupBtn").addEventListener("click", stopSetup);
  $("#closeSetupLog").addEventListener("click", () => show($("#setupLogCard"), false));
  $("#openLogsBtn").addEventListener("click", () => {
    toast(t("settings.toast.logs"), "info", 6000);
  });

  populateForms();
  on("engine", syncSettingsEngine);
  onLanguageChange(() => {
    syncSettingsEngine();
    if (state.engines) updateEngineStatus(state.engines);
    if (watchedTask) updateSetupTaskLabels();
  });
  syncSettingsEngine();
}

export function syncSettingsEngine() {
  const engine = state.engine === "flowhmr" ? "flowhmr" : "instanthmr";
  show($("#instantCard"), engine === "instanthmr");
  show($("#flowCard"), engine === "flowhmr");
  show($("#poseFilterField"), engine === "instanthmr");
  show($("#instantSmoothField"), engine === "instanthmr");
  show($("#flowManualField"), engine === "flowhmr");
  show($("#flowStaticField"), engine === "flowhmr");
  const label = engine === "instanthmr" ? "InstantHMR" : "FlowHMR";
  const text = $("#settingsContextText");
  if (text) text.innerHTML = t("settings.context", { engine: label });
  const context = $("#settingsContext");
  if (context) context.dataset.engine = engine;
}

function flowModelLabel(label) {
  const key = FLOW_MODEL_KEYS[label];
  return key ? t(key) : label;
}

function setupTitle(title) {
  if (!title) return t("settings.setupLog.title");
  if (title.startsWith("准备 InstantHMR 环境")) return t("setup.title.prepareInstant");
  if (title.startsWith("自动配置 FlowHMR 环境")) return t("setup.title.autoFlow");
  if (title.startsWith("检查 FlowHMR 环境")) return t("setup.title.checkFlow");
  if (title.startsWith("下载 InstantHMR · ")) {
    return t("setup.title.downloadInstant", { model: title.split("·").pop().trim() });
  }
  if (title === "下载 FlowHMR 公开模型") return t("setup.title.downloadFlow");
  return title;
}

function setupStatusLabel(status) {
  if (status === "running") return t("settings.setup.running");
  if (status === "done") return t("settings.setup.done");
  if (status === "cancelled") return t("settings.setup.cancelled");
  if (status === "error") return t("settings.setup.failed");
  return t("settings.setup.disconnected");
}

function updateSetupTaskLabels() {
  if (!watchedTask) return;
  $("#setupLogTitle").textContent = setupTitle(watchedTask.title || "");
  $("#setupLogStatus").textContent = setupStatusLabel(watchedTask.status);
}

export function populateForms() {
  const cfg = state.config;
  if (!cfg) return;
  const inst = cfg.engines.instanthmr;
  $("#instantModelsDir").value = inst.models_dir || "";
  $("#instantBasePython").value = inst.base_python || "python";
  $("#instantModelSelect").value = inst.model || "medium";
  $("#instantFit").value = String(inst.fit ?? 2);
  $("#instantDevice").value = inst.device || "auto";
  $("#instantCameraIndex").value = inst.camera_index ?? 0;

  const flow = cfg.engines.flowhmr;
  $("#flowRoot").value = flow.root || "";
  $("#flowPython").value = flow.python || "";
  $("#flowPort").value = flow.port || 60550;
  $("#flowBasePython").value = flow.base_python || "python";
  $("#flowDevice").value = flow.device || "auto";

  $("#defaultMaxFrames").value = cfg.job?.max_frames ?? 300;
  $("#poseFilter").value = inst.pose_filter || "one-euro";
  $("#instantSmooth").checked = Boolean(inst.smooth);
  $("#flowManualSelect").value = String(Boolean(flow.manual_select));
  $("#flowStaticCamera").value = String(Boolean(flow.static_camera));
  $("#ffmpegPath").value = cfg.ffmpeg || "";
}

export function updateEngineStatus(engines) {
  if (!engines) return;
  renderInstant(engines.instanthmr);
  renderFlow(engines.flowhmr);
}

function renderInstant(status) {
  if (!status) return;
  statusChip($("#instantReadyChip"), status.ready, t("settings.ready"), t("settings.notReady"));
  const missing = Object.entries(status.deps || {}).filter(([, ok]) => !ok).map(([name]) => name);
  const checks = [
    checkRow(t("settings.checks.project"), status.root_ok, shorten(status.root)),
    checkRow(t("settings.checks.python"), status.python_ok,
      status.python_version || shorten(status.python) || t("settings.detail.notCreated")),
    checkRow(t("settings.checks.deps"), status.deps_ok,
      missing.length ? t("settings.detail.missingDeps", { items: missing.join(", ") })
        : "onnxruntime / opencv / numpy / rfdetr"),
    checkRow(t("settings.checks.model"), status.selected_model_ok, status.selected_model || ""),
    checkRow(t("settings.checks.body"), status.body_ok, "mhr-body.onnx"),
  ];
  $("#instantChecks").innerHTML = checks.join("");
  const pathEl = $("#instantModelsPath");
  if (pathEl) {
    pathEl.textContent = status.models_dir || "—";
    pathEl.title = status.models_dir || "";
  }

  const busy = Boolean(status.setup_running);
  const rows = (status.models || []).map((model) => {
    const stateText = model.exists ? fmtBytes(model.size) : t("settings.model.notDownloaded");
    const stateClass = model.exists ? "ok" : "bad";
    const current = model.selected ? `<span class="chip accent">${t("settings.model.current")}</span>` : "";
    return `<div class="model-row">
      <div class="model-main">
        <div class="model-title">${escapeHtml(t(`tier.${model.id}`))} ${current}</div>
        <div class="model-file" title="${escapeHtml(model.path)}">${escapeHtml(model.file)}</div>
      </div>
      <span class="model-state ${stateClass}">${stateText}</span>
      <button type="button" class="btn ghost" data-download="${escapeHtml(model.id)}"
        ${model.exists || busy ? "disabled" : ""}>${model.exists ? t("settings.model.downloaded") : t("settings.model.download")}</button>
    </div>`;
  });
  $("#instantModels").innerHTML = rows.join("") || `<p class="muted small">${escapeHtml(t("settings.model.none"))}</p>`;
  $("#instantEnvBtn").textContent = busy ? t("settings.preparing") : t("settings.env");
  $("#instantEnvBtn").disabled = busy;

  const camera = status.camera || {};
  const running = camera.status === "running";
  const cameraBtn = $("#instantCameraBtn");
  if (cameraBtn) {
    cameraBtn.textContent = running ? t("settings.cameraStop") : t("settings.cameraOpen");
    cameraBtn.disabled = busy || (!status.ready && !running);
    cameraBtn.title = camera.status === "error" ? (camera.message || "") : "";
  }
  const cameraHint = $("#instantCameraHint");
  if (cameraHint) {
    if (running) {
      cameraHint.textContent = t("settings.cameraRunning", { index: camera.camera_index ?? 0 });
    } else if (camera.status === "error") {
      cameraHint.textContent = t("settings.cameraError");
    } else {
      cameraHint.textContent = t("settings.cameraHint");
    }
  }
}

function renderFlow(status) {
  if (!status) return;
  const runtime = status.inference_runtime;
  const runtimeMissing = Boolean(runtime && !runtime.installed);
  statusChip($("#flowReadyChip"), status.job_ready ?? status.ready, t("settings.ready"),
    runtimeMissing ? t("settings.notReady") : status.ready ? t("settings.needsCamera") : t("settings.notReady"));
  const srv = status.server || {};
  const serviceChip = $("#flowServiceChip");
  const serviceMap = {
    running: [t("settings.serviceRunning"), "green"],
    starting: [t("settings.serviceStarting"), "amber"],
    error: [t("settings.serviceError"), "red"],
    stopped: [t("settings.serviceStopped"), "red"],
    unknown: [t("settings.serviceUnknown"), ""],
  };
  const [serviceText, serviceClass] = serviceMap[srv.status] || serviceMap.unknown;
  serviceChip.textContent = serviceText;
  serviceChip.className = `chip service ${serviceClass}`;

  const checkRows = [
    checkRow(t("settings.checks.project"), status.root_ok, shorten(status.root)),
    checkRow(t("settings.checks.runtime"), status.python_ok,
      status.env_kind === "bundled" ? t("settings.detail.bundled") : status.python_version || t("settings.detail.notCreated")),
    runtime ? checkRow(t("settings.checks.inferenceRuntime"), runtime.installed,
      runtime.installed ? (runtime.label || runtime.selected || "—") : t("settings.runtime.missing")) : "",
    checkRow(t("settings.checks.layout"), status.core_models_ok,
      status.layout === "windows" ? t("settings.detail.windows") : t("settings.detail.upstream")),
    checkRow(t("settings.checks.core"), status.core_models_ok,
      status.core_models_ok ? t("settings.detail.allReady") : t("settings.detail.someMissing")),
  ];
  $("#flowChecks").innerHTML = checkRows.join("");
  const pathEl = $("#flowModelsPath");
  if (pathEl) {
    pathEl.textContent = status.models_dir || "—";
    pathEl.title = status.models_dir || "";
  }

  $("#flowModels").innerHTML = (status.models || []).map((model) => `
    <div class="model-row">
      <div class="model-main">
        <div class="model-title">${escapeHtml(flowModelLabel(model.label))}
          ${model.required ? `<span class="chip">${escapeHtml(t("settings.required"))}</span>` : `<span class="chip">${escapeHtml(t("settings.optional"))}</span>`}</div>
        <div class="model-file" title="${escapeHtml(model.path)}">${escapeHtml(shorten(model.path, 72))}</div>
      </div>
      <span class="model-state ${model.exists ? "ok" : "bad"}">${model.exists ? fmtBytes(model.size) : t("settings.model.missing")}</span>
      <button type="button" class="link-btn" data-open-path="${escapeHtml(model.path)}"
        title="${escapeHtml(t("settings.model.openDir"))}: ${escapeHtml(model.path)}">${t("settings.model.openDir")}</button>
    </div>
  `).join("");

  const busy = Boolean(status.setup_running);
  const button = $("#flowServiceBtn");
  if (srv.status === "running") {
    button.textContent = srv.external ? t("settings.externalRunning") : t("settings.stopService");
    button.disabled = Boolean(srv.external);
  } else if (srv.status === "starting") {
    button.textContent = t("settings.starting");
    button.disabled = true;
  } else {
    button.textContent = t("settings.startService");
    button.disabled = !status.root_ok || !status.python_ok;
  }
  $("#flowEnvBtn").disabled = busy;
  $("#flowDownloadBtn").disabled = busy || !status.can_download_models;
  $("#flowEnvBtn").textContent = busy ? t("settings.preparing") : t("settings.env");

  const manual = $("#flowManualSelect");
  const manualNote = $("#flowManualNote");
  if (status.manual_select_forced) {
    manual.value = "true";
    manual.disabled = true;
    manualNote.textContent = t("settings.manualForced");
  } else {
    manual.disabled = false;
    manualNote.textContent = t("settings.manualUpstream");
  }

  const hint = $("#flowHint");
  if (runtimeMissing) {
    hint.textContent = t("settings.flowHint.runtime");
  } else if (!status.job_ready && status.ready) {
    hint.textContent = t("settings.flowHint.camera");
  } else if (status.env_kind === "bundled") {
    hint.textContent = t("settings.flowHint.bundled");
  } else if (status.can_download_models) {
    hint.textContent = t("settings.flowHint.venv");
  } else {
    hint.textContent = t("settings.flowHint.manual");
  }
}

async function saveInstant(notify) {
  const payload = {
    engines: {
      instanthmr: {
        models_dir: $("#instantModelsDir").value.trim(),
        base_python: $("#instantBasePython").value.trim(),
        model: $("#instantModelSelect").value,
        fit: Number($("#instantFit").value),
        device: $("#instantDevice").value,
        camera_index: Math.max(0, Math.trunc(Number($("#instantCameraIndex").value) || 0)),
      },
    },
  };
  try {
    const result = await api.saveConfig(payload);
    state.config = result.config;
    state.engines = result.engines;
    updateEngineStatus(result.engines);
    emit("config", result.config);
    emit("engines", result.engines);
    if (notify) toast(t("settings.toast.savedInstant"), "success");
    return true;
  } catch (error) {
    toast(error.message, "error");
    return false;
  }
}

async function downloadInstantModel(tier, button) {
  // Downloading a row also makes it the active model for new jobs.
  $("#instantModelSelect").value = tier;
  if (!(await saveInstant(false))) return;
  button.disabled = true;
  button.textContent = t("settings.starting");
  try {
    const response = await api.startModelSetup("instanthmr", {
      model: tier,
      fit: Number($("#instantFit").value),
    });
    watchSetup(response.task);
  } catch (error) {
    toast(error.message, "error");
    button.disabled = false;
    button.textContent = t("settings.model.download");
  }
}

async function saveFlow(notify) {
  const payload = {
    engines: {
      flowhmr: {
        root: $("#flowRoot").value.trim(),
        python: $("#flowPython").value.trim(),
        port: Number($("#flowPort").value),
        base_python: $("#flowBasePython").value.trim(),
        device: $("#flowDevice").value,
      },
    },
  };
  try {
    const result = await api.saveConfig(payload);
    state.config = result.config;
    state.engines = result.engines;
    updateEngineStatus(result.engines);
    populateForms();
    emit("config", result.config);
    emit("engines", result.engines);
    if (notify) toast(t("settings.toast.savedFlow"), "success");
    return true;
  } catch (error) {
    toast(error.message, "error");
    return false;
  }
}

async function toggleFlowService() {
  const srv = state.engines?.flowhmr?.server || {};
  try {
    if (srv.status === "running" && !srv.external) {
      const result = await api.stopFlowService();
      toast(t("settings.toast.flowStopped"), "success");
      void result;
    } else {
      toast(t("settings.toast.startingFlow"), "info");
      await api.startFlowService();
    }
    await refreshEngines();
  } catch (error) {
    toast(error.message, "error", 7000);
  }
}

async function toggleInstantCamera() {
  const camera = state.engines?.instanthmr?.camera || {};
  const button = $("#instantCameraBtn");
  if (!button) return;
  button.disabled = true;
  try {
    if (camera.status === "running") {
      await api.stopInstantCamera();
      toast(t("settings.toast.cameraStopped"), "success");
    } else {
      const saved = await saveInstant(false);
      if (!saved) return;
      const index = Math.max(0, Math.trunc(Number($("#instantCameraIndex").value) || 0));
      toast(t("settings.toast.cameraStarting"), "info");
      await api.startInstantCamera({ camera_index: index });
      toast(t("settings.toast.cameraStarted", { index }), "success");
    }
    await refreshEngines();
  } catch (error) {
    toast(error.message, "error", 8000);
    await refreshEngines();
  } finally {
    button.disabled = false;
  }
}

async function startSetup(engine, kind) {
  try {
    const response = kind === "env"
      ? await api.startEnvSetup(engine)
      : await api.startModelSetup(engine, {});
    watchSetup(response.task);
  } catch (error) {
    toast(error.message, "error", 7000);
  }
}

async function saveDefaults() {
  const payload = {
    job: { max_frames: Number($("#defaultMaxFrames").value) || 0 },
    ffmpeg: $("#ffmpegPath").value.trim(),
    engines: {
      instanthmr: {
        pose_filter: $("#poseFilter").value,
        smooth: $("#instantSmooth").checked,
      },
      flowhmr: {
        manual_select: $("#flowManualSelect").value === "true",
        static_camera: $("#flowStaticCamera").value === "true",
      },
    },
  };
  try {
    const result = await api.saveConfig(payload);
    state.config = result.config;
    state.engines = result.engines;
    populateForms();
    $("#maxFrames").value = result.config.job?.max_frames ?? 300;
    updateEngineStatus(result.engines);
    emit("config", result.config);
    emit("engines", result.engines);
    toast(t("settings.toast.defaultsSaved"), "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function openFolder(engine, which) {
  try {
    const result = await api.openFolder(engine, which);
    toast(t("settings.toast.opened", { path: result.path }), "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

function openEngineModels(engine) {
  const status = state.engines?.[engine];
  const path = status?.models_dir
    || (engine === "instanthmr" ? $("#instantModelsDir").value.trim() : "");
  if (!path) {
    toast(t("settings.toast.noModelsDir"), "warn");
    return;
  }
  openModelPath(path);
}

async function openModelPath(path) {
  try {
    const result = await api.openPath(path);
    toast(result.created
      ? t("settings.toast.createdOpened", { path: result.path })
      : t("settings.toast.opened", { path: result.path }), "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function copyPath(path) {
  if (!path) {
    toast(t("settings.toast.noPath"), "warn");
    return;
  }
  try {
    await navigator.clipboard.writeText(path);
    toast(t("settings.toast.copied"), "success");
    return;
  } catch (_) {
    // Fall through to the textarea fallback (WebView2 / permission edge cases).
  }
  const textarea = document.createElement("textarea");
  textarea.value = path;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand("copy");
    toast(t("settings.toast.copied"), "success");
  } catch (_) {
    toast(t("settings.toast.path", { path }), "info", 9000);
  } finally {
    textarea.remove();
  }
}

export async function refreshEngines() {
  try {
    const engines = await api.getEngines();
    state.engines = engines;
    updateEngineStatus(engines);
    emit("engines", engines);
    return engines;
  } catch (error) {
    console.warn(error);
    return null;
  }
}

export function watchSetup(task) {
  if (!task) return;
  watchedTask = task;
  setupOffset = 0;
  $("#setupLog").textContent = "";
  $("#setupLogTitle").textContent = setupTitle(task.title || "");
  $("#setupLogSubtitle").textContent = t("settings.setupLog.live");
  $("#setupLogStatus").textContent = t("settings.setup.running");
  $("#setupLogStatus").className = "chip amber";
  $("#stopSetupBtn").disabled = false;
  show($("#setupLogCard"), true);
  $("#setupLogCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
  window.clearInterval(setupTimer);
  setupTimer = window.setInterval(pollSetup, 900);
  void pollSetup();
}

async function pollSetup() {
  if (!watchedTask) return;
  try {
    const view = await api.getSetupTask(watchedTask.id, setupOffset);
    watchedTask = { ...watchedTask, ...view };
    const log = $("#setupLog");
    if (view.lines?.length) {
      setupOffset = view.next;
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.textContent += (log.textContent ? "\n" : "") + view.lines.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
    if (view.status !== "running") {
      window.clearInterval(setupTimer);
      const ok = view.status === "done";
      $("#setupLogStatus").textContent = setupStatusLabel(view.status);
      $("#setupLogStatus").className = `chip ${ok ? "green" : view.status === "cancelled" ? "amber" : "red"}`;
      $("#stopSetupBtn").disabled = true;
      const title = setupTitle(view.title);
      toast(ok
        ? t("settings.toast.setupDone", { title })
        : t("settings.toast.setupFailed", { title, reason: view.error || t("settings.toast.seeLog") }),
        ok ? "success" : "error", ok ? 4200 : 8000);
      await refreshEngines();
    }
  } catch (error) {
    window.clearInterval(setupTimer);
    $("#setupLogStatus").textContent = t("settings.setup.disconnected");
    $("#setupLogStatus").className = "chip red";
    toast(t("settings.toast.setupLogFailed", { reason: error.message }), "error");
  }
}

async function stopSetup() {
  if (!watchedTask) return;
  try {
    await api.stopSetupTask(watchedTask.id);
    toast(t("settings.toast.setupStop"), "warn");
  } catch (error) {
    toast(error.message, "error");
  }
}
