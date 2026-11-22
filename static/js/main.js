import * as api from "./api.js";
import { state, on, setEngine } from "./state.js";
import { $, $$, show, fmtBytes, fmtDate, toast, statusLabel, escapeHtml } from "./utils.js";
import { VideoOverlay } from "./overlay.js";
import { MeshViewer } from "./viewer.js";
import { initSettings, updateEngineStatus, populateForms } from "./settings.js";
import { initI18n, t, getLang, toggleLang, onLanguageChange } from "./i18n.js";

const ALLOWED = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"]);
const TERMINAL = new Set(["done", "error", "cancelled"]);
const ENGINE_LABEL = { instanthmr: "InstantHMR", flowhmr: "FlowHMR", gemx: "GEM-X", tdpt: "TDPT" };
const FLOW_MODEL_KEYS = {
  "FlowHMR 主模型": "flowModel.main",
  "主模型配置": "flowModel.config",
  "YOLOX 检测器": "flowModel.yolox",
  "SAM 3D Body": "flowModel.sam",
  "SMPL-H / MANO 体型": "flowModel.body",
  "VGGT-Omega 相机": "flowModel.vggt",
};
const STAGE_KEYS = {
  Detection: "stage.detection",
  "人物检测": "stage.detection",
  Camera: "stage.camera",
  "相机估计": "stage.camera",
  "Body features": "stage.bodyFeatures",
  "人体特征": "stage.bodyFeatures",
  Generation: "stage.generation",
  "动作生成": "stage.generation",
  Export: "stage.export",
  "导出结果": "stage.export",
  Postprocess: "stage.postprocess",
  "动作后处理": "stage.postprocess",
  Loading: "stage.loading",
  "加载模型": "stage.loading",
  Tracking: "stage.tracking",
  "人物跟踪": "stage.tracking",
  "Static camera": "stage.staticCamera",
  "静态机位": "stage.staticCamera",
  "准备": "job.stage.prepare",
  "检测与重建": "job.stage.detect",
  "写出结果": "job.stage.write",
  "完成": "job.stage.done",
};

let overlay = null;
let viewer = null;
let uploading = false;
let viewerLoadedJobId = null;
let viewerLoadingJobId = null;
let personChoice = null;
let personJobId = null;
let trackState = { jobId: null, loaded: false, lastTry: 0 };
let flowVideoReadyJob = null;
let previousStatus = new Map();
let jobFetching = false;
let engineFetching = false;
let jobsFetching = false;
let pollTimer = 0;
let engineTimer = 0;
let jobsTimer = 0;
let lastViewerMeta = null;
let precisionInitialized = false;

const PREF_SYNC = "p3d.syncVideo";
const PREF_AUDIO = "p3d.audio";
const PREF_LOCK = "p3d.fixedView";
let syncVideoEnabled = readPref(PREF_SYNC);
let audioEnabled = readPref(PREF_AUDIO);
let viewLocked = readPref(PREF_LOCK);

document.addEventListener("DOMContentLoaded", init);

async function init() {
  initI18n();
  overlay = new VideoOverlay($("#video"), $("#videoOverlay"));
  overlay.setVisible(false);
  viewer = new MeshViewer($("#threeContainer"));
  viewer.onFrame = (frame, total, fps) => updatePlayUI(frame, total, fps);
  viewer.setActive(false);
  viewer.setViewLocked(viewLocked);
  if (syncVideoEnabled) viewer.setSyncClock(syncClock);
  updatePlaybackToggles();

  bindTabs();
  bindTheme();
  bindLanguage();
  bindEnginePicker();
  bindPrecisionControls();
  bindFileInput();
  bindStart();
  bindJobActions();
  bindJobList();
  bindViewer();
  initSettings();

  on("engines", (engines) => {
    state.engines = engines;
    updateEngineStatus(engines);
    updateEngineUI();
  });
  on("config", (config) => {
    state.config = config;
    initPrecisionDefaults();
  });

  try {
    const [config, engines] = await Promise.all([api.getConfig(), api.getEngines()]);
    state.config = config;
    state.engines = engines;
    const savedTheme = localStorage.getItem("p3d.theme");
    if (!savedTheme) applyTheme(config.app?.theme || "dark", false);
    populateForms();
    updateEngineStatus(engines);
    $("#maxFrames").value = config.job?.max_frames ?? 300;
    initPrecisionDefaults();
    const query = new URLSearchParams(window.location.search);
    const themeParam = query.get("theme");
    if (themeParam === "light" || themeParam === "dark") applyTheme(themeParam, true);
    const engineParam = query.get("engine");
    if (["instanthmr", "flowhmr", "gemx", "tdpt"].includes(engineParam)) setEngine(engineParam);
    updateEngineUI();
    await refreshJobs();
    const jobParam = query.get("job");
    const tabParam = query.get("tab");
    if (jobParam) await selectJob(jobParam, false);
    else {
      const activeJob = state.jobs.find((job) => !TERMINAL.has(job.status));
      if (activeJob) await selectJob(activeJob.id, false);
    }
    if (["video", "viewer", "settings"].includes(tabParam)) setTab(tabParam);
  } catch (error) {
    toast(t("app.initFailed", { reason: error.message }), "error", 9000);
  }

  pollTimer = window.setInterval(pollCurrentJob, 1200);
  engineTimer = window.setInterval(refreshEngineStatus, 8000);
  jobsTimer = window.setInterval(refreshJobs, 10000);
  updateEngineUI();
}

/* ------------------------------------------------------------------ tabs */
function bindTabs() {
  $$(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });
  $("#goVideoTab").addEventListener("click", () => setTab("video"));
  $("#openResultBtn").addEventListener("click", () => setTab("viewer"));
  $("#openResultBtnInline").addEventListener("click", () => setTab("viewer"));
}

function setTab(name) {
  if (!name) return;
  state.tab = name;
  $$(".tab-btn").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  if (name !== "video") $("#video").pause();
  if (name === "viewer") {
    viewer.setActive(true);
    fitSyncVideoBox();
    if (state.currentJob?.status === "done") void ensureViewer(state.currentJob);
  } else {
    viewer.pause();
    viewer.setActive(false);
    $("#syncVideo").pause();
    $("#playBtn").textContent = "▶";
  }
}

/* ------------------------------------------------------------------ theme */
function bindTheme() {
  const saved = localStorage.getItem("p3d.theme") || "dark";
  applyTheme(saved, false);
  $("#themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(next, true);
  });
}

function bindLanguage() {
  $("#langToggle").addEventListener("click", () => toggleLang());
  onLanguageChange(() => refreshLanguage());
  updateLangButton();
}

function updateLangButton() {
  const button = $("#langToggle");
  button.textContent = getLang() === "zh" ? "EN" : "中";
  button.title = t("lang.toggle");
  button.setAttribute("aria-label", t("lang.toggle"));
}

function refreshLanguage() {
  updateLangButton();
  updateEngineUI();
  if (state.engines) updateEngineStatus(state.engines);
  if (state.currentJob) {
    applyJob(state.currentJob);
  } else {
    $("#videoTitle").textContent = t("video.title");
    $("#videoSubtitle").textContent = t("video.subtitle");
  }
  if (viewerLoadedJobId && state.currentJob) {
    $("#viewerSubtitle").textContent = `${state.currentJob.filename} · ${ENGINE_LABEL[state.currentJob.engine] || state.currentJob.engine}`;
    if (lastViewerMeta) {
      $("#viewerMetaChip").textContent = t("viewer.chip", {
        frames: lastViewerMeta.num_frames,
        fps: Number(lastViewerMeta.fps).toFixed(0),
      });
      renderViewerStats(state.currentJob, lastViewerMeta);
    }
  } else if (!state.currentJob) {
    $("#viewerSubtitle").textContent = t("viewer.subtitle");
  }
  if (trackState.loaded && state.currentJob) {
    const hasSkeleton = overlay.hasSkeleton();
    const engine = state.currentJob.engine;
    $("#overlayNote").textContent = hasSkeleton
      ? (engine === "gemx" ? t("video.overlay.gemx")
        : engine === "tdpt" ? t("video.overlay.tdpt") : t("video.overlay.instant"))
      : engine === "flowhmr" ? t("video.overlay.flow") : "";
  } else if (state.currentJob) {
    $("#overlayNote").textContent = state.currentJob.status === "done"
      ? t("video.overlay.none") : t("video.overlay.reading");
  }
  renderJobs();
}

function applyTheme(theme, persist = true) {
  const value = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = value;
  if (persist) localStorage.setItem("p3d.theme", value);
  $("#themeToggle").innerHTML = value === "dark"
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"/></svg>';
  viewer?.setTheme(value);
}

/* ------------------------------------------------------------------ engine */
function bindEnginePicker() {
  $("#enginePicker").addEventListener("click", (event) => {
    const button = event.target.closest(".engine-btn");
    if (!button) return;
    setEngine(button.dataset.engine);
    updateEngineUI();
  });
  $("#engineHealth").addEventListener("click", () => setTab("settings"));
  $("#engineHint").addEventListener("click", (event) => {
    if (event.target.closest("[data-goto-settings]")) {
      event.preventDefault();
      setTab("settings");
    }
  });
}

function updateEngineUI() {
  $$(".engine-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.engine === state.engine);
  });
  $("#engineChip").textContent = ENGINE_LABEL[state.engine] || state.engine;
  const status = state.engines?.[state.engine];
  const pill = $("#engineHealth");
  const text = $("#engineHealthText");
  const ready = Boolean(status?.job_ready ?? status?.ready);
  if (!status) {
    pill.className = "health-pill";
    text.textContent = t("engine.detecting");
  } else if (ready) {
    pill.className = "health-pill ok";
    text.textContent = t("engine.ready", { engine: ENGINE_LABEL[state.engine] });
  } else {
    const hasEnv = status.root_ok && status.python_ok;
    pill.className = `health-pill ${hasEnv ? "warn" : "bad"}`;
    text.textContent = t("engine.needsSetup", { engine: ENGINE_LABEL[state.engine] });
  }
  updateEngineHint();
  updatePrecisionUI();
  updateStartState();
}

/* ------------------------------------------------------------- precision */
const ENHANCERS = [
  { key: "hamer", id: "flowHamer", label: "job.enhance.hamer" },
  { key: "torso", id: "flowTorso", label: "job.enhance.torso" },
  { key: "crossing", id: "flowCrossing", label: "job.enhance.crossing" },
];
const ENHANCE_REASONS = {
  missing: "job.enhance.reason.missing",
  unsupported: "job.enhance.reason.unsupported",
  gpu_required: "job.enhance.reason.gpu",
};

function bindPrecisionControls() {
  for (const id of ["instantTierJob", "instantFitJob", "flowStepsJob"]) {
    $(`#${id}`)?.addEventListener("change", updatePrecisionUI);
  }
}

function initPrecisionDefaults() {
  if (precisionInitialized || !state.config) return;
  const inst = state.config.engines?.instanthmr || {};
  const tierEl = $("#instantTierJob");
  const fitEl = $("#instantFitJob");
  if (tierEl) tierEl.value = inst.model || "medium";
  if (fitEl) fitEl.value = String(inst.fit ?? 2);
  const smoothEl = $("#instantSmoothJob");
  if (smoothEl) smoothEl.checked = Boolean(inst.smooth);
  const tdptSmoothEl = $("#tdptSmoothJob");
  if (tdptSmoothEl) tdptSmoothEl.checked = Boolean(state.config.engines?.tdpt?.smooth ?? true);
  setFlowStepsValue(Number(state.config.engines?.flowhmr?.steps) || 20);
  precisionInitialized = true;
}

function setFlowStepsValue(steps) {
  const select = $("#flowStepsJob");
  if (!select) return;
  const value = String(steps);
  if (!Array.from(select.options).some((option) => option.value === value)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = t("job.precision.custom", { steps });
    select.appendChild(option);
  }
  select.value = value;
}

function selectedInstantVariant() {
  const status = state.engines?.instanthmr;
  const tier = $("#instantTierJob")?.value;
  const fit = Number($("#instantFitJob")?.value ?? 0);
  return (status?.variants || []).find((v) => v.tier === tier && v.fit === fit) || null;
}

function updatePrecisionUI() {
  const instant = state.engine === "instanthmr";
  const flow = state.engine === "flowhmr";
  show($("#instantPrecisionFields"), instant);
  show($("#instantPrecisionHint"), instant);
  show($("#instantSmoothJobFields"), instant);
  show($("#flowPrecisionFields"), flow);
  show($("#gemxPrecisionFields"), state.engine === "gemx");
  show($("#tdptPrecisionFields"), state.engine === "tdpt");
  const hint = $("#instantPrecisionHint");
  if (instant && hint) {
    const variant = selectedInstantVariant();
    hint.textContent = variant
      ? t(variant.exists ? "job.precision.ready" : "job.precision.missing", { file: variant.file })
      : "";
    hint.className = `hint precision-hint${variant && !variant.exists ? " warn" : ""}`;
  }
  updateEnhanceUI();
  updateStartState();
}

function updateEnhanceUI() {
  const isFlow = state.engine === "flowhmr";
  const status = state.engines?.flowhmr;
  const supported = isFlow && status?.layout === "windows";
  show($("#flowEnhanceFields"), supported);
  show($("#flowEnhanceHint"), supported);
  if (!isFlow || !status) return;
  if (status.layout !== "windows") {
    for (const item of ENHANCERS) {
      const input = $(`#${item.id}`);
      if (input) {
        input.checked = false;
        input.disabled = true;
      }
    }
    return;
  }
  const enhancements = status.enhancements || {};
  const unavailable = [];
  for (const item of ENHANCERS) {
    const input = $(`#${item.id}`);
    if (!input) continue;
    const info = enhancements[item.key] || { available: false, reason: "unsupported" };
    input.disabled = !info.available;
    input.closest(".enhance-item")?.classList.toggle("unavailable", !info.available);
    if (info.available) {
      input.title = "";
    } else {
      input.checked = false;
      const reason = t(ENHANCE_REASONS[info.reason] || "job.enhance.reason.missing");
      input.title = reason;
      const label = t(item.label);
      unavailable.push(getLang() === "zh" ? `${label}（${reason}）` : `${label} (${reason})`);
    }
  }
  const hint = $("#flowEnhanceHint");
  if (hint) {
    const joiner = getLang() === "zh" ? "、" : ", ";
    hint.textContent = unavailable.length
      ? t("job.enhance.unavailable", { items: unavailable.join(joiner) })
      : "";
    hint.className = `hint precision-hint${unavailable.length ? " warn" : ""}`;
  }
}

function updateEngineHint() {
  const hint = $("#engineHint");
  const status = state.engines?.[state.engine];
  if (!status) {
    hint.className = "hint";
    hint.textContent = t("job.detectingEngine");
    return;
  }
  if (status.job_ready ?? status.ready) {
    hint.className = "hint";
    if (state.engine === "instanthmr") {
      hint.textContent = t("job.readyInstant", {
        model: status.selected_model,
        python: status.python_version || "venv",
        device: status.device,
      });
    } else if (state.engine === "gemx") {
      hint.textContent = t("job.readyGemx", {
        device: status.device,
        backend: status.cuda_ok ? "CUDA" : "CPU",
      });
    } else if (state.engine === "tdpt") {
      hint.textContent = t("job.readyTdpt");
    } else {
      const service = status.server?.status === "running" ? t("job.serviceRunning") : t("job.serviceAuto");
      hint.textContent = t("job.readyFlow", { service, port: status.port });
    }
    return;
  }
  const missing = [];
  if (!status.root_ok) missing.push(t("settings.checks.project"));
  if (!status.python_ok) missing.push(t("settings.checks.runtime"));
  if (state.engine === "instanthmr") {
    if (!status.deps_ok) missing.push(t("settings.checks.deps"));
    if (!status.selected_model_ok) missing.push(`${t("settings.checks.model")} ${status.selected_model}`);
    if (!status.body_ok) missing.push("mhr-body.onnx");
  } else if (state.engine === "gemx") {
    if (!status.deps_ok) missing.push(t("settings.checks.deps"));
    if (status.device_warn) missing.push("CUDA");
    for (const model of status.models || []) {
      if (model.required && !model.exists) missing.push(model.label);
    }
  } else if (state.engine === "tdpt") {
    if (!status.deps_ok) missing.push(t("settings.checks.deps"));
    if (!status.model_ok) missing.push(status.model || t("settings.checks.core"));
  } else {
    if (status.inference_runtime && !status.inference_runtime.installed) {
      missing.push(t("settings.checks.inferenceRuntime"));
    }
    for (const model of status.models || []) {
      if (!model.required || model.exists) continue;
      const key = FLOW_MODEL_KEYS[model.label];
      missing.push(key ? t(key) : model.label);
    }
    const vggt = (status.models || []).find((model) => model.label.includes("VGGT"));
    if (vggt && !vggt.exists && !status.static_camera) {
      missing.push(t("job.missingVggt"));
    }
  }
  const joiner = getLang() === "zh" ? "、" : ", ";
  hint.className = "hint warn";
  const prefix = missing.length
    ? t("job.missing", { items: escapeHtml(missing.join(joiner)) })
    : t("job.notReady");
  hint.innerHTML = `${prefix} <a href="#" data-goto-settings>${escapeHtml(t("job.goSettings"))}</a>`;
}

function updateStartState() {
  const button = $("#startJob");
  const status = state.engines?.[state.engine];
  const ready = Boolean(status?.job_ready ?? status?.ready);
  const variant = state.engine === "instanthmr" ? selectedInstantVariant() : null;
  const precisionReady = !variant || variant.exists;
  button.disabled = uploading || !state.selectedFile || !ready || !precisionReady;
  button.textContent = uploading ? t("job.uploadingButton") : t("job.start");
}

/* ------------------------------------------------------------------ file */
function bindFileInput() {
  const dropzone = $("#dropzone");
  const input = $("#fileInput");
  $("#emptyChoose").addEventListener("click", () => input.click());
  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    if (input.files?.[0]) setFile(input.files[0]);
  });
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  }));
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });
  $("#fileRemove").addEventListener("click", () => clearFile());
}

function setFile(file) {
  const suffix = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!ALLOWED.has(suffix)) {
    toast(t("job.unsupported", { ext: suffix }), "error");
    return;
  }
  state.selectedFile = file;
  $("#fileName").textContent = file.name;
  $("#fileSize").textContent = fmtBytes(file.size);
  show($("#fileRow"), true);
  show($("#uploadProgress"), false);
  updateStartState();
}

function clearFile() {
  state.selectedFile = null;
  $("#fileInput").value = "";
  show($("#fileRow"), false);
  show($("#uploadProgress"), false);
  updateStartState();
}

/* ------------------------------------------------------------------ start job */
function bindStart() {
  $("#startJob").addEventListener("click", startJob);
}

async function startJob() {
  if (uploading || !state.selectedFile) return;
  const engine = state.engine;
  const file = state.selectedFile;
  const maxFrames = Number($("#maxFrames").value) || 0;
  const precision = engine === "instanthmr"
    ? {
        model: $("#instantTierJob").value,
        fit: Number($("#instantFitJob").value),
        smooth: $("#instantSmoothJob").checked,
      }
    : engine === "flowhmr"
    ? {
        steps: Number($("#flowStepsJob").value),
        hamer: $("#flowHamer").checked,
        torsoOpt: $("#flowTorso").checked,
        crossing: $("#flowCrossing").checked,
      }
    : engine === "tdpt"
    ? { smooth: $("#tdptSmoothJob").checked }
    : {};
  uploading = true;
  show($("#uploadProgress"), true);
  $("#uploadFill").style.width = "0%";
  updateStartState();
  setTab("video");
  $("#jobProgress").classList.remove("hidden");
  $("#progressStage").textContent = t("video.uploadingStage");
  $("#progressPct").textContent = "0%";
  $("#progressFill").style.width = "0%";
  $("#progressMessage").textContent = t("video.uploading");
  $("#stageList").innerHTML = "";
  show($("#jobStatusPill"), true);
  $("#jobStatusPill").textContent = t("status.uploading");
  $("#jobStatusPill").className = "pill running";
  try {
    const job = await api.uploadJob({
      engine, file, maxFrames, autoFrames: maxFrames === 0, ...precision,
      onProgress: (ratio) => {
        const pct = Math.round(ratio * 100);
        $("#uploadFill").style.width = `${pct}%`;
        $("#progressPct").textContent = `${pct}%`;
        $("#progressFill").style.width = `${pct}%`;
      },
    });
    state.currentJobId = job.id;
    state.currentJob = job;
    personChoice = null;
    personJobId = job.id;
    flowVideoReadyJob = null;
    trackState = { jobId: null, loaded: false, lastTry: 0 };
    overlay.setTrack(null);
    previousStatus.set(job.id, job.status);
    clearFile();
    applyJob(job);
    await refreshJobs();
    toast(t("job.created"), "success");
  } catch (error) {
    toast(error.message, "error", 8000);
    $("#jobStatusPill").classList.add("hidden");
    $("#jobProgress").classList.add("hidden");
  } finally {
    uploading = false;
    show($("#uploadProgress"), false);
    updateStartState();
  }
}

/* ------------------------------------------------------------------ current job */
async function pollCurrentJob() {
  if (!state.currentJobId || jobFetching) return;
  if (state.currentJob && TERMINAL.has(state.currentJob.status)) return;
  jobFetching = true;
  try {
    const job = await api.getJob(state.currentJobId);
    applyJob(job);
    if (TERMINAL.has(job.status)) await refreshJobs();
  } catch (error) {
    console.warn(error);
  } finally {
    jobFetching = false;
  }
}

function applyJob(job) {
  const previous = previousStatus.get(job.id);
  state.currentJob = job;
  state.currentJobId = job.id;
  if (job.engine && state.engine !== job.engine) {
    setEngine(job.engine);
    updateEngineUI();
  }
  if (personJobId !== job.id) {
    personJobId = job.id;
    personChoice = null;
  }
  updateJobHeader(job);
  updateProgress(job);
  if (job.engine === "flowhmr" && job.status === "done") flowVideoReadyJob = job.id;
  ensureVideo(job);
  void ensureTrack(job);
  renderCandidates(job);
  const done = job.status === "done";
  show($("#openResultBtn"), done);
  show($("#openResultBtnInline"), done);
  if (done) {
    $("#downloadNpz").classList.remove("disabled");
    $("#downloadNpz").href = job.urls.npz;
    if (state.tab === "viewer") void ensureViewer(job);
    else if (viewerLoadedJobId !== job.id) show($("#viewerBadge"), true);
    if (previous && previous !== "done") {
      toast(t("job.doneToast"), "success", 5200);
    }
  }
  if (job.status === "error" && previous && previous !== "error") {
    toast(t("job.failedToast", { reason: job.error || job.message }), "error", 8000);
  }
  previousStatus.set(job.id, job.status);
  $("#currentJobChip").textContent = `${job.filename} · ${statusLabel(job.status)}`;
  show($("#currentJobChip"), true);
}

function updateJobHeader(job) {
  $("#videoTitle").textContent = job.filename || t("video.title");
  const subtitle = [ENGINE_LABEL[job.engine] || job.engine, statusLabel(job.status)];
  if (job.num_frames) subtitle.push(t("job.frames", { n: job.num_frames }));
  $("#videoSubtitle").textContent = subtitle.join(" · ");
  const pill = $("#jobStatusPill");
  show(pill, true);
  pill.textContent = statusLabel(job.status);
  pill.className = `pill ${job.status}`;
  if (job.status === "done") pill.className = "pill done";
  else if (job.status === "error") pill.className = "pill error";
  else if (job.status === "awaiting_person") pill.className = "pill awaiting_person";
  else pill.className = "pill running";
}

function updateProgress(job) {
  show($("#jobProgress"), true);
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  $("#progressFill").style.width = `${progress}%`;
  $("#progressPct").textContent = `${progress}%`;
  $("#progressStage").textContent = stageLabel(job.stage || statusLabel(job.status));
  const message = localizedJobMessage(job);
  $("#progressMessage").textContent = message;
  const active = !TERMINAL.has(job.status);
  show($("#cancelJob"), active);
  $("#cancelJob").disabled = !active;
  const stages = job.stages?.length ? job.stages : pseudoStages(job);
  $("#stageList").innerHTML = stages.map((stage) => {
    const status = normalizeStage(stage.status);
    return `<span class="stage-pill ${status}"><i class="stage-dot"></i>${escapeHtml(stageLabel(stage.label || stage.name || ""))}</span>`;
  }).join("");
  const log = $("#jobLog");
  if (job.logs?.length) {
    log.textContent = job.logs.slice(-80).join("\n");
    log.scrollTop = log.scrollHeight;
    show($("#jobLogDetails"), true);
  } else {
    show($("#jobLogDetails"), false);
  }
  if (job.warnings?.length) {
    const warning = getLang() === "en" ? t("job.message.warning") : job.warnings.join("；");
    $("#progressMessage").textContent = `${message}  ⚠ ${warning}`;
  }
}

function localizedJobMessage(job) {
  const raw = job.message || job.error || "";
  if (getLang() !== "en" || !/[一-鿿]/.test(raw)) return raw;
  if (job.status === "done") return t("job.message.done");
  if (job.status === "error") return t("job.message.failed");
  if (job.status === "cancelled") return t("job.message.cancelled");
  if (job.status === "awaiting_person") return t("job.message.waiting");
  return t("job.message.processing");
}

function pseudoStages(job) {
  if (job.engine === "flowhmr") return [];
  const done = job.status === "done";
  const failed = job.status === "error";
  return [
    { label: t("job.stage.prepare"), status: job.progress >= 2 ? "completed" : job.status === "preparing" ? "running" : "pending" },
    { label: t("job.stage.detect"), status: done ? "completed" : failed ? "failed" : job.status === "running" ? "running" : "pending" },
    { label: t("job.stage.write"), status: done ? "completed" : "pending" },
  ];
}

function normalizeStage(status) {
  if (["completed", "done"].includes(status)) return "completed";
  if (["running", "active"].includes(status)) return "running";
  if (["waiting_user", "awaiting_person"].includes(status)) return "waiting_user";
  if (["failed", "error"].includes(status)) return "failed";
  return "";
}

function ensureVideo(job) {
  const ready = job.video_file || (job.engine === "flowhmr" && job.engine_job_id);
  if (!ready) return;
  const version = job.engine === "flowhmr"
    ? (flowVideoReadyJob === job.id ? "v30" : "raw")
    : "final";
  const key = `${job.id}:${version}`;
  const video = $("#video");
  if (video.dataset.jobKey === key) return;
  const sameJob = video.dataset.jobId === job.id;
  const resumeTime = sameJob ? video.currentTime : 0;
  const wasPlaying = sameJob && !video.paused;
  video.dataset.jobId = job.id;
  video.dataset.jobKey = key;
  if (!sameJob) {
    overlay.setTrack(null);
    overlay.setVisible(false);
    trackState = { jobId: null, loaded: false, lastTry: 0 };
    $("#overlayNote").textContent = t("video.overlay.reading");
  }
  show($("#videoEmpty"), false);
  show($("#videoWrap"), true);
  show($("#overlayLegend"), true);
  show($("#legendSkeleton"), job.engine !== "flowhmr");
  video.pause();
  video.removeAttribute("src");
  video.load();
  video.src = `${job.urls.video}?v=${Math.round((job.created || Date.now()) * 1000)}&stage=${version}`;
  video.load();
  if (sameJob && (resumeTime > 0 || wasPlaying)) {
    video.addEventListener("loadedmetadata", () => {
      if (resumeTime > 0) video.currentTime = Math.min(resumeTime, video.duration || resumeTime);
      if (wasPlaying) video.play().catch(() => {});
    }, { once: true });
  }
}

async function ensureTrack(job) {
  if (trackState.jobId === job.id && trackState.loaded) return;
  if (!["running", "done"].includes(job.status)) return;
  const now = performance.now();
  if (job.status !== "done" && now - trackState.lastTry < 3500) return;
  trackState.lastTry = now;
  try {
    const track = await api.getTrack(job);
    trackState = { jobId: job.id, loaded: true, lastTry: now };
    overlay.setTrack(track);
    overlay.setVisible(true);
    show($("#overlayLegend"), true);
    const hasSkeleton = overlay.hasSkeleton();
    show($("#legendSkeleton"), hasSkeleton);
    $("#overlayNote").textContent = hasSkeleton
      ? (job.engine === "gemx" ? t("video.overlay.gemx")
        : job.engine === "tdpt" ? t("video.overlay.tdpt") : t("video.overlay.instant"))
      : job.engine === "flowhmr" ? t("video.overlay.flow") : "";
    if (job.engine === "flowhmr" && flowVideoReadyJob !== job.id) {
      // Detection finished, so the engine now serves its 30 fps analysis video;
      // reload so the boxes and the video share the same timeline.
      flowVideoReadyJob = job.id;
      ensureVideo(job);
    }
  } catch (_) {
    if (job.status === "done") $("#overlayNote").textContent = t("video.overlay.none");
  }
}

function renderCandidates(job) {
  const card = $("#personCard");
  const candidates = job.candidates || [];
  if (job.engine !== "flowhmr" || job.status !== "awaiting_person" || !candidates.length) {
    show(card, false);
    return;
  }
  show(card, true);
  if (personChoice === null || !candidates.some((c) => c.index === personChoice)) {
    personChoice = job.default_person ?? candidates[0].index;
  }
  $("#personGrid").innerHTML = candidates.map((candidate) => {
    const label = t("person.option", { n: candidate.index + 1 });
    const suggested = candidate.index === job.default_person ? t("person.suggested") : "";
    return `<button type="button" class="person-option ${candidate.index === personChoice ? "selected" : ""}"
      data-person="${candidate.index}">
      <img src="${candidate.thumb}" alt="${escapeHtml(label)}" loading="lazy">
      <span>${escapeHtml(label)}${escapeHtml(suggested)}</span>
    </button>`;
  }).join("");
  $("#confirmPerson").disabled = false;
}

function bindJobActions() {
  $("#cancelJob").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      const job = await api.cancelJob(state.currentJobId);
      applyJob(job);
      toast(t("job.cancelRequested"), "warn");
      await refreshJobs();
    } catch (error) {
      toast(error.message, "error");
    }
  });
  $("#personGrid").addEventListener("click", (event) => {
    const button = event.target.closest("[data-person]");
    if (!button) return;
    personChoice = Number(button.dataset.person);
    $$(".person-option", $("#personGrid")).forEach((option) => {
      option.classList.toggle("selected", Number(option.dataset.person) === personChoice);
    });
  });
  $("#confirmPerson").addEventListener("click", async () => {
    if (!state.currentJobId || personChoice === null) return;
    $("#confirmPerson").disabled = true;
    try {
      const job = await api.selectPerson(state.currentJobId, personChoice);
      trackState = { jobId: null, loaded: false, lastTry: 0 };
      overlay.setTrack(null);
      applyJob(job);
      toast(t("person.selected", { n: personChoice + 1 }), "success");
      show($("#personCard"), false);
    } catch (error) {
      toast(error.message, "error");
      $("#confirmPerson").disabled = false;
    }
  });
  $("#refreshJobs").addEventListener("click", () => refreshJobs());
}

/* ------------------------------------------------------------------ jobs list */
async function refreshJobs() {
  if (jobsFetching) return;
  jobsFetching = true;
  try {
    state.jobs = await api.getJobs();
    renderJobs();
  } catch (error) {
    console.warn(error);
  } finally {
    jobsFetching = false;
  }
}

function renderJobs() {
  const container = $("#jobList");
  if (!state.jobs.length) {
    container.innerHTML = `<p class="muted small">${escapeHtml(t("history.empty"))}</p>`;
    return;
  }
  container.innerHTML = state.jobs.map((job) => {
    const status = job.status || "queued";
    const frames = job.num_frames ? `${t("job.frames", { n: job.num_frames })} · ` : "";
    return `<div class="job-item ${job.id === state.currentJobId ? "current" : ""}" data-job-id="${job.id}">
      <span class="job-status-dot ${status}"></span>
      <div class="job-item-main">
        <div class="job-item-title" title="${escapeHtml(job.filename)}">${escapeHtml(job.filename)}</div>
        <div class="job-item-meta">
          <span class="job-engine-tag ${job.engine}">${job.engine === "flowhmr" ? "FLOWHMR" : job.engine === "gemx" ? "GEM-X" : job.engine === "tdpt" ? "TDPT" : "INSTANT"}</span>
          <span>${statusLabel(status)}</span>
          <span>${escapeHtml(frames)}${fmtDate(job.created)}</span>
        </div>
      </div>
      <button type="button" class="job-delete" data-delete="${job.id}" title="${escapeHtml(t("history.delete"))}"
        ${TERMINAL.has(status) ? "" : "disabled"}>×</button>
    </div>`;
  }).join("");
}

async function selectJob(jobId, switchTab = true) {
  try {
    const job = await api.getJob(jobId);
    state.currentJobId = job.id;
    state.currentJob = job;
    personChoice = null;
    personJobId = job.id;
    flowVideoReadyJob = null;
    applyJob(job);
    if (switchTab) setTab("video");
    renderJobs();
  } catch (error) {
    toast(t("job.readFailed", { reason: error.message }), "error");
  }
}

function stageLabel(label) {
  const key = STAGE_KEYS[label];
  return key ? t(key) : label;
}

function bindJobList() {
  $("#jobList").addEventListener("click", async (event) => {
    const deleteButton = event.target.closest("[data-delete]");
    if (deleteButton) {
      event.stopPropagation();
      const jobId = deleteButton.dataset.delete;
      if (!window.confirm(t("job.deleteConfirm"))) return;
      try {
        await api.deleteJob(jobId);
        toast(t("job.deleted"), "success");
        if (state.currentJobId === jobId) resetCurrentJob();
        await refreshJobs();
      } catch (error) {
        toast(error.message, "error");
      }
      return;
    }
    const item = event.target.closest("[data-job-id]");
    if (item) await selectJob(item.dataset.jobId);
  });
}

function resetCurrentJob() {
  state.currentJobId = null;
  state.currentJob = null;
  $("#currentJobChip").classList.add("hidden");
  $("#jobStatusPill").classList.add("hidden");
  $("#jobProgress").classList.add("hidden");
  $("#personCard").classList.add("hidden");
  $("#openResultBtn").classList.add("hidden");
  $("#openResultBtnInline").classList.add("hidden");
  $("#videoTitle").textContent = t("video.title");
  $("#videoSubtitle").textContent = t("video.subtitle");
  $("#video").removeAttribute("src");
  $("#video").load();
  $("#video").dataset.jobId = "";
  $("#video").dataset.jobKey = "";
  flowVideoReadyJob = null;
  show($("#videoWrap"), false);
  show($("#videoEmpty"), true);
  show($("#overlayLegend"), false);
  overlay.setTrack(null);
  viewer.disposeData();
  viewerLoadedJobId = null;
  lastViewerMeta = null;
  stopSyncVideo();
  updatePlaybackToggles();
  show($("#threeContainer"), false);
  show($("#viewerEmpty"), true);
  const emptyHeading = $("#viewerEmpty").querySelector("h3");
  const emptyBody = $("#viewerEmpty").querySelector("p");
  if (emptyHeading) emptyHeading.textContent = t("viewer.empty.title");
  if (emptyBody) emptyBody.textContent = t("viewer.empty.body");
  $("#viewerSubtitle").textContent = t("viewer.subtitle");
  show($("#playbar"), false);
  show($("#viewerStats"), false);
  $("#viewerMetaChip").textContent = "—";
  $("#wireframeBtn").disabled = true;
  $("#resetViewBtn").disabled = true;
  $("#downloadNpz").classList.add("disabled");
  $("#downloadNpz").href = "#";
}

/* ------------------------------------------------------------------ viewer */
function bindViewer() {
  $("#playBtn").addEventListener("click", togglePlayback);
  $("#frameSlider").addEventListener("input", () => {
    const frame = Number($("#frameSlider").value);
    if (syncVideoEnabled) {
      const video = $("#syncVideo");
      if (!video.paused) video.pause();
      viewer.pause();
      viewer.setFrame(frame);
      seekSyncVideo(frame);
      $("#playBtn").textContent = "▶";
      return;
    }
    viewer.pause();
    $("#playBtn").textContent = "▶";
    viewer.setFrame(frame);
  });
  $("#speedSelect").addEventListener("change", () => {
    const speed = currentSpeed();
    viewer.setSpeed(speed);
    $("#syncVideo").playbackRate = speed;
  });
  $("#syncVideoBtn").addEventListener("click", () => setSyncPlayback(!syncVideoEnabled));
  $("#audioBtn").addEventListener("click", toggleAudio);
  $("#lockViewBtn").addEventListener("click", toggleViewLock);
  bindSyncVideo();
  $("#wireframeBtn").addEventListener("click", () => {
    const on = viewer.toggleWireframe();
    $("#wireframeBtn").classList.toggle("primary", on);
  });
  $("#resetViewBtn").addEventListener("click", () => viewer.resetView());
  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || state.tab !== "viewer") return;
    if (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(document.activeElement?.tagName)) return;
    event.preventDefault();
    $("#playBtn").click();
  });
  $("#video").addEventListener("error", () => {
    if (!$("#video").dataset.jobId) return;
    $("#overlayNote").textContent = t("video.overlay.playback");
    show($("#overlayLegend"), true);
  });
  $("#video").addEventListener("loadedmetadata", () => overlay.draw());
  $("#video").addEventListener("timeupdate", () => overlay.draw());
}

function readPref(key) {
  try { return localStorage.getItem(key) === "1"; } catch (_) { return false; }
}

function writePref(key, value) {
  try { localStorage.setItem(key, value ? "1" : "0"); } catch (_) { /* noop */ }
}

function currentSpeed() {
  return Number($("#speedSelect").value) || 1;
}

function syncClock() {
  const video = $("#syncVideo");
  return video.readyState >= 1 ? video.currentTime : null;
}

function togglePlayback() {
  if (!state.currentJob || state.currentJob.status !== "done") return;
  if (!syncVideoEnabled) {
    const playing = viewer.toggle();
    $("#playBtn").textContent = playing ? "⏸" : "▶";
    return;
  }
  const video = $("#syncVideo");
  if (!video.paused) {
    video.pause();
    return;
  }
  if (!video.getAttribute("src")) loadSyncVideo(state.currentJob);
  seekSyncVideo(viewer.frame);
  video.play().catch((error) => {
    toast(t("viewer.syncPlayFailed", { reason: error.message }), "error", 7000);
    setSyncPlayback(false);
  });
}

function bindSyncVideo() {
  const video = $("#syncVideo");
  video.addEventListener("loadedmetadata", fitSyncVideoBox);
  window.addEventListener("resize", fitSyncVideoBox);
  video.addEventListener("play", () => {
    if (!syncVideoEnabled) return;
    viewer.play();
    $("#playBtn").textContent = "⏸";
  });
  video.addEventListener("pause", () => {
    if (!syncVideoEnabled) return;
    viewer.pause();
    $("#playBtn").textContent = "▶";
  });
  video.addEventListener("error", () => {
    if (!syncVideoEnabled) return;
    toast(t("viewer.syncUnavailable"), "warn", 6000);
    setSyncPlayback(false);
  });
}

function setSyncPlayback(enabled) {
  const job = state.currentJob;
  const loaded = Boolean(viewer?.meta && viewerLoadedJobId && viewerLoadedJobId === job?.id);
  let next = Boolean(enabled);
  if (next && !(loaded && job?.status === "done" && job?.urls?.video)) {
    toast(t("viewer.syncUnavailable"), "warn", 6000);
    next = false;
  }
  syncVideoEnabled = next;
  writePref(PREF_SYNC, next);
  viewer.setSyncClock(next ? syncClock : null);
  const video = $("#syncVideo");
  if (next) {
    loadSyncVideo(job);
    video.muted = !audioEnabled;
    video.playbackRate = currentSpeed();
    seekSyncVideo(viewer.frame);
    if (viewer.playing) {
      video.play().catch((error) => {
        toast(t("viewer.syncPlayFailed", { reason: error.message }), "error", 7000);
        setSyncPlayback(false);
      });
    }
  } else if (!video.paused) {
    video.pause();
  }
  updatePlaybackToggles();
}

function toggleAudio() {
  if (!syncVideoEnabled) return;
  audioEnabled = !audioEnabled;
  writePref(PREF_AUDIO, audioEnabled);
  const video = $("#syncVideo");
  video.muted = !audioEnabled;
  if (audioEnabled) video.volume = 1;
  updatePlaybackToggles();
}

function toggleViewLock() {
  viewLocked = !viewLocked;
  writePref(PREF_LOCK, viewLocked);
  viewer.setViewLocked(viewLocked);
  updatePlaybackToggles();
}

function updatePlaybackToggles() {
  const loaded = Boolean(viewer?.meta && viewerLoadedJobId && viewerLoadedJobId === state.currentJobId);
  const syncButton = $("#syncVideoBtn");
  syncButton.disabled = !loaded;
  syncButton.setAttribute("aria-pressed", String(syncVideoEnabled && loaded));
  const audioButton = $("#audioBtn");
  audioButton.disabled = !loaded || !syncVideoEnabled;
  audioButton.setAttribute("aria-pressed", String(audioEnabled));
  const lockButton = $("#lockViewBtn");
  lockButton.disabled = !loaded;
  lockButton.setAttribute("aria-pressed", String(viewLocked));
  show($("#syncVideoWrap"), loaded && syncVideoEnabled);
  if (loaded && syncVideoEnabled) fitSyncVideoBox();
}

function loadSyncVideo(job) {
  if (!job?.urls?.video || !viewer.meta) return;
  const video = $("#syncVideo");
  const main = $("#video");
  const source = main.currentSrc || main.src || job.urls.video;
  if (video.dataset.source !== source) {
    video.dataset.source = source;
    video.src = source;
    video.load();
  }
  video.loop = true;
  video.muted = !audioEnabled;
  video.playbackRate = currentSpeed();
}

function fitSyncVideoBox() {
  const wrap = $("#syncVideoWrap");
  const video = $("#syncVideo");
  const stage = $("#viewerStage");
  if (!video.videoWidth || !video.videoHeight || !stage.clientWidth || !stage.clientHeight) return;
  const ratio = video.videoWidth / video.videoHeight;
  const maxWidth = Math.min(300, Math.max(170, stage.clientWidth * 0.28));
  const maxHeight = Math.max(120, stage.clientHeight * 0.52);
  let width = maxWidth;
  let height = width / ratio;
  if (height > maxHeight) {
    height = maxHeight;
    width = height * ratio;
  }
  wrap.style.width = `${Math.round(width)}px`;
  wrap.style.height = `${Math.round(height)}px`;
}

function seekSyncVideo(frame) {
  const video = $("#syncVideo");
  if (!viewer.meta) return;
  const target = Math.max(0, Number(frame) || 0) / (viewer.fps || 30);
  const apply = () => {
    if (!viewer.meta) return;
    const duration = video.duration;
    const limit = Number.isFinite(duration) && duration > 0 ? Math.max(0, duration - 0.02) : target;
    const time = Math.min(target, limit);
    if (Math.abs(video.currentTime - time) > 0.02) video.currentTime = time;
  };
  if (video.readyState >= 1) apply();
  else if (video.getAttribute("src")) video.addEventListener("loadedmetadata", apply, { once: true });
}

function stopSyncVideo() {
  const video = $("#syncVideo");
  if (!video.paused) video.pause();
  video.removeAttribute("src");
  video.load();
  delete video.dataset.source;
  show($("#syncVideoWrap"), false);
}

async function ensureViewer(job) {
  if (!job || job.status !== "done") return;
  if (viewerLoadedJobId === job.id || viewerLoadingJobId === job.id) {
    show($("#viewerBadge"), false);
    return;
  }
  viewerLoadingJobId = job.id;
  show($("#viewerLoading"), true);
  show($("#viewerEmpty"), false);
  show($("#threeContainer"), true);
  show($("#syncVideoWrap"), false);
  $("#syncVideo").pause();
  try {
    viewer.setActive(true);
    const meta = await viewer.load(job);
    if (!meta) return;
    viewerLoadedJobId = job.id;
    state.viewerReadyJobId = job.id;
    lastViewerMeta = meta;
    show($("#viewerBadge"), false);
    $("#viewerSubtitle").textContent = `${job.filename} · ${ENGINE_LABEL[job.engine] || job.engine}`;
    $("#viewerMetaChip").textContent = t("viewer.chip", {
      frames: meta.num_frames,
      fps: Number(meta.fps).toFixed(0),
    });
    show($("#playbar"), true);
    $("#playBtn").textContent = "▶";
    $("#frameSlider").max = String(Math.max(0, meta.num_frames - 1));
    $("#frameSlider").value = "0";
    $("#fpsLabel").textContent = `${Number(meta.fps).toFixed(0)} fps`;
    $("#wireframeBtn").disabled = false;
    $("#resetViewBtn").disabled = false;
    $("#downloadNpz").classList.remove("disabled");
    $("#downloadNpz").href = job.urls.npz;
    renderViewerStats(job, meta);
    show($("#viewerStats"), true);
    updatePlayUI(0, meta.num_frames, meta.fps);
    viewer.setViewLocked(viewLocked);
    updatePlaybackToggles();
    if (syncVideoEnabled) {
      loadSyncVideo(job);
      seekSyncVideo(0);
    }
  } catch (error) {
    toast(t("viewer.loadFailedToast", { reason: error.message }), "error", 9000);
    show($("#viewerEmpty"), true);
    const heading = $("#viewerEmpty").querySelector("h3");
    const paragraph = $("#viewerEmpty").querySelector("p");
    if (heading) heading.textContent = t("viewer.loadFailed");
    if (paragraph) paragraph.textContent = error.message;
  } finally {
    viewerLoadingJobId = null;
    show($("#viewerLoading"), false);
  }
}

function renderViewerStats(job, meta) {
  const fps = Number(meta.fps) || 30;
  const engine = ENGINE_LABEL[job.engine] || job.engine;
  const chips = [
    engine,
    t("viewer.stats.frames", { n: meta.num_frames }),
    t("viewer.stats.seconds", { n: (meta.num_frames / fps).toFixed(1) }),
    t("viewer.stats.vertices", { n: Number(meta.num_vertices).toLocaleString() }),
    t("viewer.stats.faces", { n: Number(meta.num_faces).toLocaleString() }),
  ];
  if (meta.timings?.elapsed_s) chips.push(t("viewer.stats.processed", { n: meta.timings.elapsed_s }));
  if (meta.model) chips.push(meta.model);
  $("#viewerStats").innerHTML = chips.map((text) => `<span>${escapeHtml(text)}</span>`).join("");
}

function updatePlayUI(frame, total, fps) {
  const slider = $("#frameSlider");
  if (document.activeElement !== slider) slider.value = String(frame);
  slider.max = String(Math.max(0, total - 1));
  const seconds = frame / (Number(fps) || 30);
  const duration = Math.max(0, total - 1) / (Number(fps) || 30);
  $("#timeLabel").textContent = `${seconds.toFixed(2)}s / ${duration.toFixed(2)}s`;
  $("#fpsLabel").textContent = `${Number(fps).toFixed(0)} fps`;
}

/* ------------------------------------------------------------------ engine polling */
async function refreshEngineStatus() {
  if (engineFetching) return;
  engineFetching = true;
  try {
    const engines = await api.getEngines();
    state.engines = engines;
    updateEngineStatus(engines);
    updateEngineUI();
  } catch (error) {
    console.warn(error);
  } finally {
    engineFetching = false;
  }
}
