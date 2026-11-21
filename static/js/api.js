import { t } from "./i18n.js";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch (error) {
    throw new Error(t("api.connectFailed", { reason: error.message }));
  }
  if (!res.ok) {
    let payload = null;
    try { payload = await res.json(); } catch (_) { /* not JSON */ }
    throw new Error(payload?.error || `${res.status} ${res.statusText}`);
  }
  return res;
}

const json = (path, options) => request(path, options).then((r) => r.json());
const postJson = (path, body) => json(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body ?? {}),
});

export const getConfig = () => json("/api/config");
export const saveConfig = (payload) => postJson("/api/config", payload);
export const getEngines = () => json("/api/engines").then((r) => r.engines);
export const getJobs = () => json("/api/jobs").then((r) => r.jobs);
export const getJob = (id) => json(`/api/jobs/${id}`).then((r) => r.job);
export const deleteJob = (id) => json(`/api/jobs/${id}`, { method: "DELETE" });
export const cancelJob = (id) => postJson(`/api/jobs/${id}/cancel`, {});
export const selectPerson = (id, person) => postJson(`/api/jobs/${id}/select`, { person });

export const getMeta = (job) => json(job.urls.meta);
export const getTrack = (job) => json(job.urls.track);

export function uploadJob({ engine, file, maxFrames, autoFrames, model, fit, steps,
                            hamer, torsoOpt, crossing, onProgress }) {
  return new Promise((resolve, reject) => {
    const query = new URLSearchParams({
      engine,
      filename: file.name,
      max_frames: String(maxFrames ?? 0),
      auto_frames: autoFrames ? "1" : "0",
    });
    if (model) query.set("model", model);
    if (fit !== undefined && fit !== null) query.set("fit", String(fit));
    if (steps !== undefined && steps !== null) query.set("steps", String(steps));
    if (hamer !== undefined) query.set("hamer", hamer ? "1" : "0");
    if (torsoOpt !== undefined) query.set("torso_opt", torsoOpt ? "1" : "0");
    if (crossing !== undefined) query.set("crossing_opt", crossing ? "1" : "0");
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api/jobs?${query}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let payload = {};
      try { payload = JSON.parse(xhr.responseText || "{}"); } catch (_) { /* noop */ }
      if (xhr.status >= 200 && xhr.status < 300 && payload.job) resolve(payload.job);
      else reject(new Error(payload.error || t("api.uploadFailed", { status: xhr.status })));
    };
    xhr.onerror = () => reject(new Error(t("api.uploadNetwork")));
    xhr.onabort = () => reject(new Error(t("api.uploadAborted")));
    xhr.send(file);
  });
}

export const startEnvSetup = (engine) => postJson(`/api/setup/${engine}/env`, {});
export const startModelSetup = (engine, payload = {}) => postJson(`/api/setup/${engine}/models`, payload);
export const getSetupTask = (id, offset = 0) => json(`/api/setup/tasks/${id}?offset=${offset}`).then((r) => r.task);
export const stopSetupTask = (id) => postJson(`/api/setup/tasks/${id}/stop`, {});

export const startFlowService = () => postJson("/api/flowhmr/start", {});
export const stopFlowService = () => postJson("/api/flowhmr/stop", {});

export const openFolder = (engine, which) => postJson("/api/open-folder", { engine, which });
export const openPath = (path) => postJson("/api/open-path", { path });
