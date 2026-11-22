const listeners = new Map();

const storedEngine = localStorage.getItem("p3d.engine");
const KNOWN_ENGINES = new Set(["instanthmr", "flowhmr", "gemx"]);

export const state = {
  config: null,
  engines: null,
  engine: KNOWN_ENGINES.has(storedEngine) ? storedEngine : "instanthmr",
  jobs: [],
  currentJobId: null,
  currentJob: null,
  selectedFile: null,
  selectedPerson: null,
  tab: "video",
  viewerJobId: null,
  viewerReadyJobId: null,
  setupTaskId: null,
};

export function on(event, handler) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(handler);
  return () => listeners.get(event)?.delete(handler);
}

export function emit(event, payload) {
  for (const handler of listeners.get(event) || []) {
    try { handler(payload); } catch (error) { console.error(error); }
  }
}

export function setEngine(engine) {
  state.engine = engine;
  try { localStorage.setItem("p3d.engine", engine); } catch (_) { /* noop */ }
  emit("engine", engine);
}
