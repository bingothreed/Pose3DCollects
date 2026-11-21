# Pose3D Collects

**English** | [简体中文](README.zh-CN.md)

A local human motion reconstruction workbench that integrates **InstantHMR** and **FlowHMR** into a single interface:

- **Reference Video**: Upload/preview videos with person boxes and 2D keypoints overlaid.
- **3D Rendering**: Play reconstructed human mesh sequences with Three.js — rotate, zoom, scrub the timeline, toggle wireframe, download NPZ. You can also enable **Sync Video** to play the reference video (with an audio toggle) in a small window that follows the mesh, or **Fixed View** to lock the camera in the scene (off by default; the camera follows the person, similar to FlowHMR's "Follow" mode).
- **Settings & Resources**: Detect engine environments, one-click environment setup (the FlowHMR portable build selects and downloads PyTorch for your GPU), automatic/manual model fetching, and start/stop the FlowHMR service.

Each engine runs in its own Python environment; the Hub only handles process orchestration and HTTP proxying, so there are no torch / onnxruntime version conflicts.

> The project folder is currently named `HMRCollects`, while the product name is **Pose3D Collects**. Renaming the folder to `Pose3DCollects` does not affect operation (all paths are relative or configurable).

---

## Quick Start

Double-click **`start.bat`**.

On first run it automatically creates the Hub's `.venv` and installs Flask (the only dependency), then opens <http://127.0.0.1:7860>.

You can also start it manually:

```bat
.venv\Scripts\python.exe app.py --open
.venv\Scripts\python.exe app.py --port 8080
```

Requires Python 3.10+. To skip opening the browser automatically, run `set POSE3D_NO_OPEN=1` before `start.bat`.

### Standalone Window

Double-click **`start-desktop.bat`**:

1. Prefers a native window via pywebview + Edge WebView2 (installs the optional `pywebview` dependency on first use).
2. If WebView2 is unavailable, falls back to an Edge/Chrome `--app` window (no address bar or tabs).
3. Only if that also fails does it use the system default browser.

If a Hub started by `start.bat` is already running, the desktop entry reuses it; closing the desktop window does not stop that Hub. A Hub started by the desktop entry itself stops when the window closes. To check desktop components: `start-desktop.bat --check`.

### English / Chinese Switching

The **EN / 中** button in the top bar switches between Chinese and English, and the choice is saved in browser local storage (desktop windows keep it too). Buttons, statuses, task cards, settings tables, and toasts all switch in sync; logs and dynamic error details emitted by the engines themselves may remain in the engine's original language — see **Task Logs** for the raw text.

---

## First-Time Engine Setup

Open **Settings & Resources** at the top; the page only shows the engine currently selected in the top bar. Switch between **InstantHMR / FlowHMR** in the top bar to view the other engine's environment, models, and settings.

### InstantHMR

These locations are detected automatically by default (editable in the cards):

| Item | Default |
|---|---|
| Project directory | `..\InstantHMR` |
| Virtual environment | `<project>\.venv\Scripts\python.exe` |
| Model directory | `<project>\models` |

1. Click **Auto Setup Environment**: creates `.venv`, runs `pip install -r requirements.txt`, and verifies runtime dependencies.
2. In the model table, choose the default model and fitting method (defaults: `medium` + `fit2`), then click **Download**: fetches `instanthmr-<size>-fit<fit>.onnx` and `mhr-body.onnx` from Hugging Face.
3. Wait until the status becomes **Ready**.

Alternatively, place the model files in the model directory manually, then click **Save Settings** to refresh detection.

### FlowHMR

The **FlowHMR-Windows portable build** (`..\FlowHMR-Windows\FlowHMR`) is preferred by default — it ships with its own Python runtime and model management:

| Item | Default |
|---|---|
| Project directory | `..\FlowHMR-Windows\FlowHMR` |
| Python | `<project>\runtime\python.exe` |
| Port | `60550` |
| Model directory | `<project>\models` |

- The settings page marks where files should go with the **Model Root**; you can copy the path or open/create the folder directly, and each model row has an **Open Folder** action. Manually downloaded models just need to be placed at the full path shown in the row.
- The portable build already includes its environment; clicking **Auto Setup Environment** reuses FlowHMR's built-in runtime manager to automatically select, download, and enable the PyTorch inference runtime for your GPU (NVIDIA CUDA / AMD ROCm / Intel XPU / CPU, with resume support and SHA-256 verification). If the runtime is already ready, it only runs checks; switching runtimes automatically restarts the FlowHMR service.
- If models are missing, click **Download Public Models**, which calls FlowHMR's built-in download script (with SHA-256 verification).
- Body model packs that require login/authorization (MANO / SMPL-H) must be downloaded manually per FlowHMR's `models/README.md`; place them in the corresponding directories, then click **Save & Detect**.
- You can also set the project directory to the upstream `flowhmr` repository in settings; if there is no `runtime`, **Auto Setup Environment** creates a `.venv` and installs upstream dependencies (including PyTorch, a very large download). Upstream models must be prepared per `docs/install.md`.

The service starts automatically when you submit your first FlowHMR job; the first start loads models and may take several minutes. Service logs are at `data/logs/flowhmr-server.log`.

---

## Workflow

1. Select an engine in the top-right corner.
2. Drag a video into the right panel and set the max frames (`0` = all; long videos take longer).
3. Click **Start Reconstruction**.
4. The **Reference Video** page shows progress; for multi-person videos with **Pause for selection on multiple people** enabled, person thumbnails appear for selection.
5. When done, click **View 3D Result** to play the mesh animation on the **3D Rendering** page.

> InstantHMR reconstructs only the single most confident person per video (fast, simple UI); use FlowHMR when you need to reconstruct multiple people separately — it lists person thumbnails for selection.
> With FlowHMR, disabling **Static Camera** requires the VGGT-Omega camera model; if it is missing, the app prompts you to download it or switch to the static camera.

Recent jobs are stored in `data/jobs/<job_id>/` and remain viewable after restarting the Hub. FlowHMR jobs are also stored under its own `output/web_jobs/`.

---

## Directory Layout

```
HMRCollects/
├─ app.py                 Hub: Flask routes, uploads, status, result proxying
├─ desktop.py             Standalone window shell (WebView2 / browser app window)
├─ start.bat              One-click environment + browser launch
├─ start-desktop.bat      One-click environment + standalone window launch
├─ requirements-desktop.txt  Optional pywebview dependency
├─ hub/
│  ├─ config.py           Configuration and project auto-discovery
│  ├─ engines.py          Engine readiness checks, environment/model install tasks
│  ├─ flow_server.py      FlowHMR service lifecycle + HTTP client
│  └─ jobs.py             Unified job model, InstantHMR worker scheduling
├─ workers/
│  ├─ instanthmr_worker.py         Runs inference in the InstantHMR environment, outputs a unified result package
│  └─ download_instanthmr_models.py
├─ templates/index.html
├─ static/css/app.css
├─ static/js/             main / viewer / overlay / settings / api / state
├─ static/vendor/three/   Local Three.js (MIT)
└─ data/                  Config, jobs, logs, temporary files
```

### Unified Result Format

The InstantHMR worker writes results in the same structure as FlowHMR:

```
data/jobs/<id>/
├─ video.mp4         Reference video (transcoded to browser-playable H.264 when FFmpeg is available)
├─ track.json        2D person boxes / keypoints (at video fps)
├─ mesh/
│  ├─ meta.json      num_frames / fps / vertices / faces / bbox / root_trajectory
│  ├─ vertices.bin   float16, frame-major
│  └─ faces.bin      uint32
└─ motion.npz        vertices / joints / joints_2d / boxes / faces / fps
```

FlowHMR results are proxied by the Hub through its `/api/jobs/<id>/meta|vertices|faces|track|video|npz` endpoints, so the frontend does not need to know which engine produced the result.

---

## FAQ

- **The video processes but won't play in the browser**: Install FFmpeg and set its path in settings; the Hub transcodes to H.264 with FFmpeg. FlowHMR bundles FFmpeg (`imageio-ffmpeg`), which is usually found automatically.
- **InstantHMR reports CUDA unavailable**: Change the device to `CPU`, or check `onnxruntime-gpu` and your GPU driver; the worker automatically falls back to CPU if instantiation fails.
- **FlowHMR service fails to start**: Check `data/logs/flowhmr-server.log`; common causes are port conflicts, insufficient VRAM, or incomplete model files. You can also switch the device to `cpu` in settings.
- **Downloads are slow**: FlowHMR models can be downloaded from ModelScope/mirrors — the script prefers mirrors; InstantHMR models can be copied manually between the Hugging Face cache and the model directory.
