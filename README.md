# Pose3D Collects

**English** | [简体中文](README.zh-CN.md)

A local human motion reconstruction workbench that integrates **InstantHMR**, **FlowHMR** and **GEM-X** into a single interface:

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

Open **Settings & Resources** at the top; the page only shows the engine currently selected in the top bar. Switch engines in the top bar to view its environment, models and settings. Model directories default to the vendored engine folders and stay editable; each model folder contains a tracked `README.md` describing the expected files. When an engine/model is selected, the job panel and settings card show which environment pieces are missing; an **Auto-configure environment** button appears only while the environment is incomplete (it changes to **Environment ready** afterwards).

### InstantHMR

InstantHMR is vendored in this checkout; the runner, environment and models live together:

| Item | Default |
|---|---|
| Project directory | `engines\instanthmr` |
| Virtual environment | `<project>\.venv\Scripts\python.exe` |
| Model directory | `<project>\models` |

1. Click **Auto Setup Environment**: creates `.venv`, installs CUDA 12.6 PyTorch (from the Aliyun mirror; CPU PyTorch when no NVIDIA GPU is found), runs `pip install -r requirements.txt`, and verifies runtime dependencies. Requires Python 3.12.
2. In the model table, choose the default model and fitting method (defaults: `medium` + `fit2`), then click **Download**: fetches `instanthmr-<size>-fit<fit>.onnx` and `mhr-body.onnx` from Hugging Face.
3. Wait until the status becomes **Ready**.

Alternatively, place the model files in the model directory manually, then click **Save Settings** to refresh detection.

**Task defaults** exposes two smoothing layers: **pose smoothing** (One-Euro by default, filters root rotation and body joint angles before the mesh is built) and **output smoothing** (off by default, applies a 1€ filter to keypoints, mesh and translation). The job panel also has an **Output smoothing** switch that overrides the default for a single job; fast motion may lag slightly when it is enabled.

**Live camera:** set **Camera index** (default 0) in the InstantHMR card and click **Open live camera**. The app launches InstantHMR's own `demo.py --camera` with the current model, fit, device and smoothing settings: every captured frame is inferred immediately and the 2D/3D result updates in InstantHMR's Rerun window — **no recording or second pass**. The detector refreshes every 3rd frame while the body runs every frame. Click **Close live camera** (or exit the app) to stop. The log is `data/logs/instanthmr-camera.log`.

### FlowHMR

The FlowHMR-Windows **source** is vendored under `engines\flowhmr`; its runtime (Python 3.10 + PyTorch) is created locally and its models are downloaded into the same folder:

| Item | Default |
|---|---|
| Project directory | `engines\flowhmr` |
| Python | `<project>\.venv\Scripts\python.exe` (Python 3.10) |
| Port | `60550` |
| Model directory | `<project>\models` |

- The settings page marks where files should go with the **Model Root**; you can copy the path or open/create the folder directly, and each model row has an **Open Folder** action. Manually downloaded models just need to be placed at the full path shown in the row. Changing the model directory takes effect after restarting the FlowHMR service.
- Clicking **Auto Setup Environment** creates a Python 3.10 `.venv`, pre-installs PyTorch/torchvision 2.8.0+cu128 from the Aliyun mirror, then installs `requirements-windows.txt`, the web/perception dependencies and YOLOX. No portable `runtime\` folder is required.
- If models are missing, click **Download Public Models**, which calls FlowHMR's built-in download script (with SHA-256 verification); the full set is about 20 GB.
- Body model packs that require login/authorization (MANO / SMPL-H) must be downloaded manually per FlowHMR's `models/README.md`; place them in the corresponding directories, then click **Save & Detect**.

The service starts automatically when you submit your first FlowHMR job; the first start loads models and may take several minutes. Service logs are at `data/logs/flowhmr-server.log`.

### GEM-X

GEM-X ([NVIDIA](https://github.com/nv-tlabs/GEM-X)) runs **Body Pose** framewise on videos: YOLOX detects and tracks the primary person, then the DINOv3 + SOMA model reconstructs the full body (18,056-vertex SOMA mesh, 77 joints) in camera space. It keeps the source video's frame rate (no 30 Hz normalization — that's the Body Motion pipeline, which is not wired into the workbench yet).

| Item | Default |
|---|---|
| Project directory | `engines\gemx` |
| Python | `<project>\.venv\Scripts\python.exe` |
| Weights directory | `<project>\weights` |
| SOMA assets | `<project>\assets\soma` |

- Click **Auto Setup Environment**: creates `.venv` and installs GEM-X (`pip install -e ".[video]"`). If `nvidia-smi` finds an NVIDIA GPU, it installs the CUDA PyTorch build (`torch 2.10.0+cu128` from the Aliyun mirror, ~2.9 GB) instead of the CPU wheel. Requires Python 3.12 on this machine (the pinned CUDA wheel is cp312).
- Click **Auto Download Models** (~4.1 GB): YOLOX HumanArt detector (OpenMMLab), `body_s240000.ckpt` + `body_config.json` (Hugging Face `nvidia/GEM-X`), and the SOMA 0.3.3 assets (Hugging Face `nvidia/SOMA-X` revision `v0.3.3`, including the corrective weights). The **proxy** field is used for Hugging Face downloads (default `http://127.0.0.1:21882`); the YOLOX archive is fetched directly. Downloads are resumable and skipped when complete.
- Inference needs ~7 GB of VRAM at the default batch size; CPU works but is slow. The first job compiles SOMA's Warp kernels (~10 s) and the cache is reused afterwards. The environment setup installs GPU ONNXRuntime on NVIDIA machines, and the detector defaults to **auto** (CUDA with CPU fallback — the worker puts torch's bundled CUDA DLLs on PATH, which is all ORT-GPU needs).
- **Pose batch** and **fp16 boost** control pose throughput (measured on an RTX 3060: batch 16 + fp16 ≈ 55–60 ms/frame ≈ 17 fps, max vertex change vs fp32 < 0.3 mm). Detection on CUDA is ~70–90 ms/frame, so a 90-frame job finishes in ~30 s end-to-end (≈ 7 fps pipeline) — fast, but not live; GEM-X's DINOv3-H backbone is not a real-time model.
- Speed paths that were tested and rejected: TensorRT (ORT's TRT EP cannot build the YOLOX graph — its TopK K=8400 exceeds TRT's 3840 limit), fp16 ONNX conversion of the detector (no gain on the 3060), and pipelining detection with pose (GPU memory-bandwidth contention made both ~3× slower).
- SOMA on Windows is not part of GEM-X's upstream Linux-validated setup — this workbench integration was verified on an RTX 3060 with driver 621.24 (warp 1.19, usd-core 26.11).

> Currently GEM-X reconstructs only the single most confident (auto-tracked) person per video. Body Motion (temporal, world-space), Hand Pose, and multi-person boxes are not wired into the workbench yet.

---

## Workflow

1. Select an engine in the top-right corner.
2. Drag a video into the right panel and set the max frames (`0` = all; long videos take longer) plus this task's precision:
   - InstantHMR: model tier (Nano → Medium, more accurate as you go) and fit mode (network only / 1 crop / 2 crops).
   - FlowHMR: generation quality (Fast 10 steps / Balanced 20 / Fine 40); more steps refine motion at a higher cost.
   - GEM-X: framewise Body Pose with no extra per-task choices; the device and detector are configured in Settings.
     Optional model enhancements (Windows portable build, when the model is downloaded): HaMeR finger refinement (GPU required), torso motion optimization (`flowhmr_base`), and crossing-person optimization (SAM 2.1). They load extra models and take longer.
   If the selected model is not downloaded, the start button stays disabled and points you to Settings.
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
├─ engines/                Vendored engine sources (models/venvs ignored)
│  ├─ instanthmr/          InstantHMR package + demo.py
│  ├─ flowhmr/             FlowHMR application source + third_party
│  ├─ gemx/                GEM-X package + packaging files
│  └─ tdpt/                TDPT VNect runner + demo.py
├─ hub/
│  ├─ config.py           Configuration and embedded engine paths
│  ├─ engines.py          Engine readiness checks, environment/model install tasks
│  ├─ flow_server.py      FlowHMR service lifecycle + HTTP client
│  └─ jobs.py             Unified job model, engine worker scheduling
├─ workers/               Unified result writers and model downloaders
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
