"""Engine discovery, readiness checks and setup tasks (env + model downloads)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from . import config as cfgmod
from .utils import (file_info, kill_tree, now, probe_python_candidates, python_version,
                    resolve_command)

INSTANT_TIERS = ["nano", "tiny", "small", "medium", "medium-commercial"]
INSTANT_TIER_LABEL = {
    "nano": "Nano · 最快",
    "tiny": "Tiny · 轻量",
    "small": "Small · 均衡",
    "medium": "Medium · 最准",
    "medium-commercial": "Medium Commercial · 可商用",
}
INSTANT_MODULES = ["onnxruntime", "cv2", "numpy", "rfdetr"]
INSTANT_DOWNLOAD_MODULES = ["huggingface_hub"]

GEMX_MODULES = ["torch", "cv2", "numpy", "onnxruntime", "transformers", "soma", "gem"]

TDPT_MODULES = ["onnxruntime", "cv2", "numpy"]
TDPT_MODEL_SINGLE = "MobileNet3D3_6.onnx"
TDPT_MODEL_THREE = "ThreeDPoseTracker_3frames.onnx"
TDPT_MODEL_URL = ("https://raw.githubusercontent.com/digital-standard/ThreeDPoseTracker/"
                  "master/Assets/Model/MobileNet3D3_6.onnx")
GEMX_SOMA_FILES = [
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "correctives_model.pt",
]
GEMX_TORCH_CUDA_WHEEL = ("https://mirrors.aliyun.com/pytorch-wheels/cu128/"
                         "torch-2.10.0%2Bcu128-cp312-cp312-win_amd64.whl")


def instant_model_filename(tier: str, fit: int) -> str:
    return f"instanthmr-{tier}-fit{fit}.onnx" if fit else f"instanthmr-{tier}.onnx"


def instant_models_dir(cfg: dict) -> Path:
    engine = cfg["engines"]["instanthmr"]
    return Path(engine.get("models_dir") or Path(engine["root"]) / "models")


def instant_model_path(cfg: dict, tier: str | None = None, fit: int | None = None) -> Path:
    """Resolve one tier/fit ONNX file; missing values fall back to the config default."""
    engine = cfg["engines"]["instanthmr"]
    tier = tier or engine.get("model") or "medium"
    fit = int(engine.get("fit") or 0) if fit is None else int(fit)
    return instant_models_dir(cfg) / instant_model_filename(tier, fit)


def _module_cache_key(python: str, modules: tuple[str, ...]) -> tuple:
    return (python, modules)


_MODULE_CACHE: dict[tuple, tuple[float, dict]] = {}
_ENGINE_CACHE: dict[str, tuple[float, dict]] = {}


def clear_caches() -> None:
    _MODULE_CACHE.clear()
    _ENGINE_CACHE.clear()
    _CUDA_CACHE.clear()


def check_modules(python: str, modules: list[str], ttl: float = 90.0) -> dict:
    """Import-spec check without importing heavy packages."""
    key = _module_cache_key(python, tuple(modules))
    hit = _MODULE_CACHE.get(key)
    if hit and now() - hit[0] < ttl:
        return hit[1]
    result = {name: False for name in modules}
    if python and Path(python).is_file():
        code = (
            "import importlib.util as u, json\n"
            f"mods = {json.dumps(modules)}\n"
            "out = {}\n"
            "for m in mods:\n"
            "    try:\n"
            "        out[m] = u.find_spec(m) is not None\n"
            "    except Exception:\n"
            "        out[m] = False\n"
            "print(json.dumps(out))\n"
        )
        try:
            proc = subprocess.run([python, "-c", code], capture_output=True, text=True,
                                  timeout=45, encoding="utf-8", errors="replace")
            if proc.returncode == 0 and proc.stdout.strip():
                parsed = json.loads(proc.stdout.strip().splitlines()[-1])
                if isinstance(parsed, dict):
                    result.update({k: bool(v) for k, v in parsed.items()})
        except Exception:
            pass
    _MODULE_CACHE[key] = (now(), result)
    return result


def instant_python(cfg: dict) -> str:
    engine = cfg["engines"]["instanthmr"]
    python = engine.get("python") or ""
    if python and Path(python).is_file():
        return python
    detected = cfgmod.venv_python(Path(engine["root"]))
    return detected or python


def flow_python(cfg: dict) -> str:
    engine = cfg["engines"]["flowhmr"]
    python = engine.get("python") or ""
    if python and Path(python).is_file():
        return python
    return cfgmod.flow_python(Path(engine["root"]))


def _instant_models(cfg: dict) -> list[dict]:
    engine = cfg["engines"]["instanthmr"]
    models_dir = instant_models_dir(cfg)
    fit = int(engine.get("fit") or 0)
    models = []
    for tier in INSTANT_TIERS:
        name = instant_model_filename(tier, fit)
        path = models_dir / name
        info = file_info(path)
        models.append({
            "id": tier, "label": INSTANT_TIER_LABEL[tier], "file": name,
            "path": str(path), "fit": fit, **info,
            "selected": tier == engine.get("model"),
        })
    body = models_dir / "mhr-body.onnx"
    return models, body


def instant_status(cfg: dict) -> dict:
    engine = cfg["engines"]["instanthmr"]
    root = Path(engine["root"])
    python = instant_python(cfg)
    models, body = _instant_models(cfg)
    models_dir = instant_models_dir(cfg)
    variants = []
    for tier in INSTANT_TIERS:
        for fit_value in (0, 1, 2):
            name = instant_model_filename(tier, fit_value)
            info = file_info(models_dir / name)
            variants.append({
                "tier": tier, "fit": fit_value, "file": name,
                "exists": info["exists"], "size": info["size"],
            })
    selected = next((m for m in models if m["selected"]), models[3] if len(models) > 3 else models[0])
    deps = check_modules(python, INSTANT_MODULES) if python else {m: False for m in INSTANT_MODULES}
    dl_deps = check_modules(python, INSTANT_DOWNLOAD_MODULES) if python else {}
    root_ok = root.is_dir()
    python_ok = bool(python) and Path(python).is_file()
    deps_ok = python_ok and all(deps.values())
    model_ok = selected["exists"]
    body_ok = body.is_file()
    return {
        "id": "instanthmr",
        "name": "InstantHMR",
        "subtitle": "ONNX · 实时单目人体网格",
        "root": str(root),
        "python": python,
        "python_version": python_version(python) if python_ok else "",
        "models_dir": str(Path(engine.get("models_dir") or root / "models")),
        "model": engine.get("model"),
        "fit": int(engine.get("fit") or 0),
        "device": engine.get("device", "auto"),
        "pose_filter": engine.get("pose_filter", "one-euro"),
        "smooth": bool(engine.get("smooth", False)),
        "root_ok": root_ok,
        "python_ok": python_ok,
        "deps": deps,
        "download_deps": dl_deps,
        "deps_ok": deps_ok,
        "models": models,
        "variants": variants,
        "selected_model": selected["id"],
        "selected_model_path": selected["path"],
        "selected_model_ok": model_ok,
        "body": {"file": "mhr-body.onnx", "path": str(body), **file_info(body)},
        "body_ok": body_ok,
        "ready": root_ok and python_ok and deps_ok and model_ok and body_ok,
        "env_kind": "venv" if python_ok else ("project" if root_ok else "missing"),
        "can_create_env": root_ok,
        "setup_running": None,
    }


def flow_layout(root: Path) -> dict:
    """Where the current checkout keeps checkpoints (portable build vs upstream)."""
    if (root / "models" / "flowhmr_latest").is_dir():
        return {
            "kind": "windows",
            "main": root / "models" / "flowhmr_latest" / "flowhmr_latest.ckpt",
            "main_cfg": root / "models" / "flowhmr_latest" / "config.yml",
            "yolox": root / "models" / "yolox" / "yolox_l.pth",
            "sam": root / "models" / "sam-3d-body-dinov3" / "model.ckpt",
            "vggt": root / "models" / "vggt-omega" / "vggt_omega_1b_512.pt",
            "body": [root / "models" / "body_models" / "smplh" / "neutral" / "model.npz",
                     root / "models" / "mano_v1_2" / "models" / "MANO_RIGHT.pkl",
                     root / "models" / "mano_v1_2" / "models" / "MANO_LEFT.pkl"],
            "models_dir": root / "models",
        }
    return {
        "kind": "upstream",
        "main": root / "checkpoints" / "flowhmr_latest" / "flowhmr_latest.ckpt",
        "main_cfg": root / "checkpoints" / "flowhmr_latest" / "config.yml",
        "yolox": root / "ckpts" / "yolox" / "yolox_l.pth",
        "sam": root / "ckpts" / "sam-3d-body-dinov3" / "model.ckpt",
        "vggt": root / "ckpts" / "vggt-omega" / "vggt_omega_1b_512.pt",
        "body": [root / "assets" / "body_models" / "smplh" / "neutral" / "model.npz"],
        "models_dir": root / "checkpoints",
    }


def _model_row(label: str, path: Path, required: bool, note: str = "") -> dict:
    info = file_info(path)
    return {"label": label, "path": str(path), "required": required, "note": note, **info}


def _files_exist(*paths: Path) -> bool:
    return all(Path(path).is_file() for path in paths)


_RUNTIME_SLOT_MARKERS = (
    "torch/lib/torch_cuda.dll",
    "torch/lib/torch_cpu.dll",
    "torch/lib/torch_hip.dll",
    "torch/lib/torch_xpu.dll",
)


def _runtime_label(name: str) -> str:
    if not name:
        return ""
    if name == "cpu":
        return "CPU"
    if name == "xpu":
        return "Intel XPU"
    if name == "system":
        return "现有项目环境"
    if name.startswith("cu") and name[2:].isdigit():
        return f"CUDA {name[2:-1]}.{name[-1]}"
    if name.startswith("rocm"):
        return f"ROCm ({name})"
    return name


def _has_torch_slot(site: Path) -> bool:
    return (any((site / marker).is_file() for marker in _RUNTIME_SLOT_MARKERS)
            and (site / "torchvision" / "_C.pyd").is_file())


def flow_runtime_info(root: Path, python: str = "") -> dict | None:
    """Inference runtime state for the Windows portable FlowHMR build.

    The selected slot lives in runtime/environments/<name>/site-packages; a
    ``system`` selection uses torch from the bundled Python or configured venv.
    """
    runtime_python = root / "runtime" / "python.exe"
    if not runtime_python.is_file():
        return None
    env_dir = root / "runtime" / "environments"
    selected = ""
    try:
        data = json.loads((env_dir / "selected.json").read_text(encoding="utf-8"))
        if isinstance(data, dict):
            selected = str(data.get("version") or "")
    except (OSError, ValueError):
        pass
    installed = False
    if selected in ("", "system"):
        sites = [runtime_python.parent / "Lib" / "site-packages"]
        if python:
            sites.append(Path(python).resolve().parent.parent / "Lib" / "site-packages")
        installed = any(_has_torch_slot(site) for site in sites)
        if installed and not selected:
            selected = "system"
    else:
        slot = env_dir / selected
        installed = (slot / "installed.json").is_file() and _has_torch_slot(slot / "site-packages")
    return {
        "kind": "bundled",
        "selected": selected,
        "label": _runtime_label(selected),
        "installed": installed,
        "directory": str(env_dir),
    }


def flow_status(cfg: dict, server_status: dict | None = None) -> dict:
    engine = cfg["engines"]["flowhmr"]
    root = Path(engine["root"])
    layout = flow_layout(root)
    python = flow_python(cfg)
    root_ok = root.is_dir()
    python_ok = bool(python) and Path(python).is_file()
    models = [
        _model_row("FlowHMR 主模型", layout["main"], True, "flowhmr_latest.ckpt"),
        _model_row("主模型配置", layout["main_cfg"], True, "config.yml"),
        _model_row("YOLOX 检测器", layout["yolox"], True),
        _model_row("SAM 3D Body", layout["sam"], True, "人物特征与网格"),
        _model_row("SMPL-H / MANO 体型", layout["body"][0], True, "任一官方体型即可"),
        _model_row("VGGT-Omega 相机", layout["vggt"], False, "静态机位可跳过"),
    ]
    body_ok = any(p.is_file() for p in layout["body"])
    # The body row may point at a missing SMPL-H file while a MANO model exists.
    if body_ok and not models[4]["exists"]:
        present = next((p for p in layout["body"] if p.is_file()), layout["body"][0])
        models[4] = _model_row("SMPL-H / MANO 体型", present, True, "任一官方体型即可")
    core_ok = (layout["main"].is_file() and layout["main_cfg"].is_file()
               and layout["yolox"].is_file() and layout["sam"].is_file() and body_ok)
    vggt_ok = layout["vggt"].is_file()
    enhancements = {
        "hamer": {"available": False, "reason": "unsupported", "requires_gpu": True},
        "torso": {"available": False, "reason": "unsupported"},
        "crossing": {"available": False, "reason": "unsupported"},
    }
    if layout["kind"] == "windows":
        models_root = root / "models"
        hamer_files_ok = _files_exist(
            models_root / "hamer" / "hamer.ckpt",
            models_root / "hamer" / "model_config.yaml",
            models_root / "hamer" / "mano_mean_params.npz",
            models_root / "mano_v1_2" / "models" / "MANO_LEFT.pkl",
            models_root / "mano_v1_2" / "models" / "MANO_RIGHT.pkl",
        ) and (root / "third_party" / "hamer_inference").is_dir()
        torso_ok = _files_exist(models_root / "flowhmr_base" / "flowhmr_base.ckpt",
                                models_root / "flowhmr_base" / "config.yml")
        crossing_ok = (_files_exist(models_root / "sam2" / "sam2.1_hiera_small.pt")
                       and (root / "third_party" / "sam2_runtime").is_dir())
        device_kind = str(engine.get("device") or "auto").split(":", 1)[0]
        hamer_ok = hamer_files_ok and device_kind != "cpu"
        enhancements = {
            "hamer": {
                "available": hamer_ok,
                "reason": "" if hamer_ok else ("gpu_required" if hamer_files_ok else "missing"),
                "requires_gpu": True,
            },
            "torso": {"available": torso_ok, "reason": "" if torso_ok else "missing"},
            "crossing": {"available": crossing_ok, "reason": "" if crossing_ok else "missing"},
        }
    runtime = flow_runtime_info(root, python)
    runtime_ok = runtime is None or runtime["installed"]
    static_camera = bool(engine.get("static_camera"))
    job_ready = root_ok and python_ok and core_ok and (static_camera or vggt_ok) and runtime_ok
    return {
        "id": "flowhmr",
        "name": "FlowHMR",
        "subtitle": "视频动捕 · 时序人体网格",
        "root": str(root),
        "python": python,
        "python_version": python_version(python) if python_ok else "",
        "port": int(engine.get("port") or 60550),
        "device": engine.get("device", "auto"),
        "static_camera": static_camera,
        "manual_select": bool(engine.get("manual_select")),
        "manual_select_forced": layout["kind"] == "windows",
        "layout": layout["kind"],
        "enhancements": enhancements,
        "root_ok": root_ok,
        "python_ok": python_ok,
        "models": models,
        "core_models_ok": core_ok,
        "vggt_ok": vggt_ok,
        "inference_runtime": runtime,
        "runtime_ok": runtime_ok,
        "ready": root_ok and python_ok and core_ok,
        "job_ready": job_ready,
        "env_kind": ("bundled" if (root / "runtime" / "python.exe").is_file()
                     else "venv" if python_ok else "missing"),
        "can_create_env": root_ok,
        "can_download_models": (root / "tools" / "windows" / "download_models.py").is_file(),
        "models_dir": str(layout["models_dir"]),
        "server": server_status or {"status": "unknown"},
        "setup_running": None,
    }


def gemx_layout(root: Path, eng: dict) -> dict:
    models_dir = Path(eng.get("models_dir") or root / "weights")
    soma_dir = Path(eng.get("soma_dir") or root / "assets" / "soma")
    return {
        "models_dir": models_dir,
        "checkpoint": models_dir / (eng.get("checkpoint") or "body_s240000.ckpt"),
        "detector": models_dir / (eng.get("detector") or "yolox_humanart.onnx"),
        "soma_dir": soma_dir,
        "soma_files": [soma_dir / name for name in GEMX_SOMA_FILES],
    }


_CUDA_CACHE: dict[str, tuple[float, bool]] = {}


def _gemx_cuda_available(python: str) -> bool:
    """torch.cuda.is_available(), cached briefly (spawning torch is slow)."""
    if not python or not Path(python).is_file():
        return False
    hit = _CUDA_CACHE.get(python)
    if hit and now() - hit[0] < 300:
        return hit[1]
    code = "import torch; print('1' if torch.cuda.is_available() else '0')"
    result = False
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True,
                              timeout=60, encoding="utf-8", errors="replace")
        result = proc.returncode == 0 and proc.stdout.strip().splitlines()[-1] == "1"
    except Exception:
        pass
    _CUDA_CACHE[python] = (now(), result)
    return result


def gemx_status(cfg: dict) -> dict:
    engine = cfg["engines"]["gemx"]
    root = Path(engine["root"])
    layout = gemx_layout(root, engine)
    python = engine.get("python") or ""
    if not python or not Path(python).is_file():
        python = cfgmod.venv_python(root)
    root_ok = root.is_dir()
    python_ok = bool(python) and Path(python).is_file()
    deps = check_modules(python, GEMX_MODULES) if python_ok else {m: False for m in GEMX_MODULES}
    deps_ok = python_ok and all(deps.values())
    cuda = _gemx_cuda_available(python) if deps_ok else False
    models = [
        _model_row("Body Pose 主模型", layout["checkpoint"], True,
                   "body_s240000.ckpt（约 3.5 GB）"),
        _model_row("YOLOX 人物检测器", layout["detector"], True,
                   "yolox_humanart.onnx"),
    ]
    soma_rows = [_model_row(name, path, True, "SOMA 0.3.3 资产")
                 for name, path in zip(GEMX_SOMA_FILES, layout["soma_files"])]
    soma_ok = _files_exist(*layout["soma_files"])
    device = str(engine.get("device") or "auto")
    device_warn = device == "cuda" and not cuda
    ready = (root_ok and python_ok and deps_ok and all(m["exists"] for m in models) and soma_ok
             and not device_warn)
    return {
        "id": "gemx",
        "name": "GEM-X",
        "subtitle": "NVIDIA · 单目全身姿态",
        "root": str(root),
        "python": python,
        "python_version": python_version(python) if python_ok else "",
        "models_dir": str(layout["models_dir"]),
        "soma_dir": str(layout["soma_dir"]),
        "device": device,
        "detector_device": str(engine.get("detector_device") or "auto"),
        "pose_batch": int(engine.get("pose_batch") or 8),
        "pose_fp16": bool(engine.get("pose_fp16", True)),
        "checkpoint": str(engine.get("checkpoint") or "body_s240000.ckpt"),
        "detector": str(engine.get("detector") or "yolox_humanart.onnx"),
        "proxy": str(engine.get("proxy") or ""),
        "root_ok": root_ok,
        "python_ok": python_ok,
        "deps": deps,
        "deps_ok": deps_ok,
        "cuda_ok": cuda,
        "device_warn": device_warn,
        "models": models + soma_rows,
        "ready": ready,
        "env_kind": "venv" if python_ok else ("project" if root_ok else "missing"),
        "can_create_env": root_ok,
        "setup_running": None,
    }


def tdpt_python(cfg: dict) -> str:
    engine = cfg["engines"]["tdpt"]
    python = engine.get("python") or ""
    if python and Path(python).is_file():
        return python
    detected = cfgmod.venv_python(Path(engine["root"]))
    return detected or python


def tdpt_model_path(cfg: dict) -> Path:
    engine = cfg["engines"]["tdpt"]
    models_dir = Path(engine.get("models_dir") or Path(engine["root"]) / "models")
    return models_dir / (engine.get("model") or TDPT_MODEL_SINGLE)


def tdpt_status(cfg: dict) -> dict:
    engine = cfg["engines"]["tdpt"]
    root = Path(engine["root"])
    python = tdpt_python(cfg)
    models_dir = Path(engine.get("models_dir") or root / "models")
    model_path = tdpt_model_path(cfg)
    root_ok = root.is_dir()
    python_ok = bool(python) and Path(python).is_file()
    deps = check_modules(python, TDPT_MODULES) if python_ok else {m: False for m in TDPT_MODULES}
    deps_ok = python_ok and all(deps.values())
    model_info = file_info(model_path)
    ready = root_ok and python_ok and deps_ok and model_info["exists"]
    known = [
        (TDPT_MODEL_THREE, "TDPT 3 帧模型（ThreeDPoseLibrary）",
         "448px · 三帧窗口 · 本地转换，非自动下载"),
        (TDPT_MODEL_SINGLE, "TDPT 单帧模型（公共）",
         "224px · MobileNet3D3_6 · 约 33 MB"),
    ]
    models = []
    for name, label, note in known:
        path = models_dir / name
        models.append(_model_row(label, path, path == model_path, note))
    if model_path not in (models_dir / name for name, _, _ in known):
        models.append(_model_row("TDPT 自定义模型", model_path, True))
    return {
        "id": "tdpt",
        "name": "TDPT",
        "subtitle": "VNect · 3 帧窗口 / 单帧 3D 骨架",
        "root": str(root),
        "python": python,
        "python_version": python_version(python) if python_ok else "",
        "models_dir": str(models_dir),
        "model": engine.get("model") or TDPT_MODEL_SINGLE,
        "model_path": str(model_path),
        "device": engine.get("device", "cpu"),
        "smooth": bool(engine.get("smooth", True)),
        "root_smooth": bool(engine.get("root_smooth", False)),
        "flip_video": bool(engine.get("flip_video", True)),
        "scale": float(engine.get("scale") or 0.01),
        "camera_index": int(engine.get("camera_index") or 0),
        "proxy": str(engine.get("proxy") or ""),
        "root_ok": root_ok,
        "python_ok": python_ok,
        "deps": deps,
        "deps_ok": deps_ok,
        "models": models,
        "model_ok": model_info["exists"],
        "ready": ready,
        "env_kind": "venv" if python_ok else ("project" if root_ok else "missing"),
        # The env setup task copies the source templates from this repo and
        # creates the directory, so it works even on a fresh machine.
        "can_create_env": True,
        "can_download_models": python_ok,
        "setup_running": None,
    }


def all_status(cfg: dict, server_status: dict | None = None) -> dict:
    return {"instanthmr": instant_status(cfg), "flowhmr": flow_status(cfg, server_status),
            "gemx": gemx_status(cfg), "tdpt": tdpt_status(cfg)}


# --------------------------------------------------------------------------- setup tasks
class SetupTask:
    def __init__(self, title: str, engine: str):
        self.id = "setup_" + uuid.uuid4().hex[:10]
        self.title = title
        self.engine = engine
        self.status = "running"
        self.started = now()
        self.ended: float | None = None
        self.error: str | None = None
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.cancelled = False

    def log(self, text: str) -> None:
        with self.lock:
            for line in str(text).replace("\r", "").splitlines() or [""]:
                self.lines.append(line)
            if len(self.lines) > 4000:
                del self.lines[:len(self.lines) - 4000]

    def view(self, offset: int = 0) -> dict:
        with self.lock:
            lines = self.lines[offset:]
            return {
                "id": self.id, "title": self.title, "engine": self.engine,
                "status": self.status, "error": self.error,
                "started": self.started, "ended": self.ended,
                "lines": lines, "next": offset + len(lines),
            }


class SetupManager:
    def __init__(self):
        self.tasks: dict[str, SetupTask] = {}
        self.lock = threading.Lock()

    def active_for(self, engine: str) -> SetupTask | None:
        with self.lock:
            for task in self.tasks.values():
                if task.engine == engine and task.status == "running":
                    return task
        return None

    def get(self, task_id: str) -> SetupTask | None:
        return self.tasks.get(task_id)

    def start(self, title: str, engine: str, runner, on_success=None) -> SetupTask:
        active = self.active_for(engine)
        if active is not None:
            return active
        task = SetupTask(title, engine)
        with self.lock:
            self.tasks[task.id] = task

        def work():
            try:
                runner(task)
                if task.cancelled:
                    task.status = "cancelled"
                else:
                    if on_success:
                        on_success(task)
                    task.status = "done"
            except Exception as exc:
                task.status = "cancelled" if task.cancelled else "error"
                task.error = str(exc)
                task.log(f"[错误] {exc}")
            finally:
                task.ended = now()
                if task.process is not None and task.process.poll() is None:
                    kill_tree(task.process)

        threading.Thread(target=work, daemon=True, name=f"setup-{engine}").start()
        return task

    def stop(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if task is None or task.status != "running":
            return False
        task.cancelled = True
        kill_tree(task.process)
        return True

    def stop_all(self) -> None:
        for task in list(self.tasks.values()):
            if task.status == "running":
                self.stop(task.id)


def _setup_env(base: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(base or {})
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


def _run_step(task: SetupTask, argv: list[str], cwd: Path | None = None,
              env: dict | None = None) -> None:
    task.log("$ " + " ".join(str(a) for a in argv))
    kwargs: dict = {
        "cwd": str(cwd) if cwd else None,
        "env": env or _setup_env(),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    proc = subprocess.Popen(argv, **kwargs)
    task.process = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        task.log(line.rstrip())
        if task.cancelled:
            kill_tree(proc)
            break
    code = proc.wait()
    task.process = None
    if task.cancelled:
        raise RuntimeError("已取消")
    if code != 0:
        raise RuntimeError(f"命令失败（退出码 {code}）")


def _venv_python(root: Path) -> Path:
    return root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")


def _ensure_venv(task: SetupTask, root: Path, configured_base: str) -> Path:
    venv_python = _venv_python(root)
    if venv_python.is_file():
        task.log(f"使用已有虚拟环境: {venv_python}")
        return venv_python
    candidates = probe_python_candidates(configured_base)
    if not candidates:
        raise RuntimeError("找不到可用的 Python 3.10+，请在设置里填写 base_python 路径")
    base = candidates[0]
    task.log(f"用 {base} 创建虚拟环境: {root / '.venv'}")
    _run_step(task, resolve_command(base) + ["-m", "venv", str(root / ".venv")], cwd=root)
    if not venv_python.is_file():
        raise RuntimeError("虚拟环境创建失败")
    return venv_python


def setup_env_task(cfg: dict, manager: SetupManager, engine: str, update_config,
                   flow_server=None) -> SetupTask:
    if engine == "instanthmr":
        root = Path(cfg["engines"]["instanthmr"]["root"])
        if not root.is_dir():
            raise RuntimeError(f"InstantHMR 目录不存在: {root}")

        def runner(task: SetupTask):
            venv_python = _ensure_venv(task, root, cfg["engines"]["instanthmr"].get("base_python", "python"))
            task.log("升级 pip…")
            _run_step(task, [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=root)
            task.log("安装 requirements.txt（首次可能需要较长时间）…")
            _run_step(task, [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=root)
            task.log("校验运行依赖…")
            clear_caches()
            deps = check_modules(str(venv_python), INSTANT_MODULES, ttl=0)
            missing = [name for name, ok in deps.items() if not ok]
            if missing:
                raise RuntimeError("依赖校验未通过，缺少: " + ", ".join(missing))
            task.log("依赖校验通过: " + " / ".join(INSTANT_MODULES))
            task.log("环境准备完成。")

        def success(task: SetupTask):
            venv_python = _venv_python(root)
            if venv_python.is_file():
                update_config(lambda c: c["engines"]["instanthmr"].__setitem__("python", str(venv_python)))
            clear_caches()

        return manager.start("准备 InstantHMR 环境", engine, runner, success)

    if engine == "flowhmr":
        root = Path(cfg["engines"]["flowhmr"]["root"])
        if not root.is_dir():
            raise RuntimeError(f"FlowHMR 目录不存在: {root}")
        bundled = root / "runtime" / "python.exe"
        if bundled.is_file():
            def runner(task: SetupTask):
                task.log(f"检测到内置运行环境: {bundled}")
                task.log("正在检查推理运行环境（PyTorch）；首次会自动下载匹配显卡的版本。")
                worker = cfgmod.BASE_DIR / "workers" / "flowhmr_runtime.py"
                _run_step(task, [str(bundled), str(worker), "--root", str(root)], cwd=root,
                          env=_setup_env({
                              "HF_HOME": str(root / ".cache" / "huggingface"),
                              "TORCH_HOME": str(root / ".cache" / "torch"),
                          }))
                if not any("[runtime] restart required" in line for line in task.lines):
                    return
                if flow_server is None:
                    task.log("运行环境已更新；FlowHMR 服务下次启动时生效。")
                    return
                status = flow_server.status()
                if status.get("status") not in ("running", "starting"):
                    task.log("运行环境已更新；FlowHMR 服务下次启动时生效。")
                    return
                if status.get("external"):
                    task.log("[警告] FlowHMR 服务由外部启动，请手动重启服务以应用新的运行环境。")
                    return
                task.log("正在重启 FlowHMR 服务以应用新的运行环境…")
                flow_server.stop()
                flow_server.start()
                try:
                    flow_server.ensure(wait=True)
                    task.log("FlowHMR 服务已重启，新的运行环境已生效。")
                except Exception as exc:
                    task.log(f"[警告] FlowHMR 服务重启失败: {exc}")

            return manager.start("自动配置 FlowHMR 环境", engine, runner)

        def runner(task: SetupTask):
            venv_python = _ensure_venv(task, root, cfg["engines"]["flowhmr"].get("base_python", "python"))
            task.log("升级 pip…")
            _run_step(task, [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=root)
            task.log("安装 FlowHMR requirements.txt（含 PyTorch，体积很大）…")
            _run_step(task, [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=root)
            web_req = root / "tools" / "web" / "requirements.txt"
            if web_req.is_file():
                task.log("安装 Web/感知依赖…")
                _run_step(task, [str(venv_python), "-m", "pip", "install", "-r", str(web_req)], cwd=root)
            yolox = root / "third_party" / "YOLOX"
            if yolox.is_dir():
                task.log("安装 YOLOX（editable）…")
                _run_step(task, [str(venv_python), "-m", "pip", "install", "wheel"], cwd=root)
                _run_step(task, [str(venv_python), "-m", "pip", "install", "--no-build-isolation",
                                 "--no-deps", "-e", str(yolox)], cwd=root)
            task.log("环境准备完成。模型请用下方按钮或手动放置。")

        def success(task: SetupTask):
            venv_python = _venv_python(root)
            if venv_python.is_file():
                update_config(lambda c: c["engines"]["flowhmr"].__setitem__("python", str(venv_python)))
            clear_caches()

        return manager.start("自动配置 FlowHMR 环境", engine, runner, success)

    if engine == "gemx":
        root = Path(cfg["engines"]["gemx"]["root"])
        if not root.is_dir():
            raise RuntimeError(f"GEM-X 目录不存在: {root}")

        def has_nvidia_gpu() -> bool:
            try:
                proc = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=15)
                return proc.returncode == 0 and bool(proc.stdout.strip())
            except Exception:
                return False

        def runner(task: SetupTask):
            venv_python = _ensure_venv(task, root, cfg["engines"]["gemx"].get("base_python", "python"))
            task.log("升级 pip…")
            _run_step(task, [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=root)
            if has_nvidia_gpu():
                task.log("检测到 NVIDIA GPU：安装 CUDA 版 PyTorch（约 2.9 GB）…")
                _run_step(task, [str(venv_python), "-m", "pip", "install",
                                 GEMX_TORCH_CUDA_WHEEL], cwd=root)
            else:
                task.log("未检测到 NVIDIA GPU：使用 CPU 版 PyTorch。")
            task.log("安装 GEM-X（含 transformers / onnxruntime 等依赖）…")
            _run_step(task, [str(venv_python), "-m", "pip", "install", "-e", ".[video]"], cwd=root)
            if has_nvidia_gpu():
                task.log("检测到 NVIDIA GPU：把检测器运行时换成 GPU 版 ONNXRuntime…")
                _run_step(task, [str(venv_python), "-m", "pip", "uninstall", "-y", "onnxruntime"], cwd=root)
                _run_step(task, [str(venv_python), "-m", "pip", "install",
                                 "onnxruntime-gpu>=1.20,<2"], cwd=root)
            task.log("校验运行依赖…")
            clear_caches()
            deps = check_modules(str(venv_python), GEMX_MODULES, ttl=0)
            missing = [name for name, ok in deps.items() if not ok]
            if missing:
                raise RuntimeError("依赖校验未通过，缺少: " + ", ".join(missing))
            cuda = _gemx_cuda_available(str(venv_python))
            task.log("依赖校验通过: " + " / ".join(GEMX_MODULES))
            task.log("CUDA 可用。" if cuda else "CUDA 不可用（将使用 CPU）。")
            task.log("环境准备完成。接下来请用「自动下载模型」准备权重。")

        def success(task: SetupTask):
            venv_python = _venv_python(root)
            if venv_python.is_file():
                update_config(lambda c: c["engines"]["gemx"].__setitem__("python", str(venv_python)))
            clear_caches()

        return manager.start("准备 GEM-X 环境", engine, runner, success)

    if engine == "tdpt":
        root = Path(cfg["engines"]["tdpt"]["root"])
        templates = cfgmod.BASE_DIR / "engines" / "tdpt"
        mirror = "https://mirrors.aliyun.com/pypi/simple/"

        def runner(task: SetupTask):
            root.mkdir(parents=True, exist_ok=True)
            task.log(f"写入引擎源码模板（{templates}）…")
            for name in ("tdpt_vnect.py", "demo.py", "requirements.txt"):
                src = templates / name
                if not src.is_file():
                    raise RuntimeError(f"缺少 TDPT 源码模板: {src}")
                shutil.copyfile(src, root / name)
            venv_python = _ensure_venv(task, root, cfg["engines"]["tdpt"].get("base_python", "python"))
            task.log("升级 pip…")
            _run_step(task, [str(venv_python), "-m", "pip", "install", "--upgrade", "pip",
                             "-i", mirror], cwd=root)
            task.log("安装 requirements.txt（onnxruntime / opencv / numpy，走阿里云镜像）…")
            _run_step(task, [str(venv_python), "-m", "pip", "install",
                             "-i", mirror, "-r", "requirements.txt"], cwd=root)
            task.log("校验运行依赖…")
            clear_caches()
            deps = check_modules(str(venv_python), TDPT_MODULES, ttl=0)
            missing = [name for name, ok in deps.items() if not ok]
            if missing:
                raise RuntimeError("依赖校验未通过，缺少: " + ", ".join(missing))
            task.log("依赖校验通过: " + " / ".join(TDPT_MODULES))
            task.log("环境准备完成。接下来请用「自动下载模型」获取权重。")

        def success(task: SetupTask):
            venv_python = _venv_python(root)
            if venv_python.is_file():
                update_config(lambda c: c["engines"]["tdpt"].__setitem__("python", str(venv_python)))
            clear_caches()

        return manager.start("准备 TDPT 环境", engine, runner, success)

    raise RuntimeError(f"unknown engine: {engine}")


def setup_models_task(cfg: dict, manager: SetupManager, engine: str, payload: dict,
                      update_config) -> SetupTask:
    if engine == "instanthmr":
        eng = cfg["engines"]["instanthmr"]
        root = Path(eng["root"])
        python = instant_python(cfg)
        if not python or not Path(python).is_file():
            raise RuntimeError("InstantHMR 虚拟环境不存在，请先创建环境")
        tier = payload.get("model") or eng.get("model") or "medium"
        if tier not in INSTANT_TIERS:
            raise RuntimeError(f"未知模型: {tier}")
        fit = int(payload.get("fit", eng.get("fit") or 0))
        models_dir = Path(eng.get("models_dir") or root / "models")
        models_dir.mkdir(parents=True, exist_ok=True)
        worker = cfgmod.BASE_DIR / "workers" / "download_instanthmr_models.py"

        def runner(task: SetupTask):
            _run_step(task, [python, str(worker), "--models-dir", str(models_dir),
                             "--model", tier, "--fit", str(fit), "--body"],
                      cwd=root, env=_setup_env({
                          "HF_HOME": str(root / ".cache" / "huggingface"),
                      }))
            task.log("模型下载完成。")

        return manager.start(f"下载 InstantHMR · {tier}", engine, runner)

    if engine == "flowhmr":
        eng = cfg["engines"]["flowhmr"]
        root = Path(eng["root"])
        python = flow_python(cfg)
        script = root / "tools" / "windows" / "download_models.py"
        if not python or not Path(python).is_file():
            raise RuntimeError("FlowHMR 运行环境不存在，请先创建环境")
        if not script.is_file():
            raise RuntimeError("该 FlowHMR 版本没有自动下载脚本，请按 docs/install.md 手动准备模型，"
                               "或把已有模型放进模型目录后点“重新检测”")

        def runner(task: SetupTask):
            _run_step(task, [python, str(script)], cwd=root, env=_setup_env({
                "HF_HOME": str(root / ".cache" / "huggingface"),
                "TORCH_HOME": str(root / ".cache" / "torch"),
            }))
            task.log("模型下载完成。")

        return manager.start("下载 FlowHMR 公开模型", engine, runner)

    if engine == "gemx":
        eng = cfg["engines"]["gemx"]
        root = Path(eng["root"])
        python = eng.get("python") or ""
        if not python or not Path(python).is_file():
            python = cfgmod.venv_python(root)
        if not python or not Path(python).is_file():
            raise RuntimeError("GEM-X 虚拟环境不存在，请先创建环境")
        layout = gemx_layout(root, eng)
        worker = cfgmod.BASE_DIR / "workers" / "download_gemx_models.py"
        proxy = str(eng.get("proxy") or "").strip()

        def runner(task: SetupTask):
            env = _setup_env({
                "HF_HOME": str(root / ".cache" / "huggingface"),
            })
            if proxy:
                env.update({"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy})
            _run_step(task, [python, str(worker),
                             "--models-dir", str(layout["models_dir"]),
                             "--soma-dir", str(layout["soma_dir"]),
                             "--proxy", proxy],
                      cwd=root, env=env)
            task.log("模型下载完成。")

        return manager.start("下载 GEM-X 模型（约 4.1 GB）", engine, runner)

    if engine == "tdpt":
        eng = cfg["engines"]["tdpt"]
        root = Path(eng["root"])
        python = tdpt_python(cfg)
        if not python or not Path(python).is_file():
            raise RuntimeError("TDPT 虚拟环境不存在，请先创建环境")
        models_dir = Path(eng.get("models_dir") or root / "models")
        models_dir.mkdir(parents=True, exist_ok=True)
        worker = cfgmod.BASE_DIR / "workers" / "download_tdpt_models.py"
        proxy = str(eng.get("proxy") or "").strip()

        def runner(task: SetupTask):
            _run_step(task, [python, str(worker), "--models-dir", str(models_dir),
                             "--proxy", proxy], cwd=root)
            task.log("模型下载完成。")

        return manager.start("下载 TDPT 模型（约 33 MB）", engine, runner)

    raise RuntimeError(f"unknown engine: {engine}")
