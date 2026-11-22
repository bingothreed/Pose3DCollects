"""Configuration: defaults, project discovery, persistence.

The hub never imports the engines' Python packages. Everything is resolved by
paths on disk, so each engine can live in its own virtual environment.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
LOGS_DIR = DATA_DIR / "logs"
TMP_DIR = DATA_DIR / "tmp"
CONFIG_PATH = DATA_DIR / "config.json"

# Every engine runner is vendored in this checkout.  Models, virtual
# environments and caches live inside the same folder but are gitignored and
# are prepared by the setup/model-download tasks.
ENGINE_ROOTS = {
    "instanthmr": BASE_DIR / "engines" / "instanthmr",
    "flowhmr": BASE_DIR / "engines" / "flowhmr",
    "gemx": BASE_DIR / "engines" / "gemx",
    "tdpt": BASE_DIR / "engines" / "tdpt",
}


def venv_python(root: Path) -> str:
    """The conventional venv Python for a project (empty when there is none)."""
    options = [root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"]
    for path in options:
        if path.is_file():
            return str(path)
    return ""


def flow_python(root: Path) -> str:
    """FlowHMR's bundled portable runtime, else its venv."""
    bundled = root / "runtime" / "python.exe"
    if bundled.is_file():
        return str(bundled)
    return venv_python(root)


def _defaults() -> dict:
    instant_root = ENGINE_ROOTS["instanthmr"]
    flow_root = ENGINE_ROOTS["flowhmr"]
    gemx_root = ENGINE_ROOTS["gemx"]
    tdpt_root = ENGINE_ROOTS["tdpt"]
    return {
        "app": {
            "name": "Pose3D Collects",
            "theme": "dark",
        },
        "job": {
            "max_frames": 300,
            "auto_frames": False,
        },
        "engines": {
            "instanthmr": {
                "root": str(instant_root),
                "python": venv_python(instant_root),
                "base_python": "py -3.12",
                "models_dir": str(instant_root / "models"),
                "model": "medium",
                "fit": 2,
                "device": "auto",
                "pose_filter": "one-euro",
                "smooth": False,
                "camera_index": 0,
                "focal": None,
                "proxy": "http://127.0.0.1:21882",
            },
            "flowhmr": {
                "root": str(flow_root),
                "python": flow_python(flow_root),
                "base_python": "py -3.10",
                "models_dir": str(flow_root / "models"),
                "port": 60550,
                "device": "auto",
                "static_camera": False,
                "manual_select": True,
                "steps": 20,
                "sam_chunk_size": 16,
                "startup_timeout": 300,
                "proxy": "http://127.0.0.1:21882",
            },
            "gemx": {
                "root": str(gemx_root),
                "python": venv_python(gemx_root),
                "base_python": "py -3.12",
                "models_dir": str(gemx_root / "weights"),
                "soma_dir": str(gemx_root / "assets" / "soma"),
                "checkpoint": "body_s240000.ckpt",
                "detector": "yolox_humanart.onnx",
                "device": "auto",
                "detector_device": "auto",
                "pose_batch": 16,
                "pose_fp16": True,
                "proxy": "http://127.0.0.1:21882",
            },
            "tdpt": {
                "root": str(tdpt_root),
                "python": venv_python(tdpt_root),
                "base_python": "python",
                "models_dir": str(tdpt_root / "models"),
                "model": "MobileNet3D3_6.onnx",
                "device": "cpu",
                "smooth": True,
                "root_smooth": False,
                "kalman": True,
                "kalman_interval": 0.45,
                "kalman_noise": 0.4,
                "lowpass": True,
                "lowpass_order": 7,
                "lowpass_smooth": 0.9,
                "one_euro": True,
                "one_euro_min_cutoff": 3.5,
                "one_euro_beta": 0.005,
                "one_euro_d_cutoff": 1.2,
                "flip_video": False,
                "scale": 0.01,
                "camera_index": 0,
                "proxy": "http://127.0.0.1:21882",
            },
        },
        "ffmpeg": "",
    }


def _merge(base: dict, incoming: dict) -> dict:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def default_config() -> dict:
    return copy.deepcopy(_defaults())


def load_config() -> dict:
    cfg = default_config()
    if CONFIG_PATH.is_file():
        try:
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
        if isinstance(stored, dict):
            _merge(cfg, stored)
    # All engines are vendored in this checkout: force the source root and the
    # model directories so a stale external configuration cannot take over.
    # A working external interpreter is kept only until the local virtual
    # environment has been created by the setup task.
    for engine, root in ENGINE_ROOTS.items():
        eng = cfg["engines"][engine]
        eng["root"] = str(root)
        if not eng.get("models_dir"):
            eng["models_dir"] = str(root / ("weights" if engine == "gemx" else "models"))
        if engine == "gemx" and not eng.get("soma_dir"):
            eng["soma_dir"] = str(root / "assets" / "soma")
        detected = flow_python(root) if engine == "flowhmr" else venv_python(root)
        stored_python = eng.get("python") or ""
        if detected:
            eng["python"] = detected
        elif engine == "flowhmr":
            # The portable runtime cannot configure torch from an embedded
            # source tree; require the Python 3.10 venv created by setup.
            eng["python"] = ""
        elif stored_python and Path(stored_python).is_file():
            eng["python"] = stored_python
        else:
            eng["python"] = ""
    return cfg


def save_config(cfg: dict) -> dict:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(cfg)
    # Never store secrets; there are none today, but keep the file tidy.
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)
    return payload


def ensure_dirs() -> None:
    for path in (DATA_DIR, JOBS_DIR, LOGS_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def public_config(cfg: dict) -> dict:
    """Config safe to send to the browser (all of it today)."""
    return copy.deepcopy(cfg)
