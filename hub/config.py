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
PARENT_DIR = BASE_DIR.parent
TDPT_ROOT = BASE_DIR / "engines" / "tdpt"


def _first_dir(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _instant_candidates() -> list[Path]:
    return [
        PARENT_DIR / "InstantHMR",
        BASE_DIR / "InstantHMR",
        PARENT_DIR / "InstantHMR-main",
    ]


def _gemx_candidates() -> list[Path]:
    return [
        PARENT_DIR / "GEM-X",
        BASE_DIR / "GEM-X",
        PARENT_DIR / "gem-x",
    ]


def _flow_candidates() -> list[Path]:
    candidates = [
        PARENT_DIR / "FlowHMR-Windows" / "FlowHMR",
        PARENT_DIR / "flowhmr",
        PARENT_DIR / "FlowHMR",
        BASE_DIR / "flowhmr",
    ]
    # Prefer a checkout that can actually run on Windows right now.
    ready = [p for p in candidates if (p / "runtime" / "python.exe").is_file()]
    if ready:
        return ready + [p for p in candidates if p not in ready]
    with_models = [p for p in candidates if (p / "models" / "flowhmr_latest").is_dir()]
    return with_models + [p for p in candidates if p not in with_models]


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
    instant_root = _first_dir(_instant_candidates())
    flow_root = _first_dir(_flow_candidates())
    gemx_root = _first_dir(_gemx_candidates())
    # TDPT is a built-in engine.  Keep code, venv and models in this project
    # instead of depending on a neighbouring TDPT checkout.
    tdpt_root = TDPT_ROOT
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
                "base_python": "python",
                "models_dir": str(instant_root / "models"),
                "model": "medium",
                "fit": 2,
                "device": "auto",
                "pose_filter": "one-euro",
                "smooth": False,
                "camera_index": 0,
                "focal": None,
            },
            "flowhmr": {
                "root": str(flow_root),
                "python": flow_python(flow_root),
                "port": 60550,
                "device": "auto",
                "static_camera": False,
                "manual_select": True,
                "steps": 20,
                "sam_chunk_size": 16,
                "startup_timeout": 300,
            },
            "gemx": {
                "root": str(gemx_root),
                "python": venv_python(gemx_root),
                "base_python": "python",
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
    # Migrate legacy external-TDPT configurations.  TDPT is deliberately not
    # configurable as an external engine: a portable Pose3D Collects checkout
    # owns its runner, virtual environment and model directory.
    tdpt = cfg["engines"]["tdpt"]
    tdpt["root"] = str(TDPT_ROOT)
    tdpt["models_dir"] = str(TDPT_ROOT / "models")
    tdpt["python"] = venv_python(TDPT_ROOT)
    # Refresh auto-detected values when the stored ones are empty or stale.
    for engine, key in (("instanthmr", "python"), ("flowhmr", "python"),
                        ("gemx", "python"), ("tdpt", "python")):
        if not cfg["engines"][engine].get(key):
            root = Path(cfg["engines"][engine]["root"])
            cfg["engines"][engine][key] = flow_python(root) if engine == "flowhmr" else venv_python(root)
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
