"""Launch InstantHMR's own live-camera demo (Rerun viewer) in its venv.

The demo is a separate process: OpenCV opens the camera, InstantHMR runs
per frame, and Rerun shows the 2D/3D result. We keep the process handle so the
UI can start/stop it and surface the log when it exits.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from . import engines
from .utils import kill_tree, tail_file

_CUDA_PROBE: dict[str, bool] = {}
_CUDA_LOCK = threading.Lock()


def _cuda_available(python: str) -> bool:
    """Whether this interpreter has a working ONNX Runtime CUDA provider.

    Cached per interpreter. The probe only checks the provider list; the
    launcher also prepends the venv's NVIDIA DLL folders to PATH.
    """
    with _CUDA_LOCK:
        if python in _CUDA_PROBE:
            return _CUDA_PROBE[python]
    code = ("import onnxruntime as o; "
            "print(1 if 'CUDAExecutionProvider' in o.get_available_providers() else 0)")
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True,
                              timeout=30, encoding="utf-8", errors="replace")
        ok = proc.returncode == 0 and proc.stdout.strip().endswith("1")
    except Exception:
        ok = False
    with _CUDA_LOCK:
        _CUDA_PROBE[python] = ok
    return ok


def _cuda_bin_dirs(root: Path, python: str) -> list[str]:
    """The DLL folders run_camera.bat adds: cuDNN and friends are loaded lazily."""
    dirs = [str(Path(python).parent)]
    venv = root / ".venv"
    for pattern in ("Lib/site-packages/nvidia/*/bin", "lib/python*/site-packages/nvidia/*/bin"):
        dirs.extend(str(path) for path in sorted(venv.glob(pattern)) if path.is_dir())
    return dirs


class InstantCamera:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.pid: int | None = None
        self.state = "stopped"
        self.message = "未启动"
        self.camera_index: int | None = None
        self.model = ""
        self.device = ""
        self.started_at: float | None = None
        self.log_path = self.log_dir / "instanthmr-camera.log"

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        with self.lock:
            proc = self.process
            if proc is not None and proc.poll() is not None:
                code = proc.returncode
                self.process = None
                self.pid = None
                if code:
                    self.state = "error"
                    self.message = f"摄像头进程已退出（代码 {code}）"
                else:
                    self.state = "stopped"
                    self.message = "摄像头已关闭"
            data = {
                "status": self.state,
                "message": self.message,
                "pid": self.pid,
                "camera_index": self.camera_index,
                "model": self.model,
                "device": self.device,
                "started_at": self.started_at,
                "log": str(self.log_path),
            }
            if self.state == "error":
                data["tail"] = tail_file(self.log_path, 12)
            return data

    # ------------------------------------------------------------------ process
    def start(self, cfg: dict, camera_index: int | None = None) -> dict:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self.status()

            engine = cfg["engines"]["instanthmr"]
            root = Path(engine.get("root") or "")
            python = engines.instant_python(cfg)
            if not root.is_dir():
                raise RuntimeError(f"InstantHMR 目录不存在: {root}")
            if not python or not Path(python).is_file():
                raise RuntimeError("InstantHMR 虚拟环境不存在，请先在设置页自动配置环境")
            if not (root / "demo.py").is_file():
                raise RuntimeError(f"缺少 InstantHMR 演示脚本: {root / 'demo.py'}")
            if not engines.check_modules(python, ["rerun"]).get("rerun"):
                raise RuntimeError("缺少 rerun-sdk，请在 InstantHMR 环境执行 pip install -r requirements.txt")

            tier = engine.get("model") or "medium"
            fit = int(engine.get("fit") or 0)
            model_path = engines.instant_model_path(cfg, tier, fit)
            if not model_path.is_file():
                raise RuntimeError(f"模型未下载: {model_path.name}，请先在模型表里下载")
            body_path = model_path.with_name("mhr-body.onnx")
            if not body_path.is_file():
                raise RuntimeError(f"缺少 {body_path.name}，请先下载 MHR 体型模型")

            raw_index = engine.get("camera_index", 0) if camera_index is None else camera_index
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("摄像头编号必须是整数") from exc
            if not 0 <= index <= 15:
                raise RuntimeError("摄像头编号应在 0–15 之间")

            device = str(engine.get("device") or "auto")
            if device == "auto":
                device = "cuda" if _cuda_available(python) else "cpu"
            if device not in ("cuda", "tensorrt", "cpu", "coreml"):
                raise RuntimeError(f"不支持的设备: {device}")

            pose_filter = str(engine.get("pose_filter") or "one-euro")
            if pose_filter not in ("one-euro", "alpha-beta", "none", ""):
                raise RuntimeError(f"不支持的姿态滤波: {pose_filter}")

            cmd = [
                python, "-u", "demo.py", "--camera", str(index),
                "--model", str(model_path), "--fit", "0", "--device", device,
                "--detector", "medium", "--detector-stride", "3", "--max-persons", "1",
            ]
            if pose_filter and pose_filter != "none":
                cmd += ["--pose-filter", pose_filter]
            if engine.get("smooth"):
                cmd += ["--smooth"]
            focal = engine.get("focal")
            if focal:
                cmd += ["--focal", str(focal)]

            env = dict(os.environ)
            env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
            env["PATH"] = os.pathsep.join(_cuda_bin_dirs(root, python)
                                          + [env.get("PATH", "")])

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_path.open("a", encoding="utf-8", errors="replace")
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} camera {index} =====\n")
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            flags = 0
            if os.name == "nt":
                flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                         | subprocess.CREATE_NEW_PROCESS_GROUP)
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, creationflags=flags,
                )
            except Exception as exc:
                log.write(f"启动失败: {exc}\n")
                log.close()
                raise RuntimeError(f"无法启动摄像头演示: {exc}") from exc
            log.close()

            self.process = proc
            self.pid = proc.pid
            self.state = "running"
            self.message = f"摄像头 {index} 运行中"
            self.camera_index = index
            self.model = model_path.name
            self.device = device
            self.started_at = time.time()
            return self.status()

    def stop(self) -> dict:
        with self.lock:
            proc = self.process
            self.process = None
            self.pid = None
            self.state = "stopped"
            self.message = "摄像头已关闭"
        if proc is not None:
            kill_tree(proc)
        return self.status()

    def shutdown(self) -> None:
        with self.lock:
            proc = self.process
        if proc is not None:
            kill_tree(proc)
