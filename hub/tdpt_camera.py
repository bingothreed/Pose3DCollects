"""Launch the TDPT live-camera demo (OpenCV skeleton overlay) in its venv.

Same pattern as InstantCamera: the demo runs as a separate process, the hub
keeps the handle so the UI can start/stop it and surface the log.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from . import engines
from .utils import kill_tree, tail_file


class TDPTCamera:
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
        self.started_at: float | None = None
        self.log_path = self.log_dir / "tdpt-camera.log"

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

            engine = cfg["engines"]["tdpt"]
            root = Path(engine.get("root") or "")
            python = engines.tdpt_python(cfg)
            if not root.is_dir():
                raise RuntimeError(f"TDPT 目录不存在: {root}")
            if not python or not Path(python).is_file():
                raise RuntimeError("TDPT 虚拟环境不存在，请先在设置页自动配置环境")
            if not (root / "demo.py").is_file():
                raise RuntimeError(f"缺少 TDPT 演示脚本: {root / 'demo.py'}")

            model_path = engines.tdpt_model_path(cfg)
            if not model_path.is_file():
                raise RuntimeError(f"模型未下载: {model_path.name}，请先点「自动下载模型」")

            raw_index = engine.get("camera_index", 0) if camera_index is None else camera_index
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("摄像头编号必须是整数") from exc
            if not 0 <= index <= 15:
                raise RuntimeError("摄像头编号应在 0–15 之间")

            cmd = [python, "-u", "demo.py", "--camera", str(index),
                   "--model", str(model_path)]
            if not engine.get("smooth", True):
                cmd += ["--no-smooth"]

            env = dict(os.environ)
            env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})

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
                raise RuntimeError(f"无法启动 TDPT 摄像头演示: {exc}") from exc
            log.close()

            self.process = proc
            self.pid = proc.pid
            self.state = "running"
            self.message = f"摄像头 {index} 运行中"
            self.camera_index = index
            self.model = model_path.name
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
