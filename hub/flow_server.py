"""Lifecycle + HTTP client for the FlowHMR web server.

FlowHMR already ships a capable job server and a Three.js result API. We keep
that server in its own process (it needs its own torch/ONNX environment) and
proxy its stable endpoints through the hub.
"""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from .utils import http_json, kill_tree, now, tail_file


class FlowServer:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._ready = threading.Event()
        self._config: dict = {}
        self.state = "stopped"
        self.message = "未启动"
        self.pid: int | None = None
        self.process: subprocess.Popen | None = None
        self.external = False
        self.log_path = self.log_dir / "flowhmr-server.log"
        self._last_probe: tuple[float, bool, dict] | None = None

    # ------------------------------------------------------------------ config
    def configure(self, cfg: dict) -> None:
        engine = cfg["engines"]["flowhmr"]
        with self.lock:
            self._config = {
                "root": engine.get("root", ""),
                "python": engine.get("python", ""),
                "port": int(engine.get("port") or 60550),
                "device": engine.get("device", "auto"),
                "startup_timeout": int(engine.get("startup_timeout") or 300),
            }

    @property
    def port(self) -> int:
        return int(self._config.get("port") or 60550)

    def _base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _url(self, path: str) -> str:
        return self._base() + path

    # ------------------------------------------------------------------ health
    def probe(self, ttl: float = 1.5, timeout: float = 2.0) -> tuple[bool, dict]:
        cached = self._last_probe
        if cached and now() - cached[0] < ttl:
            return cached[1], cached[2]
        try:
            info = http_json(self._url("/api/info"), timeout=timeout)
            ok = isinstance(info, dict) and ("stages" in info or "defaults" in info)
            result = (ok, info if ok else {})
        except Exception:
            result = (False, {})
        self._last_probe = (now(), result[0], result[1])
        return result

    def status(self) -> dict:
        healthy, _ = self.probe(ttl=2.0, timeout=1.5)
        with self.lock:
            state = self.state
            # An externally started server on the same port counts as running.
            if healthy:
                state = "running"
                if self.process is None:
                    self.external = True
            elif state == "running" and self.process is not None and self.process.poll() is not None:
                state = "error"
                self.message = "FlowHMR 服务已退出"
            elif state == "running" and self.process is None:
                state = "stopped"
                self.message = "外部 FlowHMR 服务不可用"
            data = {
                "status": state,
                "message": self.message,
                "port": self.port,
                "url": self._base(),
                "pid": self.pid,
                "external": self.external,
                "log": str(self.log_path),
            }
            if state == "error":
                data["tail"] = tail_file(self.log_path, 12)
            return data

    # ------------------------------------------------------------------ process
    def start(self) -> dict:
        """Start the server without blocking (a monitor thread waits for readiness)."""
        with self.lock:
            healthy, _ = self.probe(ttl=0.0, timeout=1.0)
            if healthy:
                self.state = "running"
                self.message = "使用已在运行的服务"
                self.external = self.process is None
                self._ready.set()
                return self.status()
            if self.process is not None and self.process.poll() is None:
                return self.status()
            root = Path(self._config.get("root") or "")
            python = self._config.get("python") or ""
            if not root.is_dir():
                raise RuntimeError(f"FlowHMR 目录不存在: {root}")
            if not python or not Path(python).is_file():
                raise RuntimeError("FlowHMR 运行环境不存在，请在设置页创建环境或指定 Python")

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_path.open("a", encoding="utf-8", errors="replace")
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting FlowHMR =====\n")
            log.flush()
            env = dict(os.environ)
            env.update({
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": str(root / ".cache" / "huggingface"),
                "TORCH_HOME": str(root / ".cache" / "torch"),
                "FLOWHMR_SAM_COMPILE": "0",
                "FLOWHMR_ATTENTION_BACKEND": "query_batched",
                "FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE": "256",
            })
            cmd = [python, "-u", "app.py", "--host", "127.0.0.1", "--port", str(self.port)]
            device = self._config.get("device") or "auto"
            if device not in ("", "auto"):
                cmd += ["--device", device]
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
            self.process = subprocess.Popen(
                cmd, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, stdin=subprocess.DEVNULL,
            )
            log.close()
            self.pid = self.process.pid
            self.state = "starting"
            self.message = "正在启动 FlowHMR 服务（首次加载模型可能需要几分钟）"
            self.external = False
            self._ready.clear()
        threading.Thread(target=self._monitor, daemon=True, name="flowhmr-monitor").start()
        return self.status()

    def _monitor(self) -> None:
        timeout = self._config.get("startup_timeout") or 300
        deadline = now() + timeout
        while now() < deadline:
            proc = self.process
            if proc is not None and proc.poll() is not None:
                with self.lock:
                    self.state = "error"
                    self.message = f"FlowHMR 启动失败（退出码 {proc.returncode}），查看日志"
                self._ready.set()
                return
            healthy, _ = self.probe(ttl=0.0, timeout=1.5)
            if healthy:
                with self.lock:
                    self.state = "running"
                    self.message = "FlowHMR 服务已就绪"
                self._ready.set()
                return
            time.sleep(1.0)
        with self.lock:
            self.state = "error"
            self.message = "FlowHMR 启动超时，请查看服务日志"
        self._ready.set()

    def ensure(self, wait: bool = True, timeout: float | None = None) -> bool:
        """Make sure the server is up. Raises RuntimeError when it cannot start."""
        healthy, _ = self.probe(ttl=0.5, timeout=1.5)
        if healthy:
            with self.lock:
                self.state = "running"
                if self.process is None:
                    self.external = True
            return True
        with self.lock:
            starting = self.state == "starting" and self.process is not None and self.process.poll() is None
        if not starting:
            self.start()
        if not wait:
            return False
        limit = timeout if timeout is not None else (self._config.get("startup_timeout") or 300)
        if not self._ready.wait(limit):
            raise RuntimeError("等待 FlowHMR 服务启动超时")
        with self.lock:
            if self.state != "running":
                tail = "\n".join(tail_file(self.log_path, 20))
                raise RuntimeError(f"{self.message}\n{tail}")
        return True

    def stop(self) -> dict:
        with self.lock:
            proc = self.process
            self.process = None
            self.pid = None
            self._ready.clear()
        if proc is not None:
            kill_tree(proc)
        with self.lock:
            self.state = "stopped"
            self.message = "已停止" if proc is not None else "外部启动的服务不会被本应用停止"
            self.external = proc is None
        self._last_probe = None
        return self.status()

    def shutdown(self) -> None:
        with self.lock:
            proc = self.process
        if proc is not None:
            kill_tree(proc)

    # ------------------------------------------------------------------ client
    def _check(self) -> None:
        if not self.ensure(wait=True):
            raise RuntimeError("FlowHMR 服务不可用")

    def get_job(self, engine_id: str) -> dict:
        self._check()
        return http_json(self._url(f"/api/jobs/{engine_id}"), timeout=15)

    def cancel_job(self, engine_id: str) -> dict:
        self._check()
        return http_json(self._url(f"/api/jobs/{engine_id}/cancel"), method="POST", payload={}, timeout=20)

    def select_person(self, engine_id: str, person: int) -> dict:
        self._check()
        return http_json(self._url(f"/api/jobs/{engine_id}/select"), method="POST",
                         payload={"person": int(person)}, timeout=20)

    def delete_jobs(self, engine_ids: list[str]) -> dict:
        healthy, _ = self.probe(ttl=0.0, timeout=1.0)
        if not healthy or not engine_ids:
            return {"deleted": [], "skipped": []}
        return http_json(self._url("/api/jobs"), method="DELETE",
                         payload={"ids": engine_ids}, timeout=30)

    def upload(self, video_path: Path, filename: str, params: dict) -> str:
        """Stream a multipart upload to FlowHMR and return its job id."""
        self._check()
        boundary = "----Pose3D" + uuid.uuid4().hex
        fields = []
        for key, value in params.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            fields.append((key, str(value)))
        chunks: list[bytes] = []
        for key, value in fields:
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                f"{value}\r\n".encode("utf-8")
            )
        head = b"".join(chunks)
        file_head = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
            f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        size = len(head) + len(file_head) + video_path.stat().st_size + len(tail)

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=300)
        try:
            conn.putrequest("POST", "/api/upload")
            conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            conn.putheader("Content-Length", str(size))
            conn.endheaders()
            conn.send(head)
            conn.send(file_head)
            with video_path.open("rb") as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    conn.send(block)
            conn.send(tail)
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"FlowHMR 上传失败（HTTP {resp.status}）: "
                                   f"{body.decode('utf-8', 'replace')[:500]}")
            parsed = json.loads(body.decode("utf-8"))
            job_id = parsed.get("job_id")
            if not job_id:
                raise RuntimeError("FlowHMR 未返回任务 ID")
            return job_id
        finally:
            conn.close()

    def open_result(self, engine_id: str, kind: str, range_header: str | None = None,
                    extra_path: str = ""):
        """Open one of FlowHMR's result endpoints for streaming."""
        self._check()
        path = {
            "video": f"/api/jobs/{engine_id}/video",
            "meta": f"/api/jobs/{engine_id}/meta",
            "vertices": f"/api/jobs/{engine_id}/vertices",
            "faces": f"/api/jobs/{engine_id}/faces",
            "track": f"/api/jobs/{engine_id}/track",
            "npz": f"/api/jobs/{engine_id}/npz",
        }.get(kind)
        if kind == "person":
            path = f"/api/jobs/{engine_id}/person/{extra_path}.jpg"
        if path is None:
            raise RuntimeError(f"未知结果类型: {kind}")
        headers = {}
        if range_header:
            headers["Range"] = range_header
        req = urllib.request.Request(self._url(path), headers=headers)
        return urllib.request.urlopen(req, timeout=120)

    def snapshot(self) -> dict:
        return self.status()
