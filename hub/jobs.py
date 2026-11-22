"""Unified job registry: one record shape for every engine."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import config as cfgmod
from . import engines
from .utils import file_info, human_size, kill_tree, now

TERMINAL = {"done", "error", "cancelled"}
ACTIVE = {"queued", "preparing", "running", "awaiting_person", "cancelling"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}

FLOW_STATUS_MAP = {
    "queued": "queued",
    "running": "running",
    "waiting_user": "awaiting_person",
    "done": "done",
    "error": "error",
    "cancel_requested": "cancelling",
    "cancelled": "cancelled",
}


def _mime_for(path: Path) -> str:
    return {
        ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
        ".mpg": "video/mpeg", ".mpeg": "video/mpeg", ".wmv": "video/x-ms-wmv",
    }.get(path.suffix.lower(), "application/octet-stream")


class JobManager:
    def __init__(self, cfg: dict, flow_server):
        self.cfg = cfg
        self.flow = flow_server
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        cfgmod.ensure_dirs()
        self._recover()

    # ------------------------------------------------------------------ storage
    def _dir(self, job_id: str) -> Path:
        return cfgmod.JOBS_DIR / job_id

    def _save(self, job: dict) -> None:
        path = self._dir(job["id"]) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _recover(self) -> None:
        for directory in cfgmod.JOBS_DIR.iterdir() if cfgmod.JOBS_DIR.exists() else []:
            record = directory / "job.json"
            if not record.is_file():
                continue
            try:
                job = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(job, dict) or "id" not in job:
                continue
            if job.get("status") in ACTIVE:
                if job.get("engine") == "flowhmr" and job.get("engine_job_id"):
                    # The FlowHMR server may have kept running and still owns the job.
                    job["message"] = "应用重启，正在重新连接 FlowHMR 任务…"
                else:
                    job.update(status="error", error="应用重启，任务已中断",
                               message="应用重启，任务已中断", finished=now())
                self._save(job)
            self.jobs[job["id"]] = job

    # ------------------------------------------------------------------ create
    def submit(self, engine: str, filename: str, temp_path: Path, params: dict) -> dict:
        if engine not in ("instanthmr", "flowhmr", "gemx", "tdpt"):
            raise RuntimeError(f"未知引擎: {engine}")
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTS:
            raise RuntimeError(f"不支持的视频格式 {suffix or '(无扩展名)'}")
        job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        job_dir = self._dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / f"input{suffix}"
        shutil.move(str(temp_path), input_path)
        job = {
            "id": job_id, "engine": engine, "filename": filename,
            "status": "preparing", "stage": "准备", "message": "任务已创建",
            "error": None, "progress": 0, "created": now(), "started": now(), "finished": None,
            "params": params, "input_file": input_path.name, "video_file": None,
            "engine_job_id": None, "meta": None, "candidates": None,
            "default_person": None, "selected_person": None, "video_size": None,
            "stages": [], "logs": [], "warnings": [], "size": input_path.stat().st_size,
            "size_text": human_size(input_path.stat().st_size),
        }
        with self.lock:
            self.jobs[job_id] = job
            self._save(job)
        target = self._run_instant if engine == "instanthmr" else (
            self._run_gemx if engine == "gemx" else (
                self._run_tdpt if engine == "tdpt" else self._run_flow))
        threading.Thread(target=target, args=(job_id,), daemon=True,
                         name=f"job-{job_id}").start()
        return self.get(job_id)

    def _log(self, job_id: str, line: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            job.setdefault("logs", []).append(str(line)[:1000])
            if len(job["logs"]) > 120:
                del job["logs"][:-120]

    def _update(self, job_id: str, save: bool = True, **kw) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            if job.get("status") in TERMINAL and kw.get("status") not in TERMINAL:
                return
            job.update(kw)
            if save:
                self._save(job)

    # ------------------------------------------------------------------ instant
    def _find_ffmpeg(self) -> str:
        configured = (self.cfg.get("ffmpeg") or "").strip()
        if configured and Path(configured).is_file():
            return configured
        found = shutil.which("ffmpeg")
        if found:
            return found
        roots = [Path(self.cfg["engines"]["flowhmr"]["root"]),
                 Path(self.cfg["engines"]["instanthmr"]["root"])]
        patterns = [
            "runtime/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe",
            ".venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe",
            ".venv/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*",
        ]
        for root in roots:
            for pattern in patterns:
                for candidate in root.glob(pattern):
                    if candidate.is_file():
                        return str(candidate)
        return ""

    def _transcode(self, job_id: str, source: Path, out: Path) -> Path:
        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            self._log(job_id, "未找到 FFmpeg，直接使用原视频")
            with self.lock:
                self.jobs[job_id].setdefault("warnings", []).append(
                    "未找到 FFmpeg：参考视频可能无法在浏览器中播放")
            return source
        self._update(job_id, save=False, stage="转码", message="转码为浏览器可播放的 MP4…")
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            "-fps_mode", "passthrough", str(out),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 60,
                                  encoding="utf-8", errors="replace")
            if proc.returncode == 0 and out.is_file() and out.stat().st_size > 0:
                self._log(job_id, f"转码完成: {human_size(out.stat().st_size)}")
                return out
            self._log(job_id, (proc.stderr or "").strip()[-1200:])
        except Exception as exc:
            self._log(job_id, f"转码失败: {exc}")
        # Fall back to the original file; cv2 can still read it even if the browser cannot.
        with self.lock:
            self.jobs[job_id].setdefault("warnings", []).append(
                "转码失败：参考视频可能无法在浏览器中播放")
        return source

    def _run_instant(self, job_id: str) -> None:
        try:
            with self.lock:
                job = self.jobs[job_id]
                params = dict(job["params"])
                input_path = self._dir(job_id) / job["input_file"]
            status = engines.instant_status(self.cfg)
            tier = params.get("model") or status.get("selected_model") or "medium"
            fit = params.get("fit")
            if fit is None:
                fit = status.get("fit") or 0
            model_path = engines.instant_model_path(self.cfg, tier, int(fit))
            missing = []
            if not status["root_ok"]:
                missing.append("项目目录")
            if not status["python_ok"]:
                missing.append("Python 环境")
            if not status["deps_ok"]:
                missing.append("依赖")
            if not model_path.is_file():
                missing.append(f"模型 {model_path.name}")
            if not status["body_ok"]:
                missing.append("mhr-body.onnx")
            if missing:
                raise RuntimeError("InstantHMR 尚未就绪，缺少: " + "、".join(missing)
                                   + "（到“设置与资源”里一键准备）")
            body_path = Path(status["body"]["path"])
            out_dir = self._dir(job_id)
            video_name = "video.mp4"
            video_path = out_dir / video_name
            self._update(job_id, message="准备参考视频…", stage="准备")
            media = self._transcode(job_id, input_path, video_path)
            if media != video_path:
                video_name = media.name
            self._update(job_id, video_file=video_name, status="running", stage="检测与重建",
                         message="加载模型并处理视频…", progress=2)

            python = status["python"]
            worker = cfgmod.BASE_DIR / "workers" / "instanthmr_worker.py"
            max_frames = int(params.get("max_frames") or 0)
            cmd = [
                python, str(worker), "--project", status["root"], "--video", str(media),
                "--out", str(out_dir), "--model", str(model_path), "--body", str(body_path),
                "--device", str(params.get("device") or status.get("device") or "auto"),
                "--max-frames", str(max_frames),
            ]
            pose_filter = params.get("pose_filter", status.get("pose_filter", "one-euro"))
            if pose_filter and pose_filter != "none":
                cmd += ["--pose-filter", str(pose_filter)]
            if params.get("smooth", status.get("smooth", False)):
                cmd += ["--smooth"]
            focal = params.get("focal")
            if focal:
                cmd += ["--focal", str(focal)]
            self._log(job_id, "$ " + " ".join(cmd))
            env = dict(os.environ)
            env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                cmd, cwd=status["root"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
                stdin=subprocess.DEVNULL,
            )
            with self.lock:
                self.procs[job_id] = proc
            done_payload = None
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        self._log(job_id, line)
                        continue
                    kind = event.get("type")
                    if kind == "progress":
                        frame = int(event.get("frame") or 0)
                        total = int(event.get("total") or 0)
                        with self.lock:
                            current = int(self.jobs[job_id].get("progress") or 0)
                        if total > 0:
                            pct = 2 + int(96 * min(frame / total, 1.0))
                        else:
                            pct = max(current, min(95, 2 + frame // 10))
                        self._update(job_id, save=False, progress=pct,
                                     message=event.get("message") or (
                                         f"处理帧 {frame}/{total}" if total else f"已处理 {frame} 帧"),
                                     stage=event.get("stage") or "检测与重建")
                    elif kind == "log":
                        self._log(job_id, event.get("message", ""))
                    elif kind == "done":
                        done_payload = event.get("meta") or {}
                    elif kind == "error":
                        raise RuntimeError(event.get("message") or "InstantHMR 处理失败")
                else:
                    self._log(job_id, line)
            code = proc.wait()
            with self.lock:
                self.procs.pop(job_id, None)
            with self.lock:
                cancelled = self.jobs[job_id]["status"] == "cancelled"
            if cancelled:
                return
            if code != 0 or done_payload is None:
                tail = "\n".join(self.jobs[job_id].get("logs", [])[-8:])
                raise RuntimeError(f"InstantHMR 进程退出（{code}）\n{tail}")
            meta_path = out_dir / "mesh" / "meta.json"
            if not meta_path.is_file():
                raise RuntimeError("InstantHMR 没有生成 3D 结果")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._update(job_id, status="done", progress=100, stage="完成", message="重建完成",
                         finished=now(), meta=meta,
                         num_frames=meta.get("num_frames"), fps=meta.get("fps"))
        except Exception as exc:
            with self.lock:
                job = self.jobs.get(job_id)
                if job and job.get("status") != "cancelled":
                    job.update(status="error", error=str(exc), message=str(exc), finished=now())
                    self._save(job)
        finally:
            with self.lock:
                self.procs.pop(job_id, None)

    # ------------------------------------------------------------------ gemx
    def _run_gemx(self, job_id: str) -> None:
        try:
            with self.lock:
                job = self.jobs[job_id]
                params = dict(job["params"])
                input_path = self._dir(job_id) / job["input_file"]
            status = engines.gemx_status(self.cfg)
            missing = []
            if not status["root_ok"]:
                missing.append("项目目录")
            if not status["python_ok"]:
                missing.append("Python 环境")
            if not status["deps_ok"]:
                missing.append("依赖")
            layout = engines.gemx_layout(Path(self.cfg["engines"]["gemx"]["root"]),
                                         self.cfg["engines"]["gemx"])
            for label, path in (("Body Pose 模型", layout["checkpoint"]),
                                ("YOLOX 检测器", layout["detector"])):
                if not path.is_file():
                    missing.append(label)
            if not all(p.is_file() for p in layout["soma_files"]):
                missing.append("SOMA 资产")
            if status["device_warn"]:
                raise RuntimeError("当前设置为 CUDA 但 torch 没有可用的 CUDA；"
                                   "请到设置里把设备改为 auto 或 cpu")
            if missing:
                raise RuntimeError("GEM-X 尚未就绪，缺少: " + "、".join(missing)
                                   + "（到“设置与资源”里一键准备）")
            out_dir = self._dir(job_id)
            video_name = "video.mp4"
            video_path = out_dir / video_name
            self._update(job_id, message="准备参考视频…", stage="准备")
            media = self._transcode(job_id, input_path, video_path)
            if media != video_path:
                video_name = media.name
            self._update(job_id, video_file=video_name, status="running", stage="检测与重建",
                         message="加载模型并处理视频…", progress=2)

            python = status["python"]
            worker = cfgmod.BASE_DIR / "workers" / "gemx_worker.py"
            max_frames = int(params.get("max_frames") or 0)
            cmd = [
                python, str(worker), "--project", status["root"], "--video", str(media),
                "--out", str(out_dir), "--checkpoint", str(layout["checkpoint"]),
                "--detector", str(layout["detector"]), "--soma", str(layout["soma_dir"]),
                "--device", str(params.get("device") or status.get("device") or "auto"),
                "--detector-device", str(status.get("detector_device") or "auto"),
                "--pose-batch", str(int(status.get("pose_batch") or 8)),
                "--fp16", "1" if bool(status.get("pose_fp16", True)) else "0",
                "--max-frames", str(max_frames),
            ]
            self._log(job_id, "$ " + " ".join(cmd))
            env = dict(os.environ)
            env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                cmd, cwd=status["root"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
                stdin=subprocess.DEVNULL,
            )
            with self.lock:
                self.procs[job_id] = proc
            done_payload = None
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        self._log(job_id, line)
                        continue
                    kind = event.get("type")
                    if kind == "progress":
                        frame = int(event.get("frame") or 0)
                        total = int(event.get("total") or 0)
                        with self.lock:
                            current = int(self.jobs[job_id].get("progress") or 0)
                        if total > 0:
                            pct = 2 + int(96 * min(frame / total, 1.0))
                        else:
                            pct = max(current, min(95, 2 + frame // 10))
                        self._update(job_id, save=False, progress=pct,
                                     message=event.get("message") or (
                                         f"处理帧 {frame}/{total}" if total else f"已处理 {frame} 帧"),
                                     stage=event.get("stage") or "检测与重建")
                    elif kind == "log":
                        self._log(job_id, event.get("message", ""))
                    elif kind == "done":
                        done_payload = event.get("meta") or {}
                    elif kind == "error":
                        raise RuntimeError(event.get("message") or "GEM-X 处理失败")
                else:
                    self._log(job_id, line)
            code = proc.wait()
            with self.lock:
                self.procs.pop(job_id, None)
            with self.lock:
                cancelled = self.jobs[job_id]["status"] == "cancelled"
            if cancelled:
                return
            if code != 0 or done_payload is None:
                tail = "\n".join(self.jobs[job_id].get("logs", [])[-8:])
                raise RuntimeError(f"GEM-X 进程退出（{code}）\n{tail}")
            meta_path = out_dir / "mesh" / "meta.json"
            if not meta_path.is_file():
                raise RuntimeError("GEM-X 没有生成 3D 结果")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._update(job_id, status="done", progress=100, stage="完成", message="重建完成",
                         finished=now(), meta=meta,
                         num_frames=meta.get("num_frames"), fps=meta.get("fps"))
        except Exception as exc:
            with self.lock:
                job = self.jobs.get(job_id)
                if job and job.get("status") != "cancelled":
                    job.update(status="error", error=str(exc), message=str(exc), finished=now())
                    self._save(job)
        finally:
            with self.lock:
                self.procs.pop(job_id, None)

    # ------------------------------------------------------------------ tdpt
    def _run_tdpt(self, job_id: str) -> None:
        try:
            with self.lock:
                job = self.jobs[job_id]
                params = dict(job["params"])
                input_path = self._dir(job_id) / job["input_file"]
            status = engines.tdpt_status(self.cfg)
            missing = []
            if not status["root_ok"]:
                missing.append("项目目录")
            if not status["python_ok"]:
                missing.append("Python 环境")
            if not status["deps_ok"]:
                missing.append("依赖")
            model_path = engines.tdpt_model_path(self.cfg)
            if not model_path.is_file():
                missing.append(f"模型 {model_path.name}")
            if missing:
                raise RuntimeError("TDPT 尚未就绪，缺少: " + "、".join(missing)
                                   + "（到“设置与资源”里一键准备）")
            out_dir = self._dir(job_id)
            video_name = "video.mp4"
            video_path = out_dir / video_name
            self._update(job_id, message="准备参考视频…", stage="准备")
            media = self._transcode(job_id, input_path, video_path)
            if media != video_path:
                video_name = media.name
            self._update(job_id, video_file=video_name, status="running", stage="检测与重建",
                         message="加载模型并处理视频…", progress=2)

            python = status["python"]
            worker = cfgmod.BASE_DIR / "workers" / "tdpt_worker.py"
            max_frames = int(params.get("max_frames") or 0)
            cmd = [
                python, str(worker), "--project", status["root"], "--video", str(media),
                "--out", str(out_dir), "--model", str(model_path),
                "--max-frames", str(max_frames),
                "--smooth", "1" if params.get("smooth", status.get("smooth", True)) else "0",
                "--root-smooth",
                "1" if params.get("root_smooth",
                                  status.get("root_smooth", False)) else "0",
                "--kalman", "1" if status.get("kalman", True) else "0",
                "--kalman-interval", str(status.get("kalman_interval", 0.45)),
                "--kalman-noise", str(status.get("kalman_noise", 0.4)),
                "--lowpass", "1" if status.get("lowpass", True) else "0",
                "--lowpass-order", str(status.get("lowpass_order", 7)),
                "--lowpass-smooth", str(status.get("lowpass_smooth", 0.9)),
                "--one-euro", "1" if status.get("one_euro", True) else "0",
                "--one-euro-min-cutoff", str(status.get("one_euro_min_cutoff", 3.5)),
                "--one-euro-beta", str(status.get("one_euro_beta", 0.005)),
                "--one-euro-d-cutoff", str(status.get("one_euro_d_cutoff", 1.2)),
                "--flip", "1" if params.get("flip", status.get("flip_video", True)) else "0",
                "--scale", str(params.get("scale", status.get("scale", 0.01))),
            ]
            self._log(job_id, "$ " + " ".join(cmd))
            env = dict(os.environ)
            env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                cmd, cwd=status["root"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
                stdin=subprocess.DEVNULL,
            )
            with self.lock:
                self.procs[job_id] = proc
            done_payload = None
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        self._log(job_id, line)
                        continue
                    kind = event.get("type")
                    if kind == "progress":
                        frame = int(event.get("frame") or 0)
                        total = int(event.get("total") or 0)
                        with self.lock:
                            current = int(self.jobs[job_id].get("progress") or 0)
                        if total > 0:
                            pct = 2 + int(96 * min(frame / total, 1.0))
                        else:
                            pct = max(current, min(95, 2 + frame // 10))
                        self._update(job_id, save=False, progress=pct,
                                     message=event.get("message") or (
                                         f"处理帧 {frame}/{total}" if total else f"已处理 {frame} 帧"),
                                     stage=event.get("stage") or "检测与重建")
                    elif kind == "log":
                        self._log(job_id, event.get("message", ""))
                    elif kind == "done":
                        done_payload = event.get("meta") or {}
                    elif kind == "error":
                        raise RuntimeError(event.get("message") or "TDPT 处理失败")
                else:
                    self._log(job_id, line)
            code = proc.wait()
            with self.lock:
                self.procs.pop(job_id, None)
            with self.lock:
                cancelled = self.jobs[job_id]["status"] == "cancelled"
            if cancelled:
                return
            if code != 0 or done_payload is None:
                tail = "\n".join(self.jobs[job_id].get("logs", [])[-8:])
                raise RuntimeError(f"TDPT 进程退出（{code}）\n{tail}")
            meta_path = out_dir / "mesh" / "meta.json"
            if not meta_path.is_file():
                raise RuntimeError("TDPT 没有生成 3D 结果")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._update(job_id, status="done", progress=100, stage="完成", message="重建完成",
                         finished=now(), meta=meta,
                         num_frames=meta.get("num_frames"), fps=meta.get("fps"))
        except Exception as exc:
            with self.lock:
                job = self.jobs.get(job_id)
                if job and job.get("status") != "cancelled":
                    job.update(status="error", error=str(exc), message=str(exc), finished=now())
                    self._save(job)
        finally:
            with self.lock:
                self.procs.pop(job_id, None)

    # ------------------------------------------------------------------ flowhmr
    def _run_flow(self, job_id: str) -> None:
        try:
            with self.lock:
                job = self.jobs[job_id]
                params = dict(job["params"])
                input_path = self._dir(job_id) / job["input_file"]
                filename = job["filename"]
            self._update(job_id, stage="启动服务", message="启动 FlowHMR 服务（首次可能需要几分钟）…")
            self.flow.ensure(wait=True)
            self._update(job_id, stage="上传", message="上传视频到 FlowHMR…")
            engine_id = self.flow.upload(input_path, filename, params)
            self._update(job_id, engine_job_id=engine_id, status="queued", stage="排队",
                         message="已进入 FlowHMR 队列", progress=0)
            self._log(job_id, f"FlowHMR job: {engine_id}")
        except Exception as exc:
            with self.lock:
                job = self.jobs.get(job_id)
                if job and job.get("status") != "cancelled":
                    job.update(status="error", error=str(exc), message=str(exc), finished=now())
                    self._save(job)

    def _refresh_flow(self, job: dict) -> dict:
        engine_id = job.get("engine_job_id")
        if not engine_id or job.get("status") in TERMINAL:
            return job
        try:
            remote = self.flow.get_job(engine_id)
        except Exception as exc:
            with self.lock:
                current = self.jobs.get(job["id"])
                if current is None:
                    return job
                current["message"] = f"无法读取 FlowHMR 任务: {exc}"
                if "HTTP 404" in str(exc):
                    current.update(status="error", error="FlowHMR 任务已不存在（可能已在 FlowHMR 中删除）",
                                   message="FlowHMR 任务已不存在", finished=now())
                    self._save(current)
            return self.jobs.get(job["id"], job)
        mapped = FLOW_STATUS_MAP.get(remote.get("status"), remote.get("status") or "running")
        stages = []
        for stage in remote.get("stages") or []:
            stages.append({
                "name": stage.get("name"), "label": stage.get("label"),
                "status": stage.get("status"), "progress": stage.get("progress", 0),
            })
        update = {
            "status": mapped,
            "progress": int(remote.get("progress") or 0),
            "message": remote.get("message") or "",
            "stage": next((s["label"] for s in stages if s["status"] in ("running", "waiting_user")),
                          stages[-1]["label"] if stages else "FlowHMR"),
            "stages": stages,
            "candidates": [
                {
                    "index": int(candidate.get("index", i)),
                    "score": candidate.get("score"),
                    "frame_count": len(candidate.get("frames") or []),
                }
                for i, candidate in enumerate(remote.get("candidates") or [])
            ],
            "default_person": remote.get("default_person"),
            "selected_person": remote.get("selected_person"),
            "video_size": remote.get("video_size"),
            "queue_position": remote.get("queue_position"),
        }
        if mapped == "done":
            update["finished"] = remote.get("finished") or now()
            update["num_frames"] = remote.get("num_frames")
            update["has_fbx"] = remote.get("has_fbx")
        if mapped == "error":
            update["error"] = remote.get("error") or remote.get("message") or "FlowHMR 任务失败"
        with self.lock:
            current = self.jobs.get(job["id"])
            if current and current.get("status") not in ("cancelled",):
                current.update(update)
                if mapped in TERMINAL:
                    self._save(current)
                elif now() - getattr(self, "_last_save", 0) > 1.0:
                    self._last_save = now()
                    self._save(current)
            return self.jobs.get(job["id"], job)

    # ------------------------------------------------------------------ public
    def get(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            needs_refresh = job.get("engine") == "flowhmr" and job.get("status") not in TERMINAL
        if needs_refresh:
            self._refresh_flow(job)
        with self.lock:
            return self._serialize(self.jobs[job_id])

    def list(self) -> list[dict]:
        with self.lock:
            ordered = sorted(self.jobs.values(), key=lambda j: j.get("created", 0), reverse=True)
        out = []
        for job in ordered[:60]:
            if job.get("engine") == "flowhmr" and job.get("status") in ACTIVE:
                try:
                    job = self._refresh_flow(job)
                except Exception:
                    pass
            out.append(self._serialize(job, slim=True))
        return out

    def _serialize(self, job: dict, slim: bool = False) -> dict:
        data = {k: v for k, v in job.items() if not k.startswith("_")}
        base = f"/api/jobs/{job['id']}"
        data["urls"] = {
            "video": f"{base}/video",
            "meta": f"{base}/meta",
            "vertices": f"{base}/vertices",
            "faces": f"{base}/faces",
            "track": f"{base}/track",
            "npz": f"{base}/npz",
        }
        if not slim:
            candidates = job.get("candidates") or []
            data["candidates"] = [
                {
                    "index": int(c.get("index", i)),
                    "thumb": f"{base}/person/{int(c.get('index', i))}.jpg",
                    "frames": int(c.get("frame_count", len(c.get("frames") or []))),
                    "score": c.get("score"),
                }
                for i, c in enumerate(candidates)
            ]
            data["result_ready"] = job.get("status") == "done"
            data["job_dir"] = str(self._dir(job["id"]))
            meta = job.get("meta")
            if meta:
                data["meta"] = meta
        else:
            for key in ("logs", "stages", "candidates", "meta", "params", "warnings"):
                data.pop(key, None)
        return data

    def refresh(self, job_id: str) -> dict:
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.get("status") in TERMINAL:
                return self._serialize(job)
            job.update(status="cancelled", message="正在取消…", finished=now())
            self._save(job)
            proc = self.procs.get(job_id)
            engine_id = job.get("engine_job_id")
            engine = job.get("engine")
        if proc is not None:
            kill_tree(proc)
        if engine == "flowhmr" and engine_id:
            try:
                self.flow.cancel_job(engine_id)
            except Exception as exc:
                self._log(job_id, f"取消 FlowHMR 任务失败: {exc}")
        with self.lock:
            job = self.jobs[job_id]
            job["message"] = "已取消"
            self._save(job)
            return self._serialize(job)

    def select(self, job_id: str, person: int) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            engine_id = job.get("engine_job_id")
            engine = job.get("engine")
        if engine != "flowhmr" or not engine_id:
            raise RuntimeError("该任务不需要人物选择")
        self.flow.select_person(engine_id, person)
        with self.lock:
            job["selected_person"] = person
            job["status"] = "running"
            job["message"] = f"已选择人物 #{person}"
            self._save(job)
            return self._serialize(job)

    def delete(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.get("status") in ACTIVE:
                raise RuntimeError("任务仍在运行，请先取消")
            engine_id = job.get("engine_job_id")
            engine = job.get("engine")
            self.jobs.pop(job_id, None)
        if engine == "flowhmr" and engine_id:
            try:
                self.flow.delete_jobs([engine_id])
            except Exception:
                pass
        proc = self.procs.pop(job_id, None)
        kill_tree(proc)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)

    def shutdown(self) -> None:
        for job_id, proc in list(self.procs.items()):
            kill_tree(proc)
        self.procs.clear()


def video_mime(job: dict) -> str:
    if job.get("video_file"):
        return _mime_for(Path(job["video_file"]))
    return _mime_for(Path(job.get("input_file", "video.mp4")))


def file_size_text(path: Path) -> str:
    info = file_info(path)
    return info["size_text"]
