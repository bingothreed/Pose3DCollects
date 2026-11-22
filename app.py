"""Pose3D Collects — one local UI for InstantHMR and FlowHMR.

Run:
    python app.py                 # http://127.0.0.1:7860
    python app.py --port 8080
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from hub import config as cfgmod
from hub import engines
from hub.flow_server import FlowServer
from hub.instant_camera import InstantCamera
from hub.tdpt_camera import TDPTCamera
from hub.jobs import JobManager, VIDEO_EXTS, video_mime

BASE_DIR = cfgmod.BASE_DIR
CONFIG_LOCK = threading.RLock()
CONFIG = cfgmod.load_config()
cfgmod.ensure_dirs()

SETUP = engines.SetupManager()
FLOW = FlowServer(cfgmod.LOGS_DIR)
FLOW.configure(CONFIG)
CAMERA = InstantCamera(cfgmod.LOGS_DIR)
TDPT_CAMERA = TDPTCamera(cfgmod.LOGS_DIR)
JOBS = JobManager(CONFIG, FLOW)

MAX_UPLOAD = 24 * 1024 * 1024 * 1024  # 24 GiB

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
app.json.ensure_ascii = False


def _save_config() -> dict:
    cfgmod.save_config(CONFIG)
    FLOW.configure(CONFIG)
    engines.clear_caches()
    return cfgmod.public_config(CONFIG)


def _update_config(mutator) -> None:
    with CONFIG_LOCK:
        mutator(CONFIG)
        _save_config()


def _merge_config(payload: dict) -> None:
    """Only accept the known configuration surface."""
    if not isinstance(payload, dict):
        raise ValueError("配置必须是 JSON 对象")

    def put(section: str, key: str, cast=None):
        if section not in payload or not isinstance(payload[section], dict):
            return
        if key not in payload[section]:
            return
        value = payload[section][key]
        if cast is not None and value not in (None, ""):
            value = cast(value)
        CONFIG[section][key] = value

    for key, cast in (("ffmpeg", str),):
        if key in payload:
            CONFIG[key] = cast(payload[key] or "")

    if "app" in payload and isinstance(payload["app"], dict):
        if "theme" in payload["app"]:
            CONFIG["app"]["theme"] = str(payload["app"]["theme"])

    put("job", "max_frames", int)
    put("job", "auto_frames", bool)

    engines_payload = payload.get("engines")
    if not isinstance(engines_payload, dict):
        engines_payload = {}
    for engine in ("instanthmr", "flowhmr", "gemx", "tdpt"):
        if engine not in engines_payload or not isinstance(engines_payload[engine], dict):
            continue
        src = engines_payload[engine]
        dest = CONFIG["engines"][engine]
        for key in ("root", "python", "base_python", "models_dir", "device", "pose_filter"):
            # Source and environment are embedded/managed by the app; the
            # model directory stays user-configurable.
            if key in ("root", "python"):
                continue
            if key in src and src[key] is not None:
                dest[key] = str(src[key]).strip()
        if engine in ("instanthmr", "flowhmr") and "proxy" in src and src["proxy"] is not None:
            dest["proxy"] = str(src["proxy"]).strip()
        if "model" in src and str(src["model"]):
            dest["model"] = str(src["model"])
        if "fit" in src and src["fit"] not in (None, ""):
            dest["fit"] = int(src["fit"])
        if "camera_index" in src and src["camera_index"] not in (None, ""):
            dest["camera_index"] = int(src["camera_index"])
        if "focal" in src:
            dest["focal"] = float(src["focal"]) if src["focal"] not in (None, "") else None
        if "port" in src and src["port"] not in (None, ""):
            dest["port"] = int(src["port"])
        for key in ("static_camera", "manual_select", "smooth"):
            if key in src:
                dest[key] = bool(src[key])
        for key in ("steps", "sam_chunk_size", "startup_timeout"):
            if key in src and src[key] not in (None, ""):
                dest[key] = int(src[key])
        if engine == "gemx":
            for key in ("soma_dir", "checkpoint", "detector", "detector_device", "proxy"):
                if key in src and src[key] is not None:
                    dest[key] = str(src[key]).strip()
        if engine == "tdpt":
            for key in ("proxy",):
                if key in src and src[key] is not None:
                    dest[key] = str(src[key]).strip()
            if "flip_video" in src:
                dest["flip_video"] = bool(src["flip_video"])
            for key in ("root_smooth", "kalman", "lowpass", "one_euro"):
                if key in src:
                    dest[key] = bool(src[key])
            float_ranges = {
                "kalman_interval": (0.0001, 10.0),
                "kalman_noise": (0.0, 100.0),
                "lowpass_smooth": (0.0, 1.0),
                "one_euro_min_cutoff": (0.0001, 100.0),
                "one_euro_beta": (0.0, 100.0),
                "one_euro_d_cutoff": (0.0001, 100.0),
            }
            for key, (minimum, maximum) in float_ranges.items():
                if key in src and src[key] not in (None, ""):
                    value = float(src[key])
                    if not minimum <= value <= maximum:
                        raise ValueError(f"TDPT {key} 必须在 {minimum:g} 到 {maximum:g} 之间")
                    dest[key] = value
            if "lowpass_order" in src and src["lowpass_order"] not in (None, ""):
                dest["lowpass_order"] = max(1, min(10, int(src["lowpass_order"])))
            if "scale" in src and src["scale"] not in (None, ""):
                value = float(src["scale"])
                if value <= 0:
                    raise ValueError("TDPT 缩放值必须大于 0")
                dest["scale"] = value
            if "pose_batch" in src and src["pose_batch"] not in (None, ""):
                dest["pose_batch"] = max(1, min(32, int(src["pose_batch"])))
            if "pose_fp16" in src:
                dest["pose_fp16"] = bool(src["pose_fp16"])

    for engine in ("instanthmr", "flowhmr", "gemx", "tdpt"):
        eng = CONFIG["engines"][engine]
        root = Path(eng["root"])
        detected_python = cfgmod.flow_python(root) if engine == "flowhmr" else cfgmod.venv_python(root)
        if not eng.get("python") or not Path(eng["python"]).is_file():
            if detected_python:
                eng["python"] = detected_python
        if engine == "instanthmr":
            if not eng.get("models_dir"):
                eng["models_dir"] = str(root / "models")
        elif engine == "flowhmr":
            if not eng.get("models_dir"):
                eng["models_dir"] = str(root / "models")
        elif engine == "gemx":
            if not eng.get("models_dir"):
                eng["models_dir"] = str(root / "weights")
            if not eng.get("soma_dir"):
                eng["soma_dir"] = str(root / "assets" / "soma")
        elif engine == "tdpt":
            if not eng.get("models_dir"):
                eng["models_dir"] = str(root / "models")


def _engine_payload() -> dict:
    status = engines.all_status(CONFIG, FLOW.status())
    status["instanthmr"]["camera"] = CAMERA.status()
    status["tdpt"]["camera"] = TDPT_CAMERA.status()
    for engine_id in ("instanthmr", "flowhmr", "gemx", "tdpt"):
        active = SETUP.active_for(engine_id)
        status[engine_id]["setup_running"] = active.id if active else None
    return status


def _check_engine(engine: str, params: dict | None = None) -> None:
    if engine == "instanthmr":
        status = engines.instant_status(CONFIG)
        selected_ok = bool(status["selected_model_ok"])
        selected_label = str(status["selected_model"])
        if params and params.get("model"):
            tier = str(params["model"])
            fit = int(params.get("fit", status.get("fit") or 0))
            variant = next((v for v in status.get("variants", [])
                            if v["tier"] == tier and v["fit"] == fit), None)
            if variant is None:
                raise RuntimeError(f"未知的 InstantHMR 模型档位: {tier}")
            selected_ok = bool(variant["exists"])
            selected_label = f"{tier} · fit{fit}"
        missing = []
        if not status["root_ok"]:
            missing.append("InstantHMR 目录")
        if not status["python_ok"]:
            missing.append("虚拟环境")
        if not status["deps_ok"]:
            missing.append("Python 依赖")
        if not selected_ok:
            missing.append(f"模型 {selected_label}")
        if not status["body_ok"]:
            missing.append("mhr-body.onnx")
        if missing:
            raise RuntimeError("InstantHMR 未就绪：缺少 " + "、".join(missing)
                               + "。请到「设置与资源」准备。")
    elif engine == "flowhmr":
        status = engines.flow_status(CONFIG, FLOW.status())
        if not status["ready"]:
            missing = []
            if not status["root_ok"]:
                missing.append("FlowHMR 目录")
            if not status["python_ok"]:
                missing.append("运行环境")
            if not status.get("deps_ok", True):
                missing.append("Python 依赖")
            for item in status["models"]:
                if item["required"] and not item["exists"]:
                    missing.append(item["label"])
            raise RuntimeError("FlowHMR 未就绪：缺少 " + "、".join(missing)
                               + "。请到「设置与资源」准备。")
        if not status.get("job_ready", status["ready"]):
            raise RuntimeError("当前未开启「静态机位」，FlowHMR 需要 VGGT-Omega 相机模型。"
                               "请到设置里下载该模型，或开启静态机位后重试。")
        if params:
            labels = {"hamer": "HaMeR 手指精修", "torso": "躯干动作优化",
                      "crossing": "人物交叉优化"}
            reasons = {"unsupported": "当前构建不支持", "missing": "模型未下载",
                       "gpu_required": "需要 GPU"}
            unavailable = []
            for key in ("hamer", "torso", "crossing"):
                flag = params.get("hamer" if key == "hamer" else f"{key}_opt")
                if not flag:
                    continue
                info = (status.get("enhancements") or {}).get(key) or {}
                if not info.get("available"):
                    unavailable.append(
                        f"{labels[key]}（{reasons.get(info.get('reason'), '不可用')}）")
            if unavailable:
                raise RuntimeError("FlowHMR 增强项不可用：" + "、".join(unavailable)
                                   + "。请到「设置与资源」准备。")
    elif engine == "gemx":
        status = engines.gemx_status(CONFIG)
        missing = []
        if not status["root_ok"]:
            missing.append("GEM-X 目录")
        if not status["python_ok"]:
            missing.append("虚拟环境")
        if not status["deps_ok"]:
            missing.append("Python 依赖")
        for item in status["models"]:
            if item["required"] and not item["exists"]:
                missing.append(item["label"])
        if status["device_warn"]:
            raise RuntimeError("当前设置为 CUDA 但 torch 没有可用的 CUDA；"
                               "请到设置里把设备改为 auto 或 cpu")
        if missing:
            raise RuntimeError("GEM-X 未就绪：缺少 " + "、".join(missing)
                               + "。请到「设置与资源」准备。")
    elif engine == "tdpt":
        status = engines.tdpt_status(CONFIG)
        missing = []
        if not status["root_ok"]:
            missing.append("TDPT 目录")
        if not status["python_ok"]:
            missing.append("虚拟环境")
        if not status["deps_ok"]:
            missing.append("Python 依赖")
        if not status["model_ok"]:
            missing.append("模型 " + str(status["model"]))
        if missing:
            raise RuntimeError("TDPT 未就绪：缺少 " + "、".join(missing)
                               + "。请到「设置与资源」准备。")
    else:
        raise RuntimeError(f"未知引擎: {engine}")


def _job_params(engine: str) -> dict:
    """Per-job parameters: precision/quality choices plus the shared limits."""
    params = {
        "max_frames": int(request.args.get("max_frames", CONFIG["job"]["max_frames"]) or 0),
        "auto_frames": str(request.args.get("auto_frames", "0")).lower() in ("1", "true", "on"),
        "device": CONFIG["engines"][engine].get("device"),
    }
    if engine == "instanthmr":
        tier = (request.args.get("model") or "").strip()
        if tier:
            if tier not in engines.INSTANT_TIERS:
                raise ValueError(f"未知的 InstantHMR 模型档位: {tier}")
            params["model"] = tier
        fit_raw = request.args.get("fit")
        if fit_raw not in (None, ""):
            try:
                fit = int(fit_raw)
            except (TypeError, ValueError):
                raise ValueError(f"InstantHMR 拟合档位无效: {fit_raw}")
            if fit not in (0, 1, 2):
                raise ValueError("InstantHMR 拟合档位须为 0、1 或 2")
            params["fit"] = fit
        inst = CONFIG["engines"]["instanthmr"]
        params["pose_filter"] = inst.get("pose_filter", "one-euro")
        smooth_raw = request.args.get("smooth")
        params["smooth"] = (str(smooth_raw).lower() in ("1", "true", "on", "yes")
                            if smooth_raw not in (None, "")
                            else bool(inst.get("smooth", False)))
        params["focal"] = inst.get("focal")
    elif engine == "flowhmr":
        eng = CONFIG["engines"]["flowhmr"]
        steps_raw = request.args.get("steps")
        if steps_raw not in (None, ""):
            try:
                steps = int(steps_raw)
            except (TypeError, ValueError):
                raise ValueError(f"FlowHMR 步数无效: {steps_raw}")
            if not 1 <= steps <= 100:
                raise ValueError("FlowHMR 步数须在 1–100 之间")
            params["steps"] = steps
        else:
            params["steps"] = max(1, min(100, int(eng.get("steps") or 20)))

        def flag(name: str) -> bool:
            return str(request.args.get(name, "0")).lower() in ("1", "true", "on", "yes")

        enhance = {
            "hamer": flag("hamer"),
            "torso_opt": flag("torso_opt"),
            "crossing_opt": flag("crossing_opt"),
        }
        if any(enhance.values()) and engines.flow_layout(Path(eng["root"]), eng)["kind"] != "windows":
            raise ValueError("当前 FlowHMR 构建不支持模型增强选项（需要 Windows 便携版）")
        # Only version-agnostic fields plus the optional Windows enhancements:
        # each FlowHMR build fills the rest from its own defaults.
        params.update({
            "manual_select": bool(eng.get("manual_select")),
            "static_camera": bool(eng.get("static_camera")),
            **enhance,
        })
        if params["auto_frames"] or params["max_frames"] <= 0:
            params["auto_frames"] = True
            params["max_frames"] = 900
        else:
            params["max_frames"] = max(30, min(10000, params["max_frames"]))
    elif engine == "gemx":
        params["device"] = CONFIG["engines"]["gemx"].get("device")
    elif engine == "tdpt":
        eng = CONFIG["engines"]["tdpt"]
        params["device"] = eng.get("device") or "cpu"
        smooth_raw = request.args.get("smooth")
        params["smooth"] = (str(smooth_raw).lower() in ("1", "true", "on", "yes")
                            if smooth_raw not in (None, "")
                            else bool(eng.get("smooth", True)))
        params["flip"] = bool(eng.get("flip_video", True))
        params["scale"] = float(eng.get("scale") or 0.01)
    else:
        raise ValueError(f"未知引擎: {engine}")
    return params


# --------------------------------------------------------------------------- pages
@app.get("/")
def index():
    return send_from_directory(str(BASE_DIR / "templates"), "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


# --------------------------------------------------------------------------- config / status
@app.get("/api/config")
def api_config():
    return jsonify(cfgmod.public_config(CONFIG))


@app.post("/api/config")
def api_save_config():
    payload = request.get_json(silent=True) or {}
    try:
        _update_config(lambda c: _merge_config(payload))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(config=cfgmod.public_config(CONFIG), engines=_engine_payload())


@app.get("/api/engines")
def api_engines():
    return jsonify(engines=_engine_payload())


@app.get("/api/health")
def api_health():
    return jsonify(ok=True, app="Pose3D Collects", version="0.1.0")


# --------------------------------------------------------------------------- setup
@app.post("/api/setup/<engine>/env")
def api_setup_env(engine: str):
    try:
        task = engines.setup_env_task(CONFIG, SETUP, engine, _update_config, FLOW)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(task=task.view(0))


@app.post("/api/setup/<engine>/models")
def api_setup_models(engine: str):
    payload = request.get_json(silent=True) or {}
    try:
        task = engines.setup_models_task(CONFIG, SETUP, engine, payload, _update_config)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(task=task.view(0))


@app.get("/api/setup/tasks/<task_id>")
def api_setup_task(task_id: str):
    task = SETUP.get(task_id)
    if task is None:
        abort(404)
    offset = request.args.get("offset", default=0, type=int)
    return jsonify(task=task.view(offset))


@app.post("/api/setup/tasks/<task_id>/stop")
def api_setup_stop(task_id: str):
    ok = SETUP.stop(task_id)
    return jsonify(ok=ok)


# --------------------------------------------------------------------------- flowhmr service
@app.post("/api/flowhmr/start")
def api_flow_start():
    try:
        return jsonify(server=FLOW.start())
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/flowhmr/stop")
def api_flow_stop():
    return jsonify(server=FLOW.stop())


# --------------------------------------------------------------------------- instant camera
@app.post("/api/instanthmr/camera/start")
def api_instant_camera_start():
    payload = request.get_json(silent=True) or {}
    index = payload.get("camera_index")
    if index in ("", None):
        index = None
    try:
        return jsonify(camera=CAMERA.start(CONFIG, camera_index=index))
    except (RuntimeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/instanthmr/camera/stop")
def api_instant_camera_stop():
    return jsonify(camera=CAMERA.stop())


@app.get("/api/instanthmr/camera")
def api_instant_camera_status():
    return jsonify(camera=CAMERA.status())


# --------------------------------------------------------------------------- tdpt camera
@app.post("/api/tdpt/camera/start")
def api_tdpt_camera_start():
    payload = request.get_json(silent=True) or {}
    index = payload.get("camera_index")
    if index in ("", None):
        index = None
    try:
        return jsonify(camera=TDPT_CAMERA.start(CONFIG, camera_index=index))
    except (RuntimeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/tdpt/camera/stop")
def api_tdpt_camera_stop():
    return jsonify(camera=TDPT_CAMERA.stop())


@app.get("/api/tdpt/camera")
def api_tdpt_camera_status():
    return jsonify(camera=TDPT_CAMERA.status())


# --------------------------------------------------------------------------- jobs
@app.get("/api/jobs")
def api_jobs():
    return jsonify(jobs=JOBS.list())


@app.route("/api/jobs", methods=["POST", "PUT"])
def api_create_job():
    engine = request.args.get("engine", "instanthmr")
    if engine not in ("instanthmr", "flowhmr", "gemx", "tdpt"):
        return jsonify(error=f"未知引擎: {engine}"), 400
    filename = Path(request.args.get("filename") or "video.mp4").name
    if Path(filename).suffix.lower() not in VIDEO_EXTS:
        return jsonify(error=f"不支持的视频格式: {Path(filename).suffix or '(无扩展名)'}"), 400
    try:
        params = _job_params(engine)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    try:
        _check_engine(engine, params)
    except RuntimeError as exc:
        return jsonify(error=str(exc), code="engine_not_ready"), 409

    if request.content_length and request.content_length > MAX_UPLOAD:
        return jsonify(error="视频超过 24 GB 上限"), 413
    cfgmod.TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp = cfgmod.TMP_DIR / (uuid.uuid4().hex + Path(filename).suffix.lower())
    size = 0
    try:
        with temp.open("wb") as out:
            while True:
                chunk = request.stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise RuntimeError("视频超过 24 GB 上限")
                out.write(chunk)
        if size == 0:
            raise RuntimeError("上传的文件为空")
        job = JOBS.submit(engine, filename, temp, params)
        return jsonify(job=job)
    except RuntimeError as exc:
        temp.unlink(missing_ok=True)
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        temp.unlink(missing_ok=True)
        return jsonify(error=f"上传失败: {exc}"), 400


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    try:
        return jsonify(job=JOBS.get(job_id))
    except KeyError:
        abort(404)


@app.delete("/api/jobs/<job_id>")
def api_delete_job(job_id: str):
    try:
        JOBS.delete(job_id)
        return jsonify(ok=True)
    except KeyError:
        abort(404)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409


@app.post("/api/jobs/<job_id>/cancel")
def api_cancel_job(job_id: str):
    try:
        return jsonify(job=JOBS.cancel(job_id))
    except KeyError:
        abort(404)


@app.post("/api/jobs/<job_id>/select")
def api_select_person(job_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        person = int(payload.get("person"))
    except (TypeError, ValueError):
        return jsonify(error="person 必须是整数"), 400
    try:
        return jsonify(job=JOBS.select(job_id, person))
    except KeyError:
        abort(404)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409


# --------------------------------------------------------------------------- result files
def _proxy(kind: str, job_id: str, extra: str = ""):
    job = JOBS.jobs.get(job_id)
    if job is None:
        abort(404)
    engine_id = job.get("engine_job_id")
    if not engine_id:
        abort(404)
    try:
        upstream = FLOW.open_result(engine_id, kind, request.headers.get("Range"), extra)
    except Exception as exc:
        return jsonify(error=str(exc)), 502
    headers = {}
    for key in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges",
                "Content-Disposition", "Last-Modified", "ETag"):
        value = upstream.headers.get(key)
        if value:
            headers[key] = value
    headers.setdefault("Cache-Control", "no-store")

    def generate():
        try:
            while True:
                chunk = upstream.read(512 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            upstream.close()

    return Response(generate(), status=upstream.status, headers=headers)


@app.route("/api/jobs/<job_id>/video")
def api_job_video(job_id: str):
    job = JOBS.jobs.get(job_id)
    if job is None:
        abort(404)
    if job.get("engine") == "flowhmr":
        return _proxy("video", job_id)
    path = JOBS._dir(job_id) / (job.get("video_file") or job.get("input_file") or "")
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=video_mime(job), conditional=True, max_age=0)


@app.route("/api/jobs/<job_id>/meta")
def api_job_meta(job_id: str):
    job = JOBS.jobs.get(job_id)
    if job is None:
        abort(404)
    if job.get("engine") == "flowhmr":
        return _proxy("meta", job_id)
    path = JOBS._dir(job_id) / "mesh" / "meta.json"
    if not path.is_file():
        abort(404)
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/jobs/<job_id>/vertices")
def api_job_vertices(job_id: str):
    return _result_file(job_id, "vertices", "mesh/vertices.bin", "application/octet-stream")


@app.route("/api/jobs/<job_id>/faces")
def api_job_faces(job_id: str):
    return _result_file(job_id, "faces", "mesh/faces.bin", "application/octet-stream")


@app.route("/api/jobs/<job_id>/track")
def api_job_track(job_id: str):
    job = JOBS.jobs.get(job_id)
    if job is None:
        abort(404)
    if job.get("engine") == "flowhmr":
        return _proxy("track", job_id)
    path = JOBS._dir(job_id) / "track.json"
    if not path.is_file():
        abort(404)
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/jobs/<job_id>/npz")
def api_job_npz(job_id: str):
    job = JOBS.jobs.get(job_id)
    if job is None:
        abort(404)
    if job.get("engine") == "flowhmr":
        return _proxy("npz", job_id)
    path = JOBS._dir(job_id) / "motion.npz"
    if not path.is_file():
        abort(404)
    name = f"{Path(job['filename']).stem}_{job.get('engine') or 'instanthmr'}.npz"
    return send_file(path, mimetype="application/octet-stream", as_attachment=True,
                     download_name=name, conditional=True)


@app.route("/api/jobs/<job_id>/person/<int:index>.jpg")
def api_job_person(job_id: str, index: int):
    job = JOBS.jobs.get(job_id)
    if job is None:
        abort(404)
    if job.get("engine") == "flowhmr":
        return _proxy("person", job_id, str(index))
    abort(404)


def _result_file(job_id: str, kind: str, relative: str, mimetype: str):
    job = JOBS.jobs.get(job_id)
    if job is None:
        abort(404)
    if job.get("engine") == "flowhmr":
        return _proxy(kind, job_id)
    path = JOBS._dir(job_id) / relative
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=mimetype, conditional=True, max_age=0)


# --------------------------------------------------------------------------- misc
def _allowed_path_roots() -> list[Path]:
    roots: list[Path] = []
    for engine in CONFIG["engines"].values():
        roots.append(Path(engine["root"]))
        if engine.get("models_dir"):
            roots.append(Path(engine["models_dir"]))
        if engine.get("soma_dir"):
            roots.append(Path(engine["soma_dir"]))
    try:
        roots.append(Path(engines.flow_layout(Path(CONFIG["engines"]["flowhmr"]["root"]),
                                              CONFIG["engines"]["flowhmr"])["models_dir"]))
    except Exception:
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _within_allowed(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = str(path.resolve()).lower()
    except OSError:
        return False
    for root in roots:
        base = str(root).lower().rstrip("\\/")
        if resolved == base or resolved.startswith(base + os.sep):
            return True
    return False


def _open_directory(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        import subprocess
        subprocess.Popen(["xdg-open", str(path)])


@app.post("/api/open-folder")
def api_open_folder():
    payload = request.get_json(silent=True) or {}
    engine = payload.get("engine")
    which = payload.get("which", "models")
    if engine not in CONFIG["engines"]:
        return jsonify(error="未知引擎"), 400
    eng = CONFIG["engines"][engine]
    path = Path(eng.get("models_dir") or eng["root"]) if which == "models" else Path(eng["root"])
    target = path.resolve()
    if not _within_allowed(target, _allowed_path_roots()):
        return jsonify(error="路径不在已配置的项目目录内"), 400
    if not target.is_dir():
        return jsonify(error=f"目录不存在: {target}"), 400
    try:
        _open_directory(target)
    except Exception as exc:
        return jsonify(error=f"无法打开目录: {exc}"), 500
    return jsonify(ok=True, path=str(target))


@app.post("/api/open-path")
def api_open_path():
    """Open a model folder, creating it when the user is about to place files."""
    payload = request.get_json(silent=True) or {}
    raw = payload.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return jsonify(error="缺少路径"), 400
    path = Path(raw.strip())
    if not path.is_absolute():
        return jsonify(error="路径必须是绝对路径"), 400
    roots = _allowed_path_roots()
    if not _within_allowed(path, roots):
        return jsonify(error="路径不在已配置的引擎目录内"), 400

    created = False
    if path.exists():
        target = path.parent if path.is_file() else path
    else:
        # Treat a filename (with an extension) as "open its parent folder".
        target = path.parent if path.suffix else path
        if not _within_allowed(target, roots):
            return jsonify(error="目标目录不在已配置的引擎目录内"), 400
        try:
            target.mkdir(parents=True, exist_ok=True)
            created = True
        except OSError as exc:
            return jsonify(error=f"无法创建目录: {exc}"), 500
    try:
        _open_directory(target)
    except Exception as exc:
        return jsonify(error=f"无法打开目录: {exc}"), 500
    return jsonify(ok=True, path=str(target), created=created)


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify(error="not found"), 404
    return send_from_directory(str(BASE_DIR / "templates"), "index.html")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="上传文件过大"), 413


@atexit.register
def _shutdown():
    SETUP.stop_all()
    JOBS.shutdown()
    FLOW.shutdown()
    CAMERA.shutdown()
    TDPT_CAMERA.shutdown()


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Pose3D Collects 需要 Python 3.10 或更高版本")
    # The UI polls once a second; don't fill the log with request lines.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Pose3D Collects")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="open the browser")
    args = parser.parse_args()
    if args.open:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    print(f"\n  Pose3D Collects:  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
