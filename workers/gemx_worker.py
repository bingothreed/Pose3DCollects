"""Run GEM-X Body Pose on a video inside the GEM-X virtual environment.

The hub launches this with GEM-X's Python. It detects the primary person with
YOLOX, predicts body pose with the DINOv3/SOMA model in batches (optionally
under fp16 autocast), writes the unified result package (float16 vertex stream
+ faces + meta + 2D track) and reports progress as JSON lines on stdout.

Measured on an RTX 3060: CUDA detector ~78 ms/frame; pose x16 fp16 ~55 ms/frame
(~18 fps); fp16 changes vertices by <0.3 mm. The detector falls back to CPU
when ONNXRuntime has no CUDA execution provider (torch's bundled CUDA DLLs are
put on PATH first, which is all ORT-GPU needs).

GEM-X Body Pose geometry lives in the original camera frame (x right, y down,
z forward); it is flipped to the same y-up convention InstantHMR writes.
"""
from __future__ import annotations

import argparse
import json
import os
import site
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


def emit(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _prepend_torch_lib() -> None:
    """ORT-GPU resolves CUDA/cuDNN from torch's bundled DLL directory."""
    for base in site.getsitepackages():
        candidate = Path(base) / "torch" / "lib"
        if candidate.is_dir():
            current = os.environ.get("PATH", "")
            if str(candidate) not in current:
                os.environ["PATH"] = str(candidate) + os.pathsep + current
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GEM-X repository root")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True, help="body_s240000.ckpt path")
    parser.add_argument("--detector", required=True, help="yolox_humanart.onnx path")
    parser.add_argument("--soma", required=True, help="SOMA 0.3.3 asset directory")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--detector-device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--pose-batch", type=int, default=8, help="frames per forward batch")
    parser.add_argument("--fp16", type=int, default=1, help="1 = fp16 autocast on CUDA")
    return parser.parse_args()


def load_predictor(args, *, device: str):
    from gem.inference.pose import load_predictor as _load

    emit(type="log", message=f"加载 GEM-X Body Pose 模型（{device}）…")
    predictor, _release = _load(
        args.checkpoint, task="body", soma_assets=args.soma, device=device
    )
    return predictor


def make_predictor(args):
    attempts = ["cuda", "cpu"] if args.device == "auto" else [args.device]
    last_error = None
    for candidate in attempts:
        try:
            predictor = load_predictor(args, device=candidate)
            return predictor, candidate
        except Exception as exc:  # CUDA missing, OOM, driver mismatch, …
            last_error = exc
            emit(type="log", message=f"{candidate} 初始化失败: {exc}")
    raise RuntimeError(f"无法初始化 GEM-X: {last_error}")


def detect_annotations(args, video_path):
    from gem.inference.body import detect_body_annotations

    attempts = (["cuda", "cpu"] if args.detector_device == "auto"
                else [args.detector_device])
    for candidate in attempts:
        emit(type="log", message=f"YOLOX 检测并跟踪主要人物（{candidate}）…")
        started = time.perf_counter()
        try:
            annotations, metadata = detect_body_annotations(
                video_path, args.detector, video=True, device=candidate
            )
        except RuntimeError as exc:  # e.g. no CUDAExecutionProvider
            if candidate is attempts[-1]:
                raise
            emit(type="log", message=f"{candidate} 检测器不可用（{exc}），回退 CPU")
            continue
        elapsed = time.perf_counter() - started
        per_frame = elapsed / max(len(annotations), 1) * 1000
        emit(type="log", message=f"检测完成: {len(annotations)} 个标注帧，"
                                 f"{metadata['image_size'][0]}x{metadata['image_size'][1]}，"
                                 f"耗时 {elapsed:.0f}s（{per_frame:.0f} ms/帧）")
        return annotations, metadata
    raise RuntimeError("无法初始化 YOLOX 检测器")


def trim_video(source: Path, out: Path, max_frames: int, fps: float) -> Path:
    """Re-encode the first max_frames frames so detection does not scan further."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = None
    count = 0
    try:
        while count < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if writer is None:
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建临时视频: {out}")
            writer.write(bgr)
            count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return out


def main() -> None:
    args = parse_args()
    _prepend_torch_lib()
    project = Path(args.project).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    out = Path(args.out).resolve()
    (out / "mesh").mkdir(parents=True, exist_ok=True)

    source = Path(args.video)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    detect_input = source
    max_frames = max(0, args.max_frames)
    if max_frames and (not source_frame_count or source_frame_count > max_frames):
        trimmed = out / "detect_input.mp4"
        trim_video(source, trimmed, max_frames, fps)
        detect_input = trimmed
        emit(type="log", message=f"检测输入截取到前 {max_frames} 帧")
    annotations, _metadata = detect_annotations(args, detect_input)

    predictor, device = make_predictor(args)
    faces = np.asarray(predictor.geometry.faces.cpu(), dtype=np.uint32)
    vertices_count = int(faces.max()) + 1
    use_fp16 = bool(args.fp16) and device == "cuda"
    pose_batch = max(1, int(args.pose_batch))
    from gem.motion.frontend.joint_sets import get_observation_schema
    schema = get_observation_schema("soma77_native_v1")
    joint_names = list(schema.joint_names)
    edges = [[p, i] for i, parents in enumerate(schema.visibility_parents)
             for p in parents]
    emit(type="log", message=f"视频 {width}x{height} @ {fps:.2f} fps，"
                             f"关节 {len(joint_names)}，顶点 {vertices_count}，设备 {device}，"
                             f"批量 {pose_batch}，{'fp16' if use_fp16 else 'fp32'}")

    vertices_path = out / "mesh" / "vertices.bin"
    joints_path = out / "mesh" / "joints.bin"
    track_frames = []
    joints2d_all = []
    joints3d_all = []
    boxes_all = []
    last_vertices = None
    last_joints = None
    last_keypoints = None
    bbox_min = np.array([np.inf, np.inf, np.inf], np.float32)
    bbox_max = np.array([-np.inf, -np.inf, -np.inf], np.float32)
    root_trajectory = []
    predict_ms = []
    empty = 0
    seen_person = False
    started = time.perf_counter()
    pending: list = []

    def pack(i: int, boxes_t: torch.Tensor, prediction) -> dict:
        """One batched row -> CPU numpy payload in the unified convention."""
        verts = np.asarray(prediction.vertices_camera[i].cpu(), np.float32)
        joints = np.asarray(prediction.joints_camera[i].cpu(), np.float32)
        verts = np.stack([verts[:, 0], -verts[:, 1], -verts[:, 2]], axis=1)
        joints = np.stack([joints[:, 0], -joints[:, 1], -joints[:, 2]], axis=1)
        pixels = np.asarray(prediction.keypoints_2d[i].cpu(), np.float32)
        valid = np.asarray(prediction.projection_valid[i].cpu(), np.bool_)
        cx, cy, size = (float(v) for v in boxes_t[i].cpu())
        half = size / 2.0
        box = [cx - half, cy - half, cx + half, cy + half]
        keypoints = [
            [round(float(pixels[j, 0]), 2), round(float(pixels[j, 1]), 2)]
            if valid[j] else None
            for j in range(len(joint_names))
        ]
        return {"verts": verts, "joints": joints, "box": box, "keypoints": keypoints}

    def flush_batch() -> None:
        """Run one forward batch and write payloads in the same order."""
        nonlocal pending, last_vertices, last_joints, last_keypoints, seen_person
        if not pending:
            return
        images = [item["rgb"] for item in pending]
        boxes_t = torch.tensor([item["record"]["instances"][0]["bbox"] for item in pending],
                               dtype=torch.float32)
        intrinsics = np.asarray([item["record"]["intrinsics"] for item in pending], np.float32)
        k_t = torch.as_tensor(intrinsics)
        batch = predictor.prepare(images, boxes_t, k_t)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            prediction = predictor.predict_batch(batch)
        torch.cuda.synchronize()
        predict_ms.append((time.perf_counter() - t0) * 1000.0 / len(pending))
        for i in range(len(pending)):
            payload = pack(i, boxes_t, prediction)
            write_payload(payload)
            last_vertices, last_joints = payload["verts"], payload["joints"]
            last_keypoints = payload["keypoints"]
            seen_person = True
        del batch, prediction
        pending = []

    def write_payload(payload: dict) -> None:
        vertex_file.write(np.asarray(payload["verts"], np.float16).tobytes())
        joint_file.write(np.asarray(payload["joints"], np.float16).tobytes())
        if seen_person:
            bbox_min[...] = np.minimum(bbox_min, payload["verts"].min(axis=0))
            bbox_max[...] = np.maximum(bbox_max, payload["verts"].max(axis=0))
        root = payload["joints"][0]
        root_trajectory.append([round(float(root[0]), 3), round(float(root[2]), 3)])
        joints2d_all.append(np.asarray(
            [[0.0, 0.0] if p is None else p for p in payload["keypoints"]], np.float32))
        joints3d_all.append(payload["joints"])
        boxes_all.append([round(float(v), 2) for v in payload["box"]])
        track_frames.append({"box": payload["box"], "score": 0.0,
                             "keypoints": payload["keypoints"]})

    def write_hold() -> None:
        nonlocal last_vertices, last_joints, last_keypoints, empty
        empty += 1
        if last_vertices is None:
            last_vertices = np.zeros((18056, 3), np.float32)
            last_joints = np.zeros((77, 3), np.float32)
            last_keypoints = None
        vertex_file.write(np.asarray(last_vertices, np.float16).tobytes())
        joint_file.write(np.asarray(last_joints, np.float16).tobytes())
        if seen_person:
            bbox_min[...] = np.minimum(bbox_min, last_vertices.min(axis=0))
            bbox_max[...] = np.maximum(bbox_max, last_vertices.max(axis=0))
        root = last_joints[0]
        root_trajectory.append([round(float(root[0]), 3), round(float(root[2]), 3)])
        joints2d_all.append(np.asarray(
            [[0.0, 0.0] if p is None else p for p in last_keypoints], np.float32)
            if last_keypoints is not None else np.zeros((77, 2), np.float32))
        joints3d_all.append(last_joints)
        boxes_all.append([0, 0, 0, 0])
        track_frames.append({"box": None, "score": 0.0, "keypoints": None})

    cap = cv2.VideoCapture(str(detect_input))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with vertices_path.open("wb") as vertex_file, joints_path.open("wb") as joint_file:
        index = 0
        while True:
            if max_frames and index >= max_frames:
                break
            ok, bgr = cap.read()
            if not ok:
                break
            if index in annotations:
                pending.append({"rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                "record": annotations[index]})
                if len(pending) >= pose_batch:
                    flush_batch()
            else:
                # Unannotated frame: flush the batch first so file order holds,
                # then write the held frame.
                if pending:
                    flush_batch()
                write_hold()
            index += 1
            if index % 8 == 0 or (max_frames and index >= max_frames):
                emit(type="progress", frame=index, total=max_frames or 0,
                     stage="检测与重建",
                     message=f"处理帧 {index}" + (f"/{max_frames}" if max_frames else ""))
        if pending:
            flush_batch()
    cap.release()

    frames = len(track_frames)
    if frames == 0:
        raise RuntimeError("视频中没有可读取的帧")
    if empty == frames:
        raise RuntimeError("所有帧都没有检测到人物，请换一个更清晰的视频")
    if not np.isfinite(bbox_min).all():
        raise RuntimeError("没有生成有效的 3D 网格")

    faces.tofile(out / "mesh" / "faces.bin")
    vertices = np.fromfile(vertices_path, dtype=np.float16).reshape(frames, -1, 3)
    joints = np.fromfile(joints_path, dtype=np.float16).reshape(frames, -1, 3)
    joints2d = np.asarray(joints2d_all, np.float32)
    boxes = np.asarray(boxes_all, np.float32)
    np.savez(out / "motion.npz", vertices=vertices, faces=faces, joints=joints,
             joints_2d=joints2d, boxes=boxes, fps=np.float32(fps))

    track = {
        "fps": fps, "width": width, "height": height, "start": 0, "end": frames - 1,
        "edges": edges, "joint_names": joint_names,
        "frames": track_frames,
    }
    (out / "track.json").write_text(json.dumps(track, ensure_ascii=False), encoding="utf-8")

    predict_median = float(np.median(predict_ms)) if predict_ms else 0.0
    meta = {
        "engine": "gemx",
        "num_frames": frames,
        "num_vertices": int(vertices.shape[1]),
        "num_faces": int(faces.shape[0]),
        "fps": fps,
        "vertex_dtype": "float16",
        "face_dtype": "uint32",
        "up_axis": "y",
        "bbox_min": [round(float(v), 4) for v in bbox_min],
        "bbox_max": [round(float(v), 4) for v in bbox_max],
        "root_trajectory": root_trajectory,
        "track_start": 0,
        "track_end": frames - 1,
        "width": width,
        "height": height,
        "empty_frames": empty,
        "timings": {"predict_ms_median": round(predict_median, 2),
                    "elapsed_s": round(time.perf_counter() - started, 2)},
        "device": device,
        "precision": "fp16" if use_fp16 else "fp32",
        "pose_batch": pose_batch,
        "model": Path(args.checkpoint).name,
        "source_total_frames": source_frame_count or None,
    }
    (out / "mesh" / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    emit(type="done", meta=meta)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit(type="error", message=str(exc))
        traceback.print_exc()
        sys.exit(1)
