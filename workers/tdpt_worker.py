"""Run TDPT (ThreeDPoseTracker VNect) on a video inside the TDPT venv.

The hub launches this with the TDPT project's Python. It runs the tiny
MobileNet3D3_6.onnx on CPU per frame, decodes 24 joints + 4 derived joints,
synthesizes a capsule mesh for the 3D viewer and writes the unified result
package (float16 vertex stream + faces + meta + 2D track) with JSON lines on
stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Per-joint score below which the frame is treated as "no person" and the last
# valid pose is held. A clear full-body frame scores ~0.5-0.9.
SCORE_THRESHOLD = 0.10

# VNect outputs model units (input crop = 224 px, a person is ~140 units tall).
# The output is multiplied by --scale so it lands in the same metre-ish range
# the other engines write (default 0.01 -> person ~1.4 units), which keeps the
# 3D viewer's default camera/clip distances framing the skeleton correctly.
DEFAULT_SCALE = 0.01


def emit(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="TDPT project root")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, help="MobileNet3D3_6.onnx path")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--smooth", type=int, default=1, help="1 = One-Euro smoothing")
    parser.add_argument("--flip", type=int, default=0,
                        help="1 = flip frames vertically (Unity texture orientation)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help="output scale factor (model units -> scene units)")
    return parser.parse_args()


def load_session(model: Path):
    if not model.is_file():
        raise RuntimeError(f"模型不存在: {model}")
    ort.set_default_logger_severity(3)  # silence UTF-16 console warnings on Windows
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    probe = sess.run(None, {input_name: np.zeros((1, 3, 224, 224), np.float32)})
    by_shape = {tuple(o.shape): o for o in probe}
    heat_key = (1, 24 * 14, 14, 14)
    off_key = (1, 24 * 3 * 14, 14, 14)
    if heat_key not in by_shape or off_key not in by_shape:
        raise RuntimeError(f"模型输出与 VNect 布局不符: {list(by_shape)}")
    return sess, input_name, heat_key, off_key


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from tdpt_vnect import (ALL_JOINT_NAMES, EDGES_2D, MESH_FACES, decode, derive,
                            project2d, synthesize_mesh, OneEuro)

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

    sess, input_name, heat_key, off_key = load_session(Path(args.model))
    emit(type="log", message=f"加载 TDPT (VNect) 模型（CPU）…")
    scale = float(args.scale)
    if scale <= 0:
        raise RuntimeError("缩放值必须大于 0")
    emit(type="log", message=f"视频 {width}x{height} @ {fps:.2f} fps，"
                             f"关节 {len(ALL_JOINT_NAMES)}，缩放 {scale:g}，"
                             f"{'One-Euro 平滑' if args.smooth else '无滤波'}"
                             + ("，垂直翻转输入" if args.flip else ""))

    max_frames = max(0, args.max_frames)
    total = min(source_frame_count, max_frames) if source_frame_count and max_frames else (
        source_frame_count or max_frames)
    filters = [OneEuro() for _ in range(28)] if args.smooth else None

    vertices_path = out / "mesh" / "vertices.bin"
    joints_path = out / "mesh" / "joints.bin"
    track_frames = []
    joints2d_all = []
    boxes_all = []
    last_joints = None
    bbox_min = np.array([np.inf, np.inf, np.inf], np.float32)
    bbox_max = np.array([-np.inf, -np.inf, -np.inf], np.float32)
    root_trajectory = []
    durations = []
    empty = 0
    seen_person = False
    started = time.perf_counter()

    with vertices_path.open("wb") as vertex_file, joints_path.open("wb") as joint_file:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if args.flip:
                bgr = cv2.flip(bgr, 0)
            rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            t0 = time.perf_counter()
            outs = sess.run(None, {input_name: x})
            by_shape = {tuple(o.shape): o for o in outs}
            kps, scores = decode(by_shape[heat_key][0], by_shape[off_key][0])
            kps28 = derive(kps)
            durations.append((time.perf_counter() - t0) * 1000.0)

            mean_score = float(scores.mean())
            if mean_score >= SCORE_THRESHOLD:
                if filters is not None:
                    now = time.perf_counter()
                    for i in range(28):
                        kps28[i] = filters[i].filter(kps28[i], now)
                last_joints = kps28
                seen_person = True
            else:
                empty += 1
                if last_joints is None:
                    last_joints = np.zeros((28, 3), np.float32)

            px = project2d(last_joints, width, height)
            scaled_joints = last_joints * scale
            mesh = synthesize_mesh(scaled_joints,
                                   radius=max(3.0 * scale, 0.015))
            vertex_file.write(np.asarray(mesh, np.float16).tobytes())
            joint_file.write(np.asarray(scaled_joints, np.float16).tobytes())
            if seen_person:
                bbox_min = np.minimum(bbox_min, mesh.min(axis=0))
                bbox_max = np.maximum(bbox_max, mesh.max(axis=0))
            root = scaled_joints[24]  # hip
            root_trajectory.append([round(float(root[0]), 3), round(float(root[2]), 3)])
            keypoints = [[round(float(v[0]), 2), round(float(v[1]), 2)] for v in px]
            joints2d_all.append(np.asarray(keypoints, np.float32))
            boxes_all.append([0, 0, 0, 0])
            track_frames.append({"box": None,
                                 "score": round(mean_score, 4) if mean_score >= SCORE_THRESHOLD else 0.0,
                                 "keypoints": keypoints if seen_person else None})
            index += 1
            if index % 5 == 0 or (total and index >= total):
                pct = (index / total * 100.0) if total else 0.0
                emit(type="progress", frame=index, total=total, stage="检测与重建",
                     message=f"处理帧 {index}" + (f"/{total}" if total else ""),
                     percent=round(pct, 1))
            if max_frames and index >= max_frames:
                break
    cap.release()

    frames = len(track_frames)
    if frames == 0:
        raise RuntimeError("视频中没有可读取的帧")
    if empty == frames:
        raise RuntimeError("所有帧都没有检测到人物，请换一个更清晰的视频")
    if not np.isfinite(bbox_min).all():
        raise RuntimeError("没有生成有效的 3D 骨架")

    MESH_FACES.tofile(out / "mesh" / "faces.bin")
    vertices = np.fromfile(vertices_path, dtype=np.float16).reshape(frames, -1, 3)
    joints = np.fromfile(joints_path, dtype=np.float16).reshape(frames, -1, 3)
    joints2d = np.asarray(joints2d_all, np.float32)
    boxes = np.asarray(boxes_all, np.float32)
    np.savez(out / "motion.npz", vertices=vertices, faces=np.asarray(MESH_FACES),
             joints=joints, joints_2d=joints2d, boxes=boxes, fps=np.float32(fps))

    track = {
        "fps": fps, "width": width, "height": height, "start": 0, "end": frames - 1,
        "edges": [list(edge) for edge in EDGES_2D],
        "joint_names": ALL_JOINT_NAMES,
        "frames": track_frames,
    }
    (out / "track.json").write_text(json.dumps(track, ensure_ascii=False), encoding="utf-8")

    predict_ms = float(np.median(durations)) if durations else 0.0
    meta = {
        "engine": "tdpt",
        "num_frames": frames,
        "num_vertices": int(vertices.shape[1]),
        "num_faces": int(len(MESH_FACES)),
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
        "timings": {"predict_ms_median": round(predict_ms, 2),
                    "elapsed_s": round(time.perf_counter() - started, 2)},
        "device": "cpu",
        "model": Path(args.model).name,
        "smooth": bool(args.smooth),
        "flip": bool(args.flip),
        "scale": scale,
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
