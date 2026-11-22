"""Run InstantHMR on a video inside the InstantHMR virtual environment.

The hub launches this with that project's Python. It writes a unified result
package (float16 vertex stream + faces + meta + 2D track) and reports progress
as JSON lines on stdout.
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


def emit(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="InstantHMR repository root")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, help="local .onnx path")
    parser.add_argument("--body", required=True, help="local mhr-body.onnx path")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "tensorrt", "coreml"))
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--pose-filter", default="", choices=("", "one-euro", "alpha-beta"))
    parser.add_argument("--smooth", action="store_true",
                        help="smooth keypoints, mesh and translation across frames (1 Euro filter)")
    parser.add_argument("--focal", type=float, default=None)
    return parser.parse_args()


def make_pipeline(args):
    from instanthmr import PosePipeline

    device = args.device
    attempts = ["cuda", "cpu"] if device == "auto" else [device]
    last_error = None
    for candidate in attempts:
        try:
            emit(type="log", message=f"加载 InstantHMR（{candidate}）…")
            pipeline = PosePipeline(
                model=args.model,
                fit=0,
                device=candidate,
                detector="medium",
                det_confidence=0.5,
                max_persons=1,
                focal=args.focal,
                pose_filter=args.pose_filter or None,
            )
            return pipeline, candidate
        except Exception as exc:  # CUDA missing, driver mismatch, …
            last_error = exc
            emit(type="log", message=f"{candidate} 初始化失败: {exc}")
    raise RuntimeError(f"无法初始化 InstantHMR: {last_error}")


def main() -> None:
    args = parse_args()
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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = max(0, args.max_frames)
    limit = frame_count if frame_count > 0 else 0
    if max_frames:
        limit = min(limit, max_frames) if limit else max_frames
    total = limit if limit > 0 else (max_frames if max_frames else 0)

    pipeline, device = make_pipeline(args)
    smoother = None
    if args.smooth:
        from instanthmr.smoothing import TemporalSmoother
        smoother = TemporalSmoother()
    faces = np.asarray(pipeline.hmr.faces, dtype=np.uint32)
    emit(type="log", message=f"视频 {width}x{height} @ {fps:.2f} fps，"
                             f"{'全部 ' + str(total) + ' 帧' if total else '按上限处理'}，设备 {device}"
                             + ("，输出平滑 开启" if smoother is not None else ""))

    vertices_path = out / "mesh" / "vertices.bin"
    joints_path = out / "mesh" / "joints.bin"
    track_frames = []
    joints2d_all = []
    joints3d_all = []
    boxes_all = []
    last_vertices = None
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
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = index / fps
            result = pipeline(rgb, t)
            durations.append((result.detector_ms, result.hmr_ms))
            persons = smoother(result.persons, t) if smoother is not None else result.persons
            if persons:
                person = persons[0]
                verts = np.asarray(person.vertices_cam, np.float32)
                verts = np.stack([verts[:, 0], -verts[:, 1], -verts[:, 2]], axis=1)
                joints = np.asarray(person.joints_3d_cam, np.float32)
                joints = np.stack([joints[:, 0], -joints[:, 1], -joints[:, 2]], axis=1)
                last_vertices, last_joints = verts, joints
                seen_person = True
                box = [round(float(v), 2) for v in person.bbox]
                score = round(float(person.confidence), 4)
                keypoints = [[round(float(x), 2), round(float(y), 2)] for x, y in person.joints_2d]
                root = (joints[9] + joints[10]) * 0.5
                root_trajectory.append([round(float(root[0]), 3), round(float(root[2]), 3)])
            else:
                empty += 1
                if last_vertices is None:
                    last_vertices = np.zeros((18439, 3), np.float32)
                    last_joints = np.zeros((70, 3), np.float32)
                box, score, keypoints = None, 0.0, None
                root_trajectory.append(root_trajectory[-1] if root_trajectory else [0.0, 0.0])
            # Hold the last valid mesh on frames where the detector briefly loses the person.
            vertex_file.write(np.asarray(last_vertices, np.float16).tobytes())
            joint_file.write(np.asarray(last_joints, np.float16).tobytes())
            if seen_person:
                bbox_min = np.minimum(bbox_min, last_vertices.min(axis=0))
                bbox_max = np.maximum(bbox_max, last_vertices.max(axis=0))
            joints2d_all.append(person.joints_2d if persons else np.zeros((70, 2), np.float32))
            joints3d_all.append(last_joints)
            boxes_all.append(box if box is not None else [0, 0, 0, 0])
            track_frames.append({"box": box, "score": score, "keypoints": keypoints})
            index += 1
            if index % 3 == 0 or (total and index >= total):
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
        raise RuntimeError("没有生成有效的 3D 网格")

    faces.tofile(out / "mesh" / "faces.bin")
    vertices = np.fromfile(vertices_path, dtype=np.float16).reshape(frames, -1, 3)
    joints = np.fromfile(joints_path, dtype=np.float16).reshape(frames, -1, 3)
    joints2d = np.asarray(joints2d_all, np.float32)
    boxes = np.asarray(boxes_all, np.float32)
    np.savez(out / "motion.npz", vertices=vertices, faces=faces, joints=joints,
             joints_2d=joints2d, boxes=boxes, fps=np.float32(fps))

    from instanthmr.skeleton import JOINT_NAMES, SKELETON_EDGES
    track = {
        "fps": fps, "width": width, "height": height, "start": 0, "end": frames - 1,
        "edges": [list(edge) for edge in SKELETON_EDGES],
        "joint_names": JOINT_NAMES,
        "frames": track_frames,
    }
    (out / "track.json").write_text(json.dumps(track, ensure_ascii=False), encoding="utf-8")

    det = float(np.median([d for d, _ in durations])) if durations else 0.0
    hmr = float(np.median([h for _, h in durations])) if durations else 0.0
    meta = {
        "engine": "instanthmr",
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
        "timings": {"detector_ms": round(det, 2), "hmr_ms": round(hmr, 2),
                    "elapsed_s": round(time.perf_counter() - started, 2)},
        "device": device,
        "model": Path(args.model).name,
        "pose_filter": args.pose_filter or "none",
        "smooth": bool(args.smooth),
        "source_total_frames": frame_count or None,
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
