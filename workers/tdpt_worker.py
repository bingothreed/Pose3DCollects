"""Run TDPT (ThreeDPoseTracker VNect) on a video inside the TDPT venv.

The hub launches this with the TDPT project's Python. It supports both the
single-frame MobileNet3D3_6.onnx and the three-input ThreeDPoseLibrary model,
decodes 24 joints from the 2D and 3D heads, applies the library's
Kalman -> low-pass -> One-Euro cascade to both and derives 4 extra joints,
synthesizes a capsule mesh for the 3D viewer and writes the unified result
package (float16 vertex stream + faces + meta + 2D head track) with JSON
lines on stdout.
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
    parser.add_argument("--model", required=True,
                        help="VNect ONNX path (224 single-frame or 448 three-frame)")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--smooth", type=int, default=1,
                        help="1 = Kalman + low-pass + One-Euro post-processing")
    parser.add_argument("--root-smooth", type=int, default=0,
                        help="1 = avatar-root depth Kalman (body height -> z offset)")
    parser.add_argument("--kalman", type=int, default=1,
                        help="enable the per-joint constant-velocity Kalman stage")
    parser.add_argument("--kalman-interval", type=float, default=0.45)
    parser.add_argument("--kalman-noise", type=float, default=0.4)
    parser.add_argument("--lowpass", type=int, default=1,
                        help="enable the per-joint cascaded low-pass stage")
    parser.add_argument("--lowpass-order", type=int, default=7)
    parser.add_argument("--lowpass-smooth", type=float, default=0.9)
    parser.add_argument("--one-euro", type=int, default=1,
                        help="enable the per-joint One-Euro stage")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=3.5)
    parser.add_argument("--one-euro-beta", type=float, default=0.005)
    parser.add_argument("--one-euro-d-cutoff", type=float, default=1.2)
    parser.add_argument("--flip", type=int, default=0,
                        help="1 = flip frames vertically (Unity texture orientation)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help="output scale factor (model units -> scene units)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from tdpt_vnect import (ALL_JOINT_NAMES, EDGES_2D, MESH_FACES, PoseFilter,
                            RootDepthSmoother, VNectRunner, derive, project2d,
                            synthesize_mesh)

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

    model_path = Path(args.model)
    if not model_path.is_file():
        raise RuntimeError(f"模型不存在: {model_path}")
    runner = VNectRunner(model_path)
    emit(type="log", message=f"加载 TDPT (VNect) 模型（CPU）… {runner.model_kind}")
    scale = float(args.scale)
    if scale <= 0:
        raise RuntimeError("缩放值必须大于 0")
    emit(type="log", message=f"视频 {width}x{height} @ {fps:.2f} fps，"
                             f"关节 {len(ALL_JOINT_NAMES)}，缩放 {scale:g}，"
                             f"帧窗口 {runner.frame_window}，"
                             f"2D 头 {'有' if runner.has_2d else '无'}，"
                             f"{'2D+3D Kalman + 低通 + One-Euro' if args.smooth else '无滤波'}"
                             + ("，根节点深度平滑" if args.root_smooth else "")
                             + ("，垂直翻转输入" if args.flip else ""))

    max_frames = max(0, args.max_frames)
    total = min(source_frame_count, max_frames) if source_frame_count and max_frames else (
        source_frame_count or max_frames)
    filter_options = {
        "kalman": bool(args.kalman),
        "kalman_interval": args.kalman_interval,
        "kalman_noise": args.kalman_noise,
        "lowpass": bool(args.lowpass),
        "lpf_order": args.lowpass_order,
        "lpf_smooth": args.lowpass_smooth,
        "one_euro": bool(args.one_euro),
        "one_euro_min_cutoff": args.one_euro_min_cutoff,
        "one_euro_beta": args.one_euro_beta,
        "one_euro_d_cutoff": args.one_euro_d_cutoff,
    }
    filters3d = PoseFilter(**filter_options) if args.smooth else None
    filters2d = PoseFilter(**filter_options) if args.smooth else None
    root_smoother = RootDepthSmoother() if args.root_smooth else None

    vertices_path = out / "mesh" / "vertices.bin"
    joints_path = out / "mesh" / "joints.bin"
    track_frames = []
    joints2d_all = []
    boxes_all = []
    last_joints = None
    last_joints_2d = None
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
            t0 = time.perf_counter()
            frame = runner.infer(bgr)
            durations.append((time.perf_counter() - t0) * 1000.0)

            mean_score = float(frame.scores.mean())
            # TDPBarracudaRunner updates its filter cascade for every model
            # output, before the avatar controller performs score gating.  Do
            # the same here: this keeps Kalman's velocity state continuous and
            # makes short occlusions use its predicted trajectory instead of a
            # frozen previous pose.
            now = (index + 1) / fps
            kps3, kps2 = frame.joints3d, frame.joints2d
            if filters3d is not None:
                kps3 = filters3d.apply(kps3, now)
            if filters2d is not None and kps2 is not None:
                kps2 = filters2d.apply(kps2, now)
            predicted_joints = derive(kps3 * runner.unit_scale)
            predicted_joints_2d = (derive(kps2 * runner.unit_scale)
                                   if kps2 is not None else predicted_joints.copy())
            if mean_score >= SCORE_THRESHOLD:
                last_joints = predicted_joints
                last_joints_2d = predicted_joints_2d
                if root_smoother is not None:
                    root_smoother.update(last_joints, frame.scores)
                seen_person = True
            else:
                empty += 1
                if last_joints is None:
                    last_joints = np.zeros((28, 3), np.float32)
                else:
                    last_joints = predicted_joints
                if last_joints_2d is None:
                    last_joints_2d = np.zeros((28, 3), np.float32)
                else:
                    last_joints_2d = predicted_joints_2d

            px = project2d(last_joints_2d, width, height)
            scaled_joints = last_joints * scale
            if root_smoother is not None:
                # Root translation is in scene units, like the Unity avatar.
                scaled_joints[:, 2] += root_smoother.offset
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
        "model_kind": runner.model_kind,
        "input_size": runner.input_size,
        "frame_window": runner.frame_window,
        "heatmap": runner.heat_size,
        "smooth": bool(args.smooth),
        "two_d_head": runner.has_2d,
        "filters": ([name for name, enabled in (
            ("kalman", args.kalman), ("lowpass", args.lowpass),
            ("one_euro", args.one_euro)) if args.smooth and enabled]),
        "filters_2d": ([name for name, enabled in (
            ("kalman", args.kalman), ("lowpass", args.lowpass),
            ("one_euro", args.one_euro)) if args.smooth and runner.has_2d and enabled]),
        "filter_options": filter_options if args.smooth else {},
        "root_smooth": bool(args.root_smooth),
        "root_smooth_final_offset": (round(root_smoother.offset, 4)
                                     if root_smoother is not None else None),
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
