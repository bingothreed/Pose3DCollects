"""TDPT live camera demo (OpenCV window with skeleton overlay).

Mirrors ThreeDPoseTracker's webcam experience: per-frame VNect inference on
CPU, skeleton drawn over the frame, FPS in the title bar. Close the window
or press ESC / Q to quit.

    python demo.py --camera 0 --model models/MobileNet3D3_6.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from tdpt_vnect import (ALL_JOINT_NAMES, EDGES, IMG, decode, derive, project2d,
                        OneEuro)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--no-smooth", action="store_true", help="disable 1 Euro filter")
    parser.add_argument("--flip", action="store_true",
                        help="flip frames vertically (only if the picture is upside down)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    print(f"[tdpt] loading {model_path.name} (CPU)…", flush=True)
    ort.set_default_logger_severity(3)  # silence UTF-16 console warnings on Windows
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: np.zeros((1, 3, IMG, IMG), np.float32)})
    by_shape = {tuple(o.shape): o for o in outputs}
    heat_key = (1, 24 * 14, 14, 14)
    off_key = (1, 24 * 3 * 14, 14, 14)
    if heat_key not in by_shape or off_key not in by_shape:
        raise SystemExit("model outputs don't match VNect layout "
                         f"(got {list(by_shape)})")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    print("[tdpt] camera open — ESC/Q closes, scores: green high / red low", flush=True)

    flip = args.flip
    smooth = not args.no_smooth
    filters = [OneEuro() for _ in range(28)] if smooth else None
    scores = np.zeros(28, np.float32)
    fps_smooth = 0.0

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 0)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame, (IMG, IMG)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        outs = sess.run(None, {input_name: x})
        by_shape = {tuple(o.shape): o for o in outs}
        heat3d = by_shape[heat_key][0]
        off3d = by_shape[off_key][0]
        kps, sc = decode(heat3d, off3d)
        kps28 = derive(kps)
        scores[:24] = sc
        if filters is not None:
            now = time.perf_counter()
            for i in range(28):
                kps28[i] = filters[i].filter(kps28[i], now)

        # Draw 2D overlay on the (flipped) frame with score colors.
        px = project2d(kps28, w, h)
        palette = np.zeros((28, 3), np.uint8)
        for i in range(28):
            v = float(np.clip(scores[i] if i < 24 else scores[24], 0.0, 1.0))
            palette[i] = (int(60 * (1 - v) + 40), int(220 * v + 35), 255)
        for a, b in EDGES:
            pa = tuple(int(round(v)) for v in px[a])
            pb = tuple(int(round(v)) for v in px[b])
            cv2.line(frame, pa, pb, (120, 220, 120), 2, cv2.LINE_AA)
        for i in range(28):
            cv2.circle(frame, tuple(int(round(v)) for v in px[i]), 4,
                       tuple(int(v) for v in palette[i]), -1, cv2.LINE_AA)

        dt = time.perf_counter() - t0
        fps_smooth = fps_smooth * 0.9 + (1.0 / max(dt, 1e-6)) * 0.1
        cv2.putText(frame, f"TDPT  {fps_smooth:5.1f} fps  score {scores[:24].mean():.2f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.imshow("TDPT", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty("TDPT", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[tdpt] camera closed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[tdpt] error: {exc}", flush=True)
        sys.exit(1)
