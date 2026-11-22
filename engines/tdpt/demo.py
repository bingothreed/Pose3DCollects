"""TDPT live camera demo (OpenCV window with skeleton overlay).

Mirrors ThreeDPoseTracker's webcam experience: per-frame VNect inference on
CPU, skeleton drawn over the frame, FPS in the title bar.  The three-input
model gets a sliding window of consecutive frames; post-processing follows
ThreeDPoseLibrary (Kalman -> low-pass -> One-Euro on both the 2D and 3D
heads).  Close the window or press ESC / Q to quit.

    python demo.py --camera 0 --model models/MobileNet3D3_6.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tdpt_vnect import EDGES, PoseFilter, VNectRunner, derive, project2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--no-smooth", action="store_true",
                        help="disable Kalman / low-pass / One-Euro post-processing")
    parser.add_argument("--flip", action="store_true",
                        help="flip frames vertically (only if the picture is upside down)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    print(f"[tdpt] loading {model_path.name} (CPU)…", flush=True)
    runner = VNectRunner(model_path)
    print(f"[tdpt] model {runner.model_kind}, heatmap {runner.heat_size}x"
          f"{runner.heat_size}, 2D head {'yes' if runner.has_2d else 'no'}, "
          f"frame window {runner.frame_window}", flush=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    print("[tdpt] camera open — ESC/Q closes, scores: green high / red low", flush=True)

    flip = args.flip
    smooth = not args.no_smooth
    filters3d = PoseFilter() if smooth else None
    filters2d = PoseFilter() if smooth and runner.has_2d else None
    started = time.perf_counter()
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
        out = runner.infer(frame)
        now = t0 - started
        kps3 = out.joints3d
        kps2 = out.joints2d
        if filters3d:
            kps3 = filters3d.apply(kps3, now)
        if filters2d and kps2 is not None:
            kps2 = filters2d.apply(kps2, now)
        draw = kps2 if kps2 is not None else kps3
        kps28 = derive(draw * runner.unit_scale)
        scores[:24] = out.scores2d if out.scores2d is not None else out.scores

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
