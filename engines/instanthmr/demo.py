#!/usr/bin/env python3
"""InstantHMR demo: people in an image, a video or a live camera, shown in the Rerun viewer.

    python demo.py --image photo.jpg
    python demo.py --video clip.mp4 --pose-filter one-euro
    python demo.py --camera 0 --model small --fit 1 --device tensorrt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

from instanthmr import DEFAULT_FIT, DEFAULT_MODEL, MODELS, PosePipeline
from instanthmr.smoothing import TemporalSmoother
from instanthmr.video_writer import OverlayVideoWriter
from instanthmr.visualizer import RerunVisualizer, draw_overlay

# `--quality` shortcut -> released model name. `best` and `medium` are the same graph: the highest-accuracy
# tier (EMDB-1 PA-MPJPE 44.4 mm with --fit 2), non-commercial (SAM License). `commercial` is the
# commercially usable tier (68.3 mm).
QUALITY_ALIASES = {
    "best": "medium",
    "medium": "medium",
    "commercial": "medium-commercial",
    "small": "small",
    "tiny": "tiny",
    "nano": "nano",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image file")
    src.add_argument("--video", help="video file")
    src.add_argument("--camera", type=int, metavar="INDEX", help="live camera, e.g. 0")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"{', '.join(MODELS)}, or a path to an .onnx file (default: {DEFAULT_MODEL})")
    p.add_argument("--quality", choices=tuple(QUALITY_ALIASES),
                   help="shortcut for --model: best/medium (non-commercial, highest accuracy), "
                        "commercial, small, tiny, nano (overrides --model; the -fitN graph is picked by --fit)")
    p.add_argument("--fit", type=int, default=DEFAULT_FIT, choices=(0, 1, 2),
                   help="kinematic fit on 1 or 2 crops per person, 0 for the network alone (default: "
                        f"{DEFAULT_FIT}; ignored for an .onnx path)")
    p.add_argument("--device", default="cuda", choices=("cuda", "tensorrt", "cpu", "coreml"))
    p.add_argument("--max-persons", type=int, default=4)
    p.add_argument("--detector", default="medium", choices=("nano", "small", "medium", "base", "large"),
                   help="RF-DETR size (default: medium)")
    p.add_argument("--det-confidence", type=float, default=0.5)
    p.add_argument("--detector-stride", type=int, default=1, metavar="N",
                   help="run the detector every N frames and reuse its boxes in between")
    p.add_argument("--focal", type=float, help="camera focal length in pixels, if known")
    p.add_argument("--pose-filter", choices=("one-euro", "alpha-beta"),
                   help="video: filter each person's pose across frames (more accurate and steadier)")
    p.add_argument("--smooth", action="store_true",
                   help="video: smooth the keypoints and mesh instead (steadiest, but lags on fast motion)")
    p.add_argument("--no-mesh", action="store_true", help="show the skeleton only")
    p.add_argument("--mesh-alpha", type=float, default=0.35, help="mesh opacity (default: 0.35)")
    p.add_argument("--no-floor", action="store_true", help="3D: hide the ground grid drawn under the people")
    p.add_argument("--viewer-grid", action="store_true",
                   help="3D: show the Rerun viewer's built-in line grid (hidden by default)")
    p.add_argument("--show-camera-3d", action="store_true",
                   help="3D: also draw the camera pinhole and the frame's image plane inside the 3D view "
                        "(hidden by default, the 2D 'Camera' panel is unaffected)")
    p.add_argument("--floor-size", type=float, default=4.0, metavar="METRES",
                   help="3D: side length of the ground grid (default: 4.0)")
    p.add_argument("--save-rrd", metavar="PATH", help="also save the Rerun recording")
    p.add_argument("--no-viewer", action="store_true", help="do not open the Rerun viewer")
    p.add_argument("--audio", action="store_true",
                   help="video: start with the clip's sound already on (the control panel can flip it either "
                        "way; Rerun 0.31 has no audio archetype, so the sound cannot be embedded in the .rrd)")
    p.add_argument("--no-audio", dest="audio", action="store_false",
                   help="video: start muted, the panel's sound switch stays available (default)")
    p.add_argument("--no-panel", action="store_true",
                   help="do not open the small control panel (audio and 3D switches then only come from flags)")
    p.add_argument("--audio-device", metavar="INDEX",
                   help="index of the output device for the clip audio (see the panel's 输出 list); "
                        "default: the system default, which is often a headset you are not wearing")
    p.add_argument("--save-video", metavar="PATH",
                   help="also write the 2D view (image + boxes + skeleton + timings) to an mp4 together with "
                        "the clip's audio, so it can be replayed with sound in a normal player")
    p.add_argument("--exit-when-done", action="store_true",
                   help="quit as soon as every frame is processed (by default the viewer and the panel stay "
                        "open so the recording can be replayed with sound)")
    p.add_argument("--auto-replay", action="store_true",
                   help="when every frame is processed, immediately replay the recording from the start with "
                        "the clip's sound in step (same as clicking 从头播放 in the panel)")
    return p.parse_args()


def frames(args):
    """(index, timestamp in s, RGB frame) from the chosen source."""
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"cannot read image {args.image}")
        yield 0, 0.0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video or f'camera {args.camera}'}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, i = time.perf_counter(), 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            t = i / fps if args.video else time.perf_counter() - start
            yield i, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    if args.quality:
        args.model = QUALITY_ALIASES[args.quality]
    fit = f", kinematic fit on {args.fit} crop{'s' if args.fit > 1 else ''}" if args.fit else ""
    print(f"loading InstantHMR ({args.model}{fit}) on {args.device} and RF-DETR {args.detector}...")
    pipeline = PosePipeline(args.model, fit=args.fit, device=args.device, detector=args.detector,
                            det_confidence=args.det_confidence, max_persons=args.max_persons,
                            detector_stride=args.detector_stride, focal=args.focal, pose_filter=args.pose_filter)
    print(f"InstantHMR runs on {pipeline.hmr.provider}")
    # The `frame` timeline is a sequence, so the viewer needs the clip's rate to replay it in real
    # time -- which is what keeps the sound in step when the recording is replayed at the end.
    clip_fps = 30.0
    if args.video:
        cap = cv2.VideoCapture(args.video)
        clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    viewer = RerunVisualizer(None if args.no_mesh else pipeline.hmr.faces, args.mesh_alpha,
                             spawn=not args.no_viewer, save_path=args.save_rrd,
                             floor=not args.no_floor, viewer_grid=args.viewer_grid,
                             floor_size=args.floor_size, camera_in_3d=args.show_camera_3d,
                             fps=clip_fps)
    smoother = TemporalSmoother() if args.smooth else None
    player = panel = writer = None
    now = [0.0]
    clip_audio = None
    if args.video:
        from instanthmr.audio import has_audio, load_clip, open_clip_audio

        has_track = has_audio(args.video)
        if has_track:
            device = int(args.audio_device) if args.audio_device is not None else None
            player = open_clip_audio(args.video, device=device)
            if player is not None:
                player.set_enabled(bool(args.audio))
            if args.save_video:
                try:
                    clip_audio = load_clip(args.video)
                except Exception as e:
                    print(f"[video] could not decode the audio for --save-video: {e}")
    else:
        has_track = False
    if not args.no_panel:
        from instanthmr.control_panel import ControlPanel

        panel = ControlPanel(player, viewer if not args.no_viewer else None,
                             clock=lambda: now[0], has_audio=has_track,
                             video_path=args.save_video)

    out_fps = clip_fps if args.video else 30.0
    if args.save_video:
        if clip_audio is None and args.video:
            from instanthmr.audio import load_clip

            try:
                clip_audio = load_clip(args.video)
            except Exception:
                clip_audio = None

    history = []
    try:
        for i, t, rgb in frames(args):
            now[0] = t
            if panel is not None:
                panel.poll()
            panel_open = panel is not None and not panel.closed
            sync = (player is not None and player.enabled
                    and (True if panel is None else (panel_open and bool(panel.sync.get()))))
            if sync:
                # Audio is the master clock: hold the frames back until the sound reaches them, and never
                # let the playhead run more than `player.lead` past the frame on screen (it waits instead,
                # which is what happens on a machine slower than the clip, e.g. 30 fps input at ~18 fps).
                player.set_limit(t)
                if player.cursor > t + 2.0:   # a long stall, e.g. the model reloading: jump back
                    player.seek(t)
                while player.cursor + 0.02 < t and not player.at_end:
                    if panel_open:
                        panel.poll()
                        if panel.closed:
                            break
                    time.sleep(0.004)
            elif player is not None:
                player.set_limit(None)
            result = pipeline(rgb, t)
            persons = smoother(result.persons, t) if smoother else result.persons
            timings = {"detector": result.detector_ms, "InstantHMR": result.hmr_ms}
            viewer.log(i, rgb, persons, timings)
            if args.save_video:
                overlay = draw_overlay(rgb, persons, timings)
                if writer is None:
                    writer = OverlayVideoWriter(args.save_video, out_fps,
                                                (overlay.shape[1], overlay.shape[0]), clip_audio)
                writer.write(overlay)
            history.append((result.detector_ms, result.hmr_ms))
            if i % 30 == 0:
                print(f"frame {i:5d}  {len(persons)} person(s)  detector {result.detector_ms:5.1f} ms  "
                      f"InstantHMR {result.hmr_ms:5.1f} ms")
    except KeyboardInterrupt:
        pass
    if panel is not None:
        panel.processing = False
    if player is not None:
        gate = player.gated_ratio()
        note = f", waited for the picture {100 * gate:.0f}% of the time" if gate else ""
        print(f"[audio] playhead stopped at {player.cursor:.1f} s of {player.duration:.1f} s{note}")
        if gate > 0.15:
            print("[audio] the picture could not keep up with the clip, so the sound keeps stopping to "
                  "stay aligned; untick 严格对齐 in the panel to let it play straight through, or use a "
                  "lighter --detector / --quality for real-time")
        player.set_limit(None)
    if writer is not None:
        writer.close()
        size_mb = os.path.getsize(args.save_video) / 1024 / 1024 if os.path.exists(args.save_video) else 0.0
        has_sound = "with sound" if clip_audio is not None else "picture only, no audio decoded"
        print(f"[video] wrote {args.save_video} ({size_mb:.1f} MB, {has_sound})")
    if len(history) > 1:
        det, hmr = np.median(np.array(history[1:]), axis=0)
        print(f"median over {len(history) - 1} frames: detector {det:.1f} ms, InstantHMR {hmr:.1f} ms")

    # Keep the viewer usable after the last frame: the panel must outlive the run, otherwise there is no
    # way to hear the clip again when the recording is replayed in the viewer.
    keep_open = panel is not None and not panel.closed and not args.exit_when_done and not args.no_viewer
    if args.auto_replay and not keep_open:
        print("[panel] --auto-replay needs the viewer and the panel to stay open; not starting it")
    if keep_open:
        print("[panel] all frames logged: click 从头播放 in the panel to replay the recording from the start, "
              "with the clip's sound in step (close the panel to quit)")
        if args.auto_replay:
            panel.replay()
        try:
            while not panel.closed and panel.viewer_alive():
                panel.poll()
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        if not panel.viewer_alive():
            print("[panel] the Rerun Viewer was closed, exiting")
            return
    if panel is not None:
        panel.close()
    elif player is not None:
        player.stop()


if __name__ == "__main__":
    main()
