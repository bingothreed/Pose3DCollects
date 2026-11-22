"""Rerun viewer: the frame with 2D skeletons and timings, the 3D scene, and a latency plot."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .inference import HMRPrediction
from .skeleton import SKELETON_EDGES

MESH_COLOR = (200, 160, 130)
JOINT_COLOR = (0, 230, 0)
BONE_COLOR = (255, 230, 0)
BOX_COLOR = (0, 200, 255)
FLOOR_COLOR = (110, 110, 110)
FLOOR_MAJOR_COLOR = (175, 175, 175)
FLOOR_AXIS_X = (230, 80, 80)
FLOOR_AXIS_Z = (80, 140, 230)

# ankles, toes, heels: the lowest points of the body, used to place the floor
FLOOR_JOINTS = [13, 14, 15, 16, 17, 18, 19, 20]


class RerunVisualizer:
    def __init__(self, faces: Optional[np.ndarray] = None, mesh_alpha: float = 0.35, spawn: bool = True,
                 save_path: Optional[str] = None, floor: bool = True, viewer_grid: bool = False,
                 floor_size: float = 4.0, floor_spacing: float = 0.25, camera_in_3d: bool = False,
                 fps: float = 30.0):
        import rerun as rr

        self.rr = rr
        self.faces = faces
        self.albedo = (*MESH_COLOR, int(round(255 * min(max(mesh_alpha, 0.0), 1.0))))
        self.floor, self.viewer_grid = floor, viewer_grid
        self.floor_size, self.floor_spacing = floor_size, floor_spacing
        self.camera_in_3d = camera_in_3d
        # The time panel's playback state is part of the blueprint, and Rerun 0.31 applies changes
        # made here to a running viewer. The demo uses that to replay a finished recording in real
        # time, with the clip's sound started at the same moment. `fps` is the timeline's rate: the
        # `frame` timeline is a sequence, and the viewer advances it by this many frames per second.
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.timeline = "frame"
        self.play_state = "Following"
        self.playback_speed = 1.0
        self.loop_mode = "Off"
        if spawn:
            self._put_rerun_on_path()
        rr.init("instanthmr", spawn=spawn)
        if save_path:
            rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        self._n_logged = 0
        self._layout_sent = False
        self._floor_logged = False
        self._centre: Optional[np.ndarray] = None
        self._spread = 0.4

    @staticmethod
    def _put_rerun_on_path() -> None:
        """Make the `rerun` executable that ships inside rerun-sdk findable, so spawning works even when
        python was started without the venv's Scripts directory on PATH."""
        import os
        import shutil

        from rerun_sdk.rerun_cli import __file__ as cli_file  # noqa: F401  (imports the package path)

        if shutil.which("rerun"):
            return
        cli_dir = os.path.dirname(os.path.abspath(cli_file))
        if any(f.lower().startswith("rerun") and f.lower().endswith(".exe") for f in os.listdir(cli_dir)):
            os.environ["PATH"] = cli_dir + os.pathsep + os.environ.get("PATH", "")

    def log(self, frame: int, image_rgb: np.ndarray, persons: Sequence[HMRPrediction],
            timings: dict[str, float]) -> None:
        rr = self.rr
        rr.set_time("frame", sequence=frame)
        for name, ms in timings.items():
            rr.log(f"timing/{name}", rr.Scalars(float(ms)))
        for stale in range(len(persons), self._n_logged):
            rr.log(f"world/person_{stale}", rr.Clear(recursive=True))
        self._n_logged = len(persons)

        for i, p in enumerate(persons):
            joints = p.joints_3d_cam
            rr.log(f"world/person_{i}/joints", rr.Points3D(joints, radii=0.012, colors=JOINT_COLOR))
            rr.log(f"world/person_{i}/skeleton",
                   rr.LineStrips3D(joints[np.array(SKELETON_EDGES)], colors=BONE_COLOR, radii=0.004))
            if self.faces is not None and p.vertices_local is not None:
                rr.log(f"world/person_{i}/mesh", rr.Mesh3D(vertex_positions=p.vertices_cam,
                                                           triangle_indices=self.faces, albedo_factor=self.albedo))

        h, w = image_rgb.shape[:2]
        rr.log("camera/image", rr.Image(draw_overlay(image_rgb, persons, timings)))
        if persons:
            if self.floor and not self._floor_logged:
                self._log_floor(persons)
                self._floor_logged = True
            depth = float(np.median([p.cam_trans[2] for p in persons]))
            rr.log("camera", rr.Pinhole(width=w, height=h, focal_length=persons[0].focal_length,
                                        principal_point=[w / 2, h / 2], image_plane_distance=max(0.25 * depth, 0.2)))
            if not self._layout_sent:
                joints = persons[0].joints_3d_cam
                centre = joints.mean(0)
                self._send_layout(centre, float(np.abs(joints - centre).max()))
                self._layout_sent = True

    def _log_floor(self, persons: Sequence[HMRPrediction]) -> None:
        """A ground grid under the people, placed at the lowest foot joint (Y points down here)."""
        rr = self.rr
        pts = np.concatenate([p.joints_3d_cam for p in persons], axis=0)
        feet = pts[[i for i in FLOOR_JOINTS if i < len(pts)]]
        floor_y = float(feet[:, 1].max()) + 0.02 if len(feet) else float(pts[:, 1].max())
        cx, cz = float(np.median(pts[:, 0])), float(np.median(pts[:, 2]))

        half = self.floor_size / 2.0
        n = max(2, int(round(self.floor_size / self.floor_spacing)))
        step = self.floor_size / n
        lines, colors = [], []
        for i in range(n + 1):
            d = -half + i * step
            major = i % 4 == 0
            color = FLOOR_MAJOR_COLOR if major else FLOOR_COLOR
            x, z = cx + d, cz + d
            lines.append(np.array([[x, floor_y, cz - half], [x, floor_y, cz + half]], np.float32))
            colors.append(color)
            lines.append(np.array([[cx - half, floor_y, z], [cx + half, floor_y, z]], np.float32))
            colors.append(color)
        rr.log("world/floor", rr.LineStrips3D(lines, colors=[(*c, 255) for c in colors],
                                              radii=0.004 if n <= 24 else 0.003), static=True)
        reach = min(half, 0.5)
        rr.log("world/floor/axes", rr.Arrows3D(
            origins=np.array([[cx, floor_y, cz], [cx, floor_y, cz]], np.float32),
            vectors=np.array([[reach, 0.0, 0.0], [0.0, 0.0, reach]], np.float32),
            colors=[(*FLOOR_AXIS_X, 255), (*FLOOR_AXIS_Z, 255)], radii=0.006), static=True)

    def _send_layout(self, centre: Optional[np.ndarray] = None, spread: Optional[float] = None) -> None:
        import rerun.blueprint as rrb

        if centre is not None:
            self._centre = np.asarray(centre, dtype=np.float32)
        if spread is not None:
            self._spread = float(spread)
        centre = self._centre if self._centre is not None else np.zeros(3, dtype=np.float32)
        distance = max(4 * self._spread, 1.5)
        # The 3D view lists everything under `/`; the things a viewer-side query cannot reach are hidden
        # with blueprint `EntityBehavior` overrides instead, which is the same component the viewer's own
        # eye toggle writes. `update_layout()` re-sends this blueprint, so the demo's control panel can
        # flip them live -- Rerun 0.31 gives the viewer no way to hand a click back to us.
        overrides = {}
        if not self.camera_in_3d:
            overrides["camera"] = rrb.EntityBehavior(visible=False)
        if not self.floor:
            overrides["world/floor"] = rrb.EntityBehavior(visible=False)
        # Every blueprint sent to the viewer replaces the active one, so the time panel has to be in
        # all of them -- a layout-only blueprint would reset the playback state to "following".
        time_panel = rrb.TimePanel(timeline=self.timeline, playback_speed=self.playback_speed,
                                   fps=self.fps, play_state=self.play_state,
                                   loop_mode=self.loop_mode)
        self.rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(rrb.Spatial2DView(origin="camera/image", name="Camera"),
                         rrb.TimeSeriesView(origin="timing", name="Latency (ms)"), row_shares=[3, 1]),
            rrb.Spatial3DView(origin="/", name="3D", contents="$origin/**", overrides=overrides,
                              background=rrb.Background(color=[25, 25, 25]),
                              line_grid=rrb.LineGrid3D(visible=self.viewer_grid),
                              eye_controls=rrb.EyeControls3D(
                                  position=[float(centre[0]), float(centre[1]),
                                            float(centre[2] - distance)],
                                  look_target=centre.tolist(), eye_up=[0.0, -1.0, 0.0],
                                  kind=rrb.Eye3DKind.Orbital)),
            column_shares=[2, 3]),
            time_panel))

    def set_time_panel(self, *, play_state: Optional[str] = None, playback_speed: Optional[float] = None,
                       fps: Optional[float] = None, loop_mode: Optional[str] = None,
                       timeline: Optional[str] = None) -> None:
        """Drive the viewer's time panel (play/pause, speed, rate) live, by re-sending the blueprint.

        Rerun 0.31 reads these blueprint components on every frame: asking for ``Playing`` while the
        cursor sits at the end makes the viewer start again from the first frame, and asking for
        ``Following`` moves the cursor to the end. The control panel's replay button uses that pair
        to restart a finished recording from the beginning while it starts the sound.
        """
        if play_state is not None:
            self.play_state = str(play_state)
        if playback_speed is not None:
            self.playback_speed = float(playback_speed)
        if fps is not None:
            self.fps = float(fps)
        if loop_mode is not None:
            self.loop_mode = str(loop_mode)
        if timeline is not None:
            self.timeline = str(timeline)
        self._send_layout()

    def update_layout(self, camera_in_3d: Optional[bool] = None, floor: Optional[bool] = None,
                      viewer_grid: Optional[bool] = None) -> None:
        """Flip what the 3D view draws, live, by re-sending the blueprint."""
        if camera_in_3d is not None:
            self.camera_in_3d = bool(camera_in_3d)
        if floor is not None:
            self.floor = bool(floor)
        if viewer_grid is not None:
            self.viewer_grid = bool(viewer_grid)
        self._send_layout()


def draw_overlay(image_rgb: np.ndarray, persons: Sequence[HMRPrediction], timings: dict[str, float]) -> np.ndarray:
    """The frame with each person's box and 2D skeleton, and the stage timings top left."""
    out = image_rgb.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 400))
    for p in persons:
        x1, y1, x2, y2 = p.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thick)
        pts = p.joints_2d.astype(int)
        for a, b in SKELETON_EDGES:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), BONE_COLOR, thick, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(out, tuple(pt), thick + 1, JOINT_COLOR, -1, cv2.LINE_AA)

    lines = [f"{name:<9s}{ms:6.1f} ms" for name, ms in timings.items()]
    total = sum(timings.values())
    if total > 0:
        lines.append(f"{'total':<9s}{total:6.1f} ms ({1000 / total:.0f} fps)")
    scale = max(0.5, min(h, w) / 1200)
    line_h = int(28 * scale)
    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (int(300 * scale) + 16, 16 + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)
    for k, line in enumerate(lines):
        cv2.putText(out, line, (16, 8 + line_h * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7,
                    (255, 255, 255), max(1, round(scale * 1.5)), cv2.LINE_AA)
    return out
