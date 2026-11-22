"""TDPT (ThreeDPoseTracker) VNect decode, inference and post-processing.

Two model generations are supported:

- ``MobileNet3D3_6.onnx`` (public upstream): 1x3x224x224 single frame,
  heatmap3D (24*14, 14, 14), offset3D (3*24*14, 14, 14).
- ``ThreeDPoseTracker_3frames.onnx`` (ThreeDPoseLibrary / SDK): three
  448x448 RGB inputs ``input.1`` (current), ``input.4`` (t-1),
  ``input.7`` (t-2), heatmap3D (24*28, 28, 28), offset3D (3*24*28, 28, 28).

Channel layouts (both generations):
- heatmap3D: j*heat + z          (j = joint 0..23, z = depth slice)
- offset3D:  g*24*heat + j*heat + z,  g in {0: x, 1: y, 2: z}

Post-processing follows ThreeDPoseLibrary.TDPBarracudaRunner exactly:
Kalman (constant-velocity, 50-frame warm-up) -> cascaded low-pass
(order 7, smooth 0.9, 10-frame warm-up) -> One-Euro (3.5 / 0.005 / 1.2),
applied per joint in that order, and independently to the 2D and 3D heads
(the library keeps two sets of filters). The four derived joints are computed
from the filtered 24 joints afterwards, exactly like TDPAvatarController.

An optional avatar-root depth smoother (``RootDepthSmoother``) ports
TDPAvatarController's body-height estimate + root z Kalman; it translates the
whole skeleton in scene units and is disabled by default.

Decoded coordinates are normalised to the canonical 224-unit space so all
downstream code (scale 0.01, project2d, mesh synthesis) behaves identically
for both model generations.
"""
from __future__ import annotations

import math
from collections import deque
from typing import NamedTuple

import numpy as np

JOINTS = 24
HEAT = 14
IMG = 224

# Canonical output unit: both model generations are mapped to a 224px frame.
UNIT = IMG

JOINT_NAMES = [
    "rShldrBend", "rForearmBend", "rHand", "rThumb2", "rMid1",
    "lShldrBend", "lForearmBend", "lHand", "lThumb2", "lMid1",
    "lEar", "lEye", "rEar", "rEye", "Nose",
    "rThighBend", "rShin", "rFoot", "rToe",
    "lThighBend", "lShin", "lFoot", "lToe",
    "abdomenUpper",
]

DERIVED_NAMES = ["hip", "head", "neck", "spine"]

ALL_JOINT_NAMES = JOINT_NAMES + DERIVED_NAMES

# Stick-figure edges over the 28-joint layout (indices into ALL_JOINT_NAMES).
EDGES = [
    (26, 23),            # neck -> spine(abdomenUpper)
    (23, 24),            # spine -> hip
    (24, 19), (19, 20), (20, 21), (21, 22),   # left leg
    (24, 15), (15, 16), (16, 17), (17, 18),   # right leg
    (26, 25),            # neck -> head
    (25, 14),            # head -> nose
    (25, 10), (25, 12),  # head -> ears
    (26, 5), (5, 6), (6, 7),                   # left arm
    (26, 0), (0, 1), (1, 2),                   # right arm
]
EDGES_2D = EDGES  # the video overlay uses the same stick figure

# Capsule-mesh template: a unit cylinder along +y, RINGS x SEGS vertices, open ends.
_TUBE_RINGS = 3
_TUBE_SEGS = 8
TUBE_VERTS = _TUBE_RINGS * _TUBE_SEGS
TUBE_RADIUS = 3.0          # model units (image is 224 px across)


def _tube_template() -> tuple[np.ndarray, np.ndarray]:
    """Unit cylinder (radius 1, height 1, along +y) vertices and quad faces."""
    verts = []
    faces = []
    for ring in range(_TUBE_RINGS):
        y = ring / (_TUBE_RINGS - 1)
        for seg in range(_TUBE_SEGS):
            ang = 2 * math.pi * seg / _TUBE_SEGS
            verts.append((math.cos(ang), y, math.sin(ang)))
    for ring in range(_TUBE_RINGS - 1):
        for seg in range(_TUBE_SEGS):
            a = ring * _TUBE_SEGS + seg
            b = ring * _TUBE_SEGS + (seg + 1) % _TUBE_SEGS
            c = (ring + 1) * _TUBE_SEGS + seg
            d = (ring + 1) * _TUBE_SEGS + (seg + 1) % _TUBE_SEGS
            faces.append((a, b, c))
            faces.append((b, d, c))
    return (np.asarray(verts, np.float32),
            np.asarray(faces, np.uint32))


_TUBE_V, _TUBE_F = _tube_template()
_NUM_MESH_VERTS = len(EDGES) * TUBE_VERTS
_NUM_MESH_FACES = len(EDGES) * len(_TUBE_F)

MESH_FACES = np.empty((_NUM_MESH_FACES, 3), np.uint32)
for e, _ in enumerate(EDGES):
    start_face = e * len(_TUBE_F)
    MESH_FACES[start_face:start_face + len(_TUBE_F)] = _TUBE_F + e * TUBE_VERTS


def decode(heat3d: np.ndarray, off3d: np.ndarray, heat: int = HEAT,
           img: int = IMG, unit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Argmax + offset decode -> (24, 3) keypoints and per-joint scores.

    ``heat3d`` / ``off3d`` are channel-first (C, heat, heat).  Coordinates are
    returned in pixels of the model input (``img``) unless ``unit`` is given.
    The library filters run in this native scale; callers scale to canonical
    224 units afterwards (``VNectRunner.unit_scale``).
    """
    unit = img if unit is None else unit
    h = heat3d.reshape(JOINTS, heat, heat * heat)
    flat = h.reshape(JOINTS, -1)
    flat_idx = flat.argmax(axis=1)
    scores = flat[np.arange(JOINTS), flat_idx]
    z_idx = flat_idx // (heat * heat)
    yx = flat_idx % (heat * heat)
    y_idx = yx // heat
    x_idx = yx % heat
    ch = np.arange(JOINTS) * heat + z_idx
    stride = JOINTS * heat
    ox = off3d[ch, y_idx, x_idx]
    oy = off3d[stride + ch, y_idx, x_idx]
    oz = off3d[2 * stride + ch, y_idx, x_idx]
    scale = float(unit) / float(img)
    x = (((ox + 0.5 + x_idx) / heat) * img - img / 2) * scale
    y = (img - ((oy + 0.5 + y_idx) / heat) * img - img / 2) * scale
    z = (((oz + 0.5 + (z_idx - heat / 2)) / heat) * img) * scale
    return np.stack([x, y, z], axis=1).astype(np.float32), scores.astype(np.float32)


def decode2d(heat2d: np.ndarray, off2d: np.ndarray, heat: int = HEAT,
             img: int = IMG, unit: int | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """2D head: argmax + offset decode -> (24, 3) keypoints (z=0), scores.

    Same conventions as :func:`decode` / :func:`project2d` (x right, y up),
    i.e. the library's `224 - u` x axis is negated back to screen convention.
    """
    unit = img if unit is None else unit
    flat = heat2d.reshape(JOINTS, -1)
    flat_idx = flat.argmax(axis=1)
    scores = flat[np.arange(JOINTS), flat_idx]
    y_idx = flat_idx // heat
    x_idx = flat_idx % heat
    ch = np.arange(JOINTS)
    u = (off2d[ch, y_idx, x_idx] + 0.5 + x_idx) / heat * img
    v = (off2d[JOINTS + ch, y_idx, x_idx] + 0.5 + y_idx) / heat * img
    scale = float(unit) / float(img)
    x = (u - img / 2) * scale
    y = (img / 2 - v) * scale
    z = np.zeros(JOINTS, np.float32)
    return (np.stack([x, y, z], axis=1).astype(np.float32),
            scores.astype(np.float32))


def derive(kps: np.ndarray) -> np.ndarray:
    """24 -> 28 joints (hip, head, neck, spine per VNectBarracudaRunner)."""
    out = np.zeros((28, 3), np.float32)
    out[:24] = kps
    lc = (kps[15] + kps[19]) / 2.0
    out[24] = (kps[23] + lc) / 2.0                       # hip
    out[26] = (kps[0] + kps[5]) / 2.0                    # neck
    c_ear = (kps[12] + kps[10]) / 2.0
    hv = c_ear - out[26]
    norm = float(np.linalg.norm(hv))
    if norm > 1e-6:
        nhv = hv / norm
        nv = kps[14] - out[26]
        out[25] = out[26] + nhv * float(np.dot(nhv, nv))
    else:
        out[25] = out[26]
    out[27] = kps[23]                                     # spine
    return out


def project2d(kps: np.ndarray, width: int, height: int) -> np.ndarray:
    """Orthographic projection back to pixels (input was stretched to 224)."""
    px = (kps[:, 0] + IMG / 2) / IMG * width
    py = (IMG / 2 - kps[:, 1]) / IMG * height
    return np.stack([px, py], axis=1)


def synthesize_mesh(kps: np.ndarray, radius: float = TUBE_RADIUS) -> np.ndarray:
    """Stick figure -> capsule mesh with static topology (verts in edge order)."""
    out = np.empty((_NUM_MESH_VERTS, 3), np.float32)
    for e, (a, b) in enumerate(EDGES):
        pa = kps[a]
        pb = kps[b]
        dir_vec = pb - pa
        length = float(np.linalg.norm(dir_vec))
        if length < 1e-6:
            dir_vec = np.array([0.0, 1.0, 0.0], np.float32)
            length = 0.0
        # Orthonormal basis mapping +y onto dir_vec.
        axis_y = dir_vec / max(length, 1e-6)
        ref = np.array([1.0, 0.0, 0.0], np.float32)
        if abs(float(np.dot(axis_y, ref))) > 0.99:
            ref = np.array([0.0, 0.0, 1.0], np.float32)
        axis_x = np.cross(axis_y, ref)
        axis_x /= max(float(np.linalg.norm(axis_x)), 1e-6)
        axis_z = np.cross(axis_x, axis_y)
        rot = np.stack([axis_x, axis_y, axis_z], axis=1)  # maps +y -> dir
        tube = _TUBE_V.copy()
        tube[:, 0] *= radius
        tube[:, 1] *= length
        tube[:, 2] *= radius
        out[e * TUBE_VERTS:(e + 1) * TUBE_VERTS] = pa + tube @ rot.T
    return out


# --------------------------------------------------------------------------- filters
# Faithful ports of ThreeDPoseLibrary's KalmanFilter / LowPassFilter /
# OneEuroFilter (TDPBarracudaRunner defaults).

class KalmanFilter3D:
    """Constant-velocity 3D Kalman filter (6-state, 3-measurement).

    Port of ThreeDPoseLibrary.KalmanFilter + ConstantVelocity3DModel.
    Like the original, the first 50 measurements pass through unfiltered and
    the returned value after warm-up is the *predicted* next position.
    """

    WARMUP = 50

    def __init__(self, time_interval: float = 0.45, noise: float = 0.4):
        dt = float(time_interval)
        self.transition = np.array([
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ], np.float32)
        g = np.zeros((6, 3), np.float32)
        g[0, 0] = dt * dt / 2.0
        g[1, 0] = dt
        g[2, 1] = dt * dt / 2.0
        g[3, 1] = dt
        g[4, 2] = dt * dt / 2.0
        g[5, 2] = dt
        self.process_noise = g @ np.diag(np.full(3, noise, np.float32)) @ g.T
        self.measurement_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], np.float32)
        self.measurement_noise = np.eye(3, dtype=np.float32)
        self.state = np.zeros(6, np.float32)
        self.covariance = self.process_noise.copy()
        self.kalman_gain = np.zeros((6, 3), np.float32)
        self.count = 0
        self.predict()

    def predict(self) -> None:
        f = self.transition
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + self.process_noise
        h = self.measurement_matrix
        residual_cov = h @ self.covariance @ h.T + self.measurement_noise
        residual_inv = np.linalg.inv(residual_cov)
        self.kalman_gain = self.covariance @ h.T @ residual_inv

    def correct(self, measurement: np.ndarray) -> None:
        h = self.measurement_matrix
        innovation = np.asarray(measurement, np.float32) - h @ self.state
        self.state = self.state + self.kalman_gain @ innovation
        # ThreeDPoseLibrary uses P^T here, not P; keep it identical.
        eye = np.eye(6, dtype=np.float32)
        self.covariance = (eye - self.kalman_gain @ h) @ self.covariance.T

    def correct_and_predict(self, kp: np.ndarray) -> np.ndarray:
        kp = np.asarray(kp, np.float32)
        self.correct(kp)
        self.predict()
        if self.count >= self.WARMUP:
            return np.array([self.state[0], self.state[2], self.state[4]],
                            np.float32)
        self.count += 1
        return kp


class LowPassFilter3D:
    """Cascaded one-pole low-pass, order N (ThreeDPoseLibrary.LowPassFilter)."""

    WARMUP = 10

    def __init__(self, order: int = 7, smooth: float = 0.9):
        self.order = max(1, min(int(order), 10))
        self.smooth = float(smooth)
        self.prev = np.zeros((10, 3), np.float32)
        self.count = 0

    def correct_and_predict(self, kp: np.ndarray) -> np.ndarray:
        kp = np.asarray(kp, np.float32)
        s = self.smooth
        self.prev[0] = kp
        for i in range(1, self.order):
            self.prev[i] = self.prev[i] * (1.0 - s) + self.prev[i - 1] * s
        self.prev[0] = self.prev[0] * (1.0 - s) + self.prev[self.order - 1] * s
        if self.count >= self.WARMUP:
            return self.prev[0].copy()
        self.count += 1
        return kp


class OneEuroFilter3D:
    """One-Euro filter matching ThreeDPoseLibrary.OneEuroFilter.

    Notable quirk kept from the original: ``x_prev`` is the raw measurement,
    not the filtered output.  On the very first call the original effectively
    passes the measurement through (Unity's Time.time is already large); we do
    the same and seed the timestamp.
    """

    def __init__(self, min_cutoff: float = 3.5, beta: float = 0.005,
                 d_cutoff: float = 1.2):
        self.min_cutoff = np.full(3, float(min_cutoff), np.float32)
        self.beta = float(beta)
        self.d_cutoff = np.full(3, float(d_cutoff), np.float32)
        self.x_prev = np.zeros(3, np.float32)
        self.dx_prev = np.zeros(3, np.float32)
        self.t_prev: float | None = None

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: np.ndarray) -> np.ndarray:
        x = 2.0 * math.pi * cutoff * dt
        return x / (x + 1.0)

    @staticmethod
    def _exponential_smoothing(alpha: np.ndarray, x: np.ndarray,
                               x_prev: np.ndarray) -> np.ndarray:
        return alpha * x + (1.0 - alpha) * x_prev

    def correct_and_predict(self, kp: np.ndarray, t: float) -> np.ndarray:
        kp = np.asarray(kp, np.float32)
        if self.t_prev is None:
            self.x_prev = kp.copy()
            self.dx_prev = np.zeros(3, np.float32)
            self.t_prev = float(t)
            return kp
        dt = float(t) - self.t_prev
        if dt <= 1e-6:
            dt = 1e-6
        rate = (kp - self.x_prev) / dt
        dx = self._exponential_smoothing(
            self._smoothing_factor(dt, self.d_cutoff), rate, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx)
        out = self._exponential_smoothing(
            self._smoothing_factor(dt, cutoff), kp, self.x_prev)
        self.x_prev = kp.copy()
        self.dx_prev = dx
        self.t_prev = float(t)
        return out


class RootDepthSmoother:
    """Avatar-root depth smoothing (port of TDPAvatarController.PoseUpdate).

    The Unity controller estimates the body height from joint distances and
    turns its deviation from the expected height into a z offset for the
    avatar root:

        num2 = (tall / centerTall - 1) * DistanceToPerson
        z    = Kalman(num2 * ZMovementSensitivity)   # 0.5 s / 0.5, 50-frame warm-up

    ``DistanceToPerson`` defaults to 3 m and ``centerTall`` is
    ``56 * DistanceToPerson`` canonical units (equivalent to the library's
    336 at 3 m in 448 units).  The returned offset is in scene units and
    translates the whole skeleton in z.

    Only the height estimate and root z are ported; the avatar bone rotations
    and the foot/leg lock state (CheckLeg) do not affect a joint/mesh output.
    """

    VISIBLE_THRESHOLD = 0.05

    def __init__(self, distance_to_person: float = 3.0,
                 z_sensitivity: float = 0.5,
                 time_interval: float = 0.5, noise: float = 0.5):
        self.distance = float(distance_to_person)
        self.z_sensitivity = float(z_sensitivity)
        self.center_tall = 56.0 * self.distance
        self.tall = self.center_tall
        self.prev_tall = self.center_tall
        self.prev_head = 0.0
        self.prev_head_neck = 0.0
        self.prev_neck_spine = 0.0
        self.prev_spine_crotch = 0.0
        self.prev_thigh = 0.0
        self.prev_shin = 0.0
        self.kalman = KalmanFilter3D(time_interval, noise)
        self.offset = 0.0

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def update(self, joints28: np.ndarray, scores24: np.ndarray) -> float:
        """Advance one valid frame; returns the root z offset (scene units)."""
        p = np.asarray(joints28, np.float32)
        scores = np.asarray(scores24, np.float32)
        vis = np.zeros(28, bool)
        vis[:24] = scores >= self.VISIBLE_THRESHOLD
        vis[25] = bool(vis[11] and vis[13])   # head <- eyes
        vis[26] = bool(vis[5] and vis[0])     # neck <- shoulders
        vis[27] = bool(vis[23])               # spine <- abdomenUpper

        def seg(a: int, b: int, fallback: float) -> float:
            return self._dist(p[a], p[b]) if (vis[a] and vis[b]) else fallback

        tall_head = seg(10, 12, self.prev_head)
        tall_head_neck = seg(25, 26, self.prev_head_neck)
        tall_neck_spine = seg(26, 27, self.prev_neck_spine)
        # The original falls back to prevTallThigh for both shin segments and
        # keeps the left/right average; replicate it verbatim.
        tall_shin = (seg(16, 17, self.prev_thigh)
                     + seg(20, 21, self.prev_thigh)) * 0.5
        tall_thigh = (seg(15, 16, self.prev_thigh)
                      + seg(19, 20, self.prev_thigh)) * 0.5
        tall_spine_crotch = (
            self._dist(p[27], (p[15] + p[19]) * 0.5)
            if (vis[15] and vis[19] and vis[27])
            else self.prev_spine_crotch)

        tall = ((tall_head_neck + tall_neck_spine + tall_spine_crotch
                 + tall_thigh + tall_shin) * 0.5 + self.prev_tall * 0.5)
        num2 = (tall / self.center_tall - 1.0) * self.distance

        self.prev_tall = tall
        self.prev_head = tall_head * 0.3 + self.prev_head * 0.7
        self.prev_head_neck = tall_head_neck * 0.3 + self.prev_head_neck * 0.7
        # The library swaps these two coefficient pairs; keep for fidelity.
        self.prev_neck_spine = (self.prev_neck_spine * 0.3
                                + tall_neck_spine * 0.7)
        self.prev_spine_crotch = (self.prev_spine_crotch * 0.3
                                  + tall_spine_crotch * 0.7)
        self.prev_thigh = tall_thigh * 0.3 + self.prev_thigh * 0.7
        self.prev_shin = tall_shin * 0.3 + self.prev_shin * 0.7

        measurement = np.array([0.0, 0.0, num2 * self.z_sensitivity],
                               np.float32)
        self.offset = float(self.kalman.correct_and_predict(measurement)[2])
        return self.offset


class PoseFilter:
    """ThreeDPoseLibrary filter cascade for the 24 model joints."""

    def __init__(self, joints: int = JOINTS, kalman: bool = True,
                 lowpass: bool = True, one_euro: bool = True,
                 kalman_interval: float = 0.45, kalman_noise: float = 0.4,
                 lpf_order: int = 7, lpf_smooth: float = 0.9,
                 one_euro_min_cutoff: float = 3.5, one_euro_beta: float = 0.005,
                 one_euro_d_cutoff: float = 1.2):
        self.joints = int(joints)
        self.kalman_enabled = bool(kalman)
        self.lowpass_enabled = bool(lowpass)
        self.one_euro_enabled = bool(one_euro)
        self.kalman = ([KalmanFilter3D(kalman_interval, kalman_noise)
                        for _ in range(self.joints)] if kalman else None)
        self.lowpass = ([LowPassFilter3D(lpf_order, lpf_smooth)
                         for _ in range(self.joints)] if lowpass else None)
        self.one_euro = ([OneEuroFilter3D(one_euro_min_cutoff, one_euro_beta,
                                          one_euro_d_cutoff)
                          for _ in range(self.joints)] if one_euro else None)

    @property
    def enabled(self) -> bool:
        return self.kalman_enabled or self.lowpass_enabled or self.one_euro_enabled

    def apply(self, kps: np.ndarray, t: float) -> np.ndarray:
        """Filter (J, 3) joints.  Order matches TDPBarracudaRunner."""
        kps = np.asarray(kps, np.float32)
        if not self.enabled:
            return kps
        out = np.empty_like(kps)
        for j in range(kps.shape[0]):
            p = kps[j]
            if self.kalman is not None:
                p = self.kalman[j].correct_and_predict(p)
            if self.lowpass is not None:
                p = self.lowpass[j].correct_and_predict(p)
            if self.one_euro is not None:
                p = self.one_euro[j].correct_and_predict(p, t)
            out[j] = p
        return out


# --------------------------------------------------------------------------- runner

def _classify_3d_output(shape) -> tuple[str, int, str] | None:
    """Identify a 3D heatmap/offset output in NCHW or NHWC layout."""
    if len(shape) != 4 or not all(isinstance(d, int) for d in shape):
        return None
    dims = list(shape)
    for layout in ("nchw", "nhwc"):
        if layout == "nchw":
            c, h, w = dims[1], dims[2], dims[3]
        else:
            h, w, c = dims[1], dims[2], dims[3]
        if h != w:
            continue
        if c == 3 * JOINTS * h:
            return "offset", h, layout
        if c == JOINTS * h:
            return "heat", h, layout
    return None


def _classify_2d_output(shape, heat: int) -> tuple[str, str] | None:
    """Identify a 2D heatmap/offset output in NCHW or NHWC layout."""
    if len(shape) != 4 or not all(isinstance(d, int) for d in shape):
        return None
    dims = list(shape)
    for layout in ("nchw", "nhwc"):
        if layout == "nchw":
            c, h, w = dims[1], dims[2], dims[3]
        else:
            h, w, c = dims[1], dims[2], dims[3]
        if h == heat and w == heat:
            if c == 2 * JOINTS:
                return "offset", layout
            if c == JOINTS:
                return "heat", layout
    return None


class VNectFrame(NamedTuple):
    """One inference result in native model pixels (filters run here)."""

    joints3d: np.ndarray            # (24, 3)
    scores: np.ndarray              # (24,) 3D head scores
    joints2d: np.ndarray | None     # (24, 3) with z=0, or None
    scores2d: np.ndarray | None     # (24,) 2D head scores, or None


class VNectRunner:
    """ONNX runner with automatic model-generation detection and frame window.

    The three-input model is fed a sliding window of consecutive frames,
    exactly like TDPBarracudaRunner.UpdateTDPAsync: on the first frame all
    three inputs receive the same frame, afterwards each new frame shifts
    ``input.7 <- input.4 <- input.1 <- current``.
    """

    def __init__(self, model_path, providers: list[str] | None = None):
        import onnxruntime as ort
        ort.set_default_logger_severity(3)  # silence UTF-16 console warnings
        self.model_path = str(model_path)
        self.session = ort.InferenceSession(
            self.model_path, providers=providers or ["CPUExecutionProvider"])
        specs = self.session.get_inputs()
        if not specs:
            raise RuntimeError("模型没有输入")
        self.input_names = [spec.name for spec in specs]
        self.input_count = len(specs)
        first_shape = specs[0].shape
        if len(first_shape) != 4:
            raise RuntimeError(f"不支持的模型输入形状: {first_shape}")
        self.input_layout = "nchw" if first_shape[1] == 3 else "nhwc"
        self.input_size = int(first_shape[2] if self.input_layout == "nchw"
                              else first_shape[1])
        if self.input_size not in (224, 448):
            raise RuntimeError(f"不支持的模型输入尺寸: {self.input_size}")
        self.frame_window = self.input_count if self.input_count > 1 else 1
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_window)

        heat_spec = off_spec = None
        for spec in self.session.get_outputs():
            info = _classify_3d_output(spec.shape)
            if info is None:
                continue
            if info[0] == "heat":
                heat_spec = (spec.name, info[1], info[2])
            else:
                off_spec = (spec.name, info[1], info[2])
        if heat_spec is None or off_spec is None:
            raise RuntimeError("模型输出与 VNect 布局不符")
        if heat_spec[1] != off_spec[1]:
            raise RuntimeError("heatmap / offset 网格尺寸不一致")
        self.heat_size = heat_spec[1]
        heat2d_spec = off2d_spec = None
        for spec in self.session.get_outputs():
            info = _classify_2d_output(spec.shape, self.heat_size)
            if info is None:
                continue
            if info[0] == "heat":
                heat2d_spec = (spec.name, info[1])
            else:
                off2d_spec = (spec.name, info[1])
        self.has_2d = heat2d_spec is not None and off2d_spec is not None
        self._heat2d_spec = heat2d_spec
        self._off2d_spec = off2d_spec
        # Native model pixels -> canonical 224-unit space used by the viewer.
        self.unit_scale = UNIT / float(self.input_size)
        self._heat_spec = heat_spec
        self._off_spec = off_spec

    # ------------------------------------------------------------------ helpers
    @property
    def model_kind(self) -> str:
        if self.input_count == 1:
            return f"single-frame {self.input_size}px"
        return f"{self.input_count}-frame {self.input_size}px"

    def reset(self) -> None:
        """Drop the frame window (e.g. after a scene cut / new stream)."""
        self._frames.clear()

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        import cv2
        if bgr.shape[0] != self.input_size or bgr.shape[1] != self.input_size:
            bgr = cv2.resize(bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def _to_layout(self, hwc: np.ndarray) -> np.ndarray:
        arr = hwc.transpose(2, 0, 1) if self.input_layout == "nchw" else hwc
        return np.ascontiguousarray(arr[None], dtype=np.float32)

    def _execute(self, bgr: np.ndarray):
        frame = self._preprocess(bgr)
        if self.frame_window == 1:
            seq = [frame]
        else:
            if not self._frames:
                for _ in range(self.frame_window):
                    self._frames.append(frame)
            else:
                self._frames.append(frame)
            seq = list(reversed(self._frames))  # current, t-1, t-2
        feed = {name: self._to_layout(img)
                for name, img in zip(self.input_names, seq)}
        outputs = self.session.run(None, feed)
        return {spec.name: value
                for spec, value in zip(self.session.get_outputs(), outputs)}

    @staticmethod
    def _to_chw(value: np.ndarray, layout: str) -> np.ndarray:
        out = value[0]
        if layout == "nhwc":
            out = out.transpose(2, 0, 1)
        return np.ascontiguousarray(out)

    def infer(self, bgr: np.ndarray) -> VNectFrame:
        """Run one frame -> 2D/3D keypoints in native pixels + 3D scores.

        Apply the post-processing cascades in this native scale, then multiply
        by :attr:`unit_scale` for the canonical 224-unit output space.
        """
        raw = self._execute(bgr)
        heat = self._to_chw(raw[self._heat_spec[0]], self._heat_spec[2])
        off = self._to_chw(raw[self._off_spec[0]], self._off_spec[2])
        joints3d, scores = decode(heat, off, self.heat_size, self.input_size,
                                  unit=self.input_size)
        joints2d = scores2d = None
        if self.has_2d:
            heat2 = self._to_chw(raw[self._heat2d_spec[0]],
                                 self._heat2d_spec[1])
            off2 = self._to_chw(raw[self._off2d_spec[0]],
                                self._off2d_spec[1])
            joints2d, scores2d = decode2d(heat2, off2, self.heat_size,
                                          self.input_size,
                                          unit=self.input_size)
        return VNectFrame(joints3d, scores, joints2d, scores2d)
