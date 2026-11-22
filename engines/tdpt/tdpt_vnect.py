"""TDPT (ThreeDPoseTracker) VNect decode and utilities.

Ported from digital-standard/ThreeDPoseTracker (VNectBarracudaRunner.cs):

- Model ``MobileNet3D3_6.onnx``: 1x3x224x224 RGB input, outputs
  heatmap3D (1, 24*14, 14, 14) and offset3D (1, 24*3*14, 14, 14).
- heatmap3D channel layout: j*14 + z  (j = joint 0..23, z = depth slice)
- offset3D channel layout:  g*336 + j*14 + z,  g in {0: x, 1: y, 2: z}
- 24 model joints + 4 derived joints (hip, head, neck, spine).
"""
from __future__ import annotations

import math

import numpy as np

JOINTS = 24
DEPTH = 14
HEAT = 14
IMG = 224

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


def decode(heat3d: np.ndarray, off3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmax + offset decode -> (24, 3) keypoints and per-joint scores."""
    h = heat3d.reshape(JOINTS, DEPTH, HEAT * HEAT)
    flat = h.reshape(JOINTS, -1)
    flat_idx = flat.argmax(axis=1)
    scores = flat[np.arange(JOINTS), flat_idx]
    z_idx = flat_idx // (HEAT * HEAT)
    yx = flat_idx % (HEAT * HEAT)
    y_idx = yx // HEAT
    x_idx = yx % HEAT
    ch = np.arange(JOINTS) * DEPTH + z_idx
    ox = off3d[ch, y_idx, x_idx]
    oy = off3d[336 + ch, y_idx, x_idx]
    oz = off3d[672 + ch, y_idx, x_idx]
    x = ((ox + 0.5 + x_idx) / HEAT) * IMG - IMG / 2
    y = IMG - ((oy + 0.5 + y_idx) / HEAT) * IMG - IMG / 2
    z = ((oz + 0.5 + (z_idx - HEAT / 2)) / HEAT) * IMG
    return np.stack([x, y, z], axis=1).astype(np.float32), scores.astype(np.float32)


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


class OneEuro:
    """One-Euro filter (Casiez et al.) for 3D keypoint streams."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        dt = t - self.t_prev
        self.t_prev = t
        dx = (x - self.x_prev) / max(dt, 1e-6)
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = np.clip(self._alpha(cutoff, dt), 0.0, 1.0)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
