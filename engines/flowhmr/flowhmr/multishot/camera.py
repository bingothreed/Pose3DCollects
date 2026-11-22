"""Camera data in the paired motion's Y-up metre world; no body-model dependency."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    RT: np.ndarray
    K: np.ndarray
    camera_valid: np.ndarray
    shot_ids: np.ndarray
    image_size_hw: np.ndarray
    track_start: int = 0
    fps: int = 30
    version: int = 1

    def __post_init__(self):
        n = len(self.RT)
        if not n or self.RT.shape != (n, 4, 4) or self.K.shape != (n, 3, 3):
            raise ValueError('Camera RT/K dimensions are invalid')
        if self.camera_valid.shape != (n,) or self.camera_valid.dtype.kind != 'b' or self.shot_ids.shape != (n,):
            raise ValueError('Camera masks must match the track')
        if self.shot_ids.dtype.kind not in 'iu' or self.fps != 30 or self.version != 1 or self.track_start < 0:
            raise ValueError('Unsupported camera timeline/version')
        if np.shape(self.image_size_hw) != (2,) or np.any(np.asarray(self.image_size_hw) <= 0):
            raise ValueError('Invalid camera image dimensions')
        if not np.isfinite(self.RT).all() or not np.isfinite(self.K).all():
            raise ValueError('Non-finite camera data')
        R = self.RT[self.camera_valid, :3, :3]
        K = self.K[self.camera_valid]
        if (not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(R), 1, atol=1e-5)
                or not np.allclose(self.RT[:, 3], [0, 0, 0, 1])
                or np.any(K[:, (0, 1), (0, 1)] <= 0)
                or not np.allclose(K[:, 2], [0, 0, 1])):
            raise ValueError('Invalid camera rotation or intrinsics')

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: int(data[name]) if name in ('track_start', 'fps', 'version') else data[name]
                          for name in cls.__dataclass_fields__})

    def arrays(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    def json(self, ground_y_m=0.):
        return {**{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.arrays().items()},
                'world': 'paired_motion_y_up_metres', 'extrinsics': 'world_to_opencv_camera',
                'intrinsics': 'source_image_pixels', 'frame_index': 'track_relative',
                'ground_y_m': float(ground_y_m), 'physical_calibration_guaranteed': False}

    @property
    def centers(self):
        return -np.einsum('tji,tj->ti', self.RT[:, :3, :3], self.RT[:, :3, 3])

    def transformed(self, scale=1., translation=(0., 0., 0.)):
        scale, translation = float(scale), np.asarray(translation, dtype=np.float64)
        if (not np.isfinite(scale) or not 0.01 <= scale <= 100
                or translation.shape not in ((3,), (len(self.RT), 3)) or not np.isfinite(translation).all()):
            raise ValueError('Invalid character camera transform')
        rt = self.RT.copy()
        centers = self.centers * scale + translation
        rt[:, :3, 3] = -np.einsum('tij,tj->ti', rt[:, :3, :3], centers)
        return Camera(rt, self.K.copy(), self.camera_valid.copy(), self.shot_ids.copy(),
                      self.image_size_hw.copy(), self.track_start, self.fps, self.version)


def transform_parameters(value, camera):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - {'scale', 'translation', 'start_frame', 'frame_translations'}:
        raise ValueError('Invalid character camera transform fields')
    scale = float(value.get('scale', 1))
    if 'frame_translations' not in value and 'start_frame' not in value:
        translation = np.asarray(value.get('translation', [0, 0, 0]), dtype=float)
        if translation.shape != (3,):
            raise ValueError('Invalid character camera translation')
        return scale, translation
    start = value.get('start_frame')
    keys = np.asarray(value.get('frame_translations'), dtype=float)
    size = len(camera.RT)
    first = camera.track_start
    if ('translation' in value or keys.ndim != 2 or keys.shape[1] != 3 or not len(keys)
            or not np.isfinite(keys).all() or type(start) is not int
            or not first <= start < first + size or start + len(keys) > first + size):
        raise ValueError('Invalid character camera translation timeline')
    indices = np.clip(np.arange(first, first + size) - start, 0, len(keys) - 1)
    return scale, keys[indices]


def project(RT, K, points):
    xyz = np.einsum('tij,tkj->tki', RT[:, :3, :3], points) + RT[:, None, :3, 3]
    uvw = np.einsum('tij,tkj->tki', K, xyz)
    return uvw[..., :2] / uvw[..., 2:], xyz[..., 2]
