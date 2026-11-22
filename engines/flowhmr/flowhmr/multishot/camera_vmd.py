"""Pure-Python VMD camera writer, verified against installed mmd_tools layout.

One vertical integer FOV per shot. Optical-axis dolly preserves scale at the
reference depth. Unequal fx/fy and principal point offsets remain reported
approximations; the sidecar always retains the original complete intrinsics.
"""
from pathlib import Path
import struct

import numpy as np
from scipy.spatial.transform import Rotation

from .camera import project
from .timeline import intervals

METRES_PER_MMD_UNIT = .08
LINEAR = bytes([20, 107, 20, 107] * 6)
KEY = struct.Struct('<If3f3f24sIB')
TO_BLENDER = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def approximate(camera, joints, *, source_body_visible=None):
    joints = np.asarray(joints, dtype=float)
    n = len(camera.RT)
    if joints.ndim != 3 or joints.shape[0] != n or joints.shape[2] != 3 or not np.isfinite(joints).all():
        raise ValueError('Camera reference joints do not match the motion')
    valid = camera.camera_valid
    if not valid.any():
        raise ValueError('No valid positive-depth camera reference')
    visible = valid if source_body_visible is None else np.asarray(source_body_visible)
    if visible.shape != (n,) or visible.dtype.kind != 'b':
        raise ValueError('Camera reference visibility does not match the motion')
    reference = valid & visible
    background = valid & ~visible
    depth = np.ones(n)
    if reference.any():
        _, depths = project(camera.RT[reference], camera.K[reference], joints[reference])
        depth[reference] = np.median(depths, axis=1)
        if np.any(depth[reference] <= .001):
            raise ValueError('No valid positive-depth camera reference')
        depth[background] = np.median(depth[reference])
    h, w = map(int, camera.image_size_hw)
    K = camera.K
    fy = np.where(valid, K[:, 1, 1], 1.)
    fov = np.degrees(2 * np.arctan(h / (2 * fy)))
    angles = np.full(n, 30, dtype=np.int32)
    shot_report = []
    for a, b in intervals(valid):
        cuts = np.r_[a, np.flatnonzero(np.diff(camera.shot_ids[a:b])) + a + 1, b]
        for first, last in zip(cuts[:-1], cuts[1:]):
            angle = int(np.clip(np.floor(np.median(fov[first:last]) + .5), 1, 179))
            angles[first:last] = angle
            focal = K[first:last, 1, 1]
            shot_report.append({'shot_id': int(camera.shot_ids[first]),
                                'source_frames': [int(first + camera.track_start), int(last + camera.track_start)],
                                'vertical_fov_degrees': angle,
                                'median_rounding_error_degrees': float(angle - np.median(fov[first:last])),
                                'maximum_fov_difference_degrees': float(np.max(np.abs(angle - fov[first:last]))),
                                'focal_change_over_30_percent': bool(np.max(focal) / np.min(focal) > 1.3)})
    focal = h / (2 * np.tan(np.radians(angles) / 2))
    distance = depth * focal / fy
    # A background-only shot has no trustworthy reference plane for dolly.
    # Put the VMD target on the forward ray and keep the measured camera centre.
    distance[background] = depth[background]
    forward = camera.RT[:, 2, :3]
    # Keep the requested optical-axis approximation; don't secretly alter yaw,
    # roll or body to hide the unsupported principal point / pixel aspect.
    centers = camera.centers + forward * (depth - distance)[:, None]
    targets = centers + forward * distance[:, None]
    rt = camera.RT.copy()
    rt[:, :3, 3] = -np.einsum('tij,tj->ti', rt[:, :3, :3], centers)
    approx_K = np.zeros_like(K)
    approx_K[:, 0, 0] = approx_K[:, 1, 1] = focal
    approx_K[:, 0, 2], approx_K[:, 1, 2], approx_K[:, 2, 2] = w / 2, h / 2, 1
    projection = None
    if reference.any():
        original_uv, _ = project(camera.RT[reference], K[reference], joints[reference])
        approx_uv, approx_z = project(rt[reference], approx_K[reference], joints[reference])
        if np.any(approx_z <= .001):
            raise ValueError('Fixed-FOV camera would put body joints behind the camera')
        errors = np.linalg.norm(approx_uv - original_uv, axis=-1)
        projection = dict(p99=float(np.percentile(errors, 99)), max=float(np.max(errors)))
    report = {'method': 'per_shot_integer_vertical_fov_optical_axis_dolly', 'shots': shot_report,
              'image_size_hw': [h, w], 'required_aspect_ratio': w / h,
              'metres_per_mmd_unit': METRES_PER_MMD_UNIT,
              'max_fx_fy_relative_difference': float(np.max(np.abs(K[valid, 0, 0] / K[valid, 1, 1] - 1))),
              'max_principal_point_offset_px': float(np.max(np.abs(K[valid, :2, 2] - [w / 2, h / 2]))),
              'reference_joint_projection_difference_px': projection,
              'background_without_body_reference_ranges': [
                  [a + camera.track_start, b + camera.track_start] for a, b in intervals(background)],
              'limitations': ['one scalar FOV; no skew/principal point/pixel aspect',
                              'dolly matches reference depth, not all body/background depths'],
              'invalid_camera_ranges': [[a + camera.track_start, b + camera.track_start] for a, b in intervals(~valid)]}
    return rt, approx_K, targets, distance, angles, report


def export_camera_vmd(camera, joints, path, *, source_body_visible=None):
    rt, K, targets, distance, angles, report = approximate(camera, joints, source_body_visible=source_body_visible)
    # mmd_tools creates a YXZ parent plus a child camera rotated +90 degrees X.
    # The importer maps VMD xyz to Blender xzy without changing signs.
    rotation = np.tile(np.eye(3), (len(camera.RT), 1, 1))
    rotation[camera.camera_valid] = TO_BLENDER @ camera.RT[camera.camera_valid, :3, :3].transpose(0, 2, 1) @ np.diag([1., -1., -1.])
    parent = rotation @ Rotation.from_euler('x', -90, degrees=True).as_matrix()
    euler_yxz = Rotation.from_matrix(parent).as_euler('yxz')
    # Unwrap within each shot only; never smooth a cut.
    for a, b in intervals(camera.camera_valid):
        cuts = np.r_[a, np.flatnonzero(np.diff(camera.shot_ids[a:b])) + a + 1, b]
        for first, last in zip(cuts[:-1], cuts[1:]):
            euler_yxz[first:last] = np.unwrap(euler_yxz[first:last], axis=0)
    rotations = euler_yxz[:, [1, 2, 0]]
    points = targets * [1, 1, -1] / METRES_PER_MMD_UNIT
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as stream:
        stream.write(struct.pack('<30s20sIII', b'Vocaloid Motion Data 0002',
                                 'カメラ・照明'.encode('shift_jis'), 0, 0, int(camera.camera_valid.sum())))
        for frame in np.flatnonzero(camera.camera_valid):
            stream.write(KEY.pack(int(frame + camera.track_start), float(-distance[frame] / METRES_PER_MMD_UNIT),
                                  *points[frame], *rotations[frame], LINEAR, int(angles[frame]), 0))
        stream.write(struct.pack('<III', 0, 0, 0))
    return report
