"""One fixed similarity aligns recovered cameras to the final capture, never its pose."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from flowhmr.utils.runtime_group_scene import _camera_scale, _directions, _rotation

VERSION = 1


def project(points, rt, intrinsics):
    camera = np.einsum('tij,tkj->tki', rt[:, :3, :3], points) + rt[:, None, :3, 3]
    homogeneous = np.einsum('tij,tkj->tki', intrinsics, camera)
    return homogeneous[..., :2] / np.maximum(camera[..., 2:3], 1e-5), camera[..., 2]


def transform_camera(rt, rotation, scale, translation):
    """OpenCV world-to-camera extrinsics in the final motion's Y-up world, metres."""
    old_rotation = rt[:, :3, :3]
    centers = -np.einsum('tji,tj->ti', old_rotation, rt[:, :3, 3])
    centers = scale * (centers @ rotation.T) + translation
    new_rotation = old_rotation @ rotation.T
    result = np.tile(np.eye(4), (len(rt), 1, 1))
    result[:, :3, :3] = new_rotation
    result[:, :3, 3] = -np.einsum('tij,tj->ti', new_rotation, centers)
    return result


def validate_camera(rt, intrinsics):
    rt, intrinsics = np.asarray(rt, dtype=float), np.asarray(intrinsics, dtype=float)
    if (rt.ndim != 3 or rt.shape[1:] != (4, 4) or intrinsics.shape != (len(rt), 3, 3)
            or not len(rt) or not np.isfinite(rt).all() or not np.isfinite(intrinsics).all()):
        raise ValueError('摄像机数据的帧数、矩阵或数值无效。')
    rotation = rt[:, :3, :3]
    if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-4)
            or not np.allclose(rt[:, 3], [0, 0, 0, 1])
            or np.any(intrinsics[:, [0, 1], [0, 1]] <= 0)
            or not np.allclose(intrinsics[:, 2], [0, 0, 1])):
        raise ValueError('摄像机旋转或镜头参数无效。')
    return rt, intrinsics


def align_camera(rt, intrinsics, frames, observed, motion, *, check=lambda: None):
    """Fit seven global parameters from matching frames/joints, with robust pixel residuals.

    SAM supplies camera-space observations. Its 3D directions initialize the existing
    Procrustes fit; 2D refinement reduces sensitivity to SAM depth/shape estimates.
    No per-frame camera corrections, focal changes, or motion edits are permitted.
    """
    rt, intrinsics = validate_camera(rt, intrinsics)
    raw_frames = np.asarray(frames)
    if not np.issubdtype(raw_frames.dtype, np.integer):
        raise ValueError('人体观测必须使用整数帧索引。')
    frames = raw_frames.astype(int)
    observed, motion = np.asarray(observed, float), np.asarray(motion, float)
    if (frames.ndim != 1 or len(frames) < 3 or len(np.unique(frames)) != len(frames)
            or np.any(frames < 0) or np.any(frames >= len(rt))
            or observed.shape != (len(frames), 8, 3) or motion.shape != observed.shape
            or not np.isfinite(observed).all() or not np.isfinite(motion).all()
            or np.any(observed[..., 2] <= .05)):
        raise ValueError('缺少有效、对应同一时间的人体观测，无法对齐摄像机。')
    sample_rt = rt[frames]
    rotation = sample_rt[:, :3, :3]
    world = np.einsum('tkj,tji->tki', observed, rotation)
    a, weights = _directions(world)
    b, _ = _directions(motion)
    shared = _rotation(a, b, weights)
    centers = -np.einsum('tji,tj->ti', rotation, sample_rt[:, :3, 3]) @ shared.T
    world = world @ shared.T
    delta = (motion[:, 2:4] - world[:, 2:4]).mean(axis=1)
    moving = float(np.ptp(centers, axis=0).max()) > 1e-6
    scale = max(_camera_scale([(centers, delta)]), 1e-4) if moving else 1.
    translation = np.median(delta - scale * centers, axis=0)
    target, _ = project(observed, np.tile(np.eye(4), (len(frames), 1, 1)), intrinsics[frames])
    initial = np.r_[Rotation.from_matrix(shared).as_rotvec(), translation, np.log(scale)]

    def unpack(parameters):
        return Rotation.from_rotvec(parameters[:3]).as_matrix(), np.exp(parameters[6]) if moving else 1., parameters[3:6]

    def residual(parameters):
        check()
        adjusted = transform_camera(sample_rt, *unpack(parameters))
        uv, depth = project(motion, adjusted, intrinsics[frames])
        return np.r_[(uv - target).ravel(), (np.minimum(depth - .05, 0.) * 1000).ravel()]

    fit = least_squares(residual, initial, loss='soft_l1', f_scale=8., max_nfev=200,
                        bounds=(np.r_[np.full(6, -np.inf), -12.], np.r_[np.full(6, np.inf), 12.]))
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('摄像机与人物的整体对齐未收敛，请检查素材。')
    shared, scale, translation = unpack(fit.x)
    aligned = transform_camera(rt, shared, scale, translation)
    uv, depth = project(motion, aligned[frames], intrinsics[frames])
    if np.min(depth) <= .05:
        raise ValueError('整体对齐后人物位于摄像机后方，无法导出有效摄像机。')
    errors = np.linalg.norm(uv - target, axis=-1)
    centers = -np.einsum('tji,tj->ti', aligned[:, :3, :3], aligned[:, :3, 3])
    return dict(RT=aligned, K=intrinsics, centers=centers, world_rotation=shared,
                world_scale=np.float64(scale), world_translation=translation,
                sample_frames=frames, observed_uv=target, projected_uv=uv,
                report=dict(version=VERSION, sample_frames=len(frames), observed_joints=8,
                            median_px=float(np.median(errors)), p95_px=float(np.quantile(errors, .95)),
                            max_px=float(errors.max()), scale=float(scale), moving=moving,
                            method='one_global_similarity', optimizer_evaluations=fit.nfev))
