"""Recover a verified part of a long moving shot whose full alignment failed."""
import numpy as np

from .alignment import MH, SM, body_facing_rotation, fit_shot
from .camera import project
from .similarity import transform_camera
from .timeline import intervals


_TORSO = np.isin(SM, [1, 2, 16, 17])
_WINDOW = 90
_MIN_RUN = 45


def _visible_median(errors, visible):
    return np.ma.median(np.ma.array(errors, mask=~visible), axis=1).filled(np.inf)


def recover_supported_run(rt, K, joints, points, usable, image_size_hw, up, scale_prior, *, check, ground_min):
    """Publish one coherent raw-camera path confirmed by held-out joints and frames.

    Close-ups without enough shared torso and limb landmarks stay uncovered.
    One similarity applies to the entire verified run and never edits the motion.
    """
    if len(rt) < 2 * _WINDOW:
        return None
    # A stationary or purely rotating path has no reliable translation scale.
    centers = -np.einsum('tji,tj->ti', rt[:, :3, :3], rt[:, :3, 3])
    if np.linalg.norm(np.ptp(centers, axis=0)) < .015:
        return None
    motion, observed = joints[:, SM], points[:, MH]
    pixels, depth = project(np.broadcast_to(np.eye(4), rt.shape), K, observed)
    visible = (usable[:, None] & (depth > .05) & (pixels >= 0).all(2)
               & (pixels < np.asarray(image_size_hw)[::-1]).all(2))
    span = (np.where(visible[..., None], pixels, -np.inf).max(1)
            - np.where(visible[..., None], pixels, np.inf).min(1))
    size = np.maximum(np.linalg.norm(span, axis=1), 1.)
    good = ((visible[:, _TORSO].sum(1) >= 3) & (visible[:, ~_TORSO].sum(1) >= 3)
            & (size >= .15 * min(image_size_hw)))
    best = None
    for a, b in intervals(good):
        if b-a < _WINDOW:
            continue
        for start in np.unique(np.linspace(a, b-_WINDOW, min(7, 1+(b-a)//30), dtype=int)):
            check()
            training = np.arange(start, start+_WINDOW, 2)
            try:
                facing = body_facing_rotation(rt[training], observed[training], motion[training])
                result = fit_shot(rt[training], K[training], observed[training], motion[training],
                                  up, facing, scale_prior, image_size_hw=image_size_hw,
                                  check=check, ground_min=ground_min, fit_joints=_TORSO)
            except (ValueError, np.linalg.LinAlgError):
                continue
            path = transform_camera(rt, result['world_rotation'], result['world_scale'], result['world_translation'])
            projection, fitted_depth = project(path, K, motion)
            errors = np.linalg.norm(projection-pixels, axis=2) / size[:, None]
            torso = _visible_median(errors[:, _TORSO], visible[:, _TORSO])
            limbs = _visible_median(errors[:, ~_TORSO], visible[:, ~_TORSO])
            supported = good & (torso <= .12) & (limbs <= .16) & ((fitted_depth > .05) | ~visible).all(1)
            validation = np.arange(start+1, start+_WINDOW, 2)
            if supported[validation].mean() < .78:
                continue
            for lo, hi in intervals(supported):
                if hi-lo < _MIN_RUN or lo >= start+_WINDOW or hi <= start:
                    continue
                score = (hi-lo, -float(np.median(limbs[lo:hi])))
                if best is None or score > best[0]:
                    indices = np.arange(lo, hi)
                    held_out = validation[supported[validation]]
                    best = score, indices, path[indices], dict(
                        method='fixed_similarity_verified_on_held_out_body_evidence',
                        scale=result['world_scale'], training_frames=[int(start), int(start+_WINDOW)],
                        validation_frames=int(len(held_out)),
                        validation_limb_error_fraction=float(np.median(limbs[held_out])),
                        visible_median_px=float(np.median(np.linalg.norm(projection[indices]-pixels[indices], axis=2)[visible[indices]])),
                        raw_trajectory_preserved=True,
                    )
    if best is None:
        return None
    _, indices, path, report = best
    return indices, path, report
