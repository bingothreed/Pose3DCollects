"""Try a measured moving-camera path when fixed-camera recovery disagrees with the image."""
from dataclasses import replace

import numpy as np

from .alignment import MH, SM, body_facing_rotation, fit_shot
from .camera import project
from .gravity import MAX_DISPERSION_DEG
from .similarity import transform_camera
from .timeline import intervals

_TORSO = np.isin(SM, [16, 17, 1, 2])


def body_evidence(points, K, eligible, hw):
    """Reuse source pixels even when a camera proposal changes focal length."""
    target, depth = project(np.broadcast_to(np.eye(4), (len(K), 4, 4)), K, points[:, MH])
    visible = ((target >= 0).all(2) & (target < np.asarray(hw)[::-1]).all(2)
               & (depth > .05) & eligible[:, None])
    low = np.where(visible[..., None], target, np.inf).min(1)
    high = np.where(visible[..., None], target, -np.inf).max(1)
    span = np.where(visible.any(1), np.linalg.norm(high-low, axis=1), 0.)
    return target, visible, np.maximum(span, .10 * min(hw))


def _errors(rt, K, joints, target, visible):
    uv, depth = project(rt, K, joints[:, SM])
    return np.where(visible, np.where(depth > .05, np.linalg.norm(uv-target, axis=2), 1e6), np.nan)


def _summary(errors, span, frames):
    pixels = errors[frames]
    normalized = pixels / span[frames, None]
    return dict(median_px=float(np.nanmedian(pixels)), p90_px=float(np.nanpercentile(pixels, 90)),
                median_fraction=float(np.nanmedian(normalized)), p90_fraction=float(np.nanpercentile(normalized, 90)))


def _background_shift(textures, matcher, minimum_shift):
    import cv2
    from .background_camera import _matches

    best = 0.
    for stride in (1, 2):
        for left, right in zip(textures, textures[stride:]):
            if left is None or right is None:
                continue
            indices, other = _matches(left[1], right[1], matcher)
            if len(indices) < 24:
                continue
            x, y = left[0][indices], right[0][other]
            _, support = cv2.findHomography(x, y, cv2.RANSAC, 3.)
            if support is None:
                continue
            use = support.ravel().astype(bool)
            if use.sum() < 24 or np.min(np.ptp(x[use], axis=0) / left[3][::-1]) < .20:
                continue
            best = max(best, float(np.median(np.linalg.norm(y[use]-x[use], axis=1)) / left[-1]))
            if best >= minimum_shift:
                return best
    return best


def _background_validation(textures, sampled, rt, K, matcher):
    """Triangulate static tracks on two images and predict a third, held-out image."""
    from .background_camera import _landmarks, _matches, _project

    calibrated = [None if feature is None else
                  (feature[0], feature[1], np.diag([feature[-1], feature[-1], 1.]) @ k, *feature[3:])
                  for feature, k in zip(textures, K)]
    errors, ranges = [], []
    for start in range(len(calibrated) - 4):
        a, mid, b = start, start+2, start+4
        x, test, y = calibrated[a], calibrated[mid], calibrated[b]
        if x is None or test is None or y is None:
            continue
        world, desc = _landmarks(x, y, rt[a], rt[b], matcher)
        if len(world) < 18:
            continue
        indices, other = _matches(desc, test[1], matcher)
        pred, depth = _project(world[indices], test[2] @ rt[mid, :3])
        error = np.linalg.norm(pred-test[0][other], axis=1) / test[-1]
        good = (depth > .05) & np.isfinite(error)
        if good.sum() >= 12:
            errors.append(error[good])
            if np.percentile(error[good], 75) <= 8.:
                ranges.append([int(sampled[a]), int(sampled[b])])
    if len(errors) < 2:
        return None
    residual = np.concatenate(errors)
    median, p75 = np.percentile(residual, (50, 75))
    if median > 5. or p75 > 12.:
        return None
    return dict(heldout_matches=len(residual), heldout_views=len(errors), verified_step_ranges=ranges,
                median_px=float(median), p75_px=float(p75))


def _verify_steps(rt, K, background):
    from scipy.spatial.transform import Rotation
    from .background_camera import _camera_centers

    travel = np.linalg.norm(np.diff(_camera_centers(rt), axis=0), axis=1)
    turn = Rotation.from_matrix(rt[1:, :3, :3] @ rt[:-1, :3, :3].transpose(0,2,1)).magnitude()
    zoom = abs(np.diff(np.log(K[:, 0, 0])))
    unusual = (travel > .30) | (turn > np.deg2rad(6)) | (zoom > .10)
    # A fast move can be real. Require local held-out background support
    # instead of rejecting a measured dolly just because its speed is high.
    for a,b in background['verified_step_ranges']:
        unusual[a:b] = False
    return not unusual.any()


def _fit_path(rt, K, old_K, joints, points, visible, up, check, hw):
    # Each SAM keypoint was generated with old_K. Re-express its ray using the
    # proposed lens so the target stays on the exact same original pixel.
    rays = np.linalg.solve(K, old_K)
    observed = np.einsum('tij,tkj->tki', rays, points[:, MH])
    training = np.flatnonzero(visible[:, _TORSO].sum(1) >= 3)[::2]
    if len(training) < 8:
        return None
    motion = joints[:, SM]
    facing = body_facing_rotation(rt[training], observed[training], motion[training])
    fit = fit_shot(rt[training], K[training], observed[training], motion[training], up, facing, 1.,
                   image_size_hw=hw, fit_joints=_TORSO, ground_min=0., check=check)
    return transform_camera(rt, fit['world_rotation'], fit['world_scale'], fit['world_translation']), training


def _quality_mask(errors, span, holdout, previous, missing):
    # For a shot with no old fit, only publish coherent intervals that have
    # enough image evidence. A few bad frames cannot authorize a whole motion.
    if previous.any():
        common = holdout & previous
        if common.sum() < 8:
            return None
        before = _summary(missing, span, common)
        after = _summary(errors, span, common)
        if (after['p90_fraction'] >= .8 * before['p90_fraction']
                or after['median_fraction'] > 1.15 * before['median_fraction'] + .008
                or after['p90_fraction'] > .20):
            return None
        return previous.copy(), before, after

    normalized = np.ma.median(np.ma.masked_invalid(errors / span[:, None]), axis=1).filled(np.inf)
    supported = normalized <= .15
    mask = np.zeros(len(previous), dtype=bool)
    for a,b in intervals(supported):
        if b-a >= 15 and np.count_nonzero(holdout[a:b]) >= 7:
            mask[a:b] = True
    if not mask.any():
        return None
    after = _summary(errors, span, mask & holdout)
    if after['p90_fraction'] > .24:
        return None
    return mask, None, after


def refine_fixed_shots(video, camera, report, raw_rt, K, joints, points, boxes, shots,
                       eligible, valid, *, estimate, measure_gravity, check, notify):
    """Replace a fixed shot only with a track supported by background and body evidence."""
    import cv2
    from .background_camera import _features

    if boxes is None:
        return camera
    hw = np.asarray(camera.image_size_hw)
    target, visible, span = body_evidence(points, K, eligible, hw)
    original_errors = _errors(camera.RT, camera.K, joints, target, visible)
    detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=.018)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    cap = cv2.VideoCapture(str(video))
    candidates = {}
    proposal_rt, proposal_K = raw_rt.copy(), K.copy()
    try:
        for shot in shots:
            check()
            a, b = shot['start'], shot['end']
            sl = slice(a,b)
            seen = visible[sl].sum(1) >= 4
            if (visible[sl][:, _TORSO].sum(1) >= 3).sum() < 16:
                continue
            compared = seen & camera.camera_valid[sl]
            if compared.sum() >= .9 * seen.sum():
                if _summary(original_errors[sl], span[sl], compared)['p90_fraction'] < .12:
                    continue
            sampled = np.unique(np.linspace(a,b-1,min(15,b-a)).round().astype(int))
            pictures = []
            for frame in sampled:
                check()
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(camera.track_start+frame))
                ok, image = cap.read()
                box = boxes[frame]
                pictures.append(_features(image, box, K[frame], detector)
                                if ok and np.isfinite(box).all() else None)
            threshold = .010 * np.hypot(*hw)
            displacement = _background_shift(pictures, matcher, threshold)
            if displacement < threshold:
                continue
            try:
                notify(f'反求镜头 {shot["id"]+1} 的推拉与转向', 96)
                rt, lens = estimate(a,b)
                background = _background_validation(pictures, sampled-a, rt[sampled-a], lens[sampled-a], matcher)
            except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError, cv2.error) as exc:
                print(f'  [camera] moving proposal skipped: {type(exc).__name__}: {exc}', flush=True)
                continue
            if background is not None:
                proposal_rt[sl], proposal_K[sl] = rt, lens
                candidates[shot['id']] = dict(background, matched_displacement_px=displacement)
    finally:
        cap.release()
    if not candidates:
        return camera

    gravity = measure_gravity(proposal_rt, proposal_K, [shot for shot in shots if shot['id'] in candidates])
    final_rt, final_K, coverage = camera.RT.copy(), camera.K.copy(), camera.camera_valid.copy()
    accepted = []
    for shot in shots:
        check()
        sid = shot['id']
        if sid not in candidates:
            continue
        row = gravity.get(sid, {})
        up = np.array(row.get('mean_world_up', [0., 0., 0.]))
        if (up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < .5
                or row.get('dispersion_max_deg', np.inf) > MAX_DISPERSION_DEG):
            continue
        a, b = shot['start'], shot['end']
        sl = slice(a,b)
        lens, rt = proposal_K[sl], proposal_rt[sl]
        try:
            fitted = _fit_path(rt, lens, K[sl], joints[sl], points[sl], visible[sl], up, check, hw)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if fitted is None:
            continue
        aligned, training = fitted
        if not _verify_steps(aligned, lens, candidates[sid]):
            continue
        error = _errors(aligned, lens, joints[sl], target[sl], visible[sl])
        holdout = visible[sl].sum(1) >= 4
        holdout[training] = False
        decision = _quality_mask(error, span[sl], holdout, coverage[sl], original_errors[sl])
        if decision is None:
            continue
        mask, before, after = decision
        # A single missing detection in an otherwise verified run can follow
        # the measured background path. Keep genuine motion gaps independent.
        for lo,hi in intervals(~mask & ~eligible[sl]):
            if (0 < lo < hi < len(mask) and hi-lo <= 3 and valid[a+lo:a+hi].all()
                    and mask[lo-1] and mask[hi]):
                mask[lo:hi] = True
        final_rt[sl], final_K[sl], coverage[sl] = aligned, lens, mask
        details = dict(shot=sid, source_frames=[a,b], labels=['background_validated_motion_camera'],
                       gravity_source='image', gravity_method=row.get('method'),
                       gravity_dispersion_deg=row['dispersion_max_deg'], holdout_before=before,
                       holdout_after=after, background=candidates[sid],
                       valid_ranges=[[a+lo,a+hi] for lo,hi in intervals(mask)])
        for previous in report['shots']:
            if previous['shot'] == sid and previous.get('gravity_source'):
                previous['superseded'] = True
        report['shots'].append(details)
        accepted.append(details)
    if not accepted:
        return camera
    report['camera_path_refinement'] = dict(method='shot_trajectory_with_background_and_body_validation',
                                            accepted_shots=[row['shot'] for row in accepted])
    report['shots'].sort(key=lambda row: row['source_frames'][0])
    return replace(camera, RT=final_rt, K=final_K, camera_valid=coverage)
