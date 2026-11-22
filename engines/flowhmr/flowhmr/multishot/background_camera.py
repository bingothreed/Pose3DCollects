"""Register a short cutaway to a neighboring camera using its static scenery."""
from dataclasses import replace
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .timeline import intervals


def complete_cutaway_cameras(video, camera, raw, boxes, valid, textured, check, *, ungated=None):
    """Fill only short gaps supported by the same 3D background across the gap."""
    missing = valid & textured & ~camera.camera_valid
    if not missing.any():
        return camera, []
    ranges = []
    for a,b in intervals(missing):
        edges = [a, *(a+1+np.flatnonzero(np.diff(camera.shot_ids[a:b]))), b]
        ranges.extend((a,b) for a,b in zip(edges,edges[1:]) if b-a <= 30)
    if not ranges:
        return camera, []

    rt, covered = camera.RT.copy(), camera.camera_valid.copy()
    detector = cv2.SIFT_create(nfeatures=1400, contrastThreshold=.018)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    cap = cv2.VideoCapture(str(video))
    report = []
    try:
        for a,b in ranges:
            check()
            references, dense_references = [], []
            for edge, direction in ((a-1,-1),(b,1)):
                if not 0 <= edge < len(rt) or not camera.camera_valid[edge]:
                    continue
                nearby = edge + np.arange(min(45, edge+1 if direction<0 else len(rt)-edge))*direction
                nearby = nearby[camera.camera_valid[nearby] & (camera.shot_ids[nearby] == camera.shot_ids[edge])]
                if len(nearby) >= 8:
                    references.append(np.sort(nearby[np.linspace(0,len(nearby)-1,5,dtype=int)]))
                    dense_references.append(np.sort(nearby[np.linspace(0,len(nearby)-1,min(12,len(nearby)),dtype=int)]))
            if not references:
                continue
            texture, images = {}, {}
            targets = np.arange(a,b)
            for frame in np.unique(np.concatenate([targets, *references])):
                check()
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(camera.track_start+frame))
                ok, image = cap.read()
                if not ok:
                    continue
                images[frame] = image
                texture[frame] = _features(image, boxes[frame] if camera.camera_valid[frame] else None,
                                           camera.K[frame], detector)
            if not all(f in images for f in targets):
                continue
            best = None
            for group in references if all(texture[f] is not None for f in targets) else ():
                for first,second in combinations(group,2):
                    check()
                    if texture.get(first) is None or texture.get(second) is None:
                        continue
                    points, descriptors = _landmarks(texture[first], texture[second], rt[first], rt[second], matcher)
                    if len(points) < 24:
                        continue
                    estimate = _fit_cutaway(targets, texture, points, descriptors, raw[a:b], matcher, check)
                    if estimate is None:
                        continue
                    poses, min_points, error = estimate
                    if best is None or min_points > best[0]:
                        best = min_points, poses, error, first, second
            if best is not None:
                count, poses, error, first, second = best
                diagnostics = dict(method='background_feature_registration',
                                   reference_source_frames=[int(first+camera.track_start),int(second+camera.track_start)],
                                   minimum_background_inliers=count, median_reprojection_px=error)
            else:
                from .defocused_camera import register_defocused_cutaway
                for frame in np.unique(np.concatenate(dense_references)):
                    if frame not in images:
                        check()
                        cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame+camera.track_start))
                        ok, image = cap.read()
                        if ok:
                            images[frame] = image
                estimate = register_defocused_cutaway(images, targets, dense_references, camera, boxes, raw[a:b], check)
                if estimate is None and ungated is not None:
                    # Align to the measured neighboring path when the viewer's
                    # person-relative smoothing changed its stereo baseline.
                    estimate = register_defocused_cutaway(images, targets, dense_references, ungated, boxes, raw[a:b], check)
                    if estimate is not None:
                        estimate[1]['reference_camera'] = 'before_temporal_gate'
                if estimate is None:
                    continue
                poses, diagnostics = estimate
            rt[a:b] = poses
            covered[a:b] = True
            report.append(dict(shot=int(camera.shot_ids[a]),
                               source_frames=[int(a+camera.track_start),int(b+camera.track_start)],
                               **diagnostics))
    finally:
        cap.release()
    return replace(camera, RT=rt, camera_valid=covered), report


_BACKGROUND_BLOCK = 8
_BACKGROUND_EDGE_LIMIT = 240


def _camera_centers(poses):
    return -np.einsum('...ji,...j->...i', poses[..., :3, :3], poses[..., :3, 3])


def _same_shot_step(poses):
    rotation = poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1)
    angle = np.max(Rotation.from_matrix(rotation).magnitude())
    travel = np.max(np.linalg.norm(np.diff(_camera_centers(poses), axis=0), axis=1))
    return angle <= np.deg2rad(6) and travel <= .30


def continue_background_run(video, camera, raw, boxes, valid, textured, runs, check):
    """Register adjacent frames to a body-verified interval within the same shot.

    A block must share triangulated background with an accepted reference and
    reproject landmarks unused by its pose fit. Stop at the first failed block.
    """
    if not runs or boxes is None:
        return camera, []
    eligible = valid & textured
    covered = camera.camera_valid.copy()
    poses = camera.RT.copy()
    detector = cv2.SIFT_create(nfeatures=1400, contrastThreshold=.018)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    capture = cv2.VideoCapture(str(video))

    @lru_cache(maxsize=52)
    def sample(frame):
        check()
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(camera.track_start + frame))
        ok, image = capture.read()
        if not ok or not np.isfinite(boxes[frame]).all():
            return None
        return _features(image, boxes[frame], camera.K[frame], detector)

    reports = []
    try:
        for a, b in runs:
            shot = camera.shot_ids[a]
            for edge, direction in ((a, -1), (b - 1, 1)):
                frames = edge + direction * np.arange(1, _BACKGROUND_EDGE_LIMIT + 1)
                frames = frames[(frames >= 0) & (frames < len(covered))]
                allowed = eligible[frames] & ~covered[frames] & (camera.shot_ids[frames] == shot)
                frames = frames[:np.flatnonzero(~allowed)[0]] if not allowed.all() else frames
                for start in range(0, len(frames), _BACKGROUND_BLOCK):
                    check()
                    block = frames[start:start + _BACKGROUND_BLOCK]
                    seam = int(block[0] - direction)
                    tests = np.sort(np.r_[seam, block])
                    texture = {int(f): sample(int(f)) for f in tests}
                    if any(row is None for row in texture.values()):
                        break
                    candidates = seam - direction * np.array([0, 4, 8, 14, 23, 33])
                    references = [int(f) for f in candidates if 0 <= f < len(covered)
                                  and covered[f] and camera.shot_ids[f] == shot and sample(int(f)) is not None]
                    estimate = _continue_block(tests, seam, references, texture, sample, poses, raw, matcher, check)
                    if estimate is None:
                        break
                    fitted, count, error = estimate
                    within = tests != seam
                    poses[tests[within]] = fitted[within]
                    covered[block] = True
                    reports.append(dict(shot=int(shot), method='same_shot_background_registration',
                        source_frames=[int(block.min()) + camera.track_start, int(block.max()) + camera.track_start + 1],
                        minimum_background_inliers=count, median_reprojection_px=error))
    finally:
        capture.release()
    return replace(camera, RT=poses, camera_valid=covered), reports


def _continue_block(tests, seam, references, texture, sample, poses, raw, matcher, check):
    best = None
    for first, second in combinations(references, 2):
        check()
        world, descriptors = _landmarks(sample(first), sample(second), poses[first], poses[second], matcher)
        if len(world) < 32:
            continue
        estimate = _fit_cutaway(tests, texture, world, descriptors, raw[tests], matcher, check, hold_out=True)
        if estimate is None:
            continue
        fitted, count, error = estimate
        seam_index = int(np.searchsorted(tests, seam))
        anchor = fitted[seam_index]
        rotation = anchor[:3, :3] @ poses[seam, :3, :3].T
        angle = Rotation.from_matrix(rotation).magnitude()
        travel = np.linalg.norm(_camera_centers(anchor) - _camera_centers(poses[seam]))
        if angle > np.deg2rad(1.5) or travel > .08:
            continue
        fitted[seam_index] = poses[seam]
        if _same_shot_step(fitted) and (best is None or (count, -error) > (best[1], -best[2])):
            best = fitted, count, error
        if best is not None and best[1] >= 100 and best[2] <= 1.5:
            break
    return best


def _background_view(image, box, K):
    h, w = image.shape[:2]
    scale = min(1., 768 / w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale < 1.:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    mask = None
    if box is not None:
        x1, y1, x2, y2 = np.rint(box * scale).astype(int)
        mask = np.full_like(gray, 255)
        mask[max(0, y1-8):y2+8, max(0, x1-8):x2+8] = 0
    scaled_K = K.astype(float).copy()
    scaled_K[0] *= gray.shape[1] / w
    scaled_K[1] *= gray.shape[0] / h
    return gray, mask, scaled_K, scale


def _describe_background(background, detector):
    gray, mask, scaled_K, scale = background
    points, descriptors = detector.detectAndCompute(gray, mask)
    if descriptors is None or len(descriptors) < 24:
        return None
    return np.asarray([p.pt for p in points], dtype=float), descriptors, scaled_K, gray.shape, scale


def _features(image, box, K, detector):
    return _describe_background(_background_view(image, box, K), detector)


def _matches(first, second, matcher):
    pairs = []
    for candidates in matcher.knnMatch(first,second,k=2):
        if len(candidates)==2 and candidates[0].distance < .77*candidates[1].distance:
            pairs.append(candidates[0])
    unique = {}
    for match in sorted(pairs,key=lambda m:m.distance):
        unique.setdefault(match.trainIdx,match.queryIdx)
    return np.array(list(unique.values()),int), np.array(list(unique),int)


def _project(points, matrix):
    p = points @ matrix[:,:3].T + matrix[:,3]
    return p[:,:2] / p[:,2:], p[:,2]


def _landmarks(first, second, A, B, matcher):
    x, dx, Kx, *_ = first
    y, dy, Ky, *_ = second
    i,j = _matches(dx,dy,matcher)
    if len(i)<24:
        return np.empty((0,3)), None
    if np.linalg.norm(A[:3,:3].T@A[:3,3] - B[:3,:3].T@B[:3,3]) < .004:
        return np.empty((0,3)), None
    world, keep = _triangulate(x[i], y[j], Kx @ A[:3], Ky @ B[:3])
    return world[keep], dx[i[keep]]


def _triangulate(x, y, P, Q):
    homogeneous = cv2.triangulatePoints(P, Q, x.T, y.T).T
    finite = np.abs(homogeneous[:, 3]) > 1e-8
    world = np.zeros((len(x), 3))
    world[finite] = homogeneous[finite, :3] / homogeneous[finite, 3:]
    with np.errstate(divide='ignore', invalid='ignore'):
        u, z = _project(world, P)
        v, w = _project(world, Q)
    keep = finite & (z > .05) & (w > .05) & (np.linalg.norm(u-x, axis=1) < 1.2) & (np.linalg.norm(v-y, axis=1) < 1.2)
    return world, keep


def _fit_cutaway(targets, texture, points, descriptors, raw, matcher, check, *, hold_out=False):
    estimates, inlier_counts, errors = [], [], []
    for frame in targets:
        check()
        pixels, desc, K, shape, scale = texture[frame]
        i,j = _matches(descriptors,desc,matcher)
        if len(i) < (32 if hold_out else 24):
            return None
        fit = _fit_background_pose(points[i], pixels[j], K, shape, hold_out=hold_out)
        if fit is None:
            return None
        pose, count, residuals = fit
        estimates.append(pose)
        inlier_counts.append(count)
        errors.extend((residuals / scale).tolist())
    estimates = np.asarray(estimates)
    # The scenery solution must agree with the independently measured relative
    # camera rotations. This rejects unstable or repeated-texture matches.
    calibration = Rotation.from_matrix(raw[:,:3,:3].transpose(0,2,1) @ estimates[:,:3,:3])
    error = (calibration.mean().inv()*calibration).magnitude()
    if np.max(error) > np.deg2rad(6):
        return None
    return estimates, min(inlier_counts), float(np.median(errors))


def _fit_background_pose(obj, target, K, shape, *, hold_out):
    if hold_out:
        reserved = np.arange(len(obj)) % 4 == 0
        test_points, test_pixels = obj[reserved], target[reserved]
        obj, target = obj[~reserved], target[~reserved]
    ready, rotation, translation, inliers = cv2.solvePnPRansac(
        obj,target,K,None,iterationsCount=350,reprojectionError=2.5,
        confidence=.999,flags=cv2.SOLVEPNP_EPNP,
    )
    if not ready or len(inliers)<20 or len(inliers)<.45*len(obj):
        return None
    keep = inliers[:,0]
    if np.prod(np.ptp(target[keep],axis=0)) < .12*np.prod(shape):
        return None
    rotation, translation = cv2.solvePnPRefineLM(obj[keep],target[keep],K,None,rotation,translation)
    pose = np.eye(4)
    pose[:3,:3] = cv2.Rodrigues(rotation)[0]
    pose[:3,3] = translation[:,0]
    projected, depth = _project(obj[keep],K@pose[:3])
    if np.any(depth <= .05):
        return None
    if hold_out:
        prediction, distance = _project(test_points, K @ pose[:3])
        error = np.linalg.norm(prediction - test_pixels, axis=1)
        if ((distance > .05) & (error < 3.5)).mean() < .65:
            return None
    return pose, len(keep), np.linalg.norm(projected - target[keep], axis=1)
