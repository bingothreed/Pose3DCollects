"""Check continuity at a track handoff using only visible source landmarks."""
import cv2
import numpy as np

from .camera import project


# MHR shoulders, elbows, wrists, and hips. Indices below refer to this subset.
LANDMARKS = [5, 6, 7, 8, 62, 41, 9, 10]
ARMS = [(0, 2, 4), (1, 3, 5)]
LOOKAROUND = 12


def observed_arms(points, K, frames, split, hw):
    """Port the torso-frame comparison from PR #19 to each arm independently."""
    uv, z = project(np.broadcast_to(np.eye(4), (len(K), 4, 4)), K, points)
    visible = ((z > .05) & (uv >= 0).all(2) & (uv < np.asarray(hw)[::-1]).all(2))
    trusted = visible[:, [0, 1, 6, 7]].all(1)
    if not trusted[split-1:split+1].all():
        return ()
    shoulders = (points[:, 0]+points[:, 1])*.5
    hips = (points[:, 6]+points[:, 7])*.5
    spine = shoulders-hips
    up = spine / np.maximum(np.linalg.norm(spine, axis=1, keepdims=True), 1e-6)
    right = points[:, 1]-points[:, 0]
    right -= (right*up).sum(1, keepdims=True)*up
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
    basis = np.stack((right, up, np.cross(right, up)), axis=-1)
    trusted &= np.isfinite(basis).all((1, 2)) & (np.linalg.det(basis) > .99)
    local = np.einsum('tji,tkj->tki', basis, points-shoulders[:, None])

    confirmed = []
    for side, arm in enumerate(ARMS):
        available = trusted & visible[:, arm].all(1)
        delta = np.linalg.norm(np.diff(local[:, arm], axis=0), axis=-1).mean(1)
        supported = available[:-1] & available[1:]
        if not supported[split-1]:
            continue
        supported &= np.diff(frames) == 1
        supported[split-1] = False
        if supported.sum() >= 6 and delta[split-1] <= 3 * max(np.percentile(delta[supported], 99), 1e-6):
            confirmed.append(side)
    return tuple(confirmed)


def measure_cut_arms(engine, video, parts, camera_paths, cuts, cache, check, max_gap, *, shared_camera=None):
    if not cuts or not callable(getattr(engine.sam_extractor, 'scene_keypoints', None)):
        return {}
    requests, spans = {}, {}
    for left_index, (left, right) in enumerate(zip(parts, parts[1:])):
        last, first = left[1], right[0]
        boundaries = [c for c in cuts if last < c <= first]
        if len(boundaries) != 1 or first-last-1 > max_gap:
            continue
        cut = boundaries[0]
        left_edge = max(c for c in [left[0], *cuts] if c <= last)
        right_edge = min(c for c in [right[1]+1, *cuts] if c > first)
        outgoing = left[3][left[3] >= left_edge][-LOOKAROUND:]
        incoming = right[3][right[3] < right_edge][:LOOKAROUND]
        if min(len(outgoing), len(incoming)) < 4:
            continue
        spans[cut] = (outgoing, incoming)
        for i, part, indices in ((left_index, left, outgoing), (left_index+1, right, incoming)):
            if shared_camera is None:
                with np.load(camera_paths[i], allow_pickle=False) as camera:
                    intrinsic = camera['K']
            else:
                start, K = shared_camera
                intrinsic = K[part[0]-start:part[1]-start+1]
            for frame in indices:
                offset = frame-part[0]
                requests[int(frame)] = (part[4][offset], intrinsic[offset])
    if not requests:
        return {}

    from .runtime import save_npz

    frames = np.array(sorted(requests))
    boxes, K = map(np.stack, zip(*(requests[f] for f in frames)))
    points = None
    try:
        with np.load(cache, allow_pickle=False) as data:
            if all(np.array_equal(data[key], value) for key, value in (('frames',frames),('boxes',boxes),('K',K))):
                if data['points'].shape == (len(frames), len(LANDMARKS), 3):
                    points = data['points']
    except (OSError, ValueError, KeyError):
        pass
    cap = cv2.VideoCapture(str(video))
    hw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    try:
        if points is None:
            rows = []
            previous = -2
            with engine.scene_sam() as sam:
                for index, frame in enumerate(frames):
                    if check is not None:
                        check()
                    if frame != previous+1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f'无法读取接续原片第 {frame+1} 帧。')
                    rows.append(sam.scene_keypoints(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                                    boxes[index:index+1], K[index], LANDMARKS)[0])
                    previous = frame
            points = np.stack(rows)
            save_npz(cache, points=points, frames=frames, boxes=boxes, K=K)
    finally:
        cap.release()
    result = {}
    for cut, (outgoing, incoming) in spans.items():
        indices = np.searchsorted(frames, np.r_[outgoing, incoming])
        result[cut] = observed_arms(points[indices], K[indices], frames[indices], len(outgoing), hw)
    return result
