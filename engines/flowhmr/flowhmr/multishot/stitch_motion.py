"""Generate selected track fragments independently and place them on the source timeline."""

import os
from pathlib import Path
import shutil
from zipfile import BadZipFile

import numpy as np
from scipy.spatial.transform import Rotation

from .timeline import held_indices, intervals, validity


# 0.13 seconds on either side of a handoff at the normalized source rate.
SEAM_FRAMES = 4
MAX_DROPOUT = 3


def fragment_inputs(data, first, last, width, height):
    """Recover consecutive runs of ownership without mixing across tracks."""
    from ..utils.runtime_video_processing import VideoProcessor

    if 'fragment_frames' not in data:
        return []
    order = np.argsort(data['fragment_frames'], kind='stable')
    frames, boxes, owners = (data[key][order] for key in ('fragment_frames', 'fragment_boxes', 'fragment_ids'))
    keep = (frames >= first) & (frames <= last)
    frames, boxes, owners = frames[keep], boxes[keep], owners[keep]
    if not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError('接续片段的检测帧重叠或为空。')
    edges = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1, len(frames)]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        observed = frames[lo:hi]
        start, end = int(observed[0]), int(observed[-1])
        track = np.column_stack([np.interp(np.arange(start,end+1), observed, coord)
                                 for coord in boxes[lo:hi].T]).astype(np.float32)
        parts.append((start, end, int(owners[lo]), observed, VideoProcessor.smooth_bboxes(track, width, height)))
    return parts


def _yaw(rotvec):
    matrix = Rotation.from_rotvec(rotvec).as_matrix()
    return np.arctan2(matrix[...,0,2] - matrix[...,2,0], matrix[...,0,0] + matrix[...,2,2])


def _articulation_continuous(left, right):
    # Ignore root heading and finger curl. The body rotations include the
    # wrists, so a single arm flip can veto an otherwise similar handoff.
    def step(a, b):
        a = Rotation.from_rotvec(a[:,1:22].reshape(-1,3))
        b = Rotation.from_rotvec(b[:,1:22].reshape(-1,3))
        return (a.inv() * b).magnitude().reshape(-1,21)

    steps = [step(side[:-1], side[1:]) for side in (left, right) if len(side) > 1]
    if not steps:
        return False
    baseline = np.concatenate(steps)
    limit = np.clip(3 * np.percentile(baseline, 90, axis=0) + np.deg2rad(8),
                    np.deg2rad(20), np.deg2rad(45))
    return bool((step(left[-1:], right[:1])[0] <= limit).all())


def dropout_references(video_path, parts, first, last, cuts, check):
    """A short missing boundary frame may use a verified image from its own shot."""
    import cv2

    edges = [(first, parts[0][0], None, parts[0])]
    edges.extend((left[1]+1, right[0], left, right) for left, right in zip(parts, parts[1:]))
    edges.append((parts[-1][1]+1, last+1, parts[-1], None))
    short = [(a,b,left,right) for a,b,left,right in edges if 0 < b-a <= MAX_DROPOUT]
    if not short:
        return {}
    refs = {}
    capture = cv2.VideoCapture(str(video_path))
    try:
        for a,b,left,right in short:
            if check is not None:
                check()
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, max(first,a-1)):
                continue
            lo, hi = max(first,a-1), min(last+1,b+1)
            images = {}
            for frame in range(lo,hi):
                ok, image = capture.read()
                if not ok:
                    break
                images[frame] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for endpoint, box in ((a-1, left[4][-1] if left else None),
                                  (b, right[4][0] if right else None)):
                if box is None or endpoint not in images:
                    continue
                x1,y1,x2,y2 = np.rint(box).astype(int)
                if min(x2-x1,y2-y1) < 20:
                    continue
                def crop(frame):
                    return cv2.resize(images[frame][y1:y2,x1:x2], (64,64),
                                      interpolation=cv2.INTER_AREA).astype(np.float32)
                anchor = crop(endpoint)
                shot = np.searchsorted(cuts, endpoint, side='right')
                for frame in range(a,b):
                    if frame not in images or np.searchsorted(cuts, frame, side='right') != shot:
                        continue
                    candidate = crop(frame)
                    contrast = min(anchor.std(), candidate.std())
                    if contrast >= 8 and np.abs(candidate-anchor).mean() <= .35 * contrast:
                        old = refs.get(frame)
                        if old is None or abs(endpoint-frame) < abs(old-frame):
                            refs[frame] = endpoint
    finally:
        capture.release()
    return refs


def _include_verified_edge_frames(parts, references):
    """Supply same-shot boxes for short, visually verified boundary misses."""
    expanded = []
    for start, end, track_id, observed, boxes in parts:
        a, b = start, end
        while references.get(a-1) == start:
            a -= 1
        while references.get(b+1) == end:
            b += 1
        if (a, b) != (start, end):
            boxes = np.pad(boxes, ((start-a, b-end), (0, 0)), mode='edge')
        expanded.append((a, b, track_id, observed, boxes))
    return expanded


def _continuous_generation_parts(parts, arms, frame_limit):
    """Run a common inference window through source-verified same-person cuts."""
    groups = [[parts[0]]]
    for part in parts[1:]:
        group = groups[-1]
        start, end = part[:2]
        if start == group[-1][1]+1 and arms.get(start) == (0, 1) and end-group[0][0]+1 <= frame_limit:
            group.append(part)
        else:
            groups.append([part])
    return [
        (group[0][0], group[-1][1], group[0][2],
         np.concatenate([part[3] for part in group]), np.concatenate([part[4] for part in group]))
        if len(group) > 1 else group[0] for group in groups
    ]


def _boxes_for_shot_timeline(start, boxes, observed, shot_ranges):
    """Interpolate missing detections from the same image shot and retain their provenance."""
    end = start + len(boxes)
    aligned = boxes.copy()
    pose_support = np.isin(np.arange(start, end), observed)
    for left, right in shot_ranges:
        a, b = max(start, left), min(end, right)
        anchors = observed[(observed >= a) & (observed < b)]
        if not len(anchors):
            continue
        timeline = np.arange(a, b)
        for coord in range(4):
            aligned[a-start:b-start, coord] = np.interp(timeline, anchors, boxes[anchors-start, coord])
        # Only short detector misses may contribute to the cut's pose estimate.
        supported = pose_support[a-start:b-start]
        for u, v in intervals(~supported):
            if v-u <= 2:
                supported[u:v] = True
    return aligned, pose_support


def _ease_arm_handoff(poses, left, stop, start, right, arms):
    """Fade a rotation offset into both edges while preserving each clip's motion."""
    from .seam_observations import ARMS

    changed = []
    for arm in arms:
        joints = np.array(ARMS[arm]) + 16
        before, after = poses[stop-1, joints], poses[start, joints]
        delta = (Rotation.from_rotvec(before).inv() * Rotation.from_rotvec(after)).as_rotvec()
        angle = np.linalg.norm(delta, axis=1)
        # Retain normal single-frame steps. Only soften the excess discontinuity.
        keep = np.deg2rad(12.)
        offset = .5 * (1. - keep / np.maximum(angle, keep))[:, None] * delta
        if not offset.any():
            continue
        for rows in (np.arange(left+1,stop), np.arange(right-1,start-1,-1)):
            weight = np.arange(1,len(rows)+1) / len(rows)
            weight = weight**2 * (3. - 2. * weight)
            move = -offset if rows[0] >= start else offset
            original = poses[np.ix_(rows, joints)]
            adjustment = Rotation.from_rotvec((weight[:,None,None]*move).reshape(-1,3))
            poses[np.ix_(rows, joints)] = (Rotation.from_rotvec(original.reshape(-1,3)) *
                                           adjustment).as_rotvec().reshape(original.shape)
        changed.append(arm)
    return changed


def stitch_fragments(parts, first, last, *, cuts=(), references=None, observed_arms=None):
    """Keep normal poses and limit seam corrections to source-supported motion."""
    reference = parts[0][2]
    length = last-first+1
    poses = np.empty((length,52,3), dtype=reference['poses'].dtype)
    trans = np.empty((length,3), dtype=reference['trans'].dtype)
    contacts = np.zeros((length,4), dtype=np.float32)
    valid = np.zeros(length, dtype=bool)
    source_ranges, seams, handoffs = [], [], []
    prev_start = prev_stop = None
    for start, end, motion in parts:
        a, b = start-first, end-first+1
        if not 0 <= a < b <= length or (prev_stop is not None and a < prev_stop):
            raise ValueError('接续片段的动作区间无效。')
        poses[a:b] = motion['poses']
        trans[a:b] = motion['trans']
        valid[a:b] = True
        if 'foot_contacts' in motion:
            contacts[a:b] = motion['foot_contacts']
        source_ranges.append([start,end+1])
        if prev_stop is not None:
            cut = next((frame for frame in cuts if first+prev_stop <= frame <= start), None)
            before = poses[max(prev_start,prev_stop-5):prev_stop]
            after = poses[a:min(b,a+5)]
            same_motion = (cut is None and a-prev_stop <= MAX_DROPOUT
                           and all(frame in (references or {}) for frame in range(first+prev_stop,start))
                           and _articulation_continuous(before, after))
            yaw_target = _yaw(poses[prev_stop-1,0])
            destination = trans[prev_stop-1].copy()
            if same_motion and len(before) > 1:
                frames = a-prev_stop+1
                yaw_target += frames * np.median(np.diff(np.unwrap(_yaw(before[:,0]))))
                destination += frames * np.median(np.diff(trans[prev_stop-len(before):prev_stop], axis=0), axis=0)
            yaw = yaw_target - _yaw(poses[a,0])
            rotation = Rotation.from_rotvec([0.,yaw,0.])
            poses[a:b,0] = (rotation * Rotation.from_rotvec(poses[a:b,0])).as_rotvec()
            shifted = rotation.apply(trans[a:b])
            # Each fragment has already solved its own floor height.
            shifted[:,[0,2]] += destination[[0,2]] - shifted[0,[0,2]]
            trans[a:b] = shifted

            left = prev_stop - 1 - min(SEAM_FRAMES, (prev_stop-prev_start-1)//2)
            right = a + min(SEAM_FRAMES, (b-a-1)//2)
            # Even a same-shot ownership change must not reach into a camera cut
            # that is only a few frames from the handoff.
            left = max([left, *(frame-first for frame in cuts if left < frame-first < prev_stop)])
            right = min([right, *(frame-first-1 for frame in cuts if a < frame-first <= right)])
            blended = bool(same_motion and right-left > 1)
            handoff = dict(source_start=start, cut=cut, blended=blended)
            if blended:
                weight = np.arange(1,right-left)/(right-left)
                start_rot = Rotation.from_rotvec(poses[left])
                delta = (start_rot.inv() * Rotation.from_rotvec(poses[right])).as_rotvec()
                interp = Rotation.from_rotvec((weight[:,None,None]*delta).reshape(-1,3)).as_matrix()
                poses[left+1:right] = Rotation.from_matrix(
                    (start_rot.as_matrix()[None] @ interp.reshape(-1,52,3,3)).reshape(-1,3,3)
                ).as_rotvec().reshape(-1,52,3)
                trans[left+1:right] = trans[left] + weight[:,None]*(trans[right]-trans[left])
                contacts[left+1:right] = 0.
                valid[left+1:right] = True
                seams.append([first+left+1,first+right])
            elif cut is not None and left < prev_stop-1 and right > a:
                changed = _ease_arm_handoff(poses, left, prev_stop, a, right, (observed_arms or {}).get(cut, ()))
                if changed:
                    handoff['arm_transition'] = ['left' if arm == 0 else 'right' for arm in changed]
                    seams.append([first+left+1,first+right])
            handoffs.append(handoff)
        prev_start, prev_stop = a, b

    for frame, endpoint in (references or {}).items():
        i, j = frame-first, endpoint-first
        if not valid[i] and valid[j]:
            poses[i], trans[i], valid[i] = poses[j], trans[j], True
    hold = held_indices(valid)
    poses, trans = poses[hold], trans[hold]
    result = dict(poses=poses, trans=trans, Rh=poses[:,0].copy(), betas=reference['betas'],
                  num_frames=np.int64(length), mocap_framerate=reference['mocap_framerate'],
                  gender=reference['gender'], body_reference=reference['body_reference'])
    if not valid.all():
        result['valid_frames'] = valid
    if all('foot_contacts' in motion for _, _, motion in parts):
        result['foot_contacts'] = contacts
    return result, dict(fragments=source_ranges, seams=seams, handoffs=handoffs,
                        visible_ranges=[[first+a,first+b] for a,b in intervals(valid)])


def generate_fragments(engine, video_path, working_video, sub_dir, name, parts, first, last, *,
                       options, prepared_dir, prepared_key, cuts, notify, shots=(), image_size_hw=None,
                       camera_progress=None):
    """Run the ordinary single-track generator for every stretch of the selected identity."""
    from .runtime import estimate_shot_cameras, file_revision_digest, save_json, save_npz
    from .seam_observations import measure_cut_arms
    from ..utils.runtime_capabilities import LONG_MOTION_EXPERIMENTAL_THRESHOLD, detection_request_key
    from ..utils.runtime_torso_optimization import save_composed_motion

    reports, origin_parts, final_parts, cameras = [], [], [], []
    seed = options['seed']
    has_final = engine.postprocess is not None or options['hamer']
    references = dropout_references(working_video, parts, first, last, cuts, options['cancel_check'])
    check = options['cancel_check'] or (lambda: None)
    share_camera = (not options['static_camera'] and len(shots) > 1
                    and getattr(engine.sam_extractor, 'estimator', None) is not None)
    arms = {}
    seam_cache = Path(sub_dir) / f'{name}_seam_observations.npz'
    if share_camera:
        if prepared_dir is None:
            parts = _include_verified_edge_frames(parts, references)
        folder = Path(sub_dir) / 'linked'
        video_hash = file_revision_digest(working_video)
        raw_folder = folder / f'raw-{video_hash[:16]}'
        raw_folder.mkdir(parents=True, exist_ok=True)
        camera_options = {'params': {'static_camera': False, 'vggt_interval': options['vggt_frame_interval']}}
        n = last-first+1
        camera_progress = camera_progress or (lambda message, progress: None)
        camera_progress('按镜头估计共用相机', 0)
        raw, K, _ = estimate_shot_cameras(
            raw_folder, camera_options, shots, first, n, image_size_hw, working_video, engine.device,
            check, lambda message, pct: camera_progress(message, max(0., (pct-5)*100/13 - 100/len(shots))),
            engine._release_camera_cuda,
        )
        camera_progress('动作与反求共用相机已就绪', 100)
        ranges = np.array([[first+shot['start'], first+shot['end']] for shot in shots], dtype=np.int64)
        observed_points = np.full((n, 70, 3), np.nan, dtype=np.float32)
        detected = np.zeros(n, dtype=bool)
        try:
            arms = measure_cut_arms(engine, working_video, parts, (), cuts, seam_cache,
                                    options['cancel_check'], MAX_DROPOUT, shared_camera=(first, K))
        except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
            print(f'  [stitch] skipped image-supported arm transition: {exc}', flush=True)
        if prepared_dir is None:
            frame_limit = LONG_MOTION_EXPERIMENTAL_THRESHOLD
            if engine._staged_models:
                # Preserve the longest inference window in staged low-memory mode.
                frame_limit = min(frame_limit, max(end-start+1 for start,end,*_ in parts))
            parts = _continuous_generation_parts(parts, arms, frame_limit)
    for index, (start, end, track_id, observed, boxes) in enumerate(parts, 1):
        if share_camera and prepared_dir is None:
            boxes, pose_support = _boxes_for_shot_timeline(start, boxes, observed, ranges)
        else:
            pose_support = np.isin(np.arange(start, end+1), observed) if share_camera else None
        notify(index, len(parts))
        work = Path(sub_dir) / 'fragments' / f'{index:03d}-{start}-{end}'
        inputs = work / name
        inputs.mkdir(parents=True, exist_ok=True)
        source = inputs / f'{name}_30fps.mp4'
        if not source.is_file():
            try:
                os.link(working_video, source)
            except OSError:
                shutil.copy2(working_video, source)
        detection = inputs / f'{name}_bbox.npz'
        extra = {} if prepared_key is None else {'sam2_input_key': np.asarray(prepared_key)}
        save_npz(detection, bbox=boxes, start_end=np.array([start,end]), observed_frames=observed,
                 detection_request=detection_request_key((start,end)), track_id=np.int64(track_id), **extra)
        shared = None
        if share_camera:
            section = slice(start-first, end-first+1)
            shared = dict(frame_range=(start, end), RT=raw[section], K=K[section],
                          shot_ranges=ranges, pose_support=pose_support)
        engine.process_video(video_path, str(work), start_frame=start, end_frame=end,
                             prepared_inputs=prepared_dir is not None,
                             prepared_inputs_dir=prepared_dir, shared_camera=shared, **options)
        mode = 'static' if options['static_camera'] else 'vggt'
        cameras.append(inputs / f'{name}_{mode}_camera.npz')
        if share_camera:
            with np.load(inputs / f'{name}_sam3d_feat.pt.observations.npz', allow_pickle=False) as data:
                if (not np.array_equal(data['source_frames'], np.arange(start, end+1))
                        or not np.array_equal(data['K'], K[section])
                        or not np.array_equal(data['boxes'], boxes)):
                    raise ValueError('片段人物观测与共用相机输入不一致。')
                observed_points[section] = data['points']
            detected[observed-first] = True
        stem = f'{name}_seed{seed}'
        with np.load(inputs / f'{stem}_origin.npz', allow_pickle=False) as data:
            origin_parts.append((start,end,{key: data[key] for key in data.files}))
        if has_final:
            with np.load(inputs / f'{stem}.npz', allow_pickle=False) as data:
                final_parts.append((start,end,{key: data[key] for key in data.files}))
        if engine.postprocess is not None:
            reports.append(dict(engine.postprocess.last_report))
    if not share_camera:
        try:
            arms = measure_cut_arms(engine, working_video, parts, cameras, cuts, seam_cache,
                                    options['cancel_check'], MAX_DROPOUT)
        except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
            print(f'  [stitch] skipped image-supported arm transition: {exc}', flush=True)
    origin, report = stitch_fragments(origin_parts, first, last, cuts=cuts, references=references, observed_arms=arms)
    save_composed_motion(origin, os.path.join(sub_dir, f'{name}_seed{seed}_origin'), engine.smpl_model_path)
    if has_final:
        final, report = stitch_fragments(final_parts, first, last, cuts=cuts, references=references, observed_arms=arms)
        save_composed_motion(final, os.path.join(sub_dir, f'{name}_seed{seed}'), engine.smpl_model_path)
    if share_camera:
        motion = final if has_final else origin
        valid = validity(motion.get('valid_frames'), len(observed_points))
        observed_points[~(detected & valid)] = np.nan
        packet = folder / 'camera_inputs.npz'
        save_npz(packet, RT=raw, K=K, points=observed_points, valid=valid,
                 image_size_hw=np.array(image_size_hw), track_start=first, observed_frames=first+np.flatnonzero(detected))
        final_stem = f'{name}_seed{seed}' + ('' if has_final else '_origin')
        save_json(packet.with_suffix('.json'), dict(shots=shots, video_sha256=video_hash,
            sha256=file_revision_digest(packet), motion_sha256=file_revision_digest(Path(sub_dir)/f'{final_stem}.npz')))
        report['shared_camera_inputs'] = True
    if reports:
        engine.postprocess.last_report = {**reports[0],
            'contacts': sum(item['contacts'] for item in reports),
            'elapsed_seconds': sum(item['elapsed_seconds'] for item in reports)}
    return report
