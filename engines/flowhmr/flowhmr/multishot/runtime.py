"""Lazy I/O adapter: existing VGGT/SAM lifecycle, frozen joints, atomic output.

No generation pipeline is constructed here. The caller lends its SAM context
and inference device; CPU/XPU fallback policy remains with the existing app.
"""
import hashlib
import json
import logging
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import mkstemp
from threading import RLock
from uuid import uuid4
from zipfile import BadZipFile
import numpy as np

from .quality import camera_coverage, camera_quality
from .recovery import recover_camera
from .timeline import intervals, validity

_paired_file_lock = RLock()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=64)
def _digest_revision(path, revision):
    return digest(path)


def file_revision_digest(path):
    path = Path(path)
    stat = path.stat()
    return _digest_revision(path, (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))


class InvalidCameraArtifact(ValueError):
    """A camera file is present but cannot be paired with the current motion."""


def load_paired_result(directory):
    from .camera import Camera

    root = Path(directory)
    camera_path = root / 'camera/camera.npz'
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    try:
        # Take one snapshot and release the Windows file handle before parsing.
        with _paired_file_lock:
            payload = camera_path.read_bytes()
            with np.load(BytesIO(payload), allow_pickle=False) as arrays:
                embedded = arrays.get('paired_report_json')
            paired_json = ((root / 'camera/report.json').read_text(encoding='utf-8')
                           if embedded is None else str(embedded))
        report = json.loads(paired_json)
        expected_motion = report['motion_sha256'] if 'motion_sha256' in report else report['source']['motion_sha256']
        if expected_motion != file_revision_digest(root / 'motion.npz'):
            raise InvalidCameraArtifact('摄像机与当前动作不一致，请重新恢复摄像机。')
        camera_hash = hashlib.sha256(payload).hexdigest()
        if embedded is None and report.get('camera_sha256', camera_hash) != camera_hash:
            raise InvalidCameraArtifact('摄像机文件已损坏，请重新恢复摄像机。')
        camera = Camera.load(BytesIO(payload))
        report.update(camera_sha256=camera_hash, quality=camera_quality(report), coverage=camera_coverage(camera))
        return camera, report
    except InvalidCameraArtifact:
        raise
    except (OSError, EOFError, ValueError, KeyError, TypeError, BadZipFile) as error:
        raise InvalidCameraArtifact('摄像机或恢复报告不可用，请重新恢复摄像机。') from error


def array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def bind_linked_camera_inputs(directory, first, count):
    """Bind the shared generation observations to the delivered motion."""
    root = Path(directory)
    packet = root / 'input/linked/camera_inputs.npz'
    metadata = packet.with_suffix('.json')
    meta = json.loads(metadata.read_text(encoding='utf8'))
    if meta['sha256'] != file_revision_digest(packet):
        raise ValueError('共用相机输入校验失败。')
    with np.load(packet, allow_pickle=False) as inputs, np.load(root/'motion.npz', allow_pickle=False) as motion:
        if (int(inputs['track_start']) != first or len(inputs['RT']) != count
                or not np.array_equal(inputs['valid'], validity(motion.get('valid_frames'), count))):
            raise ValueError('共用相机输入与最终动作时间线不一致。')
    meta['motion_sha256'] = file_revision_digest(root/'motion.npz')
    save_json(metadata, meta)


def _serialize_json(value):
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Unsupported camera report value: {type(value).__name__}')
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(_serialize_json(value), encoding='utf8')
    tmp.replace(path)


def save_npz(path, **values):
    path = Path(path)
    tmp = path.with_suffix('.npz.part')
    try:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **values)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def publish_camera(folder, camera, report, *, check=lambda: None, preserve_previous=False):
    """Replace the camera and its source report together as one validated NPZ."""
    path = Path(folder) / 'camera.npz'
    published = dict(report, publication_id=uuid4().hex, coverage=camera_coverage(camera),
                     quality=camera_quality(report))
    published.pop('camera_sha256', None)
    paired_json = _serialize_json(published)
    fd, name = mkstemp(prefix='camera-', suffix='.npz.part', dir=path.parent)
    candidate = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **camera.arrays(), paired_report_json=paired_json)
            stream.flush()
            os.fsync(stream.fileno())
        published['camera_sha256'] = digest(candidate)
        with _paired_file_lock:
            if preserve_previous:
                _check_retry_result(path.parent.parent, camera, published)
            check()
            candidate.replace(path)
            report.update(published)
            try:
                # Keep a readable mirror for offline inspection; the NPZ is authoritative.
                save_json(path.with_name('report.json'), published)
            except OSError as error:
                logging.getLogger(__name__).warning('Camera report mirror unavailable: %s', error)
    finally:
        candidate.unlink(missing_ok=True)


def camera_directory(directory, job):
    root = Path(directory)
    return root / 'scene' if job.get('person_batch') == job['id'] else root


def _observation_support(job, first, count, observed_frames):
    if observed_frames is not None:
        frames = np.asarray(observed_frames)
        if (frames.ndim != 1 or frames.dtype.kind not in 'iu'
                or np.any(frames < first) or np.any(frames >= first+count)
                or np.any(frames[1:] <= frames[:-1])):
            raise ValueError('人物的原始检测帧与动作时间线不一致。')
        direct = np.zeros(count, bool)
        direct[frames.astype(np.int64) - first] = True
        return direct, 'detected_frames'
    fragments = job.get('track_fragments')
    if not fragments:
        return None, 'unknown'
    direct = np.zeros(count, bool)
    for fragment in fragments:
        a, b = int(fragment['start']), int(fragment['end'])+1
        if a >= b:
            raise ValueError('人物续接片段的检测范围无效。')
        a, b = max(a, first), min(b, first+count)
        if a < b:
            direct[a-first:b-first] = True
    return direct, 'legacy_fragment_ranges'


def _short_camera_continuity(video, first, valid, support, check):
    """Find short detector dropouts with a continuous video image on both sides."""
    import cv2

    # 100 ms at the normalized 30 fps. Longer gaps stay out of camera coverage.
    gaps = [(a, b) for a, b in intervals(~support)
            if 0 < a < b < len(valid) and b-a <= 3 and valid[a-1:b+1].all()]
    if not gaps:
        return None
    continuous = np.zeros(len(valid), bool)
    cap = cv2.VideoCapture(str(video))
    try:
        for a, b in gaps:
            check()
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, first+a-1):
                continue
            previous = None
            for _ in range(a-1, b+1):
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
                contrast = float(image.std())
                # Avoid flat/blank pictures and visible cuts. Compare change to
                # local image contrast rather than to any person's detected box.
                if contrast < 6 or (previous is not None and
                    np.abs(image-previous).mean() > .6 * (contrast + previous_contrast) / 2):
                    break
                previous, previous_contrast = image, contrast
            else:
                continuous[a:b] = True
    finally:
        cap.release()
    return continuous


def _check_retry_result(directory, camera, report):
    try:
        old_camera, previous = load_paired_result(directory)
    except (OSError, InvalidCameraArtifact):
        return
    previous_frames = np.flatnonzero(old_camera.camera_valid) + old_camera.track_start
    relative = previous_frames - camera.track_start
    in_range = (relative >= 0) & (relative < len(camera.camera_valid))
    lost = len(relative) - np.count_nonzero(camera.camera_valid[relative[in_range]])
    if lost:
        raise ValueError(f'重新计算有 {lost} 帧失去可用机位，已保留上一次可用的摄像机。')
    quality = report['quality']
    if not quality['degraded']:
        return
    old = previous['quality']
    if old['degraded']:
        # A detected cut can change shot IDs. Compare the same previously
        # available source frames so added coverage and renumbering are safe.
        current_fallback = np.isin(camera.shot_ids[relative], quality['fallback_shots'])
        old_fallback = np.isin(old_camera.shot_ids[old_camera.camera_valid], old['fallback_shots'])
        if not np.any(current_fallback & ~old_fallback):
            return
    if old['status'] != 'unknown':
        raise ValueError('重新计算仍未取得更可靠的画面重力，已保留上一次可用的摄像机。')


def _raw_camera(directory, job, first, count, hw, video, device, check, notify, release_cuda, *, separate_shot=False):
    from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera_isolated, make_static_camera
    mode = 'static' if job['params'].get('static_camera') else 'vggt'
    source = directory / f'input/input_{mode}_camera.npz'
    if not separate_shot:
        try:
            with np.load(source, allow_pickle=False) as z:
                a, b = map(int, z['frame_range'])
                if a <= first <= first+count-1 <= b and str(z['k_space']) == 'original_frame':
                    s = slice(first-a, first-a+count)
                    rt, K = z['camera_RT'][s], z['K'][s]
                    if rt.shape == (count, 4, 4) and K.shape == (count, 3, 3) and np.isfinite(rt).all() and np.isfinite(K).all():
                        notify('已复用任务的原始相机观测', 18)
                        return rt, K, 'task_cache'
        except (OSError, ValueError, KeyError, BadZipFile):
            pass
    prefix = 'raw_static' if mode == 'static' else 'raw'
    output = directory / 'camera' / f'{prefix}_{first}_{first+count-1}.npz'
    if mode == 'static':
        result = make_static_camera(hw, frame_range=(first, first+count-1), output_npz=str(output))
    else:
        result = extract_vggt_camera_isolated(str(video), output_npz=str(output),
            frame_range=(first, first+count-1), expected_len=count, device=device,
            frame_interval=job['params'].get('vggt_interval', 0), cancel_check=check,
            on_progress=lambda *args: notify(str(args[-1]), 15), release_cuda=release_cuda)
    check()
    return result.camera_RT, result.K, mode


def estimate_shot_cameras(directory, job, shots, first, count, hw, video, device, check, notify, release_cuda):
    if len(shots) == 1:
        return _raw_camera(directory, job, first, count, hw, video, device, check, notify, release_cuda)
    (directory / 'camera').mkdir(exist_ok=True)
    rt = np.empty((count, 4, 4), dtype=np.float32)
    K = np.empty((count, 3, 3), dtype=np.float32)
    for index, shot in enumerate(shots, 1):
        a, b = shot['start'], shot['end']
        def shot_progress(message, _progress):
            notify(f'镜头 {index}/{len(shots)} · {message}', 5 + 13 * index / len(shots))
        rt[a:b], K[a:b], _ = _raw_camera(directory, job, first+a, b-a, hw, video, device,
                                          check, shot_progress, release_cuda, separate_shot=True)
    return rt, K, 'per_shot'


def _observations(folder, video, boxes, K, valid, first, acquire_sam, identity, check, notify):
    import cv2
    target, proof = folder / 'observations.npz', folder / 'observations.json'
    try:
        meta = json.loads(proof.read_text(encoding='utf8'))
        legacy = {key: value for key, value in identity.items() if key != 'observation_support'}
        if meta['identity'] in (identity, legacy) and meta['sha256'] == digest(target):
            with np.load(target, allow_pickle=False) as z:
                points = z['points']
                if points.shape == (len(valid), 70, 3) and np.isfinite(points[valid]).all():
                    notify('已复用人物观测，动作保持不变', 45)
                    return points, True
    except (OSError, ValueError, KeyError, BadZipFile):
        pass
    points = np.full((len(valid), 70, 3), np.nan)
    selected = np.flatnonzero(valid)
    if not np.isfinite(boxes[valid]).all():
        raise ValueError('旧任务缺少有效的人物检测框，无法补算相机观测。')
    cap = cv2.VideoCapture(str(video))
    previous = -2
    try:
        check()
        with acquire_sam() as extractor:
            for i, frame in enumerate(selected):
                check()
                if frame != previous+1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(first+frame))
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'无法读取原视频第 {first+frame+1} 帧。')
                points[frame] = extractor.scene_keypoints(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    boxes[frame:frame+1], K[frame], list(range(70)))[0]
                previous = frame
                if i % 5 == 0 or i+1 == len(selected):
                    notify(f'补算人物观测 · {i+1}/{len(selected)} 帧', 20 + 25*(i+1)/len(selected))
    finally:
        cap.release()
    check()
    save_npz(target, points=points)
    save_json(proof, dict(identity=identity, sha256=digest(target), generation_calls=0))
    return points, False


def measure_gravity(video, rt, K, shots, first, folder, device, check, notify, *, diagnostics,
                    observation_support, person_boxes, release_cuda=None):
    from contextlib import nullcontext
    from .gravity import image_gravity

    with (release_cuda or nullcontext)():
        return image_gravity(video, rt, K, shots, first, folder, device, check, notify,
                             diagnostics=diagnostics, observation_support=observation_support,
                             **({'person_boxes': person_boxes} if person_boxes is not None else {}))


def _saved_boxes(directory, first, count):
    try:
        with np.load(directory / 'input/input_bbox.npz', allow_pickle=False) as z:
            if np.array_equal(z['start_end'], [first, first+count-1]) and z['bbox'].shape == (count, 4):
                return z['bbox']
    except (OSError, ValueError, KeyError, BadZipFile):
        pass
    return None


def _refine_fixed_path(video, folder, camera, report, rt, K, joints, points, boxes, shots,
                       eligible, valid, device, release_cuda, check, notify):
    """Use installed weights only. The action's camera mode remains independent."""
    if boxes is None or str(device).split(':', 1)[0] != 'cuda':
        return camera
    import torch
    from ..model_files import VGGT, file_present, resolve_model
    from ..runtime_rocm import uses_shared_memory
    from ..utils.runtime_vggt_camera import extract_vggt_camera_isolated
    from .fixed_camera_refinement import refine_fixed_shots

    if uses_shared_memory(torch, device) or not file_present(resolve_model(VGGT)):
        return camera

    first = camera.track_start

    def estimate(a, b):
        output = folder / f'raw_verified_motion_{first+a}_{first+b-1}.npz'
        result = extract_vggt_camera_isolated(str(video), output_npz=str(output),
            frame_range=(first+a, first+b-1), expected_len=b-a, device=device,
            frame_interval=max(3, (b-a+47)//48), cancel_check=check,
            on_progress=lambda message: notify(message, 96), release_cuda=release_cuda)
        return result.camera_RT, result.K

    def measure(proposed_rt, proposed_K, selected_shots):
        directory = folder / 'motion_gravity'
        directory.mkdir(exist_ok=True)
        return measure_gravity(video, proposed_rt, proposed_K, selected_shots, first, directory, device,
                               check, lambda message, _progress: notify(message, 97), diagnostics={},
                               observation_support=eligible, person_boxes=boxes, release_cuda=release_cuda)

    return refine_fixed_shots(video, camera, report, rt, K, joints, points, boxes, shots, eligible, valid,
                              estimate=estimate, measure_gravity=measure, check=check, notify=notify)


def build_camera(directory, job, inspection, acquire_sam, device, *, check=lambda: None,
                 notify=lambda message, progress: None, release_cuda=None):
    from contextlib import nullcontext
    from .shared import build_shared_camera
    directory = Path(directory)
    if job.get('person_batch') == job['id']:
        return build_shared_camera(directory / 'scene', check=check, notify=notify,
                                   preserve_previous=bool(job.get('camera_force')))
    folder = directory / 'camera'
    folder.mkdir(exist_ok=True)
    motion_hash = digest(directory / 'motion.npz')
    check()
    notify('读取已完成的人物动作', 2)
    path, info = inspection.load(directory, body_reference=job.get('body_reference') or 'neutral')
    joints = np.fromfile(path, dtype='<f4').reshape(info['frames'], 52, 3)
    with np.load(directory / 'motion.npz', allow_pickle=False) as z:
        valid = validity(z.get('valid_frames'), len(joints))
    try:
        with np.load(directory / 'input/camera_arm_reference.npz', allow_pickle=False) as z:
            original_arms = z['joints']
            if (str(z['motion_sha256']) == motion_hash and original_arms.shape == (len(joints), 4, 3)
                    and np.isfinite(original_arms[valid]).all()):
                joints[:, 18:22] = original_arms
    except (OSError, ValueError, KeyError, BadZipFile):
        pass
    first = int((job.get('track_range') or [0])[0])
    video = directory / 'input/input_30fps.mp4'
    # Explicit, source-identified archived input contract. It contains raw camera,
    # SAM evidence and reviewed gravity, never fitted/gated output cameras.
    archive = folder / 'inputs.npz'
    meta_path = folder / 'inputs.json'
    linked = directory / 'input/linked/camera_inputs.npz'
    linked_meta = linked.with_suffix('.json')
    gravity_diagnostic = {}
    continuity = shot_images = boxes = None
    if linked.exists() and linked_meta.exists():
        meta = json.loads(linked_meta.read_text(encoding='utf8'))
        if meta['motion_sha256'] != motion_hash or meta['sha256'] != digest(linked):
            raise ValueError('跨镜头观测与当前动作不一致，请重跑任务。')
        with np.load(linked, allow_pickle=False) as z:
            if not np.array_equal(z['valid'], valid):
                raise ValueError('跨镜头观测有效区间不一致。')
            rt, K, points, hw = (z[k] for k in ('RT', 'K', 'points', 'image_size_hw'))
            first = int(z['track_start'])
            support, support_source = _observation_support(job, first, len(joints), z.get('observed_frames'))
        shots = meta['shots']
        source = dict(raw_camera='linked_shot_cache', observations_reused=True, generation_calls=0)
        if 'video_sha256' in meta:
            if meta['video_sha256'] != file_revision_digest(video):
                raise ValueError('共用相机输入与当前原片不一致，请重跑任务。')
            from .shot_detection import camera_shots

            notify('已复用生成阶段的机位与人物观测', 45)
            _, shot_images = camera_shots(video, first, len(joints), check)
            boxes = _saved_boxes(directory, first, len(joints))
            if support is not None and not support.all():
                continuity = _short_camera_continuity(video, first, valid, support, check)
    elif archive.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf8'))
        if meta['motion_sha256'] != motion_hash or meta['sha256'] != digest(archive):
            raise ValueError('相机输入档案与冻结人物不一致。')
        with np.load(archive, allow_pickle=False) as z:
            if not np.array_equal(z['joints'][valid], joints[valid]) or not np.array_equal(z['valid'], valid):
                raise ValueError('相机输入档案的人物关节或有效范围不一致。')
            rt, K, points, hw = (z[k] for k in ('RT', 'K', 'points', 'image_size_hw'))
            support, support_source = _observation_support(job, first, len(joints), z.get('observed_frames'))
        shots = meta['shots']
        source = dict(raw_camera='archived_inputs', observations_reused=True, archive_sha256=meta['sha256'])
    else:
        import cv2
        cap = cv2.VideoCapture(str(video))
        hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if min(hw) < 1 or abs(fps-30) > .01:
            raise ValueError('此旧任务缺少可读取的 30fps 原视频。')
        boxes_path = directory / 'input/input_bbox.npz'
        if not boxes_path.is_file():
            raise ValueError('此旧任务缺少人物检测资料，请恢复原任务的检测缓存后重试；已有动作仍可导出。')
        with np.load(boxes_path, allow_pickle=False) as z:
            first, last = map(int, z['start_end'])
            boxes = z['bbox']
            support, support_source = _observation_support(job, first, len(joints), z.get('observed_frames'))
        if last-first+1 != len(joints) or len(boxes) != len(joints) or last >= frames:
            raise ValueError('人物动作与原视频检测范围不一致。')
        eligible = valid if support is None else valid & support
        if not eligible.any():
            raise ValueError('没有可用于恢复摄像机的人物检测帧。已保留人物动作。')
        from .shot_detection import camera_shots

        notify('检测原视频切镜', 5)
        shots, textured = camera_shots(video, first, len(joints), check)
        rt, K, mode = estimate_shot_cameras(directory, job, shots, first, len(joints), hw, video, device,
                                            check, notify, release_cuda or nullcontext)
        # A single shot can also contain detector dropouts in either camera mode.
        if support is not None and (mode == 'per_shot' or not support.all()):
            shot_images = textured
        identity = dict(video_sha256=digest(video), boxes=array_digest(boxes), K=array_digest(K),
                        valid=array_digest(valid), first=first, method='upstream_scene_keypoints_body_70_v1')
        if not np.array_equal(eligible, valid):
            identity['observation_support'] = array_digest(eligible)
        sidecar = directory / ('input/input_sam3d_feat' + ('_static' if job['params'].get('static_camera') else '') + '.pt.observations.npz')
        points, reused = None, False
        try:
            with np.load(sidecar, allow_pickle=False) as z:
                cached_points = z['points']
                if (np.array_equal(z['K'], K) and np.array_equal(z['boxes'], boxes)
                    and np.array_equal(z['source_frames'], np.arange(first,first+len(joints)))
                    and cached_points.shape == (len(joints),70,3) and np.isfinite(cached_points[eligible]).all()):
                    points, reused = cached_points, True
        except (OSError, ValueError, KeyError, BadZipFile):
            pass
        if points is None:
            points, reused = _observations(folder, video, boxes, K, eligible, first, acquire_sam, identity, check, notify)
        if support is not None and not support.all() and (mode == 'per_shot' or not job.get('track_fragments')):
            continuity = _short_camera_continuity(video, first, valid, support, check)
        source = dict(raw_camera=mode, observations_reused=reused, observations_identity=identity)
    eligible = valid if support is None else valid & support
    if source['raw_camera'] == 'archived_inputs' and (not job.get('camera_force') or not video.is_file()):
        gravity = {int(k): v for k, v in meta['gravity'].items()}
        gravity_diagnostic = dict(status='archived')
    else:
        gravity = measure_gravity(video, rt, K, shots, first, folder, device, check, notify,
                                  diagnostics=gravity_diagnostic,
                                  observation_support=eligible if support is not None and not eligible.all() else None,
                                  person_boxes=boxes if boxes is not None else _saved_boxes(directory, first, len(joints)),
                                  release_cuda=release_cuda)
    source['observation_support'] = dict(
        method=support_source,
        eligible_frames=int(eligible.sum()),
        ranges=[[first+a, first+b] for a, b in intervals(eligible)],
        interval_convention='half_open_source_frames',
    )
    register = None
    if shot_images is not None and job.get('shot_linking_report'):
        from .background_camera import complete_cutaway_cameras

        def register(camera, ungated, background_valid):
            return complete_cutaway_cameras(video, camera, rt, boxes, background_valid, shot_images, check,
                                            ungated=ungated)

    camera, report = recover_camera(rt, K, joints, points, valid, shots, hw,
        gravity=gravity, track_start=first, check=check, notify=notify, observation_support=support,
        image_continuity=continuity, shot_images=shot_images,
        shot_edge_limit=30 if job.get('shot_linking_report') else 9, register_cutaways=register)
    partial_runs = [row['source_frames'] for row in report['shots']
                    if 'partial_body_validated' in row.get('labels', ())]
    if partial_runs and shot_images is not None:
        from .background_camera import continue_background_run

        notify('沿同一镜头的静态背景恢复机位', 97)
        camera, background = continue_background_run(video, camera, rt, boxes, valid, shot_images, partial_runs, check)
        if background:
            report.setdefault('background_alignment', []).extend(background)
    if job['params'].get('static_camera') and video.is_file():
        camera = _refine_fixed_path(video, folder, camera, report, rt, K, joints, points,
            boxes if boxes is not None else _saved_boxes(directory, first, len(joints)),
            shots, eligible, valid, device, release_cuda or nullcontext, check, notify)
    visible_ranges = (job.get('fragment_stitching') or {}).get('visible_ranges')
    if visible_ranges is not None:
        report['visible_ranges'] = visible_ranges
    check()
    if digest(directory / 'motion.npz') != motion_hash:
        raise ValueError('相机计算期间人物动作已变化，请重新计算。')
    report.update(source=source, motion_sha256=motion_hash, body_reference=job.get('body_reference') or 'neutral')
    report['gravity'] = gravity_diagnostic
    check()
    publish_camera(folder, camera, report, check=check, preserve_previous=bool(job.get('camera_force')))
    return report
