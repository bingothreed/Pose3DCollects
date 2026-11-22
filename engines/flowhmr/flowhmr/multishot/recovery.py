"""Camera-only recovery. The boundary accepts joints, never pose/model parameters.

Image observations and gravity are supplied by the runtime adapter. Each valid
interval has one fixed similarity. A gap with detector support on both sides
within one shot shares that similarity and keeps its missing frames masked.
"""
import numpy as np

from .alignment import SM, MH, CameraFitConvergenceError, body_facing_rotation, fit_shot
from .camera import Camera, project
from .gate import gate_camera, observations
from .similarity import align_camera, transform_camera, validate_camera
from .timeline import intervals

LEGACY_SM = [16, 17, 1, 2, 4, 5, 7, 8]
LEGACY_MH = [5, 6, 9, 10, 11, 12, 13, 14]
# Loose alignment guard for disconnected observations of the same source shot.
# Measure its error relative to the observed body's on-screen span.
MAX_FRAGMENT_MEDIAN_ERROR_FRACTION = .12
# Require three observed frames that support the shared fit per rejected frame.
MIN_CONSISTENT_OBSERVATION_FRACTION = .75


class CameraCancelled(Exception):
    pass


def _fragment_errors(result, observed_frames):
    inside = result['inside']
    points = result['observed_uv']
    distance = np.linalg.norm(result['projected_uv'] - points, axis=-1)
    split = np.flatnonzero(np.diff(observed_frames) > 1) + 1
    fragments = []
    for indices in np.split(np.arange(len(inside)), split):
        seen = inside[indices]
        observed = points[indices]
        extent = (np.where(seen[..., None], observed, -np.inf).max(1)
                  - np.where(seen[..., None], observed, np.inf).min(1))
        span = float(np.median(np.linalg.norm(extent, axis=1)))
        error = distance[indices][seen]
        median = float(np.median(error))
        fragments.append(dict(
            source_frames=[int(observed_frames[indices[0]]), int(observed_frames[indices[-1]])+1],
            body_span_px=span, median_error_px=median, p95_error_px=float(np.percentile(error, 95)),
            median_error_fraction=median/max(span, 1e-6),
        ))
    return fragments


def _extend_camera(rt, K, anchors, result, image_size_hw):
    """Check that the accepted camera follows a plausible raw path over a gap."""
    try:
        raw, K = validate_camera(rt, K)
    except ValueError:
        return None
    path = transform_camera(raw, result['world_rotation'],
                            result['world_scale'], result['world_translation'])
    # Project fixed landmarks to check the camera path independently of any
    # poses generated during the gap. A cut or jump is unsafe to expose.
    uv, depth = project(path, K, np.broadcast_to(anchors, (len(path), *anchors.shape)))
    if not np.isfinite(uv).all() or np.min(depth) <= .05:
        return None
    step = np.median(np.linalg.norm(np.diff(uv, axis=0), axis=-1), axis=1)
    if step.max() > .05*np.hypot(*image_size_hw):
        return None
    return path


def _short_unreliable_observations(fragments, observed_frames):
    """Isolated detector reacquisition should not discard the rest of a shot."""
    bad = [row['source_frames'] for row in fragments
           if row['median_error_fraction'] > MAX_FRAGMENT_MEDIAN_ERROR_FRACTION]
    count = sum(b-a for a, b in bad)
    if count > 3 or len(observed_frames)-count < 12:
        return []
    for a,b in bad:
        before = observed_frames[observed_frames < a]
        after = observed_frames[observed_frames >= b]
        if len(before) and len(after) and (len(before) < 8 or len(after) < 8):
            return []
    return bad


def _continue_shot_edges(rt, K, joints, result, image_size_hw, start, end, observed, images, valid, max_gap):
    """Use the independently measured camera near the edge of its own shot."""
    for lo, hi, reference in ((start, int(observed[0]), int(observed[0])),
                              (int(observed[-1])+1, end, int(observed[-1]))):
        if not 0 < hi-lo <= max_gap or not (valid[lo:hi] & images[lo:hi]).all():
            continue
        a, b = min(lo, reference), max(hi, reference+1)
        path = _extend_camera(rt[a:b], K[a:b], joints[reference, SM], result, image_size_hw)
        if path is not None:
            yield lo, hi, path[lo-a:hi-a]


def recover_camera(rt, K, joints, points, valid, shots, image_size_hw, *,
                   gravity=None, track_start=0, check=lambda: None,
                   notify=lambda message, progress: None, observation_support=None, image_continuity=None,
                   shot_images=None, shot_edge_limit=9, register_cutaways=None):
    """Return camera and diagnostic report without mutating any caller array.

Shot bounds are track-relative, half-open. Approved archived gravity is an
immutable input of a reviewed recipe; new high-dispersion measurements fall
back to the long-window body orientation and ground constraint.
"""
    rt, K, joints, points = map(np.asarray, (rt, K, joints, points))
    valid = np.asarray(valid, dtype=bool)
    n = len(joints)
    if (joints.shape != (n, 52, 3) or points.shape != (n, 70, 3) or valid.shape != (n,)
            or rt.shape != (n, 4, 4) or K.shape != (n, 3, 3)):
        raise ValueError('相机输入与人物时间线不一致。')
    if not np.isfinite(joints[valid]).all():
        raise ValueError('冻结人物关节含无效值。')
    # Missing observations are camera gaps; the motion's validity is untouched.
    usable = valid & np.isfinite(points).all((1, 2))
    if observation_support is not None:
        support = np.asarray(observation_support)
        if support.shape != (n,) or support.dtype.kind != 'b':
            raise ValueError('相机观测帧与人物时间线不一致。')
        initial_support = np.flatnonzero(usable & support)
        usable[:] = False
        if len(initial_support):
            *_, seen = observations(joints[initial_support], points[initial_support], K[initial_support], image_size_hw)
            usable[initial_support] = seen.any(1)
    if not usable.all():
        points = np.where(usable[:, None, None], points, np.nan)
    rows, fit_segments, taken = [], [], np.zeros(n, bool)
    shot_bounds = {}
    ids = np.full(n, -1, np.int32)
    for shot in shots:
        a, b, sid = int(shot['start']), int(shot['end']), int(shot['id'])
        if not 0 <= a < b <= n or taken[a:b].any():
            raise ValueError('相机镜头范围重叠或越界。')
        taken[a:b] = True
        ids[a:b] = sid
        shot_bounds[sid] = (a, b)
        spans = intervals(usable[a:b])
        if observation_support is not None and len(spans) > 1:
            begin, end = a + spans[0][0], a + spans[-1][1]
            fit_segments.append((begin, end, sid))
        else:
            fit_segments.extend((a+x, a+y, sid) for x, y in spans)
        for x, y in intervals(valid[a:b] & ~usable[a:b]):
            rows.append(dict(shot=sid, source_frames=[a+x, a+y], labels=['no_observation']))
    if np.any(valid & ~taken):
        raise ValueError('相机镜头范围未覆盖有效人物帧。')
    # Reviewed V3 stores fixed similarities at float64 precision.
    aligned, intrinsics = rt.astype(float), K.copy()
    scales = {}
    for i, (a, b, sid) in enumerate(fit_segments):
        check()
        notify(f'摄像机整体尺度 · {i+1}/{len(fit_segments)}', 55 + 10*i/max(1, len(fit_segments)))
        observed_frames = a + np.flatnonzero(usable[a:b])
        count = len(observed_frames)
        sample = np.unique(np.linspace(0, count-1, min(80, count), dtype=int))
        sampled_frames = observed_frames[sample]
        try:
            legacy = align_camera(rt[observed_frames], K[observed_frames], sample,
                                  points[sampled_frames][:, LEGACY_MH],
                                  joints[sampled_frames][:, LEGACY_SM], check=check)
            scales[a, b] = float(legacy['world_scale'])
        except ValueError:
            # Static/short/weakly observed shots still have body-facing evidence.
            pass
    reliable = [s for s in scales.values() if .1 < s < 1000]
    prior = float(np.median(reliable)) if reliable else 1.
    initials = {}
    for a,b,_ in fit_segments:
        check()
        try:
            observed_frames = a + np.flatnonzero(usable[a:b])
            initials[a,b] = body_facing_rotation(rt[observed_frames], points[observed_frames][:, MH],
                                                 joints[observed_frames][:, SM])
        except (ValueError, np.linalg.LinAlgError):
            pass
    from .leveling import body_gravity
    from .gravity import MAX_DISPERSION_DEG
    fallback_up, leveling_report = body_gravity(rt, points, fit_segments, initials)
    fitted, camera_valid = [], np.zeros(n, bool)
    continued = np.zeros(n, bool)
    sparse_shots = []
    for i, (a, b, sid) in enumerate(fit_segments):
        check()
        notify(f'对齐原片机位 · {i+1}/{len(fit_segments)}', 65 + 25*i/max(1, len(fit_segments)))
        unsupported = np.zeros(n, dtype=bool)
        try:
            if (a,b) not in initials:
                raise ValueError('人物朝向观测不可用。')
            initial = initials[a,b]
            g = (gravity or {}).get(sid, {})
            up = np.asarray(g.get('mean_world_up', [0, 0, 0]), dtype=float)
            gravity_ok = (up.shape == (3,) and np.isfinite(up).all() and np.linalg.norm(up) > .5
                          and (float(g.get('dispersion_max_deg', 0)) <= MAX_DISPERSION_DEG or g.get('reviewed_input', False)))
            if not gravity_ok:
                up = fallback_up[a,b]
            rawscale = scales.get((a, b))
            own_scale = rawscale is not None and .1 < rawscale < 1000
            scale_prior = rawscale if own_scale else prior
            observed_frames = a + np.flatnonzero(usable[a:b])
            fit_data = (rt[observed_frames], K[observed_frames], points[observed_frames][:, MH],
                        joints[observed_frames][:, SM], up, initial)
            fit_options = dict(image_size_hw=image_size_hw, check=check, ground_min=None if gravity_ok else 0.)
            used_default_scale = False
            try:
                result = fit_shot(*fit_data, scale_prior, **fit_options)
            except CameraFitConvergenceError:
                if own_scale or scale_prior == 1.:
                    raise
                # A typical scale from other shots can be a poor prior for a
                # shot with little translation. Retry with the existing default.
                scale_prior = 1.
                result = fit_shot(*fit_data, scale_prior, **fit_options)
                used_default_scale = True
            report = result['report']
            if report['visible_min_depth_m'] <= .05:
                raise ValueError('对齐后人物位于摄像机后方。')
            report.update(shot=sid, source_frames=[a, b], labels=[],
                          gravity_source='image' if gravity_ok else 'body_long_window_and_ground',
                          gravity_method=g.get('method'),
                          gravity_evidence={key: g[key] for key in ('background_verticals', 'initial_dispersion_deg',
                                                                  'consensus_source_frames') if key in g},
                          gravity_dispersion_deg=g.get('dispersion_max_deg'),
                          gravity_sampled_source_frames=g.get('sampled_source_frames'),
                          reviewed_gravity=bool(g.get('reviewed_input', False)))
            if not gravity_ok:
                report['labels'].append('gravity_fallback')
            if used_default_scale:
                report['labels'].append('default_scale_retry')
            if len(observed_frames) < 12:
                report['labels'].append('short_shot')
            if len(observed_frames) == b-a:
                fitted.append(report)
            else:
                fragments = _fragment_errors(result, observed_frames)
                report['fragments'] = fragments
                report['maximum_fragment_median_error_fraction'] = MAX_FRAGMENT_MEDIAN_ERROR_FRACTION
                worst = max(fragments, key=lambda fragment: fragment['median_error_fraction'])
                if worst['median_error_fraction'] > MAX_FRAGMENT_MEDIAN_ERROR_FRACTION:
                    reject = (_short_unreliable_observations(fragments, observed_frames)
                              if shot_images is not None else [])
                    if reject:
                        keep = np.ones(len(observed_frames), bool)
                        for lo, hi in reject:
                            keep &= (observed_frames < lo) | (observed_frames >= hi)
                        observed_frames = observed_frames[keep]
                        result = fit_shot(rt[observed_frames], K[observed_frames], points[observed_frames][:, MH],
                                          joints[observed_frames][:, SM], up, initial, scale_prior,
                                          image_size_hw=image_size_hw, check=check, ground_min=None if gravity_ok else 0.)
                        report.update(result['report'])
                        if report['visible_min_depth_m'] <= .05:
                            raise ValueError('对齐后人物位于摄像机后方。')
                        report['unreliable_observation_ranges'] = reject
                        report['labels'].append('isolated_observations_rejected')
                        retained = _fragment_errors(result, observed_frames)
                        report['retained_fragments'] = retained
                        worst = max(retained, key=lambda fragment: fragment['median_error_fraction'])
                    if not reject and shot_images is not None:
                        rejected = [row['source_frames'] for row in fragments
                                    if row['median_error_fraction'] > MAX_FRAGMENT_MEDIAN_ERROR_FRACTION]
                        for lo, hi in rejected:
                            unsupported[lo:hi] = True
                        keep = ~unsupported[observed_frames]
                        if keep.sum() >= max(12, MIN_CONSISTENT_OBSERVATION_FRACTION * len(keep)):
                            # Keep the well-supported part of the original shared fit.
                            # A conflicting interval stays masked even if its raw path is smooth.
                            observed_frames = observed_frames[keep]
                            result['RT'] = result['RT'][keep]
                            report['unsupported_observation_ranges'] = rejected
                            report['labels'].append('inconsistent_observations_masked')
                    if (worst['median_error_fraction'] > MAX_FRAGMENT_MEDIAN_ERROR_FRACTION
                            and 'unsupported_observation_ranges' not in report):
                        begin, end = worst['source_frames']
                        report['labels'] += ['failed', 'incompatible_observation_fragments']
                        report['error'] = (
                            f'同一镜头中的人物轨迹片段无法与同一机位可靠对齐。'
                            f'原视频第 {track_start+begin+1}–{track_start+end} 帧的典型投影误差为'
                            f'{worst["median_error_px"]:.1f} 像素'
                            f'（画面内人体跨度的 {worst["median_error_fraction"]:.0%}）。'
                        )
                        from .partial_camera import recover_supported_run

                        recovered = recover_supported_run(rt[a:b], K[a:b], joints[a:b], points[a:b],
                            usable[a:b], image_size_hw, up, scale_prior, check=check,
                            ground_min=None if gravity_ok else 0.)
                        rows.append(report)
                        if recovered is not None:
                            indices, path, verified = recovered
                            indices = a + indices
                            aligned[indices] = path
                            camera_valid[indices] = True
                            recovered_range = [int(indices[0]), int(indices[-1])+1]
                            verified['training_frames'] = [a+f for f in verified['training_frames']]
                            rows.append(dict(shot=sid, source_frames=recovered_range,
                                labels=['partial_body_validated', 'temporal_gate_skipped_sparse'],
                                gravity_source=report['gravity_source'], **verified))
                        continue
                report['labels'].extend(['one_similarity_across_observation_gaps', 'temporal_gate_skipped_sparse'])
                rows.append(report)
                sparse_shots.append(sid)
                for left, right in zip(observed_frames[:-1], observed_frames[1:]):
                    lo, hi = left+1, right
                    if lo >= hi or unsupported[lo:hi].any():
                        continue
                    continuous = image_continuity is not None and image_continuity[lo:hi].all()
                    rejected = report.get('unreliable_observation_ranges', [])
                    within_shot = (shot_images is not None and hi-lo <= 7
                                   and (valid[lo:hi] & shot_images[lo:hi]).all()
                                   and any(lo <= x < y <= hi for x,y in rejected))
                    if continuous or within_shot:
                        anchors = (joints[left, SM]+joints[right, SM])*.5
                        path = _extend_camera(rt[left:right+1], K[left:right+1], anchors,
                                              result, image_size_hw)
                        if path is not None:
                            aligned[lo:hi] = path[1:-1]
                            continued[lo:hi] = True
            aligned[observed_frames] = result['RT']
            camera_valid[observed_frames] = True
            if shot_images is not None:
                start, end = shot_bounds[sid]
                for lo, hi, path in _continue_shot_edges(rt, K, joints, result, image_size_hw,
                                                        start, end, observed_frames, shot_images,
                                                        valid & ~unsupported, shot_edge_limit):
                    aligned[lo:hi] = path
                    continued[lo:hi] = True
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append(dict(shot=sid, source_frames=[a, b], labels=['failed'], error=str(exc)))
    check()
    camera_valid |= continued
    if not camera_valid.any():
        incompatible = next((row['error'] for row in rows
                             if 'incompatible_observation_fragments' in row['labels']), None)
        if incompatible:
            raise ValueError(incompatible + '已保留人物动作，可分段恢复摄像机。')
        raise ValueError('未得到有效摄像机；人物动作已保留。请检查人物可见范围与素材。')
    notify('稳定摄像机并整理逐镜头说明', 94)
    (final_rt, final_K), _, gate = gate_camera(aligned, intrinsics, joints.astype(float), points.astype(float), fitted,
                                               camera_valid, image_size_hw)
    if sparse_shots:
        gate['skipped_sparse_shots'] = sparse_shots
    decisions = {tuple(row['source_frames']): row for row in gate['shots']}
    for row in fitted:
        decision = decisions.get(tuple(row['source_frames']), {})
        labels = row['labels'] + decision.get('labels', [])
        rows.append({**row, **decision, 'labels': labels})
    check()
    camera = Camera(final_rt, final_K, camera_valid, ids, np.asarray(image_size_hw), track_start)
    report = dict(version=1, recipe='reviewed_v3_with_product_fallbacks',
                  track_start=int(track_start), fps=30, interval_convention='half_open_track_relative',
                  shots=sorted(rows, key=lambda r: r['source_frames'][0]), gate=gate, leveling=leveling_report,
                  generation_calls=0, motion_changed=False, physical_calibration_guaranteed=False)
    if continued.any():
        for row in rows:
            a, b = row['source_frames']
            if 'no_observation' in row['labels'] and continued[a:b].all():
                row['labels'].append('camera_continued_without_person_observation')
        report['camera_continuity'] = dict(
            frames=int(continued.sum()),
            source_ranges=[[a+track_start, b+track_start] for a, b in intervals(continued)],
            interval_convention='half_open_source_frames',
            method='same_similarity_on_raw_camera_path',
        )
    if register_cutaways is not None and not camera.camera_valid.all():
        ungated = Camera(aligned, intrinsics, camera_valid, ids, np.asarray(image_size_hw), track_start)
        notify('对齐只有背景的短镜头', 96)
        # A failed body fit cannot authorize raw-path continuation. Independent
        # registration may still recover its camera from static 3D landmarks.
        camera, cutaways = register_cutaways(camera, ungated, valid)
        if cutaways:
            report['background_alignment'] = cutaways
    return camera, report
