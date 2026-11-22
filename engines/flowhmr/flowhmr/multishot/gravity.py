"""GeoCalib measurements with explicit, persistent fallback diagnostics."""
import json
import numpy as np

MEASUREMENT_REVISION = 3
MAX_DISPERSION_DEG = 8.


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _degrees(cosine):
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def _consensus(directions, uncertainty):
    # A few wide shots give more reliable sky/architecture cues than closeups.
    # Bound the weights so one confidently wrong image cannot dominate.
    uncertainty = np.where(np.isfinite(uncertainty), uncertainty, .1)
    weights = np.clip(.08 / np.maximum(uncertainty, .025), .5, 2.)
    neighbors = _degrees(directions @ directions.T) <= 4.
    chosen = neighbors[np.argmax(neighbors @ weights)]
    up = _unit(np.average(directions[chosen], axis=0, weights=weights[chosen]))
    supported = (chosen.sum() >= max(4, .5 * len(directions))
                 and float(np.median(uncertainty[chosen])) <= .09)
    return up, chosen, supported


def vertical_angles(up, lines, projection):
    """Angular residual of image lines to the vertical vanishing point."""
    midpoints = (lines[:, :2] + lines[:, 2:]) * .5
    measured = _unit(lines[:, 2:] - lines[:, :2])
    vanishing = projection @ up
    predicted = _unit(vanishing[:2] - midpoints * vanishing[2])
    cross = predicted[:, 0] * measured[:, 1] - predicted[:, 1] * measured[:, 0]
    return (np.degrees(np.arctan2(cross, (predicted * measured).sum(1))) + 90) % 180 - 90


def background_verticals(image, box, K, rotation, prior, detector):
    import cv2

    width = min(768, image.shape[1])
    scale = width / image.shape[1]
    height = round(image.shape[0] * scale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    measurements = detector.detect(gray)[0]
    if measurements is None:
        return None
    lines = measurements[:, 0].astype(float)
    x1, y1, x2, y2 = lines.T
    x, y = (x1 + x2) * .5, (y1 + y2) * .5
    length = np.hypot(x2 - x1, y2 - y1)
    left, top, right, bottom = box * scale
    outside = ((np.maximum(x1, x2) < left - 5) | (np.minimum(x1, x2) > right + 5)
               | (np.maximum(y1, y2) < top - 5) | (np.minimum(y1, y2) > bottom + 5))
    good = ((length > .035 * height) & (np.abs(y2 - y1) > .8 * length) & outside
            & (x > .025 * width) & (x < .97 * width) & (y > .092 * height) & (y < .98 * height))
    projection = np.diag([scale, scale, 1.]) @ K @ rotation
    shown, length = lines[good], length[good]
    near_vertical = np.abs(vertical_angles(prior, shown, projection)) <= 13.
    if not near_vertical.any():
        return None
    return shown[near_vertical], length[near_vertical], projection


def _line_evidence(prior, frames):
    """Fit on interleaved frames and test against the other half of the shot."""
    from scipy.optimize import least_squares

    if len(frames) < 8 or sum(len(lines) for lines, _, _ in frames) < 40:
        return None, {}
    seed_axis = np.eye(3)[np.argmin(np.abs(prior))]
    tangent = _unit(np.cross(prior, seed_axis))
    basis = np.stack([tangent, np.cross(prior, tangent)], axis=1)

    def residual(up, data, weighted=False):
        return np.concatenate([
            vertical_angles(up, lines, projection)
            * (np.sqrt(length / length.sum() * min(10, len(length))) if weighted else 1.)
            for lines, length, projection in data
        ])

    def fit(data):
        solve = least_squares(lambda v: residual(_unit(prior + basis @ v), data, weighted=True),
                              np.zeros(2), loss='cauchy', f_scale=1.45, bounds=(-.32, .32))
        return _unit(prior + basis @ solve.x)

    left, right = frames[::2], frames[1::2]
    from_left, from_right = fit(left), fit(right)
    dispersion = float(_degrees(from_left @ from_right))
    original = np.abs(residual(prior, frames))
    heldout = np.abs(np.r_[residual(from_left, right), residual(from_right, left)])
    base, measured = np.percentile(original, [50, 75]), np.percentile(heldout, [50, 75])
    # A closely agreeing split is independent support even when the shot's
    # image prior is inconsistent. Weaker geometry must improve over the prior.
    better = dispersion <= MAX_DISPERSION_DEG and measured[0] <= base[0] and measured[1] <= .92 * base[1]
    supported = measured[0] <= 3.5 and measured[1] <= 6. and (dispersion <= 2.5 or better)
    details = dict(frames=len(frames), segments=len(heldout), split_difference_deg=dispersion,
                   prior_residual_deg=base.tolist(), heldout_residual_deg=measured.tolist())
    return (fit(frames) if supported else None), details


def image_gravity(video, rt, K, shots, first, folder, device, check, notify, *, diagnostics=None,
                  observation_support=None, person_boxes=None):
    from .runtime import digest, array_digest, save_json
    from .recovery import CameraCancelled
    from .. import geocalib_setup as setup, model_files, model_sources
    diagnostic = diagnostics if diagnostics is not None else {}
    diagnostic.update(status='unavailable', source_revision=setup.SOURCE_REVISION)
    check()
    reason = setup.readiness(device, verify=True)
    if reason:
        diagnostic['reason'] = reason
        notify('画面重力不可用，采用人物与地面估计：' + reason, 50)
        return {}
    weights = model_files.resolve_model(model_files.GEOCALIB)
    diagnostic['model_sha256'] = model_sources.PUBLIC_ASSETS['geocalib'][0].sha256
    cache = folder / 'gravity.json'
    identity = dict(video=digest(video), RT=array_digest(rt), K=array_digest(K), first=first,
                    shots=shots, weights=diagnostic['model_sha256'], source_revision=setup.SOURCE_REVISION,
                    measurement_revision=MEASUREMENT_REVISION)
    if observation_support is not None:
        identity['observation_support'] = array_digest(observation_support)
    if person_boxes is not None:
        identity['person_boxes'] = array_digest(person_boxes)
    try:
        old = json.loads(cache.read_text(encoding='utf8'))
        if old['identity'] == identity:
            diagnostic.update(status='measured', cache_reused=True)
            return {int(k): v for k, v in old['shots'].items()}
    except (OSError, ValueError, KeyError):
        pass
    model, cap = None, None
    result = {}
    try:
        import cv2
        import torch
        GeoCalib = setup.load_class()
        check()
        notify('估计画面重力方向', 48)
        model = GeoCalib(weights=str(weights)).to(device).eval()
        cap = cv2.VideoCapture(str(video))
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

        def read(frame):
            check()
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(first + frame))
            ok, image = cap.read()
            if not ok:
                raise ValueError('Gravity source decode failed')
            return image

        def observe(frame, image):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255
            pred = model.calibrate(tensor, priors={'focal': torch.tensor([K[frame, 1, 1]], device=device)})
            up = pred['gravity'].vec3d.detach().cpu().numpy().reshape(-1, 3)[0]
            noise = pred.get('gravity_uncertainty')
            return _unit(rt[frame, :3, :3].T @ up), (float(noise.item()) if noise is not None else np.nan)

        with torch.inference_mode():
            for shot in shots:
                a, b = shot['start'], shot['end']
                eligible = np.arange(a, b) if observation_support is None else a + np.flatnonzero(observation_support[a:b])
                if not len(eligible):
                    continue
                if observation_support is None:
                    frames = np.unique(np.linspace(a+min(2,b-a-1), b-1-min(2,b-a-1), 5, dtype=int))
                else:
                    frames = eligible[np.linspace(0, len(eligible)-1, min(5, len(eligible)), dtype=int)]
                measured = {frame: observe(frame, read(frame)) for frame in frames}
                directions = np.stack([value[0] for value in measured.values()])
                mean = _unit(np.median(directions, axis=0))
                dispersion = float(_degrees(directions @ mean).max())
                row = dict(mean_world_up=mean.tolist(), dispersion_max_deg=dispersion,
                           sampled_source_frames=(first+frames).tolist(), method='geocalib')
                if dispersion > MAX_DISPERSION_DEG and len(eligible) >= 8:
                    extra = eligible[np.linspace(0, len(eligible)-1, min(17, len(eligible)), dtype=int)]
                    frames = np.union1d(frames, extra)
                    for frame in frames:
                        if frame not in measured:
                            measured[frame] = observe(frame, read(frame))
                    directions = np.stack([measured[frame][0] for frame in frames])
                    noise = np.array([measured[frame][1] for frame in frames])
                    consensus, inliers, reliable = _consensus(directions, noise)
                    prior = consensus if reliable else _unit(np.median(directions, axis=0))
                    line_frames = []
                    if person_boxes is not None:
                        for frame in extra:
                            if not np.isfinite(person_boxes[frame]).all():
                                continue
                            evidence = background_verticals(read(frame), person_boxes[frame], K[frame],
                                                            rt[frame, :3, :3], prior, detector)
                            if evidence is not None:
                                line_frames.append(evidence)
                    image_up, support = _line_evidence(prior, line_frames)
                    row.update(sampled_source_frames=(first+frames).tolist(),
                               initial_dispersion_deg=dispersion, background_verticals=support,
                               consensus_source_frames=(first+frames[inliers]).tolist())
                    if image_up is not None:
                        row.update(mean_world_up=image_up.tolist(), method='background_verticals',
                                   dispersion_max_deg=support['split_difference_deg'])
                    elif reliable:
                        row.update(mean_world_up=consensus.tolist(), method='geocalib_consensus',
                                   dispersion_max_deg=float(_degrees(directions[inliers] @ consensus).max()))
                result[int(shot['id'])] = row
        save_json(cache, dict(identity=identity, shots=result))
        diagnostic.update(status='measured', cache_reused=False)
    except CameraCancelled:
        raise
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        diagnostic.update(status='unavailable', reason='runtime_missing' if isinstance(exc, ImportError) else 'inference_failed',
                          detail=f'{type(exc).__name__}: {exc}')
        notify(f'画面重力不可用，采用人物与地面估计：{exc}', 50)
        result = {}
    finally:
        if cap is not None:
            cap.release()
        del model
    return result
