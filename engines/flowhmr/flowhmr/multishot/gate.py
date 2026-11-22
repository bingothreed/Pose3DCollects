"""Shot-local, person-observed shared R/C/log-focal fast-motion gate.

No body fitting and no background/temporal camera optimizer. Statistics below
are observational heuristics, not metric scale ground truth. Visibility-changing
centroids remain an explicitly reported measurement limitation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GateConfig:
    sigma_frames: float = 3.0
    minimum_frames: int = 12
    slow_travel_m: float = 1.0
    slow_correlation: float = .8
    supported_fast_gain: float = .7
    cap_percentile: float = 90.
    # User visual review (2026-11-14): noisy person projection is diagnostic,
    # not an automatic veto of visibly better camera stability.
    reject_worse_residual: bool = False


SM = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
MH = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
IMPORTANCE = np.array([1, 1, .6, .6, .35, .35, 1, 1, .6, .6, .5, .5])


def project(R, C, X, K):
    q = np.einsum('tij,tkj->tki', R, X - C[:, None])
    q = np.einsum('tij,tkj->tki', K, q)
    return q[..., :2] / q[..., 2:]


def observations(joints, points, K, image_size_hw):
    H, W = image_size_hw
    X, Y = joints[:, SM], points[:, MH]
    q = np.einsum('tij,tkj->tki', K, Y)
    uv = q[..., :2] / q[..., 2:]
    inside = ((uv[..., 0] >= 0) & (uv[..., 0] < W) &
              (uv[..., 1] >= 0) & (uv[..., 1] < H) & (Y[..., 2] > .05))
    w = inside * IMPORTANCE + 1e-3
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * X).sum(1), (w[..., None] * Y).sum(1), (w[..., None] * uv).sum(1), inside


def lowpass(x, sigma):
    return gaussian_filter1d(x, sigma, axis=0, mode='nearest')


def split(R, C, K, mx, my, scale_ratio, sigma):
    C = C.copy()
    if scale_ratio != 1.:
        C *= scale_ratio
        anchored = mx - np.einsum('tji,tj->ti', R, my)
        C += np.median(anchored - C, axis=0)
    ref = R[len(R) // 2]
    rv = Rotation.from_matrix(R @ ref.T).as_rotvec()
    lf = np.log(np.stack([K[:, 0, 0], K[:, 1, 1]], axis=1))
    pairs = [(lowpass(x, sigma), x - lowpass(x, sigma)) for x in (C, rv, lf)]
    return ref, pairs, K


def compose(parts, gain):
    ref, pairs, K = parts
    C, rv, lf = [slow + gain * fast for slow, fast in pairs]
    Kn = K.copy()
    Kn[:, 0, 0], Kn[:, 1, 1] = np.exp(lf).T
    return Rotation.from_rotvec(rv).as_matrix() @ ref, C, Kn


def gain_from_person(parts, mx, target, sigma):
    R0,C0,K0 = compose(parts, 0.)
    R1,C1,K1 = compose(parts, 1.)
    q0 = project(R0,C0,mx[:, None],K0)[:, 0]
    q1 = project(R1,C1,mx[:, None],K1)[:, 0]
    u, y = [((x - lowpass(x, sigma))[2:-2]) for x in (q1 - q0, target - q0)]
    dot = float((u * y).sum())
    gain = dot / max(float((u * u).sum()), 1e-9)
    correlation = dot / np.sqrt(float((u * u).sum() * (y * y).sum()) + 1e-12)
    return float(np.clip(gain, 0., 1.)), gain, float(correlation)


def metric_arrays(R, C, K, mx, target, joints):
    residual = project(R, C, mx[:, None], K)[:, 0] - target
    d2 = residual[2:] - 2 * residual[1:-1] + residual[:-2]
    X = joints[1:-1][:, [0, 1, 2, 12, 16, 17]]
    p = [project(R[k:len(R)-2+k], C[k:len(R)-2+k], X, K[k:len(R)-2+k]).mean(1) for k in range(3)]
    return dict(residual_d2=np.linalg.norm(d2, axis=1), offset=np.linalg.norm(residual, axis=1),
                camera_pulse=np.linalg.norm(p[2] - 2*p[1] + p[0], axis=1))


def summarize(arrays):
    return dict(residual_d2_rms_px=float(np.sqrt(np.mean(arrays['residual_d2']**2))),
                median_offset_px=float(np.median(arrays['offset'])),
                camera_pulse_rms_px=float(np.sqrt(np.mean(arrays['camera_pulse']**2))))


def gate_camera(RT, K, joints, points, shots, valid, image_size_hw, config=GateConfig()):
    """Return final camera, ungated-policy proposal, and full decision evidence.

    ``shots`` are existing fitted records, with source_frames, shot and scale.
    Only wholly valid, disjoint shots are processed. Short shots/gaps stay exact.
    An optional strict guard retains the baseline on increased person residual.
    The default reports that increase without overriding visual review; neither
    metric is a perceptual quality guarantee, especially after body retargeting.
    """
    R = RT[:, :3, :3].astype(float)
    K = K.astype(float)
    C = -np.einsum('tji,tj->ti', R, RT[:, :3, 3])
    mx, my, target, visible = observations(joints, points, K, image_size_hw)
    evidence = []
    used = np.zeros(len(R), bool)
    for shot in shots:
        a, b = shot['source_frames']
        if b - a < config.minimum_frames:
            continue
        if not valid[a:b].all() or used[a:b].any():
            raise ValueError('Shot includes a gap or overlaps another shot')
        used[a:b] = True
        Rs, Cs = R[a:b], C[a:b]
        ray = np.einsum('tji,tj->ti', Rs, my[a:b])
        distance = np.linalg.norm(ray, axis=1)
        ray /= distance[:, None]
        anchored = mx[a:b] - np.einsum('tji,tj->ti', Rs, my[a:b])
        smooth = [lowpass(x, config.sigma_frames) for x in (Cs, anchored)]
        lateral = []
        for x in smooth:
            x = x - x.mean(0)
            lateral.append(x - (x*ray).sum(1, keepdims=True)*ray)
        v, p = lateral
        corr = float((v*p).sum() / np.sqrt((v*v).sum()*(p*p).sum() + 1e-12))
        travel = float(np.linalg.norm(smooth[0]-np.median(smooth[0], axis=0), axis=1).max()*2)
        trusted = travel > config.slow_travel_m and corr > config.slow_correlation
        evidence.append(dict(shot=shot['shot'], source_frames=[a,b], original_scale=shot['scale'],
            median_person_distance=float(np.median(distance)), scale_over_distance=float(shot['scale']/np.median(distance)),
            slow_travel_m=travel, slow_correlation=corr, trusted=trusted,
            visible_joint_min=int(visible[a:b].sum(1).min()), visible_joint_median=float(np.median(visible[a:b].sum(1))),
            visibility_change_frames=(np.flatnonzero(np.any(visible[a+1:b] != visible[a:b-1],axis=1))+a+1).tolist()))
    ratios = [e['scale_over_distance'] for e in evidence if e['trusted']]
    cap = float(np.percentile(ratios, config.cap_percentile)) if ratios else None
    final, proposal = dict(R=R.copy(),C=C.copy(),K=K.copy()), dict(R=R.copy(),C=C.copy(),K=K.copy())
    aggregates = {name:{k:[] for k in ['residual_d2','offset','camera_pulse']} for name in ['baseline','proposal','final']}
    for e in evidence:
        a,b = e['source_frames']
        make = lambda k: split(R[a:b],C[a:b],K[a:b],mx[a:b],my[a:b],k,config.sigma_frames)
        _, raw_gain, _ = gain_from_person(make(1.),mx[a:b],target[a:b],config.sigma_frames)
        fixed = cap is None and not e['trusted'] and raw_gain < config.supported_fast_gain
        k = 1. if fixed or e['trusted'] or raw_gain >= config.supported_fast_gain else min(1.,cap/e['scale_over_distance'])
        gain, _, corr = gain_from_person(make(k),mx[a:b],target[a:b],config.sigma_frames)
        candidate = compose(make(k),gain)
        if k == 1. and gain == 1.:
            candidate = R[a:b].copy(), C[a:b].copy(), K[a:b].copy()
        if fixed:
            # Without a reliable scale population, retain slow rotation and focal,
            # but anchor one fixed center to the observed person.
            slow_R, _, slow_K = compose(make(1.), 0.)
            anchor = mx[a:b] - np.einsum('tji,tj->ti', slow_R, my[a:b])
            candidate = slow_R, np.tile(np.median(anchor, axis=0), (b-a, 1)), slow_K
            gain = 0.
        e['labels'] = (['trusted'] if e['trusted'] else ['fixed_center'] if fixed else ['clamped'] if k < 1. else ['fast_supported'])
        if e['visible_joint_median'] < 6 or len(e['visibility_change_frames']) > .25*(b-a-1):
            e['labels'].append('low_observation')
        before = metric_arrays(R[a:b],C[a:b],K[a:b],mx[a:b],target[a:b],joints[a:b])
        after = metric_arrays(*candidate,mx[a:b],target[a:b],joints[a:b])
        worse = summarize(after)['residual_d2_rms_px'] > summarize(before)['residual_d2_rms_px'] + 1e-7
        rejected = bool(worse and config.reject_worse_residual)
        for key, value in zip(['R','C','K'],candidate):
            proposal[key][a:b] = value
            if not rejected:final[key][a:b] = value
        for label,arr in [('baseline',before),('proposal',after),('final',before if rejected else after)]:
            for key in arr:aggregates[label][key].append(arr[key])
        e.update(proposed_scale_ratio=k, proposed_scale=e['original_scale']*k, proposed_gain=gain,
                 full_scale_raw_gain=raw_gain, fast_correlation=corr, kept_baseline=rejected,
                 person_residual_worse=bool(worse),
                 decision='retain_baseline_person_residual_worse' if rejected else 'apply_shared_gate',
                 baseline=summarize(before),proposal=summarize(after),final=summarize(before if rejected else after))
    def export(v):
        rt=RT.copy();rt[used,:3,:3]=v['R'][used];rt[used,:3,3]=-np.einsum('tij,tj->ti',v['R'][used],v['C'][used])
        # Preserve untouched shots byte-for-byte as well as geometrically.
        for e in evidence:
            if (e['kept_baseline'] and v is final) or (e['proposed_scale_ratio']==1. and e['proposed_gain']==1.):
                a,b=e['source_frames'];rt[a:b]=RT[a:b]
        return rt,v['K']
    report=dict(cap_scale_over_distance=cap,trusted_shots=[e['shot'] for e in evidence if e['trusted']],
                shots=evidence,aggregate={label:summarize({k:np.concatenate(v) for k,v in arr.items()}) for label,arr in aggregates.items()} if evidence else {},
                unprocessed_frames=np.flatnonzero(~used).tolist(),pixel_coordinate_size=list(image_size_hw),
                metric_limitations=['SAM estimates are not ground truth','projected weighted 3D centroid versus weighted 2D centroid',
                                    'visibility-set changes affect centroids','small non-visible weights match Fable reference'])
    return export(final),export(proposal),report
