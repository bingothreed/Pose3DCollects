"""Refine observed wrists on a fixed-length arm using the source camera rays."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .camera import project
from .timeline import intervals


ARM_CHAINS = ((16, 18, 20, 5, 7, 62), (17, 19, 21, 6, 8, 41))
MAX_WRIST_GAP = 3
# At 30 FPS, limit the extra rotation introduced by the correction, not the underlying motion.
ARM_UPDATE_STEP = np.deg2rad(12.)
ARM_BLEND_STEP = np.deg2rad(6.)
# Budget for the correction's own second difference at 30 FPS, in arm lengths.
WRIST_CORRECTION_ACCEL = .055


def motion_skeleton(motion, model_path):
    """Read the same shaped rest joints used by SMPL-H preview and export."""
    from flowhmr.core.bodymodels.smplh_data import load_smplh_data

    model = load_smplh_data(model_path)
    betas = np.asarray(motion['betas']).reshape(-1)
    shape = np.asarray(model['shapedirs'])
    count = min(shape.shape[-1],len(betas))
    rest = np.asarray(model['J_regressor']) @ (model['v_template'] + shape[:,:,:count] @ betas[:count])
    poses = np.asarray(motion['poses'])
    parents = np.asarray(model['kintree_table'][0],dtype=int).copy()
    parents[0] = -1
    local = Rotation.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(*poses.shape[:2],3,3)
    absolute = local.copy()
    joints = np.empty_like(poses)
    joints[:,0] = motion['trans']+rest[0]
    for i, parent in enumerate(parents[1:],1):
        absolute[:,i] = absolute[:,parent]@local[:,i]
        joints[:,i] = joints[:,parent]+absolute[:,parent]@(rest[i]-rest[parent])
    return joints,absolute,parents


def _unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-9)


def _swing(before, after):
    a,b = _unit(before),_unit(after)
    axis = np.cross(a,b)
    dot = np.einsum('ij,ij->i',a,b)
    opposite = dot < -.999
    if np.any(opposite):
        alternate = np.eye(3)[np.argmin(np.abs(a[opposite]),axis=1)]
        axis[opposite] = _unit(np.cross(a[opposite],alternate))
    return Rotation.from_quat(np.column_stack([axis,np.maximum(1.+dot,0.)])).as_matrix()


def _arm_pixels(intrinsics, evidence):
    projected = np.einsum('tij,tkj->tki', intrinsics, evidence)
    with np.errstate(invalid='ignore', divide='ignore'):
        return projected[:, :, :2] / projected[:, :, 2:3]


def _bridge_arm_observations(evidence, observed, measured, camera_valid, intrinsics, span):
    """Reject an isolated wrong hand and bridge short gaps in a single shot."""
    measured = measured.copy()
    samples = np.flatnonzero(measured)
    previous, current, following = samples[:-2], samples[1:-1], samples[2:]
    fraction = ((current - previous) / (following - previous))[:, None]
    travel = observed[following, 2] - observed[previous, 2]
    expected = observed[previous, 2] + fraction * travel
    deviation = np.linalg.norm(observed[current, 2] - expected, axis=1)
    distances = np.minimum(np.linalg.norm(observed[current, 2] - observed[previous, 2], axis=1),
                           np.linalg.norm(observed[current, 2] - observed[following, 2], axis=1))
    outlier = ((current - previous <= MAX_WRIST_GAP) & (following - current <= MAX_WRIST_GAP)
               & (distances > .30 * span[current])
               & (deviation > np.maximum(.42 * span[current], .65 * np.linalg.norm(travel, axis=1))))
    measured[current[outlier]] = False
    for start, end in intervals(~measured):
        if not 0 < start < end < len(measured) or end-start > MAX_WRIST_GAP or not camera_valid[start:end].all():
            continue
        travel = np.linalg.norm(observed[end, 2] - observed[start-1, 2])
        if travel > .45 * (end-start+1) * np.median(span[start-1:end+1]):
            continue
        fraction = np.arange(1, end-start+1)[:, None, None] / (end-start+1)
        points = evidence[start-1] + fraction * (evidence[end] - evidence[start-1])
        evidence[start:end] = points
        observed[start:end] = _arm_pixels(intrinsics[start:end], points)
        measured[start:end] = True
    return measured


def _continuous_wrist_observations(observed, predicted, measured, span):
    """Break a shot's wrist track at an isolated step unsupported by its motion."""
    if len(measured) < 3:
        return measured
    pair = measured[:-1] & measured[1:]
    scale = np.maximum((span[1:] + span[:-1]) / 2, 1.)
    step = np.diff(observed[:, 2] - observed[:, 0], axis=0) / scale[:, None]
    expected = np.diff(predicted[:, 2] - predicted[:, 0], axis=0) / scale[:, None]
    speed = np.where(pair, np.linalg.norm(step, axis=1), 0.)
    window = np.lib.stride_tricks.sliding_window_view(np.pad(speed, 2), 5)
    neighbors = np.median(window[:, [0, 1, 3, 4]], axis=1)
    jump = (pair & (speed > .22) & (speed > 2.5 * neighbors + .05)
            & (speed > 1.5 * np.linalg.norm(expected, axis=1) + .08)
            & (np.linalg.norm(step - expected, axis=1) > .18))
    # Keep the underlying pose on either side of an ambiguous handoff.
    accepted = measured.copy()
    accepted[:-1] &= ~jump
    accepted[1:] &= ~jump
    return accepted


def _bounded_arm_weights(delta, accepted, limit=1.):
    """Keep correction rotations continuous while leaving smooth proposals unchanged."""
    delta = np.where(accepted[:, None, None], delta, 0.)
    amount = np.linalg.norm(delta, axis=-1).max(axis=1)
    turn = np.linalg.norm(np.diff(delta, axis=0), axis=-1).max(axis=1)
    cap = np.minimum(1., ARM_UPDATE_STEP / np.maximum(turn, 1e-8))
    slope = ARM_BLEND_STEP / np.maximum(np.maximum(amount[:-1], amount[1:]), 1e-8)
    weights = np.where(accepted, limit, 0.)
    weights[:-1] = np.minimum(weights[:-1], cap)
    weights[1:] = np.minimum(weights[1:], cap)
    # A pair is bounded by the turn of the shared correction plus the change in blend weight.
    for frame, step in enumerate(slope, 1):
        weights[frame] = min(weights[frame], weights[frame-1] + step)
    for frame in range(len(slope)-1, -1, -1):
        weights[frame] = min(weights[frame], weights[frame+1] + slope[frame])
    return weights


def _continuous_arm_solution(local, delta, accepted, joints, world, parent, camera, observed, error, shot_ids, check):
    """Accept a wrist fit only when it is spatially and temporally consistent."""
    bone = np.einsum('tnji,tnj->tni', world, np.diff(joints, axis=1))
    budget = WRIST_CORRECTION_ACCEL * np.linalg.norm(bone[1:-1], axis=-1).sum(axis=1)
    within = ((shot_ids[1:-1] >= 0) & (shot_ids[:-2] == shot_ids[1:-1]) & (shot_ids[1:-1] == shot_ids[2:]))
    accepted = accepted.copy()
    limit = np.ones(len(joints))
    displacement = np.zeros((len(joints), 3))
    while accepted.any():
        check()
        weight = _bounded_arm_weights(delta, accepted, limit)
        frames = np.flatnonzero(weight > 0)
        scaled = delta[frames] * weight[frames, None, None]
        rotations = Rotation.from_rotvec(scaled.reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3) @ local[frames]
        upper = parent[frames] @ rotations[:, 0]
        lower = upper @ rotations[:, 1]
        elbow = joints[frames, 0] + np.einsum('tij,tj->ti', upper, bone[frames, 0])
        wrist = elbow + np.einsum('tij,tj->ti', lower, bone[frames, 1])
        uv, depth = project(camera.RT[frames], camera.K[frames], wrist[:, None])
        after = np.linalg.norm(uv[:, 0] - observed[frames], axis=1)
        good = (depth[:, 0] > .05) & (after <= error[frames] + 1e-4)
        if not good.all():
            accepted[frames[~good]] = False
            continue
        displacement.fill(0.)
        displacement[frames] = wrist - joints[frames, 2]
        accel = np.linalg.norm(np.diff(displacement, n=2, axis=0), axis=1)
        spikes = np.flatnonzero(within & (accel > budget + 1e-6))
        if not len(spikes):
            return frames, rotations, after

        # Reduce the offending three-frame update together and taper its edges.
        # Test the next prediction with forward kinematics before saving it.
        ratio = .8 * budget[spikes] / accel[spikes]
        centers = spikes + 1
        taper = np.ones(len(joints))
        for offset, power in ((-3,.25), (-2,.5), (-1,1.), (0,1.), (1,1.), (2,.5), (3,.25)):
            indices = np.clip(centers + offset, 0, len(joints)-1)
            nearby = shot_ids[indices] == shot_ids[centers]
            np.minimum.at(taper, indices[nearby], ratio[nearby] ** power)
        limit *= taper
        accepted &= limit >= .005
    return np.empty(0, dtype=int), np.empty((0, 2, 3, 3)), np.empty(0)


def _target_arm(joints, observed, pixels, weight, rt, intrinsics, span):
    shoulder, elbow, wrist = np.moveaxis(joints, 1, 0)
    source_shoulder, source_elbow, source_wrist = np.moveaxis(observed, 1, 0)
    upper = np.linalg.norm(elbow - shoulder, axis=1)
    lower = np.linalg.norm(wrist - elbow, axis=1)
    source_length = (np.linalg.norm(source_elbow - source_shoulder, axis=1)
                     + np.linalg.norm(source_wrist - source_elbow, axis=1))
    size = (upper+lower)/np.maximum(source_length,1e-6)
    rotations = rt[:,:3,:3]
    camera_pos = -np.einsum('tji,tj->ti',rotations,rt[:,:3,3])
    rays = np.linalg.solve(intrinsics, np.column_stack([pixels[:, 2], np.ones(len(pixels))])[..., None])[:, :, 0]
    depth_delta = (source_wrist[:,2]-source_shoulder[:,2])*size
    depth = np.einsum('ti,ti->t',rotations[:,2],shoulder)+rt[:,2,3]
    current_depth = np.einsum('ti,ti->t',rotations[:,2],wrist)+rt[:,2,3]
    z = np.maximum(.1,.65*(depth+depth_delta)+.35*current_depth)
    direction = _unit(np.einsum('tji,tj->ti',rotations,rays))
    along = z*np.linalg.norm(rays,axis=1)
    center = np.einsum('ti,ti->t',direction,shoulder-camera_pos)
    nearest = camera_pos+center[:,None]*direction
    reach = .995*(upper+lower)
    discriminant = reach**2-np.sum((nearest-shoulder)**2,axis=1)
    radius = np.sqrt(np.maximum(discriminant,0.))
    ray_end = camera_pos+np.clip(along,center-radius,center+radius)[:,None]*direction
    endpoint = wrist + (ray_end - wrist) * weight[:, None]
    speed = endpoint-shoulder
    axis = _unit(speed)
    distance = np.clip(np.linalg.norm(speed,axis=1),np.abs(upper-lower)+.003,reach)
    wrist_target = shoulder+axis*distance[:,None]
    elbow_forward = (upper**2 - lower**2 + distance**2)/(2*distance)
    bend = np.sqrt(np.maximum(upper**2-elbow_forward**2,0.))
    observed_elbow = np.einsum('tji,tj->ti',rotations,(source_elbow-source_shoulder)*size[:,None])
    current = elbow-shoulder
    # Use the observed elbow pole increasingly as the wrist requires more correction.
    hint = current + (observed_elbow-current)*weight[:,None]
    perpendicular = hint-np.einsum('ti,ti->t',hint,axis)[:,None]*axis
    degenerate = np.linalg.norm(perpendicular,axis=1) < 1e-6
    if np.any(degenerate):
        perpendicular[degenerate] = np.cross(axis[degenerate],np.eye(3)[np.argmin(np.abs(axis[degenerate]),axis=1)])
    first = _unit(perpendicular)
    second = np.cross(axis, first)
    center = shoulder + axis * elbow_forward[:, None]
    # Search on the elbow circle so the hand stays on its ray and both bone
    # lengths stay fixed. Keep some of the observed 3D bend when depth is ambiguous.
    angles = np.radians(np.arange(-105, 106, 7))
    poles = first[:, None] * np.cos(angles)[None, :, None] + second[:, None] * np.sin(angles)[None, :, None]
    candidates = center[:, None] + poles * bend[:, None, None]
    uv, depth = project(rt, intrinsics, candidates)
    cost = np.linalg.norm(uv - pixels[:, 1:2], axis=2) / np.maximum(span[:, None], 1.)
    cost = cost**2 + (.095 * angles[None, :] / np.maximum(weight[:, None], .08))**2
    choice = np.argmin(np.where(depth > .05, cost, np.inf), axis=1)
    angle = gaussian_filter1d(angles[choice], .85, mode='nearest')
    elbow_target = center + (first * np.cos(angle)[:, None] + second * np.sin(angle)[:, None]) * bend[:, None]
    return elbow_target,wrist_target


def correct_wrists(poses, joints, absolute, parents, points, original_K, camera, support, shots, check=lambda: None):
    """Correct wrists using coherent shot evidence and bound the added arm motion."""
    output = poses.copy()
    adjusted = np.zeros((len(poses),2),bool)
    source_error = np.full((len(poses),2),np.nan)
    final_error = source_error.copy()
    h,w = camera.image_size_hw
    shot_ids = np.full(len(poses), -1, dtype=int)
    for index, shot in enumerate(shots):
        shot_ids[shot['start']:shot['end']] = index
    for side,(a,b,c,x,y,z) in enumerate(ARM_CHAINS):
        chain = joints[:,[a,b,c]]
        evidence = points[:,[x,y,z]]
        local = Rotation.from_rotvec(poses[:, [a,b]].reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3)
        delta = np.zeros((len(poses), 2, 3))
        accepted = np.zeros(len(poses), dtype=bool)
        pred, depths = project(camera.RT,camera.K,chain)
        observed = _arm_pixels(original_K, evidence)
        valid = (support & camera.camera_valid & np.isfinite(observed).all((1,2))
                 & (evidence[:,:,2]>.1).all(1) & (depths>.1).all(1))
        valid &= ((observed[:,2]>[1,1])&(observed[:,2]<[w-1,h-1])).all(1)
        length = np.linalg.norm(chain[:, 1] - chain[:, 0], axis=1) + np.linalg.norm(chain[:, 2] - chain[:, 1], axis=1)
        span = length * camera.K[:, 0, 0] / np.maximum(depths[:, 0], .1)
        shoulder_error = np.linalg.norm(pred[:, 0] - observed[:, 0], axis=1)
        valid &= shoulder_error < 1.1 * span
        valid &= np.linalg.norm(evidence[:, 1] - evidence[:, 0], axis=1) > .05
        valid &= np.linalg.norm(evidence[:, 2] - evidence[:, 1], axis=1) > .05
        for shot in shots:
            check()
            lo, hi = shot['start'], shot['end']
            measured = _bridge_arm_observations(evidence[lo:hi], observed[lo:hi], valid[lo:hi],
                camera.camera_valid[lo:hi], original_K[lo:hi], span[lo:hi])
            measured = _continuous_wrist_observations(observed[lo:hi], pred[lo:hi], measured, span[lo:hi])
            error = np.linalg.norm(pred[lo:hi, 2] - observed[lo:hi, 2], axis=1)
            # Weight measurements by local shoulder support. Close views can
            # show meaningful wrist errors well below a tenth of the arm span.
            local_span = np.maximum(span[lo:hi], 1.)
            shoulder_support = np.clip((.45 - shoulder_error[lo:hi] / local_span) / .25, 0., 1.)
            weight = np.clip((error / local_span - .04) / .18, 0., 1.)
            weight = .98 * weight * weight * (3 - 2 * weight) * shoulder_support
            for start,end in intervals(measured):
                if end-start < 2:
                    continue
                frames = np.arange(lo+start,lo+end)
                strength = gaussian_filter1d(weight[start:end], 1.15, mode='nearest')
                fade = np.arange(1, len(frames)+1) / 3
                if start > 0:
                    strength *= np.minimum(fade, 1.)
                if end < hi-lo:
                    strength *= np.minimum(fade[::-1], 1.)
                ids = strength > .04
                frames,strength = frames[ids],strength[ids]
                if not len(frames):
                    continue
                C,E,W = np.moveaxis(chain[frames],1,0)
                elbow,wrist = _target_arm(chain[frames], evidence[frames], observed[frames], strength,
                                          camera.RT[frames], camera.K[frames], span[frames])
                uv,_ = project(camera.RT[frames],camera.K[frames],wrist[:,None])
                after = np.linalg.norm(uv[:,0]-observed[frames,2],axis=1)
                good = after+.2 < error[frames-lo]
                frames = frames[good]
                if not len(frames):
                    continue
                C,E,W,elbow,wrist = (p[good] for p in (C,E,W,elbow,wrist))
                first = _swing(E-C,elbow-C)
                second = _swing((first@(W-E)[...,None])[:,:,0],wrist-elbow)
                upper = first @ absolute[frames,a]
                lower = second @ first @ absolute[frames,b]
                top = absolute[frames,parents[a]].transpose(0,2,1) @ upper
                bottom = upper.transpose(0,2,1) @ lower
                relative = np.stack([top, bottom], axis=1) @ local[frames].transpose(0, 1, 3, 2)
                delta[frames] = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 2, 3)
                accepted[frames] = True
        error = np.linalg.norm(pred[:, 2] - observed[:, 2], axis=1)
        frames, rotations, after = _continuous_arm_solution(
            local, delta, accepted, chain, absolute[:, [a,b]], absolute[:, parents[a]],
            camera, observed[:, 2], error, shot_ids, check)
        if len(frames):
            values = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 2, 3)
            output[np.ix_(frames, [a,b])] = values
            adjusted[frames,side] = True
            source_error[frames,side] = error[frames]
            final_error[frames,side] = after
    return output,adjusted,source_error,final_error


def refine_linked_motion(directory, result, job, engine, *, check, notify):
    """Refine the final generated arms with the existing shot and SAM observations."""
    from pathlib import Path
    from .background_camera import complete_cutaway_cameras
    from .motion_condition import track_shots
    from .recovery import recover_camera
    from .runtime import _short_camera_continuity, digest, measure_gravity, save_npz
    from .shot_detection import camera_shots
    from .timeline import validity
    from ..utils.runtime_torso_optimization import save_composed_motion

    directory = Path(directory)
    task = directory / 'input'
    params = job['params']
    camera_name = 'static' if params.get('static_camera') else 'vggt'
    feature_name = 'input_sam3d_feat_static' if camera_name == 'static' else 'input_sam3d_feat'
    with np.load(task / 'input_bbox.npz', allow_pickle=False) as bbox:
        first, last = map(int, bbox['start_end'])
        boxes = bbox['bbox']
        detected = bbox['observed_frames']
        shots = track_shots(bbox.get('shot_ranges'), first, last)
    if len(shots) < 2:
        return None
    with np.load(task / f'input_{camera_name}_camera.npz', allow_pickle=False) as raw:
        rt, K, dimensions = raw['camera_RT'], raw['K'], raw['image_size_hw']
        if not np.array_equal(raw['frame_range'], [first, last]):
            raise ValueError('手腕校正的相机与动作范围不一致。')
    with np.load(task / f'{feature_name}.pt.observations.npz', allow_pickle=False) as obs:
        points = obs['points']
        if (not np.array_equal(obs['boxes'], boxes) or not np.array_equal(obs['K'], K)
                or not np.array_equal(obs['source_frames'], np.arange(first, last+1))):
            raise ValueError('手腕校正的人物观测与动作范围不一致。')
    with np.load(result + '.npz', allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    joints, rotations, parents = motion_skeleton(motion, engine.smpl_model_path)
    valid = validity(motion.get('valid_frames'), len(joints))
    direct = valid & np.isin(np.arange(first, last+1), detected)
    if not direct.any():
        return None
    folder = directory / 'camera'
    folder.mkdir(exist_ok=True)
    device = engine.device if engine.device.type != 'xpu' else 'cpu'
    video = task / 'input_30fps.mp4'
    _, textured = camera_shots(video, first, len(joints), check)
    continuity = _short_camera_continuity(video, first, valid, direct, check)
    gravity = measure_gravity(video, rt, K, shots, first, folder, device, check, notify,
                              diagnostics={}, observation_support=direct if not direct.all() else None,
                              person_boxes=boxes, release_cuda=engine._release_camera_cuda)
    check()
    def register(camera, ungated, eligible):
        return complete_cutaway_cameras(video, camera, rt, boxes, eligible, textured, check,
                                        ungated=ungated)

    camera, report = recover_camera(rt, K, joints, points, valid, shots, dimensions,
                                    gravity=gravity, track_start=first, observation_support=direct,
                                    image_continuity=continuity, shot_images=textured, shot_edge_limit=30,
                                    register_cutaways=register, check=check)
    # The background solution determines the viewpoint. Observations from a
    # rejected body fit still need independent support before moving an arm.
    arm_support = direct.copy()
    for row in report['shots']:
        for lo, hi in row.get('unsupported_observation_ranges', ()):
            arm_support[lo:hi] = False
    corrected, mask, before, after = correct_wrists(motion['poses'], joints, rotations, parents, points, K,
                                                    camera, arm_support, shots, check)
    if not mask.any():
        return dict(status='no_supported_correction', adjusted_frames=[0, 0])
    check()
    save_composed_motion({**motion, 'poses': corrected}, result, engine.smpl_model_path)
    # The camera fit uses the original arm positions.
    save_npz(task / 'camera_arm_reference.npz', joints=joints[:, 18:22],
             motion_sha256=digest(Path(result + '.npz')))
    return dict(status='applied', adjusted_frames=mask.sum(axis=0).tolist(),
                before_median_px=float(np.median(before[mask])), after_median_px=float(np.median(after[mask])))
