"""Condition one continuous action on separately estimated source shots."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .timeline import intervals


def track_shots(ranges, first, last):
    """Clip absolute half-open shot ranges to the inclusive selected motion."""
    if ranges is None:
        return []
    blocks = [(max(a, first)-first, min(b, last+1)-first)
              for a,b in ranges if a <= last and b > first]
    return [dict(id=i, start=int(a), end=int(b)) for i,(a,b) in enumerate(blocks)]


def observed_motion_frames(observed, first, count):
    """Small detector misses use boxes from that shot. Longer ones have no pose evidence."""
    observed = np.isin(np.arange(first, first+count), observed)
    usable = np.ones(count, dtype=bool)
    for a,b in intervals(~observed):
        if b-a > 9:
            usable[a:b] = False
    return observed, usable


def _heading(rotations):
    return np.arctan2(rotations[:,0,2]-rotations[:,2,0], rotations[:,0,0]+rotations[:,2,2])


def _level(up):
    up = up / np.linalg.norm(up)
    forward = np.array([0.,0.,1.])
    if abs(forward @ up) > .98:
        forward = np.array([1.,0.,0.])
    forward -= (forward @ up) * up
    forward /= np.linalg.norm(forward)
    return np.stack([np.cross(up, forward), up, forward], axis=1)


def _heading_at_cut(headings, frames, cut):
    if len(headings) == 1:
        return headings[0]
    times = frames - cut
    return np.linalg.lstsq(np.column_stack((times, np.ones(len(times)))), headings, rcond=None)[0][1]


def joined_motion_camera(rt, body_orientation, observed, usable, shots, *, fit_cut_velocity=False):
    """Use the observed actor's heading to place each shot in the same world."""
    out = rt.copy()
    previous = None
    for shot in shots:
        a,b = shot['start'], shot['end']
        direct = observed[a:b] & np.isfinite(body_orientation[a:b]).all((1,2))
        if not direct.any():
            continue
        raw = Rotation.from_matrix(rt[a:b,:3,:3]).as_matrix()
        relative = raw @ raw[0].T
        headings = relative[direct].transpose(0,2,1) @ body_orientation[a:b][direct]
        up = (np.mean if fit_cut_velocity else np.median)(headings[:,:,1], axis=0)
        if np.linalg.norm(up) < .2:
            up = np.array([0.,-1.,0.])
        reference = _level(up)
        yaw = np.unwrap(_heading(reference.T @ headings))
        frames = a + np.flatnonzero(direct)
        if previous is not None:
            old_frames, old_yaw = previous
            if not fit_cut_velocity:
                correction = np.median(yaw[:6]) - np.median(old_yaw)
            elif frames[0] != old_frames[-1]+1:
                correction = yaw[0] - old_yaw[-1]
            else:
                # Compare both estimates at the same transition time. Averaging
                # each side at different times biases the alignment of turns.
                cut = (frames[0] + old_frames[-1])/2
                correction = (_heading_at_cut(yaw[:6], frames[:6], cut)
                              - _heading_at_cut(old_yaw, old_frames, cut))
            reference = reference @ Rotation.from_rotvec([0.,correction,0.]).as_matrix()
            yaw -= correction
        previous = frames[-6:], yaw[-6:]
        out[a:b,:3,:3] = relative @ reference
    for a,b in intervals(~usable):
        if a and b < len(out):
            endpoints = Rotation.from_matrix(out[[a-1,b],:3,:3])
            out[a:b,:3,:3] = Slerp([a-1,b], endpoints)(np.arange(a,b)).as_matrix()
        elif a:
            out[a:b,:3,:3] = out[a-1,:3,:3]
        elif b < len(out):
            out[a:b,:3,:3] = out[b,:3,:3]
    return out


def fill_hidden_features(tokens, usable):
    """Bridge short offscreen intervals using the adjacent observed action."""
    for a,b in intervals(~usable):
        if a and b < len(tokens):
            for frame in range(a,b):
                fraction = (frame-a+1)/(b-a+1)
                tokens[frame] = tokens[a-1].lerp(tokens[b], fraction)
        elif a:
            tokens[a:b] = tokens[a-1]
        elif b < len(tokens):
            tokens[a:b] = tokens[b]
