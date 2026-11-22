"""Body-lean continuity across cuts, with a five-second leveling window.

Only camera gravity priors change. Person gaps reset continuity. The final
camera still uses one fixed similarity per shot, not per-frame translations.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from .timeline import intervals


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def horizontal_swing(source, target):
    delta = target-source
    axis = unit(np.stack([delta[...,2], np.zeros(delta.shape[:-1]), -delta[...,0]], -1))
    sine = np.sum(axis*np.cross(source,target), axis=-1)
    cosine = np.sum(source*target,axis=-1) - np.sum(axis*source,axis=-1)*np.sum(axis*target,axis=-1)
    return Rotation.from_rotvec(axis*np.arctan2(sine,cosine)[...,None]).as_matrix()


def body_gravity(rt, points, segments, initials, *, fps=30):
    n = len(rt)
    up = np.zeros((n,3))
    usable = np.zeros(n,bool)
    for a,b,_ in segments:
        if (a,b) not in initials:
            continue
        camera_up = unit(points[a:b,[5,6]].mean(1) - points[a:b,[9,10]].mean(1))
        condition = rt[a:b,:3,:3] @ initials[a,b].T
        up[a:b] = unit(np.einsum('tji,tj->ti',condition,camera_up))
        usable[a:b] = np.isfinite(up[a:b]).all(1) & (up[a:b,1] > .1)
    lean = np.arctan2(up[:,[0,2]],up[:,1,None])
    accumulated = lean.copy()
    target = up.copy()
    cuts = []
    for start,stop in intervals(usable):
        previous = None
        for a,b,_ in segments:
            a,b = max(a,start),min(b,stop)
            if a>=b:
                continue
            count=min(6,b-a)
            offset = np.zeros(2) if previous is None else accumulated[max(previous,a-6):a].mean(0)-lean[a:a+count].mean(0)
            accumulated[a:b] = lean[a:b]+offset
            if previous is not None:
                cuts.append(a)
            previous=a
        slow=gaussian_filter1d(accumulated[start:stop],5*fps,axis=0,mode='nearest',truncate=1.)
        revised=accumulated[start:stop]-slow
        target[start:stop]=unit(np.stack([np.tan(revised[:,0]),np.ones(stop-start),np.tan(revised[:,1])],-1))
    result={}
    for a,b,_ in segments:
        if (a,b) not in initials:
            continue
        Q=initials[a,b]
        mask=usable[a:b]
        if mask.any():
            adjustment=horizontal_swing(up[a:b][mask],target[a:b][mask])
            directions=np.einsum('tji,j->ti',adjustment @ Q, np.array([0.,1.,0.]))
            result[a,b]=unit(np.median(directions,axis=0))
        else:
            result[a,b]=Q.T @ np.array([0.,1.,0.])
    return result, dict(sigma_seconds=5,support_seconds_each_side=5,edge_frames=6,
                        continuous_cut_frames=cuts,reset_at_person_gaps=True,
                        applied_to_motion=False)
