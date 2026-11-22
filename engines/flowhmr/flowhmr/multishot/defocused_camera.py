"""Recover a defocused, near-stationary cutaway from its common background."""
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .background_camera import _landmarks, _project


def register_defocused_cutaway(images, frames, references, cameras, boxes, raw, check):
    """Register visible landscape and verify the resulting camera on every frame."""
    relative = raw[:,:3,:3] @ raw[0,:3,:3].T
    if Rotation.from_matrix(relative).magnitude().max() > np.deg2rad(1.5):
        return None
    source_h, source_w = cameras.image_size_hw
    scale = min(1.,480/max(source_w,source_h))
    size = (int(round(source_w*scale)), int(round(source_h*scale)))
    w,h = size
    axes = np.array([w/source_w,h/source_h])
    K = cameras.K.astype(float).copy()
    K[:,:2] *= axes[None,:,None]
    gray = {f: cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),size,interpolation=cv2.INTER_AREA)
            for f,im in images.items()}
    soft = np.array([cv2.GaussianBlur(gray[f],(0,0),1.4) for f in frames])
    target_mask = np.std(soft,axis=0) < 18
    target = np.uint8(np.median([gray[f] for f in frames],axis=0))
    detector = cv2.SIFT_create(nfeatures=2200,contrastThreshold=.006)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    inferences = []
    for group in references:
        pool = []
        views = {}
        for reference in group:
            if reference not in gray:
                continue
            check()
            row = _reference_view(gray[reference],target,boxes[reference],axes,K[reference],K[frames[0]],detector)
            if row is not None:
                views[reference] = row
        for a,b in combinations(views,2):
            check()
            if b-a < 7:
                continue
            # Try both tracking directions, since one reference may expose
            # more of the background around the detected actor.
            for first,second in ((a,b),(b,a)):
                warp, texture = views[first]
                world,_ = _landmarks(texture,views[second][1],cameras.RT[first],cameras.RT[second],matcher)
                if len(world) < 24:
                    continue
                uv, _ = _project(world,K[first]@cameras.RT[first,:3])
                pool.extend(_match_broad_landmarks(gray[first],target,target_mask,warp,world,uv,first,second))
        if not pool:
            continue
        unique = {}
        for xyz,uv,error,first,second in sorted(pool,key=lambda row:row[2]):
            unique.setdefault(tuple(np.rint(uv/2.5).astype(int)),(xyz,uv,first,second))
        world,pixels,lo,hi = map(np.asarray,zip(*unique.values()))
        estimate = _verified_pose(world,pixels,gray,target,frames,K,relative,axes,check)
        if estimate is not None:
            poses, keep, count, median, worst = estimate
            used = np.unique(np.r_[lo[keep],hi[keep]])
            inferences.append((worst,poses,dict(
                method='defocused_background_registration',
                reference_source_frames=(used+cameras.track_start).tolist(),
                minimum_background_inliers=count, median_reprojection_px=median,
                worst_frame_p90_reprojection_px=worst)))
    if not inferences:
        return None
    _, poses, diagnostics = min(inferences,key=lambda result:result[0])
    return poses, diagnostics


def _reference_view(gray,target,box,axes,reference_K,target_K,detector):
    h,w = gray.shape
    mask = np.full_like(gray,255)
    # Title cards and watermarks at the image edges do not move with scenery.
    mask[:int(.075*h)] = 0
    mask[-int(.03*h):] = 0
    x1,y1,x2,y2 = np.rint(np.asarray(box)*np.tile(axes,2)).astype(int)
    mask[max(0,y1-5):y2+5,max(0,x1-5):x2+5] = 0
    if np.count_nonzero(mask) < .2*h*w:
        return None
    initial = (reference_K@np.linalg.inv(target_K)).astype(np.float32)[:2].copy()
    try:
        score, warp = cv2.findTransformECC(
            cv2.GaussianBlur(target,(0,0),2.),cv2.GaussianBlur(gray,(0,0),2.3),initial,
            cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),mask,5)
    except cv2.error:
        return None
    if score < .68 or np.linalg.det(warp[:,:2]) <= .2:
        return None
    keypoints, desc = detector.detectAndCompute(gray,mask)
    if desc is None or len(desc) < 24:
        return None
    return warp, (np.asarray([p.pt for p in keypoints],float),desc,reference_K,gray.shape,1.)


def _match_broad_landmarks(reference,target,mask,warp,world,uv,first,second):
    h,w = target.shape
    guess = cv2.transform(uv[None],cv2.invertAffineTransform(warp))[0]
    good = ((guess[:,0]>12) & (guess[:,0]<w-12) & (guess[:,1]>.14*h) & (guess[:,1]<h-12))
    grid = np.rint(guess[good]).astype(int)
    good[np.flatnonzero(good)[~mask[grid[:,1],grid[:,0]]]] = False
    if np.count_nonzero(good) < 20:
        return []
    initial = guess[good].astype(np.float32).reshape(-1,1,2)
    image = cv2.warpAffine(cv2.GaussianBlur(reference,(0,0),1.7),warp,(w,h),
                           flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    target = cv2.GaussianBlur(target,(0,0),1.1)
    opts = dict(winSize=(26,26),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,.02))
    matched, visible, error = cv2.calcOpticalFlowPyrLK(image,target,initial,initial.copy(),**opts)
    back, returned, _ = cv2.calcOpticalFlowPyrLK(target,image,matched,initial.copy(),**opts)
    keep = ((visible[:,0]>0) & (returned[:,0]>0) & (error[:,0]<18)
            & (np.linalg.norm(back[:,0]-initial[:,0],axis=1)<2)
            & (np.linalg.norm(matched[:,0]-initial[:,0],axis=1)<35)
            & (matched[:,0,1]>.14*h))
    return [(xyz,uv,float(error),first,second)
            for xyz,uv,error in zip(world[good][keep],matched[keep,0],error[keep,0])]


def _verified_pose(world,pixels,gray,target,frames,K,relative,axes,check):
    if len(world) < 24:
        return None
    anchor = K[frames[0]]
    yes, rvec, trans, inliers = cv2.solvePnPRansac(
        world,pixels,anchor,None,iterationsCount=7000,reprojectionError=3.5,
        confidence=.9999,flags=cv2.SOLVEPNP_EPNP)
    if not yes or len(inliers)<max(24,.28*len(world)):
        return None
    keep = inliers[:,0]
    world, pixels = world[keep], pixels[keep]
    area = np.prod(np.ptp(pixels,axis=0))/np.prod(target.shape)
    if area < .14:
        return None
    source = cv2.GaussianBlur(target,(0,0),1.1)
    observations, support = [], []
    for f in frames:
        check()
        current = cv2.GaussianBlur(gray[f],(0,0),1.1)
        matches, tracked, error = cv2.calcOpticalFlowPyrLK(
            source,current,pixels[:,None].astype(np.float32),None,winSize=(21,21),maxLevel=2)
        measured = matches[:,0]
        good = (tracked[:,0]>0) & (error[:,0]<18) & (np.linalg.norm(measured-pixels,axis=1)<4)
        if good.sum() < max(20,len(world)*.65):
            return None
        observations.append(measured)
        support.append(good)
    observed,support = np.asarray(observations),np.asarray(support)
    # Estimate the anchor with alternating frames and validate on the other
    # frames too. The median image alone can hide motion of a foreground prop.
    validation = []
    homogeneous = np.dstack((observed,np.ones(support.shape)))
    camera_rays = np.einsum('tij,tkj->tki',np.linalg.inv(K[frames]),homogeneous)
    for changes in (relative,np.broadcast_to(np.eye(3),relative.shape)):
        rays = np.einsum('tji,tkj->tki',changes,camera_rays) @ anchor.T
        pixels_aligned = rays[:,:,:2]/rays[:,:,2:3]
        indices = np.flatnonzero(support[::2].all(axis=0))
        if len(indices) < 16:
            continue
        centers = np.median(pixels_aligned[::2,indices],axis=0)
        r,t = cv2.solvePnPRefineLM(world[indices],centers,anchor,None,rvec.copy(),trans.copy())
        rotation = cv2.Rodrigues(r)[0]
        center = -rotation.T@t[:,0]
        poses = np.tile(np.eye(4),(len(frames),1,1))
        poses[:,:3,:3] = changes @ rotation
        poses[:,:3,3] = -poses[:,:3,:3]@center
        errors = []
        for i,f in enumerate(frames):
            prediction, depth = _project(world[support[i]],K[f]@poses[i,:3])
            if np.any(depth <= .05):
                break
            errors.append(np.linalg.norm((prediction-observed[i,support[i]])/axes,axis=1))
        if len(errors) == len(frames):
            worst = max(float(np.percentile(row,90)) for row in errors)
            if worst < 13:
                validation.append((worst,poses,float(np.median(np.concatenate(errors)))))
    if not validation:
        return None
    worst,poses,median = min(validation,key=lambda row:row[0])
    return poses,keep,int(support.sum(axis=1).min()),median,worst
