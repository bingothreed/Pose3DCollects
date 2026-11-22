"""A fixed shot similarity with image-derived gravity and visible-body evidence."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .similarity import transform_camera,project,validate_camera
from flowhmr.utils.runtime_group_scene import _directions,_rotation

SM=np.array([16,17,18,19,20,21,1,2,4,5,7,8])
MH=np.array([5,6,7,8,62,41,9,10,11,12,13,14])


class CameraFitConvergenceError(ValueError):
    """The shot optimizer exhausted its search without converging."""


def yaw_matrix(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def body_facing_rotation(rt,observed,motion):
    ids=[0,1,6,7,8,9,10,11]
    observed_world=np.einsum('tji,tkj->tki',rt[:,:3,:3],observed[:,ids])
    a,w=_directions(observed_world);b,_=_directions(motion[:,ids])
    return _rotation(a,b,w)


def fit_shot(rt,K,observed,motion,world_up,initial_rotation,scale_prior,*,image_size_hw,valid=None,check=lambda: None,
             ground_min=None, fit_joints=None):
    """All returned frames share one similarity; no framewise position fit."""
    rt,K=validate_camera(rt,K);n=len(rt)
    valid=np.ones(n,dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    observed=np.nan_to_num(observed);motion=np.nan_to_num(motion)
    H,W=image_size_hw
    up=np.array(world_up,dtype=float);up/=np.linalg.norm(up)
    Qup=Rotation.align_vectors(np.array([[0.,1.,0.]]),up[None])[0].as_matrix()
    D=initial_rotation@Qup.T
    initial_yaw=np.arctan2(D[0,2]-D[2,0],D[0,0]+D[2,2])
    target,_=project(observed,np.tile(np.eye(4),(n,1,1)),K)
    inside=(target[...,0]>=0)&(target[...,0]<W)&(target[...,1]>=0)&(target[...,1]<H)&(observed[...,2]>.05)&valid[:,None]
    importance=np.array([1,1,.6,.6,.35,.35,1,1,.6,.6,.5,.5])
    weight=inside*importance
    if fit_joints is not None:
        weight *= fit_joints
    if not valid.any() or np.any(weight.sum(1)[valid]<.1):raise ValueError('No visible alignment evidence in a valid frame')
    weight=weight/np.maximum(weight.sum(1,keepdims=True),1e-8)
    C=-np.einsum('tji,tj->ti',rt[:,:3,:3],rt[:,:3,3])
    Q=yaw_matrix(initial_yaw)@Qup
    rays=np.einsum('tji,tkj->tki',rt[:,:3,:3],observed)@Q.T
    delta=np.sum(weight[...,None]*(motion-rays),axis=1)
    translation=np.median((delta-scale_prior*(C@Q.T))[valid],axis=0)
    initial=np.r_[initial_yaw,translation,np.log(scale_prior)]
    def unpack(p):return yaw_matrix(p[0])@Qup,float(np.exp(p[4])),p[1:4]
    # Shoulder and hip directions retain a weak 3D facing cue in symmetric views.
    pairs=[(0,1),(6,7)]
    observed_direction=np.stack([observed[:,i]-observed[:,j] for i,j in pairs],1)
    observed_direction/=np.maximum(np.linalg.norm(observed_direction,axis=-1,keepdims=True),1e-8)
    body_direction=np.stack([motion[:,i]-motion[:,j] for i,j in pairs],1)
    body_direction/=np.maximum(np.linalg.norm(body_direction,axis=-1,keepdims=True),1e-8)
    def residual(p):
        check()
        adjusted=transform_camera(rt,*unpack(p));uv,depth=project(motion,adjusted,K)
        facing=np.einsum('tij,tkj->tki',adjusted[:,:3,:3],body_direction)
        values = np.r_[((uv-target)*np.sqrt(weight)[...,None]).ravel(),
                     ((facing-observed_direction)[valid]*60.).ravel(),
                     (np.minimum(depth-.05,0)*1000*np.sqrt(weight)).ravel(),
                     np.sqrt(n)*8*(p[4]-np.log(scale_prior))/1.5]
        if ground_min is not None:
            centers = -np.einsum('tji,tj->ti', adjusted[:,:3,:3], adjusted[:,:3,3])
            values = np.r_[values, 1000*np.minimum(centers[:,1]-ground_min, 0)]
        return values
    bounds=(np.r_[initial_yaw-np.deg2rad(25),np.full(3,-np.inf),np.log(scale_prior/20)],
            np.r_[initial_yaw+np.deg2rad(25),np.full(3,np.inf),np.log(scale_prior*20)])
    result=least_squares(residual,initial,bounds=bounds,loss='soft_l1',f_scale=10,max_nfev=1500,x_scale='jac',ftol=1e-7)
    if not result.success:
        raise CameraFitConvergenceError('Per-shot alignment did not converge: '+result.message)
    Q,scale,T=unpack(result.x);aligned=transform_camera(rt,Q,scale,T)
    uv,depth=project(motion,aligned,K);error=np.linalg.norm(uv-target,axis=-1)
    centers=-np.einsum('tji,tj->ti',aligned[:,:3,:3],aligned[:,:3,3])
    reconstructed=scale*(C@Q.T)+T
    relative_raw=rt[1:,:3,:3]@rt[:-1,:3,:3].transpose(0,2,1)
    relative_new=aligned[1:,:3,:3]@aligned[:-1,:3,:3].transpose(0,2,1)
    assert np.max(abs(centers-reconstructed))<1e-4
    rotation_error=float(np.max(abs(relative_raw-relative_new),initial=0.))
    assert rotation_error<1e-8
    return dict(RT=aligned,world_rotation=Q,world_scale=scale,world_translation=T,inside=inside,observed_uv=target,projected_uv=uv,
        report=dict(method='one_fixed_similarity_with_image_gravity_visible_pixel_fit_and_weak_3D_facing',
            scale=scale,scale_prior=scale_prior,scale_prior_log_sigma=1.5,yaw_adjustment_deg=float(np.rad2deg(result.x[0]-initial_yaw)),
            visible_median_px=float(np.median(error[inside])),visible_p95_px=float(np.percentile(error[inside],95)),
            visible_max_px=float(error[inside].max()),visible_min_depth_m=float(depth[inside].min()),
            camera_height_m=[float(centers[:,1].min()),float(centers[:,1].max())],
            raw_trajectory_preserved=True,relative_rotation_matrix_max_error=rotation_error,
            centers_similarity_max_error_m=float(np.max(abs(centers-reconstructed))),optimizer_evaluations=result.nfev,
            image_gravity_fixed_per_shot=True,minimum_visible_points=int(inside.sum(1).min()),
            facing_initialization='robust 3D directions across the entire shot; not unconstrained 2D legacy optimum',
            facing_cue_weight_px=60.,yaw_budget_from_3D_deg=25.,yaw_bound_active=bool(abs(result.x[0]-initial_yaw)>np.deg2rad(24.99)),
            lower_scale_bound_active=bool(result.x[4]<bounds[0][4]+1e-4),upper_scale_bound_active=bool(result.x[4]>bounds[1][4]-1e-4)))
