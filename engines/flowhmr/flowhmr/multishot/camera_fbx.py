"""Separate Y-up/metre camera FBX using the application's installed FBX SDK."""
import numpy as np
from scipy.spatial.transform import Rotation


def export_camera_fbx(data, path):
    import fbx
    from flowhmr.core.export.fbx_utils import create_manager, export_scene_atomic

    indices = np.flatnonzero(data.camera_valid)
    if not len(indices):
        raise ValueError('No valid camera frames')
    h, w = map(int, data.image_size_hw)
    hold = np.r_[~data.camera_valid[1:] | (data.shot_ids[1:] != data.shot_ids[:-1]), False]
    centers = data.centers
    ry90 = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    matrices = np.tile(np.eye(3), (len(data.RT), 1, 1))
    matrices[indices] = data.RT[indices, :3, :3].transpose(0, 2, 1) @ np.diag([1., -1., -1.]) @ ry90
    angles = Rotation.from_matrix(matrices).as_euler('xyz', degrees=True)
    from .timeline import intervals
    for a, b in intervals(data.camera_valid):
        cuts = np.r_[a, np.flatnonzero(np.diff(data.shot_ids[a:b])) + a + 1, b]
        for first, last in zip(cuts[:-1], cuts[1:]):
            angles[first:last] = np.rad2deg(np.unwrap(np.deg2rad(angles[first:last]), axis=0))
    focal = (data.K[:, 0, 0] + data.K[:, 1, 1]) * .5 * 36 / max(w, h)
    manager = create_manager()
    try:
        scene = fbx.FbxScene.Create(manager, 'Paired camera')
        settings = scene.GetGlobalSettings()
        settings.SetAxisSystem(fbx.FbxAxisSystem.MayaYUp)
        settings.SetSystemUnit(fbx.FbxSystemUnit.m)
        settings.SetOriginalSystemUnit(fbx.FbxSystemUnit.m)
        settings.SetTimeMode(fbx.FbxTime.ConvertFrameRateToTimeMode(data.fps))
        start, stop = fbx.FbxTime(), fbx.FbxTime()
        start.SetSecondDouble(data.track_start / data.fps)
        stop.SetSecondDouble((data.track_start + len(data.RT) - 1) / data.fps)
        span = fbx.FbxTimeSpan(start, stop)
        settings.SetTimelineDefaultTimeSpan(span)
        stack = fbx.FbxAnimStack.Create(scene, 'Camera')
        stack.SetLocalTimeSpan(span)
        stack.SetReferenceTimeSpan(span)
        layer = fbx.FbxAnimLayer.Create(scene, 'BaseLayer')
        stack.AddMember(layer)
        node = fbx.FbxNode.Create(scene, 'FlowHMR Camera')
        camera = fbx.FbxCamera.Create(scene, 'FlowHMR Camera')
        node.SetNodeAttribute(camera)
        scene.GetRootNode().AddChild(node)
        camera.ProjectionType.Set(fbx.FbxCamera.EProjectionType.ePerspective.value)
        camera.SetAspect(fbx.FbxCamera.EAspectRatioMode.eFixedResolution, float(w), float(h))
        camera.SetApertureMode(fbx.FbxCamera.EApertureMode.eFocalLength)
        camera.FilmWidth.Set(36 / 25.4)
        camera.FilmHeight.Set(36 / 25.4)
        camera.GateFit.Set((fbx.FbxCamera.EGateFit.eFitVertical if h > w else fbx.FbxCamera.EGateFit.eFitHorizontal).value)
        camera.NearPlane.Set(.01)
        camera.FarPlane.Set(10000.)
        curves = [(node.LclTranslation.GetCurve(layer, axis, True), centers[:, i]) for i, axis in enumerate('XYZ')]
        curves += [(node.LclRotation.GetCurve(layer, axis, True), angles[:, i]) for i, axis in enumerate('XYZ')]
        curves += [(camera.FocalLength.GetCurve(layer, True), focal)]
        for curve, values in curves:
            curve.KeyModifyBegin()
            for frame in indices:
                time = fbx.FbxTime()
                time.SetSecondDouble((int(frame) + data.track_start) / data.fps)
                key = curve.KeyAdd(time)[0]
                curve.KeySetValue(key, float(values[frame]))
                interpolation = fbx.FbxAnimCurveDef.EInterpolationType
                curve.KeySetInterpolation(key, interpolation.eInterpolationConstant if hold[frame]
                                          else interpolation.eInterpolationLinear)
            curve.KeyModifyEnd()
        settings.SetDefaultCamera(node.GetName())
        export_scene_atomic(manager, scene, str(path))
    finally:
        manager.Destroy()
    return {'method': 'average_fx_fy_centered_square_pixel_fbx', 'image_size_hw': [h, w],
            'required_aspect_ratio': w / h,
            'max_principal_point_offset_px': float(np.max(np.abs(data.K[indices, :2, 2] - [w / 2, h / 2]))),
            'max_focal_anisotropy_px': float(np.max(np.abs(data.K[indices, 0, 0] - data.K[indices, 1, 1]))),
            'lossless_intrinsics': 'camera.json'}
