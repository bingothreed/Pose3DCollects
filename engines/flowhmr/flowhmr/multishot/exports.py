"""Paired camera bundles. The stored motion and camera are never modified."""
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from .camera import transform_parameters
from .quality import camera_coverage, camera_quality
from .runtime import load_paired_result
from .timeline import gap_report, source_body_visibility


def camera_bundle(directory, format_name, joints, transform=None, *, paired=None):
    if format_name not in ('vmd', 'fbx', 'npz'):
        raise ValueError('Camera export supports VMD, FlowHMR FBX or NPZ')
    root = Path(directory)
    original, recovery = paired or load_paired_result(root)
    scale, translation = transform_parameters(transform, original)
    camera = original.transformed(scale, translation)
    from .ground import stored_floor
    ground = stored_floor(root)
    if translation.ndim == 1:
        ground = ground * scale + float(translation[1])
    if joints is not None:
        joints = np.asarray(joints) * scale + (translation if translation.ndim == 1 else translation[:, None])
    with np.load(root / 'motion.npz', allow_pickle=False) as motion:
        if 'trans' in motion:
            if len(motion['trans']) != len(camera.RT):
                raise ValueError('Paired camera and motion lengths differ')
            # The existing single-person motion exporters start at frame zero.
            # Preview and raw observations still use the original video offset.
            camera = replace(camera, track_start=0)
            gaps = gap_report(motion, 0)
        else:
            from .timeline import intervals
            scene = json.loads((root / 'scene.json').read_text(encoding='utf8'))
            if scene['num_frames'] != len(camera.RT):
                raise ValueError('Shared camera and scene lengths differ')
            gaps = dict(fps=30, track_start=camera.track_start, interval_convention='half_open_source_frames',
                        people=scene['people'], gaps=[[a+camera.track_start, b+camera.track_start]
                        for a,b in intervals(~camera.camera_valid)])
    report = {'camera_transform': (dict(scale=scale, translation=translation.tolist()) if translation.ndim == 1
                                  else dict(scale=scale, method='head_to_sole_and_animated_pelvis',
                                            sampled_frames=len(transform['frame_translations']),
                                            source_start_frame=transform['start_frame'])),
              'motion': gaps, 'camera': {'format': format_name},
              'timeline': {'source_track_start': original.track_start,
                           'export_track_start': camera.track_start, 'fps': camera.fps},
              'source_world': 'paired_motion_y_up_metres',
              'required_aspect_ratio': float(camera.image_size_hw[1] / camera.image_size_hw[0])}
    report['quality'] = recovery.get('quality') or camera_quality(recovery)
    report['coverage'] = recovery.get('coverage') or camera_coverage(original)
    body_visible = source_body_visibility(original, recovery)
    if (root / 'scene.json').exists():
        report['character_scale_policy'] = 'Camera uses the first selected character scale; different body sizes cannot all preserve the same projection.'
    report['recovery'] = recovery
    folder = root / 'camera/exports'
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='paired-', dir=folder) as temp:
        temp = Path(temp)
        camera.save(temp / 'camera.npz')
        (temp / 'camera.json').write_text(json.dumps({**camera.json(ground),
            'source_track_start': original.track_start, 'quality': report['quality'],
            'source_body_visible': body_visible,
            'coverage': report['coverage']},
            ensure_ascii=False, allow_nan=False), encoding='utf-8')
        if format_name == 'vmd':
            from .camera_vmd import export_camera_vmd
            report['camera'] = export_camera_vmd(camera, joints, temp / 'camera.vmd', source_body_visible=body_visible)
        elif format_name == 'fbx':
            from .camera_fbx import export_camera_fbx
            report['camera'] = export_camera_fbx(camera, temp / 'camera.fbx')
        (temp / 'export-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (temp / 'READ_ME.txt').write_text(
            'FlowHMR paired motion and camera\n'
            '30 FPS. Import the camera alongside the motion. The source start and export start are in export-report.json.\n'
            'VMD: load the character motion on the same PMX, and camera.vmd on the camera.\n'
            'Blender mmd_tools: import the PMX and both VMD files at scale 0.08, current frame 0, frame margin 0.\n'
            'Enable the model root Use Property Driver option for display and IK animation.\n'
            'VMD uses one integer vertical FOV per shot plus optical-axis dolly. FBX averages fx/fy.\n'
            'These are approximations. camera.npz and camera.json retain full intrinsics and independent camera validity.\n'
            'Camera gaps contain no keys. FBX holds the last camera until the next shot or valid key.\n'
            'camera.json includes source_body_visible for complete cutaways without body observations.\n'
            'VMD players may interpolate across gaps. Check the report for source visibility.\n'
            'A reference pose at FBX frame -1 is outside the paired camera timeline.\n'
            'The character model and textures are not included.\n', encoding='utf-8')
        stream = BytesIO()
        with ZipFile(stream, 'w', ZIP_DEFLATED) as archive:
            for path in sorted(temp.iterdir()):
                archive.write(path, path.name)
        stream.seek(0)
        return stream
