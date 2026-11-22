"""One camera from the exact shared frame used to assemble a group scene."""
import json
from pathlib import Path
import numpy as np

from .camera import Camera
from .similarity import transform_camera


def shared_frame(tracks, shared, scale, origin):
    """Complete the common horizontal fit with the first actor's ground origin."""
    ref = tracks[0]
    observed = (ref['world'] @ shared.T)[:, 2:4].mean(axis=1)
    body = ref['motion'][:, 2:4].mean(axis=1)
    center = ref['camera'] @ shared.T
    y = np.median(body[:, 1] - ref['floor'] - observed[:, 1] - scale*center[:, 1])
    return dict(rotation=shared, scale=scale, translation=np.array([-origin[0], y, -origin[1]]))


def save_shared_inputs(destination, tracks, rt, K, first, last, hw):
    from .runtime import save_npz, save_json, digest
    destination = Path(destination)
    folder = destination / 'camera'
    folder.mkdir(exist_ok=True)
    frame = tracks[0]['shared_frame']
    valid = np.zeros(last-first+1, bool)
    # One reference joint stream for the VMD optical dolly. Prefer the first
    # selected person; use other visible people where that person is absent.
    joints = np.zeros((len(valid), 52, 3), np.float32)
    from flowhmr.utils.runtime_group_scene import yaw_matrix
    for track in tracks:
        a, b = track['range']
        selection = slice(a-first, b-first+1)
        points = track['joints'] @ yaw_matrix(track['yaw']).T + track['offset']
        unseen = ~valid[selection]
        joints[selection][unseen] = points[unseen]
        valid[selection] = True
    from .timeline import held_indices
    joints = joints[held_indices(valid)]
    save_npz(folder / 'shared-frame.npz', RT=rt, K=K, image_size_hw=hw,
             world_rotation=frame['rotation'], world_scale=frame['scale'],
             world_translation=frame['translation'], camera_valid=valid,
             track_start=first, joints=joints)
    save_json(folder / 'shared-frame.json', dict(version=1,
        motion_sha256=digest(destination / 'motion.npz'), sha256=digest(folder / 'shared-frame.npz'),
        source='upstream_group_scene_common_fit', first_person=tracks[0]['job']['id'],
        body_references=[t['body_reference'] for t in tracks]))


def build_shared_camera(destination, *, check=lambda: None, notify=lambda message, progress: None,
                        preserve_previous=False):
    from .runtime import digest, publish_camera
    destination = Path(destination)
    folder = destination / 'camera'
    check()
    proof = json.loads((folder / 'shared-frame.json').read_text(encoding='utf8'))
    if proof['motion_sha256'] != digest(destination / 'motion.npz') or proof['sha256'] != digest(folder / 'shared-frame.npz'):
        raise ValueError('多人场景已变化，需要重新保存共同相机坐标。')
    notify('恢复多人场景的共同机位', 75)
    with np.load(folder / 'shared-frame.npz', allow_pickle=False) as z:
        rt = transform_camera(z['RT'], z['world_rotation'], float(z['world_scale']), z['world_translation'])
        camera = Camera(rt, z['K'], z['camera_valid'], np.zeros(len(rt), np.int32),
                        z['image_size_hw'], int(z['track_start']))
        report = dict(version=1, source=proof, motion_sha256=proof['motion_sha256'],
                      motion_changed=False, generation_calls=0,
                      shared_world=True, world_rotation=z['world_rotation'].tolist(),
                      world_scale=float(z['world_scale']), world_translation=z['world_translation'].tolist(),
                      labels=['shared_scene'], physical_calibration_guaranteed=False,
                      character_scale_policy='first_selected_character; one shared camera for all people')
    check()
    if proof['motion_sha256'] != digest(destination / 'motion.npz'):
        raise ValueError('多人场景在恢复机位期间发生变化。')
    publish_camera(folder, camera, report, check=check, preserve_previous=preserve_previous)
    return report
