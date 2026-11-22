"""Optional post-task camera endpoints; no model generation or job status mutation."""
from pathlib import Path
from threading import Lock

from flask import jsonify, request, send_file

from .exports import camera_bundle
from .quality import camera_quality
from .runtime import InvalidCameraArtifact, load_paired_result
from .tasks import ACTIVE_CAMERA_STATES, camera_revision
from .timeline import source_body_visibility


def register_camera_routes(app, get_job, job_directory, load_joints, tasks=None, *, queue_camera=None):
    export_lock = Lock()

    def root(job_id):
        if tasks is not None:
            return tasks.root(job_id)
        job = get_job(job_id)
        if job['status'] != 'done':
            raise ValueError('人物动作尚未完成。')
        base = Path(job_directory(job_id))
        return base / 'scene' if job.get('person_batch') == job_id else base

    def missing():
        return jsonify(error='此任务还没有可用摄像机。', error_code='camera_not_ready'), 409

    def invalid(error):
        return jsonify(error=str(error), error_code='camera_invalid'), 422

    @app.get('/api/jobs/<job_id>/camera')
    def paired_camera(job_id):
        try:
            from .ground import stored_floor
            directory = root(job_id)
            camera, report = load_paired_result(directory)
            return jsonify({**camera.json(stored_floor(directory)),
                            'quality': camera_quality(report), 'coverage': report.get('coverage'),
                            'source_body_visible': source_body_visibility(camera, report),
                            'revision': camera_revision(report)})
        except FileNotFoundError:
            return missing()
        except (ValueError, KeyError) as exc:
            return invalid(exc)

    @app.post('/api/jobs/<job_id>/camera/prepare')
    def prepare_paired_camera(job_id):
        body = request.get_json(silent=True) if request.get_data() else {}
        if not isinstance(body, dict) or set(body) - {'force'} or not isinstance(body.get('force', False), bool):
            return jsonify(error='摄像机恢复参数无效。', error_code='camera_options'), 400
        force = body.get('force', False)
        try:
            if tasks is not None:
                if queue_camera is not None:
                    tasks.root(job_id)
                    status = tasks.status(job_id)
                    if status['available'] and not force:
                        return jsonify(status)
                    if status['camera_state'] in ACTIVE_CAMERA_STATES:
                        return jsonify(status), 202
                    return queue_camera(job_id, force=True) if force else queue_camera(job_id)
                status = tasks.prepare(job_id, force=force)
                return jsonify(status), 200 if status['available'] and not force else 202
            if force:
                return jsonify(error='此服务不支持重新计算摄像机。', error_code='camera_retry_unavailable'), 409
            _, report = load_paired_result(root(job_id))
            return jsonify(camera_state='ready', available=True, revision=camera_revision(report),
                           quality=camera_quality(report), coverage=report.get('coverage'))
        except FileNotFoundError:
            return missing()
        except (ValueError, KeyError) as exc:
            return invalid(exc)

    @app.get('/api/jobs/<job_id>/camera/status')
    def paired_camera_status(job_id):
        return jsonify(tasks.status(job_id) if tasks else {'camera_state': 'ready'})

    @app.post('/api/jobs/<job_id>/camera/cancel')
    def cancel_paired_camera(job_id):
        return jsonify(tasks.cancel(job_id) if tasks else {'camera_state': 'ready'})

    @app.get('/api/jobs/<job_id>/camera/report')
    def paired_camera_report(job_id):
        try:
            _, report = load_paired_result(root(job_id))
            return jsonify(report)
        except FileNotFoundError:
            return missing()
        except (ValueError, KeyError) as exc:
            return invalid(exc)

    @app.post('/api/jobs/<job_id>/camera/export')
    def export_paired_camera(job_id):
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - {'format', 'transform'}:
            return jsonify(error='摄像机导出参数无效。', error_code='camera_options'), 400
        try:
            with export_lock:
                directory = root(job_id)
                paired = load_paired_result(directory)
                kind = body.get('format')
                joints = load_joints(job_id, directory) if kind == 'vmd' else None
                stream = camera_bundle(directory, kind, joints, body.get('transform'), paired=paired)
            return send_file(stream, mimetype='application/zip', as_attachment=True, download_name='camera.zip')
        except FileNotFoundError:
            return missing()
        except InvalidCameraArtifact as exc:
            return invalid(exc)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc), error_code='camera_options'), 400
