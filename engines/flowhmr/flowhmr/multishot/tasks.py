"""Queue-independent camera task service, shared by HTTP and future MCP callers."""
from hashlib import sha256
import json
from threading import RLock
import time
from zipfile import BadZipFile

from .recovery import CameraCancelled
from .runtime import camera_directory, load_paired_result
from .quality import camera_quality, ready_message

ACTIVE_CAMERA_STATES = frozenset(('queued', 'running', 'cancel_requested'))


def camera_revision(report):
    publication_id = report.get('publication_id')
    if publication_id:
        return publication_id
    return sha256(json.dumps(report, sort_keys=True).encode('utf-8')).hexdigest()


def _read_report(directory):
    try:
        _, report = load_paired_result(directory)
        return report
    except (OSError, ValueError, KeyError, BadZipFile):
        return None


def _completed_camera(report):
    quality = camera_quality(report)
    return dict(camera_state='ready', camera_progress=100, camera_force=False, camera_error=None,
                camera_available=True, camera_revision=camera_revision(report),
                camera_quality=quality, camera_message=ready_message(quality, report.get('coverage')),
                camera_finished=time.time())


class CameraTasks:
    def __init__(self, get_job, job_directory, update_job, enqueue, build):
        self.get_job, self.directory, self.update = get_job, job_directory, update_job
        self.enqueue, self.build = enqueue, build
        self.lock = RLock()

    def root(self, job_id):
        job = self.get_job(job_id)
        if job['status'] != 'done':
            raise ValueError('请等待人物动作完成，再恢复摄像机。')
        if job.get('person_batch') == job_id and job.get('scene_state') != 'ready':
            raise ValueError('请先完成多人场景对齐，再恢复共同机位。')
        return camera_directory(self.directory(job_id), job)

    def _report(self, job_id):
        try:
            directory = self.root(job_id)
        except ValueError:
            return None
        return _read_report(directory)

    def status(self, job_id):
        job = self.get_job(job_id)
        report = self._report(job_id)
        available = report is not None
        quality = camera_quality(report) if available else None
        coverage = report.get('coverage') if available else None
        state = job.get('camera_state', 'none')
        if state in ('none', 'ready'):
            state = 'ready' if available else 'none'
        return dict(camera_state=state, available=available,
                    revision=camera_revision(report) if available else None,
                    message=ready_message(quality, coverage) if state == 'ready' else job.get('camera_message', ''),
                    quality=quality, coverage=coverage,
                    progress=100 if state == 'ready' else job.get('camera_progress', 0),
                    error=job.get('camera_error'),
                    started=job.get('camera_started'), finished=job.get('camera_finished'))

    def prepare(self, job_id, *, force=False):
        with self.lock:
            self.root(job_id)
            status = self.status(job_id)
            if status['camera_state'] in ACTIVE_CAMERA_STATES or (status['available'] and not force):
                return status
            message = '等待重新计算摄像机' if status['available'] else '等待恢复摄像机'
            self.update(job_id, camera_state='queued', camera_message=message, camera_progress=0,
                        camera_force=force, camera_error=None, camera_quality=status['quality'],
                        camera_available=status['available'], camera_revision=status['revision'],
                        camera_base_revision=status['revision'],
                        camera_started=None, camera_finished=None)
            self.enqueue(('camera', job_id))
            return self.status(job_id)

    def cancel(self, job_id):
        with self.lock:
            if self.get_job(job_id).get('camera_state') in ACTIVE_CAMERA_STATES:
                self.update(job_id, camera_state='cancel_requested', camera_message='正在停止摄像机计算')
            return self.status(job_id)

    def run(self, job_id):
        def check():
            if self.get_job(job_id).get('camera_state') == 'cancel_requested':
                raise CameraCancelled()

        def notify(message, progress):
            with self.lock:
                check()
                self.update(job_id, camera_message=message, camera_progress=float(progress))

        try:
            with self.lock:
                check()
                directory = self.root(job_id)
                previous = _read_report(directory)
                previous_revision = camera_revision(previous) if previous is not None else None
                self.update(job_id, camera_state='running', camera_started=time.time(), camera_error=None,
                            camera_base_revision=previous_revision)
            build_error = None
            try:
                self.build(job_id, check=check, notify=notify)
            except Exception as exc:
                build_error = exc
            with self.lock:
                report = self._report(job_id)
                # Only the new publication can complete this attempt. Its commit wins over a late cancel.
                if report is None or camera_revision(report) == previous_revision:
                    if build_error is not None:
                        raise build_error
                    check()
                    raise ValueError('摄像机结果未完整保存。')
                self.update(job_id, **_completed_camera(report))
        except CameraCancelled:
            with self.lock:
                self.update(job_id, camera_state='cancelled', camera_force=False,
                            camera_message='摄像机计算已停止，人物动作已保留',
                            camera_error=None, camera_finished=time.time())
        except Exception as exc:
            with self.lock:
                self.update(job_id, camera_state='failed', camera_force=False, camera_message='摄像机恢复失败，可重试',
                            camera_error=f'{type(exc).__name__}: {exc}', camera_finished=time.time())


def restore_camera_state(job, directory=None):
    if job.get('camera_state') not in ACTIVE_CAMERA_STATES:
        return
    if directory is not None and 'camera_base_revision' in job:
        report = _read_report(camera_directory(directory, job))
        if report is not None and camera_revision(report) != job['camera_base_revision']:
            job.update(_completed_camera(report))
            return
    job.update(camera_state='failed', camera_force=False,
               camera_error='应用重启中断了相机计算，可重试；人物动作已保留。',
               camera_message='摄像机计算已中断', camera_finished=time.time())
