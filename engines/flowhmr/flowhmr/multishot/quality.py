"""Independent camera availability and image-gravity diagnostics."""

from .timeline import intervals


def camera_coverage(camera):
    total = len(camera.camera_valid)
    count = int(camera.camera_valid.sum())
    return dict(status='complete' if count == total else 'partial' if count else 'unavailable',
                valid_frames=count, total_frames=total,
                interval_convention='half_open_source_frames',
                missing_source_ranges=[[a+camera.track_start, b+camera.track_start]
                                       for a, b in intervals(~camera.camera_valid)])


def camera_quality(report):
    if report.get('shared_world'):
        return dict(status='not_applicable', degraded=False, reason='shared_world')
    rows = [row for row in report.get('shots', [])
            if row.get('gravity_source') and not row.get('superseded')
            and 'failed' not in row.get('labels', [])]
    fallback = sorted({row['shot'] for row in rows if row['gravity_source'] != 'image'})
    total = len({row['shot'] for row in rows})
    diagnostic = report.get('gravity', {})
    if not total:
        return dict(status='unknown', degraded=True, reason='unverified', fallback_shots=[], total_shots=0)
    reason = diagnostic.get('reason')
    if fallback and not reason:
        reason = 'unstable' if diagnostic.get('status') in ('measured', 'archived') else 'legacy_fallback'
    return dict(status=('partial' if len(fallback) < total else 'fallback') if fallback else 'image',
                degraded=bool(fallback), reason=reason if fallback else None,
                fallback_shots=fallback, total_shots=total)


def read_quality(directory):
    from .runtime import load_paired_result
    try:
        _, report = load_paired_result(directory)
        return report['quality']
    except (OSError, ValueError, TypeError, KeyError):
        return dict(status='unknown', degraded=True, reason='unverified', fallback_shots=[], total_shots=0)


def ready_message(quality, coverage=None):
    notes = []
    if coverage and coverage['status'] == 'partial':
        notes.append('部分区间无可用机位')
    if quality['degraded']:
        notes.append('画面重力已降级，地面倾角可能不准确')
    return f'摄像机已就绪（{"。".join(notes)}）' if notes else '摄像机已就绪'
