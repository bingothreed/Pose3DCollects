"""Detect abrupt cuts on the normalized source timeline."""
import cv2
import numpy as np


class ShotDetector:
    """Accumulate cut evidence without retaining video frames."""

    def __init__(self):
        self.changes = []
        self.textured = []
        self.previous = None

    def observe(self, frame):
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        self.textured.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std() >= 6)
        small = small.astype(np.float32)
        change = 0. if self.previous is None else np.abs(small - self.previous).mean()
        self.changes.append(change)
        self.previous = small

    def finish(self):
        changes = np.asarray(self.changes)
        textured = np.asarray(self.textured, dtype=bool)
        # Isolated jumps are cuts, including before a fast pan. Neighboring
        # changes suppress continuous pans and flashes.
        cuts = [0]
        for frame in np.flatnonzero(changes >= 18):
            if not textured[frame-1:frame+1].all():
                continue
            nearby = np.sort(np.r_[0., 0., changes[max(0, frame-3):frame], changes[frame+1:frame+4]])
            if changes[frame] >= max(1.5 * nearby[-1], 2.5 * nearby[-2]):
                cuts.append(int(frame))
        cuts.append(len(changes))
        return [dict(id=i, start=a, end=b) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))], textured


def camera_shots(video, first, count, check):
    """Return half-open, track-relative shots and nonblank source frames."""
    detector = ShotDetector()
    cap = cv2.VideoCapture(str(video))
    try:
        if first and (not cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                      or cap.get(cv2.CAP_PROP_POS_FRAMES) != first):
            raise ValueError(f'无法定位原视频第 {first+1} 帧。')
        for index in range(count):
            check()
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'无法读取原视频第 {first+index+1} 帧。')
            detector.observe(frame)
    finally:
        cap.release()
    return detector.finish()
