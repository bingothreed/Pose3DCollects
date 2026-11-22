"""Compare a few observations per shot using local CPU models."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


WEIGHTS = Path(__file__).resolve().parents[2] / "assets/person_reid"
SAMPLES_PER_SHOT = 7
FACE_DISAGREES = .23
FACE_AGREES = .40
BODY_DISAGREES = .35
BODY_AGREES = .65
BODY_TRACK_SWITCH = .22


@dataclass
class Appearance:
    faces: list[np.ndarray] = field(default_factory=list)
    bodies: list[np.ndarray] = field(default_factory=list)
    clipped_bodies: list[np.ndarray] = field(default_factory=list)
    clipped_torsos: list[np.ndarray] = field(default_factory=list)


def _weights(name):
    # Passing a buffer also supports non-ASCII Windows installation paths.
    return np.frombuffer((WEIGHTS / name).read_bytes(), dtype=np.uint8)


def _unit(vector):
    vector = vector.reshape(-1)
    magnitude = np.linalg.norm(vector)
    return vector / magnitude if np.isfinite(magnitude) and magnitude > 0 else None


class AppearanceExtractor:
    def __init__(self):
        empty = np.empty(0, dtype=np.uint8)
        backend, target = cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
        self.detector = cv2.FaceDetectorYN.create(
            "onnx", _weights("face_detection_yunet_2023mar.onnx"), empty,
            (320, 320), .85, .3, 50, backend, target,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            "onnx", _weights("face_recognition_sface_2021dec_int8.onnx"), empty, backend, target,
        )
        self.body_net = cv2.dnn.readNetFromONNX(_weights("person-reidentification-retail-0270-opencv.onnx"))
        self.body_net.setPreferableBackend(backend)
        self.body_net.setPreferableTarget(target)

    def face(self, crop):
        height, width = crop.shape[:2]
        scale = min(1., 512 / max(height, width))
        image = crop if scale == 1 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.detector.setInputSize(image.shape[1::-1])
        _, faces = self.detector.detect(image)
        if faces is None:
            return None

        # Ignore faces behind the tracked subject and require landmarks suitable for alignment.
        eligible = []
        for face in faces:
            x, y, w, h = face[:4] / scale
            if not (.14 * width <= x+w/2 <= .86 * width and y+h/2 <= .43 * height):
                continue
            if min(w, h) < 36 or x < 1 or y < 1 or x+w >= width or y+h >= height:
                continue
            right_eye, left_eye, nose = face[4:10].reshape(3, 2)
            axis = left_eye - right_eye
            eye_distance_squared = np.dot(axis, axis)
            if (eye_distance_squared >= (.27 * face[2]) ** 2
                    and .18 <= np.dot(nose-right_eye, axis) / eye_distance_squared <= .82):
                eligible.append(face)
        if len(eligible) != 1:
            return None
        aligned = self.recognizer.alignCrop(image, eligible[0])
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_32F).var() < 24:
            return None
        return _unit(self.recognizer.feature(aligned))

    def body(self, crop):
        self.body_net.setInput(cv2.dnn.blobFromImage(crop, size=(128, 256)))
        return _unit(self.body_net.forward())


def _observe(model, frame, box, appearance):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.clip(box, 0, [width, height, width, height]).astype(int)
    w, h = x2-x1, y2-y1
    if w < 36 or h < 50:
        return
    crop = frame[y1:y2, x1:x2]
    face = model.face(crop)
    if face is not None:
        appearance.faces.append(face)
    # Close-up or clipped crops are poor evidence for whole-body disagreement.
    if h >= 96 and .18 <= w/h <= .88 and min(x1, y1, width-x2, height-y2) > 2:
        body = model.body(crop)
        if body is not None:
            appearance.bodies.append(body)
    elif h >= 100 and .48 <= w/h <= 2.4 and y2 >= .965 * height:
        # Keep close-up descriptors separate from the full-body model evidence.
        # A crop touching the lower edge may show a jacket with no legs.
        body = model.body(crop)
        torso = model.body(crop[round(.52*h):round(.96*h), round(.2*w):round(.8*w)])
        if body is not None and torso is not None:
            appearance.clipped_bodies.append(body)
            appearance.clipped_torsos.append(torso)


def sample_appearances(video, segments, check):
    """Each segment contains frame numbers and their tracked boxes on the source timeline."""
    model = AppearanceExtractor()
    profiles = [Appearance() for _ in segments]
    samples = []
    for profile, (frames, boxes) in zip(profiles, segments):
        for i in np.linspace(0, len(frames)-1, min(len(frames), SAMPLES_PER_SHOT), dtype=int):
            samples.append((int(frames[i]), boxes[i], profile))
    samples.sort(key=lambda item: item[0])
    cap = cv2.VideoCapture(str(video))
    try:
        previous = -1
        for number, box, profile in samples:
            check()
            if number != previous:
                if 0 < number-previous <= 24:
                    for _ in range(number-previous-1):
                        if not cap.grab():
                            raise OSError("Could not advance to a person re-identification sample")
                elif not cap.set(cv2.CAP_PROP_POS_FRAMES, number):
                    raise OSError("Could not seek to a person re-identification sample")
                if cap.get(cv2.CAP_PROP_POS_FRAMES) != number:
                    raise OSError("Decoder returned the wrong person re-identification frame")
                ok, frame = cap.read()
                if not ok:
                    raise OSError("Could not read a person re-identification sample")
                previous = number
            _observe(model, frame, box, profile)
    finally:
        cap.release()
    return profiles


def profile_tracks(video, tracks, shots, check):
    """Key each view by tracker ID and absolute shot start."""
    edges = [start for start, _ in shots] + [shots[-1][1]]
    keys, segments = [], []
    for track in tracks:
        borders = track.frames.searchsorted(edges)
        for shot, first, end in zip(edges, borders, borders[1:]):
            if end-first >= 2:
                keys.append((track.track_id, shot))
                segments.append((track.frames[first:end], track.bboxes[first:end]))
    if len(segments) < 2:
        return {}
    return dict(zip(keys, sample_appearances(video, segments, check)))


def similarity(left, right, *, within):
    """Return a cosine score only when both segments show consistent features."""
    if min(len(left), len(right)) < 2:
        return None
    a, b = np.stack(left), np.stack(right)
    for vectors in (a, b):
        pairs = vectors @ vectors.T
        if np.median(pairs[np.triu_indices(len(vectors), k=1)]) < within:
            return None
    scores = a @ b.T
    return float(np.median(np.r_[scores.max(axis=0), scores.max(axis=1)]))


def face_similarity(left, right):
    return similarity(left.faces, right.faces, within=.48)


def body_similarity(left, right):
    return similarity(left.bodies, right.bodies, within=.54)
