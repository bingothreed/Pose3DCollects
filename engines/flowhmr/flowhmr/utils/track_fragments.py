"""Join one person's tracks without interpolating image positions across cuts."""
from dataclasses import replace
from itertools import combinations
import logging

import numpy as np

MAX_GAP = 15  # Missing frames at normalized 30 FPS; never bridge a long absence.
MAX_EDGE = 5
MAX_DETAIL_CUTAWAY = 90  # Three seconds on the normalized timeline.

log = logging.getLogger(__name__)


class TrackJoinError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def join_candidates(candidates, people, frame_range, *, shots=()):
    """Return one candidate covering the requested range, or refuse ambiguity.

    This validates temporal/box continuity, not identity. The user must confirm
    that the selected, non-overlapping fragments show the same person.
    """
    if len(people) < 2 or len(set(people)) != len(people):
        raise TrackJoinError("track_join_count")
    parts = []
    for index in people:
        if type(index) is not int or not 0 <= index < len(candidates):
            raise TrackJoinError("track_join_invalid")
        candidate = candidates[index]
        frames = np.asarray(candidate.get("frames", []))
        boxes = np.asarray(candidate.get("boxes", []), dtype=np.float64)
        if (frames.ndim != 1 or frames.size == 0 or frames.dtype.kind not in "iu"
                or boxes.shape != (len(frames), 4) or not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise TrackJoinError("track_join_invalid")
        order = np.argsort(frames)
        frames, boxes = frames[order], boxes[order]
        if np.any(np.diff(frames) <= 0):
            raise TrackJoinError("track_join_invalid")
        parts.append((index, frames, boxes))
    parts.sort(key=lambda part: int(part[1][0]))
    if len(shots) > 1:
        edited = _join_across_cuts(candidates, parts, frame_range, shots)
        if edited is not None:
            return edited
    for (_, before, left), (_, after, right) in zip(parts, parts[1:]):
        if after[0] <= before[-1]:
            raise TrackJoinError("track_join_overlap")
        # A close hand-off is necessary even when the user confirms identity.
        # ID switches often coincide with a few partial-body detections. Use
        # five observations so one unstable endpoint cannot veto a valid join.
        if not _nearby_boxes(left[-5:], right[:5]):
            raise TrackJoinError("track_join_jump")
    frames = np.concatenate([part[1] for part in parts])
    boxes = np.concatenate([part[2] for part in parts])
    if np.any(np.diff(frames) - 1 > MAX_GAP):
        raise TrackJoinError("track_join_gap")
    start, end = frame_range
    if (start < 0 or end < start or frames[0] < start or frames[-1] > end
            or frames[0] - start > MAX_EDGE or end - frames[-1] > MAX_EDGE):
        raise TrackJoinError("track_join_coverage")
    # Hold endpoints for at most five frames. Internal holes are interpolated
    # by VideoProcessor.track_bboxes, exactly like ordinary tracking gaps.
    observed_frames = frames.tolist()
    if frames[0] > start:
        frames = np.concatenate([[start], frames]); boxes = np.concatenate([boxes[:1], boxes])
    if frames[-1] < end:
        frames = np.concatenate([frames, [end]]); boxes = np.concatenate([boxes, boxes[-1:]])
    first = candidates[people[0]]
    return {**first, "start": int(start), "end": int(end), "num_frames": len(frames),
            "frames": frames.tolist(), "boxes": boxes.tolist(),
            "observed_frames": observed_frames,
            "fragment_indices": list(people),
            "source_tracks": [{key: candidates[index][key] for key in ("track_id", "start", "end")}
                              for index in people]}


def group_candidates(candidates, groups, frame_range, *, shots=()):
    """Resolve user-confirmed groups to one track per person in selection order."""
    if not isinstance(groups, list) or not groups or any(not isinstance(group, list) or not group for group in groups):
        raise TrackJoinError("track_join_invalid")
    indices = [index for group in groups for index in group]
    if (any(type(index) is not int or not 0 <= index < len(candidates) for index in indices)
            or len(indices) != len(set(indices))):
        raise TrackJoinError("track_join_invalid")

    resolved = []
    start, end = frame_range
    for group in groups:
        if len(group) == 1:
            resolved.append(candidates[group[0]])
            continue
        first = max(start, min(candidates[index]["start"] for index in group))
        last = min(end, max(candidates[index]["end"] for index in group))
        resolved.append(join_candidates(candidates, group, (first, last), shots=shots))
    return resolved


def _join_across_cuts(candidates, parts, frame_range, shots):
    starts = np.array([a for a, _ in shots])
    frames = np.concatenate([p[1] for p in parts])
    boxes = np.concatenate([p[2] for p in parts])
    owners = np.repeat([p[0] for p in parts], [len(p[1]) for p in parts])
    order = np.argsort(frames, kind="stable")
    frames, boxes, owners = frames[order], boxes[order], owners[order]
    shot_ids = np.searchsorted(starts, frames, side="right") - 1
    changes = owners[1:] != owners[:-1]
    cuts = shot_ids[1:] != shot_ids[:-1]
    if not np.any(changes & cuts):
        return None
    # Track lifetimes can overlap even when they describe alternating views of
    # one dancer. Two detections on the same frame still require a human choice.
    if np.any(np.diff(frames) <= 0):
        raise TrackJoinError("track_join_overlap")
    for i in np.flatnonzero(changes & ~cuts):
        if not _nearby_boxes(boxes[max(0, i-4):i+1], boxes[i+1:i+6]):
            raise TrackJoinError("track_join_jump")
    # A short cutaway can momentarily show only detail or scenery.
    if np.any(np.diff(frames) - 1 > np.where(cuts, 30, MAX_GAP)):
        raise TrackJoinError("track_join_gap")
    start, end = frame_range
    if not start <= frames[0] <= frames[-1] <= end:
        raise TrackJoinError("track_join_coverage")
    first_shot, last_shot = shot_ids[[0, -1]]
    a = max(start, shots[first_shot][0])
    b = min(end, shots[last_shot][1]-1)
    first = int(a if frames[0]-a <= 9 else frames[0])
    last = int(b if b-frames[-1] <= 9 else frames[-1])
    ranges = [[max(first, a), min(last+1, b)] for a,b in shots[first_shot:last_shot+1]]
    indices = [p[0] for p in parts]
    handoffs = np.flatnonzero(changes & cuts)
    transitions = [(int(owners[i]), int(owners[i+1]), int(starts[shot_ids[i]]), int(starts[shot_ids[i+1]]))
                   for i in handoffs]
    return {**candidates[indices[0]], "start": first, "end": last,
            "num_frames": len(frames), "frames": frames.tolist(), "boxes": boxes.tolist(),
            "observed_frames": frames.tolist(), "fragment_indices": indices,
            "shot_ranges": ranges, "cut_handoffs": frames[handoffs+1].tolist(), "cut_transitions": transitions,
            "source_tracks": [{key: candidates[i][key] for key in ("track_id", "start", "end")}
                              for i in indices]}


def _nearby_boxes(left, right):
    a, b = np.median(left, axis=0), np.median(right, axis=0)
    sizes = np.stack([a[2:] - a[:2], b[2:] - b[:2]])
    centers = np.stack([(a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2])
    ratio = np.prod(sizes[1]) / np.prod(sizes[0])
    return .5 <= ratio <= 2 and np.linalg.norm(centers[1] - centers[0]) <= .5 * np.linalg.norm(sizes.mean(axis=0))


def review_track_identities(video, result, min_length, check):
    """Split IDs that ByteTrack kept through a clearly different person at a cut."""
    import cv2

    from flowhmr.utils.track_appearance import (
        BODY_TRACK_SWITCH, FACE_AGREES, FACE_DISAGREES, body_similarity, face_similarity, profile_tracks,
    )

    last_id = max(track.track_id for track in result.tracks)
    longest = max(result.tracks, key=lambda track: (track.length, track.avg_area))
    tracks = [track for track in result.tracks if track.length >= min_length or track is longest]
    try:
        appearances = profile_tracks(video, tracks, result.shots, check)
    except (OSError, AttributeError, cv2.error) as error:
        log.warning("Person re-identification unavailable for this video: %s", error)
        return {}

    reviewed = []
    for track in result.tracks:
        views = [(shot, appearances.get((track.track_id, shot))) for shot, _ in result.shots]
        splits = []
        for (_, left), (shot, right) in zip(views, views[1:]):
            if left is None or right is None:
                continue
            face = face_similarity(left, right)
            if face is not None and face >= FACE_AGREES:
                continue
            if face is not None and face < FACE_DISAGREES:
                splits.append(shot)
                continue
            if min(len(left.bodies), len(right.bodies)) >= 3:
                body = body_similarity(left, right)
                if body is not None and body < BODY_TRACK_SWITCH:
                    splits.append(shot)
        if not splits:
            reviewed.append(track)
            continue
        start = 0
        boundaries = track.frames.searchsorted(splits).tolist() + [track.length]
        for part, (end, stop) in enumerate(zip(boundaries, splits + [result.shots[-1][1]])):
            if part:
                last_id += 1
            track_id = last_id if part else track.track_id
            boxes = track.bboxes[start:end]
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            reviewed.append(replace(track, track_id=track_id, frames=track.frames[start:end],
                                    bboxes=boxes, length=len(boxes), avg_area=float(areas.mean())))
            for shot, view in views:
                if part and view is not None and splits[part-1] <= shot < stop:
                    appearances.pop((track.track_id, shot))
                    appearances[(track_id, shot)] = view
            start = end
        log.info("Person re-identification split track %s at source frames %s",
                 track.track_id, [frame+1 for frame in splits])
    result.tracks = reviewed
    return appearances


def propose_edited_track(video, candidates, frame_range, check, *, shots, appearances):
    """Suggest one person only when the cut geometry and identity evidence agree."""
    if len(shots) < 2 or not appearances:
        return None
    from flowhmr.utils.track_appearance import (
        BODY_AGREES, BODY_DISAGREES, FACE_AGREES, FACE_DISAGREES, body_similarity, face_similarity,
    )

    try:
        candidate = join_candidates(candidates, list(range(len(candidates))), frame_range, shots=shots)
    except TrackJoinError:
        return None
    if "shot_ranges" not in candidate:
        return None

    track_ids = {row["track_id"] for row in candidates}
    face_views = [(track_id, view) for (track_id, _), view in appearances.items()
                  if track_id in track_ids and len(view.faces) >= 2]
    compared_faces = set()
    face_matches = set()
    # A distant, clear view can expose a wrong merge even if an intervening
    # crop has only legs or the two adjacent outfits look alike.
    for (a, left), (b, right) in combinations(face_views, 2):
        check()
        score = face_similarity(left, right)
        if score is None:
            continue
        if score < FACE_DISAGREES:
            return None
        if a != b:
            pair = frozenset((a, b))
            compared_faces.add(pair)
            if score >= FACE_AGREES:
                face_matches.add(pair)

    transitions = [(candidates[a]["track_id"], candidates[b]["track_id"], left, right)
                   for a, b, left, right in candidate["cut_transitions"]]
    anchored = {track for pair in face_matches for track in pair}
    handoff_frames = candidate["cut_handoffs"]
    bracketed, corroborated_handoffs = set(), set()
    for i, ((a, b, _, detail_shot), (c, d, exit_shot, next_shot)) in enumerate(zip(transitions, transitions[1:])):
        if b != c or not (frozenset((a, d)) in face_matches or a == d and a in anchored):
            continue
        bracketed.add(b)
        # Different camera distances make pixel color unreliable in a short
        # cutaway. Only substitute whole-body agreement from multiple shots of
        # the face-confirmed neighbors when the detail itself has no usable face.
        detail = appearances.get((b, detail_shot))
        if detail is None or detail.faces or detail_shot != exit_shot or next_shot - detail_shot > MAX_DETAIL_CUTAWAY:
            continue
        matches = set()
        for (track, shot), view in appearances.items():
            if track in (a, d):
                check()
                score = body_similarity(detail, view)
                if score is not None and score >= BODY_AGREES:
                    matches.add((track, shot))
        if len({shot for _, shot in matches}) >= 2 and {track for track, _ in matches} == {a, d}:
            corroborated_handoffs.update(handoff_frames[i:i+2])

    weak_pairs = []
    needs_color = False
    for frame, (a, b, left_shot, right_shot) in zip(handoff_frames, transitions):
        pair = frozenset((a, b))
        if pair in face_matches:
            continue
        if pair in compared_faces:
            return None
        left = appearances.get((a, left_shot))
        right = appearances.get((b, right_shot))
        score = body_similarity(left, right) if left is not None and right is not None else None
        if score is not None and score < BODY_DISAGREES:
            return None
        if frame in corroborated_handoffs:
            continue
        needs_color = True
        if score is None or score < BODY_AGREES:
            weak_pairs.append(pair)

    # Short detail shots can lack comparable faces and full-body crops. The
    # matching person must already be established on either side of the shot.
    supported = anchored | bracketed
    if any(pair - supported for pair in weak_pairs):
        return None
    if needs_color:
        handoffs = [frame for frame in handoff_frames if frame not in corroborated_handoffs]
        if not _matching_clothing(video, {**candidate, "cut_handoffs": handoffs}, check):
            return None
    return candidate


def _matching_clothing(video, candidate, check):
    import cv2

    frames = np.asarray(candidate["frames"])
    boxes = np.asarray(candidate["boxes"])
    cap = cv2.VideoCapture(str(video))
    distances = []

    def histogram(indices):
        samples = []
        for index in indices:
            check()
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[index])):
                continue
            ok, image = cap.read()
            if not ok:
                continue
            a, b = boxes[index, :2], boxes[index, 2:]
            x1, y1 = np.maximum(0, a + (b-a)*[.22, .22]).astype(int)
            x2, y2 = np.minimum(image.shape[1::-1], a + (b-a)*[.78, .70]).astype(int)
            if x2-x1 < 8 or y2-y1 < 8:
                continue
            colors = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
            hist = cv2.calcHist([colors], [1, 2], None, [12, 12], [64, 192, 64, 192])
            if hist.sum():
                samples.append(hist/hist.sum())
        return np.mean(samples, axis=0) if len(samples) >= 2 else None

    try:
        for frame in candidate["cut_handoffs"]:
            i = int(np.searchsorted(frames, frame))
            left = histogram(range(max(0, i-5), i))
            right = histogram(range(i, min(len(frames), i+5)))
            if left is None or right is None:
                return False
            distances.append(cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA))
    finally:
        cap.release()
    # A geometrically possible handoff is only a suggestion without matching
    # visual evidence. These are normalized color distances, not confidence.
    return bool(distances and max(distances) <= .50 and np.median(distances) <= .36)
