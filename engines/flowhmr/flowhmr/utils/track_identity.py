"""Conservative ByteTrack fragment linking using continuous SAM2 identities.

Only raw observed ByteTrack boxes belong here: do not interpolate frames or use
held crop boxes as identity evidence. SAM2 keeps its frame entries even when the
person has no visible pixels; those entries prove coverage, not a visible match.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from flowhmr.utils.runtime_video_processing import TrackInfo, TrackingResult


@dataclass(frozen=True)
class Sam2Object:
    """One uninterrupted SAM2 object, in ByteTrack's image coordinates.

    ``frames`` includes processed empty-mask frames. ``visible=False`` excludes
    a frame from matching without breaking tracking continuity. Masks, when
    supplied, supply the visible evidence even if full-body boxes also exist.
    """
    object_id: int
    frames: np.ndarray
    boxes: Optional[np.ndarray] = None
    masks: Optional[np.ndarray] = None
    visible: Optional[np.ndarray] = None

    def __post_init__(self):
        frames = np.asarray(self.frames, dtype=np.int64).reshape(-1)
        if not len(frames) or np.any(np.diff(frames) <= 0):
            raise ValueError("SAM2 object frames must be nonempty and strictly increasing")
        if self.boxes is None and self.masks is None:
            raise ValueError("SAM2 object needs boxes or masks")
        if self.boxes is not None:
            boxes = np.asarray(self.boxes, dtype=np.float32)
            if boxes.shape != (len(frames), 4) or not np.isfinite(boxes).all():
                raise ValueError("SAM2 object boxes must be finite (frames, 4)")
            object.__setattr__(self, "boxes", boxes)
        if self.masks is not None:
            from .sam2_person_inputs import PackedMasks

            masks = self.masks if isinstance(self.masks, PackedMasks) else np.asarray(self.masks)
            if masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks[..., 0]
            if masks.ndim != 3 or masks.shape[0] != len(frames):
                raise ValueError("SAM2 object masks must have shape (frames, H, W)")
            object.__setattr__(self, "masks", masks)
            object.__setattr__(self, "masks", masks)
        if self.visible is not None:
            visible = np.asarray(self.visible, dtype=bool).reshape(-1)
            if len(visible) != len(frames):
                raise ValueError("SAM2 object visibility must match frames")
            object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "frames", frames)

    def box_at(self, index: int) -> Optional[np.ndarray]:
        if self.boxes is not None:
            return self.boxes[index]
        return _mask_box(self.masks[index])

    def visible_at(self, index: int) -> bool:
        if self.visible is not None and not self.visible[index]:
            return False
        if self.masks is not None:
            return bool(np.any(self.masks[index]))
        box = self.box_at(index)
        return box is not None and bool(np.all(box[2:] > box[:2]))


@dataclass(frozen=True)
class IdentityMerge:
    canonical_id: int
    bytetrack_ids: Tuple[int, ...]
    sam2_object_ids: Tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class IdentityResolution:
    tracks: Tuple[TrackInfo, ...]
    merges: Tuple[IdentityMerge, ...]
    ambiguous: Tuple[int, ...] = ()


def _mask_box(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    # Exclusive maxima match detector xyxy box geometry.
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def _box_score(track_box: np.ndarray, object_box: np.ndarray) -> float:
    """IoU, never containment: a tiny fragment inside a body is not an anchor."""
    a, b = np.asarray(track_box), np.asarray(object_box)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    if np.any(a[2:] <= a[:2]) or np.any(b[2:] <= b[:2]):
        return 0.0
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    inter = float(np.prod(wh))
    return inter / max(float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - inter, 1e-9)


def observation_match_score(box: np.ndarray, obj: Sam2Object, index: int) -> float:
    """Visible-observation score reusable by callers; no temporal inference.

    Uses visible-mask bbox IoU plus a mask-support check (>=60% of the mask
    pixels inside the observed box). Hidden flags, empty masks and tiny
    contained fragments yield no confident anchor. Supplied held boxes cannot
    override a small mask when both mask and boxes are present.
    """
    if not obj.visible_at(index):
        return 0.0
    evidence = _mask_box(obj.masks[index]) if obj.masks is not None else obj.box_at(index)
    if evidence is None:
        return 0.0
    score = _box_score(np.asarray(box, np.float32), evidence)
    if obj.masks is not None and score > 0:
        mask = obj.masks[index]
        h, w = mask.shape
        x1, y1, x2, y2 = np.asarray(box, dtype=float)
        x1, x2 = max(0, int(np.floor(x1))), min(w, int(np.ceil(x2)))
        y1, y2 = max(0, int(np.floor(y1))), min(h, int(np.ceil(y2)))
        support = np.count_nonzero(mask[y1:y2, x1:x2]) if x2 > x1 and y2 > y1 else 0
        if support / max(np.count_nonzero(mask), 1) < .6:
            return 0.0
    return score


def _longest_run(frames):
    longest = current = 0
    previous = None
    for frame in sorted(frames):
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        previous = frame
        longest = max(longest, current)
    return longest


def _covered_gap(obj: Sam2Object, first: int, last: int) -> bool:
    if last < first:
        return True
    lo = int(np.searchsorted(obj.frames, first))
    hi = int(np.searchsorted(obj.frames, last, side="right"))
    return hi - lo == last - first + 1


def _copy_track(track):
    return replace(track, frames=np.asarray(track.frames).copy(), bboxes=np.asarray(track.bboxes).copy())


def _canonical_track(fragments, obj):
    fragments = sorted(fragments, key=lambda t: (int(t.frames.min()), int(t.track_id)))
    frames = np.concatenate([t.frames for t in fragments])
    boxes = np.concatenate([t.bboxes for t in fragments])
    order = np.argsort(frames)
    boxes, frames = boxes[order].astype(np.float32), frames[order].astype(np.int64)
    ids = tuple(dict.fromkeys(v for t in fragments for v in (t.bytetrack_ids or (t.track_id,))))
    object_ids = tuple(dict.fromkeys([v for t in fragments for v in t.sam2_object_ids] + [obj.object_id]))
    return TrackInfo(track_id=int(fragments[0].track_id), frames=frames, bboxes=boxes,
                     length=len(frames), avg_area=float(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1).mean()),
                     bytetrack_ids=ids, sam2_object_ids=object_ids)


def merge_sam2_identities(tracks: Iterable[TrackInfo], objects: Sequence[Sam2Object], *,
                          min_score: float = .25, min_anchor_frames: int = 3,
                          min_match_ratio: float = .6) -> IdentityResolution:
    """Link fragments with >=3 consecutive unique visible matches by default.

    Each match must beat every other visible SAM2 object AND simultaneous raw
    ByteTrack box by >0.15 IoU. >=60% of available visible observations must
    support the assignment. A second object's qualifying run vetoes assignment.
    A temporal overlap anywhere in one object's assigned group blocks its whole
    merge (including unsafe three-fragment chains). Coverage across each gap
    must be complete, but empty masks are allowed there. Inputs are never edited.
    """
    tracks, objects = list(tracks), list(objects)
    if min_anchor_frames < 1 or int(min_anchor_frames) != min_anchor_frames:
        raise ValueError("min_anchor_frames must be a positive integer")
    if not 0 <= min_match_ratio <= 1 or not 0 <= min_score <= 1:
        raise ValueError("match thresholds must be between 0 and 1")
    if len({o.object_id for o in objects}) != len(objects):
        raise ValueError("SAM2 object IDs must be unique within one tracking run")
    for t in tracks:
        if len(t.frames) != len(t.bboxes) or len(np.unique(t.frames)) != len(t.frames):
            raise ValueError("Track frames must be unique raw observations matching its boxes")
    positions = [{int(f): i for i, f in enumerate(o.frames)} for o in objects]
    observations = {}
    for ti, track in enumerate(tracks):
        for f, box in zip(track.frames, track.bboxes):
            observations.setdefault(int(f), []).append((ti, box))
    evidence = [[[] for _ in objects] for _ in tracks]
    potential = [[[] for _ in objects] for _ in tracks]
    available = np.zeros((len(tracks), len(objects)), int)
    ambiguous = set()
    for frame, entries in observations.items():
        scores = np.zeros((len(entries), len(objects)), float)
        for oi, obj in enumerate(objects):
            index = positions[oi].get(frame)
            if index is None or not obj.visible_at(index):
                continue
            for ei, (ti, box) in enumerate(entries):
                available[ti, oi] += 1
                scores[ei, oi] = observation_match_score(box, obj, index)
        for ei, (ti, _) in enumerate(entries):
            if not len(objects):
                continue
            oi = int(np.argmax(scores[ei]))
            score = scores[ei, oi]
            if score < min_score:
                continue
            other_obj = np.delete(scores[ei], oi).max(initial=0)
            other_track = np.delete(scores[:, oi], ei).max(initial=0)
            if score - other_obj > .15:
                potential[ti][oi].append(frame)
            if score - max(other_obj, other_track) <= .15:
                ambiguous.add(int(tracks[ti].track_id))
                continue
            evidence[ti][oi].append(frame)
    assignments = {}
    mixed_tracks = set()
    for ti, track in enumerate(tracks):
        qualifying = [oi for oi, frames in enumerate(evidence[ti])
                      if _longest_run(frames) >= min_anchor_frames]
        if len(qualifying) > 1:
            mixed_tracks.add(ti)
            ambiguous.add(int(track.track_id))
            continue
        if len(qualifying) == 1:
            oi = qualifying[0]
            if len(evidence[ti][oi]) / max(available[ti, oi], 1) >= min_match_ratio:
                assignments[ti] = oi
    groups = {}
    for ti, oi in assignments.items():
        groups.setdefault(oi, []).append(ti)
    output, merges, consumed = [], [], set()
    for oi, indices in groups.items():
        indices.sort(key=lambda ti: (int(tracks[ti].frames.min()), tracks[ti].track_id))
        # Check the whole group, not only adjacent links after splitting groups.
        max_end = -1
        conflict = False
        contenders = [ti for ti in range(len(tracks)) if ti not in mixed_tracks
                      and _longest_run(potential[ti][oi]) >= min_anchor_frames]
        contenders.sort(key=lambda ti: int(tracks[ti].frames.min()))
        for ti in contenders:
            if int(tracks[ti].frames.min()) <= max_end:
                conflict = True
            max_end = max(max_end, int(tracks[ti].frames.max()))
        if conflict:
            ambiguous.update(int(tracks[ti].track_id) for ti in contenders)
            continue
        chains = []
        for ti in indices:
            if (not chains or not _covered_gap(objects[oi],
                    int(tracks[chains[-1][-1]].frames.max()) + 1, int(tracks[ti].frames.min()) - 1)):
                chains.append([ti])
            else:
                chains[-1].append(ti)
        for chain in chains:
            merged = _canonical_track([tracks[ti] for ti in chain], objects[oi])
            output.append(merged)
            consumed.update(chain)
            if len(chain) > 1:
                merges.append(IdentityMerge(merged.track_id, merged.bytetrack_ids, merged.sam2_object_ids,
                                             int(merged.frames.min()), int(merged.frames.max())))
    output.extend(_copy_track(t) for ti, t in enumerate(tracks) if ti not in consumed)
    output.sort(key=lambda t: (int(t.frames.min()) if len(t.frames) else -1, int(t.track_id)))
    return IdentityResolution(tuple(output), tuple(merges), tuple(sorted(ambiguous)))


def merge_result_identities(result: TrackingResult, objects: Sequence[Sam2Object], *,
                            min_score: float = .25, min_anchor_frames: int = 3,
                            min_match_ratio: float = .6) -> TrackingResult:
    resolved = merge_sam2_identities(result.tracks, objects, min_score=min_score,
                                    min_anchor_frames=min_anchor_frames, min_match_ratio=min_match_ratio)
    return TrackingResult(list(resolved.tracks), result.total_frames, result.width,
                          result.height, result.scanned_range)
