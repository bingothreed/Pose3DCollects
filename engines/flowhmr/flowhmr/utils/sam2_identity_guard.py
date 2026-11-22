"""Reject sustained duplicate SAM2 identities without changing any target pixels.

This gate is independent of detector-ID linking. It compares packed visible
masks before body conditioning; small fragments and empty occlusions remain
unchanged. A passing certificate proves only absence of this clone criterion,
not semantic identity correctness.
"""
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import hashlib
import json
import os
import uuid

import numpy as np


GUARD_VERSION = 1
_POPCOUNT = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class IdentityGuardSettings:
    min_iou: float = .7
    min_intersection_fraction: float = .85
    consecutive_frames: int = 3

    def __post_init__(self):
        if not 0 <= self.min_iou <= 1 or not 0 <= self.min_intersection_fraction <= 1:
            raise ValueError("Identity guard overlap thresholds must be between zero and one")
        if type(self.consecutive_frames) is not int or self.consecutive_frames < 1:
            raise ValueError("Identity guard consecutive_frames must be a positive integer")


class Sam2IdentityCollisionError(ValueError):
    def __init__(self, collisions):
        self.collisions = collisions
        details = []
        for item in collisions:
            labels = "/".join(f"S{oid}" for oid in item["object_ids"])
            canonical = "/".join(map(str, item["canonical_ids"]))
            ranges = ", ".join(f"{first}–{last}" for first, last in item["segments"])
            details.append(f"{labels} (canonical {canonical}), source frames {ranges}")
        super().__init__("SAM2 identity collision: " + "; ".join(details)
                         + ". Sustained duplicate visible masks cannot condition separate people; "
                           "recover identities from independent unique observations and reseed SAM2 "
                           "before generating motion (frames are zero-based).")


def _packed_rows(stream, first, stop):
    if stream.packed.dtype != np.uint8:
        raise ValueError("Identity guard requires uint8 packed masks")
    rows = stream.packed[stream.first + first:stream.first + stop]
    # np.packbits pads its last byte. Ignore unused bits without modifying file.
    remainder = (stream.height * stream.width) % 8
    if remainder:
        rows = rows.copy()
        rows[:, -1] &= np.uint8((255 << (8 - remainder)) & 255)
    return rows


def _runs(indices, minimum):
    segments = []
    first = previous = None
    for value in indices:
        frame = int(value)
        if previous is not None and frame != previous + 1:
            if previous - first + 1 >= minimum:
                segments.append([first, previous])
            first = frame
        if first is None:
            first = frame
        previous = frame
    if previous is not None and previous - first + 1 >= minimum:
        segments.append([first, previous])
    return segments


def find_identity_collisions(people, streams, settings=None):
    """Scan actual packed pixels on CPU; inactive/empty rows break the run.

    Streams are keyed by canonical ID and cover the same source timeline.
    Pair criteria use strict ``>`` thresholds; no minimum area is imposed.
    Tiny identical pieces can be ambiguous briefly, so only sustained runs fail.
    """
    settings = settings or IdentityGuardSettings()
    people = list(people)
    if len({int(p["canonical_id"]) for p in people}) != len(people):
        raise ValueError("Prepared identities must have distinct canonical IDs")
    if len({int(p["object_id"]) for p in people}) != len(people):
        raise ValueError("Prepared identities must have distinct SAM2 object IDs")
    sizes = {streams[int(person["canonical_id"])].shape for person in people}
    if len(sizes) > 1:
        raise ValueError("Identity guard masks must use the same source timeline and resolution")
    collisions = []
    block_size = 32
    for left, right in combinations(people, 2):
        a, b = streams[int(left["canonical_id"])], streams[int(right["canonical_id"])]
        first = max(int(left["frame_range"][0]), int(right["frame_range"][0]), 0)
        stop = min(int(left["frame_range"][1]), int(right["frame_range"][1]), len(a) - 1) + 1
        if stop <= first:
            continue
        suspicious = []
        highest_iou = highest_fraction = 0.0
        for begin in range(first, stop, block_size):
            end = min(begin + block_size, stop)
            ap, bp = _packed_rows(a, begin, end), _packed_rows(b, begin, end)
            area_a = _POPCOUNT[ap].sum(axis=1, dtype=np.int64)
            area_b = _POPCOUNT[bp].sum(axis=1, dtype=np.int64)
            intersection = _POPCOUNT[np.bitwise_and(ap, bp)].sum(axis=1, dtype=np.int64)
            union = area_a + area_b - intersection
            fraction = intersection / np.maximum(np.minimum(area_a, area_b), 1)
            iou = intersection / np.maximum(union, 1)
            bad = (area_a > 0) & (area_b > 0) & (iou > settings.min_iou) & (fraction > settings.min_intersection_fraction)
            suspicious.extend((np.flatnonzero(bad) + begin).tolist())
            if np.any(bad):
                highest_iou = max(highest_iou, float(iou[bad].max()))
                highest_fraction = max(highest_fraction, float(fraction[bad].max()))
        segments = _runs(suspicious, settings.consecutive_frames)
        if segments:
            collisions.append(dict(object_ids=[int(left["object_id"]), int(right["object_id"])],
                                   canonical_ids=[int(left["canonical_id"]), int(right["canonical_id"])],
                                   segments=segments, max_iou=highest_iou,
                                   max_intersection_fraction=highest_fraction))
    return collisions


def validate_identity_masks(people, streams, actual_mask_hashes, *, cache_path=None, settings=None):
    """Mandatory prepared-input guard, cached by freshly computed file hashes.

    Callers must supply hashes computed from actual files in this load, not
    copy unverified manifest claims. Cache writes are optional: a read-only
    destination still performs validation normally. Both passing and failing
    scans can be reused while file content and settings remain identical.
    """
    settings = settings or IdentityGuardSettings()
    people = list(people)
    identities = []
    for person in people:
        cid = int(person["canonical_id"])
        digest = actual_mask_hashes[cid]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Identity guard needs an actual SHA256 for every mask file")
        stream = streams[cid]
        identities.append(dict(object_id=int(person["object_id"]), canonical_id=cid,
                               active_range=list(map(int, person["frame_range"])),
                               shape=list(stream.shape), first=int(stream.first), sha256=digest))
    payload = dict(version=GUARD_VERSION, settings=asdict(settings), identities=identities)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    certificate = None
    target = Path(cache_path) if cache_path is not None else None
    if target is not None:
        try:
            candidate = json.loads(target.read_text(encoding="utf-8"))
            if candidate.get("cache_key") == key and isinstance(candidate.get("collisions"), list):
                certificate = candidate
        except (OSError, ValueError, AttributeError):
            pass
    if certificate is None:
        collisions = find_identity_collisions(people, streams, settings)
        certificate = dict(cache_key=key, **payload, collisions=collisions)
        if target is not None:
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
                os.replace(temporary, target)
            except OSError:
                # Validation remains mandatory even when certificates cannot persist.
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    if certificate["collisions"]:
        raise Sam2IdentityCollisionError(certificate["collisions"])
    return certificate
