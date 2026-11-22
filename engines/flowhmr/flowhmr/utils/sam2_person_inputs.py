"""Lossless SAM2 person inputs shared by inspection and motion conditioning.

Packed masks retain original full-resolution binary pixels. Only the requested
batch is decoded, so a long 1080p clip need not materialize all masks in RAM.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PackedMasks:
    ndim = 3
    dtype = np.dtype("uint8")

    def __init__(self, packed, height, width, first=0, stop=None):
        self.packed = packed
        self.height, self.width = int(height), int(width)
        self.first, self.stop = int(first), len(packed) if stop is None else int(stop)
        if not 0 <= self.first <= self.stop <= len(packed):
            raise ValueError("Invalid packed mask slice")
        if packed.ndim != 2 or packed.shape[1] != (self.height * self.width + 7) // 8:
            raise ValueError("Packed masks do not match image dimensions")
        self.shape = (self.stop - self.first, self.height, self.width)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = np.arange(len(self))[item] + self.first
            packed = self.packed[indices]
            return np.unpackbits(packed, axis=-1, count=self.height * self.width).reshape(len(indices), self.height, self.width)
        index = int(item)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return np.unpackbits(self.packed[self.first + index], count=self.height * self.width).reshape(self.height, self.width)

    def close(self):
        """Release this mapped file after all windows sharing it are finished."""
        mapping = getattr(self.packed, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    def window(self, first, stop):
        if not 0 <= first <= stop <= len(self):
            raise ValueError("Mask window is outside prepared input")
        return PackedMasks(self.packed, self.height, self.width, self.first + first, self.first + stop)


def load_person_inputs(directory, video_path=None):
    """Load prepared tracks and masks only after non-mutating identity validation."""
    from .runtime_video_processing import TrackInfo, TrackingResult
    from .sam2_identity_guard import validate_identity_masks

    root = Path(directory)
    meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if meta.get("version") != VERSION:
        raise ValueError("Unsupported SAM2 input version")
    if video_path is not None and file_sha256(video_path) != meta["video_sha256"]:
        raise ValueError("SAM2 input belongs to a different clean video; annotated previews cannot be inference input")
    width, height = map(int, meta["image_size"])
    total = int(meta["num_frames"])
    if width < 1 or height < 1 or total < 1:
        raise ValueError("Prepared SAM2 source dimensions and frame count must be positive")
    tracks, masks, opened, actual_hashes = [], {}, [], {}
    try:
        for person in meta["people"]:
            cid = int(person["canonical_id"])
            if cid in masks:
                raise ValueError("Prepared identities must have distinct canonical IDs")
            # Always hash actual mask bytes, including old manifests without a
            # checksum. A stale certificate cannot bless an edited mask file.
            actual_hashes[cid] = file_sha256(root / person["mask_file"])
            if "mask_sha256" in person and actual_hashes[cid] != person["mask_sha256"]:
                raise ValueError(f"Prepared SAM2 input changed: {person['mask_file']}")
            if "boxes_sha256" in person and file_sha256(root / person["boxes_file"]) != person["boxes_sha256"]:
                raise ValueError(f"Prepared SAM2 input changed: {person['boxes_file']}")
            first, last = map(int, person["frame_range"])
            if not 0 <= first <= last < total:
                raise ValueError("Prepared person frame range is outside the clean source")
            with np.load(root / person["boxes_file"], allow_pickle=False) as data:
                frames = data["frames"].copy()
                boxes = data["body_boxes"].astype(np.float32)
            if not np.array_equal(frames, np.arange(first, last + 1)) or boxes.shape != (len(frames), 4):
                raise ValueError("Prepared boxes have inconsistent source frame range")
            if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError("Prepared body boxes are invalid")
            packed = np.load(root / person["mask_file"], mmap_mode="r", allow_pickle=False)
            opened.append(packed)
            stream = PackedMasks(packed, height, width)
            if len(stream) != total:
                raise ValueError("Prepared masks do not cover the clean source")
            track = TrackInfo(track_id=cid, frames=frames, bboxes=boxes,
                              length=len(frames), avg_area=float(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1).mean()),
                              bytetrack_ids=tuple(person["bytetrack_ids"]), sam2_object_ids=(int(person["object_id"]),))
            tracks.append(track)
            masks[cid] = stream
        validate_identity_masks(meta["people"], masks, actual_hashes,
                                cache_path=root / "identity_guard.json")
        result = TrackingResult(tracks, total, width, height, (0, total - 1))
        return result, masks, meta
    except BaseException:
        for packed in opened:
            mapping = getattr(packed, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        raise


def save_prepared_track(path, result, track_index, frame_range, meta):
    from .runtime_video_processing import VideoProcessor
    VideoProcessor.save_detected_track(path, result, track_index, frame_range)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    payload["sam2_input_key"] = np.asarray(meta["run_id"])
    np.savez(path, **payload)


def prepared_detection_matches(path, frame_range, meta):
    from .runtime_capabilities import cached_detection_matches
    if not cached_detection_matches(path, frame_range):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["sam2_input_key"]) == meta["run_id"]
    except (OSError, ValueError, KeyError):
        return False


def mask_stream_for_bbox(directory, bbox_file, video_path):
    """Read the selected identity from its persisted bbox, including batch children."""
    result, streams, meta = load_person_inputs(directory, video_path)
    try:
        with np.load(bbox_file, allow_pickle=False) as data:
            if "track_id" not in data:
                raise ValueError("Prepared SAM2 inputs need a selected canonical track_id")
            track_id = int(data["track_id"])
            start, end = map(int, data["start_end"])
        if track_id not in streams:
            raise ValueError("Selected identity is absent from SAM2 inputs")
        selected = streams[track_id].window(start, end + 1)
        for cid, stream in streams.items():
            if cid != track_id:
                stream.close()
        return selected, meta
    except BaseException:
        for stream in streams.values():
            stream.close()
        raise
