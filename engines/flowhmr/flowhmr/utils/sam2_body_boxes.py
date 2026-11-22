"""Causal full-body crop boxes for partly visible SAM 2 people.

No held crop.  Every frame follows the current visible mask: body SCALE comes
from the most recent reliable full-body observation, never a stale POSITION.
Vertical extent is trusted and unlimited; width is bounded by the height prior
(a person's arm span is about their body height), and a mask that expands past
that bound on one side pulls the stale opposite side in.

Visibility/identity decisions belong to the caller.  Coordinates are image-space
xyxy with positive width/height; zero or non-finite rows denote absent masks.
"""
from dataclasses import dataclass
import logging

import numpy as np

# Developer-facing only: the GUI never shows these.  Tests and CLI logs do.
logger = logging.getLogger(__name__)

# Bump when the crop policy changes; prepared-input caches must match it.
BODY_POLICY_VERSION = "causal_union_nohold_v2"
# Arm span ~= body height, so the crop width is bounded by the trusted height.
ARM_SPAN_RATIO = 1.05
# Only a substantial silhouette may pull the crop sideways; a head fragment or a
# stray speck keeps the body-scale centre.
_MIN_MASK_HEIGHT = .25
_MIN_MASK_WIDTH = .10


@dataclass(frozen=True)
class BodyBoxResult:
    """One row per input frame, without backfilling uninitialized frames.

    ``ready`` must be checked before using a box for body inference.  Before the
    first reliable full-body observation a visible fragment is returned as a
    diagnostic box; ``reference_boxes`` identifies the last reliable full-body
    box that supplied the body scale.
    """
    body_boxes: np.ndarray
    states: np.ndarray
    reference_boxes: np.ndarray
    ready: np.ndarray
    has_mask: np.ndarray
    reference_frames: tuple


def _boxes(value, count=None, name="mask_boxes"):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"{name} must have shape (frames, 4)")
    if count is not None and len(result) != count:
        raise ValueError(f"{name} must match mask_boxes frame count")
    return result


def _valid(boxes):
    return np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)


def _union(first, second):
    return np.concatenate((np.minimum(first[:2], second[:2]),
                           np.maximum(first[2:], second[2:])))


def _body_scale_box(reference, mask, extent):
    """Body-scale crop centred on the current visible mask.

    Returns ``(box, clamped)``.  ``clamped`` is true when the height prior
    bounded the width and the stale side of the reference was pulled in.
    """
    mask_size = mask[2:] - mask[:2]
    height = float(max(extent[1], mask_size[1]))
    limit = height * ARM_SPAN_RATIO
    requested = float(max(extent[0], mask_size[0]))
    centre = float((mask[0] + mask[2]) * .5)
    top = float(min(reference[1], mask[1]))
    bottom = float(max(reference[3], mask[3]))
    if requested <= limit or mask_size[0] > limit:
        # Width fits the prior, or the visible silhouette itself is wider than
        # the prior: keep the evidence and trust it.
        width = requested
        clamped = False
    else:
        left_overflow = max(0.0, float(reference[0] - mask[0]))
        right_overflow = max(0.0, float(mask[2] - reference[2]))
        width = limit
        clamped = True
        if right_overflow > 0 and left_overflow <= 0:
            # Evidence moved right: keep its right edge, drop the stale left.
            box = np.asarray([mask[2] - width, top, mask[2], bottom], dtype=np.float32)
            return _union(box, mask), clamped
        if left_overflow > 0 and right_overflow <= 0:
            box = np.asarray([mask[0], top, mask[0] + width, bottom], dtype=np.float32)
            return _union(box, mask), clamped
    box = np.asarray([centre - width * .5, top, centre + width * .5, bottom], dtype=np.float32)
    return _union(box, mask), clamped


def build_body_boxes(mask_boxes, reliable_visible, detector_boxes=None, *,
                     uninitialized="mark"):
    """Build full-body crop boxes using only observations up to each frame.

    Args:
        mask_boxes: ``(T, 4)`` CURRENT visible target SAM2 bounds.  Missing masks
            use all-zero or NaN rows.
        reliable_visible: ``(T,)`` booleans supplied by the caller: the target
            has reliable FULL-body visibility in this frame.  A missing mask
            overrides True.
        detector_boxes: optional ``(T, 4)`` correctly matched raw observations,
            unioned into reliable full-body crops and used as the fallback when
            no mask is available.
        uninitialized: ``"mark"`` returns ready=False before a reliable initial
            observation; ``"raise"`` rejects such a prefix.

    Reliable frames use the union of the mask and the matched detector box.
    Partly visible frames use a body-scale crop centred on the current mask
    (body scale from the last reliable full-body observation).  A frame without
    any usable observation keeps the previous crop.  A hidden ``logging``
    warning reports how often the arm-span prior clamped the crop.
    """
    masks = _boxes(mask_boxes)
    count = len(masks)
    flags = np.asarray(reliable_visible)
    if flags.shape != (count,) or not np.isin(flags, (False, True)).all():
        raise ValueError("reliable_visible must contain one boolean per frame")
    flags = flags.astype(bool)
    if uninitialized not in ("mark", "raise"):
        raise ValueError("uninitialized must be 'mark' or 'raise'")
    detectors = (np.full_like(masks, np.nan) if detector_boxes is None else
                 _boxes(detector_boxes, count, "detector_boxes"))
    has_mask, has_detector = _valid(masks), _valid(detectors)
    reliable = flags & has_mask
    output = np.full_like(masks, np.nan)
    references = np.full_like(masks, np.nan)
    states = np.empty(count, dtype="<U20")
    ready = np.zeros(count, dtype=bool)
    sources = []
    reference = None
    extent = None
    last_box = None
    clamped_frames = []

    for frame in range(count):
        mask = masks[frame]
        if reference is None and not reliable[frame]:
            if uninitialized == "raise":
                raise ValueError(f"frame {frame}: no reliable initial full-body box")
            if has_mask[frame]:
                output[frame] = mask
            states[frame] = "not_ready"
            sources.append(())
            continue
        if reliable[frame]:
            box = _union(mask, detectors[frame]) if has_detector[frame] else mask.copy()
            reference = box.copy()
            extent = box[2:] - box[:2]
            output[frame] = box
            references[frame] = reference
            states[frame] = "visible"
            sources.append((frame,))
        elif has_mask[frame]:
            box, clamped = _body_scale_box(reference, mask, extent)
            if clamped:
                clamped_frames.append(frame)
            output[frame] = box
            references[frame] = reference
            states[frame] = "estimated"
            sources.append(())
        elif has_detector[frame]:
            output[frame] = detectors[frame].copy()
            references[frame] = reference
            states[frame] = "detector"
            sources.append(())
        else:
            fallback = last_box if last_box is not None else reference
            output[frame] = np.array(fallback, dtype=np.float32, copy=True)
            references[frame] = reference
            states[frame] = "fallback"
            sources.append(())
        ready[frame] = True
        last_box = output[frame].copy()

    if clamped_frames:
        # Hidden from GUI users; visible in the application/CLI logs for triage.
        logger.warning("arm-span prior clamped the body crop on %d frame(s) (%d..%d, ratio %.2f)",
                       len(clamped_frames), clamped_frames[0], clamped_frames[-1], ARM_SPAN_RATIO)
    return BodyBoxResult(output, states, references, ready, has_mask, tuple(sources))
