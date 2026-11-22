"""Experimental crossing optimisation for the GUI.

Replaces plain ByteTrack boxes with SAM 2.1 person masks and two things built on
top of them:

* causal body crop boxes: median of the last five reliable full-body frames is
  held during occlusion and expanded by the currently visible mask, so a person
  who rises behind someone else keeps a body-sized crop;
* SAM 2 guided identity merge: ByteTrack fragments that one continuously tracked
  SAM2 object covers are joined, and fragments that clearly belong to somebody
  else are never used as one person's evidence.

Everything is prepared before the normal pipeline runs, so the existing GUI
pipeline picks the result up unchanged.  Failures raise ``CrossingInputsError``
and the caller falls back to the standard detection flow.
"""
from pathlib import Path
import contextlib
import gc
import hashlib
import json
import os
import shutil
import tempfile
import time

import cv2
import numpy as np
import torch

from .sam2_body_boxes import ARM_SPAN_RATIO, BODY_POLICY_VERSION, build_body_boxes
from .sam2_frame_store import bounded_state
from .sam2_person_inputs import PackedMasks, file_sha256, load_person_inputs
from .track_identity import Sam2Object, merge_sam2_identities, observation_match_score

SAM2_MODEL = "sam2.1_hiera_small"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = Path("models") / "sam2" / "sam2.1_hiera_small.pt"

MAX_PEOPLE = 6
SEED_FRAME_TOLERANCE = 10
MIN_SEED_FRAMES = 15

_MATCH_MIN_SCORE = .25
_MATCH_MIN_MARGIN = .15
_RELIABLE_SCORE = .45
_MAX_FOREIGN_FRACTION = .30
_OVERLAP_COVER = .25
_OVERLAP_RUN = 5
_ANCHOR_WINDOW = 60
# A mask this small relative to its own crop carries no usable body evidence.
_OUTAGE_MIN_PIXELS = 64
_OUTAGE_MIN_FRACTION = .01
MAX_OUTAGE_RUN = 10            # consecutive frames
MAX_OUTAGE_FRACTION = .05      # share of the whole clip


class CrossingInputsError(ValueError):
    """The clip cannot be prepared reliably; callers should fall back."""


def _bf16_autocast(device):
    """bf16 by default, plain fp32 when the backend cannot do bf16.

    ROCm builds and CPU-only installs do not always expose bf16 kernels, and a
    hidden fallback here keeps the optimisation usable there instead of failing.
    """
    try:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16)
    except (AssertionError, RuntimeError):
        pass
    return contextlib.nullcontext()


def _release_cuda():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def mask_outage_stats(mask_areas, body_boxes):
    """Find frames where the SAM2 mask is too small to describe the body crop."""
    areas = np.asarray(mask_areas, dtype=np.float64).reshape(-1)
    boxes = np.asarray(body_boxes, dtype=np.float64)
    if boxes.shape[0] != areas.size:
        raise ValueError("mask areas and body boxes must have the same length")
    width = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    height = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    threshold = np.maximum(_OUTAGE_MIN_PIXELS, _OUTAGE_MIN_FRACTION * width * height)
    outage = areas < threshold
    longest = run = start = best_start = 0
    for index, flag in enumerate(outage):
        if flag:
            if run == 0:
                start = index
            run += 1
            if run > longest:
                longest, best_start = run, start
        else:
            run = 0
    return dict(outage=outage, frames=int(np.count_nonzero(outage)),
                fraction=float(np.count_nonzero(outage) / max(outage.size, 1)),
                longest_run=int(longest),
                run_start=int(best_start) if longest else -1,
                run_end=int(best_start + longest - 1) if longest else -1)


def check_mask_quality(label, mask_areas, body_boxes):
    """Reject a person whose segmentation collapses for a meaningful stretch.

    Holding a body box while the mask is effectively empty keeps the crop on a
    stale position, which usually covers the other dancer during a crossing; the
    estimator would then predict a smooth generic motion.  Falling back to the
    standard detection flow is both safer and what users saw before the toggle.
    """
    stats = mask_outage_stats(mask_areas, body_boxes)
    if stats["longest_run"] > MAX_OUTAGE_RUN or stats["fraction"] > MAX_OUTAGE_FRACTION:
        raise CrossingInputsError(
            f"{label} 在第 {stats['run_start']}–{stats['run_end']} 帧丢失分割"
            f"（{stats['longest_run']} 帧几乎没有可见像素，占全片 {stats['fraction']:.1%}），"
            "交叉优化已回退默认流程，避免用错误的分割生成动作")
    return stats


def _pair_metrics(first, second):
    """Return (separation score, coverage of the smaller box)."""
    width_height = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]))
    intersection = float(np.prod(width_height))
    area_a = float(np.prod(first[2:] - first[:2]))
    area_b = float(np.prod(second[2:] - second[:2]))
    iou = intersection / max(area_a + area_b - intersection, 1)
    centre_distance = np.linalg.norm((first[:2] + first[2:] - second[:2] - second[2:]) / 2)
    height = max((first[3] - first[1] + second[3] - second[1]) / 2, 1)
    return float(centre_distance / height - .5 * iou), intersection / max(min(area_a, area_b), 1)


def select_seed_tracks(tracks, *, max_people=MAX_PEOPLE):
    """People present from the opening frames, ordered by first appearance."""
    seeds = [track for track in tracks
             if int(track.frames.min()) <= SEED_FRAME_TOLERANCE and len(track.frames) >= MIN_SEED_FRAMES]
    seeds.sort(key=lambda track: (int(track.frames.min()), -track.length, int(track.track_id)))
    if len(seeds) > max_people:
        raise CrossingInputsError(
            f"画面里同时出现了 {len(seeds)} 个初始人物，超过交叉优化支持的上限 {max_people} 人")
    return seeds


def _boxes_by_track(seeds):
    return {track.track_id: {int(frame): box for frame, box in zip(track.frames, track.bboxes)}
            for track in seeds}


def choose_anchor_frame(seeds):
    """Pick a separated prompt frame before the first sustained overlap.

    Returns ``(anchor, ordered_seeds, prompts)`` where prompts are xyxy boxes in
    the same order as ``ordered_seeds`` (left to right).  No post-occlusion or
    reappearance frame is ever used as a prompt.
    """
    if not seeds:
        raise CrossingInputsError("没有可用于初始化的稳定人物轨迹")
    by_track = _boxes_by_track(seeds)
    common = sorted(set.intersection(*(set(by_track[track.track_id]) for track in seeds)))
    if not common:
        raise CrossingInputsError("初始人物轨迹没有共同的起始帧，无法建立稳定身份")
    count = len(seeds)
    if count == 1:
        anchor = common[0]
    else:
        run = 0
        event = None
        for frame in common:
            boxes = [by_track[track.track_id][frame] for track in seeds]
            cover = max(_pair_metrics(boxes[i], boxes[j])[1]
                        for i in range(count) for j in range(i + 1, count))
            run = run + 1 if cover >= _OVERLAP_COVER else 0
            if run >= _OVERLAP_RUN:
                event = frame - _OVERLAP_RUN + 1
                break
        candidates = [frame for frame in common
                      if event is None or max(common[0], event - _ANCHOR_WINDOW) <= frame <= event]
        if not candidates:
            raise CrossingInputsError("没有找到人物彼此分离的初始化帧")
        anchor = max(candidates, key=lambda frame: min(
            _pair_metrics(by_track[seeds[i].track_id][frame], by_track[seeds[j].track_id][frame])[0]
            for i in range(count) for j in range(i + 1, count)))
    ordered = sorted(seeds, key=lambda track: float(by_track[track.track_id][anchor][0]))
    prompts = [by_track[track.track_id][anchor].tolist() for track in ordered]
    return int(anchor), ordered, prompts


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _extract_frames(video_path, directory, total, cancel_check):
    directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    written = 0
    try:
        for index in range(total):
            if cancel_check is not None and index % 30 == 0:
                cancel_check()
            ok, frame = capture.read()
            if not ok:
                raise CrossingInputsError(f"第 {index} 帧无法读取，视频可能已损坏")
            cv2.imwrite(str(directory / f"{index:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
    finally:
        capture.release()
    if written != total:
        raise CrossingInputsError(f"只导出了 {written}/{total} 帧")
    return directory


def _propagate_masks(predictor, frames_dir, work, anchor, prompts, total, width, height,
                     cancel_check, progress, label, precision="bf16"):
    """Run one forward and one backward pass per object and store packed masks."""
    packed_bits = (width * height + 7) // 8
    count = len(prompts)
    packed = {index: np.lib.format.open_memmap(
        work / f"object_{index}_masks.npy", mode="w+", dtype=np.uint8, shape=(total, packed_bits))
        for index in range(1, count + 1)}
    boxes = np.zeros((count, total, 4), dtype=np.float32)
    areas = np.zeros((count, total), dtype=np.int64)
    processed = np.zeros(total, dtype=bool)
    state = bounded_state(predictor, frames_dir, work / "bootstrap_frames")
    autocast = _bf16_autocast(predictor.device) if precision == "bf16" else contextlib.nullcontext()
    with torch.inference_mode(), autocast:
        for object_id, box in enumerate(prompts, 1):
            predictor.add_new_points_or_box(state, frame_idx=anchor, obj_id=object_id,
                                            box=np.asarray(box, np.float32))

        def collect(reverse):
            for frame, object_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=anchor, reverse=reverse):
                if cancel_check is not None:
                    cancel_check()
                for object_id, logit in zip(object_ids, logits):
                    mask = (logit > 0).cpu().numpy().reshape(height, width)
                    ys, xs = np.where(mask)
                    packed[int(object_id)][frame] = np.packbits(mask.reshape(-1))
                    index = int(object_id) - 1
                    areas[index, frame] = len(xs)
                    if len(xs):
                        boxes[index, frame] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                processed[frame] = True
                if progress is not None and frame % 25 == 0:
                    progress(0.35 + 0.45 * (frame / max(total, 1)), label(frame, reverse))

        collect(False)
        if anchor:
            collect(True)
    if not processed.all():
        missing = int(np.count_nonzero(~processed))
        raise CrossingInputsError(f"SAM2 有 {missing} 帧没有完成跟踪")
    for value in packed.values():
        value.flush()
    for value in packed.values():
        del value
    return boxes, areas


def prepare_crossing_inputs(processor, video_path, target_dir, *, device=None, max_people=MAX_PEOPLE,
                            progress=None, cancel_check=None):
    """Build ``sam2_inputs`` for ``video_path`` and move it into ``target_dir``.

    ``processor`` is a ready ``VideoProcessor`` (reuses the GUI detector).  The
    returned dict summarises what was produced.  Body crops follow the current
    visible mask at body scale (no held position); see ``sam2_body_boxes``.
    """
    video_path = str(video_path)
    target = Path(target_dir)
    started = time.time()

    def report(fraction, message):
        if progress is not None:
            progress(float(fraction), message)

    def check():
        if cancel_check is not None:
            cancel_check()

    report(0.02, "人物交叉优化：检测人物轨迹")
    result = processor.detect_and_track(
        video_path, cancel_check=cancel_check,
        on_progress=lambda done, total: report(0.02 + 0.18 * (done / max(total or 1, 1)),
                                               f"人物交叉优化：检测人物 {done}/{total}"))
    check()
    meta = processor.get_video_meta(video_path)
    total, width, height = int(meta.total_frames), int(meta.width), int(meta.height)
    if total <= 0 or width <= 0 or height <= 0:
        raise CrossingInputsError("无法读取视频的帧数或分辨率")
    seeds = select_seed_tracks(result.tracks, max_people=max_people)
    if len(seeds) < 2:
        raise CrossingInputsError("只检测到一个人物，交叉优化没有意义")
    anchor, seeds, prompts = choose_anchor_frame(seeds)
    report(0.22, f"人物交叉优化：从第 {anchor} 帧建立 {len(seeds)} 个身份")

    ckpt = _repo_root() / SAM2_CHECKPOINT
    if not ckpt.is_file():
        raise CrossingInputsError(f"SAM2 权重缺失：{ckpt}")
    from flowhmr.sam2_setup import crossing_issue
    from flowhmr.utils.sam2_builder import build_video_predictor

    component_issue = crossing_issue()
    if component_issue is not None:
        raise CrossingInputsError(component_issue[1])

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(tempfile.mkdtemp(prefix=".crossing-frames-", dir=str(parent)))
    work = Path(tempfile.mkdtemp(prefix=".crossing-inputs-", dir=str(parent)))
    predictor = None
    streams = []
    try:
        report(0.25, "人物交叉优化：导出视频帧")
        _extract_frames(video_path, frames_dir, total, cancel_check)
        check()
        predictor = build_video_predictor(SAM2_CONFIG, str(ckpt),
                                          device=str(device or processor.detector.device))
        predictor.eval()
        report(0.35, "人物交叉优化：SAM2 分割与跟踪")

        def label(frame, reverse):
            return f"人物交叉优化：SAM2 {'反向' if reverse else '正向'}传播第 {frame} 帧"

        precision_used = "fp32"
        for precision in ("bf16", "fp32"):
            try:
                boxes, areas = _propagate_masks(
                    predictor, frames_dir, work, anchor, prompts, total, width, height,
                    cancel_check, report, label, precision=precision)
                precision_used = precision
                break
            except CrossingInputsError:
                raise
            except Exception as error:  # noqa: BLE001 - bf16 is an optimisation, not a requirement
                if precision == "fp32":
                    raise
                print(f"[crossing] {precision} propagation failed "
                      f"({type(error).__name__}: {error}); retrying in fp32", flush=True)
                report(0.35, "人物交叉优化：bf16 不可用，改用 fp32 重试")
                _release_cuda()
                check()
        del predictor
        predictor = None
        _release_cuda()
        check()

        report(0.82, "人物交叉优化：合并断开的身份")
        streams = [PackedMasks(np.load(work / f"object_{index}_masks.npy", mmap_mode="r"), height, width)
                   for index in range(1, len(seeds) + 1)]
        objects = [Sam2Object(index + 1, np.arange(total), boxes=boxes[index], masks=streams[index])
                   for index in range(len(seeds))]
        resolution = merge_sam2_identities(result.tracks, objects)
        report(0.86, "人物交叉优化：生成尺度保持框")
        observations = {frame: [] for frame in range(total)}
        for track in result.tracks:
            for frame, box in zip(track.frames, track.bboxes):
                observations[int(frame)].append((int(track.track_id), box))
        count = len(seeds)
        matched = np.full((count, total, 4), np.nan, np.float32)
        matched_ids = np.full((count, total), -1, np.int64)
        reliable = np.zeros((count, total), dtype=bool)
        foreign = np.zeros((count, total), np.float32)
        for frame in range(total):
            if cancel_check is not None and frame % 50 == 0:
                cancel_check()
            masks = [stream[frame].astype(bool) for stream in streams]
            rows = observations[frame]
            scores = (np.array([[observation_match_score(np.asarray(box), obj, frame) for obj in objects]
                                for _, box in rows]) if rows else np.zeros((0, count)))
            for index in range(count):
                if not rows:
                    continue
                best = int(np.argmax(scores[:, index]))
                score = scores[best, index]
                other_object = np.delete(scores[best], index).max(initial=0)
                other_row = np.delete(scores[:, index], best).max(initial=0)
                if score < _MATCH_MIN_SCORE or score - max(other_object, other_row) <= _MATCH_MIN_MARGIN:
                    continue
                matched[index, frame] = rows[best][1]
                matched_ids[index, frame] = rows[best][0]
                box = rows[best][1]
                x1, y1, x2, y2 = [int(value) for value in
                                  (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))]
                own = np.count_nonzero(masks[index][y1:y2, x1:x2])
                alien_mask = np.logical_or.reduce([masks[other] for other in range(count) if other != index])
                alien = np.count_nonzero((alien_mask & ~masks[index])[y1:y2, x1:x2])
                foreign[index, frame] = alien / max(own + alien, 1)
                reliable[index, frame] = score >= _RELIABLE_SCORE and foreign[index, frame] < _MAX_FOREIGN_FRACTION

        people = []
        for index, seed in enumerate(seeds):
            policy = build_body_boxes(boxes[index], reliable[index], matched[index])
            body_boxes = policy.body_boxes
            ready = np.flatnonzero(policy.ready)
            if not len(ready):
                raise CrossingInputsError(f"第 {index + 1} 个人物没有可靠的全身起始框")
            first = int(ready[0])
            if not policy.ready[first:].all():
                raise CrossingInputsError(f"第 {index + 1} 个人物的尺度框在跟踪中失效")
            outage = check_mask_quality(f"第 {index + 1} 个人物", areas[index], body_boxes)
            accepted = [track for track in resolution.tracks if (index + 1) in track.sam2_object_ids]
            aliases = list(dict.fromkeys([int(seed.track_id)] + [
                int(raw_id) for track in accepted for raw_id in track.bytetrack_ids]))
            np.savez_compressed(
                work / f"object_{index + 1}_boxes.npz", frames=np.arange(first, total),
                body_boxes=body_boxes[first:], mask_boxes=boxes[index, first:],
                states=policy.states[first:], reference_boxes=policy.reference_boxes[first:],
                reliable_visible=reliable[index, first:], matched_raw_boxes=matched[index, first:],
                matched_raw_ids=matched_ids[index, first:], mask_area=areas[index, first:],
                foreign_fraction=foreign[index, first:])
            (work / f"object_{index + 1}_reference_frames.json").write_text(
                json.dumps([list(frames) for frames in policy.reference_frames]), encoding="utf-8")
            people.append(dict(
                object_id=index + 1, canonical_id=int(seed.track_id), bytetrack_ids=aliases,
                frame_range=[first, total - 1], mask_file=f"object_{index + 1}_masks.npy",
                mask_sha256=file_sha256(work / f"object_{index + 1}_masks.npy"),
                boxes_file=f"object_{index + 1}_boxes.npz", visible_frames=int((areas[index] > 0).sum()),
                mask_outage={key: value for key, value in outage.items() if key != "outage"}))
        for stream in streams:
            stream.close()
        streams = []

        mask_run_id = hashlib.sha256(json.dumps(
            dict(model=SAM2_MODEL, video_sha256=file_sha256(video_path), prompt_frame=anchor,
                 prompts=prompts, seed_bytetrack_ids=[int(track.track_id) for track in seeds]),
            sort_keys=True).encode()).hexdigest()
        policy = dict(version=BODY_POLICY_VERSION, reliability_iou=_RELIABLE_SCORE,
                      max_foreign_fraction=_MAX_FOREIGN_FRACTION,
                      arm_span_ratio=ARM_SPAN_RATIO)
        for person in people:
            person["boxes_sha256"] = file_sha256(work / person["boxes_file"])
        provenance = dict(
            version=1, producer="gui_crossing_v1", video=str(video_path), video_sha256=file_sha256(video_path),
            image_size=[width, height], num_frames=total, fps=30, precision=precision_used,
            checkpoint_sha256=file_sha256(ckpt), model=SAM2_MODEL, config=SAM2_CONFIG,
            prompt_frame=anchor, prompts=prompts,
            seed_bytetrack_ids=[int(track.track_id) for track in seeds],
            mask_run_id=mask_run_id, body_policy=policy,
            people=people,
            merges=[dict(canonical_id=merge.canonical_id, bytetrack_ids=list(merge.bytetrack_ids),
                         sam2_object_ids=list(merge.sam2_object_ids), start=merge.start, end=merge.end)
                    for merge in resolution.merges],
            ambiguous_bytetrack_ids=list(resolution.ambiguous))
        provenance["run_id"] = hashlib.sha256(json.dumps(
            dict(mask_run=mask_run_id, people=people, policy=policy), sort_keys=True).encode()).hexdigest()
        (work / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        report(0.92, "人物交叉优化：检查身份一致性")
        try:
            prepared, validation_streams, _ = load_person_inputs(work, video_path)
            for stream in validation_streams.values():
                stream.close()
            if len(prepared.tracks) != len(people):
                raise CrossingInputsError("准备好的身份数量与检测结果不一致")
        except CrossingInputsError:
            raise
        except Exception as error:
            raise CrossingInputsError(f"身份一致性检查未通过：{error}") from error
        check()
        if target.exists():
            shutil.rmtree(target)
        os.replace(work, target)
        work = None
        report(1.0, f"人物交叉优化：已准备 {len(people)} 个身份")
        return dict(people=len(people), frames=total, prompt_frame=anchor, precision=precision_used,
                    merges=len(provenance["merges"]), input_key=provenance["run_id"],
                    seconds=round(time.time() - started, 1))
    finally:
        for stream in streams:
            try:
                stream.close()
            except Exception:
                pass
        if predictor is not None:
            del predictor
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        shutil.rmtree(frames_dir, ignore_errors=True)
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
