// A source frame owns [frame, frame + 1). Hide gaps for that entire interval;
// hold the last pose before a gap or cut instead of blending two unrelated shots.
export function sampleFrame(frame, count, validFrames, shotIds) {
  if (!Number.isFinite(frame) || frame < 0 || frame >= count || count < 1)
    return { visible: false, lo: 0, hi: 0, alpha: 0, frame: 0 };
  const lo = Math.floor(frame), next = Math.min(lo + 1, count - 1);
  const valid = (i) => validFrames == null || !!validFrames[i];
  const visible = valid(lo);
  const continuous = visible && valid(next) && (shotIds == null || shotIds[lo] === shotIds[next]);
  const hi = continuous ? next : lo;
  const alpha = hi === lo ? 0 : frame - lo;
  return { visible, lo, hi, alpha, frame: lo + alpha };
}

// Video synchronization interpolates around frame centers. At a cut or
// visibility transition, the decoded source frame owns the boundary instead.
export function sameFrameSegment(frame, sourceFrame, count, validFrames, shotIds) {
  const a = sampleFrame(frame, count, validFrames, shotIds);
  const b = sampleFrame(sourceFrame, count, validFrames, shotIds);
  return a.visible === b.visible && (shotIds == null || shotIds[a.lo] === shotIds[b.lo]);
}

export function scenePlacement(actor, sourceFrame) {
  const shots = actor.placements;
  if (!shots?.length) return actor;
  return shots.findLast(shot => shot.start <= sourceFrame) ?? shots[0];
}

export function sceneActorVisible(actor, sourceFrame) {
  return actor.visible_ranges == null || actor.visible_ranges.some(([start, end]) => start <= sourceFrame && sourceFrame < end);
}
