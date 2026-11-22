// Localize current and persisted recovery progress without rewriting old jobs.
import { localize, message as m } from './i18n.js';
const fixed = new Map(Object.entries({
  '等待恢复摄像机':'queued', '正在停止摄像机计算':'cancel_requested',
  '摄像机已就绪':'ready', '摄像机计算已停止，人物动作已保留':'cancelled',
  '摄像机已就绪（已降级，地面倾角可能不准确）':'degraded_ready',
  '摄像机已就绪（部分区间无可用机位）':'degraded_ready',
  '摄像机恢复失败，可重试':'failed', '摄像机计算已中断':'failed',
  '读取已完成的人物动作':'reading', '已复用任务的原始相机观测':'reuse_camera',
  '已复用人物观测，动作保持不变':'reuse_observations', '估计画面重力方向':'gravity',
  '采用人物长窗口与地面估计水平':'level_fallback',
  '稳定摄像机并整理逐镜头说明':'stabilizing', '恢复多人场景的共同机位':'shared_recovery',
}));
const counters = [
  [/^摄像机整体尺度 · (\d+)\/(\d+)$/, 'scale_progress'],
  [/^对齐原片机位 · (\d+)\/(\d+)$/, 'align_progress'],
  [/^补算人物观测 · (\d+)\/(\d+) 帧$/, 'observation_progress'],
];
export function multishotMessage(value) {
  if (!value) return null;
  if (value === '正在加载人物观测模型') return m('job.model_loading');
  if (fixed.has(value)) return m(`camera.${fixed.get(value)}`);
  for (const [pattern,key] of counters) {
    const match=value.match(pattern);
    if (match) return m(`camera.${key}`, {current:match[1],total:match[2]});
  }
  if (value.startsWith('画面重力不可用，采用人物与地面估计：')) return m('camera.level_fallback');
  return null;
}
export function cameraStatusMessage(status) {
  const operation = cameraOperationMessage(status);
  if (operation) return operation;
  if (status.camera_state === 'ready' && status.quality?.degraded) return m('camera.degraded_ready');
  return multishotMessage(status.message) ?? m(`camera.${status.camera_state ?? 'none'}`);
}
export const cameraActive = (status) => ['queued', 'running', 'cancel_requested'].includes(status?.camera_state);
export const cameraAvailable = (status) => status?.available ?? (status?.camera_state === 'ready');
export function cameraStatusForJob(job) {
  return {
    camera_state: job.camera_state,
    available: job.camera_available,
    revision: job.camera_revision,
    quality: job.camera_quality,
    coverage: job.camera_coverage,
    message: job.camera_message,
    error: job.camera_error,
  };
}
export function cameraOperationMessage(status) {
  if (!cameraAvailable(status)) return '';
  if (status.camera_state === 'cancel_requested') return m('camera.cancel_recompute_old');
  if (cameraActive(status)) return m('camera.recomputing_old');
  if (status.camera_state === 'failed') return m('camera.recompute_failed_old');
  if (status.camera_state === 'cancelled') return m('camera.recompute_cancelled_old');
  return '';
}
function gravityMessage(quality) {
  if (!quality?.degraded || quality.reason === 'camera_coverage') return '';
  const key = ['weights_missing', 'weights_corrupt', 'runtime_missing', 'unsupported_device',
    'inference_failed', 'static_camera', 'unstable', 'legacy_fallback', 'unverified', 'body_ground'].includes(quality.reason)
    ? quality.reason : 'unverified';
  return m(`camera.gravity_${key}`, {count:quality.fallback_shots?.length ?? 0, total:quality.total_shots ?? 0});
}
export function cameraQualityMessage(quality, coverage) {
  const gravity = gravityMessage(quality);
  const gaps = coverage && coverage.status !== 'complete'
    ? m('camera.partial_coverage', { valid: coverage.valid_frames, total: coverage.total_frames })
    : quality?.reason === 'camera_coverage' ? m('camera.confidence_hint') : '';
  if (gravity && gaps) return () => `${localize(gravity)} ${localize(gaps)}`;
  return gravity || gaps;
}
