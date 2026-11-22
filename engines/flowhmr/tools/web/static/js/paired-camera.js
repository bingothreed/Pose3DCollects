import { sampleFrame } from './frame-sample.js';

const JOINT_COMPONENTS = 52 * 3;

// Full pixel intrinsics, independent of the viewer's free-orbit projection.
export function placedCameraTransform(transform, yaw = 0, offset = [0, 0, 0]) {
  const { scale: k, translation: b } = transform;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const rotated = [c*b[0] + s*b[2], b[1], -s*b[0] + c*b[2]];
  return { scale: k, translation: rotated.map((x, i) => x + (1-k)*offset[i]) };
}

export function characterHeightScale(sourceHead, sourceSole, targetHead, targetSole) {
  const a = sourceHead?.y - sourceSole, b = targetHead?.y - targetSole;
  const scale = b / a;
  return a > .05 && b > .05 && scale >= .01 && scale <= 100 ? scale : null;
}

export function pelvisCameraTransform(scale, motion, frame, target, placement) {
  const f = Math.max(0, Math.min(motion.frames - 1, frame));
  const lo = Math.floor(f), hi = Math.min(lo + 1, motion.frames - 1), t = f - lo;
  const translation = target.map((value, axis) => {
    const a = motion.joints[lo * JOINT_COMPONENTS + axis], b = motion.joints[hi * JOINT_COMPONENTS + axis];
    return value - scale * (a + (b - a) * t);
  });
  return placedCameraTransform({ scale, translation }, placement.yaw, placement.offset);
}

export function animatedCameraTransform(scale, start, translations) {
  return { scale, start_frame: start, frame_translations: translations };
}

export function applySourceCamera(camera, data, frame, transform = { scale: 1, translation: [0, 0, 0] }) {
  if (!data) return false;
  const { visible, lo, hi, alpha } = sampleFrame(frame, data.RT.length, data.camera_valid, data.shot_ids);
  if (!visible) return false;
  let K = data.K[lo];
  const [h, w] = data.image_size_hw;
  const { scale: k, translation: b } = transform;
  const matrixAt = (index, matrix) => {
    const R = data.RT[index];
    const C = [0, 1, 2].map((j) => -(R[0][j] * R[0][3] + R[1][j] * R[1][3] + R[2][j] * R[2][3]) * k + b[j]);
    return matrix.set(R[0][0], -R[1][0], -R[2][0], C[0],
      R[0][1], -R[1][1], -R[2][1], C[1], R[0][2], -R[1][2], -R[2][2], C[2], 0, 0, 0, 1);
  };
  camera.matrixAutoUpdate = false;
  matrixAt(lo, camera.matrix);
  if (alpha) {
    camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
    const p = camera.position.clone(), q = camera.quaternion.clone(), s = camera.scale.clone();
    matrixAt(hi, camera.matrix.clone()).decompose(p, q, s);
    camera.position.lerp(p, alpha);
    camera.quaternion.slerp(q, alpha);
    camera.matrix.compose(camera.position, camera.quaternion, camera.scale);
    K = K.map((row, i) => row.map((value, j) => value + alpha * (data.K[hi][i][j] - value)));
  }
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse.copy(camera.matrix).invert();
  camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
  const near = .001, far = 10000;
  // The depth shader reads the camera's far plane separately from the matrix.
  camera.near = near;
  camera.far = far;
  camera.projectionMatrix.set(2 * K[0][0] / w, -2 * K[0][1] / w, 1 - 2 * K[0][2] / w, 0,
    0, 2 * K[1][1] / h, 2 * K[1][2] / h - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near), 0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  return true;
}

export function sourceViewport(width, height, imageSize) {
  const aspect = imageSize[1] / imageSize[0];
  const w = Math.min(width, height * aspect), h = w / aspect;
  return [(width - w) / 2, (height - h) / 2, w, h];
}
