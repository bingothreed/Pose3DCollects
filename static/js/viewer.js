// Three.js mesh-sequence viewer shared by both engines.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const HALF = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  HALF[h] = exp === 0
    ? sign * 5.960464477539063e-8 * frac
    : exp === 31
      ? (frac ? NaN : sign * Infinity)
      : sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

const THEMES = {
  dark: {
    background: 0x0a0c12,
    mesh: 0xb9c4ff,
    meshWire: 0xd4dbff,
    gridCenter: 0x6f7cff,
    gridLine: 0x252b3d,
    shadowOpacity: 0.28,
    trail: 0x7f8dff,
    key: 0xffffff,
  },
  light: {
    background: 0xf2f4fa,
    mesh: 0x8c9bff,
    meshWire: 0x4f5ee6,
    gridCenter: 0x7c89d9,
    gridLine: 0xd4d8e4,
    shadowOpacity: 0.18,
    trail: 0x5a6bff,
    key: 0xffffff,
  },
};

export class MeshViewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.setAttribute("aria-label", "3D 人体网格视图");
    container.appendChild(this.renderer.domElement);

    const theme = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    this.themeName = theme;
    this.theme = THEMES[theme];
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.theme.background);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.04, 400);
    this.camera.position.set(2.8, 1.8, 3.8);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.085;
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 160;
    this.controls.target.set(0, 0.9, 0);

    this.scene.add(new THREE.HemisphereLight(0xf4f7ff, 0x1e2433, 1.35));
    this.keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.bias = -0.0005;
    this.keyLight.shadow.normalBias = 0.018;
    this.keyLight.target.position.set(0, 0.8, 0);
    this.scene.add(this.keyLight, this.keyLight.target);
    const fill = new THREE.DirectionalLight(0xbfd0ff, 0.6);
    fill.position.set(-5, 2.5, -3);
    const rim = new THREE.DirectionalLight(0xffffff, 0.55);
    rim.position.set(1, 4, -7);
    this.scene.add(fill, rim);

    this.grid = null;
    this._buildGrid();

    this.shadowFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(400, 400),
      new THREE.ShadowMaterial({ color: 0x000000, opacity: this.theme.shadowOpacity }),
    );
    this.shadowFloor.rotation.x = -Math.PI / 2;
    this.shadowFloor.receiveShadow = true;
    this.scene.add(this.shadowFloor);

    this.geometry = null;
    this.material = null;
    this.mesh = null;
    this.trail = null;
    this.marker = null;
    this.meta = null;
    this.vertexData = null;
    this.faces = null;
    this.positions = null;
    this.numFrames = 0;
    this.numVerts = 0;
    this.fps = 30;
    this.frame = 0;
    this.playing = false;
    this.speed = 1;
    this.acc = 0;
    this.lastTs = 0;
    this.syncClock = null;
    this.viewLocked = false;
    this.prevRoot = null;
    this.onFrame = null;
    this.loadToken = 0;
    this.active = true;
    this.center = new THREE.Vector3(0, 1, 0);
    this.size = 1.8;
    this.floorY = 0;

    this.resize = this.resize.bind(this);
    this.tick = this.tick.bind(this);
    this.observer = new ResizeObserver(this.resize);
    this.observer.observe(container);
    this.resize();
    this.raf = requestAnimationFrame(this.tick);
  }

  _buildGrid() {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry?.dispose();
      this.grid.material?.dispose?.();
    }
    const grid = new THREE.GridHelper(40, 40, this.theme.gridCenter, this.theme.gridLine);
    grid.material.transparent = true;
    grid.material.opacity = 0.34;
    grid.material.depthWrite = false;
    grid.position.y = this.floorY || 0;
    this.grid = grid;
    this.scene.add(grid);
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  async load(job) {
    const token = ++this.loadToken;
    const [metaRes, vertsRes, facesRes] = await Promise.all([
      fetch(job.urls.meta, { cache: "no-store" }),
      fetch(job.urls.vertices, { cache: "no-store" }),
      fetch(job.urls.faces, { cache: "no-store" }),
    ]);
    if (!metaRes.ok) throw new Error(`读取 meta 失败（${metaRes.status}）`);
    if (!vertsRes.ok) throw new Error(`读取顶点失败（${vertsRes.status}）`);
    if (!facesRes.ok) throw new Error(`读取面片失败（${facesRes.status}）`);
    const meta = await metaRes.json();
    const [vertexBuffer, faceBuffer] = await Promise.all([vertsRes.arrayBuffer(), facesRes.arrayBuffer()]);
    if (token !== this.loadToken) return null;

    this.disposeData();
    this.meta = meta;
    this.numFrames = Number(meta.num_frames) || 0;
    this.numVerts = Number(meta.num_vertices) || 0;
    this.fps = Number(meta.fps) || 30;
    this.frame = 0;
    this.acc = 0;
    this.playing = false;
    if (!this.numFrames || !this.numVerts) throw new Error("结果缺少网格数据");

    const dtype = meta.vertex_dtype || "float16";
    this.vertexData = dtype === "float32" ? new Float32Array(vertexBuffer) : new Uint16Array(vertexBuffer);
    this.faces = new Uint32Array(faceBuffer);

    const count = this.numVerts * 3;
    const expected = this.numFrames * count;
    if (this.vertexData.length < expected) {
      throw new Error(`顶点数据不完整：期望 ${expected}，实际 ${this.vertexData.length}`);
    }
    this.positions = new Float32Array(count);
    const geometry = new THREE.BufferGeometry();
    const position = new THREE.BufferAttribute(this.positions, 3);
    position.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", position);
    geometry.setIndex(new THREE.BufferAttribute(this.faces, 1));

    const bmin = meta.bbox_min || [-1, 0, -1];
    const bmax = meta.bbox_max || [1, 2, 1];
    this.center.set((bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2);
    this.size = Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2], 0.7);
    this.floorY = Number.isFinite(bmin[1]) ? bmin[1] : 0;
    geometry.boundingSphere = new THREE.Sphere(this.center.clone(), this.size * 2.2);

    this.material = new THREE.MeshStandardMaterial({
      color: this.theme.mesh,
      roughness: 0.58,
      metalness: 0.03,
      flatShading: true,
    });
    this.geometry = geometry;
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.scene.add(this.mesh);

    this.grid.position.y = this.floorY;
    this.shadowFloor.position.y = this.floorY - 0.002;
    const shadowSize = Math.max(2.5, this.size * 2.4);
    this.keyLight.position.set(this.size * 1.4, this.floorY + this.size * 2.4, this.size * 1.8);
    this.keyLight.target.position.copy(this.center);
    Object.assign(this.keyLight.shadow.camera, {
      left: -shadowSize, right: shadowSize, top: shadowSize, bottom: -shadowSize,
      near: 0.2, far: shadowSize * 5,
    });
    this.keyLight.shadow.camera.updateProjectionMatrix();

    this._buildTrail();
    this.setFrame(0);
    this.resetView();
    return meta;
  }

  _buildTrail() {
    const trajectory = this.meta?.root_trajectory || [];
    if (trajectory.length < 2) return;
    const points = trajectory.map(([x, z]) => new THREE.Vector3(x, this.floorY + 0.006, z));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: this.theme.trail, transparent: true, opacity: 0.55, depthWrite: false,
    });
    this.trail = new THREE.Line(geometry, material);
    this.scene.add(this.trail);
    const marker = new THREE.Mesh(
      new THREE.RingGeometry(0.055, 0.082, 40),
      new THREE.MeshBasicMaterial({ color: this.theme.trail, side: THREE.DoubleSide, depthWrite: false }),
    );
    marker.rotation.x = -Math.PI / 2;
    this.marker = marker;
    this.scene.add(marker);
  }

  setFrame(index) {
    if (!this.meta || !this.numFrames) return;
    const frame = ((Math.round(index) % this.numFrames) + this.numFrames) % this.numFrames;
    this.frame = frame;
    const count = this.numVerts * 3;
    const base = frame * count;
    if (this.vertexData instanceof Uint16Array) {
      for (let i = 0; i < count; i++) this.positions[i] = HALF[this.vertexData[base + i]];
    } else {
      this.positions.set(this.vertexData.subarray(base, base + count));
    }
    this.geometry.attributes.position.needsUpdate = true;
    const root = this.meta.root_trajectory?.[frame];
    if (this.marker && root) {
      this.marker.position.set(root[0], this.floorY + 0.012, root[1]);
    }
    this._followRoot(root);
    if (this.onFrame) this.onFrame(frame, this.numFrames, this.fps);
  }

  // Unlocked view follows the character across the scene (like FlowHMR's
  // “Follow”): shift the camera and orbit target by the root delta so the
  // user's chosen angle and zoom are preserved. Locked view stays put.
  _followRoot(root) {
    if (!root) return;
    const previous = this.prevRoot;
    this.prevRoot = new THREE.Vector2(root[0], root[1]);
    if (!previous || this.viewLocked) return;
    const dx = root[0] - previous.x;
    const dz = root[1] - previous.y;
    if (!dx && !dz) return;
    this.camera.position.x += dx;
    this.camera.position.z += dz;
    this.controls.target.x += dx;
    this.controls.target.z += dz;
  }

  play() {
    if (!this.meta) return;
    this.playing = true;
    this.lastTs = performance.now();
    this.acc = 0;
  }

  pause() {
    this.playing = false;
  }

  toggle() {
    if (this.playing) this.pause(); else this.play();
    return this.playing;
  }

  setSpeed(speed) {
    this.speed = Number(speed) || 1;
  }

  // When set, playback progress comes from this clock (seconds) instead of
  // the internal accumulator — used to follow the reference video exactly.
  setSyncClock(clock) {
    this.syncClock = typeof clock === "function" ? clock : null;
  }

  setViewLocked(locked) {
    this.viewLocked = Boolean(locked);
    this.controls.enabled = !this.viewLocked;
  }

  toggleWireframe() {
    if (!this.material) return false;
    this.material.wireframe = !this.material.wireframe;
    return this.material.wireframe;
  }

  resetView() {
    if (!this.meta) return;
    const distance = Math.max(1.2, this.size * 1.85);
    const dir = new THREE.Vector3(0.82, 0.38, 1.0).normalize();
    const target = this.center.clone();
    target.y = this.center.y - this.size * 0.04;
    this.controls.target.copy(target);
    this.camera.position.copy(target).addScaledVector(dir, distance);
    this.camera.near = Math.max(0.02, distance / 200);
    this.camera.far = distance * 30;
    this.camera.updateProjectionMatrix();
    const root = this.meta.root_trajectory?.[this.frame];
    this.prevRoot = root ? new THREE.Vector2(root[0], root[1]) : null;
    this.controls.update();
  }

  setTheme(themeName) {
    this.themeName = themeName === "light" ? "light" : "dark";
    this.theme = THEMES[this.themeName];
    this.scene.background.setHex(this.theme.background);
    if (this.material) {
      this.material.color.setHex(this.material.wireframe ? this.theme.meshWire : this.theme.mesh);
    }
    this._buildGrid();
    this.shadowFloor.material.opacity = this.theme.shadowOpacity;
    if (this.trail) this.trail.material.color.setHex(this.theme.trail);
    if (this.marker) this.marker.material.color.setHex(this.theme.trail);
  }

  disposeData() {
    this.loadToken += 1;
    this.playing = false;
    for (const object of [this.mesh, this.trail, this.marker]) {
      if (!object) continue;
      this.scene.remove(object);
      object.geometry?.dispose();
      object.material?.dispose?.();
    }
    this.mesh = this.trail = this.marker = null;
    this.geometry = this.material = null;
    this.meta = null;
    this.prevRoot = null;
    this.vertexData = null;
    this.faces = null;
    this.positions = null;
    this.numFrames = 0;
    this.numVerts = 0;
    this.frame = 0;
  }

  setActive(active) {
    this.active = Boolean(active);
    if (this.active) {
      this.resize();
      this.renderer.render(this.scene, this.camera);
    }
  }

  tick(ts) {
    this.raf = requestAnimationFrame(this.tick);
    const dt = this.lastTs ? Math.min(0.08, (ts - this.lastTs) / 1000) : 0;
    this.lastTs = ts;
    if (!this.active) return;
    if (this.playing && this.meta) {
      if (this.syncClock) {
        const seconds = this.syncClock();
        if (Number.isFinite(seconds) && seconds >= 0) {
          this.setFrame(Math.floor(seconds * this.fps + 1e-6));
        }
      } else {
        this.acc += dt * this.fps * this.speed;
        const step = Math.floor(this.acc);
        if (step > 0) {
          this.acc -= step;
          this.setFrame(this.frame + step);
        }
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.observer.disconnect();
    this.disposeData();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
