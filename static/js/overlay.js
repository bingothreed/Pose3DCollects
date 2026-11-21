// Two-dimensional person boxes / keypoints drawn on top of the reference video.
export class VideoOverlay {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.track = null;
    this.frames = [];
    this.edges = [];
    this.fps = 30;
    this.start = 0;
    this.raf = 0;
    this.visible = true;
    this._loop = this._loop.bind(this);
    this._resize = this._resize.bind(this);
    window.addEventListener("resize", this._resize);
    video.addEventListener("loadedmetadata", this._resize);
    video.addEventListener("seeked", () => this.draw());
    this._resize();
    this.raf = requestAnimationFrame(this._loop);
  }

  setTrack(track) {
    this.track = track || null;
    if (!track) {
      this.frames = [];
      this.edges = [];
      this.draw();
      return;
    }
    this.fps = Number(track.fps) || 30;
    this.start = Number(track.start) || 0;
    if (Array.isArray(track.frames)) {
      this.frames = track.frames;
    } else if (Array.isArray(track.boxes)) {
      this.frames = track.boxes.map((box) => ({ box }));
    } else {
      this.frames = [];
    }
    this.edges = Array.isArray(track.edges) ? track.edges : [];
    this.draw();
  }

  setVisible(visible) {
    this.visible = visible;
    this.canvas.classList.toggle("hidden", !visible);
    if (visible) this.draw();
  }

  hasSkeleton() {
    return this.edges.length > 0 && this.frames.some((frame) => Array.isArray(frame.keypoints));
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.video.clientWidth;
    const h = this.video.clientHeight;
    if (!w || !h) return;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw;
      this.canvas.height = ph;
    }
    this.dpr = dpr;
    this.draw();
  }

  _contentRect() {
    const cw = this.video.clientWidth;
    const ch = this.video.clientHeight;
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!cw || !ch || !vw || !vh) return null;
    const scale = Math.min(cw / vw, ch / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    return { x: (cw - dw) / 2, y: (ch - dh) / 2, scale, cw, ch };
  }

  _drawFrame() {
    const ctx = this.ctx;
    const dpr = this.dpr || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.video.clientWidth, this.video.clientHeight);
    if (!this.track || !this.frames.length) return;
    const rect = this._contentRect();
    if (!rect) return;
    const frameIndex = Math.max(0, Math.min(this.frames.length - 1,
      Math.floor(this.video.currentTime * this.fps) - this.start));
    const frame = this.frames[frameIndex];
    if (!frame) return;
    const dark = document.documentElement.dataset.theme !== "light";
    const boxColor = dark ? "rgba(139, 155, 255, 0.96)" : "rgba(74, 93, 240, 0.95)";
    const keyColor = dark ? "rgba(77, 214, 193, 0.95)" : "rgba(14, 174, 150, 0.95)";
    const labelBg = dark ? "rgba(10, 12, 20, 0.82)" : "rgba(255, 255, 255, 0.9)";
    const labelFg = dark ? "#eef1f8" : "#161922";
    ctx.save();
    ctx.translate(rect.x, rect.y);
    ctx.scale(rect.scale, rect.scale);
    if (Array.isArray(frame.keypoints)) {
      ctx.strokeStyle = keyColor;
      ctx.lineWidth = 1.6 / rect.scale;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      for (const [a, b] of this.edges) {
        const pa = frame.keypoints[a];
        const pb = frame.keypoints[b];
        if (!pa || !pb || (pa[0] === 0 && pa[1] === 0) || (pb[0] === 0 && pb[1] === 0)) continue;
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
      }
      ctx.stroke();
      ctx.fillStyle = keyColor;
      for (const point of frame.keypoints) {
        if (!point || (point[0] === 0 && point[1] === 0)) continue;
        ctx.beginPath();
        ctx.arc(point[0], point[1], 2.2 / rect.scale, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    if (Array.isArray(frame.box)) {
      const [x1, y1, x2, y2] = frame.box;
      const w = Math.max(1, x2 - x1);
      const h = Math.max(1, y2 - y1);
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2 / rect.scale;
      ctx.strokeRect(x1, y1, w, h);
      // corner ticks
      const tick = Math.min(w, h) * 0.16;
      ctx.lineWidth = 3.5 / rect.scale;
      ctx.beginPath();
      ctx.moveTo(x1, y1 + tick); ctx.lineTo(x1, y1); ctx.lineTo(x1 + tick, y1);
      ctx.moveTo(x2 - tick, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + tick);
      ctx.moveTo(x2, y2 - tick); ctx.lineTo(x2, y2); ctx.lineTo(x2 - tick, y2);
      ctx.moveTo(x1 + tick, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - tick);
      ctx.stroke();
      if (frame.score) {
        const label = `${Math.round(frame.score * 100)}%`;
        const fontPx = 11 / rect.scale;
        ctx.font = `600 ${fontPx}px "Segoe UI", "Microsoft YaHei", sans-serif`;
        const pad = 5 / rect.scale;
        const tw = ctx.measureText(label).width;
        const lw = tw + pad * 2;
        const lh = 17 / rect.scale;
        const lx = x1;
        let ly = y1 - lh - 3 / rect.scale;
        if (ly < 0) ly = y1 + 3 / rect.scale;
        ctx.fillStyle = labelBg;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(lx, ly, lw, lh, 4 / rect.scale);
        else ctx.rect(lx, ly, lw, lh);
        ctx.fill();
        ctx.fillStyle = labelFg;
        ctx.fillText(label, lx + pad, ly + lh * 0.72);
      }
    }
    ctx.restore();
  }

  _loop() {
    this.raf = requestAnimationFrame(this._loop);
    if (this.visible) this._drawFrame();
  }

  draw() {
    this._drawFrame();
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this._resize);
  }
}
