/*
 * Interactive Ewald-sphere figure for the cd-reduce docs.
 *
 * Dependency-free (Canvas 2D). Markup:
 *   <div class="ewald-figure" data-ewald data-energy="285" data-pitch="80"
 *        data-omega="15" data-mmax="4" data-fixtures="../../assets/ewald/fixtures.json">
 *     ...static fallback (replaced when this script runs)...
 *   </div>
 *
 * Geometry (sample frame, reduction convention of cd_reduce.reduction.geometry):
 *   k = 2 pi E / hc, G = 2 pi / pitch. The Ewald sphere has radius k and centre
 *   C = (k sin w, 0, -k cos w), so it passes through the origin. Rod m sits at
 *   (m G, 0, Qz) and meets the sphere at
 *     Qz = -k cos w + sqrt(k^2 - (m G - k sin w)^2)
 *   (no intersection when the square root is imaginary). The flat-detector
 *   (small-angle) limit is Qz = Qx tan w. tests/test_ewald_fixtures.py pins this
 *   closed form to compute_pixel_q; the badge re-checks it against the fixtures.
 */
(() => {
  "use strict";

  const HC = 1239.84198;
  const D2R = Math.PI / 180;
  const OMEGA_RANGE = 40;
  const ORDER_COLORS = ["#0ea5e9", "#f59e0b", "#10b981", "#f43f5e", "#8b5cf6",
                        "#14b8a6", "#eab308", "#ec4899"];

  const rodQz = (k, G, w, m) => {
    const s = Math.sin(w * D2R), c = Math.cos(w * D2R);
    const d = k * k - (m * G - k * s) ** 2;
    return d <= 0 ? null : -k * c + Math.sqrt(d);
  };
  const sheetQz = (k, w, qx, qy) => {
    const s = Math.sin(w * D2R), c = Math.cos(w * D2R);
    const d = k * k - (qx - k * s) ** 2 - qy * qy;
    return d <= 0 ? null : -k * c + Math.sqrt(d);
  };
  const orderColor = (m) => ORDER_COLORS[(Math.abs(m) - 1) % ORDER_COLORS.length];
  const fmt = (v, d = 4) => (v === null || !Number.isFinite(v) ? "—" : v.toFixed(d));

  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name, fallback) => (cs.getPropertyValue(name) || "").trim() || fallback;
    const dark = document.documentElement.classList.contains("dark");
    return {
      fg: v("--foreground", dark ? "#e5e7eb" : "#1f2933"),
      muted: v("--muted-foreground", dark ? "#9ca3af" : "#52606d"),
      border: v("--border", dark ? "#334155" : "#d9e2ec"),
      bg: v("--card", v("--background", dark ? "#0b1220" : "#ffffff")),
      accent: v("--chart-2", "#3b82f6"),
      accentStrong: v("--chart-3", "#1d4ed8"),
      font: v("--font-sans", "system-ui, sans-serif"),
      dark,
    };
  }

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, val] of Object.entries(attrs)) {
      if (key === "text") node.textContent = val;
      else node.setAttribute(key, val);
    }
    for (const child of children) node.append(child);
    return node;
  }

  class EwaldFigure {
    constructor(root) {
      this.root = root;
      const d = root.dataset;
      this.state = {
        energy: Number(d.energy || 285),
        pitch: Number(d.pitch || 80),
        omega: Number(d.omega || 15),
        mmax: Math.max(1, Math.min(8, Number(d.mmax || 4))),
        yaw: -0.62,
        elev: 0.38,
        playing: false,
        dir: 1,
      };
      this.defaults = { ...this.state };
      this.theme = readTheme();
      this.build();
      this.bind();
      this.resize();
      if (d.fixtures) this.checkFixtures(d.fixtures);
    }

    /* ---------- DOM ---------- */
    build() {
      const s = this.state;
      this.root.replaceChildren();
      this.root.classList.add("ewald-figure--live");
      this.cv3d = el("canvas", { "aria-label": "Ewald sphere and grating rods in 3D", role: "img" });
      this.cv2d = el("canvas", { "aria-label": "Ewald arcs in the Qx–Qz plane", role: "img" });
      const panel = (canvas, title, note) => el("figure", { class: "ewald-figure__panel" }, [
        el("div", { class: "ewald-figure__title", text: title }),
        canvas,
        el("figcaption", { text: note }),
      ]);
      const panels = el("div", { class: "ewald-figure__panels" }, [
        panel(this.cv3d, "Reciprocal space (sample frame)",
              "Drag to rotate · double-click to reset. Inset: the whole sphere and the zoomed patch."),
        panel(this.cv2d, "Qx–Qz: one frame and the full ω sweep",
              "Bold arc: this ω (solid exact, dashed small-angle). Shaded: all ω in ±40°."),
      ]);

      const slider = (label, min, max, step, value, unit) => {
        const input = el("input", { type: "range", min, max, step, value: String(value) });
        const out = el("output", { text: `${value}${unit}` });
        const wrap = el("label", { class: "ewald-figure__control" }, [
          el("span", { text: label }), input, out,
        ]);
        return { wrap, input, out, unit };
      };
      this.cOmega = slider("Tilt ω", -OMEGA_RANGE, OMEGA_RANGE, 0.5, s.omega, "°");
      this.cEnergy = slider("Energy", 150, 1000, 1, s.energy, " eV");
      this.cPitch = el("input", { type: "number", min: "20", max: "500", step: "1",
                                  value: String(s.pitch) });
      const pitchWrap = el("label", { class: "ewald-figure__control ewald-figure__control--num" }, [
        el("span", { text: "Pitch Λ (nm)" }), this.cPitch,
      ]);
      this.bPlay = el("button", { type: "button", class: "ewald-figure__btn", text: "Sweep ω" });
      this.bReset = el("button", { type: "button", class: "ewald-figure__btn", text: "Reset" });
      const controls = el("div", { class: "ewald-figure__controls" }, [
        this.cOmega.wrap, this.cEnergy.wrap, pitchWrap,
        el("div", { class: "ewald-figure__buttons" }, [this.bPlay, this.bReset]),
      ]);
      this.readout = el("div", { class: "ewald-figure__readout", "aria-live": "polite" });
      this.badge = el("div", { class: "ewald-figure__badge" });
      this.root.append(panels, controls, this.readout, this.badge);
    }

    bind() {
      const s = this.state;
      this.cOmega.input.addEventListener("input", () => {
        s.omega = Number(this.cOmega.input.value);
        this.draw();
      });
      this.cEnergy.input.addEventListener("input", () => {
        s.energy = Number(this.cEnergy.input.value);
        this.draw();
      });
      this.cPitch.addEventListener("input", () => {
        const v = Number(this.cPitch.value);
        if (v >= 20 && v <= 500) { s.pitch = v; this.draw(); }
      });
      this.bPlay.addEventListener("click", () => this.togglePlay());
      this.bReset.addEventListener("click", () => this.reset());

      let drag = null;
      this.cv3d.addEventListener("pointerdown", (e) => {
        drag = { x: e.clientX, y: e.clientY };
        this.cv3d.setPointerCapture(e.pointerId);
      });
      this.cv3d.addEventListener("pointermove", (e) => {
        if (!drag) return;
        s.yaw += (e.clientX - drag.x) * 0.01;
        s.elev = Math.max(-0.2, Math.min(1.45, s.elev + (e.clientY - drag.y) * 0.01));
        drag = { x: e.clientX, y: e.clientY };
        this.draw();
      });
      const end = () => { drag = null; };
      this.cv3d.addEventListener("pointerup", end);
      this.cv3d.addEventListener("pointercancel", end);
      this.cv3d.addEventListener("dblclick", () => {
        s.yaw = this.defaults.yaw;
        s.elev = this.defaults.elev;
        this.draw();
      });

      new ResizeObserver(() => this.resize()).observe(this.root);
      new MutationObserver(() => { this.theme = readTheme(); this.draw(); })
        .observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    }

    reset() {
      Object.assign(this.state, this.defaults, { playing: false });
      this.cOmega.input.value = String(this.state.omega);
      this.cEnergy.input.value = String(this.state.energy);
      this.cPitch.value = String(this.state.pitch);
      this.bPlay.textContent = "Sweep ω";
      this.draw();
    }

    togglePlay() {
      const s = this.state;
      s.playing = !s.playing;
      this.bPlay.textContent = s.playing ? "Pause" : "Sweep ω";
      if (!s.playing) return;
      const step = () => {
        if (!s.playing) return;
        s.omega += 0.35 * s.dir;
        if (Math.abs(s.omega) >= OMEGA_RANGE) {
          s.omega = Math.sign(s.omega) * OMEGA_RANGE;
          s.dir = -s.dir;
        }
        this.cOmega.input.value = String(Math.round(s.omega * 2) / 2);
        this.draw();
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    resize() {
      const dpr = window.devicePixelRatio || 1;
      for (const cv of [this.cv3d, this.cv2d]) {
        const w = cv.clientWidth || 320;
        const h = Math.round(w * 0.78);
        cv.style.height = `${h}px`;
        cv.width = Math.round(w * dpr);
        cv.height = Math.round(h * dpr);
      }
      this.dpr = dpr;
      this.draw();
    }

    /* ---------- derived quantities ---------- */
    derived() {
      const s = this.state;
      const k = (2 * Math.PI * s.energy) / HC;
      const G = (2 * Math.PI) / s.pitch;
      const X = (s.mmax + 0.7) * G;
      let Z = 0;
      for (let w = -OMEGA_RANGE; w <= OMEGA_RANGE; w += 2) {
        for (let m = -s.mmax; m <= s.mmax; m++) {
          const q = rodQz(k, G, w, m);
          if (q !== null) Z = Math.max(Z, Math.abs(q));
        }
      }
      Z = Math.max(0.35 * X, Math.min(1.1 * Z, 1.2 * X));
      return { k, G, X, Y: 0.55 * X, Z };
    }

    /* ---------- drawing ---------- */
    draw() {
      if (!this.dpr) return;
      const s = this.state;
      this.cOmega.out.textContent = `${s.omega.toFixed(1)}°`;
      this.cEnergy.out.textContent = `${s.energy.toFixed(0)} eV`;
      const g = this.derived();
      this.draw3d(g);
      this.draw2d(g);
      this.drawReadout(g);
    }

    ctx(canvas) {
      const c = canvas.getContext("2d");
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      const w = canvas.width / this.dpr, h = canvas.height / this.dpr;
      c.clearRect(0, 0, w, h);
      c.fillStyle = this.theme.bg;
      c.fillRect(0, 0, w, h);
      c.font = `12px ${this.theme.font}`;
      c.lineCap = "round";
      c.lineJoin = "round";
      return { c, w, h };
    }

    draw3d(g) {
      const { c, w, h } = this.ctx(this.cv3d);
      const t = this.theme, s = this.state;
      const cy = Math.cos(s.yaw), sy = Math.sin(s.yaw);
      const ce = Math.cos(s.elev), se = Math.sin(s.elev);
      const scale = (0.31 * Math.min(w, h * 1.2)) / g.X;
      const ox = w * 0.47, oy = h * 0.54;
      const Zr = 0.78 * g.Z; // drawn rod half-length
      // Qx right, Qy into the screen, Qz up; yaw about Qz, then elevation.
      const P = (qx, qy, qz) => {
        const x1 = qx * cy - qy * sy;
        const y1 = qx * sy + qy * cy;
        return {
          x: ox + x1 * scale,
          y: oy - (qz * ce - y1 * se) * scale,
          d: (y1 * ce + qz * se) / g.X,
        };
      };
      const fade = (d) => Math.max(0.18, Math.min(1, 0.65 - 0.45 * d));
      const stroke = (pts, color, width, alpha = 1, dash = []) => {
        if (pts.length < 2) return;
        c.globalAlpha = alpha;
        c.strokeStyle = color;
        c.lineWidth = width;
        c.setLineDash(dash);
        c.beginPath();
        pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
        c.stroke();
        c.setLineDash([]);
        c.globalAlpha = 1;
      };

      // axes
      const axes = [
        [g.X * 1.12, 0, 0, "Qx"], [0, g.Y * 1.25, 0, "Qy"], [0, 0, Zr * 1.08, "Qz"],
      ];
      for (const [ax, ay, az, name] of axes) {
        const a = P(-ax * 0.15, -ay * 0.15, -az * 0.15), b = P(ax, ay, az);
        stroke([a, b], t.muted, 1, 0.8);
        c.fillStyle = t.muted;
        c.fillText(name, b.x + 4, b.y - 4);
      }

      // Ewald sheet near the origin (wireframe)
      const nx = 16, ny = 8;
      const lines = [];
      for (let j = 0; j <= ny; j++) {
        const qy = -g.Y + (2 * g.Y * j) / ny;
        const pts = [];
        for (let i = 0; i <= 48; i++) {
          const qx = -g.X + (2 * g.X * i) / 48;
          const qz = sheetQz(g.k, s.omega, qx, qy);
          if (qz !== null && Math.abs(qz) <= Zr) pts.push(P(qx, qy, qz));
        }
        lines.push({ pts, main: j === ny / 2 });
      }
      for (let i = 0; i <= nx; i++) {
        const qx = -g.X + (2 * g.X * i) / nx;
        const pts = [];
        for (let j = 0; j <= 24; j++) {
          const qy = -g.Y + (2 * g.Y * j) / 24;
          const qz = sheetQz(g.k, s.omega, qx, qy);
          if (qz !== null && Math.abs(qz) <= Zr) pts.push(P(qx, qy, qz));
        }
        lines.push({ pts, main: false });
      }
      for (const ln of lines) {
        if (ln.main) continue;
        const d = ln.pts.length ? ln.pts[ln.pts.length >> 1].d : 0;
        stroke(ln.pts, t.accent, 1, 0.45 * fade(d));
      }

      // rods
      for (let m = -s.mmax; m <= s.mmax; m++) {
        const a = P(m * g.G, 0, -Zr), b = P(m * g.G, 0, Zr);
        const color = m === 0 ? t.muted : orderColor(m);
        stroke([a, b], color, m === 0 ? 1 : 1.6, m === 0 ? 0.45 : 0.75);
        if (m !== 0) {
          c.fillStyle = color;
          c.fillText(m > 0 ? `+${m}` : `${m}`, a.x - 7, a.y + 14);
        }
      }

      // Ewald arc at Qy = 0, drawn over the rods
      const arc = lines.find((ln) => ln.main);
      if (arc) stroke(arc.pts, t.accentStrong, 2.4, 0.95);

      // intersections
      for (let m = -s.mmax; m <= s.mmax; m++) {
        if (m === 0) continue;
        const qz = rodQz(g.k, g.G, s.omega, m);
        if (qz === null || Math.abs(qz) > Zr) continue;
        const p = P(m * g.G, 0, qz);
        c.beginPath();
        c.arc(p.x, p.y, 5.5, 0, 2 * Math.PI);
        c.fillStyle = orderColor(m);
        c.strokeStyle = t.bg;
        c.lineWidth = 2;
        c.fill();
        c.stroke();
      }
      const o = P(0, 0, 0);
      c.beginPath();
      c.arc(o.x, o.y, 3, 0, 2 * Math.PI);
      c.fillStyle = t.fg;
      c.fill();

      this.drawInset(c, w, h, g);
    }

    drawInset(c, w, h, g) {
      // Whole Ewald circle (Qy = 0 section) at true scale, with k_in and the zoom box.
      const t = this.theme, s = this.state;
      const size = Math.min(132, w * 0.28);
      const x0 = w - size - 8, y0 = 8;
      const R = size * 0.36;
      const cx = x0 + size / 2, cyy = y0 + size / 2 + R * 0.25;
      const sw = Math.sin(s.omega * D2R), cw = Math.cos(s.omega * D2R);
      // sphere centre C = (k sin w, -k cos w); origin O on the sphere
      const Ox = cx - R * sw, Oy = cyy - R * cw;
      c.fillStyle = t.bg;
      c.strokeStyle = t.border;
      c.lineWidth = 1;
      c.beginPath();
      c.rect(x0, y0, size, size);
      c.fill();
      c.stroke();
      c.strokeStyle = t.accent;
      c.globalAlpha = 0.9;
      c.beginPath();
      c.arc(cx, cyy, R, 0, 2 * Math.PI);
      c.stroke();
      c.globalAlpha = 1;
      // k_in from C to O
      c.strokeStyle = t.fg;
      c.lineWidth = 1.3;
      c.beginPath();
      c.moveTo(cx, cyy);
      c.lineTo(Ox, Oy);
      c.stroke();
      const ang = Math.atan2(Oy - cyy, Ox - cx);
      c.beginPath();
      c.moveTo(Ox, Oy);
      c.lineTo(Ox - 7 * Math.cos(ang - 0.4), Oy - 7 * Math.sin(ang - 0.4));
      c.lineTo(Ox - 7 * Math.cos(ang + 0.4), Oy - 7 * Math.sin(ang + 0.4));
      c.closePath();
      c.fillStyle = t.fg;
      c.fill();
      // zoom box around the origin
      const bx = (g.X / g.k) * R, bz = (g.Z / g.k) * R;
      c.strokeStyle = t.accentStrong;
      c.lineWidth = 1.2;
      c.strokeRect(Ox - bx, Oy - bz, 2 * bx, 2 * bz);
      c.fillStyle = t.muted;
      c.font = `11px ${t.font}`;
      c.fillText("whole sphere", x0 + 6, y0 + 14);
      c.fillText("kᵢ", (cx + Ox) / 2 + 5, (cyy + Oy) / 2);
      c.font = `12px ${t.font}`;
    }

    draw2d(g) {
      const { c, w, h } = this.ctx(this.cv2d);
      const t = this.theme, s = this.state;
      const pad = { l: 56, r: 14, t: 12, b: 38 };
      const sx = (qx) => pad.l + ((qx + g.X) / (2 * g.X)) * (w - pad.l - pad.r);
      const sz = (qz) => pad.t + ((g.Z - qz) / (2 * g.Z)) * (h - pad.t - pad.b);

      // frame and ticks
      c.strokeStyle = t.border;
      c.lineWidth = 1;
      c.strokeRect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
      c.fillStyle = t.muted;
      const tick = (v) => Math.abs(v) < 1e-9 ? "0" : v.toFixed(2);
      for (let i = -2; i <= 2; i++) {
        const qx = (i / 2) * g.X * 0.9, qz = (i / 2) * g.Z * 0.9;
        c.fillText(tick(qx), sx(qx) - 12, h - pad.b + 16);
        c.fillText(tick(qz), 22, sz(qz) + 4);
      }
      c.fillText("Qx (nm⁻¹)", (pad.l + w - pad.r) / 2 - 24, h - 6);
      c.save();
      c.translate(13, (pad.t + h - pad.b) / 2 + 24);
      c.rotate(-Math.PI / 2);
      c.fillText("Qz (nm⁻¹)", 0, 0);
      c.restore();

      c.save();
      c.beginPath();
      c.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
      c.clip();

      // butterfly: arcs for every omega in the sweep
      const arcPts = (omega) => {
        const pts = [];
        for (let i = 0; i <= 120; i++) {
          const qx = -g.X + (2 * g.X * i) / 120;
          const qz = sheetQz(g.k, omega, qx, 0);
          if (qz !== null) pts.push([sx(qx), sz(qz)]);
        }
        return pts;
      };
      const poly = (pts, color, width, alpha = 1, dash = []) => {
        c.globalAlpha = alpha;
        c.strokeStyle = color;
        c.lineWidth = width;
        c.setLineDash(dash);
        c.beginPath();
        pts.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
        c.stroke();
        c.setLineDash([]);
        c.globalAlpha = 1;
      };
      for (let wv = -OMEGA_RANGE; wv <= OMEGA_RANGE; wv += 1) {
        poly(arcPts(wv), t.accent, 1.2, 0.13);
      }

      // rods and their sampled Qz range over the sweep
      for (let m = -s.mmax; m <= s.mmax; m++) {
        const x = sx(m * g.G);
        poly([[x, sz(-g.Z)], [x, sz(g.Z)]], m === 0 ? t.muted : orderColor(m),
             m === 0 ? 1 : 1.3, m === 0 ? 0.35 : 0.5);
      }

      // current arc: exact and small-angle
      poly(arcPts(s.omega), t.accentStrong, 2.4, 0.95);
      const tn = Math.tan(s.omega * D2R);
      poly([[sx(-g.X), sz(-g.X * tn)], [sx(g.X), sz(g.X * tn)]], t.fg, 1.2, 0.7, [5, 4]);

      for (let m = -s.mmax; m <= s.mmax; m++) {
        if (m === 0) continue;
        const qz = rodQz(g.k, g.G, s.omega, m);
        if (qz === null) continue;
        c.beginPath();
        c.arc(sx(m * g.G), sz(qz), 5, 0, 2 * Math.PI);
        c.fillStyle = orderColor(m);
        c.strokeStyle = t.bg;
        c.lineWidth = 2;
        c.fill();
        c.stroke();
      }
      c.restore();
    }

    drawReadout(g) {
      const s = this.state;
      const lam = HC / s.energy;
      const ms = [];
      for (let m = -s.mmax; m <= s.mmax; m++) if (m !== 0) ms.push(m);
      const cell = (v) => `<td>${fmt(v)}</td>`;
      const head = ms.map((m) => `<th style="color:${orderColor(m)}">${m > 0 ? "+" : ""}${m}</th>`);
      const exact = ms.map((m) => rodQz(g.k, g.G, s.omega, m));
      const flat = ms.map((m) => m * g.G * Math.tan(s.omega * D2R));
      this.readout.innerHTML =
        `<div><b>k</b> = ${fmt(g.k, 3)} nm⁻¹ · <b>λ</b> = ${fmt(lam, 3)} nm · `
        + `<b>G</b> = ${fmt(g.G, 4)} nm⁻¹</div>`
        + `<table class="ewald-figure__table"><thead><tr><th>order m</th>${head.join("")}</tr></thead>`
        + `<tbody><tr><th>Q<sub>z</sub> exact</th>${exact.map(cell).join("")}</tr>`
        + `<tr><th>Q<sub>z</sub> small-angle</th>${flat.map(cell).join("")}</tr></tbody></table>`
        + `<div class="ewald-figure__muted">Q<sub>z</sub> in nm⁻¹; “—” means the rod does not reach the sphere at this ω.</div>`;
    }

    async checkFixtures(url) {
      try {
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        let maxErr = 0, n = 0, ok = true;
        for (const cse of data.cases) {
          const k = (2 * Math.PI * cse.E) / data.hc;
          const q = rodQz(k, (2 * Math.PI) / cse.pitch, cse.omega, cse.m);
          n += 1;
          if (cse.qz === null || q === null) ok = ok && cse.qz === q;
          else maxErr = Math.max(maxErr, Math.abs(q - cse.qz));
        }
        ok = ok && maxErr < 1e-9;
        this.badge.textContent = ok
          ? `✓ Same formula as cd_reduce’s geometry: ${n} reference cases, max |ΔQz| = ${maxErr.toExponential(1)} nm⁻¹`
          : `✗ Figure disagrees with the reference cases (max |ΔQz| = ${maxErr.toExponential(1)} nm⁻¹)`;
        this.badge.classList.toggle("is-bad", !ok);
      } catch (err) {
        /* offline or file:// — leave the badge empty */
      }
    }
  }

  function init() {
    document.querySelectorAll(".ewald-figure[data-ewald]").forEach((root) => {
      if (!root.dataset.ewaldReady) {
        root.dataset.ewaldReady = "1";
        new EwaldFigure(root);
      }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
