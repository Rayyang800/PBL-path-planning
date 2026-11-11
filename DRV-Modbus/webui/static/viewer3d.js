// 立體模型檢視器 (純 WebGL,不需外部函式庫)
// 讀 runs/<id>/data/model.ply (scan3d.py 產生;紙面座標 mm:x 右、y 下、z = 離紙面高度)
// 左鍵拖曳 = 旋轉、右鍵 / Shift+拖曳 = 平移、滾輪 = 縮放、雙擊 = 重設
"use strict";
const Viewer3D = (() => {
  let gl, canvas, prog, lineProg, buf = {}, nPts = 0, nLines = 0, info = null;
  const st = { yaw: -0.6, pitch: 0.75, dist: 300, tx: 0, ty: 0, tz: 10, size: 2.5, mode: "rgb", objOnly: false, zmax: 30, ctr: [0, 0] };

  // ---------- 小型矩陣工具 (column-major) ----------
  const persp = (fov, a, n, f) => { const t = 1 / Math.tan(fov / 2); return [t / a, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0]; };
  const mul = (a, b) => { const o = new Array(16).fill(0); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k]; return o; };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(...a) || 1; return a.map(v => v / l); };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function lookAt(eye, tgt, up) {
    const z = norm(sub(eye, tgt)), x = norm(cross(up, z)), y = cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
  }

  const VS = `attribute vec3 p; attribute vec3 c; uniform mat4 M; uniform float size; uniform float zmax; uniform int mode; uniform float minz;
    varying vec3 vc; varying float keep;
    vec3 turbo(float t){ t=clamp(t,0.,1.); return clamp(vec3(0.13572138+t*(4.6153926+t*(-42.66032258+t*(132.13108234+t*(-152.94239396+t*59.28637943)))),
      0.09140261+t*(2.19418839+t*(4.84296658+t*(-14.18503333+t*(4.27729857+t*2.82956604)))),
      0.1066733+t*(12.64194608+t*(-60.58204836+t*(110.36276771+t*(-89.90310912+t*27.34824973))))),0.,1.); }
    void main(){ gl_Position = M*vec4(p,1.); gl_PointSize = size; keep = p.z >= minz ? 1. : 0.;
      vc = mode==1 ? turbo(p.z/zmax) : c; }`;
  const FS = `precision mediump float; varying vec3 vc; varying float keep; void main(){ if(keep<0.5) discard; gl_FragColor=vec4(vc,1.); }`;
  const LVS = `attribute vec3 p; attribute vec3 c; uniform mat4 M; varying vec3 vc; void main(){ gl_Position=M*vec4(p,1.); vc=c; }`;
  const LFS = `precision mediump float; varying vec3 vc; void main(){ gl_FragColor=vec4(vc,1.); }`;

  function compile(vs, fs) {
    const mk = (t, s) => { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh; };
    const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p); return p;
  }

  function init(el) {
    canvas = el;
    gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) { canvas.replaceWith(Object.assign(document.createElement("div"), { textContent: "瀏覽器不支援 WebGL" })); return false; }
    prog = compile(VS, FS); lineProg = compile(LVS, LFS);
    let drag = null;
    canvas.addEventListener("contextmenu", e => e.preventDefault());
    canvas.addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; });
    window.addEventListener("mouseup", () => { drag = null; });
    window.addEventListener("mousemove", e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        // 畫面右方 = (cos yaw, sin yaw, 0);畫面上方 = (-sin yaw·sin pitch, cos yaw·sin pitch, cos pitch)
        const k = st.dist / 700, cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), sp = Math.sin(st.pitch), cp = Math.cos(st.pitch);
        st.tx += -dx * cy * k + dy * (-sy * sp) * k;
        st.ty += -dx * sy * k + dy * (cy * sp) * k;
        st.tz += dy * cp * k;
      } else { st.yaw -= dx * 0.008; st.pitch = Math.max(-0.05, Math.min(1.55, st.pitch + dy * 0.008)); }
      draw();
    });
    canvas.addEventListener("wheel", e => { e.preventDefault(); st.dist *= Math.exp(e.deltaY * 0.001); st.dist = Math.max(20, Math.min(3000, st.dist)); draw(); }, { passive: false });
    canvas.addEventListener("dblclick", () => { reset(); draw(); });
    new ResizeObserver(() => draw()).observe(canvas);
    return true;
  }

  function reset() { st.yaw = -0.6; st.pitch = 0.75; st.tx = 0; st.ty = 0; st.tz = st.zmax * 0.4; st.dist = st.extent * 2.2 || 300; }
  function view(name) {
    if (name === "top") { st.yaw = 0; st.pitch = 1.55; }
    else if (name === "front") { st.yaw = 0; st.pitch = 0.02; }
    else if (name === "side") { st.yaw = Math.PI / 2; st.pitch = 0.02; }
    else reset();
    draw();
  }

  function upload(name, data, size) {
    buf[name] = buf[name] || gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf[name]); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  }

  // 紙面 (x 右, y 下) -> 世界 (X 右, Y 往紙的上方, Z 高度),以物體為中心
  const W = (x, y, z) => [x - st.ctr[0], -(y - st.ctr[1]), z];

  function parsePly(text) {
    const end = text.indexOf("end_header");
    const head = text.slice(0, end), n = +(/element vertex (\d+)/.exec(head) || [0, 0])[1];
    const lines = text.slice(text.indexOf("\n", end) + 1).split("\n");
    const P = new Float32Array(n * 3), C = new Float32Array(n * 3);
    let k = 0;
    for (let i = 0; i < lines.length && k < n; i++) {
      const s = lines[i].split(" "); if (s.length < 6) continue;
      P[k * 3] = +s[0]; P[k * 3 + 1] = +s[1]; P[k * 3 + 2] = +s[2];
      C[k * 3] = s[3] / 255; C[k * 3 + 1] = s[4] / 255; C[k * 3 + 2] = s[5] / 255; k++;
    }
    return { P: P.subarray(0, k * 3), C: C.subarray(0, k * 3), n: k };
  }

  function buildLines(P, m) {
    const L = [], Cc = [];
    const seg = (a, b, col) => { L.push(...a, ...b); Cc.push(...col, ...col); };
    // 紙面格線 (10 mm),範圍 = 點雲外框
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (let i = 0; i < P.length; i += 3) { x0 = Math.min(x0, P[i]); x1 = Math.max(x1, P[i]); y0 = Math.min(y0, P[i + 1]); y1 = Math.max(y1, P[i + 1]); }
    const g = [0.55, 0.6, 0.65];
    for (let x = Math.ceil(x0 / 10) * 10; x <= x1; x += 10) seg(W(x, y0, 0), W(x, y1, 0), g);
    for (let y = Math.ceil(y0 / 10) * 10; y <= y1; y += 10) seg(W(x0, y, 0), W(x1, y, 0), g);
    // 座標軸:紅 = 紙面 x、綠 = 紙面 y (往下)、藍 = 高度
    const o = [x0, y0];
    seg(W(o[0], o[1], 0), W(o[0] + 20, o[1], 0), [1, 0.2, 0.2]);
    seg(W(o[0], o[1], 0), W(o[0], o[1] + 20, 0), [0.2, 0.9, 0.2]);
    seg(W(o[0], o[1], 0), W(o[0], o[1], 20), [0.3, 0.5, 1]);
    // 量測結果:底面外框 (橘,z=0)、頂面 (綠,z = 頂面高度) 與四條垂直邊
    if (m) {
      const box = (b, z, col) => { for (let i = 0; i < 4; i++) seg(W(...b[i], z), W(...b[(i + 1) % 4], z), col); };
      if (m.footprint_rect) box(m.footprint_rect.box, 0.2, [1, 0.55, 0]);
      if (m.top_rect) {
        box(m.top_rect.box, m.h_top_mm, [0.1, 1, 0.3]);
        for (let i = 0; i < 4; i++) seg(W(...m.top_rect.box[i], 0), W(...m.top_rect.box[i], m.h_top_mm), [0.1, 0.8, 0.3]);
      }
    }
    return { L: new Float32Array(L), C: new Float32Array(Cc) };
  }

  async function load(runId) {
    const base = `/files/runs/${encodeURIComponent(runId)}/data/`;
    const r = await fetch(base + "model.ply");
    if (!r.ok) throw new Error("這個紀錄沒有 model.ply");
    const ply = parsePly(await r.text());
    let res = null;
    try { const rr = await fetch(base + "result.json"); if (rr.ok) res = await rr.json(); } catch (e) { /* 沒有也可以 */ }
    const m = res && (res.object || res.cube);
    // 中心 = 物體 (頂面) 中心;沒有就用點雲中心
    let cx = 0, cy = 0, zmax = 1, zs = [];
    for (let i = 0; i < ply.n; i++) { cx += ply.P[i * 3]; cy += ply.P[i * 3 + 1]; zs.push(ply.P[i * 3 + 2]); }
    st.ctr = m && m.top_rect ? m.top_rect.center : [cx / ply.n, cy / ply.n];
    zs.sort((a, b) => a - b); zmax = Math.max(5, zs[Math.floor(zs.length * 0.995)] || 5);
    st.zmax = zmax;
    const P = new Float32Array(ply.n * 3);
    let ext = 0;
    for (let i = 0; i < ply.n; i++) {
      const w = W(ply.P[i * 3], ply.P[i * 3 + 1], ply.P[i * 3 + 2]);
      P.set(w, i * 3); ext = Math.max(ext, Math.hypot(w[0], w[1]));
    }
    st.extent = Math.min(ext, 150);
    upload("p", P); upload("c", ply.C); nPts = ply.n;
    const ln = buildLines(ply.P, m);
    upload("lp", ln.L); upload("lc", ln.C); nLines = ln.L.length / 3;
    info = { run: runId, n: ply.n, result: res, m };
    reset(); draw();
    return info;
  }

  function bindAttr(p, name, b) {
    const loc = gl.getAttribLocation(p, name); gl.bindBuffer(gl.ARRAY_BUFFER, buf[b]);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  }

  function draw() {
    if (!gl) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.11, 0.14, 0.16, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    if (!nPts) return;
    const tgt = [st.tx, st.ty, st.tz];
    const eye = [tgt[0] + st.dist * Math.cos(st.pitch) * Math.sin(st.yaw), tgt[1] - st.dist * Math.cos(st.pitch) * Math.cos(st.yaw),
      tgt[2] + st.dist * Math.sin(st.pitch)];
    const up = st.pitch > 1.5 ? [Math.sin(st.yaw), Math.cos(st.yaw), 0] : [0, 0, 1];
    const M = mul(persp(0.8, w / Math.max(1, h), 2, 5000), lookAt(eye, tgt, up));
    gl.useProgram(lineProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "M"), false, new Float32Array(M));
    bindAttr(lineProg, "p", "lp"); bindAttr(lineProg, "c", "lc");
    gl.drawArrays(gl.LINES, 0, nLines);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "M"), false, new Float32Array(M));
    gl.uniform1f(gl.getUniformLocation(prog, "size"), st.size * dpr);
    gl.uniform1f(gl.getUniformLocation(prog, "zmax"), st.zmax);
    gl.uniform1f(gl.getUniformLocation(prog, "minz"), st.objOnly ? 1.5 : -1e9);
    gl.uniform1i(gl.getUniformLocation(prog, "mode"), st.mode === "height" ? 1 : 0);
    bindAttr(prog, "p", "p"); bindAttr(prog, "c", "c");
    gl.drawArrays(gl.POINTS, 0, nPts);
  }

  function set(k, v) { st[k] = v; draw(); }
  return { init, load, view, set, draw, get info() { return info; } };
})();
