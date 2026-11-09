"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
let VIEW = "color", CFG = null, WORKPIECES = {}, TASKS = {}, LAST_STATUS = null;

function toast(msg, ms = 2600) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._h); toast._h = setTimeout(() => t.classList.remove("show"), ms);
}
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.ok === false) throw new Error(j.error || r.statusText);
  return j;
}
const fmt = (v, d = 1) => (v === null || v === undefined || v === "") ? "—" : (typeof v === "number" ? v.toFixed(d) : String(v));
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function badge(el, text, cls) { el.textContent = text; el.className = "badge " + (cls || ""); }
function kv(el, rows) { el.innerHTML = rows.map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${v}</div>`).join(""); }
function ago(iso) {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return `${Math.round(s)} 秒前`; if (s < 5400) return `${Math.round(s / 60)} 分前`;
  if (s < 172800) return `${Math.round(s / 3600)} 小時前`; return `${Math.round(s / 86400)} 天前`;
}

// ---------------- 影像 ----------------
function setView(v) {
  VIEW = v;
  $$("#view-tabs button").forEach(b => b.classList.toggle("active", b.dataset.view === v));
  $("#video").src = `/stream.mjpg?view=${v}&overlay=${$("#chk-overlay").checked ? 1 : 0}&t=${Date.now()}`;
}
$$("#view-tabs button").forEach(b => b.onclick = () => setView(b.dataset.view));
$("#chk-overlay").onchange = () => setView(VIEW);
$("#video").onerror = () => setTimeout(() => setView(VIEW), 2000);

// ---------------- 狀態 ----------------
async function pollStatus() {
  try {
    const s = await api("/api/status");
    LAST_STATUS = s;
    renderStatus(s);
  } catch (e) {
    badge($("#b-robot"), "伺服器離線", "bad");
  }
  setTimeout(pollStatus, 1000);
}
function renderStatus(s) {
  const r = s.robot || {};
  if (r.ok) {
    const flagTxt = r.flag === 1 ? "1 (可接受指令)" : `${r.flag} (忙碌或異常)`;
    badge($("#b-robot"), `手臂 ${r.flag === 1 ? "待命" : "旗標 " + r.flag}${r.at_home ? " · home" : ""}`, r.flag === 1 ? "ok" : "warn");
    $("#robot-msg").textContent = r.msg;
    const p = r.pose || [];
    kv($("#robot-kv"), [
      ["位置 (mm)", p.slice(0, 3).map(v => v.toFixed(1)).join(", ")],
      ["姿態 (度)", p.slice(3).map(v => v.toFixed(1)).join(", ")],
      ["旗標 0x031F", flagTxt],
      ["離 home", r.home_dpos !== undefined ? `${r.home_dpos} mm / ${r.home_dang}°${r.at_home ? " ✓" : ""}` : "—"],
      ["夾爪 (0x02FE)", `${esc(r.gripper)} (${r.do})`],
    ]);
  } else {
    badge($("#b-robot"), "手臂 " + (s.task.running ? "任務中" : "離線"), s.task.running ? "warn" : "bad");
    $("#robot-msg").textContent = r.msg || "";
    kv($("#robot-kv"), [["狀態", esc(r.msg || "—")]]);
  }
  const c = s.camera || {};
  badge($("#b-camera"), `相機 ${c.state}${c.fps ? " · " + c.fps + " fps" : ""}`, c.state === "串流中" ? "ok" : (c.state.startsWith("已釋放") ? "warn" : "bad"));
  const pp = s.paper || {};
  badge($("#b-paper"), pp.ok ? `校正紙 4/4 · ${fmt(pp.rms_mm, 2)} mm` : "校正紙 未完整", pp.ok ? "ok" : "warn");
  $("#paper-line").textContent = `${pp.desc || "—"}${pp.aruco && pp.aruco.length ? "  |  舊 ArUco: " + pp.aruco.join(", ") : ""}` +
    (c.info ? `  |  ${c.info.name || ""} ${c.info.serial || ""} USB ${c.info.usb || ""}` : "");
  const t = s.task || {};
  if (t.running) {
    badge($("#b-task"), "任務:" + t.current.label + " (" + t.current.mode + ")", "warn");
    $("#task-box").innerHTML = `<b>${esc(t.current.label)}</b> (${esc(t.current.mode)}) · PID ${t.current.pid} · 已執行 ${Math.round(s.time - t.current.started)} 秒
      <pre class="cmd">${esc(t.current.cmdline)}</pre>`;
  } else {
    badge($("#b-task"), "任務 無", "");
    $("#task-box").innerHTML = t.last ? `上一個:<b>${esc(t.last.label)}</b> (${esc(t.last.mode)}),結束碼 ${t.last.exit_code}` +
      (t.last.run ? ` · 紀錄 <a href="#" data-run="${esc(t.last.run)}">${esc(t.last.run)}</a>` : "") : "沒有任務在執行";
  }
  renderKeybar(t);
  $$("[data-launch]").forEach(b => b.disabled = !!t.running);
  $$("[data-grip]").forEach(b => b.disabled = !!t.running || !r.ok);
  $("#btn-kill").disabled = !t.running;
  renderCalib(s.calib || {});
}
function renderCalib(cal) {
  const p = cal.pen_paper, w = cal.warp_plane, rows = [];
  if (p) {
    const err = p.loo_max_mm != null ? `留一法最大 ${fmt(p.loo_max_mm, 2)} mm` :
      (p.est_err_at_100mm != null ? `估 100 mm 外 ${fmt(p.est_err_at_100mm, 1)} mm` : "—");
    rows.push(["筆尖校正", `<b>${esc(p.method || "?")}</b>${p.fit ? " / " + esc(p.fit) : ""} · ${ago(p.created)}`],
      ["精度", `RMS ${fmt(p.rms_mm, 2)} mm · ${err}`],
      ["點數 / 範圍", `${fmt(p.n_points, 0)} 點${p.span_mm ? " · " + p.span_mm.map(v => v.toFixed(0)).join("×") + " mm" : ""}`],
      ["z_touch", fmt(p.z_touch, 2)],
      ["來源", p.source_run ? `<a href="#" data-run="${esc(p.source_run)}">${esc(p.source_run)}</a> (共 ${p.n_versions} 版)` : "—"]);
  } else rows.push(["筆尖校正", "<span class='muted'>尚未完成任何一次</span>"]);
  rows.push(["舊 ArUco 平面", w ? `${ago(w.created)} (粗略換算用)` : "—"]);
  kv($("#calib-kv"), rows);
}

// ---------------- 任務執行中的按鍵提示列 ----------------
const K = s => s.split(" ").map(k => `<kbd>${esc(k)}</kbd>`).join(" ");
const COMMON_KEYS = [["Ctrl+C", "停止→抬起→回 home"], ["空白", "暫停選單"], ["o g", "夾爪開/夾"]];
const PROBE_KEYS = [["Enter", "往下"], ["2 0.5", "改步距"], ["u", "往上"], ["x 2", "水平微調"], ["c", "記下高度"]];
const TASK_KEYS = {
  sheet_calib: { hint: "第一點:人工下降到剛好畫得出來按 c,之後自動", keys: [...PROBE_KEYS, ...COMMON_KEYS] },
  quick_calib: { hint: "第一點:人工下降到剛好點出顏色按 c,之後自動", keys: [...PROBE_KEYS, ...COMMON_KEYS] },
  go_home: { hint: "確認路徑安全後在視窗按 Enter 開始移動", keys: [["Enter", "開始"], ["q", "取消"], ["Ctrl+C", "停止"]] },
  gripper_ctl: { hint: "按鍵直接生效,不用 Enter", keys: [["o", "張開"], ["g", "夾緊"], ["s", "讀狀態"], ["q", "離開"]] },
  seam_boards: { hint: "唯讀,不動手臂;結束後按任意鍵關閉視窗", keys: [] },
  warp_sort: { hint: "dry-run 不動手臂;--go 時每步確認", keys: [["Enter", "確認"], ["Ctrl+C", "停止"]] },
};
function renderKeybar(t) {
  const bar = $("#keybar");
  if (!t.running) { bar.hidden = true; return; }
  const spec = TASK_KEYS[t.current.task] || { hint: "", keys: COMMON_KEYS };
  bar.innerHTML = `<span class="title">▶ ${esc(t.current.label)} (${esc(t.current.mode)}) — 請在新開的主控台視窗操作</span>` +
    (spec.hint ? `<span class="item">${esc(spec.hint)}</span>` : "") +
    spec.keys.map(([k, d]) => `<span class="item">${k.includes("+") ? k.split("+").map(x => `<kbd>${esc(x)}</kbd>`).join("+") : K(k)} ${esc(d)}</span>`).join("") +
    `<span class="item warn-txt">急停在手邊</span>`;
  bar.hidden = false;
}
$("#btn-keys-toggle").onclick = () => {
  const body = $("#keys-body"), hide = !body.hidden;
  body.hidden = hide; $("#btn-keys-toggle").textContent = hide ? "展開" : "收起";
  try { localStorage.setItem("keysHidden", hide ? "1" : "0"); } catch (e) { /* 無痕模式等 */ }
};
try { if (localStorage.getItem("keysHidden") === "1") { $("#keys-body").hidden = true; $("#btn-keys-toggle").textContent = "展開"; } } catch (e) { /* ignore */ }

// ---------------- 紀錄 ----------------
async function loadRuns() {
  try {
    const runs = await api("/api/runs?limit=40");
    $("#runs-body").innerHTML = runs.map(r => {
      const sum = Object.entries(r.summary || {}).map(([k, v]) => `${k}=${typeof v === "number" ? (+v).toFixed(2) : v}`).join(" ");
      return `<tr data-run="${esc(r.id)}"><td>${esc((r.started || "").replace("T", " ").slice(5, 19))}</td>
        <td>${esc(r.kind)}${r.label ? " <span class='muted'>" + esc(r.label) + "</span>" : ""}</td>
        <td><span class="st ${esc(r.status)}">${esc(r.status)}</span></td><td class="sum">${esc(sum)}</td><td>${r.n_snap || ""}</td></tr>`;
    }).join("") || `<tr><td colspan="5" class="muted">還沒有紀錄</td></tr>`;
  } catch (e) { toast("讀取紀錄失敗:" + e.message); }
}
document.addEventListener("click", e => {
  const a = e.target.closest("[data-run]");
  if (a) { e.preventDefault(); openRun(a.dataset.run); }
});
async function openRun(id) {
  try {
    const d = await api("/api/run?id=" + encodeURIComponent(id));
    const m = d.meta;
    $("#dlg-title").textContent = id;
    const imgs = d.images.filter(n => /\.(png|jpe?g)$/i.test(n));
    const res = d.results["marks_result.json"] || d.results["result.json"];
    let resTxt = "";
    if (res) {
      const keys = ["ok", "fit", "rms_mm", "max_mm", "loo_mean_mm", "loo_max_mm", "n_points", "rotation_deg", "rotation_vs_rough_deg",
        "est_err_at_100mm", "gap_mm", "length_mm", "reason"];
      resTxt = keys.filter(k => res[k] !== undefined).map(k => `${k}: ${typeof res[k] === "number" ? res[k].toFixed(3) : res[k]}`).join("\n");
    }
    $("#dlg-body").innerHTML = `
      <div class="kv">
        <div class="k">類型 / 狀態</div><div class="v">${esc(m.kind)} · <span class="st ${esc(m.status)}">${esc(m.status)}</span></div>
        <div class="k">時間</div><div class="v">${esc(m.started)} → ${esc(m.ended || "")}</div>
        <div class="k">指令</div><div class="v">${esc((m.script || {}).name)} ${esc(((m.script || {}).argv || []).join(" "))}</div>
        <div class="k">程式版本</div><div class="v">${m.git ? esc((m.git.commit || "").slice(0, 8)) + (m.git.dirty_files && m.git.dirty_files.length ? " (有未提交修改)" : "") : "—"}</div>
        <div class="k">相機</div><div class="v">${m.camera ? esc(m.camera.name + " " + m.camera.serial) : "—"}</div>
        <div class="k">事件數</div><div class="v">${d.n_events}(含手臂指令與軌跡)</div>
        <div class="k">結果</div><div class="v">${esc(JSON.stringify(m.result || {}))}</div>
      </div>
      ${resTxt ? `<h2 style="margin-top:10px">分析結果</h2><pre class="cmd">${esc(resTxt)}</pre>` : ""}
      <h2 style="margin-top:10px">影像 (${imgs.length})</h2>
      <div class="thumbs">${imgs.map(n => `<figure><a href="/files/runs/${encodeURIComponent(id)}/images/${encodeURIComponent(n)}" target="_blank">
        <img loading="lazy" src="/files/runs/${encodeURIComponent(id)}/images/${encodeURIComponent(n)}"></a><figcaption>${esc(n)}</figcaption></figure>`).join("")}</div>
      ${d.data.length ? `<div class="small muted">data/:${d.data.map(n => `<a target="_blank" href="/files/runs/${encodeURIComponent(id)}/data/${encodeURIComponent(n)}">${esc(n)}</a>`).join("、")}</div>` : ""}
      ${d.console ? `<h2 style="margin-top:10px">終端輸出 (最後 120 行)</h2><pre class="console">${esc(d.console)}</pre>` : ""}`;
    $("#dlg-run").showModal();
  } catch (e) { toast("讀取失敗:" + e.message); }
}
$("#dlg-close").onclick = () => $("#dlg-run").close();
$("#btn-refresh-runs").onclick = loadRuns;

// ---------------- 設定表單 ----------------
const FIELDS = [
  ["sec", "工件"],
  ["workpiece.type", "工件類型", "select"],
  ["workpiece.surface_h", "工件表面比校正紙高 (mm)", "number"],
  ["seam.board_h", "板厚 (縫合線擷取視差修正, mm)", "number"],
  ["workpiece.notes", "備註", "text", "full"],
  ["sec", "共通"],
  ["common.speed", "手臂速度 (1~30)", "number"],
  ["common.pen_loaded", "筆已夾好 (--pen-loaded)", "checkbox"],
  ["sec", "護貝紙整張校正"],
  ["sheet.cols", "格點 欄", "number"], ["sheet.rows", "格點 列", "number"],
  ["sheet.margin", "離 tag 內緣 (mm)", "number"], ["sheet.stroke", "十字半長 (mm)", "number"],
  ["sheet.hop", "點間抬高 (mm)", "number"], ["sheet.press", "多壓 (mm)", "number"],
  ["sheet.touch_all", "每點人工確認高度", "checkbox"], ["sheet.no_refine", "不回 home 修正", "checkbox"],
  ["sec", "快速校正"],
  ["quick.step", "偏移 step (mm)", "number"], ["quick.hop", "點間抬高 (mm)", "number"],
  ["quick.start_x", "第一點紙面 X (空 = 紙中心)", "number"], ["quick.start_y", "第一點紙面 Y", "number"],
];
function getPath(o, p) { return p.split(".").reduce((a, k) => (a || {})[k], o); }
function buildForm() {
  $("#cfg-form").innerHTML = FIELDS.map(f => {
    if (f[0] === "sec") return `<div class="sec">${esc(f[1])}</div>`;
    const [path, label, type, full] = f, v = getPath(CFG, path);
    if (type === "select") return `<label>${esc(label)}<select name="${path}">${Object.entries(WORKPIECES).map(([k, t]) =>
      `<option value="${k}" ${k === v ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></label>`;
    if (type === "checkbox") return `<label class="inline"><input type="checkbox" name="${path}" ${v ? "checked" : ""}> ${esc(label)}</label>`;
    return `<label class="${full ? "full" : ""}">${esc(label)}<input name="${path}" type="${type}" step="any" value="${v ?? ""}"></label>`;
  }).join("");
}
function readForm() {
  const out = {};
  $$("#cfg-form [name]").forEach(el => {
    const [sec, key] = el.name.split(".");
    out[sec] = out[sec] || {};
    let v = el.type === "checkbox" ? el.checked : el.value;
    if (el.type === "number") v = v === "" ? null : Number(v);
    out[sec][key] = v;
  });
  return out;
}
async function loadConfig() {
  const j = await api("/api/config");
  CFG = j.config; WORKPIECES = j.workpieces; TASKS = j.tasks;
  buildForm();
}
$("#btn-save-cfg").onclick = async () => {
  const f = readForm();
  if (f.common.speed < 1 || f.common.speed > 30) return toast("速度需在 1~30");
  try { const j = await api("/api/config", f); CFG = j.config; $("#cfg-msg").textContent = "已儲存 " + new Date().toLocaleTimeString(); toast("設定已儲存"); }
  catch (e) { toast("儲存失敗:" + e.message); }
};

// ---------------- 動作 ----------------
function confirmDlg(title, text, cmd) {
  return new Promise(res => {
    $("#cf-title").textContent = title; $("#cf-text").textContent = text; $("#cf-cmd").textContent = cmd || "";
    $("#cf-cmd").style.display = cmd ? "" : "none";
    const d = $("#dlg-confirm");
    $("#cf-yes").onclick = () => { d.close(); res(true); };
    $("#cf-no").onclick = () => { d.close(); res(false); };
    d.showModal();
  });
}
$$("[data-launch]").forEach(b => b.onclick = async () => {
  const task = b.dataset.launch, mode = b.dataset.mode, info = TASKS[task] || {};
  // 先存目前表單,讓任務用最新設定
  try { const j = await api("/api/config", readForm()); CFG = j.config; } catch (e) { return toast("設定儲存失敗:" + e.message); }
  const pv = await api(`/api/preview_cmd?task=${task}&mode=${mode}`).catch(() => ({ cmd: "" }));
  const moves = mode === "go" || info.always_moves;
  if (moves) {
    const r = LAST_STATUS && LAST_STATUS.robot;
    const warn = r && r.ok && r.flag !== 1 ? "\n⚠ 手臂旗標不是 1,請先在教導器清除警報。" : "";
    const ok = await confirmDlg(`執行「${info.label}」— 手臂會動`,
      "手放急停旁、確認路徑無障礙。任務會在新視窗執行,請在那個視窗操作 (Ctrl+C = 停止回 home、空白鍵 = 暫停選單)。" + warn, pv.cmd);
    if (!ok) return;
  }
  try { await api("/api/launch", { task, mode }); toast(`已啟動「${info.label}」,請看新開的主控台視窗`); }
  catch (e) { toast("啟動失敗:" + e.message, 4000); }
});
$$("[data-grip]").forEach(b => b.onclick = async () => {
  const st = b.dataset.grip;
  if (st === "open" && !(await confirmDlg("夾爪張開", "夾著的筆或物體會掉下來,確定?"))) return;
  try { await api("/api/gripper", { state: st }); toast(st === "open" ? "夾爪張開" : "夾爪夾緊"); } catch (e) { toast("失敗:" + e.message); }
});
$("#btn-stop").onclick = async () => {
  try { const j = await api("/api/robot/stop", {}); toast(j.note, 5000); } catch (e) { toast("送出失敗:" + e.message + " — 請按實體急停!", 6000); }
};
$("#btn-kill").onclick = async () => {
  if (!(await confirmDlg("強制結束任務程式", "會直接關掉任務視窗;手臂若正在移動不一定會停 (請同時準備急停)。建議優先在任務視窗按 Ctrl+C。"))) return;
  try { await api("/api/kill", {}); toast("已結束任務程式"); } catch (e) { toast("失敗:" + e.message); }
};
$("#btn-snap").onclick = async () => {
  try { const j = await api("/api/snapshot", { label: "" }); toast("已存快照:" + j.run); loadRuns(); } catch (e) { toast("快照失敗:" + e.message); }
};

// ---------------- 啟動 ----------------
setView("color");
loadConfig().catch(e => toast("讀取設定失敗:" + e.message));
pollStatus();
loadRuns();
setInterval(loadRuns, 8000);
