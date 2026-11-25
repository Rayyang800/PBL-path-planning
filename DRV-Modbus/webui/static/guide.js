"use strict";
// 三種工件的操作圖解:全部用 SVG 畫,不用外部圖檔 (repo 公開,不放實拍照片)。
const Guide = (() => {
  const line = (d, cls) => `<path class="${cls}" d="${d}"/>`;
  const text = (x, y, t, cls = "g-t", anchor = "middle") => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${t}</text>`;
  const svg = (body, label) => `<svg viewBox="0 0 200 136" role="img" aria-label="${label}">${body}</svg>`;
  const arrow = (d) => `<path class="g-arrow" d="${d}" marker-end="url(#g-head)"/>`;

  // 校正紙 + 四個 tag
  const paper = () => `<rect class="g-paper" x="12" y="8" width="176" height="120" rx="3"/>` +
    [[20, 16], [164, 16], [20, 104], [164, 104]].map(([x, y]) =>
      `<rect class="g-tag" x="${x}" y="${y}" width="16" height="16"/><rect class="g-paper" x="${x + 5}" y="${y + 5}" width="6" height="6"/>`).join("");

  const top = {
    A: (path) => `<rect class="g-obj" x="66" y="30" width="34" height="76"/><rect class="g-obj2" x="100" y="30" width="34" height="76"/>` +
      (path ? line("M100 36V100", "g-path") : ""),
    B: (path) => `<rect class="g-obj" x="54" y="36" width="45" height="64"/><rect class="g-obj" x="101" y="36" width="45" height="64"/>` +
      (path ? line("M100 40V96", "g-path") : ""),
    C: (path) => `<circle class="g-obj" cx="100" cy="68" r="38"/><circle class="g-obj2" cx="100" cy="68" r="32"/>` +
      (path ? `<circle class="g-path" cx="100" cy="68" r="35"/>` : ""),
    D: (path) => `<circle class="g-obj" cx="100" cy="68" r="34"/><circle class="g-obj2" cx="100" cy="68" r="28"/>` +
      (path ? `<circle class="g-path" cx="100" cy="68" r="43"/>` : ""),
  };

  // 側面:筆尖在 (x, y),deg = 往外傾斜
  const pen = (x, y, deg = 0) => `<g transform="translate(${x} ${y}) rotate(${deg})">` +
    `<rect class="g-grip" x="-17" y="-88" width="34" height="38" rx="3"/><rect class="g-pen" x="-6" y="-54" width="12" height="40"/>` +
    `<path class="g-pen" d="M-6 -14H6L0 0Z"/></g>`;
  const side = {
    A: () => line("M50 126L100 92L150 126Z", "g-obj") + pen(100, 92),
    B: () => `<rect class="g-obj" x="40" y="112" width="58" height="14"/><rect class="g-obj" x="102" y="112" width="58" height="14"/>` + pen(100, 112),
    C: () => `<rect class="g-obj" x="46" y="96" width="86" height="30" rx="2"/>` + pen(130, 96),
    D: () => `<rect class="g-obj" x="30" y="100" width="72" height="26" rx="2"/>` + pen(116, 126, 18),
  };

  const button = (x, y, w, label, primary) =>
    `<rect class="${primary ? "g-btn-p" : "g-btn"}" x="${x}" y="${y}" width="${w}" height="22" rx="5"/>` +
    text(x + w / 2, y + 15, label, primary ? "g-t-inv" : "g-t");

  const INFO = {
    A: { name: "A 脊線", place: "工件放在四個 tag 中間,脊線朝上", see: "洋紅線要在脊線上", touch: "筆尖剛好碰到脊線", after: "一次連續畫完一條線" },
    B: { name: "B 縫合", place: "兩片板併好,放在四個 tag 中間", see: "橘線要在縫上、綠圈是起點", touch: "筆尖剛好碰到板面", after: "一次連續畫完一條縫" },
    C: { name: "C 罐頭頂面", place: "罐頭放在四個 tag 中間", see: "洋紅圓要在捲邊上", touch: "筆尖剛好碰到捲邊", after: "繞頂面畫一圈" },
    D: { name: "C 罐頭底面 (傾斜)", place: "先用尺量筆長填進設定;罐頭放中間,周圍 15 cm 淨空", see: "洋紅圓在罐頭外、終端顯示「檢查:通過」", touch: "先看空走一圈沒碰到,再讓筆尖碰紙", after: "傾斜繞底面外圍一圈" },
  };

  function steps(k) {
    const w = INFO[k];
    return [
      { title: "放工件", desc: w.place, art: paper() + top[k](false) },
      { title: "按「預覽」", desc: `手臂不會動。${w.see}`,
        art: paper() + top[k](true) + button(72, 110, 56, "預覽", false) },
      { title: "按「開始」,再按 Enter", desc: "手放急停旁。網頁確認後,到新開的黑色視窗按 Enter。",
        art: button(58, 8, 84, "開始", true) + arrow("M100 34V46") +
          `<rect class="g-console" x="22" y="52" width="156" height="76" rx="5"/>` +
          text(32, 74, "確認…手放急停旁", "g-t-con", "start") + text(32, 92, "按 Enter 開始 &gt;", "g-t-con", "start") +
          `<rect class="g-key" x="118" y="100" width="52" height="22" rx="4"/>` + text(144, 115, "Enter") },
      { title: "往下,碰到按 c", desc: `每按一次 Enter 往下一小步。${w.touch}就輸入 c 再按 Enter。`,
        art: line("M12 126H188", "g-floor") + side[k]() + arrow("M26 24V62") +
          text(26, 78, "Enter") + `<rect class="g-key" x="160" y="20" width="26" height="26" rx="4"/>` + text(173, 38, "c") },
      { title: "自動畫完、量偏差", desc: `${w.after},自動回 home。結果在下方「執行紀錄」。`,
        art: paper() + top[k](true) + `<circle class="g-ok" cx="164" cy="68" r="14"/>` + line("M157 68L162 74L172 61", "g-tick") },
    ];
  }

  function show(k) {
    document.querySelectorAll("#guide-tabs button").forEach(b => b.classList.toggle("active", b.dataset.guide === k));
    document.querySelector("#guide-steps").innerHTML = steps(k).map((s, i) =>
      `<li class="g-step"><div class="g-art">${svg(s.art, s.title)}</div>` +
      `<div class="g-title"><span class="g-num">${i + 1}</span>${s.title}</div><div class="small muted">${s.desc}</div></li>`).join("");
  }

  function icons() {
    document.querySelectorAll("[data-icon]").forEach(el => {
      const k = el.dataset.icon;
      el.innerHTML = svg(paper() + top[k](true), INFO[k].name);
    });
  }

  function init() {
    document.querySelector("#guide-tabs").innerHTML = Object.entries(INFO).map(([k, w]) =>
      `<button type="button" data-guide="${k}">${w.name}</button>`).join("");
    document.querySelectorAll("#guide-tabs button").forEach(b => { b.onclick = () => show(b.dataset.guide); });
    icons();
    show("A");
    const box = document.querySelector("#guide");
    // 第一次打開網頁時展開;收起來之後就記住
    try { box.open = localStorage.getItem("guideClosed") !== "1"; } catch (e) { box.open = true; }
    box.addEventListener("toggle", () => {
      try { localStorage.setItem("guideClosed", box.open ? "0" : "1"); } catch (e) { /* 無痕模式等 */ }
    });
    document.querySelectorAll("[data-help]").forEach(b => {
      b.onclick = () => { box.open = true; show(b.dataset.help); box.scrollIntoView({ block: "start" }); };
    });
  }
  return { init };
})();
Guide.init();
