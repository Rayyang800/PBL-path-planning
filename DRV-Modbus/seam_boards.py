# -*- coding: utf-8 -*-
"""
兩片平面黑板的縫合線擷取 (工件 B:縫合塗)。純影像運算,不動手臂。

流程:
  1. PaperFrame 偵測 A4 紙的 4 個 tag -> 紙面 mm 座標
  2. 只看紙面範圍內 (扣掉 tag 本體),用 Otsu 自動門檻分出「暗」的板子
  3. 取最大的兩塊,各自用 minAreaRect 擬合成長方形 (在紙面 mm 座標裡擬合,板厚造成的視差已修正)
  4. 找兩塊板「相對」的兩條邊 (幾乎平行、有重疊),縫合線 = 兩邊的中線
     縫寬 (gap) 太小 (影像上看不到白紙) 時會失敗並提示:請在兩板間留 2~3 mm 縫

用法:
  python seam_boards.py --image frame.png --board-h 5      # 分析一張圖
  python seam_boards.py --camera --board-h 5               # 抓一張 RealSense 畫面分析 (只讀相機)
輸出:預設存成一筆 runs/<時間>_seam/ 紀錄 (原始影像、疊圖、縫合線 CSV、版面檔複本、meta.json)。
     --no-save 則只寫 seam_result.json 與 seam_overlay.png 到 --out。
--board-h = 板子厚度 (mm),請用游標卡尺量;不知道時先填 0,但厚度越大誤差越大 (見 PaperFrame 視差修正)。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from paper_frame import PaperFrame, DEFAULT_LAYOUT
from run_log import RunLog, RUNS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def paper_mask(shape, pf, inset_mm=3.0, tag_pad_mm=3.0):
    """紙面範圍的遮罩 (影像座標),扣掉各 tag 本體 (外擴 tag_pad_mm)。"""
    W, H = pf.W, pf.H
    rect = np.float32([[inset_mm, inset_mm], [W - inset_mm, inset_mm], [W - inset_mm, H - inset_mm], [inset_mm, H - inset_mm]])
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.int32(np.round(pf.paper_to_pix(rect))), 255)
    t = pf.layout["tag_mm"]
    for tag in pf.layout["tags"]:
        c = np.array(tag["corners_mm"], dtype=np.float64)
        ctr = c.mean(axis=0)
        c = ctr + (c - ctr) * ((t / 2 + tag_pad_mm) / (t / 2))
        cv2.fillConvexPoly(m, np.int32(np.round(pf.paper_to_pix(c))), 0)
    return m


def find_boards(frame, pf, board_h_mm=0.0, min_area_mm2=1500.0):
    """回傳 (boards, info)。boards: 依面積由大到小,每塊含 box(4x2 mm)、area、fill、contour。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    m = paper_mask(gray.shape, pf)
    vals = gray[m > 0].reshape(-1, 1)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((gray < thr) & (m > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(cv2.morphologyEx(dark, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boards = []
    for c in cnts:
        pts = pf.pix_to_paper(c.reshape(-1, 2), h=board_h_mm)
        area = _poly_area(pts)
        if area < min_area_mm2:
            continue
        rect = cv2.minAreaRect(pts.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.float64)
        boards.append({"area": area, "box": box, "fill": area / max(rect[1][0] * rect[1][1], 1e-6),
                       "size_mm": sorted([float(rect[1][0]), float(rect[1][1])], reverse=True), "contour": c})
    boards.sort(key=lambda b: -b["area"])
    return boards, {"otsu_thr": float(thr), "dark_mask": dark}


def _edges(box):
    return [(box[i], box[(i + 1) % 4]) for i in range(4)]


def facing_edges(box_a, box_b, max_angle_deg=15.0, min_overlap_mm=10.0):
    """兩板相對的邊:幾乎平行、沿邊方向有重疊,取兩邊垂直距離 (縫寬) 最小者。"""
    best = None
    for a0, a1 in _edges(box_a):
        for b0, b1 in _edges(box_b):
            da, db = a1 - a0, b1 - b0
            ang = np.degrees(np.arctan2(abs(da[0] * db[1] - da[1] * db[0]), float(da @ db)))
            ang = min(ang, 180.0 - ang)
            if ang > max_angle_deg:
                continue
            u = da / np.linalg.norm(da)
            if float(u @ db) < 0:
                db, b0, b1 = -db, b1, b0                      # 讓 b 與 a 同向
            u = (da / np.linalg.norm(da) + db / np.linalg.norm(db))
            u /= np.linalg.norm(u)
            n = np.array([-u[1], u[0]])
            gap = abs(float(((b0 + b1) / 2 - (a0 + a1) / 2) @ n))
            ia = sorted([float(a0 @ u), float(a1 @ u)]); ib = sorted([float(b0 @ u), float(b1 @ u)])
            lo, hi = max(ia[0], ib[0]), min(ia[1], ib[1])
            if hi - lo < min_overlap_mm:
                continue
            if best is None or gap < best["gap_mm"]:
                best = {"gap_mm": gap, "a": (a0, a1), "b": (b0, b1), "u": u, "n": n, "lo": lo, "hi": hi, "angle_deg": float(ang)}
    return best


def seam_midline(fe, step_mm=2.0):
    """兩條相對邊的中線 (紙面 mm 折線,依 u 方向由小到大)。"""
    u = fe["u"]
    (a0, a1), (b0, b1) = fe["a"], fe["b"]
    s = np.arange(fe["lo"], fe["hi"] + 1e-6, step_mm)
    if s[-1] < fe["hi"] - 1e-6:
        s = np.append(s, fe["hi"])

    def on_line(p0, p1):
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        t = (s - float(p0 @ u)) / float(d @ u)
        return p0[None, :] + t[:, None] * d[None, :]
    return 0.5 * (on_line(a0, a1) + on_line(b0, b1))


def touching_seam(frame, pf, board, board_h_mm=0.0, step_mm=2.0, ppm=4.0, margin_mm=4.0,
                  min_contrast=3.0, min_score=4.0, max_tilt_deg=6.0):
    """兩板貼合 (中間看不到白紙) 時:在那一大塊暗區裡找一條「筆直、比兩側更暗或更亮」的細線當縫。
    做法:把板子區域校正成紙面俯視圖 (ppm 像素/mm) 並轉到板子的長方形軸向,
    對兩個軸向、±max_tilt_deg 的小角度做投影剖面,找最明顯的凹/凸線;再分段細修成折線。
    回傳 dict(ok, seam_paper_mm, contrast, score, ...) 或 dict(ok=False, reason)。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    box = board["box"].astype(np.float64)
    ctr = box.mean(0)
    e = box[1] - box[0]
    u = e / np.linalg.norm(e)                                         # 板子的一個邊方向
    v = np.array([-u[1], u[0]])
    bw, bh = float(np.ptp(box @ u)), float(np.ptp(box @ v))
    W, H = int((bw + 2) * ppm), int((bh + 2) * ppm)                   # 校正圖:板子中心在中央、軸向對齊影像軸

    def board_to_paper(xy):                                           # 校正圖像素 -> 紙面 mm
        b = (np.asarray(xy, float) - [W / 2, H / 2]) / ppm
        return ctr + b[:, :1] * u + b[:, 1:2] * v
    # 校正圖每個像素在原始影像的位置
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    paper = board_to_paper(np.stack([gx.ravel(), gy.ravel()], 1))
    pix = pf.paper_to_pix(paper, h=board_h_mm).reshape(H, W, 2).astype(np.float32)
    rect = cv2.remap(gray.astype(np.float32), pix[..., 0], pix[..., 1], cv2.INTER_LINEAR, borderValue=255)
    m = int(margin_mm * ppm)
    core = rect[m + int(ppm):H - m - int(ppm), m + int(ppm):W - m - int(ppm)]
    if core.shape[0] < 20 or core.shape[1] < 20:
        return {"ok": False, "reason": "板子太小"}
    best = None
    for axis in (0, 1):                                              # 0: 縫沿影像 y (找列剖面);1: 縫沿影像 x
        img = core if axis == 0 else core.T
        h, w = img.shape
        yy = np.arange(h) - h / 2
        for tilt in np.arange(-max_tilt_deg, max_tilt_deg + 1e-6, 0.25):
            shift = np.tan(np.radians(tilt)) * yy                     # 斜線:每列平移
            xs = np.arange(w)[None, :] - shift[:, None]
            idx = np.round(xs).astype(int)
            ok = (idx >= 0) & (idx < w)
            prof = np.bincount(idx[ok], weights=img[ok], minlength=w) / np.maximum(np.bincount(idx[ok], minlength=w), 1)
            lo, hi = int(0.2 * w), int(0.8 * w)
            win = int(5 * ppm)
            for sign in (1, -1):                                      # 1: 暗線, -1: 亮線
                for x in range(lo, hi):
                    side = np.r_[prof[max(0, x - 3 * win):max(0, x - win // 2)], prof[x + win // 2:x + 3 * win]]
                    if len(side) < win:
                        continue
                    c = sign * (np.median(side) - prof[x])
                    noise = np.std(np.diff(side)) / np.sqrt(2) + 1e-3
                    score = c / noise
                    if best is None or score > best["score"]:
                        best = {"score": float(score), "contrast": float(c), "axis": axis, "tilt": float(tilt), "x": x,
                                "sign": sign, "w": w, "h": h}
    if best is None or best["contrast"] < min_contrast or best["score"] < min_score:
        return {"ok": False, "reason": f"看不到貼合的縫 (最明顯的線對比 {best['contrast'] if best else 0:.1f} 灰階、"
                                       f"信噪比 {best['score'] if best else 0:.1f};需要 ≥ {min_contrast:g} / ≥ {min_score:g})",
                "touch_best": best}
    # 分段細修:每 step_mm 一段,在粗線 ±2 mm 內找極值 (拋物線內插)
    img = core if best["axis"] == 0 else core.T
    h, w = img.shape
    seg = max(2, int(step_mm * ppm))
    pts, seg_contrast = [], []
    win = int(5 * ppm)
    for y0 in range(0, h - seg + 1, seg):
        yc = y0 + seg / 2
        x_line = best["x"] + np.tan(np.radians(best["tilt"])) * (yc - h / 2)
        a, b = int(max(1, x_line - 2 * ppm)), int(min(w - 2, x_line + 2 * ppm))
        if b - a < 3:
            continue
        row = img[y0:y0 + seg].mean(0) * best["sign"]
        xl = int(round(x_line))
        side = np.r_[row[max(0, xl - 3 * win):max(0, xl - win // 2)], row[xl + win // 2:xl + 3 * win]]
        p = img[y0:y0 + seg, a:b + 1].mean(0) * best["sign"]
        k = int(np.argmin(p))
        seg_contrast.append(float(np.median(side) - p[k]) if len(side) else 0.0)
        if 0 < k < len(p) - 1:
            den = p[k - 1] - 2 * p[k] + p[k + 1]
            k = k + (0.5 * (p[k - 1] - p[k + 1]) / den if abs(den) > 1e-6 else 0.0)
        pts.append((a + k, yc))
    pts = np.array(pts, float)
    if len(pts) < 5:
        return {"ok": False, "reason": "貼合縫太短"}
    continuity = float(np.mean(np.array(seg_contrast) >= min_contrast * 0.5))
    if continuity < 0.7:                                             # 只有一小段有變化 (例如板子錯開的白紙缺口、刮痕) 不算縫
        return {"ok": False, "reason": f"候選線只有 {continuity * 100:.0f}% 的長度看得到 (需要 ≥ 70%),不像一條貫穿的縫",
                "touch_best": best}
    # 直線擬合 + 去除離群 (筆跡、刮痕) 後重擬
    for _ in range(2):
        A = np.c_[pts[:, 1], np.ones(len(pts))]
        coef, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
        res = pts[:, 0] - A @ coef
        keep = np.abs(res) < max(1.0, 3 * np.median(np.abs(res)))
        pts = pts[keep]
    fit_rms_mm = float(np.sqrt(np.mean((pts[:, 0] - np.c_[pts[:, 1], np.ones(len(pts))] @ coef) ** 2)) / ppm)
    ys = np.arange(pts[0, 1], pts[-1, 1] + 1e-6, step_mm * ppm)
    line = np.c_[np.polyval(coef, ys), ys]
    if best["axis"] == 1:
        line = line[:, ::-1]
    line += [m + int(ppm), m + int(ppm)]                               # core -> 校正圖座標
    seam_h = board_to_paper(line)                                     # 紙面 mm (板頂高度已在 remap 時修正)
    length = float(np.linalg.norm(seam_h[-1] - seam_h[0]))
    return {"ok": True, "seam_paper_mm": seam_h, "contrast": best["contrast"], "score": best["score"],
            "polarity": "dark" if best["sign"] == 1 else "bright", "fit_rms_mm": fit_rms_mm, "length_mm": length,
            "continuity": continuity}


def analyze(frame, pf, board_h_mm=0.0, step_mm=2.0, allow_touching=True):
    """主要入口。成功回傳 dict(ok=True, seam_paper_mm=Nx2 ...);失敗回傳 dict(ok=False, reason=...)。
    兩板之間看得到白紙縫 -> 用兩塊板的相對邊 (mode=gap);只找到一大塊 -> 找貼合的細線 (mode=touching)。"""
    if not pf.detect(frame):
        return {"ok": False, "reason": "紙上 4 個 tag 沒有全部偵測到:" + pf.describe()}
    boards, info = find_boards(frame, pf, board_h_mm)
    if len(boards) == 1 and allow_touching:
        t = touching_seam(frame, pf, boards[0], board_h_mm, step_mm)
        if t["ok"]:
            s = t["seam_paper_mm"]
            return {"ok": True, "mode": "touching", "paper": pf.describe(), "seam_paper_mm": s, "gap_mm": 0.0,
                    "length_mm": t["length_mm"], "angle_deg": float(np.degrees(np.arctan2(s[-1][1] - s[0][1], s[-1][0] - s[0][0]))),
                    "boards": boards, "board_h_mm": board_h_mm, "otsu_thr": info["otsu_thr"],
                    "touching": {k: t[k] for k in ("contrast", "score", "polarity", "fit_rms_mm", "continuity")}}
        return {"ok": False, "reason": "只找到 1 塊板子,也找不到貼合的縫:" + t["reason"] + "。可以在兩板間留 2~3 mm 縫",
                "paper": pf.describe(), "boards": boards}
    if len(boards) < 2:
        return {"ok": False, "reason": f"只找到 {len(boards)} 塊板子。兩板貼太緊(看不到白紙縫)時會被當成一塊,請留 2~3 mm 縫;"
                                         f"或板子被 tag 遮住/太小。", "paper": pf.describe(), "boards": boards}
    a, b = boards[0], boards[1]
    fe = facing_edges(a["box"], b["box"])
    if fe is None:
        return {"ok": False, "reason": "找不到兩板相對且平行的邊 (夾角 >15° 或沒有重疊 ≥10 mm)", "paper": pf.describe(), "boards": boards}
    seam = seam_midline(fe, step_mm)
    return {"ok": True, "paper": pf.describe(), "seam_paper_mm": seam, "gap_mm": fe["gap_mm"],
            "length_mm": float(np.linalg.norm(seam[-1] - seam[0])), "angle_deg": float(np.degrees(np.arctan2(seam[-1][1] - seam[0][1], seam[-1][0] - seam[0][0]))),
            "boards": boards, "board_h_mm": board_h_mm, "otsu_thr": info["otsu_thr"], "edges": fe}


def overlay(frame, pf, res):
    out = frame.copy()
    for b in res.get("boards", [])[:2]:
        cv2.polylines(out, [np.int32(np.round(pf.paper_to_pix(b["box"], h=res.get("board_h_mm", 0))))], True, (0, 200, 0), 2)
    if res.get("ok"):
        p = np.int32(np.round(pf.paper_to_pix(res["seam_paper_mm"], h=res["board_h_mm"])))
        cv2.polylines(out, [p], False, (0, 120, 255), 2)
        for q in p[::10]:
            cv2.circle(out, tuple(int(v) for v in q), 3, (0, 0, 255), -1)
        cv2.putText(out, f"seam {res['length_mm']:.1f}mm gap {res['gap_mm']:.1f}mm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 255), 2)
    return out


CAPTURE = {}             # grab_frame 最近一次的完整快照 (含深度) 與相機資訊


def grab_frame():
    from rs_capture import grab_once
    snap, info = grab_once(n_depth=15, warmup=40)
    CAPTURE.update(snap=snap, info=info)
    return snap["color"]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--camera", action="store_true")
    ap.add_argument("--board-h", type=float, default=0.0, help="板子厚度 mm (用卡尺量)")
    ap.add_argument("--step-mm", type=float, default=2.0)
    ap.add_argument("--label", default="", help="這次紀錄的標籤 (例如 boards_gap3)")
    ap.add_argument("--runs-dir", default=RUNS_DIR, help="紀錄資料夾 (預設 runs/)")
    ap.add_argument("--no-save", action="store_true", help="不建立 runs/ 紀錄,只把結果寫到 --out")
    ap.add_argument("--out", default=HERE, help="--no-save 時的輸出資料夾")
    args = ap.parse_args()

    frame = grab_frame() if args.camera else cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit("讀不到影像")
    pf = PaperFrame()
    res = analyze(frame, pf, args.board_h, args.step_mm)
    print(res.get("paper", pf.describe()))
    if not res["ok"]:
        print("失敗:", res["reason"])
    else:
        print(f"縫合線: 長 {res['length_mm']:.1f} mm,縫寬 {res['gap_mm']:.2f} mm,方向 {res['angle_deg']:.1f}°,{len(res['seam_paper_mm'])} 點")
        print("兩板尺寸 (mm):", [[round(v, 1) for v in b["size_mm"]] for b in res["boards"][:2]])
    ov = overlay(frame, pf, res)
    summary = {"frame": "paper_mm", "ok": bool(res["ok"]), "board_h_mm": args.board_h,
               "reason": res.get("reason"), "paper": res.get("paper", pf.describe())}
    if res["ok"]:
        summary.update({"gap_mm": res["gap_mm"], "length_mm": res["length_mm"], "angle_deg": res["angle_deg"],
                        "boards_size_mm": [b["size_mm"] for b in res["boards"][:2]],
                        "boards_box_paper_mm": [b["box"] for b in res["boards"][:2]], "otsu_thr": res["otsu_thr"]})

    if args.no_save:
        with open(os.path.join(args.out, "seam_result.json"), "w", encoding="utf-8") as f:
            json.dump(dict(summary, seam=np.round(res["seam_paper_mm"], 3).tolist() if res["ok"] else None), f, indent=1, default=float)
        cv2.imencode(".png", ov)[1].tofile(os.path.join(args.out, "seam_overlay.png"))
        print("疊圖:", os.path.join(args.out, "seam_overlay.png"))
        return

    hw = {"camera": {"source": "realsense" if args.camera else os.path.abspath(args.image), "resolution": [frame.shape[1], frame.shape[0]],
                     "K": pf.K.tolist()}}
    if pf.ok:
        hw["paper_frame"] = {"rms_mm": pf.rms_mm, "cam_h_mm": pf.cam_h, "tilt_deg": pf.tilt_deg, "mm_per_px": pf.mm_per_pixel(),
                             "tag_corners_px": {str(k): v.tolist() for k, v in pf.found.items()}}
    with RunLog("seam", label=args.label, runs_dir=args.runs_dir, params=vars(args), hardware=hw) as log:
        log.add_input(DEFAULT_LAYOUT)
        if args.camera:
            log.save_snapshot("frame_raw", CAPTURE["snap"], CAPTURE["info"])     # 彩色 + 深度
        else:
            log.save_image("frame_raw", frame)
        log.save_image("overlay.jpg", ov)          # 疊圖僅供目視,用 JPG 省空間;原始影像維持無損 PNG
        log.event("analyze", ok=bool(res["ok"]), reason=res.get("reason"))
        if res["ok"]:
            log.save_csv("seam_paper_mm", ["x_mm", "y_mm"], res["seam_paper_mm"].tolist())
        log.save_json("result", summary)
        log.finish("ok" if res["ok"] else "failed", {k: summary.get(k) for k in ("ok", "gap_mm", "length_mm", "angle_deg", "reason")})
        print("已存紀錄:", log.path)


if __name__ == "__main__":
    main()
