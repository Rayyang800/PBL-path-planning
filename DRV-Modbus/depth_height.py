# -*- coding: utf-8 -*-
"""
離線:從紀錄裡的深度快照量「工件比校正紙高多少」。不連線手臂/相機。

  python depth_height.py runs\\<id>                         # 用 home_before (沒有就用第一個有深度的快照)
  python depth_height.py runs\\<id> --snap home_after --at 128 105 --r 6   # 量紙面 (128,105) mm 附近半徑 6 mm
  python depth_height.py runs\\<id> --map                   # 另存高度圖 depth_height.jpg 到該紀錄的 data/

做法:4 個 AprilTag 定出紙面 -> 取紙上 (tag 內緣再縮 5 mm) 的深度點,用穩健擬合求紙面平面
     (先全部擬合,再只留離平面近的點重擬,排除工件) -> 每個像素到平面的距離 = 高度 (mm,高出紙面為正)。
"""
import argparse
import os
import sys

import cv2
import numpy as np

from paper_frame import PaperFrame, DEFAULT_LAYOUT
from run_log import load_run, load_snapshot

HERE = os.path.dirname(os.path.abspath(__file__))


def backproject(mm, K):
    v, u = np.nonzero(~np.isnan(mm))
    z = mm[v, u]
    return u, v, np.stack([(u - K["ppx"]) / K["fx"] * z, (v - K["ppy"]) / K["fy"] * z, z], 1)


def fit_plane(P):
    c = P.mean(0)
    Q = P - c
    _, Vt = np.linalg.eigh(Q.T @ Q)          # 3x3 共變異矩陣,最小特徵值的特徵向量 = 法向量 (不要對 N×3 做完整 SVD)
    n = Vt[:, 0]
    return c, n * (-1 if n[2] > 0 else 1)          # 法向量指向相機 (z 減少的方向) -> 高出紙面為正


def height_map(color, mm, K, layout=DEFAULT_LAYOUT, shrink=5.0):
    pf = PaperFrame(layout=layout, K=np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]]))
    if not pf.detect(color):
        raise RuntimeError("校正紙: " + pf.describe())
    xs = [c[0] for t in pf.layout["tags"] for c in t["corners_mm"]]
    ys = [c[1] for t in pf.layout["tags"] for c in t["corners_mm"]]
    tag = pf.layout["tag_mm"]
    box = (min(xs) + tag + shrink, min(ys) + tag + shrink, max(xs) - tag - shrink, max(ys) - tag - shrink)
    poly = pf.paper_to_pix([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
    mask = np.zeros(mm.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.int32(poly), 1)
    inside = mm.copy()
    inside[mask == 0] = np.nan
    u, v, P = backproject(inside, K)
    if len(P) < 500:
        raise RuntimeError(f"紙上有效深度點太少 ({len(P)})")
    c, n = fit_plane(P)
    for _ in range(4):                               # 只留離平面近的點 (排除工件) 重擬
        d = (P - c) @ n
        keep = np.abs(d - np.median(d)) < max(1.5, 2.5 * np.median(np.abs(d - np.median(d))))
        c, n = fit_plane(P[keep])
    ua, va, Pa = backproject(mm, K)
    H = np.full(mm.shape, np.nan, np.float32)
    H[va, ua] = (Pa - c) @ n
    d = H[v, u][keep]
    stats = {"paper_rms_mm": float(np.sqrt(np.mean(d ** 2))), "paper_points": int(keep.sum()),
             "valid_frac_on_paper": float(len(P) / max(1, mask.sum())), "cam_to_plane_mm": float(abs(c @ n)),
             "pnp_cam_h_mm": float(pf.cam_h)}
    return H, pf, stats


def board_thickness(H, contour, inner_px=8, ring_px=(12, 40)):
    """板厚估計 = 板子內部高度中位數 - 板子外圍一圈紙的高度中位數 (以附近的紙為基準,較不受紙面不平與反光影響)。
    contour: 板子在影像中的輪廓 (seam_boards.find_boards 的 contour)。回傳 dict(t_mm, spread_mm, ...) 或 None。
    實測 (11-10,5 mm 珍珠板):估 5.1 mm,左右上下半各 4.6~5.7 mm。"""
    inner = np.zeros(H.shape, np.uint8)
    cv2.drawContours(inner, [contour], -1, 1, -1)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))   # noqa: E731
    core = cv2.erode(inner, k(inner_px))
    ring = cv2.dilate(inner, k(ring_px[1])) & (1 - cv2.dilate(inner, k(ring_px[0])))
    hb, hp = H[(core > 0) & ~np.isnan(H)], H[(ring > 0) & ~np.isnan(H)]
    if hb.size < 500 or hp.size < 500:
        return None
    ref = float(np.median(hp))
    ys, xs = np.nonzero(core)
    xm, ym = np.median(xs), np.median(ys)
    parts = []
    for sel in (xs < xm, xs >= xm, ys < ym, ys >= ym):
        v = H[ys[sel], xs[sel]]
        v = v[~np.isnan(v)]
        if v.size > 100:
            parts.append(float(np.median(v)) - ref)
    t = float(np.median(hb)) - ref
    return {"t_mm": t, "spread_mm": float(np.ptp(parts)) if parts else None, "parts_mm": parts,
            "board_px": int(hb.size), "ring_px": int(hp.size)}


def depth_seam(H, pf, box_mm, board_h_mm=0.0, ppm=2.0, margin_mm=6.0, side_mm=(12.0, 30.0),
               min_dip_mm=0.6, max_tilt_deg=6.0, step_mm=5.0, parallel_to_box=True):
    """用高度圖找兩板之間的縫:板子範圍內一條「筆直的凹陷」(貼合縫在深度上會暈成 ~±8 mm 寬、~2 mm 深的 V 形)。
    box_mm: 板子 (兩板合起來) 的 4x2 紙面長方形 (seam_boards.find_boards 的 box)。
    回傳 dict(ok, seam_paper_mm, dip_mm, fit_rms_mm, ...)。中心用整條剖面的拋物線頂點,所以凹陷暈開也找得到中心。
    parallel_to_box=True (預設):縫的方向 = 板子外框的邊 (貼合縫一定和板邊平行),深度只決定左右位置。
      11-10 實測:方向也讓深度決定時會偏 2~6 度 (凹陷寬 ±8 mm、只深 1.6 mm,角度太敏感);位置則穩定 (與影像縫差 +0.3~0.8 mm)。"""
    box = np.asarray(box_mm, float)
    ctr = box.mean(0)
    e = box[1] - box[0]
    u = e / np.linalg.norm(e)
    v = np.array([-u[1], u[0]])
    bw, bh = float(np.ptp(box @ u)), float(np.ptp(box @ v))
    W, Hh = int(bw * ppm), int(bh * ppm)
    gx, gy = np.meshgrid(np.arange(W), np.arange(Hh))
    b = (np.stack([gx.ravel(), gy.ravel()], 1) - [W / 2, Hh / 2]) / ppm
    paper = ctr + b[:, :1] * u + b[:, 1:2] * v
    pix = pf.paper_to_pix(paper, h=board_h_mm).reshape(Hh, W, 2).astype(np.float32)
    R = cv2.remap(H.astype(np.float32), pix[..., 0], pix[..., 1], cv2.INTER_NEAREST, borderValue=float("nan"))
    m = int(margin_mm * ppm)
    core = R[m:Hh - m, m:W - m]
    if core.shape[0] < 20 or core.shape[1] < 20:
        return {"ok": False, "reason": "板子太小"}
    s0, s1 = int(side_mm[0] * ppm), int(side_mm[1] * ppm)
    best = None
    for axis in (0, 1):
        img = core if axis == 0 else core.T
        h, w = img.shape
        yy = np.arange(h) - h / 2
        for tilt in ([0.0] if parallel_to_box else np.arange(-max_tilt_deg, max_tilt_deg + 1e-6, 0.5)):
            idx = np.round(np.arange(w)[None, :] - (np.tan(np.radians(tilt)) * yy)[:, None]).astype(int)
            ok = (idx >= 0) & (idx < w) & ~np.isnan(img)
            cnt = np.bincount(idx[ok], minlength=w)
            prof = np.bincount(idx[ok], weights=img[ok], minlength=w) / np.maximum(cnt, 1)
            prof[cnt < 0.3 * h] = np.nan
            for x in range(max(s1, int(0.15 * w)), min(w - s1, int(0.85 * w))):
                side = np.r_[prof[x - s1:x - s0], prof[x + s0:x + s1]]
                side = side[~np.isnan(side)]
                if len(side) < (s1 - s0) or np.isnan(prof[x]):
                    continue
                dip = float(np.median(side) - prof[x])
                if best is None or dip > best["dip"]:
                    best = {"dip": dip, "axis": axis, "tilt": float(tilt), "x": x}
    if best is None or best["dip"] < min_dip_mm:
        return {"ok": False, "reason": f"高度圖上找不到明顯的凹陷 (最深 {best['dip'] if best else 0:.2f} mm,需要 ≥ {min_dip_mm:g})"}
    img = core if best["axis"] == 0 else core.T
    h, w = img.shape
    yy = np.arange(h) - h / 2
    half = int(7 * ppm)

    def global_fit(tilt):
        """整條縫一起:該傾角的投影剖面,在凹陷 ±7 mm 內擬合拋物線 -> (中心 x, 曲率 c2, 擬合殘差)。"""
        idx = np.round(np.arange(w)[None, :] - (np.tan(np.radians(tilt)) * yy)[:, None]).astype(int)
        ok = (idx >= 0) & (idx < w) & ~np.isnan(img)
        cnt = np.bincount(idx[ok], minlength=w)
        prof = np.bincount(idx[ok], weights=img[ok], minlength=w) / np.maximum(cnt, 1)
        xs = np.arange(max(0, best["x"] - half), min(w, best["x"] + half + 1))
        good = cnt[xs] >= 0.3 * h
        if good.sum() < 5:
            return None
        cc = np.polyfit(xs[good] - best["x"], prof[xs][good], 2)
        if cc[0] <= 0:
            return None
        res = prof[xs][good] - np.polyval(cc, xs[good] - best["x"])
        return best["x"] - cc[1] / (2 * cc[0]), cc[0], float(np.sqrt(np.mean(res ** 2)))
    # 傾角細搜 (0.1 度):凹陷最「尖」(曲率最大) 的角度 = 投影方向與縫平行
    tilts = [0.0] if parallel_to_box else np.arange(best["tilt"] - 1.0, best["tilt"] + 1.0001, 0.1)
    cand = [(t, global_fit(t)) for t in tilts]
    cand = [(t, g) for t, g in cand if g is not None]
    if cand:
        best["tilt"], (best["xc"], _, _) = max(cand, key=lambda tg: tg[1][1])
    else:
        best["xc"] = float(best["x"])
    seg = max(2, int(step_mm * ppm))
    pts, dips = [], []
    for y0 in range(0, h - seg + 1, seg):
        yc = y0 + seg / 2
        xl = best["xc"] + np.tan(np.radians(best["tilt"])) * (yc - h / 2)
        a, bb = int(xl - half), int(xl + half)
        if a < s1 or bb > w - s1:
            continue
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = np.nanmedian(img[y0:y0 + seg, a - s1:bb + s1], axis=0)
        xs = np.arange(a - s1, bb + s1)
        okp = ~np.isnan(p)
        if okp.sum() < len(p) * 0.6:
            continue
        inner = (xs >= a) & (xs <= bb) & okp
        if inner.sum() < 5:
            continue
        c2, c1, c0 = np.polyfit(xs[inner] - xl, p[inner], 2)       # 凹陷 -> 開口向上的拋物線
        if c2 <= 0:
            continue
        xv = xl - c1 / (2 * c2)
        if abs(xv - xl) > half:
            continue
        sidev = np.r_[p[: s1 - s0], p[-(s1 - s0):]]
        dips.append(float(np.nanmedian(sidev) - np.polyval([c2, c1, c0], xv - xl)))
        pts.append((xv, yc))
    pts = np.array(pts, float)
    if len(pts) < 4:
        return {"ok": False, "reason": "凹陷不連續 (有效段太少)"}
    # 縫的位置與方向用「整條一起」的結果 (x = xc + tan(tilt)*(y - h/2));分段結果只用來檢查是否筆直
    coef = np.array([np.tan(np.radians(best["tilt"])), best["xc"] - np.tan(np.radians(best["tilt"])) * h / 2])
    seg_res = pts[:, 0] - np.c_[pts[:, 1], np.ones(len(pts))] @ coef
    fit_rms = float(np.sqrt(np.median(seg_res ** 2)) / ppm)
    ys = np.arange(pts[0, 1], pts[-1, 1] + 1e-6, 2.0 * ppm)
    line = np.c_[np.polyval(coef, ys), ys]
    if best["axis"] == 1:
        line = line[:, ::-1]
    line = line + m
    bl = (line - [W / 2, Hh / 2]) / ppm
    seam = ctr + bl[:, :1] * u + bl[:, 1:2] * v
    return {"ok": True, "seam_paper_mm": seam, "dip_mm": float(np.median(dips)) if dips else best["dip"],
            "fit_rms_mm": fit_rms, "n_segments": int(len(pts)), "length_mm": float(np.linalg.norm(seam[-1] - seam[0]))}


def line_offset(seam_a, seam_b):
    """兩條縫 (紙面 mm 折線) 的平均垂直距離與夾角。"""
    a, b = np.asarray(seam_a, float), np.asarray(seam_b, float)
    ua = (a[-1] - a[0]) / np.linalg.norm(a[-1] - a[0])
    ub = (b[-1] - b[0]) / np.linalg.norm(b[-1] - b[0])
    n = np.array([-ua[1], ua[0]])
    off = float(np.median((b - a[0]) @ n))
    ang = float(np.degrees(np.arccos(np.clip(abs(ua @ ub), -1, 1))))
    return off, ang


def measure(H, pf, at_mm, r_mm):
    """紙面 at_mm 附近半徑 r_mm 的高度 (中位數、四分位距、有效比例)。"""
    cu = pf.paper_to_pix([at_mm])[0]
    rp = max(2, int(r_mm / pf.mm_per_pixel()))
    m = np.zeros(H.shape, np.uint8)
    cv2.circle(m, tuple(int(x) for x in cu), rp, 1, -1)
    vals = H[(m > 0)]
    good = vals[~np.isnan(vals)]
    if good.size == 0:
        return {"ok": False, "valid_frac": 0.0}
    q1, med, q3 = np.percentile(good, [25, 50, 75])
    return {"ok": True, "height_mm": float(med), "iqr_mm": float(q3 - q1), "valid_frac": float(good.size / vals.size),
            "pixel": cu.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--snap", help="快照名稱 (預設 home_before 或第一個有深度的)")
    ap.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"), help="要量的紙面座標 mm (預設紙中心)")
    ap.add_argument("--r", type=float, default=6.0, help="量測半徑 mm")
    ap.add_argument("--layout", help="版面檔 (預設用紀錄 inputs/ 裡的,沒有就用 print/lab_sheet_layout.json)")
    ap.add_argument("--map", action="store_true", help="另存高度圖")
    args = ap.parse_args()
    meta, _ = load_run(args.run)
    snaps = [s for s in meta.get("snapshots", []) if "depth" in s]
    if not snaps:
        sys.exit("這筆紀錄沒有深度快照 (v2 之前的紀錄只有彩色)")
    name = args.snap or next((s["name"] for s in snaps if s["name"] == "home_before"), snaps[0]["name"])
    color, mm, dmeta = load_snapshot(args.run, name)
    layout = args.layout or next((os.path.join(args.run, "inputs", k) for k in meta.get("inputs", {}) if k.endswith("layout.json")),
                                 os.path.join(HERE, "print", "lab_sheet_layout.json"))
    H, pf, st = height_map(color, mm, dmeta["intrinsics"], layout)
    print(f"快照 {name}:紙上有效深度 {st['valid_frac_on_paper'] * 100:.0f}%,紙面平面殘差 RMS {st['paper_rms_mm']:.2f} mm,"
          f"相機到紙 {st['cam_to_plane_mm']:.1f} mm (PnP {st['pnp_cam_h_mm']:.1f})")
    at = args.at or (pf.W / 2, pf.H / 2)
    m = measure(H, pf, at, args.r)
    if m["ok"]:
        print(f"紙面 ({at[0]:.0f},{at[1]:.0f}) 半徑 {args.r:g} mm:高度 {m['height_mm']:+.2f} mm (四分位距 {m['iqr_mm']:.2f},"
              f"有效 {m['valid_frac'] * 100:.0f}%)")
    else:
        print("該處沒有有效深度 (黑色或反光表面常見)")
    if args.map:
        vis = np.clip((np.nan_to_num(H, nan=-99) + 5) / 25 * 255, 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis[np.isnan(H)] = (40, 40, 40)
        out = os.path.join(args.run, "data", f"depth_height_{name}.jpg")
        cv2.imencode(".jpg", np.hstack([color, vis]))[1].tofile(out)
        print("高度圖 (-5 ~ +20 mm,灰 = 無深度):", out)


if __name__ == "__main__":
    main()
