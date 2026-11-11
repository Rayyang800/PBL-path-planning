# -*- coding: utf-8 -*-
"""
多視角 RGB-D 立體重建 (任何放在校正紙上的物體:立方體、L 型工件…;不下筆、不畫東西)。
結果 (data/model.ply + result.json) 可在網頁控制台的「立體模型」視窗旋轉檢視。

流程
  0. 手臂在 home:拍彩色 + 深度 -> 用校正紙 AprilTag 求相機位姿 -> 點雲轉到「紙面座標」-> 找出比紙面高的物體 (位置、最高點)
  1. 規劃掃描點:相機在物體正上方 + 周圍一圈 (預設 8 個,半徑 --radius),相機離紙 --cam-h (預設 380 mm,近距離深度較準)
     姿態固定 (和 home 一樣朝下,只平移、不傾斜)
  2. 逐點移動 (MovL,速度 --speed) -> 停穩 -> 拍彩色 + 多幀深度 -> 記錄手臂位姿
  3. 每張深度用「該張看到的 tag」求相機位姿 (少於 3 個 tag 時改用手臂位移推算),轉到紙面座標;
     再用該張的紙面點擬合平面做修正 -> 全部合併成一個點雲 (model.ply) 與高度圖
  4. 對齊:各近拍視角的頂面中心對到正上方視角 (修正位姿誤差,只修水平)
  5. 量測 (任何形狀):最高點、體積、底面外框;有平的頂面時另量頂面高度、長寬、平整度;每個視角各自量一次
  6. 傾斜視角 (--tilts,預設 20,35 度):先在物體上方原地小角度試轉 3 次,用 tag 量出「相機在手臂末端的位置/方向」
     與手臂歐拉角慣例 (手眼);再規劃繞物體一圈、相機傾斜往物體中心看的視角 (越斜越低,看得到側面),逐點拍。
     相機離 TCP 約 150 mm,斜看時會轉手腕 (最多 --max-roll 90°) 讓相機在靠物體那側,TCP 才不用跑太遠;
     每圈的行走順序讓手腕轉角單調變化 (不會來回大轉,保護相機線)。試轉在較高處做 (--probe-lift),
     繞兩個不同軸,看不到 3 個 tag 就減半角度重試;轉軸太相近 / 解出的位置不合理就不做傾斜
  7. 回 home (物體較大時,周圍視角半徑自動加大)

安全
  - 筆尖任何時候都比「物體最高點 + --clearance」(預設 40 mm) 高;不夠高的掃描點會自動抬高
  - 相機離物體頂面至少 --min-range (預設 250 mm;D435 深度在 848x480 太近會量不到/不準)
  - 掃描點不會比 home 高;離 home 水平超過 --max-travel (預設 150 mm) 的點會略過
    (相機在筆旁邊約 140 mm,相機在物體上方時筆尖不在物體上方,但仍保持上面的淨空)
  - 點與點之間都是直線 (MovL),兩端都夠高,中間也一定夠高
  - 傾斜時 TCP 原地轉,筆尖 (在 TCP 下方) 只會變高;所以傾斜視角也用同一個條件:TCP z - z_touch >= 物體高 + 淨空
  - 試轉的手眼誤差太大 (旋轉 > 3° 或位置 > 4 mm) 就不做傾斜視角;手臂不接受的姿態 (超出可達範圍) 自動略過

  python scan3d.py                       # dry-run:home 拍照、找物體、印掃描計畫 (存 scan3d_plan.png),不動手臂
  python scan3d.py --go                  # 執行 (按 Enter 才開始動)
  python scan3d.py --go --radius 50 --n-ring 6 --cam-h 400
  python scan3d.py --reprocess runs\\<id>  # 離線重算 (不連手臂/相機),結果存到新紀錄

按鍵 (本視窗):Ctrl+C = 停止 -> 回 home (已拍的視角照樣合併);移動中 空白/Esc/p = 停止並選單 (r 重拍這點 / s 跳過 / h 回 home 合併 / q 結束)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import depth_height
from paper_frame import FAMILIES
from run_log import RunLog, CalibrationStore, load_snapshot

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = os.path.join(HERE, "print", "lab_sheet_layout.json")
CAM_RANGE_MM = (150.0, 1000.0)


# ======================= 幾何 =======================
def K_of(intr):
    return np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])


def tag_pose(color, layout, K):
    """用畫面中看得到的 tag (每個 4 角) 做 PnP。回傳 dict(R, t, C, n_tags, ids, rms_px) 或 None (少於 2 個 tag)。
    紙面座標:x 右、y 下、z 指向桌面 (相機在紙上方 z<0);高度 h = -z。"""
    d = cv2.aruco.Dictionary_get(FAMILIES[layout["family"]])
    p = cv2.aruco.DetectorParameters_create()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=p)
    cands = {}
    for c, i in zip(corners, ids if ids is not None else []):
        cands.setdefault(int(i[0]), []).append(c[0])
    obj, img, used = [], [], []
    for t in layout["tags"]:
        cs = cands.get(t["id"], [])
        if len(cs) != 1:                    # 看不到,或畫面中有重複 ID (別張紙):不用
            continue
        obj += [[x, y, 0.0] for x, y in t["corners_mm"]]
        img += cs[0].tolist()
        used.append(t["id"])
    if len(used) < 2:
        return None
    obj, img = np.float64(obj), np.float64(img)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1))))
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    return {"R": R, "t": t, "C": (-R.T @ t), "n_tags": len(used), "ids": used, "rms_px": rms}


def pose_from_center(R, C):
    return {"R": R, "t": -R @ C, "C": np.asarray(C, float)}


def view_points(color, mm, K, pose, stride=2, scale=1.0):
    """深度 (mm, NaN = 無效) -> 紙面座標點 (N,2)、高度 (N,)、顏色 (N,3 RGB)、相機距離 (N,)。
    scale:深度比例修正 (深度 x scale 後才轉座標;見 depth_scale_from_paper)。"""
    sub = np.full_like(mm, np.nan)
    sub[::stride, ::stride] = mm[::stride, ::stride]
    sub[(sub < CAM_RANGE_MM[0]) | (sub > CAM_RANGE_MM[1])] = np.nan
    u, v, P = depth_height.backproject(sub, {"fx": K[0, 0], "fy": K[1, 1], "ppx": K[0, 2], "ppy": K[1, 2]})
    P = P * scale
    X = (pose["R"].T @ (P.T - pose["t"].reshape(3, 1))).T
    rgb = color[v, u][:, ::-1]
    return X[:, :2], -X[:, 2], rgb, P[:, 2]


def depth_scale_from_paper(mm, K, pose, paper_wh, exclude_box=None, edge=8.0):
    """深度和 tag (PnP) 的距離比例常差 2~3% (11-11 實測:相機離紙 507 mm 時深度多 13 mm)。
    只修高度偏移的話,離光軸 100 mm 的點水平也會差 ~2.6 mm,所以先求比例 s:讓紙面上的深度點 x s 後剛好落在紙面 (h = 0)。"""
    xy, h, _, _ = view_points(np.zeros((mm.shape[0], mm.shape[1], 3), np.uint8), mm, K, pose)
    sub = np.full_like(mm, np.nan)
    sub[::2, ::2] = mm[::2, ::2]
    sub[(sub < CAM_RANGE_MM[0]) | (sub > CAM_RANGE_MM[1])] = np.nan
    _, _, P = depth_height.backproject(sub, {"fx": K[0, 0], "fy": K[1, 1], "ppx": K[0, 2], "ppy": K[1, 2]})
    a = (pose["R"].T @ P.T)[2]                                     # 紙面 z 分量 (乘上 s 後減 b = 紙面 z)
    b = float((pose["R"].T @ pose["t"])[2])
    W, H = paper_wh
    m = (xy[:, 0] > edge) & (xy[:, 0] < W - edge) & (xy[:, 1] > edge) & (xy[:, 1] < H - edge)
    if exclude_box is not None:
        x0, y0, x1, y1 = exclude_box
        m &= ~((xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1))
    m &= np.abs(h) < 40
    if m.sum() < 500:
        return 1.0
    r = b / a[m]
    s_ = float(np.median(r))
    for _ in range(3):                                            # 排除物體 / 雜點:只留 h 接近 0 的點
        hz = -(s_ * a[m] - b)
        keep = np.abs(hz) < max(2.0, 3 * 1.4826 * np.median(np.abs(hz)))
        if keep.sum() < 300:
            break
        s_ = float(np.median(r[keep]))
    return s_ if 0.9 < s_ < 1.1 else 1.0


def fit_paper_plane(xy, h, paper_wh, exclude_box=None, edge=8.0):
    """在紙面範圍內 (排除物體附近) 用穩健最小平方擬合 h = a x + b y + c。回傳 (a, b, c, rms, n)。"""
    W, H = paper_wh
    m = (xy[:, 0] > edge) & (xy[:, 0] < W - edge) & (xy[:, 1] > edge) & (xy[:, 1] < H - edge) & (np.abs(h) < 15)
    if exclude_box is not None:
        x0, y0, x1, y1 = exclude_box
        m &= ~((xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1))
    X, Z = xy[m], h[m]
    if len(Z) < 500:
        return None
    keep = np.ones(len(Z), bool)
    for _ in range(4):
        A = np.c_[X[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, Z[keep], rcond=None)
        r = np.c_[X, np.ones(len(X))] @ coef - Z
        mad = np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < max(1.5, 3.0 * 1.4826 * mad)
    rms = float(np.sqrt(np.mean(r[keep] ** 2)))
    return float(coef[0]), float(coef[1]), float(coef[2]), rms, int(keep.sum())


def find_object(xy, h, paper_wh, cell=1.0, min_h=3.0, min_area_mm2=150.0, tag_mm=29.0, margins=(14.0, 13.0)):
    """紙面上比紙高 min_h 的最大一塊 -> dict(center, bbox, h_max, area_mm2, mask, ...) 或 None。"""
    W, H = paper_wh
    gw, gh = int(W / cell), int(H / cell)
    ix, iy = (xy[:, 0] / cell).astype(int), (xy[:, 1] / cell).astype(int)
    ok = (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh)
    grid = np.zeros((gh, gw), np.float32)
    cnt = np.zeros((gh, gw), np.float32)
    np.add.at(grid, (iy[ok], ix[ok]), h[ok])
    np.add.at(cnt, (iy[ok], ix[ok]), 1)
    hm = np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan)
    m = np.nan_to_num(hm, nan=0) > min_h
    # tag 區域 (四角) 不算
    e = int((max(margins) + tag_mm + 3) / cell)
    for ys in (slice(0, e), slice(gh - e, gh)):
        for xs in (slice(0, e), slice(gw - e, gw)):
            m[ys, xs] = False
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = float(st[k, cv2.CC_STAT_AREA] * cell * cell)
    if area < min_area_mm2:
        return None
    x, y, w, hh = st[k, :4]
    vals = hm[lab == k]
    vals = vals[~np.isnan(vals)]
    return {"center": [float(cen[k][0] * cell), float(cen[k][1] * cell)],
            "bbox": [float(x * cell), float(y * cell), float((x + w) * cell), float((y + hh) * cell)],
            "h_max": float(np.percentile(vals, 99)), "h_median": float(np.median(vals)), "area_mm2": area}


def measure_object(xy, h, cell=0.5, min_h=3.0):
    """點雲 (紙面座標) -> 最高點、頂面 (最高的平台) 高度與長寬、底面外框、體積。任何形狀都可以用;
    立方體 / 方塊看 top_rect,其他形狀看 h_max、footprint、volume。回傳 dict 或 None。"""
    obj = h > min_h
    if obj.sum() < 300:
        return None
    # 只留最大的一塊 (ROI 裡可能有別的東西或雜點)
    o = xy[obj].min(0) - 3
    ij = ((xy - o) / 1.0).astype(int)
    gw, gh = ij[obj].max(0) + 4
    g = np.zeros((gh, gw), np.uint8)
    g[ij[obj, 1], ij[obj, 0]] = 1
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    inb = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < gw) & (ij[:, 1] < gh)
    comp = np.zeros(len(h), bool)
    comp[inb] = lab[ij[inb, 1], ij[inb, 0]] == k
    obj &= comp
    if obj.sum() < 300:
        return None
    hs = h[obj]
    p99 = float(np.percentile(hs, 99.5))
    lo = max(min_h, 0.5 * p99)
    bins = np.arange(lo, p99 + 2.0, 0.5)
    if len(bins) < 3:
        return None
    hist, edges = np.histogram(hs, bins)
    hist = np.convolve(hist, [1, 2, 1], "same")
    peak = float(edges[int(np.argmax(hist))] + 0.25)
    band = max(2.0, 0.04 * peak)
    near = obj & (np.abs(h - peak) < band)
    h_top = float(np.median(h[near]))
    top = obj & (np.abs(h - h_top) < band)

    def rect_of(mask_pts, close=5):
        P = xy[mask_pts]
        if len(P) < 100:
            return None
        o = P.min(0) - 5
        ij = ((P - o) / cell).astype(int)
        gw, gh = ij.max(0) + 6
        g = np.zeros((gh, gw), np.uint8)
        g[ij[:, 1], ij[:, 0]] = 1
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(g)
        if n < 2:
            return None
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cx, cy), (w, l), ang = cv2.minAreaRect(max(cs, key=cv2.contourArea))
        box = cv2.boxPoints(((cx, cy), (w, l), ang)) * cell + o
        if w < l:
            w, l, ang = l, w, ang + 90.0
        return {"center": (np.array([cx, cy]) * cell + o).tolist(), "long_mm": float(w * cell), "short_mm": float(l * cell),
                "angle_deg": float(((ang + 90) % 180) - 90), "box": box.tolist(),
                "area_mm2": float((lab == k).sum() * cell * cell)}

    rt = rect_of(top)
    rf = rect_of(obj)
    ht = h[top]
    A = np.c_[xy[top], np.ones(top.sum())]
    coef, *_ = np.linalg.lstsq(A, ht, rcond=None)
    tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    flat = float(np.std(ht - A @ coef))
    side = int((obj & (h < h_top - band - 2)).sum())
    # 體積:1 mm 格子,每格取平均高度 (看不到的側面下方當作實心)
    P = xy[obj]
    o2 = P.min(0)
    ij2 = ((P - o2) / 1.0).astype(int)
    gw2, gh2 = ij2.max(0) + 1
    hs2 = np.zeros((gh2, gw2))
    ns2 = np.zeros((gh2, gw2))
    np.add.at(hs2, (ij2[:, 1], ij2[:, 0]), h[obj])
    np.add.at(ns2, (ij2[:, 1], ij2[:, 0]), 1)
    hm2 = np.where(ns2 > 0, hs2 / np.maximum(ns2, 1), 0)
    filled = cv2.morphologyEx((ns2 > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hm2 = np.where((filled > 0) & (ns2 == 0), cv2.blur(hm2, (3, 3)), hm2)
    vol = float(hm2.sum() / 1000.0)
    return {"h_max_mm": float(np.percentile(hs, 99.5)), "volume_cm3": vol, "h_top_mm": h_top, "top_flatness_mm": flat, "top_tilt_deg": tilt, "top_points": int(top.sum()),
            "side_points": side, "top_rect": rt, "footprint_rect": rf}


# ======================= 合併 / 輸出 =======================
def icp_rigid(src, dst_tree, dst, iters=40, dists=(20.0, 12.0, 8.0, 5.0, 3.0), max_shift=30.0, max_rot_deg=8.0):
    """點對點 ICP (剛體 3D,修剪對應)。src / dst:(N,3) [x, y, h]。回傳 (R, t, rms, inlier_frac) 或 None。
    用在位姿不準的視角 (手眼 / 手臂推算):把它對到用 tag 定位的正上方視角合成的點雲上;紙面約束高度與傾斜,物體約束水平。"""
    Rt, tt = np.eye(3), np.zeros(3)
    cur = src.copy()
    per = max(1, iters // len(dists))
    rms, frac = None, 0.0
    for dmax in dists:
        for _ in range(per):
            d, j = dst_tree.query(cur, distance_upper_bound=dmax)
            ok = np.isfinite(d)
            if ok.sum() < 200:
                return None
            P, Q = cur[ok], dst[j[ok]]
            pc, qc = P.mean(0), Q.mean(0)
            U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
            Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            Ri = Vt.T @ Dm @ U.T
            ti = qc - Ri @ pc
            cur = cur @ Ri.T + ti
            Rt, tt = Ri @ Rt, Ri @ tt + ti
            rms = float(np.sqrt(np.mean(d[ok] ** 2)))
            frac = float(ok.mean())
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1))))
    if np.linalg.norm(tt + (Rt - np.eye(3)) @ src.mean(0)) > max_shift or ang > max_rot_deg:
        return None
    return Rt, tt, rms, frac


def anchor_xy(xy, h):
    """對齊用的錨點:頂面 (最高的平台) 中心;沒有明顯平台時用上半部 (高於最高點 60%) 的重心。
    頂面朝向相機,各視角都看得完整,所以中心穩定;側面每個視角看到的不一樣,不能拿來對齊。
    11-11 實測 (立方體 9 視角):頂面中心對齊後合併長寬 33.5x31.0 -> 32.6x31.4 (單一視角約 30x28);
    高度格相位相關反而變大 (35x35),已不用。"""
    m = measure_object(xy, h)
    if m and m["top_rect"] and m["top_points"] > 200:
        return np.array(m["top_rect"]["center"]), "top"
    if m:
        up = h > 0.6 * m["h_max_mm"]
        if up.sum() > 100:
            return xy[up].mean(0), "upper"
    return None, None


def process_views(views, layout, obj_prior=None, roi_margin=80.0, log=None, register=True):
    """views: [{name, color, mm, K, pose, pose_source}];回傳 (all_xy, all_h, all_rgb, per_view_rows, obj)。"""
    W, H = layout["paper_mm"]
    rows, kept = [], []
    obj = obj_prior
    # 合併只用近距離的視角:home (~520 mm) 的深度會低估高度,和近拍混在一起會把頂面拉低;home 仍單獨量一次當比較
    only_home = all(v["name"] == "home" for v in views)
    for v in views:
        ex = None
        if obj is not None:
            b = obj["bbox"]
            ex = (b[0] - 15, b[1] - 15, b[2] + 15, b[3] + 15)
        sc = depth_scale_from_paper(v["mm"], v["K"], v["pose"], (W, H), ex)
        xy, h, rgb, z = view_points(v["color"], v["mm"], v["K"], v["pose"], scale=sc)
        pl = fit_paper_plane(xy, h, (W, H), ex)
        r = {"view": v["name"], "pose_source": v["pose_source"], "cam_h_mm": float(-v["pose"]["C"][2]), "depth_scale": sc,
             "tilt": v.get("tilt"),
             "n_points": int(len(h))}
        if pl is None:
            r["error"] = "紙面點太少,這個視角不用"
            rows.append(r)
            continue
        a, b_, c, rms, n = pl
        h = h - (a * xy[:, 0] + b_ * xy[:, 1] + c)
        r.update(plane_offset_mm=c, plane_tilt_deg=float(np.degrees(np.arctan(np.hypot(a, b_)))), paper_rms_mm=rms)
        if obj is None:
            obj = find_object(xy, h, (W, H))
            if obj is None:
                r["error"] = "找不到比紙面高的物體"
                rows.append(r)
                continue
        bx = obj["bbox"]
        roi = ((xy[:, 0] > bx[0] - roi_margin) & (xy[:, 0] < bx[2] + roi_margin) &
               (xy[:, 1] > bx[1] - roi_margin) & (xy[:, 1] < bx[3] + roi_margin) & (h > -10) &
               (h < obj["h_max"] * 1.5 + 30))
        r["fused"] = bool(only_home or v["name"] != "home")
        rows.append(r)
        kept.append((r, xy[roi], h[roi], rgb[roi]))

    # ---- 對齊 1:位姿不是 tag 定的視角 (手眼 / 手臂推算,傾斜視角多半看不到 3 個 tag) -> ICP 對到 tag 視角的點雲 ----
    fused = [k for k in kept if k[0]["fused"]]
    ref_set = [k for k in fused if k[0]["pose_source"] == "tags"]
    if register and ref_set and any(k[0]["pose_source"] != "tags" for k in fused):
        from scipy.spatial import cKDTree
        # 用 ROI 內全部點 (紙面 + 物體)。只用物體點時,長條形物體 (屋脊) 會沿長度方向滑動 (11-11 試過:滑 26 mm),
        # 所以位姿先用 refine_view_poses 的手眼模型 (誤差 ~5 mm),ICP 只做 10 mm 以內的小修正
        dst = np.concatenate([np.c_[k[1], k[2]] for k in ref_set])
        dst = dst[np.random.default_rng(0).choice(len(dst), min(len(dst), 120000), replace=False)]
        tree = cKDTree(dst)
        for k in fused:
            if k[0]["pose_source"] == "tags":
                continue
            src = np.c_[k[1], k[2]]
            sub = src[np.random.default_rng(1).choice(len(src), min(len(src), 15000), replace=False)]
            res = icp_rigid(sub, tree, dst, dists=(12.0, 8.0, 5.0, 3.0), max_shift=10.0, max_rot_deg=4.0)
            if res is None:
                k[0]["icp"] = None
                continue
            Ri, ti, rms, frac = res
            P = src @ Ri.T + ti
            k[1][:] = P[:, :2]
            k[2][:] = P[:, 2]
            k[0]["icp"] = {"shift_mm": np.round(ti + (Ri - np.eye(3)) @ sub.mean(0), 2).tolist(),
                           "rot_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(Ri) - 1) / 2, -1, 1)))), 2),
                           "rms_mm": round(rms, 2), "inliers": round(frac, 2)}
            k[0]["registered"] = "icp"

    # ---- 對齊 2:以「正上方」視角 (c) 為參考,其他 tag 視角把頂面中心對上去 (只修水平偏移) ----
    if register and len(fused) > 1:
        ref_k = next((k for k in fused if k[0]["view"] == "c"), fused[0])
        ref, how = anchor_xy(ref_k[1], ref_k[2])
        if ref is not None:
            ref_k[0]["reg_shift_mm"] = [0.0, 0.0]
            for k in fused:
                if k is ref_k or k[0].get("registered") == "icp":
                    continue
                a, how2 = anchor_xy(k[1], k[2])
                lim = 12.0 if k[0]["pose_source"] != "tags" else 4.0     # tag 位姿誤差小,只允許小修正
                if a is None or how2 != how or np.hypot(*(a - ref)) > lim:
                    k[0]["reg_shift_mm"] = None
                    continue
                d = a - ref
                k[1][:] = k[1] - d
                k[0]["reg_shift_mm"] = [round(float(d[0]), 2), round(float(d[1]), 2)]

    XY, HH, RGB = [], [], []
    for r, xy, h, rgb in kept:
        if r.get("fused") and "icp" in r and r["icp"] is None and r.get("tilt"):
            r["fused"] = False                     # 傾斜視角位姿不準又對不上:不要把歪的點混進模型
            r["error"] = "ICP 對不上,未併入"
        m = measure_object(xy, h)
        if m:
            r.update(h_top_mm=m["h_top_mm"], h_max_mm=m["h_max_mm"], top_flatness_mm=m["top_flatness_mm"],
                     side_points=m["side_points"], cam_to_top_mm=float(r["cam_h_mm"] - m["h_top_mm"]))
            if m["top_rect"]:
                r.update(top_long_mm=m["top_rect"]["long_mm"], top_short_mm=m["top_rect"]["short_mm"],
                         top_angle_deg=m["top_rect"]["angle_deg"])
        if r["fused"]:
            XY.append(xy)
            HH.append(h)
            RGB.append(rgb)
        if log is not None:
            log.event("view_processed", **{k: vv for k, vv in r.items()})
    if not XY:
        return None, None, None, rows, obj
    return np.concatenate(XY), np.concatenate(HH), np.concatenate(RGB), rows, obj


def voxel_down(xy, h, rgb, vox=1.0):
    P = np.c_[xy, h]
    key = np.floor(P / vox).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Ps = np.zeros((len(cnt), 3))
    Cs = np.zeros((len(cnt), 3))
    np.add.at(Ps, inv, P)
    np.add.at(Cs, inv, rgb.astype(float))
    return Ps / cnt[:, None], (Cs / cnt[:, None]).astype(np.uint8), cnt


def write_ply(path, P, C):
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\ncomment paper frame mm: x right, y down, z = height above paper\n"
                f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(P, C):
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {r} {g} {b}\n")


def height_image(xy, h, m, cell=0.5, scale=3):
    o = xy.min(0) - 2
    ij = ((xy - o) / cell).astype(int)
    gw, gh = ij.max(0) + 3
    s = np.zeros((gh, gw))
    n = np.zeros((gh, gw))
    np.add.at(s, (ij[:, 1], ij[:, 0]), h)
    np.add.at(n, (ij[:, 1], ij[:, 0]), 1)
    hm = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    top = (m or {}).get("h_top_mm") or np.nanmax(hm)
    img = np.clip(np.nan_to_num(hm, nan=0) / max(top + 3, 1) * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    img[n == 0] = (40, 40, 40)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for key, col in (("footprint_rect", (0, 140, 255)), ("top_rect", (0, 255, 0))):
        r = (m or {}).get(key)
        if r:
            pts = ((np.array(r["box"]) - o) / cell * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, col, 2)
    if m and m.get("top_rect"):
        t = m["top_rect"]
        txt = f"top {t['long_mm']:.1f} x {t['short_mm']:.1f} mm  H {m['h_top_mm']:.1f} mm"
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def summarize(m, rows):
    lines = ["", "================ 重建結果 (紙面座標,mm;任何形狀) ================"]
    if not m:
        return "\n".join(lines + ["量不到物體 (點太少)"])
    lines.append(f"最高點 (離紙面)   : {m['h_max_mm']:.2f} mm   體積約 {m['volume_cm3']:.1f} cm³ (底下看不到的部分當實心)")
    lines.append(f"頂面高度 (離紙面) : {m['h_top_mm']:.2f} mm   平整度 {m['top_flatness_mm']:.2f} mm、傾斜 {m['top_tilt_deg']:.1f}°")
    if m["top_rect"]:
        t = m["top_rect"]
        lines.append(f"頂面長 x 寬       : {t['long_mm']:.1f} x {t['short_mm']:.1f} mm  (轉角 {t['angle_deg']:+.1f}°,"
                     f"中心 ({t['center'][0]:.1f}, {t['center'][1]:.1f}))")
    if m["footprint_rect"]:
        f = m["footprint_rect"]
        lines.append(f"底面外框 (含側面) : {f['long_mm']:.1f} x {f['short_mm']:.1f} mm")
    lines.append(f"側面點數          : {m['side_points']}  (越多代表側面拍得越完整)")
    lines.append("")
    lines.append(" 視角    位姿來源  相機離紙  離頂面  頂面高   頂面長x寬        紙面殘差  深度比例  對齊位移 (mm)")
    for r in rows:
        if "error" in r:
            lines.append(f" {r['view']:7s} {r['pose_source']:7s} {r.get('cam_h_mm', float('nan')):7.0f}   -- {r['error']}")
            continue
        sh = r.get("reg_shift_mm")
        lines.append(f" {r['view'] + ('' if r.get('fused') else '*'):7s} {r['pose_source']:7s} {r['cam_h_mm']:7.0f} {r.get('cam_to_top_mm', float('nan')):7.0f} "
                     f"{r.get('h_top_mm', float('nan')):7.2f}   {r.get('top_long_mm', float('nan')):5.1f} x "
                     f"{r.get('top_short_mm', float('nan')):5.1f}     {r.get('paper_rms_mm', float('nan')):.2f}     {r.get('depth_scale', 1):.4f}"
                     + (f"   對齊 {sh[0]:+.1f},{sh[1]:+.1f}" if sh else ("   對齊失敗(未修正)" if "reg_shift_mm" in r else ""))
                     + (f"   ICP 移 {np.round(r['icp']['shift_mm'], 1).tolist()} 轉 {r['icp']['rot_deg']}° 殘差 {r['icp']['rms_mm']}"
                        if r.get("icp") else ""))
    lines.append(" (* = 距離較遠,只單獨量、不併入模型)")
    hv = [r["h_top_mm"] for r in rows if "h_top_mm" in r and r.get("fused")]
    if len(hv) > 1:
        lines.append(f"近拍各視角頂面高: 平均 {np.mean(hv):.2f}、標準差 {np.std(hv):.2f}、範圍 {np.ptp(hv):.2f} mm")
    lines.append("之後用尺量實際長寬高,和這裡比較 (紀錄的 result.json 有全部數字)")
    return "\n".join(lines)


def finish_outputs(log, views, layout, home_obj):
    xy, h, rgb, rows, obj = process_views(views, layout, home_obj, log=log)
    log.save_json("views_measure", rows)
    if xy is None:
        print(summarize(None, rows))
        return None
    m = measure_object(xy, h)
    P, C, cnt = voxel_down(xy, h, rgb, 1.0)
    os.makedirs(os.path.join(log.path, "data"), exist_ok=True)
    write_ply(os.path.join(log.path, "data", "model.ply"), P, C)
    np.savez_compressed(os.path.join(log.path, "data", "cloud_raw.npz"), xy=xy.astype(np.float32),
                        h=h.astype(np.float32), rgb=rgb)
    img = height_image(xy, h, m)
    log.save_image("heightmap.png", img)
    cv2.imencode(".png", img)[1].tofile(os.path.join(HERE, "scan3d_result.png"))
    result = {"object": m, "cube": m, "object_home": home_obj, "fused_points": int(len(h)), "voxels_1mm": int(len(P)),
              "per_view": rows}
    log.save_json("result", result)
    print(summarize(m, rows))
    return m


# ======================= 實機 =======================
def plan_views(obj, home_pose, tcp_home, A, tcal, z_touch, args, paper_wh, targets=None):
    """回傳 (views[{name, cam_xy, tcp}], info)。相機位置 -> 筆尖紙面位置 -> 手臂 TCP (用筆尖校正 A, t)。"""
    Ainv = np.linalg.inv(A)
    C_home = home_pose["C"]
    H_home = float(-C_home[2])
    pen_home = Ainv @ (np.array(tcp_home[:2]) - tcal)            # home 時筆尖在紙面的位置
    offset = C_home[:2] - pen_home                              # 相機中心相對筆尖 (紙面,姿態固定所以不變)
    cam_above_pen = H_home - (tcp_home[2] - z_touch)
    h_obj = obj["h_max"]
    need = max(h_obj + args.min_range, h_obj + args.clearance + cam_above_pen)
    Hs = max(args.cam_h, need)
    notes = []
    if Hs > args.cam_h + 0.1:
        notes.append(f"相機高度由 {args.cam_h:g} 抬到 {Hs:.0f} mm (物體高 {h_obj:.0f} mm,要保持筆尖淨空 / 深度最短距離)")
    if Hs > H_home:
        Hs = H_home
        notes.append("需要的高度比 home 還高,改用 home 高度掃描")
    z_scan = z_touch + (Hs - cam_above_pen)
    pen_tip_h = Hs - cam_above_pen
    ctr = np.array(obj["center"])
    bx = obj["bbox"]
    half_diag = 0.5 * float(np.hypot(bx[2] - bx[0], bx[3] - bx[1]))
    radius = max(args.radius, half_diag + 20.0)          # 物體越大,周圍視角離越遠 (相機要能看到側面)
    if radius > args.radius + 0.1:
        notes.append(f"物體較大 (對角線一半 {half_diag:.0f} mm),周圍視角半徑由 {args.radius:g} 加大到 {radius:.0f} mm")
    if targets is not None:                              # 呼叫端指定相機位置 [(name, 紙面 xy)] (例如 ridge_trace 的快速掃描)
        targets = [(n_, np.asarray(q_, float)) for n_, q_ in targets]
    else:
        targets = [("c", ctr)]
    for k in range(args.n_ring if len(targets) == 1 and targets[0][0] == "c" else 0):
        a = 2 * np.pi * k / max(1, args.n_ring)
        targets.append((f"r{k}", ctr + radius * np.array([np.cos(a), np.sin(a)])))
    W, H = paper_wh
    out = []
    for name, q in targets:
        pen = q - offset
        xy = A @ pen + tcal
        d_home = float(np.hypot(xy[0] - tcp_home[0], xy[1] - tcp_home[1]))
        if d_home > args.max_travel:
            notes.append(f"{name}: 離 home 水平 {d_home:.0f} mm 超過 --max-travel {args.max_travel:g},略過")
            continue
        out.append({"name": name, "cam_xy": q.tolist(), "pen_paper": pen.tolist(), "tcp": [float(xy[0]), float(xy[1]), float(z_scan)]})
    info = {"cam_h_scan": Hs, "z_scan": z_scan, "pen_tip_above_paper": pen_tip_h, "pen_clearance_over_object": pen_tip_h - h_obj,
            "cam_above_pen": cam_above_pen, "cam_offset_from_pen_paper": offset.tolist(), "h_obj": h_obj,
            "cam_h_home": H_home, "radius": radius, "notes": notes}
    return out, info


def draw_plan(color, pose, K, obj, views, info):
    img = color.copy()
    b = obj["bbox"]
    box3 = np.float64([[b[0], b[1], -obj["h_max"]], [b[2], b[1], -obj["h_max"]], [b[2], b[3], -obj["h_max"]],
                       [b[0], b[3], -obj["h_max"]]])
    rv, _ = cv2.Rodrigues(pose["R"])
    pts, _ = cv2.projectPoints(box3, rv, pose["t"], K, np.zeros(5))
    cv2.polylines(img, [np.int32(pts.reshape(-1, 2))], True, (0, 255, 0), 2)
    for v in views:
        for key, col, rad in (("cam_xy", (0, 140, 255), 9), ("pen_paper", (255, 0, 255), 4)):
            p, _ = cv2.projectPoints(np.float64([[*v[key], 0.0]]), rv, pose["t"], K, np.zeros(5))
            p = tuple(int(x) for x in p.ravel())
            cv2.circle(img, p, rad, col, 2)
            if key == "cam_xy":
                cv2.putText(img, v["name"], (p[0] + 10, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(img, f"object H~{obj['h_max']:.0f}mm  cam_h {info['cam_h_scan']:.0f}mm  pen clearance {info['pen_clearance_over_object']:.0f}mm",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "orange = camera centers, magenta = pen tip positions", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 0, 255), 1, cv2.LINE_AA)
    return img


def capture(cam, info, n_depth):
    s = cam.grab(n_depth=n_depth, settle=0.8, flush=5)
    mm = s["depth"].astype(np.float32) * float(info["depth_scale_m"]) * 1000.0
    mm[s["depth"] == 0] = np.nan
    return s, mm


def choose_pose(color, layout, K, predicted, min_tags=3):
    tp = tag_pose(color, layout, K)
    if tp is not None and tp["n_tags"] >= min_tags and tp["rms_px"] < 2.0:
        d = float(np.linalg.norm(tp["C"] - predicted["C"])) if predicted is not None else None
        return tp, "tags", tp, d
    if predicted is None:
        return None, "none", tp, None
    d = float(np.linalg.norm(tp["C"] - predicted["C"])) if tp is not None else None
    return predicted, "arm", tp, d


def refine_view_poses(views, probe_recs, layout, A, conv, log=None):
    """用所有看得到 >= 2 個 tag 的視角 + 試轉,一起解「手眼 + 紙面/手臂」完整模型 (scan_tilt.refine_full),
    再用它推算「tag 不足 3 個」的視角位姿。11-11 實測:原地試轉解的手眼讓傾斜視角差 20~40 mm;
    用全部視角 (手腕轉 ±90° 槓桿大) 解完,留一法約 2~9 mm。views 需要 "tcp"。回傳模型摘要或 None。"""
    import scan_tilt as stl
    recs = list(probe_recs)
    for v in views:
        if v.get("tcp") is None:
            continue
        tp = tag_pose(v["color"], layout, v["K"])
        v["_tp"] = tp
        if tp is not None and tp["rms_px"] < 1.5:
            recs.append({"tcp": v["tcp"], "R": tp["R"], "C": tp["C"]})
    if len(recs) < 6:
        return None
    R_PB, s0 = stl.paper_to_base(A)
    fs = stl.refine_full(recs, R_PB, s0, conv)
    if fs is None or fs["rms_mm"] > 12.0:
        print(f"  [位姿模型] 殘差太大 ({fs['rms_mm'] if fs else float('nan'):.1f} mm),不用")
        return None
    n_upd = 0
    for v in views:
        if v.get("tcp") is None or v["pose_source"] == "tags":
            continue
        v["pose"] = stl.predict_full(fs, v["tcp"], conv)
        v["pose_source"] = "model"
        n_upd += 1
    summ = {"n_records": len(recs), "used": fs["n"], "rms_mm": fs["rms_mm"], "rot_rms_deg": fs["rot_rms_deg"],
            "p_cam_in_tool_mm": fs["p"].tolist(), "scale": fs["s"], "updated_views": n_upd}
    print(f"  [位姿模型] {len(recs)} 筆 tag 資料解手眼:殘差 {fs['rms_mm']:.1f} mm / {fs['rot_rms_deg']:.2f}°,"
          f"相機在工具座標 {np.round(fs['p'], 1).tolist()},更新 {n_upd} 個視角的位姿")
    if log is not None:
        log.event("pose_model", **summ)
        log.save_json("pose_model", {**summ, "R_PB": fs["R_PB"].tolist(), "T": fs["T"].tolist(), "Rtc": fs["Rtc"].tolist(),
                                     "conv": conv})
    return summ


def fix_arm_poses(captured, log=None):
    """手臂位移推算的位姿,用同一類 (正上方 / 傾斜) tag 視角的平均偏差修正
    (11-11 實測:正上方推算和 tag 差 7~10 mm,多半是固定偏差;傾斜視角的手眼推算差 ~19 mm)。"""
    for fam in (False, True):
        _fix_family([v for v in captured if bool(v.get("tilt")) == fam], log)


def _fix_family(captured, log=None):
    tagv = [v for v in captured if v.get("pose_source") == "tags" and v.get("pred_C") is not None]
    armv = [v for v in captured if v.get("pose_source") in ("arm", "handeye")]
    if not tagv or not armv:
        return
    bias = np.mean([v["pose"]["C"] - v["pred_C"] for v in tagv], axis=0)
    R_mean = tagv[0]["pose"]["R"]
    for v in armv:
        if v["pose_source"] == "arm":
            v["pose"] = pose_from_center(R_mean, v["pred_C"] + bias)
        else:                                      # 傾斜視角:旋轉用手眼推算的,只修位置偏差
            v["pose"] = pose_from_center(v["pose"]["R"], v["pred_C"] + bias)
        v["pose_source"] += "+bias"
        if log is not None:
            log.event("arm_pose_fixed", name=v["name"], bias=bias.tolist(), C=v["pose"]["C"].tolist(), t=v["pose"]["t"].tolist())


PROBE_RECS = []          # 試轉時看到 >= 3 個 tag 的紀錄 (給 refine_view_poses 用)
HAND_EYE_CONV = ["xyz"]  # 試轉判斷出的歐拉角慣例


def tilt_phase(args, arm, c, cam, info, K, layout, captured, views, obj_plan, tcp_home, A, z_touch, log):
    """傾斜視角:試轉求手眼 -> 規劃周圍往物體看的視角 (含較低、較斜的) -> 逐點拍。回傳 (status, reason)。"""
    import scan_tilt as stl
    from drv_modbus import send, request
    from pen_calibrate import Paused, ask
    base = list(next((v for v in views if v["name"] == "c"), views[0])["tcp"])
    home_ori = list(tcp_home[3:])
    # 試轉在比較高的地方做:視野大,傾斜時 4 個 tag 比較不會跑出畫面 (11-11:在 380 mm 繞 y 轉 10° 就只剩 2 個 tag)
    base[2] = min(float(tcp_home[2]), base[2] + args.probe_lift)
    R_PB, s = stl.paper_to_base(A)
    print(f"\n=== 傾斜視角:先在物體上方 (抬高 {args.probe_lift:g} mm) 原地試轉 ±{args.probe_deg:g}° (繞兩個軸) 量手眼關係 ===")
    probes = []
    R0 = stl.R_tool(home_ori, "xyz")
    n_try = 0
    for axis, sign in ((None, 0), ([1, 0, 0], 1), ([0, 1, 0], 1), ([1, 0, 0], -1), ([0, 1, 0], -1)):
        for deg in ((0.0,) if axis is None else (args.probe_deg, args.probe_deg / 2)):   # 看不到 3 個 tag 就減半角度再試
            ori = home_ori if axis is None else stl.angles_of(stl.tilt_base(axis, sign * deg) @ R0, "xyz", near=home_ori)
            try:
                arm.move(*base[:3], ori=ori)
            except send.CommandIgnored as e:
                arm.stop_and_check("probe_ignored")
                print("  試轉沒有執行:", e)
                continue
            s_, mmv = capture(cam, info, args.n_depth)
            tcp_now = request.Get_TCP_Pose(c)
            tp = tag_pose(s_["color"], layout, K)
            name = f"probe{n_try}"
            n_try += 1
            log.save_snapshot(name, s_, info)
            log.event("probe", name=name, tcp=tcp_now, tags=tp["ids"] if tp else [], rms_px=tp["rms_px"] if tp else None,
                      R=tp["R"].tolist() if tp else None, C=tp["C"].tolist() if tp else None, axis=axis, deg=sign * deg)
            if tp is None or tp["n_tags"] < 3:
                print(f"  {name}:繞 {axis} 轉 {sign * deg:+g}° 只看到 tag {tp['ids'] if tp else []},不足 3 個")
                continue
            probes.append({"tcp": tcp_now, "R": tp["R"], "C": tp["C"]})
            PROBE_RECS.append({"tcp": tcp_now, "R": tp["R"], "C": tp["C"]})
            print(f"  {name}:繞 {axis or '-'} 轉 {sign * deg:+g}°,姿態 ({tcp_now[3]:.1f}, {tcp_now[4]:.1f}, {tcp_now[5]:.1f}),tag {tp['ids']}")
            break
    arm.move(*base[:3], ori=home_ori)
    try:
        he = stl.solve_handeye(probes, R_PB, s)
    except ValueError as e:
        print("  手眼無法求解:", e, " -> 略過傾斜視角")
        return "ok", None
    HAND_EYE_CONV[0] = he["conv"]
    log.save_json("hand_eye", {"conv": he["conv"], "R_tool_cam": he["R_tool_cam"].tolist(), "p_cam_in_tool_mm": he["p"].tolist(),
                               "rot_err_deg": he["rot_err_deg"], "pos_rms_mm": he["pos_rms_mm"], "R_PB": R_PB.tolist(), "s": s,
                               "probes": [{"tcp": p["tcp"], "C": p["C"].tolist()} for p in probes]})
    errs = ", ".join(f"{k} {v:.2f}°" for k, v in he["rot_err_deg"].items())
    print(f"  歐拉角慣例 {he['conv']} (各慣例旋轉誤差 {errs});"
          f"相機在工具座標 {np.round(he['p'], 1).tolist()} mm,位置殘差 {he['pos_rms_mm']:.2f} mm")
    if he["rot_err_deg"][he["conv"]] > 3.0 or he["pos_rms_mm"] > 4.0:
        print("  手眼誤差太大 (旋轉 > 3° 或位置 > 4 mm),為了安全不做傾斜視角")
        return "ok", None
    kin = stl.Kinematics(he, R_PB, s, probes[0]["tcp"], probes[0]["C"])
    W, H = layout["paper_mm"]
    tcal = np.asarray(args._tcal, float)

    def region(tcp):
        # TCP 正下方 (筆的位置,近似) 不能離開校正紙超過 --tilt-margin:紙外有輸送帶、框架,不要讓手臂跑過去
        pp = np.linalg.solve(A, np.asarray(tcp[:2]) - tcal)
        m = args.tilt_margin
        return -m <= pp[0] <= W + m and -m <= pp[1] <= H + m

    tv, notes = stl.plan_tilted(kin, obj_plan["center"], obj_plan["h_max"], args.tilts, args.n_tilt, args.view_dist,
                                z_touch, args.clearance, tcp_home, args.tilt_travel, min_range=args.min_range, region=region,
                                rolls=[r for r in stl.ROLLS if abs(r) <= args.max_roll],
                                ref_xy=(A @ np.asarray(obj_plan["center"], float) + tcal).tolist())
    for n_ in notes:
        print("  注意:", n_)
    log.save_json("tilt_plan", tv)
    if not tv:
        print("  沒有安全可行的傾斜視角")
        return "ok", None
    for v in tv:
        print(f"    {v['name']:7s} 傾斜 {v['tilt']:4.0f}° 手腕轉 {v['roll']:+4.0f}° 離物體 {v['dist']:3.0f} mm  TCP ({v['tcp'][0]:6.1f}, "
              f"{v['tcp'][1]:6.1f}, {v['tcp'][2]:6.1f}) 離物體中心 {v['tcp_from_home']:3.0f} mm  筆尖離紙 {v['pen_tip_above_paper']:3.0f} mm")
    print(f"  規劃 {len(tv)} 個傾斜視角:傾斜 {sorted({round(v['tilt']) for v in tv})} 度、相機離物體 "
          f"{min(v['dist'] for v in tv):.0f}~{max(v['dist'] for v in tv):.0f} mm、筆尖最低離紙 "
          f"{min(v['pen_tip_above_paper'] for v in tv):.0f} mm (物體規劃高 {obj_plan['h_max']:.0f} mm)")
    ask("確認周圍淨空 (手臂會傾斜、相機會降低),手放急停旁後按 Enter 開始傾斜掃描 ")
    i = 0
    while i < len(tv):
        v = tv[i]
        try:
            print(f"\n[傾斜 {i + 1}/{len(tv)}] {v['name']}:傾斜 {v['tilt']:.0f}°、離物體 {v['dist']:.0f} mm")
            try:
                arm.move(*v["tcp"][:3], ori=v["tcp"][3:])
            except send.CommandIgnored as e:
                arm.stop_and_check("tilt_ignored")
                log.event("tilt_skipped", name=v["name"], error=str(e))
                # 11-11:走到 t40_8 途中停住 (多半是奇異點 / 關節極限),之後所有 MovL 都被拒絕,連回 home 也不行。
                # 所以只要一次沒走到,就不再送傾斜指令,直接結束傾斜階段。
                print("  手臂沒有走到 (可能碰到奇異點 / 關節極限):", e)
                print("  不再送傾斜指令,結束傾斜階段 (已拍的照樣合併)")
                return "partial", "tilt_unreachable"
            s_, mmv = capture(cam, info, args.n_depth)
            tcp_now = request.Get_TCP_Pose(c)
            pred = kin.camera_pose(tcp_now)
            pose, src, tp, diff = choose_pose(s_["color"], layout, K, pred)
            if src == "arm":
                src = "handeye"
            log.save_snapshot(v["name"], s_, info)
            log.event("scan_view", name=v["name"], tcp=tcp_now, pose_source=src, pred_C=pred["C"].tolist(),
                      R=pose["R"].tolist(), t=pose["t"].tolist(), C=pose["C"].tolist(), tags=tp["ids"] if tp else [],
                      tag_rms_px=tp["rms_px"] if tp else None, tags_vs_arm_mm=diff, tilt=v["tilt"])
            print(f"  拍好:位姿用 {'tag ' + str(tp['ids']) if src == 'tags' else '手眼推算 (tag 不足 3 個)'}"
                  + (f",tag 與手眼推算差 {diff:.1f} mm" if diff is not None and src == "tags" else ""))
            captured.append({"name": v["name"], "color": s_["color"], "mm": mmv, "K": K, "pose": pose,
                             "pose_source": src, "pred_C": pred["C"], "tilt": v["tilt"], "tcp": tcp_now})
            i += 1
        except Paused as e:
            if str(e) == "ctrl_c":
                raise
            ch = ""
            while ch not in ("r", "s", "h", "q"):
                ch = ask("暫停:r = 重拍這點 / s = 跳過 / h = 回 home 並合併已拍的 / q = 回 home 結束 > ").strip().lower()
            if ch == "s":
                i += 1
            elif ch == "h":
                return "partial", "user_home"
            elif ch == "q":
                return "aborted", "user_quit"
    return "ok", None


def run_live(args):
    from pymodbus.client import ModbusTcpClient
    import warp_pick as wp
    from drv_modbus import send, request
    from rs_capture import RSCamera
    from pen_calibrate import Arm, Paused, ask

    with open(args.layout, encoding="utf-8") as f:
        layout = json.load(f)
    W, H = layout["paper_mm"]
    try:
        cal = CalibrationStore().load_latest("pen_paper")
    except FileNotFoundError:
        sys.exit("還沒有筆尖校正 (calibrations/pen_paper):需要它才知道筆尖和手臂的關係")
    A, tcal, z_touch = np.array(cal["data"]["A"]), np.array(cal["data"]["t"]), cal["data"].get("z_touch")
    args._tcal = tcal.tolist()
    if z_touch is None:
        sys.exit("筆尖校正沒有 z_touch")
    calib = wp.load_calibration()
    home = calib["home"]
    c = ModbusTcpClient(host=wp.ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    tcp = request.Get_TCP_Pose(c)
    ok, dpos, dang = wp.check_home(calib, tcp)
    flag = request.Get_Pose_Flag(c)
    print(f"TCP 與 home 差: 位置 {dpos:.1f} mm, 角度 {dang:.1f} 度;旗標 {flag}")
    if not ok:
        sys.exit("手臂不在 home (先回 home)")
    if flag != 1:
        sys.exit("旗標不是 1,請先在教導器清除警報")

    cam = RSCamera()
    info = cam.info()
    K = K_of(info["color_intrinsics"])
    snap, mm = capture(cam, info, args.n_depth)
    hp = tag_pose(snap["color"], layout, K)
    if hp is None or hp["n_tags"] < 4:
        cam.close()
        sys.exit("home 要看到校正紙 4 個 tag 才能定位 (看到 %s)" % (hp["ids"] if hp else "少於 2 個"))
    home_view = {"name": "home", "color": snap["color"], "mm": mm, "K": K, "pose": hp, "pose_source": "tags"}
    sc = depth_scale_from_paper(mm, K, hp, (W, H))
    xy, h, _, _ = view_points(snap["color"], mm, K, hp, scale=sc)
    pl = fit_paper_plane(xy, h, (W, H))
    if pl is None:
        cam.close()
        sys.exit("紙面深度點太少")
    h = h - (pl[0] * xy[:, 0] + pl[1] * xy[:, 1] + pl[2])
    obj = find_object(xy, h, (W, H))
    if obj is None:
        cam.close()
        sys.exit("校正紙上找不到比紙面高 3 mm 以上的物體")
    # home 距離遠,深度會低估高度 (11-10:5 mm 量成 3.2):規劃用的物體高度加 20% 安全餘量
    obj_plan = dict(obj, h_max=max(obj["h_max"] * 1.2 + 5.0, args.obj_h or 0.0))
    views, pinfo = plan_views(obj_plan, hp, tcp, A, tcal, z_touch, args, (W, H))
    print(f"\n物體:中心 ({obj['center'][0]:.0f}, {obj['center'][1]:.0f}) mm、home 量到最高約 {obj['h_max']:.1f} mm"
          f" (規劃用 {obj_plan['h_max']:.0f} mm)、面積 {obj['area_mm2']:.0f} mm²")
    print(f"掃描:相機離紙 {pinfo['cam_h_scan']:.0f} mm (TCP z {pinfo['z_scan']:.1f}),筆尖離紙 {pinfo['pen_tip_above_paper']:.0f} mm,"
          f"比物體高 {pinfo['pen_clearance_over_object']:.0f} mm;{len(views)} 個掃描點,半徑 {args.radius:g} mm")
    for n_ in pinfo["notes"]:
        print("  注意:", n_)
    for v in views:
        print(f"  {v['name']:3s} 相機 ({v['cam_xy'][0]:6.1f}, {v['cam_xy'][1]:6.1f})  TCP ({v['tcp'][0]:6.1f}, {v['tcp'][1]:6.1f}, {v['tcp'][2]:6.1f})")
    plan_img = draw_plan(snap["color"], hp, K, obj_plan, views, pinfo)
    cv2.imencode(".png", plan_img)[1].tofile(os.path.join(HERE, "scan3d_plan.png"))
    plan = {"object_home": obj, "object_plan": obj_plan, "views": views, "plan": pinfo, "K": K.tolist(),
            "home_pose": {"R": hp["R"].tolist(), "t": hp["t"].tolist(), "n_tags": hp["n_tags"], "rms_px": hp["rms_px"]},
            "tcp_home": tcp, "pen_calibration": {"created": cal["created"], "z_touch": z_touch}}

    with RunLog("scan3d", label=args.label or ("" if args.go else "dryrun"), params=vars(args)) as log:
        log.add_input(args.layout)
        log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
        log.event("tcp", pose=tcp, flag=flag)
        log.save_snapshot("home", snap, info)
        log.save_image("plan.jpg", plan_img)
        log.save_json("plan", plan)
        if not args.go:
            m = finish_outputs(log, [home_view], layout, obj)
            log.finish("dry-run", {"views": len(views), "h_top_home": (m or {}).get("h_top_mm")})
            cam.close()
            c.close()
            print("\n(dry-run) 沒有移動手臂。看 scan3d_plan.png:橘圈 = 相機要去的位置、洋紅 = 筆尖位置。確認後加 --go")
            print("紀錄:", log.path)
            return
        if not views:
            cam.close()
            sys.exit("沒有可用的掃描點")
        arm = Arm(c, log, home[3:], args.speed)
        captured = [home_view]
        status, reason = "ok", None
        print("\n[Ctrl+C = 停止並回 home] [移動中 空白/Esc/p = 選單]  這個程式不會下筆")
        try:
            ask(f"確認筆尖下方淨空、手放急停旁後按 Enter 開始掃描 ({len(views)} 點,速度 {args.speed}) ")
            i = 0
            while i < len(views):
                v = views[i]
                try:
                    print(f"\n[{i + 1}/{len(views)}] {v['name']}:移到 TCP ({v['tcp'][0]:.1f}, {v['tcp'][1]:.1f}, {v['tcp'][2]:.1f})")
                    arm.move(*v["tcp"], ori=home[3:])
                    s, mmv = capture(cam, info, args.n_depth)
                    tcp_now = request.Get_TCP_Pose(c)
                    d_tcp = np.array(tcp_now[:3]) - np.array(tcp[:3])
                    dpen = np.linalg.solve(A, d_tcp[:2])          # 手臂位移 -> 紙面位移
                    pred_C = hp["C"] + np.array([dpen[0], dpen[1], -d_tcp[2]])
                    pred = pose_from_center(hp["R"], pred_C)
                    pose, src, tp, diff = choose_pose(s["color"], layout, K, pred)
                    log.save_snapshot(v["name"], s, info)
                    rec = {"name": v["name"], "tcp": tcp_now, "pose_source": src, "pred_C": pred_C.tolist(), "R": pose["R"].tolist(),
                           "t": pose["t"].tolist(), "C": pose["C"].tolist(), "tags": tp["ids"] if tp else [],
                           "tag_rms_px": tp["rms_px"] if tp else None, "tags_vs_arm_mm": diff}
                    log.event("scan_view", **rec)
                    print(f"  拍好:位姿用 {'tag ' + str(tp['ids']) if src == 'tags' else '手臂位移推算 (tag 不足 3 個)'}"
                          + (f",tag 與手臂推算差 {diff:.1f} mm" if diff is not None and src == "tags" else ""))
                    captured.append({"name": v["name"], "color": s["color"], "mm": mmv, "K": K, "pose": pose,
                                     "pose_source": src, "pred_C": pred_C, "tcp": tcp_now})
                    i += 1
                except Paused as e:
                    if str(e) == "ctrl_c":
                        raise
                    ch = ""
                    while ch not in ("r", "s", "h", "q"):
                        ch = ask("暫停:r = 重拍這點 / s = 跳過 / h = 回 home 並合併已拍的 / q = 回 home 結束 > ").strip().lower()
                    if ch == "s":
                        i += 1
                    elif ch == "h":
                        status, reason = "partial", "user_home"
                        break
                    elif ch == "q":
                        status, reason = "aborted", "user_quit"
                        break
            if status == "ok" and args.tilts:
                st2, rs2 = tilt_phase(args, arm, c, cam, info, K, layout, captured, views, obj_plan, tcp, A, z_touch, log)
                if st2 != "ok":
                    status, reason = st2, rs2
        except (Paused, KeyboardInterrupt):
            status, reason = "partial", "ctrl_c"
            print("\nCtrl+C:停止 -> 回 home (已拍的視角照樣合併)")
        home_ok = True
        try:
            print("\n回 home ...")
            arm.move(*home[:3], ori=home[3:])
        except (Paused, KeyboardInterrupt):
            print("回 home 被中斷,手臂停在目前位置 (之後用回 home 任務)")
            status, home_ok = "aborted", False
        except (send.CommandIgnored, TimeoutError) as e:
            home_ok = False
            log.event("home_failed", error=str(e))
            print("\n!!! 回 home 沒有執行:", e)
            print("!!! 手臂可能停在奇異點 / 關節極限附近,直線移動 (MovL) 被拒絕。")
            p_now = request.Get_TCP_Pose(c)
            if p_now:
                print(f"!!! 目前 TCP {[round(x, 1) for x in p_now]};筆尖離紙約 {p_now[2] - z_touch:.0f} mm "
                      f"(物體約 {obj['h_max']:.0f} mm)")
            print("!!! 請先看教導器有沒有警報,有的話先清除。")
            status = "partial"
            try:
                ch = ask("按 Enter 改用關節移動 (MovP,速度 10) 回 home;路徑不是直線,手放急停旁 (q = 不動,自己處理) > ")
                if ch.strip().lower() != "q":
                    arm.move(*home[:3], mov=send.MovP, speed=10, ori=home[3:])
                    home_ok = True
                    print("已回 home")
            except (Paused, KeyboardInterrupt, send.CommandIgnored, TimeoutError) as e2:
                print("!!! 關節移動也沒有完成:", e2, " -> 請用教導器把手臂移回安全位置")
        cam.close()
        c.close()
        if refine_view_poses(captured, PROBE_RECS, layout, A, HAND_EYE_CONV[0], log) is None:
            fix_arm_poses(captured, log)
        if status == "aborted" and reason == "user_quit":
            log.finish("aborted", {"reason": reason, "views": len(captured) - 1})
            print("紀錄:", log.path)
            return
        m = finish_outputs(log, captured, layout, obj)
        res = {"views": len(captured) - 1, "reason": reason}
        if m:
            res.update(h_top_mm=m["h_top_mm"], top_long_mm=(m["top_rect"] or {}).get("long_mm"),
                       top_short_mm=(m["top_rect"] or {}).get("short_mm"), top_flatness_mm=m["top_flatness_mm"])
        res["home_ok"] = home_ok
        log.finish(status, res)
        print("結果圖: scan3d_result.png;點雲: data/model.ply (可用 MeshLab / CloudCompare 開)")
        print("紀錄:", log.path)


def run_reprocess(args):
    from run_log import load_run
    run = args.reprocess
    meta, _ = load_run(run)
    with open(os.path.join(run, "data", "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    with open(args.layout, encoding="utf-8") as f:
        layout = json.load(f)
    K = np.array(plan["K"])
    ev = []
    with open(os.path.join(run, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "scan_view":
                ev.append(e)
    views = []
    color, mm, _ = load_snapshot(run, "home")
    hp = tag_pose(color, layout, K)
    views.append({"name": "home", "color": color, "mm": mm, "K": K, "pose": hp, "pose_source": "tags"})
    fixed = {}
    with open(os.path.join(run, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e2 = json.loads(line)
            if e2.get("type") == "arm_pose_fixed":
                fixed[e2["name"]] = e2
    for e in ev:
        d = e.get("data", e)
        if d["name"] in fixed:
            d = dict(d, C=fixed[d["name"]]["C"], t=fixed[d["name"]]["t"], pose_source=d["pose_source"] + "+bias")
        color, mm, _ = load_snapshot(run, d["name"])
        pose = {"R": np.array(d["R"]), "t": np.array(d["t"]), "C": np.array(d["C"])}
        views.append({"name": d["name"], "color": color, "mm": mm, "K": K, "pose": pose, "pose_source": d["pose_source"],
                      "pred_C": np.array(d["pred_C"]) if d.get("pred_C") else None, "tilt": d.get("tilt"),
                      "tcp": d.get("tcp")})
    probe_recs = []
    with open(os.path.join(run, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e2 = json.loads(line)
            if e2.get("type") == "probe" and e2.get("C") is not None and len(e2.get("tags") or []) >= 3:
                probe_recs.append({"tcp": e2["tcp"], "R": np.array(e2["R"]), "C": np.array(e2["C"])})
    conv = "xyz"
    hp_path = os.path.join(run, "data", "hand_eye.json")
    if os.path.isfile(hp_path):
        with open(hp_path, encoding="utf-8") as f:
            conv = json.load(f).get("conv", "xyz")
    with open(os.path.join(run, "inputs", "pen_paper.json"), encoding="utf-8") as f:
        A_run = np.array(json.load(f)["data"]["A"])
    for v in views:                                # 先還原成「原始」來源,讓模型重新決定
        v["pose_source"] = v["pose_source"].replace("+bias", "")
    if args.pose != "raw" and refine_view_poses(views, probe_recs, layout, A_run, conv) is None and not fixed:
        fix_arm_poses(views)
    print(f"{run}:{len(views)} 個視角")
    with RunLog("scan3d_reprocess", label=os.path.basename(run.rstrip("\\/")), params=vars(args)) as log:
        m = finish_outputs(log, views, layout, plan["object_home"])
        log.finish("ok", {"source_run": run, "h_top_mm": (m or {}).get("h_top_mm")})
        print("紀錄:", log.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="實際移動手臂掃描 (預設 dry-run)")
    ap.add_argument("--cam-h", type=float, default=380.0, help="掃描時相機離紙面高度 mm (home 約 520)")
    ap.add_argument("--radius", type=float, default=60.0, help="周圍一圈的相機位置離物體中心 mm")
    ap.add_argument("--n-ring", type=int, default=8, help="周圍一圈幾個視角 (0 = 只拍正上方)")
    ap.add_argument("--clearance", type=float, default=40.0, help="筆尖至少比物體最高點高多少 mm")
    ap.add_argument("--min-range", type=float, default=230.0, help="相機離物體頂面至少 mm (D435 在 848x480 約 200 mm 以內量不到)")
    ap.add_argument("--obj-h", type=float, default=0.0, help="物體高度上限 mm (規劃安全高度用;0 = 用 home 深度估計 x1.2 + 5)")
    ap.add_argument("--tilts", default="25,40",
                    help="傾斜視角:相機光軸偏離垂直的角度 (度,逗號分隔;空字串 = 不做)。先原地試轉量手眼,再往物體看")
    ap.add_argument("--n-tilt", type=int, default=10, help="每個傾斜角度繞一圈幾個視角")
    ap.add_argument("--view-dist", type=float, default=290.0, help="傾斜視角時相機離物體中心 mm (不安全會自動推遠)")
    ap.add_argument("--probe-deg", type=float, default=8.0, help="試轉角度 (度;看不到 3 個 tag 時自動減半)")
    ap.add_argument("--tilt-travel", type=float, default=260.0, help="傾斜視角時 TCP 離物體中心的最大水平距離 mm")
    ap.add_argument("--max-roll", type=float, default=90.0, help="傾斜視角時手腕最多轉幾度 (注意相機 USB 線與氣管)")
    ap.add_argument("--tilt-margin", type=float, default=60.0, help="傾斜視角時 TCP 正下方最多可超出校正紙多少 mm")
    ap.add_argument("--probe-lift", type=float, default=60.0, help="試轉時比正上方視角抬高多少 mm (視野較大)")
    ap.add_argument("--n-depth", type=int, default=10, help="每個視角取幾幀深度做中位數")
    ap.add_argument("--max-travel", type=float, default=150.0, help="掃描點離 home 的最大水平距離 mm")
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--label", default="")
    ap.add_argument("--reprocess", default=None, help="離線重算某次紀錄 (runs\\<id>)")
    ap.add_argument("--pose", choices=["auto", "raw"], default="auto",
                    help="--reprocess 時:auto = 用全部 tag 資料重解手眼模型修正位姿;raw = 用當時記錄的位姿")
    args = ap.parse_args()
    if args.reprocess:
        return run_reprocess(args)
    args.tilts = [float(x) for x in str(args.tilts).replace(" ", "").split(",") if x]
    if any(not 5 <= t <= 50 for t in args.tilts) or not 0 <= args.n_tilt <= 16 or not 3 <= args.probe_deg <= 15:
        sys.exit("--tilts 每個 5~50 度、--n-tilt 0~16、--probe-deg 3~15")
    if args.n_tilt == 0:
        args.tilts = []
    if not 1 <= args.speed <= 30:
        sys.exit("速度需在 1~30")
    if args.clearance < 20:
        sys.exit("--clearance 至少 20 mm")
    if not 0 <= args.n_ring <= 12 or not 0 <= args.radius <= 120:
        sys.exit("--n-ring 0~12、--radius 0~120")
    run_live(args)


measure_cube = measure_object

if __name__ == "__main__":
    main()
