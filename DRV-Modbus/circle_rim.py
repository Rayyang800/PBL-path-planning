# -*- coding: utf-8 -*-
"""
工件 C (圓形,例如鮪魚罐頭) 頂面外圈的擷取 (純計算,可離線測試)。

輸入:點雲 (紙面座標 xy mm、高度 h mm)
  1. 取物體最大一塊、頂部的點 (h > top_frac x 最高點)
  2. 以初估圓心為中心,每 5° 一個扇形,取該扇形頂部點的外緣半徑 (97 百分位) -> 外緣點
  3. 外緣點做穩健圓擬合 (代數最小平方 + 剔除離群) -> 圓心、外緣半徑
  4. 捲邊高度:外緣往內 1~6 mm 的環內,高度 90 百分位
"""
import numpy as np

import ridge


def fit_circle(P):
    """代數 (Kasa) 圓擬合。回傳 (cx, cy, r)。"""
    x, y = P[:, 0], P[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    return float(cx), float(cy), float(np.sqrt(c[2] + cx ** 2 + cy ** 2))


def robust_circle(P, it=4):
    keep = np.ones(len(P), bool)
    for _ in range(it):
        cx, cy, r = fit_circle(P[keep])
        res = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) < max(0.8, 3 * mad)
        if new.sum() < 8 or (new == keep).all():
            break
        keep = new
    return cx, cy, r, float(np.sqrt(np.mean(res[keep] ** 2))), keep


def extract_rim(xy, h, top_frac=0.7, sector_deg=5.0):
    """回傳 dict(center, r_outer, rms_mm, h_rim_mm, h_top_mm, n_edge, coverage) 或 None。"""
    xy, h = np.asarray(xy, float), np.asarray(h, float)
    if len(h) < 500:
        return None
    hmax = float(np.percentile(h, 99.5))
    if hmax < 5:
        return None
    top = h > top_frac * hmax
    top &= ridge.largest_component(xy, top)
    if top.sum() < 300:
        return None
    P = xy[top]
    c0 = P.mean(0)
    ang = np.arctan2(P[:, 1] - c0[1], P[:, 0] - c0[0])
    rad = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
    edge = []
    for a0 in np.arange(-np.pi, np.pi, np.radians(sector_deg)):
        m = (ang >= a0) & (ang < a0 + np.radians(sector_deg))
        if m.sum() < 10:
            continue
        r97 = np.percentile(rad[m], 97)
        a_mid = a0 + np.radians(sector_deg) / 2
        edge.append(c0 + r97 * np.array([np.cos(a_mid), np.sin(a_mid)]))
    n_sec = int(round(360 / sector_deg))
    if len(edge) < 0.5 * n_sec:
        return None
    E = np.array(edge)
    cx, cy, r, rms, keep = robust_circle(E)
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    ring = (d > r - 6) & (d < r - 1)
    h_rim = float(np.percentile(h[ring], 90)) if ring.sum() > 50 else hmax
    inner = d < 0.5 * r
    return {"center": [cx, cy], "r_outer": r, "rms_mm": rms, "n_edge": int(keep.sum()), "coverage": len(edge) / n_sec,
            "h_rim_mm": h_rim, "h_top_mm": float(np.median(h[inner])) if inner.sum() > 50 else None, "h_max_mm": hmax}


def circle_path(center, r, h, step_mm=4.0, start_deg=0.0, overlap_deg=8.0, ccw=True):
    """圓周等距點 (N,3) [x, y, h];多走 overlap_deg 讓頭尾接起來。"""
    n = max(12, int(np.ceil(2 * np.pi * r / step_mm)))
    span = 2 * np.pi + np.radians(overlap_deg)
    k = np.arange(int(np.ceil(n * span / (2 * np.pi))) + 1)
    a = np.radians(start_deg) + (1 if ccw else -1) * k * (2 * np.pi / n)
    return np.c_[center[0] + r * np.cos(a), center[1] + r * np.sin(a), np.full(len(a), h)]


def refine_with_color(img, pose, K, center, r0, h, search=(-3.0, 6.0), step_deg=3.0):
    """用彩色影像修正圓心 / 半徑:沿每個方向找「最外側的強邊緣」(罐頭和背景的交界),反投影到捲邊高度再擬合圓。
    深度在物體邊緣有一側會缺 / 偏 (RealSense 雙鏡頭遮擋),11-18 罐頭:深度圓心偏了約 2 mm;彩色邊緣擬合殘差 0.17 mm。
    回傳 dict(center, r_outer, rms_mm, kept_frac, shift_mm) 或 None。"""
    import cv2
    import ridge_check as rc
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    H, W = g.shape
    edges = []
    rs = np.arange(r0 + search[0], r0 + search[1], 0.2)
    for a in np.radians(np.arange(0, 360, step_deg)):
        P = np.c_[center[0] + rs * np.cos(a), center[1] + rs * np.sin(a), np.full(len(rs), h)]
        uv = rc.project(pose, K, P)
        dirv = uv[-1] - uv[0]
        dirv = dirv / max(1e-9, np.linalg.norm(dirv))
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        if ui.min() < 0 or vi.min() < 0 or ui.max() >= W or vi.max() >= H:
            continue
        vals = np.abs(gx[vi, ui] * dirv[0] + gy[vi, ui] * dirv[1])
        strong = np.nonzero(vals > 0.4 * vals.max())[0]
        k = int(strong[-1]) if len(strong) else int(np.argmax(vals))
        while k + 1 < len(vals) and vals[k + 1] > vals[k]:
            k += 1
        edges.append([center[0] + rs[k] * np.cos(a), center[1] + rs[k] * np.sin(a)])
    if len(edges) < 0.6 * 360 / step_deg:
        return None
    E = np.array(edges)
    cx, cy, r, rms, keep = robust_circle(E)
    return {"center": [cx, cy], "r_outer": r, "rms_mm": rms, "kept_frac": float(keep.mean()),
            "shift_mm": [cx - center[0], cy - center[1]]}
