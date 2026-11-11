# -*- coding: utf-8 -*-
"""
立體工件 (A:L 型 / 屋脊形) 的脊線擷取 (純計算,可離線測試)。

輸入:點雲 (紙面座標 xy mm、高度 h mm,高出紙面為正;scan3d 的合併結果或單一視角)
做法
  1. 取物體點 (h > 30% 最高點),PCA 求物體的長軸 u (脊線方向的初值)
  2. 沿 u 每 slice_mm 切一片;每片取「最高的那些點」(h >= 該片最高 - top_band) 的橫向位置與高度中位數 -> 脊線點
  3. 去掉兩端 (高度掉下來的部分、end_margin),用穩健直線擬合 (3D) -> 脊線;同時保留逐片的折線 (給彎的脊線用)
  4. 另外用兩個平面 (脊線兩側斜面) 擬合,交線當交叉驗證:和逐片結果的水平差、高度差、兩面夾角
"""
import numpy as np


def largest_component(xy, mask, cell=1.0):
    import cv2
    o = xy[mask].min(0) - 3
    ij = ((xy - o) / cell).astype(int)
    gw, gh = ij[mask].max(0) + 4
    g = np.zeros((gh, gw), np.uint8)
    g[ij[mask, 1], ij[mask, 0]] = 1
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g)
    if n < 2:
        return mask
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    inb = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < gw) & (ij[:, 1] < gh)
    out = np.zeros(len(mask), bool)
    out[inb] = lab[ij[inb, 1], ij[inb, 0]] == k
    return out


def _robust_line(s, y, it=4):
    """y = a + b·s,穩健 (MAD 剔除)。"""
    keep = np.ones(len(s), bool)
    for _ in range(it):
        A = np.c_[np.ones(keep.sum()), s[keep]]
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = y - (coef[0] + coef[1] * s)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) * 1.4826
        keep = np.abs(r) < max(0.6, 3 * mad)
    return coef, r, keep


def _plane_fit(P):
    c = P.mean(0)
    w, V = np.linalg.eigh((P - c).T @ (P - c))
    n = V[:, 0]
    return c, n * (1 if n[2] > 0 else -1)


def extract_ridge(xy, h, slice_mm=2.0, top_band=1.5, end_margin=6.0, min_frac=0.3):
    """回傳 dict 或 None:
      line_xyh : 直線脊線的兩端點 [[x, y, h], [x, y, h]] (已去掉兩端 end_margin)
      poly_xyh : 逐片脊線點 (N,3)
      dir_xy   : 脊線水平方向 (單位向量)、length_mm、h_mean_mm、slope_deg (沿脊線的坡度)
      lateral_rms_mm / height_rms_mm : 逐片點相對直線的殘差 (越小越直、越可信)
      planes   : 兩側斜面擬合結果 (夾角、交線和直線脊線的水平/高度差)"""
    xy, h = np.asarray(xy, float), np.asarray(h, float)
    if len(h) < 500:
        return None
    hmax = float(np.percentile(h, 99.5))
    obj = h > min_frac * hmax
    if obj.sum() < 300:
        return None
    obj &= largest_component(xy, obj)              # 只留最大的一塊 (11-11:ROI 邊緣的電線讓脊線多出 60 mm)
    if obj.sum() < 300:
        return None
    P, H = xy[obj], h[obj]
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    u = V[:, 1]                                     # 長軸
    v = np.array([-u[1], u[0]])
    s = (P - c) @ u
    t = (P - c) @ v
    pts = []
    for s0 in np.arange(s.min(), s.max(), slice_mm):
        m = (s >= s0) & (s < s0 + slice_mm)
        if m.sum() < 30:
            continue
        hm = np.percentile(H[m], 98)
        top = m & (H >= hm - top_band)
        if top.sum() < 5:
            continue
        pts.append([s0 + slice_mm / 2, float(np.median(t[top])), float(np.median(H[top])), int(m.sum())])
    if len(pts) < 8:
        return None
    pts = np.array(pts)
    # 兩端:高度明顯低於中段的片 (斜面收尾、雜點) 去掉
    hmid = np.median(pts[:, 2])
    good = pts[:, 2] > hmid - max(3.0, 0.15 * hmid)
    idx = np.nonzero(good)[0]
    lo, hi = pts[idx[0], 0] + end_margin, pts[idx[-1], 0] - end_margin
    sel = good & (pts[:, 0] >= lo) & (pts[:, 0] <= hi)
    if sel.sum() < 5:
        return None
    S, T, Hs = pts[sel, 0], pts[sel, 1], pts[sel, 2]
    ct, rt, kt = _robust_line(S, T)
    ch, rh, kh = _robust_line(S, Hs)
    ends = []
    for se in (lo, hi):
        tt = ct[0] + ct[1] * se
        ends.append([*(c + se * u + tt * v), ch[0] + ch[1] * se])
    ends = np.array(ends)
    d = ends[1, :2] - ends[0, :2]
    L = float(np.linalg.norm(d))
    poly = np.array([[*(c + S[i] * u + T[i] * v), Hs[i]] for i in range(len(S))])
    out = {"line_xyh": ends.tolist(), "poly_xyh": poly.tolist(), "dir_xy": (d / L).tolist(), "length_mm": L,
           "h_mean_mm": float(np.mean(ends[:, 2])), "slope_deg": float(np.degrees(np.arctan2(ends[1, 2] - ends[0, 2], L))),
           "lateral_rms_mm": float(np.sqrt(np.mean(rt[kt] ** 2))), "height_rms_mm": float(np.sqrt(np.mean(rh[kh] ** 2))),
           "n_slices": int(sel.sum()), "h_max_mm": hmax}
    # ---- 交叉驗證:兩側斜面 ----
    side = obj & (h < hmax - 3.0) & (h > 0.3 * hmax)
    Q = np.c_[xy[side], h[side]]
    tq = (xy[side] - c) @ v - (ct[0] + ct[1] * ((xy[side] - c) @ u))   # 相對脊線的橫向位置
    planes = None
    if (tq < -2).sum() > 300 and (tq > 2).sum() > 300:
        c1, n1 = _plane_fit(Q[tq < -2])
        c2, n2 = _plane_fit(Q[tq > 2])
        ang = float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1, 1))))
        dirl = np.cross(n1, n2)
        if np.linalg.norm(dirl) > 1e-6:
            dirl /= np.linalg.norm(dirl)
            # 交線上一點:解 n1·x = n1·c1, n2·x = n2·c2, dirl·x = dirl·(脊線中點)
            mid = ends.mean(0)
            M = np.vstack([n1, n2, dirl])
            p0 = np.linalg.solve(M, [n1 @ c1, n2 @ c2, dirl @ mid])
            off = p0 - mid
            planes = {"angle_between_deg": ang, "ridge_angle_deg": 180.0 - ang,
                      "lateral_diff_mm": float(off[:2] @ np.array([-out["dir_xy"][1], out["dir_xy"][0]])),
                      "height_diff_mm": float(off[2]),
                      "dir_diff_deg": float(np.degrees(np.arccos(np.clip(abs(dirl[:2] @ np.array(out["dir_xy"])) /
                                                                         max(1e-9, np.linalg.norm(dirl[:2])), -1, 1))))}
    out["planes"] = planes
    return out


def path_along(line_xyh, step_mm=2.0, reverse=False):
    """直線脊線 -> 等距點 (N,3) [x, y, h]。"""
    a, b = (np.asarray(line_xyh[1]), np.asarray(line_xyh[0])) if reverse else (np.asarray(line_xyh[0]), np.asarray(line_xyh[1]))
    L = np.linalg.norm(b[:2] - a[:2])
    n = max(2, int(np.ceil(L / step_mm)) + 1)
    return np.array([a + (b - a) * k / (n - 1) for k in range(n)])
