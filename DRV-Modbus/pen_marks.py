# -*- coding: utf-8 -*-
"""
筆尖校正的離線分析:在 home 拍的照片裡找出筆留下的墨點 (dot) 或「+」(plus),求「紙面 mm -> 手臂 TCP 指令 XY」的轉換。不連線手臂。

原理
  pen_calibrate.py 在手臂座標 C_i 畫「+」(手臂移動 TCP 到 C_i,筆尖實際落在 C_i + 筆尖偏移)。
  照片 -> 紙面座標 P_i (用 4 個 AprilTag)。擬合 C = A·P + t (仿射,至少 3 點;建議 5 點看殘差)。
  筆尖偏移、紙面旋轉、tag 間距的比例誤差都包含在 A、t 裡,之後「要筆畫在紙面 P」就送 TCP 到 A·P + t。
  接觸高度 z_i 再擬合平面 z = a·x + b·y + c (手臂座標),得到紙面傾斜。

  python pen_marks.py runs\\<pen_calib 的資料夾>          # 分析並存進 calibrations/pen_paper/
  python pen_marks.py runs\\<...> --no-store              # 只分析不存版本
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from paper_frame import PaperFrame

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- rough pixel <-> robot (舊 calibration.json,只用來預測「+」大概在哪) ----------
def rough_homography(calib):
    """回傳 (H 像素->手臂XY, H 反向)。誤差可達 ±20 mm,只用於找 ROI 與筆畫方向。"""
    ids = ("1", "2", "3", "4")
    src = np.float32([calib["tag_pixels"][i] for i in ids])
    dst = np.float32([calib["tag_robot_xyz"][i][:2] for i in ids])
    H = cv2.getPerspectiveTransform(src, dst)
    return H, np.linalg.inv(H)


def apply_h(H, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


# ---------- cross detection ----------
def find_cross(gray, center, dir_a, dir_b, radius, exclude=None):
    """在 center 附近 (半徑 radius px) 找「+」的中心。dir_a / dir_b = 兩筆在影像中的方向 (單位向量)。
    回傳 dict(ok, uv, n_dark, peak_a, peak_b, reason)。"""
    h, w = gray.shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, x1, y0, y1 = max(0, cx - radius), min(w, cx + radius + 1), max(0, cy - radius), min(h, cy + radius + 1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return {"ok": False, "reason": "ROI 超出影像"}
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    bg, dark = np.median(roi), np.percentile(roi, 1)
    if bg - dark < 50:
        return {"ok": False, "reason": f"ROI 內沒有明顯筆跡 (背景 {bg:.0f}, 最暗 {dark:.0f})"}
    m = roi < (bg + dark) / 2
    if exclude is not None:
        m &= exclude[y0:y1, x0:x1] == 0
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return {"ok": False, "reason": f"筆跡像素太少 ({len(xs)})"}
    p = np.stack([xs + x0, ys + y0], 1).astype(np.float64)
    na = np.array([-dir_a[1], dir_a[0]])            # 垂直於 a 筆的法向 -> a 筆的像素在 na·p 上集中
    nb = np.array([-dir_b[1], dir_b[0]])

    def peak(v):
        lo = np.floor(v.min()) - 2
        hist = np.bincount((v - lo).astype(int))
        sm = np.convolve(hist, np.ones(3), "same")
        k = int(np.argmax(sm))
        sel = np.abs(v - (lo + k + 0.5)) <= 4
        return float(v[sel].mean()), float(sm[k] / max(1, len(v)))

    a0, fa = peak(p @ na)
    b0, fb = peak(p @ nb)
    M = np.array([na, nb])
    uv = np.linalg.solve(M, np.array([a0, b0]))
    return {"ok": True, "uv": uv.tolist(), "n_dark": int(len(xs)), "peak_a": fa, "peak_b": fb}


def find_dot(gray, center, radius, exclude=None, new_ink=None, min_area=4, max_area=1500):
    """在 center 附近找墨點:ROI 內的暗色連通區,取最靠近 center 的一塊,回傳面積加權質心。
    new_ink: 「點之後比點之前變暗」的遮罩 (有的話只看新墨跡,不會被紙上舊髒點干擾)。"""
    h, w = gray.shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, x1, y0, y1 = max(0, cx - radius), min(w, cx + radius + 1), max(0, cy - radius), min(h, cy + radius + 1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return {"ok": False, "reason": "ROI 超出影像"}
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    # 墨點只有幾十個像素,用百分位數會被背景淹沒;改用輕微模糊後的最小值 (去掉單點雜訊)
    bg, dark = np.median(roi), float(cv2.blur(roi, (3, 3)).min())
    if bg - dark < 40:
        return {"ok": False, "reason": f"ROI 內沒有明顯墨點 (背景 {bg:.0f}, 最暗 {dark:.0f})"}
    thr = (bg + dark) / 2
    m = (roi < thr).astype(np.uint8)
    if exclude is not None:
        m &= (exclude[y0:y1, x0:x1] == 0).astype(np.uint8)
    if new_ink is not None:
        m &= (new_ink[y0:y1, x0:x1] > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    cands = [k for k in range(1, n) if min_area <= stats[k, cv2.CC_STAT_AREA] <= max_area]
    if not cands:
        return {"ok": False, "reason": "ROI 內找不到大小合理的墨點"}
    loc = np.array([cx - x0, cy - y0], float)
    k = min(cands, key=lambda i: np.linalg.norm(cents[i] - loc))
    ys, xs = np.nonzero(lab == k)
    wgt = thr - roi[ys, xs]                       # 越黑權重越大
    u = float((xs * wgt).sum() / wgt.sum()) + x0
    v = float((ys * wgt).sum() / wgt.sum()) + y0
    return {"ok": True, "uv": [u, v], "area_px": int(stats[k, cv2.CC_STAT_AREA]), "n_candidates": len(cands)}


def new_ink_mask(before, after, pf_after, pf_before, min_drop=35, return_drop=False):
    """把 before 用兩張圖的紙面單應對齊到 after,回傳「after 明顯比 before 暗」的遮罩
    (return_drop=True 時另回傳變暗量,可當權重算中心)。"""
    Hb2a = pf_after.H0inv @ pf_before.H0                 # before 像素 -> 紙面 -> after 像素
    gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    ga = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpPerspective(gb, Hb2a, (ga.shape[1], ga.shape[0]), borderValue=255)
    warped = cv2.erode(warped, np.ones((3, 3), np.uint8))   # 舊的暗點稍微擴大,容忍 1 px 對位誤差
    drop = warped.astype(np.int16) - ga.astype(np.int16)
    mask = (drop > min_drop).astype(np.uint8) * 255
    return (mask, np.clip(drop, 0, None).astype(np.float32)) if return_drop else mask


def ink_mask(img, ink="any"):
    """ink: any = 任何顏色 (只靠「變暗」判斷);red = 只算紅色;dark = 只算很暗的。"""
    if ink == "any":
        return np.ones(img.shape[:2], np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if ink == "red":
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        return (((h <= 10) | (h >= 165)) & (s >= 70) & (v >= 40)).astype(np.uint8)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 90).astype(np.uint8)


def new_ink_blobs(ref, img, ink="any", min_area=6, min_drop=25):
    """同一個相機位置拍的 ref (下降前) 與 img (下降後):找新出現的筆跡 (點或小十字)。
    回傳 [(面積, (u, v))...] 由大到小。"""
    m = ink_mask(img, ink)
    if ink != "any":
        m &= 1 - cv2.dilate(ink_mask(ref, ink), np.ones((5, 5), np.uint8))   # 先前就有的同色墨跡排除
    g0 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    m &= ((g0 - g1) > min_drop).astype(np.uint8)                                # 而且要真的變暗
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs = [(int(stats[k, cv2.CC_STAT_AREA]), tuple(cents[k])) for k in range(1, n)
             if stats[k, cv2.CC_STAT_AREA] >= min_area]
    return sorted(blobs, reverse=True)


def tag_exclusion_mask(shape, pf, grow_mm=4.0):
    """把 4 個 tag (外擴 grow_mm) 畫成遮罩,避免把 tag 當成筆跡。"""
    mask = np.zeros(shape[:2], np.uint8)
    for t in pf.layout["tags"]:
        c = np.array(t["corners_mm"], float)
        cen = c.mean(0)
        c = cen + (c - cen) * (1 + 2 * grow_mm / pf.layout["tag_mm"])
        cv2.fillConvexPoly(mask, np.int32(np.round(pf.paper_to_pix(c))), 255)
    return mask


# ---------- fitting ----------
def fit_affine(P, C):
    """C ≈ A·P + t;回傳 (A 2x2, t 2, 殘差 (N,2))。"""
    P, C = np.asarray(P, float), np.asarray(C, float)
    X = np.hstack([P, np.ones((len(P), 1))])
    sol, *_ = np.linalg.lstsq(X, C, rcond=None)
    A, t = sol[:2].T, sol[2]
    return A, t, C - (P @ A.T + t)


def describe_affine(A):
    """分解出旋轉角、兩軸比例、是否鏡像 (紙面 y 向下、手臂右手系,通常會是鏡像)。"""
    sx, sy = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
    ang = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    skew = np.degrees(np.arccos(np.clip(abs(A[:, 0] @ A[:, 1]) / (sx * sy), -1, 1)))
    return {"rotation_deg": float(ang), "scale_x": float(sx), "scale_y": float(sy),
            "axis_angle_deg": float(skew), "mirrored": bool(np.linalg.det(A) < 0)}


def fit_rigid(P, C, allow_mirror=True):
    """C ≈ R·P + t,R 為正交矩陣 (比例固定 1:1;紙面 mm 與手臂 mm 同尺度)。點很集中時比仿射穩定。
    回傳 (R 2x2, t, 殘差 (N,2))。"""
    P, C = np.asarray(P, float), np.asarray(C, float)
    pm, cm = P.mean(0), C.mean(0)
    U, _, Vt = np.linalg.svd((P - pm).T @ (C - cm))
    R = (U @ Vt).T
    if not allow_mirror and np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    t = cm - R @ pm
    return R, t, C - (P @ R.T + t)


def match_pattern(blobs_uv, center_cmd, offsets_cmd, H_rough):
    """把偵測到的墨點 (像素) 對應到指令點。點的排列 = 中心 + offsets_cmd (手臂 mm)。
    先找「到其他點距離總和最小」的當中心,再依粗略換算預測的方向分配其餘點 (粗略換算只用來定方向,不用絕對位置)。
    回傳 [uv 或 None] (順序同 [center] + offsets)。"""
    if len(blobs_uv) == 0:
        return [None] * (1 + len(offsets_cmd))
    B = np.asarray(blobs_uv, float)
    d = np.linalg.norm(B[:, None] - B[None], axis=2)
    ci = int(np.argmin(d.sum(1)))
    out = [B[ci].tolist()]
    Hinv = np.linalg.inv(H_rough)
    c_pix = apply_h(Hinv, [center_cmd])[0]
    used = {ci}
    for off in offsets_cmd:
        exp = apply_h(Hinv, [np.asarray(center_cmd) + off])[0] - c_pix      # 這個點相對中心在影像中的預期向量
        best, bk = None, None
        for k in range(len(B)):
            if k in used:
                continue
            v = B[k] - B[ci]
            cosang = v @ exp / (np.linalg.norm(v) * np.linalg.norm(exp) + 1e-9)
            ratio = np.linalg.norm(v) / (np.linalg.norm(exp) + 1e-9)
            if cosang > np.cos(np.radians(30)) and 0.5 < ratio < 2.0:
                score = cosang - abs(np.log(ratio))
                if best is None or score > best:
                    best, bk = score, k
        if bk is None:
            out.append(None)
        else:
            used.add(bk)
            out.append(B[bk].tolist())
    return out


def fit_plane(xy, z):
    X = np.hstack([np.asarray(xy, float), np.ones((len(xy), 1))])
    (a, b, c), *_ = np.linalg.lstsq(X, np.asarray(z, float), rcond=None)
    res = np.asarray(z, float) - X @ [a, b, c]
    tilt = float(np.degrees(np.arctan(np.hypot(a, b))))
    return {"a": float(a), "b": float(b), "c": float(c), "tilt_deg": tilt, "residual_mm": res.tolist()}


# ---------- whole analysis ----------
def analyze(frame, pf, points, H_rough, stroke_mm=12.0, roi_mm=30.0, mark="plus", before=None, offset=(0.0, 0.0)):
    """points: [{"robot_xy":[x,y], "contact_z":z}, ...]。mark = "plus" (畫十字) 或 "dot" (點一下)。
    before: 點之前在 home 拍的影像 (dot 模式建議提供,用來排除紙上原有的髒點)。
    offset: 已知的「指令 - 實際落點」偏移 (手臂 mm),只用來預測標記在影像中的位置。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not pf.detect(frame):
        return {"ok": False, "reason": pf.describe()}
    excl = tag_exclusion_mask(frame.shape, pf)
    new_ink = None
    if before is not None:
        pfb = PaperFrame(layout=pf.layout_path, K=pf.K)
        if pfb.detect(before):
            new_ink = new_ink_mask(before, frame, pf, pfb)
            pf.detect(frame)
    Hinv = np.linalg.inv(H_rough)
    mmpp = pf.mm_per_pixel()
    radius = int(roi_mm / mmpp)
    det = []
    for k, pt in enumerate(points):
        x, y = np.asarray(pt["robot_xy"], float) - np.asarray(offset, float)
        c0, cx, cy = apply_h(Hinv, [[x, y], [x + 10, y], [x, y + 10]])
        if mark == "dot":
            r = find_dot(gray, c0, radius, excl, new_ink)
        else:
            da, db = (cx - c0) / np.linalg.norm(cx - c0), (cy - c0) / np.linalg.norm(cy - c0)
            r = find_cross(gray, c0, da, db, radius, excl)
        r["index"], r["predicted_uv"] = k, c0.tolist()
        if r["ok"]:
            r["paper_mm"] = pf.pix_to_paper([r["uv"]])[0].tolist()
        det.append(r)
    good = [d for d in det if d["ok"]]
    out = {"ok": False, "detections": det, "mm_per_px": mmpp, "paper_frame": pf.describe()}
    if len(good) < 3:
        out["reason"] = f"只找到 {len(good)} 個標記,至少要 3 個"
        return out
    P = [d["paper_mm"] for d in good]
    C = [points[d["index"]]["robot_xy"] for d in good]
    A, t, res = fit_affine(P, C)
    err = np.linalg.norm(res, axis=1)
    out.update({"ok": True, "A": A.tolist(), "t": t.tolist(), "affine": describe_affine(A),
                "residual_mm": err.tolist(), "rms_mm": float(np.sqrt((err ** 2).mean())),
                "max_mm": float(err.max()), "n_points": len(good)})
    zs = [(points[d["index"]]["robot_xy"], points[d["index"]].get("contact_z")) for d in good]
    zs = [(xy, z) for xy, z in zs if z is not None]
    if len(zs) >= 3:
        out["plane"] = fit_plane([a for a, _ in zs], [b for _, b in zs])
    return out


def overlay(frame, res):
    v = frame.copy()
    for d in res.get("detections", []):
        pu = tuple(int(round(x)) for x in d["predicted_uv"])
        cv2.circle(v, pu, 6, (0, 200, 255), 2)
        if d["ok"]:
            u = tuple(int(round(x)) for x in d["uv"])
            cv2.drawMarker(v, u, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(v, f"#{d['index']}", (u[0] + 10, u[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(v, f"#{d['index']} ?", (pu[0] + 10, pu[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    if res.get("ok"):
        cv2.putText(v, f"RMS {res['rms_mm']:.2f} mm  max {res['max_mm']:.2f} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return v


def summarize(res):
    lines = [res.get("paper_frame", "")]
    for d in res.get("detections", []):
        lines.append(f"  標記 #{d['index']}: " + (f"像素 ({d['uv'][0]:.1f},{d['uv'][1]:.1f}) -> 紙面 "
                     f"({d['paper_mm'][0]:.1f},{d['paper_mm'][1]:.1f}) mm" if d["ok"] else "找不到 - " + d["reason"]))
    if not res.get("ok"):
        lines.append("分析失敗: " + res.get("reason", ""))
        return "\n".join(lines)
    a = res["affine"]
    lines.append(f"紙面 -> 手臂: 旋轉 {a['rotation_deg']:.1f}°、比例 {a['scale_x']:.4f}/{a['scale_y']:.4f}、"
                 f"兩軸夾角 {a['axis_angle_deg']:.1f}°、鏡像 {a['mirrored']}")
    lines.append(f"擬合殘差 RMS {res['rms_mm']:.2f} mm、最大 {res['max_mm']:.2f} mm ({res['n_points']} 點;"
                 f"3 點時殘差必為 0,看不出誤差)")
    if "plane" in res:
        p = res["plane"]
        lines.append(f"接觸高度平面: 傾斜 {p['tilt_deg']:.2f}°,殘差 {', '.join(f'{r:+.2f}' for r in p['residual_mm'])} mm")
    return "\n".join(lines)


def main():
    from run_log import CalibrationStore, load_run
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="pen_calibrate.py 產生的 runs\\... 資料夾")
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args()
    meta, _ = load_run(args.run)
    with open(os.path.join(args.run, "data", "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    rd = lambda n: cv2.imdecode(np.fromfile(os.path.join(args.run, "images", n), np.uint8), cv2.IMREAD_COLOR)  # noqa
    frame, before = rd("home_after.png"), rd("home_before.png")
    pf = PaperFrame(layout=os.path.join(args.run, "inputs", os.path.basename(plan["layout"])))
    res = analyze(frame, pf, plan["points"], np.array(plan["H_rough"]), plan.get("stroke_mm", 12.0),
                  mark=plan.get("mark", "plus"), before=before, offset=plan.get("predict_offset", (0.0, 0.0)))
    print(summarize(res))
    cv2.imencode(".jpg", overlay(frame, res))[1].tofile(os.path.join(args.run, "images", "marks_overlay.jpg"))
    with open(os.path.join(args.run, "data", "marks_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("疊圖:", os.path.join(args.run, "images", "marks_overlay.jpg"))
    if res.get("ok") and not args.no_store:
        p = CalibrationStore().save("pen_paper", {
            "A": res["A"], "t": res["t"], "plane": res.get("plane"), "rms_mm": res["rms_mm"],
            "home": plan["home"], "orientation": plan["orientation"], "press_mm": plan["press_mm"],
            "layout": plan["layout"]}, source_run=meta["run_id"],
            notes="紙面 mm -> TCP 指令 XY: C = A·P + t (筆夾在夾爪,方向固定為 orientation)")
        print("已存校正版本:", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
