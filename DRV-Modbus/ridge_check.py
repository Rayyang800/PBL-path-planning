# -*- coding: utf-8 -*-
"""
脊線描繪的回饋分析:比較描繪前 / 後的 home 照片,找出新畫上的筆跡,量它離規劃脊線多遠。

做法
  1. 兩張照片各自用校正紙 tag 求相機位姿 (手臂回 home 的位置可能差一點點,所以各算各的)
  2. 規劃的脊線 (紙面座標 x, y, 高度 h) 投影到照片上,只看它附近 --corridor mm 的範圍 (校正紙上的紅十字不會被算進來)
  3. 新筆跡 = 描繪後「變紅」或「變暗」的像素 (紅色 = R 比 G、B 多;兩張相減,只留增加的)
  4. 筆跡像素反投影回紙面:射線和「脊線高度 + 接觸修正」的水平面求交點 -> 紙面 xy
  5. 沿脊線每 bin_mm 一段:取該段筆跡的橫向位置中位數 -> 偏差;有筆跡的段數比例 -> 涵蓋率

  python ridge_check.py runs\\<ridge_trace 紀錄>        # 離線分析某次描繪 (不需要手臂 / 相機)
  python ridge_check.py --latest                        # 分析最新一次 ridge_trace
結果存在該紀錄的 data/trace_check.json 與 images/trace_check.jpg (原紀錄只新增檔案,不改原本的)。
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def redness(img):
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    return r - 0.5 * (g + b)


def project(pose, K, P_xyh):
    P = np.c_[P_xyh[:, 0], P_xyh[:, 1], -P_xyh[:, 2]].astype(np.float64)
    rv, _ = cv2.Rodrigues(pose["R"])
    pix, _ = cv2.projectPoints(P, rv, pose["t"], K, np.zeros(5))
    return pix.reshape(-1, 2)


def backproject_to_height(pose, K, uv, h):
    """像素 (N,2) -> 紙面 xy (N,2):相機射線和 z = -h 的水平面 (紙面座標 z 朝下) 的交點。h 可為 (N,) 陣列。"""
    R, t = pose["R"], pose["t"]
    C = -R.T @ t
    d = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] @ R    # R^T · ray
    lam = (-np.asarray(h, float) - C[2]) / d[:, 2]
    return (C[None, :2] + lam[:, None] * d[:, :2])


def analyze(before, after, layout, K, line_xyh, dz=0.0, corridor=12.0, bin_mm=2.0, min_red=8.0, min_dark=12.0):
    """回傳 (result dict, overlay 影像)。line_xyh:規劃脊線兩端 [[x,y,h],[x,y,h]] (紙面 mm)。dz:接觸修正 (實際尖端比量到的高多少)。"""
    import scan3d
    pb, pa = scan3d.tag_pose(before, layout, K), scan3d.tag_pose(after, layout, K)
    if pb is None or pa is None:
        return {"ok": False, "reason": "前或後的照片看不到校正紙 tag"}, after.copy()
    L = np.asarray(line_xyh, float)
    a, b = L[0], L[1]
    u = (b[:2] - a[:2]) / np.linalg.norm(b[:2] - a[:2])
    n = np.array([-u[1], u[0]])
    length = float(np.linalg.norm(b[:2] - a[:2]))

    # 走廊遮罩 (在「描繪後」照片上)
    def corridor_poly(pose, half):
        ext = 6.0
        pts = []
        for s_, w_ in ((-ext, -half), (length + ext, -half), (length + ext, half), (-ext, half)):
            xy = a[:2] + s_ * u + w_ * n
            hh = a[2] + (b[2] - a[2]) * np.clip(s_ / length, 0, 1) + dz
            pts.append([xy[0], xy[1], hh])
        return project(pose, K, np.array(pts))

    mask = np.zeros(after.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.int32(corridor_poly(pa, corridor)), 1)
    # 前後對齊:用兩次的 tag 位姿把「前」照片換到「後」的視角 (只在走廊附近,近似平面單應)
    src = corridor_poly(pb, corridor + 10)
    dst = corridor_poly(pa, corridor + 10)
    Hm, _ = cv2.findHomography(np.float32(src), np.float32(dst))
    before_w = cv2.warpPerspective(before, Hm, (after.shape[1], after.shape[0]))
    d_red = redness(after) - redness(before_w)
    d_dark = cv2.cvtColor(before_w, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 門檻跟著雜訊走:筆跡很淡 (筆輕輕碰到) 時固定門檻會漏掉 (11-18 第 1 次:變紅只有 ~20)
    inm = mask > 0

    def thr(x, floor):
        v_ = x[inm]
        med = float(np.median(v_))
        sig = 1.4826 * float(np.median(np.abs(v_ - med)))
        return max(floor, med + 6.0 * sig)
    t_red, t_dark = thr(d_red, min_red), thr(d_dark, min_dark)
    ink = ((d_red > t_red) | ((d_dark > t_dark) & (d_red > 0.5 * t_red))) & inm
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    v, uu = np.nonzero(ink)
    res = {"ok": False, "ink_pixels": int(len(v)), "plan_length_mm": length, "dz_mm": dz, "corridor_mm": corridor,
           "thr_red": t_red, "thr_dark": t_dark}
    view = after.copy()
    plan_pix = project(pa, K, np.array([[*a[:2], a[2] + dz], [*b[:2], b[2] + dz]]))
    if len(v) < 30:
        res["reason"] = "走廊內幾乎沒有新筆跡 (筆沒碰到、或顏色和工件太像)"
        cv2.line(view, tuple(np.int32(plan_pix[0])), tuple(np.int32(plan_pix[1])), (255, 0, 255), 1)
        return res, view
    # 反投影到「脊線高度 + dz」:先用規劃線上的高度 (依沿線位置內插)
    xy0 = backproject_to_height(pa, K, np.c_[uu, v].astype(float), a[2] + dz)
    s0 = (xy0 - a[:2]) @ u
    h_at = a[2] + (b[2] - a[2]) * np.clip(s0 / length, 0, 1) + dz
    xy = backproject_to_height(pa, K, np.c_[uu, v].astype(float), h_at)
    s = (xy - a[:2]) @ u
    t = (xy - a[:2]) @ n
    bins = np.arange(0, length + 1e-6, bin_mm)
    rows = []
    for s_lo in bins[:-1]:
        m = (s >= s_lo) & (s < s_lo + bin_mm)
        if m.sum() >= 3:
            rows.append([s_lo + bin_mm / 2, float(np.median(t[m])), int(m.sum())])
    n_bins = len(bins) - 1
    res["coverage"] = len(rows) / max(1, n_bins)
    if len(rows) < 3:
        res["reason"] = "筆跡太少,無法算偏差"
        return res, view
    R_ = np.array(rows)
    dev = R_[:, 1]
    gaps, run, prev = [], 0, None
    covered = set(np.round((R_[:, 0] - bin_mm / 2) / bin_mm).astype(int))
    for k in range(n_bins):
        if k not in covered:
            run += 1
        elif run:
            gaps.append(run * bin_mm)
            run = 0
    if run:
        gaps.append(run * bin_mm)
    res.update(ok=True, mean_abs_dev_mm=float(np.mean(np.abs(dev))), p90_abs_dev_mm=float(np.percentile(np.abs(dev), 90)),
               max_abs_dev_mm=float(np.max(np.abs(dev))), bias_mm=float(np.mean(dev)),
               dev_std_mm=float(np.std(dev)), ink_from_mm=float(R_[0, 0] - bin_mm / 2), ink_to_mm=float(R_[-1, 0] + bin_mm / 2),
               longest_gap_mm=float(max(gaps) if gaps else 0.0), profile=[[round(r[0], 1), round(r[1], 2), r[2]] for r in rows])
    # 畫圖:規劃線 (洋紅)、偵測到的筆跡 (綠)、每段中位數 (黃點)
    view[ink] = (0, 255, 0)
    cv2.line(view, tuple(np.int32(plan_pix[0])), tuple(np.int32(plan_pix[1])), (255, 0, 255), 1)
    med = np.array([[*(a[:2] + r[0] * u + r[1] * n), a[2] + (b[2] - a[2]) * r[0] / length + dz] for r in rows])
    for q in project(pa, K, med):
        cv2.circle(view, tuple(np.int32(q)), 2, (0, 230, 255), -1)
    txt = (f"mean {res['mean_abs_dev_mm']:.2f} mm  p90 {res['p90_abs_dev_mm']:.2f}  bias {res['bias_mm']:+.2f}  "
           f"coverage {res['coverage'] * 100:.0f}%")
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(view, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return res, view


def crop_around(img, pix, pad=90):
    x0, y0 = np.int32(np.maximum(pix.min(0) - pad, 0))
    x1, y1 = np.int32(np.minimum(pix.max(0) + pad, [img.shape[1], img.shape[0]]))
    return img[y0:y1, x0:x1]


def summarize(r):
    if not r.get("ok"):
        return f"筆跡分析:無法量測 ({r.get('reason', '?')})"
    return (f"筆跡分析:離規劃脊線 平均 {r['mean_abs_dev_mm']:.2f} mm、90% 在 {r['p90_abs_dev_mm']:.2f} mm 內、最大 {r['max_abs_dev_mm']:.2f} mm;"
            f"整體偏 {r['bias_mm']:+.2f} mm (正 = 往規劃線左側);涵蓋 {r['coverage'] * 100:.0f}%、"
            f"最長斷線 {r['longest_gap_mm']:.0f} mm")


def check_pairs(pairs, layout, K, line_xyh, dz, log=None):
    """pairs = [(before, after, name)],依序試 (正上方優先),取第一個量得到的。結果與圖存進 log。"""
    import scan3d
    best, all_res = None, {}
    for before, after, name in pairs:
        r, view = analyze(before, after, layout, K, line_xyh, dz=dz)
        r["view"] = name
        all_res[name] = {k: v for k, v in r.items() if k != "profile"}
        if log is not None:
            pose = scan3d.tag_pose(after, layout, K)
            if pose is not None:
                pix = project(pose, K, np.array([[*p[:2], p[2] + dz] for p in line_xyh]))
                log.save_image(f"trace_check_{name}.jpg", crop_around(view, pix))
        if best is None and r.get("ok"):
            best = r
    best = best or (dict(next(iter(all_res.values())), view=None) if all_res else None)
    if log is not None and best is not None:
        log.save_json("trace_check", {"best": best, "all": all_res})
        log.event("trace_check", **{k: v for k, v in best.items() if k != "profile"})
    return best


def analyze_run(run, save=True):
    """離線分析一次 ridge_trace 紀錄。回傳結果 dict。"""
    from run_log import load_snapshot
    with open(os.path.join(run, "data", "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    dz = 0.0
    with open(os.path.join(run, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "contact":
                dz = float(e.get("dz", 0.0))
    with open(os.path.join(run, "inputs", "lab_sheet_layout.json"), encoding="utf-8") as f:
        layout = json.load(f)
    import scan3d
    from run_log import list_snapshots
    names = list_snapshots(run)
    pair = ("c", "c_after") if "c_after" in names and "c" in names else ("home", "home_after")
    if pair[1] not in names:
        raise SystemExit("這次紀錄沒有描繪後的照片")
    before, _, dmeta = load_snapshot(run, pair[0])
    after, _, _ = load_snapshot(run, pair[1])
    K = scan3d.K_of(dmeta["intrinsics"])
    line = plan.get("line_draw_xyh") or plan["ridge"]["line_xyh"]
    res, view = analyze(before, after, layout, K, line, dz=dz)
    res["run"] = os.path.basename(run.rstrip("\\/"))
    res["view"] = pair[0]
    if save:
        with open(os.path.join(run, "data", "trace_check.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        pose = scan3d.tag_pose(after, layout, K)
        if pose is not None:
            pix = project(pose, K, np.array([[*p[:2], p[2] + dz] for p in line]))
            cv2.imencode(".jpg", crop_around(view, pix))[1].tofile(os.path.join(run, "images", "trace_check.jpg"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="runs\\<ridge_trace 紀錄>")
    ap.add_argument("--latest", action="store_true", help="分析最新一次 ridge_trace")
    args = ap.parse_args()
    run = args.run
    if args.latest or not run:
        cands = sorted(glob.glob(os.path.join(HERE, "runs", "*_ridge_trace*")))
        cands = [c for c in cands if os.path.isfile(os.path.join(c, "images", "home_after.png"))]
        if not cands:
            sys.exit("找不到有描繪後照片的 ridge_trace 紀錄")
        run = cands[-1]
    r = analyze_run(run)
    print(run)
    print(summarize(r))
    if r.get("ok"):
        print("圖:", os.path.join(run, "images", "trace_check.jpg"))


if __name__ == "__main__":
    main()
