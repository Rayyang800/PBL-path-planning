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
