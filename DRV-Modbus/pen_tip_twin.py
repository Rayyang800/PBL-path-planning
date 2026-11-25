# -*- coding: utf-8 -*-
"""
筆尖 3D 位置 (工具座標) 不用傾斜碰點,改用「數位分身」的算法從現有資料推出來 (不動手臂)。

方法 (同 Delta_Arm_IsaacSim/real2sim/twin.py,11-12 驗證過):home 時相機在手臂座標的位置有兩種算法
  (1) 手眼:相機 = 法蘭(home) + R_home · p_cam          (scan3d 的 data/hand_eye.json)
  (2) 紙面:相機 = 紙面->手臂(PnP 相機中心)              (筆尖校正 A、t、z_touch + home 照片的 tag PnP)
筆尖校正量的是「筆尖」碰紙時的法蘭位置,所以 (1) - (2) = R_home · p_tip,直接解出筆尖在工具座標的位置 p_tip。

只要「筆尖校正 (護貝紙)」是現在這次夾筆做的、相機沒拆過,就不用再做 pen_pivot.py 的傾斜碰點。

11-21 用 11-18 的資料算 (5 次 run x 2 次手眼):
  - 筆尖水平位置 (x, y) 各次一致 ±0.6 mm
  - 筆長 (法蘭下方) 依手眼不同:235.5 (11-11 手眼) / 246.8 (11-18 手眼),差 11 mm
    -> 用尺量「法蘭面到筆尖」的垂直距離 (--pen-len) 決定;沒量時取兩者平均,誤差當 ±6 mm
  - 誤差對傾斜描繪的影響:筆長誤差 d -> 筆尖側向偏 d·sin(傾角)。±6 mm、傾斜 15° -> ±1.6 mm
    (高度不受影響太多:d·(1-cos) < 0.3 mm,而且下筆還是人工確認接觸)

  python pen_tip_twin.py                 # 算、比較各手眼,不存
  python pen_tip_twin.py --save          # 存成 calibrations/pen_tip (只供查看;circle_trace 傾斜時自己即時算,不讀這份)
  python pen_tip_twin.py --pen-len 238 --save    # 用尺量的筆長 (法蘭面 -> 筆尖,mm) 取代手眼的 z
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from run_log import CalibrationStore

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def jload(*p):
    with open(os.path.join(*p), encoding="utf-8") as f:
        return json.load(f)


def R_xyz(a):
    return Rotation.from_euler("xyz", a, degrees=True).as_matrix()      # = Rz Ry Rx (手眼 conv 'xyz')


def estimate(home_pnp, tcp, cal, hand_eyes_list, pen_len=None):
    """這次的 home 照片 PnP + 護貝紙校正 -> 筆尖 p (法蘭座標)。circle_trace 傾斜時自動呼叫,不用先存檔。
    同一天的手眼先平均,再讓各天等權 (11-11 有兩次,不該算兩票)。
    回傳 dict(p, z_unc, z_source, by_day)。"""
    A, t, z_touch = np.array(cal["A"], float), np.array(cal["t"], float), float(cal["z_touch"])
    T, R = np.array(tcp[:3], float), R_xyz(tcp[3:])
    s = float(np.mean(np.linalg.svd(A, compute_uv=False)))
    C = np.asarray(home_pnp["C"], float)
    cam_from_paper = np.array([*(A @ C[:2] + t), z_touch + s * (-C[2])])
    days = {}
    for he in hand_eyes_list:
        p = R.T @ (T + R @ np.array(he["p_cam_in_tool_mm"], float) - cam_from_paper)
        days.setdefault(he["run"][:8], []).append(p)
    if not days:
        raise ValueError("沒有手眼資料")
    by_day = np.array([np.mean(v, axis=0) for v in days.values()])
    p = by_day.mean(0)
    z_unc, src = max(2.0, float(np.ptp(by_day[:, 2]) / 2)) if len(by_day) > 1 else 6.0, "手眼平均"
    if pen_len:
        # 11-25:分身推的筆長 (手眼的相機高度定不準) 比實際短約 40 mm;只提醒,以尺量為準
        if abs(pen_len - p[2]) > 20:
            print(f"  注意:尺量筆長 {pen_len:g} 和分身推算的 {p[2]:.1f} mm 差 {pen_len - p[2]:+.0f} mm,以尺量為準 (分身的筆長不可靠)")
        p[2], z_unc, src = float(pen_len), 2.0, "尺量"
    if not (30 < np.hypot(p[0], p[1]) < 100 and 180 < p[2] < 310):
        raise ValueError(f"推算的筆尖位置不合理 {np.round(p, 1).tolist()},請重做護貝紙校正")
    return {"p": p, "z_unc": float(z_unc), "z_source": src,
            "by_day": {k: np.mean(v, axis=0).tolist() for k, v in days.items()}}


def runs_with_home(created):
    """有 images/home.png、且 inputs/pen_paper.json 是指定版本的 run (新 -> 舊)。"""
    out = []
    for r in sorted(glob.glob(os.path.join(RUNS, "*")), reverse=True):
        try:
            if os.path.isfile(os.path.join(r, "images", "home.png")) and \
                    jload(r, "inputs", "pen_paper.json").get("created") == created:
                out.append(r)
        except (OSError, ValueError):
            continue
    return out


def hand_eyes():
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*_scan3d", "data", "hand_eye.json"))):
        he = jload(p)
        pc = np.array(he.get("p_cam_in_tool_mm", [np.inf] * 3), float)
        if he.get("conv") != "xyz" or not np.all(np.isfinite(pc)) or np.linalg.norm(pc) > 500:
            continue                                     # 失敗的手眼 (例如 11-11 11:33 那次 p 是幾十萬 mm)
        out.append((os.path.basename(os.path.dirname(os.path.dirname(p))), he))
    return out


def tip_from(run, he, cal):
    import scan3d
    meta = jload(run, "meta.json")
    layout = jload(run, "inputs", "lab_sheet_layout.json")
    ci = meta["camera"]["color_intrinsics"]
    K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]], float)
    img = cv2.imdecode(np.fromfile(os.path.join(run, "images", "home.png"), np.uint8), cv2.IMREAD_COLOR)
    pose = scan3d.tag_pose(img, layout, K)
    if pose is None or pose["n_tags"] < 4:
        return None
    A, t, z_touch = np.array(cal["A"], float), np.array(cal["t"], float), float(cal["z_touch"])
    home = np.array(cal["home"], float)
    R_home = R_xyz(home[3:])
    s = float(np.mean(np.linalg.svd(A, compute_uv=False)))           # 手臂 mm / 紙面 mm (約 1.022)
    C = pose["C"]                                                      # 紙面座標,z<0 在紙上方
    cam_from_paper = np.array([*(A @ C[:2] + t), z_touch + s * (-C[2])])
    cam_world = home[:3] + R_home @ np.array(he["p_cam_in_tool_mm"], float)
    return R_home.T @ (cam_world - cam_from_paper), pose["rms_px"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="用哪次 run 的 home 照片 (預設:用目前筆尖校正版本的所有 run 平均)")
    ap.add_argument("--handeye", help="用哪次 scan3d 的手眼 (預設:全部可用的手眼平均,並列出差異)")
    ap.add_argument("--pen-len", type=float, help="用尺量:法蘭面到筆尖的距離 (mm,筆垂直時),取代手眼推出的 z")
    ap.add_argument("--save", action="store_true", help="存成 calibrations/pen_tip")
    args = ap.parse_args()

    calrec = CalibrationStore().load_latest("pen_paper")
    cal, created = calrec["data"], calrec.get("created")
    runs = [os.path.join(RUNS, args.run)] if args.run else runs_with_home(created)[:6]
    hes = [(h, he) for h, he in hand_eyes() if not args.handeye or h == args.handeye]
    if not runs:
        sys.exit(f"找不到用目前筆尖校正 ({created}) 拍過 home 照片的 run;先跑一次任何任務的 dry-run (會拍 home)")
    if not hes:
        sys.exit("找不到可用的手眼 (scan3d 的 data/hand_eye.json)")
    print(f"筆尖校正 (護貝紙):{created},留一法最大 {cal.get('loo_max_mm', float('nan')):.2f} mm")

    table = {}
    for h, he in hes:
        ps = []
        for r in runs:
            res = tip_from(r, he, cal)
            if res is not None:
                ps.append(res[0])
        if ps:
            P = np.array(ps)
            table[h] = P
            m = P.mean(0)
            print(f"  手眼 {h} (殘差 {he.get('pos_rms_mm', float('nan')):.2f} mm):筆尖 {np.round(m, 1).tolist()} mm,"
                  f"離 J6 {np.hypot(m[0], m[1]):.1f}、法蘭下 {m[2]:.1f};{len(ps)} 張 home 照片間差 ±{np.ptp(P, 0).max() / 2:.1f} mm")
    if not table:
        sys.exit("沒有任何 home 照片看到 4 個 tag")
    means = np.array([P.mean(0) for P in table.values()])
    p = means.mean(0)
    xy_spread = float(np.ptp(means[:, :2], 0).max() / 2) if len(means) > 1 else 0.6
    z_spread = float(np.ptp(means[:, 2]) / 2) if len(means) > 1 else 6.0
    if args.pen_len is not None:
        if abs(args.pen_len - p[2]) > max(15.0, 2 * z_spread + 5):
            print(f"  注意:尺量筆長 {args.pen_len:g} 和手眼推的 {p[2]:.1f} 差很多,請再量一次 (法蘭面 -> 筆尖,筆垂直)")
        p[2] = args.pen_len
        z_unc, z_src = 2.0, "尺量"
    else:
        z_unc, z_src = max(z_spread, 2.0), "手眼平均"
    print(f"\n採用筆尖 p = {np.round(p, 1).tolist()} mm (水平 ±{max(xy_spread, 0.6):.1f}、筆長 ±{z_unc:.1f},筆長來源:{z_src})")
    for deg in (10, 15, 20):
        e = np.hypot(max(xy_spread, 0.6) * 2 * np.sin(np.radians(deg) / 2), z_unc * np.sin(np.radians(deg)))
        print(f"  傾斜 {deg:2d}° 時筆尖位置誤差約 ±{e:.1f} mm")
    if args.pen_len is None and z_spread > 3:
        print("  建議:用尺量法蘭面到筆尖的距離,加 --pen-len 重算 (兩次手眼的筆長差 "
              f"{2 * z_spread:.0f} mm,用尺可以分辨)")
    if not args.save:
        print("\n(沒有存;確認後加 --save)")
        return
    data = {"p_tool_mm": p.tolist(), "conv": "xyz", "method": "twin",
            "rms_mm": float(np.hypot(max(xy_spread, 0.6), z_unc)),
            "uncertainty_mm": {"xy": max(xy_spread, 0.6), "z": z_unc, "z_source": z_src},
            "pen_paper_created": created, "home_runs": [os.path.basename(r) for r in runs],
            "hand_eyes": {h: np.round(P.mean(0), 2).tolist() for h, P in table.items()}}
    path = CalibrationStore().save("pen_tip", data, notes="數位分身算法 (手眼 + 筆尖校正 + home PnP),不需傾斜碰點")
    print("已存:", path)


if __name__ == "__main__":
    main()
