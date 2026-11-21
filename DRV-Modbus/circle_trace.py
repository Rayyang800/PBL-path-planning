# -*- coding: utf-8 -*-
"""
沿圓形工件頂面外圈描繪 (工件 C:鮪魚罐頭頂面的捲邊)。筆垂直、不傾斜。

流程 (--go)
  0. home 拍照 -> 找物體 -> 粗估圓
  1. 快速掃描 (--views,預設 1):相機在罐頭正上方 380 mm 拍深度 (朝下、不傾斜)
  2. 頂部點的外緣做圓擬合 -> 圓心、外緣半徑;捲邊高度 = 外緣往內 1~6 mm 的高度 90 百分位 x 深度增益
     描繪半徑 = 外緣半徑 - --inset (預設 2.5 mm,讓筆落在捲邊上)
  3. 移到起點上方安全高度 -> 降到「估計捲邊高度 + approach」-> 人工每步 0.5 mm 往下,筆尖剛好碰到按 c
  4. 沿圓周每 --step-mm 一點 MovL (圓是曲線,台達 GO 一次只能一個目標,所以是多段直線;--step-mm 4 時弦高 < 0.1 mm)
  5. 抬起 -> 回正上方拍照 -> 量筆跡離規劃圓的徑向偏差 / 涵蓋率 (data/circle_check.json、images/circle_check.jpg) -> 回 home

傾斜描繪 (--tilt-deg,例如 20):筆往圓心外倒 (筆身在外、筆尖朝內下壓),每一點姿態 = 垂直姿態繞該點圓切線轉 tilt。
  需要先做 pen_pivot.py (筆尖 3D 校正);TCP = 垂直時的 TCP + (R0 - R)·p。換姿態都在高處原地轉。
  手臂走不到某個傾斜姿態 (關節極限 / 奇異點) 就停止、抬起 (傾斜時抬到 home 高度) 轉回垂直、回 home。

  python circle_trace.py                 # dry-run:只用 home 一張找圓、出計畫圖 (circle_trace_plan.png),不動手臂
  python circle_trace.py --go            # 執行
按鍵:Ctrl+C = 停止 -> 抬起回 home;移動中 空白/Esc/p = 選單 (r 繼續 / q 結束)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import circle_rim
import pen_offset
import ridge_check
import scan3d
from ridge_trace import height_gain, quick_targets
from run_log import RunLog, CalibrationStore

HERE = os.path.dirname(os.path.abspath(__file__))


def draw_plan(color, pose, K, rim, path, r_draw):
    img = color.copy()
    for rad, col in ((rim["r_outer"], (0, 255, 0)), (r_draw, (255, 0, 255))):
        P = circle_rim.circle_path(rim["center"], rad, path[0][2], 2.0, overlap_deg=0)
        cv2.polylines(img, [np.int32(ridge_check.project(pose, K, P))], True, col, 1)
    s0 = ridge_check.project(pose, K, path[:1])[0]
    cv2.circle(img, tuple(int(v) for v in s0), 8, (0, 255, 0), 2)
    cv2.putText(img, "start", (int(s0[0]) + 10, int(s0[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def analyze_circle(before, after, layout, K, center, r, h, corridor=8.0, bin_deg=5.0):
    """筆跡離規劃圓的徑向偏差 (正 = 往外)。回傳 (dict, overlay)。"""
    pb, pa = scan3d.tag_pose(before, layout, K), scan3d.tag_pose(after, layout, K)
    if pb is None or pa is None:
        return {"ok": False, "reason": "前或後的照片看不到校正紙 tag"}, after.copy()

    def ring(pose, rr):
        return ridge_check.project(pose, K, circle_rim.circle_path(center, rr, h, 2.0, overlap_deg=0))

    mask = np.zeros(after.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(ring(pa, r + corridor))], 1)
    cv2.fillPoly(mask, [np.int32(ring(pa, max(1.0, r - corridor)))], 0)
    src = np.vstack([ring(pb, r + corridor + 8)[::6], ring(pb, max(1.0, r - corridor - 8))[::6]])
    dst = np.vstack([ring(pa, r + corridor + 8)[::6], ring(pa, max(1.0, r - corridor - 8))[::6]])
    Hm, _ = cv2.findHomography(np.float32(src), np.float32(dst))
    bw = cv2.warpPerspective(before, Hm, (after.shape[1], after.shape[0]))
    d_red = ridge_check.redness(after) - ridge_check.redness(bw)
    d_dark = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inm = mask > 0

    def thr(x, floor):
        v_ = x[inm]
        med = float(np.median(v_))
        return max(floor, med + 6.0 * 1.4826 * float(np.median(np.abs(v_ - med))))
    t_red, t_dark = thr(d_red, 8.0), thr(d_dark, 12.0)
    ink = ((d_red > t_red) | ((d_dark > t_dark) & (d_red > 0.5 * t_red))) & inm
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    v, u = np.nonzero(ink)
    res = {"ok": False, "ink_pixels": int(len(v)), "r_plan_mm": r, "corridor_mm": corridor, "thr_red": t_red}
    view = after.copy()
    cv2.polylines(view, [np.int32(ring(pa, r))], True, (255, 0, 255), 1)
    if len(v) < 30:
        res["reason"] = "走廊內幾乎沒有新筆跡"
        return res, view
    xy = ridge_check.backproject_to_height(pa, K, np.c_[u, v].astype(float), h)
    rel = xy - np.asarray(center)
    ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    dr = np.hypot(rel[:, 0], rel[:, 1]) - r
    rows = []
    for a0 in np.arange(-180, 180, bin_deg):
        m = (ang >= a0) & (ang < a0 + bin_deg)
        if m.sum() >= 3:
            rows.append([a0 + bin_deg / 2, float(np.median(dr[m])), int(m.sum())])
    n_bins = int(360 / bin_deg)
    res["coverage"] = len(rows) / n_bins
    if len(rows) < 3:
        res["reason"] = "筆跡太少"
        return res, view
    R_ = np.array(rows)
    dev = R_[:, 1]
    # 筆跡圓 (中位數點) 擬合:看是圓心偏還是半徑偏
    pts = np.c_[center[0] + (r + dev) * np.cos(np.radians(R_[:, 0])), center[1] + (r + dev) * np.sin(np.radians(R_[:, 0]))]
    cx, cy, rr = circle_rim.fit_circle(pts) if len(pts) >= 5 else (np.nan, np.nan, np.nan)
    res.update(ok=True, mean_abs_dev_mm=float(np.mean(np.abs(dev))), p90_abs_dev_mm=float(np.percentile(np.abs(dev), 90)),
               max_abs_dev_mm=float(np.max(np.abs(dev))), radial_bias_mm=float(np.mean(dev)),
               ink_circle={"center_shift_mm": [float(cx - center[0]), float(cy - center[1])], "r_mm": float(rr)},
               profile=[[round(a, 1), round(d, 2), n] for a, d, n in rows])
    view[ink] = (0, 255, 0)
    for q in ridge_check.project(pa, K, np.c_[pts, np.full(len(pts), h)]):
        cv2.circle(view, tuple(np.int32(q)), 2, (0, 230, 255), -1)
    return res, view


def pen_points(T, Rm, p_tip, body_mm=120.0, n=7):
    """TCP 位置 T、姿態 Rm -> 筆尖與筆身 (從筆尖往 TCP 方向 body_mm) 的取樣點 (手臂座標)。"""
    tip = np.asarray(T) + Rm @ p_tip
    u = -(Rm @ p_tip) / np.linalg.norm(p_tip)               # 筆尖 -> TCP 方向
    return np.array([tip + u * body_mm * k / (n - 1) for k in range(n)])


def rotation_margin(T, R_from, R_to, p_tip, R0, A, tcal, z_touch, center, r_obj, h_obj, steps=12,
                    clear_obj=15.0, clear_paper=10.0):
    """在 TCP = T 原地從 R_from 轉到 R_to (球面內插),筆尖 / 筆身離物體 (圓柱:圓心、半徑、高) 與紙面的最小餘裕 (mm)。
    負值 = 會撞。座標換算:垂直姿態時筆尖紙面座標 P、高度 h <-> 手臂 (A P + t + (R0 p)_xy, z_touch + h + (R0 p)_z)。"""
    from scipy.spatial.transform import Rotation, Slerp
    sl = Slerp([0, 1], Rotation.from_matrix([R_from, R_to]))
    off = R0 @ p_tip
    worst = np.inf
    for k in range(steps + 1):
        Rm = sl([k / steps]).as_matrix()[0]
        for q in pen_points(T, Rm, p_tip):
            h = q[2] - z_touch - off[2]
            P = np.linalg.solve(A, q[:2] - tcal - off[:2])
            inside = np.hypot(*(P - np.asarray(center))) < r_obj + 5.0
            worst = min(worst, (h - h_obj - clear_obj) if inside else (h - clear_paper))
    return worst


def housing_margin(lowest_tcp_z, paper_z, object_height, reaches):
    """包覆球以 TCP 為球心。全段 MovL 的最低 TCP 高度由兩端決定，與姿態無關。"""
    return lowest_tcp_z - max(reaches) - (paper_z + object_height + 15.0)


def safe_rot_height(T_xy, z0, R_from, R_to, z_max, **kw):
    """從 z0 開始每次抬 5 mm,直到這次轉動的餘裕 >= 0;超過 z_max 回傳 None。"""
    z = z0
    while z <= z_max:
        if rotation_margin(np.array([T_xy[0], T_xy[1], z]), R_from, R_to, **kw) >= 0:
            return z
        z += 5.0
    return None


def summarize_check(r):
    if not r.get("ok"):
        return f"筆跡分析:無法量測 ({r.get('reason', '?')})"
    ic = r["ink_circle"]
    return (f"筆跡分析:離規劃圓 平均 {r['mean_abs_dev_mm']:.2f} mm、90% 在 {r['p90_abs_dev_mm']:.2f} mm 內、最大 {r['max_abs_dev_mm']:.2f} mm;"
            f"徑向整體 {r['radial_bias_mm']:+.2f} mm (正 = 偏外);筆跡圓心偏 ({ic['center_shift_mm'][0]:+.2f}, {ic['center_shift_mm'][1]:+.2f}) mm;"
            f"涵蓋 {r['coverage'] * 100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--views", type=int, choices=[1, 3, 5], default=1, help="快速掃描幾個正上方視角")
    ap.add_argument("--cam-h", type=float, default=380.0)
    ap.add_argument("--radius", type=float, default=60.0, help="快速掃描:周圍視角離圓心 mm")
    ap.add_argument("--inset", type=float, default=2.5, help="描繪半徑 = 外緣半徑 - inset (mm)")
    ap.add_argument("--clearance", type=float, default=30.0)
    ap.add_argument("--approach", type=float, default=5.0)
    ap.add_argument("--height-gain", type=float, default=None)
    ap.add_argument("--press", type=float, default=0.0)
    ap.add_argument("--step-mm", type=float, default=8.0,
                    help="圓周上每幾 mm 一點 (8 mm 在半徑 37 mm 時弦高 0.2 mm;每段手臂要約 2 秒加減速,段數越少越快)")
    ap.add_argument("--no-color", action="store_true", help="不用彩色邊緣修正圓心 (只用深度)")
    ap.add_argument("--no-pen-offset", action="store_true", help="不套用筆跡回饋學到的筆尖偏移")
    ap.add_argument("--tilt-deg", type=float, default=0.0,
                    help="筆往圓心外傾斜幾度 (筆身往外倒、筆尖朝內下壓);0 = 垂直。需要先做 pen_pivot.py 筆尖 3D 校正")
    ap.add_argument("--gripper-reach", type=float, help="TCP 到夾爪最遠外緣的實測上界 (mm，含張開狀態)")
    ap.add_argument("--camera-reach", type=float, help="TCP 到相機與支架最遠外緣的實測上界 (mm)")
    ap.add_argument("--tilt-in", action="store_true", help="改成往圓心內傾斜 (筆身往內倒)")
    ap.add_argument("--start-deg", type=float, default=None, help="起點角度 (紙面座標;預設 = 離 home 最近的方向)")
    ap.add_argument("--cw", action="store_true", help="順時針 (預設逆時針)")
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--n-depth", type=int, default=10)
    ap.add_argument("--layout", default=scan3d.LAYOUT)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    args.min_range, args.max_travel, args.n_ring = 230.0, 150.0, 0
    if not 0.0 <= args.inset <= 8.0:
        sys.exit("--inset 需在 0~8 mm")

    if not np.isfinite(args.tilt_deg) or args.tilt_deg < 0:
        ap.error("傾角必須是有限的非負數")
    if args.tilt_deg > 0:
        import pen_pivot
        reaches = (args.gripper_reach, args.camera_reach)
        if any(r is None or not np.isfinite(r) or r <= 0 for r in reaches):
            ap.error("傾斜描繪需要 --gripper-reach 和 --camera-reach 的實測上界")
        tip = pen_pivot.load_tip()
        if tip is None:
            ap.error("傾斜描繪需要筆尖 3D 校正")
        if not np.isfinite(args.approach) or args.approach < 0:
            ap.error("接近高度必須是有限的非負數")
    from pymodbus.client import ModbusTcpClient
    import warp_pick as wp
    from drv_modbus import send, request
    from rs_capture import RSCamera
    from pen_calibrate import Arm, Paused, ask
    from pen_quick_calib import Quit, probe_with_nudge

    with open(args.layout, encoding="utf-8") as f:
        layout = json.load(f)
    W, H = layout["paper_mm"]
    try:
        cal = CalibrationStore().load_latest("pen_paper")
    except FileNotFoundError:
        sys.exit("還沒有筆尖校正 (calibrations/pen_paper)")
    A, tcal, z_touch = np.array(cal["data"]["A"]), np.array(cal["data"]["t"]), cal["data"].get("z_touch")
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
    K = scan3d.K_of(info["color_intrinsics"])
    snap, mm = scan3d.capture(cam, info, args.n_depth)
    hp = scan3d.tag_pose(snap["color"], layout, K)
    if hp is None or hp["n_tags"] < 4:
        cam.close()
        sys.exit("home 要看到校正紙 4 個 tag")
    home_view = {"name": "home", "color": snap["color"], "mm": mm, "K": K, "pose": hp, "pose_source": "tags"}
    xy0, h0, _, _, obj = scan3d.process_views([home_view], layout)
    rim0 = circle_rim.extract_rim(xy0, h0) if obj is not None else None
    if rim0 is None:
        cam.close()
        sys.exit("home 照片找不到圓形物體的頂面")
    print(f"[home 一張] 圓心 ({rim0['center'][0]:.1f}, {rim0['center'][1]:.1f})、外緣半徑 {rim0['r_outer']:.1f} mm、"
          f"捲邊高 {rim0['h_rim_mm']:.1f} mm (未修正)")

    with RunLog("circle_trace", label=args.label or ("" if args.go else "dryrun"), params=vars(args)) as log:
        log.add_input(args.layout)
        log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
        log.event("tcp", pose=tcp, flag=flag)
        log.save_snapshot("home", snap, info)
        log.save_json("rim_home", rim0)
        obj_plan = dict(obj, h_max=max(obj["h_max"], rim0["h_max_mm"]) * 1.15 + 5.0)
        arm = Arm(c, log, home[3:], args.speed)
        views_done = [home_view]
        rim, src, pose_src, img_src = rim0, "home", hp, snap["color"]
        status, res = "failed", {}
        try:
            if args.go:
                obj_c = dict(obj_plan, center=rim0["center"])
                targets = quick_targets(obj_c, [1.0, 0.0], args.views, args.radius)
                views, pinfo = scan3d.plan_views(obj_plan, hp, tcp, A, tcal, z_touch, args, (W, H), targets=targets)
                if args.tilt_deg > 0:
                    import scan_tilt as stl
                    paper_z = z_touch + (stl.R_tool(home[3:], tip["conv"]) @ tip["p"])[2]
                    scan_low = min([tcp[2], home[2]] + [v["tcp"][2] for v in views])
                    clearance = housing_margin(scan_low, paper_z, obj_plan["h_max"], reaches)
                    if not np.isfinite(clearance) or clearance < 0:
                        raise Quit("快速掃描的夾爪／相機淨空不足")
                print(f"\n快速掃描 {len(views)} 點:相機離紙 {pinfo['cam_h_scan']:.0f} mm,筆尖比物體高 "
                      f"{pinfo['pen_clearance_over_object']:.0f} mm,全部朝正下方 (不傾斜)")
                ask("確認筆已夾好、手放急停旁後按 Enter 開始快速掃描 ")
                for v in views:
                    arm.move(*v["tcp"], ori=home[3:])
                    s_, mmv = scan3d.capture(cam, info, args.n_depth)
                    tcp_now = request.Get_TCP_Pose(c)
                    d_tcp = np.array(tcp_now[:3]) - np.array(tcp[:3])
                    dpen = np.linalg.solve(A, d_tcp[:2])
                    pred_C = hp["C"] + np.array([dpen[0], dpen[1], -d_tcp[2]])
                    pose, psrc, tp, _ = scan3d.choose_pose(s_["color"], layout, K, scan3d.pose_from_center(hp["R"], pred_C))
                    log.save_snapshot(v["name"], s_, info)
                    views_done.append({"name": v["name"], "color": s_["color"], "mm": mmv, "K": K, "pose": pose,
                                       "pose_source": psrc, "pred_C": pred_C, "tcp": tcp_now})
                    print(f"  {v['name']} 拍好 ({'tag' if psrc == 'tags' else '手臂推算'})")
                scan3d.fix_arm_poses(views_done, log)
                xy, h, rgb, rows, _ = scan3d.process_views(views_done, layout, obj, log=log)
                rr = circle_rim.extract_rim(xy, h) if xy is not None else None
                if rr is not None:
                    rim, src = rr, f"quick{len(views_done) - 1}"
                    cv_ = next(v for v in views_done if v["name"] == "c")
                    pose_src, img_src = cv_["pose"], cv_["color"]
            cam_h_src = float(-pose_src["C"][2])
            gain, gain_src = height_gain(cam_h_src, args.height_gain)
            h_draw = rim["h_rim_mm"] * gain
            # 圓心 / 半徑用彩色邊緣修正 (深度在邊緣有一側偏,11-18 罐頭圓心偏 ~2 mm 造成偏心)
            if not args.no_color:
                rcol = circle_rim.refine_with_color(img_src, pose_src, K, rim["center"], rim["r_outer"], h_draw)
                if rcol and rcol["rms_mm"] < 1.0 and rcol["kept_frac"] > 0.6 and np.hypot(*rcol["shift_mm"]) < 5.0 \
                        and abs(rcol["r_outer"] - rim["r_outer"]) < 4.0:
                    print(f"  彩色邊緣修正:圓心移 ({rcol['shift_mm'][0]:+.2f}, {rcol['shift_mm'][1]:+.2f}) mm、外緣半徑 "
                          f"{rim['r_outer']:.2f} → {rcol['r_outer']:.2f} mm (擬合殘差 {rcol['rms_mm']:.2f} mm)")
                    rim = dict(rim, center_depth=rim["center"], r_outer_depth=rim["r_outer"], center=rcol["center"],
                               r_outer=rcol["r_outer"], color_rms_mm=rcol["rms_mm"])
                else:
                    print("  彩色邊緣修正不可靠,用深度的圓", (rcol or {}).get("rms_mm"))
            r_draw = rim["r_outer"] - args.inset
            print(f"\n[{src}] 圓心 ({rim['center'][0]:.1f}, {rim['center'][1]:.1f})、外緣半徑 {rim['r_outer']:.1f} mm "
                  f"(擬合殘差 {rim['rms_mm']:.2f} mm)、描繪半徑 {r_draw:.1f} mm")
            print(f"  捲邊高度 {rim['h_rim_mm']:.2f} x {gain:.3f} = {h_draw:.2f} mm (增益:{gain_src});頂蓋中間 "
                  f"{(rim['h_top_mm'] or float('nan')) * gain:.1f} mm")
            # 起點:離 home 最近的方向 (手臂少繞)
            if args.start_deg is None:
                pen_home = np.linalg.solve(A, np.array(tcp[:2]) - tcal)
                d = pen_home - np.array(rim["center"])
                start_deg = float(np.degrees(np.arctan2(d[1], d[0])))
            else:
                start_deg = args.start_deg
            path3 = circle_rim.circle_path(rim["center"], r_draw, h_draw, args.step_mm, start_deg, ccw=not args.cw)
            o_pen, o_note, _ = (np.zeros(2), "已關閉", 0) if args.no_pen_offset else pen_offset.learned()
            if np.any(o_pen != 0):
                print(f"  筆尖偏移修正 ({o_note}):路徑移 ({-o_pen[0]:+.2f}, {-o_pen[1]:+.2f}) mm")
            path_robot = np.array([[*(A @ (p[:2] - o_pen) + tcal), z_touch + p[2]] for p in path3])
            # ---- 傾斜:每一點的姿態 = 垂直姿態繞「該點圓切線」轉 tilt;TCP = 垂直時的 TCP + (R0 - R)·p (p = 筆尖在工具座標) ----
            oris = [list(home[3:])] * len(path_robot)
            if args.tilt_deg > 0:
                import scan_tilt as stl
                if args.tilt_deg > 35:
                    print(f"  注意:傾斜 {args.tilt_deg:g}° 超過筆尖校正測過的範圍 / 11-11 曾在 40° 卡在奇異點")
                p_tip = tip["p"]
                R0 = stl.R_tool(home[3:], tip["conv"])
                oris, adj = [], []
                for pt in path3:
                    rad_p = (pt[:2] - np.asarray(rim["center"])) / np.linalg.norm(pt[:2] - np.asarray(rim["center"]))
                    rad_b = A @ rad_p
                    rad_b = np.array([rad_b[0], rad_b[1], 0.0]) / np.linalg.norm(rad_b)
                    t_b = np.cross([0.0, 0.0, 1.0], rad_b)                       # 圓切線 (手臂座標)
                    best = None
                    for sg in (1, -1):
                        Rk = stl.tilt_base(t_b, sg * args.tilt_deg) @ R0
                        tipdir = Rk @ (p_tip / np.linalg.norm(p_tip))           # TCP -> 筆尖的方向
                        inward = -(tipdir[:2] @ rad_b[:2])                        # 筆尖往圓心那側 = 筆身往外倒
                        score = -inward if args.tilt_in else inward
                        if best is None or score > best[0]:
                            best = (score, Rk)
                    Rk = best[1]
                    oris.append(stl.angles_of(Rk, tip["conv"], near=home[3:]))
                    adj.append((R0 - Rk) @ p_tip)
                path_robot = path_robot + np.array(adj)
                print(f"  傾斜 {args.tilt_deg:g}° ({'往內' if args.tilt_in else '往圓心外'}倒);筆尖校正 p = "
                      f"{np.round(p_tip, 1).tolist()} mm (殘差 {tip.get('rms_mm', float('nan')):.2f} mm);"
                      f"TCP 修正量最大 {np.max(np.linalg.norm(adj, axis=1)):.0f} mm")
            view = draw_plan(img_src, pose_src, K, dict(rim), path3, r_draw)
            log.save_image("plan.jpg", view)
            cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "circle_trace_plan.png"))
            z_safe = min(float(home[2]), z_touch + max(h_draw, obj["h_max"]) + args.clearance + 5.0)
            plan = {"rim": rim, "rim_home": rim0, "source": src, "r_draw": r_draw, "h_draw": h_draw, "height_gain": gain,
                    "start_deg": start_deg, "path_paper_xyh": path3.tolist(), "path_robot": path_robot.tolist(),
                    "z_touch": z_touch, "z_safe": z_safe, "object_home": obj, "cam_h_mm": cam_h_src,
                    "pen_offset_applied": o_pen.tolist(), "pen_offset_note": o_note,
                    "tilt_deg": args.tilt_deg, "oris": oris}
            log.save_json("plan", plan)
            log.save_csv("path_robot_mm", ["x", "y", "z_est"], path_robot.tolist())
            print(f"  路徑 {len(path3)} 點 (每 {args.step_mm:g} mm,含頭尾重疊);起點 TCP ({path_robot[0][0]:.1f}, "
                  f"{path_robot[0][1]:.1f}),估計接觸 z {path_robot[0][2]:.2f};移動高度 z {z_safe:.1f}")
            if args.tilt_deg > 0:
                # 人工找接觸和下壓最多到各點預估高度以下 4 mm。包覆球全程高過工件頂面。
                margin = housing_margin(float(np.min(path_robot[:, 2])) - 4.0,
                                        z_touch + (R0 @ p_tip)[2],
                                        max(obj_plan["h_max"], h_draw, obj["h_max"] + 3.0), reaches)
                if args.go:
                    margin = min(margin, housing_margin(scan_low, paper_z,
                                     max(obj_plan["h_max"], h_draw, obj["h_max"] + 3.0), reaches))
                log.event("housing_clearance", margin_mm=margin, reaches_mm=reaches)
                if not np.isfinite(margin) or margin < 0:
                    raise Quit(f"夾爪／相機包覆球淨空不足 ({margin:.1f} mm)")
            if not args.go:
                status, res = "dry-run", {"r_outer": rim["r_outer"], "h_rim_mm": rim["h_rim_mm"]}
                print("\n(dry-run) 沒有移動手臂。看 circle_trace_plan.png:綠 = 外緣、洋紅 = 要畫的圓、綠圈 = 起點。確認後加 --go")
                raise StopIteration

            # ---------- 接觸 ----------
            x0, y0, zr0 = path_robot[0]
            ask(f"\n確認洋紅圓在罐頭捲邊上 (circle_trace_plan.png)、手放急停旁後按 Enter:移到起點上方 z {z_safe:.1f} ")
            z_hi = max(z_safe, zr0 + args.approach + 10.0)
            if args.tilt_deg > 0:
                # 碰撞檢查:起點 (垂直 -> 傾斜) 與終點 (傾斜 -> 垂直) 的原地轉動,模擬筆尖 / 筆身掃過的位置。
                # 11-18 pen_pivot 事故:傾斜轉回垂直時筆尖會往下掉 ~38 mm;另外筆尖若不在 TCP 正下方,轉動也可能往下。
                kw = dict(p_tip=p_tip, R0=R0, A=A, tcal=tcal, z_touch=z_touch, center=rim["center"],
                          r_obj=rim["r_outer"], h_obj=max(h_draw, obj["h_max"]) + 3.0)
                R_start = stl.R_tool(oris[0], tip["conv"])
                R_end = stl.R_tool(oris[-1], tip["conv"])
                zs = safe_rot_height(path_robot[0][:2], z_hi, R0, R_start, float(home[2]), **kw)
                ze = safe_rot_height(path_robot[-1][:2], z_hi, R_end, R0, float(home[2]), **kw)
                if zs is None or ze is None:
                    raise Quit("找不到安全的轉動高度 (筆尖 / 筆身會碰到工件),不執行")
                z_hi = max(zs, ze)
                # 描繪中每段:姿態也在變,檢查兩點之間筆尖會不會掉下去 (相對兩端的直線)
                dev = []
                from scipy.spatial.transform import Rotation as _Ro, Slerp as _Sl
                for k in range(1, len(path_robot)):
                    Ra, Rb = stl.R_tool(oris[k - 1], tip["conv"]), stl.R_tool(oris[k], tip["conv"])
                    sl = _Sl([0, 1], _Ro.from_matrix([Ra, Rb]))
                    for f_ in (0.25, 0.5, 0.75):
                        T_ = path_robot[k - 1] + (path_robot[k] - path_robot[k - 1]) * f_
                        tip_ = T_ + sl([f_]).as_matrix()[0] @ p_tip
                        lin = (path_robot[k - 1] + Ra @ p_tip) * (1 - f_) + (path_robot[k] + Rb @ p_tip) * f_
                        dev.append(tip_[2] - lin[2])
                print(f"  碰撞檢查:轉動高度 z {z_hi:.1f} (垂直時筆尖離紙 {z_hi - z_touch:.0f} mm);"
                      f"描繪中段間筆尖高度偏差 {min(dev):+.2f} ~ {max(dev):+.2f} mm (假設控制器姿態做球面內插)")
                log.event("tilt_safety", z_rot=z_hi, seg_dev_min=min(dev), seg_dev_max=max(dev))
            arm.move(x0, y0, z_hi)
            if args.tilt_deg > 0:
                arm.move(x0, y0, z_hi, ori=oris[0])           # 在高處原地轉 (繞 TCP,筆尖只會變高)
                arm.ori = list(oris[0])                       # 人工找接觸時保持起點的傾斜
            z_start = zr0 + args.approach
            z_floor = zr0 - 4.0
            arm.move(x0, y0, z_start)
            print(f"\n=== 已在起點上方 {args.approach:g} mm (z = {z_start:.2f});每步 0.5 mm 往下 (Enter),"
                  f"筆尖剛好碰到捲邊就按 c (最低到 z {z_floor:.2f}) ===")
            genuine = False
            while True:
                x, y, z_c = probe_with_nudge(arm, x0, y0, z_start, 0.5, z_floor, z_hi)
                if z_c < z_start - 0.05:
                    genuine = True
                    break
                if ask("  還沒往下就按 c:筆可能已經壓著。確定剛好碰到?(y / Enter = 往上 3 mm 再找) ").strip().lower() == "y":
                    break
                z_start = z_c + 3.0
                if z_start > z_hi:
                    raise Quit("起點已抬到移動高度,請檢查高度")
                arm.move(x, y, z_start)
                x0, y0 = x, y
            shift = np.array([x - path_robot[0][0], y - path_robot[0][1]])
            dz = z_c - zr0
            log.event("contact", robot_xy=[x, y], z=z_c, z_est=zr0, dz=dz, shift=shift.tolist())
            h_contact = z_c - z_touch
            print(f"  接觸 z {z_c:.2f} (估計 {zr0:.2f},差 {dz:+.2f} mm);接觸換算捲邊高 {h_contact:.2f} mm")
            if args.tilt_deg == 0 and genuine and rim["h_rim_mm"] > 5:
                g_obs = h_contact / rim["h_rim_mm"]
                if 0.9 < g_obs < 1.25:
                    CalibrationStore().save("ridge_height_gain", {"gain": g_obs, "h_contact_mm": h_contact,
                                                                  "h_apex_raw_mm": rim["h_rim_mm"], "cam_h_mm": cam_h_src,
                                                                  "object": "circle_rim"},
                                            source_run=log.path, notes="circle_trace 人工確認接觸")
                    print(f"  這次增益 {g_obs:.3f} (已記錄)")
            path_draw = path_robot.copy()
            path_draw[:, :2] += shift
            path_draw[:, 2] = np.maximum(path_robot[:, 2] + dz - args.press, path_robot[:, 2] - 4.0)
            if args.tilt_deg > 0:
                end_height = safe_rot_height(path_draw[-1, :2], z_hi, R_end, R0, float(home[2]), **kw)
                if end_height is None:
                    raise Quit("微調後的終點缺少轉回垂直的淨空")
                z_hi = end_height
            # ---------- 描繪 ----------
            done, k, t0 = 0, 1, time.time()
            while k < len(path_draw):
                try:
                    arm.move(*map(float, path_draw[k]), ori=oris[k])
                    done = k
                    k += 1
                except send.CommandIgnored as e:
                    arm.stop_and_check("circle_ignored")
                    raise Quit(f"手臂沒有走到第 {k} 點 (傾斜姿態可能碰到關節極限 / 奇異點):{e}")
                except Paused as e:
                    if str(e) == "ctrl_c":
                        raise
                    ch = ""
                    while ch not in ("r", "q"):
                        ch = ask(f"暫停 (畫到 {done}/{len(path_draw) - 1}):r = 繼續 / q = 抬起回 home > ").strip().lower()
                    if ch == "q":
                        raise Quit("使用者結束")
                    arm.move(*map(float, path_draw[done]), ori=oris[done])
            log.event("trace_done", points=done, of=len(path_draw) - 1, seconds=round(time.time() - t0, 1))
            print(f"  描繪完成 {done} 段,{time.time() - t0:.0f} 秒")
            arm.lift_to(z_hi)
            if args.tilt_deg > 0:
                pcur = __import__("pen_pivot").read_pose(c)
                arm.move(pcur[0], pcur[1], pcur[2], ori=home[3:])    # 高處轉回垂直
                arm.ori = list(home[3:])
            check_pairs = []
            c_view = next((v for v in views_done if v["name"] == "c" and v.get("tcp")), None)
            if c_view is not None:
                try:
                    arm.move(*c_view["tcp"][:3], ori=home[3:])
                    s_c, _ = scan3d.capture(cam, info, 5)
                    log.save_snapshot("c_after", s_c, info)
                    check_pairs.append((c_view["color"], s_c["color"], "c"))
                except send.CommandIgnored as e:
                    print("  回正上方拍照沒有完成:", e)
            arm.move(*home[:3], ori=home[3:])
            time.sleep(1.0)
            s_after, _ = scan3d.capture(cam, info, 5)
            log.save_snapshot("home_after", s_after, info)
            check_pairs.append((snap["color"], s_after["color"], "home"))
            status, res = "ok", {"points_done": done, "dz_mm": dz}
            best = None
            for before, after, name in check_pairs:
                r_, v_ = analyze_circle(before, after, layout, K, [rim["center"][0] + 0, rim["center"][1]], r_draw, h_draw + dz)
                r_["view"] = name
                pose_a = scan3d.tag_pose(after, layout, K)
                if pose_a is not None:
                    pix = ridge_check.project(pose_a, K, path3)
                    log.save_image(f"circle_check_{name}.jpg", ridge_check.crop_around(v_, pix))
                if best is None and r_.get("ok"):
                    best = r_
            best = best or r_
            log.save_json("circle_check", best)
            print("\n" + summarize_check(best))
            if best.get("ok") and best.get("coverage", 0) >= 0.5:
                sh = best["ink_circle"]["center_shift_mm"]
                if pen_offset.record(np.eye(2), sh, o_pen, "circle", run=log.path, extra={"coverage": best["coverage"]}):
                    print(f"  回饋:筆跡圓心偏 ({sh[0]:+.2f}, {sh[1]:+.2f}) mm 已記錄,下次自動修正筆尖偏移")
            res.update({k_: best.get(k_) for k_ in ("mean_abs_dev_mm", "p90_abs_dev_mm", "max_abs_dev_mm", "radial_bias_mm",
                                                    "coverage")}, check_view=best.get("view"))
        except StopIteration:
            pass
        except (Quit, Paused, KeyboardInterrupt) as e:
            print("\n停止:", e, "-> 垂直抬起回 home")
            status, res = "aborted", {"reason": str(e)}
            try:
                if not args.go:
                    raise Paused("dry-run 沒有移動手臂")
                # 傾斜時:先維持姿態抬到 home 高度再轉回垂直 (11-18:轉回垂直時筆尖會往下掉約 38 mm)
                arm.lift_to(float(home[2]) if args.tilt_deg > 0
                            else min(float(home[2]), z_touch + obj_plan["h_max"] + args.clearance))
                pcur = __import__("pen_pivot").read_pose(c)
                arm.move(pcur[0], pcur[1], pcur[2], ori=home[3:])        # 先轉回垂直 (傾斜時)
                arm.move(*home[:3], ori=home[3:])
            except (Paused, KeyboardInterrupt, send.CommandIgnored, TimeoutError):
                print("已停止,手臂停在目前位置 (之後用回 home)")
        cam.close()
        c.close()
        res.update(source=src, r_outer=rim["r_outer"], h_rim_raw=rim["h_rim_mm"])
        log.finish(status, res)
        print("紀錄:", log.path)


if __name__ == "__main__":
    main()
