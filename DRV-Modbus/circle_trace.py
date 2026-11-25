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

傾斜描繪 (--tilt-deg 5~20,或 --target bottom = 底面外圍,預設 15 度):
  - 只需要護貝紙筆尖校正:筆尖 3D 位置由這次的 home 照片 + 分身的手眼自動推出 (pen_tip_twin.estimate),不做傾斜碰點。
  - 開始前用分身檢查整套動作和「每一點中止時怎麼回 home」(tilt_plan / twin_check),沒過就不動。
  - 換姿態一律繞筆尖轉 (筆尖留在工件上方原處);先空走一圈,再按 Enter 下降,人工確認接觸 (c) 才畫。
  - 底面外圍畫在紙上;離罐壁多遠由工具外形自動算 (15 度約 11 mm,受筆身粗細限制,貼不到牆角)。

連續運動 (--smooth):控制器 RL MovL + PASS,見 smooth_motion.py。控制器沒設定好時自動改用逐點。

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
    ap.add_argument("--target", choices=["rim", "bottom"], default="rim", help="rim = 頂面捲邊;bottom = 底面外圍 (紙上)")
    ap.add_argument("--tilt-deg", type=float, default=None,
                    help="筆往圓心外傾斜幾度 (5~20);0 = 垂直。預設:rim 0、bottom 15")
    ap.add_argument("--bottom-offset", type=float, default=None,
                    help="底面模式:筆尖中心離罐壁 mm (不能比分身算的最小值小;預設 = 最小值)")
    ap.add_argument("--pen-len", type=float, default=None,
                    help="尺量筆長:法蘭面到筆尖 mm (筆垂直時)。傾斜描繪一定要給:11-25 實機分身推算短了約 40 mm,筆跡往內偏 11 mm")
    ap.add_argument("--air-lift", type=float, default=None,
                    help="空走一圈時筆尖比描繪高幾 mm (預設 20)。拿掉罐頭、設 3 可以看筆尖和紙的距離是否整圈一樣")
    ap.add_argument("--smooth", action="store_true", help="連續運動 (控制器要先裝 loader,見 smooth_motion.py)")
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

    if args.tilt_deg is None:
        args.tilt_deg = 15.0 if args.target == "bottom" else 0.0
    if not np.isfinite(args.tilt_deg) or args.tilt_deg < 0:
        ap.error("傾角必須是有限的非負數")
    if args.target == "bottom" and args.tilt_deg == 0:
        ap.error("底面外圍一定要傾斜 (垂直時夾爪和筆身的檢查不在這個流程裡)")
    if not np.isfinite(args.approach) or args.approach < 0:
        ap.error("接近高度必須是有限的非負數")
    if args.tilt_deg > 0 and not args.pen_len:
        ap.error("傾斜描繪要先用尺量筆長 (法蘭面到筆尖,筆垂直),填 --pen-len / 網頁進階設定「尺量筆長」。\n"
                 "11-25 實機:分身推算的筆長短了約 40 mm,筆尖整圈往罐頭內偏 11 mm、差點壓壞筆")
    if args.air_lift is not None and not 2.0 <= args.air_lift <= 60.0:
        ap.error("--air-lift 需在 2~60 mm")
    if args.tilt_deg > 0:
        args.speed = min(args.speed, 8)                 # 傾斜還沒上機驗證:限速
    import smooth_motion
    import tilt_plan
    import pen_tip_twin
    from pymodbus.client import ModbusTcpClient
    import warp_pick as wp
    from drv_modbus import send, request
    from pen_calibrate import refresh
    from rs_capture import RSCamera
    from pen_calibrate import Arm, MOVE_ERRORS, Paused, ask
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
        tp, sm, stuck = None, None, False
        tilted = args.tilt_deg > 0
        try:
            if tilted:
                # 動之前先用 home 一張的結果完整檢查一次,不通過就連掃描都不做
                import twin_check
                twin = twin_check.Twin()
                tip = pen_tip_twin.estimate(hp, tcp, cal["data"], twin.model["hand_eyes"], args.pen_len)
                print(f"  筆尖 (分身推算) {np.round(tip['p'], 1).tolist()} mm,筆長來源 {tip['z_source']} ±{tip['z_unc']:.1f}")
            if args.smooth and args.go:
                sm = smooth_motion.Smooth(c, log)
                try:
                    ask("連續運動自我測試 (不動手臂):會啟動控制器 RL 900,之後描繪會覆蓋點位表 501 起的暫存點。按 Enter ")
                    refresh(c)
                    sm.preflight()
                    print("  連續運動可以用")
                except smooth_motion.SmoothUnavailable as e:
                    print("  連續運動不能用,改成逐點描繪:", e)
                    sm = None
            if args.go:
                obj_c = dict(obj_plan, center=rim0["center"])
                targets = quick_targets(obj_c, [1.0, 0.0], args.views, args.radius)
                views, pinfo = scan3d.plan_views(obj_plan, hp, tcp, A, tcal, z_touch, args, (W, H), targets=targets)
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
                    pose, psrc, _, _ = scan3d.choose_pose(s_["color"], layout, K, scan3d.pose_from_center(hp["R"], pred_C))
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
            h_path = 0.0 if args.target == "bottom" else h_draw
            o_pen, o_note, _ = (np.zeros(2), "已關閉", 0) if args.no_pen_offset else pen_offset.learned("circle")
            if np.any(o_pen != 0):
                print(f"  筆尖偏移修正 ({o_note}):路徑移 ({-o_pen[0]:+.2f}, {-o_pen[1]:+.2f}) mm")

            def make_path(r):
                return circle_rim.circle_path(rim["center"], r, h_path, args.step_mm, start_deg, ccw=not args.cw)

            offset, precheck = None, None
            if tilted:
                try:
                    tp = tilt_plan.TiltPlan(cal["data"], tcp, tip, rim["center"], rim["r_outer"],
                                            max(h_draw, obj["h_max"]) + 3.0, args.tilt_deg, twin=twin, inward=args.tilt_in)
                    if args.target == "bottom":
                        least = tilt_plan.bottom_offset(tp, make_path, o_pen)
                        if args.bottom_offset is not None and args.bottom_offset < least:
                            raise tilt_plan.PlanError(f"離罐壁 {args.bottom_offset:g} mm 太近,最少要 {least:g} mm")
                        offset = args.bottom_offset or least
                        r_draw = rim["r_outer"] + offset
                    tp.set_path(make_path(r_draw), o_pen, args.target)
                    print("  分身執行前檢查中 (可達、關節、奇異點、碰撞、每一點的中止路徑) ...")
                    precheck = tp.verify(start=list(__import__("pen_pivot").read_pose(c)) if args.go else None,
                                         air_lift=args.air_lift)
                except tilt_plan.PlanError as e:
                    print("  分身執行前檢查:不通過 ->", e)
                    log.event("precheck", ok=False, reason=str(e))
                    raise Quit(str(e))
                print(f"  分身執行前檢查:通過 (關節最小餘裕 {precheck['margin_deg']:.1f} 度、筆尖側向誤差約 "
                      f"±{precheck['tip_err_mm']:.1f} mm" + (f"、離罐壁 {offset:g} mm" if offset else "") + ")")
                log.event("precheck", pen_tip=np.asarray(tip["p"]).tolist(), z_unc=tip["z_unc"], bottom_offset=offset,
                          **precheck)
            path3 = make_path(r_draw)
            if tilted:
                path_robot = np.array([p[:3] for p in tp.draw])
                oris = [p[3:] for p in tp.draw]
            else:
                path_robot = np.array([[*(A @ (p[:2] - o_pen) + tcal), z_touch + p[2]] for p in path3])
                oris = [list(home[3:])] * len(path_robot)
            view = draw_plan(img_src, pose_src, K, dict(rim), path3, r_draw)
            log.save_image("plan.jpg", view)
            cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "circle_trace_plan.png"))
            z_safe = min(float(home[2]), z_touch + max(h_draw, obj["h_max"]) + args.clearance + 5.0)
            plan = {"rim": rim, "rim_home": rim0, "source": src, "r_draw": r_draw, "h_draw": h_draw, "height_gain": gain,
                    "start_deg": start_deg, "path_paper_xyh": path3.tolist(), "path_robot": path_robot.tolist(),
                    "z_touch": z_touch, "z_safe": z_safe, "object_home": obj, "cam_h_mm": cam_h_src,
                    "pen_offset_applied": o_pen.tolist(), "pen_offset_note": o_note,
                    "tilt_deg": args.tilt_deg, "oris": oris, "target": args.target, "bottom_offset": offset,
                    "h_path": h_path, "precheck": precheck, "smooth": sm is not None}
            log.save_json("plan", plan)
            log.save_csv("path_robot_mm", ["x", "y", "z_est"], path_robot.tolist())
            print(f"  路徑 {len(path3)} 點 (每 {args.step_mm:g} mm,含頭尾重疊);起點 TCP ({path_robot[0][0]:.1f}, "
                  f"{path_robot[0][1]:.1f}),估計接觸 z {path_robot[0][2]:.2f};移動高度 z {z_safe:.1f}")
            if not args.go:
                status, res = "dry-run", {"r_outer": rim["r_outer"], "h_rim_mm": rim["h_rim_mm"]}
                print("\n(dry-run) 沒有移動手臂。看 circle_trace_plan.png:綠 = 外緣、洋紅 = 要畫的圓、綠圈 = 起點。確認後加 --go")
                raise StopIteration

            def follow(points, what):
                """目前在 points[0]:沿 points 走完。連續運動可用就用,否則逐點。回傳走到第幾點。"""
                k, t0 = 1, time.time()
                how = "smooth" if sm else "segmented"
                while k < len(points):
                    if sm:
                        refresh(c)
                        why, n = sm.run(points[k:], min(args.speed, smooth_motion.MAX_SPEED), poll=arm._poll)
                        if why == "ctrl_c":
                            raise Paused("ctrl_c")
                        k = len(points) if why == "done" else k + n
                        if why == "done":
                            break
                    else:
                        try:
                            arm.move(*map(float, points[k][:3]), ori=list(points[k][3:]))
                            k += 1
                            continue
                        except send.CommandIgnored as e:
                            arm.stop_and_check("circle_ignored")
                            raise Quit(f"手臂沒有走到第 {k} 點 (可能碰到關節極限 / 奇異點):{e}")
                        except Paused as e:
                            if str(e) == "ctrl_c":
                                raise
                    ch = ""
                    while ch not in ("r", "q"):
                        ch = ask(f"暫停 ({what} {k - 1}/{len(points) - 1}):r = 繼續 / q = 抬起回 home > ").strip().lower()
                    if ch == "q":
                        raise Quit("使用者結束")
                sec = time.time() - t0
                log.event("path_done", what=what, how=how, seconds=round(sec, 1), points=len(points) - 1)
                print(f"  {what}完成:{len(points) - 1} 段,{sec:.0f} 秒 ({'連續' if sm else '逐點'})")
                return len(points) - 1

            # ---------- 接觸 ----------
            x0, y0, zr0 = path_robot[0]
            z_floor = zr0 - (2.0 if args.target == "bottom" else 4.0)
            allow = None
            if tilted:
                above, up = tp.entry()
                lift = tilt_plan.AIR_LIFT if args.air_lift is None else args.air_lift
                air = [[p[0], p[1], p[2] + lift] + p[3:] for p in tp.draw]
                ask(f"\n確認洋紅圓位置 (circle_trace_plan.png)、工件和校正紙沒動、周圍沒有其他東西,手放急停旁。\n"
                    f"按 Enter:移到起點上方、繞筆尖傾斜 {args.tilt_deg:g} 度,先在高 {lift:g} mm 處空走一圈 (不碰工件)。"
                    f"筆尖和紙的距離整圈都應該差不多是 {lift:g} mm,不是就按 Ctrl+C ")
                arm.move(*above[:3], ori=list(home[3:]))
                arm.move(*up[:3], ori=up[3:])
                arm.ori = list(up[3:])
                arm.move(*air[0][:3], ori=air[0][3:])
                follow(air, "空走")
                end_up = tp.raised(air[-1])
                arm.move(*end_up[:3], ori=end_up[3:])
                arm.move(*up[:3], ori=up[3:])
                ask("空走時夾爪、相機、線材都沒有碰到東西,而且圓的位置正確?按 Enter 下降找接觸 (不對就按 Ctrl+C) ")
                z_hi = up[2]
                allow = lambda x, y, z: tp.allowed([x, y, z] + list(arm.ori))
            else:
                ask(f"\n確認洋紅圓在罐頭捲邊上 (circle_trace_plan.png)、手放急停旁後按 Enter:移到起點上方 z {z_safe:.1f} ")
                z_hi = max(z_safe, zr0 + args.approach + 10.0)
                arm.move(x0, y0, z_hi)
            z_start = zr0 + args.approach
            arm.move(x0, y0, z_start)
            print(f"\n=== 已在起點上方 {args.approach:g} mm (z = {z_start:.2f});每步 0.5 mm 往下 (Enter),"
                  f"筆尖剛好碰到{'紙' if args.target == 'bottom' else '捲邊'}就按 c (最低到 z {z_floor:.2f}) ===")
            genuine = False
            while True:
                x, y, z_c = probe_with_nudge(arm, x0, y0, z_start, 0.5, z_floor, z_hi, allow=allow)
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
            if not tilted and genuine and rim["h_rim_mm"] > 5:
                g_obs = h_contact / rim["h_rim_mm"]
                if 0.9 < g_obs < 1.25:
                    CalibrationStore().save("ridge_height_gain", {"gain": g_obs, "h_contact_mm": h_contact,
                                                                  "h_apex_raw_mm": rim["h_rim_mm"], "cam_h_mm": cam_h_src,
                                                                  "object": "circle_rim"},
                                            source_run=log.path, notes="circle_trace 人工確認接觸")
                    print(f"  這次增益 {g_obs:.3f} (已記錄)")
            path_draw = path_robot.copy()
            path_draw[:, :2] += shift
            path_draw[:, 2] = np.maximum(path_robot[:, 2] + dz - args.press, path_robot[:, 2] - (zr0 - z_floor))
            draw6 = [list(map(float, p)) + list(o) for p, o in zip(path_draw, oris)]
            if tilted:
                try:
                    tp.verify(draw6, air_lift=args.air_lift)    # 微調 + 接觸高度之後再檢查一次
                except tilt_plan.PlanError as e:
                    raise Quit(f"微調後的路徑沒通過檢查:{e}")
            # ---------- 描繪 ----------
            t0 = time.time()
            done = follow(draw6, "描繪")
            log.event("trace_done", points=done, of=len(path_draw) - 1, seconds=round(time.time() - t0, 1))
            if tilted:
                refresh(c)
                for p in tp.escape(list(request.Get_TCP_Pose(c)))[:2]:
                    arm.move(*p[:3], ori=p[3:])
                arm.ori = list(home[3:])
            else:
                arm.lift_to(z_hi)
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
                r_, v_ = analyze_circle(before, after, layout, K, [rim["center"][0] + 0, rim["center"][1]], r_draw, h_path + dz)
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
            if not tilted and best.get("ok") and best.get("coverage", 0) >= 0.5:    # 傾斜的誤差含筆長,不混進垂直的偏移
                sh = best["ink_circle"]["center_shift_mm"]
                if pen_offset.record(np.eye(2), sh, o_pen, "circle", run=log.path, extra={"coverage": best["coverage"]}):
                    print(f"  回饋:筆跡圓心偏 ({sh[0]:+.2f}, {sh[1]:+.2f}) mm 已記錄,下次自動修正筆尖偏移")
            res.update({k_: best.get(k_) for k_ in ("mean_abs_dev_mm", "p90_abs_dev_mm", "max_abs_dev_mm", "radial_bias_mm",
                                                    "coverage")}, check_view=best.get("view"))
        except StopIteration:
            pass
        except smooth_motion.SmoothStuck as e:
            stuck = True
            status, res = "failed", {"reason": str(e)}
            print("\n!!! 連續運動狀態不明:", e, "\n!!! 不再送任何移動指令。請按實體急停,在教導器確認 RL 已停止後再回 home")
        except (Quit, Paused, KeyboardInterrupt, smooth_motion.SmoothFailed,
                smooth_motion.SmoothUnavailable) + MOVE_ERRORS as e:
            status, res = "aborted", {"reason": str(e)}
            if not args.go:
                print("\n停止:", e, "(dry-run,手臂沒有動)")
            else:
                print("\n停止:", e, "-> 垂直抬起回 home")
                try:
                    refresh(c)
                    now = __import__("pen_pivot").read_pose(c)
                    if now is None:
                        raise ConnectionError("讀不到手臂位置")
                    if tp is not None:
                        # 傾斜:維持姿態抬到工件上方 -> 繞筆尖轉回垂直 -> 回 home (抬到 home 高度在圓上很多點到不了)
                        for p in tp.escape(list(now)):
                            arm.move(*p[:3], ori=p[3:])
                    else:
                        arm.lift_to(min(float(home[2]), z_touch + obj_plan["h_max"] + args.clearance))
                        arm.move(*home[:3], ori=home[3:])
                except (Paused, KeyboardInterrupt) + MOVE_ERRORS:
                    print("已停止,手臂停在目前位置 (之後用回 home)")
        cam.close()
        c.close()
        res.update(target=args.target, tilt_deg=args.tilt_deg, stuck=stuck, source=src, r_outer=rim["r_outer"], h_rim_raw=rim["h_rim_mm"])
        log.finish(status, res)
        print("紀錄:", log.path)


if __name__ == "__main__":
    main()
