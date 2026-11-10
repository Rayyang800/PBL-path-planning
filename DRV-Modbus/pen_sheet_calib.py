# -*- coding: utf-8 -*-
"""
護貝校正紙的筆尖校正 (整張紙都可以畫,白板筆畫完可擦掉)。

和 pen_quick_calib 的差別:點分散在整張紙 (預設 3x3 格子),旋轉與比例都量得準;
擬合完整仿射 (含 tag 間距估計的比例誤差),並用「留一法」報告真實的預測誤差。

流程 (--go)
  0. 手臂在 home:拍「之前」照片 (彩色 + 深度)
  1. 第一點 (最靠近紙中心的格點) 上方 -> (可選) 裝筆 -> 人工下降,可 x/y 微調,剛好畫得出來按 c -> z_touch
     在第一點畫小十字
  2. (預設) 修正:垂直抬起 -> 回 home 拍照 -> 找到第一個十字的實際位置 -> 修正其餘點的指令 (舊 ArUco 粗略換算可能差 ±20 mm)
  3. 其餘點:抬 hop mm 平移 -> 下到 z_touch 畫小十字 -> 抬起 (固定高度,不拍照;--touch-all 則每點人工確認高度)
  4. 垂直抬起 -> 回 home 拍「之後」照片 -> 前後相減找十字 -> 仿射擬合 + 留一法 -> 存 calibrations/pen_paper/ (method = sheet)
  5. 校正完可以擦掉筆跡 (紙不要移動;擦不擦都不影響這次結果)

  python pen_sheet_calib.py                                  # dry-run:印計畫、存紀錄與 pen_sheet_plan.png
  python pen_sheet_calib.py --go --pen-loaded                # 執行 (3x3)
  python pen_sheet_calib.py --go --pen-loaded --grid 4 3 --margin 8
  python pen_sheet_calib.py --analyze runs\\<id>             # 重新分析

按鍵:Ctrl+C = 停止 -> 垂直抬起 -> 回 home 結束 (已畫的會拍照,可 --analyze);
      後續點移動中按 空白/Esc/p = 停止並選單 (r 重做 / s 跳過 / h 回 home 用已完成點分析 / q 結束);移動中 o/g = 夾爪開/夾
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient

import pen_marks
import warp_pick as wp
from drv_modbus import send, request
from paper_frame import PaperFrame
from pen_calibrate import Arm, Paused, ask, grab, LAST_GRAB, LAYOUT, MAX_SPEED, safe_region, small_cross
from pen_quick_calib import Quit, probe_with_nudge, rough_rotation, angle_of, point_menu
from run_log import RunLog, CalibrationStore, load_run

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- 規劃 ----------------
def grid_points(pf, cols, rows, margin):
    """可畫範圍 (tag 內緣再縮 margin) 內的 cols x rows 格點 (紙面 mm)。
    第一點 = 最靠近紙中心的格點,其餘依蛇形順序 (相鄰點距離短)。"""
    x0, y0, x1, y1 = safe_region(pf, margin)
    xs = np.linspace(x0, x1, cols) if cols > 1 else [(x0 + x1) / 2]
    ys = np.linspace(y0, y1, rows) if rows > 1 else [(y0 + y1) / 2]
    snake = [(float(x), float(y)) for j, y in enumerate(ys) for x in (xs if j % 2 == 0 else xs[::-1])]
    c = np.array([pf.W / 2, pf.H / 2])
    first = min(range(len(snake)), key=lambda i: np.hypot(*(np.array(snake[i]) - c)))
    return [snake[first]] + [p for i, p in enumerate(snake) if i != first]


def paper_to_robot(pf, H_rough, paper_xy, offset, surface_h=0.0):
    pix = pf.paper_to_pix(paper_xy, surface_h)
    return pen_marks.apply_h(H_rough, pix) + np.asarray(offset, float)


# ---------------- 分析 ----------------
def analyze(before, after, layout, points, H_rough, offset, radius_mm, surface_h=0.0):
    """points: [{"index", "robot_xy", ...}];offset = 已知的「指令 - 粗略換算的落點」(手臂 mm)。"""
    pfa, pfb = PaperFrame(layout=layout), PaperFrame(layout=layout)
    if not (pfa.detect(after) and pfb.detect(before)):
        return {"ok": False, "reason": "前後照片沒有都看到 4 個 tag:" + pfa.describe() + " / " + pfb.describe()}
    excl = pen_marks.tag_exclusion_mask(after.shape, pfa, grow_mm=3.0)
    blobs = pen_marks.new_ink_components(before, after, pfa, pfb, min_drop=30, min_area=4, exclude=excl)
    Hinv = np.linalg.inv(H_rough)
    pred = [pen_marks.apply_h(Hinv, [np.asarray(p["robot_xy"]) - offset])[0] for p in points]
    r_px = radius_mm / pfa.mm_per_pixel()
    idx = pen_marks.assign_nearest(blobs, pred, r_px)
    det = []
    for p, pu, k in zip(points, pred, idx):
        d = {"index": p["index"], "predicted_uv": pu.tolist(), "ok": k is not None}
        if k is None:
            d["reason"] = f"預測位置 {radius_mm:g} mm 內找不到筆跡 (太淡、沒畫到、或被擦掉)"
        else:
            d.update(uv=list(blobs[k][1]), area_px=blobs[k][0],
                     paper_mm=pfa.pix_to_paper([blobs[k][1]], surface_h)[0].tolist(),
                     pred_err_mm=float(np.hypot(*(np.array(blobs[k][1]) - pu)) * pfa.mm_per_pixel()))
        det.append(d)
    good = [(d, p) for d, p in zip(det, points) if d["ok"]]
    res = {"ok": False, "detections": det, "n_blobs": len(blobs), "paper_frame": pfa.describe(),
           "mm_per_px": pfa.mm_per_pixel()}
    if len(good) < 3:
        res["reason"] = f"只找到 {len(good)} 個十字,至少要 3 個"
        return res
    P = np.array([d["paper_mm"] for d, _ in good])
    C = np.array([p["robot_xy"] for _, p in good])
    fits = {"rigid": pen_marks.fit_rigid}
    if len(good) >= 4:
        fits["affine"] = pen_marks.fit_affine
    out = {}
    for name, f in fits.items():
        A, t, r = f(P, C)
        e = np.linalg.norm(r, axis=1)
        loo = pen_marks.loo_errors(P, C, f) if len(good) >= (4 if name == "rigid" else 5) else None
        out[name] = {"A": A.tolist(), "t": t.tolist(), "rms_mm": float(np.sqrt((e ** 2).mean())),
                     "max_mm": float(e.max()), "residual_mm": e.tolist(),
                     "loo_mean_mm": float(np.mean(loo)) if loo else None, "loo_max_mm": float(np.max(loo)) if loo else None,
                     "affine": pen_marks.describe_affine(A)}
    # 有留一法時用留一法挑模型;點少時用剛體 (參數少、比較不會亂外插)
    best = "affine" if ("affine" in out and out["affine"]["loo_mean_mm"] is not None
                        and out["affine"]["loo_mean_mm"] <= out["rigid"]["loo_mean_mm"]) else "rigid"
    A = np.array(out[best]["A"])
    U, _, Vt = np.linalg.svd(A)
    Rr = rough_rotation(pfa, H_rough, P.mean(0))
    rot = angle_of(U @ Vt)
    res.update({"ok": True, "fit": best, "fits": out, "A": out[best]["A"], "t": out[best]["t"],
                "rms_mm": out[best]["rms_mm"], "max_mm": out[best]["max_mm"],
                "loo_mean_mm": out[best]["loo_mean_mm"], "loo_max_mm": out[best]["loo_max_mm"],
                "n_points": len(good), "rotation_deg": rot,
                "rotation_vs_rough_deg": float((rot - angle_of(Rr) + 180) % 360 - 180),
                "span_mm": [float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1]))],
                "mark_area_px": [d.get("area_px") for d in det]})
    return res


def summarize(res):
    lines = [res.get("paper_frame", "")]
    for d in res.get("detections", []):
        if d["ok"]:
            lines.append(f"  點 #{d['index']}: 紙面 ({d['paper_mm'][0]:6.1f},{d['paper_mm'][1]:6.1f}) mm  十字面積 {d['area_px']:4d} px"
                         f"  (與預測差 {d['pred_err_mm']:.1f} mm)")
        else:
            lines.append(f"  點 #{d['index']}: {d['reason']}")
    if not res.get("ok"):
        lines.append("分析失敗:" + res.get("reason", ""))
        return "\n".join(lines)
    for name, f in res["fits"].items():
        a = f["affine"]
        loo = "" if f["loo_mean_mm"] is None else f",留一法 平均 {f['loo_mean_mm']:.2f} / 最大 {f['loo_max_mm']:.2f} mm"
        lines.append(f"  [{name}] 殘差 RMS {f['rms_mm']:.2f} / 最大 {f['max_mm']:.2f} mm{loo};"
                     f"比例 {a['scale_x']:.4f}/{a['scale_y']:.4f},兩軸夾角 {a['axis_angle_deg']:.2f}°")
    lines.append(f"採用 {res['fit']}:{res['n_points']} 點,涵蓋 {res['span_mm'][0]:.0f} x {res['span_mm'][1]:.0f} mm,"
                 f"旋轉 {res['rotation_deg']:.1f}°(與舊 ArUco 差 {res['rotation_vs_rough_deg']:+.1f}°"
                 + ("" if abs(res["rotation_vs_rough_deg"]) < 5 else ",差很多,請看 marks_overlay.jpg") + ")")
    areas = [a for a in res["mark_area_px"] if a]
    if areas and min(areas) < 0.4 * np.median(areas):
        lines.append("注意:有些十字明顯比較小/淡 -> 那裡的紙面可能比較低 (紙不平),可用 --touch-all")
    return "\n".join(lines)


def overlay(img, res):
    v = img.copy()
    for d in res.get("detections", []):
        pu = tuple(int(round(x)) for x in d["predicted_uv"])
        cv2.circle(v, pu, 14, (0, 200, 255), 1)
        if d["ok"]:
            u = tuple(int(round(x)) for x in d["uv"])
            cv2.drawMarker(v, u, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(v, f"#{d['index']}", (u[0] + 9, u[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    if res.get("ok"):
        txt = f"{res['fit']} RMS {res['rms_mm']:.2f} mm"
        if res.get("loo_max_mm") is not None:
            txt += f"  LOO max {res['loo_max_mm']:.2f} mm"
        cv2.putText(v, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return v


def store(res, extra, source_run):
    return CalibrationStore().save("pen_paper", dict({
        "A": res["A"], "t": res["t"], "fit": res["fit"], "method": "sheet", "rms_mm": res["rms_mm"],
        "loo_max_mm": res["loo_max_mm"], "n_points": res["n_points"], "span_mm": res["span_mm"]}, **extra),
        source_run=source_run,
        notes="紙面 mm -> TCP 指令 XY: C = A·P + t (護貝紙整張畫十字校正);z_touch 為紙面接觸高度")


def find_first_mark(before, img, layout, cmd, H_rough, radius_mm=35.0):
    """修正用:在 img 中找第一個十字,回傳 (指令 - 粗略換算落點) 偏移 (手臂 mm) 與像素位置。"""
    pfa, pfb = PaperFrame(layout=layout), PaperFrame(layout=layout)
    if not (pfa.detect(img) and pfb.detect(before)):
        return None, None
    excl = pen_marks.tag_exclusion_mask(img.shape, pfa, grow_mm=3.0)
    blobs = pen_marks.new_ink_components(before, img, pfa, pfb, min_drop=30, min_area=4, exclude=excl)
    pred = pen_marks.apply_h(np.linalg.inv(H_rough), [cmd])[0]
    k = pen_marks.assign_nearest(blobs, [pred], radius_mm / pfa.mm_per_pixel())[0]
    if k is None:
        return None, None
    uv = blobs[k][1]
    return np.asarray(cmd, float) - pen_marks.apply_h(H_rough, [uv])[0], uv


def reanalyze(run_dir):
    meta, _ = load_run(run_dir)
    with open(os.path.join(run_dir, "data", "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    rd = lambda n: cv2.imdecode(np.fromfile(os.path.join(run_dir, "images", n), np.uint8), cv2.IMREAD_COLOR)  # noqa
    before, after = rd("home_before.png"), rd("home_after.png")
    res = analyze(before, after, os.path.join(run_dir, "inputs", plan["layout"]), plan["points"],
                  np.array(plan["H_rough"]), np.array(plan["offset"]), plan["radius_mm"], plan["surface_h"])
    print(summarize(res))
    cv2.imencode(".jpg", overlay(after, res))[1].tofile(os.path.join(run_dir, "images", "marks_overlay.jpg"))
    return meta, plan, res


# ---------------- 主程式 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pen-loaded", action="store_true")
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--grid", type=int, nargs=2, default=[3, 3], metavar=("COLS", "ROWS"), help="格點數 (預設 3 3)")
    ap.add_argument("--margin", type=float, default=12.0, help="可畫範圍 = tag 內緣再往內縮幾 mm")
    ap.add_argument("--stroke", type=float, default=4.0, help="小十字每筆半長 (mm)")
    ap.add_argument("--hop", type=float, default=5.0, help="點與點之間抬多高平移 (mm)")
    ap.add_argument("--press", type=float, default=0.0, help="比 z_touch 再往下多少 (mm)")
    ap.add_argument("--no-refine", action="store_true", help="不在第一點後回 home 修正 (較快,但其餘點可能偏 ±20 mm)")
    ap.add_argument("--touch-all", action="store_true", help="每一點都人工確認高度 (紙不平時用)")
    ap.add_argument("--surface-h", type=float, default=0.0, help="畫的表面比 tag 紙面高多少 mm (護貝紙本身通常 0)")
    ap.add_argument("--z-floor", type=float, default=None)
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--label", default="")
    ap.add_argument("--analyze", help="重新分析 runs\\<id>")
    args = ap.parse_args()
    if args.analyze:
        meta, plan, res = reanalyze(args.analyze)
        if res.get("ok") and ask("存成新的校正版本? (y/N) ").strip().lower() == "y":
            print("已存:", store(res, {"z_touch": plan["z_touch"], "home": plan["home"], "orientation": plan["orientation"],
                                       "layout": plan["layout"]}, meta["run_id"]))
        return
    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED}")
    cols, rows = args.grid
    if cols * rows < 3 or cols < 1 or rows < 1 or cols * rows > 25:
        sys.exit("--grid 點數需在 3~25")

    calib = wp.load_calibration()
    home, ori = calib["home"], calib["home"][3:]
    z_safe = min(home[2], calib["drop"][2])
    z_floor = args.z_floor if args.z_floor is not None else calib["z_pick"] - 2.0
    H_rough, _ = pen_marks.rough_homography(calib)

    c = ModbusTcpClient(host=wp.ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    tcp = request.Get_TCP_Pose(c)
    ok, dpos, dang = wp.check_home(calib, tcp)
    flag = request.Get_Pose_Flag(c)
    print(f"TCP 與 home 差: 位置 {dpos:.1f} mm, 角度 {dang:.1f} 度;旗標 {flag}")
    if not ok:
        sys.exit("手臂不在 home (python go_home.py)")
    if flag != 1:
        sys.exit("旗標不是 1,請先在教導器清除警報")

    before = grab()
    snap_b, cam_info = LAST_GRAB["snap"], LAST_GRAB["info"]
    pf = PaperFrame(layout=args.layout)
    if not pf.detect(before):
        with RunLog("pen_sheet", label=args.label, params=vars(args)) as log:
            log.save_snapshot("home_before", snap_b, cam_info)
            log.finish("failed", {"reason": "校正紙: " + pf.describe()})
        sys.exit("校正紙: " + pf.describe())

    paper_pts = grid_points(pf, cols, rows, args.margin)
    offset = np.zeros(2)
    robot = paper_to_robot(pf, H_rough, paper_pts, offset, args.surface_h)
    view = before.copy()
    x0, y0, x1, y1 = safe_region(pf, args.margin)
    cv2.polylines(view, [np.int32(pf.paper_to_pix([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))], True, (0, 200, 0), 1)
    for k, p in enumerate(paper_pts):
        u = pf.paper_to_pix([p])[0]
        cv2.drawMarker(view, tuple(int(v) for v in u), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(view, str(k), (int(u[0]) + 7, int(u[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "pen_sheet_plan.png"))
    span = np.ptp(np.array(paper_pts), axis=0)
    print(f"校正紙: {pf.describe()}")
    print(f"{cols}x{rows} = {len(paper_pts)} 點,涵蓋紙面 {span[0]:.0f} x {span[1]:.0f} mm (綠框 = 可畫範圍,tag 內緣再縮 {args.margin:g} mm)")
    print(f"第一點 #0 紙面 ({paper_pts[0][0]:.0f},{paper_pts[0][1]:.0f}) -> 手臂約 ({robot[0][0]:.1f},{robot[0][1]:.1f})"
          f"{'' if args.no_refine else ';畫完第一點會回 home 修正其餘點位'}")
    print(f"安全高度 {z_safe:.1f}、高度下限 {z_floor:.1f}、點間抬 {args.hop:g} mm、十字半長 {args.stroke:g} mm。預覽: pen_sheet_plan.png")
    if not args.go:
        with RunLog("pen_sheet", label=args.label or "dryrun", params=vars(args)) as log:
            log.add_input(args.layout)
            log.add_input(wp.CALIB)
            log.event("tcp", pose=tcp, flag=flag)
            log.save_snapshot("home_before", snap_b, cam_info)
            log.save_image("plan.jpg", view)
            log.save_json("plan_preview", {"paper_pts": paper_pts, "robot_xy": robot.tolist(), "H_rough": H_rough.tolist()})
            log.finish("dry-run")
            print("紀錄 (含深度):", log.path)
        print("\n(dry-run) 沒有移動手臂。加 --go 執行")
        c.close()
        return

    radius_mm = 20.0 if not args.no_refine else 30.0   # 修正後仍有粗略換算的旋轉誤差 (離第一點 200 mm 約 5 mm)
    params = vars(args).copy()
    params.update(z_safe=z_safe, z_floor=z_floor)
    with RunLog("pen_sheet", label=args.label, params=params,
                hardware={"robot_ip": wp.ROBOT_IP, "camera": "RealSense D435 1280x720"}) as log:
        log.add_input(args.layout)
        log.add_input(wp.CALIB)
        log.event("tcp", pose=tcp, flag=flag)
        log.save_snapshot("home_before", snap_b, cam_info)
        arm = Arm(c, log, ori, args.speed)
        points, z_touch = [], None

        def save_plan():
            log.save_json("plan", {"points": points, "paper_targets": paper_pts, "H_rough": H_rough.tolist(),
                                   "offset": np.asarray(offset).tolist(), "radius_mm": radius_mm,
                                   "surface_h": args.surface_h, "z_touch": z_touch, "press": args.press,
                                   "stroke": args.stroke, "home": home, "orientation": ori,
                                   "layout": os.path.basename(args.layout)})

        print("\n[Ctrl+C = 停止並回 home 結束] [空白/Esc/p = 停止並選單] [移動中 o/g = 夾爪開/夾]")
        try:
            x, y = map(float, robot[0])
            ask("\n確認安全 (手放急停旁) 後按 Enter:移到第一點上方安全高度 ")
            arm.move(x, y, z_safe, mov=send.MovP)
            if not args.pen_loaded:
                arm.grip(close=False)
                ask("把白板筆放進夾爪 (筆尖朝下、露出夾爪下方),扶好後按 Enter ")
                for s in (3, 2, 1):
                    print(f"  {s} ... 手離開夾爪")
                    time.sleep(1)
                arm.grip(close=True)
            print("\n=== 第一點:人工下降,剛好畫得出來時按 c ===")
            x, y, z_touch = probe_with_nudge(arm, x, y, z_safe, 20.0, z_floor, z_safe)
            z_draw = z_touch - args.press
            if z_draw < z_floor:
                raise Quit(f"畫的高度低於下限 {z_floor:.2f}")
            arm.move(x, y, z_touch + args.hop)
            small_cross(arm, x, y, z_draw, args.stroke)
            log.event("contact", index=0, robot_xy=[x, y], z=z_draw)
            points.append({"index": 0, "robot_xy": [x, y], "contact_z": z_draw, "paper_target": paper_pts[0]})

            if not args.no_refine:
                print("\n修正:回 home 拍照,找第一個十字")
                arm.lift_to(z_safe)
                arm.move(*home[:3], mov=send.MovP, ori=home[3:])
                img = grab()
                log.save_snapshot("home_first", LAST_GRAB["snap"], LAST_GRAB["info"])
                off, uv = find_first_mark(before, img, args.layout, [x, y], H_rough)
                if off is None:
                    raise Quit("回 home 後找不到第一個十字 (太淡或沒畫到?);可加 --press 0.5 或 --touch-all 重來")
                offset = off
                log.event("refine", offset=offset, mark_uv=uv)
                print(f"  粗略換算偏差 ({offset[0]:+.1f}, {offset[1]:+.1f}) mm,其餘點已修正")
                robot = paper_to_robot(pf, H_rough, paper_pts, offset, args.surface_h)
                bad = [k for k, p in enumerate(paper_pts) if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1)]
                if bad:
                    raise Quit(f"點 {bad} 超出可畫範圍")
                nx, ny = map(float, robot[1])
                arm.move(nx, ny, z_safe, mov=send.MovP)
                arm.move(nx, ny, z_touch + args.hop)

            k = 1
            while k < len(paper_pts):
                x, y = map(float, robot[k])
                try:
                    if args.touch_all:
                        arm.move(x, y, z_touch + 3.0)
                        print(f"\n=== 點 #{k}:人工確認高度 (從上一點高度 +3 mm),c = 剛好畫得出來 ===")
                        x, y, zk = probe_with_nudge(arm, x, y, z_touch + 3.0, 0.5, z_floor, z_safe)
                        z_touch = zk
                        z_draw = zk - args.press
                    else:
                        print(f"  點 #{k}: ({x:.1f}, {y:.1f}) z = {z_draw:.2f}")
                        arm.move(x, y, z_touch + args.hop)
                    small_cross(arm, x, y, z_draw, args.stroke)
                    arm.move(x, y, z_touch + args.hop)
                except (Paused, KeyboardInterrupt) as e:
                    if isinstance(e, KeyboardInterrupt) or str(e) == "ctrl_c":
                        if isinstance(e, KeyboardInterrupt):
                            arm.stop_and_check("ctrl_c")
                        raise Quit("Ctrl+C")
                    ch = point_menu(arm)
                    if ch == "r":
                        arm.lift_to(z_touch + args.hop)
                        continue
                    if ch == "s":
                        k += 1
                        continue
                    if ch == "q":
                        raise Quit("使用者結束")
                    break
                log.event("contact", index=k, robot_xy=[x, y], z=z_draw)
                points.append({"index": k, "robot_xy": [x, y], "contact_z": z_draw, "paper_target": paper_pts[k]})
                k += 1
            arm.lift_to(z_safe)
            arm.move(*home[:3], mov=send.MovP, ori=home[3:])
        except (Quit, Paused, KeyboardInterrupt) as e:
            print("\n結束:", e, "-> 垂直抬起回 home")
            at_home = False
            try:
                arm.lift_to(z_safe)
                arm.move(*home[:3], mov=send.MovP, ori=home[3:])
                at_home = True
            except (Paused, KeyboardInterrupt):
                print("已停止,手臂停在目前位置 (之後用 python go_home.py)")
            save_plan()
            if at_home and points:
                try:
                    grab()
                    log.save_snapshot("home_after", LAST_GRAB["snap"], LAST_GRAB["info"])
                    print("已拍下目前的筆跡,可用 python pen_sheet_calib.py --analyze", log.path)
                except Exception as ee:              # noqa: BLE001
                    print("拍照失敗:", ee)
            log.finish("aborted", {"reason": str(e), "points_done": len(points)})
            c.close()
            return

        save_plan()
        time.sleep(1.0)
        after = grab()
        log.save_snapshot("home_after", LAST_GRAB["snap"], LAST_GRAB["info"])
        res = analyze(before, after, args.layout, points, H_rough, offset, radius_mm, args.surface_h)
        log.save_json("marks_result", res)
        log.save_image("marks_overlay.jpg", overlay(after, res))
        print("\n" + summarize(res))
        if res.get("ok"):
            p = store(res, {"z_touch": z_touch, "home": home, "orientation": ori,
                            "layout": os.path.basename(args.layout)}, os.path.basename(log.path))
            print("已存校正版本:", os.path.relpath(p, HERE))
            log.finish("ok", {"fit": res["fit"], "rms_mm": res["rms_mm"], "loo_max_mm": res["loo_max_mm"],
                              "n_points": res["n_points"]})
            print("\n校正完成。紙不要移動;護貝紙上的筆跡可以擦掉 (不擦也可以)。")
        else:
            print("分析失敗,可用 python pen_sheet_calib.py --analyze", log.path, "重新分析")
            log.finish("ok", {"analysis": "failed", "reason": res.get("reason")})
        print("紀錄:", log.path)
    c.close()


if __name__ == "__main__":
    main()
