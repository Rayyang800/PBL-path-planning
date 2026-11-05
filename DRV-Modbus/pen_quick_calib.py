# -*- coding: utf-8 -*-
"""
快速筆尖校正 (人只找一次高度,之後不用相機,最後拍一張照片算)。

流程 (--go)
  0. 手臂在 home:拍「之前」照片 (校正紙 4 個 AprilTag 要看得到)
  1. 移到第一點上方安全高度;(可選) 開夾爪裝筆
  2. 人工逐步下降,可用 x/y 微調位置讓筆尖落在工件膠帶上;剛好點出顏色時按 c -> 記下 z_touch
  3. 假設工作面等高:之後每一點都用 z_touch,不拍照。抬 hop mm -> 水平移 step mm -> 下到 z_touch 點一下 -> 抬起
     點的排列:第一點 + 手臂 +X / +Y / -X / -Y 各偏 step (全部在第一點 step 範圍內,不會跑出工件)
  4. 垂直抬到安全高度 -> 回 home -> 拍「之後」照片 -> 前後相減找墨點 -> 求「紙面 mm -> TCP 指令 XY」(旋轉 + 平移,比例 1:1)
     存 calibrations/pen_paper/ (method = quick)

  python pen_quick_calib.py                         # dry-run:印計畫、存 pen_quick_plan.png
  python pen_quick_calib.py --go --pen-loaded       # 執行
  python pen_quick_calib.py --go --pen-loaded --start-paper 120 90 --surface-h 5   # 第一點在紙面 (120,90),工件高 5 mm
  python pen_quick_calib.py --analyze runs\\<id>    # 重新分析某次紀錄

按鍵:Ctrl+C = 停止 -> 垂直抬起 -> 回 home 結束;空白/Esc/p = 停止並選單;移動中 o/g = 夾爪開/夾
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
from pen_calibrate import Arm, Paused, ask, grab, LAST_GRAB, LAYOUT, MAX_SPEED
from run_log import RunLog, CalibrationStore, load_run

HERE = os.path.dirname(os.path.abspath(__file__))
OFFSET_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]          # 手臂 +X, +Y, -X, -Y

PROBE_HELP = ("  Enter=往下一步  數字(如 2 / 0.5)=改步距並往下  u=往上一步\n"
              "  x 2 / y -3 = 水平微調 (mm,每次最多 5)  o / g = 夾爪開 / 夾\n"
              "  c = 筆尖剛好點出顏色 (之後所有點都用這個高度)  q = 結束")


class Quit(Exception):
    pass


def probe_with_nudge(arm, x, y, z, step, z_floor, z_safe):
    """人工下降 + 水平微調。回傳 (x, y, z_touch)。"""
    print(PROBE_HELP)
    while True:
        s = ask(f"    位置 ({x:.1f}, {y:.1f})  z = {z:.2f} (步距 {step:g}) > ").strip().lower()
        if s == "c":
            return x, y, z
        if s == "q":
            raise Quit("使用者結束")
        if s in ("o", "g"):
            arm.grip(close=s == "g", delay=0.5)
            continue
        if s[:1] in ("x", "y") and len(s.split()) == 2:
            try:
                d = float(s.split()[1])
            except ValueError:
                print(PROBE_HELP)
                continue
            if abs(d) > 5:
                print("    微調每次最多 5 mm")
                continue
            if z < z_safe - 1:
                arm.move(x, y, z + 2.0)            # 先離開紙面 2 mm 再水平移,避免拖出一條線
            x, y = (x + d, y) if s[0] == "x" else (x, y + d)
            arm.move(x, y, z + 2.0 if z < z_safe - 1 else z)
            if z < z_safe - 1:
                arm.move(x, y, z)
            continue
        if s == "u":
            nz = min(z + step, z_safe)
        else:
            if s:
                try:
                    step = float(s)
                except ValueError:
                    print(PROBE_HELP)
                    continue
                if not 0.1 <= step <= 30:
                    print("    步距需在 0.1~30 mm")
                    continue
            nz = z - step
        if nz < z_floor:
            print(f"    拒絕:{nz:.2f} 低於高度下限 {z_floor:.2f} (可用 --z-floor 調整)")
            continue
        arm.move(x, y, nz)
        z = nz


def rough_rotation(pf, H_rough, center_paper):
    """用粗略換算 (紙面 -> 像素 -> 手臂) 在中心附近的方向,當作旋轉的交叉檢查。回傳 2x2 正交矩陣。"""
    p = np.array(center_paper, float)
    pts = pf.paper_to_pix([p, p + [10, 0], p + [0, 10]])
    r = pen_marks.apply_h(H_rough, pts)
    J = np.stack([(r[1] - r[0]) / 10, (r[2] - r[0]) / 10], 1)
    U, _, Vt = np.linalg.svd(J)
    return U @ Vt


def angle_of(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def analyze(before, after, layout, points, H_rough, step, surface_h, roi_mm=None):
    """points[0] = 第一點,其後依 OFFSET_DIRS。回傳結果 dict。"""
    pfa, pfb = PaperFrame(layout=layout), PaperFrame(layout=layout)
    if not (pfa.detect(after) and pfb.detect(before)):
        return {"ok": False, "reason": "前後照片沒有都看到 4 個 tag:" + pfa.describe() + " / " + pfb.describe()}
    ink, drop = pen_marks.new_ink_mask(before, after, pfa, pfb, min_drop=30, return_drop=True)
    c_cmd = np.array(points[0]["robot_xy"])
    c_pix = pen_marks.apply_h(np.linalg.inv(H_rough), [c_cmd])[0]
    roi_mm = roi_mm or (1.5 * step + 30.0)                 # 粗略換算可能差 ±20 mm
    r_px = int(roi_mm / pfa.mm_per_pixel())
    roi = np.zeros(ink.shape, np.uint8)
    cv2.circle(roi, tuple(int(v) for v in c_pix), r_px, 255, -1)
    m = ((ink > 0) & (roi > 0)).astype(np.uint8)          # 不做形態學 (偶數核會讓中心偏 1 px)
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs = []
    for k in range(1, n):
        if st[k, 4] < 3:
            continue
        ys, xs = np.nonzero(lab == k)
        w = drop[ys, xs]
        blobs.append((int(st[k, 4]), np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])))
    blobs.sort(key=lambda b: -b[0])
    blobs = blobs[:len(points) + 3]
    uv = pen_marks.match_pattern([b[1] for b in blobs], c_cmd,
                                 [np.array(p["robot_xy"]) - c_cmd for p in points[1:]], H_rough)
    det = [{"index": k, "ok": u is not None, "uv": u, "predicted_uv": c_pix.tolist(),
            "reason": "" if u is not None else "找不到對應的墨點"} for k, u in enumerate(uv)]
    good = [d for d in det if d["ok"]]
    for d in good:
        d["paper_mm"] = pfa.pix_to_paper([d["uv"]], surface_h)[0].tolist()
    res = {"ok": False, "detections": det, "n_blobs": len(blobs), "roi_mm": roi_mm,
           "paper_frame": pfa.describe(), "mm_per_px": pfa.mm_per_pixel()}
    if len(good) < 3:
        res["reason"] = f"只對應到 {len(good)} 個墨點 (偵測到 {len(blobs)} 個),至少要 3 個"
        return res
    P = [d["paper_mm"] for d in good]
    C = [points[d["index"]]["robot_xy"] for d in good]
    R, t, resid = pen_marks.fit_rigid(P, C)
    err = np.linalg.norm(resid, axis=1)
    rms = float(np.sqrt((err ** 2).mean()))
    Rr = rough_rotation(pfa, H_rough, P[0])
    dang = (angle_of(R) - angle_of(Rr) + 180) % 360 - 180
    res.update({"ok": True, "A": R.tolist(), "t": t.tolist(), "rms_mm": rms, "max_mm": float(err.max()),
                "residual_mm": err.tolist(), "n_points": len(good), "rotation_deg": angle_of(R),
                "mirrored": bool(np.linalg.det(R) < 0), "rough_rotation_deg": angle_of(Rr),
                "rotation_vs_rough_deg": float(dang), "mirror_matches_rough": bool(np.linalg.det(R) * np.linalg.det(Rr) > 0),
                # 旋轉誤差約 rms/step (rad) -> 離第一點 100 mm 處的估計誤差
                "est_err_at_100mm": float(rms + 100.0 * rms / max(step, 1e-6)),
                "first_point_paper_mm": P[0]})
    return res


def summarize(res):
    lines = [res.get("paper_frame", "")]
    for d in res.get("detections", []):
        lines.append(f"  點 #{d['index']}: " + (f"像素 ({d['uv'][0]:.1f},{d['uv'][1]:.1f}) -> 紙面 "
                     f"({d['paper_mm'][0]:.1f},{d['paper_mm'][1]:.1f}) mm" if d["ok"] else d["reason"]))
    if not res.get("ok"):
        lines.append("分析失敗:" + res.get("reason", ""))
        return "\n".join(lines)
    lines.append(f"紙面 -> 手臂:旋轉 {res['rotation_deg']:.1f}°、鏡像 {res['mirrored']};"
                 f"殘差 RMS {res['rms_mm']:.2f} mm、最大 {res['max_mm']:.2f} mm ({res['n_points']} 點)")
    lines.append(f"與舊 ArUco 粗略換算的旋轉差 {res['rotation_vs_rough_deg']:+.1f}°"
                 + ("" if abs(res["rotation_vs_rough_deg"]) < 5 and res["mirror_matches_rough"]
                    else "  <-- 差很多,請檢查墨點對應 (marks_overlay.jpg)"))
    lines.append(f"估計誤差:第一點附近 ≈ {res['rms_mm']:.1f} mm,離第一點 100 mm ≈ {res['est_err_at_100mm']:.1f} mm")
    return "\n".join(lines)


def overlay(img, res, points):
    v = img.copy()
    for d in res.get("detections", []):
        if d["ok"]:
            u = tuple(int(round(x)) for x in d["uv"])
            cv2.circle(v, u, 7, (0, 0, 255), 2)
            cv2.putText(v, f"#{d['index']}", (u[0] + 8, u[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if res.get("ok"):
        A, t = np.array(res["A"]), np.array(res["t"])
        cv2.putText(v, f"RMS {res['rms_mm']:.2f} mm  rot {res['rotation_deg']:.1f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return v


def store(res, meta_extra, source_run):
    return CalibrationStore().save("pen_paper", dict({
        "A": res["A"], "t": res["t"], "rms_mm": res["rms_mm"], "method": "quick",
        "est_err_at_100mm": res["est_err_at_100mm"], "first_point_paper_mm": res["first_point_paper_mm"]},
        **meta_extra), source_run=source_run,
        notes="紙面 mm -> TCP 指令 XY: C = A·P + t (A 為旋轉/鏡像,比例 1:1);z_touch 為工作面接觸高度")


def reanalyze(run_dir):
    meta, _ = load_run(run_dir)
    with open(os.path.join(run_dir, "data", "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    rd = lambda n: cv2.imdecode(np.fromfile(os.path.join(run_dir, "images", n), np.uint8), cv2.IMREAD_COLOR)  # noqa
    before, after = rd("home_before.png"), rd("home_after.png")
    res = analyze(before, after, os.path.join(run_dir, "inputs", plan["layout"]), plan["points"],
                  np.array(plan["H_rough"]), plan["step"], plan["surface_h"])
    print(summarize(res))
    cv2.imencode(".jpg", overlay(after, res, plan["points"]))[1].tofile(os.path.join(run_dir, "images", "marks_overlay.jpg"))
    return meta, plan, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pen-loaded", action="store_true")
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--step", type=float, default=10.0, help="其他點離第一點多遠 (mm,手臂 X/Y 方向)")
    ap.add_argument("--hop", type=float, default=3.0, help="點與點之間抬多高 (mm)")
    ap.add_argument("--press", type=float, default=0.0, help="比 z_touch 再往下多少 (mm;預設 0 = 和第一點一樣)")
    ap.add_argument("--dwell", type=float, default=0.5, help="每點停留秒數")
    ap.add_argument("--start-paper", type=float, nargs=2, metavar=("X", "Y"), help="第一點的紙面座標 mm (預設紙中心)")
    ap.add_argument("--surface-h", type=float, default=0.0, help="點的表面比校正紙高多少 mm (點在工件上時填工件厚度)")
    ap.add_argument("--z-floor", type=float, default=None)
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--label", default="")
    ap.add_argument("--analyze", help="重新分析 runs\\<id>")
    args = ap.parse_args()
    if args.analyze:
        meta, plan, res = reanalyze(args.analyze)
        if res.get("ok") and ask("存成新的校正版本? (y/N) ").strip().lower() == "y":
            print("已存:", store(res, {"z_touch": plan["z_touch"], "surface_h": plan["surface_h"],
                                       "home": plan["home"], "orientation": plan["orientation"]}, meta["run_id"]))
        return
    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED}")
    if not 3 <= args.step <= 30:
        sys.exit("--step 需在 3~30 mm")

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

    before = grab()                                   # 彩色 + 15 張深度中位數
    snap_b, cam_info = LAST_GRAB["snap"], LAST_GRAB["info"]
    pf = PaperFrame(layout=args.layout)
    if not pf.detect(before):
        with RunLog("pen_quick", label=args.label, params=vars(args)) as log:   # 失敗也留下照片與深度
            log.save_snapshot("home_before", snap_b, cam_info)
            log.finish("failed", {"reason": "校正紙: " + pf.describe()})
        sys.exit("校正紙: " + pf.describe())
    start_paper = args.start_paper or (pf.W / 2, pf.H / 2)
    x0, y0 = pen_marks.apply_h(H_rough, pf.paper_to_pix([start_paper], args.surface_h))[0]
    plan_xy = [(x0, y0)] + [(x0 + dx * args.step, y0 + dy * args.step) for dx, dy in OFFSET_DIRS]

    view = before.copy()
    Hinv = np.linalg.inv(H_rough)
    for k, (x, y) in enumerate(plan_xy):
        u = pen_marks.apply_h(Hinv, [(x, y)])[0]
        cv2.circle(view, tuple(int(v) for v in u), 5, (0, 0, 255), -1)
        cv2.putText(view, str(k), (int(u[0]) + 6, int(u[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    rr = int((args.step + 20) / pf.mm_per_pixel())
    u0 = pen_marks.apply_h(Hinv, [plan_xy[0]])[0]
    cv2.circle(view, tuple(int(v) for v in u0), rr, (0, 200, 255), 1)
    cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "pen_quick_plan.png"))
    print(f"校正紙: {pf.describe()}")
    print(f"第一點 (預估,可能差 ±20 mm,現場用 x/y 微調):紙面 ({start_paper[0]:.0f},{start_paper[1]:.0f}) -> 手臂 "
          f"({x0:.1f},{y0:.1f});其餘 4 點在手臂 ±X/±Y {args.step:g} mm。橘色圈 = 第一點 ±{args.step + 20:g} mm 範圍")
    print(f"安全高度 {z_safe:.1f}、高度下限 {z_floor:.1f}、點間抬 {args.hop:g} mm、下壓 {args.press:g} mm。預覽: pen_quick_plan.png")
    if not args.go:
        with RunLog("pen_quick", label=args.label or "dryrun", params=vars(args)) as log:
            log.add_input(args.layout)
            log.add_input(wp.CALIB)
            log.event("tcp", pose=tcp, flag=flag)
            log.save_snapshot("home_before", snap_b, cam_info)
            log.save_image("plan.jpg", view)
            log.save_json("plan_preview", {"plan_xy": np.asarray(plan_xy).tolist(), "start_paper": list(start_paper),
                                           "H_rough": H_rough.tolist(), "z_safe": z_safe, "z_floor": z_floor})
            log.finish("dry-run")
            print("紀錄 (含深度):", log.path)
        print("\n(dry-run) 沒有移動手臂。工件要大於第一點 ±(step + 5) mm 的範圍。加 --go 執行")
        c.close()
        return

    params = vars(args).copy()
    params.update(z_safe=z_safe, z_floor=z_floor)
    with RunLog("pen_quick", label=args.label, params=params,
                hardware={"robot_ip": wp.ROBOT_IP, "camera": "RealSense D435 1280x720"}) as log:
        log.add_input(args.layout)
        log.add_input(wp.CALIB)
        log.event("tcp", pose=tcp, flag=flag)
        log.save_snapshot("home_before", snap_b, cam_info)
        arm = Arm(c, log, ori, args.speed)
        points, z_touch = [], None
        print("\n[Ctrl+C = 停止並回 home 結束] [空白/Esc/p = 停止並選單] [移動中 o/g = 夾爪開/夾]")
        try:
            ask("\n確認安全 (手放急停旁) 後按 Enter:移到第一點上方安全高度 ")
            arm.move(x0, y0, z_safe, mov=send.MovP)
            if not args.pen_loaded:
                arm.grip(close=False)
                ask("把白板筆放進夾爪 (筆尖朝下、露出夾爪下方),扶好後按 Enter ")
                for s in (3, 2, 1):
                    print(f"  {s} ... 手離開夾爪")
                    time.sleep(1)
                arm.grip(close=True)
            print("\n=== 第一點:人工下降,剛好點出顏色時按 c ===")
            x0, y0, z_touch = probe_with_nudge(arm, x0, y0, z_safe, 20.0, z_floor, z_safe)
            if z_touch - args.press < z_floor:
                raise Quit(f"點的高度低於下限 {z_floor:.2f}")
            time.sleep(args.dwell)
            log.event("contact", index=0, robot_xy=[x0, y0], z=z_touch)
            points.append({"index": 0, "robot_xy": [x0, y0], "contact_z": z_touch})
            arm.move(x0, y0, z_touch + args.hop)
            k = 1
            while k <= len(OFFSET_DIRS):
                dx, dy = OFFSET_DIRS[k - 1]
                x, y = x0 + dx * args.step, y0 + dy * args.step
                try:
                    print(f"  點 #{k}: ({x:.1f}, {y:.1f}) z = {z_touch - args.press:.2f}")
                    arm.move(x, y, z_touch + args.hop)
                    arm.move(x, y, z_touch - args.press)
                    time.sleep(args.dwell)
                    arm.move(x, y, z_touch + args.hop)
                except (Paused, KeyboardInterrupt) as e:
                    if isinstance(e, KeyboardInterrupt) or str(e) == "ctrl_c":
                        if isinstance(e, KeyboardInterrupt):
                            arm.stop_and_check("ctrl_c")
                        raise Quit("Ctrl+C")
                    ch = ""
                    while ch not in ("r", "s", "h", "q"):
                        ch = ask("暫停:r = 重做這一點 / s = 跳過 / h = 回 home 分析已完成的點 / q = 結束 > ").strip().lower()
                    if ch == "r":
                        continue
                    if ch == "s":
                        k += 1
                        continue
                    if ch == "q":
                        raise Quit("使用者結束")
                    break                               # h:用已完成的點分析
                log.event("contact", index=k, robot_xy=[x, y], z=z_touch - args.press)
                points.append({"index": k, "robot_xy": [x, y], "contact_z": z_touch - args.press})
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
            log.save_json("plan", {"points": points, "H_rough": H_rough.tolist(), "step": args.step,
                                   "surface_h": args.surface_h, "z_touch": z_touch, "press": args.press,
                                   "home": home, "orientation": ori, "layout": os.path.basename(args.layout)})
            if at_home and points:                    # 已點的點也拍下來,之後可 --analyze 離線分析
                try:
                    grab()
                    log.save_snapshot("home_after", LAST_GRAB["snap"], LAST_GRAB["info"])
                except Exception as ee:              # noqa: BLE001
                    print("拍照失敗:", ee)
            log.finish("aborted", {"reason": str(e), "points_done": len(points)})
            c.close()
            return

        plan = {"points": points, "H_rough": H_rough.tolist(), "step": args.step, "surface_h": args.surface_h,
                "z_touch": z_touch, "press": args.press, "home": home, "orientation": ori,
                "layout": os.path.basename(args.layout)}
        log.save_json("plan", plan)
        time.sleep(1.0)
        after = grab()
        log.save_snapshot("home_after", LAST_GRAB["snap"], LAST_GRAB["info"])
        res = analyze(before, after, args.layout, points, H_rough, args.step, args.surface_h)
        log.save_json("marks_result", res)
        log.save_image("marks_overlay.jpg", overlay(after, res, points))
        print("\n" + summarize(res))
        if res.get("ok"):
            p = store(res, {"z_touch": z_touch, "surface_h": args.surface_h, "home": home, "orientation": ori,
                            "layout": os.path.basename(args.layout)}, os.path.basename(log.path))
            print("已存校正版本:", os.path.relpath(p, HERE))
            log.finish("ok", {"rms_mm": res["rms_mm"], "n_points": res["n_points"],
                              "rotation_vs_rough_deg": res["rotation_vs_rough_deg"]})
        else:
            print("分析失敗,可用 python pen_quick_calib.py --analyze", log.path, "重新分析")
            log.finish("ok", {"analysis": "failed", "reason": res.get("reason")})
        print("紀錄:", log.path)
    c.close()


if __name__ == "__main__":
    main()
