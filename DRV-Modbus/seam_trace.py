# -*- coding: utf-8 -*-
"""
沿縫描繪 (工件 B:兩片板縫合塗;T16)。

流程 (--go)
  0. 手臂在 home:拍照 (彩色 + 15 幀深度) -> 找縫 (seam_boards,白紙縫或貼合縫都可)
     -> 深度估板厚 (板子內部 - 周圍一圈紙;也可 --board-h 手動指定) -> 用筆尖校正 (calibrations/pen_paper) 換成手臂路徑
  1. 移到縫起點上方安全高度 -> 自動降到「估計板面 + approach」(預設 3 mm)
  2. 人工確認接觸:每步 0.5 mm 往下,剛好畫得出來按 c (可 x/y 微調;微調量會套用到整條路徑)
  3. 固定這個高度,一次 MovL 從起點畫到終點 (縫是直線,和工件 A 脊線一樣;--segmented = 每 --step-mm 停一次的舊做法)
     -> 抬起 -> 回 home 拍照
  4. 比對前後照片:畫出的筆跡離縫多遠 (平均/最大 mm、涵蓋率)。黑板上看不清楚時會提示目視確認
  5. 筆跡回饋:量到的整體側向偏移記到 calibrations/pen_xy_offset (pen_offset.py,和 A 脊線 / C 罐頭共用),
     下次規劃路徑時自動扣掉 (--no-pen-offset 關閉)

  python seam_trace.py                   # dry-run:找縫、估板厚、印路徑,存 seam_trace_plan.png 與紀錄
  python seam_trace.py --go              # 執行
  python seam_trace.py --go --board-h 5  # 手動指定板厚 (不用深度估計)
  python seam_trace.py --go --reverse    # 從另一端開始畫

按鍵 (本視窗):Ctrl+C = 停止 -> 抬起回 home;移動中 空白/Esc/p = 停止並選單 (r 繼續 / h 回 home 分析 / q 結束);移動中 o/g = 夾爪
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient

import depth_height
import pen_marks
import pen_offset
import seam_boards
import warp_pick as wp
from drv_modbus import send, request
from paper_frame import PaperFrame
from pen_calibrate import Arm, MOVE_ERRORS, Paused, ask, grab, LAST_GRAB, LAYOUT, MAX_SPEED
from pen_quick_calib import Quit, probe_with_nudge
from run_log import RunLog, CalibrationStore

HERE = os.path.dirname(os.path.abspath(__file__))


def depth_mm(snap, info):
    if snap.get("depth") is None or not info or not info.get("depth_scale_m"):
        return None
    mm = snap["depth"].astype(np.float32) * float(info["depth_scale_m"]) * 1000.0
    mm[snap["depth"] == 0] = np.nan
    return mm


DEPTH_CACHE = {}          # estimate_thickness 算出的高度圖 (H, pf),給深度找縫用


def choose_seam(source, color_res, depth_res, max_diff_mm=1.5):
    """依 --seam-source 選縫。回傳 (seam 紙面 mm 或 None, 說明)。
    auto:影像縫為主 (較精準 ~0.3 mm),深度縫獨立確認;影像找不到時用深度縫 (~±1 mm)。"""
    c_ok, d_ok = bool(color_res and color_res.get("ok")), bool(depth_res and depth_res.get("ok"))
    msg = []
    if c_ok and d_ok:
        off, ang = depth_height.line_offset(color_res["seam_paper_mm"], depth_res["seam_paper_mm"])
        msg.append(f"影像縫與深度縫:側向差 {off:+.2f} mm、夾角 {ang:.2f} 度")
        if abs(off) > max_diff_mm or ang > 3.0:
            msg.append(f"  注意:兩者差超過 {max_diff_mm:g} mm 或 3 度,請看預覽圖確認是哪一條對")
    note = ("\n" + "\n".join(msg)) if msg else ""
    if source == "color":
        return (np.asarray(color_res["seam_paper_mm"], float) if c_ok else None), ("使用影像縫" if c_ok else "影像找不到縫") + note
    if source == "depth":
        if d_ok:
            return np.asarray(depth_res["seam_paper_mm"], float), "使用深度縫 (凹陷 %.2f mm)" % depth_res["dip_mm"] + note
        return None, "深度找不到縫:" + (depth_res or {}).get("reason", "沒有深度資料")
    if c_ok:
        return np.asarray(color_res["seam_paper_mm"], float), \
            "使用影像縫" + ("" if d_ok else " (深度沒找到縫,無法交叉確認)") + note
    if d_ok:
        return np.asarray(depth_res["seam_paper_mm"], float), "影像找不到縫 -> 改用深度縫 (凹陷 %.2f mm,精度約 ±1 mm)" % depth_res["dip_mm"]
    return None, "影像與深度都找不到縫"


def estimate_thickness(color, snap, info, layout, pf_boards):
    """回傳 (t_mm 或 None, 說明)。"""
    mm = depth_mm(snap, info)
    if mm is None:
        return None, "沒有深度資料"
    try:
        H, pf, st = depth_height.height_map(color, mm, info["color_intrinsics"], layout)
    except Exception as e:              # noqa: BLE001
        return None, f"高度圖失敗: {e}"
    if not pf_boards:
        return None, "沒有找到板子"
    est = depth_height.board_thickness(H, pf_boards[0]["contour"])
    DEPTH_CACHE.update(H=H, pf=pf)
    if est is None:
        return None, "板子或周圍紙的有效深度太少"
    return est, f"深度估板厚 {est['t_mm']:.2f} mm (分區差 {est['spread_mm'] or 0:.2f} mm,紙面殘差 {st['paper_rms_mm']:.2f} mm)"


def dist_to_polyline(Q, L):
    """點 Q (N,2) 到折線 L (M,2) 的最短距離。"""
    Q, L = np.asarray(Q, float), np.asarray(L, float)
    best = np.full(len(Q), np.inf)
    for a, b in zip(L[:-1], L[1:]):
        ab = b - a
        t = np.clip(((Q - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        best = np.minimum(best, np.linalg.norm(Q - (a + t[:, None] * ab), axis=1))
    return best


def analyze_trace(before, after, layout, seam_mm, board_h, band_mm=6.0):
    """畫出的筆跡相對縫的偏差。"""
    pfa, pfb = PaperFrame(layout=layout), PaperFrame(layout=layout)
    if not (pfa.detect(after) and pfb.detect(before)):
        return {"ok": False, "reason": "前後照片沒有都看到 4 個 tag"}
    # 黑板上用白/銀筆會「變亮」,紅/黑筆會「變暗」:兩個方向都算 (前後照片先用紙面單應對齊)
    Hb2a = pfa.H0inv @ pfb.H0
    gb = cv2.warpPerspective(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), Hb2a, (after.shape[1], after.shape[0]))
    ga = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ga, gb)
    diff = cv2.subtract(diff, cv2.dilate(cv2.absdiff(cv2.GaussianBlur(gb, (3, 3), 0), gb), np.ones((3, 3), np.uint8)))  # 扣掉邊緣的對位誤差
    ink = (diff > 20).astype(np.uint8)
    ink &= (pen_marks.tag_exclusion_mask(after.shape, pfa, grow_mm=3.0) == 0).astype(np.uint8)
    ys, xs = np.nonzero(ink)
    if len(xs) < 20:
        return {"ok": False, "reason": "看不到新的筆跡 (黑色板子上筆跡對比太低?請目視確認)"}
    P = pfa.pix_to_paper(np.stack([xs, ys], 1).astype(np.float32), board_h)
    d = dist_to_polyline(P, seam_mm)
    near = P[d < band_mm]
    if len(near) < 20:
        return {"ok": False, "reason": f"縫附近 {band_mm:g} mm 內沒有筆跡 (畫偏很多或看不到)", "n_ink": int(len(xs))}
    dn = d[d < band_mm]
    # 涵蓋率:沿縫每 2 mm 一段,有筆跡 (離縫 < 2 mm) 的段數比例
    S = np.asarray(seam_mm, float)
    seg_len = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]
    u = (S[-1] - S[0]) / max(np.linalg.norm(S[-1] - S[0]), 1e-9)
    s_ink = (near - S[0]) @ u
    bins = np.arange(0, seg_len[-1] + 2, 2.0)
    covered = np.unique(np.digitize(s_ink[dn < 2.0], bins))
    # 筆跡中心線相對縫的有號偏移 (縫法向)
    nrm = np.array([-u[1], u[0]])
    signed = (near - S[0]) @ nrm
    return {"ok": True, "mean_dev_mm": float(np.median(dn)), "p90_dev_mm": float(np.percentile(dn, 90)),
            "offset_mm": float(np.median(signed)), "coverage": float(min(1.0, len(covered) / max(1, len(bins) - 1))),
            "n_ink_near": int(len(near))}


def summarize(res):
    if not res.get("ok"):
        return "偏差分析:" + res.get("reason", "")
    return (f"筆跡離縫:中位數 {res['mean_dev_mm']:.2f} mm、90% 在 {res['p90_dev_mm']:.2f} mm 內;"
            f"整體側偏 {res['offset_mm']:+.2f} mm;沿縫涵蓋 {res['coverage'] * 100:.0f}%")


def plan_from(color, snap, info, pf, args, A, tcal, z_paper, close=False):
    """由一張 (彩色 + 深度) 快照規劃:板厚、縫 (影像/深度)、手臂路徑。close=True 表示是近距離拍的 (深度較準)。"""
    boards, _ = seam_boards.find_boards(color, pf)
    est, est_msg = estimate_thickness(color, snap, info, args.layout, boards)
    print(("[近距離] " if close else "") + est_msg)
    # 11-10 實測:home (相機離紙 ~520 mm) 深度把 5 mm 板量成 ~3.1 mm;下降 150 mm (~370 mm) 量到 4.89 ± 0.05 mm。
    approach = max(args.approach, 3.0)
    if args.board_h is not None:
        board_h = args.board_h
        print(f"使用手動板厚 {board_h:g} mm" + (f" (深度估 {est['t_mm']:.2f},差 {board_h - est['t_mm']:+.2f})" if est else ""))
        if est and abs(board_h - est["t_mm"]) > (1.0 if close else 2.0):
            print("  注意:手動板厚和深度估計差 " + ("超過 1 mm (近距離深度通常準到 ±0.2 mm,請確認板厚或工件是否放平)"
                                                    if close else "超過 2 mm (home 高度的深度會低估約 2 mm)"))
    elif est is not None and 1.0 <= est["t_mm"] <= 30.0 and (est["spread_mm"] or 0) <= 1.5:
        board_h = round(est["t_mm"], 2)
        approach += 2.0 if close else 4.0                    # 深度可能低估:多留餘裕,由人工往下找
        print(f"  沒有填板厚,暫用深度估計 {board_h:.2f} mm,自動下降多留 {2 if close else 4} mm 餘裕")
    else:
        sys.exit("深度估的板厚不合理 (" + (f"{est['t_mm']:.2f} mm" if est else "無") + "),請用卡尺量板厚後加 --board-h <mm>"
                 " (或在網頁設定填板厚)")
    if board_h <= 0:
        sys.exit("板厚必須 > 0")
    # 高度下限:手動板厚 -> 不低於板面 2 mm;深度估計 -> 不低於紙面 +1 mm
    z_floor = z_paper + (max(1.0, board_h - 2.0) if args.board_h is not None else 1.0)
    res = seam_boards.analyze(color, PaperFrame(layout=args.layout), board_h_mm=board_h, step_mm=args.step_mm)
    dres = None
    if DEPTH_CACHE.get("H") is not None and boards:
        dres = depth_height.depth_seam(DEPTH_CACHE["H"], DEPTH_CACHE["pf"], boards[0]["box"], board_h)
    seam, seam_msg = choose_seam(args.seam_source, res, dres)
    print(seam_msg)
    if seam is None:
        sys.exit("找不到縫:" + (res.get("reason") or "") + " / 深度:" + ((dres or {}).get("reason") or "無深度"))
    if not res["ok"]:                                   # 只有深度縫:疊圖需要的欄位
        res = dict(res, ok=True, mode="depth", seam_paper_mm=seam, board_h_mm=board_h, gap_mm=0.0,
                   length_mm=float(np.linalg.norm(seam[-1] - seam[0])))
    if args.reverse:
        seam = seam[::-1]
    # 筆跡回饋學到的筆尖水平偏移 (紙面 mm):路徑先減掉,筆跡就落在縫上 (同 ridge_trace)
    o_pen, o_note, _ = (np.zeros(2), "已關閉", 0) if args.no_pen_offset else pen_offset.learned()
    if np.any(o_pen != 0) and not close:
        print(f"筆尖偏移修正 ({o_note}):路徑移 ({-o_pen[0]:+.2f}, {-o_pen[1]:+.2f}) mm")
    path = (seam - o_pen) @ A.T + tcal
    z_est = z_paper + board_h
    view = seam_boards.overlay(color, pf, res)
    if dres and dres.get("ok"):
        cv2.polylines(view, [np.int32(np.round(pf.paper_to_pix(dres["seam_paper_mm"], board_h)))], False, (255, 0, 255), 1)
    p0 = pf.paper_to_pix([seam[0]], board_h)[0]
    cv2.circle(view, (int(p0[0]), int(p0[1])), 10, (0, 255, 0), 3)
    cv2.putText(view, "start", (int(p0[0]) + 12, int(p0[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"縫 ({res.get('mode', 'gap')}):長 {res['length_mm']:.1f} mm,{len(seam)} 點;板厚 {board_h:.2f} mm")
    print(f"路徑 (手臂):起點 ({path[0][0]:.1f}, {path[0][1]:.1f}) -> 終點 ({path[-1][0]:.1f}, {path[-1][1]:.1f})")
    print(f"紙面接觸高度 {z_paper:.2f} + 板厚 = 估計板面 {z_est:.2f};自動降到 {z_est + approach:.2f} 後人工確認"
          f";高度下限 {z_floor:.2f}" + ("" if close else "。預覽: seam_trace_plan.png (綠圈 = 起點)"))
    return {"board_h": board_h, "approach": approach, "z_floor": z_floor, "res": res, "dres": dres, "seam": seam,
            "path": path, "z_est": z_est, "view": view, "est": est, "seam_msg": seam_msg, "o_pen": o_pen, "o_note": o_note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--board-h", type=float, default=None, help="板厚 mm (預設用深度估計)")
    ap.add_argument("--approach", type=float, default=3.0, help="自動降到估計板面上方幾 mm,之後人工確認")
    ap.add_argument("--press", type=float, default=0.0, help="接觸高度再往下壓幾 mm 畫線")
    ap.add_argument("--step-mm", type=float, default=2.0, help="路徑點間距 (mm)")
    ap.add_argument("--reverse", action="store_true", help="從縫的另一端開始")
    ap.add_argument("--segmented", action="store_true",
                    help="逐點描繪 (每 --step-mm 停一次,舊做法);預設連續:直線縫一次 MovL 畫完")
    ap.add_argument("--no-pen-offset", action="store_true", help="不套用筆跡回饋學到的筆尖偏移")
    ap.add_argument("--look-dz", type=float, default=150.0,
                    help="--go 時先從 home 垂直下降多少 mm 近距離重新量測 (板厚、縫);0 = 不下降。dry-run 不會下降")
    ap.add_argument("--seam-source", choices=["auto", "color", "depth"], default="auto",
                    help="縫從哪裡來:auto (預設,影像為主、深度確認;影像找不到用深度) / color / depth")
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED}")
    try:
        cal = CalibrationStore().load_latest("pen_paper")
    except FileNotFoundError:
        sys.exit("還沒有筆尖校正 (calibrations/pen_paper)")
    A, tcal = np.array(cal["data"]["A"]), np.array(cal["data"]["t"])
    z_paper = cal["data"].get("z_touch")
    if z_paper is None:
        sys.exit("筆尖校正沒有 z_touch")

    calib = wp.load_calibration()
    home, ori = calib["home"], calib["home"][3:]
    z_safe = min(home[2], calib["drop"][2])

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

    color = grab()
    snap, info = LAST_GRAB["snap"], LAST_GRAB["info"]
    pf = PaperFrame(layout=args.layout)
    if not pf.detect(color):
        sys.exit("校正紙: " + pf.describe())
    P = plan_from(color, snap, info, pf, args, A, tcal, z_paper, close=False)
    board_h, approach, z_floor, res, dres, seam, path, z_est, view, est, seam_msg, o_pen = (
        P[k] for k in ("board_h", "approach", "z_floor", "res", "dres", "seam", "path", "z_est", "view", "est", "seam_msg",
                       "o_pen"))
    cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "seam_trace_plan.png"))

    plan = {"seam_paper_mm": seam.tolist(), "path_robot": path.tolist(), "board_h": board_h, "board_h_source":
            "manual" if args.board_h is not None else "depth", "depth_estimate": est, "z_paper": z_paper, "z_est": z_est,
            "seam_mode": res.get("mode", "gap"), "touching": res.get("touching"), "seam_source": args.seam_source,
            "seam_choice": seam_msg, "pen_offset_applied": np.asarray(o_pen).tolist(), "pen_offset_note": P["o_note"],
            "continuous": not args.segmented, "depth_seam": ({k: v for k, v in dres.items() if k != "seam_paper_mm"} if dres else None), "calibration":
            {"created": cal["created"], "source_run": cal.get("source_run"), "method": cal["data"].get("method")},
            "layout": os.path.basename(args.layout)}
    if not args.go:
        with RunLog("seam_trace", label=args.label or "dryrun", params=vars(args)) as log:
            log.add_input(args.layout)
            log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
            log.save_snapshot("home_before", snap, info)
            log.save_image("plan.jpg", view)
            log.save_json("plan", plan)
            log.save_csv("path_robot_mm", ["x", "y"], path.tolist())
            log.finish("dry-run")
            print("紀錄:", log.path)
        print("\n(dry-run) 沒有移動手臂。確認橘線在縫上、綠圈是起點後加 --go")
        c.close()
        return

    with RunLog("seam_trace", label=args.label, params=vars(args)) as log:
        log.add_input(args.layout)
        log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
        log.event("tcp", pose=tcp, flag=flag)
        log.save_snapshot("home_before", snap, info)
        log.save_image("plan.jpg", view)
        log.save_json("plan", plan)
        arm = Arm(c, log, ori, args.speed)
        print("\n[Ctrl+C = 停止並回 home] [移動中 空白/Esc/p = 停止並選單] [移動中 o/g = 夾爪]")
        done = 0
        try:
            ask("確認筆已夾好、手放急停旁後按 Enter 開始" + (f" (先垂直下降 {args.look_dz:g} mm 近距離量測) " if args.look_dz > 0 else " "))
            if args.look_dz > 0:
                look = [home[0], home[1], home[2] - args.look_dz]
                print(f"\n=== 近距離量測:垂直下降到 z {look[2]:.1f} ===")
                arm.move(*look, ori=home[3:])
                cimg = grab()
                csnap, cinfo = LAST_GRAB["snap"], LAST_GRAB["info"]
                log.save_snapshot("close", csnap, cinfo)
                pfc = PaperFrame(layout=args.layout)
                if not pfc.detect(cimg):
                    print("  近距離看不到 4 個 tag (" + pfc.describe() + "),沿用 home 的規劃")
                else:
                    try:
                        Pc = plan_from(cimg, csnap, cinfo, pfc, args, A, tcal, z_paper, close=True)
                    except SystemExit as e:                    # 近距離規劃失敗 -> 沿用 home 的,不要把手臂丟在半空中
                        Pc = None
                        print("  近距離規劃失敗:", e, " -> 沿用 home 的規劃")
                    if Pc is not None:
                        off, ang = depth_height.line_offset(seam, Pc["seam"])
                        print(f"  近距離 vs home:縫側向差 {off:+.2f} mm、夾角 {ang:.2f} 度;板厚 {Pc['board_h']:.2f} (home {board_h:.2f})")
                        if abs(off) > 3.0 or ang > 5.0:
                            print("  注意:近距離和 home 找到的縫差很多,採用近距離的結果 (較準),畫的時候請注意")
                        board_h, approach, z_floor, res, dres, seam, path, z_est, est, seam_msg = (
                            Pc[k] for k in ("board_h", "approach", "z_floor", "res", "dres", "seam", "path", "z_est", "est",
                                            "seam_msg"))
                        log.save_image("plan_close.jpg", Pc["view"])
                        log.save_json("plan_close", {"seam_paper_mm": seam.tolist(), "path_robot": path.tolist(),
                                                     "board_h": board_h, "depth_estimate": est, "seam_choice": seam_msg,
                                                     "vs_home": {"offset_mm": off, "angle_deg": ang},
                                                     "depth_seam": ({k: v for k, v in dres.items() if k != "seam_paper_mm"}
                                                                    if dres else None)})
                        log.save_csv("path_robot_mm_close", ["x", "y"], path.tolist())
            x0, y0 = map(float, path[0])
            arm.move(x0, y0, z_safe, mov=send.MovP)
            z_start = z_est + approach
            arm.move(x0, y0, z_start)
            print(f"\n=== 已降到估計板面上方 {approach:g} mm (z = {z_start:.2f}),請每步 0.5 mm 往下 (Enter),"
                  f"筆尖『剛好』畫得出來就按 c ===")
            while True:
                x, y, z_c = probe_with_nudge(arm, x0, y0, z_start, 0.5, z_floor, z_safe)
                if z_c < z_start - 0.05:
                    break
                # 一到開始高度就按 c:筆可能早就壓在板子上 (板厚低估),不能直接用
                if ask("  還沒往下就按 c:筆尖可能已經壓進板子。確定筆尖『剛好』碰到?(y = 確定 / Enter = 先往上 2 mm 再找) ")\
                        .strip().lower() == "y":
                    break
                z_start = z_c + 2.0
                arm.move(x, y, z_start)
                x0, y0 = x, y
            shift = np.array([x - float(path[0][0]), y - float(path[0][1])])   # 相對原本起點的微調量
            if np.any(shift != 0):
                print(f"  起點微調 ({shift[0]:+.1f}, {shift[1]:+.1f}) mm,套用到整條路徑")
                path = path + shift
            z_draw = z_c - args.press
            log.event("contact", robot_xy=[x, y], z=z_c, z_est=z_est, shift=shift)
            if abs(z_c - z_est) > 2.5:
                print(f"  注意:實際接觸高度 {z_c:.2f} 和估計板面 {z_est:.2f} 差 {z_c - z_est:+.2f} mm")
            k = 1
            if not args.segmented:
                # 連續描繪:縫是直線,一次 MovL 從起點走到終點,路徑和逐點相同,只是中間不停 (同 ridge_trace)。
                end = [float(path[-1][0]), float(path[-1][1]), z_draw]
                t0 = time.time()
                while True:
                    try:
                        arm.move(*end)
                        break
                    except Paused as e:
                        if str(e) == "ctrl_c":
                            raise
                        ch = ""
                        while ch not in ("r", "q"):
                            ch = ask("暫停:r = 從目前位置繼續畫到終點 / q = 抬起回 home > ").strip().lower()
                        if ch == "q":
                            raise Quit("使用者結束")
                done = k = len(path) - 1
                k += 1
                L = float(np.linalg.norm(np.asarray(end[:2]) - np.asarray(path[0])))
                log.event("continuous_trace", seconds=round(time.time() - t0, 2), length_mm=L)
                print(f"  連續描繪完成:{L:.0f} mm,{time.time() - t0:.1f} 秒")
            while k < len(path):
                try:
                    arm.move(float(path[k][0]), float(path[k][1]), z_draw)
                    done = k
                    k += 1
                except (Paused, KeyboardInterrupt) as e:
                    if isinstance(e, KeyboardInterrupt) or str(e) == "ctrl_c":
                        if isinstance(e, KeyboardInterrupt):
                            arm.stop_and_check("ctrl_c")
                        raise Quit("Ctrl+C")
                    ch = ""
                    while ch not in ("r", "h", "q"):
                        ch = ask(f"暫停 (畫到第 {done}/{len(path) - 1} 點):r = 繼續畫 / h = 回 home 分析已畫部分 / q = 結束 > ").strip().lower()
                        if ch in ("o", "g"):
                            arm.grip(close=ch == "g", delay=0.5)
                    if ch == "r":
                        p = request.Get_TCP_Pose(c)
                        arm.move(p[0], p[1], z_draw + 3)             # 先抬一點,再回到停下的點重新下筆
                        arm.move(float(path[done][0]), float(path[done][1]), z_draw + 3)
                        arm.move(float(path[done][0]), float(path[done][1]), z_draw)
                        continue
                    if ch == "q":
                        raise Quit("使用者結束")
                    break
            log.event("trace_done", points=done, of=len(path) - 1)
            arm.lift_to(z_safe)
            arm.move(*home[:3], mov=send.MovP, ori=home[3:])
        except (Quit, Paused, KeyboardInterrupt) + MOVE_ERRORS as e:
            print("\n停止:", e, "-> 垂直抬起回 home")
            try:
                arm.lift_to(z_safe)
                arm.move(*home[:3], mov=send.MovP, ori=home[3:])
            except (Paused, KeyboardInterrupt) + MOVE_ERRORS:
                print("已停止,手臂停在目前位置 (之後用 python go_home.py)")
            log.finish("aborted", {"reason": str(e), "points_done": done})
            c.close()
            return
        time.sleep(1.0)
        after = grab()
        log.save_snapshot("home_after", LAST_GRAB["snap"], LAST_GRAB["info"])
        r = analyze_trace(color, after, args.layout, seam, board_h)
        log.save_json("trace_result", r)
        view2 = after.copy()
        pf2 = PaperFrame(layout=args.layout)
        if pf2.detect(after):
            cv2.polylines(view2, [np.int32(pf2.paper_to_pix(seam, board_h))], False, (0, 120, 255), 1)
        log.save_image("trace_overlay.jpg", view2)
        print("\n" + summarize(r))
        # 筆跡回饋:縫法向的整體偏移 -> 記錄,下次自動修正 (涵蓋率夠才信)
        if r.get("ok") and r.get("coverage", 0) >= 0.6 and np.isfinite(r.get("offset_mm", np.nan)):
            S_ = np.asarray(seam, float)
            u_ = (S_[-1] - S_[0]) / max(np.linalg.norm(S_[-1] - S_[0]), 1e-9)
            if pen_offset.record([[-u_[1], u_[0]]], [r["offset_mm"]], o_pen, "seam", run=log.path,
                                 extra={"coverage": r["coverage"]}):
                print(f"  回饋:筆跡整體偏 {r['offset_mm']:+.2f} mm 已記錄,下次自動修正筆尖偏移")
        log.finish("ok", {"points_done": done, **{k: r.get(k) for k in ("ok", "mean_dev_mm", "p90_dev_mm", "offset_mm",
                                                                        "coverage", "reason")}})
        print("紀錄:", log.path)
    c.close()


if __name__ == "__main__":
    main()
