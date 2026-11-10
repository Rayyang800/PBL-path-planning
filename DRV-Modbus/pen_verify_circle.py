# -*- coding: utf-8 -*-
"""
驗證「相機視覺 -> 筆尖」轉換:相機在 home 找到紙上的十字 -> 用最新的筆尖校正 (calibrations/pen_paper) 換成手臂座標
-> 繞著它畫一個圓 -> 回 home 拍照 -> 量「畫出的圓心」與「十字中心」差多少 mm。

  python pen_verify_circle.py                       # dry-run:找出紙上所有十字並編號,存 pen_verify_plan.png
  python pen_verify_circle.py --target 2 --go       # 繞 2 號十字畫圓 (預設半徑 8 mm)
  python pen_verify_circle.py --target auto --go    # 自動選最靠近紙中心的十字
  python pen_verify_circle.py --analyze runs\\<id>  # 重新分析

判讀:圓心誤差 ≲ 1 mm 表示視覺 -> 筆尖轉換可用;半徑誤差反映比例與筆跡粗細。
按鍵 (在本視窗):Ctrl+C = 停止 -> 抬起回 home;移動中 空白/Esc/p = 停止並選單;移動中 o/g = 夾爪開/夾
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
from pen_calibrate import Arm, Paused, ask, grab, LAST_GRAB, LAYOUT, MAX_SPEED, safe_region
from pen_quick_calib import Quit
from run_log import RunLog, CalibrationStore, load_run

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- 視覺 ----------------
def find_marks(img, pf, margin=0.0, min_area=15, max_area=3000):
    """在紙上 (tag 內緣縮 margin、排除 tag) 找深色筆跡 (十字、點)。回傳 [{uv, paper_mm, area}],依紙面 y、x 排序。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, y0, x1, y1 = safe_region(pf, margin)
    poly = pf.paper_to_pix([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    roi = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(roi, np.int32(poly), 1)
    roi &= (pen_marks.tag_exclusion_mask(img.shape, pf, grow_mm=3.0) == 0).astype(np.uint8)
    bg = cv2.medianBlur(gray.astype(np.uint8), 31).astype(np.float32)      # 局部背景 (護貝反光不均也能用)
    drop = np.clip(bg - gray, 0, None)
    m = ((drop > 35) & (roi > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        if not (min_area <= a <= max_area) or max(w, h) > 4 * max(1, min(w, h)):
            continue
        ys, xs = np.nonzero(lab == k)
        wt = drop[ys, xs]
        uv = ((xs * wt).sum() / wt.sum(), (ys * wt).sum() / wt.sum())
        out.append({"uv": [float(uv[0]), float(uv[1])], "paper_mm": pf.pix_to_paper([uv])[0].tolist(), "area": int(a)})
    out.sort(key=lambda d: (round(d["paper_mm"][1] / 20), d["paper_mm"][0]))
    for i, d in enumerate(out):
        d["id"] = i
    return out


def fit_circle(pts):
    """最小平方圓擬合 (Kasa)。pts: (N,2) -> (cx, cy, r, rms)。"""
    P = np.asarray(pts, float)
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(c + cx ** 2 + cy ** 2)
    res = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
    return float(cx), float(cy), float(r), float(np.sqrt((res ** 2).mean()))


def analyze(before, after, layout, target_paper, radius_mm):
    pfa, pfb = PaperFrame(layout=layout), PaperFrame(layout=layout)
    if not (pfa.detect(after) and pfb.detect(before)):
        return {"ok": False, "reason": "前後照片沒有都看到 4 個 tag"}
    ink = pen_marks.new_ink_mask(before, after, pfa, pfb, min_drop=30)
    ink &= (pen_marks.tag_exclusion_mask(after.shape, pfa, grow_mm=3.0) == 0).astype(np.uint8) * 255
    # 只看目標附近 (半徑 ±60%) 的圓環,排除其他雜訊
    tp = np.array(target_paper)
    ys, xs = np.nonzero(ink)
    if len(xs) < 30:
        return {"ok": False, "reason": f"找不到新畫的圓 (新筆跡像素 {len(xs)})"}
    P = pfa.pix_to_paper(np.stack([xs, ys], 1).astype(np.float32))
    d = np.hypot(P[:, 0] - tp[0], P[:, 1] - tp[1])
    ring = P[(d > radius_mm * 0.4) & (d < radius_mm * 1.6 + 6)]
    if len(ring) < 30:
        return {"ok": False, "reason": f"目標附近的圓環筆跡太少 ({len(ring)} 點):圓可能偏很多或沒畫出來"}
    cx, cy, r, rms = fit_circle(ring)
    # 涵蓋角度:圓有沒有完整畫出來 (紙不平時可能斷斷續續)
    ang = np.degrees(np.arctan2(ring[:, 1] - cy, ring[:, 0] - cx))
    coverage = len(np.unique(np.floor((ang + 180) / 10))) / 36.0
    err = np.array([cx, cy]) - tp
    return {"ok": True, "circle_center_paper_mm": [cx, cy], "circle_radius_mm": r, "circle_fit_rms_mm": rms,
            "target_paper_mm": tp.tolist(), "center_error_mm": float(np.hypot(*err)), "center_error_xy_mm": err.tolist(),
            "radius_error_mm": float(r - radius_mm), "coverage": float(coverage), "n_ring_px": int(len(ring)),
            "mm_per_px": float(pfa.mm_per_pixel())}


def summarize(res):
    if not res.get("ok"):
        return "分析失敗:" + res.get("reason", "")
    e = res["center_error_mm"]
    verdict = "通過 (≤ 1 mm)" if e <= 1.0 else ("可接受 (≤ 2 mm)" if e <= 2.0 else "偏差偏大 (> 2 mm)")
    return (f"圓心與十字中心差 {e:.2f} mm (紙面 x {res['center_error_xy_mm'][0]:+.2f}, y {res['center_error_xy_mm'][1]:+.2f}) -> {verdict}\n"
            f"畫出的半徑 {res['circle_radius_mm']:.2f} mm (指令 {res['circle_radius_mm'] - res['radius_error_mm']:.1f},差 "
            f"{res['radius_error_mm']:+.2f});圓擬合殘差 {res['circle_fit_rms_mm']:.2f} mm;圓周涵蓋 {res['coverage'] * 100:.0f}%"
            + ("  (圓沒畫完整:紙面可能不平或筆壓不夠)" if res["coverage"] < 0.7 else ""))


def overlay(img, pf, res, target):
    v = img.copy()
    t = target["uv"]
    cv2.drawMarker(v, (int(t[0]), int(t[1])), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    if res.get("ok"):
        c = pf.paper_to_pix([res["circle_center_paper_mm"]])[0]
        e = pf.paper_to_pix([[res["circle_center_paper_mm"][0] + res["circle_radius_mm"], res["circle_center_paper_mm"][1]]])[0]
        cv2.circle(v, (int(c[0]), int(c[1])), int(np.hypot(*(e - c))), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(v, (int(c[0]), int(c[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(v, f"center err {res['center_error_mm']:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return v


def choose_target(marks, target, pf):
    if not marks:
        return None
    if target in (None, "", "auto"):
        c = np.array([pf.W / 2, pf.H / 2])
        return min(marks, key=lambda d: np.hypot(*(np.array(d["paper_mm"]) - c)))
    k = int(target)
    return next((d for d in marks if d["id"] == k), None)


# ---------------- 主程式 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--target", default="auto", help="十字編號 (dry-run 的預覽圖上有編號) 或 auto = 最靠近紙中心")
    ap.add_argument("--radius", type=float, default=8.0, help="圓半徑 (mm)")
    ap.add_argument("--segments", type=int, default=24, help="圓分成幾段直線 (MovL)")
    ap.add_argument("--dz", type=float, default=0.0, help="畫圓高度 = 校正的 z_touch - dz (紙較低處可給 0.5~1)")
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--label", default="")
    ap.add_argument("--analyze", help="重新分析 runs\\<id>")
    args = ap.parse_args()
    if args.analyze:
        rd = lambda n: cv2.imdecode(np.fromfile(os.path.join(args.analyze, "images", n), np.uint8), cv2.IMREAD_COLOR)  # noqa
        with open(os.path.join(args.analyze, "data", "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
        res = analyze(rd("home_before.png"), rd("home_after.png"), os.path.join(args.analyze, "inputs", plan["layout"]),
                      plan["target"]["paper_mm"], plan["radius"])
        print(summarize(res))
        return
    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED}")
    try:
        cal = CalibrationStore().load_latest("pen_paper")
    except FileNotFoundError:
        sys.exit("還沒有筆尖校正 (calibrations/pen_paper),請先做 pen_sheet_calib / pen_quick_calib")
    A, t = np.array(cal["data"]["A"]), np.array(cal["data"]["t"])
    z_touch = cal["data"].get("z_touch")
    if z_touch is None:
        sys.exit("校正檔沒有 z_touch")
    z_draw = z_touch - args.dz

    calib = wp.load_calibration()
    home, ori = calib["home"], calib["home"][3:]
    z_safe = min(home[2], calib["drop"][2])
    z_floor = calib["z_pick"] - 2.0
    if z_draw < z_floor:
        sys.exit(f"畫圓高度 {z_draw:.2f} 低於下限 {z_floor:.2f}")

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
        sys.exit("校正紙: " + pf.describe())
    marks = find_marks(before, pf)
    view = before.copy()
    for d in marks:
        u = d["uv"]
        cv2.circle(view, (int(u[0]), int(u[1])), 12, (0, 200, 255), 2)
        cv2.putText(view, str(d["id"]), (int(u[0]) + 12, int(u[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    target = choose_target(marks, args.target, pf)
    print(f"校正: {cal['created']} ({cal['data'].get('method')}/{cal['data'].get('fit')},留一法最大 "
          f"{cal['data'].get('loo_max_mm') or 0:.2f} mm),z_touch {z_touch:.2f},畫圓高度 {z_draw:.2f}")
    print(f"紙上找到 {len(marks)} 個筆跡:")
    for d in marks:
        print(f"  #{d['id']}: 紙面 ({d['paper_mm'][0]:.1f}, {d['paper_mm'][1]:.1f}) mm,面積 {d['area']} px")
    if target is None:
        cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "pen_verify_plan.png"))
        sys.exit(f"找不到目標十字 (--target {args.target});看 pen_verify_plan.png 的編號")
    P = np.array(target["paper_mm"])
    center = A @ P + t
    pts = [A @ (P + args.radius * np.array([np.cos(a), np.sin(a)])) + t
           for a in np.linspace(0, 2 * np.pi, args.segments + 1)]
    x0, y0, x1, y1 = safe_region(pf, -10.0)          # 圓可以稍微超出格點範圍,但不能出紙
    for a in np.linspace(0, 2 * np.pi, 13):
        q = P + (args.radius + 2) * np.array([np.cos(a), np.sin(a)])
        if not (x0 <= q[0] <= x1 and y0 <= q[1] <= y1):
            sys.exit(f"圓會超出可畫範圍,換一個比較靠內側的十字或縮小 --radius")
    ring = pf.paper_to_pix([P + args.radius * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 60)])
    cv2.polylines(view, [np.int32(ring)], True, (255, 0, 255), 2)
    cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "pen_verify_plan.png"))
    print(f"\n目標 #{target['id']} 紙面 ({P[0]:.1f}, {P[1]:.1f}) -> 手臂圓心 ({center[0]:.1f}, {center[1]:.1f}),"
          f"半徑 {args.radius:g} mm,{args.segments} 段。預覽 (紫色圓): pen_verify_plan.png")

    plan = {"target": target, "radius": args.radius, "segments": args.segments, "center_robot": center.tolist(),
            "path_robot": [p.tolist() for p in pts], "z_draw": z_draw, "calibration": {"created": cal["created"],
            "source_run": cal.get("source_run"), "A": A.tolist(), "t": t.tolist()}, "layout": os.path.basename(args.layout)}
    if not args.go:
        with RunLog("pen_verify", label=args.label or "dryrun", params=vars(args)) as log:
            log.add_input(args.layout)
            log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
            log.save_snapshot("home_before", snap_b, cam_info)
            log.save_image("plan.jpg", view)
            log.save_json("plan", plan)
            log.finish("dry-run")
            print("紀錄:", log.path)
        print("\n(dry-run) 沒有移動手臂。確認紫色圓繞著要的十字後,加 --target <編號> --go")
        c.close()
        return

    with RunLog("pen_verify", label=args.label, params=vars(args)) as log:
        log.add_input(args.layout)
        log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
        log.event("tcp", pose=tcp, flag=flag)
        log.save_snapshot("home_before", snap_b, cam_info)
        log.save_image("plan.jpg", view)
        log.save_json("plan", plan)
        arm = Arm(c, log, ori, args.speed)
        print("\n[Ctrl+C = 停止並回 home] [移動中 空白/Esc/p = 停止] [移動中 o/g = 夾爪]")
        try:
            ask("確認筆已夾好、手放急停旁後按 Enter 開始 ")
            sx, sy = map(float, pts[0])
            arm.move(sx, sy, z_safe, mov=send.MovP)
            arm.move(sx, sy, z_draw + args.hop)
            arm.move(sx, sy, z_draw)
            for p in pts[1:]:
                arm.move(float(p[0]), float(p[1]), z_draw)
            arm.move(*map(float, pts[-1]), z_draw + args.hop)
            arm.lift_to(z_safe)
            arm.move(*home[:3], mov=send.MovP, ori=home[3:])
        except (Quit, Paused, KeyboardInterrupt) as e:
            print("\n停止:", e, "-> 垂直抬起回 home")
            try:
                arm.lift_to(z_safe)
                arm.move(*home[:3], mov=send.MovP, ori=home[3:])
            except (Paused, KeyboardInterrupt):
                print("已停止,手臂停在目前位置 (之後用 python go_home.py)")
            log.finish("aborted", {"reason": str(e)})
            c.close()
            return
        time.sleep(1.0)
        after = grab()
        log.save_snapshot("home_after", LAST_GRAB["snap"], LAST_GRAB["info"])
        res = analyze(before, after, args.layout, P.tolist(), args.radius)
        log.save_json("verify_result", res)
        pf.detect(after)
        log.save_image("verify_overlay.jpg", overlay(after, pf, res, target))
        print("\n" + summarize(res))
        log.finish("ok" if res.get("ok") else "failed", {k: res.get(k) for k in ("ok", "center_error_mm", "radius_error_mm",
                                                                                "coverage", "reason")})
        print("紀錄:", log.path)
    c.close()


if __name__ == "__main__":
    main()
