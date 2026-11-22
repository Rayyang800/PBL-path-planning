# -*- coding: utf-8 -*-
"""
筆尖 3D 校正 (pivot / 工具中心點校正):讓筆傾斜時也知道筆尖在哪裡。

原理
  筆尖相對 TCP 的位置 p (工具座標) 是固定的。筆尖碰同一個點 X、用不同姿態 R_i:
      TCP_i + R_i · p = X        (i = 0..n)
  相減消掉 X:  TCP_i - TCP_0 = -(R_i - R_0) · p  -> 最小平方解 p (需要繞兩個不同軸的傾斜)
  之後任何姿態 R:要筆尖到 T,TCP = T - R·p。

流程 (--go)
  1. 移到校正紙上一個空白處 (--at,預設紙面 (148, 185) mm) 上方,垂直往下人工找接觸 (c) -> 在紙上點一個點
  2. 抬到轉動高度,依序傾斜 (預設繞手臂 x、y 軸 ±12°):用目前的 p 估計 TCP,先停在估計位置上方 --approach (40) mm,
     你用 Enter (往下)、x / y (水平微調)、u (往上) 讓筆尖剛好碰到那個點的中心,按 c
  3. 每多一個姿態就重算 p;最後存 calibrations/pen_tip/ (p、各點殘差、歐拉角慣例)

安全:所有轉動與水平移動都在「垂直時筆尖離紙 --rot-height (100 mm)」的高度做,只在目標正上方垂直下降。
     (11-18 事故:原本抬 30 mm 就轉回垂直,但傾斜轉回垂直時筆尖會往下掉約 38 mm,筆壓進紙約 8 mm。
      「繞 TCP 轉筆尖只會變高」只對垂直 -> 傾斜成立,反方向會往下掉同樣多。)
     傾斜後先停在估計筆尖高度上方 --approach (預設 40 mm),由人工往下 (預設每步 2 mm,接近時輸入 0.5 改小步)。
     第一次 p 未知,先用 --guess-len (預設 260 mm,由相機手眼估計) 估計;估錯的部分用 x / y 微調補,第 3 點起會用實測重算。

  python pen_pivot.py --go
  python pen_pivot.py --go --tilt 15 --approach 30
按鍵:Enter 往下 / 數字 改步距 / u 往上 / x 2、y -3 水平微調 / c 對準了 / q 結束;Ctrl+C = 停止 -> 抬起回 home
"""
import argparse
import json
import os
import sys

import numpy as np

import scan_tilt as stl
from run_log import RunLog, CalibrationStore

HERE = os.path.dirname(os.path.abspath(__file__))
CONV = "xyz"          # 11-11 試轉實測:台達 DRV 的 (rx, ry, rz) 是外旋 xyz (旋轉誤差 0.6°,另一種 16°)


def solve_tip(recs):
    """recs: [{tcp:[x,y,z,rx,ry,rz]}] (同一個筆尖點)。回傳 (p (3,), X (3,), 殘差 rms, 各點殘差, 奇異值)。"""
    R = [stl.R_tool(r["tcp"][3:], CONV) for r in recs]
    T = [np.asarray(r["tcp"][:3], float) for r in recs]
    # 未知數 p (3) + X (3):  R_i p - X = -T_i
    M = np.vstack([np.hstack([R[i], -np.eye(3)]) for i in range(len(recs))])
    b = np.concatenate([-T[i] for i in range(len(recs))])
    sv = np.linalg.svd(M, compute_uv=False)
    x, *_ = np.linalg.lstsq(M, b, rcond=None)
    res = np.array([np.linalg.norm(R[i] @ x[:3] - x[3:] + T[i]) for i in range(len(recs))])
    return x[:3], x[3:], float(np.sqrt(np.mean(res ** 2))), res.tolist(), sv.tolist()


POSES = ("垂直", "X+", "Y−", "X−", "Y+")


def inspect_points(recs):
    """同一 run 的有效接觸。舊紀錄可能在讀不到 TCP 時記下 null。"""
    points = []
    labels = []
    for rec in recs:
        tcp = np.asarray(rec.get("tcp"), dtype=float)
        if tcp.shape != (6,) or not np.all(np.isfinite(tcp)) or "tilt" not in rec:
            continue
        tilt = rec["tilt"]
        if tilt is None:
            name = "垂直"
        else:
            axis, angle = tilt
            if axis not in ([1, 0, 0], [0, 1, 0]) or not np.isfinite(angle) or angle == 0:
                continue
            name = ("X" if axis[0] else "Y") + ("+" if angle > 0 else "−")
        labels.append(name)
        points.append(rec)
    missing = [name for name in POSES if name not in labels]
    report = {"count": len(points), "missing": missing, "fit_ok": False}
    if missing:
        return report
    p, X, rms, res, sv = solve_tip(points)
    report.update(p_tool_mm=p.tolist(), X_robot=X.tolist(), rms_mm=rms, res_mm=res,
                  singular_values=sv, fit_ok=rms < 2.0 and min(sv) > 0.1)
    return report


def check_reference(report, reference):
    """以獨立量得的基準點（機器人基座座標 mm）評估新接觸，不加入擬合。"""
    target = np.asarray(reference["target_robot_mm"], dtype=float)
    poses = np.asarray(reference["tcp_poses"], dtype=float)
    if target.shape not in ((2,), (3,)) or not np.isfinite(target).all():
        raise ValueError("target_robot_mm 需為有限的 XY 或 XYZ")
    if poses.ndim != 2 or poses.shape[1] != 6 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("tcp_poses 需有至少一組實測的六維 TCP 位姿")
    predicted = [pose[:3] + stl.R_tool(pose[3:], CONV) @ report["p_tool_mm"] for pose in poses]
    errors = np.linalg.norm(np.asarray(predicted)[:, :len(target)] - target, axis=1)
    return {"space": "XY" if len(target) == 2 else "XYZ", "errors_mm": errors.tolist(),
            "max_mm": float(errors.max()), "rms_mm": float(np.sqrt(np.mean(errors ** 2)))}


def read_pose(c, tries=3):
    """讀 TCP;連線閒置 (等人工輸入超過 ~20 秒) 被手臂端關掉時自動重連再讀。11-18:按 c 後讀不到位置而當掉。"""
    from drv_modbus import request
    from drv_modbus.conn import refresh
    for _ in range(tries):
        p = request.Get_TCP_Pose(c)
        if p is not None:
            return list(p)
        refresh(c)
    raise ConnectionError("讀不到手臂位置 (重連也失敗),請檢查網路線")


def load_tip():
    """讀最新的筆尖校正。回傳 dict(p, conv, ...) 或 None。"""
    try:
        d = CalibrationStore().load_latest("pen_tip")["data"]
        return {**d, "p": np.array(d["p_tool_mm"], float)}
    except (FileNotFoundError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--go", action="store_true")
    mode.add_argument("--check-run", metavar="RUN", help="離線檢查單次校正 run 的 events.jsonl")
    ap.add_argument("--reference", metavar="JSON", help="另一次對點的獨立量測，搭配 --check-run")
    ap.add_argument("--at", type=float, nargs=2, default=[148.0, 185.0],
                    help="紙面上點點的位置 (mm),要是空白處。預設在下方兩個 tag 中間 (離手臂基座近,11-18 在 (148,35) 往外伸碰到奇異點)")
    ap.add_argument("--tilt", type=float, default=12.0,
                    help="傾斜角度 (度)。12° 時手臂末端只要移 ~5 cm;20° 要 ~9 cm 且曾碰到奇異點")
    ap.add_argument("--guess-len", type=float, default=260.0,
                    help="第一次估計:筆尖在 TCP 正下方幾 mm。11-11 手眼:相機在 TCP 下方約 93 mm、相機比筆尖高約 168 mm -> 約 260")
    ap.add_argument("--approach", type=float, default=40.0,
                    help="傾斜後先停在估計筆尖高度上方幾 mm (筆長估錯 30 mm、傾斜 20° 時高度約差 28 mm)")
    ap.add_argument("--rot-height", type=float, default=100.0,
                    help="換姿態 (轉動) 的高度:垂直時筆尖離紙幾 mm。所有轉動都在這個 TCP 高度做,水平移動也在這個高度")
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.reference and not args.check_run:
        ap.error("--reference 需搭配 --check-run")
    if args.check_run:
        with open(os.path.join(args.check_run, "events.jsonl"), encoding="utf-8") as f:
            events = (json.loads(line) for line in f if line.strip())
            report = inspect_points(e for e in events if e.get("type") == "pivot_point")
        if args.reference and report["fit_ok"]:
            with open(args.reference, encoding="utf-8") as f:
                report["reference"] = check_reference(report, json.load(f))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["fit_ok"] else 1
    if not 5 <= args.tilt <= 35:
        sys.exit("--tilt 需在 5~35 度")

    from pymodbus.client import ModbusTcpClient
    import warp_pick as wp
    from drv_modbus import send, request
    from pen_calibrate import Arm, Paused, ask
    from pen_quick_calib import Quit, probe_with_nudge

    try:
        cal = CalibrationStore().load_latest("pen_paper")
    except FileNotFoundError:
        sys.exit("還沒有筆尖校正 (calibrations/pen_paper)")
    A, tcal, z_touch = np.array(cal["data"]["A"]), np.array(cal["data"]["t"]), cal["data"]["z_touch"]
    calib = wp.load_calibration()
    home = calib["home"]
    c = ModbusTcpClient(host=wp.ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    tcp = read_pose(c)
    ok, dpos, dang = wp.check_home(calib, tcp)
    if not ok or request.Get_Pose_Flag(c) != 1:
        sys.exit("手臂要在 home、旗標 1")
    x0, y0 = A @ np.array(args.at) + tcal
    R0 = stl.R_tool(home[3:], CONV)
    p_guess = np.array([0.0, 0.0, args.guess_len])            # 工具 z 朝下 (rx ~ 180) -> 筆尖在 +z_tool
    print(f"點點位置:紙面 {args.at} -> TCP ({x0:.1f}, {y0:.1f}),紙面接觸 z {z_touch:.2f}")
    print(f"初始估計:筆尖在 TCP 下方 {args.guess_len:g} mm (估錯的水平部分用 x / y 微調補)")
    if not args.go:
        print("(dry-run) 沒有移動手臂。加 --go 執行")
        return

    with RunLog("pen_pivot", label=args.label, params=vars(args)) as log:
        log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
        arm = Arm(c, log, home[3:], args.speed)
        recs = []
        z_up = z_touch + 30.0
        z_rot = z_touch + args.rot_height
        completed = False
        try:
            print("\n注意:轉動時筆尖會掃過約 ±9 cm 的範圍。請先把工作區的其他物件 (例如罐頭) 移開,"
                  "或確認點點位置附近 15 cm 內沒有東西。")
            ask("確認紙面那一點是空白的、周圍淨空、手放急停旁,按 Enter 開始 (先垂直點一個點) ")
            arm.move(x0, y0, z_up)
            arm.move(x0, y0, z_touch + 3.0)
            print("\n=== 垂直:每步 0.5 mm 往下,筆尖剛好點到紙就按 c (這個點之後要對準) ===")
            x, y, zc = probe_with_nudge(arm, x0, y0, z_touch + 3.0, 0.5, z_touch - 3.0, z_up, paper_A=A)
            t_now = read_pose(c)
            recs.append({"tcp": t_now, "tilt": None})
            log.event("pivot_point", tcp=t_now, tilt=None)
            T0 = np.array(t_now[:3])
            p_est = p_guess
            X_est = T0 + R0 @ p_est
            # 11-18 事故:原本「抬 30 mm 就在原地轉回垂直」,但從傾斜轉回垂直時筆尖會往下掉 ~38 mm -> 筆壓進紙 ~8 mm。
            # 現在所有轉動與水平移動都在 z_rot (垂直時筆尖離紙 --rot-height,預設 100 mm) 做,只在目標正上方垂直下降。
            z_rot = zc + args.rot_height
            arm.move(x, y, z_rot)
            # 順序:11-18 實測繞 y 軸 +20° 要往外伸 ~9 cm,手臂走不到 -> 排最後;先取 x+、y- 以取得兩個不同轉軸
            for i, (axis, sgn) in enumerate((([1, 0, 0], 1), ([0, 1, 0], -1), ([1, 0, 0], -1), ([0, 1, 0], 1)), 1):
                print(f"接下來取樣 {POSES[i]}，已取得 {len(recs)}/5 個接觸姿態")
                Ri = stl.tilt_base(axis, sgn * args.tilt) @ R0
                ori = stl.angles_of(Ri, CONV, near=home[3:])
                T_pred = X_est - Ri @ p_est
                cur = read_pose(c)
                print(f"\n--- 傾斜:繞手臂 {'x' if axis[0] else 'y'} 軸 {sgn * args.tilt:+g}° ---")
                try:
                    arm.move(cur[0], cur[1], max(cur[2], z_rot), ori=list(cur[3:]))   # 先確定在轉動高度
                    arm.move(cur[0], cur[1], max(cur[2], z_rot), ori=ori)            # 在高處原地轉
                    arm.move(T_pred[0], T_pred[1], max(cur[2], z_rot), ori=ori)      # 高處水平移到目標正上方
                    arm.move(T_pred[0], T_pred[1], T_pred[2] + args.approach, ori=ori)   # 垂直下降到估計位置上方
                except send.CommandIgnored as e:
                    arm.stop_and_check("pivot_ignored")
                    print("  手臂沒有走到這個姿態 (可能接近奇異點 / 可達範圍邊緣),略過:", e)
                    p_ = read_pose(c)
                    arm.move(p_[0], p_[1], max(p_[2], z_rot), ori=list(p_[3:]))
                    arm.move(p_[0], p_[1], max(p_[2], z_rot), ori=home[3:])
                    continue
                arm.ori = list(ori)                            # 人工微調時保持這個姿態
                print(f"  ★ 這一步要做的:讓【傾斜的筆尖】剛好碰在【一開始垂直點的那個點】上 (紙面 {args.at})。\n"
                      f"    程式只能猜筆尖在哪 (停在猜的位置上方 {args.approach:g} mm),猜不準的部分由你修正:\n"
                      f"    1) Enter 往下 (每步 2 mm;輸入 5 加快、靠近紙時輸入 0.5)\n"
                      f"    2) 筆尖快碰到紙時,看筆尖在點的哪一邊,用 w / s / a / d 移 (例如 a 2 = 往紙的左邊 2 mm)\n"
                      f"    3) 筆尖剛好碰在那個點上 (不壓、不歪) 就按 c;不想做了輸入 q (會抬起、轉回垂直、回 home)")
                xi, yi, zi = probe_with_nudge(arm, T_pred[0], T_pred[1], T_pred[2] + args.approach, 2.0,
                                              T_pred[2] - 30.0, T_pred[2] + args.approach + 5, paper_A=A)
                t_now = read_pose(c)
                recs.append({"tcp": t_now, "tilt": [axis, sgn * args.tilt]})
                log.event("pivot_point", tcp=t_now, tilt=[axis, sgn * args.tilt])
                arm.move(xi, yi, max(zi + 30.0, z_rot))          # 先 (維持傾斜) 垂直抬到轉動高度
                arm.move(xi, yi, max(zi + 30.0, z_rot), ori=home[3:])   # 再在高處轉回垂直 (筆尖會往下掉 ~40 mm,仍離紙 ~100 mm)
                arm.ori = list(home[3:])
                if len(recs) >= 3:
                    p_fit, X_fit, rms, res, sv = solve_tip(recs)
                    if min(sv) <= 0.1 or rms >= 2.0:
                        print("姿態多樣性或殘差未達標，保留先前的接近位置估計")
                        continue
                    p_est, X_est = p_fit, X_fit
                    print(f"  目前估計:筆尖在工具座標 {np.round(p_est, 1).tolist()} mm,殘差 {rms:.2f} mm")
            arm.move(*home[:3], ori=home[3:])
            completed = True
        except (Quit, Paused, KeyboardInterrupt, ConnectionError, send.CommandIgnored, TimeoutError) as e:
            print("\n停止:", e, "-> 抬起回 home")
            try:
                p_ = read_pose(c)
                recovery_z = max(p_[2], z_rot)
                arm.move(p_[0], p_[1], recovery_z, ori=list(p_[3:]))
                arm.move(p_[0], p_[1], recovery_z, ori=home[3:])
                arm.move(*home[:3], ori=home[3:])
            except Exception:                  # noqa: BLE001
                print("回 home 沒完成,請用回 home 任務")
        c.close()
        report = inspect_points(recs)
        log.save_json("pivot_fit", report)
        if not completed or report["missing"]:
            print("校正未完成；缺少姿態：", "、".join(report["missing"]) or "無，流程已中斷")
            log.finish("aborted", report)
            return
        print(f"各姿態殘差 {np.round(report['res_mm'], 2).tolist()} mm，"
              f"RMS {report['rms_mm']:.2f} mm，最小奇異值 {min(report['singular_values']):.3f}")
        if not report["fit_ok"]:
            print("未通過擬合門檻，請檢查各姿態的接觸紀錄")
            log.finish("failed", report)
            return
        data = dict(report, conv=CONV, n=len(recs), tilt_deg=args.tilt, points=recs)
        log.save_json("pen_tip", data)
        CalibrationStore().save("pen_tip", data, source_run=log.path, notes="五姿態擬合；獨立對點另測")
        log.finish("ok", report)
        print("已存五姿態擬合結果。請使用另一個已知點量測實際定位誤差。")
        print("紀錄:", log.path)


if __name__ == "__main__":
    sys.exit(main())
