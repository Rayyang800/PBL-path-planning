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
按鍵:Enter 往下 / 數字 改步距 / u 往上 / x 2、y -3 水平微調 / c 對準了 / q 結束;Ctrl+C = 中止，現場確認姿態
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
    ap.add_argument("--go", action="store_true")
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
            for axis, sgn in (([1, 0, 0], 1), ([0, 1, 0], -1), ([1, 0, 0], -1), ([0, 1, 0], 1)):
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
        except (Paused, KeyboardInterrupt, ConnectionError, send.CommandIgnored, TimeoutError) as e:
            print("校正已中斷，請現場確認姿態與手臂狀態:", e)
        except Quit as e:
            print("\n停止:", e, "-> 抬起回 home")
            try:
                p_ = read_pose(c)
                arm.move(p_[0], p_[1], p_[2] + args.rot_height, ori=list(p_[3:]))     # 維持姿態抬高,再轉回垂直
                arm.move(p_[0], p_[1], p_[2] + args.rot_height, ori=home[3:])
                arm.move(*home[:3], ori=home[3:])
            except Exception:                  # noqa: BLE001
                print("回 home 沒完成,請用回 home 任務")
        c.close()
        if len(recs) < 3:
            log.finish("aborted", {"points": len(recs)})
            print("點數不夠 (需要垂直 + 至少兩個不同軸的傾斜),沒有存校正")
            return
        p_est, X_est, rms, res, sv = solve_tip(recs)
        print(f"\n筆尖在工具座標 {np.round(p_est, 2).tolist()} mm (長 {np.linalg.norm(p_est):.1f} mm);"
              f"各姿態殘差 {np.round(res, 2).tolist()} mm,RMS {rms:.2f} mm;奇異值 {np.round(sv, 2).tolist()}")
        data = {"p_tool_mm": p_est.tolist(), "conv": CONV, "rms_mm": rms, "res_mm": res, "n": len(recs),
                "tilt_deg": args.tilt, "points": recs, "X_robot": X_est.tolist()}
        log.save_json("pen_tip", data)
        if rms < 2.0 and min(sv) > 0.1:
            CalibrationStore().save("pen_tip", data, source_run=log.path, notes="pen_pivot.py")
            print("已存 calibrations/pen_tip (傾斜描繪會用)")
            log.finish("ok", {"p_tool_mm": p_est.tolist(), "rms_mm": rms})
        else:
            print("殘差太大或姿態不夠多樣,沒有存;請重做 (對準時看清楚點的中心)")
            log.finish("failed", {"rms_mm": rms})
        print("紀錄:", log.path)


if __name__ == "__main__":
    main()
