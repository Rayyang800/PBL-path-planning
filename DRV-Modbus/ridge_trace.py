# -*- coding: utf-8 -*-
"""
沿脊線描繪 (工件 A:L 型 / 屋脊形的脊線塗)。

流程 (--go)
  0. home 拍照 -> 找物體 -> 粗估脊線方向
  1. 快速掃描 (--views,預設 5):相機在物體正上方 + 沿脊線前後 + 垂直脊線左右,全部「朝正下方、只平移」
     (不傾斜 -> 不會碰到奇異點;11-11 離線比較:正上方 1 張脊線差 ~1.7 mm,5 張 < 0.4 mm,和 21 個視角的完整掃描比)
     --from-run runs\\<scan3d 紀錄>:直接用之前掃好的模型 (物體不能動過;會先用 home 照片檢查位置)
  2. 從點雲擷取脊線 (ridge.extract_ridge:逐片取最高點 + 穩健直線;兩側斜面交線交叉驗證) -> 每 --step-mm 一點
  3. 移到起點上方安全高度 -> 降到「脊線高度 + approach」-> 人工每步 0.5 mm 往下,筆尖剛好碰到按 c
     (深度在尖角會被抹圓,量到的脊線比真正的尖端低幾 mm,所以接觸高度一定要人工確認)
  4. 用「接觸高度 - 量到的起點高度」當修正量,沿脊線 MovL (高度跟著量到的脊線坡度) -> 抬起 -> 回 home 拍照

安全
  - 移到起點前,筆尖一直比物體最高點高 --clearance (預設 30 mm);只在起點正上方垂直下降
  - 人工下降的下限 = 量到的起點高度 - 4 mm;描繪時每點高度不會低於「量到的脊線 - 4 mm」
  - 掃描點用 scan3d 同一套規則 (筆尖淨空、相機最短距離、離 home 最大距離)

  python ridge_trace.py                          # dry-run:只用 home 一張找脊線、出計畫圖 (ridge_trace_plan.png),不動手臂
  python ridge_trace.py --go                     # 快速掃描 5 點 -> 找脊線 -> 人工確認接觸 -> 沿脊線畫
  python ridge_trace.py --go --views 1           # 只拍正上方一張 (最快)
  python ridge_trace.py --go --from-run runs\\20261111-115626_scan3d_reprocess_...   # 不掃描,用之前的模型

按鍵:Ctrl+C = 停止 -> 抬起回 home;移動中 空白/Esc/p = 選單 (r 繼續 / q 結束);o/g = 夾爪
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import ridge
import scan3d
from run_log import RunLog, CalibrationStore

HERE = os.path.dirname(os.path.abspath(__file__))


def quick_targets(obj, dir_xy, n, r):
    ctr = np.array(obj["center"], float)
    u = np.asarray(dir_xy, float)
    v = np.array([-u[1], u[0]])
    t = [("c", ctr)]
    if n >= 3:
        t += [("perp+", ctr + r * v), ("perp-", ctr - r * v)]
    if n >= 5:
        t += [("along+", ctr + r * u), ("along-", ctr - r * u)]
    return t


def draw_ridge_on(color, pose, K, line, path, h_obj):
    img = color.copy()
    rv, _ = cv2.Rodrigues(pose["R"])
    P = np.array([[p[0], p[1], -p[2]] for p in path], float)
    pix, _ = cv2.projectPoints(P, rv, pose["t"], K, np.zeros(5))
    pix = pix.reshape(-1, 2)
    cv2.polylines(img, [np.int32(pix)], False, (255, 0, 255), 2)
    cv2.circle(img, tuple(int(v) for v in pix[0]), 9, (0, 255, 0), 2)
    cv2.putText(img, "start", (int(pix[0][0]) + 10, int(pix[0][1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def summarize_ridge(r):
    pl = r.get("planes") or {}
    s = (f"脊線:長 {r['length_mm']:.1f} mm、平均高 {r['h_mean_mm']:.2f} mm、沿線坡度 {r['slope_deg']:+.2f}°、"
         f"直線殘差 橫向 {r['lateral_rms_mm']:.2f} / 高度 {r['height_rms_mm']:.2f} mm ({r['n_slices']} 片)")
    if pl:
        s += (f"\n  兩側斜面:夾角 {pl['ridge_angle_deg']:.1f}°;斜面交線與脊線 橫向差 {pl['lateral_diff_mm']:+.2f} mm、"
              f"高度差 {pl['height_diff_mm']:+.2f} mm (尖角被深度抹圓,實際尖端可能比量到的高,所以接觸要人工確認)")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--views", type=int, choices=[1, 3, 5], default=5, help="快速掃描幾個正上方視角")
    ap.add_argument("--from-run", default=None, help="用之前 scan3d 的模型 (data/cloud_raw.npz),不重新掃描")
    ap.add_argument("--cam-h", type=float, default=380.0)
    ap.add_argument("--radius", type=float, default=60.0, help="快速掃描:周圍視角離物體中心 mm")
    ap.add_argument("--clearance", type=float, default=30.0, help="移動時筆尖至少比物體最高點高 mm")
    ap.add_argument("--approach", type=float, default=3.0, help="自動降到量到的脊線上方幾 mm,之後人工確認")
    ap.add_argument("--press", type=float, default=0.0, help="接觸高度再往下壓幾 mm 畫線")
    ap.add_argument("--step-mm", type=float, default=2.0)
    ap.add_argument("--end-margin", type=float, default=8.0, help="脊線兩端各留多少 mm 不畫")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--n-depth", type=int, default=10)
    ap.add_argument("--layout", default=scan3d.LAYOUT)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    # scan3d.plan_views 需要的參數
    args.min_range, args.max_travel, args.n_ring = 230.0, 150.0, 0

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
    if z_touch is None:
        sys.exit("筆尖校正沒有 z_touch")
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
    xy0, h0, _, rows0, obj = scan3d.process_views([home_view], layout)
    if obj is None or xy0 is None:
        cam.close()
        sys.exit("校正紙上找不到比紙面高的物體")
    r0 = ridge.extract_ridge(xy0, h0, end_margin=args.end_margin)
    if r0 is None:
        cam.close()
        sys.exit("home 照片找不到脊線 (物體太小或不是屋脊形)")
    print(f"\n[home 一張] " + summarize_ridge(r0))

    with RunLog("ridge_trace", label=args.label or ("" if args.go else "dryrun"), params=vars(args)) as log:
        log.add_input(args.layout)
        log.add_input(os.path.join(HERE, "calibrations", "pen_paper", "latest.json"), as_name="pen_paper.json")
        log.event("tcp", pose=tcp, flag=flag)
        log.save_snapshot("home", snap, info)
        log.save_json("ridge_home", r0)
        obj_plan = dict(obj, h_max=max(obj["h_max"], r0["h_max_mm"]) * 1.15 + 5.0)
        arm = Arm(c, log, home[3:], args.speed)
        views_done = [home_view]
        ridge_res, src = r0, "home"

        try:
            # ---------- 1. 取得點雲 ----------
            if args.from_run:
                npz = os.path.join(args.from_run, "data", "cloud_raw.npz")
                pj = os.path.join(args.from_run, "data", "plan.json")
                if not os.path.isfile(npz):
                    raise SystemExit("找不到 " + npz)
                with open(pj, encoding="utf-8") as f:
                    old_obj = json.load(f)["object_home"]
                move = float(np.hypot(*(np.array(old_obj["center"]) - np.array(obj["center"]))))
                print(f"  --from-run:物體中心和那次相差 {move:.1f} mm")
                if move > 3.0:
                    raise SystemExit("物體和那次掃描時的位置差超過 3 mm,不能用舊模型 (請重新掃描)")
                d = np.load(npz)
                rr = ridge.extract_ridge(d["xy"], d["h"], end_margin=args.end_margin)
                if rr is None:
                    raise SystemExit("舊模型找不到脊線")
                ridge_res, src = rr, "from_run"
                log.add_input(npz, as_name="cloud_raw_from_run.npz")
            elif args.go:
                targets = quick_targets(obj, r0["dir_xy"], args.views, args.radius)
                views, pinfo = scan3d.plan_views(obj_plan, hp, tcp, A, tcal, z_touch, args, (W, H), targets=targets)
                for n_ in pinfo["notes"]:
                    print("  注意:", n_)
                print(f"\n快速掃描 {len(views)} 點:相機離紙 {pinfo['cam_h_scan']:.0f} mm,筆尖比物體高 "
                      f"{pinfo['pen_clearance_over_object']:.0f} mm,全部朝正下方 (不傾斜)")
                ask("確認筆已夾好、手放急停旁後按 Enter 開始快速掃描 ")
                for v in views:
                    print(f"  {v['name']}:移到 ({v['tcp'][0]:.1f}, {v['tcp'][1]:.1f}, {v['tcp'][2]:.1f})")
                    arm.move(*v["tcp"], ori=home[3:])
                    s_, mmv = scan3d.capture(cam, info, args.n_depth)
                    tcp_now = request.Get_TCP_Pose(c)
                    d_tcp = np.array(tcp_now[:3]) - np.array(tcp[:3])
                    dpen = np.linalg.solve(A, d_tcp[:2])
                    pred_C = hp["C"] + np.array([dpen[0], dpen[1], -d_tcp[2]])
                    pose, psrc, tp, diff = scan3d.choose_pose(s_["color"], layout, K, scan3d.pose_from_center(hp["R"], pred_C))
                    log.save_snapshot(v["name"], s_, info)
                    log.event("scan_view", name=v["name"], tcp=tcp_now, pose_source=psrc, pred_C=pred_C.tolist(),
                              R=pose["R"].tolist(), t=pose["t"].tolist(), C=pose["C"].tolist())
                    views_done.append({"name": v["name"], "color": s_["color"], "mm": mmv, "K": K, "pose": pose,
                                       "pose_source": psrc, "pred_C": pred_C, "tcp": tcp_now})
                    print(f"    拍好 ({'tag' if psrc == 'tags' else '手臂推算'})")
                scan3d.fix_arm_poses(views_done, log)
                xy, h, rgb, rows, _ = scan3d.process_views(views_done, layout, obj, log=log)
                log.save_json("views_measure", rows)
                rr = ridge.extract_ridge(xy, h, end_margin=args.end_margin) if xy is not None else None
                if rr is None:
                    print("  快速掃描找不到脊線,改用 home 一張的結果")
                else:
                    ridge_res, src = rr, f"quick{len(views_done) - 1}"
                    P, C, _ = scan3d.voxel_down(xy, h, rgb, 1.0)
                    os.makedirs(os.path.join(log.path, "data"), exist_ok=True)
                    scan3d.write_ply(os.path.join(log.path, "data", "model.ply"), P, C)
            # ---------- 2. 路徑 ----------
            print(f"\n[{src}] " + summarize_ridge(ridge_res))
            if src != "home":
                d0 = np.array(ridge_res["line_xyh"]) - np.array(r0["line_xyh"])
                print(f"  和 home 一張比:兩端水平差 {np.linalg.norm(d0[0, :2]):.1f} / {np.linalg.norm(d0[1, :2]):.1f} mm、"
                      f"高度差 {d0[0, 2]:+.1f} / {d0[1, 2]:+.1f} mm")
            path3 = ridge.path_along(ridge_res["line_xyh"], args.step_mm, args.reverse)        # 紙面 [x, y, h]
            path_robot = np.array([[*(A @ p[:2] + tcal), z_touch + p[2]] for p in path3])
            view = draw_ridge_on(snap["color"], hp, K, ridge_res["line_xyh"], path3, obj["h_max"])
            log.save_image("plan.jpg", view)
            cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "ridge_trace_plan.png"))
            h_top = max(float(np.max(path3[:, 2])), obj["h_max"])
            z_safe = min(float(home[2]), z_touch + h_top + max(args.clearance, 20.0) + 5.0)
            plan = {"ridge": ridge_res, "ridge_home": r0, "source": src, "path_paper_xyh": path3.tolist(),
                    "path_robot": path_robot.tolist(), "z_touch": z_touch, "z_safe": z_safe, "object_home": obj}
            log.save_json("plan", plan)
            log.save_csv("path_robot_mm", ["x", "y", "z_est"], path_robot.tolist())
            print(f"  路徑 {len(path3)} 點 (每 {args.step_mm:g} mm);起點 TCP ({path_robot[0][0]:.1f}, {path_robot[0][1]:.1f}),"
                  f"估計接觸 z {path_robot[0][2]:.2f};移動高度 z {z_safe:.1f} (筆尖離紙 {z_safe - z_touch:.0f} mm)")
            if not args.go:
                cam.close()
                log.finish("dry-run", {"length_mm": ridge_res["length_mm"], "h_mean_mm": ridge_res["h_mean_mm"]})
                print("\n(dry-run) 沒有移動手臂。看 ridge_trace_plan.png:洋紅線 = 要畫的脊線、綠圈 = 起點。確認後加 --go")
                print("紀錄:", log.path)
                c.close()
                return

            # ---------- 3. 接觸 ----------
            x0, y0, zr0 = path_robot[0]
            ask(f"\n確認洋紅線在脊線上 (ridge_trace_plan.png)、手放急停旁後按 Enter:移到起點上方 z {z_safe:.1f} ")
            arm.move(x0, y0, z_safe)
            z_start = zr0 + args.approach
            z_floor = zr0 - 4.0
            arm.move(x0, y0, z_start)
            print(f"\n=== 已在脊線起點上方 {args.approach:g} mm (z = {z_start:.2f});每步 0.5 mm 往下 (Enter),"
                  f"筆尖剛好碰到脊線就按 c (最低到 z {z_floor:.2f}) ===")
            while True:
                x, y, z_c = probe_with_nudge(arm, x0, y0, z_start, 0.5, z_floor, z_safe)
                if z_c < z_start - 0.05:
                    break
                if ask("  還沒往下就按 c:筆尖可能已經壓在脊線上 (尖端比量到的高)。確定剛好碰到?(y / Enter = 往上 3 mm 再找) ")\
                        .strip().lower() == "y":
                    break
                z_start = z_c + 3.0
                if z_start > z_safe:
                    raise Quit("起點已抬到移動高度,脊線可能比估計高很多,請檢查")
                arm.move(x, y, z_start)
                x0, y0 = x, y
            shift = np.array([x - path_robot[0][0], y - path_robot[0][1]])
            dz = z_c - zr0
            log.event("contact", robot_xy=[x, y], z=z_c, z_est=zr0, dz=dz, shift=shift.tolist())
            print(f"  接觸 z {z_c:.2f} (量到的 {zr0:.2f},差 {dz:+.2f} mm);水平微調 ({shift[0]:+.1f}, {shift[1]:+.1f}) 套用到整條")
            path_draw = path_robot.copy()
            path_draw[:, :2] += shift
            path_draw[:, 2] = np.maximum(path_robot[:, 2] + dz - args.press, path_robot[:, 2] - 4.0)
            # ---------- 4. 描繪 ----------
            done = 0
            k = 1
            while k < len(path_draw):
                try:
                    arm.move(*map(float, path_draw[k]))
                    done = k
                    k += 1
                except Paused as e:
                    if str(e) == "ctrl_c":
                        raise
                    ch = ""
                    while ch not in ("r", "q"):
                        ch = ask(f"暫停 (畫到 {done}/{len(path_draw) - 1}):r = 繼續 / q = 抬起回 home > ").strip().lower()
                    if ch == "q":
                        raise Quit("使用者結束")
                    p = request.Get_TCP_Pose(c)
                    arm.move(p[0], p[1], p[2] + 3)
                    arm.move(float(path_draw[done][0]), float(path_draw[done][1]), float(path_draw[done][2]) + 3)
                    arm.move(*map(float, path_draw[done]))
            log.event("trace_done", points=done, of=len(path_draw) - 1)
            arm.lift_to(z_safe)
            arm.move(*home[:3], ori=home[3:])
            status, res = "ok", {"points_done": done, "dz_mm": dz}
        except (Quit, Paused, KeyboardInterrupt) as e:
            print("\n停止:", e, "-> 垂直抬起回 home")
            status, res = "aborted", {"reason": str(e)}
            try:
                arm.lift_to(min(float(home[2]), z_touch + obj_plan["h_max"] + args.clearance))
                arm.move(*home[:3], ori=home[3:])
            except (Paused, KeyboardInterrupt, send.CommandIgnored, TimeoutError):
                print("已停止,手臂停在目前位置 (之後用回 home)")
        except SystemExit as e:
            status, res = "failed", {"error": str(e)}
            print(e)
            try:
                arm.move(*home[:3], ori=home[3:])
            except Exception:                  # noqa: BLE001
                print("回 home 沒完成,請用回 home 任務")
        time.sleep(1.0)
        try:
            s_after, _ = scan3d.capture(cam, info, 5)
            log.save_snapshot("home_after", s_after, info)
        except Exception:                      # noqa: BLE001
            pass
        cam.close()
        c.close()
        res.update(source=src, length_mm=ridge_res["length_mm"], h_mean_mm=ridge_res["h_mean_mm"])
        log.finish(status, res)
        print("紀錄:", log.path)


if __name__ == "__main__":
    main()
