# -*- coding: utf-8 -*-
"""
近距離深度量測:手臂從 home 垂直往下 --dz mm (預設 150,相機離紙 ~510 -> ~360 mm),比較兩個高度量到的板厚與深度縫。
深度誤差約與距離平方成正比,近一點理論上更準。

  python depth_closeup.py                # 下降 150 mm (按 Enter 才動)
  python depth_closeup.py --dz 200       # 下降 200 mm (4 個 tag 可能拍不全)

流程:home 拍 3 組 -> MovL 垂直下降 (速度 10) -> 拍 3 組 -> MovL 回 home -> 印比較表。全部存在 runs/<時間>_depth_closeup/。
按 Ctrl+C:送停止指令 (手臂停在原地,之後用 python go_home.py 回去)。
"""
import argparse
import sys
import time

import numpy as np
from pymodbus.client import ModbusTcpClient

import depth_height
import seam_boards
import warp_pick as wp
from drv_modbus import send, request
from drv_modbus.conn import refresh
from paper_frame import PaperFrame
from rs_capture import RSCamera
from run_log import RunLog

LAYOUT = wp.os.path.join(wp.HERE, "print", "lab_sheet_layout.json")


def measure(cam, info, log, tag, n, board_h, truth):
    rows = []
    for k in range(n):
        s = cam.grab(n_depth=15, settle=0.5 if k == 0 else 0.0, flush=5)
        log.save_snapshot(f"{tag}_{k}", s, info)
        mm = s["depth"].astype(np.float32) * float(info["depth_scale_m"]) * 1000.0
        mm[s["depth"] == 0] = np.nan
        r = {"tag": tag, "k": k}
        try:
            H, pf, st = depth_height.height_map(s["color"], mm, info["color_intrinsics"], LAYOUT)
            boards, _ = seam_boards.find_boards(s["color"], pf, board_h)
            r.update(cam_to_paper_mm=st["cam_to_plane_mm"], paper_rms_mm=st["paper_rms_mm"])
            if boards:
                est = depth_height.board_thickness(H, boards[0]["contour"])
                if est:
                    r.update(t_mm=est["t_mm"], t_err=est["t_mm"] - truth)
                col = seam_boards.analyze(s["color"], PaperFrame(layout=LAYOUT), board_h_mm=board_h)
                ds = depth_height.depth_seam(H, pf, boards[0]["box"], board_h)
                r.update(depth_seam_ok=bool(ds["ok"]), dip_mm=ds.get("dip_mm"), color_seam_ok=bool(col["ok"]))
                if ds["ok"] and col["ok"]:
                    r["seam_offset_mm"], r["seam_angle_deg"] = depth_height.line_offset(col["seam_paper_mm"], ds["seam_paper_mm"])
        except Exception as e:                  # noqa: BLE001
            r["error"] = str(e)[:120]
        print(f"  [{tag} #{k}] 相機離紙 {r.get('cam_to_paper_mm', float('nan')):.0f} mm | 板厚 {r.get('t_mm', float('nan')):.2f} "
              f"(誤差 {r.get('t_err', float('nan')):+.2f}) | 紙面雜訊 {r.get('paper_rms_mm', float('nan')):.2f} | 深度縫 "
              + (f"凹陷 {r['dip_mm']:.2f}、與影像縫差 {r.get('seam_offset_mm', float('nan')):+.2f} mm / {r.get('seam_angle_deg', float('nan')):.1f}°"
                 if r.get("depth_seam_ok") else "找不到") + (f" | 錯誤 {r['error']}" if "error" in r else ""))
        rows.append(r)
    return rows


def stats(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz", type=float, default=150.0, help="從 home 垂直往下多少 mm (50~250)")
    ap.add_argument("--n", type=int, default=3, help="每個高度拍幾組")
    ap.add_argument("--board-h", type=float, default=5.0, help="板厚設定 / 真值 (mm)")
    ap.add_argument("--speed", type=int, default=10)
    args = ap.parse_args()
    if not 50 <= args.dz <= 250:
        sys.exit("--dz 需在 50~250 mm")
    calib = wp.load_calibration()
    home = calib["home"]
    c = ModbusTcpClient(host=wp.ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    tcp = request.Get_TCP_Pose(c)
    ok, dpos, dang = wp.check_home(calib, tcp)
    if not ok:
        sys.exit(f"手臂不在 home (差 {dpos:.1f} mm / {dang:.1f}°),請先 python go_home.py")
    if request.Get_Pose_Flag(c) != 1:
        sys.exit("旗標不是 1,請先在教導器清除警報")
    low = list(home)
    low[2] = home[2] - args.dz
    cam = RSCamera()
    info = cam.info()
    with RunLog("depth_closeup", params=vars(args)) as log:
        try:
            print(f"=== 1/3 home (z {home[2]:.1f}) 量測 ===")
            r_home = measure(cam, info, log, "home", args.n, args.board_h, args.board_h)
            input(f"\n=== 2/3 手臂將垂直往下 {args.dz:g} mm 到 z {low[2]:.1f} (MovL,速度 {args.speed})。"
                  "確認下方淨空、手放急停旁後按 Enter (Ctrl+C 取消) ")
            refresh(c)
            send.Go_Position(c, *low, args.speed, send.MovL)
            r_low = measure(cam, info, log, "low", args.n, args.board_h, args.board_h)
            print("\n=== 3/3 回 home ===")
            refresh(c)
            send.Go_Position(c, *home, args.speed, send.MovL)
        except (KeyboardInterrupt, send.Interrupted):
            print("\n已停止 (手臂停在目前位置,之後用 python go_home.py 回 home)")
            log.finish("aborted", {"reason": "ctrl_c"})
            cam.close()
            return
        except (send.CommandIgnored, TimeoutError) as e:
            print("\n手臂沒有完成移動:", e)
            log.finish("failed", {"error": str(e)})
            cam.close()
            return
        cam.close()
        print(f"\n================ 比較 (真值 {args.board_h:g} mm) ================")
        res = {}
        for name, rows in (("home", r_home), (f"下降 {args.dz:g} mm", r_low)):
            d, _ = stats(rows, "cam_to_paper_mm")
            t, ts = stats(rows, "t_mm")
            pr, _ = stats(rows, "paper_rms_mm")
            so, sos = stats(rows, "seam_offset_mm")
            sa, _ = stats(rows, "seam_angle_deg")
            dip, _ = stats(rows, "dip_mm")
            print(f"{name:12s} 相機離紙 {d:5.0f} mm | 板厚 {t:5.2f} ± {ts:.2f} (誤差 {t - args.board_h:+.2f}) | 紙面雜訊 {pr:.2f} | "
                  f"深度縫凹陷 {dip:.2f}、與影像縫差 {so:+.2f} ± {sos:.2f} mm / {sa:.1f}°")
            res[name] = {"cam_to_paper_mm": d, "t_mean": t, "t_std": ts, "t_err": t - args.board_h, "paper_rms": pr,
                         "seam_offset": so, "seam_offset_std": sos, "seam_angle": sa, "dip": dip}
        log.save_json("comparison", {"home": r_home, "low": r_low, "summary": res})
        log.finish("ok", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()})
        print("紀錄:", log.path)
    c.close()


if __name__ == "__main__":
    main()
