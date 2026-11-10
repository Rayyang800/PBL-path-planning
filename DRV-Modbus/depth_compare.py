# -*- coding: utf-8 -*-
"""
比較 RealSense 深度設定對「板厚估計」的影響 (只讀相機,不動手臂)。
手臂在 home、校正紙與工件放好。對每組設定重複估板厚 N 次,看誰最接近真值、最穩定。

  python depth_compare.py --truth 5             # 已知板厚 5 mm
  python depth_compare.py --truth 5 --trials 8

設定 (彩色固定 1280x720,深度對齊到彩色):
  A 1280x720 深度、無濾波 (目前)
  B 848x480 深度、無濾波 (官方建議的 D435 最佳深度解析度)
  C 848x480 + spatial + temporal 濾波
  D C + High Accuracy 預設 + 雷射功率最大
  E 848x480 + High Accuracy + 雷射最大、無濾波
結果與每組的快照存在 runs/<時間>_depth_compare/。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

import depth_height
import seam_boards
from paper_frame import PaperFrame
from run_log import RunLog

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = os.path.join(HERE, "print", "lab_sheet_layout.json")

CONFIGS = [
    ("A_1280_raw", (1280, 720), False, False),
    ("B_848_raw", (848, 480), False, False),
    ("C_848_filter", (848, 480), True, False),
    ("D_848_filter_HA_laser", (848, 480), True, True),
    ("E_848_HA_laser", (848, 480), False, True),
]


def run_config(name, dres, use_filters, ha_laser, trials, n_frames, color_info):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, dres[0], dres[1], rs.format.z16, 30)
    prof = pipe.start(cfg)
    try:
        sensor = prof.get_device().first_depth_sensor()
        scale = sensor.get_depth_scale()
        if sensor.supports(rs.option.visual_preset):
            preset = rs.rs400_visual_preset.high_accuracy if ha_laser else rs.rs400_visual_preset.default
            sensor.set_option(rs.option.visual_preset, int(preset))
        if sensor.supports(rs.option.laser_power):
            r = sensor.get_option_range(rs.option.laser_power)
            sensor.set_option(rs.option.laser_power, r.max if ha_laser else r.default)
        laser = sensor.get_option(rs.option.laser_power) if sensor.supports(rs.option.laser_power) else None
        align = rs.align(rs.stream.color)
        spat, temp = rs.spatial_filter(), rs.temporal_filter()
        it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = {"fx": it.fx, "fy": it.fy, "ppx": it.ppx, "ppy": it.ppy}
        for _ in range(40):                         # 等曝光/預設穩定
            pipe.wait_for_frames(5000)
        results, last = [], None
        for t in range(trials):
            deps, color = [], None
            for _ in range(n_frames):
                fr = align.process(pipe.wait_for_frames(5000))
                d = fr.get_depth_frame()
                if use_filters:
                    d = temp.process(spat.process(d))
                deps.append(np.asanyarray(d.get_data()).astype(np.float32))
                color = np.asanyarray(fr.get_color_frame().get_data()).copy()
            D = np.stack(deps)
            D[D == 0] = np.nan
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mm = np.nanmedian(D, axis=0) * scale * 1000.0
            pf = PaperFrame(layout=LAYOUT)
            if not pf.detect(color):
                results.append({"ok": False, "reason": "paper"})
                continue
            boards, _ = seam_boards.find_boards(color, pf)
            try:
                H, _, st = depth_height.height_map(color, mm, K, LAYOUT)
                est = depth_height.board_thickness(H, boards[0]["contour"]) if boards else None
            except Exception as e:              # noqa: BLE001
                results.append({"ok": False, "reason": str(e)})
                continue
            if est is None:
                results.append({"ok": False, "reason": "no estimate"})
                continue
            valid = float(np.mean(~np.isnan(mm)))
            results.append({"ok": True, "t_mm": est["t_mm"], "spread_mm": est["spread_mm"], "paper_rms_mm": st["paper_rms_mm"],
                            "valid_frac": valid, "cam_to_plane_mm": st["cam_to_plane_mm"]})
            raw = np.nan_to_num(mm / (scale * 1000.0), nan=0).round().astype(np.uint16)
            last = ({"color": color, "depth": raw, "n_depth": n_frames}, H)
        return results, last, {"laser_power": laser, "depth_res": list(dres), "filters": use_filters,
                               "high_accuracy": ha_laser, "color_intrinsics": K, "depth_scale_m": scale}
    finally:
        pipe.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=float, default=5.0, help="已知板厚 (mm)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--frames", type=int, default=15, help="每次估計用幾張深度取中位數")
    ap.add_argument("--only", nargs="*", help="只跑某些設定 (名稱開頭,例如 B D)")
    args = ap.parse_args()
    if not rs.context().query_devices():
        sys.exit("找不到 RealSense")
    summary = []
    with RunLog("depth_compare", params=vars(args)) as log:
        for name, dres, filt, ha in CONFIGS:
            if args.only and not any(name.startswith(o) for o in args.only):
                continue
            print(f"\n=== {name} ===")
            t0 = time.time()
            res, last, meta = run_config(name, dres, filt, ha, args.trials, args.frames, None)
            ts = [r["t_mm"] for r in res if r["ok"]]
            for r in res:
                print("   ", f"板厚 {r['t_mm']:+.2f} mm  分區差 {r['spread_mm']:.2f}  紙面殘差 {r['paper_rms_mm']:.2f}  有效 {r['valid_frac'] * 100:.0f}%"
                      if r["ok"] else f"失敗: {r['reason']}")
            row = {"config": name, **meta, "n_ok": len(ts), "trials": res}
            if ts:
                row.update(mean=float(np.mean(ts)), std=float(np.std(ts)), range=float(np.ptp(ts)),
                           err=float(np.mean(ts) - args.truth), max_abs_err=float(np.max(np.abs(np.array(ts) - args.truth))),
                           paper_rms=float(np.mean([r["paper_rms_mm"] for r in res if r["ok"]])))
                print(f"  -> 平均 {row['mean']:.2f} mm (誤差 {row['err']:+.2f}),標準差 {row['std']:.2f},最大誤差 {row['max_abs_err']:.2f},"
                      f"紙面殘差 {row['paper_rms']:.2f} mm  ({time.time() - t0:.0f} s)")
            summary.append(row)
            if last is not None:
                log.save_snapshot(name, last[0], {"depth_scale_m": meta["depth_scale_m"], "color_intrinsics": meta["color_intrinsics"]})
        log.save_json("summary", summary)
        good = [r for r in summary if r.get("n_ok")]
        if good:
            best = min(good, key=lambda r: r["max_abs_err"] + r["std"])
            print("\n================ 總結 (真值 %.1f mm) ================" % args.truth)
            print(f"{'設定':24s} {'平均':>7s} {'誤差':>7s} {'標準差':>7s} {'最大誤差':>8s} {'紙面殘差':>8s} {'雷射':>6s}")
            for r in good:
                print(f"{r['config']:24s} {r['mean']:7.2f} {r['err']:+7.2f} {r['std']:7.2f} {r['max_abs_err']:8.2f} {r['paper_rms']:8.2f} "
                      f"{r['laser_power'] or 0:6.0f}" + ("   <-- 最佳" if r is best else ""))
            log.finish("ok", {"best": best["config"], "best_max_abs_err": best["max_abs_err"], "truth": args.truth})
        print("紀錄:", log.path)


if __name__ == "__main__":
    main()
