# -*- coding: utf-8 -*-
"""pen_sheet_calib 合成測試 (不需要手臂與相機)。  python tests\\test_sheet_calib.py
模擬完整流程:粗略換算偏 15 mm 且旋轉差 1.5° -> 第一點十字 -> 修正 -> 其餘格點 -> 分析;
紙上有擦過留下的淡痕與髒點。檢查:修正後落點接近目標、擬合還原真實轉換、留一法誤差小。"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import pen_sheet_calib as sc                                  # noqa: E402
from paper_frame import PaperFrame                            # noqa: E402
from test_pen_marks import paper_canvas, to_camera, PPM       # noqa: E402


def run(grid=(3, 3), scale_err=1.0, seed=0):
    lay = os.path.join(ROOT, "print", "lab_sheet_layout.json")
    layout = json.load(open(lay, encoding="utf-8"))
    W, H = layout["paper_mm"]
    a = np.radians(1.2)
    A = (np.array([[0, -1], [-1, 0]]) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])) * scale_err
    t = np.array([600.0, 60.0]) + [3.2, -1.7]
    Ainv = np.linalg.inv(A)
    land = lambda cmd: Ainv @ (np.asarray(cmd) - t)                       # noqa: E731  指令 -> 實際落點 (紙面)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]]) * PPM
    dst = np.float32([[240, 140], [770, 136], [775, 498], [236, 503]])
    Hp = cv2.getPerspectiveTransform(src, dst)
    cam = lambda marks, s: to_camera(paper_canvas(layout, [(m, np.degrees(np.arctan2(*(Ainv @ [1, 0])[::-1])))  # noqa
                                                               for m in marks], stroke_mm=4.0, ink_mm=1.0,
                                                  dirt=dirt), Hp, s)
    dirt = [((W / 2 + 3, H / 2 - 20), 1.0), ((60, 150), 1.6)]           # 擦不乾淨的痕跡/髒點
    # 粗略換算:偏 (12,-9) mm、再多轉 1.5°
    b = np.radians(1.5)
    Rb = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
    pix_of = lambda P: cv2.perspectiveTransform(np.float32(P).reshape(-1, 1, 2) * PPM, Hp).reshape(-1, 2)  # noqa
    corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    H_rough = cv2.getPerspectiveTransform(pix_of(corners), np.float32([Rb @ (A @ p + t) + [12.0, -9.0] for p in corners]))

    before = cam([], seed + 1)
    pf = PaperFrame(layout=lay)
    assert pf.detect(before)
    targets = sc.grid_points(pf, *grid, 12.0)
    robot = sc.paper_to_robot(pf, H_rough, targets, np.zeros(2))
    cmd0 = robot[0]
    first = cam([land(cmd0)], seed + 2)
    off, _ = sc.find_first_mark(before, first, lay, cmd0, H_rough)
    assert off is not None
    robot = sc.paper_to_robot(pf, H_rough, targets, off)
    cmds = [cmd0] + list(robot[1:])
    landed = [land(cc) for cc in cmds]
    miss = [float(np.hypot(*(landed[k] - targets[k]))) for k in range(1, len(targets))]
    after = cam(landed, seed + 3)
    points = [{"index": k, "robot_xy": np.asarray(cc).tolist()} for k, cc in enumerate(cmds)]
    res = sc.analyze(before, after, lay, points, H_rough, off, 20.0)
    return res, A, t, targets, miss, W, H


def main():
    for grid, s in (((3, 3), 1.0), ((4, 3), 1.006), ((2, 2), 1.0)):
        res, A, t, targets, miss, W, H = run(grid, s)
        print(f"\n[格點 {grid[0]}x{grid[1]},真實比例 {s}]")
        print(f"修正後落點離目標:最大 {max(miss):.1f} mm (未修正時粗略換算可差 15+ mm)")
        print(sc.summarize(res))
        assert res["ok"] and res["n_points"] == grid[0] * grid[1], res.get("reason")
        Af, tf = np.array(res["A"]), np.array(res["t"])
        test = np.array([[50.0, 50.0], [W - 50, H - 50], [W / 2, H / 2], [W - 50, 50]])
        err = [float(np.linalg.norm(Af @ p + tf - (A @ p + t))) for p in test]
        print("整張紙 4 個測試點的 TCP 指令誤差:", ", ".join(f"{e:.2f}" for e in err), "mm")
        assert max(miss) < 8.0 and max(err) < 0.6      # 修正只消除位移;粗略換算的旋轉誤差在 120 mm 外約 3 mm
        if res["loo_max_mm"] is not None:
            assert res["loo_max_mm"] < 0.8
    print("\nPASS")


if __name__ == "__main__":
    main()
