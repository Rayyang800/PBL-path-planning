# -*- coding: utf-8 -*-
"""pen_quick_calib 分析測試 (不需要手臂與相機)。  python tests\\test_quick_calib.py
5 個墨點 (第一點 + 手臂 ±X/±Y 10 mm),粗略換算故意偏 15 mm,旁邊有舊髒點;檢查墨點對應與旋轉/平移還原。"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import pen_quick_calib as q                                  # noqa: E402
from test_pen_marks import paper_canvas, to_camera, PPM       # noqa: E402


def run(step, rot_deg, dot_mm=1.2, seed=0):
    lay_path = os.path.join(ROOT, "print", "lab_sheet_layout.json")
    with open(lay_path, encoding="utf-8") as f:
        layout = json.load(f)
    W, H = layout["paper_mm"]
    a = np.radians(rot_deg)
    A = np.array([[0, -1], [-1, 0]]) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = np.array([600.0, 60.0]) + [3.2, -1.7]                 # 含筆尖偏移
    Ainv = np.linalg.inv(A)
    P0 = np.array([W / 2 - 20, H / 2 + 10])
    C0 = A @ P0 + t
    cmds = [C0] + [C0 + np.array(d) * step for d in q.OFFSET_DIRS]
    paper = [Ainv @ (cc - t) for cc in cmds]
    dirt = [((P0[0] + 6, P0[1] - 14), 1.0), ((P0[0] - 25, P0[1] + 3), 1.5)]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]]) * PPM
    dst = np.float32([[240, 140], [770, 136], [775, 498], [236, 503]])
    Hp = cv2.getPerspectiveTransform(src, dst)
    before = to_camera(paper_canvas(layout, [], dirt=dirt), Hp, seed + 1)
    after = to_camera(paper_canvas(layout, [], dots=[(p, dot_mm) for p in paper], dirt=dirt), Hp, seed + 2)
    pix_of = lambda P: cv2.perspectiveTransform(np.float32(P).reshape(-1, 1, 2) * PPM, Hp).reshape(-1, 2)  # noqa
    corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    H_rough = cv2.getPerspectiveTransform(pix_of(corners), np.float32([A @ p + t + [12.0, -9.0] for p in corners]))
    points = [{"index": k, "robot_xy": cc.tolist(), "contact_z": 470.0} for k, cc in enumerate(cmds)]
    res = q.analyze(before, after, lay_path, points, H_rough, step, 0.0)
    return res, A, t, P0, W, H


def main():
    for step, rot in ((10.0, 1.5), (10.0, -3.0), (15.0, 0.5)):
        res, A, t, P0, W, H = run(step, rot)
        print(f"\n[step {step:g} mm, 真實旋轉偏 {rot:+.1f}°]")
        print(q.summarize(res))
        assert res["ok"] and res["n_points"] == 5, res.get("reason")
        R, tt = np.array(res["A"]), np.array(res["t"])
        tests = [P0, P0 + [20, 0], P0 + [0, 20], P0 + [60, 40]]
        err = [float(np.linalg.norm(R @ p + tt - (A @ p + t))) for p in tests]
        print("TCP 指令誤差 (第一點、+20、+20、~72 mm 外):", ", ".join(f"{e:.2f}" for e in err), "mm")
        assert err[0] < 0.5 and err[1] < 0.8 and abs(res["rotation_vs_rough_deg"]) < 5
    print("\nPASS")


if __name__ == "__main__":
    main()
