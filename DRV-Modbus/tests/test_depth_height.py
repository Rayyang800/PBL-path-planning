# -*- coding: utf-8 -*-
"""depth_height 測試:合成「傾斜 5° 的紙 + 一塊高 6 mm 的工件 + 深度雜訊 1.5 mm + 破洞」,
存成 v2 紀錄後用 depth_height 讀回量高度。  python tests\\test_depth_height.py"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import depth_height as dh                                 # noqa: E402
from run_log import RunLog                                # noqa: E402
from test_pen_marks import paper_canvas, PPM              # noqa: E402

K = {"fx": 914.4, "fy": 912.9, "ppx": 642.2, "ppy": 369.5}


def main():
    lay = os.path.join(ROOT, "print", "lab_sheet_layout.json")
    layout = json.load(open(lay, encoding="utf-8"))
    W, H = layout["paper_mm"]
    # 相機座標:紙面點 (x, y) -> 3D,紙中心在相機前 528 mm,繞 x 軸傾斜 5 度
    a = np.radians(5.0)
    Rm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    def to_cam(x, y, h=0.0):
        p = np.array([x - W / 2, y - H / 2, -h])
        return Rm @ p + [0, 0, 528.0]
    # 彩色:用 3D 投影出的紙角做單應
    corners = [(0, 0), (W, 0), (W, H), (0, H)]
    pix = np.float32([[K["fx"] * q[0] / q[2] + K["ppx"], K["fy"] * q[1] / q[2] + K["ppy"]] for q in (to_cam(*c) for c in corners)])
    Hp = cv2.getPerspectiveTransform(np.float32(corners) * PPM, pix)
    color = cv2.cvtColor(cv2.warpPerspective(paper_canvas(layout, []), Hp, (1280, 720), borderValue=120), cv2.COLOR_GRAY2BGR)
    # 深度:逐像素射線與「紙面 / 工件頂面」求交
    Hinv = np.linalg.inv(Hp)
    vv, uu = np.mgrid[0:720, 0:1280]
    pts = cv2.perspectiveTransform(np.float32(np.stack([uu, vv], -1)).reshape(-1, 1, 2), Hinv).reshape(720, 1280, 2) / PPM
    work = (pts[..., 0] > 120) & (pts[..., 0] < 180) & (pts[..., 1] > 80) & (pts[..., 1] < 130)   # 工件 60x50 mm
    n = Rm @ [0, 0, -1.0]
    p0 = to_cam(W / 2, H / 2)
    ray = np.stack([(uu - K["ppx"]) / K["fx"], (vv - K["ppy"]) / K["fy"], np.ones_like(uu, float)], -1)
    z_paper = (p0 @ n) / (ray @ n)                                  # 射線參數 = 深度 z
    z_top = ((p0 + n * 6.0) @ n) / (ray @ n)
    z = np.where(work, z_top, z_paper) + np.random.RandomState(0).normal(0, 1.5, z_paper.shape)
    z[200:230, 300:340] = 0                                          # 破洞
    depth = np.clip(z, 0, 65535).round().astype(np.uint16)
    tmp = tempfile.mkdtemp(prefix="depthh_")
    with RunLog("unit_depth", runs_dir=tmp, console=False) as log:
        log.add_input(lay)
        log.save_snapshot("home_before", {"color": color, "depth": depth, "n_depth": 15},
                          {"depth_scale_m": 0.001, "color_intrinsics": K})
        path = log.path
    color2, mm, dmeta = __import__("run_log").load_snapshot(path, "home_before")
    Hm, pf, st = dh.height_map(color2, mm, dmeta["intrinsics"], os.path.join(path, "inputs", "lab_sheet_layout.json"))
    on = dh.measure(Hm, pf, (150, 105), 6)
    off = dh.measure(Hm, pf, (80, 150), 6)
    print(f"紙面殘差 RMS {st['paper_rms_mm']:.2f} mm,相機到紙 {st['cam_to_plane_mm']:.1f} (PnP {st['pnp_cam_h_mm']:.1f})")
    print(f"工件上: {on['height_mm']:+.2f} mm (真值 +6.00);紙上: {off['height_mm']:+.2f} mm (真值 0)")
    assert abs(on["height_mm"] - 6.0) < 0.8 and abs(off["height_mm"]) < 0.8
    print("PASS")


if __name__ == "__main__":
    main()
