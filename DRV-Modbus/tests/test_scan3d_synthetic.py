# -*- coding: utf-8 -*-
"""scan3d 的離線測試 (不需手臂/相機):量測、對齊錨點、規劃的安全淨空、手臂位姿修正。

  python tests/test_scan3d_synthetic.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scan3d  # noqa: E402


def sample_box(cx, cy, L, Wd, Hh, ang_deg=0.0, step=0.7, noise=0.3, rng=None):
    """紙面 (200x150 範圍) + 一個長方體:頂面與四個側面的點 (像多視角看到的)。"""
    rng = rng or np.random.default_rng(0)
    a = np.radians(ang_deg)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    gx, gy = np.meshgrid(np.arange(cx - 60, cx + 60, step), np.arange(cy - 60, cy + 60, step))
    P = np.c_[gx.ravel(), gy.ravel()]
    loc = (P - [cx, cy]) @ Rm
    inside = (np.abs(loc[:, 0]) <= L / 2) & (np.abs(loc[:, 1]) <= Wd / 2)
    h = np.where(inside, Hh, 0.0)
    pts, hs = [P], [h]
    for sx, sy, n in ((1, 0, Wd), (-1, 0, Wd), (0, 1, L), (0, -1, L)):       # 側面
        t = np.arange(-n / 2, n / 2, step)
        z = np.arange(0.5, Hh, step)
        T, Z = np.meshgrid(t, z)
        if sx:
            lp = np.c_[np.full(T.size, sx * L / 2), T.ravel()]
        else:
            lp = np.c_[T.ravel(), np.full(T.size, sy * Wd / 2)]
        pts.append(lp @ Rm.T + [cx, cy])
        hs.append(Z.ravel())
    xy, hh = np.vstack(pts), np.concatenate(hs)
    return xy + rng.normal(0, noise * 0.3, xy.shape), hh + rng.normal(0, noise, hh.shape)


def test_cube_measure():
    xy, h = sample_box(150, 100, 30, 30, 30, ang_deg=20)
    m = scan3d.measure_object(xy, h)
    assert abs(m["h_top_mm"] - 30) < 0.5, m["h_top_mm"]
    t = m["top_rect"]
    assert abs(t["long_mm"] - 30) < 2.0 and abs(t["short_mm"] - 30) < 2.0, (t["long_mm"], t["short_mm"])
    assert abs(m["volume_cm3"] - 27.0) < 3.5, m["volume_cm3"]
    print(f"  立方體 30mm:頂 {t['long_mm']:.1f} x {t['short_mm']:.1f}、高 {m['h_top_mm']:.2f}、體積 {m['volume_cm3']:.1f} cm³")


def test_box_and_L_shape():
    xy, h = sample_box(150, 100, 60, 25, 15, ang_deg=-10)
    m = scan3d.measure_object(xy, h)
    t = m["top_rect"]
    assert abs(t["long_mm"] - 60) < 2.5 and abs(t["short_mm"] - 25) < 2.0, (t["long_mm"], t["short_mm"])
    # L 型:低的長板 + 高的短塊 -> 頂面 = 高的那塊,最高點 = 高塊高度
    xy1, h1 = sample_box(150, 100, 80, 40, 10)
    xy2, h2 = sample_box(175, 100, 30, 40, 35, rng=np.random.default_rng(1))
    keep1 = ~((np.abs(xy1[:, 0] - 175) < 15) & (np.abs(xy1[:, 1] - 100) < 20))
    xy, h = np.vstack([xy1[keep1], xy2]), np.concatenate([h1[keep1], h2])
    m = scan3d.measure_object(xy, h)
    assert abs(m["h_max_mm"] - 35) < 1.5 and abs(m["h_top_mm"] - 35) < 1.0, (m["h_max_mm"], m["h_top_mm"])
    f = m["footprint_rect"]
    assert abs(f["long_mm"] - 80) < 4 and abs(f["short_mm"] - 40) < 3, (f["long_mm"], f["short_mm"])
    print(f"  L 型:最高 {m['h_max_mm']:.1f}、底面 {f['long_mm']:.1f} x {f['short_mm']:.1f}")


def test_anchor_shift():
    xy, h = sample_box(150, 100, 30, 30, 30)
    a0, how = scan3d.anchor_xy(xy, h)
    a1, how1 = scan3d.anchor_xy(xy + [3.0, -2.0], h)
    assert how == how1 == "top"
    assert np.allclose(a1 - a0, [3.0, -2.0], atol=0.4), a1 - a0


def test_plan_clearance():
    R = np.eye(3)
    R[1, 1] = R[2, 2] = -1                      # 相機朝下 (紙面 z 指向桌面)
    home_pose = scan3d.pose_from_center(R, np.array([150.0, 100.0, -507.0]))
    A = np.array([[0.0, -1.0], [-1.0, 0.0]])
    tcal = np.array([610.0, 84.0])
    z_touch = 470.8
    tcp_home = [364.5, -83.0, 809.4, -179.2, 0.6, -107.8]
    for h_obj in (10.0, 60.0, 150.0):
        args = argparse.Namespace(cam_h=380.0, min_range=250.0, clearance=40.0, radius=60.0, n_ring=8, max_travel=150.0)
        views, info = scan3d.plan_views({"center": [150, 100], "h_max": h_obj, "bbox": [135, 85, 165, 115]},
                                        home_pose, tcp_home, A, tcal, z_touch, args, (296.3, 210.6))
        pen_h = views[0]["tcp"][2] - z_touch
        assert pen_h >= h_obj + 40 - 1e-6 or info["cam_h_scan"] >= info["cam_h_home"] - 1e-6, (h_obj, pen_h)
        assert info["cam_h_scan"] - h_obj >= 250 - 1e-6 or info["cam_h_scan"] >= info["cam_h_home"] - 1e-6
        assert all(v["tcp"][2] <= tcp_home[2] + 1e-6 for v in views)
        print(f"  物體 {h_obj:.0f} mm:相機離紙 {info['cam_h_scan']:.0f}、筆尖淨空 {info['pen_clearance_over_object']:.0f} mm、{len(views)} 點")


def test_fix_arm_poses():
    R = np.eye(3)
    bias = np.array([4.0, -6.0, 1.0])
    cap = []
    for k in range(4):
        pred = np.array([100.0 + k * 10, 50.0, -380.0])
        cap.append({"pose_source": "tags", "pred_C": pred, "pose": scan3d.pose_from_center(R, pred + bias)})
    pred = np.array([140.0, 60.0, -380.0])
    cap.append({"name": "x", "pose_source": "arm", "pred_C": pred, "pose": scan3d.pose_from_center(R, pred)})
    scan3d.fix_arm_poses(cap)
    assert cap[-1]["pose_source"] == "arm+bias" and np.allclose(cap[-1]["pose"]["C"], pred + bias)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
