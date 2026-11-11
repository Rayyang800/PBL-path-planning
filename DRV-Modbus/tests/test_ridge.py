# -*- coding: utf-8 -*-
"""ridge.extract_ridge 的離線測試:合成屋脊 (∧) 形點雲,脊線位置 / 高度 / 方向要對;再加雜點與一條電線。

  python tests/test_ridge.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ridge  # noqa: E402


def make_roof(cx=150, cy=110, L=120, half_w=40, H=38, ang_deg=5.0, slope_along=0.0, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    a = np.radians(ang_deg)
    u = np.array([np.sin(a), np.cos(a)])          # 脊線方向 (大致沿紙面 y)
    v = np.array([u[1], -u[0]])
    gx, gy = np.meshgrid(np.arange(cx - 100, cx + 100, 0.8), np.arange(cy - 100, cy + 100, 0.8))
    P = np.c_[gx.ravel(), gy.ravel()]
    s = (P - [cx, cy]) @ u
    t = (P - [cx, cy]) @ v
    inside = (np.abs(s) < L / 2) & (np.abs(t) < half_w)
    h = np.where(inside, (H + slope_along * s) * (1 - np.abs(t) / half_w), 0.0)
    h = h + rng.normal(0, noise, h.shape)
    return P, h, u, np.array([cx, cy])


def test_roof():
    P, h, u, ctr = make_roof()
    r = ridge.extract_ridge(P, h)
    e = np.array(r["line_xyh"])
    d = np.array(r["dir_xy"])
    assert abs(abs(d @ u) - 1) < 1e-3, d
    v = np.array([u[1], -u[0]])
    lat = [(p[:2] - ctr) @ v for p in e]
    assert max(abs(x) for x in lat) < 0.8, lat
    assert abs(r["h_mean_mm"] - 38) < 2.0, r["h_mean_mm"]       # 尖端附近取「最高的一帶」中位數,會略低於尖端
    assert 90 < r["length_mm"] < 112, r["length_mm"]
    assert r["planes"] and abs(r["planes"]["ridge_angle_deg"] - 2 * np.degrees(np.arctan(40 / 38))) < 3
    print(f"  屋脊:長 {r['length_mm']:.1f}、高 {r['h_mean_mm']:.2f}、橫向偏 {np.round(lat, 2)}、斜面夾角 {r['planes']['ridge_angle_deg']:.1f}°")


def test_roof_with_wire_and_slope():
    P, h, u, ctr = make_roof(slope_along=0.02)
    wire = (np.abs(P[:, 1] - 15) < 1.5) & (P[:, 0] > 60)            # 遠處一條電線 (高 20 mm)
    h = h.copy()
    h[wire] = 20.0
    r = ridge.extract_ridge(P, h)
    assert r["length_mm"] < 115, r["length_mm"]                      # 不能把電線算進來
    assert abs(r["slope_deg"] - np.degrees(np.arctan(0.02))) < 0.6, r["slope_deg"]
    path = ridge.path_along(r["line_xyh"], 2.0)
    assert np.allclose(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1), 2.0, atol=0.1)
    print(f"  有電線 + 坡度:長 {r['length_mm']:.1f}、坡度 {r['slope_deg']:.2f}°、路徑 {len(path)} 點")


if __name__ == "__main__":
    test_roof()
    print("PASS test_roof")
    test_roof_with_wire_and_slope()
    print("PASS test_roof_with_wire_and_slope")
