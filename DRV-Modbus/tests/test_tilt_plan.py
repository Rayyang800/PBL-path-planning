# -*- coding: utf-8 -*-
"""傾斜描繪的規劃與執行前檢查 (純計算,不碰手臂)。  python tests\\test_tilt_plan.py
校正、home、罐頭用 2026-11-18 的實測值。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import circle_rim  # noqa: E402
import tilt_plan  # noqa: E402
import twin_check  # noqa: E402

HOME = [364.477, -83.009, 809.429, -179.198, 0.559, -107.773]
CAL = {"A": [[-0.0027471250329477387, -1.0190530301444314], [-1.0250426430135346, 0.011669952203069545]],
       "t": [610.571900650227, 84.44396101433077], "z_touch": 473.08}
TIP = {"p": np.array([-44.5, 46.2, 240.9]), "z_unc": 5.8}
CENTER, R_OBJ, H_OBJ = [140.5, 112.6], 39.3, 47.7


def make(deg, off=None):
    tp = tilt_plan.TiltPlan(CAL, HOME, TIP, CENTER, R_OBJ, H_OBJ, deg)
    path = lambda r: circle_rim.circle_path(CENTER, r, 0.0, 8.0, 0.0)
    if off is None:
        off = tilt_plan.bottom_offset(tp, path, [0, 0])
    tp.set_path(path(R_OBJ + off), [0, 0], "bottom")
    return tp, off


def main():
    tw = twin_check.Twin()
    q = tw.home_q(HOME)
    p, R = tw.fk(q)
    assert np.linalg.norm(p - HOME[:3]) < 0.05
    print("OK 分身解得出 home")

    tp, off = make(15)
    assert 8 <= off <= 14, off
    assert tp.verify()["ok"]
    print(f"OK 15 度底面:離罐壁 {off} mm,整套動作通過")

    for P, pose in zip(tp.X, tp.draw):
        assert np.linalg.norm(tp.tip_world(pose) - P) < 1e-6
        up, vert, home = tp.escape(pose)
        assert np.linalg.norm(tp.tip_world(vert) - tp.tip_world(up)) < 1e-6      # 繞筆尖轉:筆尖不動
        assert tp.tip_height(up) >= H_OBJ + tilt_plan.CLEAR - 1e-6
        assert home == HOME and tp.tip_world(pose)[2] < tp.tip_world(up)[2]
    print("OK 每一點:筆尖在目標上、抬起後繞筆尖轉回垂直")

    tp, _ = make(15, off=2.0)
    try:
        tp.verify()
        raise AssertionError("貼著罐壁 2 mm 應該被擋")
    except tilt_plan.PlanError:
        print("OK 太靠近罐壁 -> 擋下")

    for deg in (25, 45):
        try:
            make(deg)[0].verify()
            raise AssertionError(f"{deg} 度應該被擋")
        except tilt_plan.PlanError:
            print(f"OK {deg} 度 -> 擋下")

    tp, _ = make(15)
    down = [[p[0], p[1], p[2] - 25.0] + p[3:] for p in tp.draw]
    try:
        tp.verify(down)
        raise AssertionError("壓進桌面應該被擋")
    except tilt_plan.PlanError:
        print("OK 壓進紙面 -> 擋下")
    print("PASS")


if __name__ == "__main__":
    main()
