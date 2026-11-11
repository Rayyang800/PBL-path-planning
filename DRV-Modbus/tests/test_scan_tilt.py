# -*- coding: utf-8 -*-
"""scan_tilt 的離線測試:用一台「假手臂」(已知歐拉慣例、手眼) 產生試轉量測,確認能解回來、規劃的視角真的看向物體且筆尖安全。

  python tests/test_scan_tilt.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scan_tilt as st  # noqa: E402

A = np.array([[-0.0027, -1.019], [-1.025, 0.0117]])
TCP_HOME = [364.5, -83.0, 809.4, -179.2, 0.6, -107.8]
Z_TOUCH = 470.8


class FakeArm:
    def __init__(self, conv, p, Rtc):
        self.conv, self.p, self.Rtc = conv, np.asarray(p, float), Rtc
        self.R_PB, self.s = st.paper_to_base(A)
        # 紙面原點在手臂座標的位置 (任意,只要一致)
        self.O_B = np.array([610.0, 84.0, Z_TOUCH - 150.0])

    def measure(self, tcp):
        """TCP -> tag PnP 會量到的 (R, C) (紙面座標)。"""
        Rt = st.R_tool(tcp[3:], self.conv)
        C_B = np.asarray(tcp[:3]) + Rt @ self.p
        C_P = self.R_PB.T @ (C_B - self.O_B) / self.s
        R_P_cam = self.R_PB.T @ (Rt @ self.Rtc)
        return {"R": R_P_cam.T, "C": C_P}


def make(conv):
    Rtc = st.Rotation.from_euler("xyz", [1.0, -1.5, 92.0], degrees=True).as_matrix()   # 相機在工具上轉了約 90°
    arm = FakeArm(conv, [8.0, -140.0, -30.0], Rtc)
    base = [360.0, -95.0, 682.0, *TCP_HOME[3:]]
    probes = []
    R0 = st.R_tool(base[3:], conv)
    for axis, deg in ((None, 0), ([1, 0, 0], 10), ([0, 1, 0], 10), ([1, 0, 0], -10)):
        R = R0 if axis is None else st.tilt_base(axis, deg) @ R0
        tcp = base[:3] + st.angles_of(R, conv)
        m = arm.measure(tcp)
        probes.append({"tcp": tcp, "R": m["R"], "C": m["C"]})
    return arm, probes


def test_solve_both_conventions():
    for conv in st.CONVENTIONS:
        arm, probes = make(conv)
        he = st.solve_handeye(probes, arm.R_PB, arm.s)
        assert he["conv"] == conv, (conv, he["rot_err_deg"])
        assert np.allclose(he["p"], arm.p, atol=0.5), he["p"]
        assert he["pos_rms_mm"] < 0.1
        print(f"  慣例 {conv}:解出 p = {np.round(he['p'], 2)},旋轉誤差 {he['rot_err_deg']}")


def test_plan_looks_at_object_and_safe():
    arm, probes = make("xyz")
    he = st.solve_handeye(probes, arm.R_PB, arm.s)
    kin = st.Kinematics(he, arm.R_PB, arm.s, probes[0]["tcp"], probes[0]["C"])
    obj, h = [180.0, 105.0], 40.0
    views, notes = st.plan_tilted(kin, obj, h, [20, 35], 8, 330.0, Z_TOUCH, 40.0, TCP_HOME, 250.0)
    assert len(views) >= 12, notes
    O = np.array([obj[0], obj[1], -h / 2])
    for v in views:
        m = arm.measure(v["tcp"])                      # 假手臂真的走到這個 TCP 時,相機在哪、看哪
        axis = m["R"].T @ np.array([0, 0, 1.0])        # 相機光軸 (紙面座標)
        to_obj = (O - m["C"]) / np.linalg.norm(O - m["C"])
        ang = np.degrees(np.arccos(np.clip(axis @ to_obj, -1, 1)))
        assert ang < 1.0, (v["name"], ang)
        assert v["tcp"][2] - Z_TOUCH >= h + 40 - 1e-6
        pred = kin.camera_pose(v["tcp"])
        assert np.linalg.norm(pred["C"] - m["C"]) < 0.5
    tl = sorted({round(v["tilt"]) for v in views})
    print(f"  {len(views)} 個傾斜視角,傾斜 {tl} 度,距離 {min(v['dist'] for v in views):.0f}~{max(v['dist'] for v in views):.0f} mm;"
          f"筆尖最低離紙 {min(v['pen_tip_above_paper'] for v in views):.0f} mm")


if __name__ == "__main__":
    test_solve_both_conventions()
    print("PASS test_solve_both_conventions")
    test_plan_looks_at_object_and_safe()
    print("PASS test_plan_looks_at_object_and_safe")
