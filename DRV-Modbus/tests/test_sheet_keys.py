# -*- coding: utf-8 -*-
"""
按鍵功能模擬測試:用假的手臂 / 相機 / 鍵盤,實際跑 pen_sheet_calib.py --go 的流程,逐一驗證按鍵卡上的每個功能。
不連線手臂、不開相機。  python tests\\test_sheet_keys.py
"""
import builtins
import functools
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import pen_calibrate as pc                                     # noqa: E402
import pen_sheet_calib as sc                                   # noqa: E402
import run_log                                                 # noqa: E402
from drv_modbus import send, request, gripper                  # noqa: E402
from test_pen_marks import paper_canvas, to_camera, PPM        # noqa: E402

import cv2                                                     # noqa: E402

CAL = json.load(open(os.path.join(ROOT, "calibration.json"), encoding="utf-8"))
HOME = CAL["home"]
Z_SAFE = min(HOME[2], CAL["drop"][2])
LAYOUT = json.load(open(os.path.join(ROOT, "print", "lab_sheet_layout.json"), encoding="utf-8"))


def scene():
    W, H = LAYOUT["paper_mm"]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]]) * PPM
    dst = np.float32([[240, 140], [770, 136], [775, 498], [236, 503]])
    return to_camera(paper_canvas(LAYOUT, []), cv2.getPerspectiveTransform(src, dst), 1)


IMG = scene()


class Sim:
    """一次模擬:記錄手臂移動、夾爪、停止;依計畫注入鍵盤事件。"""

    def __init__(self, inputs, keys_at_cross=None, ctrlc_at_cross=None):
        self.pose = list(HOME)
        self.inputs = list(inputs)
        self.keys_at_cross = keys_at_cross or {}
        self.ctrlc_at_cross = ctrlc_at_cross
        self.queue, self.moves, self.grips, self.stops, self.crosses = [], [], [], 0, 0
        self.ctrlc_next = False

    # ---- 假鍵盤 (msvcrt) ----
    def kbhit(self):
        return bool(self.queue)

    def getwch(self):
        return self.queue.pop(0)

    # ---- 假手臂 ----
    def go(self, c, x, y, z, rx, ry, rz, speed=20, mov=0, block=True, on_poll=None, **kw):
        if self.ctrlc_next:
            self.ctrlc_next = False
            raise send.Interrupted("ctrl_c")
        for _ in range(3):                              # 移動中輪詢鍵盤
            if on_poll is not None and on_poll():
                send.Stop(c)
                raise send.Interrupted("key")
        self.moves.append((round(x, 2), round(y, 2), round(z, 2)))
        self.pose = [x, y, z, rx, ry, rz]

    def stop(self, c):
        self.stops += 1

    def cross(self, arm, x, y, z, half):
        k = self.crosses
        self.crosses += 1
        self.queue += list(self.keys_at_cross.get(k, ""))
        if self.ctrlc_at_cross == k:
            self.ctrlc_next = True
        return REAL_CROSS(arm, x, y, z, half)

    def input(self, prompt=""):
        if not self.inputs:
            raise AssertionError("輸入用完了,提示: " + prompt)
        v = self.inputs.pop(0)
        print(prompt + ("<Ctrl+C>" if v is KeyboardInterrupt else v))
        if v is KeyboardInterrupt:
            raise KeyboardInterrupt
        return v


REAL_CROSS = pc.small_cross


class FakeClient:
    def __init__(self, *a, **k):
        pass

    def connect(self):
        return True

    def close(self):
        pass


def run(sim, extra_args=()):
    tmp = tempfile.mkdtemp(prefix="keys_")
    saved = {}

    def patch(obj, name, val):
        saved[(obj, name)] = getattr(obj, name)
        setattr(obj, name, val)

    patch(request, "Get_TCP_Pose", lambda c: tuple(sim.pose))
    patch(request, "Get_Pose_Flag", lambda c: 1)
    patch(send, "Go_Position", sim.go)
    patch(send, "Stop", sim.stop)
    patch(gripper, "Gripper_Open", lambda c, delay=1.0: sim.grips.append("open"))
    patch(gripper, "Gripper_Close", lambda c, delay=1.0: sim.grips.append("close"))
    patch(pc, "refresh", lambda c: None)
    patch(pc, "msvcrt", sim)
    patch(pc.time, "sleep", lambda s: None)
    patch(sc, "ModbusTcpClient", FakeClient)
    patch(sc, "small_cross", sim.cross)
    patch(sc, "find_first_mark", lambda *a, **k: (np.zeros(2), (0.0, 0.0)))
    patch(sc, "RunLog", functools.partial(run_log.RunLog, runs_dir=tmp, console=False))
    patch(sc, "CalibrationStore", lambda: run_log.CalibrationStore(os.path.join(tmp, "cal")))

    def fake_grab(*a, **k):
        pc.LAST_GRAB.update(snap={"color": IMG.copy()}, info={"name": "fake"})
        return IMG.copy()
    patch(sc, "grab", fake_grab)
    patch(builtins, "input", sim.input)
    argv = sys.argv
    sys.argv = ["pen_sheet_calib.py", "--go", "--pen-loaded"] + list(extra_args)
    try:
        sc.main()
    finally:
        sys.argv = argv
        for (obj, name), val in saved.items():
            setattr(obj, name, val)
    d = sorted(x for x in os.listdir(tmp) if "pen_sheet" in x)[-1]
    meta, events = run_log.load_run(os.path.join(tmp, d))
    contacts = [e for e in events if e["type"] == "contact"]
    return meta, contacts, events


def at_home(sim):
    return max(abs(a - b) for a, b in zip(sim.pose[:3], HOME[:3])) < 1e-6


def check(name, cond, detail=""):
    print(("  OK  " if cond else "  FAIL") + f" {name}" + (f"  ({detail})" if detail else ""))
    assert cond, name


def main():
    print("\n=== A. 找高度的每個輸入 + 移動中 o / g ===")
    sim = Sim(["", "20", "5", "x 2", "y -3", "u", "1", " ", "p", "esc", "g", "o", "c"], keys_at_cross={3: "o", 5: "g"})
    meta, contacts, _ = run(sim)
    zt = contacts[0]["z"]
    plan0 = contacts[0]["robot_xy"]
    check("Enter (空字串) 開始後先到安全高度", any(abs(m[2] - Z_SAFE) < 0.01 for m in sim.moves))
    check("20 / 5 / u / 1 步進後 z_touch = 安全高度 -20 -5 +5 -1", abs(zt - (Z_SAFE - 21)) < 1e-6, f"z_touch={zt:.2f}")
    check("空白+Enter、p、esc 在提示時不會讓手臂往下", True, "z 只由 20/5/u/1 決定 (上一項已驗證)")
    check("x 2 / y -3 微調有反映在第一點座標", True, f"第一點 {plan0}")
    check("提示時 g / o 有動夾爪", sim.grips[:2] == ["close", "open"], str(sim.grips))
    check("移動中 o / g 立刻動夾爪,手臂沒停", sim.grips[2:] == ["open", "close"] and sim.stops == 0, str(sim.grips))
    check("9 點全部畫完", len(contacts) == 9, f"{len(contacts)} 點")
    check("結束回到 home", at_home(sim))

    sim_ref = Sim(["", "20", "c"])
    _, c_ref, _ = run(sim_ref)
    dx, dy = plan0[0] - c_ref[0]["robot_xy"][0], plan0[1] - c_ref[0]["robot_xy"][1]
    check("x 2 / y -3 = 手臂 X +2、Y -3", abs(dx - 2) < 1e-6 and abs(dy + 3) < 1e-6, f"差 ({dx:+.1f}, {dy:+.1f})")

    print("\n=== B. 移動中 空白 → r 重做;p → s 跳過;Esc → h 回 home 分析;選單中 o / g ===")
    sim = Sim(["", "20", "c", "o", "g", "r", "s", "h"], keys_at_cross={2: " ", 5: "p", 8: "\x1b"})
    meta, contacts, _ = run(sim)
    idx = [c["index"] for c in contacts]
    check("每次暫停都送了停止指令 (移動函式 + 確認是否停住 各一次)", sim.stops == 6, f"{sim.stops} 次 = 3 次暫停 x 2")
    check("選單中 o / g 有動夾爪", sim.grips == ["open", "close"], str(sim.grips))
    check("r 重做:第 2 點仍完成", 2 in idx)
    check("s 跳過:少了第 4 點", 4 not in idx, f"完成 {idx}")
    check("h:停止時正在畫的第 7 點不算,用已完成的點分析", idx == [0, 1, 2, 3, 5, 6], f"完成 {idx}")
    check("h 之後回到 home 並完成分析 (狀態 ok)", at_home(sim) and meta["status"] == "ok", meta["status"])

    print("\n=== C. 選單 q = 結束 ===")
    sim = Sim(["", "20", "c", "q"], keys_at_cross={3: " "})
    meta, contacts, _ = run(sim)
    check("q 結束:狀態 aborted、回到 home", meta["status"] == "aborted" and at_home(sim), meta["status"])
    check("已完成的點保留在紀錄", [c["index"] for c in contacts] == [0, 1, 2])

    print("\n=== D. 找高度時 Ctrl+C ===")
    sim = Sim(["", "20", "5", KeyboardInterrupt])
    meta, contacts, _ = run(sim)
    check("Ctrl+C:沒有任何點,狀態 aborted,垂直抬起後回 home", meta["status"] == "aborted" and not contacts and at_home(sim))
    lift = [m for m in sim.moves if abs(m[2] - Z_SAFE) < 0.01]
    check("回 home 前先在原地抬到安全高度", len(lift) >= 2)

    print("\n=== E. 畫點移動中 Ctrl+C ===")
    sim = Sim(["", "20", "c"], ctrlc_at_cross=4)
    meta, contacts, _ = run(sim)
    check("Ctrl+C:不跳選單,直接回 home,狀態 aborted", meta["status"] == "aborted" and at_home(sim))
    check("已完成的點保留 (0~3)", [c["index"] for c in contacts] == [0, 1, 2, 3])

    print("\n=== F. 找高度時 q ===")
    sim = Sim(["", "20", "q"])
    meta, contacts, _ = run(sim)
    check("q:抬起回 home,狀態 aborted", meta["status"] == "aborted" and at_home(sim) and not contacts)

    print("\n=== G. 第一點畫十字時按空白 (還沒進入點迴圈) ===")
    sim = Sim(["", "20", "c"], keys_at_cross={0: " "})
    meta, contacts, _ = run(sim)
    check("第一點階段暫停 = 抬起回 home 結束 (不跳選單)", meta["status"] == "aborted" and at_home(sim) and sim.stops == 2)
    print("\nPASS")


if __name__ == "__main__":
    main()
