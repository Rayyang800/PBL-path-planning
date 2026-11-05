# -*- coding: utf-8 -*-
"""紀錄 v2 測試 (不碰手臂/相機):深度快照存讀、console.log、current()、
send.Go_Position / gripper 自動寫 cmd_move / tcp_trace / move_done / cmd_gripper、run_main + sys.exit。
  python tests\\test_run_log_v2.py"""
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_log                                       # noqa: E402
from run_log import RunLog, load_run, load_snapshot  # noqa: E402
from drv_modbus import send, request, gripper        # noqa: E402


class FakeClient:
    def __init__(self):
        self.writes = []

    def write_register(self, a, v, s):
        self.writes.append((a, v))

    def write_registers(self, a, v, s):
        self.writes.append((a, "payload"))


def main():
    tmp = tempfile.mkdtemp(prefix="runlog2_")
    color = np.random.RandomState(0).randint(0, 255, (72, 128, 3), np.uint8)
    depth = np.full((72, 128), 520, np.uint16)
    depth[10:20, 10:20] = 0                                    # 無效區
    depth[30:40, 60:80] = 515                                  # 高 5 單位的「工件」
    info = {"serial": "TEST", "depth_scale_m": 0.001, "color_intrinsics": {"fx": 91.0, "fy": 91.0, "ppx": 64, "ppy": 36}}

    # 假手臂:移動 0.35 秒
    state = {"t0": None}

    def pose(c):
        if state["t0"] is None:
            return [100.0, 0, 500, -179, 0.5, -107]
        k = min(1.0, (time.time() - state["t0"]) / 0.35)
        return [100.0, 0, 500 - 20 * k, -179, 0.5, -107]
    request.Get_TCP_Pose = pose
    request.Get_Pose_Flag = lambda c: 1 if state["t0"] and time.time() - state["t0"] > 0.36 else 2

    with RunLog("unit", runs_dir=tmp, params={"x": 1}) as log:
        assert run_log.current() is log
        print("這行要進 console.log")
        log.save_snapshot("home", {"color": color, "depth": depth, "n_depth": 15, "depth_valid_frac": 0.98,
                                   "time": time.time()}, info)
        log.save_snapshot("color_only", color)
        c = FakeClient()
        state["t0"] = time.time()
        send.Go_Position(c, 100, 0, 480, -179, 0.5, -107, 15, send.MovL)
        gripper.Gripper_Close(c, delay=0)
        path = log.path
    assert run_log.current() is None

    meta, events = load_run(path)
    types = [e["type"] for e in events]
    print("事件:", {t: types.count(t) for t in dict.fromkeys(types)})
    assert {"snapshot", "cmd_move", "tcp_trace", "move_done", "cmd_gripper"} <= set(types)
    assert types.count("tcp_trace") >= 2
    assert meta["camera"]["serial"] == "TEST" and len(meta["snapshots"]) == 2 and meta["schema"] == 2
    assert "git" in meta
    c2, mm, dm = load_snapshot(path, "home")
    assert np.array_equal(c2, color) and np.isnan(mm[15, 15]) and abs(mm[0, 0] - 520.0) < 1e-3
    assert abs((mm[0, 0] - mm[35, 70]) - 5.0) < 1e-3 and dm["n_frames_median"] == 15
    c3, mm3, _ = load_snapshot(path, "color_only")
    assert mm3 is None and np.array_equal(c3, color)
    with open(os.path.join(path, "console.log"), encoding="utf-8") as f:
        log_text = f.read()
    assert "這行要進 console.log" in log_text and "Move done!" in log_text
    print("OK 深度快照 / console.log / 自動軌跡 / 夾爪")

    # run_main + sys.exit("原因")
    sys.argv = ["fake_script.py", "--go"]
    try:
        run_log.run_main("exit_test", lambda: sys.exit("手臂不在 home"), runs_dir=tmp)
    except SystemExit:
        pass
    last = sorted(d for d in os.listdir(tmp) if "exit_test" in d)[-1]
    m2, _ = load_run(os.path.join(tmp, last))
    assert m2["status"] == "exited" and "home" in m2["result"]["exit"], m2["status"]
    print("OK run_main + sys.exit -> status exited")
    print("PASS")


if __name__ == "__main__":
    main()
