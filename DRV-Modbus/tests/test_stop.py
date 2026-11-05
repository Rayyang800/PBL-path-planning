# -*- coding: utf-8 -*-
"""停止指令測試 (假的 Modbus 連線,不碰手臂)。  python tests\\test_stop.py
檢查:移動中 on_poll 回傳 True 或發生 Ctrl+C 時,Go_Position 會寫 0x0300 = 0 並丟 Interrupted。"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drv_modbus import send, request   # noqa: E402


class FakeClient:
    def __init__(self):
        self.writes = []

    def write_register(self, addr, val, slave):
        self.writes.append((addr, val))

    def write_registers(self, addr, vals, slave):
        self.writes.append((addr, "payload"))


def main():
    pose = [100.0, 0.0, 500.0, -179.0, 0.5, -107.0]
    state = {"t0": time.time()}
    request.Get_TCP_Pose = lambda c: pose if time.time() - state["t0"] < 0.05 else [100.0, 0.0, 490.0, -179.0, 0.5, -107.0]
    request.Get_Pose_Flag = lambda c: 2           # 一直在移動中

    # 1. 按鍵中斷
    c = FakeClient()
    n = {"k": 0}

    def poll():
        n["k"] += 1
        return n["k"] > 3
    try:
        send.Go_Position(c, 100, 0, 400, -179, 0.5, -107, 15, send.MovL, on_poll=poll)
        raise AssertionError("應該要中斷")
    except send.Interrupted as e:
        assert str(e) == "key" and c.writes[-1] == (0x0300, 0), c.writes
        print("OK 按鍵 -> 寫 0x0300=0,Interrupted('key')")

    # 2. Ctrl+C (在等待迴圈中發生 KeyboardInterrupt)
    c = FakeClient()

    def poll_ctrl_c():
        raise KeyboardInterrupt
    try:
        send.Go_Position(c, 100, 0, 400, -179, 0.5, -107, 15, send.MovL, on_poll=poll_ctrl_c)
        raise AssertionError("應該要中斷")
    except send.Interrupted as e:
        assert str(e) == "ctrl_c" and c.writes[-1] == (0x0300, 0), c.writes
        print("OK Ctrl+C -> 寫 0x0300=0,Interrupted('ctrl_c')")
    print("PASS")


if __name__ == "__main__":
    main()
