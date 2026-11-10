# -*- coding: utf-8 -*-
"""
鍵盤即時控制夾爪 (不移動手臂)。
  python gripper_ctl.py
  o = 張開 (DO7)   g = 夾緊 (DO8)   s = 顯示目前 DO 狀態   q 或 Ctrl+C = 離開
按鍵直接生效,不用按 Enter。手臂閒置斷線時會自動重連。
"""
import sys

from pymodbus.client import ModbusTcpClient

from drv_modbus import gripper
from drv_modbus.conn import refresh
from keyin import wait_key

HELP = "鍵盤控制夾爪 (不會動手臂):o = 張開   g = 夾緊   s = 讀目前狀態   q 或 Ctrl+C = 離開"

ROBOT_IP = "192.168.1.1"


def show(c):
    refresh(c)
    r = c.read_holding_registers(gripper.DO_REGISTER, 1, slave=2)
    v = None if r.isError() else r.registers[0]
    state = {gripper.DO_OPEN: "張開", gripper.DO_CLOSE: "夾緊", 0: "全部關閉"}.get(v, f"其他 ({v})")
    print(f"  DO 0x02FE = {v} -> {state}")


def main():
    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    print(HELP)
    show(c)
    try:
        while True:
            ch = wait_key().lower()
            if ch in ("q", "\x03"):
                break
            if ch in ("o", "g"):
                refresh(c)
                (gripper.Gripper_Close if ch == "g" else gripper.Gripper_Open)(c, delay=0)
                print("送出:" + ("夾緊 (DO8, 128)" if ch == "g" else "張開 (DO7, 64)"))
                show(c)
            elif ch == "s":
                show(c)
            elif ch.strip():
                print(f"  ({ch!r} 沒有功能) " + HELP)
    except KeyboardInterrupt:
        pass
    finally:
        c.close()


if __name__ == "__main__":
    main()
