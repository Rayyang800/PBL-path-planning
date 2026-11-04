"""
低速回到 calibration.json 記錄的 home 位姿 (warp_pick.py 需要手臂在這個位姿)。
移動前會顯示起點與終點並要求按 Enter 確認。請手放急停旁、確認路徑上沒有障礙物。

  python go_home.py            # 速度 10
  python go_home.py --speed 20
"""
import argparse
import json
import os
import sys

from pymodbus.client import ModbusTcpClient

from drv_modbus import send, request
from drv_modbus.conn import refresh

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_IP = "192.168.1.1"
MAX_SPEED = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=int, default=40)
    args = ap.parse_args()
    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED}")

    with open(os.path.join(HERE, "calibration.json"), encoding="utf-8") as f:
        home = json.load(f)["home"]

    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    try:
        print("目前位置:", [round(v, 1) for v in request.Get_TCP_Pose(c)])
        print("home    :", [round(v, 1) for v in home])
        print(f"速度 {args.speed},MovP (關節式移動,路徑不一定是直線)")
        if input("確認路徑安全後按 Enter 移動 (q 取消): ").strip().lower() == "q":
            return
        refresh(c)   # 等待期間手臂可能已關閉閒置連線
        send.Go_Position(c, *home, args.speed, send.MovP)
        print("已到達 home:", [round(v, 1) for v in request.Get_TCP_Pose(c)])
    finally:
        c.close()


if __name__ == "__main__":
    main()
