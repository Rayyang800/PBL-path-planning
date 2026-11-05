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
from run_log import RunLog

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
        cal = json.load(f)
    home = cal["home"]
    z_safe = min(home[2], cal["drop"][2])

    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    try:
        cur = request.Get_TCP_Pose(c)
        print("目前位置:", [round(v, 1) for v in cur])
        print("home    :", [round(v, 1) for v in home])
        low = cur[2] < z_safe - 1.0
        if low:
            print(f"目前低於安全高度 {z_safe:.1f}:會先『垂直往上』(MovL) 到安全高度,再回 home")
        print(f"速度 {args.speed},MovP (關節式移動,路徑不一定是直線)")
        if input("確認路徑安全後按 Enter 移動 (q 取消): ").strip().lower() == "q":
            return
        refresh(c)   # 等待期間手臂可能已關閉閒置連線
        with RunLog("go_home", params=vars(args), hardware={"robot_ip": ROBOT_IP}) as log:   # 指令與軌跡自動記錄
            log.add_input(os.path.join(HERE, "calibration.json"))
            log.event("tcp", pose=cur, flag=request.Get_Pose_Flag(c))
            try:
                if low:
                    send.Go_Position(c, cur[0], cur[1], z_safe, *cur[3:], min(args.speed, 20), send.MovL)
                send.Go_Position(c, *home, args.speed, send.MovP)
            except (send.CommandIgnored, TimeoutError) as e:
                log.finish("failed", {"error": str(e)})
                sys.exit("\n未到達 home: " + str(e))
            except send.Interrupted:
                log.finish("aborted", {"error": "ctrl_c"})
                sys.exit("\n已按 Ctrl+C:送出停止指令,手臂停在目前位置")
            end = request.Get_TCP_Pose(c)
            print("已到達 home:", [round(v, 1) for v in end])
            log.finish("ok", {"end_pose": end})
    finally:
        c.close()


if __name__ == "__main__":
    main()
