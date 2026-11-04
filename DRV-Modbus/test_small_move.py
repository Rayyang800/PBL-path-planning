"""
最小移動測試 (診斷『手臂不動』用):從目前位置只往下 10 mm,再回到原位。
每一步前都要按 Enter,速度 5,等待上限 20 秒,並每秒印出到位旗標 (1=閒置/到位, 2=忙碌或異常)。

  python test_small_move.py
結果判讀:
  - 連 10 mm 都不動  -> 手臂目前不接受 Modbus 移動 (模式/警報/伺服/使用權問題),請看教導器
  - 10 mm 能動        -> Modbus 控制正常,先前不動是『目標點不可達』
"""
import time

from pymodbus.client import ModbusTcpClient

from drv_modbus import send, request
from drv_modbus.conn import refresh

ROBOT_IP = "192.168.1.1"
SPEED = 5
DZ = -10.0   # mm


def move_and_watch(c, pose, name):
    refresh(c)
    print(f"\n送出移動: {name} -> {[round(v, 2) for v in pose]}")
    send.Go_Position(c, *pose, SPEED, send.MovP, block=False)
    for sec in range(21):
        flag = request.Get_Pose_Flag(c)
        now = request.Get_TCP_Pose(c)
        print(f"  {sec:>2}s  旗標={flag}  位置={[round(v, 2) for v in now]}")
        if flag == 1 and sec > 0:
            print("  -> 到位")
            return True
        time.sleep(1)
    print("  -> 20 秒內沒有到位")
    return False


def main():
    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        print("連線失敗")
        return
    try:
        start = request.Get_TCP_Pose(c)
        print("起點:", [round(v, 2) for v in start], " 旗標:", request.Get_Pose_Flag(c))
        down = list(start)
        down[2] += DZ
        if input(f"\n往下 {abs(DZ)} mm 並回原位,速度 {SPEED}。手放急停旁,按 Enter 開始 (q 取消): ").strip().lower() == "q":
            return
        if move_and_watch(c, down, "往下 10mm"):
            input("\n按 Enter 回到原位: ")
            move_and_watch(c, start, "回原位")
    finally:
        c.close()


if __name__ == "__main__":
    main()
