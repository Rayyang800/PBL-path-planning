"""
台達 DRV 取放程式 (對應 Project20261007 / main.lua)

流程:
  中心 -> 抬起p1 -> 開夾爪 -> p1(夾取) -> 抬起p1
       -> 抬起p2 -> p2 -> 抬起p2
       -> 抬起p3 -> p3 -> 抬起p3
       -> 抬起p4 -> p4(放開) -> 抬起p4 -> 中心

夾爪: 0x02FE 寫 64 = 開 (DO7), 寫 128 = 夾 (DO8),見 drv_modbus/gripper.py

執行前請確認:低速、手放急停旁、工作區無障礙物。
用法:
  python pick_place_p1_p4.py            # 連續執行,速度 25 (開始前確認一次)
  python pick_place_p1_p4.py --step     # 單步確認模式 (每個動作按 Enter)
  python pick_place_p1_p4.py --speed 10 # 指定速度 (預設 25)
"""
import argparse
import sys

from pymodbus.client import ModbusTcpClient
from drv_modbus import send, request
from drv_modbus import gripper
from drv_modbus.conn import refresh

ROBOT_IP = "192.168.1.1"
DEFAULT_SPEED = 70     # main.lua 為 SpdJ(10),已調整為 25
MAX_SPEED = 100          # 防呆上限

# 起始與結束時是否回到中心點 (main.lua 沒有這一步,但中心點是已測試過的安全位置)
USE_CENTER = True

Z_LIFT = 545.995
CENTER = (430.967, 33.29, 519.334, -177.557, 1.262, 133.355)
P1 = (497.074, -107.626, 418.139, -176.697, -0.722, 130.487)
P2 = (313.6, -107.626, 418.139, -176.698, -0.722, 130.487)
P3 = (313.6, 143.18, 418.14, -176.698, -0.722, 130.487)
P4 = (498.048, 143.18, 418.14, -176.698, -0.722, 130.487)


def lift(p):
    return (p[0], p[1], Z_LIFT, p[3], p[4], p[5])


class Runner:
    def __init__(self, c, speed, step_confirm):
        self.c = c
        self.speed = speed
        self.step_confirm = step_confirm

    def _confirm(self, msg):
        if self.step_confirm:
            ans = input(f"[下一步] {msg}  (Enter 執行 / q 中止) ")
            if ans.strip().lower() == "q":
                raise KeyboardInterrupt
        else:
            print(f"[執行] {msg}")

    def move(self, name, pose):
        self._confirm(f"移動到 {name}  {pose}")
        refresh(self.c)   # 手臂會關閉閒置連線,動作前先重連
        send.Go_Position(self.c, *pose, self.speed, send.MovP)

    def grip_open(self):
        self._confirm("夾爪打開 (DO7)")
        refresh(self.c)
        gripper.Gripper_Open(self.c)

    def grip_close(self):
        self._confirm("夾爪夾緊 (DO8)")
        refresh(self.c)
        gripper.Gripper_Close(self.c)


def run(r):
    if USE_CENTER:
        r.move("中心", CENTER)

    # 取 p1
    r.grip_open()
    r.move("抬起p1", lift(P1))
    r.move("p1", P1)
    r.grip_close()
    r.move("抬起p1", lift(P1))

    # 經過 p2、p3 (夾著物件)
    for name, p in (("p2", P2), ("p3", P3)):
        r.move(f"抬起{name}", lift(p))
        r.move(name, p)
        r.move(f"抬起{name}", lift(p))

    # 放 p4
    r.move("抬起p4", lift(P4))
    r.move("p4", P4)
    r.grip_open()
    r.move("抬起p4", lift(P4))

    if USE_CENTER:
        r.move("中心", CENTER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", action="store_true", help="單步確認模式 (每個動作按 Enter)")
    ap.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    args = ap.parse_args()

    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED} 之間")

    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線失敗,請檢查網路線與 IP (192.168.1.1)")

    try:
        print("目前位置:", request.Get_TCP_Pose(c))
        print(f"速度: {args.speed}  模式: {'單步確認' if args.step else '連續'}")
        input("確認安全 (手放急停旁、無障礙物) 後按 Enter 開始 ")
        run(Runner(c, args.speed, step_confirm=args.step))
        print("完成")
    except KeyboardInterrupt:
        print("\n使用者中止 (手臂會停在目前位置,夾爪狀態不變)")
    finally:
        c.close()


if __name__ == "__main__":
    main()
