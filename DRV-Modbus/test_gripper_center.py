"""
測試:移動到中心點,並開合夾爪 (0x02FE: 開=64 (DO7), 夾=128 (DO8))。
每一步都需要按 Enter 確認,輸入 q 即中止並關閉連線。
執行前:低速、手放急停旁、中心點周圍無障礙、夾爪前方無物件。
"""
from pymodbus.client import ModbusTcpClient
from drv_modbus import send, request
from drv_modbus import gripper
from drv_modbus.conn import refresh

ROBOT_IP = "192.168.1.1"
SPEED = 10
CENTER = (430.967, 33.29, 519.334, -177.557, 1.262, 133.355)
CYCLES = 2


def ask(c, msg):
    ans = input(f"\n[{msg}]  Enter 繼續 / q 中止 : ")
    if ans.strip().lower() == "q":
        raise KeyboardInterrupt
    refresh(c)   # 手臂會關閉閒置連線,等待後先重連


def read_do(c):
    r = c.read_holding_registers(gripper.DO_REGISTER, 1, 2)
    return None if r.isError() else r.registers[0]


def main():
    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        print("連線失敗")
        return
    try:
        print("目前位置:", request.Get_TCP_Pose(c))
        print("0x02FE 目前值:", read_do(c))

        ask(c, f"以速度 {SPEED} MovP 移動到中心點 {CENTER}")
        send.Go_Position(c, *CENTER, SPEED, send.MovP)
        print("到達後位置:", request.Get_TCP_Pose(c))

        for i in range(CYCLES):
            ask(c, f"第 {i + 1}/{CYCLES} 次:夾爪打開 (寫入 {gripper.DO_OPEN}, DO7)")
            gripper.Gripper_Open(c)
            print("0x02FE 讀回:", read_do(c))

            ask(c, f"第 {i + 1}/{CYCLES} 次:夾爪夾緊 (寫入 {gripper.DO_CLOSE}, DO8)")
            gripper.Gripper_Close(c)
            print("0x02FE 讀回:", read_do(c))

        print("\n測試完成。若開/夾方向相反,請對調 gripper.py 的 DO_OPEN 與 DO_CLOSE。")
    except KeyboardInterrupt:
        print("\n中止")
    finally:
        # 結束時不主動改變夾爪狀態,只關閉連線
        c.close()


if __name__ == "__main__":
    main()
