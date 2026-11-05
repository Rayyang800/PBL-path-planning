"""
夾爪 (雙 DO),與吸盤共用同一個 DO 輸出暫存器 0x02FE (位元遮罩):
    DO1 = 1 (吸盤), ..., DO7 = 64, DO8 = 128
對應 main.lua:
    開 = DO(8,"OFF") + DO(7,"ON")  -> 寫入 64
    夾 = DO(7,"OFF") + DO(8,"ON")  -> 寫入 128

注意:這個暫存器一次決定所有 DO,寫入夾爪值時吸盤 (DO1) 會被關閉。
"""
import time

DO_REGISTER = 0x02FE
DO_OPEN = 64     # DO7 ON
DO_CLOSE = 128   # DO8 ON


def _log(state):
    try:
        import run_log
        lg = run_log.current()
        if lg is not None:
            lg.event("cmd_gripper", state=state)
    except Exception:        # noqa: BLE001 - 紀錄失敗不能影響夾爪
        pass


def Gripper_Open(c, delay=1.0):
    c.write_register(DO_REGISTER, DO_OPEN, 2)
    _log("open")
    time.sleep(delay)


def Gripper_Close(c, delay=1.0):
    c.write_register(DO_REGISTER, DO_CLOSE, 2)
    _log("close")
    time.sleep(delay)
