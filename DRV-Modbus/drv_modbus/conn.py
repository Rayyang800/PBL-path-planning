"""
台達手臂的 Modbus 連線閒置約 20 秒就會被手臂端關閉。
在『等使用者按 Enter / 想一段時間』之後、讀寫手臂之前,先呼叫 refresh(c) 重新連線。
"""


def refresh(c):
    c.close()
    if not c.connect():
        raise ConnectionError("重新連線手臂失敗,請檢查網路線與 IP")
    return c
