"""
唯讀監看手臂狀態 (不會移動手臂): 每秒印出到位旗標與 TCP 位姿,Ctrl+C 結束。
旗標 1 = 閒置/可接受移動指令; 2 = 忙碌或異常 (之前的指令卡住)。
你在教導器上清除警報、切換模式時,可以用它看旗標是否恢復成 1。
"""
import time

from pymodbus.client import ModbusTcpClient

from drv_modbus import request
from drv_modbus.conn import refresh

c = ModbusTcpClient(host="192.168.1.1", port=502, timeout=3)
c.connect()
last = None
try:
    while True:
        try:
            refresh(c)
            flag = request.Get_Pose_Flag(c)
            pose = request.Get_TCP_Pose(c)
            mark = "" if flag == last else "   <-- 旗標改變"
            print(f"{time.strftime('%H:%M:%S')}  旗標={flag}  位置={[round(v, 1) for v in pose] if pose else None}{mark}")
            last = flag
        except Exception as e:  # noqa: BLE001
            print("讀取失敗:", e)
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    c.close()
