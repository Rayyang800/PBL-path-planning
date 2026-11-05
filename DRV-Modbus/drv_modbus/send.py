import time
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from drv_modbus import request

MovP = 0
MovL = 1

class CommandIgnored(TimeoutError):
    """手臂沒有執行移動指令 (旗標沒變、位置沒動)。繼承 TimeoutError,既有腳本的 except TimeoutError 都會接到。"""


class Interrupted(Exception):
    """on_poll 要求中斷 (已送出停止指令)。"""


TRACE_PERIOD = 0.1      # 移動中記錄 TCP 軌跡的間隔 (秒)


def _log(type_, **data):
    """寫進目前進行中的 run_log 紀錄 (沒有就略過;紀錄失敗不能影響手臂控制)。"""
    try:
        import run_log
        lg = run_log.current()
        if lg is not None:
            lg.event(type_, **data)
    except Exception:        # noqa: BLE001
        pass


def Stop(c):
    """送停止指令 (0x0300 = 0)。[未驗證] 對 MovP/MovL 是否有效;呼叫端應讀回位置確認。實體急停仍是最可靠的方式。"""
    c.write_register(0x0300, 0, 2)
    _log("cmd_stop")


def _pos_dist(a, b):
    return max(abs(p - q) for p, q in zip(a[:3], b[:3]))


def Go_Position(c, x, y, z, rx, ry, rz, speed = 20, mov = 0, block = True, timeout = 120,
                start_timeout = 2.0, arrive_tol = 1.0, on_poll = None):
    """送出移動指令;block=True 時等待到位並確認位置。
    旗標在送出前後都可能是 1,所以不能只看旗標:先等「旗標變忙碌或位置開始改變」,
    start_timeout 秒內都沒有 -> 手臂沒接受指令,丟 CommandIgnored。
    on_poll(): 等待期間反覆呼叫;回傳 True 或按 Ctrl+C -> 送停止指令並丟 Interrupted。
    有進行中的 run_log 紀錄時,自動記錄 cmd_move / tcp_trace (約 10 Hz) / move_done / move_error。"""
    t0 = time.time()
    _log("cmd_move", pose=[x, y, z, rx, ry, rz], speed=speed, mov="MovL" if mov == MovL else "MovP",
         frame="robot_mm_deg", block=block)
    try:
        end = _go(c, x, y, z, rx, ry, rz, speed, mov, block, timeout, start_timeout, arrive_tol, on_poll)
        if block:
            _log("move_done", pose=end, dt=round(time.time() - t0, 3))
    except KeyboardInterrupt:
        Stop(c)
        _log("move_error", error="ctrl_c", dt=round(time.time() - t0, 3))
        raise Interrupted("ctrl_c")
    except Interrupted:
        Stop(c)
        _log("move_error", error="key", dt=round(time.time() - t0, 3))
        raise Interrupted("key")
    except Exception as e:
        _log("move_error", error=repr(e), dt=round(time.time() - t0, 3))
        raise


def _go(c, x, y, z, rx, ry, rz, speed, mov, block, timeout, start_timeout, arrive_tol, on_poll):
    target = (x, y, z, rx, ry, rz)
    start = request.Get_TCP_Pose(c) if block else None
    last_trace = [0.0]

    def trace(flag, pose=None):
        now = time.time()
        if now - last_trace[0] >= TRACE_PERIOD:
            last_trace[0] = now
            _log("tcp_trace", pose=pose if pose is not None else request.Get_TCP_Pose(c), flag=flag)
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
    builder.add_32bit_int(int(x  * 1000))
    builder.add_32bit_int(int(y  * 1000))
    builder.add_32bit_int(int(z  * 1000))
    builder.add_32bit_int(int(rx * 1000))
    builder.add_32bit_int(int(ry * 1000))
    builder.add_32bit_int(int(rz * 1000))
    payload = builder.to_registers()
    c.write_register(0x0324, speed, 2)
    c.write_registers(0x0330, payload, 2)

    if mov == MovP:
        print("MovP")
        c.write_register(0x0300, 301, 2)   
    if mov == MovL:
        print("MovL")
        c.write_register(0x0300, 302, 2)   

    print("Start moving...")
    if block == False:
        return

    t0 = time.time()
    if start is not None and _pos_dist(start, target) > arrive_tol:
        while True:   # 等手臂開始動
            f = request.Get_Pose_Flag(c)
            p = request.Get_TCP_Pose(c)
            trace(f, p)
            if f != 1 or (p is not None and _pos_dist(p, start) > 0.3):
                break
            if time.time() - t0 > start_timeout:
                raise CommandIgnored(
                    f"手臂沒有執行移動指令 ({start_timeout:.0f} 秒內旗標維持 {f}、位置沒變)。可能原因:"
                    "教導器不在外部/遠端控制模式、伺服未開、目標超出可達範圍。請看教導器畫面")
            if on_poll is not None and on_poll():
                raise Interrupted()
            time.sleep(0.02)

    pos_flag = 2
    while pos_flag != 1:
        if on_poll is not None and on_poll():
            raise Interrupted()
        pos_flag = request.Get_Pose_Flag(c)
        trace(pos_flag)
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"手臂 {timeout} 秒內沒有到位 (到位旗標={pos_flag})。"
                "請查看教導器是否有警報、目標是否超出可達範圍,清除後再試")
        time.sleep(0.02)
    end = request.Get_TCP_Pose(c)
    if end is not None and _pos_dist(end, target) > arrive_tol:
        raise CommandIgnored(
            f"旗標顯示到位,但位置 {[round(v, 1) for v in end[:3]]} 與目標 {[round(v, 1) for v in target[:3]]} "
            f"差 {_pos_dist(end, target):.1f} mm (可能被手臂拒絕或中途停止)")
    print("Move done!")
    return end

def Suction_ON(c):
    c.write_register(0x02FE, 1, 2)
    _log("cmd_suction", state="on")

def Suction_OFF(c):
    c.write_register(0x02FE, 0, 2)
    _log("cmd_suction", state="off")

def Jog_Position(c, x, y, z, rx, ry, rz):
    if x > 0:
        c.write_registers(0x0300, 601, 2)
    if x < 0:
        c.write_registers(0x0300, 602, 2)

    if y > 0:
        c.write_registers(0x0300, 603, 2)
    if y < 0:
        c.write_registers(0x0300, 604, 2)

    if z > 0:
        c.write_registers(0x0300, 605, 2)
    if z < 0:
        c.write_registers(0x0300, 606, 2)

    if rx > 0:
        c.write_registers(0x0300, 607, 2)
    if rx < 0:
        c.write_registers(0x0300, 608, 2)

    if ry > 0:
        c.write_registers(0x0300, 609, 2)
    if ry < 0:
        c.write_registers(0x0300, 610, 2)

    if rz > 0:
        c.write_registers(0x0300, 611, 2)
    if rz < 0:
        c.write_registers(0x0300, 612, 2)

def Jog_Stop(c):
    c.write_registers(0x0300, 0, 2)