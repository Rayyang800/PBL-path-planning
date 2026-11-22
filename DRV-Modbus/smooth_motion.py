# -*- coding: utf-8 -*-
"""
連續路徑 (點和點之間不停):控制器上的 RL 程式用 MovL + PASS() 走整條路徑。

方法與 Lua 程式來自 DanielYut/DeltaArm_Smooth (third_party/deltaarm_smooth/SOURCE.md),這裡是配合本專案規則重寫的 Python 端。
和原版 delta_smooth.py 不同的地方:
  1. 不送 Motion Stop 1000 (團隊決定不做中斷指令)。Ctrl+C / 暫停 = 寫取消旗標,控制器不再送下一段,
     走完已送出的段 (約 1~2 段) 停下,停妥後才關 RL,之後照常抬起回 home。
  2. 控制器上有別的 RL 程式在跑時拒絕執行 (原版會直接停掉它)。
  3. 手臂停妥、RL 關掉以前不送任何 GO 指令 (RL 執行時控制器不接受 GO);確認不了就丟 SmoothStuck。
  4. 寫進點位表的點連姿態一起讀回比對 (原版只比 XYZ)。

控制器要先做一次 (本專案的程式不會自動做):裝 loader.lua 到 RL ID 900、教模板點 P1、切 Auto。
會被覆蓋:點位表 slot0+1 起 (預設 501 起,每個路徑點一格)、Device Buffer 0x1000~0x1FFF。
注意:loader 會執行任何從 Modbus 送進去的 Lua,只在實驗室內網用。

11-22:只用模擬控制器測過,還沒上過實機。SPD(%) 和 GO 指令速度 % 是否一樣快也還沒驗證。

  python smooth_motion.py diag      # 只讀:操作模式、RL 狀態、RL ID
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LUA = os.path.join(HERE, "third_party", "deltaarm_smooth", "path_runner.lua")

REG_POSE, REG_MOTION, REG_OP_MODE = 0x00F0, 0x00E0, 0x0139
REG_RL_STATUS, REG_RL_ID, REG_RL_MODE = 0x0213, 0x0220, 0x0228
BUF, PTS, MSG = 0x1000, 0x1020, 0x1F00
REC = 16
MAX_PTS = (MSG - 1 - PTS) // REC
VER, LOADER_READY = 2, 1001
ST_RUN, ST_DONE, ST_TEST_DONE, ST_ERR = 2, 3, 6, 9
MAX_SPEED = 15


class SmoothUnavailable(RuntimeError):
    """還沒開始動就發現不能用:呼叫端可以改用逐點描繪。"""


class SmoothFailed(RuntimeError):
    """路徑中途失敗,但手臂已停妥、RL 已關:可以照常抬起回 home。"""


class SmoothStuck(RuntimeError):
    """確認不了手臂停妥或 RL 關掉:不能再送 GO,要現場處理 (實體急停)。"""


def _dw(v):
    v = int(round(v * 1000)) & 0xFFFFFFFF
    return [v & 0xFFFF, v >> 16]


def _from_dw(lo, hi):
    v = (hi << 16) | lo
    return (v - (1 << 32) if v >= (1 << 31) else v) / 1000.0


class Smooth:
    def __init__(self, client, log=None, rl_id=900, template=1, slot0=500, lua=LUA):
        self.c, self.log = client, log
        self.rl_id, self.template, self.slot0, self.lua = int(rl_id), int(template), int(slot0), lua

    # ---------- Modbus ----------
    def _io(self, fn, what):
        last = None
        for _ in range(3):
            try:
                r = fn()
                if r is not None and not r.isError():
                    return r
                last = r
            except Exception as e:          # noqa: BLE001 - 連線斷掉時重連再試
                last = e
            self.c.close()
            time.sleep(0.2)
            self.c.connect()
        raise IOError(f"{what} 失敗:{last}")

    def read(self, addr, n=1):
        return list(self._io(lambda: self.c.read_holding_registers(addr, n, 2), f"讀 0x{addr:04X}").registers)

    def write(self, addr, v):
        if isinstance(v, list):
            return self._io(lambda: self.c.write_registers(addr, v, 2), f"寫 0x{addr:04X}")
        return self._io(lambda: self.c.write_register(addr, v, 2), f"寫 0x{addr:04X}")

    def pose(self):
        r = self.read(REG_POSE, 12)
        return [_from_dw(r[i], r[i + 1]) for i in range(0, 12, 2)]

    def message(self):
        n = min(self.read(MSG - 1)[0], 250)
        out = []
        for i in range(0, n, 10):
            out += self.read(MSG + i, min(10, n - i))
        return bytes(v & 0xFF for v in out).decode("utf-8", errors="replace")

    def _event(self, name, **kw):
        if self.log is not None:
            self.log.event(name, **kw)

    # ---------- 狀態 ----------
    def diag(self):
        return {"mode": self.read(REG_OP_MODE)[0], "rl_status": self.read(REG_RL_STATUS)[0],
                "rl_id": self.read(REG_RL_ID)[0], "program": self.read(BUF + 2)[0]}

    def _cmd(self, cmd, want, timeout=5.0):
        self.write(BUF, cmd)
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.read(BUF)[0]
            if st == want:
                return
            if st == ST_ERR:
                raise SmoothUnavailable("控制器回報:" + self.message())
            time.sleep(0.02)
        raise SmoothUnavailable(f"控制器沒有回應 (指令 {cmd})")

    def _upload(self):
        with open(self.lua, "rb") as f:
            lines = [ln.strip() for ln in f.read().replace(b"\r\n", b"\n").split(b"\n")]
        src = b"\n".join(ln for ln in lines if ln and not ln.startswith(b"--"))
        self.write(BUF, 23)
        time.sleep(0.1)
        size = (MSG - 1 - PTS) * 2
        for off in range(0, len(src), size):
            part = src[off:off + size]
            b = part + b"\0" if len(part) % 2 else part
            words = [b[i] | (b[i + 1] << 8) for i in range(0, len(b), 2)]
            for i in range(0, len(words), 10):
                self.write(PTS + i, words[i:i + 10])
            self.write(BUF + 1, len(part))
            self._cmd(20, 21, 10.0)
        self.write(BUF, 22)
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self.read(BUF + 2)[0] == VER:
                return
            if self.read(BUF)[0] == ST_ERR:
                raise SmoothUnavailable("Lua 沒跑起來:" + self.message())
            time.sleep(0.05)
        raise SmoothUnavailable("Lua 沒跑起來")

    def start(self):
        d = self.diag()
        if d["mode"] != 3:
            raise SmoothUnavailable("控制器不在 Auto 模式")
        if d["rl_status"] != 0 and d["rl_id"] != self.rl_id:
            raise SmoothUnavailable(f"控制器上有別的 RL 程式 (ID {d['rl_id']}) 在執行,不去動它")
        if d["rl_status"] == 1:
            if d["program"] == VER:
                return
            if d["program"] == LOADER_READY:
                return self._upload()
            raise SmoothUnavailable("RL 在跑不認得的版本,請先在 DRAStudio / 教導器按 Stop")
        self.write(BUF + 2, 0)
        self.write(BUF, 0)
        self.write(REG_RL_ID, self.rl_id)
        for seq in ((2,), (6,), (1, 2)):             # Run -> Open & Run -> Open 再 Run
            for v in seq:
                self.write(REG_RL_MODE, v)
                time.sleep(0.3)
            t0 = time.time()
            while time.time() - t0 < 3.0:
                v = self.read(BUF + 2)[0]
                if v == VER:
                    return
                if v == LOADER_READY:
                    return self._upload()
                time.sleep(0.1)
        raise SmoothUnavailable(f"RL {self.rl_id} 沒有啟動 (還沒裝 loader?)")

    def _still(self, timeout=10.0):
        t0, last, since = time.time(), self.pose(), time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.1)
            p = self.pose()
            if max(abs(a - b) for a, b in zip(p, last)) > 0.05 or self.read(REG_MOTION)[0] != 0:
                since = time.time()
            elif time.time() - since >= 0.4:
                return True
            last = p
        return False

    def release(self):
        """手臂停妥後關掉我們的 RL,GO 指令才能用。"""
        try:
            if self.read(REG_RL_STATUS)[0] == 0:
                return
            if self.read(REG_RL_ID)[0] != self.rl_id:
                return
            if not self._still():
                raise SmoothStuck("手臂還沒停妥,不關 RL")
            self.write(REG_RL_MODE, 3)
            t0 = time.time()
            while time.time() - t0 < 3.0:
                if self.read(REG_RL_STATUS)[0] != 1:
                    return
                time.sleep(0.05)
        except IOError as e:
            raise SmoothStuck(f"連線失敗,無法確認手臂狀態:{e}")
        raise SmoothStuck("RL 關不掉")

    def preflight(self):
        """不動手臂:啟動 RL、確認控制器看到的座標和 Modbus 一致。成功才回傳;結束時一律關 RL。"""
        try:
            try:
                self.start()
                p = self.pose()
                self._write_points([p], 1)
                self._cmd(5, ST_TEST_DONE)
                r = self.read(0x1010, 12)
                seen = [_from_dw(r[i], r[i + 1]) for i in range(6, 12, 2)]
                if max(abs(a - b) for a, b in zip(seen, p[:3])) > 1.0:
                    raise SmoothUnavailable("RL 和 Modbus 的座標系 (UF / TF) 不一致")
            except IOError as e:
                raise SmoothUnavailable(str(e))
        finally:
            self.release()

    # ---------- 執行 ----------
    def _write_points(self, pts, speed):
        regs = []
        for p in pts:
            for v in p:
                regs += _dw(v)
            regs += [int(round(speed * 10)), 0, 10, 0]        # 速度 0.1 %、MovL、PASS
        for i in range(0, len(regs), 10):
            self.write(PTS + i, regs[i:i + 10])
        self.write(BUF + 1, len(pts))
        self.write(BUF + 10, [1, 100, self.template, self.slot0])     # PassMode DEC、重疊 100 %

    def run(self, pts, speed, poll=None, timeout=300.0):
        """pts:不含目前位置的 [x, y, z, rx, ry, rz]。poll() 回傳 True 或 Ctrl+C = 取消。
        回傳 (原因 'done' / 'key' / 'ctrl_c', 走到第幾點)。回傳時手臂已停妥、RL 已關。"""
        pts = [[float(v) for v in p] for p in pts]
        if not pts or len(pts) > MAX_PTS:
            raise SmoothUnavailable(f"點數要在 1~{MAX_PTS}")
        if not 0 < speed <= MAX_SPEED:
            raise SmoothUnavailable(f"速度要在 1~{MAX_SPEED}")
        sent, why, idx, err, armed = False, "done", 0, None, False
        try:
            self.start()
            armed = True
            self._write_points(pts, speed)
            self.write(BUF + 3, 0)
            self.write(BUF + 4, 0)
            sent = True
            self.write(BUF, 1)
            self._event("smooth_start", n=len(pts), speed=speed)
            t0, last_trace, cancel_t = time.time(), 0.0, None
            while True:
                try:
                    st = self.read(BUF)[0]
                    if st == ST_DONE:
                        break
                    if st == ST_ERR:
                        err = self.message() or "RL 回報錯誤"
                        break
                    if self.read(REG_RL_STATUS)[0] != 1:
                        err = "RL 中途停了 (看教導器警報)"
                        break
                    now = time.time()
                    if now - last_trace >= 0.1:
                        last_trace = now
                        self._event("tcp_trace", pose=self.pose(), flag=None)
                    if cancel_t is None and ((poll is not None and poll()) or now - t0 > timeout):
                        why = "key" if now - t0 <= timeout else "timeout"
                        self.write(BUF + 4, 1)
                        cancel_t = now
                    elif cancel_t is not None and now - cancel_t > 20.0:
                        raise SmoothStuck("取消後 20 秒還沒停")
                    time.sleep(0.03)
                except KeyboardInterrupt:
                    why = "ctrl_c"
                    self.write(BUF + 4, 1)
                    cancel_t = cancel_t or time.time()
                    print("\n取消:走完已送出的這一段就停,再抬起回 home (緊急時按實體急停)")
            idx = min(len(pts), self.read(BUF + 3)[0])
            self.release()
            armed = False
        except KeyboardInterrupt:                   # 在上傳 / 寫點 / 關 RL 時按的
            why = "ctrl_c"
            try:
                if sent:
                    self.write(BUF + 4, 1)
                    t1 = time.time()
                    while self.read(BUF)[0] not in (ST_DONE, ST_ERR):
                        if time.time() - t1 > 20.0:
                            raise SmoothStuck("取消後 20 秒還沒停")
                        time.sleep(0.05)
                self.release()
            except IOError as e:
                raise SmoothStuck(str(e))
            return why, idx
        except IOError as e:
            if sent:
                raise SmoothStuck(f"執行中斷線:{e}")
            if armed:
                self.release()
            raise SmoothUnavailable(str(e))
        except SmoothUnavailable:
            self.release()
            raise
        self._event("smooth_end", reason=why, index=idx, error=err)
        if err or why == "timeout":
            raise SmoothFailed(err or "逾時")
        if why == "done":
            end = self.pose()
            if max(abs(a - b) for a, b in zip(end[:3], pts[-1][:3])) > 1.5:
                raise SmoothFailed("走完但不在終點")
        return why, idx


def main():
    if sys.argv[1:] != ["diag"]:
        sys.exit("用法:python smooth_motion.py diag  (只讀,不動手臂)")
    from pymodbus.client import ModbusTcpClient
    c = ModbusTcpClient(host="192.168.1.1", port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    d = Smooth(c).diag()
    print("操作模式", {1: "T1", 2: "T2", 3: "Auto"}.get(d["mode"], d["mode"]), "| RL 狀態",
          {0: "關閉", 1: "執行中", 2: "中斷點", 3: "暫停"}.get(d["rl_status"], d["rl_status"]), "| RL ID", d["rl_id"])
    print("載入器有沒有裝好,要第一次勾「連續運動」執行時的不動手臂自我測試才知道。")
    c.close()


if __name__ == "__main__":
    main()
