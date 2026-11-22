-- ============================================================
-- 台達 DRV 連續路徑執行器 (MS2 機器人語言)
-- 不用下載這個檔：控制器上裝 loader.lua (只裝一次)，delta_smooth.py 每次啟動時會把這個檔
-- 透過 Modbus 送進去執行。改完這個檔直接跑 Python 就生效。
--
-- 為什麼：Modbus GO 指令一次只能送一個點，每點都會停。RL 的 MovL + PASS() 讓控制器
-- 把前後兩段疊在一起，整條路徑連續不停。
--
-- 交握 (Device Buffer 0x1000~0x1FFF，16-bit word；32-bit 數值低位字在前)：
--   0x1000  指令/狀態：Python 寫 1 = 執行路徑、5 = 自我測試
--                      這裡回 2 = 執行中、3 = 完成、6 = 測試完成、9 = 錯誤
--   0x1001  點數 N
--   0x1002  版本號 VER (Python 用來確認程式在跑)
--   0x1003  目前執行到第幾點
--   0x1004  [本專案修改] 1 = 取消:不再送下一段,走完已送出的段停下 (不是 Motion Stop)
--   0x100A  PASS 模式：0 = DISTANT (轉角磨圓，每段仍會加減速)、1 = DEC (速度疊合，連續不減速)
--   0x100B  DEC 模式疊合 % (1~100，100 = 前段減速和下段加速完全重疊)
--   0x100C  模板點編號 (0 = 用 P1)：新點複製它的 UF/TF/手臂姿態 (Elbow/Shoulder/Flip)
--   0x100D  暫存點起始編號 (0 = 500)：第 i 點寫到 (起始 + i)，這段點位表會被覆蓋
--   0x1010  自我測試回傳：第 1 點 X/Y/Z、目前位置 X/Y/Z (各 2 word，0.001 mm)
--   0x1020  起每點 16 word：X Y Z RX RY RZ (各 2 word，0.001 mm / 0.001 度)、
--           速度 (0.1 %)、種類 (0 = MovL、1 = MovP)、疊合 (0 = 到位才走；DISTANT 時 = 0.1 mm)、保留 (共 16 word)
--   0x1EFF  訊息長度；0x1F00 起 訊息文字 (錯誤原因)
-- ============================================================

local VER = 2
local BASE = 0x1000
local PTS = 0x1020
local REC = 16
local MSG = 0x1F00

local function u16(v)
    if v < 0 then v = v + 65536 end
    return v
end

local function s32(lo, hi)
    local v = u16(lo) + u16(hi) * 65536
    if v >= 2147483648 then v = v - 4294967296 end
    return v
end

local function w16(v)
    if v > 32767 then v = v - 65536 end
    return v
end

local function writeDW(addr, val)
    local v = math.floor(val * 1000 + 0.5)
    if v < 0 then v = v + 4294967296 end
    WriteModbus(addr, "W", w16(v % 65536))
    WriteModbus(addr + 1, "W", w16(math.floor(v / 65536)))
end

local function report(msg)
    msg = string.sub(tostring(msg), 1, 250)
    WriteModbus(MSG - 1, "W", #msg)
    for i = 1, #msg do
        WriteModbus(MSG + i - 1, "W", string.byte(msg, i))
    end
end

-- 一次讀 8 個 word (控制器一次讀太多可能會錯)
local function readRec(i)
    local a = PTS + (i - 1) * REC
    local r = MultiReadModbus(a, 8, "W")
    local q = MultiReadModbus(a + 8, 8, "W")
    return {
        x = s32(r[1], r[2]) / 1000, y = s32(r[3], r[4]) / 1000, z = s32(r[5], r[6]) / 1000,
        rx = s32(r[7], r[8]) / 1000, ry = s32(q[1], q[2]) / 1000, rz = s32(q[3], q[4]) / 1000,
        spd = u16(q[5]) / 10, kind = u16(q[6]), blend = u16(q[7]) / 10,
    }
end

-- MovL/MovP 要用「點位表編號」(自己組的 table 不行)：複製模板點，再改座標
local function setSlot(id, tmpl, p)
    CopyPoint(id, tmpl)
    WritePoint(id, "X", p.x)
    WritePoint(id, "Y", p.y)
    WritePoint(id, "Z", p.z)
    WritePoint(id, "RX", p.rx)
    WritePoint(id, "RY", p.ry)
    WritePoint(id, "RZ", p.rz)
    local function adiff(a, b) return math.abs((a - b + 180) % 360 - 180) end
    if adiff(ReadPoint(id, "RX"), p.rx) > 0.05 or adiff(ReadPoint(id, "RY"), p.ry) > 0.05
        or adiff(ReadPoint(id, "RZ"), p.rz) > 0.05 then
        error("point " .. id .. " orientation mismatch")
    end
    local bx, by, bz = ReadPoint(id, "X"), ReadPoint(id, "Y"), ReadPoint(id, "Z")
    if math.abs(bx - p.x) > 0.05 or math.abs(by - p.y) > 0.05 or math.abs(bz - p.z) > 0.05 then
        error("point " .. id .. " write mismatch: got " .. tostring(bx) .. "," .. tostring(by) .. "," .. tostring(bz))
    end
end

local STAGE = ""

local function runPath()
    local n = u16(ReadModbus(BASE + 1, "W"))
    local dec = u16(ReadModbus(BASE + 10, "W")) == 1
    local otime = u16(ReadModbus(BASE + 11, "W"))
    local tmpl = u16(ReadModbus(BASE + 12, "W"))
    local slot0 = u16(ReadModbus(BASE + 13, "W"))
    if tmpl == 0 then tmpl = 1 end
    if slot0 == 0 then slot0 = 500 end
    if otime < 1 or otime > 100 then otime = 100 end

    -- 先把所有點寫進點位表 (手臂還沒動)，動起來後迴圈只剩 MovL，下一段才來得及接上
    local recs = {}
    for i = 1, n do
        STAGE = "load " .. i
        local p = readRec(i)
        if p.spd <= 0 then p.spd = 5 end
        p.id = slot0 + i
        setSlot(p.id, tmpl, p)
        recs[i] = p
    end

    STAGE = "PassMode"
    if dec then
        PassMode("DEC", 100)
        SetOverlapTime(otime)
    else
        PassMode("DISTANT", 100)
    end

    for i = 1, n do
        local p = recs[i]
        if ReadModbus(BASE + 4, "W") == 1 then break end
        WriteModbus(BASE + 3, "W", i)
        local pass = p.blend > 0 and i < n
        if pass and not dec then
            SetOverlapDistance(math.max(1, p.blend))
        end
        STAGE = (p.kind == 1 and "MovP " or "MovL ") .. i
        if p.kind == 1 then
            if pass then MovP(p.id, SPD(p.spd) + PASS()) else MovP(p.id, SPD(p.spd)) end
        else
            if pass then MovL(p.id, SPD(p.spd) + PASS()) else MovL(p.id, SPD(p.spd)) end
        end
    end
    STAGE = "done"
end

local function selfTest()
    local p = readRec(1)
    writeDW(0x1010, p.x)
    writeDW(0x1012, p.y)
    writeDW(0x1014, p.z)
    writeDW(0x1016, RobotX())
    writeDW(0x1018, RobotY())
    writeDW(0x101A, RobotZ())
end

local function traceback(e)
    if debug ~= nil and debug.traceback ~= nil then
        return debug.traceback(tostring(e), 2)
    end
    return tostring(e)
end

local function try(fn, name)
    STAGE = ""
    local ok, err = xpcall(fn, traceback)
    if not ok then
        pcall(report, name .. " error @" .. STAGE .. ": " .. tostring(err))
    end
    return ok
end

WriteModbus(BASE, "W", 0)
WriteModbus(BASE + 3, "W", 0)
WriteModbus(BASE + 2, "W", VER)

while true do
    local cmd = ReadModbus(BASE, "W")
    if cmd == 1 then
        WriteModbus(BASE, "W", 2)
        WriteModbus(BASE, "W", try(runPath, "runPath") and 3 or 9)
    elseif cmd == 5 then
        WriteModbus(BASE, "W", try(selfTest, "selfTest") and 6 or 9)
    end
    DELAY(0.02)
end
