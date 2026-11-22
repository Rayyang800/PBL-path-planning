-- ============================================================
-- RL 載入器 (台達 DRV，MS2)：每台控制器只要用 DRAStudio 下載一次，之後不用再改。
-- 真正的程式 (path_runner.lua) 由 Python (delta_smooth.py) 每次啟動時透過 Modbus 送進來，
-- 這裡用 load() 編譯後執行。所以改 path_runner.lua 不用開 DRAStudio，其他電腦也不用裝 DRAStudio。
--
-- 交握 (Device Buffer，16-bit word)：
--   0x1002  1001 = 載入器在等程式 (程式跑起來後由程式自己改寫成它的 VER)
--   0x1000  Python 寫 20 = 收一段、22 = 編譯並執行、23 = 清掉重來
--           載入器回 21 = 這段收到了、9 = 錯誤 (訊息在 0x1EFF/0x1F00)
--   0x1001  這段的位元組數
--   0x1020  起 這段的內容，每個 word 放 2 個位元組 (低位元組在前)
-- ============================================================

local BASE = 0x1000
local DATA = 0x1020
local MSG = 0x1F00
local LOADER_READY = 1001
local BLK = 10                  -- MultiReadModbus 一次讀幾個 word (10 已確認可用)

local function u16(v)
    if v < 0 then v = v + 65536 end
    return v
end

local function report(msg)
    msg = string.sub(tostring(msg), 1, 250)
    WriteModbus(MSG - 1, "W", #msg)
    for i = 1, #msg do
        WriteModbus(MSG + i - 1, "W", string.byte(msg, i))
    end
end

local parts = {}

local function receive()
    local nbytes = u16(ReadModbus(BASE + 1, "W"))
    local nwords = math.floor((nbytes + 1) / 2)
    local bytes = {}
    local w = 0
    while w < nwords do
        local k = math.min(BLK, nwords - w)
        local r = MultiReadModbus(DATA + w, k, "W")
        for j = 1, k do
            local v = u16(r[j])
            bytes[#bytes + 1] = string.char(v % 256)
            bytes[#bytes + 1] = string.char(math.floor(v / 256))
        end
        w = w + k
    end
    parts[#parts + 1] = string.sub(table.concat(bytes), 1, nbytes)
end

local compile = loadstring or load

WriteModbus(BASE, "W", 0)
WriteModbus(BASE + 2, "W", LOADER_READY)

while true do
    local cmd = ReadModbus(BASE, "W")
    if cmd == 20 then
        local ok, err = pcall(receive)
        if ok then
            WriteModbus(BASE, "W", 21)
        else
            report("loader receive error: " .. tostring(err))
            WriteModbus(BASE, "W", 9)
        end
    elseif cmd == 23 then
        parts = {}
        WriteModbus(BASE, "W", 0)
    elseif cmd == 22 then
        local src = table.concat(parts)
        parts = {}
        local fn, err = compile(src, "=path_runner")
        if fn == nil then
            report("compile error: " .. tostring(err))
            WriteModbus(BASE, "W", 9)
        else
            WriteModbus(BASE, "W", 0)
            local ok, rerr = pcall(fn)          -- 正常情況 path_runner 的主迴圈不會回來
            report("program ended: " .. tostring(rerr))
            WriteModbus(BASE, "W", 9)
            WriteModbus(BASE + 2, "W", LOADER_READY)
        end
    end
    DELAY(0.02)
end
