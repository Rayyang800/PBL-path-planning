# -*- coding: utf-8 -*-
"""
主控台單鍵輸入 (Windows msvcrt)。

VS Code / Windows Terminal 等終端機會自己送出「跳脫序列」(Esc 後面緊接 [ 或 O,例如視窗焦點事件 Esc[I、方向鍵),
直接用 msvcrt.getwch() 會把它當成使用者按了 Esc。read_key() 會把整串吃掉並回傳 None。
"""
import time

try:
    import msvcrt
except ImportError:          # 非 Windows
    msvcrt = None


def read_key():
    """有按鍵就回傳該字元 (單獨按 Esc 回傳 '\\x1b');沒有按鍵、或是終端機跳脫序列/功能鍵則回傳 None。不會阻塞。"""
    if not (msvcrt and msvcrt.kbhit()):
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):                 # 功能鍵/方向鍵:後面還有一個碼
        if msvcrt.kbhit():
            msvcrt.getwch()
        return None
    if ch == "\x1b":
        time.sleep(0.03)                       # 跳脫序列的後續字元會緊接著到
        if msvcrt.kbhit():
            nxt = msvcrt.getwch()
            if nxt in ("[", "O"):
                t0 = time.time()
                while time.time() - t0 < 0.1:   # 吃到序列結尾 (字母或 ~)
                    if msvcrt.kbhit():
                        c = msvcrt.getwch()
                        if c.isalpha() or c == "~":
                            break
                return None
            return nxt                         # Esc 後面跟一般字元:當成那個字元
    return ch


def wait_key(poll=0.02):
    """阻塞等待一個「真的」按鍵。"""
    while True:
        k = read_key()
        if k is not None:
            return k
        time.sleep(poll)
