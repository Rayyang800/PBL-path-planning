# -*- coding: utf-8 -*-
"""
筆跡回饋 -> 筆尖水平偏移 (紙面座標 mm) 的學習與修正。

每次描繪後的筆跡分析 (ridge_check / circle_trace) 會量到「筆跡相對幾何目標」的偏移:
  - 脊線:只知道垂直脊線方向的偏移 (n · o = b)
  - 圓:  筆跡圓心相對目標圓心的 2D 偏移 (兩個方向都有)
把最近幾筆一起做最小平方,得到筆尖的系統偏移 o;下次描繪時路徑先減掉 o,筆跡就會落在目標上。
11-18 實測:脊線 3 次都往同一側偏 0.66~1.06 mm、罐頭圓心偏 (+0.8, +0.9) mm (扣掉深度圓心誤差後) -> 筆尖校正有約 1 mm 固定偏移。

紀錄存在 calibrations/pen_xy_offset/ (每筆:n 方向、b 量到的偏移 (已加回當時套用的修正)、來源)。
"""
import json

import numpy as np

from run_log import CalibrationStore

NAME = "pen_xy_offset"
MAX_ABS = 3.0          # 超過 3 mm 的修正不信 (可能是目標本身找錯)


def record(n_list, b_list, applied, kind, run=None, extra=None):
    """n_list: 方向 (k,2);b_list: 量到的筆跡偏移 (k,) (相對幾何目標,當時已套用 applied 修正)。"""
    applied = np.asarray(applied if applied is not None else [0.0, 0.0], float)
    n = np.asarray(n_list, float).reshape(-1, 2)
    b = np.asarray(b_list, float).reshape(-1) + n @ applied          # 換成「沒修正時」的偏移
    if np.any(np.abs(b) > MAX_ABS):
        return False
    CalibrationStore().save(NAME, {"n": n.tolist(), "b": b.tolist(), "kind": kind, **(extra or {})}, source_run=run,
                            notes="筆跡回饋分析自動記錄")
    return True


def learned(n_last=8):
    """回傳 (o (2,), 說明, 筆數)。資料不夠 (某方向沒有約束) 時該方向給 0。"""
    st = CalibrationStore()
    N, B, k = [], [], 0
    for p_ in reversed(st.history(NAME)):
        try:
            with open(p_, encoding="utf-8") as f:
                d = json.load(f)["data"]
        except (OSError, ValueError, KeyError):
            continue
        N += d["n"]
        B += d["b"]
        k += 1
        if k >= n_last:
            break
    if not N:
        return np.zeros(2), "沒有回饋紀錄,不修正", 0
    N, B = np.array(N), np.array(B)
    o, *_ = np.linalg.lstsq(N, B, rcond=None)
    sv = np.linalg.svd(N, compute_uv=False)
    if len(sv) < 2 or sv[-1] < 0.3 * sv[0]:
        # 只有一個方向的資料 (例如只有脊線):只修那個方向 (最小範數解本來就是這樣)
        note = f"最近 {k} 筆回饋 (只有一個方向有資料)"
    else:
        note = f"最近 {k} 筆回饋"
    if np.linalg.norm(o) > MAX_ABS:
        return np.zeros(2), "回饋算出的偏移太大,不修正", k
    return o, note, k
