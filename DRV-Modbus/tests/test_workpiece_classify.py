# -*- coding: utf-8 -*-
"""工件自動判斷的決策規則 (不需要相機)。  python tests\\test_workpiece_classify.py
數值取自 2026-11-25 用 runs/ 實機照片量到的範圍 (34 次 A / B / C 全對,沒有工件的 12 張都回「無法判斷」)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workpiece_classify as wc  # noqa: E402

CASES = [
    ("A 屋脊 (典型)", {"ok": True, "h_max": 27.5, "bbox_mm": [89, 123], "circle": {"r": 36, "rms": 15.2},
                    "ridge": {"length": 104, "lateral_rms": 0.94}, "seam": {"ok": True}}, "A"),
    ("A 屋脊 (最差那次)", {"ok": True, "h_max": 26.6, "bbox_mm": [90, 124], "circle": {"r": 36, "rms": 14.8},
                      "ridge": {"length": 100, "lateral_rms": 1.23}, "seam": {"ok": True}}, "A"),
    ("B 縫合 (板子被當成物體)", {"ok": True, "h_max": 8.0, "bbox_mm": [48, 93], "ridge": {"length": 40, "lateral_rms": 15.2},
                         "seam": {"ok": True}}, "B"),
    ("B 縫合 (板子太薄,沒當成物體)", {"ok": True, "h_max": 0.0, "seam": {"ok": True}}, "B"),
    ("C 罐頭 (最差那次)", {"ok": True, "h_max": 42.5, "bbox_mm": [79, 78], "circle": {"r": 40, "rms": 1.25},
                      "ridge": {"length": 60, "lateral_rms": 8.2}, "seam": {"ok": True}}, "C"),
    ("空的校正紙", {"ok": True, "h_max": 0.0, "seam": {"ok": False}}, None),
    ("立方體 (不是 A/B/C)", {"ok": True, "h_max": 28.0, "bbox_mm": [30, 30], "seam": {"ok": False}}, None),
    ("看不到 tag", {"ok": False, "reason": "看不到校正紙 4 個 tag"}, None),
]


def main():
    for name, f, want in CASES:
        r = wc.decide(f)
        assert r["kind"] == want, (name, r)
        if want:
            assert r["task"] == wc.TASKS[want]
        print(f"OK {name} -> {r['kind'] or '無法判斷'}")
    print("PASS")


if __name__ == "__main__":
    main()
