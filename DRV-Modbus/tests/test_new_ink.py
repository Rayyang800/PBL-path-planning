# -*- coding: utf-8 -*-
"""自動探觸的墨點偵測測試。  python tests\\test_new_ink.py
ref = 下降前 (白紙 + 紅色筆尖 + 一個舊紅點),img = 碰紙後 (多一個小紅點、亮度略變、雜訊)。"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pen_marks   # noqa: E402

RED = (40, 40, 210)


def scene(seed, gain=1.0, new_dot=None, dot_r=3):
    img = np.full((720, 1280, 3), 225, np.uint8)
    cv2.rectangle(img, (600, 0), (640, 300), (60, 60, 60), -1)        # 筆身 / 夾爪
    cv2.circle(img, (620, 305), 8, RED, -1)                          # 紅色筆尖 (兩張都有)
    cv2.circle(img, (400, 500), 4, RED, -1)                          # 舊的紅點 (兩張都有)
    if new_dot:
        cv2.circle(img, new_dot, dot_r, RED, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8).astype(np.float32) * gain
    img += np.random.RandomState(seed).normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ref = scene(0)
    none = pen_marks.new_ink_blobs(ref, scene(1, gain=0.97))
    print("沒碰到 (亮度 -3%):", none)
    assert not none
    for r in (2, 3, 5):
        hit = pen_marks.new_ink_blobs(ref, scene(2, gain=1.02, new_dot=(650, 330), dot_r=r))
        print(f"碰到 (點半徑 {r} px):", hit[:2])
        assert hit and abs(hit[0][1][0] - 650) < 2 and abs(hit[0][1][1] - 330) < 2
    dark = pen_marks.new_ink_blobs(scene(0), scene(3, new_dot=(650, 330)), ink="dark")
    print("dark 模式 (紅點不夠黑,應該抓不到或很小):", dark[:1])
    print("PASS")


if __name__ == "__main__":
    main()
