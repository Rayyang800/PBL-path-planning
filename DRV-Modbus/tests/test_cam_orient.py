# -*- coding: utf-8 -*-
"""相機反裝 180 度測試 (不需要手臂與相機)。  python tests\\test_cam_orient.py
合成一張有 ID 1~4 (5x5_100) 與一個綠色圓點的影像當作「校正時 (正裝)」,
再把影像轉 180 度當作「相機反裝」,檢查:auto 能判斷、轉正後 tag 像素與校正一致、換算的手臂座標相同。"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cam_orient                                   # noqa: E402
import warp_pick as wp                              # noqa: E402
from record_apriltags import make_detector          # noqa: E402

DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
# 不對稱擺放 (模擬相機沒對正),位置: tag id -> 左上角像素
POS = {1: (900, 120), 2: (880, 520), 3: (180, 110), 4: (220, 540)}
ROBOT = {"1": [383.0, -215.0, 400.0], "2": [384.0, 75.0, 400.0], "3": [620.0, -212.0, 400.0], "4": [621.0, 78.0, 400.0]}


def scene():
    img = np.full((720, 1280, 3), 200, np.uint8)
    for i, (x, y) in POS.items():
        img[y - 15:y + 105, x - 15:x + 105] = 255
        img[y:y + 90, x:x + 90] = cv2.cvtColor(cv2.aruco.drawMarker(DICT, i, 90), cv2.COLOR_GRAY2BGR)
    cv2.circle(img, (610, 300), 22, (60, 170, 40), -1)
    return img


def main():
    dicts, params = make_detector("5x5_100")
    det = lambda f: wp.tag_pixel_centers(f, dicts, params)   # noqa: E731
    up = scene()
    calib = {"tag_pixels": {str(k): v for k, v in det(up).items()}, "tag_robot_xyz": ROBOT}
    assert sorted(calib["tag_pixels"]) == ["1", "2", "3", "4"]
    green = [o for o in __import__("color_detect").detect_colors(up, 300, ["green"])][0]["center"]
    xy_ref = wp.pix_to_robot(wp.make_transform(calib, det(up)), green)

    # 1. 正裝 auto -> 不轉
    _, on, msg = cam_orient.orient_frame(up, det, "auto", calib["tag_pixels"])
    print(msg); assert not on

    # 2. 反裝 auto -> 轉 180,轉正後 tag 像素與校正時一致
    flipped = cv2.rotate(up, cv2.ROTATE_180)
    fixed, on, msg = cam_orient.orient_frame(flipped, det, "auto", calib["tag_pixels"])
    print(msg); assert on
    live = det(fixed)
    d = max(np.linalg.norm(np.subtract(live[i], calib["tag_pixels"][str(i)])) for i in (1, 2, 3, 4))
    print(f"轉正後 tag 像素與校正差 最大 {d:.2f} px"); assert d < 1.0

    # 3. 換算的手臂座標:轉正後 / 不轉直接算 都要等於正裝結果
    g_fixed = __import__("color_detect").detect_colors(fixed, 300, ["green"])[0]["center"]
    xy_fixed = wp.pix_to_robot(wp.make_transform(calib, live), g_fixed)
    g_raw = __import__("color_detect").detect_colors(flipped, 300, ["green"])[0]["center"]
    xy_raw = wp.pix_to_robot(wp.make_transform(calib, det(flipped)), g_raw)
    print("綠點手臂座標  正裝 (%.2f, %.2f)  反裝轉正 (%.2f, %.2f)  反裝不轉 (%.2f, %.2f)" % (*xy_ref, *xy_fixed, *xy_raw))
    assert np.allclose(xy_ref, xy_fixed, atol=0.5) and np.allclose(xy_ref, xy_raw, atol=0.5)

    # 4. 強制設定與 raw_point / rotated_K
    assert cam_orient.decide("on")[0] and not cam_orient.decide("off")[0]
    assert cam_orient.raw_point((0, 0), up.shape) == (1279, 719)
    K2 = cam_orient.rotated_K(np.eye(3) * [[1], [1], [1]] + [[0, 0, 638.875], [0, 0, 364.43], [0, 0, 0]], up.shape)
    assert abs(K2[0, 2] - 640.125) < 1e-9 and abs(K2[1, 2] - 354.57) < 1e-9
    print("\nPASS")


if __name__ == "__main__":
    main()
