# -*- coding: utf-8 -*-
"""
相機安裝方向 (正裝 / 轉 180 度裝) 的共用處理。

做法:相機若反裝,就在「取像後第一步」把影像轉 180 度,之後所有程式看到的影像都和正裝時一樣
(畫面方向、tag 像素位置、校正檔裡的 tag_pixels 都能直接沿用)。

設定來源 (優先順序高到低):
  1. 命令列 --rot180 on / off / auto
  2. camera.json 的 {"rotate_180": true/false}  (python cam_orient.py --set on/off 寫入)
  3. auto:有 calibration.json 時,比較畫面中 tag 的排列與校正時是否上下左右顛倒,自動判斷

注意
  - 像素 -> 手臂座標的轉換本來就是用「當下畫面」的 4 個 tag 算的,所以不論轉不轉都會對;
    轉 180 度主要是讓畫面方向、tag 偏移檢查 (warp_sort) 和顯示跟以前一致。
  - 有用到相機內參的地方 (深度反投影、PnP) 要用 raw_point() 把座標換回原始影像,或用 rotated_K() 換內參。
  - 反裝通常也會讓相機相對夾爪的位置改變,第一次反裝後請重做 warp_calibrate.py --pixels-only
    (只重記 home 畫面的 tag 像素,tag 手臂座標、z_pick、drop 保留)。

  python cam_orient.py --show          # 顯示目前設定
  python cam_orient.py --set on        # 記住「相機反裝」
  python cam_orient.py --set off       # 記住「相機正裝」
  python cam_orient.py --set auto      # 刪除設定,改回自動判斷
"""
import argparse
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_JSON = os.path.join(HERE, "camera.json")
MODES = ("auto", "on", "off")
AUTO_SCORE = 0.5      # 排列相似度 > +0.5 視為同方向,< -0.5 視為轉 180 度,中間無法判斷


def saved_mode():
    try:
        with open(CAMERA_JSON, encoding="utf-8") as f:
            v = json.load(f).get("rotate_180")
    except (OSError, ValueError):
        return "auto"
    return "auto" if v is None else ("on" if v else "off")


def save_mode(mode):
    if mode == "auto":
        if os.path.exists(CAMERA_JSON):
            os.remove(CAMERA_JSON)
        return
    with open(CAMERA_JSON, "w", encoding="utf-8") as f:
        json.dump({"rotate_180": mode == "on"}, f, indent=2)


def add_arg(ap):
    ap.add_argument("--rot180", choices=MODES, default=None,
                    help=f"相機轉 180 度安裝時影像要不要轉回來 (預設讀 camera.json,目前={saved_mode()})")


def rotate(img, on=True):
    return cv2.rotate(img, cv2.ROTATE_180) if on else img


def raw_point(uv, shape, on=True):
    """轉正後影像的像素 -> 原始影像像素 (反向也一樣)。shape = 影像 shape。"""
    if not on:
        return tuple(uv)
    h, w = shape[:2]
    return (w - 1 - uv[0], h - 1 - uv[1])


def rotated_K(K, shape):
    """影像轉 180 度後對應的內參 (主點移到對稱位置)。"""
    h, w = shape[:2]
    K2 = np.array(K, float).copy()
    K2[0, 2] = w - 1 - K2[0, 2]
    K2[1, 2] = h - 1 - K2[1, 2]
    return K2


def layout_score(live, ref):
    """比較兩組 tag 中心 ({id: (u,v)}) 相對於各自重心的方向:+1 同方向,-1 轉了 180 度。需要 >= 2 個共同 tag。"""
    ids = [i for i in live if str(i) in ref or i in ref]
    if len(ids) < 2:
        return None
    a = np.array([live[i] for i in ids], float)
    b = np.array([ref[str(i)] if str(i) in ref else ref[i] for i in ids], float)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / den) if den > 0 else None


def decide(mode, live_raw=None, ref=None):
    """回傳 (是否轉 180, 說明)。live_raw = 原始影像中的 tag 中心;ref = 校正檔的 tag_pixels。"""
    mode = mode or saved_mode()
    if mode in ("on", "off"):
        return mode == "on", f"相機方向: {'反裝 (影像轉 180 度)' if mode == 'on' else '正裝'} [設定]"
    if not live_raw or not ref:
        return False, "相機方向: 正裝 [auto,沒有可比較的 tag,預設不轉]"
    s = layout_score(live_raw, ref)
    if s is None:
        return False, "相機方向: 正裝 [auto,共同 tag 不足 2 個,預設不轉]"
    if s >= AUTO_SCORE:
        return False, f"相機方向: 正裝 [auto,tag 排列與校正時相同,相似度 {s:+.2f}]"
    if s <= -AUTO_SCORE:
        return True, f"相機方向: 反裝,影像轉 180 度 [auto,tag 排列與校正時顛倒,相似度 {s:+.2f}]"
    raise SystemExit(f"無法自動判斷相機方向 (tag 排列相似度 {s:+.2f})。"
                     "請加 --rot180 on (反裝) 或 --rot180 off (正裝),或用 cam_orient.py --set 記住")


def orient_frame(frame, detect_fn, mode=None, ref=None):
    """取像後呼叫:回傳 (轉正後影像, 是否轉了, 說明)。detect_fn(frame) -> {id: (u,v)} 用於 auto 判斷。"""
    mode = mode or saved_mode()
    live = detect_fn(frame) if mode == "auto" else None
    on, msg = decide(mode, live, ref)
    return rotate(frame, on), on, msg


def main():
    ap = argparse.ArgumentParser(description="設定相機安裝方向")
    ap.add_argument("--set", choices=MODES)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.set:
        save_mode(args.set)
    print("目前設定:", {"auto": "auto (依 tag 排列自動判斷)", "on": "on (相機反裝,影像轉 180 度)",
                       "off": "off (相機正裝)"}[saved_mode()])


if __name__ == "__main__":
    main()
