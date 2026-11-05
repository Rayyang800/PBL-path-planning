"""
warp 取放校正 (對應 warp_suction_example.py 裡手動填的 o_point / e_point / home / drop / z_height)。

**這支程式不會移動手臂**:手臂由你用教導器 / DRAStudio 移動,程式只讀取 TCP 位姿與相機畫面。
結果存到 calibration.json,供 warp_pick.py 使用;每次也會在 calibrations/warp_plane/ 留一個版本。

流程:
  1. 把手臂移到「觀察位姿 home」,讓相機完整看到 4 個 ArUco (ID 1~4) -> 記錄 home 與 4 個 tag 的像素中心
  2. 依序把吸盤尖端 (TCP) 對準 tag 1、2、3、4 的中心 -> 記錄各 tag 的手臂座標
  3. 把吸盤降到「吸取高度」(吸盤剛好接觸物體頂面) -> 記錄 z_pick
  4. 把手臂移到「放置點上方」 -> 記錄 drop
注意:相機裝在手臂上時,之後執行 warp_pick.py 必須回到同一個 home 位姿。

  python warp_calibrate.py                    # 完整 4 步驟
  python warp_calibrate.py --pixels-only      # 只重做步驟 1 (相機改裝/轉 180 度後用;tag 沒移動時)
  python warp_calibrate.py --rot180 on        # 相機反裝 (也可先 python cam_orient.py --set on 記住)
"""
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient

import cam_orient
import run_log
from rs_capture import RSCamera
from drv_modbus import request
from drv_modbus.conn import refresh
from record_apriltags import make_detector, detect
from run_log import CalibrationStore

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calibration.json")
ROBOT_IP = "192.168.1.1"
FAMILY = "5x5_100"


def read_tcp(c):
    refresh(c)   # 手臂會關閉閒置連線,讀取前先重連
    pose = request.Get_TCP_Pose(c)
    if pose is None:
        sys.exit("讀不到手臂位姿")
    pose = [float(v) for v in pose]
    run_log.current() and run_log.current().event("tcp", pose=pose)
    return pose


def centers(frame, dicts, params):
    corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dicts, params)
    return {i: c.mean(axis=0).tolist() for c, i in zip(corners, ids)}


def grab(pipe):
    return np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data())


def tag_pixels(pipe, dicts, params, rot, n=20):
    """平均 n 張 (已轉正) 影像的 tag 中心像素;回傳 {id: [u, v]}。"""
    acc = {}
    for _ in range(n):
        for i, uv in centers(cam_orient.rotate(grab(pipe), rot), dicts, params).items():
            acc.setdefault(i, []).append(uv)
    return {i: np.mean(v, axis=0).tolist() for i, v in acc.items() if len(v) >= n // 2}


def ask(msg):
    ans = input(f"\n{msg}\n  完成後按 Enter (q 放棄): ")
    if ans.strip().lower() == "q":
        sys.exit("已放棄,未儲存")


def load_old():
    if not os.path.exists(OUT):
        return None
    with open(OUT, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels-only", action="store_true",
                    help="只更新 home 與 tag 像素 (tag 手臂座標、z_pick、drop 沿用舊的 calibration.json)")
    cam_orient.add_arg(ap)
    args = ap.parse_args()
    old = load_old()
    if args.pixels_only and old is None:
        sys.exit("沒有舊的 calibration.json,不能用 --pixels-only")

    dicts, params = make_detector(FAMILY)
    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")

    cam = RSCamera()                    # 彩色 + 深度
    pipe = cam.pipe
    try:
        ask("步驟1/4: 用教導器把手臂移到『觀察位姿 home』,確認相機完整看到 ID 1~4 四個 tag")
        run_log.snapshot("home", cam.grab(n_depth=15, flush=5), cam.info())
        rot, msg = cam_orient.decide(args.rot180, centers(grab(pipe), dicts, params),
                                     old.get("tag_pixels") if old else None)
        print(msg)
        pix = tag_pixels(pipe, dicts, params, rot)
        if sorted(pix) != [1, 2, 3, 4]:
            sys.exit(f"只偵測到 ID {sorted(pix)},需要 1~4 全部可見;請調整後重跑")
        home = read_tcp(c)
        print("home TCP:", [round(v, 3) for v in home])
        print("tag 像素中心:", {k: [round(x, 1) for x in v] for k, v in pix.items()})

        if args.pixels_only:
            tags, z_pick, drop = old["tag_robot_xyz"], old["z_pick"], old["drop"]
            if old.get("tag_pixels"):
                d = np.mean([np.linalg.norm(np.subtract(pix[i], old["tag_pixels"][str(i)])) for i in (1, 2, 3, 4)])
                print(f"與舊校正的 tag 像素平均差 {d:.1f} px")
        else:
            tags = {}
            for i in (1, 2, 3, 4):
                ask(f"步驟2/4: 移動吸盤尖端,對準 tag {i} 的中心 (貼近但勿壓壞)")
                p = read_tcp(c)
                tags[str(i)] = p[:3]
                print(f"tag {i} 手臂座標:", [round(v, 3) for v in p[:3]])

            ask("步驟3/4: 把吸盤降到『吸取高度』 (吸盤剛好接觸物體頂面)")
            z_pick = read_tcp(c)[2]
            print("z_pick:", round(z_pick, 3))

            ask("步驟4/4: 把手臂移到『放置點』上方的安全高度")
            drop = read_tcp(c)
            print("drop:", [round(v, 3) for v in drop])
    finally:
        cam.close()
        c.close()

    data = {"family": FAMILY, "home": home, "tag_pixels": pix, "tag_robot_xyz": tags,
            "z_pick": z_pick, "drop": drop, "camera_rotate_180": bool(rot)}
    if old is not None:
        bak = os.path.join(HERE, time.strftime("calibration.bak_%Y%m%d-%H%M%S.json"))
        shutil.copy2(OUT, bak)
        print("舊檔備份:", os.path.basename(bak))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    ver = CalibrationStore().save("warp_plane", data,
                                  notes="pixels-only" if args.pixels_only else "full", extra={"camera_rotate_180": bool(rot)})
    print("\n已儲存:", OUT)
    print("版本紀錄:", os.path.relpath(ver, HERE))
    if run_log.current():
        run_log.current().save_json("calibration", data)


if __name__ == "__main__":
    run_log.run_main("warp_calibrate", main)
