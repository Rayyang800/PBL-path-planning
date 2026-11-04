"""
用 4 個 ArUco 做透視轉換 (warp),把影像中的顏色物體換算成手臂 XY,再用夾爪取放 (0x02FE 寫 64=開 DO7, 128=夾 DO8)。
對應 warp_suction_example.py (原本用吸盤),但座標全部來自 warp_calibrate.py 產生的 calibration.json。

預設只「計算並印出座標」,不移動手臂 (dry-run)。加 --go 才會移動。
用法:
  python warp_pick.py --color red            # 只印出物體的手臂座標 (不動手臂)
  python warp_pick.py --color red --go       # 實際取放,每步需按 Enter 確認
  python warp_pick.py --color red --go --auto --speed 25   # 連續執行

執行前:手臂必須在 calibration.json 的 home 位姿 (相機在同一個位置),4 個 tag 與物體都在畫面中。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient

import cam_orient
from drv_modbus import send, request
from drv_modbus import gripper
from drv_modbus.conn import refresh
from color_detect import detect_colors, COLOR_RANGES
from record_apriltags import make_detector, detect

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, "calibration.json")
ROBOT_IP = "192.168.1.1"
MAX_SPEED = 50
HOME_POS_TOL = 5.0     # mm,目前位置與 home 的容許誤差 (xyz)
HOME_ANG_TOL = 2.0     # 度
BOUND_MARGIN = 30.0    # mm,物體可超出 4 個 tag 範圍的容許量


def load_calibration():
    if not os.path.exists(CALIB):
        sys.exit("找不到 calibration.json,請先執行 warp_calibrate.py")
    with open(CALIB, encoding="utf-8") as f:
        return json.load(f)


def is_convex_quad(pts):
    """pts 依 1,2,4,3 的順序繞一圈;回傳是否為凸四邊形 (蝴蝶結/交叉會是 False)。"""
    signs = []
    for k in range(4):
        a, b, c = pts[k], pts[(k + 1) % 4], pts[(k + 2) % 4]
        signs.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0)
    return all(signs) or not any(signs)


def make_transform(calib, live_pix):
    """用「目前偵測到的 4 個 tag 像素中心」對應「校正時記錄的 tag 手臂 XY」,求像素 -> 手臂 XY 的透視轉換。"""
    ids = [1, 2, 3, 4]
    src = np.float32([live_pix[i] for i in ids])
    dst = np.float32([calib["tag_robot_xyz"][str(i)][:2] for i in ids])
    order = [0, 1, 3, 2]   # 依 tag 1,2,4,3 繞一圈
    if not is_convex_quad(src[order]):
        sys.exit("目前畫面中 4 個 tag 的位置排列異常 (tag 被誤認或被遮住?)")
    if not is_convex_quad(dst[order]):
        sys.exit("calibration.json 中 4 個 tag 的手臂座標排成交叉形狀,代表校正時 tag 的對應順序錯了 "
                 "(例如 tag 3、4 對調)。請對照畫面中各 tag 的實際位置重新校正")
    return cv2.getPerspectiveTransform(src, dst)


def pix_to_robot(m, uv):
    p = cv2.perspectiveTransform(np.float32([[uv]]), m)
    return float(p[0][0][0]), float(p[0][0][1])


def tag_pixel_centers(frame, dicts, params):
    corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dicts, params)
    return {i: c.mean(axis=0).tolist() for c, i in zip(corners, ids)}


def objects_in_workspace(frame, live_pix, color, min_area):
    """只保留在 4 個 tag 圍成的四邊形內的色塊;回傳面積由大到小。"""
    quad = np.int32([live_pix[i] for i in (1, 2, 4, 3)])
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return detect_colors(masked, min_area, [color])


def check_home(calib, tcp):
    h = calib["home"]
    dpos = max(abs(a - b) for a, b in zip(tcp[:3], h[:3]))
    dang = max(abs(a - b) for a, b in zip(tcp[3:], h[3:]))
    return dpos <= HOME_POS_TOL and dang <= HOME_ANG_TOL, dpos, dang


class Mover:
    def __init__(self, c, speed, step):
        self.c, self.speed, self.step = c, speed, step

    def _confirm(self, msg):
        if self.step:
            if input(f"[下一步] {msg}  (Enter 執行 / q 中止) ").strip().lower() == "q":
                raise KeyboardInterrupt
        else:
            print(f"[執行] {msg}")

    def go(self, name, x, y, z, ori):
        self._confirm(f"移動到 {name} ({x:.1f}, {y:.1f}, {z:.1f})")
        refresh(self.c)   # 等待確認期間手臂可能已關閉閒置連線
        send.Go_Position(self.c, x, y, z, *ori, self.speed, send.MovP)

    def grip_open(self):
        self._confirm("夾爪打開 (DO7)")
        refresh(self.c)
        gripper.Gripper_Open(self.c)

    def grip_close(self):
        self._confirm("夾爪夾緊 (DO8)")
        refresh(self.c)
        gripper.Gripper_Close(self.c)


def pick_and_place(mv, calib, x, y):
    home, drop, z_pick = calib["home"], calib["drop"], calib["z_pick"]
    # 安全高度用『放置點高度』(校正時手臂實際到過的高度),不用 home 高度:
    # home 在手臂可達範圍的最上緣,從那裡水平移動容易超出範圍被手臂拒絕
    ori, z_safe = home[3:], min(home[2], drop[2])
    if z_safe - z_pick < 50:
        sys.exit(f"安全高度 {z_safe:.0f} 與吸取高度 {z_pick:.0f} 相差不到 50 mm,請重新校正")
    mv.go("物體上方", x, y, z_safe, ori)
    mv.grip_open()                      # 下降前先張開夾爪
    mv.go("夾取高度", x, y, z_pick, ori)
    mv.grip_close()
    mv.go("抬起", x, y, z_safe, ori)
    mv.go("放置點上方", drop[0], drop[1], drop[2], ori)
    mv.go("放置高度", drop[0], drop[1], z_pick, ori)
    mv.grip_open()
    mv.go("抬起", drop[0], drop[1], drop[2], ori)
    mv.go("home", home[0], home[1], home[2], ori)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", required=True, choices=sorted(COLOR_RANGES))
    ap.add_argument("--min-area", type=float, default=300)
    ap.add_argument("--go", action="store_true", help="實際移動手臂取放 (預設只印出座標)")
    ap.add_argument("--auto", action="store_true", help="--go 時連續執行,不逐步確認")
    ap.add_argument("--speed", type=int, default=25)
    cam_orient.add_arg(ap)
    args = ap.parse_args()
    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED}")

    calib = load_calibration()
    dicts, params = make_detector(calib.get("family", "5x5_100"))

    c = ModbusTcpClient(host=ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    try:
        tcp = request.Get_TCP_Pose(c)
        ok, dpos, dang = check_home(calib, tcp)
        print(f"目前 TCP 與 home 差: 位置 {dpos:.1f} mm, 角度 {dang:.1f} 度")
        if not ok:
            sys.exit("手臂不在 home 位姿 (相機位置不同,座標會錯)。請先用教導器回到 home 再執行")

        pipe.start(cfg)
        for _ in range(30):
            pipe.wait_for_frames(5000)
        frame = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data())
        frame, _, msg = cam_orient.orient_frame(frame, lambda f: tag_pixel_centers(f, dicts, params),
                                                args.rot180, calib.get("tag_pixels"))
        print(msg)

        live = tag_pixel_centers(frame, dicts, params)
        if not all(i in live for i in (1, 2, 3, 4)):
            sys.exit(f"只看到 tag {sorted(live)},需要 1~4 都在畫面中")
        m = make_transform(calib, live)

        objs = objects_in_workspace(frame, live, args.color, args.min_area)
        if not objs:
            sys.exit(f"工作區內沒有 {args.color} 物體")

        xy = np.array([calib["tag_robot_xyz"][str(i)][:2] for i in (1, 2, 3, 4)])
        lo, hi = xy.min(axis=0) - BOUND_MARGIN, xy.max(axis=0) + BOUND_MARGIN
        results = []
        for o in objs:
            x, y = pix_to_robot(m, o["center"])
            inside = bool((lo <= [x, y]).all() and ([x, y] <= hi).all())
            results.append((x, y, inside, o))
            print(f"{args.color} 物體 面積={o['area']:.0f}px 像素=({o['center'][0]:.0f},{o['center'][1]:.0f}) "
                  f"-> 手臂 X={x:.1f} Y={y:.1f} mm {'' if inside else ' [超出工作區,忽略]'}")

        view = frame.copy()
        for x, y, inside, o in results:
            cv2.drawContours(view, [o["contour"]], -1, (0, 255, 0) if inside else (0, 0, 255), 2)
        cv2.polylines(view, [np.int32([live[i] for i in (1, 2, 4, 3)])], True, (255, 0, 0), 2)
        cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "warp_pick_last.png"))
        print("已存辨識結果圖: warp_pick_last.png")

        targets = [r for r in results if r[2]]
        if not targets:
            sys.exit("沒有在工作區內的物體")
        if not args.go:
            print("\n(dry-run) 沒有移動手臂。確認座標合理後,加 --go 才會取放")
            return

        x, y = targets[0][0], targets[0][1]
        print(f"\n即將取放: 吸取 ({x:.1f}, {y:.1f}),速度 {args.speed}")
        input("確認安全 (手放急停旁、路徑無障礙) 後按 Enter 開始 ")
        pick_and_place(Mover(c, args.speed, step=not args.auto), calib, x, y)
        print("完成")
    except KeyboardInterrupt:
        print("\n使用者中止 (手臂停在目前位置,夾爪狀態不變)")
    except TimeoutError as e:
        print("\n" + str(e))
    finally:
        try:
            pipe.stop()
        except RuntimeError:
            pass
        c.close()


if __name__ == "__main__":
    main()
