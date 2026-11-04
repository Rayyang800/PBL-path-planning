"""
自動分類:綠色筆蓋 -> 夾到 tag 1、2 中間;藍色筆蓋 -> 夾到 tag 3、4 中間 (用夾爪)。
座標來自 calibration.json (warp_calibrate.py 產生),換算方式同 warp_pick.py。

預設只「辨識並印出計畫」,不移動手臂 (dry-run),並存 warp_sort_last.png。加 --go 才會移動。
用法:
  python warp_sort.py                        # dry-run: 印出兩個筆蓋的位置與目標點
  python warp_sort.py --go --speed 15 --step # 第一次實際執行: 低速、每步按 Enter 確認
  python warp_sort.py --go --speed 50        # 連續執行 (開始前仍會確認一次)
  python warp_sort.py --go --only green      # 只處理綠色

動作: 物體上方 -> 張開夾爪 -> 下降 -> 夾緊 -> 抬起 -> 目標上方 -> 下降 -> 放開 -> 抬起 ... 最後回 home。
執行前:手臂必須在 home 位姿 (calibration.json),4 個 tag 與兩個筆蓋都在畫面中。
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient

import cam_orient
import warp_pick as wp
from drv_modbus import request
from drv_modbus.conn import refresh
from record_apriltags import make_detector, detect

HERE = os.path.dirname(os.path.abspath(__file__))

# 顏色 -> (HSV 範圍, 放置位置為哪兩個 tag 的中點)
JOBS = [
    # 綠色筆蓋偏青綠 (H≈83~90, S≈70),板面是低飽和灰綠 (S<35),所以 S 下限要高於板面
    ("green", ((35, 55, 40), (92, 255, 255)), (1, 2)),
    ("blue", ((95, 110, 60), (130, 255, 255)), (3, 4)),
]
MIN_AREA = 500            # 像素,小於此視為污點 (筆蓋面積約 800~1300)
MAX_AREA = 6000
DIAM_RANGE = (28, 80)     # 外接圓直徑 (像素),筆蓋約 35~50
TAG_SHIFT_WARN = 10.0     # px,tag 與校正時位置的平均偏移超過這個就警告
Z_SAFE_MIN_GAP = 50.0     # mm,安全高度至少比夾取高度高
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def workspace_mask(shape, corners, live):
    """4 個 tag 中心圍成的四邊形,扣掉各 tag 本體 (外擴 35%)。"""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.int32([live[i] for i in (1, 2, 4, 3)]), 255)
    for c in corners:
        hull = cv2.convexHull(c.astype(np.float32))
        cen = hull.mean(axis=0)
        cv2.fillConvexPoly(mask, np.int32(cen + (hull - cen) * 1.35), 0)
    return mask


CLOSE_KERNEL = {   # 筆蓋從上方看可能只剩細環或破碎的色塊 (中間有標誌),要用大一點的核補成實心
    "black": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    "green": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
}


def clean(m, name):
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, CLOSE_KERNEL.get(name, KERNEL))


def find_caps(frame, corners, live):
    """回傳 {顏色: {center, area, diam, contour}};找不到的顏色為 None。"""
    work = workspace_mask(frame.shape, corners, live)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    masks = {name: cv2.inRange(hsv, np.array(lo), np.array(hi)) for name, (lo, hi), _ in JOBS}
    if "black" in masks and "blue" in masks:   # 藍色筆蓋的暗部/陰影不要被當成黑色
        masks["black"] = cv2.bitwise_and(masks["black"], cv2.bitwise_not(cv2.dilate(masks["blue"], KERNEL, iterations=2)))

    found = {}
    for name, _, _ in JOBS:
        m = clean(cv2.bitwise_and(masks[name], work), name)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            a = cv2.contourArea(c)
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if MIN_AREA <= a <= MAX_AREA and DIAM_RANGE[0] <= 2 * r <= DIAM_RANGE[1]:
                mo = cv2.moments(c)
                cands.append({"center": (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]),
                              "area": a, "diam": 2 * r, "contour": c})
        if len(cands) > 1:
            print(f"注意: 找到 {len(cands)} 個 {name} 物體,使用面積最大的")
        found[name] = max(cands, key=lambda o: o["area"]) if cands else None
    return found


def midpoint(calib, a, b):
    pa, pb = calib["tag_robot_xyz"][str(a)][:2], calib["tag_robot_xyz"][str(b)][:2]
    return (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2


def run_job(mv, calib, z_safe, z_pick_h, z_place_h, pick_xy, place_xy):
    ori = calib["home"][3:]
    mv.go("物體上方", *pick_xy, z_safe, ori)
    mv.grip_open()
    mv.go("夾取高度", *pick_xy, z_pick_h, ori)
    mv.grip_close()
    mv.go("抬起", *pick_xy, z_safe, ori)
    mv.go("目標上方", *place_xy, z_safe, ori)
    mv.go("放置高度", *place_xy, z_place_h, ori)
    mv.grip_open()
    mv.go("抬起", *place_xy, z_safe, ori)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="實際移動手臂 (預設只印出計畫)")
    ap.add_argument("--step", action="store_true", help="每個動作前按 Enter 確認")
    ap.add_argument("--speed", type=int, default=25)
    ap.add_argument("--only", choices=[j[0] for j in JOBS], help="只處理某個顏色")
    ap.add_argument("--z-offset", type=float, default=10.0,
                    help="基準高度 = 校正的 z_pick + 這個值 (mm)。z_pick 若是吸盤碰到板面的高度,夾爪手指會貼到板子,所以基準預設抬高 10 mm")
    ap.add_argument("--pick-down", type=float, default=10.0,
                    help="夾取高度 = 基準高度 - 這個值 (mm),預設 10 (=1 cm)")
    ap.add_argument("--place-down", type=float, default=9.0,
                    help="放置高度 = 基準高度 - 這個值 (mm),預設 9 (=0.9 cm)")
    ap.add_argument("--ignore-shift", action="store_true", help="忽略 tag 位置與校正時不同的警告")
    cam_orient.add_arg(ap)
    args = ap.parse_args()
    if not 1 <= args.speed <= wp.MAX_SPEED:
        sys.exit(f"速度需在 1~{wp.MAX_SPEED}")

    calib = wp.load_calibration()
    home, drop = calib["home"], calib["drop"]
    z_base = calib["z_pick"] + args.z_offset
    z_pick_h = z_base - args.pick_down
    z_place_h = z_base - args.place_down
    z_safe = min(home[2], drop[2])
    if z_safe - z_base < Z_SAFE_MIN_GAP:
        sys.exit(f"安全高度 {z_safe:.0f} 與基準高度 {z_base:.0f} 相差不到 {Z_SAFE_MIN_GAP:.0f} mm,請重新校正")
    for label, z in (("夾取", z_pick_h), ("放置", z_place_h)):
        if z < calib["z_pick"] - 0.5:
            sys.exit(f"{label}高度 {z:.1f} 低於校正的 z_pick {calib['z_pick']:.1f} (板面高度),會壓到板子。"
                     f"請減少 --{'pick' if label == '夾取' else 'place'}-down 或確認 z_pick")

    c = ModbusTcpClient(host=wp.ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    try:
        tcp = request.Get_TCP_Pose(c)
        if tcp is None:
            sys.exit("讀不到手臂位姿")
        ok, dpos, dang = wp.check_home(calib, tcp)
        print(f"目前 TCP 與 home 差: 位置 {dpos:.1f} mm, 角度 {dang:.1f} 度")
        if not ok:
            sys.exit("手臂不在 home 位姿 (相機位置不同,座標會錯)。請先回到 home (python go_home.py)")
        if request.Get_Pose_Flag(c) != 1:
            sys.exit("手臂到位旗標不是 1 (忙碌或異常),請先在教導器清除警報後再執行")

        pipe.start(cfg)
        for _ in range(30):
            pipe.wait_for_frames(5000)
        frame = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data())
        pipe.stop()

        dicts, params = make_detector(calib.get("family", "5x5_100"))
        frame, _, msg = cam_orient.orient_frame(frame, lambda f: wp.tag_pixel_centers(f, dicts, params),
                                                args.rot180, calib.get("tag_pixels"))
        print(msg)
        corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dicts, params)
        live = {i: cc.mean(axis=0).tolist() for cc, i in zip(corners, ids)}
        if not all(i in live for i in (1, 2, 3, 4)):
            sys.exit(f"只看到 tag {sorted(live)},需要 1~4 都在畫面中")

        shifts = [np.linalg.norm(np.array(live[i]) - np.array(calib["tag_pixels"][str(i)])) for i in (1, 2, 3, 4)]
        shift_warn = float(np.mean(shifts)) > TAG_SHIFT_WARN
        if shift_warn:
            print(f"\n警告: 4 個 tag 的位置和校正時平均差 {np.mean(shifts):.0f} px (約 {np.mean(shifts) * 0.575:.0f} mm)。")
            print("  - 若是『板子/tag 被移動過』: 座標會整體偏移,請重新執行 warp_calibrate.py")
            print("  - 若只是相機位置略不同、tag 沒動: 轉換會自動補償,可以繼續")
            print("  - 若剛把相機改裝 (例如轉 180 度): 執行 warp_calibrate.py --pixels-only 更新 tag 像素即可\n")

        m = wp.make_transform(calib, live)
        caps = find_caps(frame, corners, live)

        xy = np.array([calib["tag_robot_xyz"][str(i)][:2] for i in (1, 2, 3, 4)])
        lo, hi = xy.min(axis=0) - wp.BOUND_MARGIN, xy.max(axis=0) + wp.BOUND_MARGIN
        m_inv = np.linalg.inv(m)
        view = frame.copy()
        cv2.polylines(view, [np.int32([live[i] for i in (1, 2, 4, 3)])], True, (255, 0, 0), 2)

        plan = []
        print("辨識結果:")
        for name, _, (ta, tb) in JOBS:
            if args.only and name != args.only:
                continue
            tx, ty = midpoint(calib, ta, tb)
            tpx = cv2.perspectiveTransform(np.float32([[[tx, ty]]]), m_inv)[0][0]
            cv2.drawMarker(view, (int(tpx[0]), int(tpx[1])), (0, 200, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(view, f"{name} target", (int(tpx[0]) - 40, int(tpx[1]) - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            o = caps[name]
            if o is None:
                print(f"  {name}: 找不到筆蓋,略過")
                continue
            x, y = wp.pix_to_robot(m, o["center"])
            inside = bool((lo <= [x, y]).all() and ([x, y] <= hi).all())
            print(f"  {name}: 像素=({o['center'][0]:.0f},{o['center'][1]:.0f}) 直徑={o['diam']:.0f}px "
                  f"-> 手臂 X={x:.1f} Y={y:.1f}   放到 tag{ta}/{tb} 中點 X={tx:.1f} Y={ty:.1f}"
                  f"{'' if inside else '  [超出工作區,略過]'}")
            cv2.drawContours(view, [o["contour"]], -1, (0, 255, 0) if inside else (0, 0, 255), 2)
            cv2.putText(view, name, (int(o["center"][0]) - 20, int(o["center"][1]) - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if inside:
                plan.append((name, (x, y), (tx, ty)))
        cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "warp_sort_last.png"))
        print("已存辨識結果圖: warp_sort_last.png")
        print(f"高度: 安全 {z_safe:.1f} / 夾取 {z_pick_h:.1f} / 放置 {z_place_h:.1f} mm (校正 z_pick = {calib['z_pick']:.1f})")

        if not plan:
            sys.exit("沒有可處理的筆蓋")
        if not args.go:
            print("\n(dry-run) 沒有移動手臂。確認座標與圖片合理後,加 --go 才會執行")
            return

        if shift_warn and not args.ignore_shift:
            if input("tag 位置與校正時不同。確定座標仍然正確 (tag 沒被移動) 才輸入 y 繼續: ").strip().lower() != "y":
                sys.exit("已取消")
        print(f"\n即將執行 {len(plan)} 個取放,速度 {args.speed},{'單步確認' if args.step else '連續執行'}")
        input("確認安全 (手放急停旁、路徑無障礙) 後按 Enter 開始 ")
        mv = wp.Mover(c, args.speed, step=args.step)
        for name, pick_xy, place_xy in plan:
            print(f"\n=== {name} 筆蓋: ({pick_xy[0]:.1f},{pick_xy[1]:.1f}) -> ({place_xy[0]:.1f},{place_xy[1]:.1f}) ===")
            run_job(mv, calib, z_safe, z_pick_h, z_place_h, pick_xy, place_xy)
        mv.go("home", home[0], home[1], home[2], home[3:])
        print("\n完成")
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
