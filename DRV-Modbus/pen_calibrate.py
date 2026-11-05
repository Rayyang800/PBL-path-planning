# -*- coding: utf-8 -*-
"""
筆尖探針式校正 (人在迴路):夾爪夾白板筆,在校正紙上 5 個點各點一下 (或 --mark plus 畫「+」),回 home 拍照,
用 pen_marks.py 求「紙面 mm -> TCP 指令 XY」與紙面高度平面。不需要舊的 ArUco 校正是否準確。

預設 dry-run:只拍照、印出計畫、存預覽圖,不動手臂。加 --go 才執行。
  python pen_calibrate.py                      # dry-run
  python pen_calibrate.py --go                 # 執行 (含裝筆)
  python pen_calibrate.py --go --pen-loaded    # 筆已經夾好
  python pen_calibrate.py --go --points 3      # 只畫 3 個點 (看不出殘差,不建議)

流程 (--go)
  0. 檢查:旗標 1、手臂在 home、4 個 AprilTag 可見 -> 拍 home_before
  1. 移到紙中心上方安全高度 -> 夾爪張開 -> 人把筆放進夾爪 (筆尖朝下) -> Enter -> 倒數 3 秒 -> 夾緊
  2. 每個點:移到上方 -> 人工逐步下降 (Enter 下降一步,數字=改步距,u 上升,c 確認剛好碰到,q 中止)
     -> 下壓 press_mm 停 0.5 秒留下墨點 (plus 模式則畫手臂 X、Y 方向各一筆) -> 抬起
  3. 回 home -> 拍 home_after -> 自動分析 -> 存 calibrations/pen_paper/
安全:沒有接觸感測,碰紙完全由人判斷;有高度下限 (--z-floor);速度上限 30。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient

import pen_marks
import warp_pick as wp
from drv_modbus import send, request, gripper
from drv_modbus.conn import refresh
from paper_frame import PaperFrame
from run_log import RunLog, CalibrationStore

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = os.path.join(HERE, "print", "lab_sheet_layout.json")
MAX_SPEED = 30


class Abort(Exception):
    def __init__(self, msg="", lift=True):
        super().__init__(msg)
        self.lift = lift            # 結束前要不要垂直抬到安全高度


class Camera:
    """持續開著的 RealSense (自動探觸要反覆拍照,每次重開太慢)。"""

    def __init__(self):
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipe.start(cfg)
        for _ in range(30):
            self.pipe.wait_for_frames(5000)

    def frame(self, settle=0.4, flush=6):
        time.sleep(settle)                       # 等手臂停穩
        for _ in range(flush):                   # 丟掉緩衝區裡移動中的舊影像
            self.pipe.wait_for_frames(5000)
        return np.asanyarray(self.pipe.wait_for_frames(5000).get_color_frame().get_data()).copy()

    def close(self):
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


def small_cross(arm, x, y, z, half):
    """在高度 z 畫一個小十字 (手臂 X、Y 方向各一筆,半長 half)。筆沒碰到紙就什麼都不會留下。"""
    up = z + 2.0
    for dx, dy in ((half, 0.0), (0.0, half)):
        arm.move(x - dx, y - dy, up)
        arm.move(x - dx, y - dy, z)
        arm.move(x + dx, y + dy, z)
        arm.move(x + dx, y + dy, up)


LIFT_MARK_UV = {}        # lift 模式:第一次找到筆跡時記住它在畫面中的位置 (相機和筆固定在一起,之後都在附近)


def lift_roi(shape, border=30, radius=90):
    m = np.zeros(shape[:2], np.uint8)
    if "uv" in LIFT_MARK_UV:
        cv2.circle(m, tuple(int(v) for v in LIFT_MARK_UV["uv"]), radius, 1, -1)
    else:
        m[border:-border, border:-border] = 1          # 還不知道位置:整張畫面,但排除邊緣 (手常從邊緣伸進來)
    return m


BASE_LOOK = {}           # base 模式:移動方向 "dir" (main 設定) 與第一次找到的距離 "dist",之後每一步都相同
BASE_TAG_IDS = [2, 3]    # 要找的 5x5 ArUco (--base-tags 可改)


def base_tags_visible(frame):
    from record_apriltags import make_detector, detect
    if "det" not in BASE_LOOK:
        BASE_LOOK["det"] = make_detector("5x5_100")
    corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), *BASE_LOOK["det"])
    return [i for i in ids if i in BASE_TAG_IDS]


def base_look_pose(arm, cam, x, y, z_look, step=10.0, max_dist=None):
    """從 (x, y, z_look) 沿 BASE_LOOK["dir"] (朝 BASE_TAG_IDS 的中點) 水平移動,每 step mm 拍一張,
    看到其中一個 tag 就停。第一次搜尋後記住距離。回傳 (lx, ly)。"""
    d = np.asarray(BASE_LOOK["dir"], float)
    max_dist = max_dist or BASE_LOOK.get("max", 250.0)
    if "dist" in BASE_LOOK:
        dist = BASE_LOOK["dist"]
        lx, ly = x + d[0] * dist, y + d[1] * dist
        arm.move(lx, ly, z_look)
        return lx, ly
    print(f"    第一次:往 ArUco {BASE_TAG_IDS} 方向每 {step:g} mm 拍照,直到看到其中一個 (最多 {max_dist:g} mm)")
    dist, last_ok = 0.0, (x, y)
    while dist < max_dist:
        dist += step
        lx, ly = x + d[0] * dist, y + d[1] * dist
        try:
            arm.move(lx, ly, z_look)
        except send.CommandIgnored as e:
            print(f"    手臂到不了 ({lx:.1f}, {ly:.1f}):{e};停在上一個位置")
            dist -= step
            break
        last_ok = (lx, ly)
        seen = base_tags_visible(cam.frame(settle=0.3, flush=4))
        if seen:
            print(f"    移動 {dist:g} mm 看到 ArUco {seen},之後都在這個相對位置拍照")
            break
    else:
        print(f"    走了 {max_dist:g} mm 還沒看到 ArUco,就用這個位置拍照")
    BASE_LOOK["dist"] = dist
    return last_ok


def auto_probe(arm, cam, log, x, y, z_start, z_floor, max_steps, ink, home, z_safe, roi_mask, step=1.0,
               cross_half=4.0, look="base", look_dz=80.0, base_lift=10.0):
    """自動探觸:每次往下 step mm 畫一個小十字 -> 拍照 -> 與下降前同一位置拍的照片比對有沒有新筆跡。
    look = "base" (預設):抬 base_lift mm 後往手臂基座方向水平移動,直到看到 ArUco 1/2,在那裡拍;
           "lift":在這一點正上方 z_start + look_dz 拍;"home":垂直抬起後回 home 拍 (看得到整張紙)。
    每一步的照片都存在 images/probe_*.jpg。回傳接觸高度;max_steps 步內沒看到回傳 None (交給人工)。"""
    z_look = min(z_start + (base_lift if look == "base" else look_dz), z_safe)

    def shoot():
        if look == "home":
            arm.move(x, y, z_safe)
            arm.move(*home[:3], mov=send.MovP, ori=home[3:])
        elif look == "base":
            arm.move(x, y, z_look)                      # 先垂直抬起,再水平往基座
            base_look_pose(arm, cam, x, y, z_look)
        else:
            arm.move(x, y, z_look)
        return cam.frame()

    tag = f"probe_{len([f for f in os.listdir(os.path.join(log.path, 'images')) if f.endswith('_ref.jpg')])}"
    ref = shoot()
    log.save_image(f"{tag}_ref.jpg", ref)
    mask = roi_mask if look == "home" else None
    z = z_start
    for k in range(max_steps):
        if z < z_floor:
            print(f"    到達高度下限 {z_floor:.2f},停止自動探觸")
            return None
        if look == "home":
            arm.move(x, y, z_safe, mov=send.MovP)
        elif look == "base":
            arm.move(x, y, z_look)                      # 先水平回到點的正上方,再往下
        if cross_half > 0:
            small_cross(arm, x, y, z, cross_half)
        else:
            arm.move(x, y, z)
        img = shoot()
        log.save_image(f"{tag}_z{z:.1f}.jpg", img)
        m = mask if mask is not None else lift_roi(img.shape)
        blobs = [b for b in pen_marks.new_ink_blobs(ref, img, ink, min_area=12)
                 if m[int(round(b[1][1])), int(round(b[1][0]))]]
        log.event("auto_probe", z=z, look=look, blobs=blobs[:3])
        if blobs:
            area, uv = blobs[0]
            print(f"    z = {z:.2f}: 看到新筆跡 (面積 {area} px,影像 ({uv[0]:.0f},{uv[1]:.0f}))")
            if look != "home" and "uv" not in LIFT_MARK_UV:
                LIFT_MARK_UV["uv"] = uv
            if look == "base":
                arm.move(x, y, z_look)                  # 水平回到點的正上方
            return z
        print(f"    z = {z:.2f}: 沒有新筆跡,再往下 {step:g} mm")
        z -= step
    print(f"    {max_steps} 步內沒看到筆跡,改人工判斷。"
          + ("可能是拍照位置看不到筆尖下方,下次試 --look home (照片在 images/probe_*)" if look != "home" else ""))
    if look == "base":
        arm.move(x, y, z_look)
    return None


def grab(n_warm=20):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        for _ in range(n_warm):
            pipe.wait_for_frames(5000)
        return np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data()).copy()
    finally:
        pipe.stop()


def plan_points(pf, args):
    """回傳紙面目標點 (mm)。left:從紙中心往左 (影像左方 = 紙面 -x) 排 cols 欄、往下排 rows 列,間距 step。"""
    cx, cy = pf.W / 2 + args.shift_x, pf.H / 2 + args.shift_y
    if args.pattern == "spread":
        dx, dy = args.spread_x, args.spread_y
        pts = [(cx, cy), (cx - dx, cy - dy), (cx + dx, cy - dy), (cx + dx, cy + dy), (cx - dx, cy + dy)]
        return pts[:args.points]
    s = args.step
    if args.pattern == "cross":     # 紙中心的上、右、下、左各一點 (順時針,相鄰點移動距離短)
        return [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
    return [(cx - i * s, cy + j * s) for j in range(args.rows) for i in range(args.cols)]


def safe_region(pf, margin=12.0):
    """紙面上可以點的範圍:4 個 tag 內緣再往內縮 margin (mm)。回傳 (xmin, ymin, xmax, ymax)。"""
    xs = [c[0] for t in pf.layout["tags"] for c in t["corners_mm"]]
    ys = [c[1] for t in pf.layout["tags"] for c in t["corners_mm"]]
    tag = pf.layout["tag_mm"]
    return (min(xs) + tag + margin, min(ys) + tag + margin, max(xs) - tag - margin, max(ys) - tag - margin)


def load_previous(run_dir):
    """讀回中斷的 pen_calib:已完成的點 (contact 後有往下壓的才算) 與當時的 home_before 影像。"""
    from run_log import load_run
    meta, events = load_run(run_dir)
    pts, last_contact = [], None
    for e in events:
        if e["type"] == "prev_point":          # 這次本身也是接續來的:先前的點一起帶過去
            p = {k: e[k] for k in ("robot_xy", "contact_z", "from_run")}
            p["index"] = len(pts)
            pts.append(p)
        elif e["type"] == "contact":
            last_contact = e
        elif e["type"] == "cmd_move" and last_contact is not None:
            if e["pose"][2] < last_contact["z"] - 0.05:      # 接觸後有往下壓 (點或畫線) = 已經留下標記
                pts.append({"index": len(pts), "robot_xy": last_contact["robot_xy"], "contact_z": last_contact["z"],
                            "from_run": meta["run_id"]})
                last_contact = None
    before = cv2.imdecode(np.fromfile(os.path.join(run_dir, "images", "home_before.png"), np.uint8), cv2.IMREAD_COLOR)
    return meta, pts, before


try:
    import msvcrt
except ImportError:          # 非 Windows
    msvcrt = None

PAUSE_KEYS = (" ", "\x1b", "p", "P")


class Paused(Exception):
    """使用者在移動中按了暫停鍵 (或 Ctrl+C);已送出停止指令。"""


def drain_keys():
    while msvcrt and msvcrt.kbhit():
        msvcrt.getwch()


def ask(prompt):
    drain_keys()            # 移動中按的鍵不要被當成下一個問題的答案
    return input(prompt)


class Arm:
    def __init__(self, c, log, ori, speed):
        self.c, self.log, self.ori, self.speed = c, log, list(ori), speed

    def _poll(self):
        """移動中每 20 ms 呼叫一次:暫停鍵 -> True;o / g -> 立刻開 / 夾夾爪 (手臂繼續動)。"""
        while msvcrt and msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in PAUSE_KEYS:
                return True
            if ch in ("o", "O", "g", "G"):
                self.grip(close=ch in ("g", "G"), delay=0.0)
                print(f"\n[夾爪] {'夾緊' if ch in ('g', 'G') else '張開'}")
        return False

    def stop_and_check(self, why):
        """送停止指令,再讀兩次位置確認手臂真的停了。"""
        try:
            send.Stop(self.c)
        except Exception:     # noqa: BLE001 - 連線斷了也要繼續往下提示
            refresh(self.c)
            send.Stop(self.c)
        p1 = request.Get_TCP_Pose(self.c)
        time.sleep(0.6)
        p2 = request.Get_TCP_Pose(self.c)
        moving = p1 is not None and p2 is not None and send._pos_dist(p1, p2) > 0.5
        self.log.event("stop", reason=why, pose=p2, still_moving=moving)
        if moving:
            print("\n!!! 停止指令後手臂仍在移動,請立刻按實體急停 !!!")
        else:
            print(f"\n[已停止] 位置 {[round(v, 1) for v in p2[:3]] if p2 else '?'}")
        return p2

    def move(self, x, y, z, mov=send.MovL, speed=None, ori=None):
        ori = ori or self.ori
        refresh(self.c)
        self.log.event("cmd_move", pose=[x, y, z, *ori], speed=speed or self.speed,
                       mov="MovL" if mov == send.MovL else "MovP", frame="robot_mm_deg")
        try:
            send.Go_Position(self.c, x, y, z, *ori, speed or self.speed, mov, on_poll=self._poll)
        except send.Interrupted as e:
            self.stop_and_check(str(e))
            raise Paused(str(e))          # "ctrl_c" 或 "key"
        if abs(z - getattr(self, "_last_z", z)) < 1.0:
            time.sleep(0.3)       # 很小的移動:旗標可能來不及變化,多等一下
        self._last_z = z
        tcp = request.Get_TCP_Pose(self.c)
        self.log.event("tcp", pose=tcp, flag=request.Get_Pose_Flag(self.c))
        return tcp

    def lift_to(self, z_safe):
        """在目前 XY 垂直抬到 z_safe (低於才動);避免筆在紙上斜著拖過去。"""
        refresh(self.c)
        p = request.Get_TCP_Pose(self.c)
        if p is not None and p[2] < z_safe - 0.5:
            self.move(p[0], p[1], z_safe, ori=list(p[3:]))

    def grip(self, close, delay=1.0):
        if delay:                 # 移動中 (delay=0) 不重連,避免打斷正在進行的輪詢
            refresh(self.c)
        (gripper.Gripper_Close if close else gripper.Gripper_Open)(self.c, delay=delay)
        self.log.event("cmd_gripper", state="close" if close else "open")


HELP = ("  Enter=往下一步  數字(如 2 / 0.5)=改步距並往下  u=往上一步  c=筆尖剛好碰到紙  a=從這裡開始自動探觸\n"
        "  o=夾爪張開  g=夾爪夾緊  q=中止")


class StartAuto(Exception):
    def __init__(self, z):
        self.z = z


def probe(arm, x, y, z_start, step, z_floor, z_safe):
    """人工逐步下降,回傳接觸高度。"""
    z = z_start
    print(HELP)
    while True:
        s = ask(f"    z = {z:.2f} mm (步距 {step:g}) > ").strip().lower()
        if s == "c":
            return z
        if s == "a":
            raise StartAuto(z)
        if s in ("o", "g"):
            arm.grip(close=s == "g", delay=0.5)
            print(f"    [夾爪] {'夾緊' if s == 'g' else '張開'}")
            continue
        if s == "q":
            raise Abort("使用者中止")
        if s == "u":
            nz = min(z + step, z_safe)
        else:
            if s:
                try:
                    step = float(s)
                except ValueError:
                    print(HELP)
                    continue
                if not 0.1 <= step <= 30:
                    print("    步距需在 0.1~30 mm")
                    continue
            nz = z - step
        if nz < z_floor:
            print(f"    拒絕:{nz:.2f} 低於高度下限 {z_floor:.2f} (可用 --z-floor 調整)")
            continue
        arm.move(x, y, nz)
        z = nz


def tap_dot(arm, x, y, z_c, press, lift, dwell=0.5):
    """從接觸高度再下壓 press,停 dwell 秒留下墨點,再抬起。"""
    arm.move(x, y, z_c - press)
    time.sleep(dwell)
    arm.move(x, y, z_c + lift)


def draw_plus(arm, x, y, z_c, press, half, lift, x_dir, y_dir):
    zd, zu = z_c - press, z_c + lift
    for d in (x_dir, y_dir):
        a, b = (x - d[0] * half, y - d[1] * half), (x + d[0] * half, y + d[1] * half)
        arm.move(*a, zu)
        arm.move(*a, zd)
        arm.move(*b, zd)
        arm.move(*b, zu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pen-loaded", action="store_true", help="筆已夾好,略過裝筆")
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--pattern", choices=["cross", "left", "spread"], default="cross",
                    help="cross (預設):紙中心上下左右各偏 step 共 4 點;left:往左往下排格子;spread:中心 + 四角 5 點")
    ap.add_argument("--step", type=float, default=20.0, help="cross / left 模式的偏移量 (紙面 mm)")
    ap.add_argument("--cols", type=int, default=3, help="left 模式往左幾欄 (含中心)")
    ap.add_argument("--rows", type=int, default=2, help="left 模式往下幾列 (至少 2,否則點在一直線上算不出轉換)")
    ap.add_argument("--shift-x", type=float, default=0.0, help="整組點的起點相對紙中心平移 (紙面 mm,+ = 影像右方)")
    ap.add_argument("--shift-y", type=float, default=0.0, help="整組點的起點相對紙中心平移 (紙面 mm,+ = 影像下方)")
    ap.add_argument("--points", type=int, default=5, choices=[3, 4, 5], help="spread 模式的點數")
    ap.add_argument("--spread-x", type=float, default=75.0, help="spread 模式:點距紙中心的 x 距離 (紙面 mm)")
    ap.add_argument("--spread-y", type=float, default=45.0, help="spread 模式:點距紙中心的 y 距離 (紙面 mm)")
    ap.add_argument("--resume", help="接續中斷的 runs\\...pen_calib 資料夾:沿用已點好的點,並用它修正後續點位")
    ap.add_argument("--probe", choices=["auto", "manual"], default="auto",
                    help="auto (預設):每次下 1 mm,用相機看有沒有新墨點判斷接觸;manual:人按 c 判斷")
    ap.add_argument("--ink", choices=["any", "red", "dark"], default="any",
                    help="auto 探觸怎麼認筆跡:any (預設,任何顏色只要變暗) / red / dark")
    ap.add_argument("--probe-cross", type=float, default=4.0,
                    help="auto 探觸每一步畫的小十字半長 (mm);0 = 只往下點一下")
    ap.add_argument("--look", choices=["base", "lift", "home"], default="base",
                    help="auto 探觸拍照位置:base (預設) = 抬 --base-lift 後往基座水平移動到看見 ArUco 1/2;"
                         "lift = 原地抬高 --look-dz;home = 回 home (慢但看得到整張紙)")
    ap.add_argument("--base-lift", type=float, default=10.0, help="base 模式水平移動前先抬多少 (mm)")
    ap.add_argument("--base-dir", choices=["down", "tags"], default="down",
                    help="base 模式移動方向:down (預設) = 畫面往下 (朝手臂基座);tags = 朝 --base-tags 的中點")
    ap.add_argument("--base-tags", type=int, nargs="+", default=[1, 2],
                    help="base 模式:看到這些 ArUco 其中一個就停 (預設 1 2 = 畫面下方靠基座的兩個)")
    ap.add_argument("--base-max", type=float, default=250.0, help="base 模式最多水平移動多少 (mm)")
    ap.add_argument("--look-dz", type=float, default=80.0, help="lift 模式抬高多少拍照 (mm)")
    ap.add_argument("--probe-step", type=float, default=3.0, help="auto 探觸每次往下多少 (mm)")
    ap.add_argument("--auto-max", type=int, default=8, help="auto:最多自動下降幾步 (每步都回 home 拍照),超過改人工")
    ap.add_argument("--auto-start", type=float, default=3.0, help="auto:從上一點接觸高度上方幾 mm 開始")
    ap.add_argument("--mark", choices=["dot", "plus"], default="plus", help="plus=畫十字 (預設);dot=只點一下")
    ap.add_argument("--stroke", type=float, default=10.0, help="「+」每筆的半長 (mm,只用於 --mark plus)")
    ap.add_argument("--press", type=float, default=None,
                    help="接觸高度再往下壓多少畫線 (mm);預設 auto 0.5 (偵測到墨點時已經碰到)、manual 1.0")
    ap.add_argument("--lift", type=float, default=5.0, help="畫線之間抬起高度 (mm,相對接觸高度)")
    ap.add_argument("--z-floor", type=float, default=None, help="高度下限 (預設 = 舊校正 z_pick - 2)")
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if not 1 <= args.speed <= MAX_SPEED:
        sys.exit(f"速度需在 1~{MAX_SPEED}")
    if args.press is None:
        # auto:偵測到筆跡的高度已經壓在紙上 (最多 probe_step mm),大十字就畫在同一高度,不再多壓
        args.press = 0.0 if args.probe == "auto" else 1.0

    calib = wp.load_calibration()          # 只用 home、安全高度、粗略像素->手臂
    home, ori = calib["home"], calib["home"][3:]
    z_safe = min(home[2], calib["drop"][2])
    z_floor = args.z_floor if args.z_floor is not None else calib["z_pick"] - 2.0
    H_rough, _ = pen_marks.rough_homography(calib)

    c = ModbusTcpClient(host=wp.ROBOT_IP, port=502, timeout=3)
    if not c.connect():
        sys.exit("連線手臂失敗")
    tcp = request.Get_TCP_Pose(c)
    ok, dpos, dang = wp.check_home(calib, tcp)
    flag = request.Get_Pose_Flag(c)
    print(f"TCP 與 home 差: 位置 {dpos:.1f} mm, 角度 {dang:.1f} 度;旗標 {flag}")
    if not ok:
        sys.exit("手臂不在 home (python go_home.py)")
    if flag != 1:
        sys.exit("旗標不是 1,請先在教導器清除警報")

    frame0 = grab()
    pf = PaperFrame(layout=args.layout)
    if not pf.detect(frame0):
        sys.exit("校正紙: " + pf.describe())
    print("校正紙:", pf.describe())

    # ---- 接續:讀回已點好的點,在目前照片中找到它們,求「指令 -> 實際落點」的偏移 ----
    prev_pts, before_img, offset, prev_meta = [], frame0, np.zeros(2), None
    prev_paper = []
    if args.resume:
        prev_meta, prev_pts, before_img = load_previous(args.resume)
        if not prev_pts:
            sys.exit(f"{args.resume} 裡沒有已完成的點")
        res0 = pen_marks.analyze(frame0, PaperFrame(layout=args.layout), prev_pts, H_rough, mark="dot",
                                 before=before_img)
        offs = []
        for d, p in zip(res0["detections"], prev_pts):
            if not d["ok"]:
                sys.exit(f"照片裡找不到先前點 #{d['index']} 的墨點 ({d['reason']});紙是否被移動?")
            seen = pen_marks.apply_h(H_rough, [d["uv"]])[0]          # 墨點在粗略手臂座標的位置
            offs.append(np.array(p["robot_xy"]) - seen)
            prev_paper.append(d["paper_mm"])
            print(f"先前點 #{d['index']}: 指令 ({p['robot_xy'][0]:.1f},{p['robot_xy'][1]:.1f}) z={p['contact_z']:.2f} "
                  f"-> 墨點在紙面 ({d['paper_mm'][0]:.1f},{d['paper_mm'][1]:.1f}) mm")
        offset = np.mean(offs, axis=0)
        print(f"指令與實際落點的偏移 (粗略換算誤差 + 筆尖偏移): ({offset[0]:+.1f}, {offset[1]:+.1f}) mm,後續點位已修正")
        pf.detect(frame0)

    paper_pts = plan_points(pf, args)
    # 跳過和已點過的點太近的目標
    paper_pts = [p for p in paper_pts if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 8 for q in prev_paper)]
    if args.pattern == "left" and args.rows < 2 and len(prev_pts) + len(paper_pts) >= 3:
        print("注意: 只有 1 列,所有點在一直線上,算不出垂直方向的轉換")
    pix = pf.paper_to_pix(paper_pts)
    robot = pen_marks.apply_h(H_rough, pix) + offset
    xmin, ymin, xmax, ymax = safe_region(pf)
    # 手臂 X、Y 方向單位向量 (畫「+」用)
    x_dir, y_dir = (1.0, 0.0), (0.0, 1.0)

    view = frame0.copy()
    sr = pf.paper_to_pix([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    cv2.polylines(view, [np.int32(sr)], True, (0, 200, 0), 2)
    roi_mask = np.zeros(frame0.shape[:2], np.uint8)          # 自動探觸只在這個範圍內找新墨點
    cv2.fillConvexPoly(roi_mask, np.int32(sr), 1)
    print(f"\n可點範圍 (紙面 mm): x {xmin:.0f}~{xmax:.0f}, y {ymin:.0f}~{ymax:.0f} (綠框)")
    print(f"計畫 ({len(paper_pts)} 個新點 + {len(prev_pts)} 個先前點,速度 {args.speed},安全高度 {z_safe:.1f},"
          f"高度下限 {z_floor:.1f},下壓 {args.press} mm):")
    bad = False
    for k, (p, u, r) in enumerate(zip(paper_pts, pix, robot)):
        inside = xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
        bad |= not inside
        print(f"  #{k} 紙面 ({p[0]:.0f},{p[1]:.0f}) -> 像素 ({u[0]:.0f},{u[1]:.0f}) -> 手臂 X={r[0]:.1f} Y={r[1]:.1f}"
              + ("" if inside else "  [超出可點範圍!]"))
        col = (0, 0, 255) if inside else (0, 140, 255)
        cv2.drawMarker(view, tuple(int(v) for v in u), col, cv2.MARKER_CROSS, 30, 2)
        cv2.putText(view, f"#{k}", (int(u[0]) + 8, int(u[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    for q in prev_paper:
        u = pf.paper_to_pix([q])[0]
        cv2.circle(view, tuple(int(v) for v in u), 10, (255, 0, 0), 2)
    if args.probe == "auto" and args.look == "base":
        BASE_TAG_IDS[:] = args.base_tags
        center = np.mean(robot, axis=0)
        if args.base_dir == "down":                     # 畫面往下 100 px 對應的手臂方向
            cpix = np.mean(pix, axis=0)
            dvec = pen_marks.apply_h(H_rough, [cpix + [0, 100]])[0] - pen_marks.apply_h(H_rough, [cpix])[0]
            target = center + dvec
        else:
            target = np.mean([calib["tag_robot_xyz"][str(i)][:2] for i in args.base_tags], axis=0)
            dvec = target - center
        if np.linalg.norm(dvec) < 20:
            sys.exit(f"ArUco {args.base_tags} 的中點離點位太近 ({np.linalg.norm(dvec):.0f} mm),方向不明確;請改 --base-tags")
        BASE_LOOK["dir"] = (dvec / np.linalg.norm(dvec)).tolist()
        BASE_LOOK["max"] = args.base_max
        Hinv = np.linalg.inv(H_rough)
        a0, a1 = pen_marks.apply_h(Hinv, [center - offset, center - offset + dvec / np.linalg.norm(dvec) * 80])
        cv2.arrowedLine(view, tuple(int(v) for v in a0), tuple(int(v) for v in a1), (255, 0, 255), 3, tipLength=0.25)
        cv2.putText(view, f"look until ArUco {args.base_tags}", (int(a1[0]) + 8, int(a1[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        print(f"拍照方向 (紫色箭頭): {'畫面往下' if args.base_dir == 'down' else f'往 ArUco {args.base_tags} 中點'},"
              f"看到 ArUco {args.base_tags} 其中一個就停;"
              f"手臂方向 ({BASE_LOOK['dir'][0]:+.2f},{BASE_LOOK['dir'][1]:+.2f}),先抬 {args.base_lift:g} mm,最多 {args.base_max:g} mm")
    if bad:
        cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "pen_calibrate_plan.png"))
        sys.exit("有點超出紙上可點範圍 (見 pen_calibrate_plan.png);請減少 --step / --cols / --rows")
    cv2.imencode(".png", view)[1].tofile(os.path.join(HERE, "pen_calibrate_plan.png"))
    print("預覽圖: pen_calibrate_plan.png")
    if not args.go:
        print("\n(dry-run) 沒有移動手臂。確認點位在紙上、遠離 tag 後,加 --go 執行")
        c.close()
        return

    params = vars(args).copy()
    params.update(z_safe=z_safe, z_floor=z_floor)
    with RunLog("pen_calib", label=args.label, params=params,
                hardware={"robot_ip": wp.ROBOT_IP, "camera": "RealSense D435 1280x720"}) as log:
        log.add_input(args.layout)
        log.add_input(wp.CALIB)
        log.save_image("home_before", before_img)          # 接續時 = 最早那次 (還沒有任何墨點) 的照片
        if args.resume:
            log.save_image("home_resume", frame0)
            log.event("note", msg="resume", from_run=prev_meta["run_id"], offset=offset)
            for p in prev_pts:
                log.event("prev_point", **p)
        log.event("tcp", pose=tcp, flag=flag)
        arm = Arm(c, log, ori, args.speed)
        points = list(prev_pts)
        n0 = len(prev_pts)
        cam = Camera() if args.probe == "auto" else None
        try:
            cx, cy = robot[0]
            ask("\n確認安全 (手放急停旁) 後按 Enter:移到紙中心上方安全高度 ")
            arm.move(cx, cy, z_safe, mov=send.MovP)
            if not args.pen_loaded:
                arm.grip(close=False)
                ask("把白板筆放進夾爪 (筆尖朝下、筆身垂直、筆尖露出夾爪下方),扶好後按 Enter ")
                for s in (3, 2, 1):
                    print(f"  {s} ... 手離開夾爪")
                    time.sleep(1)
                arm.grip(close=True)
                ask("確認筆已夾穩、筆尖朝下後按 Enter 開始 ")
            z_c_prev = prev_pts[-1]["contact_z"] if prev_pts else None
            print("\n[Ctrl+C = 停止並直接回 home 結束 (可 --resume 接續)]"
                  "\n[空白鍵 / Esc / p = 停止並選擇:重做、改人工、跳過、回 home、結束]"
                  "\n[o / g = 隨時張開 / 夾緊夾爪 (移動中按,手臂不會停)]")
            force_manual = set()
            j = 0
            while j < len(robot):
                x, y, k = float(robot[j][0]), float(robot[j][1]), n0 + j
                try:
                    z_c = run_point(arm, cam, log, args, j, k, x, y, z_c_prev, z_safe, z_floor, home,
                                    roi_mask, pix, pf, x_dir, y_dir, manual=j in force_manual)
                except (Paused, KeyboardInterrupt) as e:
                    ctrl_c = isinstance(e, KeyboardInterrupt) or str(e) == "ctrl_c"
                    if isinstance(e, KeyboardInterrupt):
                        arm.stop_and_check("ctrl_c")
                    if ctrl_c:
                        print("Ctrl+C:垂直抬到安全高度後直接回 home (再按一次 Ctrl+C 會原地停止)")
                        ch = "h"
                    else:
                        ch = pause_menu(arm)
                    if ch == "r":
                        continue
                    if ch == "m":
                        force_manual.add(j)
                        continue
                    if ch == "s":
                        log.event("skip", index=k)
                        j += 1
                        continue
                    arm.lift_to(z_safe)
                    if ch == "h":
                        arm.move(*home[:3], mov=send.MovP, ori=home[3:])
                    raise Abort("使用者暫停後結束", lift=False)
                points.append({"index": k, "paper_target_mm": list(paper_pts[j]), "robot_xy": [x, y], "contact_z": z_c})
                z_c_prev = z_c
                j += 1
            print("\n回 home 拍照")
            arm.move(*home[:3], mov=send.MovP, ori=home[3:])
        except Abort as e:
            print("\n結束:", e)
            if e.lift:
                arm.lift_to(z_safe)
            log.finish("aborted", {"reason": str(e), "points_done": len(points)})
            print(f"已完成 {len(points)} 點;之後回 home 用 --resume {log.path} 接續")
            if cam:
                cam.close()
            c.close()
            return
        except (Paused, KeyboardInterrupt) as e:      # 在裝筆、回 home 等步驟中按暫停,或回 home 途中又按 Ctrl+C
            if isinstance(e, KeyboardInterrupt):
                arm.stop_and_check("ctrl_c")
            print("已停止,程式結束 (手臂停在目前位置;回 home 請用 python go_home.py,會先垂直抬起)")
            log.finish("aborted", {"reason": "paused outside point loop", "points_done": len(points)})
            if cam:
                cam.close()
            c.close()
            return
        except BaseException:
            if cam:
                cam.close()
            raise
        _finish(args, log, cam, points, H_rough, offset, home, ori, before_img)
    c.close()


def pause_menu(arm):
    print("\n=== 已暫停 ===\n"
          "  r = 重做這一點 (先垂直抬到安全高度)\n"
          "  m = 這一點改人工控制下降\n"
          "  s = 跳過這一點\n"
          "  h = 結束並回 home (已完成的點保留,可 --resume 接續)\n"
          "  q = 結束,停在安全高度\n"
          "  o / g = 夾爪張開 / 夾緊 (選完繼續留在這個選單)")
    while True:
        ch = ask("選擇 > ").strip().lower()
        if ch in ("o", "g"):
            arm.grip(close=ch == "g", delay=0.5)
            print(f"  [夾爪] {'夾緊' if ch == 'g' else '張開'}")
            continue
        if ch in ("r", "m", "s", "h", "q"):
            return ch


def run_point(arm, cam, log, args, j, k, x, y, z_c_prev, z_safe, z_floor, home, roi_mask, pix, pf,
              x_dir, y_dir, manual=False):
    """完成一個點:找接觸高度 -> 畫標記 -> 抬到安全高度。回傳接觸高度。"""
    print(f"\n=== 點 #{k}  手臂 X={x:.1f} Y={y:.1f}{'  (人工)' if manual else ''} ===")
    arm.lift_to(z_safe)
    arm.move(x, y, z_safe, mov=send.MovP)
    z_c, auto_from = None, None
    if manual:
        z0 = z_c_prev + 3.0 if z_c_prev is not None else z_safe
        if z0 < z_safe:
            arm.move(x, y, z0)
        print("  人工下降:碰到紙按 c (a = 從目前高度改自動探觸)")
        try:
            z_c = probe(arm, x, y, z0, 0.5 if z_c_prev is not None else 20.0, z_floor, z_safe)
        except StartAuto as e:
            auto_from = e.z
    elif args.probe == "auto" and z_c_prev is not None:
        auto_from = z_c_prev + args.auto_start
        print(f"  自動探觸:從 z = {auto_from:.2f} 開始每次下 {args.probe_step:g} mm 畫小十字,畫完拍照 ({args.look})")
    elif z_c_prev is None:
        print("  第一點:人工從安全高度往下 (建議 20 -> 5 -> 2)。"
              + ("筆尖離紙約 5 mm 時按 a 開始自動探觸" if args.probe == "auto" else "碰到紙按 c"))
        try:
            z_c = probe(arm, x, y, z_safe, 20.0, z_floor, z_safe)
        except StartAuto as e:
            auto_from = e.z
    else:
        arm.move(x, y, z_c_prev + 3.0)
        print("  已降到上一點接觸高度 +3 mm")
        z_c = probe(arm, x, y, z_c_prev + 3.0, 0.5, z_floor, z_safe)

    if auto_from is not None:
        if cam is None:
            raise Abort("沒有開相機,不能自動探觸 (--probe manual 時請按 c)")
        local = np.zeros_like(roi_mask)              # 只看預定點附近 35 mm (排除光線變化等雜訊)
        cv2.circle(local, tuple(int(v) for v in pix[j]), int(35.0 / pf.mm_per_pixel()), 1, -1)
        z_c = auto_probe(arm, cam, log, x, y, auto_from, z_floor, args.auto_max, args.ink,
                         home, z_safe, roi_mask & local, cross_half=args.probe_cross,
                         look=args.look, look_dz=args.look_dz, base_lift=args.base_lift, step=args.probe_step)
        if args.look == "home":
            arm.move(x, y, z_safe, mov=send.MovP)    # 目前在 home,回到點的上方
        if z_c is None:
            last = auto_from - (args.auto_max - 1) * args.probe_step
            arm.move(x, y, last + 2.0)
            print("  請人工判斷:從最後探觸高度 +2 mm 往下,碰到按 c")
            try:
                z_c = probe(arm, x, y, last + 2.0, 0.5, z_floor, z_safe)
            except StartAuto:
                raise Abort("自動探觸失敗後又選擇自動")
        else:
            arm.move(x, y, z_c + args.lift)

    if z_c - args.press < z_floor:
        raise Abort(f"畫線高度 {z_c - args.press:.2f} 低於下限 {z_floor:.2f}")
    log.event("contact", index=k, robot_xy=[x, y], z=z_c)
    if args.mark == "dot":
        print(f"  接觸高度 {z_c:.2f},點一下")
        tap_dot(arm, x, y, z_c, args.press, args.lift)
    else:
        print(f"  接觸高度 {z_c:.2f},畫「+」")
        draw_plus(arm, x, y, z_c, args.press, args.stroke, args.lift, x_dir, y_dir)
    arm.move(x, y, z_safe)
    return z_c


def _finish(args, log, cam, points, H_rough, offset, home, ori, before_img):
    """所有點完成、手臂在 home:拍照、分析、存校正版本。"""
    plan = {"points": points, "H_rough": H_rough.tolist(), "mark": args.mark, "stroke_mm": args.stroke,
            "press_mm": args.press, "predict_offset": np.asarray(offset).tolist(),
            "home": home, "orientation": ori, "layout": os.path.basename(args.layout)}
    log.save_json("plan", plan)
    if cam:
        frame1 = cam.frame(settle=1.0, flush=15)
        cam.close()
    else:
        time.sleep(1.0)
        frame1 = grab()
    log.save_image("home_after", frame1)
    res = pen_marks.analyze(frame1, PaperFrame(layout=args.layout), points, H_rough, args.stroke,
                            mark=args.mark, before=before_img, offset=offset)
    log.save_json("marks_result", res)
    log.save_image("marks_overlay.jpg", pen_marks.overlay(frame1, res))
    print("\n" + pen_marks.summarize(res))
    if res.get("ok"):
        p = CalibrationStore().save("pen_paper", {
            "A": res["A"], "t": res["t"], "plane": res.get("plane"), "rms_mm": res["rms_mm"],
            "home": home, "orientation": ori, "press_mm": args.press, "layout": os.path.basename(args.layout)},
            source_run=os.path.basename(log.path),
            notes="紙面 mm -> TCP 指令 XY: C = A·P + t (筆夾在夾爪,方向固定為 orientation)")
        print("已存校正版本:", os.path.relpath(p, HERE))
        log.finish("ok", {"rms_mm": res["rms_mm"], "max_mm": res["max_mm"], "n_points": res["n_points"]})
    else:
        print("分析失敗,可修改後用 python pen_marks.py", log.path, "重新分析")
        log.finish("ok", {"analysis": "failed", "reason": res.get("reason")})
    print("紀錄:", log.path)


if __name__ == "__main__":
    main()
