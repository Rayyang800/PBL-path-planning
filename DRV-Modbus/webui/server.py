# -*- coding: utf-8 -*-
"""
本機 Web 控制台 (只聽 127.0.0.1):即時 RealSense 影像、手臂狀態、校正結果、任務啟動、工件設定、執行紀錄。

  conda activate delta_arm
  cd DRV-Modbus
  python webui\\server.py              # 開 http://127.0.0.1:8080 (會自動開瀏覽器)
  python webui\\server.py --port 8090 --no-browser --no-robot --no-camera

設計
  - 只用 Python 標準函式庫 + 專案既有模組 (不用裝 Flask);網頁不連外部 CDN。
  - 相機同時只能給一支程式用:啟動會用相機的任務前,伺服器先釋放相機,任務結束後自動接回。
  - 任務在「新的主控台視窗」執行 (校正要用鍵盤:Enter / c / 空白 / Ctrl+C / o / g),結束後按任意鍵關閉視窗。
  - 手臂狀態唯讀輪詢 (0.5 s);任務執行中暫停輪詢,避免兩條 Modbus 連線互相干擾。
  - 網頁上的「送停止」寫 0x0300 = 0 [對 MovP/MovL 是否有效未驗證],實體急停才是可靠的停止方式。
  - 伺服器本身的動作 (啟動任務、夾爪、停止、快照) 都記在 runs/<時間>_webui/ (§0.4)。
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse, unquote

import cv2
import numpy as np

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_log                                           # noqa: E402
from run_log import RunLog, CalibrationStore, RUNS_DIR    # noqa: E402

CONFIG_PATH = os.path.join(WEB, "job_config.json")
STATIC = os.path.join(WEB, "static")
LAYOUT = os.path.join(ROOT, "print", "lab_sheet_layout.json")
ROBOT_IP = "192.168.1.1"

DEFAULT_CONFIG = {
    "workpiece": {"type": "B", "surface_h": 0.0, "notes": "", "depth_bias": 1.9},
    "common": {"speed": 15, "pen_loaded": True},
    "sheet": {"cols": 3, "rows": 3, "margin": 12.0, "stroke": 4.0, "hop": 5.0, "press": 0.0, "touch_all": False,
              "no_refine": False},
    "quick": {"step": 10.0, "hop": 3.0, "press": 0.0, "start_x": None, "start_y": None},
    "seam": {"board_h": 5.0, "step_mm": 2.0, "approach": 3.0, "press": 0.0, "reverse": False, "source": "auto", "look_dz": 150.0},
    "verify": {"target": "auto", "radius": 8.0, "dz": 0.0},
    "ridge": {"views": 5, "approach": 3.0, "press": 0.0, "step_mm": 2.0, "end_margin": 8.0, "reverse": False},
    "scan": {"cam_h": 380.0, "radius": 60.0, "n_ring": 8, "clearance": 40.0, "obj_h": 0.0,
             "tilts": "25,40", "n_tilt": 10, "view_dist": 290.0},
}
WORKPIECES = {"A": "A 立體 L 型 (脊線塗)", "B": "B 兩片板縫合塗", "C": "C 平面其他形狀 (邊緣塗)"}


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CONFIG_PATH)


# ======================= 相機 =======================
class CameraWorker(threading.Thread):
    def __init__(self, enabled=True):
        super().__init__(daemon=True)
        self.want = enabled
        self.cam = None
        self.info = None
        self.state = "啟動中" if enabled else "已停用 (--no-camera)"
        self.latest = None          # {"color", "depth"(uint16 或無), "time"}
        self.version = 0
        self.fps = 0.0
        self.lock = threading.Lock()
        self._capture_req = None    # (n_depth, Event, holder)

    def release(self, timeout=8.0):
        """釋放相機給其他程式用;回傳是否已釋放。"""
        self.want = False
        t0 = time.time()
        while self.cam is not None and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.cam is None

    def resume(self):
        self.want = True

    def capture(self, n_depth=15, timeout=10.0):
        """請相機執行緒拍一張多幀深度中位數的快照 (給『存快照』用)。"""
        if self.cam is None:
            return None
        ev, holder = threading.Event(), {}
        self._capture_req = (n_depth, ev, holder)
        ev.wait(timeout)
        return holder.get("snap")

    def _close(self):
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:       # noqa: BLE001
                pass
            self.cam = None

    def run(self):
        from rs_capture import RSCamera
        t_last, n = time.time(), 0
        while True:
            if not self.want:
                if self.cam is not None:
                    self._close()
                if not self.state.startswith("已停用"):
                    self.state = "已釋放 (任務使用中)"
                time.sleep(0.2)
                continue
            if self.cam is None:
                try:
                    # 先用「列出裝置」檢查 (很快);沒接相機時直接開 pipeline 會卡住整個 Python 十幾秒
                    import pyrealsense2 as rs
                    if len(rs.context().query_devices()) == 0:
                        self.state = "未連接 (找不到 RealSense)"
                        time.sleep(2.0)
                        continue
                    self.cam = RSCamera(depth=True, warmup=10)
                    self.info = self.cam.info()
                    self.state = "串流中"
                except Exception as e:      # noqa: BLE001
                    self.state = f"未連接 ({str(e)[:60]})"
                    self.cam = None
                    time.sleep(3.0)
                    continue
            try:
                req = self._capture_req
                if req is not None:
                    self._capture_req = None
                    req[2]["snap"] = self.cam.grab(n_depth=req[0])
                    req[1].set()
                snap = self.cam.grab(n_depth=1)
            except Exception as e:          # noqa: BLE001
                self.state = f"讀取失敗 ({str(e)[:60]})"
                self._close()
                time.sleep(1.0)
                continue
            with self.lock:
                self.latest = snap
                self.version += 1
            n += 1
            if time.time() - t_last >= 1.0:
                self.fps, n, t_last = n / (time.time() - t_last), 0, time.time()


# ======================= 影像處理 (疊圖 / 深度 / 高度) =======================
class Renderer:
    def __init__(self, camw):
        self.camw = camw
        self.cache = {}             # (view, overlay) -> (version, jpeg)
        self.paper = {"ok": False, "desc": "尚未偵測"}
        self._paper_ver = -99
        self._pf = None
        self._aruco = None
        self.height = None          # (version, height_map, stats)
        self._height_req = 0.0
        self._work_t = 0.0
        self.auto_analyze = True    # 相機串流中每 ~4 秒分析一次工件 (板厚、影像縫、深度縫)
        self.work = {"ok": False, "msg": "等待相機"}
        self.lock = threading.Lock()
        threading.Thread(target=self._height_loop, daemon=True).start()

    def _detect(self, snap, ver):
        if ver - self._paper_ver < 3 and self._pf is not None:
            return
        from paper_frame import PaperFrame
        from record_apriltags import make_detector, detect
        self._paper_ver = ver
        try:
            pf = PaperFrame(layout=LAYOUT)
            ok = pf.detect(snap["color"])
            self._pf = pf
            if self._aruco is None:
                self._aruco = make_detector("5x5_100")
            corners, ids, _ = detect(cv2.cvtColor(snap["color"], cv2.COLOR_BGR2GRAY), *self._aruco)
            self.paper = {"ok": bool(ok), "desc": pf.describe(), "aruco": sorted(int(i) for i in ids),
                          "aruco_corners": [c.tolist() for c in corners], "ids": [int(i) for i in ids],
                          "cam_h": float(pf.cam_h) if ok else None, "rms_mm": float(pf.rms_mm) if ok else None,
                          "mm_per_px": float(pf.mm_per_pixel()) if ok else None}
        except Exception as e:              # noqa: BLE001
            self.paper = {"ok": False, "desc": f"偵測錯誤: {e}"}

    def _overlay(self, img):
        pf, p = self._pf, self.paper
        for c, i in zip(p.get("aruco_corners", []), p.get("ids", [])):
            pts = np.int32(c)
            cv2.polylines(img, [pts], True, (255, 160, 0), 2)
            cv2.putText(img, f"A{i}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
        if pf is not None and pf.found:
            for tid, c in pf.found.items():
                cv2.polylines(img, [np.int32(c)], True, (0, 220, 0), 2)
                cv2.putText(img, f"T{tid}", tuple(np.int32(c[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        if pf is not None and pf.ok:
            W, H = pf.W, pf.H
            for x in np.arange(0, W + 0.1, 50):
                a, b = pf.paper_to_pix([(x, 0), (x, H)])
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (200, 200, 200), 1, cv2.LINE_AA)
            for y in np.arange(0, H + 0.1, 50):
                a, b = pf.paper_to_pix([(0, y), (W, y)])
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (200, 200, 200), 1, cv2.LINE_AA)
            from pen_calibrate import safe_region
            x0, y0, x1, y1 = safe_region(pf, 12.0)
            q = pf.paper_to_pix([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            cv2.polylines(img, [np.int32(q)], True, (0, 200, 255), 1, cv2.LINE_AA)
            o, ax, ay = pf.paper_to_pix([(0, 0), (40, 0), (0, 40)])
            cv2.arrowedLine(img, tuple(np.int32(o)), tuple(np.int32(ax)), (0, 0, 255), 2, tipLength=0.25)
            cv2.arrowedLine(img, tuple(np.int32(o)), tuple(np.int32(ay)), (0, 255, 0), 2, tipLength=0.25)
            txt = f"paper RMS {pf.rms_mm:.2f}mm  cam {pf.cam_h:.0f}mm  {pf.mm_per_pixel():.3f}mm/px"
        else:
            txt = "paper: " + ("tags " + str(sorted(pf.found)) if pf is not None else "-")
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _height_loop(self):
        """每 ~4 秒分析一次工件 (相機串流中才做):高度圖、板厚 (深度)、影像縫、深度縫。結果給狀態與疊圖用。"""
        import depth_height
        import seam_boards
        from paper_frame import PaperFrame
        self.work = {"ok": False, "msg": "等待相機"}
        while True:
            time.sleep(0.5)
            if self.camw.cam is None or not self.camw.want or not self.auto_analyze:
                continue
            if time.time() - self._work_t < 4.0 and time.time() - self._height_req > 0.5:
                continue
            self._work_t = time.time()
            snap = self.camw.capture(n_depth=8, timeout=5.0)    # 8 幀深度中位數 (單幀雜訊太大)
            info = self.camw.info
            if snap is None or snap.get("depth") is None or info is None:
                continue
            cfg = load_config()
            board_h = float(cfg["seam"].get("board_h") or 0) or 5.0
            bias = float(cfg["workpiece"].get("depth_bias", 0.0) or 0.0)
            w = {"ok": False, "t": time.time()}
            try:
                mm = snap["depth"].astype(np.float32) * float(info["depth_scale_m"]) * 1000.0
                mm[snap["depth"] == 0] = np.nan
                H, pf, st = depth_height.height_map(snap["color"], mm, info["color_intrinsics"], LAYOUT)
                self.height = (self.camw.version, H, st)
                w.update(paper_rms_mm=st["paper_rms_mm"])
                boards, _ = seam_boards.find_boards(snap["color"], pf, board_h)
                if not boards:
                    w["msg"] = "紙上沒有找到工件 (暗色板子)"
                else:
                    b = boards[0]
                    w.update(ok=True, msg="", board_size_mm=b["size_mm"], n_boards=len(boards))
                    est = depth_height.board_thickness(H, b["contour"])
                    if est:
                        w.update(t_raw_mm=est["t_mm"], t_spread_mm=est["spread_mm"], depth_bias_mm=bias,
                                 t_corrected_mm=est["t_mm"] + bias, board_h_setting=board_h)
                    col = seam_boards.analyze(snap["color"], PaperFrame(layout=LAYOUT), board_h_mm=board_h)
                    dsm = depth_height.depth_seam(H, pf, b["box"], board_h)
                    w["color_seam"] = {"ok": bool(col["ok"]), "mode": col.get("mode"), "reason": col.get("reason"),
                                       "length_mm": col.get("length_mm")}
                    w["depth_seam"] = {"ok": bool(dsm["ok"]), "reason": dsm.get("reason"), "dip_mm": dsm.get("dip_mm"),
                                       "length_mm": dsm.get("length_mm")}
                    lines = {"board_h": board_h}
                    if col["ok"]:
                        lines["color"] = np.asarray(col["seam_paper_mm"]).tolist()
                    if dsm["ok"]:
                        lines["depth"] = np.asarray(dsm["seam_paper_mm"]).tolist()
                    if col["ok"] and dsm["ok"]:
                        off, ang = depth_height.line_offset(col["seam_paper_mm"], dsm["seam_paper_mm"])
                        w["seam_diff"] = {"offset_mm": off, "angle_deg": ang, "agree": abs(off) <= 1.5 and ang <= 3.0}
                    self.seam_lines = lines
                    self._seam_pf = pf
            except Exception as e:          # noqa: BLE001
                w = {"ok": False, "msg": f"分析錯誤: {str(e)[:80]}", "t": time.time()}
            self.work = w

    def _draw_seams(self, img):
        """在影像上畫影像縫 (橘) 與深度縫 (洋紅)。"""
        lines, pf = getattr(self, "seam_lines", None), getattr(self, "_seam_pf", None)
        if not lines or pf is None:
            return img
        bh = lines.get("board_h", 0)
        for key, col in (("depth", (255, 0, 255)), ("color", (0, 140, 255))):
            if key in lines:
                p = np.int32(np.round(pf.paper_to_pix(np.asarray(lines[key]), bh)))
                cv2.polylines(img, [p], False, col, 2, cv2.LINE_AA)
        return img

    def jpeg(self, view="color", overlay=True, width=960):
        with self.camw.lock:
            snap, ver = self.camw.latest, self.camw.version
        if snap is None:
            return self._placeholder(self.camw.state)
        key = (view, overlay, width)
        with self.lock:
            c = self.cache.get(key)
            if c is not None and c[0] == ver:
                return c[1]
        if view == "depth" and snap.get("depth") is not None:
            d = snap["depth"].astype(np.float32)
            valid = d > 0
            lo, hi = (np.percentile(d[valid], [2, 98]) if valid.any() else (0, 1))
            vis = cv2.applyColorMap(np.clip((d - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            vis[~valid] = (40, 40, 40)
            img = vis
            sc = float((self.camw.info or {}).get("depth_scale_m") or 0.001) * 1000.0
            cv2.putText(img, f"depth {lo * sc:.0f}~{hi * sc:.0f} mm", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif view == "height":
            self._height_req = time.time()
            h = self.height
            img = snap["color"].copy()
            if h is not None and h[1] is not None:
                Hm = h[1]
                vis = cv2.applyColorMap(np.clip((np.nan_to_num(Hm, nan=-99) + 5) / 25 * 255, 0, 255).astype(np.uint8),
                                        cv2.COLORMAP_TURBO)
                m = ~np.isnan(Hm)
                img[m] = (0.35 * img[m] + 0.65 * vis[m]).astype(np.uint8)
                st = h[2]
                txt = f"height -5~+20mm  paper RMS {st['paper_rms_mm']:.2f}mm  orange=image seam  magenta=depth seam"
            else:
                txt = "height: " + (h[2].get("error", "computing") if h else "computing (need 4 tags + depth)")
            if overlay:
                img = self._draw_seams(img)
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(img, txt[:110], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            img = snap["color"].copy()
            if overlay:
                self._detect(snap, ver)
                img = self._overlay(img)
                img = self._draw_seams(img)
        if width and img.shape[1] > width:
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = buf.tobytes()
        with self.lock:
            self.cache[key] = (ver, data)
        return data

    @staticmethod
    def _placeholder(msg):
        img = np.full((540, 960, 3), 45, np.uint8)
        cv2.putText(img, "CAMERA", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (180, 180, 180), 3)
        ascii_msg = msg.encode("ascii", "ignore").decode() or "see status panel"
        cv2.putText(img, ascii_msg[:60], (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return cv2.imencode(".jpg", img)[1].tobytes()


# ======================= 手臂狀態 =======================
class RobotWorker(threading.Thread):
    def __init__(self, enabled=True, ip=ROBOT_IP):
        super().__init__(daemon=True)
        self.enabled, self.ip = enabled, ip
        self.paused = False
        self.state = {"ok": False, "msg": "啟動中" if enabled else "已停用 (--no-robot)"}
        self.c = None
        self.lock = threading.Lock()        # 任何 Modbus 讀寫都要拿這個鎖

    def _connect(self):
        from pymodbus.client import ModbusTcpClient
        c = ModbusTcpClient(host=self.ip, port=502, timeout=1.5)
        if not c.connect():
            raise ConnectionError(f"連不上 {self.ip}:502")
        return c

    def disconnect(self):
        with self.lock:
            if self.c is not None:
                try:
                    self.c.close()
                except Exception:           # noqa: BLE001
                    pass
                self.c = None

    def run(self):
        from drv_modbus import request
        try:
            with open(os.path.join(ROOT, "calibration.json"), encoding="utf-8") as f:
                home = json.load(f)["home"]
        except Exception:                   # noqa: BLE001
            home = None
        while True:
            if not self.enabled:
                time.sleep(1.0)
                continue
            if self.paused:
                self.disconnect()
                self.state = dict(self.state, ok=False, msg="任務執行中,暫停讀取 (避免干擾)")
                time.sleep(0.5)
                continue
            try:
                with self.lock:
                    if self.c is None:
                        self.c = self._connect()
                    pose = request.Get_TCP_Pose(self.c)
                    flag = request.Get_Pose_Flag(self.c)
                    r = self.c.read_holding_registers(0x02FE, 1, 2)
                    do = None if r.isError() else r.registers[0]
                if pose is None:
                    raise ConnectionError("讀不到位姿")
                st = {"ok": True, "msg": "連線中", "pose": [round(v, 2) for v in pose], "flag": flag, "do": do,
                      "gripper": {64: "張開", 128: "夾緊", 0: "無輸出", 1: "吸盤"}.get(do, str(do)), "t": time.time()}
                if home:
                    st["home_dpos"] = round(max(abs(a - b) for a, b in zip(pose[:3], home[:3])), 1)
                    st["home_dang"] = round(max(abs(a - b) for a, b in zip(pose[3:], home[3:])), 1)
                    st["at_home"] = st["home_dpos"] <= 5 and st["home_dang"] <= 2
                self.state = st
                time.sleep(0.5)
            except Exception as e:          # noqa: BLE001
                self.disconnect()
                self.state = {"ok": False, "msg": f"離線 ({str(e)[:60]})", "t": time.time()}
                time.sleep(2.0)

    def with_client(self, fn):
        """用獨立的短連線執行寫入 (停止、夾爪)。"""
        from pymodbus.client import ModbusTcpClient
        with self.lock:
            c = ModbusTcpClient(host=self.ip, port=502, timeout=2)
            if not c.connect():
                raise ConnectionError(f"連不上 {self.ip}:502")
            try:
                return fn(c)
            finally:
                c.close()


# ======================= 任務 =======================
def task_registry(cfg):
    """回傳 {task_id: {label, script, uses_camera, moves, build(mode) -> argv}}。"""
    cm, sh, qk, sm, wp_ = cfg["common"], cfg["sheet"], cfg["quick"], cfg["seam"], cfg["workpiece"]
    pen = ["--pen-loaded"] if cm.get("pen_loaded") else []
    spd = ["--speed", str(int(cm.get("speed", 15)))]

    def sheet(mode):
        a = ["--grid", str(int(sh["cols"])), str(int(sh["rows"])), "--margin", str(sh["margin"]), "--stroke", str(sh["stroke"]),
             "--hop", str(sh["hop"]), "--press", str(sh["press"]), "--surface-h", str(wp_.get("surface_h", 0))] + spd + pen
        a += ["--touch-all"] if sh.get("touch_all") else []
        a += ["--no-refine"] if sh.get("no_refine") else []
        return a + (["--go"] if mode == "go" else [])

    def quick(mode):
        a = ["--step", str(qk["step"]), "--hop", str(qk["hop"]), "--press", str(qk["press"]),
             "--surface-h", str(wp_.get("surface_h", 0))] + spd + pen
        if qk.get("start_x") not in (None, "") and qk.get("start_y") not in (None, ""):
            a += ["--start-paper", str(qk["start_x"]), str(qk["start_y"])]
        return a + (["--go"] if mode == "go" else [])

    vf = cfg.get("verify", DEFAULT_CONFIG["verify"])
    sc = cfg.get("scan", DEFAULT_CONFIG["scan"])
    rg = cfg.get("ridge", DEFAULT_CONFIG["ridge"])

    def verify(mode):
        a = ["--target", str(vf.get("target") or "auto"), "--radius", str(vf.get("radius", 8.0)),
             "--dz", str(vf.get("dz", 0.0))] + spd
        return a + (["--go"] if mode == "go" else [])

    return {
        "verify_circle": {"label": "畫圓驗證 (視覺 -> 筆尖)", "script": "pen_verify_circle.py", "uses_camera": True,
                          "moves": True, "build": verify},
        "sheet_calib": {"label": "護貝紙整張校正", "script": "pen_sheet_calib.py", "uses_camera": True, "moves": True,
                        "build": sheet},
        "quick_calib": {"label": "快速校正 (工件上 ±step)", "script": "pen_quick_calib.py", "uses_camera": True,
                        "moves": True, "build": quick},
        "go_home": {"label": "回 home", "script": "go_home.py", "uses_camera": False, "moves": True, "always_moves": True,
                    "build": lambda m: ["--speed", str(min(40, int(cm.get("speed", 15)) * 2))]},
        "gripper_ctl": {"label": "鍵盤控制夾爪", "script": "gripper_ctl.py", "uses_camera": False, "moves": False,
                        "build": lambda m: []},
        "seam_trace": {"label": "沿縫描繪", "script": "seam_trace.py", "uses_camera": True, "moves": True,
                       "build": lambda m: ((["--board-h", str(sm["board_h"])] if float(sm.get("board_h") or 0) > 0 else [])
                                           + ["--approach", str(sm.get("approach", 3.0)), "--press", str(sm.get("press", 0.0)),
                                              "--step-mm", str(sm.get("step_mm", 2.0)), "--speed", str(min(15, int(cm.get("speed", 10))))]
                                           + ["--seam-source", str(sm.get("source") or "auto")]
                                           + ["--look-dz", str(float(sm.get("look_dz") if sm.get("look_dz") is not None else 150.0))]
                                           + (["--reverse"] if sm.get("reverse") else []) + (["--go"] if m == "go" else []))},
        "depth_closeup": {"label": "近距離深度量測 (手臂下降 150 mm)", "script": "depth_closeup.py", "uses_camera": True,
                          "moves": True, "always_moves": True,
                          "build": lambda m: ["--dz", "150", "--board-h", str(float(sm.get("board_h") or 5.0))]},
        "ridge_trace": {"label": "沿脊線描繪 (工件 A)", "script": "ridge_trace.py", "uses_camera": True, "moves": True,
                        "build": lambda m: ["--views", str(int(rg.get("views", 5))), "--approach", str(rg.get("approach", 3.0)),
                                            "--press", str(rg.get("press", 0.0)), "--step-mm", str(rg.get("step_mm", 2.0)),
                                            "--end-margin", str(rg.get("end_margin", 8.0)),
                                            "--speed", str(min(15, int(cm.get("speed", 10))))]
                                           + (["--reverse"] if rg.get("reverse") else []) + (["--go"] if m == "go" else [])},
        "scan3d": {"label": "多視角立體重建 (不下筆)", "script": "scan3d.py", "uses_camera": True, "moves": True,
                   "build": lambda m: ["--cam-h", str(sc.get("cam_h", 380.0)), "--radius", str(sc.get("radius", 60.0)),
                                       "--n-ring", str(int(sc.get("n_ring", 8))), "--clearance", str(max(20.0, float(sc.get("clearance", 40.0)))),
                                       "--obj-h", str(float(sc.get("obj_h") or 0.0)),
                                       "--tilts", str(sc.get("tilts") if sc.get("tilts") is not None else "25,40").replace(" ", "") or "",
                                       "--n-tilt", str(int(sc.get("n_tilt", 10))), "--view-dist", str(float(sc.get("view_dist") or 290.0)),
                                       "--speed", str(min(20, int(cm.get("speed", 15))))] + (["--go"] if m == "go" else [])},
        "depth_compare": {"label": "深度設定比較 (唯讀)", "script": "depth_compare.py", "uses_camera": True, "moves": False,
                          "build": lambda m: ["--truth", str(float(sm.get("board_h") or 5.0))]},
        "seam_boards": {"label": "縫合線擷取 (唯讀)", "script": "seam_boards.py", "uses_camera": True, "moves": False,
                        "build": lambda m: ["--camera", "--board-h", str(sm.get("board_h", 0)), "--step-mm", str(sm.get("step_mm", 2)),
                                            "--label", f"wp{wp_.get('type', '')}"]},
        "warp_sort": {"label": "筆蓋分類 (舊 ArUco)", "script": "warp_sort.py", "uses_camera": True, "moves": True,
                      "build": lambda m: spd + (["--go"] if m == "go" else [])},
    }


class TaskManager:
    def __init__(self, camw, robw, slog):
        self.camw, self.robw, self.slog = camw, robw, slog
        self.proc = None
        self.info = None
        self.last = None
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is not None:
                self._finished()
            return {"running": self.proc is not None, "current": self.info, "last": self.last}

    def _finished(self):
        code = self.proc.returncode
        info = dict(self.info, ended=time.time(), exit_code=code)
        info["run"] = newest_run_since(info["started"] - 1)
        self.last = info
        self.slog.event("task_end", **{k: v for k, v in info.items() if k != "argv"})
        self.proc, self.info = None, None
        self.robw.paused = False
        self.camw.resume()

    def launch(self, task_id, mode):
        cfg = load_config()
        reg = task_registry(cfg)
        if task_id not in reg:
            raise ValueError("未知任務: " + task_id)
        t = reg[task_id]
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("已有任務在執行:" + self.info["label"])
            if self.proc is not None:
                self._finished()
            if t["uses_camera"] and not self.camw.release():
                raise RuntimeError("相機釋放逾時")
            self.robw.paused = True
            self.robw.disconnect()
            argv = [sys.executable, os.path.join(ROOT, t["script"])] + t["build"](mode)
            # cmd 的特殊字元 (> < | & ^) 不能出現在 title 指令裡:"->" 曾被當成輸出導向,產生名為「筆尖)」的空檔
            title = "".join(ch for ch in f"{t['label']} ({mode})" if ch not in '<>|&^"%')
            # cmd /c "... & pause":結束後視窗留著讓人看結果,按任意鍵關閉
            line = subprocess.list2cmdline(argv)
            cmd = f'title {title} & {line} & echo. & echo ===== 程式已結束,按任意鍵關閉視窗 ===== & pause >nul'
            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            self.proc = subprocess.Popen(["cmd", "/c", cmd], cwd=ROOT, creationflags=flags, env=env)
            self.info = {"task": task_id, "label": t["label"], "mode": mode, "argv": argv, "pid": self.proc.pid,
                         "started": time.time(), "cmdline": line}
            self.slog.event("task_start", task=task_id, mode=mode, argv=argv)
            return self.info

    def kill(self):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return False
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            self.slog.event("task_kill", pid=self.proc.pid)
        return True


# ======================= 紀錄 / 校正 =======================
def list_runs(limit=40):
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for d in sorted(os.listdir(RUNS_DIR), reverse=True)[:limit * 2]:
        p = os.path.join(RUNS_DIR, d, "meta.json")
        try:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        res = m.get("result") or {}
        out.append({"id": d, "kind": m.get("kind"), "label": m.get("label"), "status": m.get("status"),
                    "started": m.get("started"), "ended": m.get("ended"),
                    "summary": {k: res[k] for k in list(res)[:6] if not isinstance(res[k], (dict, list))},
                    "n_snap": len(m.get("snapshots", [])), "schema": m.get("schema")})
        if len(out) >= limit:
            break
    return out


def list_models(limit=30):
    """有立體模型 (data/model.ply) 的紀錄,新到舊。"""
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for d in sorted(os.listdir(RUNS_DIR), reverse=True):
        if not os.path.isfile(os.path.join(RUNS_DIR, d, "data", "model.ply")):
            continue
        try:
            with open(os.path.join(RUNS_DIR, d, "meta.json"), encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            m = {}
        out.append({"id": d, "kind": m.get("kind"), "label": m.get("label"), "status": m.get("status"),
                    "started": m.get("started"), "result": m.get("result")})
        if len(out) >= limit:
            break
    return out


def newest_run_since(t0):
    if not os.path.isdir(RUNS_DIR):
        return None
    best = None
    for d in os.listdir(RUNS_DIR):
        p = os.path.join(RUNS_DIR, d)
        try:
            if os.path.getmtime(os.path.join(p, "meta.json")) >= t0 and "_webui" not in d:
                best = d if best is None or d > best else best
        except OSError:
            pass
    return best


def run_detail(run_id):
    path = safe_join(RUNS_DIR, run_id)
    if path is None or not os.path.isdir(path):
        raise FileNotFoundError(run_id)
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    imgs = sorted(os.listdir(os.path.join(path, "images"))) if os.path.isdir(os.path.join(path, "images")) else []
    data = sorted(os.listdir(os.path.join(path, "data"))) if os.path.isdir(os.path.join(path, "data")) else []
    console = ""
    cp = os.path.join(path, "console.log")
    if os.path.isfile(cp):
        with open(cp, encoding="utf-8", errors="replace") as f:
            console = "".join(f.readlines()[-120:])
    results = {}
    for name in ("marks_result.json", "verify_result.json", "trace_result.json", "result.json", "plan.json"):
        fp = os.path.join(path, "data", name)
        if os.path.isfile(fp) and os.path.getsize(fp) < 400_000:
            with open(fp, encoding="utf-8") as f:
                results[name] = json.load(f)
    n_events = 0
    ep = os.path.join(path, "events.jsonl")
    if os.path.isfile(ep):
        with open(ep, encoding="utf-8") as f:
            n_events = sum(1 for _ in f)
    return {"id": run_id, "meta": meta, "images": imgs, "data": data, "console": console, "results": results,
            "n_events": n_events}


def calib_summary():
    out = {}
    store = CalibrationStore()
    for name in ("pen_paper", "warp_plane"):
        try:
            rec = store.load_latest(name)
        except FileNotFoundError:
            out[name] = None
            continue
        d = rec.get("data", {})
        out[name] = {"created": rec.get("created"), "source_run": rec.get("source_run"), "n_versions": len(store.history(name)),
                     "method": d.get("method"), "fit": d.get("fit"), "rms_mm": d.get("rms_mm"),
                     "loo_max_mm": d.get("loo_max_mm"), "n_points": d.get("n_points"), "z_touch": d.get("z_touch"),
                     "span_mm": d.get("span_mm"), "est_err_at_100mm": d.get("est_err_at_100mm")}
    return out


def safe_join(base, rel):
    p = os.path.realpath(os.path.join(base, rel))
    return p if p.startswith(os.path.realpath(base) + os.sep) or p == os.path.realpath(base) else None


# ======================= HTTP =======================
class App:
    camw = robw = rend = tasks = slog = None


MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".json": "application/json; charset=utf-8",
        ".log": "text/plain; charset=utf-8", ".ply": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    server_version = "DeltaArmWebUI/1"

    def log_message(self, fmt, *a):          # 不要把每個請求印到終端
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if path is None or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            data = f.read()
        self._send(200, data, MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream"))

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._file(os.path.join(STATIC, "index.html"))
            if u.path.startswith("/static/"):
                return self._file(safe_join(STATIC, unquote(u.path[len("/static/"):])))
            if u.path == "/stream.mjpg":
                return self._stream(q.get("view", "color"), q.get("overlay", "1") == "1")
            if u.path == "/frame.jpg":
                return self._send(200, App.rend.jpeg(q.get("view", "color"), q.get("overlay", "1") == "1", 1280), "image/jpeg")
            if u.path == "/api/status":
                return self._send(200, status_payload())
            if u.path == "/api/runs":
                return self._send(200, list_runs(int(q.get("limit", 40))))
            if u.path == "/api/models":
                return self._send(200, list_models(int(q.get("limit", 30))))
            if u.path == "/api/run":
                return self._send(200, run_detail(q["id"]))
            if u.path == "/api/config":
                return self._send(200, {"config": load_config(), "workpieces": WORKPIECES,
                                        "tasks": {k: {kk: vv for kk, vv in v.items() if kk != "build"}
                                                  for k, v in task_registry(load_config()).items()}})
            if u.path == "/api/preview_cmd":
                reg = task_registry(load_config())
                t = reg[q["task"]]
                return self._send(200, {"cmd": subprocess.list2cmdline(["python", t["script"]] + t["build"](q.get("mode", "dry")))})
            if u.path.startswith("/files/runs/"):
                return self._file(safe_join(RUNS_DIR, unquote(u.path[len("/files/runs/"):])))
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as e:              # noqa: BLE001
            return self._send(500, {"error": str(e), "trace": traceback.format_exc()[-800:]})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._json_body()
            if u.path == "/api/config":
                cfg = load_config()
                for k, v in body.items():
                    if k in cfg and isinstance(v, dict):
                        cfg[k].update(v)
                save_config(cfg)
                App.slog.event("config_saved", config=cfg)
                return self._send(200, {"ok": True, "config": cfg})
            if u.path == "/api/launch":
                info = App.tasks.launch(body["task"], body.get("mode", "dry"))
                return self._send(200, {"ok": True, "task": info})
            if u.path == "/api/kill":
                return self._send(200, {"ok": App.tasks.kill()})
            if u.path == "/api/robot/stop":
                from drv_modbus import send
                App.robw.with_client(send.Stop)          # run_log.current() = 伺服器紀錄 -> 自動寫 cmd_stop
                return self._send(200, {"ok": True, "note": "已寫 0x0300=0 (未驗證是否能停止移動中的手臂,請以實體急停為主)"})
            if u.path == "/api/gripper":
                if App.tasks.status()["running"]:
                    raise RuntimeError("任務執行中,請在任務視窗用 o / g 控制夾爪")
                from drv_modbus import gripper
                fn = gripper.Gripper_Close if body.get("state") == "close" else gripper.Gripper_Open
                App.robw.with_client(lambda c: fn(c, delay=0))
                return self._send(200, {"ok": True})
            if u.path == "/api/snapshot":
                return self._send(200, take_snapshot(body.get("label", "")))
            return self._send(404, {"error": "not found"})
        except Exception as e:              # noqa: BLE001
            return self._send(400, {"ok": False, "error": str(e)})

    def _stream(self, view, overlay):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while True:
                data = App.rend.jpeg(view, overlay)
                if data is not last:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() +
                                     b"\r\n\r\n" + data + b"\r\n")
                    last = data
                time.sleep(0.07 if App.camw.latest is not None else 0.5)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return


def status_payload():
    cam = App.camw
    return {"time": time.time(),
            "camera": {"state": cam.state, "fps": round(cam.fps, 1), "info": {k: cam.info.get(k) for k in ("name", "serial", "usb", "firmware")}
                       if cam.info else None, "has_depth": bool(cam.latest and cam.latest.get("depth") is not None)},
            "paper": {k: v for k, v in App.rend.paper.items() if k not in ("aruco_corners",)},
            "work": App.rend.work,
            "robot": App.robw.state, "task": App.tasks.status(), "calib": calib_summary(),
            "server_run": os.path.basename(App.slog.path)}


def take_snapshot(label=""):
    snap = App.camw.capture(n_depth=15)
    if snap is None:
        raise RuntimeError("相機沒有串流 (未連接或任務使用中)")
    with RunLog("web_snapshot", label=label, console=False,
                params={"robot": App.robw.state, "paper": {k: v for k, v in App.rend.paper.items() if k != "aruco_corners"}}) as log:
        log.add_input(LAYOUT)
        if App.robw.state.get("ok"):
            log.event("tcp", pose=App.robw.state.get("pose"), flag=App.robw.state.get("flag"))
        log.save_snapshot("frame", snap, App.camw.info)
        path = log.path
    App.slog.event("snapshot", run=os.path.basename(path))
    return {"ok": True, "run": os.path.basename(path)}


def fix_stale_sessions():
    """先前的 Web 控制台若被直接關掉 (沒按 Ctrl+C),它的紀錄會一直是 running;啟動時改成 aborted。"""
    if not os.path.isdir(RUNS_DIR):
        return
    for d in os.listdir(RUNS_DIR):
        if not d.endswith("_webui"):
            continue
        p = os.path.join(RUNS_DIR, d, "meta.json")
        try:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("status") == "running":
                m.update(status="aborted", notes=(m.get("notes") or "") + " 伺服器未正常結束 (下次啟動時標記)")
                with open(p + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2, ensure_ascii=False)
                os.replace(p + ".tmp", p)
        except (OSError, ValueError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-robot", action="store_true", help="不連手臂 (只看相機與紀錄)")
    ap.add_argument("--no-camera", action="store_true", help="不開相機")
    ap.add_argument("--robot-ip", default=ROBOT_IP)
    args = ap.parse_args()

    fix_stale_sessions()
    with RunLog("webui", params=vars(args)) as slog:
        App.slog = slog
        App.camw = CameraWorker(enabled=not args.no_camera)
        App.robw = RobotWorker(enabled=not args.no_robot, ip=args.robot_ip)
        App.rend = Renderer(App.camw)
        App.tasks = TaskManager(App.camw, App.robw, slog)
        App.camw.start()
        App.robw.start()
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        srv.daemon_threads = True
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Web 控制台:{url}  (Ctrl+C 結束)")
        if not args.no_browser:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            print("\n結束 Web 控制台")
        finally:
            srv.server_close()
            App.camw.release(timeout=3)


if __name__ == "__main__":
    main()
