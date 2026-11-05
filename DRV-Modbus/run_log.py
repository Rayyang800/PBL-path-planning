# -*- coding: utf-8 -*-
"""
執行紀錄 (RunLog) 與校正版本庫 (CalibrationStore)。純檔案,不連線手臂/相機。

目的:每次執行/校正都留下「可重現、可比對、可匯出給模擬軟體」的完整紀錄,且**不覆蓋舊資料**。

資料夾結構
  runs/<YYYYMMDD-HHMMSS>_<kind>[_<label>]/
      meta.json      誰、何時、參數、環境、使用的輸入檔雜湊、結果、狀態 (running/ok/failed/aborted/dry-run)
      events.jsonl   時間序事件,一行一個 JSON:{"t":秒,"ts":ISO時間,"type":...,...}
      inputs/        執行當下使用的校正檔、版面檔等的複本 (含 sha256)
      images/        原始影像 (raw) 與疊圖 (overlay)
      data/          路徑、偏差等 CSV/JSON
  calibrations/<name>/<YYYYMMDD-HHMMSS>.json   每次校正一個新檔 (含 supersedes 指向前一版)
  calibrations/<name>/latest.json              最新版的複本 (Windows 不用 symlink)

約定
  - 單位:手臂座標 mm / 度;紙面座標 mm;影像 px。座標資料一律在欄位或檔名註明 frame (robot / paper / pixel)。
  - 事件 type 建議:cmd_move{pose,speed,mov,frame}、cmd_gripper{state}、tcp{pose,flag}、note、error。
  - 先存原始資料 (raw 影像、指令、回讀位姿),衍生結果 (偏差、統計) 可以事後重算。
  - 永不覆蓋:同名資料夾自動加 -1、-2。

v2 (11-05) 新增
  - save_snapshot(name, snap, cam_info):彩色 images/<name>.png + 深度 depth/<name>.png (uint16,對齊彩色,
    單位見 depth/<name>.json 的 depth_scale_m) + 相機內參;load_snapshot(run, name) 讀回 (深度轉成 mm,0 -> NaN)。
  - console.log:執行期間終端機的所有輸出 (console=False 可關)。
  - meta.git:commit 與是否有未提交修改;meta.camera:相機序號/韌體/內參 (第一次 save_snapshot 時寫入)。
  - current():目前進行中的紀錄。drv_modbus.send.Go_Position 與 gripper 會自動把指令、移動中的軌跡
    (tcp_trace,約 10 Hz)、夾爪動作寫進去,舊腳本只要包在 with RunLog(...) 裡就有完整紀錄。
"""
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
CALIB_DIR = os.path.join(HERE, "calibrations")
SCHEMA = 2
_CURRENT = []            # 進行中的 RunLog (堆疊)


def current():
    """目前進行中的紀錄;沒有則 None。"""
    return _CURRENT[-1] if _CURRENT else None


def _git_info():
    import subprocess
    try:
        kw = dict(cwd=HERE, capture_output=True, text=True, timeout=5)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], **kw).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], **kw).stdout.strip()
        return {"commit": commit or None, "dirty_files": dirty.splitlines()[:50] if dirty else []}
    except Exception:        # noqa: BLE001 - 沒有 git 也照常紀錄
        return None


class _Tee:
    """把 stdout/stderr 同時寫到 console.log。"""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        try:
            self.f.write(s)
            self.f.flush()
        except ValueError:   # 檔案已關
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def _now():
    return dt.datetime.now().astimezone()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    os.replace(tmp, path)


def _packages():
    out = {}
    try:
        from importlib import metadata
    except ImportError:          # pragma: no cover
        return out
    for pkg in ("opencv-contrib-python", "numpy", "scipy", "pymodbus", "pyrealsense2", "matplotlib"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:        # noqa: BLE001
            pass
    return out


def _unique_dir(base):
    path, n = base, 1
    while os.path.exists(path):
        path = f"{base}-{n}"
        n += 1
    os.makedirs(path)
    return path


class RunLog:
    """with RunLog("seam", label="two_boards", params=vars(args)) as log: ...  (離開 with 時自動 finish)"""

    def __init__(self, kind, label="", runs_dir=RUNS_DIR, params=None, hardware=None, script=None, argv=None,
                 console=True):
        self._t0 = time.time()
        started = _now()
        name = f"{started.strftime('%Y%m%d-%H%M%S')}_{kind}" + (f"_{label}" if label else "")
        os.makedirs(runs_dir, exist_ok=True)
        self.path = _unique_dir(os.path.join(runs_dir, name))
        for sub in ("inputs", "images", "data"):
            os.makedirs(os.path.join(self.path, sub))
        script = script or (sys.argv[0] if sys.argv and sys.argv[0] else None)
        self.meta = {
            "schema": SCHEMA, "run_id": os.path.basename(self.path), "kind": kind, "label": label,
            "started": started.isoformat(timespec="seconds"), "ended": None, "status": "running",
            "script": {"name": os.path.basename(script) if script else None,
                       "sha256": sha256_file(script) if script and os.path.isfile(script) else None,
                       "argv": list(argv if argv is not None else sys.argv[1:])},
            "env": {"python": sys.version.split()[0], "platform": platform.platform(), "packages": _packages()},
            "hardware": hardware or {}, "params": params or {}, "inputs": {}, "result": None, "notes": "",
            "git": _git_info(), "camera": None, "snapshots": [],
        }
        self._events = open(os.path.join(self.path, "events.jsonl"), "a", encoding="utf-8")
        self._finished = False
        self._tee = None
        if console:
            self._console = open(os.path.join(self.path, "console.log"), "a", encoding="utf-8")
            self._tee = (sys.stdout, sys.stderr)
            sys.stdout, sys.stderr = _Tee(sys.stdout, self._console), _Tee(sys.stderr, self._console)
        self._write_meta()
        _CURRENT.append(self)

    # ----- writing helpers (return path relative to the run folder) -----
    def _write_meta(self):
        _dump(os.path.join(self.path, "meta.json"), self.meta)

    def event(self, type_, **data):
        if self._finished:
            return
        rec = {"t": round(time.time() - self._t0, 3), "ts": _now().isoformat(timespec="milliseconds"), "type": type_}
        rec.update(data)
        self._events.write(json.dumps(rec, ensure_ascii=False, default=_default) + "\n")
        self._events.flush()

    def add_input(self, path, as_name=None):
        """把執行當下使用的輸入檔 (校正檔、版面檔) 複製進 inputs/,並記錄 sha256。"""
        name = as_name or os.path.basename(path)
        dst = os.path.join(self.path, "inputs", name)
        shutil.copy2(path, dst)
        self.meta["inputs"][name] = {"source": os.path.abspath(path), "sha256": sha256_file(dst)}
        self._write_meta()
        return os.path.join("inputs", name)

    def save_image(self, name, img, ext=".png"):
        rel = os.path.join("images", name if os.path.splitext(name)[1] else name + ext)
        ok, buf = cv2.imencode(os.path.splitext(rel)[1], img)
        if not ok:
            raise ValueError("影像編碼失敗: " + rel)
        buf.tofile(os.path.join(self.path, rel))
        return rel

    def save_snapshot(self, name, snap, cam_info=None, color_ext=".png"):
        """存 rs_capture 的快照:彩色 images/<name><ext>、深度 depth/<name>.png (uint16) + depth/<name>.json。
        snap 也可以只是 BGR 影像 (沒有深度)。回傳彩色的相對路徑。"""
        if isinstance(snap, np.ndarray):
            snap = {"color": snap}
        rel = self.save_image(name, snap["color"], color_ext)
        rec = {"name": name, "color": rel.replace("\\", "/"), "time": snap.get("time")}
        if cam_info and self.meta.get("camera") is None:
            self.meta["camera"] = cam_info
        if snap.get("depth") is not None:
            os.makedirs(os.path.join(self.path, "depth"), exist_ok=True)
            drel = os.path.join("depth", name + ".png")
            ok, buf = cv2.imencode(".png", snap["depth"])
            if not ok:
                raise ValueError("深度編碼失敗: " + drel)
            buf.tofile(os.path.join(self.path, drel))
            info = cam_info or self.meta.get("camera") or {}
            dmeta = {"depth_scale_m": info.get("depth_scale_m"), "aligned_to": "color",
                     "intrinsics": info.get("color_intrinsics"), "n_frames_median": snap.get("n_depth"),
                     "valid_frac": snap.get("depth_valid_frac"), "time": snap.get("time")}
            _dump(os.path.join(self.path, "depth", name + ".json"), dmeta)
            rec.update({"depth": drel.replace("\\", "/"), "depth_valid_frac": snap.get("depth_valid_frac")})
        self.meta["snapshots"].append(rec)
        self._write_meta()
        self.event("snapshot", **rec)
        return rel

    def save_json(self, name, obj):
        rel = os.path.join("data", name if name.endswith(".json") else name + ".json")
        _dump(os.path.join(self.path, rel), obj)
        return rel

    def save_csv(self, name, header, rows):
        rel = os.path.join("data", name if name.endswith(".csv") else name + ".csv")
        with open(os.path.join(self.path, rel), "w", encoding="utf-8-sig", newline="") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + "\n")
        return rel

    def finish(self, status="ok", result=None, notes=None):
        if self._finished:
            return
        self.meta.update({"ended": _now().isoformat(timespec="seconds"), "status": status})
        if result is not None:
            self.meta["result"] = result
        if notes:
            self.meta["notes"] = notes
        self._write_meta()
        self._events.close()
        self._finished = True
        if self._tee:
            sys.stdout, sys.stderr = self._tee
            self._console.close()
            self._tee = None
        if self in _CURRENT:
            _CURRENT.remove(self)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if not self._finished:
            if et is None:
                self.finish("ok")
            elif et is SystemExit:                    # 腳本用 sys.exit("原因") 結束
                code = ev.code if ev is not None else None
                self.event("exit", code=str(code))
                self.finish("ok" if code in (0, None) else "exited", {"exit": str(code)})
            else:
                self.event("error", error=repr(ev))
                self.finish("aborted" if et is KeyboardInterrupt else "failed", {"error": repr(ev)})
        return False


def load_run(path):
    """讀回一次紀錄:回傳 (meta, events)。"""
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    events = []
    p = os.path.join(path, "events.jsonl")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
    return meta, events


def run_main(kind, main, **kw):
    """舊腳本一行接上紀錄:if __name__ == "__main__": run_main("warp_sort", main)
    (-h/--help 不建立紀錄)。移動、夾爪、snapshot() 都會自動寫進這筆紀錄。"""
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        return main()
    with RunLog(kind, params={"argv": sys.argv[1:]}, **kw):
        return main()


def snapshot(name, snap, cam_info=None, color_ext=".png"):
    """存進目前進行中的紀錄 (沒有紀錄就略過)。"""
    lg = current()
    if lg is not None:
        return lg.save_snapshot(name, snap, cam_info, color_ext)


def load_snapshot(path, name):
    """讀回 save_snapshot 存的快照:回傳 (color BGR, depth_mm float32 或 None (0 -> NaN), depth_meta dict 或 None)。"""
    meta, _ = load_run(path)
    rec = next((s for s in meta.get("snapshots", []) if s["name"] == name), None)
    color_rel = rec["color"] if rec else os.path.join("images", name + ".png")
    color = cv2.imdecode(np.fromfile(os.path.join(path, color_rel), np.uint8), cv2.IMREAD_COLOR)
    if not rec or "depth" not in rec:
        return color, None, None
    raw = cv2.imdecode(np.fromfile(os.path.join(path, rec["depth"]), np.uint8), cv2.IMREAD_UNCHANGED)
    with open(os.path.join(path, "depth", name + ".json"), encoding="utf-8") as f:
        dmeta = json.load(f)
    mm = raw.astype(np.float32) * float(dmeta["depth_scale_m"] or 0.001) * 1000.0
    mm[raw == 0] = np.nan
    return color, mm, dmeta


def list_snapshots(path):
    return [s["name"] for s in load_run(path)[0].get("snapshots", [])]


class CalibrationStore:
    """校正結果的版本庫:每次存成新檔,舊版保留;latest.json 永遠是最新版。"""

    def __init__(self, root=CALIB_DIR):
        self.root = root

    def _dir(self, name):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        return d

    def history(self, name):
        d = os.path.join(self.root, name)
        if not os.path.isdir(d):
            return []
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json") and f != "latest.json")

    def save(self, name, data, source_run=None, notes="", created=None, extra=None):
        d = self._dir(name)
        created = created or _now()
        base = os.path.join(d, created.strftime("%Y%m%d-%H%M%S"))
        path, n = base + ".json", 1
        while os.path.exists(path):
            path = f"{base}-{n}.json"
            n += 1
        prev = self.history(name)
        rec = {"schema": SCHEMA, "name": name, "created": created.isoformat(timespec="seconds"),
               "source_run": source_run, "supersedes": os.path.basename(prev[-1]) if prev else None,
               "notes": notes, "data": data}
        if extra:
            rec.update(extra)
        _dump(path, rec)
        shutil.copy2(path, os.path.join(d, "latest.json"))
        return path

    def load_latest(self, name):
        p = os.path.join(self.root, name, "latest.json")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"沒有校正紀錄: {name}")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def import_legacy(self, name, path, notes="由既有檔匯入 (時間取檔案修改時間)"):
        """把既有的單一校正檔 (如 calibration.json) 匯入版本庫;同內容已匯入過就略過。回傳新檔路徑或 None。"""
        digest = sha256_file(path)
        for p in self.history(name):
            with open(p, encoding="utf-8") as f:
                if json.load(f).get("legacy_sha256") == digest:
                    return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path)).astimezone()
        return self.save(name, data, notes=notes, created=mtime,
                         extra={"legacy_source": os.path.abspath(path), "legacy_sha256": digest})
