# -*- coding: utf-8 -*-
"""
RealSense 取像共用模組:彩色 + 對齊到彩色的深度 + 相機資訊,方便存成可離線重算的紀錄。

  cam = RSCamera()                    # 開彩色 + 深度 (深度開不了時自動只用彩色)
  snap = cam.grab(n_depth=15)         # {"color": BGR, "depth": uint16 (對齊彩色,15 張逐像素中位數,0 = 無效), ...}
  info = cam.info()                   # 序號、韌體、內參、深度單位、深度->彩色外參
  cam.close()
  snap, info = grab_once(n_depth=15)  # 開 -> 拍 -> 關

存檔用 run_log.RunLog.save_snapshot(name, snap, info);讀回用 run_log.load_snapshot(run, name)。
"""
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:          # 離線分析的電腦可以沒有 pyrealsense2
    rs = None


def _intr(i):
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
            "model": str(i.model), "coeffs": list(i.coeffs)}


class RSCamera:
    def __init__(self, depth=True, width=1280, height=720, fps=30, warmup=30):
        if rs is None:
            raise RuntimeError("沒有安裝 pyrealsense2")
        self.pipe = rs.pipeline()
        self.depth = depth
        self.profile = None
        if depth:
            try:
                self.profile = self._start(True, width, height, fps)
            except RuntimeError as e:
                if "No device" in str(e):
                    raise
                print("[rs_capture] 深度串流開不了,只用彩色:", e)
                self.depth = False
        if self.profile is None:
            self.profile = self._start(False, width, height, fps)
        self.align = rs.align(rs.stream.color) if self.depth else None
        for _ in range(warmup):                       # 等自動曝光穩定
            self.pipe.wait_for_frames(5000)

    def _start(self, depth, w, h, fps):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if depth:
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        return self.pipe.start(cfg)

    def info(self):
        dev = self.profile.get_device()
        get = lambda k: dev.get_info(k) if dev.supports(k) else None   # noqa: E731
        cs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        out = {"name": get(rs.camera_info.name), "serial": get(rs.camera_info.serial_number),
               "firmware": get(rs.camera_info.firmware_version), "usb": get(rs.camera_info.usb_type_descriptor),
               "color_intrinsics": _intr(cs.get_intrinsics()), "depth": self.depth}
        if self.depth:
            ds = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            ex = ds.get_extrinsics_to(cs)
            out.update({"depth_scale_m": dev.first_depth_sensor().get_depth_scale(),
                        "depth_intrinsics_raw": _intr(ds.get_intrinsics()),
                        "depth_aligned_to": "color",
                        "depth_to_color": {"rotation": list(ex.rotation), "translation_m": list(ex.translation)}})
        return out

    def grab(self, n_depth=1, settle=0.0, flush=0):
        """拍一張彩色 + n_depth 張深度 (逐像素中位數,忽略 0)。"""
        if settle:
            time.sleep(settle)
        for _ in range(flush):
            self.pipe.wait_for_frames(5000)
        color, deps = None, []
        for k in range(max(1, n_depth if self.depth else 1)):
            fr = self.pipe.wait_for_frames(5000)
            if self.align is not None:
                fr = self.align.process(fr)
                d = fr.get_depth_frame()
                if d:
                    deps.append(np.asanyarray(d.get_data()).copy())
            if color is None:
                color = np.asanyarray(fr.get_color_frame().get_data()).copy()
        snap = {"color": color, "time": time.time(), "n_depth": len(deps)}
        if deps:
            D = np.stack(deps).astype(np.float32)
            D[D == 0] = np.nan
            with np.errstate(all="ignore"):
                med = np.nanmedian(D, axis=0) if len(deps) > 1 else D[0]
                valid = np.mean(~np.isnan(D), axis=0)
            snap["depth"] = np.nan_to_num(med, nan=0).round().astype(np.uint16)
            snap["depth_valid_frac"] = float(np.mean(valid > 0.5))
        return snap

    def close(self):
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


def grab_once(depth=True, n_depth=15, warmup=30):
    cam = RSCamera(depth=depth, warmup=warmup)
    try:
        return cam.grab(n_depth=n_depth), cam.info()
    finally:
        cam.close()
