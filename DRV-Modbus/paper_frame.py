# -*- coding: utf-8 -*-
"""
A4 校正紙的座標系 (純影像運算,不連線手臂/相機)。

用紙角的 4 個 AprilTag (16 個角點) 求「影像像素 <-> 紙面 mm」的單應矩陣 H0 (紙面高度 0)。
物體若高出紙面 h mm (例如珍珠板厚度),影像會往外偏移 (視差);這裡用相機中心位置做修正:
    q_h = C_xy + (q_0 - C_xy) * (1 - h / H_cam)
    q_0 : 用 H0 把像素換算到紙面高度 0 的位置      C_xy, H_cam : 相機中心在紙面座標的位置與離紙高度 (由 solvePnP 估計)
H0 由 16 個角點直接擬合 (不依賴內參);PnP 只用來估相機中心 (影響的只是視差修正量)。

紙面座標:原點在紙的左上角 (人正看紙面),x 向右、y 向下,單位 mm。
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAYOUT = os.path.join(HERE, "print", "a4_layout.json")
# D435i 彩色內參 @1280x720 (2026-10-13 量測)
DEFAULT_K = np.array([[911.49, 0.0, 638.875], [0.0, 911.71, 364.43], [0.0, 0.0, 1.0]])
FAMILIES = {"36h11": cv2.aruco.DICT_APRILTAG_36h11, "25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "16h5": cv2.aruco.DICT_APRILTAG_16h5}


class PaperFrame:
    def __init__(self, layout=DEFAULT_LAYOUT, K=DEFAULT_K, dist=None):
        self.layout_path = layout
        with open(layout, encoding="utf-8") as f:
            self.layout = json.load(f)
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.zeros(5) if dist is None else np.array(dist, dtype=np.float64)
        self.dict = cv2.aruco.Dictionary_get(FAMILIES[self.layout["family"]])
        self.params = cv2.aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.W, self.H = self.layout["paper_mm"]
        self.ok = False
        self.missing = []
        self.found = {}

    # ---------- detection ----------
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dict, parameters=self.params)
        self.found = {int(i[0]): c[0] for c, i in zip(corners, ids)} if ids is not None else {}
        need = [t["id"] for t in self.layout["tags"]]
        self.missing = [i for i in need if i not in self.found]
        self.ok = False
        if self.missing:
            return False
        img_pts = np.float32([p for t in self.layout["tags"] for p in self.found[t["id"]].tolist()])
        obj_pts = np.float32([p for t in self.layout["tags"] for p in t["corners_mm"]])
        self.H0, _ = cv2.findHomography(img_pts, obj_pts, 0)
        self.H0inv = np.linalg.inv(self.H0)
        proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), self.H0).reshape(-1, 2)
        self.rms_mm = float(np.sqrt(np.mean(np.sum((proj - obj_pts) ** 2, axis=1))))

        obj3 = np.hstack([obj_pts, np.zeros((len(obj_pts), 1), np.float32)])
        ok, rvec, tvec = cv2.solvePnP(obj3, img_pts, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        C = (-R.T @ tvec).ravel()                       # 相機中心 (紙面座標,z 軸指向桌面,所以在紙上方 z<0)
        self.cam_xy = C[:2]
        self.cam_h = float(-C[2])                       # 離紙高度 (mm)
        axis = R.T @ np.array([0.0, 0.0, 1.0])          # 光軸方向 (紙面座標)
        self.tilt_deg = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0, 1))))
        self.ok = bool(ok and self.cam_h > 50)
        return self.ok

    def describe(self):
        if not self.ok:
            return f"未偵測到完整 4 個 tag (缺 {self.missing},看到 {sorted(self.found)})"
        return (f"tag 4/4,擬合殘差 RMS {self.rms_mm:.3f} mm,相機離紙 {self.cam_h:.0f} mm,"
                f"光軸偏離紙面法線 {self.tilt_deg:.1f}°")

    # ---------- mapping ----------
    def _scale(self, h):
        return 1.0 - float(h) / self.cam_h

    def pix_to_paper(self, uv, h=0.0):
        """像素 (N,2) -> 紙面 mm (N,2);h = 該點高出紙面的高度 (mm)。"""
        uv = np.float32(uv).reshape(-1, 1, 2)
        q0 = cv2.perspectiveTransform(uv, self.H0).reshape(-1, 2)
        return self.cam_xy + (q0 - self.cam_xy) * self._scale(h)

    def paper_to_pix(self, xy, h=0.0):
        """紙面 mm (N,2) -> 像素 (N,2)。"""
        xy = np.float64(xy).reshape(-1, 2)
        q0 = self.cam_xy + (xy - self.cam_xy) / self._scale(h)
        return cv2.perspectiveTransform(np.float32(q0).reshape(-1, 1, 2), self.H0inv).reshape(-1, 2)

    def mm_per_pixel(self):
        """紙面中心附近的解析度 (mm/px)。"""
        c = np.float32([[self.W / 2, self.H / 2], [self.W / 2 + 10, self.H / 2]])
        p = self.paper_to_pix(c)
        return 10.0 / float(np.linalg.norm(p[1] - p[0]))
