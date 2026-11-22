# -*- coding: utf-8 -*-
"""
傾斜描繪的路徑與執行前檢查 (純計算,不動手臂、不碰相機)。只需要護貝紙筆尖校正 + twin_model.json。

座標:
  護貝紙校正給「筆垂直時,筆尖在紙面 (x, y)、高 h -> 法蘭 T0 = (A (x, y) + t, z_touch + h)」。
  筆尖在手臂座標 X = T0 + R0 p (p = 筆尖在法蘭座標,pen_tip_twin 由 home 照片推出)。
  任何姿態 R 要讓筆尖到 X:T = X - R p。

動作 (和 11-18 差點壓壞筆的做法不同):
  - 換姿態是「繞筆尖轉」:TCP 一邊移一邊轉,筆尖留在原地 (高出工件 CLEAR mm)。
    繞 TCP 原地轉時筆尖會掃 6~9 cm、往下掉最多約 38 mm,而且很多點抬不到夠高 (分身算過)。
  - 進入:垂直到起點正上方 -> 繞筆尖轉成傾斜 -> 垂直下降。離開 / Ctrl+C:維持姿態垂直抬起 -> 繞筆尖轉回垂直 -> 回 home。
  - 圓上「每一點」的離開路徑都先算過可行,才准開始。
"""
import numpy as np
from scipy.spatial.transform import Rotation

import twin_check

CLEAR = 15.0            # 換姿態時筆尖高出工件頂面 mm
AIR_LIFT = 20.0         # 空走一圈時比描繪高 mm
MARGIN_WALL = 2.0       # 工具外形和工件的基本餘裕 mm,再加上筆尖位置的不確定度
MAX_OFFSET = 30.0       # 底面模式離罐壁最遠找到幾 mm
TILT_RANGE = (5.0, 25.0)


class PlanError(ValueError):
    pass


class TiltPlan:
    def __init__(self, cal, home, tip, center, r_obj, h_obj, tilt_deg, twin=None, inward=False):
        """cal:pen_paper 的 data;tip:pen_tip_twin.estimate() 的結果;center、r_obj 紙面 mm;h_obj 工件高 mm。"""
        if not TILT_RANGE[0] <= tilt_deg <= TILT_RANGE[1]:
            raise PlanError(f"傾角只開放 {TILT_RANGE[0]:g}~{TILT_RANGE[1]:g} 度 (分身:30 度以上圓上有些點到不了)")
        self.A, self.t, self.z_touch = np.array(cal["A"], float), np.array(cal["t"], float), float(cal["z_touch"])
        self.home = [float(v) for v in home]
        self.R0 = twin_check.R_of(self.home[3:])
        self.p = np.asarray(tip["p"], float)
        self.tilt, self.inward = float(tilt_deg), inward
        self.err = float(0.6 + tip["z_unc"] * np.sin(np.radians(tilt_deg)))      # 筆尖側向不確定度
        self.center, self.r_obj, self.h_obj = np.asarray(center, float), float(r_obj), float(h_obj)
        self.h_clear = self.h_obj + CLEAR
        self.twin = twin or twin_check.Twin()
        self.q_home = self.twin.home_q(self.home)
        self.tool, self.nib = self.twin.tool_points(self.p)
        self.center_w = self.A @ self.center + self.t + (self.R0 @ self.p)[:2]
        self.paper_z = self.z_touch + (self.R0 @ self.p)[2]
        self.scale = float(np.mean(np.linalg.svd(self.A, compute_uv=False)))    # 手臂 mm / 紙面 mm
        self.target, self.path, self.X = None, None, None

    # ---------- 幾何 ----------
    def R_at(self, xy):
        u = self.A @ (np.asarray(xy) - self.center)
        u = np.r_[u / np.linalg.norm(u), 0.0]
        axis = np.cross([0.0, 0.0, 1.0], u)
        best = None
        for s in (1.0, -1.0):
            R = Rotation.from_rotvec(s * np.radians(self.tilt) * axis).as_matrix() @ self.R0
            score = -float(R[:, 2] @ u)          # 法蘭 z = 往筆尖;朝圓心 = 筆身在外
            score = -score if self.inward else score
            if best is None or score > best[0]:
                best = (score, R)
        return best[1]

    def tip_world(self, pose):
        return np.asarray(pose[:3]) + twin_check.R_of(pose[3:]) @ self.p

    def tip_height(self, pose):
        return float(self.tip_world(pose)[2] - self.paper_z)

    def pose(self, X, R):
        return [float(v) for v in np.asarray(X) - R @ self.p] + twin_check.ori_of(R, near=self.home[3:])

    def vertical(self, X):
        return [float(v) for v in np.asarray(X) - self.R0 @ self.p] + self.home[3:]

    def set_path(self, path_xyh, o_pen, target):
        """path_xyh:紙面幾何目標;o_pen:筆跡回饋學到的偏移 (校正座標先扣掉,碰撞用幾何位置算)。"""
        self.target, self.path = target, np.asarray(path_xyh, float)
        q = self.path[:, :2] - np.asarray(o_pen, float)
        self.X = np.c_[q @ self.A.T + self.t, self.z_touch + self.path[:, 2]] + self.R0 @ self.p
        self.Rs = [self.R_at(p[:2]) for p in self.path]
        self.o_w = self.A @ np.asarray(o_pen, float)          # 命令位置 -> 實際位置
        self.draw = [self.pose(X, R) for X, R in zip(self.X, self.Rs)]
        return self

    def shifted(self, shift_xyz):
        """人工微調 / 接觸高度套用到整圈。"""
        d = np.asarray(shift_xyz, float)
        return [[float(v) for v in np.array(p[:3]) + d] + p[3:] for p in self.draw]

    # ---------- 動作 ----------
    def raised(self, pose, h=None):
        dz = max(0.0, (self.h_clear if h is None else h) - self.tip_height(pose))
        return [pose[0], pose[1], pose[2] + dz] + list(pose[3:])

    def escape(self, pose):
        """任何姿態 -> 安全回 home 的路點:垂直抬起、繞筆尖轉回垂直、回 home。"""
        up = self.raised(pose)
        return [up, self.vertical(self.tip_world(up)), list(self.home)]

    def entry(self, draw=None):
        """垂直起點上方、傾斜起點上方。"""
        up = self.raised((draw or self.draw)[0])
        return self.vertical(self.tip_world(up)), up

    # ---------- 檢查 ----------
    def hit(self, samples):
        """工具和紙面 / 罐頭的干涉。回傳 None 或 (段, 原因)。"""
        if not samples:
            return None
        Rs = np.array([s[2] for s in samples])
        X = np.einsum("nij,kj->nki", Rs, self.tool) + np.array([s[1] for s in samples])[:, None, :]
        X[..., :2] += self.o_w
        h = X[..., 2] - self.paper_z
        body = ~self.nib
        low = (h[:, body] < 2.0).any(1)
        if low.any():
            return samples[int(np.argmax(low))][0], "工具會碰到紙面 / 桌面"
        rad = np.hypot(X[..., 0] - self.center_w[0], X[..., 1] - self.center_w[1])
        m = MARGIN_WALL + self.err
        bad = (rad < self.r_obj * self.scale + m) & (h < self.h_obj + MARGIN_WALL)
        if self.target == "rim":
            bad = bad & body                                    # 頂面模式:筆尖本來就要碰捲邊
        bad = bad.any(1)
        if bad.any():
            return samples[int(np.argmax(bad))][0], "工具離罐頭太近"
        return None

    def check_moves(self, poses, seed):
        r = self.twin.follow(poses, seed)
        if not r["ok"]:
            return r
        hit = self.hit(r["samples"])
        if hit:
            r.update(ok=False, seg=hit[0], reason=hit[1])
        return r

    def allowed(self, pose):
        """人工微調的單一位置可不可以去。"""
        R = twin_check.R_of(pose[3:])
        _, ok = self.twin.ik(pose[:3], R, self.q_start)
        return ok and self.hit([(0, np.asarray(pose[:3]), R)]) is None

    def verify(self, draw=None, start=None):
        """整套動作都檢查:進入、空走、描繪、回起點,以及每一點中止時回 home。失敗丟 PlanError。"""
        draw = draw or self.draw
        above, up = self.entry(draw)
        air = [[p[0], p[1], p[2] + AIR_LIFT] + p[3:] for p in draw]
        seq = [list(start or self.home), above, up] + air + [self.raised(air[-1]), up, draw[0]] + draw[1:]
        r = self.check_moves(seq, self.q_home)
        if not r["ok"]:
            raise PlanError(f"執行前檢查沒過 (第 {r['seg']}/{len(seq) - 1} 段):{r['reason']}")
        margin = r["margin"]
        qs = r["qs"][-len(draw):]
        for k, (p, q) in enumerate(zip(draw, qs)):
            e = self.check_moves([p] + self.escape(p), q)
            if not e["ok"]:
                raise PlanError(f"第 {k} 點如果中止會回不了 home:{e['reason']}")
        self.q_start = qs[0]
        return {"ok": True, "margin_deg": margin, "n_samples": len(r["samples"]), "tip_err_mm": self.err}


def bottom_offset(plan, make_path, o_pen):
    """底面模式:工具不碰罐壁的最小距離 (筆尖中心到罐壁 mm,每 0.5 mm 找)。"""
    for off in np.arange(3.0, MAX_OFFSET + 0.1, 0.5):
        plan.set_path(make_path(plan.r_obj + off), o_pen, "bottom")
        samples = [(i, np.array(p[:3]), twin_check.R_of(p[3:])) for i, p in enumerate(plan.draw)]
        if plan.hit(samples) is None:
            return float(off + 0.5)                  # 再留半格:點和點之間是弦,會往內一點
    raise PlanError(f"離罐壁 {MAX_OFFSET:g} mm 內找不到不碰撞的底面路徑")
