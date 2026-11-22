# -*- coding: utf-8 -*-
"""
執行前檢查 (數位分身,不用開 Isaac、不動手臂):可達性、關節餘裕、手腕奇異點、工具和工件 / 紙面的距離。

資料在 twin_model.json (由 Desktop\\Delta_Arm_IsaacSim 的 urdf\\drv70l_twin.urdf 匯出,見檔內 source):
  - 運動鏈:台達型錄尺寸 + 實機驗證過的關節方向
  - 手眼:相機在法蘭座標的位置 (3 次實測),給 pen_tip_twin 推筆尖
  - 工具外形:分身原本的夾指太細 (寬 20 x 厚 8 mm、筆半徑 6 mm),11-22 依實機照片改成保守的圓柱包絡。
    是照片估的 (±5 mm),所以檢查另外留餘裕,第一次傾斜一定先「空走一圈」。

11-22 用 runs/ 的實機紀錄回測 IK:4 次姿態造成的拒絕 / 卡住全部被判不可行;
580 個實機走得到的目標有 14 個被誤判不可行 (偏保守,會多擋,不是少擋)。
限制:不知道周圍其他東西 (輸送帶、鋁架、線材);控制器 MovL 假設是位置直線 + 姿態球面內插。
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_JOINT_MARGIN = 8.0      # 度
MIN_WRIST = 10.0            # |J5| 小於這個 = 靠近手腕奇異點
MAX_JOINT_STEP = 12.0       # 相鄰取樣點 (3 mm / 2 度) 關節跳這麼多 = 換解 / 奇異


def R_of(ori):
    return Rotation.from_euler("xyz", ori, degrees=True).as_matrix()


def ori_of(R, near=None):
    a = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    if near is not None:
        a = a + 360.0 * np.round((np.asarray(near) - a) / 360.0)
    return [float(v) for v in a]


class Twin:
    def __init__(self, path=os.path.join(HERE, "twin_model.json")):
        with open(path, encoding="utf-8") as f:
            self.model = json.load(f)
        self.chain = [(np.array(j["xyz"]), np.array(j["R"]),
                       np.array(j["axis"]) if j["axis"] is not None else None) for j in self.model["chain"]]
        self.lo, self.hi = np.array(self.model["limits_deg"]).T

    def fk(self, q, jac=False):
        """關節 (度) -> 法蘭位置 (mm)、姿態;jac=True 另外回傳 6x6 Jacobian (弧度)。"""
        R, p, axes, pts, k = np.eye(3), np.zeros(3), [], [], 0
        rad = np.radians(q)
        for xyz, Rj, axis in self.chain:
            p = p + R @ xyz
            R = R @ Rj
            if axis is not None:
                axes.append(R @ axis)
                pts.append(p)
                a, b, c = axis
                S = np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])
                R = R @ (np.eye(3) + np.sin(rad[k]) * S + (1 - np.cos(rad[k])) * S @ S)
                k += 1
        if not jac:
            return p, R
        z = np.array(axes)
        return p, R, np.vstack([np.cross(z, p - np.array(pts)).T, z.T])

    def ik(self, T, Rm, seed, iters=60):
        """回傳 (q 度, 是否解得到)。位置以公尺計權重,和姿態 (弧度) 同一量級。"""
        q = np.array(seed, float)
        w = np.array([1e-3] * 3 + [1.0] * 3)
        for _ in range(iters):
            p, R, J = self.fk(q, jac=True)
            e = np.r_[np.asarray(T) - p, 0.5 * np.sum(np.cross(R.T, np.asarray(Rm).T), axis=0)]
            if np.max(np.abs(e[:3])) < 0.02 and np.max(np.abs(e[3:])) < 2e-4:
                return q, True
            Jw, ew = J * w[:, None], e * w
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-5 * np.eye(6), ew)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + np.degrees(dq), self.lo, self.hi)
        return q, False

    def home_q(self, home):
        q, ok = self.ik(home[:3], R_of(home[3:]), self.model["home_q_deg"], iters=200)
        if not ok:
            raise ValueError("分身解不出 home 姿態 (home 或型號和 twin_model.json 不符)")
        return q

    def tool_points(self, p_tip):
        """工具外形取樣點 (法蘭座標),依這次的筆尖位置擺放。回傳 (點, 是否屬於筆尖段)。"""
        pts, nib = [], []
        px, py, pz = p_tip
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        for s in self.model["tool_envelope"]["along_pen"]:          # from / to:從筆尖往法蘭 mm
            hs = np.linspace(s["from"], s["to"], max(2, int(np.ceil((s["to"] - s["from"]) / 4.0)) + 1))
            for h in hs:
                for a in ang:
                    pts.append([px + s["r"] * np.cos(a), py + s["r"] * np.sin(a), pz - h])
                    nib.append(bool(s.get("nib")))
        for b in self.model["tool_envelope"]["boxes"]:               # 相機等:法蘭座標的方塊
            g = np.stack(np.meshgrid(*[np.linspace(-v / 2, v / 2, 5) for v in b["size"]], indexing="ij"), -1).reshape(-1, 3)
            for q in g @ np.array(b["R"]).T + np.array(b["center"]):
                pts.append(q)
                nib.append(False)
        return np.array(pts), np.array(nib)

    def follow(self, poses, seed):
        """照 MovL 走過 poses (各 [x, y, z, rx, ry, rz]),每 3 mm / 2 度取樣。
        回傳 dict(ok, reason, margin, seg, q, qs=各端點的關節, samples=[(段, T, R)])。"""
        poses = np.asarray(poses, float)
        rots = Rotation.from_euler("xyz", poses[:, 3:], degrees=True)
        q, ok = self.ik(poses[0, :3], rots[0].as_matrix(), seed, iters=200)
        res = {"ok": True, "reason": "", "margin": 999.0, "seg": 0, "q": q, "qs": [q], "samples": []}
        if not ok:
            res.update(ok=False, reason="起點到不了")
            return res
        for i in range(1, len(poses)):
            ang = np.degrees((rots[i - 1].inv() * rots[i]).magnitude())
            n = int(max(1, np.ceil(np.linalg.norm(poses[i, :3] - poses[i - 1, :3]) / 3.0), np.ceil(ang / 2.0)))
            sl = Slerp([0, 1], Rotation.from_matrix([rots[i - 1].as_matrix(), rots[i].as_matrix()]))
            f = np.linspace(0, 1, n + 1)[1:]
            for s, Rm in zip(f, sl(f).as_matrix()):
                T = poses[i - 1, :3] * (1 - s) + poses[i, :3] * s
                qn, ok = self.ik(T, Rm, q)
                m = float(np.min(np.minimum(qn - self.lo, self.hi - qn)))
                why = ""
                if not ok:
                    why = "手臂到不了 (可達範圍 / 關節極限)"
                elif np.max(np.abs(qn - q)) > MAX_JOINT_STEP:
                    why = "關節突然大轉 (換解 / 奇異點)"
                elif abs(qn[4]) < MIN_WRIST:
                    why = "靠近手腕奇異點"
                # home 本身 J5 餘裕只有 7.6 度:往餘裕變大的方向離開、或最後一段回 home 都是實機常走的
                elif m < MIN_JOINT_MARGIN and m < min(self.home_margin, float(np.min(np.minimum(q - self.lo, self.hi - q)))) - 0.05:
                    why = f"關節餘裕只剩 {m:.1f} 度"
                if why:
                    res.update(ok=False, reason=why, seg=i, q=qn)
                    return res
                res["margin"] = min(res["margin"], m)
                res["samples"].append((i, T, Rm))
                q = qn
            res["qs"].append(q)
        res["q"] = q
        return res

    @property
    def home_margin(self):
        q = np.array(self.model["home_q_deg"])
        return float(np.min(np.minimum(q - self.lo, self.hi - q)))
