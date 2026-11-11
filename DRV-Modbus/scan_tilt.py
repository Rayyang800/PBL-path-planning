# -*- coding: utf-8 -*-
"""
傾斜視角掃描的幾何 (給 scan3d.py 用;純數學,可離線測試)。

座標系
  B : 手臂基座 (mm,z 朝上)。TCP 姿態 (rx, ry, rz) 度 -> R_B_tool (歐拉角慣例由「試轉」實測決定)
  P : 校正紙 (mm,x 右、y 下、z 指向桌面;高度 h = -z)。PnP:X_cam = R X_P + t,相機中心 C_P = -R^T t
  P -> B 的方向:由筆尖校正 A (紙面 -> 手臂 XY,2x2) 的極分解得到旋轉 U 與比例 s;紙面 z 朝下 -> 手臂 -z

手眼 (相機在工具座標的位置 p 與方向 R_tool_cam)
  在固定 TCP 位置做幾個小角度試轉,每個姿態用 tag 量相機位姿:
    C_B(i) - C_B(0) = (TCP(i) - TCP(0)) + (R_i - R_0) p        (差值:未知的絕對偏移都消掉)
    R_B_cam(i) = R_B_tool(i) · R_tool_cam
  先用旋轉判斷歐拉角慣例 (哪個慣例預測的相機轉動最接近實測),再用最小平方解 p。

筆尖安全
  TCP 繞自己轉時,筆尖 (在 TCP 下方 d) 只會變高:tip_z = TCP_z - d·cos(tilt) >= TCP_z - d,
  所以只要 TCP_z - z_touch >= 物體高 + 淨空,任何傾斜下筆尖都比物體高 (z_touch = 筆尖碰紙時的 TCP z)。
"""
import numpy as np
from scipy.spatial.transform import Rotation

CONVENTIONS = ("xyz", "XYZ")      # 外旋 xyz (= 內旋 ZYX,R = Rz·Ry·Rx) / 內旋 XYZ (R = Rx·Ry·Rz)


def R_tool(angles_deg, conv):
    return Rotation.from_euler(conv, angles_deg, degrees=True).as_matrix()


def angles_of(R, conv, near=None):
    """旋轉矩陣 -> 歐拉角 (度)。near:參考角度 (例如 home),每個角度加減 360 讓它最接近參考,
    避免 rx 從 -179 跳到 +178 (控制器若直接對角度做線性內插,會讓手腕轉一大圈)。"""
    a = [float(v) for v in Rotation.from_matrix(R).as_euler(conv, degrees=True)]
    if near is not None:
        a = [v + 360.0 * round((n - v) / 360.0) for v, n in zip(a, near)]
    return a


def paper_to_base(A):
    """A (2x2) -> (R_PB 3x3 旋轉, 比例 s)。"""
    u, sv, vt = np.linalg.svd(np.asarray(A, float))
    U = u @ vt
    R = np.eye(3)
    R[:2, :2] = U
    R[2, 2] = -1.0
    if np.linalg.det(R) < 0:                  # A 若沒有鏡射 (不該發生),退回只翻 z 會變成反射:改翻 y
        R[1, :] *= -1
    return R, float(np.mean(sv))


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def tilt_base(axis_B, deg):
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis_B, float) / np.linalg.norm(axis_B)).as_matrix()


def solve_handeye(probes, R_PB, s):
    """probes: [{tcp:[x,y,z,rx,ry,rz], R:(PnP 3x3), C:(紙面相機中心)}],第 0 個是基準 (未傾斜)。
    回傳 dict(conv, R_tool_cam, p, rot_err_deg (各慣例), pos_rms_mm) 或丟 ValueError。"""
    if len(probes) < 3:
        raise ValueError("試轉至少要 3 個姿態 (基準 + 2 個不同軸)")
    Rcam = [R_PB @ np.asarray(p["R"]).T for p in probes]           # R_B_cam
    errs = {}
    for conv in CONVENTIONS:
        Rt = [R_tool(p["tcp"][3:], conv) for p in probes]
        Rtc = Rt[0].T @ Rcam[0]
        e = [rot_angle_deg(Rt[i] @ Rtc, Rcam[i]) for i in range(1, len(probes))]
        errs[conv] = float(np.max(e))
    conv = min(errs, key=errs.get)
    Rt = [R_tool(p["tcp"][3:], conv) for p in probes]
    Rtc_all = [Rt[i].T @ Rcam[i] for i in range(len(probes))]
    Rtc = Rotation.from_matrix(np.array(Rtc_all)).mean().as_matrix()
    M, b = [], []
    C0 = s * (R_PB @ np.asarray(probes[0]["C"], float))
    T0 = np.asarray(probes[0]["tcp"][:3], float)
    for i in range(1, len(probes)):
        Ci = s * (R_PB @ np.asarray(probes[i]["C"], float))
        Ti = np.asarray(probes[i]["tcp"][:3], float)
        M.append(Rt[i] - Rt[0])
        b.append((Ci - C0) - (Ti - T0))
    M, b = np.vstack(M), np.concatenate(b)
    sv = np.linalg.svd(M, compute_uv=False)
    if sv[-1] < 0.08:
        # 11-11 實測:只剩繞同一軸的試轉 (+x / -x),沿轉軸方向的分量解不出來,最小平方給出 8 萬 mm 的荒謬值
        raise ValueError(f"試轉的轉軸太相近,相機位置有一個方向解不出來 (奇異值 {np.round(sv, 3).tolist()});需要繞兩個不同軸的試轉")
    p, *_ = np.linalg.lstsq(M, b, rcond=None)
    if np.linalg.norm(p) > 400:
        raise ValueError(f"算出的相機位置離 TCP {np.linalg.norm(p):.0f} mm,不合理")
    res = M @ p - b
    return {"conv": conv, "R_tool_cam": Rtc, "p": p, "rot_err_deg": errs, "pos_rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "singular_values": sv.tolist()}


class Kinematics:
    """手眼 + 紙面/手臂關係 -> 由 TCP 推相機位姿、由想要的相機位姿反推 TCP。"""

    def __init__(self, he, R_PB, s, tcp_ref, C_P_ref):
        self.conv, self.Rtc, self.p = he["conv"], he["R_tool_cam"], np.asarray(he["p"], float)
        self.R_PB, self.s = R_PB, s
        self.tcp_ref = np.asarray(tcp_ref, float)
        self.C_P_ref = np.asarray(C_P_ref, float)
        self.R_ref = R_tool(self.tcp_ref[3:], self.conv)

    def camera_pose(self, tcp):
        """TCP -> PnP 形式的相機位姿 dict(R, t, C) (紙面座標)。"""
        tcp = np.asarray(tcp, float)
        Rt = R_tool(tcp[3:], self.conv)
        dB = (tcp[:3] - self.tcp_ref[:3]) + (Rt - self.R_ref) @ self.p
        C = self.C_P_ref + self.R_PB.T @ dB / self.s
        R_P_cam = self.R_PB.T @ (Rt @ self.Rtc)
        R = R_P_cam.T
        return {"R": R, "t": -R @ C, "C": C}

    def tcp_for(self, C_P, axis_P, roll_deg=0.0):
        """想要相機中心在紙面 C_P、光軸朝紙面方向 axis_P -> (TCP [x,y,z,rx,ry,rz], 傾斜角)。
        roll_deg:先把工具繞自己的 z 軸轉 (手腕轉),相機會繞到 TCP 的另一側;再用最小旋轉把光軸轉到 axis_P。
        相機離 TCP 約 150 mm,斜看時不轉手腕的話 TCP 要跑到很外面;轉手腕讓相機在靠物體那側,TCP 就不用跑那麼遠。"""
        R_base = self.R_ref @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
        a_ref = (R_base @ self.Rtc)[:, 2]
        a_new = self.R_PB @ (np.asarray(axis_P, float) / np.linalg.norm(axis_P))
        v = np.cross(a_ref, a_new)
        sn, cs = np.linalg.norm(v), float(np.dot(a_ref, a_new))
        Ralign = np.eye(3) if sn < 1e-9 else Rotation.from_rotvec(v / sn * np.arctan2(sn, cs)).as_matrix()
        Rt = Ralign @ R_base
        dB = self.s * (self.R_PB @ (np.asarray(C_P, float) - self.C_P_ref))
        T = self.tcp_ref[:3] + dB - (Rt - self.R_ref) @ self.p
        near = list(self.tcp_ref[3:])
        ang = angles_of(Rt, self.conv, near=near)
        tilt = float(np.degrees(np.arccos(np.clip(abs((self.R_ref @ self.Rtc)[:, 2] @ a_new), -1, 1))))
        return [*map(float, T), *ang], tilt


ROLLS = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)


def plan_tilted(kin, obj_center, h_obj, tilts, n_per, dist, z_touch, clearance, tcp_home, max_travel,
                max_dist=480.0, min_range=250.0, az0=0.0, region=None, rolls=ROLLS, ref_xy=None):
    """相機在物體周圍、往物體中心看。tilts = 光軸偏離垂直的角度 (度)。
    每個視角:距離從 dist 開始,試幾個手腕轉角 (rolls),取「安全且 TCP 離 home 最近」的;都不行就把相機推遠,
    超過 max_dist 就略過。region(tcp) -> bool:額外的工作範圍限制 (例如筆尖不能離開白板上方)。
    ref_xy:max_travel 從哪裡量 (手臂 XY);預設 home。斜看時 TCP 一定在物體外側,用物體中心量比較合理。回傳 (views, notes)。"""
    ref = np.asarray(ref_xy if ref_xy is not None else tcp_home[:2], float)
    O = np.array([obj_center[0], obj_center[1], -0.5 * h_obj])        # 紙面座標 (z 朝下)
    views, notes = [], []
    prev_roll = 0.0
    for ti, tilt in enumerate(tilts):
        e = np.radians(90.0 - tilt)
        for k in range(n_per):
            phi = np.radians(az0 + 360.0 * k / n_per + (180.0 / n_per if ti % 2 else 0.0))
            v = np.array([np.cos(e) * np.cos(phi), np.cos(e) * np.sin(phi), -np.sin(e)])    # 物體 -> 相機
            name = f"t{int(round(tilt))}_{k}"
            D = max(dist, min_range + 0.5 * h_obj)
            best, why = None, ""
            while D <= max_dist and best is None:
                C = O + D * v
                for roll in rolls:
                    tcp, tilt_real = kin.tcp_for(C, -v, roll)
                    pen_h = tcp[2] - z_touch
                    if pen_h < h_obj + clearance or -C[2] < h_obj + clearance + 30:
                        why = "筆尖 / 相機淨空不夠"
                        continue
                    d_home = float(np.hypot(tcp[0] - ref[0], tcp[1] - ref[1]))
                    if d_home > max_travel:
                        why = f"TCP 離物體中心太遠 ({d_home:.0f} mm)"
                        continue
                    if region is not None and not region(tcp):
                        why = "超出工作範圍"
                        continue
                    score = d_home + 0.5 * abs(roll) + 0.8 * abs(roll - prev_roll)   # 相鄰視角手腕別轉太多
                    if best is None or score < best[0]:
                        best = (score, tcp, tilt_real, roll, pen_h, d_home)
                if best is None:
                    D += 10.0
            if best is None:
                notes.append(f"{name}: 略過 ({why})")
                continue
            _, tcp, tilt_real, roll, pen_h, d_home = best
            prev_roll = roll
            views.append({"name": name, "tcp": tcp, "cam_P": (O + D * v).tolist(), "dist": float(D), "tilt": float(tilt_real),
                          "roll": float(roll), "pen_tip_above_paper": float(pen_h), "tcp_from_home": d_home})
    return order_rings(views), notes


def order_rings(views):
    """每個傾斜角的一圈:選起點與方向 (循環平移 / 反向),讓手腕轉角 (roll) 變化總量最小、和上一圈的結尾銜接。
    手腕轉角大致跟著方位角走,一圈走完需要轉 360°,超過允許範圍;所以一圈要從某處斷開,單調地從 -90 轉到 +90。"""
    out, prev = [], 0.0
    rings = {}
    for v in views:
        rings.setdefault(v["name"].split("_")[0], []).append(v)
    for key in rings:
        L = rings[key]
        best = None
        for rev in (False, True):
            base = L[::-1] if rev else L
            for k in range(len(base)):
                seq = base[k:] + base[:k]
                r = [prev] + [v["roll"] for v in seq]
                cost = sum(abs(r[i + 1] - r[i]) for i in range(len(r) - 1))
                if best is None or cost < best[0]:
                    best = (cost, seq)
        out += best[1]
        prev = best[1][-1]["roll"]
    return out


def refine_handeye_abs(records, R_PB, s, conv, min_sv=0.3):
    """用「所有看得到 tag 的視角」(含只有 2 個 tag 的、傾斜 + 手腕轉的) 一起解手眼:
        s·R_PB·C_P(i) - TCP(i) = R_i · p + c0        (p = 相機在工具座標的位置,c0 = 紙面原點相關的固定偏移)
    手腕轉 ±90° 的視角槓桿大,比原地小角度試轉準很多 (11-11:試轉解出的 p 讓傾斜視角推算差 20~40 mm)。
    records: [{tcp, R (PnP), C (紙面)}]。回傳 dict(p, c0, Rtc, rms_mm, n, sv, used) 或 None (轉動不夠多樣)。"""
    if len(records) < 4:
        return None
    Rt = [R_tool(r["tcp"][3:], conv) for r in records]
    Y = [s * (R_PB @ np.asarray(r["C"], float)) - np.asarray(r["tcp"][:3], float) for r in records]
    keep = np.ones(len(records), bool)
    for _ in range(4):
        M = np.vstack([np.hstack([Rt[i], np.eye(3)]) for i in range(len(records)) if keep[i]])
        y = np.concatenate([Y[i] for i in range(len(records)) if keep[i]])
        sv = np.linalg.svd(M, compute_uv=False)
        if sv[-1] < min_sv:
            return None
        x, *_ = np.linalg.lstsq(M, y, rcond=None)
        res = np.array([np.linalg.norm(Rt[i] @ x[:3] + x[3:] - Y[i]) for i in range(len(records))])
        thr = max(4.0, 3.0 * np.median(res[keep]))
        new = res < thr
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    Rcam = [R_PB @ np.asarray(r["R"]).T for r in records]
    Rtc = Rotation.from_matrix(np.array([Rt[i].T @ Rcam[i] for i in range(len(records)) if keep[i]])).mean().as_matrix()
    return {"p": x[:3], "c0": x[3:], "Rtc": Rtc, "rms_mm": float(np.sqrt(np.mean(res[keep] ** 2))), "n": int(keep.sum()),
            "sv": sv.tolist(), "used": keep.tolist(), "res_mm": res.tolist()}


def predict_abs(he, tcp, R_PB, s, conv):
    """refine_handeye_abs 的結果 + TCP -> PnP 形式的相機位姿 dict(R, t, C) (紙面座標)。"""
    tcp = np.asarray(tcp, float)
    Rt = R_tool(tcp[3:], conv)
    C = R_PB.T @ (tcp[:3] + Rt @ he["p"] + he["c0"]) / s
    R = (R_PB.T @ (Rt @ he["Rtc"])).T
    return {"R": R, "t": -R @ C, "C": C}


def refine_full(records, R_PB0, s0, conv, rot_w=5.0):
    """完整手眼 + 紙面/手臂關係一起解 (非線性最小平方):
        位置:s·R_PB·C_P(i) + T = TCP(i) + R_i·p
        方向:R_PB·R_P_cam(i) = R_i·R_tc
    未知數:R_PB (3)、log s (1)、T (3)、p (3)、R_tc (3)。比 refine_handeye_abs 多解了紙面相對手臂的傾斜
    (筆尖校正 A 只有 XY,紙面和手臂 XY 平面差 1~2° 時,斜看 300 mm 外的點就會差好幾 mm)。
    rot_w:方向殘差每 1° 當成幾 mm。回傳 dict(R_PB, s, T, p, Rtc, rms_mm, rot_rms_deg, res_mm, used) 或 None。"""
    from scipy.optimize import least_squares
    if len(records) < 5:
        return None
    Rt = [R_tool(r["tcp"][3:], conv) for r in records]
    Tc = np.array([r["tcp"][:3] for r in records], float)
    Cp = np.array([r["C"] for r in records], float)
    Rpc = [np.asarray(r["R"]).T for r in records]                 # R_P_cam
    # 初值:用固定 R_PB 的線性解
    lin = refine_handeye_abs(records, R_PB0, s0, conv, min_sv=0.0)
    p0 = lin["p"] if lin else np.zeros(3)
    T0 = -(lin["c0"] if lin else np.zeros(3))
    x0 = np.concatenate([Rotation.from_matrix(R_PB0).as_rotvec(), [np.log(s0)], T0, p0,
                         Rotation.from_matrix(lin["Rtc"] if lin else np.eye(3)).as_rotvec()])
    keep = np.ones(len(records), bool)

    def unpack(x):
        return (Rotation.from_rotvec(x[:3]).as_matrix(), float(np.exp(x[3])), x[4:7], x[7:10],
                Rotation.from_rotvec(x[10:13]).as_matrix())

    def resid(x, idx):
        RPB, s, T, p, Rtc = unpack(x)
        out = []
        for i in idx:
            out.append(s * RPB @ Cp[i] + T - (Tc[i] + Rt[i] @ p))
            E = (RPB @ Rpc[i]).T @ (Rt[i] @ Rtc)
            out.append(rot_w * np.degrees(Rotation.from_matrix(E).as_rotvec()))
        return np.concatenate(out)

    x = x0
    for _ in range(3):
        idx = np.nonzero(keep)[0]
        x = least_squares(resid, x, args=(idx,), loss="soft_l1", f_scale=3.0).x
        RPB, s, T, p, Rtc = unpack(x)
        pos = np.array([np.linalg.norm(s * RPB @ Cp[i] + T - (Tc[i] + Rt[i] @ p)) for i in range(len(records))])
        thr = max(3.0, 3.0 * np.median(pos[keep]))
        new = pos < thr
        if new.sum() < 5 or (new == keep).all():
            break
        keep = new
    rot = np.array([np.degrees(np.linalg.norm(Rotation.from_matrix((RPB @ Rpc[i]).T @ (Rt[i] @ Rtc)).as_rotvec()))
                    for i in range(len(records))])
    return {"R_PB": RPB, "s": s, "T": T, "p": p, "Rtc": Rtc, "rms_mm": float(np.sqrt(np.mean(pos[keep] ** 2))),
            "rot_rms_deg": float(np.sqrt(np.mean(rot[keep] ** 2))), "res_mm": pos.tolist(), "used": keep.tolist(), "n": int(keep.sum())}


def predict_full(fs, tcp, conv):
    """refine_full 的結果 + TCP -> PnP 形式的相機位姿 dict(R, t, C) (紙面座標)。"""
    tcp = np.asarray(tcp, float)
    Rt = R_tool(tcp[3:], conv)
    C = fs["R_PB"].T @ (tcp[:3] + Rt @ fs["p"] - fs["T"]) / fs["s"]
    R = (fs["R_PB"].T @ (Rt @ fs["Rtc"])).T
    return {"R": R, "t": -R @ C, "C": C}
