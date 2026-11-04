# -*- coding: utf-8 -*-
"""
合成場景測試 (不需要手臂與相機):
  以針孔相機 (D435i 內參、離紙 450 mm、略傾斜、紙旋轉 90°) 渲染 A4 紙 + 4 個 tag + 兩塊有厚度的黑板,
  再用 paper_frame / seam_boards 還原縫合線,與已知的真值比較。

  python tests\\test_seam_synthetic.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import a4_calib_sheet as sheet                      # noqa: E402
from paper_frame import PaperFrame, DEFAULT_K       # noqa: E402
import seam_boards                                  # noqa: E402

WI, HI = 1280, 720
K = DEFAULT_K


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def render(layout, cam_xy, cam_h, theta_deg, tilt_deg, boards, t_mm, seed=0):
    """boards: [(x0,y0,x1,y1) mm]。板子頂面高出紙面 t_mm。紙面座標 z 軸指向桌面 (相機在 z<0)。"""
    W, Hh = layout["paper_mm"]
    tag_mm = layout["tag_mm"]
    u, v = np.meshgrid(np.arange(WI), np.arange(HI))
    rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    R = rot_z(np.radians(theta_deg)) @ rot_x(np.radians(tilt_deg))
    rays = rays_c @ R.T
    C = np.array([cam_xy[0], cam_xy[1], -cam_h])

    def hit(z0):
        lam = (z0 - C[2]) / rays[..., 2]
        return C[0] + lam * rays[..., 0], C[1] + lam * rays[..., 1]

    x0, y0 = hit(0.0)
    xt, yt = hit(-t_mm)
    img = np.full((HI, WI), 90.0, np.float32)
    paper = (x0 >= 0) & (x0 <= W) & (y0 >= 0) & (y0 <= Hh)
    img[paper] = 235
    for tag in layout["tags"]:
        ox, oy = tag["origin_mm"]
        s = tag_mm / 8.0
        inside = paper & (x0 >= ox) & (x0 < ox + tag_mm) & (y0 >= oy) & (y0 < oy + tag_mm)
        j = np.clip(((x0 - ox) / s).astype(int), 0, 7)
        i = np.clip(((y0 - oy) / s).astype(int), 0, 7)
        black = sheet.tag_modules(layout["family"], tag["id"])[i, j] == 1
        img[inside & black] = 25
        img[inside & ~black] = 240
    for bx0, by0, bx1, by1 in boards:
        img[(xt >= bx0) & (xt <= bx1) & (yt >= by0) & (yt <= by1)] = 30
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    img += np.random.RandomState(seed).normal(0, 2.0, img.shape).astype(np.float32)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def run_case(name, gap_mm, t_true, t_assumed, layout, save=None, cam_xy=(118.0, 135.0), quiet=False):
    a = (20.0, 60.0, 100.0, 200.0)                       # 板 A:x 20~100,y 60~200
    xb0 = 100.0 + gap_mm
    b = (xb0, 75.0, xb0 + 82.0, 215.0)                   # 板 B:與 A 錯開 15 mm
    seam_x = 100.0 + gap_mm / 2
    frame = render(layout, cam_xy=cam_xy, cam_h=450.0, theta_deg=90.0, tilt_deg=3.0, boards=[a, b], t_mm=t_true)
    if quiet:
        res = seam_boards.analyze(frame, PaperFrame(), board_h_mm=t_assumed)
        res["max_err"] = float(np.abs(res["seam_paper_mm"][:, 0] - seam_x).max()) if res["ok"] else None
        return res
    pf = PaperFrame()
    res = seam_boards.analyze(frame, pf, board_h_mm=t_assumed)
    print(f"[{name}] 縫寬 {gap_mm} mm,板厚真值 {t_true} / 程式假設 {t_assumed}")
    print("   ", pf.describe())
    if not res["ok"]:
        print("    失敗:", res["reason"])
        return res
    s = res["seam_paper_mm"]
    ex = np.abs(s[:, 0] - seam_x)
    y0, y1 = sorted([s[0, 1], s[-1, 1]])
    print(f"    縫位置誤差 平均 {ex.mean():.2f} mm / 最大 {ex.max():.2f} mm;縫寬量測 {res['gap_mm']:.2f} mm (真值 {gap_mm});"
          f"起訖 y = {y0:.1f}~{y1:.1f} (真值 75~200);長度 {res['length_mm']:.1f} mm")
    if save:
        cv2.imencode(".png", seam_boards.overlay(frame, pf, res))[1].tofile(save)
        cv2.imencode(".png", frame)[1].tofile(save.replace(".png", "_raw.png"))      # 原始影像,可拿去餵 seam_boards.py --image
    return res


def main():
    pf0 = PaperFrame()
    layout = pf0.layout
    out = os.path.join(ROOT, "tests", "out")
    os.makedirs(out, exist_ok=True)
    r1 = run_case("A 理想:薄板、縫 3 mm", 3.0, 0.0, 0.0, layout, os.path.join(out, "case_a.png"))
    r2 = run_case("B 板厚 8 mm、程式不知道厚度 (未修正視差)", 3.0, 8.0, 0.0, layout, os.path.join(out, "case_b.png"))
    r3 = run_case("C 板厚 8 mm、程式填入厚度 8 (有修正視差)", 3.0, 8.0, 8.0, layout, os.path.join(out, "case_c.png"))
    # 視差:相機中心離縫較遠時影響才明顯 (縫位置誤差 ≈ 離相機中心的水平距離 × 板厚 / 相機離紙高度)
    run_case("B2 相機偏在紙左上、板厚 8 mm、未修正視差", 3.0, 8.0, 0.0, layout, cam_xy=(30.0, 50.0))
    run_case("C2 同上、有修正視差 (填入厚度 8)", 3.0, 8.0, 8.0, layout, cam_xy=(30.0, 50.0))
    run_case("B3 板厚 15 mm、相機偏在紙左上、未修正", 3.0, 15.0, 0.0, layout, cam_xy=(30.0, 50.0))
    run_case("C3 同上、有修正", 3.0, 15.0, 15.0, layout, cam_xy=(30.0, 50.0))
    print("\n縫寬掃描 (薄板;能否分成兩塊板):")
    for g in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        r = run_case("", g, 0.0, 0.0, layout, quiet=True)
        print(f"    縫 {g:.1f} mm -> " + (f"成功,最大位置誤差 {r['max_err']:.2f} mm,量到縫寬 {r['gap_mm']:.2f} mm" if r["ok"] else "失敗 (被當成一塊)"))
    ok = r1["ok"] and r3["ok"]
    if ok:
        e1 = np.abs(r1["seam_paper_mm"][:, 0] - 101.5).max()
        e3 = np.abs(r3["seam_paper_mm"][:, 0] - 101.5).max()
        print("\n結論: 理想情況最大誤差 %.2f mm;有視差修正 %.2f mm" % (e1, e3))
        assert e1 < 0.6 and e3 < 0.8, "誤差超過預期"
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
