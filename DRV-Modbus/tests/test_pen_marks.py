# -*- coding: utf-8 -*-
"""pen_marks 合成測試 (不需要手臂與相機)。  python tests\\test_pen_marks.py
用實驗室紙的版面 (print/lab_sheet_layout.json) 畫一張紙:4 個 tag + 5 個「+」,以單應投影成相機影像。
已知 紙面 -> 手臂 的仿射 (含筆尖偏移),粗略換算故意偏 15 mm,檢查能否還原。"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import a4_calib_sheet as sheet                    # noqa: E402
import pen_marks                                  # noqa: E402
from paper_frame import PaperFrame               # noqa: E402

PPM = 6.0   # 合成紙 每 mm 像素


def paper_canvas(layout, marks, stroke_mm=12.0, ink_mm=1.6, dots=(), dirt=()):
    W, H = layout["paper_mm"]
    img = np.full((int(H * PPM), int(W * PPM)), 238, np.uint8)
    t = layout["tag_mm"]
    for tag in layout["tags"]:
        mod = sheet.tag_modules(layout["family"], tag["id"])     # 8x8, 1 = 黑
        n = mod.shape[0]
        # 角點順序 [TL,TR,BR,BL] (tag 本身方向) 對應的紙面座標
        dst = np.float32(tag["corners_mm"]) * PPM
        src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
        tile = np.where(mod == 1, 20, 245).astype(np.uint8)
        tile = cv2.resize(tile, (n * 40, n * 40), interpolation=cv2.INTER_NEAREST)
        M = cv2.getPerspectiveTransform(src * 40, dst)
        warped = cv2.warpPerspective(tile, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tile, 255), M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        img[mask > 0] = warped[mask > 0]
    th = max(1, int(ink_mm * PPM))
    for (x, y), rot in marks:      # rot: 筆畫方向 (度),模擬手臂 X 軸在紙面的方向
        d1 = np.array([np.cos(np.radians(rot)), np.sin(np.radians(rot))]) * stroke_mm
        d2 = np.array([-d1[1], d1[0]])
        c = np.array([x, y])
        for d in (d1, d2):
            cv2.line(img, tuple(np.int32((c - d) * PPM)), tuple(np.int32((c + d) * PPM)), 35, th, cv2.LINE_AA)
    for (x, y), r_mm in list(dots) + list(dirt):
        cv2.circle(img, (int(round(x * PPM * 16)), int(round(y * PPM * 16))), int(r_mm * PPM * 16), 35, -1,
                   cv2.LINE_AA, shift=4)
    return img


def to_camera(canvas, Hp, seed):
    cam = cv2.warpPerspective(canvas, Hp, (1280, 720), borderValue=120, flags=cv2.INTER_AREA)
    cam = cv2.GaussianBlur(cam, (0, 0), 0.8)
    cam = np.clip(cam + np.random.RandomState(seed).normal(0, 2, cam.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cam, cv2.COLOR_GRAY2BGR)


def main():
    lay_path = os.path.join(ROOT, "print", "lab_sheet_layout.json")
    with open(lay_path, encoding="utf-8") as f:
        layout = json.load(f)
    W, H = layout["paper_mm"]
    # 真實的 紙面 -> 手臂 TCP 指令 (含筆尖偏移 (+3.2, -1.7)):紙 x 向右 ≈ 手臂 -Y,紙 y 向下 ≈ 手臂 -X (鏡像)
    ang = np.radians(1.5)
    A_true = np.array([[0, -1], [-1, 0]]) @ np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    t_true = np.array([600.0, 60.0]) + [3.2, -1.7]
    cx, cy = W / 2, H / 2
    paper_pts = [(cx, cy), (cx - 75, cy - 45), (cx + 75, cy - 45), (cx + 75, cy + 45), (cx - 75, cy + 45)]
    robot_cmd = [A_true @ p + t_true for p in paper_pts]
    stroke_dir = np.degrees(np.arctan2(*(np.linalg.inv(A_true) @ [1, 0])[::-1]))   # 手臂 X 方向在紙面的角度
    canvas = paper_canvas(layout, [(p, stroke_dir) for p in paper_pts])

    # 紙面 mm -> 相機像素:旋轉、縮放 1.75 px/mm、輕微透視
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]]) * PPM
    dst = np.float32([[240, 140], [770, 136], [775, 498], [236, 503]])
    Hp = cv2.getPerspectiveTransform(src, dst)
    cam = cv2.warpPerspective(canvas, Hp, (1280, 720), borderValue=120, flags=cv2.INTER_AREA)
    cam = cv2.GaussianBlur(cam, (0, 0), 0.8)
    cam = np.clip(cam + np.random.RandomState(1).normal(0, 2, cam.shape), 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(cam, cv2.COLOR_GRAY2BGR)

    # 粗略 像素 -> 手臂:真實值再故意偏 15 mm
    pix_of_paper = lambda P: cv2.perspectiveTransform(np.float32(P).reshape(-1, 1, 2) * PPM, Hp).reshape(-1, 2)  # noqa
    corners_paper = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    H_rough = cv2.getPerspectiveTransform(pix_of_paper(corners_paper),
                                          np.float32([A_true @ p + t_true + [12.0, -9.0] for p in corners_paper]))
    points = [{"robot_xy": c.tolist(), "contact_z": 412.0 + 0.004 * c[0] - 0.002 * c[1]} for c in robot_cmd]

    pf = PaperFrame(layout=lay_path)
    res = pen_marks.analyze(frame, pf, points, H_rough, 12.0)
    print(pen_marks.summarize(res))
    out = os.path.join(ROOT, "tests", "out")
    os.makedirs(out, exist_ok=True)
    cv2.imencode(".png", frame)[1].tofile(os.path.join(out, "pen_marks_raw.png"))
    cv2.imencode(".jpg", pen_marks.overlay(frame, res))[1].tofile(os.path.join(out, "pen_marks_overlay.jpg"))
    assert res["ok"] and res["n_points"] == 5, res.get("reason")
    A, t = np.array(res["A"]), np.array(res["t"])
    test = np.array([[40.0, 40.0], [W - 40, H - 40], [cx, cy]])
    err = np.linalg.norm(test @ A.T + t - (test @ A_true.T + t_true), axis=1)
    print(f"\n與真值比較:紙面上 3 個測試點的 TCP 指令誤差 {', '.join(f'{e:.2f}' for e in err)} mm")
    print(f"平面傾斜 {res['plane']['tilt_deg']:.3f}° (真值 {np.degrees(np.arctan(np.hypot(0.004, 0.002))):.3f}°)")
    assert err.max() < 0.6 and res["rms_mm"] < 0.5

    # ---- 點一下 (dot):墨點直徑 3 mm;紙上原本就有髒點,其中一個離 #1 只有 7 mm ----
    print("\n[dot 模式]")
    dirt = [((paper_pts[1][0] + 7, paper_pts[1][1] - 3), 1.2), ((paper_pts[3][0] - 10, paper_pts[3][1] + 8), 2.0)]
    before = to_camera(paper_canvas(layout, [], dirt=dirt), Hp, 2)
    after = to_camera(paper_canvas(layout, [], dots=[(p, 1.5) for p in paper_pts], dirt=dirt), Hp, 3)
    res_d = pen_marks.analyze(after, PaperFrame(layout=lay_path), points, H_rough, mark="dot", before=before)
    print(pen_marks.summarize(res_d))
    cv2.imencode(".jpg", pen_marks.overlay(after, res_d))[1].tofile(os.path.join(out, "pen_dots_overlay.jpg"))
    assert res_d["ok"] and res_d["n_points"] == 5, res_d.get("reason")
    A, t = np.array(res_d["A"]), np.array(res_d["t"])
    err_d = np.linalg.norm(test @ A.T + t - (test @ A_true.T + t_true), axis=1)
    print(f"與真值比較 (dot):{', '.join(f'{e:.2f}' for e in err_d)} mm")
    assert err_d.max() < 0.6 and res_d["rms_mm"] < 0.5
    # 沒有 before 影像時,髒點可能被誤認;只要確認程式不會當掉
    pen_marks.analyze(after, PaperFrame(layout=lay_path), points, H_rough, mark="dot")
    print("PASS")


if __name__ == "__main__":
    main()
