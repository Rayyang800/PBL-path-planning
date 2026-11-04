# -*- coding: utf-8 -*-
"""
產生列印用的 A4 校正紙:四角各一個 AprilTag (預設 36h11、邊長 28 mm),輸出向量 PDF + 預覽 PNG + 版面 JSON。

  python a4_calib_sheet.py                    # 直式 210x297,tag 28 mm,距邊 12 mm,ID 0~3
  python a4_calib_sheet.py --landscape        # 橫式 297x210
  python a4_calib_sheet.py --tag-mm 28 --margin-mm 12

列印注意:務必選「實際大小 / 100%」,不要「符合頁面」。紙上有 100 mm 比例尺,印完用尺量。
若量出來的 tag 邊長不是 28.0 mm,用 --tag-mm 填實際值重新產生版面 JSON (PDF 不必重印,只要 JSON 與實物一致)。

版面座標 (paper mm):原點在紙的左上角 (人正看紙面),x 向右、y 向下。
tag 角點順序 = TL, TR, BR, BL (與 OpenCV aruco 偵測順序一致)。
"""
import argparse
import json
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "print")
FAMILIES = {"36h11": cv2.aruco.DICT_APRILTAG_36h11, "25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "16h5": cv2.aruco.DICT_APRILTAG_16h5}


def tag_modules(family, tag_id):
    """回傳 8x8 的 0/1 陣列 (1 = 黑),含 1 格外框。"""
    d = cv2.aruco.Dictionary_get(FAMILIES[family])
    n = d.markerSize + 2
    img = cv2.aruco.drawMarker(d, tag_id, n * 10, borderBits=1)
    return (img[5::10, 5::10] == 0).astype(int)


def build_layout(w_mm, h_mm, tag_mm, margin_mm, ids, family):
    t, m = tag_mm, margin_mm
    tl = [(m, m), (w_mm - m - t, m), (w_mm - m - t, h_mm - m - t), (m, h_mm - m - t)]   # TL, TR, BR, BL
    names = ["TL", "TR", "BR", "BL"]
    tags = []
    for (x, y), tid, nm in zip(tl, ids, names):
        corners = [[x, y], [x + t, y], [x + t, y + t], [x, y + t]]
        tags.append({"id": int(tid), "pos": nm, "origin_mm": [x, y], "corners_mm": corners})
    return {"family": family, "paper_mm": [w_mm, h_mm], "tag_mm": t, "margin_mm": m, "tags": tags}


def draw_sheet(layout, pdf_path, png_path):
    w, h = layout["paper_mm"]
    t = layout["tag_mm"]
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    patches = []
    for tag in layout["tags"]:
        mods = tag_modules(layout["family"], tag["id"])
        s = t / mods.shape[0]
        x0, y0 = tag["origin_mm"]
        for i in range(mods.shape[0]):
            for j in range(mods.shape[1]):
                if mods[i, j]:
                    e = 0.015     # 相鄰黑格略微重疊,避免部分檢視器出現髮絲縫
                    patches.append(Rectangle((x0 + j * s - e, y0 + i * s - e), s + 2 * e, s + 2 * e))
        ax.text(x0 + t / 2, y0 + t + 2.6, f"ID {tag['id']}", fontsize=6, ha="center", va="top", color="0.35")
    ax.add_collection(PatchCollection(patches, facecolor="black", edgecolor="none", linewidth=0))

    # 100 mm 比例尺 (置於下緣中央)
    cx, y = w / 2, h - layout["margin_mm"] - t - 14
    ax.plot([cx - 50, cx + 50], [y, y], color="black", linewidth=0.6, solid_capstyle="butt")
    for k in range(11):
        L = 3.0 if k % 5 == 0 else 1.8
        ax.plot([cx - 50 + 10 * k] * 2, [y, y - L], color="black", linewidth=0.5)
    ax.text(cx, y + 2.5, "100 mm", fontsize=7, ha="center", va="top")
    ax.text(cx, y + 8, "列印請選「實際大小 100%」,勿縮放 / Print at 100%, do not fit to page",
            fontsize=6, ha="center", va="top", color="0.35")
    ax.text(w / 2, layout["margin_mm"] - 3, "▲ 上 TOP (ID %d / %d)" % (layout["tags"][0]["id"], layout["tags"][1]["id"]),
            fontsize=6, ha="center", va="bottom", color="0.35")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-mm", type=float, default=28.0, help="tag 黑框外邊長 (mm)")
    ap.add_argument("--margin-mm", type=float, default=12.0, help="tag 離紙邊 (mm)")
    ap.add_argument("--ids", type=int, nargs=4, default=[0, 1, 2, 3], help="TL TR BR BL 的 tag ID")
    ap.add_argument("--family", default="36h11", choices=sorted(FAMILIES))
    ap.add_argument("--landscape", action="store_true")
    ap.add_argument("--json-only", action="store_true", help="只重寫版面 JSON (tag 實測尺寸與預設不同時用)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    w, h = (297.0, 210.0) if args.landscape else (210.0, 297.0)
    layout = build_layout(w, h, args.tag_mm, args.margin_mm, args.ids, args.family)
    os.makedirs(args.out, exist_ok=True)
    jp = os.path.join(args.out, "a4_layout.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    print("版面 JSON:", jp)
    if args.json_only:
        return
    pdf, png = os.path.join(args.out, "a4_calib_sheet.pdf"), os.path.join(args.out, "a4_calib_sheet_preview.png")
    draw_sheet(layout, pdf, png)
    print("PDF  :", pdf)
    print("預覽 :", png)
    print(f"紙 {w:.0f}x{h:.0f} mm,tag {args.tag_mm} mm,family {args.family},ID(TL,TR,BR,BL)={args.ids}")


if __name__ == "__main__":
    main()
