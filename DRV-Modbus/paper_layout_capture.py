# -*- coding: utf-8 -*-
"""
從相機畫面建立「實際這張校正紙」的版面檔 (給 paper_frame.PaperFrame 用)。不連線手臂、不移動手臂。

適用:紙不是用 a4_calib_sheet.py 印的,或不確定 tag 的排法/方向。
做法:
  - 紙面座標軸依「home 拍到的畫面」定義:x 沿 tag 左上 -> 右上,y 沿 左上 -> 左下 (原點在紙左上角)。
  - tag 中心間距 dx (x 方向) / dy (y 方向):有尺量值就用 --dx/--dy;沒有就用 PnP 由 tag 邊長估計 (約 ±1%)。
    之後筆尖校正是「紙面 -> 手臂」的仿射擬合,間距的比例誤差會被吸收,所以估計值也能用。
  - 每個 tag 的 4 個角點順序依偵測結果對應,所以 tag 印成任何旋轉方向都可以。

  python paper_layout_capture.py --tag-mm 29                      # 估計間距
  python paper_layout_capture.py --tag-mm 29 --dx 238.5 --dy 155  # 用尺量值 (tag 左上角到左上角)
  python paper_layout_capture.py --image snap.png --tag-mm 29     # 用存好的影像
輸出 print/lab_sheet_layout.json (之後 pen_calibrate / pen_marks / seam_boards 用 --layout 指定)。
"""
import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np

from paper_frame import FAMILIES, DEFAULT_K

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "print", "lab_sheet_layout.json")


def detect(frame, family):
    d = cv2.aruco.Dictionary_get(FAMILIES[family])
    p = cv2.aruco.DetectorParameters_create()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=p)
    return {int(i[0]): c[0].astype(np.float64) for c, i in zip(corners, ids)} if ids is not None else {}


def assign_positions(found):
    """4 個 tag -> {"TL","TR","BR","BL": id} (依影像位置)。"""
    cen = {i: c.mean(axis=0) for i, c in found.items()}
    ids = sorted(cen, key=lambda i: cen[i][1])
    top, bot = sorted(ids[:2], key=lambda i: cen[i][0]), sorted(ids[2:], key=lambda i: cen[i][0])
    return {"TL": top[0], "TR": top[1], "BL": bot[0], "BR": bot[1]}


def pnp_spacing(found, pos, tag_mm, K):
    s = tag_mm
    obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float32)
    P = {}
    for i, c in found.items():
        _, _, tv = cv2.solvePnP(obj, c.astype(np.float32), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        P[i] = tv.ravel()
    d = lambda a, b: float(np.linalg.norm(P[pos[a]] - P[pos[b]]))   # noqa: E731
    return (d("TL", "TR") + d("BL", "BR")) / 2, (d("TL", "BL") + d("TR", "BR")) / 2, P


def build(found, pos, tag_mm, dx, dy, margin_x, margin_y, family):
    cen = {i: c.mean(axis=0) for i, c in found.items()}
    ex = cen[pos["TR"]] - cen[pos["TL"]] + cen[pos["BR"]] - cen[pos["BL"]]
    ey = cen[pos["BL"]] - cen[pos["TL"]] + cen[pos["BR"]] - cen[pos["TR"]]
    ex, ey = ex / np.linalg.norm(ex), ey / np.linalg.norm(ey)
    origin = {"TL": (margin_x, margin_y), "TR": (margin_x + dx, margin_y),
              "BL": (margin_x, margin_y + dy), "BR": (margin_x + dx, margin_y + dy)}
    tags = []
    for p, i in pos.items():
        ox, oy = origin[p]
        corners = []
        for q in found[i]:
            v = q - cen[i]
            sx, sy = (1 if v @ ex > 0 else 0), (1 if v @ ey > 0 else 0)
            corners.append([ox + sx * tag_mm, oy + sy * tag_mm])
        if len({tuple(c) for c in corners}) != 4:
            sys.exit(f"tag {i} 的角點方向判斷失敗 (tag 相對紙面約 45 度?)")
        tags.append({"id": i, "pos": p, "origin_mm": [ox, oy], "corners_mm": corners})
    W, H = dx + tag_mm + 2 * margin_x, dy + tag_mm + 2 * margin_y
    return {"family": family, "paper_mm": [round(W, 2), round(H, 2)], "tag_mm": tag_mm,
            "margin_mm": [margin_x, margin_y], "tags": tags, "source": "paper_layout_capture.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-mm", type=float, required=True, help="tag 黑框外邊長 (mm)")
    ap.add_argument("--dx", type=float, help="左上 tag 到右上 tag 的間距 (mm,角到角)")
    ap.add_argument("--dy", type=float, help="左上 tag 到左下 tag 的間距 (mm,角到角)")
    ap.add_argument("--margin-x", type=float, default=14.0, help="tag 外框到紙左右邊的距離 (mm),只影響紙面範圍")
    ap.add_argument("--margin-y", type=float, default=13.0, help="tag 外框到紙上下邊的距離 (mm)")
    ap.add_argument("--family", default="36h11", choices=sorted(FAMILIES))
    ap.add_argument("--image", help="用存好的影像 (預設用 RealSense 拍)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    K = DEFAULT_K
    if args.image:
        frame = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        found = detect(frame, args.family)
    else:
        import pyrealsense2 as rs
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        prof = pipe.start(cfg)
        it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1]])
        try:
            for _ in range(30):
                pipe.wait_for_frames(5000)
            acc = {}
            for _ in range(20):
                frame = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data())
                for i, c in detect(frame, args.family).items():
                    acc.setdefault(i, []).append(c)
            found = {i: np.mean(v, axis=0) for i, v in acc.items() if len(v) >= 10}
        finally:
            pipe.stop()
    if len(found) != 4:
        sys.exit(f"需要剛好 4 個 {args.family} tag,看到 {sorted(found)}")
    pos = assign_positions(found)
    edx, edy, _ = pnp_spacing(found, pos, args.tag_mm, K)
    dx, dy = args.dx or edx, args.dy or edy
    print("tag 位置 (影像):", {p: i for p, i in pos.items()})
    print(f"PnP 估計間距 dx={edx:.1f} dy={edy:.1f} mm;採用 dx={dx:.1f} dy={dy:.1f} ({'尺量' if args.dx else '估計'})")
    lay = build(found, pos, args.tag_mm, dx, dy, args.margin_x, args.margin_y, args.family)
    lay["spacing_source"] = "ruler" if (args.dx and args.dy) else "pnp_estimate"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(lay, f, indent=2, ensure_ascii=False)
    # 自我檢查:用這個版面求單應,看 16 個角點殘差
    img = np.float32([p for t in lay["tags"] for p in found[t["id"]]])
    obj = np.float32([p for t in lay["tags"] for p in t["corners_mm"]])
    H, _ = cv2.findHomography(img, obj, 0)
    res = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2) - obj
    print(f"角點擬合殘差 RMS {np.sqrt((res ** 2).sum(1).mean()):.3f} mm (大於 0.5 表示間距或 tag 邊長不對)")
    print("已存:", args.out)


if __name__ == "__main__":
    main()
