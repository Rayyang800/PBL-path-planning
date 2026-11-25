# -*- coding: utf-8 -*-
"""
自動判斷工件是 A (立體屋脊形)、B (兩板縫合) 還是 C (圓形罐頭)。不動手臂:只用一張正上方的彩色 + 深度。

判斷依據 (門檻由 runs/ 裡的實機照片定出,見 tests/test_workpiece_classify.py):
  - 高度:B 的板子只有 2~5 mm;A、C 都高於 15 mm。home 高度的深度會低估,所以門檻留很寬。
  - B:矮,而且彩色影像找得到縫 (貼合的暗線或兩板間的白紙縫)。
  - C:頂面外緣擬合成圓的殘差小、半徑合理 (罐頭約 40 mm)。
  - A:找得到一條直的脊線 (兩側斜面交線)。
  兩種都像時選分數高的;都不像就回報「無法判斷」,不猜。

  python workpiece_classify.py --run runs/<某次紀錄>   # 用紀錄裡的 home 照片判斷 (不需要相機、手臂)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"A": "A 立體工件 (沿脊線)", "B": "B 兩板縫合 (沿縫)", "C": "C 罐頭 (沿頂面外圈)"}
TASKS = {"A": "ridge_trace", "B": "seam_trace", "C": "circle_trace"}
LOW_MM = 12.0          # 比這個矮 = 板子類


def features(color, depth_raw, info, layout_path):
    """回傳判斷用的量 (dict)。depth_raw:uint16 深度 (和彩色對齊),info:相機資訊 (含 depth_scale_m、color_intrinsics),
    layout_path:校正紙版面 json 的路徑。"""
    import circle_rim
    import ridge
    import scan3d
    import seam_boards
    from paper_frame import PaperFrame

    f = {"ok": False}
    with open(layout_path, encoding="utf-8") as f_:
        layout = json.load(f_)
    K = scan3d.K_of(info["color_intrinsics"])
    pose = scan3d.tag_pose(color, layout, K)
    if pose is None or pose["n_tags"] < 4:
        f["reason"] = "看不到校正紙 4 個 tag"
        return f
    mm = depth_raw.astype(np.float32) * float(info["depth_scale_m"]) * 1000.0
    mm[depth_raw == 0] = np.nan
    view = {"name": "home", "color": color, "mm": mm, "K": K, "pose": pose, "pose_source": "tags"}
    xy, h, _, _, obj = scan3d.process_views([view], layout)
    f.update(ok=True, cam_h_mm=float(-pose["C"][2]))
    if obj is not None and xy is not None:
        f.update(h_max=float(obj["h_max"]), area_mm2=float(obj["area_mm2"]))
        b = obj["bbox"]
        f["bbox_mm"] = [float(b[2] - b[0]), float(b[3] - b[1])]
        rim = circle_rim.extract_rim(xy, h)
        if rim is not None:
            f["circle"] = {"r": float(rim["r_outer"]), "rms": float(rim["rms_mm"]), "coverage": float(rim.get("coverage") or 0)}
        rd = ridge.extract_ridge(xy, h)
        if rd is not None:
            f["ridge"] = {"length": float(rd["length_mm"]), "lateral_rms": float(rd["lateral_rms_mm"]),
                          "height_rms": float(rd["height_rms_mm"])}
    else:
        f["h_max"] = 0.0
    pf = PaperFrame(layout=layout_path)
    if pf.detect(color):
        r = seam_boards.analyze(color, pf, board_h_mm=max(2.0, min(f.get("h_max", 0.0), 8.0)))
        f["seam"] = {"ok": bool(r.get("ok")), "mode": r.get("mode"), "reason": None if r.get("ok") else r.get("reason")}
    return f


def decide(f):
    """features -> dict(kind 'A'/'B'/'C'/None, name, task, scores, reasons)。"""
    if not f.get("ok"):
        return {"kind": None, "reasons": [f.get("reason", "影像不完整")], "scores": {}}
    h = f.get("h_max", 0.0)
    sc, why = {"A": 0.0, "B": 0.0, "C": 0.0}, {"A": [], "B": [], "C": []}
    if h < LOW_MM:
        sc["B"] += 1.0
        why["B"].append(f"很矮 (最高約 {h:.1f} mm,板子)")
        if f.get("seam", {}).get("ok"):
            sc["B"] += 1.0
            why["B"].append("影像找得到縫")
    else:
        c, r = f.get("circle"), f.get("ridge")
        # 實機照片:罐頭的圓殘差 0.84~1.21 mm,屋脊工件硬擬合成圓 14.8~16.3 mm;
        # 脊線偏直線:屋脊 0.74~1.23 mm,罐頭 8~16 mm。門檻 3 mm 兩邊都留很大餘裕,超過就不給分。
        if c and 25.0 <= c["r"] <= 70.0 and c["rms"] < 3.0:
            q = 1.0 - c["rms"] / 3.0
            sc["C"] += 1.0 + q
            why["C"].append(f"頂面外緣接近圓 (半徑 {c['r']:.0f} mm、殘差 {c['rms']:.2f} mm)")
        if r and r["length"] >= 30.0 and r["lateral_rms"] < 3.0:
            q = 1.0 - r["lateral_rms"] / 3.0
            sc["A"] += 1.0 + q
            why["A"].append(f"有一條直的脊線 (長 {r['length']:.0f} mm、偏直線 {r['lateral_rms']:.2f} mm)")
        bw, bh = f.get("bbox_mm", [0, 0])
        if min(bw, bh) > 0:
            ar = max(bw, bh) / min(bw, bh)
            if ar < 1.25:
                sc["C"] += 0.5
                why["C"].append(f"外框接近正方 (長寬比 {ar:.2f})")
            elif ar > 1.6:
                sc["A"] += 0.5
                why["A"].append(f"外框細長 (長寬比 {ar:.2f})")
    best = max(sc, key=sc.get)
    ranked = sorted(sc.values(), reverse=True)
    if ranked[0] < 1.5 or ranked[0] - ranked[1] < 0.5:
        return {"kind": None, "scores": sc, "reasons": ["沒有一種明顯符合"] + [f"{k}: {'、'.join(v) or '不符合'}" for k, v in why.items()],
                "features": f}
    return {"kind": best, "name": NAMES[best], "task": TASKS[best], "scores": sc, "reasons": why[best], "features": f}


def classify(color, depth_raw, info, layout_path):
    return decide(features(color, depth_raw, info, layout_path))


def from_run(run):
    """用一次紀錄裡的 home 照片 (home.png / home_before.png) 判斷。"""
    import cv2
    for name in ("home", "home_before", "frame"):
        p = os.path.join(run, "images", name + ".png")
        if os.path.exists(p):
            break
    else:
        raise FileNotFoundError("紀錄裡沒有 home 照片")
    color = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.fromfile(os.path.join(run, "depth", name + ".png"), np.uint8), cv2.IMREAD_UNCHANGED)
    with open(os.path.join(run, "depth", name + ".json"), encoding="utf-8") as f_:
        dmeta = json.load(f_)
    with open(os.path.join(run, "meta.json"), encoding="utf-8") as f_:
        info = dict(json.load(f_).get("camera") or {})
    info["depth_scale_m"] = dmeta.get("depth_scale_m") or info.get("depth_scale_m")
    if dmeta.get("intrinsics"):
        info["color_intrinsics"] = dmeta["intrinsics"]
    lay = os.path.join(run, "inputs", "lab_sheet_layout.json")
    if not os.path.exists(lay):
        lay = os.path.join(HERE, "print", "lab_sheet_layout.json")
    return classify(color, depth, info, lay)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    r = from_run(ap.parse_args().run)
    print("判斷:", r.get("name") or "無法判斷", "|", "、".join(r["reasons"]), "| 分數", r["scores"])


if __name__ == "__main__":
    sys.exit(main())
