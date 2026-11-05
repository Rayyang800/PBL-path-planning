"""
用 RealSense 偵測 AprilTag,記錄每個 tag 在「相機座標系」的空間位置,
並同時記錄當下手臂 TCP 位姿 (只讀取 Modbus,不會控制手臂)。

操作 (預覽視窗):
  SPACE  記錄一次 (平均 --frames 張影像,降低雜訊)
  q/Esc  離開

輸出 (records/ 資料夾):
  apriltag_records.csv     每個 tag 一列
  apriltag_records.jsonl   每次記錄一行完整資料
  snapshot_XXX.png         記錄當下的畫面

座標說明:
  cam_x/y/z = tag 中心點在相機座標系的位置 (公尺,x 右、y 下、z 前),由深度圖反投影得到。
  tcp_*     = 記錄當下手臂 TCP 位姿 (mm, deg)。要換算到手臂基座座標,還需要手眼校正。
  若給 --tag-size (公尺),另外用 PnP 求出 tag 的姿態 (rvec/tvec)。

用法:
  python record_apriltags.py                      # 預覽,按 SPACE 記錄
  python record_apriltags.py --tag-size 0.05      # 同時算 tag 姿態 (tag 邊長 5 cm)
  python record_apriltags.py --family 36h11       # 指定字典 (預設 5x5_100;AprilTag=36h11;不確定用 auto)
  python record_apriltags.py --no-robot           # 不讀手臂位姿
  python record_apriltags.py --once               # 無視窗,直接記錄一次
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

import cam_orient

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "records")
ROBOT_IP = "192.168.1.1"

FAMILIES = {
    "5x5_100": cv2.aruco.DICT_5X5_100,   # 專案 aruco_img/ 的圖用這個
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "aruco_original": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,   # AprilTag
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
}


def make_detector(family):
    """family 可為字典名稱或 'auto' (同時嘗試所有字典)。回傳 ([(name, dict)], params)。"""
    names = list(FAMILIES) if family == "auto" else [family]
    p = cv2.aruco.DetectorParameters_create()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return [(n, cv2.aruco.Dictionary_get(FAMILIES[n])) for n in names], p


def detect(gray, dicts, params):
    """依序嘗試各字典,回傳第一個有偵測到的結果 (corners, ids, 字典名稱)。"""
    for name, d in dicts:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
        if ids is not None:
            return [c[0] for c in corners], [int(i[0]) for i in ids], name
    return [], [], None


def depth_at(depth_img, u, v, scale, win=3):
    """以 (u,v) 為中心 (2*win+1)^2 視窗的深度中位數 (公尺);無有效深度回傳 None。"""
    h, w = depth_img.shape
    u, v = int(round(u)), int(round(v))
    patch = depth_img[max(0, v - win):min(h, v + win + 1), max(0, u - win):min(w, u + win + 1)]
    vals = patch[patch > 0]
    if vals.size < 3:
        return None
    return float(np.median(vals)) * scale


def tag_observations(corners, ids, depth_img, scale, intr, tag_size, K, D):
    """回傳 {id: {...}} 單張影像中每個 tag 的觀測。"""
    obs = {}
    for c, i in zip(corners, ids):
        u, v = c.mean(axis=0)
        z = depth_at(depth_img, u, v, scale)
        xyz = None
        if z is not None:
            xyz = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], z)
        o = {"px": [float(u), float(v)], "xyz": xyz, "corners": c.tolist()}
        if tag_size:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(c.reshape(1, 4, 2), tag_size, K, D)
            o["rvec"] = rvec[0][0].tolist()
            o["tvec"] = tvec[0][0].tolist()
        obs[i] = o
    return obs


def read_tcp(ip):
    """唯讀取得手臂 TCP 位姿 (mm, deg);失敗回傳 None。"""
    try:
        from pymodbus.client import ModbusTcpClient
        from drv_modbus import request
        c = ModbusTcpClient(host=ip, port=502, timeout=2)
        if not c.connect():
            return None
        try:
            pose = request.Get_TCP_Pose(c)
        finally:
            c.close()
        return list(pose) if pose else None
    except Exception as e:  # noqa: BLE001
        print("讀取手臂位姿失敗:", e)
        return None


def average(per_frame, n_frames):
    """把多張影像的觀測平均成每個 tag 一筆。"""
    ids = sorted({i for f in per_frame for i in f})
    result = {}
    for i in ids:
        obs = [f[i] for f in per_frame if i in f]
        xyzs = np.array([o["xyz"] for o in obs if o["xyz"] is not None])
        rec = {
            "id": i,
            "seen_frames": len(obs),
            "total_frames": n_frames,
            "px": np.mean([o["px"] for o in obs], axis=0).tolist(),
        }
        if len(xyzs):
            rec["cam_xyz"] = xyzs.mean(axis=0).tolist()
            rec["cam_xyz_std"] = xyzs.std(axis=0).tolist()
        else:
            rec["cam_xyz"] = None
            rec["cam_xyz_std"] = None
        if "rvec" in obs[0]:
            rec["rvec"] = np.mean([o["rvec"] for o in obs], axis=0).tolist()
            rec["tvec"] = np.mean([o["tvec"] for o in obs], axis=0).tolist()
        result[i] = rec
    return result


def save_record(snap_id, tags, tcp, image, args, family=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    img_name = f"snapshot_{snap_id:03d}.png"
    cv2.imwrite(os.path.join(OUT_DIR, img_name), image)

    record = {"snapshot": snap_id, "time": stamp, "family": family or args.family,
              "tag_size_m": args.tag_size, "tcp_pose_mm_deg": tcp,
              "image": img_name, "tags": list(tags.values())}
    with open(os.path.join(OUT_DIR, "apriltag_records.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = os.path.join(OUT_DIR, "apriltag_records.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["snapshot", "time", "tag_id", "cam_x_m", "cam_y_m", "cam_z_m",
                        "std_x", "std_y", "std_z", "seen_frames", "px_u", "px_v",
                        "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", "image"])
        for t in tags.values():
            xyz = t["cam_xyz"] or [""] * 3
            std = t["cam_xyz_std"] or [""] * 3
            w.writerow([snap_id, stamp, t["id"], *xyz, *std,
                        f'{t["seen_frames"]}/{t["total_frames"]}', *t["px"],
                        *(tcp if tcp else [""] * 6), img_name])
    return record


def print_summary(tags, tcp):
    print("-" * 60)
    print("手臂 TCP (mm,deg):", None if tcp is None else [round(v, 3) for v in tcp])
    if not tags:
        print("沒有偵測到 tag")
        return
    for t in tags.values():
        if t["cam_xyz"]:
            x, y, z = t["cam_xyz"]
            sx, sy, sz = t["cam_xyz_std"]
            print(f"tag {t['id']:>3}: 相機座標 x={x:+.4f} y={y:+.4f} z={z:+.4f} m "
                  f"(std {sx*1000:.1f}/{sy*1000:.1f}/{sz*1000:.1f} mm, 看到 {t['seen_frames']}/{t['total_frames']} 幀)")
        else:
            print(f"tag {t['id']:>3}: 偵測到但沒有有效深度")
    pts = {i: np.array(t["cam_xyz"]) for i, t in tags.items() if t["cam_xyz"]}
    for a, b in itertools.combinations(sorted(pts), 2):
        print(f"  距離 tag{a} <-> tag{b}: {np.linalg.norm(pts[a] - pts[b])*1000:.1f} mm")


def grab(pipe, align):
    frames = align.process(pipe.wait_for_frames(5000))
    c, d = frames.get_color_frame(), frames.get_depth_frame()
    if not c or not d:
        return None, None
    return np.asanyarray(c.get_data()), np.asanyarray(d.get_data())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="5x5_100", choices=sorted(FAMILIES) + ["auto"],
                    help="標記字典 (預設 5x5_100;AprilTag 用 36h11;不確定用 auto)")
    ap.add_argument("--tag-size", type=float, default=None, help="tag 黑框外邊長 (公尺),用於 PnP 姿態")
    ap.add_argument("--frames", type=int, default=15, help="每次記錄平均的影像張數")
    ap.add_argument("--no-robot", action="store_true", help="不讀取手臂位姿")
    ap.add_argument("--robot-ip", default=ROBOT_IP)
    ap.add_argument("--once", action="store_true", help="無視窗,直接記錄一次")
    cam_orient.add_arg(ap)
    args = ap.parse_args()
    rot = (args.rot180 or cam_orient.saved_mode()) == "on"

    aruco_dict, params = make_detector(args.family)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    D = np.array(intr.coeffs)

    snap_id = 0
    if os.path.exists(os.path.join(OUT_DIR, "apriltag_records.jsonl")):
        with open(os.path.join(OUT_DIR, "apriltag_records.jsonl"), encoding="utf-8") as f:
            snap_id = sum(1 for _ in f)

    def record_now():
        nonlocal snap_id
        per_frame, last, found_fam, depths = [], None, None, []
        for _ in range(args.frames):
            color, depth = grab(pipe, align)
            if color is None:
                continue
            depths.append(depth.copy())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, fam = detect(gray, aruco_dict, params)
            per_frame.append(tag_observations(corners, ids, depth, scale, intr, args.tag_size, K, D))
            last = (color, corners, ids)
            found_fam = fam or found_fam
        tags = average(per_frame, len(per_frame))
        tcp = None if args.no_robot else read_tcp(args.robot_ip)
        snap_id += 1
        shot = last[0].copy()
        cv2.aruco.drawDetectedMarkers(shot, [c.reshape(1, 4, 2).astype(np.float32) for c in last[1]],
                                      np.array(last[2]).reshape(-1, 1) if last[2] else None)
        save_record(snap_id, tags, tcp, shot, args, found_fam)
        # v2:另存原始彩色 (無標註)、深度中位數 (uint16,對齊彩色) 與內參,方便離線重算
        D = np.stack(depths).astype(np.float32)
        D[D == 0] = np.nan
        with np.errstate(all="ignore"):
            dmed = np.nan_to_num(np.nanmedian(D, axis=0), nan=0).round().astype(np.uint16)
        base = os.path.join(OUT_DIR, f"snapshot_{snap_id:03d}")
        cv2.imencode(".png", last[0])[1].tofile(base + "_raw.png")
        cv2.imencode(".png", dmed)[1].tofile(base + "_depth.png")
        with open(base + "_camera.json", "w", encoding="utf-8") as f:
            json.dump({"depth_scale_m": scale, "aligned_to": "color", "n_frames_median": len(depths),
                       "intrinsics": {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                                      "width": intr.width, "height": intr.height, "coeffs": list(intr.coeffs)},
                       "tcp_pose_mm_deg": tcp}, f, indent=2)
        print(f"\n[已記錄 #{snap_id}] -> {OUT_DIR} (含 _raw.png / _depth.png / _camera.json)")
        print_summary(tags, tcp)

    try:
        for _ in range(30):  # 等自動曝光穩定
            pipe.wait_for_frames(5000)

        if args.once:
            record_now()
            return

        print("預覽中: SPACE=記錄  q/Esc=離開")
        while True:
            color, depth = grab(pipe, align)
            if color is None:
                continue
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, fam = detect(gray, aruco_dict, params)
            view = color.copy()
            obs = tag_observations(corners, ids, depth, scale, intr, None, K, D)
            for o in obs.values():
                cv2.polylines(view, [np.array(o["corners"], dtype=np.int32)], True, (0, 255, 0), 2)
                cv2.circle(view, (int(o["px"][0]), int(o["px"][1])), 4, (0, 0, 255), -1)
            view = cam_orient.rotate(view, rot)     # 只轉「顯示」;偵測與 3D 計算都用原始影像 (相機座標系)
            for i, o in obs.items():
                u, v = (int(round(x)) for x in cam_orient.raw_point(o["px"], view.shape, rot))
                txt = f"id{i}"
                if o["xyz"]:
                    txt += " %.3f %.3f %.3f" % tuple(o["xyz"])
                cv2.putText(view, txt, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(view, f"tags: {len(obs)} [{fam or args.family}]  SPACE=record  q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("apriltag", cv2.resize(view, (960, 540)))
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                record_now()
            elif k in (ord("q"), 27):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
