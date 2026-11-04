"""
多物體顏色辨識 (HSV)。找出畫面中各個有顏色的物體,標示顏色、位置與大小;
用 RealSense 時會再用深度圖算出每個物體在相機座標系的 3D 位置 (公尺)。

用法:
  python color_detect.py                  # RealSense 即時預覽,q 離開,s 存圖並印出清單
  python color_detect.py --image image.png --show   # 分析單張圖片並顯示,結果存成 *_colors.png
  python color_detect.py --min-area 800   # 忽略面積小於 800 像素的色塊

只讀取相機,不會連線或控制手臂。
HSV 範圍 (OpenCV: H 0~179) 與光源有關,偏差時請調整下方 COLOR_RANGES。
"""
import argparse
import os
import sys

import cv2
import numpy as np

import cam_orient

# 每個顏色可有多段 (lower, upper) 的 HSV 範圍;紅色色相在 0 附近會繞回 179,所以有兩段
COLOR_RANGES = {
    "red":    [((0, 110, 70), (8, 255, 255)), ((170, 110, 70), (179, 255, 255))],
    "orange": [((9, 120, 90), (20, 255, 255))],
    "yellow": [((21, 100, 100), (35, 255, 255))],
    "green":  [((36, 70, 60), (85, 255, 255))],
    "blue":   [((95, 110, 60), (130, 255, 255))],
    "purple": [((131, 70, 50), (160, 255, 255))],
}
DRAW_BGR = {
    "red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 255, 255),
    "green": (0, 200, 0), "blue": (255, 80, 0), "purple": (200, 0, 160),
}
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def color_mask(hsv, name):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[name]:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL)


def detect_colors(bgr, min_area=400, colors=None):
    """回傳依面積由大到小排序的物體清單:
    {color, area, center(u,v), bbox(x,y,w,h), contour}"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    objs = []
    for name in colors or COLOR_RANGES:
        cnts, _ = cv2.findContours(color_mask(hsv, name), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            m = cv2.moments(c)
            u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            objs.append({"color": name, "area": float(area), "center": (u, v),
                         "bbox": cv2.boundingRect(c), "contour": c})
    return sorted(objs, key=lambda o: -o["area"])


def draw(bgr, objs, xyz=None):
    out = bgr.copy()
    for k, o in enumerate(objs):
        col = DRAW_BGR[o["color"]]
        cv2.drawContours(out, [o["contour"]], -1, col, 2)
        u, v = int(o["center"][0]), int(o["center"][1])
        cv2.circle(out, (u, v), 4, (255, 255, 255), -1)
        label = f"{o['color']}"
        if xyz and xyz[k] is not None:
            label += " %.3f,%.3f,%.3f" % tuple(xyz[k])
        x, y, _, _ = o["bbox"]
        cv2.putText(out, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out


def print_objs(objs, xyz=None):
    if not objs:
        print("沒有找到色塊")
        return
    for k, o in enumerate(objs):
        s = f"#{k+1} {o['color']:<7} 面積={o['area']:>7.0f}px  中心=({o['center'][0]:.0f},{o['center'][1]:.0f})"
        if xyz and xyz[k] is not None:
            s += "  相機座標=(%.3f, %.3f, %.3f) m" % tuple(xyz[k])
        print(s)


def run_image(path, min_area, show=False):
    # 與課程範例相同: img = cv2.imread(...) -> cv2.imshow(...) -> cv2.waitKey(...)
    # 用 fromfile+imdecode,避免 cv2.imread 在 Windows 讀不了中文路徑
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        img = None
    if img is None:
        sys.exit(f"讀不到圖片 (相對路徑需放在目前資料夾下): {path}")
    objs = detect_colors(img, min_area)
    print_objs(objs)
    result = draw(img, objs)
    out = os.path.splitext(path)[0] + "_colors.png"
    cv2.imencode(".png", result)[1].tofile(out)
    print("已存:", out)
    if show:
        cv2.imshow("img", result)
        cv2.waitKey(10000)   # 顯示 10 秒,按任意鍵可提前關閉
        cv2.destroyAllWindows()


def run_camera(min_area, rot=False):
    import pyrealsense2 as rs
    print("相機方向:", "反裝 (影像轉 180 度)" if rot else "正裝")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def xyz_of(depth, u, v):
        h, w = depth.shape
        u, v = int(u), int(v)
        p = depth[max(0, v - 3):min(h, v + 4), max(0, u - 3):min(w, u + 4)]
        p = p[p > 0]
        if p.size < 3:
            return None
        return rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(np.median(p)) * scale)

    try:
        print("預覽中: s=存圖並印清單  q/Esc=離開")
        while True:
            fr = align.process(pipe.wait_for_frames(5000))
            c, d = fr.get_color_frame(), fr.get_depth_frame()
            if not c or not d:
                continue
            color, depth = np.asanyarray(c.get_data()), np.asanyarray(d.get_data())
            color = cam_orient.rotate(color, rot)            # 相機反裝時轉正;深度仍用原始座標查
            objs = detect_colors(color, min_area)
            xyz = [xyz_of(depth, *cam_orient.raw_point(o["center"], color.shape, rot)) for o in objs]
            view = draw(color, objs, xyz)
            cv2.putText(view, f"objects: {len(objs)}  s=save  q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("colors", cv2.resize(view, (960, 540)))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                print_objs(objs, xyz)
                cv2.imwrite("color_detect_snapshot.png", view)
                print("已存: color_detect_snapshot.png")
            elif k in (ord("q"), 27):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="分析單張圖片 (不使用相機)")
    ap.add_argument("--min-area", type=float, default=400, help="最小色塊面積 (像素)")
    ap.add_argument("--show", action="store_true", help="分析圖片後用視窗顯示結果 10 秒")
    cam_orient.add_arg(ap)
    args = ap.parse_args()
    if args.image:
        run_image(args.image, args.min_area, args.show)
    else:
        run_camera(args.min_area, (args.rot180 or cam_orient.saved_mode()) == "on")


if __name__ == "__main__":
    main()
