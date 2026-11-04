"""
課程投影片範例:用 HSV 篩選紅色並畫出輪廓。
  python red_detect_demo.py                 # 讀 apple.jpg
  python red_detect_demo.py test_food.jpg   # 指定圖片 (需放在目前資料夾下)
  python red_detect_demo.py 圖片 --no-show  # 不開視窗,只存結果 *_red_*.png
"""
import sys

import cv2
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
show = "--no-show" not in sys.argv
path = args[0] if args else "apple.jpg"

# fromfile+imdecode: 同 cv2.imread,但支援 Windows 中文路徑
try:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
except OSError:
    image = None
if image is None:
    sys.exit(f"讀不到圖片: {path} (圖片須放在程式目錄下)")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_hsv1 = np.array([0, 120, 70])      # 色相, 飽和度, 亮度
upper_hsv1 = np.array([10, 255, 255])
lower_hsv2 = np.array([170, 120, 70])
upper_hsv2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
mask = mask1 + mask2                      # 篩選紅色部分遮罩

# 輪廓檢測方法及輪廓逼近模式
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()
count = 0
for contour in contours:
    if cv2.contourArea(contour) > 500:    # 過濾小輪廓
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
        count += 1
print(f"圖片 {image.shape[1]}x{image.shape[0]},紅色遮罩像素 {int(np.count_nonzero(mask))},面積>500 的輪廓 {count} 個")

scale = 0.5
image_resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
mask_resized = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
output_resized = cv2.resize(output, (0, 0), fx=scale, fy=scale)

stem = path.rsplit(".", 1)[0]
cv2.imencode(".png", mask)[1].tofile(f"{stem}_red_mask.png")
cv2.imencode(".png", output)[1].tofile(f"{stem}_red_contours.png")
print(f"已存: {stem}_red_mask.png, {stem}_red_contours.png")

if show:
    cv2.imshow("Original Image", image_resized)
    cv2.imshow("Mask", mask_resized)
    cv2.imshow("Apple Contours", output_resized)
    cv2.waitKey(0)                        # 按任意鍵關閉
    cv2.destroyAllWindows()
