# PBL 專題:智慧化焊道/塗膠路徑規劃

台達 DRV 手臂 + Intel RealSense D435i (裝在手臂上) + 夾爪夾白板筆,以視覺辨識工件並規劃塗膠路徑。

- **專案規範與現況**:[`PROJECT_SPEC.md`](PROJECT_SPEC.md)(決策、暫存器表、程式清單、已知陷阱、待辦)
- **簡報**:`PBL專題規劃報告.pptx`、`PBL專題規劃報告_三種工件版.pptx`
- **程式**:[`DRV-Modbus/`](DRV-Modbus/),以 [liaochikon/DRV-Modbus](https://github.com/liaochikon/DRV-Modbus)(MIT License)為基礎擴充

## 環境安裝 (Windows)

```bash
conda create -n delta_arm python=3.8 -y
conda activate delta_arm
pip install pymodbus==3.6.9 opencv-contrib-python==4.6.0.66 scipy pyrealsense2
```

筆電有線網路設為 `192.168.1.x`(手臂 `192.168.1.1:502`)。

## 常用指令 (在 `DRV-Modbus/` 下)

```bash
python watch_status.py          # 唯讀:看手臂位姿與旗標
python warp_sort.py             # dry-run:辨識並印出計畫,不動手臂
python warp_sort.py --go --speed 25
python warp_calibrate.py --pixels-only   # 相機改裝後重記 tag 像素
python cam_orient.py --set on            # 相機反裝 180 度
```

**安全**:會動手臂的程式先 dry-run;手放急停旁;旗標 `0x031F` 不是 1 時先在教導器清除警報。

## 不在 repo 裡的東西

`runs/`、`records/`、測試輸出影像與舊版簡報未上傳(見 `.gitignore`)。
`calibration.json` 是本組工作台的校正結果,換工作台或移動 tag 後要重新校正。
