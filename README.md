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
python go_home.py                        # 回 home (低於安全高度會先垂直抬起)
python gripper_ctl.py                    # 鍵盤控制夾爪:o 開 / g 夾 / q 離開
python pen_quick_calib.py --pen-loaded       # 快速筆尖校正 dry-run (看 pen_quick_plan.png)
python pen_quick_calib.py --go --pen-loaded  # 人工找一次高度按 c,其餘 4 點自動,最後拍照計算
```

護貝校正紙 (白板筆可擦) 用整張紙校正,精度與範圍較好:

```bash
python pen_sheet_calib.py --pen-loaded       # dry-run (看 pen_sheet_plan.png,3x3 小十字)
python pen_sheet_calib.py --go --pen-loaded  # 第一點人工找高度按 c -> 回 home 修正 -> 其餘自動 -> 分析 (留一法誤差)
```

筆尖校正執行中:Ctrl+C = 停止並回 home;空白鍵 = 暫停選單;o / g = 夾爪開 / 夾。
舊的自動探觸版 `pen_calibrate.py` 仍保留。詳見 `PROJECT_SPEC.md` T15、D13。

**安全**:會動手臂的程式先 dry-run;手放急停旁;旗標 `0x031F` 不是 1 時先在教導器清除警報。

## 資料紀錄(每次都完整記錄)

**每一次實作——包含 dry-run、失敗、中途中止——都會自動在 `DRV-Modbus/runs/<時間>_<類型>/` 留下完整紀錄**,可離線重算:

- 彩色照片 + **對齊的深度圖**(15 幀中位數)+ 相機內參與序號
- 每個手臂指令、**移動中約 10 Hz 的位姿軌跡**、到位/失敗/中斷、夾爪動作
- 終端機完整輸出 `console.log`、程式 git 版本、當下使用的校正檔複本、分析結果

離線工具:`pen_quick_calib.py --analyze runs\<id>`、`depth_height.py runs\<id>`(量工件高度)。
新寫的程式也必須接上紀錄,規則見 `PROJECT_SPEC.md` §0.4、§15。**`runs/` 不上 GitHub,實作後請另行備份。**

## 不在 repo 裡的東西

`runs/`、`records/`、測試輸出影像與舊版簡報未上傳(見 `.gitignore`)。
`calibration.json` 是本組工作台的校正結果,換工作台或移動 tag 後要重新校正。

## Web 控制台

```bash
python webui\server.py        # 在 DRV-Modbus 下執行,開 http://127.0.0.1:8080 (只限本機)
```

即時 RealSense (彩色 / 深度 / 相對紙面高度圖)、手臂狀態、最新校正結果、工件與任務設定、一鍵啟動校正 (新主控台視窗,鍵盤操作照舊)、執行紀錄瀏覽、存快照。
右側有鍵盤按鍵說明,任務執行中頂端會顯示該任務的按鍵提示。「送停止指令」未驗證,實體急停為主。
