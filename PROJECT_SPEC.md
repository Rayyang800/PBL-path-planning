# 專案規格與 AI 交接文件:視覺導引塗膠/焊道路徑規劃系統

> 版本:2026-11-05 | GitHub:https://github.com/Rayyang800/PBL-path-planning (public) | 用途:讓未來的 AI(或新組員)讀完這份文件就能接手並執行任務
> 專案根目錄:`C:\Users\ray20\Desktop\Delta_Arm\`(程式在 `DRV-Modbus\`)
> 使用者語言:**繁體中文**(對使用者的輸出一律用繁體中文;程式識別字、指令用英文)

標記說明:`[已驗證]` 實機或實測確認過 | `[未驗證]` 寫了但沒實機跑過 | `[假設]` 推論,需確認 | `[待確認]` 需問使用者/教授

---

## 0. 給 AI 的必讀規則(先讀這一節)

### 0.1 硬性規則(違反可能造成設備損壞)

1. **未經使用者明確同意,絕不執行任何會讓手臂、夾爪、吸盤動作的程式**。這包含 `--go`、`send.Go_Position`、`Gripper_*`、`Suction_*`、任何寫入 Modbus 暫存器的動作。使用者說「不要動手臂」時,只能做**唯讀**操作(讀位姿、讀旗標、拍照)。
2. 所有會動的腳本**預設 dry-run**,必須加 `--go` 才會動。新寫的腳本也要遵守。
3. 速度:新腳本預設 ≤ 25,程式內上限 50。第一次實機執行用 `--speed 15 --step`(單步確認)。
4. 不要猜 Modbus 暫存器位址去寫入。沒有在第 5 節列為 `[已驗證]` 的位址,只能用**讀取**探查。
5. 手臂處於異常狀態時(到位旗標 `0x031F` ≠ 1 且手臂沒動),**不要重複送指令**;請使用者在教導器看警報並清除(見 §8)。
6. 寫入或修改使用者的檔案前先看過內容。**不要覆蓋**使用者放在 `Downloads\` 的簡報(例如 `PBL專題規劃報告_CIROS確認版.pptx`),另存新檔。
7. 回報結果要誠實:沒實機跑過就說「未驗證」,不要說「應該可以動」。

### 0.2 工作習慣(從過去對話歸納)

- 使用者喜歡**小步迭代**:先 dry-run、看辨識圖、再決定是否動手臂。
- 動手臂前使用者會手放急停旁;他用教導器(Teach Pendant)或 DRAStudio 手動移動手臂,通常**不是**由程式移動。
- 使用者偏好少問、多做,但**涉及動作的地方一定要確認**。
- 終端機輸出中文可能亂碼(Windows 主控台碼頁),這只是顯示問題:執行 Python 前先 `[Console]::OutputEncoding=[Text.Encoding]::UTF8; $env:PYTHONIOENCODING="utf-8"`。
- 暫存檔放系統指定的 scratchpad 或 `%TEMP%`,不要散落在專案資料夾;用完要清(`__pycache__`、除錯圖)。

### 0.3 每次完成工作後

更新本文件的 §7(程式清單與狀態)、§11(待辦)、§13(變更紀錄)。

### 0.4 資料紀錄是硬性規定(每次實作都要完整紀錄)

> **每一次實作(含 dry-run、失敗、中途中止)都必須留下完整、可離線重算的紀錄。沒存的資料補不回來。**

1. **會動手臂、拍照或做校正的程式,一律包在 `with RunLog(...)`(或 `run_log.run_main(...)`)裡執行**。新腳本不接紀錄 = 未完成。
2. **拍照一律用 `rs_capture`,存 `save_snapshot`(彩色 + 對齊深度 + 相機內參)**,不要只存彩色 JPG。
3. **手臂指令一律經過 `send.Go_Position` / `gripper.*`**,它們會自動記錄指令、約 10 Hz 軌跡、到位/失敗、夾爪動作;不要繞過它們直接寫暫存器。
4. 校正結果用 `CalibrationStore` 存新版本(不覆蓋);用到的校正檔與版面檔用 `log.add_input()` 複製進紀錄。
5. 紀錄內容與格式見 **§15**;離線重算工具:`pen_quick_calib.py --analyze`、`pen_marks.py`、`depth_height.py`、`run_log.load_run / load_snapshot`。
6. `runs/` 不上 GitHub(太大),**每次實作後由使用者另行備份**。

---

## 1. 專案概述

| 項目 | 內容 |
|---|---|
| 課程 | 國立中央大學機械系 林錦德老師,智慧製造 PBL 專題(第 9~16 週) |
| 題目 | 115 年 PBL:**視覺輔助加工(焊接或塗膠)**,出題單位:工研院智慧機械科技中心 |
| 本組主題 | 智慧化焊道/塗膠路徑規劃 |
| 驗證方式 | 用**夾爪夾白板筆**代替膠槍/焊槍,以筆跡驗證路徑 |
| 設備 | 台達 DRV 六軸手臂、Intel RealSense D435i(裝在手臂上)、雙 DO 夾爪、4 個 ArUco 標記的工作板 |
| 團隊 | 組長蔡典儒、黃柏翰、楊軒睿、林宥丞 |

官方題目重點(摘要):焊接/塗膠需精準辨識兩個待接合物體的接觸面位置,否則路徑偏移影響品質;工件可能任意擺放甚至公差大,傳統精準定位難以奏效,核心挑戰是**具容錯能力與現場適應性**的自動化系統。教育部計畫新增目標為虛實整合(Physical AI)。**教授之後可能提供更詳細的規定**(`[待確認]`,見 §12)。

### 1.1 已定案的決策(不要重新討論,除非使用者推翻)

| # | 決策 | 日期 |
|---|---|---|
| D1 | 用白板筆模擬塗膠/焊接,驗證的是**路徑規劃與安全流程**,不是製程 | 10-30 |
| D2 | 驗證**三種工件**:A 立體 L 型(脊線塗)、B 兩片珍珠板平面方板(縫合塗)、C 平面其他形狀(邊緣塗) | 11-04 |
| D3 | **工件類型由操作者預先指定**,系統暫時**不自動判斷類型** | 11-04 |
| D4 | 模擬軟體用 **CIROS**,與實機**明確劃分、不做軟體串接**:CIROS 驗證的結果由**人工**整理成實機的安全限制 | 11-04 |
| D5 | **不寫 AI 學習式接合面辨識**(簡報與文件都不提) | 11-04 |
| D6 | 夾爪用 DO7(開)/DO8(夾),吸盤(DO1)目前不使用 | 10-14 |
| D7 | 初始校正改用「**校正紙(四角 AprilTag 36h11)+ 筆尖探針法**」:手臂用筆在紙上畫標記,在 home 拍照,由「指令 TCP XY ↔ 標記紙面 mm」擬合仿射轉換(筆尖偏移含在內);**不依賴力/碰撞回饋**(§8 #11)。實作為 `pen_calibrate.py` + `pen_marks.py`;**實機尚未完整跑完一次** | 11-04 → 11-05 |
| D10 | 實驗室實際使用的校正紙**不是** `a4_calib_sheet.py` 印的:A4 橫放、**tag 29 mm**、ID 0 右上 / 1 左上 / 2 左下 / 3 右下(從 home 看)。版面由 `paper_layout_capture.py` 從相機量出 → `print/lab_sheet_layout.json`(間距 PnP 估 239.3 × 155.6 mm,±1%;仿射擬合會吸收比例誤差)。紙放在工作板中央,**會蓋住舊 ArUco 3、4** | 11-04 |
| D11 | **接觸判斷用影像**(仍無力回饋):自動探觸每次往下 `--probe-step`(3 mm)畫小十字 → 抬 10 mm → 往畫面下方(手臂 −X,朝基座)移動直到看見 ArUco 1/2 → 拍照與下降前比對「新變暗的像素」(任何顏色)→ 有就記錄 z。第一點由人工下降到筆尖離紙約 5 mm 再按 `a` 交給自動 | 11-05 |
| D13 | **快速筆尖校正 `pen_quick_calib.py` 取代 D11 成為主要流程**:人只找一次接觸高度 `z_touch`(工作面視為等高),其餘 4 點用同一 z、不拍照,點在第一點 ±X/±Y `--step`(10 mm)內(不跑出工件、筆不再越壓越深);最後回 home 拍一張,前後相減找墨點,擬合**旋轉+平移(比例 1:1)**。理由:D11 太慢、壓壞筆尖、點太分散會塗到底下的 A4。點最好畫在工件的膠帶上(可擦) | 11-05 |
| D12 | 舊 `calibration.json`(ArUco 1~4)在筆尖校正流程中**只當粗略換算**(預測點位、找 ROI,誤差可達 ±20 mm);精確轉換以 `calibrations/pen_paper/latest.json` 為準 | 11-04 |
| D8 | 第一個驗證工件先做 **B(兩片黑色平面板縫合塗)**,板子小於 A4 | 11-04 |
| D9 | **每次執行與校正都要留完整紀錄且不覆蓋**(彩色+深度快照、相機參數、指令與 10 Hz 軌跡、夾爪、終端輸出、git 版本、校正版本),dry-run/失敗/中止也存;格式見 §15、規則見 §0.4。用途:離線測試、重現、統計、手動餵給 CIROS、比對模擬與實機的落差 | 11-04 → 11-05 v2 |

### 1.2 範圍

**納入**:4 個 ArUco 界定的單一工作區;三種工件的塗佈線(脊線/縫合線/邊緣線);CIROS 虛擬驗證(與實機分開);白板筆描繪與拍照檢測;現有 DRV 手臂、D435i、夾爪。
**未涵蓋**:實際點膠/焊接製程(膠量、熔池);曲面工件;自動判斷工件類型;運動中的工件(輸送帶);工業級精度認證。

### 1.2.1 KPI(初訂,W9 量測基準後確認 `[待確認]`)

| 指標 | 目標 |
|---|---|
| 路徑生成成功率 | ≥ 90%(3 種工件 × 每種 20 次) |
| 平均描繪偏差 | ≤ 3 mm;最大 ≤ 5 mm |
| 實機撞機 | 0 次(安全限制須先經 CIROS 驗證) |
| 全流程 | 1 鍵:拍照 → 路徑 → 安全檢查 → 描繪 → 檢測報告,不需示教 |

注意:目前手動校正誤差約 **10 mm**,所以 3 mm 目標必須靠**偏差回饋補償**才可能達成。

---

## 2. 三種驗證工件

工件類型由操作者預先指定(D3),程式以參數/列舉接收,不做分類。

| 代號 | 工件 | 塗佈方式 | 塗佈線 | 取得方式 | 特別注意 |
|---|---|---|---|---|---|
| A | 立體 L 型工件 | 脊線塗 | L 型兩面相交的脊線 | 影像找脊線位置 + **深度相機量測高度 Z** | 工作板校正只含平面 XY;脊線在板面上方,需用深度補 Z,否則筆尖壓/離 |
| B | 兩片珍珠板(平面方板) | 縫合塗 | 兩片方板之間的縫 | 影像找縫線位置 | 白色板材對比低,縫線難分割;可用深色底板 |
| C | 平面其他形狀 | 邊緣塗 | 工件外緣,向外偏移一小段 | 影像擷取外輪廓 + 計算偏移路徑 | 輪廓平滑與重取樣、轉角處理 |

尺寸與擺放:`[待確認]`(L 型尺寸、珍珠板厚度與大小、其他形狀範例)。目前相機畫面中的實驗物是兩個圓形筆蓋(綠、藍),只用於驗證顏色辨識與座標換算,**不是**上述三種工件。

---

## 3. 系統架構

### 3.1 實機流程(6 步,閉迴路)

```
1 拍照定位   ArUco 4 角把像素換算成手臂座標(透視轉換)
2 擷取塗佈線 依「指定的工件類型」取得脊線/縫合線/邊緣線
3 生成路徑   平滑、重取樣、偏移、定起點
4 安全檢查   工作區邊界、高度下限、速度上限;超限 → 停止並重新規劃
5 手臂描繪   夾爪持筆,沿路徑移動
6 筆跡檢測   拍照比對規劃路徑,算出偏差 → 回饋修正下一次路徑
```

### 3.2 虛擬線(CIROS,與實機分開,D4)

```
1 建立工作站模型  手臂、工作板、夾爪、相機架、三種工件
2 規劃代表性路徑  三種工件各一條標準路徑
3 模擬檢查        可達性、關節限制、奇異點、碰撞
        │ (人工)把結果整理成實機安全限制:工作區邊界、安全高度、速度
        ▼
實機:低速單步描繪 → 量測筆跡偏差 → 修正路徑與安全限制
```

**不串接**的意思:不寫 CIROS ↔ Python 的自動匯入/匯出。實機程式只吃「人工設定的安全參數」(建議集中在一個設定檔,見 T08)。

---

## 4. 硬體與環境

### 4.1 手臂

| 項目 | 值 |
|---|---|
| 型號 | 台達 DRV 六軸(控制器 DCV)[待確認:具體型號] |
| 位址 | `192.168.1.1`,Modbus TCP 埠 `502`,slave/unit id = `2` |
| 筆電網卡 | 乙太網路 `192.168.1.25/24`(Realtek GbE);Wi-Fi `192.168.0.158` |
| 連線特性 | **閒置約 20 秒手臂會關閉連線**(已實測)→ 等待使用者後、讀寫前必須重連 `refresh(c)` |
| 控制軟體 | DRAStudio(專案例 `D:\Project20261007`:`main.lua` 用 `MovP(點位號)`、`DO(7/8,"ON/OFF")`、`SpdJ(10)`、`AccJ(35)`;點位存二進位 `PointFile.bin`,**未解析**) |
| 重要 | 手臂**必須在可接受 Modbus 移動指令的模式**(自動/遠端、伺服 ON、無警報)。使用者在教導器手動操作後可能把模式留在手動 |

### 4.2 相機

| 項目 | 值 |
|---|---|
| 型號 | Intel RealSense **D435i**,韌體 5.16.0.1(已是建議版本),USB 3.2 |
| 安裝 | 鎖在手臂上(eye-in-hand),從 home 位姿俯視工作板 |
| 串流 | 彩色 1280×720 BGR8 @30;深度 1280×720 z16 @30(`rs.align` 對齊彩色);深度比例 0.001 m |
| 彩色內參(10-13 量測) | fx 911.49、fy 911.71、cx 638.88、cy 364.43,畸變近 0 |
| 限制 | **同一時間只能一個程式使用相機**(關掉 RealSense Viewer 與其他 Python) |
| 比例 | 工作板平面上約 **0.575 mm/像素** |

### 4.3 夾爪與吸盤

| 項目 | 值 |
|---|---|
| 夾爪 | 雙 DO:**DO7 ON = 開**、**DO8 ON = 夾**(依 `main.lua`;使用者回報開合成功 `[已驗證]`,方向未獨立確認) |
| 吸盤 | DO1(目前不用) |
| 寫入方式 | 都是寫**同一個暫存器 `0x02FE`**,值為位元遮罩:DO1=1、DO7=64、DO8=128。一次寫入決定所有 DO → **寫夾爪值會同時關掉吸盤** |
| 夾筆治具 | 尚未設計(T03);需吸收 Z 高度誤差(彈簧緩衝) |
| 夾爪開口 | 圓形筆蓋直徑約 20~25 mm 的情況下使用;`[待確認]` 實際最大開口 |

### 4.4 軟體環境

| 項目 | 值 |
|---|---|
| OS | Windows 11 Home |
| Python | Miniconda 環境 **`delta_arm`**,Python **3.8.20**:`C:\Users\ray20\miniconda3\envs\delta_arm\python.exe` |
| 套件 | pymodbus **3.6.9**、opencv-contrib-python **4.6.0.66**(用舊 API:`cv2.aruco.Dictionary_get`、`DetectorParameters_create`、`estimatePoseSingleMarkers`)、numpy 1.24.4、scipy 1.10.1、pyrealsense2 2.55.1.6486、pynput 1.8.2、matplotlib 3.7.5、ipykernel |
| 啟用 | Anaconda Prompt:`conda activate delta_arm`;PowerShell 預設找不到 conda(使用者未開 ExecutionPolicy,**AI 不要代改系統安全設定**),AI 請直接呼叫上面的 python.exe |
| IDE | VS Code,工作區檔 `Delta_Arm.code-workspace`(直譯器已指向 delta_arm) |
| 上游函式庫 | `github.com/liaochikon/DRV-Modbus`(已 clone;本地修改見 §7.1) |
| Git | 只有 2 個上游 commit;本地新增檔案**都未納入版控**(`git status` 顯示 untracked) |
| 陷阱 | Windows 上 `cv2.imread` 讀不了中文路徑 → 用 `cv2.imdecode(np.fromfile(...))`;寫檔用 `cv2.imencode(...)[1].tofile(...)` |

---

## 5. Modbus 暫存器表

呼叫慣例:`c.read_holding_registers(addr, count, 2)`、`c.write_register(addr, value, 2)`(第 3 參數是 slave id 2)。32 位元整數用 `BinaryPayloadBuilder/Decoder(byteorder=BIG, wordorder=LITTLE)`,數值單位 ×1000(mm、度)。

| 位址 | 讀/寫 | 用途 | 狀態 |
|---|---|---|---|
| `0x00F0`(12 個) | 讀 | TCP 位姿 x,y,z,rx,ry,rz(各 int32 ×0.001 → mm/度) | [已驗證] |
| `0x031F`(1 個) | 讀 | **到位旗標**:`1` = 閒置/已到位;`2` = 忙碌**或異常卡住** | [已驗證](卡 2 的情況見 §8) |
| `0x0300` | 寫 | 指令碼:`301` = MovP、`302` = MovL、`601~612` = Jog、`0` = 停止。送出後控制器會把它清回 0 | 301、302 [已驗證](11-04 pen_calibrate 的 MovL 步進有動);Jog [未驗證];**`0` 能否中途停止 MovP/MovL [未驗證]**(`send.Stop()` 會在送出後讀回位置確認) |
| `0x0324` | 寫 | 速度(以 1~50 使用;確切單位 `[假設]` 為百分比) | [已驗證可寫入並讀回] |
| `0x0330`(12 個) | 寫 | 目標位姿(同 TCP 格式),先寫目標、再寫速度、最後寫 `0x0300` 指令碼 | [已驗證] |
| `0x02FE` | 寫 | DO 輸出位元遮罩(見 §4.3) | [已驗證] |

其餘位址(例如 `0x02FA` 觀察到值 8、`0x0318~0x0323` 狀態區)用途**未知**,不要寫入。

**旗標競態 `[已確認,已修正 11-04]`**:原本 `Go_Position` 送出後只等旗標 = 1,而旗標本來就是 1,手臂沒動也會印 `Move done!`(11-04 `go_home.py` 實際發生)。現在:位移 > 1 mm 時先等「旗標變 ≠1 或位置變化 > 0.3 mm」,2 秒內都沒有 → `CommandIgnored`(繼承 `TimeoutError`);到位後核對位置,與目標差 > 1 mm 也丟 `CommandIgnored`。≤ 1 mm 的小步不做開始檢查(呼叫端另外 sleep 0.3 s)。

---

## 6. 座標系與校正

### 6.1 基本觀念

- 手臂座標單位 mm、度;姿態 (rx, ry, rz) 在整個流程中**固定為 home 的姿態**(筆/夾爪朝下),只改 xyz。
- 工作板四角各貼一個 ArUco(`DICT_5X5_100`,ID 1~4,圖檔在 `DRV-Modbus\aruco_img\{1..4}_200.png`)。
- **像素→手臂 XY**:每次執行用「當下偵測到的 4 個 tag 中心像素」對應「校正時記錄的 tag 手臂 XY」,`cv2.getPerspectiveTransform` 得到單應矩陣(自動補償相機小位移;**不補償 tag 本身被移動**)。
- 相機在手臂上,所以**拍照時手臂必須回到校正時的 home 位姿**(`warp_pick/warp_sort` 會檢查,位置誤差 >5 mm 或角度 >2° 就拒絕)。
- **物體高度造成的視差**:轉換對應的是 tag 所在的平面。物體頂面高出平面時,影像位置會往外偏移,偏移量 ≈ 離畫面中心的距離 × 物高 ÷ (相機離板高度 − 物高)。相機離板約 **400 mm**(home 的 z=809 與板面 z=415 之差,為粗估)。例:物高 25 mm、離中心 230 mm 時約偏 **15 mm**;離中心 37 mm 時約 2.5 mm。物高 100 mm 在 230 mm 處偏移可達約 78 mm。所以**立體 L 型與靠邊的物體必須用深度修正**,不能只靠平面單應。

### 6.2 目前的 `calibration.json`(2026-10-21 10:51,最後修改)

```json
{
  "family": "5x5_100",
  "home": [364.477, -83.009, 809.429, -179.198, 0.559, -107.773],
  "tag_pixels": {"1": [835.3, 616.3], "2": [329.2, 613.0], "3": [824.0, 187.1], "4": [318.6, 185.0]},
  "tag_robot_xyz": {
    "1": [377.784, -215.061, 415.347],
    "2": [389.495,   75.695, 415.347],
    "3": [620.585, -209.743, 415.347],
    "4": [620.585,   75.695, 415.347]
  },
  "z_pick": 415.347,
  "drop": [620.585, -209.743, 567.777, -179.198, 0.559, -107.773]
}
```

- **畫面中的 tag 位置**:ID1 右下、ID2 左下、ID3 右上、ID4 左上(從 home 看)。
- **軸向對應**:影像 x 增加 → 手臂 Y 減少;影像 y 增加(往下)→ 手臂 X 減少。
- tag 中心間距:Y 方向約 **291 mm**、X 方向約 **243 mm**。
- `z_pick` = 校正時 TCP 在**板面(tag 平面)**的高度,所有 tag 的 z 也是這個值。
- `drop` 目前正好在 **tag 3 正上方**(z=567.8);`warp_pick/warp_sort` 用 `drop` 的 z 當**安全高度**。
- **home 的 z=809.4 是手臂可達範圍的最上緣**:實測從 home 水平側移到同高度的點會被手臂拒絕。大範圍水平移動請用較低的安全高度(567.8)。
- 備份:`calibration.before_swap.json` 是 tag 3、4 手臂座標對調前的**錯誤**版本(蝴蝶結形狀),保留作為反例,`warp_pick` 的凸四邊形檢查會擋下它。

### 6.3 校正品質 `[已知問題]`

1. 手動用 TCP 對準 tag 中心,誤差約 **±10 mm**(例如 tag1/tag2 的 X 差 11.7 mm)。
2. 2026-10-28 的 dry-run 中,4 個 tag 比校正時**整體往左偏約 28 px(≈16 mm)**(四個一致)。若板子/tag 被移動,**所有座標會整體偏 16 mm → 必須重新校正**(`warp_calibrate.py`)。若只是相機略不同則轉換會補償。`warp_sort` 在 `--go` 時會要求使用者輸入 `y` 確認。
3. `records/` 內只有 **1 個**手臂位姿的 tag 紀錄(snapshot #4、#5,TCP = 397.9, −93.0, 689.5, 175.4, −1.7, −108.0)。若要做真正的手眼校正(求相機→手臂轉換),需要**多個不同位姿**(建議 ≥ 8 組,含旋轉)。目前流程用的是「平面單應」,不是手眼校正。

---

## 7. 程式碼清單(`DRV-Modbus\`)

所有路徑相對於 `C:\Users\ray20\Desktop\Delta_Arm\DRV-Modbus\`。執行前:`conda activate delta_arm` 並 `cd` 到該資料夾。

### 7.1 函式庫

| 檔案 | 內容 | 備註 |
|---|---|---|
| `drv_modbus/send.py` | `Go_Position(c,x,y,z,rx,ry,rz,speed=20,mov=MovP,block=True,timeout=120,start_timeout=2,arrive_tol=1,on_poll=None)`、`Stop(c)`、`Suction_ON/OFF`、`Jog_*`;例外 `CommandIgnored`、`Interrupted` | **本地修改**:逾時、開始/到位檢查(§5);`on_poll()` 回傳 True 或 Ctrl+C → 寫 `0x0300=0` 並丟 `Interrupted("key"/"ctrl_c")`。舊腳本 `except TimeoutError` 會接到 `CommandIgnored`,但**不會**接到 `Interrupted`(會印 traceback,停止指令已送出) |
| `drv_modbus/request.py` | `Get_TCP_Pose(c)` → `(x,y,z,rx,ry,rz)`、`Get_Pose_Flag(c)` | 上游原檔;讀取失敗回傳 `None` 並印 `request error!` |
| `drv_modbus/gripper.py`(新) | `Gripper_Open(c)`、`Gripper_Close(c)`(寫 `0x02FE` = 64/128,sleep 1 秒) | 見 §4.3 |
| `drv_modbus/conn.py`(新) | `refresh(c)`:關閉並重連 | 任何「等待使用者/長時間思考」之後、讀寫之前都要呼叫 |
| `landmark/aruco.py`、`landmark/util.py` | 上游 ArUco 工具 | |
| `realsense/realsense.py` | 上游相機封裝 | **import 時就啟動 pipeline**(沒接相機會報 `No device connected`;已有其他程式占用也會失敗)。新程式請自己建 pipeline |

### 7.2 腳本(皆為新增,未版控)

| 檔案 | 功能 | 會動手臂? | 狀態 |
|---|---|---|---|
| `warp_calibrate.py` | 互動式校正:使用者用教導器移到 home → 記錄 tag 像素與 home;再依序把 TCP 對準 tag 1~4 中心、吸取高度、放置點;寫 `calibration.json` | **否**(只讀) | [已驗證] 使用者已用它產生目前的校正檔 |
| `warp_pick.py` | `--color <c>` 找單一顏色物體 → 手臂 XY;`--go` 才取放(吸盤→已改**夾爪**) | `--go` 才動 | dry-run [已驗證];`--go` 曾在 10-21 卡住(手臂拒絕指令,見 §8);夾爪取放流程**未完整驗證** |
| `warp_sort.py` | 依顏色分類:**綠色筆蓋 → tag 1/2 中點**、**藍色 → tag 3/4 中點**;夾取高度 = `z_pick`+10−`--pick-down`(10)、放置高度 = `z_pick`+10−`--place-down`(9);安全高度 = `min(home z, drop z)` | `--go` 才動 | dry-run [已驗證](10-28 / 11-04 皆辨識正確);實機 **[未驗證]** |
| `go_home.py` | 回到 `calibration.json` 的 home,預設速度 **40**(使用者改的),上限 50;**低於安全高度 567.8 時先垂直 MovL 抬起**再 MovP 回 home;沒動會報錯 | 是(需 Enter 確認) | [已驗證] 11-04 從 (364,−7,709) 回 home 成功(第一次手臂沒執行指令,教導器處理後第二次成功) |
| `gripper_ctl.py` | 鍵盤即時控制夾爪:`o` 開、`g` 夾、`s` 讀 DO、`q`/Esc 離開(不需 Enter;Windows `msvcrt`) | 只動夾爪 | [未驗證] |
| `paper_layout_capture.py` | 從相機(或 `--image`)量實際校正紙的 tag 排列與方向 → `print/lab_sheet_layout.json`;`--tag-mm` 必填,`--dx/--dy` 可填尺量間距(否則 PnP 估) | 否 | [已驗證] 11-04 實拍:角點擬合殘差 0.19 mm |
| `pen_calibrate.py` | 筆尖探觸校正(D7、D11)。預設 dry-run(存 `pen_calibrate_plan.png`)。`--go` 流程:移到第一點上方 → (可選)開夾爪裝筆、倒數 3 秒夾緊 → 每點找接觸高度 → 畫大十字(半長 10 mm)→ 全部完成回 home 拍照 → `pen_marks.analyze` → 存 `calibrations/pen_paper/`。點位 `--pattern cross`(預設,紙中心上右下左各偏 `--step` 20 mm)/ `left` / `spread`,`--shift-x/y`;只允許在 tag 內緣再縮 12 mm 的範圍內。探觸 `--probe auto`(預設)/`manual`;`--look base`(預設)/`lift`/`home`;`--probe-step 3`、`--auto-max 8`、`--base-lift 10`、`--base-tags 1 2`、`--ink any`。`--resume <run>` 接續中斷的紀錄(用先前墨點修正偏移)。**按鍵**:Ctrl+C = 停止→垂直抬起→回 home 結束;空白/Esc/p = 停止並選單(r 重做、m 改人工、s 跳過、h 回 home、q 結束、o/g 夾爪);移動中 `o`/`g` 即時開夾夾爪(手臂不停) | `--go` 才動 | 部分實機:MovL 步進、裝筆、人工下降 [已驗證];自動探觸 `look=lift/home` 跑過但**未成功**(§8 #14、#15);`look=base`、停止鍵、選單、完整分析 **[未驗證]** |
| `pen_quick_calib.py` | **快速筆尖校正(D13,主要流程)**。dry-run 存 `pen_quick_plan.png`。`--go`:home 拍前照 → 第一點上方 → (可選)裝筆 → 人工下降(`x 2`/`y -3` 微調,最多 5 mm、會先抬 2 mm)按 `c` 記 `z_touch` → 其餘 4 點:抬 `--hop` 3 mm、移 `--step` 10 mm、下到 `z_touch - --press`(預設 0)停 `--dwell` 0.5 s → 抬到安全高度回 home 拍後照 → `analyze()`:前後相減(變暗量加權中心)、`match_pattern` 依粗略方向配對、`fit_rigid`(可鏡像)、與舊 ArUco 旋轉交叉檢查(>5° 警告)、估計離第一點 100 mm 的誤差 → 存 `calibrations/pen_paper/`(`method: quick`)。`--start-paper X Y` 第一點紙面座標;`--surface-h` 點在工件上時填工件厚度(視差);`--analyze runs\<id>` 重新分析。按鍵:Ctrl+C / 找第一點時的空白鍵 → 抬起回 home 結束;後 4 點中空白鍵 → r 重做 / s 跳過 / h 回 home 用已完成點分析 / q 結束 | `--go` 才動 | 合成測試 PASS(`tests/test_quick_calib.py`:5 點全對,72 mm 外誤差 ≤0.1 mm);**實機未跑(連 dry-run 都還沒,11-05 手臂離線)** |
| `pen_marks.py` | 離線分析:`find_cross` / `find_dot`(前後照片對齊相減 `new_ink_mask`)、`new_ink_blobs`(同一位置前後照片的新筆跡)、`fit_affine`、`fit_rigid`(比例 1:1,可鏡像)、`match_pattern`(中心+方向配對)、`fit_plane`、`analyze`、`overlay`;CLI `python pen_marks.py runs\<pen_calib>` 重新分析 | 否 | PASS(`tests/test_pen_marks.py`、`tests/test_new_ink.py`);真實照片:前後相減找到紅點 39 px 與灰點 9 px |
| `tests/test_pen_marks.py` / `test_new_ink.py` / `test_stop.py` | 十字與墨點擬合(含 7 mm 外的舊髒點)/ 新筆跡偵測 / 假 Modbus 驗證停止指令會寫入 | 否 | PASS |
| `record_apriltags.py` | 即時預覽 ArUco/AprilTag,空白鍵記錄相機座標 + 手臂 TCP → `records/*.csv/.jsonl` | **否** | [已驗證] |
| `color_detect.py` | HSV 多色塊偵測(red/orange/yellow/green/blue/purple),可即時或 `--image` | 否 | [已驗證] |
| `red_detect_demo.py` | 課程投影片「紅色輪廓」範例的實作 | 否 | [已驗證] |
| `watch_status.py` | 每秒印旗標與位姿(唯讀),Ctrl+C 結束 | 否 | [已驗證] |
| `a4_calib_sheet.py` | 產生列印用 A4 校正紙:向量 PDF(`print/a4_calib_sheet.pdf`)+ 預覽 PNG + 版面 JSON(`print/a4_layout.json`)。預設直式 210×297、tag 28.0 mm、距邊 12 mm、36h11 ID 0~3(TL,TR,BR,BL)、含 100 mm 比例尺。`--json-only --tag-mm <實測>` 可在實測尺寸不同時只重寫版面 | 否 | PDF 預覽已檢視;**尚未列印實測** |
| `paper_frame.py` | `PaperFrame`:偵測紙上 4 tag → 影像↔紙面 mm 單應(16 角點擬合);`pix_to_paper(uv, h)` 含**板厚視差修正**(用 PnP 估相機中心);`describe()` 印殘差與相機離紙高度 | 否 | 合成場景驗證;**無真實影像** |
| `seam_boards.py` | 兩片黑板縫合線擷取:Otsu 分出暗板 → minAreaRect → 找相對的兩邊 → 中線(紙面 mm 折線)。`--image` 或 `--camera`(只讀相機)、`--board-h <板厚mm>`、`--label`。**預設存成 `runs/` 紀錄**(§15);`--no-save` 才寫 `seam_result.json` / `seam_overlay.png` 到 `--out` | 否 | 合成場景驗證;**無真實影像** |
| `tests/test_seam_synthetic.py` | 合成場景測試(針孔相機渲染紙+tag+有厚度的板,與真值比較);輸出 `tests/out/case_*.png`(疊圖)與 `case_*_raw.png`(原始影像) | 否 | PASS:理想誤差 ≈0.4 mm;視差修正後 ≤0.3 mm |
| `run_log.py` | `RunLog`(執行紀錄,context manager,例外也會留 failed/aborted 紀錄)、`CalibrationStore`(校正版本庫,`import_legacy` 匯入舊檔)、`load_run` | 否 | PASS(`tests/test_run_log.py`);目前只有 `seam_boards.py` 使用 |
| `cam_orient.py` | 相機反裝 180 度:`orient_frame()`(auto 依 tag 排列判斷)、`decide()`、`rotate()`、`raw_point()`、`rotated_K()`;`--set on/off/auto` 寫 `camera.json` | 否 | PASS(`tests/test_cam_orient.py`,合成影像);實機未測 |
| `tests/test_run_log.py` | 紀錄模組測試(使用暫存資料夾,不碰實際 `runs/`) | 否 | PASS |
| `test_small_move.py` | 往下 10 mm 再回,速度 5,每秒印旗標(診斷用) | 是 | 10-21 執行時手臂**完全沒動**(旗標卡 2) |
| `test_gripper_center.py` | 移到中心點再開合夾爪 2 輪 | 是 | 使用者回報開合成功 |
| `pick_place_p1_p4.py` | 依 main.lua 的 p1~p4 + 中心點取放(座標為早期手動記錄) | 是 | 使用者回報「沒問題」,**無日誌佐證**。⚠ 使用者把預設速度改成 **70**、上限 **100**(其他腳本上限是 50) |
| `warp_suction_example.py` 等 | 上游範例,**座標是原作者的,不可直接用** | — | 僅供參考 |

### 7.3 `warp_sort.py` 辨識規則(目前)

- 工作區 = 4 個 tag 中心圍成的四邊形,**扣掉各 tag 本體**(外擴 35%)。
- 綠色 HSV:H 35–92、S ≥ 55、V ≥ 40(筆蓋偏青綠,H≈83、S≈65;板面是低飽和灰綠 S<35,污點直徑 16~19 px)。
- 藍色 HSV:H 95–130、S ≥ 110、V ≥ 60。
- 候選篩選:面積 500~6000 px、外接圓直徑 28~80 px;多個候選取面積最大。
- 程式內保留黑色的處理(V≤100、閉合核 13×13、排除藍色暗部),但 `JOBS` 目前**沒有啟用黑色**(原本黑→tag 1/2,後來使用者改成綠色)。圓形筆蓋從上方看可能只剩細環,所以要用大閉合核補成實心。
- 目標點 = 兩個 tag 手臂座標的中點:綠 → (383.64, −69.68)、藍 → (620.59, −67.02)(依目前校正檔)。
- 保護:拒絕執行的條件 = 不在 home、旗標≠1、tag 沒看全、4 tag 手臂座標排成交叉形(凸性檢查)、物體超出工作區(+30 mm)、夾取/放置高度低於 `z_pick`、安全高度與基準高度差 <50 mm。

### 7.4 執行範例

```powershell
# 唯讀:看辨識結果(不動手臂)
python warp_sort.py
# 第一次實機:低速、單步確認、只做一個
python warp_sort.py --go --speed 15 --step --only green
# 唯讀:持續監看手臂狀態
python watch_status.py
# 低速回 home
python go_home.py --speed 10
```

---

## 8. 已知問題與陷阱(實際踩過的)

| # | 現象 | 原因 | 處理 |
|---|---|---|---|
| 1 | `request error!` / 讀不到位姿 | 手臂**閒置約 20 秒**就關閉 Modbus 連線 | 讀寫前 `refresh(c)`;已在各腳本加入 |
| 2 | 送出移動後手臂不動,旗標一直 **2**,後續指令也不動 | 手臂收到指令後進入異常/拒絕狀態(例如目標在可達範圍外)並**鎖住**,需在教導器清除;10-28 起旗標已恢復 1 | 請使用者看教導器警報與模式(自動/遠端、伺服 ON);AI **不要**反覆送指令;用 `watch_status.py` 觀察旗標何時回 1 |
| 3 | 從 home(z=809)水平移動被拒 | home 在可達範圍最上緣 | 安全高度改用 `drop` 的 z(567.8) |
| 4 | `realsense.py` import 就報 `No device connected` | 模組載入時啟動 pipeline | 自建 pipeline;確認沒有別的程式占用相機 |
| 5 | 辨識結果離譜(X=13735 mm) | 校正檔中 tag 3、4 手臂座標對調,轉換矩陣成蝴蝶結 | 已修正並加凸性檢查;重新校正時對照 §6.2 的 tag 位置圖 |
| 6 | OpenCV 讀不了中文檔名 | Windows | 用 `imdecode(np.fromfile())` |
| 7 | `0x02FE` 寫夾爪值會關吸盤 | 同一個位元遮罩暫存器 | 不要同時用吸盤與夾爪 |
| 8 | 圓形筆蓋從上方只剩黑色細環,偵測不到 | 形態學開運算把細環切碎 | 黑色用較大的閉合核(已在 `warp_sort.py`) |
| 9 | 輸出中文亂碼 | 主控台碼頁 | 見 §0.2 |
| 10 | 長中文單行讓 Python 3.8 報 `Non-UTF-8 code`(檔案實為合法 UTF-8) | 推測是 3.8 分詞器處理長行的問題 | 檔案開頭加 `# -*- coding: utf-8 -*-` 後正常 |
| 11 | **沒有可用的接觸/力/碰撞訊號** | Modbus 只確認到位姿(`0x00F0`)與到位旗標(`0x031F`);位置控制下,筆碰到紙不會有任何訊號。DRV 若有碰撞保護 `[假設]` 也是急停式、需教導器復歸(10-21 已見過鎖死情形),不能當感測器 | **不要寫「往下直到有回饋」**。改用人在迴路步進下降、或用筆跡/影像推算接觸高度;查 DRAStudio 手冊確認是否有碰撞偵測 `[待確認]` |
| 12 | 相機轉 180 度安裝 | 影像上下左右顛倒;像素→手臂的 homography 由當下 4 個 tag 求得,**本身不受影響**,但畫面方向、`warp_sort` 的 tag 偏移檢查會不同,且改裝通常會改變相機相對夾爪的位置 | 用 `cam_orient.py`(取像後先轉正;auto 依 tag 排列與 `calibration.json` 比較判斷);改裝後跑 `warp_calibrate.py --pixels-only`。有用內參的地方(深度反投影、PnP)要用 `raw_point()`/`rotated_K()`;`paper_frame`/`seam_boards` 用原始影像做 PnP,本來就不受方向影響,未接入 |
| 13 | `go_home.py` 印「已到達 home」但位置沒變 | 手臂沒執行指令(旗標維持 1,速度暫存器有寫入);加上旗標競態(§5) | 已修正 `Go_Position`。手臂不執行時看教導器:模式(外部/遠端)、伺服 ON、警報 |
| 14 | 自動探觸「抬高拍照」看不到筆跡 | 相機裝在夾爪**前側**(+X 方向):手臂在紙上方時,相機視野在筆尖前方;抬 40/80 mm 拍到的是紙外的鋁軌,紙只在畫面下緣(`runs/20261104-195948_pen_calib/images/touch_0_hit.jpg`)。結果程式一路往下 9 mm | 改成 `--look base`(往基座方向移到看見 ArUco 1/2)或 `--look home`(慢但確定看得到整張紙) |
| 15 | 自動探觸把**人的手**當成紅點 | 紅色 HSV 範圍包含膚色;當時 ROI 是整張畫面 | ROI 限制在紙上可點範圍 + 預定點 35 mm 內(home)、或排除畫面邊緣 30 px 並記住第一次的位置(base/lift);最小面積 12 px。**那次 run 的 z=468.8 是錯的,不要拿來 `--resume`** |
| 16 | 只找紅色會漏掉筆跡 | 筆輕碰留下的是灰點,不是紅點(紅點反而是那次出錯時留下的) | `--ink any`:只看「變暗 > 25」;探觸改畫小十字(半長 4 mm)比點容易辨識 |
| 17 | 原本 Ctrl+C 不會讓手臂停 | Ctrl+C 只中斷 Python,控制器會把已收到的移動走完 | `Go_Position` 中斷時寫 `0x0300=0`,再讀兩次位置確認;**仍未實機驗證,實體急停是唯一可靠的停止方式** |
| 18 | D11 自動探觸實際使用的問題(使用者回報 11-05) | ① 慢:相機無法即時看筆尖,每步都要移動拍照;② 沒有力回饋,手臂一直往下壓,**白板筆尖常被壓壞**,點有粗有細;③ 點分散在紙上,常超出工件或快出 A4,塗到底下的紙 | 改用 D13 `pen_quick_calib.py`:只找一次高度、固定 z、點集中在 ±10 mm |
| 19 | 墨點中心整體偏 ~0.8 mm(合成測試發現) | 用 2×2(偶數)核做形態學開運算,中心偏 1 px | 不做形態學,改用變暗量加權中心 |

---

## 9. 專案管理資訊

### 9.1 課程時程(官方) 與本組對應

週三進度報告。日期假設 W16 = 12/23(週三),往前推 `[待確認]`。W12 的 11/26 是**週四**(使用者提供的初型報告日期),與週三 11/25 不一致 → 簡報標「11/25~26」。

| 週 | 日期 | 官方進度 | 繳交 | 本組當週產出 |
|---|---|---|---|---|
| W9 | 11/4 | 專題說明與分組、規格與規劃討論 | 作業7 專題規劃報告 | 確認規格與目標、治具草圖、塗佈線擷取初版 |
| W10 | 11/11 | 專題進度1:工作站設計、規劃與虛擬驗證 | 作業8 進度報告1(提案修正) | CIROS 工作站模型、路徑規劃 v1 |
| W11 | 11/18 | 專題進度2:同上 | 作業9 進度報告2 | 代表性路徑與碰撞檢查、治具完成 |
| W12 | 11/25~26 | **查核點1**:虛擬工作站設計與驗證報告 | 作業10 虛擬工作站報告 | 三種工件的影像處理與路徑生成、CIROS 驗證報告(初型) |
| W13 | 12/2 | 專題進度3:實體工作站部署 | 作業11 進度報告3 | 治具上機、平面與筆尖校正、低速首次實機描繪 |
| W14 | 12/9 | 專題進度4:實體工作站部署 | 作業12 進度報告4 | 筆跡檢測、偏差補償、安全限制調整、一鍵流程 |
| W15 | 12/16 | **查核點2**:實體部署與測試報告 | 作業13 實體工作站整合報告 | 3 種工件 × 20 次實驗與統計 |
| W16 | 12/23 | PBL 專題成果報告與技術展示 | 作業14 期末報告 | 期末報告、現場展示、專案檔案 |

### 9.2 每次進度報告的固定內容(課程規定)

1. 專題目標與進度規劃(甘特圖) 2. 上次報告摘要 3. **本次進度內容:每個人都要列出實際進行的工作項目,並以文字說明工作心得** 4. 下次預期成果。

### 9.3 期末繳交(課程規定)

| 項目 | 規格 | 主責(暫定) | 預計(暫定) |
|---|---|---|---|
| 口頭報告投影片 | PowerPoint;需涵蓋製程特性、規格、製造程序、工作站構成、時序控制與模擬動畫 | 蔡典儒 | W16 |
| 書面報告 | Word、制式規格、**內含個人 300 字心得** | 蔡典儒彙整、每人自寫 | 初稿 W15、定稿 W16 |
| 專案相關檔案 | CIROS 專案檔、Python 程式、視覺模組 | 楊軒睿整理、林宥丞建模 | W15 |
| 模擬動畫影片 | MP4、5 分鐘內、VGA 以上、≤ 100 MB | 林宥丞 | 腳本 W14、成片 W15 |
| 專案海報 | PDF | 黃柏翰 | W15 |

### 9.4 WBS 與責任分派(暫定,可依專長調整)

| 編號 | 工作項目 | 主責 | 協助 |
|---|---|---|---|
| 1.1 | 週報與簡報 | 蔡典儒 | 全員 |
| 1.2 | 期末繳交文件 | 蔡典儒 | 全員 |
| 2.1 | 夾筆治具 | 林宥丞 | 蔡典儒 |
| 2.2 | 筆尖偏移校正 | 林宥丞 | 黃柏翰 |
| 3.1 | 平面校正 | 黃柏翰 | 林宥丞 |
| 3.2 | 塗佈線擷取 | 黃柏翰 | 楊軒睿 |
| 3.3 | 筆跡擷取 | 黃柏翰 | 林宥丞 |
| 4.1 | 平滑與重取樣 | 楊軒睿 | 黃柏翰 |
| 4.2 | 偏移與起點 | 楊軒睿 | 蔡典儒 |
| 5.1 | CIROS 工作站建模 | 林宥丞 | 蔡典儒 |
| 5.2 | 碰撞檢查 | 楊軒睿 | 林宥丞 |
| 5.3 | 安全限制設定 | 蔡典儒 | 楊軒睿、林宥丞 |
| 6.1 | 軌跡執行 | 蔡典儒 | 楊軒睿 |
| 6.2 | 安全與例外 | 黃柏翰 | 蔡典儒 |
| 6.3 | 一鍵流程 | 蔡典儒 | 黃柏翰、楊軒睿 |
| 7.1 | 偏差補償 | 楊軒睿 | 黃柏翰、蔡典儒 |
| 7.2 | 重複性實驗 | 林宥丞 | 蔡典儒、黃柏翰、楊軒睿 |

甘特(以週計):規格與 KPI [W9];夾筆治具 [W9–W11];塗佈線擷取 [W9–W11];路徑規劃 [W10–W12];CIROS 工作站建模 [W10–W11];代表性路徑與碰撞檢查 [W11–W12];實體部署與校正 [W13–W14];筆跡檢測與偏差補償 [W13–W14];一鍵流程與例外 [W13–W14];重複性實驗 [W14–W15];期末文件 [W14–W16]。

---

## 10. 風險登錄

| # | 風險 | 等級 | 影響 | 對策 |
|---|---|---|---|---|
| 1 | 定位誤差過大 | 高 | 手動校正約 10 mm、工作區曾位移約 16 mm,偏差目標難達成 | 固定工作板與 tag、標準化校正流程、用筆跡偏差做補償 |
| 2 | 筆尖接觸不穩 | 高 | 高度太低壓筆、太高斷線;L 型脊線高度需另行量測 | 治具加彈簧緩衝;深度相機量測脊線高度;實測調整 |
| 3 | 手臂軌跡與通訊不穩 | 中 | 逐點等待到位使轉角不順;連線閒置約 20 秒斷線;指令被拒後手臂卡異常狀態 | 動作前重連、移動逾時保護、狀態監看;備案:輸出 RL 程式由控制器執行 |
| 4 | 影像分割不穩定 | 中 | 陰影、污漬、光線變化;白色珍珠板縫線對比低 | 高對比乾淨背景、固定光源、面積與形狀檢查;縫線處用深色底板 |
| 5 | 虛實落差導致撞機 | 高 | CIROS 與實機不串接,安全限制靠人工整理,模型不準就可能撞板面、治具、線材 | 以實測校正模型尺寸與座標;Z 下限與工作區保護;實機首跑低速單步、手放急停 |
| 6 | CIROS 的機型支援 | 中 | 若機型庫沒有 DRV 型號,虛擬驗證無法如期完成 | W9 確認;備案自建手臂模型,或用 Python 做簡化碰撞檢查 |
| 7 | 時程與設備限制 | 高 | 手臂與相機僅實驗室可用且四人共用,實體階段只有 W13~W15 三週 | 前四週以虛擬與離線開發為主;排定實機使用時段;每週里程碑檢視 |
| 8 | 需求變更 | 中 | 教授之後可能提供更詳細的規定 | 模組化設計;每週確認需求與進度 |

---

## 11. 待辦清單(給 AI 的工作包)

分類:`[唯讀]` 不會動設備;`[動作]` 需使用者同意並在場。建議順序:T01 → T02 → T05 → T08 → …

### T01 [唯讀] 重新校正前的現況檢查
- 目的:確認 tag 位移(約 28 px)是板子動了還是相機動了。
- 做法:跑 `python warp_sort.py`(dry-run)看「tag 位置差」;請使用者說明板子有沒有被移動。若板子動了 → 通知使用者重新執行 `warp_calibrate.py`(**使用者操作**)。
- 完成條件:使用者確認校正檔對應目前板子位置。

### T02 [唯讀→動作] 驗證「實機能不能動」
- 目的:10-21 之後手臂是否已恢復接受 Modbus 移動指令。
- 做法:`python watch_status.py` 確認旗標 =1 → 使用者同意後 `python test_small_move.py`(下 10 mm、速度 5)。
- 完成條件:位置確實變化 10 mm 並回原位;若失敗,依 §8 #2 處理並記錄。

### T03 (2.1, 2.2) 夾筆治具與筆尖校正 — 林宥丞
- 需要:設計/製作治具(彈簧緩衝),量測筆尖相對 TCP 的偏移(x,y,z),寫入設定檔。
- AI 可做:寫「筆尖偏移補償」函式(輸入 TCP 目標點 + 偏移 → 實際送出的點)、整理量測步驟。

### T04 (3.2) 塗佈線擷取 — 黃柏翰
- 建議新檔 `workpiece.py`:`WorkpieceType` 列舉(L_RIDGE/SEAM/EDGE)與 `extract_coating_line(frame_bgr, depth, K, calib, wtype)` → `(N,3)` 手臂座標點列。
- C(邊緣):分割(顏色/閾值)→ 最大輪廓 → 平滑。
- B(縫合):找兩片板之間的縫(白板低對比:先試邊緣/Hough/深色底板);輸出直線點列。
- A(脊線):影像找脊線 + 深度取 Z;需決定脊線偵測方法(深度梯度/法向變化)。
- 完成條件:對每種工件離線影像輸出的點列與人工標註的差 ≤ 2 mm(像素→mm 以 0.575 mm/px)。

### T05 (4.1, 4.2) 路徑規劃 — 楊軒睿
- 建議新檔 `path_planner.py`:`plan_path(points_xyz, step_mm=2.0, offset_mm=0.0, closed=False)` → 平滑(移動平均/Savitzky-Golay)、依弧長重取樣、向外偏移(`shapely` 或 `cv2`)、選起點(最近 home / 轉角處)。
- 完成條件:對三種輪廓輸出點距均勻(誤差 <10%)、轉角不產生自交;純函式,可離線單元測試。

### T06 (6.1) 軌跡執行 — 蔡典儒 `[動作]`
- 問題:`Go_Position` 是逐點「寫目標 → 等到位」,轉角不順。
- 研究:查台達 DRV 是否有連續運動/融合(blending)指令的 Modbus 位址(**目前未知,不要猜**);備案:產生 DRAStudio 的 RL 程式(`main.lua` 格式)由使用者下載到控制器執行;或調整點距。
- 完成條件:沿 100 mm 方形路徑連續描繪,角落速度不為零且筆跡連續。

### T07 (6.2) 安全模組 — 黃柏翰
- 現況:安全檢查散落在 `warp_pick.py`、`warp_sort.py`、`pick_place_p1_p4.py`(且速度上限不一致:50/50/100)。
- 建議新檔 `safety.py`:`SafetyLimits`(工作區邊界、Z 下限、安全高度、速度上限、home 容許誤差)與 `check_pose(pose)`、`check_path(poses)`;所有會動的腳本改用它。
- 完成條件:所有腳本共用同一份限制;超限一律拒絕並說明原因。

### T08 (5.3) 安全參數設定檔 — 蔡典儒
- 建議 `safety_limits.json`,欄位來自 CIROS 驗證結果(**人工填寫**,D4):`workspace_xy_bounds`、`z_floor`、`z_safe`、`speed_cap`、`notes`(來源/日期)。
- 完成條件:`safety.py` 讀取此檔;檔案有版本與來源註記。

### T09 (5.1, 5.2) CIROS 建模與檢查 — 林宥丞 / 楊軒睿
- AI 可做:整理建模所需尺寸清單(手臂 DH/底座位置、工作板、夾爪、相機架、三種工件)、**代表性路徑座標表**(從 `calibration.json` 與 T05 產出)、碰撞檢查清單、查核點1 報告大綱。
- AI **不要**寫 CIROS ↔ Python 自動串接(D4)。

### T10 (3.3, 7.1) 筆跡檢測與偏差量測 — 黃柏翰 / 楊軒睿
- 建議 `pen_check.py`:描繪前後各拍一張(手臂回 home),差分取筆跡 → 單應矩陣轉手臂 XY → 與規劃路徑算偏差(平均、最大、各點)。
- 偏差定義:筆跡骨架上每點到規劃折線的最近距離(mm)。輸出 CSV + 疊圖。
- 回饋補償:先做全域平移/旋轉修正(最小平方),再看是否需要更細的補償。
- 完成條件:對已知偏移的合成資料能回復偏移量(誤差 <0.5 mm)。

### T11 (6.3) 一鍵流程 — 蔡典儒 `[動作]`
- `run_job.py --type {l_ridge|seam|edge} --go --speed N`:拍照 → 擷取 → 規劃 → 安全檢查 → (dry-run 顯示並確認)→ 描繪 → 檢測 → 報告。
- 完成條件:一個指令完成並產出偏差報告;任何一步失敗都停止並說明。

### T12 (7.2) 重複性實驗 — 林宥丞 `[動作]`
- 3 種工件 × 20 次,記錄成功與否、平均/最大偏差、失敗原因;輸出統計表與圖。
- 實驗日誌格式:每次一列(時間、工件類型、擺放位置、成功/失敗、偏差、備註)。

### T13 [唯讀] 手眼校正資料(選配)
- 若 KPI 需要更好精度:用 `record_apriltags.py` 在多個手臂位姿記錄(目前只有 1 組),求相機→手臂轉換(`cv2.calibrateHandEye`),比較與單應法的精度。

### T14 報告與文件
- 每週進度報告投影片(§9.2 格式)、期末 Word 報告、海報、影片腳本。簡報風格見 §14。

### T15 [動作] 初始校正:校正紙 + 筆尖探針(D7、D10、D11)— **進行中**
- **現況(11-05)**:`pen_calibrate.py`(D11 自動探觸)實機一次都沒做完,問題見 §8 #18。**改用 `pen_quick_calib.py`(D13)**,合成測試已過,實機尚未執行。下一步:
  ```powershell
  python go_home.py                                  # 低於安全高度會先垂直抬起
  python pen_quick_calib.py --pen-loaded             # dry-run,看 pen_quick_plan.png:第一點要落在工件 (膠帶) 上,離邊 ≥ step+5 mm
  python pen_quick_calib.py --pen-loaded --start-paper <X> <Y> --surface-h <工件厚度>   # 第一點不在工件上時調整
  python pen_quick_calib.py --go --pen-loaded        # 人工下降,剛好點出顏色按 c,其餘自動
  ```
  第一次跑時:手放急停旁;在後 4 點移動中按一次空白鍵,確認 `0x0300=0` 真的能停(§8 #17)。
- **完成條件**:5 點都對應到、RMS ≤ 0.5 mm、與舊 ArUco 旋轉差 < 5°、`calibrations/pen_paper/latest.json`(`method: quick`)產生;之後 T16 用它把紙面縫合線轉成手臂 XY(並用同一個 `z_touch`)。
- 注意:點集中在 ±10 mm,旋轉誤差會放大到遠處(程式會印「離第一點 100 mm 估計誤差」);要畫的範圍大時加大 `--step`(工件夠大的話)。
- 以下為 11-04 規劃時的原始描述(部分已被上面取代,例如改用實驗室的 29 mm 紙、改成自動探觸):
- **前置**:列印 `print/a4_calib_sheet.pdf`(實際大小 100%,用尺量 100 mm 比例尺與 tag 邊長;實測不是 28.0 就 `python a4_calib_sheet.py --json-only --tag-mm <實測>`)。紙平貼在工作板中央、四邊貼膠帶,**不要遮住外圍舊 ArUco**。從 home 拍照確認 `PaperFrame.describe()` 顯示 4/4 tag、殘差 RMS < 0.5 mm。
- **動作流程(每步需使用者確認,預設 dry-run)**:`go_home` → 移到「裝筆位置」(安全高度,例如紙中心上方 z=567.8,**不要在 home 最高點裝筆**)→ 夾爪開 → 使用者放筆並**手離開** → Enter、倒數 3 秒後夾緊 → **人在迴路下降找接觸**:從 z_plane+30 mm 起,以 5→2→1→0.5 mm 步進、每步按 Enter,使用者目視確認筆尖是否碰紙;**硬下限 z_plane−2 mm**(`z_plane` = `calibration.json` 的 `z_pick`)→ 記錄 `z_touch`,抬 3 mm → 在紙面分散的 ≥ 3 個點(建議 5 個)各畫一個「+」(兩筆各約 10 mm,中心 = 指令的 TCP XY)→ 回 home → 拍照。
- **離線分析**(可先寫,用合成影像測):偵測紙上 4 tag → 紙面 mm;偵測各「+」中心 → 紙面 mm;與指令的手臂 XY 配對擬合 2D 轉換(相似/仿射)→ 得到**筆尖 XY 偏移、轉換殘差(mm)、z_touch 的平面擬合(紙面傾斜)**;輸出 `pen_calibration.json`。
- **完成條件**:殘差 RMS ≤ 1 mm;後續描繪腳本讀取 `pen_calibration.json`。
- **注意**:沒有力/碰撞回饋(§8 #11),不要做自動下探。剛性筆硬壓會鎖死手臂或損壞筆;建議彈簧筆套或用筆頭較軟的筆,並以 z_touch 為基準只壓 ≤0.3 mm。

### T16 [唯讀→動作] 先導實驗:兩片黑色平面板縫合(D8,工件 B)
- **擺法**:兩板放在 A4 紙上、不遮住 tag,兩板之間**留 ≥ 2~3 mm 白紙縫**(合成測試:縫 ≤1 mm 會被當成一塊;1.5 mm 以上可分,實拍光線較差,所以建議 ≥2~3 mm)。用游標卡尺量**板厚 t**。
- **唯讀**:`python seam_boards.py --camera --board-h <t>` → 檢查 `seam_overlay.png`、`seam_result.json`(紙面 mm 縫合線)。
- **動作(T15 完成後)**:縫合線經 `pen_calibration.json` 轉成手臂 XY;Z = 紙面 z + t + 落筆量;低速單步描繪;拍照量偏差。
- **筆跡可見度**:黑板上看不到黑/藍白板筆 → 用白色/銀色油性筆,或畫在板邊內側 3 mm 的偏移線,或貼白紙條;先在邊角料試筆。
- **視差**:板厚 15 mm、相機中心離縫較遠時,不修正誤差可達 **2.2 mm**(合成測試),填入 `--board-h` 後 ≤ 0.3 mm。板厚必須量。
- 已知限制:板子側壁在影像周邊會被當成暗區,輪廓邊緣可能外擴 1~2 mm(兩側對稱時不影響中線);量到的縫寬偏大約 0.4 mm(合成)。

### T17 紀錄與匯出(D9)— 蔡典儒
- **把 `RunLog` 接到所有動作/校正腳本**:`warp_calibrate`、`warp_sort`、`warp_pick`、`pick_place_p1_p4`、`go_home`,以及之後的 `pen_calibrate`、`run_job`。每次 `Go_Position` 前後各記一筆 `cmd_move` 與回讀 `tcp`(含旗標),夾爪動作記 `cmd_gripper`;dry-run 也存(`finish(status="dry-run")`)。建議寫一個薄包裝 `logged_move(log, c, pose, speed, mov)` 避免每支腳本重複。**接動作腳本需使用者在場做實機測試,先用 dry-run 驗證格式。**
- **腳本改讀版本庫**:新腳本讀 `calibrations/<name>/latest.json`,並把「用了哪一版」寫進 run 的 `meta.inputs`。既有腳本仍讀根目錄 `calibration.json`(尚未改)。
- **`export_path.py`(新)**:輸入一筆 run → 輸出給模擬軟體手動匯入的路徑檔(手臂座標系 CSV:`i,x,y,z,rx,ry,rz,speed`,檔頭註明 frame/單位/使用的校正版本)。CIROS 實際可匯入的格式 `[待確認]`;D4 不做自動串接。
- **虛實落差比對(風險 5)**:`compare_sim_real.py`(新),讀模擬匯出的軌跡與實機 `events.jsonl` 的回讀 TCP,逐點算位置差。
- 完成條件:一次完整的 dry-run 與一次實機測試都產生符合 §15 的紀錄,且能用 `load_run()` 讀回。

---

## 12. 待確認問題(AI 不要自己決定,請問使用者)

1. W12 初型報告日期:11/25(週三)還是 11/26(週四)?W16 是否確為 12/23?
2. 三種工件的實際尺寸與擺放方式(L 型尺寸、珍珠板大小與厚度、C 類範例形狀)。
3. CIROS 是否有台達 DRV 機型?模型從哪裡取得?
4. 教授後續提供的更詳細規定(題目、評分、格式)。
5. 夾爪最大開口與夾持筆的方式;筆的種類與長度。
6. 是否可以用 RL 程式(DRAStudio)上傳到控制器執行?
7. KPI 數值是否已與教授確認?
8. 板子與 tag 是否固定?(§6.3)
9. 課程簡報寫的繳交項目是「CIROS 專案」——目前已與此一致(D4)。
10. ~~A4 紙的放置方向~~ → 已知:橫放,蓋住舊 ArUco 3、4(D10)。紙上另貼了一張較小的白紙當「每次換新的白紙」用。
11. 兩片黑板的**厚度、大小、材質是否反光**。
12. 能在黑板上留下可見筆跡的筆(白/銀色油性筆?);筆的種類與長度、夾持方式。
13. ~~tag 邊長~~ → 使用者告知 **29 mm**(D10)。仍可用尺量 tag 0→1、1→2 角到角距離,以 `paper_layout_capture.py --dx --dy` 取代 PnP 估計值。
15. 筆的實際顏色與筆頭硬度(探觸每步 3 mm 時會壓進紙 0~3 mm)。
16. 往基座方向拍照時(D11),筆尖離紙只有 10 mm 並經過 ArUco 1/2 卡片上方——卡片是否平貼、有無高於 10 mm 的障礙物。
14. DRV 是否有碰撞偵測(DRAStudio 手冊);若有,靈敏度與觸發後的復歸方式。

---

## 13. 變更紀錄

| 日期 | 內容 |
|---|---|
| 2026-10-13 | 安裝環境(conda `delta_arm`、DRV-Modbus、相依套件);確認手臂 Modbus 連線可讀位姿 |
| 10-14 | 夾爪 DO7/DO8 經 `0x02FE` 位元遮罩定義;`test_gripper_center.py` 使用者回報開合成功 |
| 10-21 | 新增 `record_apriltags.py`、`color_detect.py`、`warp_calibrate.py`、`warp_pick.py`、`go_home.py`、`watch_status.py`、`test_small_move.py`;發現並修正 tag 3、4 座標對調;發現 Modbus 閒置 20 秒斷線,加入 `refresh()`;`Go_Position` 加逾時;手臂拒絕指令後卡旗標 2 |
| 10-28 | 新增 `warp_sort.py`(綠→tag1/2 中點、藍→tag3/4 中點);夾取高度下降 10 mm、放置高度下降 9 mm 的參數化;旗標已恢復 1 |
| 11-04 | 簡報更新:三種工件、CIROS 不串接、移除 AI 學習、工件類型預先指定;建立本文件 |
| 11-04 | (手臂暫時不在)規劃筆尖探針式初始校正(D7、D8);新增 `a4_calib_sheet.py`、`paper_frame.py`、`seam_boards.py`、`tests/test_seam_synthetic.py`(僅合成場景驗證,尚無真實影像);新增 T15、T16 |
| 11-04 | 新增執行紀錄與校正版本庫 `run_log.py`(D9、§15)、`tests/test_run_log.py`;`seam_boards.py` 改為預設存 `runs/` 紀錄;把既有 `calibration.json` 匯入 `calibrations/warp_plane/` 當作基準版(2026-10-21);新增 T17 |
| 11-04 | 手臂連線檢查(唯讀):ping/Modbus 正常,但旗標 **2**、位姿 (444.0, 0.0, 753.9, −180, −90, −16) 不在 home;新增相機反裝 180 度支援 `cam_orient.py`(`--rot180 auto/on/off`、`camera.json`),接入 `warp_pick`/`warp_sort`/`warp_calibrate`/`color_detect`/`record_apriltags`;`warp_calibrate.py --pixels-only`(只重記 home 與 tag 像素)並自動備份舊檔、存版本;`tests/test_cam_orient.py` PASS |
| 11-04 | 專案上 GitHub(`Rayyang800/PBL-path-planning`,public;協作者 `70130H940410`)。`DRV-Modbus/.git` 改名 `.git_upstream` 保留原作者歷史 |
| 11-04 | 實機:相機接上(D435,USB 3.2)、手臂旗標恢復 1、`go_home` 成功;發現實驗室校正紙為 29 mm 橫放(D10)→ `paper_layout_capture.py`;修正 `Go_Position` 旗標競態(§8 #13);新增 `pen_calibrate.py` / `pen_marks.py`(先畫「+」→ 改點一下 → 改自動探觸畫小十字)、`--resume`、`--pattern cross` |
| 11-05 | 自動探觸問題排查(§8 #14–16)→ `--look base`(往畫面下方直到看見 ArUco 1/2)、`--ink any`、`--probe-step 3`;即時中斷(Ctrl+C 回 home、空白鍵選單)與夾爪 `o/g`、`gripper_ctl.py`;`tests/test_new_ink.py`、`test_stop.py`;push `7af52c8` |
| 11-05 | 使用者回報 D11 太慢、壓壞筆尖、點跑出工件(§8 #18)→ 新增 `pen_quick_calib.py`(D13)、`pen_marks.fit_rigid` / `match_pattern`、`new_ink_mask(return_drop=True)`、`tests/test_quick_calib.py`(PASS);修正墨點中心偏 0.8 mm(§8 #19)。手臂當時離線,實機未測 |
| 11-05 | 舊紀錄分析:10-21 的 `records/` tag 深度 → 深度相對精度約 ±1 mm(4 tag 共平面殘差 ±1 mm、傾角 5.3° vs 手臂 4.9°)。**紀錄 v2**:`rs_capture.py`、`run_log`(深度快照、console.log、git、current()、run_main)、`send`/`gripper` 自動記軌跡;接入 §15.3.1 所列腳本;新增 `depth_height.py`、`tests/test_run_log_v2.py`、`tests/test_depth_height.py`(全部 9 個測試 PASS)。實機/實際相機尚未用 v2 跑過 |

---

## 14. 附錄

### 14.1 簡報(PowerPoint)現況

| 檔案 | 位置 | 說明 |
|---|---|---|
| `PBL專題規劃報告_CIROS確認版.pptx` | `C:\Users\ray20\Downloads\` | **使用者自己修改的版本**(基準,不要覆蓋) |
| `PBL專題規劃報告_三種工件版.pptx` | `Delta_Arm\` | 以上述版本為底加入三種工件等修改,共 19 張(目前最新) |
| `PBL專題規劃報告.pptx`、`_v1`、`_v2`、`_v3` | `Delta_Arm\` | 較舊版本 |

簡報風格(若再生成):16:9;字型 **微軟正黑體**;主題色 dk1 `1C2529`、lt1 `FFFFFF`、dk2 `3A4A52`、lt2 `EDF2F3`、accent1(橘)`D9541A`、accent2(青)`12707A`、accent3 `6B7B83`、accent4(琥珀)`F2B134`、accent5(綠)`3F9C6B`、accent6(紅)`B23A48`;標題與內文皆用**全形標點**;不用標題底線或彩色邊條;每頁一個視覺元素;**直接編輯使用者的檔案並保留其措辭**,不要重建覆蓋。

### 14.2 專案資料夾索引

```
Delta_Arm\
├─ PROJECT_SPEC.md              ← 本文件
├─ Delta_Arm.code-workspace
├─ PBL專題規劃報告*.pptx
└─ DRV-Modbus\
   ├─ drv_modbus\   send.py request.py gripper.py conn.py
   ├─ landmark\ realsense\ aruco_img\
   ├─ calibration.json  calibration.before_swap.json(錯誤範例)
   ├─ records\      apriltag_records.csv/.jsonl、快照
   ├─ warp_calibrate.py warp_pick.py warp_sort.py go_home.py watch_status.py
   ├─ record_apriltags.py color_detect.py red_detect_demo.py
   ├─ test_gripper_center.py test_small_move.py pick_place_p1_p4.py
   └─ warp_pick_last.png warp_sort_last.png(最近一次辨識結果圖)
```

### 14.3 名詞

| 詞 | 意思 |
|---|---|
| TCP | 工具中心點(夾爪/筆尖),手臂座標以它為準 |
| home | 校正與拍照用的觀察位姿 `(364.5, −83.0, 809.4, −179.2, 0.56, −107.8)` |
| 塗佈線 | 要描繪/塗膠的線:脊線、縫合線、邊緣線 |
| 到位旗標 | `0x031F`;1 閒置、2 忙碌或異常 |
| MovP / MovL | 點對點(關節式)/ 直線移動 |
| 查核點1/2 | 課程 W12 / W15 的指定報告 |
| 單應(homography) | 平面上像素↔手臂 XY 的透視轉換 |

---

## 15. 資料紀錄與重現規範(D9)

> **硬性規定(§0.4):每次實作都完整紀錄** —— 彩色 + 深度 + 相機參數、每個手臂指令與移動軌跡、夾爪、按鍵中斷、終端輸出、程式版本、使用的校正檔;dry-run / 失敗 / 中止也一樣。

**為什麼要存**:(1) 重現結果與實驗統計(T12);(2) 之後手動餵給 CIROS 當輸入(D4:不自動串接,由人匯入檔案);(3) 把「模擬預測」與「實機實際」逐點比對,量化風險 5 的虛實落差;(4) 報告的圖表與數據;(5) 出問題時還原現場。**沒存的資料補不回來。**

### 15.1 資料夾結構(皆在 `DRV-Modbus\` 下,實作在 `run_log.py`)

```
runs\<YYYYMMDD-HHMMSS>_<kind>[_<label>]\     每次執行/校正一個資料夾,不覆蓋(同名自動加 -1、-2)
    meta.json     誰、何時、參數、環境(Python/套件版本)、使用的輸入檔 sha256、結果、狀態
                  status = running | ok | failed | aborted | dry-run(程式當掉也會留下 running/failed)
    events.jsonl  時間序事件,一行一個 JSON:{"t":秒,"ts":ISO時間,"type":...,...}
    inputs\       執行當下使用的校正檔、版面檔的**複本**
    images\       原始影像(PNG,無損)與疊圖(JPG,僅目視)
    depth\        (v2) <快照名>.png = 對齊彩色的深度 uint16(多幀逐像素中位數,0 = 無效)+ <快照名>.json(depth_scale_m、內參、幀數、有效比例)
    data\         路徑、偏差等 CSV/JSON
    console.log   (v2) 執行期間終端機的完整輸出
    meta.json 另含 (v2):git(commit、未提交的檔案)、camera(序號/韌體/USB/彩色與深度內參/深度單位/深度→彩色外參)、snapshots(快照清單)
calibrations\<name>\<YYYYMMDD-HHMMSS>.json     每次校正一個新檔,含 supersedes(前一版檔名)、source_run
calibrations\<name>\latest.json                最新版複本(Windows 不用 symlink)
```

`kind`:`calibration`(平面/筆尖校正)、`seam`(縫合線擷取)、`exec`(描繪)、`experiment`(T12 重複性實驗,一次一個 run + 彙總 CSV)。校正名稱約定:`warp_plane`(舊 `calibration.json` 匯入,基準 2026-10-21)、`pen`(T15)、`safety_limits`(T08)。

### 15.2 各類紀錄必存的原始資料

| kind | 必存 |
|---|---|
| 校正 | home 原始影像;tag 偵測角點;每個接觸點的**指令位姿與回讀 TCP**、`z_touch`;標記影像;擬合結果與殘差(mm) |
| seam | 原始影像(`frame_raw.png`);板厚等參數;`paper_frame`(殘差、相機離紙高度、傾角、mm/px、tag 角點);縫合線折線(紙面 mm) |
| exec | 規劃路徑(紙面與手臂座標兩份,CSV:`i,x,y,z,rx,ry,rz`,mm/度);每個指令(`cmd_move`:pose、speed、mov)與**每步前後回讀的 TCP 與旗標**;夾爪事件;當下使用的安全限制複本;描繪前後照片;偏差結果 |
| experiment | 上述 exec 的完整紀錄 + 一列彙總(工件類型、擺放位置、成功/失敗、平均/最大偏差、失敗原因) |

原則:**先存原始、衍生結果可事後重算**;單位固定(手臂 mm/度、紙面 mm、影像 px),座標資料標明 frame ∈ `robot_mm_deg` / `paper_mm` / `pixel`。

### 15.3 使用方式

```python
from run_log import RunLog, CalibrationStore, load_run
with RunLog("exec", label="seam_B_01", params=vars(args), hardware={...}) as log:
    log.add_input("calibrations/pen/latest.json")          # 記錄用了哪一版 (複製 + sha256)
    log.save_image("frame_before", img)                    # images/frame_before.png
    log.event("cmd_move", pose=[x, y, z, rx, ry, rz], speed=15, mov="MovL", frame="robot_mm_deg")
    log.event("tcp", pose=readback_pose, flag=flag)
    log.save_csv("path_robot_mm", ["x", "y", "z", "rx", "ry", "rz"], rows)
    # 離開 with 自動 finish;例外會留下 failed/aborted 紀錄並照常往外丟
CalibrationStore().save("pen", data, source_run=log.meta["run_id"])
meta, events = load_run("runs/20261104-155747_seam")
```

### 15.3.1 v2 自動紀錄(11-05)

- **`run_log.current()`**:進行中的紀錄。`drv_modbus.send.Go_Position` 會自動寫 `cmd_move`、移動中約 10 Hz 的 `tcp_trace`(pose + flag)、`move_done`(終點、耗時)或 `move_error`;`Stop` 寫 `cmd_stop`;`gripper.Gripper_*` 寫 `cmd_gripper`;吸盤寫 `cmd_suction`。**任何腳本只要在 `with RunLog(...)` 裡移動手臂就有完整軌跡**。
- **舊腳本一行接上**:`if __name__ == "__main__": run_log.run_main("warp_sort", main)`(`sys.exit("原因")` → status `exited` 並記錄原因;`-h` 不建紀錄)。
- **相機**:`rs_capture.RSCamera` / `grab_once(n_depth=15)` → `log.save_snapshot(name, snap, cam_info)`;任何地方可用 `run_log.snapshot(name, snap, info)` 存進目前紀錄。讀回:`color, depth_mm, dmeta = load_snapshot(run, name)`(深度 mm,無效 = NaN)。
- **已接上**:`pen_quick_calib`(dry-run 也存;中止後若已回 home 會補拍 `home_after` 供 `--analyze`)、`pen_calibrate`(探觸每步含深度 JPG+PNG)、`go_home`、`warp_sort`、`warp_pick`、`warp_calibrate`(home 快照、每次讀 TCP、校正結果)、`seam_boards --camera`(含深度)、`record_apriltags`(另存 `records/snapshot_###_raw.png`、`_depth.png`、`_camera.json`)。**未接**:`pick_place_p1_p4`、`test_*` 診斷腳本、`watch_status`。
- **離線工具**:`depth_height.py runs\<id> [--snap 名] [--at X Y] [--r 6] [--map]` → 從深度快照擬合紙面平面,量指定紙面位置的高度(工件厚度),合成測試 6 mm 量到 +6.04。

### 15.4 維護規則

- 容量:彩色 PNG 約 1.2 MB + 深度 PNG 約 0.3~0.8 MB/快照;`pen_calibrate` 自動探觸每步一組(JPG + 深度)。`runs/` 已在 `.gitignore`,請另行備份。
- **不要手動修改 run 內的檔案**;要刪除舊紀錄前先問使用者。
- 11-05 之前的紀錄(schema 1)沒有深度、軌跡與 console.log。
- 驗證:`python tests\test_run_log.py`、`tests\test_run_log_v2.py`、`tests\test_depth_height.py`。
