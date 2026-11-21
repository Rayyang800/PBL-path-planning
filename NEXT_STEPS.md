# 接下來要做的事 (交接用,2026-11-21 更新)

> 每完成一項就劃掉 / 更新日期。手臂動作一律要在任務視窗按 Enter 確認;Ctrl+C = 抬起回 home。

## 已完成 (11-21,尚未上機、尚未 commit)
- 工件 B `seam_trace.py`:預設一次 MovL 連續畫完 (`--segmented` = 舊逐點)、套用 / 記錄筆跡回饋筆尖偏移 (`pen_offset`,和 A、C 共用)。
  離線用 20261118-161715_seam_trace 的照片測過規劃 (偏移修正 -0.89, +1.17 mm)。
- 網頁:A / B / C 放在同一張卡「三種工件:一鍵描繪」,格式一致;B 板厚預設 0 = 深度自動估 (job_config.json 也改成 0)。
- 新增 `pen_tip_twin.py` (網頁「筆尖 3D 位置:分身算法」):不動手臂,用手眼 + 護貝紙筆尖校正 + home 照片 PnP
  推出筆尖在工具座標的位置 (方法來自 Desktop\Delta_Arm_IsaacSim\real2sim\twin.py)。
  11-18 資料:筆尖水平 (-44.5, 46.2) ±0.4 mm;筆長 235 (11-11 手眼) / 247 (11-18 手眼) mm,差 12 mm。
  傾斜 15° 時筆尖誤差約 ±1.5 mm。**尚未 --save**。

## 下一步 (依順序)
1. **上機測 B**:網頁「B 兩板縫合」dry-run -> 執行。確認連續畫、偏差分析、回饋記錄正常。
2. **用尺量筆長** (法蘭面 -> 筆尖,筆垂直),填網頁設定「尺量筆長」,按「筆尖 3D 位置」的「存檔」
   (換筆 / 重夾後:先護貝紙校正 -> 任一任務 dry-run 拍 home -> 再存檔)。
3. **circle_trace 加「底面外圍」模式**:目標圓 = 罐頭外緣半徑 + 偏移,高度 = 紙面 (h=0);筆往圓心外傾斜。
   碰撞檢查要把「筆在罐頭旁邊」算進去 (目前 rotation_margin 把圓柱內 r+5 mm 都當成要高於罐頭,底面模式會誤判);
   筆身半徑約 7 mm,傾斜 15° 時筆尖要離罐壁約 >= 7/cos - h·tan 的距離,先在 dry-run 算出可達的最小偏移。
4. **把 Isaac 分身的執行前檢查接上傾斜路徑**:`Delta_Arm_IsaacSim\real2sim\precheck.py` 目前只吃 ridge_trace 的
   plan.json (垂直姿態);改成讀 circle_trace plan 的 path_robot + oris,檢查 IK、關節餘裕、J5 奇異點、輪盤 / 夾爪碰撞。
   要用 env_isaac61 的 python (delta_arm 的 OpenCV 4.6 跑 twin.py 的 tag_pnp 會出錯)。
5. ~~Codex 改動~~ (11-21 已處理):停止指令回到 0x0300=0、停止後不丟例外、circle_trace 中止一律抬起
   (傾斜時先抬到 home 高度) 轉回垂直回 home。保留:傾斜需 --gripper-reach / --camera-reach (網頁設定有)、pen_pivot 五姿態。
6. 之後:重複性實驗 (A、C 各 10 次)、B 換淺色板量偏差、PROJECT_SPEC / README 更新、git commit。

## 備註
- 簡報 / 講稿:`PBL專題規劃報告_三種工件版.pptx` (48 頁,W12 查核點 1)、`PBL進度報告3_講稿.docx` 已完成,未 commit。
- 查核點日期:簡報時程寫 11/26,週報是週三;封面只寫「W12 查核點 1」。
