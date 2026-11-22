# 來源

- 作者:DanielYut,https://github.com/DanielYut/DeltaArm_Smooth (經由 fork https://github.com/Rayyang800/deltaarm_smooth)
- 版本:commit 7b6995b (2026-11-19)
- 授權:原 repo 沒有 LICENSE 檔。本 repo 擁有者 (Rayyang800) 於 2026-11-22 表示:持有作者同意使用的對話紀錄 (紀錄不放進公開 repo)。
  這是給本專案的同意,不等於開源授權;其他人要沿用這兩個 Lua 檔請先問原作者。

## 這裡放了什麼

| 檔案 | 和原版的差別 |
|---|---|
| `loader.lua` | 原樣 |
| `path_runner.lua` | VER 2:加取消旗標 0x1004、點位讀回時連姿態一起比對 |

**沒有**放進來:`uploader/` (台達 DLL、RLUpload.exe,屬於廠商檔案或來源無法確認的執行檔)、原版 `delta_smooth.py`
(出錯時送 Motion Stop 1000、會停掉別人的 RL 程式,和本專案規則不合;改用 `../../smooth_motion.py`)。

## 控制器一次性設定 (由組員在現場做,程式不會自動做)

1. DRAStudio 開一個 RL 專案,把 `loader.lua` 貼進 main.lua,下載到 RL ID 900 (先確認 900 沒有別人的專案)。
   或照原 repo 的 install_loader.py 做 (會執行它附的 exe,並改變控制器內容)。
2. 教模板點 P1:用平常描繪的姿態 (同一個手肘 / 手腕組態、同一個 UF / TF)。
3. 確認點位表 501 號以後沒有教好的點 (會被覆蓋)。
4. 控制器切 Auto,`python smooth_motion.py diag` 確認。
