# -*- coding: utf-8 -*-
"""RunLog / CalibrationStore 測試 (使用暫存資料夾,不碰實際 runs/ 與 calibrations/)。  python tests\\test_run_log.py"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from run_log import RunLog, CalibrationStore, load_run, sha256_file   # noqa: E402


def main():
    tmp = tempfile.mkdtemp(prefix="runlog_")
    runs, cal = os.path.join(tmp, "runs"), os.path.join(tmp, "cal")
    src = os.path.join(tmp, "calibration.json")
    with open(src, "w", encoding="utf-8") as f:
        json.dump({"home": [1.0, 2.0, 3.0], "測試": "中文"}, f, ensure_ascii=False)

    # 1. 正常一次
    with RunLog("seam", label="unit", runs_dir=runs, params={"speed": 15, "arr": np.arange(3)},
                hardware={"robot_ip": "192.168.1.1"}) as log:
        rel = log.add_input(src)
        log.event("cmd_move", pose=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), speed=15, mov="MovL", frame="robot_mm_deg")
        log.event("tcp", pose=[1, 2, 3, 4, 5, 6], flag=1)
        log.save_image("frame_raw", np.zeros((10, 20, 3), np.uint8))
        log.save_json("result", {"mean_dev_mm": 1.23})
        log.save_csv("path_paper_mm", ["x_mm", "y_mm"], [(1.0, 2.0), (3.5, 4.25)])
        run_path = log.path
    meta, events = load_run(run_path)
    assert meta["status"] == "ok" and meta["ended"] and meta["kind"] == "seam", meta["status"]
    assert meta["params"]["arr"] == [0, 1, 2] and meta["hardware"]["robot_ip"] == "192.168.1.1"
    assert meta["inputs"]["calibration.json"]["sha256"] == sha256_file(src)
    assert [e["type"] for e in events] == ["cmd_move", "tcp"] and events[0]["pose"][2] == 3.0
    for rel in ("images/frame_raw.png", "data/result.json", "data/path_paper_mm.csv", "inputs/calibration.json", "meta.json", "events.jsonl"):
        assert os.path.isfile(os.path.join(run_path, rel)), rel
    with open(os.path.join(run_path, "inputs", "calibration.json"), encoding="utf-8") as f:
        assert json.load(f)["測試"] == "中文"
    print("OK 正常紀錄:", os.path.basename(run_path))

    # 2. 例外要留下 failed 紀錄,且例外照常往外丟
    try:
        with RunLog("exec", runs_dir=runs) as log2:
            log2.event("note", msg="before crash")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    m2, e2 = load_run(log2.path)
    assert m2["status"] == "failed" and "boom" in m2["result"]["error"] and e2[-1]["type"] == "error"
    print("OK 例外 -> status =", m2["status"])

    # 3. Ctrl+C -> aborted
    try:
        with RunLog("exec", runs_dir=runs) as log3:
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert load_run(log3.path)[0]["status"] == "aborted"
    print("OK Ctrl+C -> aborted")

    # 4. 同一秒內建立兩次 -> 不覆蓋
    a, b = RunLog("dup", runs_dir=runs), RunLog("dup", runs_dir=runs)
    assert a.path != b.path
    a.finish(); b.finish()
    print("OK 不覆蓋:", os.path.basename(a.path), os.path.basename(b.path))

    # 5. 校正版本庫
    store = CalibrationStore(cal)
    p1 = store.save("pen", {"offset_xy_mm": [0.5, -0.2]}, source_run="r1")
    p2 = store.save("pen", {"offset_xy_mm": [0.6, -0.1]}, source_run="r2", notes="第二次")
    assert len(store.history("pen")) == 2 and p1 != p2
    latest = store.load_latest("pen")
    assert latest["data"]["offset_xy_mm"] == [0.6, -0.1] and latest["supersedes"] == os.path.basename(p1)
    print("OK 版本庫: 2 版,latest = 第二版,supersedes =", latest["supersedes"])

    # 6. 匯入既有校正檔 (冪等)
    q1 = store.import_legacy("warp_plane", src)
    q2 = store.import_legacy("warp_plane", src)
    assert q1 and q2 is None and len(store.history("warp_plane")) == 1
    assert store.load_latest("warp_plane")["data"]["home"] == [1.0, 2.0, 3.0]
    print("OK 匯入既有檔 (重複匯入會略過)")
    print("\nPASS")


if __name__ == "__main__":
    main()
