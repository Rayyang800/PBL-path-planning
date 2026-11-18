"""以合成位姿測試五姿態擬合與獨立對點，不連接手臂。"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pen_pivot import CONV, check_reference, inspect_points


class PivotTest(unittest.TestCase):
    def setUp(self):
        self.tip = np.array([14., -8., 255.])
        pivot = np.array([430., -75., 220.])
        base = Rotation.from_euler(CONV, [-179., 0.5, -108.], degrees=True)
        self.rows = []
        for axis, angle in (([1, 0, 0], 0), ([1, 0, 0], 12), ([0, 1, 0], -12),
                            ([1, 0, 0], -12), ([0, 1, 0], 12)):
            rot = Rotation.from_rotvec(np.radians(np.array(axis) * angle)) * base
            pose = [*(pivot - rot.apply(self.tip)), *rot.as_euler(CONV, degrees=True)]
            self.rows.append({"tcp": pose, "tilt": [axis, angle] if angle else None})

    def test_missing_direction_and_invalid_pose(self):
        report = inspect_points(self.rows[:4] + [self.rows[1], {"tcp": None, "tilt": None}])
        self.assertEqual(report["missing"], ["Y+"])
        self.assertFalse(report["fit_ok"])

    def test_fit_and_independent_dimensions(self):
        report = inspect_points(self.rows)
        self.assertTrue(report["fit_ok"])
        np.testing.assert_allclose(report["p_tool_mm"], self.tip)
        target = np.array([460., -50., 230.])
        # 獨立對點的真實高度偏離基準 3 mm。
        pose = [*(target + [0, 0, 3] - self.tip), 0, 0, 0]
        for point, expected in ((target[:2], 0), (target, 3)):
            result = check_reference(report, {"target_robot_mm": point, "tcp_poses": [pose]})
            self.assertAlmostEqual(result["max_mm"], expected)

    def test_bad_contact_and_degenerate_orientations(self):
        self.rows[1]["tcp"][0] += 20
        self.assertFalse(inspect_points(self.rows)["fit_ok"])
        for row in self.rows:
            row["tcp"] = self.rows[0]["tcp"]
        self.assertFalse(inspect_points(self.rows)["fit_ok"])


if __name__ == "__main__":
    unittest.main()
