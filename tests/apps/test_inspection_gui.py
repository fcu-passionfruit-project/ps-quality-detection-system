import unittest
from apps.inspection_gui import _format_stage_summary, _validation_summary


class GUIResultSummaryTests(unittest.TestCase):
    def test_formats_model_results_and_failures(self):
        cases = (
            (
                "roi",
                {"status": "ok", "confidence": 0.984745},
                "ROI：PF，信心值：0.984745",
            ),
            (
                "color",
                {
                    "status": "ok",
                    "class_name": "good_color",
                    "confidence": 0.998,
                },
                "顏色：good_color，信心值：0.998000",
            ),
            (
                "color",
                {
                    "status": "ok",
                    "class_name": "bad_color",
                    "confidence": 0.975,
                },
                "顏色：bad_color，信心值：0.975000",
            ),
            (
                "wrinkle",
                {
                    "status": "ok",
                    "class_name": "true(有皺褶)",
                    "confidence": 0.997,
                },
                "皺褶：true(有皺褶)，信心值：0.997000",
            ),
            (
                "wrinkle",
                {
                    "status": "ok",
                    "class_name": "false(沒皺褶)",
                    "confidence": 0.991,
                },
                "皺褶：false(沒皺褶)，信心值：0.991000",
            ),
            (
                "defect",
                {"status": "ok", "detections": []},
                "局部瑕疵：未偵測到",
            ),
            (
                "color",
                {"status": "skipped", "reason": "model_missing"},
                "顏色：略過（model_missing）",
            ),
            (
                "wrinkle",
                {"status": "error", "reason": "inference_failed"},
                "皺褶：失敗（inference_failed）",
            ),
        )
        for role, result, expected in cases:
            with self.subTest(role=role, expected=expected):
                self.assertEqual(_format_stage_summary(role, result), expected)

        report = {
            "mode": "pipeline",
            "roi": {"status": "ok", "confidence": 0.984745},
            "models": {
                "color": cases[1][1],
                "wrinkle": cases[3][1],
                "defect": cases[5][1],
            },
        }
        self.assertEqual(
            _validation_summary(report),
            "　｜　".join((cases[0][2], cases[1][2], cases[3][2], cases[5][2])),
        )


