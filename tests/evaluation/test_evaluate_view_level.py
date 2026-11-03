import unittest

from passionfruit.evaluation.evaluate_view_level import (
    any_defect_metrics,
    class_metrics,
    quadratic_weighted_kappa,
    select_per_class_plan,
    severity_level,
    uniform_plan,
)


def _record(image, gt_abrasion, pred_abrasion):
    zeros = {"anthracnose": 0.0, "insect_bite": 0.0, "insect_track": 0.0}
    return {
        "image": image,
        "gt": {"abrasion": gt_abrasion, **zeros},
        "pred": {"abrasion": [pred_abrasion], **{name: [0.0] for name in zeros}},
    }


class ViewLevelEvaluationTests(unittest.TestCase):
    def test_severity_level_uses_zero_as_none_and_inclusive_bounds(self):
        bounds = (0.01, 0.05)
        self.assertEqual([severity_level(v, bounds) for v in (0.0, 0.01, 0.05, 0.06)], [0, 1, 2, 3])

    def test_quadratic_weighted_kappa_perfect_and_penalizes_distant_errors(self):
        self.assertAlmostEqual(quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)
        near = quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 2])
        far = quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 0])
        self.assertGreater(near, far)

    def test_class_metrics_counts_presence_and_area_error(self):
        records = [
            _record("a", 0.02, 0.03),  # TP
            _record("b", 0.02, 0.0),   # FN
            _record("c", 0.0, 0.01),   # FP
            _record("d", 0.0, 0.0),    # TN
        ]
        metrics = class_metrics(records, "abrasion", 0, (0.01, 0.05))
        self.assertEqual((metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["area_mae_pct"], (0.01 + 0.02 + 0.01) / 4 * 100)
        self.assertAlmostEqual(metrics["area_mae_positive_pct"], (0.01 + 0.02) / 2 * 100)

    def test_any_defect_merges_classes(self):
        metrics = any_defect_metrics([_record("a", 0.02, 0.0), _record("b", 0.0, 0.0)], uniform_plan(0))
        self.assertEqual((metrics["tp"], metrics["fn"], metrics["tn"]), (0, 1, 1))

    def test_per_class_plan_picks_threshold_that_removes_false_positive(self):
        from passionfruit.evaluation import evaluate_view_level as module

        records = []
        for index, (gt, preds) in enumerate([(0.02, [0.02, 0.02]), (0.0, [0.01, 0.0])]):
            zeros = {"anthracnose": 0.0, "insect_bite": 0.0, "insect_track": 0.0}
            records.append({
                "image": str(index),
                "gt": {"abrasion": gt, **zeros},
                "pred": {"abrasion": preds, **{name: [0.0, 0.0] for name in zeros}},
            })
        bounds = {name: (0.01, 0.05) for name in module.CLASS_NAMES}
        original = module.THRESHOLDS
        module.THRESHOLDS = (0.1, 0.5)
        try:
            plan = select_per_class_plan(records, bounds)
        finally:
            module.THRESHOLDS = original
        self.assertEqual(plan["abrasion"], 1)


if __name__ == "__main__":
    unittest.main()
