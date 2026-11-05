from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from passionfruit.evaluation.evaluate_experiment_d_area import CLASS_NAMES, EvaluationError
from passionfruit.evaluation.evaluate_local_defect_pixels import parse_args
from passionfruit.evaluation.pixel_data import restore_masks, roi_mask
from passionfruit.evaluation.pixel_metrics import measure, summarize_rows
from passionfruit.evaluation.pixel_postprocessing import django_deployed_masks, load_django_reference, reference_summary

ROOT = Path(__file__).resolve().parents[2]


def result_for(masks, classes, shape=(7, 9), names=None):
    boxes = torch.zeros((len(classes), 6))
    if classes:
        boxes[:, :4] = torch.tensor([0, 0, shape[1], shape[0]])
        boxes[:, 4] = 0.8
        boxes[:, 5] = torch.tensor(classes)
    return Results(np.zeros((*shape, 3), np.uint8), path="fixture.png",
                   names=names or dict(enumerate(CLASS_NAMES)), boxes=boxes,
                   masks=torch.tensor(masks) if len(classes) else None)


class DjangoDeployedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.django_root = ROOT.parent / "PassionFruit_IoT_hardware"
        if not (cls.django_root / "Django_Server/fruit_app/detection.py").is_file():
            raise unittest.SkipTest("Original sibling Django source is required for reference-equivalence tests")
        cls.reference = load_django_reference(cls.django_root)

    def test_fractional_masks_strict_threshold_and_nearest_match_original(self):
        raw = np.array([[[.5, .50001, .49], [.75, 0, 1]]], dtype=np.float32)
        result = result_for(raw, [0])
        roi = np.ones((7, 9), dtype=np.uint8) * 255
        original = result.masks.data.clone()
        got = django_deployed_masks(result, roi, self.reference)
        expected = cv2.resize(raw[0], (9, 7), interpolation=cv2.INTER_NEAREST) > .5
        np.testing.assert_array_equal(got.unions[0], expected)
        self.assertEqual(int(expected.sum()), 30)
        self.assertTrue(torch.equal(original, result.masks.data))
        self.assertTrue(all(r["union_xor_pixels"] == 0 and r["ratio_exact_match"] for r in got.consistency))

    def test_same_class_overlap_and_other_class_overlap_are_independent(self):
        raw = np.zeros((3, 7, 9), np.uint8)
        raw[0, 1:4, 1:4] = 1
        raw[1, 2:5, 2:5] = 1
        raw[2] = raw[0]
        got = django_deployed_masks(result_for(raw, [0, 0, 1]), np.ones((7, 9), bool), self.reference)
        self.assertEqual([int(u.sum()) for u in got.unions], [14, 9, 0, 0])
        self.assertEqual(got.feature_ratios["abrasion"], round(14/63, 8))
        self.assertEqual(got.feature_ratios["anthracnose"], round(9/63, 8))

    def test_roi_clipping_happens_after_resize_and_threshold(self):
        raw = np.ones((1, 2, 3), np.float32)
        roi = roi_mask((7, 9), "django-ellipse")
        got = django_deployed_masks(result_for(raw, [3]), roi, self.reference)
        np.testing.assert_array_equal(got.unions[3], roi)
        self.assertEqual(got.feature_ratios["insect_track"], 1.0)

    def test_deployed_feature_rounding_is_separate_from_metric_ratio(self):
        raw = np.array([[[1, 0, 0]]], np.uint8)
        roi = np.ones((1, 3), bool)
        got = django_deployed_masks(result_for(raw, [0], shape=(1, 3)), roi, self.reference)
        self.assertEqual(got.feature_ratios["abrasion"], .33333333)
        measured = measure(np.zeros((1, 3), bool), got.unions[0], roi)
        self.assertEqual(measured["pred_area_ratio"], 1/3)
        self.assertNotEqual(measured["pred_area_ratio"], got.feature_ratios["abrasion"])

    def test_empty_predictions_match_original_zero_unions(self):
        got = django_deployed_masks(result_for([], []), np.ones((7, 9), bool), self.reference)
        self.assertEqual(got.instances.shape, (0, 7, 9))
        self.assertTrue(all(not u.any() for u in got.unions))
        self.assertEqual(got.feature_ratios, dict.fromkeys(CLASS_NAMES, 0.0))

    def test_reference_uses_model_name_semantics(self):
        names = {0: "insect_track", 1: "insect_bite", 2: "anthracnose", 3: "abrasion"}
        got = django_deployed_masks(result_for(np.ones((1, 7, 9), np.uint8), [0], names=names), np.ones((7, 9), bool), self.reference)
        self.assertEqual([int(u.sum()) for u in got.unions], [0, 0, 0, 63])

    def test_deployed_does_not_remove_letterbox_padding(self):
        raw = np.zeros((1, 20, 20), np.uint8); raw[0, 7:13, 4:12] = 1
        result = result_for(raw, [0], shape=(5, 10))
        got = django_deployed_masks(result, np.ones((5, 10), bool), self.reference)
        scaled = restore_masks(result.masks.data, (5, 10))[0]
        direct = cv2.resize(raw[0], (10, 5), interpolation=cv2.INTER_NEAREST) > .5
        np.testing.assert_array_equal(got.unions[0], direct)
        self.assertFalse(np.array_equal(got.unions[0], scaled))

    def test_zero_roi_and_missing_masks_fail_explicitly(self):
        with self.assertRaises(EvaluationError):
            django_deployed_masks(result_for([], []), np.zeros((7, 9), bool), self.reference)
        result = result_for(np.ones((1, 7, 9), np.uint8), [0]); result.masks = None
        with self.assertRaises(EvaluationError):
            django_deployed_masks(result, np.ones((7, 9), bool), self.reference)

    def test_reference_profiler_is_restored_even_if_original_raises(self):
        result = result_for(np.ones((1, 7, 9), np.uint8), [0]); result.masks = None
        previous = sys.getprofile()
        with self.assertRaises(ValueError):
            reference_summary(self.reference, result, np.ones((7, 9), bool))
        self.assertIs(sys.getprofile(), previous)

    def test_module_loading_keeps_path_and_bytecode_policy(self):
        before, bytecode = sys.path.copy(), sys.dont_write_bytecode
        self.assertIs(load_django_reference(self.django_root), self.reference)
        self.assertEqual(sys.path, before)
        self.assertEqual(sys.dont_write_bytecode, bytecode)

    def test_original_roi_preparation_matches_geometry_for_known_crop(self):
        image = np.full((20, 30, 3), 80, np.uint8)
        result = result_for(np.ones((1, 20, 30), np.uint8), [0], shape=(20, 30))
        result.boxes.data[0, :4] = torch.tensor([2.4, 3.6, 26.2, 18.1])
        info, artifacts = self.reference._prepare_roi(image, result, 0.0)
        self.assertEqual(info["crop_bbox"], [2, 4, 26, 18])
        expected = roi_mask((14, 24), "django-ellipse")
        np.testing.assert_array_equal(artifacts["roi_mask"] > 0, expected)
        np.testing.assert_array_equal(artifacts["masked_roi"], cv2.bitwise_and(image[4:18, 2:26], image[4:18, 2:26], mask=expected.astype(np.uint8)*255))

    def test_cli_keeps_scale_masks_default_and_accepts_deployed(self):
        self.assertEqual(parse_args(["--output", "unused"]).mask_mode, "scale-masks")
        self.assertEqual(parse_args(["--output", "unused", "--mask-mode", "django-deployed"]).mask_mode, "django-deployed")

    def test_original_defect_predict_call_options(self):
        class FakeModel:
            def predict(self, source, **options):
                self.options = options
                self.source = source
                return ["prediction"] * len(source)
        model = FakeModel()
        images = [np.zeros((7, 9, 3), np.uint8)]
        values, _ = self.reference._predict_batch(model, "defect", images, "cpu", {"defect": {"imgsz": 1024}})
        self.assertEqual(values, ["prediction"])
        self.assertEqual(model.options, dict(device="cpu", verbose=False, imgsz=1024))
        self.assertIs(model.source, images)

    def test_summaries_never_mix_modes(self):
        rows = []
        for mode, pixels in (("scale-masks", 1), ("django-deployed", 2)):
            gt = np.zeros((2, 2), bool)
            pred = gt.copy(); pred.flat[:pixels] = True
            row = dict(split="valid", model_name="fixture", class_id=0, mask_mode=mode)
            row.update(measure(gt, pred, np.ones((2, 2), bool)))
            rows.append(row)
        summary = {r["mask_mode"]: r for r in summarize_rows(rows)}
        self.assertEqual(summary["scale-masks"]["MAE_all"], .25)
        self.assertEqual(summary["django-deployed"]["MAE_all"], .5)


if __name__ == "__main__":
    unittest.main()
