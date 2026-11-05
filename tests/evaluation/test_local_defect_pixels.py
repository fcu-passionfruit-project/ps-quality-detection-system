from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from passionfruit.evaluation.evaluate_experiment_d_area import EvaluationError
from passionfruit.evaluation.evaluate_local_defect_pixels import checkpoint_reference, write_csv
from passionfruit.evaluation.pixel_data import load_gt, restore_masks, roi_mask
from passionfruit.evaluation.pixel_metrics import measure, pixel_scores, summarize_rows, union_instances


class PixelMetricsTests(unittest.TestCase):
    def setUp(self):
        self.roi = np.ones((40, 50), dtype=bool)
        self.gt = np.zeros((40, 50), dtype=bool)
        self.gt[:20, :] = True  # exactly 1000 pixels

    def test_identical(self):
        row = measure(self.gt, self.gt, self.roi)
        self.assertEqual((row["Pixel_Precision"], row["Pixel_Recall"], row["absolute_area_error"]), (1, 1, 0))

    def test_inside(self):
        p = self.gt.copy(); p[10:] = False
        row = measure(self.gt, p, self.roi)
        self.assertEqual((row["Pixel_Precision"], row["Pixel_Recall"], row["absolute_area_error"]), (1, .5, .25))

    def test_contains(self):
        row = measure(self.gt, self.roi, self.roi)
        self.assertEqual((row["Pixel_Precision"], row["Pixel_Recall"], row["absolute_area_error"]), (.5, 1, .5))

    def test_required_three_fragments_and_single_mask_equivalence(self):
        parts = np.zeros((3, 40, 50), dtype=bool)
        parts[0].flat[:300] = True
        parts[1].flat[300:550] = True
        parts[2].flat[550:750] = True
        self.assertEqual([int(p.sum()) for p in parts], [300, 250, 200])
        unions = union_instances(parts, np.array([3, 3, 3]), self.gt.shape)
        row = measure(self.gt, unions[3], self.roi)
        self.assertEqual(row["pred_area_pixels"], 750)
        self.assertEqual(row["intersection_pixels"], 750)
        self.assertEqual(row["Pixel_Precision"], 1.0)
        self.assertEqual(row["Pixel_Recall"], .75)
        one = union_instances(unions[3][None], np.array([3]), self.gt.shape)[3]
        self.assertEqual(row, measure(self.gt, one, self.roi))
        self.assertTrue(all(not u.any() for u in unions[:3]))

    def test_overlapping_instances_count_once(self):
        p = np.stack([self.gt, self.gt, self.gt])
        u = union_instances(p, np.array([0, 0, 0]), self.gt.shape)[0]
        self.assertEqual(int(u.sum()), 1000)
        self.assertEqual(measure(self.gt, u, self.roi)["absolute_area_error"], 0)

    def test_complete_miss(self):
        r = measure(self.gt, np.zeros_like(self.gt), self.roi)
        self.assertEqual((r["Pixel_Precision"], r["Pixel_Recall"], r["absolute_area_error"], r["fn_pixels"]), (0, 0, .5, 1000))

    def test_false_positive(self):
        r = measure(np.zeros_like(self.gt), self.gt, self.roi)
        self.assertEqual((r["Pixel_Precision"], r["Pixel_Recall"], r["absolute_area_error"], r["fp_pixels"]), (0, None, .5, 1000))

    def test_both_empty(self):
        z = np.zeros_like(self.gt)
        r = measure(z, z, self.roi)
        self.assertEqual((r["absolute_area_error"], r["intersection_pixels"]), (0, 0))
        self.assertIsNone(r["Pixel_Precision"])
        self.assertIsNone(r["Pixel_Recall"])
        self.assertIsNone(r["Pixel_F1"])

    def test_same_area_wrong_location(self):
        p = ~self.gt
        r = measure(self.gt, p, self.roi)
        self.assertEqual((r["absolute_area_error"], r["Pixel_Precision"], r["Pixel_Recall"]), (0, 0, 0))

    def test_multiclass_masks_overlap_independently(self):
        u = union_instances(np.stack([self.gt, self.gt, self.roi, ~self.gt]), np.arange(4), self.gt.shape)
        self.assertEqual([int(p.sum()) for p in u], [1000, 1000, 2000, 1000])
        self.assertEqual([measure(self.gt, p, self.roi)["intersection_pixels"] for p in u], [1000, 1000, 1000, 0])

    def test_roi_zero_and_mismatched_grid_fail(self):
        with self.assertRaisesRegex(EvaluationError, "ROI"):
            measure(self.gt, self.gt, np.zeros_like(self.roi))
        with self.assertRaises(EvaluationError):
            measure(self.gt, self.gt[:, :10], self.roi)

    def test_roi_clips_both_masks(self):
        roi = self.gt.copy()
        r = measure(self.roi, self.gt, roi)
        self.assertEqual((r["roi_pixels"], r["gt_area_ratio"], r["pred_area_ratio"]), (1000, 1, 1))
        self.assertEqual(r["gt_pixels_outside_roi"], 1000)

    def test_union_rejects_invalid_classes_and_shape(self):
        for classes in (np.array([4]), np.array([.5])):
            with self.assertRaises(EvaluationError):
                union_instances(self.gt[None], classes, self.gt.shape)
        with self.assertRaises(EvaluationError):
            union_instances(self.gt[None], np.array([0, 1]), self.gt.shape)

    def test_official_unletterbox_nearest_recovers_known_positions(self):
        original = np.zeros((5, 10), dtype=np.uint8)
        original[1:4, 2:6] = 1
        enlarged = cv2.resize(original, (20, 10), interpolation=cv2.INTER_NEAREST)
        padded = np.pad(enlarged, ((5, 5), (0, 0)))
        restored = restore_masks(torch.tensor(padded[None]), (5, 10))[0]
        np.testing.assert_array_equal(restored, original.astype(bool))
        # Direct resize fails even though its output shape matches the GT.
        direct = cv2.resize(padded, (10, 5), interpolation=cv2.INTER_NEAREST)
        self.assertFalse(np.array_equal(direct, original))

    def test_odd_letterbox_padding_is_removed(self):
        # 6x10 -> 12x20 plus 3/4 pixels padding = 19x20.
        original = np.zeros((6, 10), dtype=np.uint8); original[1:5, 2:7] = 1
        large = cv2.resize(original, (20, 12), interpolation=cv2.INTER_NEAREST)
        padded = np.pad(large, ((3, 4), (0, 0)))
        np.testing.assert_array_equal(restore_masks(torch.tensor(padded[None]), original.shape)[0], original.astype(bool))

    def test_retina_checks_original_shape(self):
        np.testing.assert_array_equal(restore_masks(torch.tensor(self.gt[None]), self.gt.shape, True)[0], self.gt)
        with self.assertRaises(EvaluationError):
            restore_masks(torch.tensor(self.gt[None]), (20, 20), True)

    def test_micro_is_pixel_sum_and_subset_denominators_csv_roundtrip(self):
        rows = []
        for index, (gt, pred) in enumerate(((self.gt, self.gt), (self.gt, self.roi), (np.zeros_like(self.gt), self.gt))):
            r = dict(split="valid", model_name="fixture", class_id=0)
            r.update(measure(gt, pred, self.roi)); rows.append(r)
        summary = summarize_rows(rows)[0]
        self.assertEqual(summary["Micro_Pixel_Precision"], .5)  # 2000/4000; not mean of P
        self.assertEqual(summary["Micro_Pixel_Recall"], 1)
        self.assertEqual(summary["MAE_gt_positive"], .25)
        self.assertEqual(summary["MAE_gt_negative"], .5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_csv(path, rows)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(summarize_rows(list(csv.DictReader(stream))), summarize_rows(rows))

    def test_empty_subset_is_na(self):
        r = dict(split="test", model_name="fixture", class_id=2)
        r.update(measure(self.gt, self.gt, self.roi))
        self.assertIsNone(summarize_rows([r])[0]["MAE_gt_negative"])
        self.assertEqual(pixel_scores(0, 0, 1), {"Pixel_Precision": 0, "Pixel_Recall": 0, "Pixel_F1": 0})

    def test_gt_all_classes_union_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.txt"
            path.write_text("\n".join(f"{c} 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5" for c in [0, 0, 1, 2, 3]), encoding="utf-8")
            masks, counts = load_gt(path, (10, 10))
            self.assertEqual(counts, [2, 1, 1, 1])
            self.assertEqual([int(p.sum()) for p in masks.values()], [25] * 4)
            path.write_text("", encoding="utf-8")
            masks, counts = load_gt(path, (10, 10))
            self.assertEqual(counts, [0] * 4)
            self.assertTrue(all(not p.any() for p in masks.values()))
            with self.assertRaises(EvaluationError):
                load_gt(Path(tmp) / "missing.txt", (10, 10))

    def test_bad_polygons_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.txt"
            for text in ("4 0 0 1 0 1 1", "0 0 0 1 1", "0 nan 0 1 0 1 1", "0 0 0 2 0 1 1", "0 0 0 0 0 0 0"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(EvaluationError):
                    load_gt(path, (10, 10))

    def test_ellipse_exact_django_geometry(self):
        for h, w in ((11, 22), (500, 721), (1, 1)):
            expected = np.zeros((h, w), np.uint8)
            cv2.ellipse(expected, (w // 2, h // 2), (max(1, int(w / 2 * 1.01)), max(1, int(h / 2 * 1.01))), 0, 0, 360, 255, -1)
            np.testing.assert_array_equal(roi_mask((h, w), "django-ellipse"), expected > 0)

    def test_lfs_pointer_identifies_target_not_pointer_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Defect.pt"
            oid = "a" * 64
            path.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 23427096\n", encoding="utf-8")
            ref = checkpoint_reference(path)
            self.assertEqual(ref["kind"], "git-lfs-pointer")
            self.assertEqual(ref["target_sha256"], oid)
            self.assertNotEqual(ref["file_sha256"], oid)


if __name__ == "__main__":
    unittest.main()
