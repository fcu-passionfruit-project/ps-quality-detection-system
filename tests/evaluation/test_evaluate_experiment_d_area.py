from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from passionfruit.evaluation.evaluate_experiment_d_area import (
    ModelSpec,
    area_record,
    dataset_roi_mask,
    rasterize_yolo_polygons,
    summarize,
    threshold_grid,
    union_prediction_masks,
)


class ExperimentDAreaEvaluationTests(unittest.TestCase):
    def test_threshold_grid_is_inclusive(self):
        values = threshold_grid(0.05, 0.50, 0.025)
        self.assertEqual(len(values), 19)
        self.assertEqual(values[0], 0.05)
        self.assertEqual(values[-1], 0.50)

    def test_gt_and_prediction_use_same_class_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            label = Path(temporary) / "image.txt"
            label.write_text(
                "0 0 0 0.5 0 0.5 0.5 0 0.5\n"
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                encoding="utf-8",
            )
            gt = rasterize_yolo_polygons(label, (5, 5))["abrasion"]
            first = np.zeros((5, 5), dtype=bool); first[:3, :3] = True
            second = np.zeros((5, 5), dtype=bool); second[1:4, 1:4] = True
            pred = union_prediction_masks(
                np.asarray([0, 0]), np.asarray([0.3, 0.8]),
                np.stack([first, second]), "abrasion", 0.25, (5, 5),
            )
            self.assertGreater(np.count_nonzero(gt), 0)
            self.assertEqual(np.count_nonzero(pred), 14)

    def test_both_empty_spatial_metrics_are_excluded(self):
        spec = ModelSpec("D0", 0, Path("best.pt"), "hash")
        empty = np.zeros((4, 4), dtype=bool)
        record = area_record(
            split="val", model=spec, mode="global", threshold=0.2,
            class_threshold=0.2, image="x.jpg", class_name="abrasion",
            gt_mask=empty, pred_mask=empty, roi_mask=np.ones((4, 4), dtype=bool),
        )
        self.assertEqual(record["union_iou"], "")
        self.assertEqual(record["dice"], "")

    def test_summary_zero_gt_fp_and_positive_only(self):
        spec = ModelSpec("D0", 0, Path("best.pt"), "hash")
        records = []
        for class_name in ("abrasion", "anthracnose", "insect_bite", "insect_track"):
            gt = np.zeros((2, 2), dtype=bool)
            pred = np.zeros((2, 2), dtype=bool)
            if class_name == "abrasion":
                pred[0, 0] = True
            records.append(area_record(
                split="val", model=spec, mode="global", threshold=0.2,
                class_threshold=0.2, image="zero.jpg", class_name=class_name,
                gt_mask=gt, pred_mask=pred, roi_mask=np.ones((2, 2), dtype=bool),
            ))
            gt2 = np.zeros((2, 2), dtype=bool); gt2[0, 0] = True
            records.append(area_record(
                split="val", model=spec, mode="global", threshold=0.2,
                class_threshold=0.2, image="positive.jpg", class_name=class_name,
                gt_mask=gt2, pred_mask=pred, roi_mask=np.ones((2, 2), dtype=bool),
            ))
        summary = {row["class"]: row for row in summarize(records, 0.2)}
        self.assertEqual(summary["abrasion"]["zero_gt_fp_rate"], 1.0)
        self.assertEqual(summary["abrasion"]["mean_zero_gt_fp_area"], 0.25)
        self.assertEqual(summary["anthracnose"]["zero_gt_fp_rate"], 0.0)
        self.assertEqual(summary["anthracnose"]["positive_only_area_mae"], 0.25)

    def test_roi_denominator_clips_gt_and_prediction_without_brightness_logic(self):
        spec = ModelSpec("D0", 0, Path("best.pt"), "hash")
        roi = dataset_roi_mask((10, 20))
        gt = np.ones((10, 20), dtype=bool)
        pred = np.zeros((10, 20), dtype=bool)
        pred[roi] = True
        record = area_record(
            split="val", model=spec, mode="global", threshold=0.2,
            class_threshold=0.2, image="dark-fruit.jpg", class_name="abrasion",
            gt_mask=gt, pred_mask=pred, roi_mask=roi,
        )
        self.assertEqual(record["fruit_area_px"], np.count_nonzero(roi))
        self.assertEqual(record["gt_area_px"], np.count_nonzero(roi))
        self.assertEqual(record["gt_area_ratio"], 1.0)
        self.assertEqual(record["pred_area_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
