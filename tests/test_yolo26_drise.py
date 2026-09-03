from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from drise.yolo_seg import (  # noqa: E402
    DriseError,
    InstancePrediction,
    compute_target_similarity,
    mask_iou,
    normalize_saliency,
    select_target_instance,
)


def _instance(index: int, class_id: int, confidence: float, mask: np.ndarray):
    names = ("abrasion", "anthracnose", "insect_bite", "insect_track")
    return InstancePrediction(
        index=index,
        class_id=class_id,
        class_name=names[class_id],
        confidence=confidence,
        box_xyxy=np.zeros(4),
        mask=mask.astype(bool),
    )


class MaskIouTests(unittest.TestCase):
    def test_identical_masks_equal_one(self):
        mask = np.array([[1, 0], [1, 0]], dtype=bool)
        self.assertEqual(mask_iou(mask, mask), 1.0)

    def test_non_overlapping_masks_equal_zero(self):
        first = np.array([[1, 0], [0, 0]], dtype=bool)
        second = np.array([[0, 0], [0, 1]], dtype=bool)
        self.assertEqual(mask_iou(first, second), 0.0)

    def test_partial_overlap_matches_hand_calculation(self):
        first = np.array([[1, 1], [0, 0]], dtype=bool)
        second = np.array([[0, 1], [1, 0]], dtype=bool)
        self.assertAlmostEqual(mask_iou(first, second), 1 / 3)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(DriseError, "座標/尺寸不一致"):
            mask_iou(np.zeros((2, 2)), np.zeros((3, 2)))


class TargetMatchingTests(unittest.TestCase):
    def test_mask_iou_has_priority_over_candidate_confidence(self):
        target_mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        target = _instance(1, 3, 0.747, target_mask)
        wrong_high_confidence = _instance(
            0, 3, 0.99, np.array([[0, 0, 0], [1, 1, 0]])
        )
        correct_lower_confidence = _instance(1, 3, 0.60, target_mask)
        similarity, matched, iou = compute_target_similarity(
            target, [wrong_high_confidence, correct_lower_confidence]
        )
        self.assertIs(matched, correct_lower_confidence)
        self.assertEqual(iou, 1.0)
        self.assertAlmostEqual(similarity, 0.60)

    def test_other_classes_do_not_match(self):
        mask = np.ones((2, 2), dtype=bool)
        similarity, matched, iou = compute_target_similarity(
            _instance(0, 3, 0.8, mask), [_instance(1, 2, 0.99, mask)]
        )
        self.assertEqual((similarity, matched, iou), (0.0, None, 0.0))

    def test_ambiguous_target_class_requires_index(self):
        mask = np.ones((2, 2), dtype=bool)
        instances = [_instance(0, 3, 0.8, mask), _instance(1, 3, 0.7, mask)]
        with self.assertRaisesRegex(DriseError, "請明確提供 --target-index"):
            select_target_instance(instances, None, "insect_track")

    def test_missing_target_index_lists_available_instances(self):
        mask = np.ones((2, 2), dtype=bool)
        instances = [_instance(0, 3, 0.8, mask), _instance(1, 0, 0.7, mask)]
        with self.assertRaisesRegex(
            DriseError,
            r"Target instance index 3 does not exist\. Available instances: 0, 1",
        ):
            select_target_instance(instances, 3, None)

    def test_no_predictions_is_an_error(self):
        with self.assertRaisesRegex(DriseError, "原始圖片沒有任何 prediction"):
            select_target_instance([], 0, None)


class DeterminismAndFusionTests(unittest.TestCase):
    def test_torch_seed_reproduces_microsoft_mask_generation(self):
        try:
            from vision_explanation_methods.explanations.drise import generate_mask
        except Exception as error:
            self.skipTest(f"optional vision-explanation-methods unavailable: {error}")
        import torch

        torch.manual_seed(7)
        first = generate_mask((4, 4), (12, 14), 4, "cpu")
        torch.manual_seed(7)
        second = generate_mask((4, 4), (12, 14), 4, "cpu")
        self.assertTrue(torch.equal(first, second))

    def test_streaming_centering_and_normalization(self):
        try:
            import torch
            from vision_explanation_methods.explanations.drise import (
                MaskAffinityRecord,
                saliency_fusion,
            )
        except Exception as error:
            self.skipTest(f"optional vision-explanation-methods unavailable: {error}")
        masks = [
            np.array([[1, 0], [0, 1]], dtype=float),
            np.array([[0, 1], [1, 0]], dtype=float),
            np.array([[1, 1], [0, 0]], dtype=float),
        ]
        scores = [0.2, 0.7, 0.5]
        weighted = sum(score * mask for score, mask in zip(scores, masks))
        raw = weighted - np.mean(scores) * sum(masks)
        expected = (raw - raw.min()) / (raw.max() - raw.min())
        np.testing.assert_allclose(normalize_saliency(raw), expected, atol=1e-7)
        records = [
            MaskAffinityRecord(
                mask=torch.from_numpy(np.repeat(mask[None, :, :], 3, axis=0)),
                affinity_scores=[torch.tensor([score], dtype=torch.float64)],
            )
            for score, mask in zip(scores, masks)
        ]
        microsoft = saliency_fusion(records, "cpu")[0][0]["detection"][0].numpy()
        np.testing.assert_allclose(normalize_saliency(raw), microsoft, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
