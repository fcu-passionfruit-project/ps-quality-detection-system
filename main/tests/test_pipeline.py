import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))

from image_processing import (  # noqa: E402
    apply_circular_mask,
    batch_preprocess,
    process_grayscale_clahe,
    select_best_detection,
    validate_parameters,
    write_image,
)
from model_manager import (  # noqa: E402
    ModelTaskMismatch,
    PassionFruitModelManager,
)
from pipeline import run_pipeline  # noqa: E402


class FakeBoxes:
    def __init__(self, xyxy, confidence):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(confidence, dtype=np.float32)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    def __init__(self, xyxy=(), confidence=()):
        self.boxes = FakeBoxes(xyxy, confidence)


class FakeManager:
    device = "cpu"


class ImageProcessingTests(unittest.TestCase):
    def test_mask_grayscale_and_parameter_validation(self):
        image = np.full((80, 100, 3), 180, dtype=np.uint8)
        masked = apply_circular_mask(image, 1.01)
        gray = process_grayscale_clahe(masked, 2.0, 8)
        self.assertEqual(masked.shape, image.shape)
        self.assertEqual(gray.shape, image.shape[:2])
        self.assertEqual(gray.dtype, np.uint8)
        with self.assertRaises(ValueError):
            validate_parameters(0, 0, 1.01, 2.0, 8)

    def test_selects_highest_confidence_detection(self):
        result = FakeResult(
            [[1, 2, 30, 40], [5, 6, 70, 75]],
            [0.25, 0.91],
        )
        detection = select_best_detection(result)
        self.assertEqual(detection["index"], 1)
        self.assertAlmostEqual(detection["confidence"], 0.91, places=5)
        self.assertEqual(select_best_detection(FakeResult()), None)

    def test_recursive_unicode_batch_and_collision_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "輸入" / "train" / "平滑"
            output = root / "輸出"
            source.mkdir(parents=True)
            image_path = source / "百香果.jpg"
            write_image(image_path, np.full((60, 80, 3), 150, dtype=np.uint8))

            def detect(_image, _confidence):
                return FakeResult([[10, 5, 70, 55]], [0.95])

            report = batch_preprocess(
                detect,
                root / "輸入",
                output,
                mode="wrinkle",
            )
            expected = output / "train" / "平滑" / "百香果.jpg"
            self.assertTrue(expected.is_file())
            self.assertEqual(report["counts"]["ok"], 1)
            self.assertTrue((output / "batch_report.json").is_file())
            with self.assertRaises(FileExistsError):
                batch_preprocess(detect, root / "輸入", output)


class PipelineTests(unittest.TestCase):
    def test_missing_roi_stops_downstream_and_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "原圖.jpg"
            write_image(
                image_path, np.full((50, 50, 3), 120, dtype=np.uint8)
            )
            report, artifacts = run_pipeline(
                image_path,
                {"roi": "", "color": "color.pt", "wrinkle": "", "defect": ""},
                manager=FakeManager(),
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["reason"], "roi_model_missing")
            self.assertEqual(
                report["models"]["color"]["reason"], "roi_unavailable"
            )
            self.assertIn("original", artifacts)
            json.dumps(report, ensure_ascii=False)

    def test_model_manager_cpu_fallback_and_task_validation(self):
        with patch("model_manager.torch.cuda.is_available", return_value=False):
            manager = PassionFruitModelManager()
        self.assertEqual(manager.device, "cpu")
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "wrong.pt"
            model_path.touch()
            wrong_model = type("WrongModel", (), {"task": "classify"})()
            with patch("model_manager.YOLO", return_value=wrong_model):
                with self.assertRaises(ModelTaskMismatch):
                    manager.load("roi", model_path)


if __name__ == "__main__":
    unittest.main()
