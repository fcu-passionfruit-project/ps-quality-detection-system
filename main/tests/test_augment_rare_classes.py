import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from ultralytics.data.utils import check_det_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.augment_rare_classes import augment_dataset, transform_point  # noqa: E402


class RareClassAugmentationTests(unittest.TestCase):
    def test_transform_point(self):
        self.assertEqual(transform_point(0.2, 0.3, "hflip"), (0.8, 0.3))
        self.assertEqual(transform_point(0.2, 0.3, "vflip"), (0.2, 0.7))
        self.assertEqual(transform_point(0.2, 0.3, "rotate90"), (0.7, 0.2))
        self.assertEqual(transform_point(0.2, 0.3, "rotate180"), (0.8, 0.7))
        self.assertEqual(transform_point(0.2, 0.3, "rotate270"), (0.3, 0.8))

    def test_augments_train_only_and_uses_insect_bite_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "v2_defect_raw"
            output = root / "v2_defect_targeted"
            self._make_dataset(source)
            valid_before = (source / "valid/images/valid.jpg").read_bytes()
            test_before = (source / "test/labels/test.txt").read_bytes()

            report = augment_dataset(source, output, seed=0)

            self.assertEqual(report["planned_generated_images"], 12)
            self.assertEqual(report["after"]["train"]["images"], 16)
            self.assertEqual(len(list((output / "train/images").glob("*__rare_aug_*"))), 12)
            self.assertEqual((output / "valid/images/valid.jpg").read_bytes(), valid_before)
            self.assertEqual((output / "test/labels/test.txt").read_bytes(), test_before)
            manifest = json.loads((output / "augmentation_manifest.json").read_text(encoding="utf-8"))
            both_entries = [item for item in manifest["generated"] if item["source_image"].endswith("both.jpg")]
            self.assertEqual(len(both_entries), 5)
            self.assertTrue(all(item["target_classes"] == ["anthracnose", "insect_bite"] for item in both_entries))
            data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("train: train/images", data_yaml)
            self.assertIn("val: valid/images", data_yaml)
            checked = check_det_dataset(output / "data.yaml", autodownload=False)
            self.assertEqual(Path(checked["train"]), output / "train/images")

    def test_dry_run_does_not_create_output_and_invalid_polygon_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "v2_defect_raw"
            output = root / "v2_defect_targeted"
            self._make_dataset(source)
            report = augment_dataset(source, output, dry_run=True)
            self.assertEqual(report["planned_generated_images"], 12)
            self.assertFalse(output.exists())

            (source / "train/labels/anthracnose.txt").write_text(
                "1 1.2 0.2 0.3 0.2 0.3 0.4\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "座標必須介於 0 到 1"):
                augment_dataset(source, output, dry_run=True)

    @staticmethod
    def _make_dataset(root: Path) -> None:
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
        (root / "data.yaml").write_text(
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            "nc: 4\n"
            "names: [abrasion, anthracnose, insect_bite, insect_track]\n",
            encoding="utf-8",
        )
        train_labels = {
            "anthracnose": "1 0.1 0.2 0.3 0.2 0.3 0.4\n",
            "insect_bite": "2 0.2 0.1 0.4 0.1 0.4 0.3\n",
            "both": "1 0.1 0.2 0.3 0.2 0.3 0.4\n2 0.2 0.1 0.4 0.1 0.4 0.3\n",
            "abrasion": "0 0.1 0.1 0.2 0.1 0.2 0.2\n",
        }
        for stem, label in train_labels.items():
            Image.new("RGB", (12, 8), (100, 120, 140)).save(root / f"train/images/{stem}.jpg")
            (root / f"train/labels/{stem}.txt").write_text(label, encoding="utf-8")
        for split in ("valid", "test"):
            Image.new("RGB", (12, 8), (80, 90, 100)).save(root / f"{split}/images/{split}.jpg")
            (root / f"{split}/labels/{split}.txt").write_text(
                "0 0.1 0.1 0.2 0.1 0.2 0.2\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main()
