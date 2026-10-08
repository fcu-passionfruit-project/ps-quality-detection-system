import tempfile
import unittest
from pathlib import Path



from passionfruit.data_preparation.generate_class_aware_train_list import (  # noqa: E402
    CLASS_REPEAT_FACTORS,
    generate_class_aware_train_list,
)


class ClassAwareTrainListTests(unittest.TestCase):
    def test_generates_max_factor_list_and_instance_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "local_defects"
            self._make_dataset(dataset)
            original_contents = self._source_contents(dataset)

            report = generate_class_aware_train_list(dataset)

            output = dataset / "experiments/train_B_class_aware.txt"
            self.assertEqual(report["original_train_images"], 6)
            self.assertEqual(report["generated_train_entries"], 13)
            self.assertEqual(report["images_by_repeat_factor"], {1: 3, 2: 1, 4: 2})
            self.assertEqual(report["images_containing"], {0: 3, 1: 2, 2: 2, 3: 1})
            self.assertEqual(report["null_images"], 1)
            self.assertEqual(report["original_exposure"], {0: 4, 1: 2, 2: 2, 3: 1})
            self.assertEqual(report["sampled_exposure"], {0: 8, 1: 6, 2: 8, 3: 1})

            entries = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(entries.count("./../train/images/abrasion.jpg"), 1)
            self.assertEqual(entries.count("./../train/images/anthracnose.jpg"), 2)
            self.assertEqual(entries.count("./../train/images/insect_bite.jpg"), 4)
            self.assertEqual(entries.count("./../train/images/mixed_rare.jpg"), 4)
            self.assertNotIn("valid", "\n".join(entries))
            self.assertNotIn("test", "\n".join(entries))
            self.assertEqual(self._source_contents(dataset), original_contents)

    def test_rejects_unknown_class_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "local_defects"
            self._make_dataset(dataset)
            (dataset / "train/labels/abrasion.txt").write_text(
                "4 0.1 0.1 0.2 0.1 0.2 0.2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "未知 class ID 4"):
                generate_class_aware_train_list(dataset)

            self.assertFalse((dataset / "experiments").exists())

    def test_rejects_missing_label_and_invalid_factor(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "local_defects"
            self._make_dataset(dataset)
            (dataset / "train/labels/null.txt").unlink()

            with self.assertRaisesRegex(ValueError, "缺少對應 label"):
                generate_class_aware_train_list(dataset)

            invalid_factors = dict(CLASS_REPEAT_FACTORS)
            invalid_factors[2] = 0
            with self.assertRaisesRegex(ValueError, "大於等於 1"):
                generate_class_aware_train_list(dataset, repeat_factors=invalid_factors)

    @staticmethod
    def _make_dataset(root: Path) -> None:
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)

        train_labels = {
            "abrasion": (
                "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
                "0 0.3 0.3 0.4 0.3 0.4 0.4\n"
            ),
            "anthracnose": (
                "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
                "1 0.3 0.3 0.4 0.3 0.4 0.4\n"
            ),
            "insect_bite": (
                "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
                "2 0.3 0.3 0.4 0.3 0.4 0.4\n"
            ),
            "insect_track": "3 0.1 0.1 0.2 0.1 0.2 0.2\n",
            "mixed_rare": (
                "1 0.1 0.1 0.2 0.1 0.2 0.2\n"
                "2 0.3 0.3 0.4 0.3 0.4 0.4\n"
            ),
            "null": "",
        }
        for stem, label in train_labels.items():
            (root / f"train/images/{stem}.jpg").write_bytes(b"test image")
            (root / f"train/labels/{stem}.txt").write_text(label, encoding="utf-8")

        for split in ("valid", "test"):
            (root / f"{split}/images/{split}.jpg").write_bytes(b"evaluation image")
            (root / f"{split}/labels/{split}.txt").write_text(
                "0 0.1 0.1 0.2 0.1 0.2 0.2\n",
                encoding="utf-8",
            )

    @staticmethod
    def _source_contents(root: Path) -> dict[Path, bytes]:
        return {
            path.relative_to(root): path.read_bytes()
            for split in ("train", "valid", "test")
            for path in (root / split).rglob("*")
            if path.is_file()
        }


if __name__ == "__main__":
    unittest.main()
