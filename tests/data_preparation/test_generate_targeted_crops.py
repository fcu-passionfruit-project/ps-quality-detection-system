import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image



from passionfruit.data_preparation.generate_targeted_crops import (  # noqa: E402
    clip_polygon_to_rectangle,
    generate_targeted_crops,
    polygon_area,
)


class TargetedCropTests(unittest.TestCase):
    def test_polygon_clipping_preserves_true_polygon_shape(self):
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 5.0), (0.0, 10.0)]

        clipped = clip_polygon_to_rectangle(polygon, 2.0, 2.0, 8.0, 8.0)

        self.assertGreaterEqual(len(clipped), 3)
        self.assertTrue(all(2.0 <= x <= 8.0 and 2.0 <= y <= 8.0 for x, y in clipped))
        self.assertGreater(polygon_area(clipped), 0.0)
        self.assertNotEqual(clipped, [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)])

    def test_generates_instance_crops_preserves_other_classes_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first/local_defects"
            second = root / "second/local_defects"
            self._make_dataset(first)
            self._make_dataset(second)
            first_weight = root / "first/yolo26s-seg.pt"
            second_weight = root / "second/yolo26s-seg.pt"
            first_weight.write_bytes(b"same initial weight")
            second_weight.write_bytes(b"same initial weight")
            first_source = self._source_contents(first)
            second_source = self._source_contents(second)

            first_report = generate_targeted_crops(
                dataset=first,
                weight=first_weight,
                minimum_crop_size=20,
                seed=0,
            )
            second_report = generate_targeted_crops(
                dataset=second,
                weight=second_weight,
                minimum_crop_size=20,
                seed=0,
            )

            output = first / "experiments/exp_C_targeted_crop"
            self.assertEqual(first_source, self._source_contents(first))
            self.assertEqual(second_source, self._source_contents(second))
            self.assertEqual(len(list((output / "images").glob("*.png"))), 4)
            self.assertEqual(len(list((output / "labels").glob("*.txt"))), 4)
            self.assertEqual(
                len((first / "experiments/train_C_targeted_crop.txt").read_text(encoding="utf-8").splitlines()),
                7,
            )

            with (output / "manifest.csv").open(encoding="utf-8", newline="") as stream:
                manifest = list(csv.DictReader(stream))
            self.assertEqual(
                [row["target_class_name"] for row in manifest].count("anthracnose"), 1
            )
            self.assertEqual(
                [row["target_class_name"] for row in manifest].count("insect_bite"), 3
            )
            bite_rectangles = {
                (row["crop_x1"], row["crop_y1"], row["crop_x2"], row["crop_y2"])
                for row in manifest
                if row["target_class_name"] == "insect_bite"
            }
            self.assertEqual(len(bite_rectangles), 3)
            self.assertTrue(
                all(
                    int(row["crop_width"]) < int(row["original_width"])
                    and int(row["crop_height"]) < int(row["original_height"])
                    for row in manifest
                )
            )
            self.assertTrue(all(row["source_image"].startswith("train/images/") for row in manifest))

            for label in (output / "labels").glob("*.txt"):
                class_ids = []
                for line in label.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    class_ids.append(int(fields[0]))
                    coordinates = [float(value) for value in fields[1:]]
                    self.assertTrue(all(0.0 <= value <= 1.0 for value in coordinates))
                    self.assertGreaterEqual(len(coordinates), 6)
                self.assertIn(0, class_ids, "crop 中可見的非 target abrasion 必須保留")

            validation = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
            checks = validation["checks"]
            self.assertEqual(checks["malformed_labels"], 0)
            self.assertEqual(checks["out_of_range_coordinates"], 0)
            self.assertEqual(checks["target_class_instance_missing"], 0)
            self.assertEqual(
                first_report["statistics"]["experiment_c"]["generated_crop_images"], 4
            )
            self.assertEqual(
                first_report["reproducibility"]["experiment_dataset_sha256"],
                second_report["reproducibility"]["experiment_dataset_sha256"],
            )
            self.assertEqual(
                (output / "manifest.csv").read_bytes(),
                (second / "experiments/exp_C_targeted_crop/manifest.csv").read_bytes(),
            )

    def test_rejects_existing_outputs_and_invalid_source_polygon(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "local_defects"
            self._make_dataset(dataset)
            weight = root / "yolo26s-seg.pt"
            weight.write_bytes(b"weight")
            generate_targeted_crops(dataset=dataset, weight=weight, minimum_crop_size=20)

            with self.assertRaisesRegex(ValueError, "--overwrite"):
                generate_targeted_crops(dataset=dataset, weight=weight, minimum_crop_size=20)

            invalid = root / "invalid/local_defects"
            self._make_dataset(invalid)
            (invalid / "train/labels/fruit_001_img_01.txt").write_text(
                "1 1.2 0.4 0.5 0.4 0.5 0.5\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "介於 0 到 1"):
                generate_targeted_crops(dataset=invalid, weight=weight, minimum_crop_size=20)

    def test_experiment_d_replays_c_random_realization_without_rng_cascade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "local_defects"
            self._make_dataset(dataset)
            weight = root / "yolo26s-seg.pt"
            weight.write_bytes(b"paired initial weight")
            generate_targeted_crops(
                dataset=dataset,
                weight=weight,
                minimum_crop_size=20,
                seed=0,
            )

            report = generate_targeted_crops(
                dataset=dataset,
                weight=weight,
                minimum_crop_size=20,
                context_scale=6.0,
                seed=0,
                experiment="D",
                paired_reference_dir=Path("experiments/exp_C_targeted_crop"),
                enforce_experiment_d_contract=False,
            )

            output = dataset / "experiments/exp_D_context6"
            replay = json.loads(
                (output / "experiment_c_replay_report.json").read_text(
                    encoding="utf-8"
                )
            )
            audit = json.loads(
                (output / "experiment_c_vs_d_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            with (output / "paired_parameters.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                paired = list(csv.DictReader(stream))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(replay["record_sequence_same"], 4)
            self.assertEqual(replay["crop_rectangles_same"], 4)
            self.assertEqual(replay["generated_image_geometry_and_pixels_same"], 4)
            self.assertEqual(replay["labels_same"], 4)
            self.assertEqual(len(paired), 4)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(
                audit["duplicates"]["duplicate_rectangle_extra_records"], 0
            )
            self.assertEqual(
                audit["duplicates"]["exact_duplicate_image_extra_records"], 0
            )
            self.assertEqual(
                audit["random_pairing"]["context_variation_paired"]
                + audit["random_pairing"]["records_unable_to_remain_fully_paired"],
                4,
            )
            self.assertIn("experiment_d", report["statistics"])
            self.assertIn("crop_exposure", report)

    @staticmethod
    def _make_dataset(root: Path) -> None:
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
        (root / "data.yaml").write_text(
            "train: ../train/images\n"
            "val: ../valid/images\n"
            "test: ../test/images\n\n"
            "nc: 4\n"
            "names: ['abrasion', 'anthracnose', 'insect_bite', 'insect_track']\n",
            encoding="utf-8",
        )
        train_labels = {
            "fruit_001_img_01": (
                "0 0.20 0.20 0.80 0.20 0.80 0.80 0.20 0.80\n"
                "1 0.40 0.40 0.50 0.40 0.50 0.50 0.40 0.50\n"
                "2 0.55 0.45 0.65 0.45 0.65 0.55 0.55 0.55\n"
            ),
            "abrasion_only": "0 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20\n",
            "null": "",
        }
        for index, (stem, label) in enumerate(train_labels.items()):
            Image.new("RGB", (100, 100), (100 + index, 120, 140)).save(
                root / f"train/images/{stem}.jpg"
            )
            (root / f"train/labels/{stem}.txt").write_text(label, encoding="utf-8")
        for index, split in enumerate(("valid", "test")):
            stem = f"fruit_00{index + 2}_img_01"
            Image.new("RGB", (100, 100), (80 + index, 90, 100)).save(
                root / f"{split}/images/{stem}.jpg"
            )
            (root / f"{split}/labels/{stem}.txt").write_text(
                "3 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20\n",
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
