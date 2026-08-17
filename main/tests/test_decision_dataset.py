import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))

from decision_dataset import (  # noqa: E402
    DEFECT_CLASSES,
    EXPECTED_CLASSES,
    FruitInput,
    classes_match_role,
    generate_decision_dataset,
    plan_layout_migration,
    read_dataset_rows,
    report_to_row,
    sha256_file,
    validate_dataset_layout,
    write_dataset_rows,
    write_yaml_atomic,
)
from model_manager import ROLE_TASKS  # noqa: E402


def successful_report():
    return {
        "roi": {"status": "ok", "roi_area_ratio": 0.25},
        "models": {
            "color": {
                "status": "ok",
                "class_name": "bad_color",
                "confidence": 0.8123456,
            },
            "wrinkle": {
                "status": "ok",
                "class_name": "false(沒皺褶)",
                "confidence": 0.9345678,
            },
            "defect": {
                "status": "ok",
                "area_ratios": {
                    name: index / 100
                    for index, name in enumerate(DEFECT_CLASSES, start=1)
                },
                "max_confidences": {
                    name: 0.5 + index / 100
                    for index, name in enumerate(DEFECT_CLASSES, start=1)
                },
            },
        },
    }


class DecisionDatasetTests(unittest.TestCase):
    def test_roi_accepts_pf_or_roi_as_its_single_class(self):
        self.assertTrue(classes_match_role("roi", {0: "PF"}))
        self.assertTrue(classes_match_role("roi", {0: "ROI"}))
        self.assertFalse(classes_match_role("roi", {0: "fruit"}))
        self.assertFalse(classes_match_role("roi", {0: "PF", 1: "ROI"}))

    def _make_layout(self, root: Path) -> Path:
        dataset = root / "DecisionLayerTrain"
        for split in ("train", "valid", "test"):
            (dataset / split).mkdir(parents=True)
        fruit = dataset / "train" / "fruit_001"
        fruit.mkdir()
        for view_id in (1, 2, 3):
            (fruit / f"img_{view_id:02d}.jpg").write_bytes(
                f"image-{view_id}".encode()
            )
        return dataset

    def _make_manifest(self, root: Path) -> Path:
        models = root / "models"
        models.mkdir()
        records = {}
        for role, task in ROLE_TASKS.items():
            weight = models / f"{role}.pt"
            weight.write_bytes(role.encode())
            records[role] = {
                "status": "approved_for_use",
                "path": f"models/{role}.pt",
                "sha256": sha256_file(weight),
                "task": task,
                "architecture": f"fake-{role}",
                "imgsz": 640,
                "classes": {
                    index: name
                    for index, name in enumerate(sorted(EXPECTED_CLASSES[role]))
                },
                "confidence_threshold": 0.25 if role in {"roi", "defect"} else None,
                "size_bytes": weight.stat().st_size,
            }
        path = models / "manifest.yaml"
        write_yaml_atomic(path, {"schema_version": 1, "models": records})
        return path

    def test_report_row_has_fixed_schema_values(self):
        fruit = FruitInput("train", 7, {1: Path("img_01.jpg")})
        row = report_to_row(successful_report(), fruit, 1, "3")

        self.assertEqual(row["fruit_id"], "7")
        self.assertEqual(row["color_bad"], "1")
        self.assertEqual(row["wrinkle"], "0")
        self.assertEqual(row["abrasion_area_ratio"], "0.01000000")
        self.assertEqual(row["abrasion_max_confidence"], "0.510000")
        self.assertEqual(row["grade"], "3")

    def test_layout_validation_requires_three_exact_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._make_layout(Path(temporary))
            fruits, errors = validate_dataset_layout(dataset)
            self.assertEqual(len(fruits), 1)
            self.assertEqual(errors, [])
            (dataset / "train" / "fruit_001" / "extra.jpg").write_bytes(b"x")
            fruits, errors = validate_dataset_layout(dataset)
            self.assertEqual(fruits, [])
            self.assertIn("未知：extra.jpg", errors[0]["reason"])

    def test_migration_plan_preserves_split_and_removes_grade_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                (root / split).mkdir()
            source = root / "train" / "上等" / "fruit_001"
            source.mkdir(parents=True)
            for view_id in (1, 2, 3):
                (source / f"img_{view_id:02d}.jpg").write_bytes(b"image")

            moves, errors = plan_layout_migration(root)

            self.assertEqual(errors, [])
            self.assertEqual(len(moves), 1)
            self.assertEqual(Path(moves[0]["target"]), root / "train" / "fruit_001")
            self.assertTrue(source.is_dir())

    def test_generation_resume_grade_preservation_and_image_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._make_layout(root)
            manifest = self._make_manifest(root)
            output = root / "decision_datasets"
            reports = [successful_report() for _ in range(3)]

            with patch("decision_dataset.run_pipeline_batch", return_value=(reports, [])):
                result = generate_decision_dataset(
                    repo_root=root,
                    dataset_root=dataset,
                    model_manifest_path=manifest,
                    output_root=output,
                    dataset_version="v001",
                    load_weights=False,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(result["processed"], 3)
            csv_path = output / "v001" / "decision_dataset.csv"
            self.assertTrue(csv_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            rows = read_dataset_rows(csv_path)
            rows[(1, 2)]["grade"] = "2"
            write_dataset_rows(csv_path, rows)

            with patch("decision_dataset.run_pipeline_batch") as inference:
                resumed = generate_decision_dataset(
                    repo_root=root,
                    dataset_root=dataset,
                    model_manifest_path=manifest,
                    output_root=output,
                    dataset_version="v001",
                    load_weights=False,
                )
            inference.assert_not_called()
            self.assertEqual(resumed["processed"], 0)

            changed = dataset / "train" / "fruit_001" / "img_02.jpg"
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "--replace-image"):
                generate_decision_dataset(
                    repo_root=root,
                    dataset_root=dataset,
                    model_manifest_path=manifest,
                    output_root=output,
                    dataset_version="v001",
                    fruit_id=1,
                    view_id=2,
                    force=True,
                    load_weights=False,
                )
            with patch(
                "decision_dataset.run_pipeline_batch",
                return_value=([successful_report()], []),
            ):
                generate_decision_dataset(
                    repo_root=root,
                    dataset_root=dataset,
                    model_manifest_path=manifest,
                    output_root=output,
                    dataset_version="v001",
                    fruit_id=1,
                    view_id=2,
                    force=True,
                    replace_image=True,
                    load_weights=False,
                )
            self.assertEqual(read_dataset_rows(csv_path)[(1, 2)]["grade"], "")

    def test_one_invalid_report_does_not_poison_other_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._make_layout(root)
            manifest = self._make_manifest(root)
            invalid = successful_report()
            invalid["models"]["defect"] = {
                "status": "ok",
                "area_ratios": {},
                "max_confidences": {},
            }
            reports = [successful_report(), invalid, successful_report()]

            with patch(
                "decision_dataset.run_pipeline_batch", return_value=(reports, [])
            ):
                result = generate_decision_dataset(
                    repo_root=root,
                    dataset_root=dataset,
                    model_manifest_path=manifest,
                    output_root=root / "decision_datasets",
                    dataset_version="v001",
                    load_weights=False,
                )

            rows = read_dataset_rows(Path(result["output"]))
            self.assertEqual(rows[(1, 1)]["status"], "ok")
            self.assertEqual(rows[(1, 2)]["status"], "error")
            self.assertEqual(rows[(1, 3)]["status"], "ok")
            self.assertEqual(result["errors"], 1)


if __name__ == "__main__":
    unittest.main()
