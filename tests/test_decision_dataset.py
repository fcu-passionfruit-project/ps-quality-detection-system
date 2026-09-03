import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))

from decision_dataset import (  # noqa: E402
    DEFECT_CLASSES,
    EXPECTED_CLASSES,
    FruitInput,
    GenerationMode,
    LayoutPlan,
    approve_model,
    build_generation_request,
    generate_dataset,
    load_yaml,
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
from decision_dataset.gui import (  # noqa: E402
    LayoutPreviewState,
    generation_request_from_gui,
)
from decision_dataset_cli import build_parser, request_from_args  # noqa: E402


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

    def test_manifest_candidate_can_be_approved_with_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._make_manifest(root)
            manifest = load_yaml(manifest_path)
            manifest["models"]["roi"]["status"] = "candidate"
            write_yaml_atomic(manifest_path, manifest)

            with patch(
                "decision_dataset.manifest.validate_model_record",
                return_value=(manifest["models"]["roi"], []),
            ):
                approved = approve_model(manifest_path, root, "roi")

            self.assertEqual(approved["status"], "approved_for_use")
            self.assertEqual(
                load_yaml(manifest_path)["models"]["roi"]["status"],
                "approved_for_use",
            )
            self.assertTrue(manifest_path.with_suffix(".yaml.bak").is_file())

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

            with patch("decision_dataset.generation.run_pipeline_batch", return_value=(reports, [])):
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

            with patch("decision_dataset.generation.run_pipeline_batch") as inference:
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
                "decision_dataset.generation.run_pipeline_batch",
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
                "decision_dataset.generation.run_pipeline_batch", return_value=(reports, [])
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

    def test_all_five_modes_map_to_engine_flags(self):
        expected = {
            GenerationMode.RESUME: (False, False, False),
            GenerationMode.ONLY_ERRORS: (True, False, False),
            GenerationMode.RERUN_FRUIT: (False, True, False),
            GenerationMode.RERUN_VIEW: (False, True, False),
            GenerationMode.REPLACE_IMAGE: (False, True, True),
        }
        for mode, flags in expected.items():
            request = build_generation_request(
                repo_root=".", dataset_root=".", model_manifest_path="manifest.yaml",
                output_root=".", dataset_version="v001", mode=mode,
                fruit_id=1 if mode in {GenerationMode.RERUN_FRUIT, GenerationMode.RERUN_VIEW, GenerationMode.REPLACE_IMAGE} else None,
                view_id=2 if mode == GenerationMode.RERUN_VIEW else None,
            )
            actual = request.engine_flags()
            self.assertEqual((actual["only_errors"], actual["force"], actual["replace_image"]), flags)

    def test_layout_preview_is_invalidated_when_path_changes(self):
        state = LayoutPreviewState(LayoutPlan(Path("dataset"), tuple(), tuple()))
        self.assertTrue(state.can_apply)
        state.invalidate()
        self.assertFalse(state.can_apply)

    def test_gui_and_cli_build_identical_generation_request(self):
        values = {
            "dataset_root": "DecisionLayerTrain", "manifest": "models/manifest.yaml",
            "output_root": "decision_datasets", "version": "v002",
            "labels_from": "v001", "mode": "重跑單張", "fruit_id": "12", "view_id": "3",
        }
        gui_request = generation_request_from_gui(values, MAIN_DIR.parent)
        args = build_parser().parse_args([
            "generate", "--version", "v002", "--dataset-root", "DecisionLayerTrain",
            "--model-manifest", "models/manifest.yaml", "--output-root", "decision_datasets",
            "--mode", "rerun_view", "--fruit-id", "12", "--view-id", "3", "--labels-from", "v001",
        ])
        self.assertEqual(gui_request, request_from_args(args))

    def test_safe_stop_finishes_current_fruit_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._make_layout(root)
            second = dataset / "train" / "fruit_002"
            second.mkdir()
            for view_id in (1, 2, 3):
                (second / f"img_{view_id:02d}.jpg").write_bytes(f"two-{view_id}".encode())
            manifest = self._make_manifest(root)
            cancel = threading.Event()

            def infer(*_args, **_kwargs):
                cancel.set()
                return [successful_report() for _ in range(3)], []

            request = build_generation_request(
                repo_root=root, dataset_root=dataset, model_manifest_path=manifest,
                output_root=root / "out", dataset_version="v001", load_weights=False,
            )
            with patch("decision_dataset.generation.run_pipeline_batch", side_effect=infer) as inference:
                result = generate_dataset(request, cancel_requested=cancel.is_set)

            self.assertTrue(result.stopped)
            self.assertEqual(result.processed, 3)
            self.assertEqual(inference.call_count, 1)
            rows = read_dataset_rows(result.output)
            self.assertEqual(set(rows), {(1, 1), (1, 2), (1, 3)})


if __name__ == "__main__":
    unittest.main()
