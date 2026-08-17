from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from image_processing import DEFAULT_PARAMETERS
from model_manager import PassionFruitModelManager, ROLE_TASKS
from pipeline import DEFECT_CLASSES, run_pipeline_batch


SPLITS = ("train", "valid", "test")
VIEWS = (1, 2, 3)
LEGACY_GRADES = ("上等", "中等", "下等", "加工")
FRUIT_PATTERN = re.compile(r"^fruit_(\d+)$")
VERSION_PATTERN = re.compile(r"^v\d{3,}$")
VIEW_FILES = {view_id: f"img_{view_id:02d}.jpg" for view_id in VIEWS}
EXPECTED_CLASSES = {
    "roi": {"PF"},
    "color": {"bad_color", "good_color"},
    "wrinkle": {"false(沒皺褶)", "true(有皺褶)"},
    "defect": set(DEFECT_CLASSES),
}
EXPECTED_CLASS_VARIANTS = {
    "roi": ({"PF"}, {"ROI"}),
    "color": (EXPECTED_CLASSES["color"],),
    "wrinkle": (EXPECTED_CLASSES["wrinkle"],),
    "defect": (EXPECTED_CLASSES["defect"],),
}
FIELDNAMES = [
    "fruit_id",
    "view_id",
    "split",
    "fruit_area_ratio",
    "color_bad",
    "color_confidence",
    "wrinkle",
    "wrinkle_confidence",
    "abrasion_area_ratio",
    "abrasion_max_confidence",
    "anthracnose_area_ratio",
    "anthracnose_max_confidence",
    "insect_bite_area_ratio",
    "insect_bite_max_confidence",
    "insect_track_area_ratio",
    "insect_track_max_confidence",
    "status",
    "error_reason",
    "grade",
]


@dataclass(frozen=True)
class FruitInput:
    split: str
    fruit_id: int
    images: dict[int, Path]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: str | Path, value: dict) -> None:
    target = Path(path)
    _atomic_replace(
        target,
        lambda temporary: temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        ),
    )


def load_yaml(path: str | Path) -> dict:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"找不到 manifest：{target}")
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest 根節點必須是 mapping：{target}")
    return value


def write_yaml_atomic(path: str | Path, value: dict, *, backup: bool = True) -> None:
    target = Path(path)
    if backup and target.exists():
        shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
    _atomic_replace(
        target,
        lambda temporary: temporary.write_text(
            yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        ),
    )


def normalize_classes(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def classes_match_role(role: str, names) -> bool:
    actual = set(normalize_classes(names).values())
    return any(actual == variant for variant in EXPECTED_CLASS_VARIANTS[role])


def validate_model_bundle(
    manifest: dict,
    repo_root: str | Path,
    *,
    load_weights: bool = True,
) -> tuple[dict[str, dict], list[str]]:
    root = Path(repo_root).resolve()
    records = manifest.get("models")
    errors: list[str] = []
    if not isinstance(records, dict):
        return {}, ["manifest 缺少 models mapping。"]
    manager = PassionFruitModelManager(device="cpu") if load_weights else None
    resolved: dict[str, dict] = {}
    for role in ROLE_TASKS:
        record = records.get(role)
        if not isinstance(record, dict):
            errors.append(f"manifest 缺少 models.{role}。")
            continue
        if record.get("status") != "approved_for_use":
            errors.append(f"models.{role}.status 必須是 approved_for_use。")
        relative_path = record.get("path")
        if not relative_path:
            errors.append(f"models.{role}.path 不可為空。")
            continue
        if Path(str(relative_path)).is_absolute():
            errors.append(f"models.{role}.path 必須是 repository 相對路徑。")
        path = (root / str(relative_path)).resolve()
        models_root = (root / "models").resolve()
        if models_root not in path.parents:
            errors.append(f"models.{role}.path 必須位於 models/：{path}")
        if not path.is_file():
            errors.append(f"找不到 {role} 權重：{path}")
            continue
        expected_hash = str(record.get("sha256") or "").lower()
        actual_hash = sha256_file(path)
        if expected_hash != actual_hash:
            errors.append(f"{role} 權重 SHA-256 不符。")
        if record.get("size_bytes") != path.stat().st_size:
            errors.append(f"models.{role}.size_bytes 與權重檔不符。")
        if (
            not isinstance(record.get("architecture"), str)
            or not record["architecture"].strip()
        ):
            errors.append(f"models.{role}.architecture 不可為空。")
        if record.get("task") != ROLE_TASKS[role]:
            errors.append(
                f"models.{role}.task 應為 {ROLE_TASKS[role]}，目前是 {record.get('task')}。"
            )
        if not isinstance(record.get("imgsz"), int) or record["imgsz"] <= 0:
            errors.append(f"models.{role}.imgsz 必須是正整數。")
        threshold = record.get("confidence_threshold")
        if role in {"roi", "defect"} and not (
            isinstance(threshold, (int, float)) and 0 < float(threshold) <= 1
        ):
            errors.append(f"models.{role}.confidence_threshold 必須介於 0 與 1。")
        if role in {"color", "wrinkle"} and threshold is not None:
            errors.append(f"models.{role}.confidence_threshold 必須是 null。")
        declared_classes = normalize_classes(record.get("classes") or {})
        if not classes_match_role(role, declared_classes):
            errors.append(f"models.{role}.classes 與正式類別不符。")
        if manager is not None:
            try:
                model = manager.load(role, path)
                actual_classes = normalize_classes(getattr(model, "names", {}))
                if not classes_match_role(role, actual_classes):
                    errors.append(f"{role} 權重內的 classes 與正式類別不符。")
            except Exception as error:
                errors.append(f"{role} 權重載入失敗：{type(error).__name__}: {error}")
        resolved[role] = {**record, "resolved_path": str(path)}
    return resolved, errors


def validate_model_record(
    role: str,
    record: dict,
    repo_root: str | Path,
    *,
    load_weights: bool = True,
    require_approved: bool = True,
) -> tuple[dict | None, list[str]]:
    """Validate one manifest record without requiring a complete bundle."""
    if role not in ROLE_TASKS:
        return None, [f"未知模型角色：{role}"]
    manifest = {"models": {role: record}}
    records = manifest["models"]
    errors: list[str] = []
    root = Path(repo_root).resolve()
    models_root = (root / "models").resolve()
    candidate = records[role]
    if require_approved and candidate.get("status") != "approved_for_use":
        errors.append(f"models.{role}.status 必須是 approved_for_use。")
    relative_path = candidate.get("path")
    if not relative_path:
        return None, [*errors, f"models.{role}.path 不可為空。"]
    path = (root / str(relative_path)).resolve()
    if Path(str(relative_path)).is_absolute():
        errors.append(f"models.{role}.path 必須是 repository 相對路徑。")
    if models_root not in path.parents:
        errors.append(f"models.{role}.path 必須位於 models/：{path}")
    if not path.is_file():
        return None, [*errors, f"找不到 {role} 權重：{path}"]
    if str(candidate.get("sha256") or "").lower() != sha256_file(path):
        errors.append(f"{role} 權重 SHA-256 不符。")
    if candidate.get("size_bytes") != path.stat().st_size:
        errors.append(f"models.{role}.size_bytes 與權重檔不符。")
    if (
        not isinstance(candidate.get("architecture"), str)
        or not candidate["architecture"].strip()
    ):
        errors.append(f"models.{role}.architecture 不可為空。")
    if candidate.get("task") != ROLE_TASKS[role]:
        errors.append(
            f"models.{role}.task 應為 {ROLE_TASKS[role]}，"
            f"目前是 {candidate.get('task')}。"
        )
    if not isinstance(candidate.get("imgsz"), int) or candidate["imgsz"] <= 0:
        errors.append(f"models.{role}.imgsz 必須是正整數。")
    threshold = candidate.get("confidence_threshold")
    if role in {"roi", "defect"} and not (
        isinstance(threshold, (int, float)) and 0 < float(threshold) <= 1
    ):
        errors.append(f"models.{role}.confidence_threshold 必須介於 0 與 1。")
    if role in {"color", "wrinkle"} and threshold is not None:
        errors.append(f"models.{role}.confidence_threshold 必須是 null。")
    if not classes_match_role(role, candidate.get("classes") or {}):
        errors.append(f"models.{role}.classes 與正式類別不符。")
    if load_weights:
        try:
            manager = PassionFruitModelManager(device="cpu")
            model = manager.load(role, path)
            actual_classes = normalize_classes(getattr(model, "names", {}))
            if not classes_match_role(role, actual_classes):
                errors.append(f"{role} 權重內的 classes 與正式類別不符。")
        except Exception as error:
            errors.append(f"{role} 權重載入失敗：{type(error).__name__}: {error}")
    return {**candidate, "resolved_path": str(path)}, errors


def validate_dataset_layout(dataset_root: str | Path) -> tuple[list[FruitInput], list[dict]]:
    root = Path(dataset_root).resolve()
    fruits: list[FruitInput] = []
    errors: list[dict] = []
    seen: dict[int, str] = {}
    if not root.is_dir():
        return [], [{"split": "", "fruit_id": "", "reason": f"找不到資料集：{root}"}]
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            errors.append({"split": split, "fruit_id": "", "reason": "缺少 split 資料夾"})
            continue
        for entry in sorted(split_dir.iterdir(), key=lambda item: item.name):
            if entry.is_file() and entry.name.lower() == "desktop.ini":
                continue
            match = FRUIT_PATTERN.fullmatch(entry.name) if entry.is_dir() else None
            if match is None:
                errors.append(
                    {"split": split, "fruit_id": "", "reason": f"未知項目：{entry.name}"}
                )
                continue
            fruit_id = int(match.group(1))
            if fruit_id in seen:
                errors.append(
                    {
                        "split": split,
                        "fruit_id": fruit_id,
                        "reason": f"fruit_id 已出現在 {seen[fruit_id]}",
                    }
                )
                continue
            seen[fruit_id] = split
            images = {view: entry / filename for view, filename in VIEW_FILES.items()}
            missing = [path.name for path in images.values() if not path.is_file()]
            allowed = set(VIEW_FILES.values()) | {"desktop.ini"}
            unexpected = sorted(
                path.name for path in entry.iterdir() if path.name not in allowed
            )
            if missing or unexpected:
                details = []
                if missing:
                    details.append(f"缺少：{', '.join(missing)}")
                if unexpected:
                    details.append(f"未知：{', '.join(unexpected)}")
                errors.append(
                    {"split": split, "fruit_id": fruit_id, "reason": "；".join(details)}
                )
                continue
            fruits.append(FruitInput(split=split, fruit_id=fruit_id, images=images))
    fruits.sort(key=lambda fruit: (SPLITS.index(fruit.split), fruit.fruit_id))
    return fruits, errors


def plan_layout_migration(dataset_root: str | Path) -> tuple[list[dict], list[dict]]:
    root = Path(dataset_root).resolve()
    moves: list[dict] = []
    errors: list[dict] = []
    targets: set[Path] = set()
    seen_fruit_ids: dict[int, str] = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            errors.append({"split": split, "source": "", "target": "", "reason": "缺少 split"})
            continue
        for flattened in split_dir.iterdir():
            match = FRUIT_PATTERN.fullmatch(flattened.name) if flattened.is_dir() else None
            if match is None:
                continue
            fruit_id = int(match.group(1))
            previous = seen_fruit_ids.setdefault(fruit_id, split)
            if previous != split:
                errors.append(
                    {
                        "split": split,
                        "source": str(flattened),
                        "target": "",
                        "reason": f"fruit_id 已出現在 {previous}",
                    }
                )
        for grade in LEGACY_GRADES:
            grade_dir = split_dir / grade
            if not grade_dir.exists():
                continue
            if not grade_dir.is_dir():
                errors.append(
                        {
                            "split": split,
                            "source": str(grade_dir),
                            "target": "",
                            "reason": "級距路徑不是資料夾",
                        }
                )
                continue
            for source in sorted(grade_dir.iterdir(), key=lambda item: item.name):
                if source.is_file() and source.name.lower() == "desktop.ini":
                    continue
                if not source.is_dir() or FRUIT_PATTERN.fullmatch(source.name) is None:
                    errors.append(
                        {
                            "split": split,
                            "source": str(source),
                            "target": "",
                            "reason": "未知項目",
                        }
                    )
                    continue
                allowed = set(VIEW_FILES.values()) | {"desktop.ini"}
                actual = {item.name for item in source.iterdir()}
                missing = sorted(set(VIEW_FILES.values()) - actual)
                unexpected = sorted(actual - allowed)
                if missing or unexpected:
                    details = []
                    if missing:
                        details.append(f"缺少：{', '.join(missing)}")
                    if unexpected:
                        details.append(f"未知：{', '.join(unexpected)}")
                    errors.append(
                        {
                            "split": split,
                            "source": str(source),
                            "target": str(split_dir / source.name),
                            "reason": "；".join(details),
                        }
                    )
                    continue
                target = split_dir / source.name
                fruit_id = int(FRUIT_PATTERN.fullmatch(source.name).group(1))
                previous = seen_fruit_ids.setdefault(fruit_id, split)
                if previous != split:
                    errors.append(
                        {
                            "split": split,
                            "source": str(source),
                            "target": str(target),
                            "reason": f"fruit_id 已出現在 {previous}",
                        }
                    )
                    continue
                if target.exists() or target in targets:
                    errors.append(
                        {
                            "split": split,
                            "source": str(source),
                            "target": str(target),
                            "reason": "目標衝突",
                        }
                    )
                    continue
                targets.add(target)
                moves.append(
                    {
                        "split": split,
                        "source": str(source),
                        "target": str(target),
                        "reason": "",
                    }
                )
    return moves, errors


def apply_layout_migration(dataset_root: str | Path, moves: list[dict]) -> Path:
    root = Path(dataset_root).resolve()
    mapping_path = root / "layout_migration.csv"
    if mapping_path.exists():
        raise FileExistsError(f"移動對照表已存在：{mapping_path}")
    for move in moves:
        source = Path(move["source"])
        target = Path(move["target"])
        if not source.is_dir():
            raise FileNotFoundError(f"搬移來源不存在：{source}")
        if target.exists():
            raise FileExistsError(f"搬移目標已存在：{target}")
    write_dict_rows(mapping_path, ["split", "source", "target", "reason"], moves)
    for move in moves:
        source = Path(move["source"])
        target = Path(move["target"])
        shutil.move(str(source), str(target))
    for split in SPLITS:
        for grade in LEGACY_GRADES:
            grade_dir = root / split / grade
            if not grade_dir.is_dir():
                continue
            desktop = grade_dir / "desktop.ini"
            remaining = [path for path in grade_dir.iterdir() if path != desktop]
            if not remaining:
                if desktop.exists():
                    desktop.unlink()
                grade_dir.rmdir()
    return mapping_path


def write_dict_rows(path: str | Path, fieldnames: list[str], rows: list[dict]) -> None:
    target = Path(path)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    _atomic_replace(target, write)


def read_dataset_rows(path: str | Path) -> dict[tuple[int, int], dict]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError("decision_dataset.csv 欄位與正式 schema 不符。")
        rows = {}
        for line, row in enumerate(reader, start=2):
            grade = row["grade"].strip()
            if grade not in {"", "0", "1", "2", "3"}:
                raise ValueError(
                    f"grade 只能是空白或 0/1/2/3：第 {line} 列 "
                    f"fruit_id={row['fruit_id']} view_id={row['view_id']}"
                )
            key = (int(row["fruit_id"]), int(row["view_id"]))
            if key in rows:
                raise ValueError(f"CSV 出現重複鍵：fruit_id={key[0]} view_id={key[1]}")
            rows[key] = row
        return rows


def write_dataset_rows(path: str | Path, rows: dict[tuple[int, int], dict]) -> None:
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            SPLITS.index(row["split"]),
            int(row["fruit_id"]),
            int(row["view_id"]),
        ),
    )
    write_dict_rows(path, FIELDNAMES, ordered)


def _empty_features() -> dict[str, str]:
    return {name: "" for name in FIELDNAMES[3:16]}


def report_to_row(report: dict, fruit: FruitInput, view_id: int, grade: str) -> dict:
    row = {
        "fruit_id": str(fruit.fruit_id),
        "view_id": str(view_id),
        "split": fruit.split,
        **_empty_features(),
        "status": "error",
        "error_reason": "",
        "grade": grade,
    }
    roi = report.get("roi") or {}
    models = report.get("models") or {}
    failed = []
    if roi.get("status") != "ok":
        failed.append(f"roi:{roi.get('reason') or 'unknown'}")
    for role in ("color", "wrinkle", "defect"):
        summary = models.get(role) or {}
        if summary.get("status") != "ok":
            failed.append(f"{role}:{summary.get('reason') or 'unknown'}")
    if failed:
        row["error_reason"] = " | ".join(failed)
        return row
    color = models["color"]
    wrinkle = models["wrinkle"]
    if color["class_name"] not in EXPECTED_CLASSES["color"]:
        raise ValueError(f"未知顏色類別：{color['class_name']}")
    if wrinkle["class_name"] not in EXPECTED_CLASSES["wrinkle"]:
        raise ValueError(f"未知皺褶類別：{wrinkle['class_name']}")
    defect = models["defect"]
    row.update(
        fruit_area_ratio=f"{float(roi['roi_area_ratio']):.8f}",
        color_bad="1" if color["class_name"] == "bad_color" else "0",
        color_confidence=f"{float(color['confidence']):.6f}",
        wrinkle="1" if wrinkle["class_name"] == "true(有皺褶)" else "0",
        wrinkle_confidence=f"{float(wrinkle['confidence']):.6f}",
        status="ok",
        error_reason="",
    )
    for class_name in DEFECT_CLASSES:
        row[f"{class_name}_area_ratio"] = (
            f"{float(defect['area_ratios'][class_name]):.8f}"
        )
        row[f"{class_name}_max_confidence"] = (
            f"{float(defect['max_confidences'][class_name]):.6f}"
        )
    return row


def _image_key(fruit_id: int, view_id: int) -> str:
    return f"{fruit_id}:{view_id}"


def build_dataset_manifest(
    version: str,
    source_manifest: dict,
    source_manifest_path: Path,
    fruits: list[FruitInput],
    parameters: dict,
) -> dict:
    images = {}
    for fruit in fruits:
        for view_id, path in fruit.images.items():
            images[_image_key(fruit.fruit_id, view_id)] = {
                "split": fruit.split,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": 1,
        "dataset_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_manifest_sha256": sha256_file(source_manifest_path),
        "model_bundle": source_manifest,
        "parameters": parameters,
        "input_images": images,
    }


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根節點必須是 object：{path}")
    return value


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        timestamp = datetime.now(timezone.utc).isoformat()
        stream.write(f"{timestamp} {message}\n")


def generate_decision_dataset(
    *,
    repo_root: str | Path,
    dataset_root: str | Path,
    model_manifest_path: str | Path,
    output_root: str | Path,
    dataset_version: str,
    validate_only: bool = False,
    fruit_id: int | None = None,
    view_id: int | None = None,
    force: bool = False,
    only_errors: bool = False,
    replace_image: bool = False,
    labels_from: str | None = None,
    manager: PassionFruitModelManager | None = None,
    load_weights: bool = True,
) -> dict:
    root = Path(repo_root).resolve()
    dataset = Path(dataset_root).resolve()
    source_manifest_path = Path(model_manifest_path).resolve()
    output = Path(output_root).resolve()
    manifest = load_yaml(source_manifest_path)
    if VERSION_PATTERN.fullmatch(dataset_version) is None:
        raise ValueError("--version 必須符合 v001、v002 這類格式。")
    records, model_errors = validate_model_bundle(
        manifest, root, load_weights=load_weights
    )
    fruits, layout_errors = validate_dataset_layout(dataset)
    if validate_only:
        return {
            "ok": not model_errors and not layout_errors,
            "model_errors": model_errors,
            "layout_errors": layout_errors,
            "fruit_count": len(fruits),
            "image_count": len(fruits) * 3,
        }
    if model_errors:
        raise ValueError("Model Bundle 驗證失敗：\n" + "\n".join(model_errors))
    if view_id is not None and fruit_id is None:
        raise ValueError("指定 --view-id 時也必須指定 --fruit-id。")
    if only_errors and (fruit_id is not None or force or replace_image):
        raise ValueError("--only-errors 不可與 fruit selector、--force 或 --replace-image 並用。")
    if replace_image and (fruit_id is None or not force):
        raise ValueError("--replace-image 必須與 --fruit-id 及 --force 並用。")

    version_dir = output / dataset_version
    version_dir.mkdir(parents=True, exist_ok=True)
    csv_path = version_dir / "decision_dataset.csv"
    output_manifest_path = version_dir / "dataset_manifest.json"
    error_path = version_dir / "validation_errors.csv"
    log_path = version_dir / "processing.log"
    write_dict_rows(error_path, ["split", "fruit_id", "reason"], layout_errors)

    parameters = {
        **DEFAULT_PARAMETERS,
        "models": {
            role: {
                "imgsz": int(records[role]["imgsz"]),
                "confidence": records[role].get("confidence_threshold"),
            }
            for role in ROLE_TASKS
        },
    }
    if output_manifest_path.exists():
        output_manifest = _load_json(output_manifest_path)
        if output_manifest.get("model_manifest_sha256") != sha256_file(
            source_manifest_path
        ):
            raise ValueError("目前 Model Bundle 與資料版本不同，請建立新版本。")
    else:
        output_manifest = build_dataset_manifest(
            dataset_version, manifest, source_manifest_path, fruits, parameters
        )
        write_json_atomic(output_manifest_path, output_manifest)

    rows = read_dataset_rows(csv_path)
    source_labels: dict[tuple[int, int], str] = {}
    if labels_from:
        source_dir = output / labels_from
        source_rows = read_dataset_rows(source_dir / "decision_dataset.csv")
        source_data_manifest = _load_json(source_dir / "dataset_manifest.json")
        for key, row in source_rows.items():
            current = output_manifest["input_images"].get(_image_key(*key), {})
            previous = source_data_manifest.get("input_images", {}).get(
                _image_key(*key), {}
            )
            if current.get("sha256") == previous.get("sha256"):
                source_labels[key] = row.get("grade", "")

    selected_fruits = fruits
    if fruit_id is not None:
        selected_fruits = [fruit for fruit in fruits if fruit.fruit_id == fruit_id]
        if not selected_fruits:
            raise ValueError(f"找不到 fruit_id={fruit_id}。")

    hash_scope = selected_fruits if fruit_id is not None else fruits
    changed_images = []
    for fruit in hash_scope:
        scoped_views = [view_id] if view_id is not None else VIEWS
        for current_view in scoped_views:
            key_text = _image_key(fruit.fruit_id, current_view)
            saved_hash = output_manifest["input_images"].get(key_text, {}).get(
                "sha256"
            )
            if saved_hash != sha256_file(fruit.images[current_view]):
                changed_images.append(key_text)
    if changed_images and not replace_image:
        preview = ", ".join(changed_images[:10])
        suffix = " ..." if len(changed_images) > 10 else ""
        raise ValueError(
            f"照片內容已變更：{preview}{suffix}；"
            "請指定 --fruit-id、--force 與 --replace-image。"
        )

    model_paths = {role: records[role]["resolved_path"] for role in ROLE_TASKS}
    model_options = {
        role: {
            "imgsz": int(records[role]["imgsz"]),
            "confidence": records[role].get("confidence_threshold"),
        }
        for role in ROLE_TASKS
    }
    manager = manager or PassionFruitModelManager()
    processed = skipped = errors = 0

    for fruit in selected_fruits:
        selected_views = list(VIEWS)
        if view_id is not None:
            selected_views = [view_id]
        keys = [(fruit.fruit_id, current_view) for current_view in selected_views]
        if only_errors:
            selected_views = [
                current_view
                for current_view in selected_views
                if rows.get((fruit.fruit_id, current_view), {}).get("status") == "error"
            ]
        elif fruit_id is None:
            selected_views = [
                current_view
                for current_view in selected_views
                if (fruit.fruit_id, current_view) not in rows
            ]
        elif not force:
            successful = [
                key for key in keys if rows.get(key, {}).get("status") == "ok"
            ]
            if successful:
                raise ValueError("指定目標已有成功結果；重跑必須加上 --force。")
        if not selected_views:
            skipped += len(keys)
            continue

        for current_view in selected_views:
            key_text = _image_key(fruit.fruit_id, current_view)
            current_hash = sha256_file(fruit.images[current_view])
            saved_hash = output_manifest["input_images"].get(key_text, {}).get(
                "sha256"
            )
            if saved_hash != current_hash:
                if not replace_image:
                    raise ValueError(
                        f"照片內容已變更：fruit_id={fruit.fruit_id} view_id={current_view}；"
                        "請使用 --replace-image。"
                    )
                output_manifest["input_images"][key_text] = {
                    "split": fruit.split,
                    "sha256": current_hash,
                }
                if (fruit.fruit_id, current_view) in rows:
                    rows[(fruit.fruit_id, current_view)]["grade"] = ""
                source_labels.pop((fruit.fruit_id, current_view), None)
        if replace_image:
            write_json_atomic(output_manifest_path, output_manifest)

        paths = [fruit.images[current_view] for current_view in selected_views]
        try:
            reports, _ = run_pipeline_batch(
                paths,
                model_paths,
                parameters=DEFAULT_PARAMETERS,
                manager=manager,
                model_options=model_options,
            )
            if len(reports) != len(selected_views):
                raise RuntimeError(
                    f"batch report 數量 {len(reports)} 與輸入數量 "
                    f"{len(selected_views)} 不一致。"
                )
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            reports = [
                {
                    "roi": {"status": "error", "reason": reason},
                    "models": {},
                }
                for _ in selected_views
            ]

        for current_view, report in zip(selected_views, reports, strict=True):
            key = (fruit.fruit_id, current_view)
            existing_grade = rows.get(key, {}).get("grade", "")
            grade = existing_grade or source_labels.get(key, "")
            try:
                row = report_to_row(report, fruit, current_view, grade)
            except Exception as error:
                row = {
                    "fruit_id": str(fruit.fruit_id),
                    "view_id": str(current_view),
                    "split": fruit.split,
                    **_empty_features(),
                    "status": "error",
                    "error_reason": f"{type(error).__name__}: {error}",
                    "grade": grade,
                }
            rows[key] = row
            processed += 1
            errors += row["status"] == "error"
        write_dataset_rows(csv_path, rows)
        _append_log(log_path, f"fruit_id={fruit.fruit_id} views={selected_views}")

    schema_path = version_dir / "decision_dataset.schema.md"
    if not schema_path.exists():
        schema_path.write_text(dataset_schema_markdown(), encoding="utf-8")
    remaining_errors = sum(row["status"] == "error" for row in rows.values())
    return {
        "ok": remaining_errors == 0 and not layout_errors,
        "processed": processed,
        "skipped": skipped,
        "processed_errors": errors,
        "errors": remaining_errors,
        "layout_errors": len(layout_errors),
        "output": str(csv_path),
    }


def dataset_schema_markdown() -> str:
    fields = "\n".join(f"{index}. `{name}`" for index, name in enumerate(FIELDNAMES, 1))
    return f"""# Decision Dataset Schema

`grade`：空白代表尚未標註；`0 = 加工`、`1 = 下等`、`2 = 中等`、`3 = 上等`。

`fruit_id`、`view_id`、`split`、`status`、`error_reason` 與 `grade` 是管理或標籤欄位，不可作為 XGBoost 特徵。第一版 baseline 僅使用 `fruit_area_ratio`、`color_bad`、`wrinkle` 與四個 `*_area_ratio`。

推論成功但沒有偵測到某瑕疵時，該類 area ratio 與 max confidence 為 `0.0`；ROI 或模型推論失敗時，所有模型特徵留空並以 `status=error` 記錄原因。

## 欄位順序

{fields}
"""
