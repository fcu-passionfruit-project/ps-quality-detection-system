from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from passionfruit.inference.image_processing import DEFAULT_PARAMETERS
from passionfruit.inference.model_manager import PassionFruitModelManager, ROLE_TASKS
from passionfruit.inference.pipeline import DEFECT_CLASSES, run_pipeline_batch

from ._shared import (
    EXPECTED_CLASSES,
    FIELDNAMES,
    SPLITS,
    VERSION_PATTERN,
    VIEWS,
    load_yaml,
    sha256_file,
    write_dict_rows,
    write_json_atomic,
)
from .layout import validate_dataset_layout
from .manifest import validate_model_bundle
from .types import (
    CancelCheck,
    FruitInput,
    GenerationMode,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
)


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
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCheck | None = None,
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
    stopped = False
    if only_errors:
        total = sum(
            rows.get((fruit.fruit_id, current_view), {}).get("status") == "error"
            for fruit in selected_fruits
            for current_view in ([view_id] if view_id is not None else VIEWS)
        )
    elif fruit_id is None:
        total = sum(
            (fruit.fruit_id, current_view) not in rows
            for fruit in selected_fruits
            for current_view in VIEWS
        )
    else:
        total = 1 if view_id is not None else 3
    if progress:
        progress(
            GenerationProgress(
                "started",
                processed=0,
                total=total,
                errors=0,
                message="開始產生決策資料集。",
            )
        )

    for fruit in selected_fruits:
        if cancel_requested and cancel_requested():
            stopped = True
            break
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
        if progress:
            progress(
                GenerationProgress(
                    "fruit_completed",
                    processed=processed,
                    total=total,
                    errors=errors,
                    fruit_id=fruit.fruit_id,
                    message=f"fruit_id={fruit.fruit_id} 已完成 checkpoint。",
                )
            )

    schema_path = version_dir / "decision_dataset.schema.md"
    if not schema_path.exists():
        schema_path.write_text(dataset_schema_markdown(), encoding="utf-8")
    remaining_errors = sum(row["status"] == "error" for row in rows.values())
    if progress:
        progress(
            GenerationProgress(
                "stopped" if stopped else "completed",
                processed=processed,
                total=total,
                errors=remaining_errors,
                message="已安全停止。" if stopped else "決策資料集處理完成。",
            )
        )
    return {
        "ok": not stopped and remaining_errors == 0 and not layout_errors,
        "processed": processed,
        "skipped": skipped,
        "processed_errors": errors,
        "errors": remaining_errors,
        "layout_errors": len(layout_errors),
        "output": str(csv_path),
        "stopped": stopped,
    }


def validate_generation(request: GenerationRequest) -> dict:
    request.validate_mode()
    return generate_decision_dataset(
        repo_root=request.repo_root,
        dataset_root=request.dataset_root,
        model_manifest_path=request.model_manifest_path,
        output_root=request.output_root,
        dataset_version=request.dataset_version,
        validate_only=True,
        fruit_id=request.fruit_id,
        view_id=request.view_id,
        labels_from=request.labels_from,
        load_weights=request.load_weights,
    )


def generate_dataset(
    request: GenerationRequest,
    *,
    manager: PassionFruitModelManager | None = None,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCheck | None = None,
) -> GenerationResult:
    request.validate_mode()
    flags = request.engine_flags()
    result = generate_decision_dataset(
        repo_root=request.repo_root,
        dataset_root=request.dataset_root,
        model_manifest_path=request.model_manifest_path,
        output_root=request.output_root,
        dataset_version=request.dataset_version,
        fruit_id=request.fruit_id,
        view_id=request.view_id,
        labels_from=request.labels_from,
        manager=manager,
        load_weights=request.load_weights,
        progress=progress,
        cancel_requested=cancel_requested,
        **flags,
    )
    return GenerationResult(**result)


def dataset_schema_markdown() -> str:
    fields = "\n".join(f"{index}. `{name}`" for index, name in enumerate(FIELDNAMES, 1))
    return f"""# Decision Dataset Schema

`grade`：空白代表尚未標註；`0 = 加工`、`1 = 下等`、`2 = 中等`、`3 = 上等`。

`fruit_id`、`view_id`、`split`、`status`、`error_reason` 與 `grade` 是管理或標籤欄位，不可作為 XGBoost 特徵。第一版 baseline 僅使用 `fruit_area_ratio`、`color_bad`、`wrinkle` 與四個 `*_area_ratio`。

推論成功但沒有偵測到某瑕疵時，該類 area ratio 與 max confidence 為 `0.0`；ROI 或模型推論失敗時，所有模型特徵留空並以 `status=error` 記錄原因。

## 欄位順序

{fields}
"""
