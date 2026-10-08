from __future__ import annotations

import copy
from pathlib import Path

from ultralytics import YOLO

from passionfruit.inference.model_manager import PassionFruitModelManager, ROLE_TASKS

from ._shared import (
    EXPECTED_CLASS_VARIANTS,
    classes_match_role,
    load_yaml,
    normalize_classes,
    sha256_file,
    write_yaml_atomic,
)


ROLES = tuple(ROLE_TASKS)


def _validate_record_fields(
    role: str,
    record: dict,
    repo_root: Path,
    *,
    require_approved: bool,
) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    if require_approved and record.get("status") != "approved_for_use":
        errors.append(f"models.{role}.status 必須是 approved_for_use。")
    relative_path = record.get("path")
    if not relative_path:
        return None, [*errors, f"models.{role}.path 不可為空。"]
    if Path(str(relative_path)).is_absolute():
        errors.append(f"models.{role}.path 必須是 repository 相對路徑。")
    path = (repo_root / str(relative_path)).resolve()
    models_root = (repo_root / "models").resolve()
    if models_root not in path.parents:
        errors.append(f"models.{role}.path 必須位於 models/：{path}")
    if not path.is_file():
        return None, [*errors, f"找不到 {role} 權重：{path}"]
    if str(record.get("sha256") or "").lower() != sha256_file(path):
        errors.append(f"{role} 權重 SHA-256 不符。")
    if record.get("size_bytes") != path.stat().st_size:
        errors.append(f"models.{role}.size_bytes 與權重檔不符。")
    architecture = record.get("architecture")
    if not isinstance(architecture, str) or not architecture.strip():
        errors.append(f"models.{role}.architecture 不可為空。")
    if record.get("task") != ROLE_TASKS[role]:
        errors.append(
            f"models.{role}.task 應為 {ROLE_TASKS[role]}，"
            f"目前是 {record.get('task')}。"
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
    if not classes_match_role(role, record.get("classes") or {}):
        errors.append(f"models.{role}.classes 與正式類別不符。")
    return path, errors


def validate_model_record(
    role: str,
    record: dict,
    repo_root: str | Path,
    *,
    load_weights: bool = True,
    require_approved: bool = True,
) -> tuple[dict | None, list[str]]:
    if role not in ROLE_TASKS:
        return None, [f"未知模型角色：{role}"]
    root = Path(repo_root).resolve()
    path, errors = _validate_record_fields(
        role, record, root, require_approved=require_approved
    )
    if path is None:
        return None, errors
    if load_weights:
        try:
            manager = PassionFruitModelManager(device="cpu")
            model = manager.load(role, path)
            if not classes_match_role(role, getattr(model, "names", {})):
                errors.append(f"{role} 權重內的 classes 與正式類別不符。")
        except Exception as error:
            errors.append(f"{role} 權重載入失敗：{type(error).__name__}: {error}")
    return {**record, "resolved_path": str(path)}, errors


def validate_model_bundle(
    manifest: dict,
    repo_root: str | Path,
    *,
    load_weights: bool = True,
) -> tuple[dict[str, dict], list[str]]:
    records = manifest.get("models")
    if not isinstance(records, dict):
        return {}, ["manifest 缺少 models mapping。"]
    root = Path(repo_root).resolve()
    manager = PassionFruitModelManager(device="cpu") if load_weights else None
    resolved: dict[str, dict] = {}
    errors: list[str] = []
    for role in ROLE_TASKS:
        record = records.get(role)
        if not isinstance(record, dict):
            errors.append(f"manifest 缺少 models.{role}。")
            continue
        path, record_errors = _validate_record_fields(
            role, record, root, require_approved=True
        )
        errors.extend(record_errors)
        if path is None:
            continue
        if manager is not None:
            try:
                model = manager.load(role, path)
                if not classes_match_role(role, getattr(model, "names", {})):
                    errors.append(f"{role} 權重內的 classes 與正式類別不符。")
            except Exception as error:
                errors.append(f"{role} 權重載入失敗：{type(error).__name__}: {error}")
        resolved[role] = {**record, "resolved_path": str(path)}
    return resolved, errors


def inspect_model(
    repo_root: str | Path,
    role: str,
    model_path: str | Path,
    imgsz: int,
    confidence_threshold: float | None,
) -> dict:
    if role not in ROLE_TASKS:
        raise ValueError(f"未知模型角色：{role}")
    if imgsz <= 0:
        raise ValueError("imgsz 必須是正整數。")
    if confidence_threshold is not None and not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold 必須介於 0 與 1。")
    root = Path(repo_root).resolve()
    resolved = Path(model_path).expanduser().resolve()
    models_root = (root / "models").resolve()
    if models_root not in resolved.parents:
        raise ValueError(f"權重檔必須位於 {models_root}")
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到權重檔：{resolved}")
    model = YOLO(str(resolved))
    task = getattr(model, "task", None)
    if task != ROLE_TASKS[role]:
        raise ValueError(
            f"{role} 應為 {ROLE_TASKS[role]} 模型，權重內是 {task or 'unknown'}。"
        )
    classes = normalize_classes(getattr(model, "names", {}))
    if not classes_match_role(role, classes):
        allowed = [sorted(variant) for variant in EXPECTED_CLASS_VARIANTS[role]]
        raise ValueError(f"{role} classes 不符：{sorted(classes.values())}；允許：{allowed}")
    if role in {"roi", "defect"} and confidence_threshold is None:
        raise ValueError(f"{role} 必須指定 confidence_threshold。")
    if role in {"color", "wrinkle"} and confidence_threshold is not None:
        raise ValueError(f"{role} 是分類模型，不可指定 confidence_threshold。")
    checkpoint = getattr(model, "ckpt", {}) or {}
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    architecture = Path(str(train_args.get("model") or resolved.stem)).stem
    return {
        "status": "candidate",
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
        "task": task,
        "architecture": architecture,
        "imgsz": imgsz,
        "classes": classes,
        "confidence_threshold": confidence_threshold,
        "size_bytes": resolved.stat().st_size,
    }


def load_or_create_manifest(path: str | Path) -> dict:
    target = Path(path)
    if target.exists():
        return load_yaml(target)
    return {"schema_version": 1, "bundle_version": "unreleased", "models": {}}


def update_manifest(
    manifest_path: str | Path,
    repo_root: str | Path,
    role: str,
    model_path: str | Path,
    imgsz: int,
    confidence_threshold: float | None,
    *,
    dry_run: bool = False,
) -> dict:
    target = Path(manifest_path).resolve()
    manifest = copy.deepcopy(load_or_create_manifest(target))
    record = inspect_model(
        repo_root, role, model_path, imgsz, confidence_threshold
    )
    manifest.setdefault("models", {})[role] = record
    if not dry_run:
        write_yaml_atomic(target, manifest, backup=True)
    return record


def approve_model(
    manifest_path: str | Path,
    repo_root: str | Path,
    role: str,
    *,
    dry_run: bool = False,
) -> dict:
    target = Path(manifest_path).resolve()
    manifest = copy.deepcopy(load_yaml(target))
    models = manifest.get("models")
    record = models.get(role) if isinstance(models, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"manifest 缺少 models.{role}。")
    if record.get("status") != "candidate":
        raise ValueError(f"models.{role}.status 必須是 candidate。")
    approved = {**record, "status": "approved_for_use"}
    _, errors = validate_model_record(role, approved, repo_root)
    if errors:
        raise ValueError("\n".join(errors))
    models[role] = approved
    if not dry_run:
        write_yaml_atomic(target, manifest, backup=True)
    return approved


def validate_manifest(
    manifest_path: str | Path,
    repo_root: str | Path,
    *,
    load_weights: bool = True,
) -> tuple[dict[str, dict], list[str]]:
    return validate_model_bundle(
        load_yaml(manifest_path), repo_root, load_weights=load_weights
    )
