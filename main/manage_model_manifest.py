from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ultralytics import YOLO

from decision_dataset import (
    EXPECTED_CLASS_VARIANTS,
    classes_match_role,
    load_yaml,
    normalize_classes,
    sha256_file,
    validate_model_bundle,
    validate_model_record,
    write_yaml_atomic,
)
from model_manager import ROLE_TASKS


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLES = tuple(ROLE_TASKS)


def _path_in_models(path: Path) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    models_root = (REPO_ROOT / "models").resolve()
    if models_root not in resolved.parents:
        raise ValueError(f"權重檔必須位於 {models_root}")
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到權重檔：{resolved}")
    return resolved, resolved.relative_to(REPO_ROOT).as_posix()


def inspect_weight(role: str, path: Path, imgsz: int, threshold: float | None) -> dict:
    resolved, relative = _path_in_models(path)
    model = YOLO(str(resolved))
    task = getattr(model, "task", None)
    if task != ROLE_TASKS[role]:
        raise ValueError(
            f"{role} 應為 {ROLE_TASKS[role]} 模型，權重內是 {task or 'unknown'}。"
        )
    classes = normalize_classes(getattr(model, "names", {}))
    if not classes_match_role(role, classes):
        expected = [sorted(variant) for variant in EXPECTED_CLASS_VARIANTS[role]]
        raise ValueError(
            f"{role} classes 不符：{sorted(classes.values())}；"
            f"允許：{expected}"
        )
    if role in {"roi", "defect"} and threshold is None:
        raise ValueError(f"{role} 必須指定 --confidence-threshold。")
    if role in {"color", "wrinkle"} and threshold is not None:
        raise ValueError(f"{role} 是分類模型，不可指定 --confidence-threshold。")
    checkpoint = getattr(model, "ckpt", {}) or {}
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    architecture = Path(str(train_args.get("model") or resolved.stem)).stem
    return {
        "status": "candidate",
        "path": relative,
        "sha256": sha256_file(resolved),
        "task": task,
        "architecture": architecture,
        "imgsz": imgsz,
        "classes": classes,
        "confidence_threshold": threshold,
        "size_bytes": resolved.stat().st_size,
    }


def _load_or_create(path: Path) -> dict:
    if path.exists():
        return load_yaml(path)
    return {"schema_version": 1, "bundle_version": "unreleased", "models": {}}


def _print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="維護決策層 Model Bundle manifest。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "models" / "manifest.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show", help="顯示 manifest。")
    subparsers.add_parser("validate", help="驗證全部四個模型。")

    update = subparsers.add_parser("update", help="載入權重並更新一個 candidate。")
    update.add_argument("--role", required=True, choices=ROLES)
    update.add_argument("--model", required=True, type=Path)
    update.add_argument("--imgsz", required=True, type=int)
    update.add_argument("--confidence-threshold", type=float)
    update.add_argument("--dry-run", action="store_true")

    approve = subparsers.add_parser("approve", help="核准一個 candidate 權重。")
    approve.add_argument("--role", required=True, choices=ROLES)
    approve.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    try:
        manifest = _load_or_create(manifest_path)
        if args.command == "show":
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        if args.command == "validate":
            _, errors = validate_model_bundle(manifest, REPO_ROOT)
            _print_errors(errors)
            if errors:
                return 1
            print("Model Bundle 驗證通過。")
            return 0
        updated = copy.deepcopy(manifest)
        models = updated.setdefault("models", {})
        if args.command == "update":
            if args.imgsz <= 0:
                raise ValueError("--imgsz 必須是正整數。")
            if args.confidence_threshold is not None and not (
                0 < args.confidence_threshold <= 1
            ):
                raise ValueError("--confidence-threshold 必須介於 0 與 1。")
            models[args.role] = inspect_weight(
                args.role, args.model, args.imgsz, args.confidence_threshold
            )
        else:
            record = models.get(args.role)
            if not isinstance(record, dict):
                raise ValueError(f"manifest 缺少 models.{args.role}。")
            if record.get("status") != "candidate":
                raise ValueError(f"models.{args.role}.status 必須是 candidate。")
            candidate = {**record, "status": "approved_for_use"}
            _, errors = validate_model_record(
                args.role, candidate, REPO_ROOT, require_approved=True
            )
            if errors:
                _print_errors(errors)
                return 1
            models[args.role] = candidate
        if args.dry_run:
            print(json.dumps(models[args.role], ensure_ascii=False, indent=2))
            print("預覽完成，未修改 manifest。")
            return 0
        write_yaml_atomic(manifest_path, updated, backup=True)
        print(f"已更新 {manifest_path}，舊版備份為 {manifest_path.name}.bak。")
        return 0
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
