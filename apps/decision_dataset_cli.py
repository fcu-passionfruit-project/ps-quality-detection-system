from __future__ import annotations

import argparse
import json
from pathlib import Path

from passionfruit.decision_dataset import (
    GenerationMode,
    ROLES,
    apply_layout,
    approve_model,
    build_generation_request,
    generate_dataset,
    load_or_create_manifest,
    plan_layout,
    update_manifest,
    validate_generation,
    validate_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="決策資料集統一維護工具。")
    commands = parser.add_subparsers(dest="command", required=True)

    layout = commands.add_parser("layout", help="預覽或整理資料夾結構。")
    layout.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "DecisionLayerTrain")
    layout_mode = layout.add_mutually_exclusive_group(required=True)
    layout_mode.add_argument("--dry-run", action="store_true")
    layout_mode.add_argument("--apply", action="store_true")

    manifest = commands.add_parser("manifest", help="維護 Model Manifest。")
    manifest.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.yaml")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_commands.add_parser("show")
    manifest_commands.add_parser("validate")
    update = manifest_commands.add_parser("update")
    update.add_argument("--role", required=True, choices=ROLES)
    update.add_argument("--model", required=True, type=Path)
    update.add_argument("--imgsz", required=True, type=int)
    update.add_argument("--confidence-threshold", type=float)
    update.add_argument("--dry-run", action="store_true")
    approve = manifest_commands.add_parser("approve")
    approve.add_argument("--role", required=True, choices=ROLES)
    approve.add_argument("--dry-run", action="store_true")

    generate = commands.add_parser("generate", help="產生或更新決策資料集 CSV。")
    generate.add_argument("--version", required=True)
    generate.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "DecisionLayerTrain")
    generate.add_argument("--model-manifest", type=Path, default=REPO_ROOT / "models" / "manifest.yaml")
    generate.add_argument("--output-root", type=Path, default=REPO_ROOT / "decision_datasets")
    generate.add_argument("--mode", choices=[mode.value for mode in GenerationMode], default=GenerationMode.RESUME.value)
    generate.add_argument("--fruit-id", type=int)
    generate.add_argument("--view-id", type=int, choices=(1, 2, 3))
    generate.add_argument("--labels-from")
    generate.add_argument("--validate-only", action="store_true")
    return parser


def request_from_args(args: argparse.Namespace):
    return build_generation_request(
        repo_root=REPO_ROOT,
        dataset_root=args.dataset_root,
        model_manifest_path=args.model_manifest,
        output_root=args.output_root,
        dataset_version=args.version,
        mode=args.mode,
        fruit_id=args.fruit_id,
        view_id=args.view_id,
        labels_from=args.labels_from,
        validate_only=args.validate_only,
    )


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "layout":
            plan = plan_layout(args.dataset_root)
            _print_json({"moves": list(plan.moves), "errors": list(plan.errors)})
            if plan.errors:
                return 1
            if args.apply:
                print(f"對照表：{apply_layout(plan)}")
            return 0

        if args.command == "manifest":
            if args.manifest_command == "show":
                _print_json(load_or_create_manifest(args.manifest))
                return 0
            if args.manifest_command == "validate":
                _, errors = validate_manifest(args.manifest, REPO_ROOT)
                _print_json({"ok": not errors, "errors": errors})
                return 0 if not errors else 1
            if args.manifest_command == "update":
                record = update_manifest(
                    args.manifest,
                    REPO_ROOT,
                    args.role,
                    args.model,
                    args.imgsz,
                    args.confidence_threshold,
                    dry_run=args.dry_run,
                )
            else:
                record = approve_model(
                    args.manifest,
                    REPO_ROOT,
                    args.role,
                    dry_run=args.dry_run,
                )
            _print_json(record)
            return 0

        request = request_from_args(args)
        result = validate_generation(request) if args.validate_only else generate_dataset(request).to_dict()
        _print_json(result)
        return 0 if result["ok"] else 1
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
