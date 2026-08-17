from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_dataset import generate_decision_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用四個 YOLO 模型批次產生決策層主 CSV。"
    )
    parser.add_argument("--version", required=True, help="資料版本，例如 v001。")
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / "DecisionLayerTrain"
    )
    parser.add_argument(
        "--model-manifest", type=Path, default=REPO_ROOT / "models" / "manifest.yaml"
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "decision_datasets"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fruit-id", type=int)
    parser.add_argument("--view-id", type=int, choices=(1, 2, 3))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-errors", action="store_true")
    parser.add_argument("--replace-image", action="store_true")
    parser.add_argument("--labels-from", help="複製照片 hash 相同的舊版本 grade。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_decision_dataset(
            repo_root=REPO_ROOT,
            dataset_root=args.dataset_root,
            model_manifest_path=args.model_manifest,
            output_root=args.output_root,
            dataset_version=args.version,
            validate_only=args.validate_only,
            fruit_id=args.fruit_id,
            view_id=args.view_id,
            force=args.force,
            only_errors=args.only_errors,
            replace_image=args.replace_image,
            labels_from=args.labels_from,
        )
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
