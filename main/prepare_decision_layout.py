from __future__ import annotations

import argparse
from pathlib import Path

from decision_dataset import apply_layout_migration, plan_layout_migration


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="移除舊級距資料夾層，保留 train/valid/test 與 fruit_id。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "DecisionLayerTrain",
        help="資料集根目錄。",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="只預覽，不搬移。")
    mode.add_argument("--apply", action="store_true", help="套用搬移。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    moves, errors = plan_layout_migration(args.dataset_root)
    print(f"預計搬移 {len(moves)} 個果實資料夾。")
    for item in errors:
        print(f"ERROR [{item['split']}] {item['source']}: {item['reason']}")
    if errors:
        print("結構驗證失敗，未修改任何檔案。")
        return 1
    if args.dry_run:
        for move in moves:
            print(f"MOVE {move['source']} -> {move['target']}")
        print("預覽完成，未修改任何檔案。")
        return 0
    mapping_path = apply_layout_migration(args.dataset_root, moves)
    print(f"搬移完成：{len(moves)} 個果實資料夾。")
    print(f"對照表：{mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
