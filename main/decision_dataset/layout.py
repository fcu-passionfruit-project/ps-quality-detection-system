from __future__ import annotations

import shutil
from pathlib import Path

from ._shared import (
    FRUIT_PATTERN,
    LEGACY_GRADES,
    SPLITS,
    VIEW_FILES,
    write_dict_rows,
)
from .types import FruitInput, LayoutPlan


def validate_dataset_layout(
    dataset_root: str | Path,
) -> tuple[list[FruitInput], list[dict]]:
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


def plan_layout(dataset_root: str | Path) -> LayoutPlan:
    root = Path(dataset_root).resolve()
    moves: list[dict] = []
    errors: list[dict] = []
    targets: set[Path] = set()
    seen_fruit_ids: dict[int, str] = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            errors.append(
                {"split": split, "source": "", "target": "", "reason": "缺少 split"}
            )
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
                match = FRUIT_PATTERN.fullmatch(source.name) if source.is_dir() else None
                if match is None:
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
                fruit_id = int(match.group(1))
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
    return LayoutPlan(root, tuple(moves), tuple(errors))


def apply_layout(plan: LayoutPlan) -> Path:
    if plan.errors:
        raise ValueError("Layout Plan 含有錯誤，不可套用。")
    root = plan.dataset_root.resolve()
    mapping_path = root / "layout_migration.csv"
    if mapping_path.exists():
        raise FileExistsError(f"移動對照表已存在：{mapping_path}")
    for move in plan.moves:
        source = Path(move["source"])
        target = Path(move["target"])
        if not source.is_dir():
            raise FileNotFoundError(f"搬移來源不存在：{source}")
        if target.exists():
            raise FileExistsError(f"搬移目標已存在：{target}")
    write_dict_rows(
        mapping_path,
        ["split", "source", "target", "reason"],
        list(plan.moves),
    )
    for move in plan.moves:
        shutil.move(move["source"], move["target"])
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
