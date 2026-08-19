from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import yaml

from pipeline import DEFECT_CLASSES

from .types import FruitInput


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_replace(path: Path, write) -> None:
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
    atomic_replace(
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
    atomic_replace(
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


def write_dict_rows(path: str | Path, fieldnames: list[str], rows: list[dict]) -> None:
    target = Path(path)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    atomic_replace(target, write)
