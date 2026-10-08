from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_DATASET = Path("training/packages/local_defects")
DEFAULT_OUTPUT = Path("experiments/train_B_class_aware.txt")
DEFAULT_FACTOR = 1
CLASS_NAMES = {
    0: "abrasion",
    1: "anthracnose",
    2: "insect_bite",
    3: "insect_track",
}
CLASS_REPEAT_FACTORS = {
    0: 1,
    1: 2,
    2: 4,
    3: 1,
}
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class TrainImage:
    image_path: Path
    label_path: Path
    instance_counts: Counter[int]
    repeat_factor: int


def _parse_segmentation_label(label_path: Path) -> Counter[int]:
    """Parse one YOLO segmentation label and return instance counts by class."""
    instance_counts: Counter[int] = Counter()

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"無法讀取 label：{label_path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2 != 0:
            raise ValueError(
                f"YOLO segmentation label 格式錯誤：{label_path} 第 {line_number} 行；"
                "預期為 class_id 加上至少 3 組 polygon 座標。"
            )

        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"class ID 不是整數：{label_path} 第 {line_number} 行：{fields[0]!r}"
            ) from exc

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"未知 class ID {class_id}：{label_path} 第 {line_number} 行；"
                f"只允許 {sorted(CLASS_NAMES)}。"
            )

        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(
                f"polygon 座標不是數字：{label_path} 第 {line_number} 行。"
            ) from exc

        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"polygon 座標必須介於 0 到 1：{label_path} 第 {line_number} 行。"
            )

        instance_counts[class_id] += 1

    return instance_counts


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise ValueError(f"找不到 images 目錄：{images_dir}")

    images = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"images 目錄內沒有支援的圖片：{images_dir}")

    stems = Counter(path.stem for path in images)
    duplicate_stems = sorted(stem for stem, count in stems.items() if count > 1)
    if duplicate_stems:
        raise ValueError(
            "train images 含有重複 stem，無法唯一對應 label："
            + ", ".join(duplicate_stems)
        )
    return images


def _validate_repeat_factors(repeat_factors: Mapping[int, int]) -> dict[int, int]:
    if set(repeat_factors) != set(CLASS_NAMES):
        raise ValueError(
            f"sampling factor 必須完整定義 class ID {sorted(CLASS_NAMES)}。"
        )

    validated: dict[int, int] = {}
    for class_id, factor in repeat_factors.items():
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
            raise ValueError(
                f"{CLASS_NAMES[class_id]} 的 sampling factor 必須是大於等於 1 的整數。"
            )
        validated[class_id] = factor
    return validated


def _resolve_output(dataset: Path, output: Path | None) -> Path:
    if output is None:
        return (dataset / DEFAULT_OUTPUT).resolve()
    if not output.is_absolute():
        output = Path.cwd() / output
    return output.resolve()


def _to_ultralytics_list_entry(image_path: Path, output_parent: Path) -> str:
    relative_path = Path(os.path.relpath(image_path, output_parent)).as_posix()
    return f"./{relative_path}"


def _resolve_list_entry(entry: str, output_parent: Path) -> Path:
    if not entry.startswith("./"):
        raise ValueError(f"image list 項目必須以 './' 開頭：{entry!r}")
    return (output_parent / entry[2:]).resolve()


def _collect_split_images(dataset: Path, split: str) -> set[Path]:
    split_dir = dataset / split / "images"
    if not split_dir.is_dir():
        raise ValueError(f"找不到 {split} images 目錄：{split_dir}")
    return {
        path.resolve()
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _validate_output_location(dataset: Path, output: Path) -> None:
    if output.suffix.lower() != ".txt":
        raise ValueError(f"輸出檔必須使用 .txt 副檔名：{output}")
    for split in ("train", "valid", "test"):
        split_root = (dataset / split).resolve()
        if output == split_root or output.is_relative_to(split_root):
            raise ValueError(
                f"輸出清單不可寫入原始 {split} split：{output}"
            )


def _build_train_records(
    dataset: Path,
    repeat_factors: Mapping[int, int],
) -> list[TrainImage]:
    images_dir = dataset / "train" / "images"
    labels_dir = dataset / "train" / "labels"
    if not labels_dir.is_dir():
        raise ValueError(f"找不到 train labels 目錄：{labels_dir}")

    images = _list_images(images_dir)
    expected_labels = {f"{image_path.stem}.txt" for image_path in images}
    actual_labels = {
        path.name for path in labels_dir.iterdir() if path.is_file() and path.suffix == ".txt"
    }
    missing_labels = sorted(expected_labels - actual_labels)
    orphan_labels = sorted(actual_labels - expected_labels)
    if missing_labels:
        raise ValueError("train image 缺少對應 label：" + ", ".join(missing_labels))
    if orphan_labels:
        raise ValueError("train labels 含有無對應 image 的檔案：" + ", ".join(orphan_labels))

    records: list[TrainImage] = []
    for image_path in images:
        label_path = (labels_dir / f"{image_path.stem}.txt").resolve()
        instance_counts = _parse_segmentation_label(label_path)
        factor = max(
            (repeat_factors[class_id] for class_id in instance_counts),
            default=DEFAULT_FACTOR,
        )
        records.append(
            TrainImage(
                image_path=image_path,
                label_path=label_path,
                instance_counts=instance_counts,
                repeat_factor=factor,
            )
        )
    return records


def _validate_entries(
    entries: list[str],
    records: list[TrainImage],
    output_parent: Path,
    valid_images: set[Path],
    test_images: set[Path],
) -> None:
    resolved_entries = [_resolve_list_entry(entry, output_parent) for entry in entries]
    missing_paths = sorted({path for path in resolved_entries if not path.is_file()})
    if missing_paths:
        raise ValueError(
            "輸出清單包含不存在的圖片：" + ", ".join(str(path) for path in missing_paths)
        )

    expected_counts = Counter(
        {record.image_path: record.repeat_factor for record in records}
    )
    actual_counts = Counter(resolved_entries)
    missing_train_images = sorted(set(expected_counts) - set(actual_counts))
    unexpected_images = sorted(set(actual_counts) - set(expected_counts))
    if missing_train_images:
        raise ValueError(
            "輸出清單遺漏 train image："
            + ", ".join(str(path) for path in missing_train_images)
        )
    if unexpected_images:
        raise ValueError(
            "輸出清單包含非 train image："
            + ", ".join(str(path) for path in unexpected_images)
        )
    if actual_counts != expected_counts:
        mismatches = [
            f"{path.name}: expected {expected_counts[path]}, got {actual_counts[path]}"
            for path in sorted(expected_counts)
            if expected_counts[path] != actual_counts[path]
        ]
        raise ValueError("輸出清單 repeat factor 不正確：" + "; ".join(mismatches))

    evaluation_images = valid_images | test_images
    leaked_images = sorted(set(actual_counts) & evaluation_images)
    if leaked_images:
        raise ValueError(
            "輸出清單混入 valid/test image："
            + ", ".join(str(path) for path in leaked_images)
        )


def _atomic_write_lines(output: Path, entries: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write("\n".join(entries) + "\n")
        temporary_path.chmod(0o644)
        temporary_path.replace(output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def generate_class_aware_train_list(
    dataset: Path,
    output: Path | None = None,
    repeat_factors: Mapping[int, int] | None = None,
) -> dict[str, object]:
    """Validate a YOLO segmentation dataset and write a repeated train image list."""
    dataset = dataset.resolve()
    if not dataset.is_dir():
        raise ValueError(f"找不到 Dataset 目錄：{dataset}")

    factors = _validate_repeat_factors(
        CLASS_REPEAT_FACTORS if repeat_factors is None else repeat_factors
    )
    output = _resolve_output(dataset, output)
    _validate_output_location(dataset, output)

    records = _build_train_records(dataset, factors)
    valid_images = _collect_split_images(dataset, "valid")
    test_images = _collect_split_images(dataset, "test")

    entries: list[str] = []
    for record in records:
        entry = _to_ultralytics_list_entry(record.image_path, output.parent)
        entries.extend([entry] * record.repeat_factor)

    _validate_entries(
        entries,
        records,
        output.parent,
        valid_images,
        test_images,
    )
    _atomic_write_lines(output, entries)

    written_entries = output.read_text(encoding="utf-8").splitlines()
    if written_entries != entries:
        raise RuntimeError(f"寫入後驗證失敗，輸出內容與預期不符：{output}")
    _validate_entries(
        written_entries,
        records,
        output.parent,
        valid_images,
        test_images,
    )

    images_by_repeat_factor = Counter(record.repeat_factor for record in records)
    images_containing: Counter[int] = Counter()
    original_exposure: Counter[int] = Counter()
    sampled_exposure: Counter[int] = Counter()
    null_images = 0
    for record in records:
        if not record.instance_counts:
            null_images += 1
        for class_id, instance_count in record.instance_counts.items():
            images_containing[class_id] += 1
            original_exposure[class_id] += instance_count
            sampled_exposure[class_id] += instance_count * record.repeat_factor

    return {
        "dataset": dataset,
        "output": output,
        "original_train_images": len(records),
        "generated_train_entries": len(entries),
        "images_by_repeat_factor": dict(images_by_repeat_factor),
        "images_containing": dict(images_containing),
        "null_images": null_images,
        "original_exposure": dict(original_exposure),
        "sampled_exposure": dict(sampled_exposure),
    }


def print_report(report: Mapping[str, object]) -> None:
    images_by_factor = report["images_by_repeat_factor"]
    images_containing = report["images_containing"]
    original_exposure = report["original_exposure"]
    sampled_exposure = report["sampled_exposure"]
    if not all(
        isinstance(value, dict)
        for value in (
            images_by_factor,
            images_containing,
            original_exposure,
            sampled_exposure,
        )
    ):
        raise TypeError("統計報告格式錯誤。")

    print(f"Dataset: {report['dataset']}")
    print(f"Output: {report['output']}")
    print(f"Original train images: {report['original_train_images']}")
    print(f"Generated train entries: {report['generated_train_entries']}")

    print("\nImages by repeat factor:")
    for factor in sorted(images_by_factor):
        print(f"{factor}x: {images_by_factor[factor]}")

    print("\nImages containing:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name}: {images_containing.get(class_id, 0)}")
    print(f"null: {report['null_images']}")

    print("\nOriginal annotation instances by class:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name}: {original_exposure.get(class_id, 0)}")

    print("\nTotal exposure by class after sampling:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name}: {sampled_exposure.get(class_id, 0)}")

    print("\nIntegrity checks: passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "為 YOLO segmentation train split 產生 class-aware oversampling image list；"
            "不修改或複製原始圖片與 labels。"
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root（預設：{DEFAULT_DATASET}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "輸出 .txt；相對路徑以目前工作目錄解析。"
            f"預設寫入 Dataset 下的 {DEFAULT_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--anthracnose-factor",
        type=int,
        default=CLASS_REPEAT_FACTORS[1],
        help="anthracnose repeat factor（預設：2）",
    )
    parser.add_argument(
        "--insect-bite-factor",
        type=int,
        default=CLASS_REPEAT_FACTORS[2],
        help="insect_bite repeat factor（預設：4）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repeat_factors = dict(CLASS_REPEAT_FACTORS)
    repeat_factors[1] = args.anthracnose_factor
    repeat_factors[2] = args.insect_bite_factor

    try:
        report = generate_class_aware_train_list(
            dataset=args.dataset,
            output=args.output,
            repeat_factors=repeat_factors,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
