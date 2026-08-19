from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TOOL_VERSION = 1


@dataclass(frozen=True)
class Recipe:
    name: str
    operation: str
    brightness: float


@dataclass(frozen=True)
class LabelObject:
    class_id: int
    points: tuple[tuple[float, float], ...]


RECIPES = (
    Recipe("hflip_brightness_090", "hflip", 0.90),
    Recipe("rotate90_brightness_110", "rotate90", 1.10),
    Recipe("vflip_brightness_095", "vflip", 0.95),
    Recipe("rotate180_brightness_105", "rotate180", 1.05),
    Recipe("rotate270_brightness_100", "rotate270", 1.00),
)


def _class_names(data: dict) -> dict[int, str]:
    names = data.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("data.yaml 的 names 必須是 list 或 class-index mapping。")


def _load_data_yaml(dataset_root: Path) -> tuple[dict, dict[int, str]]:
    data_path = dataset_root / "data.yaml"
    if not data_path.is_file():
        raise ValueError(f"找不到 Dataset 設定檔：{data_path}")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("data.yaml 必須是 YAML mapping。")
    return data, _class_names(data)


def parse_label(path: Path, class_names: dict[int, str]) -> tuple[LabelObject, ...]:
    objects: list[LabelObject] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise ValueError(f"{path}:{line_number} 不是有效的 YOLO segmentation polygon。")
        try:
            class_value = float(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} 含有無法解析的數字。") from error
        class_id = int(class_value)
        if class_value != class_id or class_id not in class_names:
            raise ValueError(f"{path}:{line_number} 的 class id {tokens[0]} 不在 data.yaml names。")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number} 的 polygon 座標必須介於 0 到 1。")
        points = tuple(zip(coordinates[0::2], coordinates[1::2], strict=True))
        objects.append(LabelObject(class_id, points))
    return tuple(objects)


def transform_point(x: float, y: float, operation: str) -> tuple[float, float]:
    if operation == "hflip":
        transformed = (1.0 - x, y)
    elif operation == "vflip":
        transformed = (x, 1.0 - y)
    elif operation == "rotate90":
        transformed = (1.0 - y, x)
    elif operation == "rotate180":
        transformed = (1.0 - x, 1.0 - y)
    elif operation == "rotate270":
        transformed = (y, 1.0 - x)
    else:
        raise ValueError(f"不支援的幾何轉換：{operation}")
    return tuple(min(1.0, max(0.0, value)) for value in transformed)


def transform_objects(objects: tuple[LabelObject, ...], operation: str) -> tuple[LabelObject, ...]:
    return tuple(
        LabelObject(obj.class_id, tuple(transform_point(x, y, operation) for x, y in obj.points))
        for obj in objects
    )


def _write_label(path: Path, objects: tuple[LabelObject, ...]) -> None:
    lines = []
    for obj in objects:
        values = [str(obj.class_id)]
        for x, y in obj.points:
            values.extend((f"{x:.8f}", f"{y:.8f}"))
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _image_pairs(dataset_root: Path, split: str, class_names: dict[int, str]) -> list[tuple[Path, Path, tuple[LabelObject, ...]]]:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"Dataset 缺少 {split}/images 或 {split}/labels。")
    images = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    by_stem: dict[str, Path] = {}
    for image_path in images:
        if image_path.stem in by_stem:
            raise ValueError(f"{image_dir} 有重複 stem：{image_path.stem}")
        by_stem[image_path.stem] = image_path
    extra_labels = [path for path in label_dir.glob("*.txt") if path.stem not in by_stem]
    if extra_labels:
        raise ValueError(f"標註沒有對應影像：{extra_labels[0]}")
    pairs = []
    for stem in sorted(by_stem):
        label_path = label_dir / f"{stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"影像沒有對應標註：{by_stem[stem]}")
        pairs.append((by_stem[stem], label_path, parse_label(label_path, class_names)))
    return pairs


def _statistics(pairs: list[tuple[Path, Path, tuple[LabelObject, ...]]], class_names: dict[int, str]) -> dict:
    image_counts = Counter()
    instance_counts = Counter()
    empty_images = 0
    for _, _, objects in pairs:
        if not objects:
            empty_images += 1
        present = {obj.class_id for obj in objects}
        image_counts.update(present)
        instance_counts.update(obj.class_id for obj in objects)
    return {
        "images": len(pairs),
        "empty_images": empty_images,
        "class_image_counts": {class_names[index]: image_counts[index] for index in sorted(class_names)},
        "class_instance_counts": {class_names[index]: instance_counts[index] for index in sorted(class_names)},
    }


def _recipes_for(image_path: Path, dataset_root: Path, copies: int, seed: int) -> tuple[Recipe, ...]:
    relative = image_path.relative_to(dataset_root).as_posix()
    digest = hashlib.sha256(f"{seed}:{relative}".encode()).digest()
    start = int.from_bytes(digest[:4], "big") % len(RECIPES)
    return tuple(RECIPES[(start + index) % len(RECIPES)] for index in range(copies))


def _transform_image(source: Path, destination: Path, recipe: Recipe) -> None:
    transpose = {
        "hflip": Image.Transpose.FLIP_LEFT_RIGHT,
        "vflip": Image.Transpose.FLIP_TOP_BOTTOM,
        "rotate90": Image.Transpose.ROTATE_270,
        "rotate180": Image.Transpose.ROTATE_180,
        "rotate270": Image.Transpose.ROTATE_90,
    }[recipe.operation]
    with Image.open(source) as opened:
        image = opened.convert("RGB").transpose(transpose)
        image = ImageEnhance.Brightness(image).enhance(recipe.brightness)
        save_options = {"quality": 95, "subsampling": 0} if destination.suffix.lower() in {".jpg", ".jpeg"} else {}
        image.save(destination, **save_options)


def _write_output_data_yaml(output_root: Path, source_data: dict) -> None:
    output_data = dict(source_data)
    output_data["path"] = output_root.resolve().as_posix()
    output_data["train"] = "train/images"
    output_data["val"] = "valid/images"
    output_data["test"] = "test/images"
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(output_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def augment_dataset(
    input_root: Path,
    output_root: Path,
    *,
    anthracnose_copies: int = 2,
    insect_bite_copies: int = 5,
    seed: int = 0,
    dry_run: bool = False,
) -> dict:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if input_root == output_root:
        raise ValueError("輸入與輸出 Dataset 不可使用相同路徑。")
    if input_root in output_root.parents:
        raise ValueError("輸出 Dataset 不可建立在輸入 Dataset 內部。")
    if not input_root.is_dir():
        raise ValueError(f"找不到輸入 Dataset：{input_root}")
    if output_root.exists():
        raise ValueError(f"輸出路徑已存在，為避免覆寫而停止：{output_root}")
    if anthracnose_copies < 0 or insect_bite_copies < 0:
        raise ValueError("擴增份數不可小於 0。")

    source_data, class_names = _load_data_yaml(input_root)
    name_to_id = {name: index for index, name in class_names.items()}
    missing = {"anthracnose", "insect_bite"} - name_to_id.keys()
    if missing:
        raise ValueError(f"data.yaml 缺少必要類別：{', '.join(sorted(missing))}")

    split_pairs = {split: _image_pairs(input_root, split, class_names) for split in ("train", "valid", "test")}
    anthracnose_id = name_to_id["anthracnose"]
    insect_bite_id = name_to_id["insect_bite"]
    generated = []
    planned_names = {image_path.stem for image_path, _, _ in split_pairs["train"]}
    for image_path, _, objects in split_pairs["train"]:
        present = {obj.class_id for obj in objects}
        copies = insect_bite_copies if insect_bite_id in present else anthracnose_copies if anthracnose_id in present else 0
        target_classes = [class_names[index] for index in sorted(present & {anthracnose_id, insect_bite_id})]
        for index, recipe in enumerate(_recipes_for(image_path, input_root, copies, seed), 1):
            generated_stem = f"{image_path.stem}__rare_aug_{index:02d}"
            if generated_stem in planned_names:
                raise ValueError(f"擴增檔名衝突：{generated_stem}")
            planned_names.add(generated_stem)
            generated.append(
                {
                    "source_image": image_path.relative_to(input_root).as_posix(),
                    "generated_image": f"train/images/{generated_stem}{image_path.suffix.lower()}",
                    "generated_label": f"train/labels/{generated_stem}.txt",
                    "target_classes": target_classes,
                    "recipe": asdict(recipe),
                    "objects": objects,
                }
            )

    report = {
        "tool_version": TOOL_VERSION,
        "input": input_root.as_posix(),
        "output": output_root.as_posix(),
        "dry_run": dry_run,
        "seed": seed,
        "copies": {"anthracnose": anthracnose_copies, "insect_bite": insect_bite_copies},
        "before": {split: _statistics(pairs, class_names) for split, pairs in split_pairs.items()},
        "planned_generated_images": len(generated),
    }
    if dry_run:
        return report

    shutil.copytree(input_root, output_root, ignore=shutil.ignore_patterns("*.cache", "desktop.ini"))
    _write_output_data_yaml(output_root, source_data)
    manifest_entries = []
    for item in generated:
        source_image = input_root / item["source_image"]
        destination_image = output_root / item["generated_image"]
        destination_label = output_root / item["generated_label"]
        recipe = Recipe(**item["recipe"])
        _transform_image(source_image, destination_image, recipe)
        _write_label(destination_label, transform_objects(item["objects"], recipe.operation))
        manifest_entries.append({key: value for key, value in item.items() if key != "objects"})

    output_pairs = {split: _image_pairs(output_root, split, class_names) for split in ("train", "valid", "test")}
    report["after"] = {split: _statistics(pairs, class_names) for split, pairs in output_pairs.items()}
    report["generated"] = manifest_entries
    (output_root / "augmentation_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只對 YOLO segmentation train split 的少數類別來源影像做離線擴增。")
    parser.add_argument("--input", required=True, type=Path, help="Roboflow 原始解析度匯出的 Dataset 根目錄。")
    parser.add_argument("--output", required=True, type=Path, help="新 Dataset 輸出目錄；必須尚未存在。")
    parser.add_argument("--anthracnose-copies", type=int, default=2)
    parser.add_argument("--insect-bite-copies", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = augment_dataset(
            args.input,
            args.output,
            anthracnose_copies=args.anthracnose_copies,
            insect_bite_copies=args.insect_bite_copies,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
