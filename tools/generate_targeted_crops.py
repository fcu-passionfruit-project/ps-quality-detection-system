from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml
from PIL import Image


DEFAULT_DATASET = Path("training/packages/local_defects")
DEFAULT_OUTPUT_DIR = Path("experiments/exp_C_targeted_crop")
DEFAULT_TRAIN_LIST = Path("experiments/train_C_targeted_crop.txt")
DEFAULT_DATA_YAML = Path("data_exp_C.yaml")
DEFAULT_WEIGHT = Path("yolo26s-seg.pt")
CLASS_NAMES = {
    0: "abrasion",
    1: "anthracnose",
    2: "insect_bite",
    3: "insect_track",
}
TARGET_CROPS_PER_INSTANCE = {1: 1, 2: 3}
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SOURCE_MANIFEST_FIELDS = ("split", "kind", "path", "size_bytes", "sha256")
CROP_MANIFEST_FIELDS = (
    "output_image",
    "output_label",
    "source_image",
    "source_label",
    "target_class_id",
    "target_class_name",
    "target_instance_index",
    "target_label_index",
    "crop_index",
    "original_width",
    "original_height",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_width",
    "crop_height",
    "context_scale",
    "effective_context_scale",
    "jitter",
    "jitter_x",
    "jitter_y",
    "seed",
    "fruit_id",
    "view_id",
)
FRUIT_VIEW_PATTERN = re.compile(
    r"^fruit_(\d+)_img_(0?[123])(?:_jpg)?(?:\.rf\.[A-Za-z0-9]+)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolygonInstance:
    class_id: int
    points: tuple[tuple[float, float], ...]
    label_index: int
    class_instance_index: int


@dataclass(frozen=True)
class SourceImage:
    image_path: Path
    label_path: Path
    width: int
    height: int
    instances: tuple[PolygonInstance, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_records(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(
            dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def _deduplicate_polygon(
    points: Sequence[tuple[float, float]], tolerance: float = 1e-9
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for point in points:
        if not result or not (
            math.isclose(point[0], result[-1][0], abs_tol=tolerance)
            and math.isclose(point[1], result[-1][1], abs_tol=tolerance)
        ):
            result.append(point)
    if len(result) > 1 and math.isclose(
        result[0][0], result[-1][0], abs_tol=tolerance
    ) and math.isclose(result[0][1], result[-1][1], abs_tol=tolerance):
        result.pop()
    return result


def _clip_edge(
    points: Sequence[tuple[float, float]],
    inside,
    intersect,
) -> list[tuple[float, float]]:
    if not points:
        return []
    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def clip_polygon_to_rectangle(
    points: Sequence[tuple[float, float]],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> list[tuple[float, float]]:
    """Clip a polygon with Sutherland-Hodgman against an axis-aligned rectangle."""

    def vertical_intersection(
        start: tuple[float, float], end: tuple[float, float], boundary: float
    ) -> tuple[float, float]:
        delta = end[0] - start[0]
        if math.isclose(delta, 0.0):
            return boundary, start[1]
        ratio = (boundary - start[0]) / delta
        return boundary, start[1] + ratio * (end[1] - start[1])

    def horizontal_intersection(
        start: tuple[float, float], end: tuple[float, float], boundary: float
    ) -> tuple[float, float]:
        delta = end[1] - start[1]
        if math.isclose(delta, 0.0):
            return start[0], boundary
        ratio = (boundary - start[1]) / delta
        return start[0] + ratio * (end[0] - start[0]), boundary

    clipped = list(points)
    clipped = _clip_edge(
        clipped,
        lambda point: point[0] >= x1,
        lambda start, end: vertical_intersection(start, end, x1),
    )
    clipped = _clip_edge(
        clipped,
        lambda point: point[0] <= x2,
        lambda start, end: vertical_intersection(start, end, x2),
    )
    clipped = _clip_edge(
        clipped,
        lambda point: point[1] >= y1,
        lambda start, end: horizontal_intersection(start, end, y1),
    )
    clipped = _clip_edge(
        clipped,
        lambda point: point[1] <= y2,
        lambda start, end: horizontal_intersection(start, end, y2),
    )
    return _deduplicate_polygon(clipped)


def _read_label(
    label_path: Path, width: int, height: int, min_polygon_area: float
) -> tuple[PolygonInstance, ...]:
    class_indices: Counter[int] = Counter()
    instances: list[PolygonInstance] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"無法讀取 label：{label_path}: {exc}") from exc

    for label_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(
                f"YOLO segmentation label 格式錯誤：{label_path} 第 {label_index + 1} 行"
            )
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"class ID 不是整數：{label_path} 第 {label_index + 1} 行"
            ) from exc
        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"未知 class ID {class_id}：{label_path} 第 {label_index + 1} 行"
            )
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(
                f"polygon 座標不是數字：{label_path} 第 {label_index + 1} 行"
            ) from exc
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"polygon 座標必須介於 0 到 1：{label_path} 第 {label_index + 1} 行"
            )
        points = _deduplicate_polygon(
            [
                (coordinates[index] * width, coordinates[index + 1] * height)
                for index in range(0, len(coordinates), 2)
            ]
        )
        if len(points) < 3 or polygon_area(points) < min_polygon_area:
            raise ValueError(
                f"polygon 無效或面積近乎 0：{label_path} 第 {label_index + 1} 行"
            )
        instances.append(
            PolygonInstance(
                class_id=class_id,
                points=tuple(points),
                label_index=label_index,
                class_instance_index=class_indices[class_id],
            )
        )
        class_indices[class_id] += 1
    return tuple(instances)


def _list_split_sources(
    dataset: Path, split: str, min_polygon_area: float
) -> list[SourceImage]:
    images_dir = dataset / split / "images"
    labels_dir = dataset / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"找不到 {split} images/labels 目錄：{dataset}")
    images = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    stems = Counter(path.stem for path in images)
    duplicate_stems = sorted(stem for stem, count in stems.items() if count > 1)
    if duplicate_stems:
        raise ValueError(f"{split} images 有重複 stem：{', '.join(duplicate_stems)}")
    expected_labels = {f"{path.stem}.txt" for path in images}
    actual_labels = {
        path.name
        for path in labels_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }
    missing = sorted(expected_labels - actual_labels)
    orphan = sorted(actual_labels - expected_labels)
    if missing or orphan:
        raise ValueError(
            f"{split} image/label 不完整；missing={missing}；orphan={orphan}"
        )
    sources: list[SourceImage] = []
    for image_path in images:
        label_path = (labels_dir / f"{image_path.stem}.txt").resolve()
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except OSError as exc:
            raise ValueError(f"圖片無法解析：{image_path}: {exc}") from exc
        instances = _read_label(label_path, width, height, min_polygon_area)
        sources.append(SourceImage(image_path, label_path, width, height, instances))
    if not sources:
        raise ValueError(f"{split} split 沒有圖片：{images_dir}")
    return sources


def _relative(path: Path, dataset: Path) -> str:
    return path.resolve().relative_to(dataset.resolve()).as_posix()


def _collect_source_manifest(
    dataset: Path, split_sources: Mapping[str, Sequence[SourceImage]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split in ("train", "valid", "test"):
        for source in split_sources[split]:
            for kind, path in (("image", source.image_path), ("label", source.label_path)):
                records.append(
                    {
                        "split": split,
                        "kind": kind,
                        "path": _relative(path, dataset),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return records


def _parse_fruit_view(image_path: Path) -> tuple[str, str]:
    match = FRUIT_VIEW_PATTERN.fullmatch(image_path.stem)
    if not match:
        return "", ""
    return str(int(match.group(1))), str(int(match.group(2)))


def _axis_crop(
    extent: int,
    box_min: float,
    box_max: float,
    desired_length: int,
    maximum_crop_fraction: float,
    center_offset: float,
) -> tuple[int, int]:
    target_min = max(0, math.floor(box_min))
    target_max = min(extent, math.ceil(box_max))
    target_length = max(1, target_max - target_min)
    maximum_length = max(target_length, math.floor(extent * maximum_crop_fraction))
    length = min(extent, maximum_length, max(target_length, desired_length))
    ideal_start = (box_min + box_max) / 2.0 - length / 2.0 + center_offset
    minimum_start = max(0, math.ceil(box_max - length))
    maximum_start = min(extent - length, math.floor(box_min))
    if minimum_start > maximum_start:
        raise ValueError("無法建立完整包含 target polygon 的 crop。")
    start = min(max(round(ideal_start), minimum_start), maximum_start)
    return int(start), int(start + length)


def _choose_crop_rectangle(
    source: SourceImage,
    target: PolygonInstance,
    sampled_context_scale: float,
    minimum_crop_size: int,
    maximum_crop_fraction: float,
    jitter_fraction: float,
    rng: random.Random,
) -> tuple[int, int, int, int, float, float, float]:
    xs = [point[0] for point in target.points]
    ys = [point[1] for point in target.points]
    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    target_extent = max(box_width, box_height)
    desired_length = max(minimum_crop_size, math.ceil(target_extent * sampled_context_scale))
    maximum_jitter = desired_length * jitter_fraction
    jitter_x = rng.uniform(-maximum_jitter, maximum_jitter)
    jitter_y = rng.uniform(-maximum_jitter, maximum_jitter)
    crop_x1, crop_x2 = _axis_crop(
        source.width,
        min(xs),
        max(xs),
        desired_length,
        maximum_crop_fraction,
        jitter_x,
    )
    crop_y1, crop_y2 = _axis_crop(
        source.height,
        min(ys),
        max(ys),
        desired_length,
        maximum_crop_fraction,
        jitter_y,
    )
    effective_scale = max(crop_x2 - crop_x1, crop_y2 - crop_y1) / target_extent
    return (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
        effective_scale,
        jitter_x,
        jitter_y,
    )


def _format_label(
    instances: Sequence[PolygonInstance],
    rectangle: tuple[int, int, int, int],
    min_polygon_area: float,
) -> tuple[str, dict[int, tuple[tuple[float, float], ...]]]:
    x1, y1, x2, y2 = rectangle
    crop_width = x2 - x1
    crop_height = y2 - y1
    lines: list[str] = []
    retained: dict[int, tuple[tuple[float, float], ...]] = {}
    for instance in instances:
        clipped = clip_polygon_to_rectangle(instance.points, x1, y1, x2, y2)
        if len(clipped) < 3 or polygon_area(clipped) < min_polygon_area:
            continue
        normalized = tuple(
            (
                min(1.0, max(0.0, (point[0] - x1) / crop_width)),
                min(1.0, max(0.0, (point[1] - y1) / crop_height)),
            )
            for point in clipped
        )
        if len(normalized) < 3 or polygon_area(normalized) <= 1e-12:
            continue
        coordinates = " ".join(f"{value:.8f}" for point in normalized for value in point)
        lines.append(f"{instance.class_id} {coordinates}")
        retained[instance.label_index] = normalized
    return ("\n".join(lines) + ("\n" if lines else "")), retained


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_csv(path: Path, fields: Sequence[str], records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _validate_data_yaml(dataset: Path) -> dict[str, object]:
    data_path = dataset / "data.yaml"
    try:
        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"無法解析原始 data.yaml：{data_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"原始 data.yaml 根節點必須是 mapping：{data_path}")
    expected = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 4,
        "names": list(CLASS_NAMES.values()),
    }
    for key, expected_value in expected.items():
        if data.get(key) != expected_value:
            raise ValueError(
                f"原始 data.yaml 的 {key} 不符合 Experiment C contract："
                f"expected={expected_value!r}, got={data.get(key)!r}"
            )
    return data


def _build_experiment_yaml(dataset: Path, train_list: Path, data_yaml: Path) -> None:
    original = (dataset / "data.yaml").read_text(encoding="utf-8")
    replacement = _relative(train_list, dataset)
    lines = original.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("train:"):
            lines[index] = f"train: {replacement}"
            replaced = True
            break
    if not replaced:
        raise ValueError("原始 data.yaml 缺少 train 設定。")
    _atomic_write_text(data_yaml, "\n".join(lines) + "\n")


def _list_entry(image_path: Path, train_list: Path) -> str:
    relative = Path(os.path.relpath(image_path, train_list.parent)).as_posix()
    return f"./{relative}"


def _dataset_statistics(sources: Sequence[SourceImage]) -> dict[str, object]:
    image_counts: Counter[int] = Counter()
    instances: Counter[int] = Counter()
    null_images = 0
    for source in sources:
        present = {instance.class_id for instance in source.instances}
        if not present:
            null_images += 1
        image_counts.update(present)
        instances.update(instance.class_id for instance in source.instances)
    classes = {
        CLASS_NAMES[class_id]: {
            "images_containing_class": image_counts[class_id],
            "instances": instances[class_id],
            "effective_exposure": instances[class_id],
        }
        for class_id in CLASS_NAMES
    }
    classes["null"] = {
        "images_containing_class": null_images,
        "instances": 0,
        "effective_exposure": null_images,
    }
    return {"images": len(sources), "classes": classes}


def _validate_split_safety(
    split_sources: Mapping[str, Sequence[SourceImage]],
    source_manifest: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    image_hashes: dict[str, set[str]] = {}
    parsed_ids: dict[str, set[tuple[str, str]]] = {}
    unparsed: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        image_hashes[split] = {
            str(record["sha256"])
            for record in source_manifest
            if record["split"] == split and record["kind"] == "image"
        }
        identities: set[tuple[str, str]] = set()
        unparsed_count = 0
        for source in split_sources[split]:
            identity = _parse_fruit_view(source.image_path)
            if identity == ("", ""):
                unparsed_count += 1
            else:
                identities.add(identity)
        parsed_ids[split] = identities
        unparsed[split] = unparsed_count

    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    hash_overlap = {
        f"{left}_{right}": len(image_hashes[left] & image_hashes[right])
        for left, right in pairs
    }
    identity_overlap = {
        f"{left}_{right}": len(parsed_ids[left] & parsed_ids[right])
        for left, right in pairs
    }
    if any(hash_overlap.values()) or any(identity_overlap.values()):
        raise ValueError(
            f"train/valid/test overlap：hash={hash_overlap}, parsed_identity={identity_overlap}"
        )
    return {
        "exact_image_sha256_overlap": hash_overlap,
        "reliably_parsed_fruit_view_overlap": identity_overlap,
        "unparseable_fruit_view_images": unparsed,
    }


def _validate_generated_dataset(
    dataset: Path,
    output_dir: Path,
    train_list: Path,
    data_yaml: Path,
    split_sources: Mapping[str, Sequence[SourceImage]],
    source_manifest_before: Sequence[Mapping[str, object]],
    crop_records: Sequence[Mapping[str, object]],
    min_polygon_area: float,
) -> dict[str, object]:
    source_manifest_after = _collect_source_manifest(dataset, split_sources)
    if list(source_manifest_before) != source_manifest_after:
        raise RuntimeError("原始 train/valid/test 在 generation 過程中被修改。")

    images = sorted((output_dir / "images").glob("*.png"))
    labels = sorted((output_dir / "labels").glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems or len(images) != len(crop_records):
        raise ValueError(
            f"generated image/label pair 不完整：images={len(images)}, labels={len(labels)}, "
            f"manifest={len(crop_records)}"
        )

    source_by_label = {
        _relative(source.label_path, dataset): source
        for source in split_sources["train"]
    }
    malformed = 0
    out_of_range = 0
    missing_target = 0
    generated_sources_not_train = 0
    for record in crop_records:
        source_label_name = str(record["source_label"])
        source = source_by_label.get(source_label_name)
        if source is None:
            generated_sources_not_train += 1
            continue
        output_image = dataset / str(record["output_image"])
        output_label = dataset / str(record["output_label"])
        if not output_image.is_file() or not output_label.is_file():
            raise ValueError(f"manifest 指向不存在的 generated pair：{record}")
        with Image.open(output_image) as image:
            expected_size = (int(record["crop_width"]), int(record["crop_height"]))
            if image.size != expected_size:
                raise ValueError(
                    f"generated crop 尺寸與 manifest 不符：{output_image}"
                )
        try:
            output_instances = _read_label(
                output_label,
                int(record["crop_width"]),
                int(record["crop_height"]),
                min_polygon_area,
            )
        except ValueError:
            malformed += 1
            continue
        crop_width = int(record["crop_width"])
        crop_height = int(record["crop_height"])
        if any(
            point[0] < -1e-9
            or point[0] > crop_width + 1e-9
            or point[1] < -1e-9
            or point[1] > crop_height + 1e-9
            for instance in output_instances
            for point in instance.points
        ):
            out_of_range += 1
        target_label_index = int(record["target_label_index"])
        target = next(
            (item for item in source.instances if item.label_index == target_label_index),
            None,
        )
        if target is None:
            missing_target += 1
            continue
        rectangle = (
            int(record["crop_x1"]),
            int(record["crop_y1"]),
            int(record["crop_x2"]),
            int(record["crop_y2"]),
        )
        if not all(
            rectangle[0] - 1e-7 <= point[0] <= rectangle[2] + 1e-7
            and rectangle[1] - 1e-7 <= point[1] <= rectangle[3] + 1e-7
            for point in target.points
        ):
            missing_target += 1
            continue
        expected_area = polygon_area(target.points)
        matching_areas = [
            polygon_area(instance.points)
            for instance in output_instances
            if instance.class_id == target.class_id
        ]
        if not any(math.isclose(area, expected_area, rel_tol=1e-5, abs_tol=1e-3) for area in matching_areas):
            missing_target += 1

    if any((malformed, out_of_range, missing_target, generated_sources_not_train)):
        raise ValueError(
            "generated Dataset 驗證失敗："
            f"malformed={malformed}, out_of_range={out_of_range}, "
            f"missing_target={missing_target}, source_not_train={generated_sources_not_train}"
        )

    entries = train_list.read_text(encoding="utf-8").splitlines()
    resolved_entries = [
        (train_list.parent / entry[2:]).resolve()
        if entry.startswith("./")
        else Path("/__invalid_train_list_entry__")
        for entry in entries
    ]
    original_train = {source.image_path for source in split_sources["train"]}
    generated = {path.resolve() for path in images}
    if len(entries) != len(original_train) + len(generated):
        raise ValueError("Experiment C train list entry 數量不正確。")
    if Counter(resolved_entries) != Counter({path: 1 for path in original_train | generated}):
        raise ValueError("Experiment C train list 必須包含每張原圖一次及每張 crop 一次。")

    experiment_yaml = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    baseline_yaml = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if experiment_yaml.get("train") != _relative(train_list, dataset):
        raise ValueError("data_exp_C.yaml train 未指向 Experiment C train list。")
    for key in ("val", "test", "nc", "names", "roboflow"):
        if experiment_yaml.get(key) != baseline_yaml.get(key):
            raise ValueError(f"data_exp_C.yaml 的 {key} 與 baseline 不同。")

    all_sources = list(split_sources["train"])
    generated_sources: list[SourceImage] = []
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = output_dir / "labels" / f"{image_path.stem}.txt"
        generated_sources.append(
            SourceImage(
                image_path,
                label_path,
                width,
                height,
                _read_label(label_path, width, height, min_polygon_area),
            )
        )

    split_safety = _validate_split_safety(
        split_sources, source_manifest_before
    )
    return {
        "checks": {
            "source_splits_unchanged": True,
            "generated_sources_are_train_only": generated_sources_not_train == 0,
            "original_train_set_preserved": True,
            "original_full_images_repeated": 0,
            "generated_image_label_pairs_complete": True,
            "malformed_labels": malformed,
            "out_of_range_coordinates": out_of_range,
            "missing_image_or_label": 0,
            "target_class_instance_missing": missing_target,
            "data_yaml_matches_baseline_valid_test_and_classes": True,
            "split_safety": split_safety,
        },
        "statistics": {
            "baseline": {
                "train_list_entries": len(split_sources["train"]),
                "unique_original_images": len(split_sources["train"]),
                "generated_crop_images": 0,
                **_dataset_statistics(split_sources["train"]),
            },
            "experiment_c": {
                "train_list_entries": len(entries),
                "unique_original_images": len(split_sources["train"]),
                "generated_crop_images": len(generated_sources),
                **_dataset_statistics(all_sources + generated_sources),
            },
        },
    }


def _training_input_hash(
    source_manifest: Sequence[Mapping[str, object]],
    crop_records: Sequence[Mapping[str, object]],
    dataset: Path,
    train_list: Path,
    data_yaml: Path,
) -> str:
    records: list[dict[str, object]] = [
        {"scope": "source", **dict(record)} for record in source_manifest
    ]
    for crop in crop_records:
        for kind in ("output_image", "output_label"):
            relative_path = str(crop[kind])
            path = dataset / relative_path
            records.append(
                {
                    "scope": "generated",
                    "kind": kind,
                    "path": relative_path,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for kind, path in (("train_list", train_list), ("data_yaml", data_yaml)):
        records.append(
            {
                "scope": "configuration",
                "kind": kind,
                "path": _relative(path, dataset),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sha256_records(sorted(records, key=lambda item: json.dumps(item, sort_keys=True)))


def generate_targeted_crops(
    dataset: Path = DEFAULT_DATASET,
    output_dir: Path | None = None,
    train_list: Path | None = None,
    data_yaml: Path | None = None,
    weight: Path = DEFAULT_WEIGHT,
    crops_per_instance: Mapping[int, int] | None = None,
    context_scale: float = 4.0,
    context_scale_variation: float = 0.5,
    center_jitter: float = 0.10,
    minimum_crop_size: int = 64,
    maximum_crop_fraction: float = 0.90,
    min_polygon_area: float = 1.0,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    dataset = dataset.resolve()
    if not dataset.is_dir():
        raise ValueError(f"找不到 Dataset：{dataset}")
    output_dir = (
        (dataset / (output_dir or DEFAULT_OUTPUT_DIR)).resolve()
        if output_dir is None or not output_dir.is_absolute()
        else output_dir.resolve()
    )
    train_list = (
        (dataset / (train_list or DEFAULT_TRAIN_LIST)).resolve()
        if train_list is None or not train_list.is_absolute()
        else train_list.resolve()
    )
    data_yaml = (
        (dataset / (data_yaml or DEFAULT_DATA_YAML)).resolve()
        if data_yaml is None or not data_yaml.is_absolute()
        else data_yaml.resolve()
    )
    weight = weight.resolve()

    for path in (output_dir, train_list, data_yaml):
        if not path.is_relative_to(dataset):
            raise ValueError(f"Experiment C 輸出必須位於 Dataset root：{path}")
        for split in ("train", "valid", "test"):
            split_root = (dataset / split).resolve()
            if path == split_root or path.is_relative_to(split_root):
                raise ValueError(f"Experiment C 不可寫入原始 {split} split：{path}")
    if (
        output_dir == dataset
        or train_list.suffix.lower() != ".txt"
        or data_yaml.suffix.lower() not in {".yaml", ".yml"}
    ):
        raise ValueError("Experiment C 輸出路徑或副檔名無效。")
    if not weight.is_file():
        raise ValueError(f"找不到初始權重：{weight}")

    counts = dict(TARGET_CROPS_PER_INSTANCE if crops_per_instance is None else crops_per_instance)
    if set(counts) != {1, 2} or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValueError("crop 數量必須只定義 anthracnose/insect_bite，且為非負整數。")
    if context_scale <= 1.0:
        raise ValueError("context_scale 必須大於 1，才能保留 defect 周圍果皮。")
    if context_scale_variation < 0 or context_scale - context_scale_variation <= 1.0:
        raise ValueError("context_scale_variation 必須非負且最小 scale 必須大於 1。")
    if not 0.0 <= center_jitter <= 0.5:
        raise ValueError("center_jitter 必須介於 0 與 0.5。")
    if not 0.0 < maximum_crop_fraction < 1.0:
        raise ValueError("maximum_crop_fraction 必須大於 0 且小於 1。")
    if minimum_crop_size < 1 or min_polygon_area <= 0:
        raise ValueError("minimum_crop_size 與 min_polygon_area 必須為正數。")

    _validate_data_yaml(dataset)
    split_sources = {
        split: _list_split_sources(dataset, split, min_polygon_area)
        for split in ("train", "valid", "test")
    }
    source_manifest = _collect_source_manifest(dataset, split_sources)
    split_safety = _validate_split_safety(split_sources, source_manifest)

    existing = [path for path in (output_dir, train_list, data_yaml) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "Experiment C 輸出已存在；若要以相同參數安全重建，請加 --overwrite："
            + ", ".join(str(path) for path in existing)
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "images").mkdir()
    (output_dir / "labels").mkdir()

    rng = random.Random(seed)
    crop_records: list[dict[str, object]] = []
    try:
        for source in split_sources["train"]:
            fruit_id, view_id = _parse_fruit_view(source.image_path)
            with Image.open(source.image_path) as source_image:
                source_image.load()
                for target in source.instances:
                    crop_count = counts.get(target.class_id, 0)
                    used_rectangles: set[tuple[int, int, int, int]] = set()
                    for crop_index in range(crop_count):
                        rectangle: tuple[int, int, int, int] | None = None
                        selection: tuple[int, int, int, int, float, float, float] | None = None
                        sampled_scale = context_scale
                        for _attempt in range(100):
                            sampled_scale = rng.uniform(
                                context_scale - context_scale_variation,
                                context_scale + context_scale_variation,
                            )
                            selection = _choose_crop_rectangle(
                                source,
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                maximum_crop_fraction,
                                center_jitter,
                                rng,
                            )
                            rectangle = selection[:4]
                            if rectangle not in used_rectangles:
                                break
                        if (
                            rectangle is None
                            or selection is None
                            or rectangle in used_rectangles
                        ):
                            raise ValueError(
                                f"無法為 {source.image_path.name} 的 {CLASS_NAMES[target.class_id]} "
                                "產生互不相同的 deterministic crops；請調整 context/jitter。"
                            )
                        used_rectangles.add(rectangle)
                        x1, y1, x2, y2, effective_scale, jitter_x, jitter_y = selection
                        label_content, retained = _format_label(
                            source.instances, rectangle, min_polygon_area
                        )
                        if target.label_index not in retained:
                            raise RuntimeError(
                                f"target polygon 未保留：{source.label_path} label index {target.label_index}"
                            )
                        output_stem = (
                            f"{source.image_path.stem}__tc_c{target.class_id}_"
                            f"i{target.class_instance_index:03d}_n{crop_index:02d}"
                        )
                        output_image = output_dir / "images" / f"{output_stem}.png"
                        output_label = output_dir / "labels" / f"{output_stem}.txt"
                        source_image.crop(rectangle).save(output_image, format="PNG")
                        _atomic_write_text(output_label, label_content)
                        crop_records.append(
                            {
                                "output_image": _relative(output_image, dataset),
                                "output_label": _relative(output_label, dataset),
                                "source_image": _relative(source.image_path, dataset),
                                "source_label": _relative(source.label_path, dataset),
                                "target_class_id": target.class_id,
                                "target_class_name": CLASS_NAMES[target.class_id],
                                "target_instance_index": target.class_instance_index,
                                "target_label_index": target.label_index,
                                "crop_index": crop_index,
                                "original_width": source.width,
                                "original_height": source.height,
                                "crop_x1": x1,
                                "crop_y1": y1,
                                "crop_x2": x2,
                                "crop_y2": y2,
                                "crop_width": x2 - x1,
                                "crop_height": y2 - y1,
                                "context_scale": f"{sampled_scale:.8f}",
                                "effective_context_scale": f"{effective_scale:.8f}",
                                "jitter": f"{jitter_x:.8f},{jitter_y:.8f}",
                                "jitter_x": f"{jitter_x:.8f}",
                                "jitter_y": f"{jitter_y:.8f}",
                                "seed": seed,
                                "fruit_id": fruit_id,
                                "view_id": view_id,
                            }
                        )

        source_manifest_path = output_dir / "source_splits_manifest.csv"
        crop_manifest_path = output_dir / "manifest.csv"
        _write_csv(source_manifest_path, SOURCE_MANIFEST_FIELDS, source_manifest)
        _write_csv(crop_manifest_path, CROP_MANIFEST_FIELDS, crop_records)

        train_entries = [
            _list_entry(source.image_path, train_list)
            for source in split_sources["train"]
        ] + [
            _list_entry(dataset / str(record["output_image"]), train_list)
            for record in crop_records
        ]
        _atomic_write_text(train_list, "\n".join(train_entries) + "\n")
        _build_experiment_yaml(dataset, train_list, data_yaml)

        validation = _validate_generated_dataset(
            dataset,
            output_dir,
            train_list,
            data_yaml,
            split_sources,
            source_manifest,
            crop_records,
            min_polygon_area,
        )
        validation["checks"]["split_safety"] = split_safety
        validation["statistics_definition"] = {
            "effective_exposure": (
                "該 class 在實際 train list 全部 entries 中的 segmentation instance 總數；"
                "Experiment C 沒有 repeated entries。"
            ),
            "null_effective_exposure": "train list 中 label 為空的 image entry 數。",
        }

        split_hashes = {
            split: sha256_records(
                record for record in source_manifest if record["split"] == split
            )
            for split in ("train", "valid", "test")
        }
        cli_arguments = {
            "dataset": _relative(dataset, dataset),
            "output_dir": _relative(output_dir, dataset),
            "train_list": _relative(train_list, dataset),
            "data_yaml": _relative(data_yaml, dataset),
            "weight": os.path.relpath(weight, Path.cwd()),
            "anthracnose_crops_per_instance": counts[1],
            "insect_bite_crops_per_instance": counts[2],
            "context_scale": context_scale,
            "context_scale_variation": context_scale_variation,
            "center_jitter": center_jitter,
            "minimum_crop_size": minimum_crop_size,
            "maximum_crop_fraction": maximum_crop_fraction,
            "min_polygon_area": min_polygon_area,
            "seed": seed,
            "overwrite": overwrite,
        }
        reproducibility = {
            "experiment": "v21_C_targeted_crop",
            "baseline": "v19_defect_epoch200_pw050_cp01",
            "seed": seed,
            "source_split_hashes": split_hashes,
            "source_splits_manifest_sha256": sha256_file(source_manifest_path),
            "experiment_train_list_sha256": sha256_file(train_list),
            "generated_crop_manifest_sha256": sha256_file(crop_manifest_path),
            "experiment_dataset_sha256": _training_input_hash(
                source_manifest, crop_records, dataset, train_list, data_yaml
            ),
            "initial_weight": {
                "path": os.path.relpath(weight, Path.cwd()),
                "size_bytes": weight.stat().st_size,
                "sha256": sha256_file(weight),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "pytorch": importlib.metadata.version("torch"),
                "ultralytics": importlib.metadata.version("ultralytics"),
                "pillow": importlib.metadata.version("Pillow"),
                "pyyaml": importlib.metadata.version("PyYAML"),
            },
            "generator_cli_arguments": cli_arguments,
        }
        validation["reproducibility"] = reproducibility
        _write_json(output_dir / "validation_report.json", validation)
        _write_json(output_dir / "reproducibility.json", reproducibility)
        return validation
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        train_list.unlink(missing_ok=True)
        data_yaml.unlink(missing_ok=True)
        raise


def print_report(report: Mapping[str, object]) -> None:
    statistics = report["statistics"]
    reproducibility = report["reproducibility"]
    if not isinstance(statistics, dict) or not isinstance(reproducibility, dict):
        raise TypeError("Experiment C report 格式錯誤。")
    baseline = statistics["baseline"]
    experiment = statistics["experiment_c"]
    print("Experiment C targeted crop Dataset validation: PASSED")
    print(f"Original train images: {baseline['unique_original_images']}")
    print(f"Generated crop images: {experiment['generated_crop_images']}")
    print(f"Experiment C train entries: {experiment['train_list_entries']}")
    print("\nClass exposure (baseline -> Experiment C):")
    for class_name in (*CLASS_NAMES.values(), "null"):
        before = baseline["classes"][class_name]["effective_exposure"]
        after = experiment["classes"][class_name]["effective_exposure"]
        print(f"{class_name}: {before} -> {after}")
    print("\nSHA-256:")
    print(f"Dataset: {reproducibility['experiment_dataset_sha256']}")
    print(f"Train list: {reproducibility['experiment_train_list_sha256']}")
    print(f"Crop manifest: {reproducibility['generated_crop_manifest_sha256']}")
    print(f"Initial weight: {reproducibility['initial_weight']['sha256']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "建立 Experiment C：以 v19 原始 train split 為 baseline，"
            "只新增 anthracnose/insect_bite instance targeted crops。"
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-list", type=Path, default=None)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--anthracnose-crops-per-instance", type=int, default=1)
    parser.add_argument("--insect-bite-crops-per-instance", type=int, default=3)
    parser.add_argument("--context-scale", type=float, default=4.0)
    parser.add_argument("--context-scale-variation", type=float, default=0.5)
    parser.add_argument("--center-jitter", type=float, default=0.10)
    parser.add_argument("--minimum-crop-size", type=int, default=64)
    parser.add_argument("--maximum-crop-fraction", type=float, default=0.90)
    parser.add_argument("--min-polygon-area", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="以相同路徑重建既有 Experiment C 產物；不修改原始 splits。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = generate_targeted_crops(
            dataset=args.dataset,
            output_dir=args.output_dir,
            train_list=args.train_list,
            data_yaml=args.data_yaml,
            weight=args.weight,
            crops_per_instance={
                1: args.anthracnose_crops_per_instance,
                2: args.insect_bite_crops_per_instance,
            },
            context_scale=args.context_scale,
            context_scale_variation=args.context_scale_variation,
            center_jitter=args.center_jitter,
            minimum_crop_size=args.minimum_crop_size,
            maximum_crop_fraction=args.maximum_crop_fraction,
            min_polygon_area=args.min_polygon_area,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
