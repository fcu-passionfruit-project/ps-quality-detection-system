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
DEFAULT_EXPERIMENT = "C"
EXPERIMENT_NAMES = {
    "C": "v21_C_targeted_crop",
    "D": "v22_D_context6",
}
EXPERIMENT_OUTPUTS = {
    "C": (
        Path("experiments/exp_C_targeted_crop"),
        Path("experiments/train_C_targeted_crop.txt"),
        Path("data_exp_C.yaml"),
    ),
    "D": (
        Path("experiments/exp_D_context6"),
        Path("experiments/train_D_context6.txt"),
        Path("data_exp_D.yaml"),
    ),
}
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
PAIRED_PARAMETER_FIELDS = (
    "record_sequence",
    "record_id",
    "source_image",
    "source_label",
    "target_class_id",
    "target_class_name",
    "target_instance_index",
    "target_label_index",
    "crop_index",
    "seed",
    "c_nominal_context_scale",
    "c_context_variation_delta",
    "c_actual_context_scale",
    "c_jitter_x_normalized",
    "c_jitter_y_normalized",
    "c_jitter_x_pixels",
    "c_jitter_y_pixels",
    "c_accepted_attempt_index",
    "c_crop_x1",
    "c_crop_y1",
    "c_crop_x2",
    "c_crop_y2",
    "c_desired_crop_size",
    "c_boundary_or_clamp",
    "d_nominal_context_scale",
    "d_context_variation_delta",
    "d_actual_context_scale",
    "d_jitter_x_normalized",
    "d_jitter_y_normalized",
    "d_jitter_x_pixels",
    "d_jitter_y_pixels",
    "d_accepted_attempt_index",
    "d_retry_count",
    "d_retry_reason",
    "d_duplicate_with_record_sequence",
    "d_paired_first_attempt_crop_x1",
    "d_paired_first_attempt_crop_y1",
    "d_paired_first_attempt_crop_x2",
    "d_paired_first_attempt_crop_y2",
    "d_paired_first_attempt_boundary_or_clamp",
    "d_crop_x1",
    "d_crop_y1",
    "d_crop_x2",
    "d_crop_y2",
    "d_desired_crop_size",
    "d_boundary_or_clamp",
    "context_variation_paired",
    "jitter_x_paired",
    "jitter_y_paired",
    "actual_scale_plus_two",
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


def _target_extent(target: PolygonInstance) -> float:
    xs = [point[0] for point in target.points]
    ys = [point[1] for point in target.points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _desired_crop_size(
    target: PolygonInstance,
    sampled_context_scale: float,
    minimum_crop_size: int,
) -> int:
    return max(
        minimum_crop_size,
        math.ceil(_target_extent(target) * sampled_context_scale),
    )


def _normalized_jitter(
    target: PolygonInstance,
    sampled_context_scale: float,
    minimum_crop_size: int,
    jitter_fraction: float,
    jitter_pixels: float,
) -> float:
    maximum_jitter = (
        _desired_crop_size(target, sampled_context_scale, minimum_crop_size)
        * jitter_fraction
    )
    if math.isclose(maximum_jitter, 0.0):
        return 0.0
    return jitter_pixels / maximum_jitter


def _choose_crop_rectangle_from_realization(
    source: SourceImage,
    target: PolygonInstance,
    sampled_context_scale: float,
    minimum_crop_size: int,
    maximum_crop_fraction: float,
    jitter_fraction: float,
    jitter_x_normalized: float,
    jitter_y_normalized: float,
) -> tuple[int, int, int, int, float, float, float]:
    xs = [point[0] for point in target.points]
    ys = [point[1] for point in target.points]
    target_extent = _target_extent(target)
    desired_length = _desired_crop_size(
        target, sampled_context_scale, minimum_crop_size
    )
    maximum_jitter = desired_length * jitter_fraction
    jitter_x = jitter_x_normalized * maximum_jitter
    jitter_y = jitter_y_normalized * maximum_jitter
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


def _geometry_details(
    source: SourceImage,
    target: PolygonInstance,
    sampled_context_scale: float,
    minimum_crop_size: int,
    maximum_crop_fraction: float,
    jitter_x: float,
    jitter_y: float,
    rectangle: tuple[int, int, int, int],
) -> dict[str, object]:
    xs = [point[0] for point in target.points]
    ys = [point[1] for point in target.points]
    desired_length = _desired_crop_size(
        target, sampled_context_scale, minimum_crop_size
    )
    x1, y1, x2, y2 = rectangle
    ideal_x1 = (min(xs) + max(xs)) / 2.0 - (x2 - x1) / 2.0 + jitter_x
    ideal_y1 = (min(ys) + max(ys)) / 2.0 - (y2 - y1) / 2.0 + jitter_y
    width_limited = x2 - x1 < desired_length
    height_limited = y2 - y1 < desired_length
    x_clamped = not math.isclose(x1, round(ideal_x1), abs_tol=1e-9)
    y_clamped = not math.isclose(y1, round(ideal_y1), abs_tol=1e-9)
    return {
        "desired_crop_size": desired_length,
        "width_limited": width_limited,
        "height_limited": height_limited,
        "x_clamped": x_clamped,
        "y_clamped": y_clamped,
        "boundary_or_clamp": width_limited or height_limited or x_clamped or y_clamped,
        "maximum_crop_fraction": maximum_crop_fraction,
    }


def _target_record_id(
    dataset: Path,
    source: SourceImage,
    target: PolygonInstance,
    crop_index: int,
) -> str:
    identity = {
        "source_image": _relative(source.image_path, dataset),
        "source_label": _relative(source.label_path, dataset),
        "target_class_id": target.class_id,
        "target_instance_index": target.class_instance_index,
        "target_label_index": target.label_index,
        "crop_index": crop_index,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _record_local_rng(seed: int, record_id: str, retry_index: int) -> random.Random:
    material = f"{seed}:{record_id}:{retry_index}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(local_seed)


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
    experiment_key: str,
    experiment_label: str,
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
    neighboring_label_mismatch = 0
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
        rectangle = (
            int(record["crop_x1"]),
            int(record["crop_y1"]),
            int(record["crop_x2"]),
            int(record["crop_y2"]),
        )
        expected_label_content, _ = _format_label(
            source.instances, rectangle, min_polygon_area
        )
        if output_label.read_text(encoding="utf-8") != expected_label_content:
            neighboring_label_mismatch += 1
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

    if any(
        (
            malformed,
            out_of_range,
            missing_target,
            neighboring_label_mismatch,
            generated_sources_not_train,
        )
    ):
        raise ValueError(
            "generated Dataset 驗證失敗："
            f"malformed={malformed}, out_of_range={out_of_range}, "
            f"missing_target={missing_target}, "
            f"neighboring_label_mismatch={neighboring_label_mismatch}, "
            f"source_not_train={generated_sources_not_train}"
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
        raise ValueError(f"Experiment {experiment_label} train list entry 數量不正確。")
    if Counter(resolved_entries) != Counter({path: 1 for path in original_train | generated}):
        raise ValueError(
            f"Experiment {experiment_label} train list 必須包含每張原圖一次及每張 crop 一次。"
        )

    experiment_yaml = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    baseline_yaml = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if experiment_yaml.get("train") != _relative(train_list, dataset):
        raise ValueError(
            f"Experiment {experiment_label} data YAML train 未指向對應 train list。"
        )
    for key in ("val", "test", "nc", "names", "roboflow"):
        if experiment_yaml.get(key) != baseline_yaml.get(key):
            raise ValueError(
                f"Experiment {experiment_label} data YAML 的 {key} 與 baseline 不同。"
            )

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
    target_records = Counter(
        int(record["target_class_id"]) for record in crop_records
    )
    unique_targets = {
        (
            str(record["source_label"]),
            int(record["target_class_id"]),
            int(record["target_instance_index"]),
            int(record["target_label_index"]),
        )
        for record in crop_records
    }
    unique_targets_by_class = Counter(item[1] for item in unique_targets)
    generated_statistics = _dataset_statistics(generated_sources)
    neighboring_instances = {
        CLASS_NAMES[class_id]: (
            int(generated_statistics["classes"][CLASS_NAMES[class_id]]["instances"])
            - target_records[class_id]
        )
        for class_id in CLASS_NAMES
    }
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
            "neighboring_labels_match_generator_policy": (
                neighboring_label_mismatch == 0
            ),
            "neighboring_label_mismatch": neighboring_label_mismatch,
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
            experiment_key: {
                "train_list_entries": len(entries),
                "unique_original_images": len(split_sources["train"]),
                "generated_crop_images": len(generated_sources),
                **_dataset_statistics(all_sources + generated_sources),
            },
        },
        "crop_exposure": {
            "target_crop_records": {
                CLASS_NAMES[class_id]: target_records[class_id]
                for class_id in CLASS_NAMES
            },
            "unique_target_instances": {
                CLASS_NAMES[class_id]: unique_targets_by_class[class_id]
                for class_id in CLASS_NAMES
            },
            "generated_crop_total_label_instances": {
                CLASS_NAMES[class_id]: int(
                    generated_statistics["classes"][CLASS_NAMES[class_id]]["instances"]
                )
                for class_id in CLASS_NAMES
            },
            "generated_crop_neighboring_instances": neighboring_instances,
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


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise ValueError(f"無法讀取 CSV：{path}: {exc}") from exc


def _crop_exposure_from_manifest(
    dataset: Path,
    manifest: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    total_labels: Counter[int] = Counter()
    target_records: Counter[int] = Counter()
    unique_targets: set[tuple[str, int, int, int]] = set()
    for record in manifest:
        target_class_id = int(record["target_class_id"])
        target_records[target_class_id] += 1
        unique_targets.add(
            (
                str(record["source_label"]),
                target_class_id,
                int(record["target_instance_index"]),
                int(record["target_label_index"]),
            )
        )
        label_path = dataset / str(record["output_label"])
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                total_labels[int(raw_line.split()[0])] += 1
    unique_by_class = Counter(item[1] for item in unique_targets)
    return {
        "target_crop_records": {
            CLASS_NAMES[class_id]: target_records[class_id]
            for class_id in CLASS_NAMES
        },
        "unique_target_instances": {
            CLASS_NAMES[class_id]: unique_by_class[class_id]
            for class_id in CLASS_NAMES
        },
        "generated_crop_total_label_instances": {
            CLASS_NAMES[class_id]: total_labels[class_id]
            for class_id in CLASS_NAMES
        },
        "generated_crop_neighboring_instances": {
            CLASS_NAMES[class_id]: total_labels[class_id] - target_records[class_id]
            for class_id in CLASS_NAMES
        },
    }


def _replay_experiment_c(
    dataset: Path,
    split_sources: Mapping[str, Sequence[SourceImage]],
    reference_dir: Path,
    counts: Mapping[int, int],
    context_scale_variation: float,
    center_jitter: float,
    minimum_crop_size: int,
    maximum_crop_fraction: float,
    min_polygon_area: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    manifest_path = reference_dir / "manifest.csv"
    reproducibility_path = reference_dir / "reproducibility.json"
    source_manifest_path = reference_dir / "source_splits_manifest.csv"
    if not manifest_path.is_file() or not reproducibility_path.is_file():
        raise ValueError(f"Experiment C reference artifacts 不完整：{reference_dir}")
    reference_manifest = _read_csv_records(manifest_path)
    try:
        reference_reproducibility = json.loads(
            reproducibility_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法解析 Experiment C reproducibility：{exc}") from exc
    reference_arguments = reference_reproducibility.get("generator_cli_arguments")
    if not isinstance(reference_arguments, dict):
        raise ValueError("Experiment C reproducibility 缺少 generator_cli_arguments。")
    expected_same = {
        "anthracnose_crops_per_instance": counts[1],
        "insect_bite_crops_per_instance": counts[2],
        "context_scale_variation": context_scale_variation,
        "center_jitter": center_jitter,
        "minimum_crop_size": minimum_crop_size,
        "maximum_crop_fraction": maximum_crop_fraction,
        "min_polygon_area": min_polygon_area,
        "seed": seed,
    }
    mismatches = {
        key: {"reference": reference_arguments.get(key), "requested": value}
        for key, value in expected_same.items()
        if reference_arguments.get(key) != value
    }
    reference_context_scale = reference_arguments.get("context_scale")
    if reference_context_scale != 4.0:
        mismatches["context_scale"] = {
            "reference": reference_context_scale,
            "expected": 4.0,
        }
    if mismatches:
        raise ValueError(
            "Experiment C replay 參數與 reference 不同："
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    current_source_manifest = _collect_source_manifest(dataset, split_sources)
    if source_manifest_path.is_file():
        current_source_manifest_sha256 = sha256_file(source_manifest_path)
        if (
            current_source_manifest_sha256
            != reference_reproducibility.get("source_splits_manifest_sha256")
        ):
            raise ValueError("Experiment C source manifest hash 與 reproducibility 不一致。")
    reference_source_manifest = _read_csv_records(source_manifest_path)
    normalized_reference_source_manifest = [
        {
            "split": record["split"],
            "kind": record["kind"],
            "path": record["path"],
            "size_bytes": int(record["size_bytes"]),
            "sha256": record["sha256"],
        }
        for record in reference_source_manifest
    ]
    if list(current_source_manifest) != normalized_reference_source_manifest:
        raise ValueError("目前 source Dataset 無法逐筆對上 Experiment C source manifest。")

    rng = random.Random(seed)
    replay_records: list[dict[str, object]] = []
    rectangle_matches = 0
    geometry_matches = 0
    label_matches = 0
    image_pixel_matches = 0
    sequence_matches = 0
    exceptions: list[dict[str, object]] = []

    for source in split_sources["train"]:
        with Image.open(source.image_path) as source_image:
            source_image.load()
            for target in source.instances:
                crop_count = counts.get(target.class_id, 0)
                used_rectangles: set[tuple[int, int, int, int]] = set()
                for crop_index in range(crop_count):
                    sampled_scale = float(reference_context_scale)
                    selection = None
                    rectangle = None
                    accepted_attempt = -1
                    for attempt in range(100):
                        sampled_scale = rng.uniform(
                            float(reference_context_scale) - context_scale_variation,
                            float(reference_context_scale) + context_scale_variation,
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
                            accepted_attempt = attempt
                            break
                    if selection is None or rectangle is None or accepted_attempt < 0:
                        raise ValueError("Experiment C replay 無法重現 accepted crop。")
                    used_rectangles.add(rectangle)
                    sequence = len(replay_records)
                    if sequence >= len(reference_manifest):
                        raise ValueError("Experiment C replay 產生多於 manifest 的 records。")
                    reference = reference_manifest[sequence]
                    record_id = _target_record_id(
                        dataset, source, target, crop_index
                    )
                    expected_identity = (
                        _relative(source.image_path, dataset),
                        _relative(source.label_path, dataset),
                        str(target.class_id),
                        str(target.class_instance_index),
                        str(target.label_index),
                        str(crop_index),
                    )
                    actual_identity = tuple(
                        reference[field]
                        for field in (
                            "source_image",
                            "source_label",
                            "target_class_id",
                            "target_instance_index",
                            "target_label_index",
                            "crop_index",
                        )
                    )
                    sequence_same = actual_identity == expected_identity
                    sequence_matches += int(sequence_same)
                    reference_rectangle = tuple(
                        int(reference[field])
                        for field in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")
                    )
                    rectangle_same = rectangle == reference_rectangle
                    rectangle_matches += int(rectangle_same)
                    x1, y1, x2, y2, effective_scale, jitter_x, jitter_y = selection
                    geometry_same = all(
                        (
                            reference["context_scale"] == f"{sampled_scale:.8f}",
                            reference["effective_context_scale"]
                            == f"{effective_scale:.8f}",
                            reference["jitter_x"] == f"{jitter_x:.8f}",
                            reference["jitter_y"] == f"{jitter_y:.8f}",
                            int(reference["crop_width"]) == x2 - x1,
                            int(reference["crop_height"]) == y2 - y1,
                        )
                    )
                    geometry_matches += int(geometry_same)
                    label_content, retained = _format_label(
                        source.instances, rectangle, min_polygon_area
                    )
                    if target.label_index not in retained:
                        raise ValueError("Experiment C replay target polygon 遺失。")
                    reference_label = dataset / reference["output_label"]
                    label_same = (
                        reference_label.read_text(encoding="utf-8") == label_content
                    )
                    label_matches += int(label_same)
                    reference_image = dataset / reference["output_image"]
                    with Image.open(reference_image) as saved_image:
                        saved_image.load()
                        replay_image = source_image.crop(rectangle)
                        image_same = (
                            saved_image.mode == replay_image.mode
                            and saved_image.size == replay_image.size
                            and saved_image.tobytes() == replay_image.tobytes()
                        )
                    image_pixel_matches += int(image_same)
                    if not all(
                        (sequence_same, rectangle_same, geometry_same, label_same, image_same)
                    ):
                        exceptions.append(
                            {
                                "record_sequence": sequence,
                                "record_id": record_id,
                                "sequence_same": sequence_same,
                                "rectangle_same": rectangle_same,
                                "geometry_same": geometry_same,
                                "label_same": label_same,
                                "image_pixels_same": image_same,
                            }
                        )
                    details = _geometry_details(
                        source,
                        target,
                        sampled_scale,
                        minimum_crop_size,
                        maximum_crop_fraction,
                        jitter_x,
                        jitter_y,
                        rectangle,
                    )
                    replay_records.append(
                        {
                            "record_sequence": sequence,
                            "record_id": record_id,
                            "source_image": expected_identity[0],
                            "source_label": expected_identity[1],
                            "target_class_id": target.class_id,
                            "target_class_name": CLASS_NAMES[target.class_id],
                            "target_instance_index": target.class_instance_index,
                            "target_label_index": target.label_index,
                            "crop_index": crop_index,
                            "seed": seed,
                            "nominal_context_scale": float(reference_context_scale),
                            "context_variation_delta": sampled_scale
                            - float(reference_context_scale),
                            "actual_context_scale": sampled_scale,
                            "jitter_x_normalized": _normalized_jitter(
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                center_jitter,
                                jitter_x,
                            ),
                            "jitter_y_normalized": _normalized_jitter(
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                center_jitter,
                                jitter_y,
                            ),
                            "jitter_x_pixels": jitter_x,
                            "jitter_y_pixels": jitter_y,
                            "accepted_attempt_index": accepted_attempt,
                            "rectangle": rectangle,
                            "geometry": details,
                        }
                    )

    total = len(reference_manifest)
    report = {
        "status": "PASS" if not exceptions and len(replay_records) == total else "FAIL",
        "reference_directory": _relative(reference_dir, dataset),
        "reference_manifest_sha256": sha256_file(manifest_path),
        "records_expected": total,
        "records_replayed": len(replay_records),
        "record_sequence_same": sequence_matches,
        "crop_rectangles_same": rectangle_matches,
        "generated_image_geometry_and_pixels_same": image_pixel_matches,
        "manifest_geometry_same": geometry_matches,
        "labels_same": label_matches,
        "exceptions": exceptions,
    }
    if report["status"] != "PASS":
        raise ValueError(
            "Experiment C deterministic replay 失敗："
            + json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
    return replay_records, report, reference_reproducibility


def _validate_experiment_d_target_contract(
    replay_records: Sequence[Mapping[str, object]],
) -> None:
    target_records = Counter(
        int(record["target_class_id"]) for record in replay_records
    )
    unique_targets = {
        (
            str(record["source_label"]),
            int(record["target_class_id"]),
            int(record["target_instance_index"]),
            int(record["target_label_index"]),
        )
        for record in replay_records
    }
    unique_by_class = Counter(item[1] for item in unique_targets)
    actual = {
        "crop_target_records": len(replay_records),
        "unique_target_instances": len(unique_targets),
        "anthracnose_unique_targets": unique_by_class[1],
        "anthracnose_target_records": target_records[1],
        "insect_bite_unique_targets": unique_by_class[2],
        "insect_bite_target_records": target_records[2],
    }
    expected = {
        "crop_target_records": 132,
        "unique_target_instances": 80,
        "anthracnose_unique_targets": 54,
        "anthracnose_target_records": 54,
        "insect_bite_unique_targets": 26,
        "insect_bite_target_records": 78,
    }
    if actual != expected:
        raise ValueError(
            "Experiment D target contract 不符："
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


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
    experiment: str = DEFAULT_EXPERIMENT,
    paired_reference_dir: Path | None = None,
    enforce_experiment_d_contract: bool = True,
) -> dict[str, object]:
    experiment = experiment.upper()
    if experiment not in EXPERIMENT_NAMES:
        raise ValueError(f"未知 experiment：{experiment!r}")
    experiment_name = EXPERIMENT_NAMES[experiment]
    experiment_key = f"experiment_{experiment.lower()}"
    default_output_dir, default_train_list, default_data_yaml = EXPERIMENT_OUTPUTS[
        experiment
    ]
    dataset = dataset.resolve()
    if not dataset.is_dir():
        raise ValueError(f"找不到 Dataset：{dataset}")
    output_dir = (
        (dataset / (output_dir or default_output_dir)).resolve()
        if output_dir is None or not output_dir.is_absolute()
        else output_dir.resolve()
    )
    train_list = (
        (dataset / (train_list or default_train_list)).resolve()
        if train_list is None or not train_list.is_absolute()
        else train_list.resolve()
    )
    data_yaml = (
        (dataset / (data_yaml or default_data_yaml)).resolve()
        if data_yaml is None or not data_yaml.is_absolute()
        else data_yaml.resolve()
    )
    weight = weight.resolve()
    if paired_reference_dir is not None:
        paired_reference_dir = (
            (dataset / paired_reference_dir).resolve()
            if not paired_reference_dir.is_absolute()
            else paired_reference_dir.resolve()
        )
    if experiment == "D" and paired_reference_dir is None:
        raise ValueError("Experiment D 必須提供 --paired-reference-dir。")
    if experiment != "D" and paired_reference_dir is not None:
        raise ValueError("--paired-reference-dir 只適用於 Experiment D。")
    if experiment == "D" and not math.isclose(context_scale, 6.0):
        raise ValueError("Experiment D nominal context_scale 必須是 6.0。")

    for path in (output_dir, train_list, data_yaml):
        if not path.is_relative_to(dataset):
            raise ValueError(f"Experiment {experiment} 輸出必須位於 Dataset root：{path}")
        for split in ("train", "valid", "test"):
            split_root = (dataset / split).resolve()
            if path == split_root or path.is_relative_to(split_root):
                raise ValueError(f"Experiment {experiment} 不可寫入原始 {split} split：{path}")
    if (
        output_dir == dataset
        or train_list.suffix.lower() != ".txt"
        or data_yaml.suffix.lower() not in {".yaml", ".yml"}
    ):
        raise ValueError(f"Experiment {experiment} 輸出路徑或副檔名無效。")
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

    replay_records: list[dict[str, object]] = []
    replay_report: dict[str, object] | None = None
    reference_reproducibility: dict[str, object] | None = None
    if paired_reference_dir is not None:
        replay_records, replay_report, reference_reproducibility = (
            _replay_experiment_c(
                dataset,
                split_sources,
                paired_reference_dir,
                counts,
                context_scale_variation,
                center_jitter,
                minimum_crop_size,
                maximum_crop_fraction,
                min_polygon_area,
                seed,
            )
        )
        if experiment == "D" and enforce_experiment_d_contract:
            _validate_experiment_d_target_contract(replay_records)
        reference_weight = reference_reproducibility.get("initial_weight")
        if not isinstance(reference_weight, dict):
            raise ValueError("Experiment C reproducibility 缺少 initial_weight。")
        current_weight_sha256 = sha256_file(weight)
        if reference_weight.get("sha256") != current_weight_sha256:
            raise ValueError(
                "Experiment D initial weight 與 Experiment C 不同："
                f"C={reference_weight.get('sha256')}, D={current_weight_sha256}"
            )

    existing = [path for path in (output_dir, train_list, data_yaml) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"Experiment {experiment} 輸出已存在；若要安全重建，請加 --overwrite："
            + ", ".join(str(path) for path in existing)
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "images").mkdir()
    (output_dir / "labels").mkdir()

    rng = random.Random(seed)
    crop_records: list[dict[str, object]] = []
    paired_parameter_records: list[dict[str, object]] = []
    try:
        for source in split_sources["train"]:
            fruit_id, view_id = _parse_fruit_view(source.image_path)
            with Image.open(source.image_path) as source_image:
                source_image.load()
                for target in source.instances:
                    crop_count = counts.get(target.class_id, 0)
                    used_rectangles: set[tuple[int, int, int, int]] = set()
                    rectangle_record_sequences: dict[
                        tuple[int, int, int, int], int
                    ] = {}
                    for crop_index in range(crop_count):
                        rectangle: tuple[int, int, int, int] | None = None
                        selection: tuple[int, int, int, int, float, float, float] | None = None
                        sampled_scale = context_scale
                        accepted_attempt = -1
                        retry_reason = ""
                        duplicate_with_sequence: int | str = ""
                        paired_first_rectangle: tuple[int, int, int, int] | None = None
                        paired_first_details: dict[str, object] | None = None
                        reference_record: Mapping[str, object] | None = None
                        record_id = _target_record_id(
                            dataset, source, target, crop_index
                        )
                        if replay_records:
                            sequence = len(crop_records)
                            if sequence >= len(replay_records):
                                raise ValueError(
                                    "Experiment D target sequence 多於 Experiment C replay。"
                                )
                            reference_record = replay_records[sequence]
                            if reference_record["record_id"] != record_id:
                                raise ValueError(
                                    f"Experiment D target sequence mismatch：record {sequence}"
                                )
                            sampled_scale = context_scale + float(
                                reference_record["context_variation_delta"]
                            )
                            selection = _choose_crop_rectangle_from_realization(
                                source,
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                maximum_crop_fraction,
                                center_jitter,
                                float(reference_record["jitter_x_normalized"]),
                                float(reference_record["jitter_y_normalized"]),
                            )
                            rectangle = selection[:4]
                            paired_first_rectangle = rectangle
                            paired_first_details = _geometry_details(
                                source,
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                maximum_crop_fraction,
                                selection[5],
                                selection[6],
                                rectangle,
                            )
                            if rectangle not in used_rectangles:
                                accepted_attempt = 0
                            else:
                                retry_reason = "paired_first_attempt_duplicate_rectangle"
                                duplicate_with_sequence = rectangle_record_sequences[
                                    rectangle
                                ]
                                for retry_index in range(1, 100):
                                    local_rng = _record_local_rng(
                                        seed, record_id, retry_index
                                    )
                                    sampled_scale = local_rng.uniform(
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
                                        local_rng,
                                    )
                                    rectangle = selection[:4]
                                    if rectangle not in used_rectangles:
                                        accepted_attempt = retry_index
                                        break
                        else:
                            for attempt in range(100):
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
                                    accepted_attempt = attempt
                                    break
                        if (
                            rectangle is None
                            or selection is None
                            or rectangle in used_rectangles
                            or accepted_attempt < 0
                        ):
                            raise ValueError(
                                f"無法為 {source.image_path.name} 的 {CLASS_NAMES[target.class_id]} "
                                "產生互不相同的 deterministic crops；請調整 context/jitter。"
                            )
                        used_rectangles.add(rectangle)
                        rectangle_record_sequences[rectangle] = len(crop_records)
                        x1, y1, x2, y2, effective_scale, jitter_x, jitter_y = selection
                        if reference_record is not None:
                            d_jitter_x_normalized = _normalized_jitter(
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                center_jitter,
                                jitter_x,
                            )
                            d_jitter_y_normalized = _normalized_jitter(
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                center_jitter,
                                jitter_y,
                            )
                            d_details = _geometry_details(
                                source,
                                target,
                                sampled_scale,
                                minimum_crop_size,
                                maximum_crop_fraction,
                                jitter_x,
                                jitter_y,
                                rectangle,
                            )
                            c_details = reference_record["geometry"]
                            if not isinstance(c_details, dict):
                                raise TypeError("Experiment C replay geometry 格式錯誤。")
                            c_rectangle = reference_record["rectangle"]
                            if not isinstance(c_rectangle, tuple):
                                raise TypeError("Experiment C replay rectangle 格式錯誤。")
                            if paired_first_rectangle is None or paired_first_details is None:
                                raise TypeError("Experiment D paired first attempt 格式錯誤。")
                            c_delta = float(
                                reference_record["context_variation_delta"]
                            )
                            c_jitter_x_normalized = float(
                                reference_record["jitter_x_normalized"]
                            )
                            c_jitter_y_normalized = float(
                                reference_record["jitter_y_normalized"]
                            )
                            paired_parameter_records.append(
                                {
                                    "record_sequence": reference_record["record_sequence"],
                                    "record_id": record_id,
                                    "source_image": reference_record["source_image"],
                                    "source_label": reference_record["source_label"],
                                    "target_class_id": target.class_id,
                                    "target_class_name": CLASS_NAMES[target.class_id],
                                    "target_instance_index": target.class_instance_index,
                                    "target_label_index": target.label_index,
                                    "crop_index": crop_index,
                                    "seed": seed,
                                    "c_nominal_context_scale": f"{float(reference_record['nominal_context_scale']):.8f}",
                                    "c_context_variation_delta": f"{c_delta:.17g}",
                                    "c_actual_context_scale": f"{float(reference_record['actual_context_scale']):.8f}",
                                    "c_jitter_x_normalized": f"{c_jitter_x_normalized:.17g}",
                                    "c_jitter_y_normalized": f"{c_jitter_y_normalized:.17g}",
                                    "c_jitter_x_pixels": f"{float(reference_record['jitter_x_pixels']):.8f}",
                                    "c_jitter_y_pixels": f"{float(reference_record['jitter_y_pixels']):.8f}",
                                    "c_accepted_attempt_index": reference_record["accepted_attempt_index"],
                                    "c_crop_x1": c_rectangle[0],
                                    "c_crop_y1": c_rectangle[1],
                                    "c_crop_x2": c_rectangle[2],
                                    "c_crop_y2": c_rectangle[3],
                                    "c_desired_crop_size": c_details["desired_crop_size"],
                                    "c_boundary_or_clamp": c_details["boundary_or_clamp"],
                                    "d_nominal_context_scale": f"{context_scale:.8f}",
                                    "d_context_variation_delta": f"{sampled_scale - context_scale:.17g}",
                                    "d_actual_context_scale": f"{sampled_scale:.8f}",
                                    "d_jitter_x_normalized": f"{d_jitter_x_normalized:.17g}",
                                    "d_jitter_y_normalized": f"{d_jitter_y_normalized:.17g}",
                                    "d_jitter_x_pixels": f"{jitter_x:.8f}",
                                    "d_jitter_y_pixels": f"{jitter_y:.8f}",
                                    "d_accepted_attempt_index": accepted_attempt,
                                    "d_retry_count": accepted_attempt,
                                    "d_retry_reason": retry_reason,
                                    "d_duplicate_with_record_sequence": duplicate_with_sequence,
                                    "d_paired_first_attempt_crop_x1": paired_first_rectangle[0],
                                    "d_paired_first_attempt_crop_y1": paired_first_rectangle[1],
                                    "d_paired_first_attempt_crop_x2": paired_first_rectangle[2],
                                    "d_paired_first_attempt_crop_y2": paired_first_rectangle[3],
                                    "d_paired_first_attempt_boundary_or_clamp": paired_first_details[
                                        "boundary_or_clamp"
                                    ],
                                    "d_crop_x1": x1,
                                    "d_crop_y1": y1,
                                    "d_crop_x2": x2,
                                    "d_crop_y2": y2,
                                    "d_desired_crop_size": d_details["desired_crop_size"],
                                    "d_boundary_or_clamp": d_details["boundary_or_clamp"],
                                    "context_variation_paired": math.isclose(
                                        sampled_scale - context_scale,
                                        c_delta,
                                        rel_tol=0.0,
                                        abs_tol=1e-15,
                                    ),
                                    "jitter_x_paired": math.isclose(
                                        d_jitter_x_normalized,
                                        c_jitter_x_normalized,
                                        rel_tol=0.0,
                                        abs_tol=1e-15,
                                    ),
                                    "jitter_y_paired": math.isclose(
                                        d_jitter_y_normalized,
                                        c_jitter_y_normalized,
                                        rel_tol=0.0,
                                        abs_tol=1e-15,
                                    ),
                                    "actual_scale_plus_two": math.isclose(
                                        sampled_scale,
                                        float(reference_record["actual_context_scale"])
                                        + 2.0,
                                        rel_tol=0.0,
                                        abs_tol=1e-12,
                                    ),
                                }
                            )
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
        paired_parameters_path = output_dir / "paired_parameters.csv"
        replay_report_path = output_dir / "experiment_c_replay_report.json"
        if paired_parameter_records:
            _write_csv(
                paired_parameters_path,
                PAIRED_PARAMETER_FIELDS,
                paired_parameter_records,
            )
        if replay_report is not None:
            _write_json(replay_report_path, replay_report)

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
            experiment_key,
            experiment,
        )
        validation["checks"]["split_safety"] = split_safety
        validation["statistics_definition"] = {
            "effective_exposure": (
                "該 class 在實際 train list 全部 entries 中的 segmentation instance 總數；"
                f"Experiment {experiment} 沒有 repeated full-image entries。"
            ),
            "null_effective_exposure": "train list 中 label 為空的 image entry 數。",
            "target_crop_records": (
                "由 target selection 主動建立的 crop records；與 crop label 內的 total "
                "instance exposure 分開計算。"
            ),
            "generated_crop_neighboring_instances": (
                "generated crop labels 的 class instance 總數減去該 class target crop records；"
                "包含因 field-of-view 擴大自然進入 crop 的合法 annotations。"
            ),
        }

        pairing_audit_path = output_dir / "experiment_c_vs_d_audit.json"
        pairing_audit: dict[str, object] | None = None
        if paired_parameter_records:
            if paired_reference_dir is None:
                raise RuntimeError("paired generation 缺少 reference directory。")
            reference_crop_exposure = _crop_exposure_from_manifest(
                dataset,
                _read_csv_records(paired_reference_dir / "manifest.csv"),
            )
            context_paired = sum(
                str(record["context_variation_paired"]) == "True"
                or record["context_variation_paired"] is True
                for record in paired_parameter_records
            )
            jitter_x_paired = sum(
                str(record["jitter_x_paired"]) == "True"
                or record["jitter_x_paired"] is True
                for record in paired_parameter_records
            )
            jitter_y_paired = sum(
                str(record["jitter_y_paired"]) == "True"
                or record["jitter_y_paired"] is True
                for record in paired_parameter_records
            )
            scale_plus_two = sum(
                str(record["actual_scale_plus_two"]) == "True"
                or record["actual_scale_plus_two"] is True
                for record in paired_parameter_records
            )
            retry_records = [
                record
                for record in paired_parameter_records
                if int(record["d_retry_count"]) > 0
            ]
            pairing_exceptions = [
                {
                    "record_sequence": record["record_sequence"],
                    "record_id": record["record_id"],
                    "retry_count": record["d_retry_count"],
                    "reason": record["d_retry_reason"],
                    "duplicate_with_record_sequence": record[
                        "d_duplicate_with_record_sequence"
                    ],
                    "paired_first_attempt_rectangle": [
                        record["d_paired_first_attempt_crop_x1"],
                        record["d_paired_first_attempt_crop_y1"],
                        record["d_paired_first_attempt_crop_x2"],
                        record["d_paired_first_attempt_crop_y2"],
                    ],
                    "context_variation_paired": record["context_variation_paired"],
                    "jitter_x_paired": record["jitter_x_paired"],
                    "jitter_y_paired": record["jitter_y_paired"],
                    "actual_scale_plus_two": record["actual_scale_plus_two"],
                }
                for record in paired_parameter_records
                if not all(
                    bool(record[field])
                    for field in (
                        "context_variation_paired",
                        "jitter_x_paired",
                        "jitter_y_paired",
                        "actual_scale_plus_two",
                    )
                )
            ]
            rectangle_groups: dict[tuple[str, int, int, int, int], list[int]] = {}
            for index, record in enumerate(crop_records):
                key = (
                    str(record["source_image"]),
                    int(record["crop_x1"]),
                    int(record["crop_y1"]),
                    int(record["crop_x2"]),
                    int(record["crop_y2"]),
                )
                rectangle_groups.setdefault(key, []).append(index)
            duplicate_rectangle_groups = [
                {"key": list(key), "record_sequences": sequences}
                for key, sequences in rectangle_groups.items()
                if len(sequences) > 1
            ]
            image_hash_groups: dict[str, list[int]] = {}
            for index, record in enumerate(crop_records):
                digest = sha256_file(dataset / str(record["output_image"]))
                image_hash_groups.setdefault(digest, []).append(index)
            duplicate_image_groups = [
                {"sha256": digest, "record_sequences": sequences}
                for digest, sequences in image_hash_groups.items()
                if len(sequences) > 1
            ]
            boundary_records = sum(
                bool(record["d_boundary_or_clamp"])
                for record in paired_parameter_records
            )
            boundary_status_changed = sum(
                bool(record["c_boundary_or_clamp"])
                != bool(record["d_boundary_or_clamp"])
                for record in paired_parameter_records
            )
            total_records = len(paired_parameter_records)
            pairing_audit = {
                "status": "PASS",
                "fully_paired": not pairing_exceptions,
                "experiment_c_replay": replay_report,
                "target_pairing": {
                    "records": total_records,
                    "record_sequence_same": total_records,
                    "unique_target_instances": sum(
                        validation["crop_exposure"]["unique_target_instances"].values()
                    ),
                    "target_crop_records_by_class": validation["crop_exposure"][
                        "target_crop_records"
                    ],
                },
                "random_pairing": {
                    "context_variation_paired": context_paired,
                    "jitter_x_paired": jitter_x_paired,
                    "jitter_y_paired": jitter_y_paired,
                    "d_actual_scale_equals_c_plus_2": scale_plus_two,
                    "retry_records": len(retry_records),
                    "records_unable_to_remain_fully_paired": len(pairing_exceptions),
                    "d_boundary_or_clamp_records": boundary_records,
                    "c_to_d_boundary_or_clamp_status_changed": boundary_status_changed,
                    "exceptions": pairing_exceptions,
                },
                "duplicates": {
                    "duplicate_rectangle_groups": len(duplicate_rectangle_groups),
                    "duplicate_rectangle_extra_records": sum(
                        len(group["record_sequences"]) - 1
                        for group in duplicate_rectangle_groups
                    ),
                    "duplicate_rectangles": duplicate_rectangle_groups,
                    "exact_duplicate_image_groups": len(duplicate_image_groups),
                    "exact_duplicate_image_extra_records": sum(
                        len(group["record_sequences"]) - 1
                        for group in duplicate_image_groups
                    ),
                    "exact_duplicate_images": duplicate_image_groups,
                },
                "exposure": {
                    "experiment_c": reference_crop_exposure,
                    "experiment_d": validation["crop_exposure"],
                },
            }
            _write_json(pairing_audit_path, pairing_audit)
            validation["checks"]["experiment_c_replay"] = replay_report
            validation["checks"]["random_pairing"] = pairing_audit[
                "random_pairing"
            ]
            validation["checks"]["duplicates"] = pairing_audit["duplicates"]
        validation["status"] = "PASS"

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
            "experiment": experiment,
            "paired_reference_dir": (
                _relative(paired_reference_dir, dataset)
                if paired_reference_dir is not None
                else None
            ),
        }
        reproducibility = {
            "experiment": experiment_name,
            "baseline": "v19_defect_epoch200_pw050_cp01",
            "seed": seed,
            "source_dataset_sha256": sha256_records(source_manifest),
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
        if paired_parameter_records:
            reproducibility["paired_generation"] = {
                "reference_experiment": reference_reproducibility.get("experiment")
                if reference_reproducibility is not None
                else None,
                "reference_dataset_sha256": reference_reproducibility.get(
                    "experiment_dataset_sha256"
                )
                if reference_reproducibility is not None
                else None,
                "paired_parameters_path": _relative(
                    paired_parameters_path, dataset
                ),
                "paired_parameters_sha256": sha256_file(paired_parameters_path),
                "experiment_c_replay_report_path": _relative(
                    replay_report_path, dataset
                ),
                "experiment_c_replay_report_sha256": sha256_file(
                    replay_report_path
                ),
                "experiment_c_vs_d_audit_path": _relative(
                    pairing_audit_path, dataset
                ),
                "experiment_c_vs_d_audit_sha256": sha256_file(
                    pairing_audit_path
                ),
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
        raise TypeError("targeted crop report 格式錯誤。")
    baseline = statistics["baseline"]
    experiment_name = str(reproducibility["experiment"])
    experiment_key = next(
        key for key in statistics if str(key).startswith("experiment_")
    )
    experiment = statistics[experiment_key]
    print(
        f"{experiment_name} targeted crop Dataset validation: "
        f"{report.get('status', 'PASS')}"
    )
    print(f"Original train images: {baseline['unique_original_images']}")
    print(f"Generated crop images: {experiment['generated_crop_images']}")
    print(f"Experiment train entries: {experiment['train_list_entries']}")
    print("\nTotal label exposure (baseline -> experiment):")
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
            "建立 rare-class targeted crop Dataset；Experiment D 可 replay "
            "Experiment C 的 stochastic realization。"
        )
    )
    parser.add_argument(
        "--experiment",
        choices=("C", "D", "c", "d"),
        default=DEFAULT_EXPERIMENT,
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
        "--paired-reference-dir",
        type=Path,
        default=None,
        help="Experiment D 用的既有 Experiment C artifact 目錄。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="安全重建既有 experiment 產物；不修改原始 splits。",
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
            experiment=args.experiment,
            paired_reference_dir=args.paired_reference_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
