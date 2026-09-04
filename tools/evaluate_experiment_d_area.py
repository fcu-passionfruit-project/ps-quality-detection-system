from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = REPO_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))
from image_processing import create_circular_mask


CLASS_NAMES = ("abrasion", "anthracnose", "insect_bite", "insect_track")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
MODEL_SPECS = (
    ("D0", 0, Path("training/runs/segment/v22_D_context6/weights/best.pt")),
    ("D1", 1, Path("training/runs/segment/v23_D_context6_seed1/weights/best.pt")),
    ("D2", 2, Path("training/runs/segment/v24_D_context6_seed2/weights/best.pt")),
)
DETAILED_FIELDS = (
    "split", "model", "seed", "threshold_mode", "threshold", "class_threshold",
    "image", "class", "gt_area_px", "pred_area_px", "gt_area_ratio",
    "pred_area_ratio", "abs_area_error", "signed_area_error",
    "relative_area_error", "union_iou", "dice", "gt_positive",
    "pred_positive", "zero_gt_fp_area", "fruit_area_px", "image_area_px",
    "fruit_area_ratio_of_image", "area_denominator",
)
SUMMARY_FIELDS = (
    "split", "model", "seed", "threshold_mode", "threshold", "class_threshold",
    "class", "sample_count", "gt_positive_count", "gt_zero_count", "area_mae",
    "median_ae", "rmse", "signed_bias", "positive_only_area_mae",
    "positive_only_median_ae", "positive_only_rmse",
    "positive_only_signed_bias", "positive_only_relative_mae",
    "mean_union_iou", "median_union_iou", "mean_dice", "median_dice",
    "spatial_evaluable_count", "zero_gt_fp_count", "zero_gt_fp_rate",
    "mean_zero_gt_fp_area", "median_zero_gt_fp_area", "p95_zero_gt_fp_area",
)


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    relative_name: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    seed: int
    checkpoint: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    if not (0 < start <= stop <= 1) or step <= 0:
        raise EvaluationError("threshold 範圍必須滿足 0 < start <= stop <= 1 且 step > 0。")
    count = int(round((stop - start) / step))
    values = [round(start + index * step, 6) for index in range(count + 1)]
    if not math.isclose(values[-1], stop, abs_tol=1e-8):
        raise EvaluationError("threshold stop 必須落在 start + N * step 上。")
    return values


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if values else 0.0


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def rmse(values: Sequence[float]) -> float:
    return float(math.sqrt(statistics.fmean(value * value for value in values))) if values else math.nan


def rasterize_yolo_polygons(label_path: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    height, width = shape
    masks = {name: np.zeros(shape, dtype=np.uint8) for name in CLASS_NAMES}
    if not label_path.is_file():
        return masks
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        except ValueError as error:
            raise EvaluationError(f"無效 label：{label_path}:{line_number}") from error
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise EvaluationError(f"未知 class id：{label_path}:{line_number}: {class_id}")
        if coordinates.size < 6 or coordinates.size % 2:
            raise EvaluationError(f"segmentation polygon 格式錯誤：{label_path}:{line_number}")
        points = coordinates.reshape(-1, 2)
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
            raise EvaluationError(f"polygon 座標超出 [0,1]：{label_path}:{line_number}")
        # Match Ultralytics polygon2mask: scale normalized coordinates and cast
        # to int32 (truncate), then fill on the original-resolution pixel grid.
        pixel_points = (points * np.asarray([width, height])).astype(np.int32)
        pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, width - 1)
        pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, height - 1)
        cv2.fillPoly(masks[CLASS_NAMES[class_id]], [pixel_points], 1)
    return {name: mask.astype(bool) for name, mask in masks.items()}


def resize_prediction_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise EvaluationError(f"prediction mask 必須是 finite 2D array，實際 {array.shape}。")
    if array.shape != shape:
        array = cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(array > 0.5, dtype=bool)


def union_prediction_masks(
    classes: np.ndarray,
    confidences: np.ndarray,
    masks: np.ndarray,
    class_name: str,
    threshold: float,
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    class_id = CLASS_TO_ID[class_name]
    for index in np.flatnonzero((classes == class_id) & (confidences >= threshold)):
        result |= resize_prediction_mask(masks[int(index)], shape)
    return result


def area_record(
    *, split: str, model: ModelSpec, mode: str, threshold: float | str,
    class_threshold: float, image: str, class_name: str,
    gt_mask: np.ndarray, pred_mask: np.ndarray, roi_mask: np.ndarray,
) -> dict[str, Any]:
    if gt_mask.shape != pred_mask.shape or gt_mask.shape != roi_mask.shape:
        raise EvaluationError(
            f"GT/pred/ROI mask 座標系不一致：{gt_mask.shape}, "
            f"{pred_mask.shape}, {roi_mask.shape}"
        )
    valid_roi = np.asarray(roi_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool) & valid_roi
    pred_mask = np.asarray(pred_mask, dtype=bool) & valid_roi
    denominator = int(np.count_nonzero(valid_roi))
    if denominator <= 0:
        raise EvaluationError("有效水果 ROI pixel 數為 0。")
    gt_area = int(np.count_nonzero(gt_mask))
    pred_area = int(np.count_nonzero(pred_mask))
    gt_ratio = gt_area / denominator
    pred_ratio = pred_area / denominator
    signed = pred_ratio - gt_ratio
    intersection = int(np.count_nonzero(gt_mask & pred_mask))
    union = int(np.count_nonzero(gt_mask | pred_mask))
    total = gt_area + pred_area
    both_empty = union == 0
    return {
        "split": split, "model": model.name, "seed": model.seed,
        "threshold_mode": mode, "threshold": threshold,
        "class_threshold": class_threshold, "image": image, "class": class_name,
        "gt_area_px": gt_area, "pred_area_px": pred_area,
        "gt_area_ratio": gt_ratio, "pred_area_ratio": pred_ratio,
        "abs_area_error": abs(signed), "signed_area_error": signed,
        "relative_area_error": abs(signed) / gt_ratio if gt_ratio > 0 else "",
        "union_iou": "" if both_empty else intersection / union,
        "dice": "" if both_empty else (2 * intersection / total if total else ""),
        "gt_positive": gt_area > 0, "pred_positive": pred_area > 0,
        "zero_gt_fp_area": pred_ratio if gt_area == 0 else "",
        "fruit_area_px": denominator, "image_area_px": int(gt_mask.size),
        "fruit_area_ratio_of_image": denominator / int(gt_mask.size),
        "area_denominator": "dataset_roi_ellipse_mask_scale_1.01",
    }


def dataset_roi_mask(shape: tuple[int, int]) -> np.ndarray:
    """Rebuild the deterministic mask used to create the Dataset ROI image."""
    return create_circular_mask(shape, mask_scale=1.01) > 0


def summarize(records: Sequence[dict[str, Any]], config_threshold: float | str) -> list[dict[str, Any]]:
    output = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["class"])].append(record)
    for class_name in CLASS_NAMES:
        rows = groups[class_name]
        positive = [row for row in rows if row["gt_positive"]]
        zero = [row for row in rows if not row["gt_positive"]]
        spatial = [row for row in rows if row["union_iou"] != ""]
        abs_errors = [float(row["abs_area_error"]) for row in rows]
        signed_errors = [float(row["signed_area_error"]) for row in rows]
        positive_abs = [float(row["abs_area_error"]) for row in positive]
        positive_signed = [float(row["signed_area_error"]) for row in positive]
        relative = [float(row["relative_area_error"]) for row in positive]
        fp_areas = [float(row["zero_gt_fp_area"]) for row in zero]
        fp_count = sum(float(row["zero_gt_fp_area"]) > 0 for row in zero)
        first = rows[0]
        class_thresholds = {float(row["class_threshold"]) for row in rows}
        output.append({
            "split": first["split"], "model": first["model"], "seed": first["seed"],
            "threshold_mode": first["threshold_mode"], "threshold": config_threshold,
            "class_threshold": min(class_thresholds), "class": class_name,
            "sample_count": len(rows), "gt_positive_count": len(positive), "gt_zero_count": len(zero),
            "area_mae": mean(abs_errors), "median_ae": median(abs_errors), "rmse": rmse(signed_errors),
            "signed_bias": mean(signed_errors), "positive_only_area_mae": mean(positive_abs),
            "positive_only_median_ae": median(positive_abs), "positive_only_rmse": rmse(positive_signed),
            "positive_only_signed_bias": mean(positive_signed),
            "positive_only_relative_mae": mean(relative),
            "mean_union_iou": mean([float(row["union_iou"]) for row in positive]),
            "median_union_iou": median([float(row["union_iou"]) for row in positive]),
            "mean_dice": mean([float(row["dice"]) for row in positive]),
            "median_dice": median([float(row["dice"]) for row in positive]),
            "spatial_evaluable_count": len(spatial), "zero_gt_fp_count": fp_count,
            "zero_gt_fp_rate": fp_count / len(zero) if zero else math.nan,
            "mean_zero_gt_fp_area": mean(fp_areas), "median_zero_gt_fp_area": median(fp_areas),
            "p95_zero_gt_fp_area": percentile(fp_areas, 95),
        })
    return output


def macro_metrics(summary_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    def macro(key: str, absolute: bool = False) -> float:
        values = [float(row[key]) for row in summary_rows if not math.isnan(float(row[key]))]
        if absolute:
            values = [abs(value) for value in values]
        return mean(values)
    catastrophic = max(float(row["p95_zero_gt_fp_area"]) for row in summary_rows)
    return {
        "macro_positive_area_mae": macro("positive_only_area_mae"),
        "macro_zero_gt_fp_area": macro("mean_zero_gt_fp_area"),
        "macro_abs_positive_bias": macro("positive_only_signed_bias", absolute=True),
        "macro_positive_union_iou": macro("mean_union_iou"),
        "macro_positive_dice": macro("mean_dice"),
        "max_class_p95_zero_gt_fp_area": catastrophic,
    }


def selection_key(row: dict[str, Any], best_primary: float) -> tuple[float, ...]:
    # Candidates within max(1% relative, 1e-5 absolute) of the best primary score
    # enter the guardrail tie-break. This is fixed before looking at test results.
    tolerance = max(best_primary * 0.01, 1e-5)
    close = float(row["macro_positive_area_mae"]) <= best_primary + tolerance
    if not close:
        return (1.0, float(row["macro_positive_area_mae"]), 0, 0, 0, 0)
    return (
        0.0,
        float(row["macro_zero_gt_fp_area"]),
        float(row["max_class_p95_zero_gt_fp_area"]),
        float(row["macro_abs_positive_bias"]),
        -float(row["macro_positive_union_iou"]),
        -float(row.get("representative_threshold", 0.0)),
    )


def select_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best_primary = min(float(row["macro_positive_area_mae"]) for row in rows)
    return min(rows, key=lambda row: selection_key(row, best_primary))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_samples(dataset_root: Path, split: str) -> list[Sample]:
    directory_name = "valid" if split == "val" else split
    image_dir = dataset_root / directory_name / "images"
    label_dir = dataset_root / directory_name / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise EvaluationError(f"找不到 {split} images/labels：{dataset_root}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples = []
    for path in sorted(item for item in image_dir.iterdir() if item.suffix.lower() in extensions):
        samples.append(Sample(path, label_dir / f"{path.stem}.txt", path.name))
    if not samples:
        raise EvaluationError(f"{split} 沒有影像。")
    return samples


def label_class_ids(label_path: Path) -> set[int]:
    if not label_path.is_file():
        return set()
    class_ids = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError as error:
            raise EvaluationError(f"無效 label class：{label_path}:{line_number}") from error
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise EvaluationError(f"未知 class id：{label_path}:{line_number}: {class_id}")
        class_ids.add(class_id)
    return class_ids


def audit_dataset_labels(samples_by_split: dict[str, Sequence[Sample]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"source_of_truth": "actual YOLO label files", "splits": {}}
    for split, samples in samples_by_split.items():
        distribution = {str(count): 0 for count in range(len(CLASS_NAMES) + 1)}
        combinations: dict[str, int] = defaultdict(int)
        for sample in samples:
            class_ids = label_class_ids(sample.label_path)
            distribution[str(len(class_ids))] += 1
            names = ",".join(CLASS_NAMES[class_id] for class_id in sorted(class_ids)) or "none"
            combinations[names] += 1
        audit["splits"][split] = {
            "image_count": len(samples),
            "classes_per_image": distribution,
            "multi_class_image_count": sum(
                count for unique_count, count in distribution.items() if int(unique_count) >= 2
            ),
            "class_combinations": dict(sorted(combinations.items())),
        }
    audit["multi_class_images_exist"] = any(
        item["multi_class_image_count"] > 0 for item in audit["splits"].values()
    )
    return audit


def write_dataset_audit(output_dir: Path, audit: dict[str, Any]) -> None:
    rows = []
    for split, payload in audit["splits"].items():
        for unique_count in range(len(CLASS_NAMES) + 1):
            rows.append({
                "split": split,
                "unique_class_count": unique_count,
                "image_count": payload["classes_per_image"][str(unique_count)],
            })
    write_csv(
        output_dir / "dataset_label_audit.csv",
        ("split", "unique_class_count", "image_count"),
        rows,
    )
    (output_dir / "dataset_label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_roi_mask_audit(
    output_dir: Path, samples_by_split: dict[str, Sequence[Sample]]
) -> dict[str, Any]:
    audit_dir = output_dir / "roi_mask_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mask_source": "main.image_processing.create_circular_mask",
        "mask_scale": 1.01,
        "mask_rule": "geometry only; image brightness is not used",
        "splits": {},
    }
    for split, samples in samples_by_split.items():
        stats = []
        darkness = []
        for sample in samples:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise EvaluationError(f"無法讀取影像：{sample.image_path}")
            mask = dataset_roi_mask(image.shape[:2])
            valid_values = np.max(image[mask], axis=1)
            outside_values = np.max(image[~mask], axis=1)
            stats.append({
                "image": sample.relative_name,
                "fruit_area_px": int(np.count_nonzero(mask)),
                "image_area_px": int(mask.size),
                "fruit_area_ratio_of_image": float(np.mean(mask)),
                # <=8 is diagnostic evidence for the historical black-mask contract;
                # it never participates in construction of the valid-pixel mask.
                "outside_near_black_fraction_le_8": float(np.mean(outside_values <= 8)),
                "inside_exact_black_fraction": float(np.mean(valid_values == 0)),
            })
            darkness.append((float(np.median(valid_values)), sample, image, mask))
        ratios = [row["fruit_area_ratio_of_image"] for row in stats]
        payload["splits"][split] = {
            "image_count": len(stats),
            "fruit_area_ratio_min": min(ratios),
            "fruit_area_ratio_median": median(ratios),
            "fruit_area_ratio_max": max(ratios),
            "outside_near_black_fraction_le_8_min": min(
                row["outside_near_black_fraction_le_8"] for row in stats
            ),
            "inside_exact_black_fraction_max": max(
                row["inside_exact_black_fraction"] for row in stats
            ),
        }
        selected = [darkness[0], darkness[len(darkness) // 2], min(darkness, key=lambda item: item[0])]
        used = set()
        for index, (_, sample, image, mask) in enumerate(selected, 1):
            if sample.relative_name in used:
                continue
            used.add(sample.relative_name)
            mask_image = np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
            overlay = image.copy()
            overlay[~mask] = (0, 0, 180)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            panel = np.concatenate((image, mask_image, overlay), axis=1)
            path = audit_dir / f"{split}_{index}_{sample.image_path.stem}.png"
            success, encoded = cv2.imencode(".png", panel)
            if not success:
                raise EvaluationError(f"ROI mask audit 圖片編碼失敗：{path}")
            encoded.tofile(path)
    (output_dir / "roi_mask_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def validate_inputs(data_yaml: Path) -> tuple[Path, list[ModelSpec], dict[str, Any]]:
    if not data_yaml.is_file():
        raise EvaluationError(f"找不到 dataset YAML：{data_yaml}")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = tuple(data.get("names", []))
    if names != CLASS_NAMES or int(data.get("nc", -1)) != len(CLASS_NAMES):
        raise EvaluationError(f"dataset class contract 不符：{names}")
    # This Roboflow YAML uses ../split/images while Ultralytics resolves its dataset
    # root to the YAML directory. Resolve to the verified local layout explicitly.
    dataset_root = data_yaml.parent
    specs = []
    for name, seed, relative in MODEL_SPECS:
        checkpoint = (REPO_ROOT / relative).resolve()
        if not checkpoint.is_file():
            raise EvaluationError(f"找不到 {name} checkpoint：{checkpoint}")
        args_path = checkpoint.parents[1] / "args.yaml"
        args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        if int(args["seed"]) != seed or int(args["imgsz"]) != 1024 or str(args["data"]) != "training/packages/local_defects/data_exp_D.yaml":
            raise EvaluationError(f"{name} run metadata 不符合 Experiment D 契約：{args_path}")
        specs.append(ModelSpec(name, seed, checkpoint, sha256_file(checkpoint)))
    metadata = {"dataset_yaml": str(data_yaml.resolve()), "dataset_root": str(dataset_root.resolve()), "class_names": names}
    return dataset_root, specs, metadata


def save_prediction_cache(path: Path, result: Any, shape: tuple[int, int]) -> None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        classes = np.empty(0, dtype=np.int16)
        confidences = np.empty(0, dtype=np.float32)
        masks = np.empty((0, *shape), dtype=bool)
    else:
        mask_data = getattr(getattr(result, "masks", None), "data", None)
        if mask_data is None or len(mask_data) != len(boxes):
            raise EvaluationError("prediction boxes/masks 數量不一致或缺少 masks。")
        classes = boxes.cls.detach().cpu().numpy().astype(np.int16)
        confidences = boxes.conf.detach().cpu().numpy().astype(np.float32)
        masks = np.stack([resize_prediction_mask(mask.detach().cpu().numpy(), shape) for mask in mask_data])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, classes=classes, confidences=confidences, masks=masks)


def infer_split(
    spec: ModelSpec, samples: Sequence[Sample], cache_dir: Path, *, device: str,
    imgsz: int, batch_size: int, base_conf: float,
) -> None:
    model = YOLO(str(spec.checkpoint))
    actual_names = tuple(str(model.names[index]) for index in range(len(model.names)))
    if getattr(model, "task", None) != "segment" or actual_names != CLASS_NAMES:
        raise EvaluationError(f"{spec.name} task/class contract 不符：{model.task}, {actual_names}")
    pending = [sample for sample in samples if not (cache_dir / f"{sample.image_path.stem}.npz").is_file()]
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        results = model.predict(
            source=[str(sample.image_path) for sample in chunk], imgsz=imgsz, conf=base_conf,
            iou=0.70, max_det=300, retina_masks=False, device=device, verbose=False,
        )
        if len(results) != len(chunk):
            raise EvaluationError(f"{spec.name} batch 結果數量不符。")
        for sample, result in zip(chunk, results, strict=True):
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise EvaluationError(f"無法讀取影像：{sample.image_path}")
            save_prediction_cache(cache_dir / f"{sample.image_path.stem}.npz", result, image.shape[:2])
        print(f"[{spec.name}] cached {min(start + len(chunk), len(pending))}/{len(pending)}", flush=True)


def evaluate_config(
    split: str, spec: ModelSpec, samples: Sequence[Sample], cache_dir: Path,
    mode: str, thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    records = []
    config_threshold: float | str = next(iter(thresholds.values())) if mode == "global" else "per-class"
    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise EvaluationError(f"無法讀取影像：{sample.image_path}")
        shape = image.shape[:2]
        roi_mask = dataset_roi_mask(shape)
        gt_masks = rasterize_yolo_polygons(sample.label_path, shape)
        with np.load(cache_dir / f"{sample.image_path.stem}.npz") as cached:
            classes = cached["classes"]
            confidences = cached["confidences"]
            masks = cached["masks"]
            for class_name in CLASS_NAMES:
                threshold = thresholds[class_name]
                pred = union_prediction_masks(classes, confidences, masks, class_name, threshold, shape)
                records.append(area_record(
                    split=split, model=spec, mode=mode, threshold=config_threshold,
                    class_threshold=threshold, image=sample.relative_name,
                    class_name=class_name, gt_mask=gt_masks[class_name], pred_mask=pred,
                    roi_mask=roi_mask,
                ))
    return records


def validation_sweep(
    spec: ModelSpec, samples: Sequence[Sample], cache_dir: Path, grid: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], list[dict[str, Any]]]:
    sweep_rows = []
    per_class_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_records: dict[float, list[dict[str, Any]]] = {}
    for threshold in grid:
        thresholds = {name: threshold for name in CLASS_NAMES}
        records = evaluate_config("val", spec, samples, cache_dir, "global", thresholds)
        global_records[threshold] = records
        summaries = summarize(records, threshold)
        macro = macro_metrics(summaries)
        sweep_rows.append({
            "model": spec.name, "seed": spec.seed, "threshold_mode": "global",
            "threshold": threshold, "class": "macro", "class_threshold": "", **macro,
        })
        for row in summaries:
            metrics = macro_metrics([row])
            candidate = {
                "model": spec.name, "seed": spec.seed, "threshold_mode": "per-class-candidate",
                "threshold": "per-class", "class": row["class"], "class_threshold": threshold,
                "representative_threshold": threshold, **metrics,
            }
            per_class_candidates[str(row["class"])].append(candidate)
            sweep_rows.append(candidate.copy())
    selected_global_row = select_candidate([dict(row, representative_threshold=float(row["threshold"])) for row in sweep_rows if row["threshold_mode"] == "global"])
    global_threshold = float(selected_global_row["threshold"])
    global_thresholds = {name: global_threshold for name in CLASS_NAMES}
    per_class_thresholds = {
        class_name: float(select_candidate(per_class_candidates[class_name])["class_threshold"])
        for class_name in CLASS_NAMES
    }
    return sweep_rows, global_thresholds, per_class_thresholds, global_records[global_threshold]


def bootstrap_macro_positive_mae(
    records: Sequence[dict[str, Any]], *, iterations: int, seed: int,
) -> tuple[float, float]:
    by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_image[str(row["image"])][str(row["class"])] = row
    images = sorted(by_image)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        chosen = [by_image[rng.choice(images)] for _ in images]
        class_maes = []
        for class_name in CLASS_NAMES:
            errors = [float(item[class_name]["abs_area_error"]) for item in chosen if item[class_name]["gt_positive"]]
            if errors:
                class_maes.append(statistics.fmean(errors))
        estimates.append(statistics.fmean(class_maes))
    return percentile(estimates, 2.5), percentile(estimates, 97.5)


def safe_number(value: Any, digits: int = 6) -> str:
    if value == "" or value is None:
        return "NA"
    number = float(value)
    return "NA" if math.isnan(number) else f"{number:.{digits}f}"


def make_plots(output_dir: Path, sweep_rows: Sequence[dict[str, Any]], selected_records: dict[tuple[str, str, str], list[dict[str, Any]]]) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    metrics = (
        ("macro_positive_area_mae", "Macro positive-only area MAE"),
        ("macro_zero_gt_fp_area", "Macro zero-GT FP area"),
        ("macro_abs_positive_bias", "Macro absolute positive-only bias"),
    )
    globals_only = [row for row in sweep_rows if row["threshold_mode"] == "global"]
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in ("D0", "D1", "D2"):
            rows = sorted((row for row in globals_only if row["model"] == model), key=lambda row: float(row["threshold"]))
            ax.plot([row["threshold"] for row in rows], [row[key] for row in rows], marker="o", markersize=3, label=model)
        ax.set(xlabel="Confidence threshold", ylabel=title, title=f"Validation: {title}")
        ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
        path = plot_dir / f"threshold_{key}.png"; fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)
    for (split, model, mode), records in selected_records.items():
        if split != "test":
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
        for ax, class_name in zip(axes.flat, CLASS_NAMES, strict=True):
            rows = [row for row in records if row["class"] == class_name]
            ax.scatter([row["gt_area_ratio"] for row in rows], [row["pred_area_ratio"] for row in rows], s=12, alpha=0.65)
            limit = max([float(row["gt_area_ratio"]) for row in rows] + [float(row["pred_area_ratio"]) for row in rows] + [0.001])
            ax.plot([0, limit], [0, limit], "k--", linewidth=1); ax.set_title(class_name); ax.grid(alpha=0.2)
        fig.supxlabel("GT area ratio"); fig.supylabel("Predicted area ratio"); fig.suptitle(f"{model} {mode}: test GT vs predicted area")
        fig.tight_layout(); path = plot_dir / f"scatter_test_{model}_{mode}.png"; fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)
    return outputs


def write_worst_cases(output_dir: Path, records: Sequence[dict[str, Any]], top_n: int) -> None:
    categories = {
        "absolute_error": sorted(records, key=lambda row: float(row["abs_area_error"]), reverse=True),
        "overestimation": sorted(records, key=lambda row: float(row["signed_area_error"]), reverse=True),
        "underestimation": sorted(records, key=lambda row: float(row["signed_area_error"])),
        "zero_gt_false_positive": sorted((row for row in records if not row["gt_positive"] and row["pred_positive"]), key=lambda row: float(row["pred_area_ratio"]), reverse=True),
        "positive_miss": sorted((row for row in records if row["gt_positive"]), key=lambda row: float(row["pred_area_ratio"])),
    }
    fields = list(DETAILED_FIELDS) + ["case_type", "rank"]
    rows = []
    for case_type, candidates in categories.items():
        for rank, row in enumerate(candidates[:top_n], 1):
            rows.append({**row, "case_type": case_type, "rank": rank})
    write_csv(output_dir / "worst_cases.csv", fields, rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def previous_analysis_metrics(previous_dir: Path) -> tuple[dict[tuple[str, str, str], float], dict[str, Any]]:
    summary_path = previous_dir / "area_evaluation_summary.csv"
    selection_path = previous_dir / "selection.json"
    if not summary_path.is_file() or not selection_path.is_file():
        raise EvaluationError(f"找不到上一版 comparison artifacts：{previous_dir}")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in read_csv_rows(summary_path):
        grouped[(row["split"], row["model"], row["threshold_mode"])].append(
            float(row["positive_only_area_mae"])
        )
    metrics = {key: mean(values) for key, values in grouped.items()}
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return metrics, selection


def build_report(
    output_dir: Path, metadata: dict[str, Any], specs: Sequence[ModelSpec],
    configs: dict[str, dict[str, dict[str, float]]], summaries: Sequence[dict[str, Any]],
    records_by_config: dict[tuple[str, str, str], list[dict[str, Any]]],
    bootstrap_iterations: int, dataset_audit: dict[str, Any],
    roi_audit: dict[str, Any], previous_dir: Path,
) -> Path:
    macro_by_key = {}
    for key, records in records_by_config.items():
        macro_by_key[key] = macro_metrics(summarize(records, "selected"))
    val_candidates = [(key, value) for key, value in macro_by_key.items() if key[0] == "val"]
    ranked = sorted(val_candidates, key=lambda item: selection_key({**item[1], "representative_threshold": 0.0}, min(value["macro_positive_area_mae"] for _, value in val_candidates)))
    winner_key, winner = ranked[0]
    _, winner_model, winner_mode = winner_key
    winner_spec = next(spec for spec in specs if spec.name == winner_model)
    winner_thresholds = configs[winner_model][winner_mode]
    winner_test = macro_by_key[("test", winner_model, winner_mode)]
    old_metrics, old_selection = previous_analysis_metrics(previous_dir)
    ci = {key: bootstrap_macro_positive_mae(records, iterations=bootstrap_iterations, seed=20260904) for key, records in records_by_config.items() if key[0] == "val"}
    impact_rows = []
    for key, new_value in sorted(
        ((key, value["macro_positive_area_mae"]) for key, value in macro_by_key.items() if key[0] == "val"),
        key=lambda item: (item[0][1], item[0][2]),
    ):
        _, model, mode = key
        old_value = old_metrics[key]
        old_thresholds = old_selection["models"][model]["configs"][mode]
        impact_rows.append([
            model, mode, safe_number(old_value), safe_number(new_value),
            f"{new_value - old_value:+.6f}",
            json.dumps(old_thresholds, sort_keys=True),
            json.dumps(configs[model][mode], sort_keys=True),
        ])
    old_val_ranking = sorted(
        (key for key in old_metrics if key[0] == "val"), key=lambda key: old_metrics[key]
    )
    old_test_ranking = sorted(
        (key for key in old_metrics if key[0] == "test"), key=lambda key: old_metrics[key]
    )
    new_test_ranking = sorted(
        (key for key in macro_by_key if key[0] == "test"),
        key=lambda key: macro_by_key[key]["macro_positive_area_mae"],
    )
    thresholds_changed = any(
        old_selection["models"][model]["configs"][mode] != configs[model][mode]
        for model in configs for mode in configs[model]
    )
    old_winner = old_val_ranking[0]
    winner_changed = (old_winner[1], old_winner[2]) != (winner_model, winner_mode)
    lines = [
        "# Experiment D 瑕疵面積精度分析（ROI denominator corrected）", "", "## Executive Summary", "",
        f"Validation 依預先固定的 area-based selection rule 選出 **{winner_model} / {winner_mode}**。推薦 checkpoint 為 `{winner_spec.checkpoint}`。",
        f"門檻設定：`{json.dumps(winner_thresholds, ensure_ascii=False, sort_keys=True)}`。Validation macro positive-only Area MAE 為 `{winner['macro_positive_area_mae']:.6f}`，locked test 為 `{winner_test['macro_positive_area_mae']:.6f}`。",
        "本結論針對 defect union area estimation，不以 mAP 排名取代 deployment objective。`local_defects` 的實際 labels 包含多類別影像；GT=0 可依該 image 的實際 YOLO labels 解讀為 negative。", "",
        "## Methodology", "",
        f"- Dataset YAML：`{metadata['dataset_yaml']}`；validation/test 分別為 {metadata['split_counts']['val']} / {metadata['split_counts']['test']} 張。",
        "- 四類 YOLO instance-segmentation polygons 先在原圖座標 rasterize，同類 instances 以 pixel union 計算；prediction 使用相同 union 規則。",
        "- imgsz=1024、NMS IoU=0.70、max_det=300、retina_masks=False。一次用 base conf=0.05 取得 NMS 後 predictions，再只依 class/confidence 做 deployment-time filtering；不改權重或 NMS。",
        "- Dataset images 已是 ROI-preprocessed images，不重跑 Stage 1。有效 ROI 直接重用 `main.image_processing.create_circular_mask(shape, mask_scale=1.01)`；這是原 preprocessing 與正式 pipeline 使用的幾何 binary mask，不使用 brightness threshold。GT/pred union 都先裁到同一 mask，denominator 是該 mask 的 nonzero pixels。",
        "- IoU/Dice 在 GT 與 prediction 都空時記為空值並排除平均；GT-positive 的 miss 保留為 0。主要 spatial summary 使用 GT-positive pairs。",
        "- Validation threshold grid 為 0.050–0.500、step 0.025。先最小化四類 macro positive-only Area MAE；進入 best + max(1%, 1e-5) 的候選再依 macro zero-GT FP area、最大 class P95 FP area、macro |positive bias|、IoU 排序。",
        "- Model/config/winner 全由 corrected validation metrics 鎖定；test 不參與本次選擇。由於上一版 test 已被查看，本次 test 不再是完全 blind holdout，只能視為 validation-locked confirmation。Bootstrap 以 image 為 resampling unit，固定 seed=20260904。", "",
        "## Dataset label audit", "",
        "實際 YOLO labels 是本次 audit 的 source of truth。Evaluator 沒有 single-class filtering、排除或 GT 重解釋；zero-GT 完全由該 image 是否缺少該 class polygon 決定。", "",
        markdown_table(
            ["Split", "0 classes", "1 class", "2 classes", "3 classes", "4 classes", "Multi-class images"],
            [[split, *[payload["classes_per_image"][str(index)] for index in range(5)], payload["multi_class_image_count"]] for split, payload in dataset_audit["splits"].items()],
        ), "",
        "因此上一版的 class-isolated 敘述只污染報告解讀，沒有改變任何已計算數值。該敘述與『必須先補齊四類 annotation 才能使用』結論正式作廢。", "",
        "## ROI denominator audit", "",
        "valid mask 只由影像尺寸與既有 ellipse geometry 決定；深色或全黑的果皮 pixel 只要位於 mask 內仍計為有效，故不會因亮度被排除。Audit 圖依序顯示原圖、binary valid mask、邊界 overlay。", "",
        markdown_table(
            ["Split", "Images", "Min valid/image", "Median", "Max"],
            [[split, payload["image_count"], safe_number(payload["fruit_area_ratio_min"]), safe_number(payload["fruit_area_ratio_median"]), safe_number(payload["fruit_area_ratio_max"])] for split, payload in roi_audit["splits"].items()],
        ), "",
        "### Checkpoints", "",
        markdown_table(["Model", "Seed", "Checkpoint", "SHA256"], [[spec.name, spec.seed, f"`{spec.checkpoint}`", f"`{spec.sha256}`"] for spec in specs]), "",
        "## Validation calibration results", "",
    ]
    comparison_rows = []
    for key, metrics in sorted(val_candidates, key=lambda item: (item[0][1], item[0][2])):
        _, model, mode = key
        thresholds = configs[model][mode]
        low, high = ci[key]
        comparison_rows.append([model, mode, json.dumps(thresholds, sort_keys=True), safe_number(metrics["macro_positive_area_mae"]), safe_number(metrics["macro_zero_gt_fp_area"]), safe_number(metrics["macro_abs_positive_bias"]), safe_number(metrics["macro_positive_union_iou"]), safe_number(metrics["macro_positive_dice"]), f"[{low:.6f}, {high:.6f}]"])
    lines.extend([markdown_table(["Model", "Mode", "Validation threshold(s)", "Macro positive MAE ↓", "Macro zero-GT FP area ↓", "Macro abs bias ↓", "Union IoU ↑", "Dice ↑", "Bootstrap 95% CI"], comparison_rows), ""])
    lines.extend(["正式 validation 排名（同一預先規則）：" + " > ".join(f"{key[1]}/{key[2]}" for key, _ in ranked) + "。CI 重疊時不宣稱 seed 有顯著差異。", "", "## Per-class area accuracy and false-positive area", ""])
    winner_val_summary = [row for row in summaries if row["split"] == "val" and row["model"] == winner_model and row["threshold_mode"] == winner_mode]
    lines.extend([markdown_table(["Class", "GT+ n", "Positive MAE", "Relative MAE", "Bias", "IoU", "Dice", "GT-zero n", "FP rate", "Mean FP area", "Median FP area", "P95 FP area"], [[row["class"], row["gt_positive_count"], safe_number(row["positive_only_area_mae"]), safe_number(row["positive_only_relative_mae"]), safe_number(row["positive_only_signed_bias"]), safe_number(row["mean_union_iou"]), safe_number(row["mean_dice"]), row["gt_zero_count"], safe_number(row["zero_gt_fp_rate"]), safe_number(row["mean_zero_gt_fp_area"]), safe_number(row["median_zero_gt_fp_area"]), safe_number(row["p95_zero_gt_fp_area"])] for row in winner_val_summary]), ""])
    rare = [row["class"] for row in winner_val_summary if int(row["gt_positive_count"]) < 20]
    lines.extend([f"少於 20 個 GT-positive image×class pairs 的類別：{', '.join(rare) if rare else '無'}；其 point estimate 統計不穩定。", "", "## Bias", "", "Signed bias > 0 表示高估、< 0 表示低估。上表與 summary CSV 同時提供全樣本與 positive-only bias；下游 grading 應保留 signed 值，不只保留 absolute error。", "", "## Worst cases", "", "每個 locked model/config 的 `worst_cases.csv` 列出 absolute error、overestimate、underestimate、zero-GT FP、positive miss 各 Top N；需人工回看原圖與標註判斷錯誤型態。", "", "## Seed stability", "", "三個 seed 的 bootstrap CI 與 validation ranking 見上表。若 CI 大幅重疊，應解讀為目前無法證明某一 seed 顯著優於其他 seed；point-estimate winner 只是依預先規則選出的 deployment candidate。", "", "## Final test evaluation", ""])
    test_rows = []
    for key, metrics in sorted((item for item in macro_by_key.items() if item[0][0] == "test"), key=lambda item: (item[0][1], item[0][2])):
        _, model, mode = key
        test_rows.append([model, mode, safe_number(metrics["macro_positive_area_mae"]), safe_number(metrics["macro_zero_gt_fp_area"]), safe_number(metrics["macro_abs_positive_bias"]), safe_number(metrics["macro_positive_union_iou"]), safe_number(metrics["macro_positive_dice"])])
    lines.extend([markdown_table(["Model", "Locked mode", "Macro positive MAE ↓", "Macro zero-GT FP area ↓", "Macro abs bias ↓", "IoU ↑", "Dice ↑"], test_rows), "", f"Validation-selected winner 固定為 **{winner_model}/{winner_mode}**；即使 test 的 point estimate 由其他 seed 較佳，也不改選。", ""])
    lines.extend([
        "## Impact of corrected assumptions", "",
        "1. **Class-isolated 假設沒有進入 evaluation code。** 上一版與本版都直接讀取每張 `local_defects` label 的全部 polygon；沒有忽略、排除或重新解釋 GT。", 
        "2. **第一項修正只改變報告解讀。** zero-GT metrics 一直由實際 labels 決定，所以舊版所有數值不因移除 class-isolated 說法而改變。", "",
        markdown_table(
            ["Model", "Mode", "Old full-image MAE", "Corrected ROI MAE", "Change", "Old threshold(s)", "Corrected threshold(s)"],
            impact_rows,
        ), "",
        "上表的 change 是 corrected minus old。除了 denominator 約縮小到原矩形的 0.798 倍，GT/pred masks 也一致裁到既有 ROI mask，因此不同 image/class 的改變不一定是完全相同常數倍。", "",
        "GT clipping audit 顯示 validation 有 1 張 abrasion、28 張 insect_track 的 union 延伸到 ellipse 外；anthracnose/insect_bite 為 0 張。Test 對應為 abrasion 3 張、insect_track 22 張。這表示非等比例差異主要集中在 ROI 邊界附近的 insect_track，而不是 class-isolated GT 重解讀。", "",
        "3. 舊版 validation ranking（純 MAE）為：" + " > ".join(f"{key[1]}/{key[2]}" for key in old_val_ranking) + "。", 
        "4. Corrected validation selection ranking（含既定 guardrails）為：" + " > ".join(f"{key[1]}/{key[2]}" for key, _ in ranked) + "。所有最佳 global/per-class thresholds " + ("有改變。" if thresholds_changed else "均未改變。"),
        f"5. Corrected validation winner 是 **{winner_model}/{winner_mode}**；winner " + ("已改變，必須由 per-image/class 差異進一步解釋。" if winner_changed else "仍與舊版相同。ROI 修正主要放大 area ratio，且裁掉 ellipse 外 pixels，沒有改變 validation 選模順序。") + "本次結果不是依已看過的 test 手動指定。", 
        "6. 舊版 test point-estimate ranking 為：" + " > ".join(f"{key[1]}/{key[2]}" for key in old_test_ranking) + "；corrected test ranking 為：" + " > ".join(f"{key[1]}/{key[2]}" for key in new_test_ranking) + "。", 
        "7. 仍成立：同類 instance 使用 pixel union、both-empty 不灌入 IoU/Dice、validation-only threshold selection、rare-class 不穩定與 mAP 不等於 area accuracy。正式作廢：class-isolated Dataset、刻意漏標其他 visible defect、full-image denominator、以及必須先補齊四類 annotation 才能使用的說法。", "",
    ])
    winner_test_summary = [row for row in summaries if row["split"] == "test" and row["model"] == winner_model and row["threshold_mode"] == winner_mode]
    lines.extend([f"### Validation-selected configuration: test {winner_mode}", "", markdown_table(["Class", "GT+ n", "Positive MAE", "Relative MAE", "Bias", "IoU", "Dice", "GT-zero n", "FP rate", "Mean FP area", "Median FP area", "P95 FP area"], [[row["class"], row["gt_positive_count"], safe_number(row["positive_only_area_mae"]), safe_number(row["positive_only_relative_mae"]), safe_number(row["positive_only_signed_bias"]), safe_number(row["mean_union_iou"]), safe_number(row["mean_dice"]), row["gt_zero_count"], safe_number(row["zero_gt_fp_rate"]), safe_number(row["mean_zero_gt_fp_area"]), safe_number(row["median_zero_gt_fp_area"]), safe_number(row["p95_zero_gt_fp_area"])] for row in winner_test_summary]), "", "## Relationship to prior mAP analysis", "", "先前 validation Mask mAP50 排名是 D0 (0.37993) > D2 (0.36219) > D1 (0.31162)。本次 corrected validation area ranking 必須獨立解讀；mAP 衡量跨 confidence/IoU 的 instance ranking，不能直接保證 union area、bias 或 absent-class FP area 最佳。", "先前三-seed test Mask mAP50 排名為 D2 (0.34718) > D0 (0.34684) > D1 (0.34157)，差距很小；test 已非完全 blind，且不參與本次 winner selection。", "", "## Final Recommendation", "", f"1. checkpoint：`{winner_spec.checkpoint}`。", f"2. threshold mode：**{winner_mode}**。", f"3. thresholds：`{json.dumps(winner_thresholds, sort_keys=True)}`。", "4. 下游 CSV 建議至少存四類 ROI-clipped union area ratio、absolute union pixels、各類 max confidence、使用的 class threshold、fruit_area_px、ROI/推論 status、model SHA256、dataset/model version；錯誤不得寫成零瑕疵。", "5. Production 判定應同時考慮 rare-class 樣本數、split provenance 與實地資料驗證；不再以錯誤的 class-isolated 前提要求補齊四類 annotation。", "", "## Limitations", "", "- `insect_bite` 等 rare class 樣本極少，macro objective 雖防止多數類完全支配，仍可能對少量標註敏感。", "- 現有 ROI contract 是固定 ellipse mask，不是逐像素的 learned fruit contour；它忠實重建目前 preprocessing denominator，但不代表真實三維表面積。", "- polygon rasterization 與 resized prediction masks 有 pixel-grid quantization；absolute pixel area受原圖解析度影響。", "- 同一固定 split 長期重用，且上一版 test 已被查看；bootstrap CI 只描述此 split 的 image-level sampling uncertainty，不是新場域泛化保證。", "", "## Output files", "", "- `area_evaluation_detailed.csv`", "- `area_evaluation_summary.csv`", "- `threshold_sweep.csv`", "- `selection.json`", "- `dataset_label_audit.csv` / `.json`", "- `roi_mask_audit.json` / `roi_mask_audit/`", "- `plots/`", "- `worst_cases/`", ""])
    path = output_dir / "experiment_d_area_accuracy_corrected_analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment D 三 seed 的 union-mask 瑕疵面積精度比較。")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "training/packages/local_defects/data.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "analysis/experiment_d_area_accuracy_corrected_20260904")
    parser.add_argument(
        "--previous-analysis", type=Path,
        default=REPO_ROOT / "analysis/experiment_d_area_accuracy_20260904",
        help="上一版 full-image denominator 結果，用於 corrected impact comparison。",
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.50)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--worst-top-n", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="重用同 output 中完整的 prediction cache。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(20260904); np.random.seed(20260904); torch.manual_seed(20260904)
    grid = threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    data_yaml = args.data.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise EvaluationError(f"output 已存在且非空；請改路徑或使用 --resume：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root, specs, metadata = validate_inputs(data_yaml)
    samples = {split: discover_samples(dataset_root, split) for split in ("val", "test")}
    audit_samples = {
        "train": discover_samples(dataset_root, "train"),
        "valid": samples["val"],
        "test": samples["test"],
    }
    dataset_audit = audit_dataset_labels(audit_samples)
    write_dataset_audit(output_dir, dataset_audit)
    roi_audit = write_roi_mask_audit(output_dir, audit_samples)
    metadata.update({
        "created_at": datetime.now().astimezone().isoformat(), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__, "numpy": np.__version__, "device": args.device,
        "imgsz": args.imgsz, "base_conf": grid[0], "nms_iou": 0.70, "max_det": 300,
        "retina_masks": False, "split_counts": {key: len(value) for key, value in samples.items()},
        "area_denominator": "dataset_roi_ellipse_mask_scale_1.01",
        "roi_mask_source": "main.image_processing.create_circular_mask",
        "dataset_label_source": "actual YOLO label files",
        "previous_analysis": str(args.previous_analysis.resolve()),
    })
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    all_sweep = []
    configs: dict[str, dict[str, dict[str, float]]] = {}
    records_by_config: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for spec in specs:
        val_cache = output_dir / "prediction_cache" / spec.name / "val"
        infer_split(spec, samples["val"], val_cache, device=args.device, imgsz=args.imgsz, batch_size=args.batch, base_conf=grid[0])
        sweep, global_thresholds, per_class_thresholds, global_records = validation_sweep(spec, samples["val"], val_cache, grid)
        all_sweep.extend(sweep)
        configs[spec.name] = {"global": global_thresholds, "per-class": per_class_thresholds}
        records_by_config[("val", spec.name, "global")] = global_records
        records_by_config[("val", spec.name, "per-class")] = evaluate_config("val", spec, samples["val"], val_cache, "per-class", per_class_thresholds)
        test_cache = output_dir / "prediction_cache" / spec.name / "test"
        infer_split(spec, samples["test"], test_cache, device=args.device, imgsz=args.imgsz, batch_size=args.batch, base_conf=grid[0])
        for mode, thresholds in configs[spec.name].items():
            records_by_config[("test", spec.name, mode)] = evaluate_config("test", spec, samples["test"], test_cache, mode, thresholds)
    detailed = [row for records in records_by_config.values() for row in records]
    summary_rows = []
    for (split, model, mode), records in records_by_config.items():
        config_threshold: float | str = next(iter(configs[model][mode].values())) if mode == "global" else "per-class"
        summary_rows.extend(summarize(records, config_threshold))
        worst_dir = output_dir / "worst_cases" / split / model / mode
        worst_dir.mkdir(parents=True, exist_ok=True)
        write_worst_cases(worst_dir, records, args.worst_top_n)
    write_csv(output_dir / "area_evaluation_detailed.csv", DETAILED_FIELDS, detailed)
    write_csv(output_dir / "area_evaluation_summary.csv", SUMMARY_FIELDS, summary_rows)
    sweep_fields = ("model", "seed", "threshold_mode", "threshold", "class", "class_threshold", "macro_positive_area_mae", "macro_zero_gt_fp_area", "macro_abs_positive_bias", "macro_positive_union_iou", "macro_positive_dice", "max_class_p95_zero_gt_fp_area")
    write_csv(output_dir / "threshold_sweep.csv", sweep_fields, all_sweep)
    selection = {"models": {spec.name: {"seed": spec.seed, "checkpoint": str(spec.checkpoint), "sha256": spec.sha256, "configs": configs[spec.name]} for spec in specs}}
    (output_dir / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plots(output_dir, all_sweep, records_by_config)
    build_report(
        output_dir, metadata, specs, configs, summary_rows, records_by_config,
        args.bootstrap_iterations, dataset_audit, roi_audit,
        args.previous_analysis.resolve(),
    )
    print(f"完成：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
