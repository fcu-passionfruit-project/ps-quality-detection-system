from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
import yaml
from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = {
    0: "abrasion",
    1: "anthracnose",
    2: "insect_bite",
    3: "insect_track",
}
RUNS = {
    "v19": Path("training/runs/segment/v19_defect_epoch200_pw050_cp01"),
    "v20": Path("training/runs/segment/v20_B_class_aware"),
    "v21": Path("training/runs/segment/v21_C_targeted_crop"),
}
DEFAULT_DATASET = Path("training/packages/local_defects")
DEFAULT_OUTPUT = Path("training/analysis/v19-v20-v21-targeted-crop")
DEFAULT_DATA_YAML = DEFAULT_DATASET / "data.yaml"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
METRIC_COLUMNS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)
LOSS_COLUMNS = (
    "train/box_loss",
    "train/seg_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/l1_loss",
    "train/sem_loss",
    "val/box_loss",
    "val/seg_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/l1_loss",
    "val/sem_loss",
)


@dataclass(frozen=True)
class Instance:
    class_id: int
    mask: np.ndarray
    box: np.ndarray


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_images(dataset: Path, split: str) -> list[Path]:
    images_dir = dataset / split / "images"
    images = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"找不到 {split} images：{images_dir}")
    return images


def _label_path(dataset: Path, split: str, image_path: Path) -> Path:
    return dataset / split / "labels" / f"{image_path.stem}.txt"


def _box_from_mask(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return np.zeros(4, dtype=np.float64)
    return np.array(
        [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1],
        dtype=np.float64,
    )


def _read_ground_truth(
    dataset: Path, split: str, image_path: Path
) -> list[Instance]:
    width, height = Image.open(image_path).size
    label_path = _label_path(dataset, split, image_path)
    if not label_path.exists():
        raise ValueError(f"找不到 label：{label_path}")

    instances: list[Instance] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"label 格式錯誤：{label_path}:{line_number}")
        class_id = int(fields[0])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"未知 class ID：{label_path}:{line_number}: {class_id}")
        coordinates = np.asarray([float(value) for value in fields[1:]])
        points = coordinates.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygon = np.rint(points).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], color=1)
        instances.append(
            Instance(class_id=class_id, mask=mask.astype(bool), box=_box_from_mask(mask))
        )
    return instances


def _read_predictions(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or not len(result.boxes):
        return []
    if result.masks is None:
        raise ValueError(f"segmentation prediction 沒有 masks：{result.path}")

    height, width = result.orig_shape
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    polygons = result.masks.xy
    if not (len(classes) == len(boxes) == len(polygons)):
        raise ValueError(f"prediction boxes/masks 數量不一致：{result.path}")

    predictions: list[dict[str, Any]] = []
    for class_id, confidence, box, polygon in zip(
        classes, confidences, boxes, polygons
    ):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.rint(polygon).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], color=1)
        predictions.append(
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": box.astype(np.float64),
                "mask": mask.astype(bool),
            }
        )
    return predictions


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def _match_instances(
    ground_truth: Sequence[Instance],
    predictions: Sequence[dict[str, Any]],
    kind: str,
    iou_threshold: float,
    class_aware: bool = True,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            if class_aware and gt.class_id != prediction["class_id"]:
                continue
            if kind == "mask":
                iou = _mask_iou(gt.mask, prediction["mask"])
            elif kind == "box":
                iou = _box_iou(gt.box, prediction["box"])
            else:
                raise ValueError(f"未知 matching kind：{kind}")
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches, matched_gt, matched_predictions


def _empty_class_counts() -> dict[str, dict[str, int]]:
    return {
        class_name: {"gt": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
        for class_name in CLASS_NAMES.values()
    }


def _audit_predictions(
    model: YOLO,
    dataset: Path,
    split: str,
    images: Sequence[Path],
    device: str,
    confidence: float,
    match_iou: float,
    nms_iou: float,
    imgsz: int,
    batch: int,
) -> dict[str, Any]:
    counts = {
        "mask": _empty_class_counts(),
        "box": _empty_class_counts(),
    }
    background = {
        "images": 0,
        "images_with_fp": 0,
        "total_fp_predictions": 0,
        "fp_by_class": Counter(),
        "mask_area_ratios": [],
    }
    duplicate_predictions = Counter()
    mask_size = {
        "matched": 0,
        "too_small_lt_0.5x": 0,
        "too_large_gt_2x": 0,
        "area_ratios": [],
    }
    per_image: dict[str, Any] = {}

    result_stream = model.predict(
        source=[str(path) for path in images],
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=confidence,
        iou=nms_iou,
        max_det=300,
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    result_by_path = {Path(result.path).resolve(): result for result in result_stream}
    if set(result_by_path) != set(images):
        missing = sorted(str(path) for path in set(images) - set(result_by_path))
        raise ValueError(f"prediction output 缺少 images：{missing[:5]}")

    for image_path in images:
        ground_truth = _read_ground_truth(dataset, split, image_path)
        predictions = _read_predictions(result_by_path[image_path])
        for kind in ("mask", "box"):
            class_counts = counts[kind]
            for gt in ground_truth:
                class_counts[CLASS_NAMES[gt.class_id]]["gt"] += 1
            for prediction in predictions:
                class_counts[CLASS_NAMES[prediction["class_id"]]]["predictions"] += 1

            matches, matched_gt, matched_predictions = _match_instances(
                ground_truth,
                predictions,
                kind=kind,
                iou_threshold=match_iou,
                class_aware=True,
            )
            for gt_index, _, _ in matches:
                class_counts[CLASS_NAMES[ground_truth[gt_index].class_id]]["tp"] += 1
            for gt_index, gt in enumerate(ground_truth):
                if gt_index not in matched_gt:
                    class_counts[CLASS_NAMES[gt.class_id]]["fn"] += 1
            for pred_index, prediction in enumerate(predictions):
                if pred_index not in matched_predictions:
                    class_counts[CLASS_NAMES[prediction["class_id"]]]["fp"] += 1

            if kind == "mask":
                for gt_index, pred_index, _ in matches:
                    gt_area = ground_truth[gt_index].mask.sum()
                    pred_area = predictions[pred_index]["mask"].sum()
                    area_ratio = float(pred_area / gt_area) if gt_area else 0.0
                    mask_size["matched"] += 1
                    mask_size["area_ratios"].append(area_ratio)
                    if area_ratio < 0.5:
                        mask_size["too_small_lt_0.5x"] += 1
                    if area_ratio > 2.0:
                        mask_size["too_large_gt_2x"] += 1

                for pred_index, prediction in enumerate(predictions):
                    if pred_index in matched_predictions:
                        continue
                    if any(
                        gt.class_id == prediction["class_id"]
                        and _mask_iou(gt.mask, prediction["mask"]) >= match_iou
                        for gt in ground_truth
                    ):
                        duplicate_predictions[CLASS_NAMES[prediction["class_id"]]] += 1

        mask_matches, matched_gt, matched_predictions = _match_instances(
            ground_truth,
            predictions,
            kind="mask",
            iou_threshold=match_iou,
            class_aware=True,
        )
        match_by_gt = {
            gt_index: (pred_index, iou)
            for gt_index, pred_index, iou in mask_matches
        }
        gt_records = []
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in match_by_gt:
                pred_index, iou = match_by_gt[gt_index]
                gt_records.append(
                    {
                        "gt_index": gt_index,
                        "class_id": gt.class_id,
                        "class_name": CLASS_NAMES[gt.class_id],
                        "correct": True,
                        "mask_iou": iou,
                        "confidence": predictions[pred_index]["confidence"],
                        "predicted_class": CLASS_NAMES[predictions[pred_index]["class_id"]],
                    }
                )
                continue

            best_any: tuple[float, dict[str, Any]] | None = None
            for prediction in predictions:
                iou = _mask_iou(gt.mask, prediction["mask"])
                if best_any is None or iou > best_any[0]:
                    best_any = (iou, prediction)
            record = {
                "gt_index": gt_index,
                "class_id": gt.class_id,
                "class_name": CLASS_NAMES[gt.class_id],
                "correct": False,
                "mask_iou": 0.0,
                "confidence": None,
                "predicted_class": None,
            }
            if best_any is not None:
                record.update(
                    {
                        "best_any_mask_iou": best_any[0],
                        "best_any_confidence": best_any[1]["confidence"],
                        "best_any_predicted_class": CLASS_NAMES[
                            best_any[1]["class_id"]
                        ],
                    }
                )
            gt_records.append(record)

        unmatched_predictions = [
            {
                "class_name": CLASS_NAMES[prediction["class_id"]],
                "confidence": prediction["confidence"],
                "mask_area_ratio": float(
                    prediction["mask"].sum() / prediction["mask"].size
                ),
            }
            for pred_index, prediction in enumerate(predictions)
            if pred_index not in matched_predictions
        ]
        per_image[image_path.name] = {
            "gt_count": len(ground_truth),
            "prediction_count": len(predictions),
            "mask_tp": len(mask_matches),
            "mask_fp": len(predictions) - len(matched_predictions),
            "mask_fn": len(ground_truth) - len(matched_gt),
            "ground_truth": gt_records,
            "unmatched_predictions": unmatched_predictions,
        }

        if not ground_truth:
            background["images"] += 1
            if predictions:
                background["images_with_fp"] += 1
            background["total_fp_predictions"] += len(predictions)
            for prediction in predictions:
                background["fp_by_class"][CLASS_NAMES[prediction["class_id"]]] += 1
                background["mask_area_ratios"].append(
                    float(prediction["mask"].sum() / prediction["mask"].size)
                )

    background["fp_per_background_image"] = (
        background["total_fp_predictions"] / background["images"]
        if background["images"]
        else 0.0
    )
    background["fp_image_rate"] = (
        background["images_with_fp"] / background["images"]
        if background["images"]
        else 0.0
    )
    if background["mask_area_ratios"]:
        background["median_fp_mask_area_ratio"] = statistics.median(
            background["mask_area_ratios"]
        )
    else:
        background["median_fp_mask_area_ratio"] = None
    if mask_size["area_ratios"]:
        mask_size["median_area_ratio"] = statistics.median(mask_size["area_ratios"])
    else:
        mask_size["median_area_ratio"] = None

    return {
        "settings": {
            "confidence": confidence,
            "match_iou": match_iou,
            "nms_iou": nms_iou,
            "imgsz": imgsz,
            "batch": batch,
        },
        "counts": counts,
        "background_only_images": background,
        "duplicate_predictions": duplicate_predictions,
        "matched_mask_size": mask_size,
        "per_image": per_image,
    }


def _curve_payload(metrics: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for curve_name, curve_result in zip(metrics.curves, metrics.curves_results):
        x_values, y_values, x_label, y_label = curve_result
        x_array = np.asarray(x_values)
        y_array = np.asarray(y_values)
        macro = y_array.mean(axis=0) if y_array.ndim == 2 else y_array
        maximum_index = int(np.nanargmax(macro))
        payload[curve_name] = {
            "x_label": x_label,
            "y_label": y_label,
            "x": x_array,
            "per_class_y": y_array,
            "macro_y": macro,
            "macro_max": float(macro[maximum_index]),
            "macro_max_x": float(x_array[maximum_index]),
        }
    return payload


def _evaluate_model(
    run_id: str,
    run_path: Path,
    data_yaml: Path,
    dataset: Path,
    output: Path,
    device: str,
    imgsz: int,
    batch: int,
    audit_confidence: float,
    match_iou: float,
    nms_iou: float,
) -> dict[str, Any]:
    weight = (run_path / "weights" / "best.pt").resolve()
    if not weight.is_file():
        raise ValueError(f"找不到 best.pt：{weight}")
    model = YOLO(weight)
    result: dict[str, Any] = {
        "run": run_id,
        "run_path": run_path.resolve(),
        "best_weight": weight,
        "best_weight_sha256": _sha256(weight),
        "splits": {},
    }
    for split in ("val", "test"):
        images = _list_images(dataset, "valid" if split == "val" else "test")
        save_name = f"{run_id}_{split}"
        torch.manual_seed(0)
        np.random.seed(0)
        metrics = model.val(
            data=data_yaml.resolve(),
            split=split,
            imgsz=imgsz,
            batch=batch,
            workers=0,
            device=device,
            conf=None,
            iou=nms_iou,
            max_det=300,
            plots=True,
            save_json=False,
            verbose=True,
            project=(output / "ultralytics").resolve(),
            name=save_name,
            exist_ok=True,
        )
        summary = metrics.summary(normalize=False, decimals=10)
        per_class_results = []
        for class_id, class_name in CLASS_NAMES.items():
            class_summary = next(
                row for row in summary if row["Class"] == class_name
            )
            (
                box_precision,
                box_recall,
                box_map50,
                box_map50_95,
                mask_precision,
                mask_recall,
                mask_map50,
                mask_map50_95,
            ) = metrics.class_result(class_id)
            per_class_results.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "images": class_summary["Images"],
                    "instances": class_summary["Instances"],
                    "box_precision": box_precision,
                    "box_recall": box_recall,
                    "box_map50": box_map50,
                    "box_map50_95": box_map50_95,
                    "mask_precision": mask_precision,
                    "mask_recall": mask_recall,
                    "mask_map50": mask_map50,
                    "mask_map50_95": mask_map50_95,
                }
            )
        confusion = metrics.confusion_matrix.matrix
        audit = _audit_predictions(
            model=model,
            dataset=dataset,
            split="valid" if split == "val" else "test",
            images=images,
            device=device,
            confidence=audit_confidence,
            match_iou=match_iou,
            nms_iou=nms_iou,
            imgsz=imgsz,
            batch=batch,
        )
        result["splits"][split] = {
            "official_settings": {
                "curve_confidence_floor": 0.001,
                "confusion_matrix_confidence": 0.25,
                "confusion_matrix_box_iou": 0.45,
                "nms_iou": nms_iou,
                "imgsz": imgsz,
                "batch": batch,
                "workers": 0,
                "device": device,
                "max_det": 300,
            },
            "results_dict": metrics.results_dict,
            "summary": summary,
            "per_class_results": per_class_results,
            "curves": _curve_payload(metrics),
            "confusion_matrix": confusion,
            "audit": audit,
        }
    return result


def _float_row(row: dict[str, str]) -> dict[str, float]:
    return {key.strip(): float(value.strip()) for key, value in row.items()}


def _rolling_mean(values: Sequence[float], window: int) -> list[float]:
    if len(values) < window:
        return []
    return [statistics.fmean(values[index - window + 1 : index + 1]) for index in range(window - 1, len(values))]


def _training_dynamics(repo: Path, output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    figure, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=False)
    colors = {"v19": "#355C7D", "v20": "#C06C84", "v21": "#2A9D8F"}
    plotted_loss_columns = (
        "train/seg_loss",
        "val/seg_loss",
        "train/cls_loss",
        "val/cls_loss",
    )

    for run_id, relative_run in RUNS.items():
        run_path = repo / relative_run
        args = yaml.safe_load((run_path / "args.yaml").read_text(encoding="utf-8"))
        with (run_path / "results.csv").open(encoding="utf-8", newline="") as stream:
            rows = [_float_row(row) for row in csv.DictReader(stream)]
        available = set(rows[0])
        fitness = [
            row["metrics/mAP50-95(B)"] + row["metrics/mAP50-95(M)"]
            for row in rows
        ]
        best_fitness_index = int(np.argmax(fitness))
        mask_values = [row["metrics/mAP50-95(M)"] for row in rows]
        best_mask_index = int(np.argmax(mask_values))
        target_95 = max(mask_values) * 0.95
        target_99 = max(mask_values) * 0.99

        losses = {}
        for column in LOSS_COLUMNS:
            if column not in available:
                losses[column] = {"available": False}
                continue
            values = [row[column] for row in rows]
            minimum_index = int(np.argmin(values))
            loss_record: dict[str, Any] = {
                "available": True,
                "start": values[0],
                "minimum": values[minimum_index],
                "minimum_epoch": minimum_index + 1,
                "at_best_fitness_epoch": values[best_fitness_index],
                "final": values[-1],
            }
            if column.startswith("val/"):
                for window in (10, 20):
                    rolling = _rolling_mean(values, window)
                    rolling_index = int(np.argmin(rolling))
                    loss_record[f"rolling_{window}_minimum"] = rolling[rolling_index]
                    loss_record[f"rolling_{window}_minimum_end_epoch"] = (
                        rolling_index + window
                    )
            losses[column] = loss_record

        metrics = {}
        for column in METRIC_COLUMNS:
            values = [row[column] for row in rows]
            metrics[column] = {
                "at_best_fitness_epoch": values[best_fitness_index],
                "at_best_mask_epoch": values[best_mask_index],
                "final": values[-1],
                "last_20_std": statistics.pstdev(values[-20:]),
                "last_20_min": min(values[-20:]),
                "last_20_max": max(values[-20:]),
            }

        result[run_id] = {
            "run_path": run_path.resolve(),
            "args": args,
            "epochs_requested": int(args["epochs"]),
            "epochs_completed": len(rows),
            "early_stopping": len(rows) < int(args["epochs"]),
            "best_fitness_definition": "Box mAP50-95 + Mask mAP50-95",
            "best_fitness_epoch": best_fitness_index + 1,
            "best_fitness": fitness[best_fitness_index],
            "best_mask_map50_95_epoch": best_mask_index + 1,
            "best_mask_map50_95": mask_values[best_mask_index],
            "first_epoch_at_95_percent_best_mask": next(
                index + 1 for index, value in enumerate(mask_values) if value >= target_95
            ),
            "first_epoch_at_99_percent_best_mask": next(
                index + 1 for index, value in enumerate(mask_values) if value >= target_99
            ),
            "best_fitness_wall_time_seconds": rows[best_fitness_index]["time"],
            "final_wall_time_seconds": rows[-1]["time"],
            "losses": losses,
            "metrics": metrics,
            "available_columns": sorted(available),
        }

        epochs = np.arange(1, len(rows) + 1)
        axes[0, 0].plot(
            epochs,
            [row["metrics/mAP50-95(M)"] for row in rows],
            label=run_id,
            color=colors[run_id],
        )
        axes[0, 1].plot(
            epochs,
            [row["metrics/recall(M)"] for row in rows],
            label=run_id,
            color=colors[run_id],
        )
        for axis, column in zip(axes[1:].flat, plotted_loss_columns):
            axis.plot(
                epochs,
                [row[column] for row in rows],
                label=run_id,
                color=colors[run_id],
            )

    titles = (
        "Validation Mask mAP50-95",
        "Validation Mask Recall",
        "Train Segmentation Loss",
        "Validation Segmentation Loss",
        "Train Classification Loss",
        "Validation Classification Loss",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure_path = output / "figures" / "training_dynamics.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    result["figure"] = figure_path.resolve()
    return result


def _compare_ground_truth_patterns(evaluations: dict[str, Any]) -> dict[str, Any]:
    patterns: dict[str, list[dict[str, Any]]] = {
        "v19_wrong_v20_right_v21_right": [],
        "v19_wrong_v20_wrong_v21_right": [],
        "v19_right_v20_wrong_v21_right": [],
        "v19_right_v20_right_v21_wrong": [],
    }
    desired = {
        (False, True, True): "v19_wrong_v20_right_v21_right",
        (False, False, True): "v19_wrong_v20_wrong_v21_right",
        (True, False, True): "v19_right_v20_wrong_v21_right",
        (True, True, False): "v19_right_v20_right_v21_wrong",
    }
    for split in ("val", "test"):
        per_run = {
            run_id: evaluations[run_id]["splits"][split]["audit"]["per_image"]
            for run_id in RUNS
        }
        common_images = set.intersection(*(set(records) for records in per_run.values()))
        for image_name in sorted(common_images):
            gt_counts = {
                len(per_run[run_id][image_name]["ground_truth"]) for run_id in RUNS
            }
            if len(gt_counts) != 1:
                raise ValueError(f"GT record 數量不一致：{split}/{image_name}")
            for gt_index in range(gt_counts.pop()):
                records = {
                    run_id: per_run[run_id][image_name]["ground_truth"][gt_index]
                    for run_id in RUNS
                }
                correctness = tuple(records[run_id]["correct"] for run_id in RUNS)
                pattern_name = desired.get(correctness)
                if pattern_name is None:
                    continue
                patterns[pattern_name].append(
                    {
                        "split": split,
                        "image": image_name,
                        "gt_index": gt_index,
                        "class_name": records["v19"]["class_name"],
                        "models": records,
                    }
                )
    return patterns


def _comparison_figures(evaluations: dict[str, Any], output: Path) -> dict[str, Path]:
    colors = {"v19": "#355C7D", "v20": "#C06C84", "v21": "#2A9D8F"}
    figure_paths: dict[str, Path] = {}

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for column, split in enumerate(("val", "test")):
        for run_id in RUNS:
            curves = evaluations[run_id]["splits"][split]["curves"]
            pr_curve = curves["Precision-Recall(M)"]
            axes[0, column].plot(
                pr_curve["x"],
                pr_curve["macro_y"],
                label=run_id,
                color=colors[run_id],
            )
            f1_curve = curves["F1-Confidence(M)"]
            axes[1, column].plot(
                f1_curve["x"],
                f1_curve["macro_y"],
                label=(
                    f"{run_id}: {f1_curve['macro_max']:.3f} "
                    f"@ {f1_curve['macro_max_x']:.3f}"
                ),
                color=colors[run_id],
            )
        axes[0, column].set_title(f"{split}: Mask PR")
        axes[0, column].set_xlabel("Recall")
        axes[0, column].set_ylabel("Precision")
        axes[1, column].set_title(f"{split}: Mask F1-confidence")
        axes[1, column].set_xlabel("Confidence")
        axes[1, column].set_ylabel("Macro F1")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    threshold_path = output / "figures" / "mask_pr_f1_comparison.png"
    figure.savefig(threshold_path, dpi=160)
    plt.close(figure)
    figure_paths["mask_pr_f1"] = threshold_path.resolve()

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_positions = np.arange(len(RUNS))
    run_ids = list(RUNS)
    for axis, split in zip(axes, ("val", "test")):
        totals = [
            evaluations[run_id]["splits"][split]["audit"]
            ["background_only_images"]["total_fp_predictions"]
            for run_id in run_ids
        ]
        affected = [
            evaluations[run_id]["splits"][split]["audit"]
            ["background_only_images"]["images_with_fp"]
            for run_id in run_ids
        ]
        axis.bar(x_positions - 0.18, totals, 0.36, label="Total FP predictions")
        axis.bar(x_positions + 0.18, affected, 0.36, label="Images with FP")
        axis.set_xticks(x_positions, run_ids)
        axis.set_title(f"{split}: null/background images @ conf=0.25")
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    background_path = output / "figures" / "background_fp_comparison.png"
    figure.savefig(background_path, dpi=160)
    plt.close(figure)
    figure_paths["background_fp"] = background_path.resolve()
    return figure_paths


def _dataset_verification(repo: Path) -> dict[str, Any]:
    dataset = repo / DEFAULT_DATASET
    yaml_paths = {
        "v19": dataset / "data.yaml",
        "v20": dataset / "data_exp_B.yaml",
        "v21": dataset / "data_exp_C.yaml",
    }
    result: dict[str, Any] = {"yamls": {}, "splits": {}}
    for run_id, yaml_path in yaml_paths.items():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        result["yamls"][run_id] = {
            "path": yaml_path.resolve(),
            "sha256": _sha256(yaml_path),
            "data": data,
        }
    for split in ("valid", "test"):
        images = _list_images(dataset, split)
        labels = [_label_path(dataset, split, image_path) for image_path in images]
        result["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "image_names_sha256": hashlib.sha256(
                "\n".join(path.name for path in images).encode("utf-8")
            ).hexdigest(),
            "image_content_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(_sha256(path)) for path in images)
            ).hexdigest(),
            "label_content_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(_sha256(path)) for path in labels)
            ).hexdigest(),
            "class_instances": Counter(
                CLASS_NAMES[instance.class_id]
                for image_path in images
                for instance in _read_ground_truth(dataset, split, image_path)
            ),
            "background_images": sum(
                not _label_path(dataset, split, image_path)
                .read_text(encoding="utf-8")
                .strip()
                for image_path in images
            ),
        }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只讀分析 v19 baseline、v20 class-aware sampling 與 "
            "v21 rare-class targeted crop 的既有 best.pt。"
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="evaluation batch；Ultralytics trainer 對 train batch=4 使用 val batch=8。",
    )
    parser.add_argument("--audit-confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    output = args.output
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("目前環境無法使用 MPS；請在可存取 Apple GPU 的 shell 執行。")
    dataset = (repo / DEFAULT_DATASET).resolve()
    data_yaml = (repo / DEFAULT_DATA_YAML).resolve()

    training = _training_dynamics(repo, output)
    dataset_verification = _dataset_verification(repo)
    evaluations = {
        run_id: _evaluate_model(
            run_id=run_id,
            run_path=(repo / relative_run).resolve(),
            data_yaml=data_yaml,
            dataset=dataset,
            output=output,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            audit_confidence=args.audit_confidence,
            match_iou=args.match_iou,
            nms_iou=args.nms_iou,
        )
        for run_id, relative_run in RUNS.items()
    }
    patterns = _compare_ground_truth_patterns(evaluations)
    comparison_figures = _comparison_figures(evaluations, output)
    payload = {
        "metadata": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": args.device,
            "class_names": CLASS_NAMES,
        },
        "dataset_verification": dataset_verification,
        "training_dynamics": training,
        "evaluations": evaluations,
        "representative_patterns": patterns,
        "comparison_figures": comparison_figures,
    }
    _write_json(output / "analysis.json", payload)
    print(f"分析完成：{output / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
