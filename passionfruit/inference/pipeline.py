from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from passionfruit.inference.image_processing import (
    DEFAULT_PARAMETERS,
    annotate_roi,
    extract_masked_roi,
    process_grayscale_clahe,
    read_image,
    select_best_detection,
    validate_parameters,
    write_image,
)
from passionfruit.inference.model_manager import PassionFruitModelManager


ROLE_LABELS = {
    "roi": "ROI",
    "color": "顏色",
    "wrinkle": "皺褶",
    "defect": "局部瑕疵",
}

DEFECT_CLASSES = (
    "abrasion",
    "anthracnose",
    "insect_bite",
    "insect_track",
)


def _error_reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _class_name(names, index: int) -> str:
    return str(names[index] if isinstance(names, dict) else names[index])


def _classification_summary(result, elapsed_ms: float) -> dict:
    probabilities = getattr(result, "probs", None)
    if probabilities is None:
        raise ValueError("分類模型沒有回傳 probabilities。")
    index = int(probabilities.top1)
    confidence = float(probabilities.top1conf.item())
    return {
        "status": "ok",
        "class_name": _class_name(result.names, index),
        "class_index": index,
        "confidence": round(confidence, 6),
        "elapsed_ms": round(elapsed_ms, 2),
        "reason": None,
    }


def _segmentation_summary(
    result, elapsed_ms: float, roi_mask: np.ndarray | None
) -> dict:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    detections = []
    area_ratios = {class_name: 0.0 for class_name in DEFECT_CLASSES}
    max_confidences = {class_name: 0.0 for class_name in DEFECT_CLASSES}
    if boxes is not None and len(boxes):
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        if masks is None or getattr(masks, "data", None) is None:
            raise ValueError("局部瑕疵模型有偵測結果，但沒有回傳 segmentation masks。")
        mask_data = masks.data.detach().cpu().numpy()
        if len(mask_data) != len(classes):
            raise ValueError("局部瑕疵模型回傳的 boxes 與 masks 數量不一致。")
        denominator = (
            int(np.count_nonzero(roi_mask))
            if roi_mask is not None
            else int(result.orig_shape[0] * result.orig_shape[1])
        )
        if denominator <= 0:
            raise ValueError("果實 ROI 面積為 0，無法計算局部瑕疵比例。")
        target_shape = (
            roi_mask.shape if roi_mask is not None else tuple(result.orig_shape)
        )
        union_masks = {
            class_name: np.zeros(target_shape, dtype=bool)
            for class_name in DEFECT_CLASSES
        }
        class_indexes = {}
        for index, (class_index, confidence) in enumerate(
            zip(classes, confidences, strict=False)
        ):
            class_index = int(class_index)
            class_name = _class_name(result.names, class_index)
            if class_name not in union_masks:
                raise ValueError(f"未知的局部瑕疵類別：{class_name}")
            resized = cv2.resize(
                mask_data[index],
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            instance_mask = resized > 0.5
            if roi_mask is not None:
                instance_mask &= roi_mask > 0
            union_masks[class_name] |= instance_mask
            class_indexes[class_name] = class_index
            max_confidences[class_name] = max(
                max_confidences[class_name], round(float(confidence), 6)
            )

        for class_name in DEFECT_CLASSES:
            numerator = int(np.count_nonzero(union_masks[class_name]))
            area_ratio = round(min(1.0, numerator / denominator), 8)
            area_ratios[class_name] = area_ratio
            if class_name not in class_indexes:
                continue
            detections.append(
                {
                    "class_name": class_name,
                    "class_index": class_indexes[class_name],
                    "confidence": max_confidences[class_name],
                    "mask_area_ratio": area_ratio,
                }
            )
    return {
        "status": "ok",
        "detections": detections,
        "area_ratios": area_ratios,
        "max_confidences": max_confidences,
        "elapsed_ms": round(elapsed_ms, 2),
        "reason": None,
    }


def _predict(
    manager: PassionFruitModelManager,
    role: str,
    model_path: str,
    image: np.ndarray,
    *,
    confidence: float | None = None,
    roi_mask: np.ndarray | None = None,
    imgsz: int | None = None,
) -> tuple[dict, object]:
    started = perf_counter()
    result = manager.predict(
        role, model_path, image, confidence=confidence, imgsz=imgsz
    )
    elapsed_ms = (perf_counter() - started) * 1000
    if role in {"color", "wrinkle"}:
        return _classification_summary(result, elapsed_ms), result
    if role == "defect":
        return _segmentation_summary(result, elapsed_ms, roi_mask), result
    detection = select_best_detection(result)
    return (
        {
            "status": "ok" if detection else "error",
            "reason": None if detection else "roi_not_detected",
            "elapsed_ms": round(elapsed_ms, 2),
            "detection": detection,
        },
        result,
    )


def _predict_batch(
    manager: PassionFruitModelManager,
    role: str,
    model_path: str,
    images: list[np.ndarray],
    *,
    confidence: float | None = None,
    imgsz: int | None = None,
) -> tuple[list, float]:
    started = perf_counter()
    results = manager.predict_batch(
        role, model_path, images, confidence=confidence, imgsz=imgsz
    )
    elapsed_ms = (perf_counter() - started) * 1000
    return results, elapsed_ms


def _settings(parameters: dict | None) -> dict:
    values = {**DEFAULT_PARAMETERS, **(parameters or {})}
    values["clahe_tile_size"] = int(values["clahe_tile_size"])
    validate_parameters(
        values["confidence"],
        values["shrink_ratio"],
        values["mask_scale"],
        values["clahe_clip_limit"],
        values["clahe_tile_size"],
    )
    return values


def _role_option(
    model_options: dict[str, dict] | None,
    role: str,
    name: str,
    default=None,
):
    return (model_options or {}).get(role, {}).get(name, default)


def _prepare_roi(
    source_image: np.ndarray,
    roi_model_path: str,
    manager: PassionFruitModelManager,
    parameters: dict,
) -> tuple[dict, dict[str, np.ndarray]] | tuple[dict, None]:
    if not roi_model_path:
        return {"status": "error", "reason": "roi_model_missing"}, None
    try:
        summary, _ = _predict(
            manager,
            "roi",
            roi_model_path,
            source_image,
            confidence=parameters["confidence"],
        )
        detection = summary.get("detection")
        if not detection:
            return {
                "status": "error",
                "reason": summary["reason"],
                "elapsed_ms": summary["elapsed_ms"],
            }, None
        masked, mask, roi = extract_masked_roi(
            source_image,
            detection,
            shrink_ratio=parameters["shrink_ratio"],
            mask_scale=parameters["mask_scale"],
        )
        roi.update(status="ok", reason=None, elapsed_ms=summary["elapsed_ms"])
        return roi, {
            "roi_annotated": annotate_roi(source_image, roi),
            "masked_roi": masked,
            "roi_mask": mask,
        }
    except Exception as error:
        return {"status": "error", "reason": _error_reason(error)}, None


def run_pipeline(
    image_path: str | Path,
    model_paths: dict[str, str],
    *,
    parameters: dict | None = None,
    manager: PassionFruitModelManager | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    started = perf_counter()
    settings = _settings(parameters)
    manager = manager or PassionFruitModelManager()
    source = Path(image_path).resolve()
    source_image = read_image(source)
    report = {
        "ok": False,
        "source": str(source),
        "mode": "pipeline",
        "device": manager.device,
        "model_paths": {
            role: str(Path(path).resolve()) if path else None
            for role, path in model_paths.items()
        },
        "parameters": settings,
        "roi": None,
        "models": {},
    }
    artifacts = {"original": source_image}
    roi, prepared = _prepare_roi(
        source_image, model_paths.get("roi", ""), manager, settings
    )
    report["roi"] = roi
    if prepared is None:
        report["reason"] = roi["reason"]
        for role in ("color", "wrinkle", "defect"):
            report["models"][role] = {
                "status": "skipped",
                "reason": "roi_unavailable",
            }
        report["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        return report, artifacts

    artifacts.update(prepared)
    gray = process_grayscale_clahe(
        prepared["masked_roi"],
        settings["clahe_clip_limit"],
        settings["clahe_tile_size"],
    )
    artifacts["wrinkle_gray"] = gray
    inputs = {
        "color": prepared["masked_roi"],
        "wrinkle": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "defect": prepared["masked_roi"],
    }
    for role in ("color", "wrinkle", "defect"):
        model_path = model_paths.get(role, "")
        if not model_path:
            report["models"][role] = {
                "status": "skipped",
                "reason": "model_missing",
            }
            continue
        try:
            summary, result = _predict(
                manager,
                role,
                model_path,
                inputs[role],
                roi_mask=prepared["roi_mask"],
            )
            report["models"][role] = summary
            if role == "defect":
                artifacts["defect_annotated"] = result.plot()
        except Exception as error:
            report["models"][role] = {
                "status": "error",
                "reason": _error_reason(error),
            }

    report["ok"] = not any(
        item["status"] == "error" for item in report["models"].values()
    )
    report["reason"] = None if report["ok"] else "model_inference_failed"
    report["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
    return report, artifacts


def run_pipeline_batch(
    image_paths: list[str | Path],
    model_paths: dict[str, str],
    *,
    parameters: dict | None = None,
    manager: PassionFruitModelManager | None = None,
    model_options: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict[str, np.ndarray]]]:
    started = perf_counter()
    settings = _settings(parameters)
    manager = manager or PassionFruitModelManager()

    sources = [Path(path).resolve() for path in image_paths]
    source_images = [read_image(source) for source in sources]

    reports: list[dict] = [
        {
            "ok": False,
            "source": str(source),
            "mode": "pipeline_batch",
            "device": manager.device,
            "model_paths": {
                role: str(Path(path).resolve()) if path else None
                for role, path in model_paths.items()
            },
            "parameters": settings,
            "roi": None,
            "models": {},
        }
        for source in sources
    ]
    artifacts_list: list[dict[str, np.ndarray]] = [
        {"original": image} for image in source_images
    ]

    roi_model_path = model_paths.get("roi", "")
    if not roi_model_path:
        for index in range(len(sources)):
            reports[index]["roi"] = {
                "status": "error",
                "reason": "roi_model_missing",
            }
            reports[index]["reason"] = "roi_model_missing"
            for role in ("color", "wrinkle", "defect"):
                reports[index]["models"][role] = {
                    "status": "skipped",
                    "reason": "roi_unavailable",
                }
            reports[index]["elapsed_ms"] = round(
                (perf_counter() - started) * 1000, 2
            )
        return reports, artifacts_list

    try:
        roi_results, roi_elapsed_ms = _predict_batch(
            manager,
            "roi",
            roi_model_path,
            source_images,
            confidence=_role_option(
                model_options, "roi", "confidence", settings["confidence"]
            ),
            imgsz=_role_option(model_options, "roi", "imgsz"),
        )
    except Exception as error:
        reason = _error_reason(error)
        for index in range(len(sources)):
            reports[index]["roi"] = {"status": "error", "reason": reason}
            reports[index]["reason"] = reason
            for role in ("color", "wrinkle", "defect"):
                reports[index]["models"][role] = {
                    "status": "skipped",
                    "reason": "roi_unavailable",
                }
            reports[index]["elapsed_ms"] = round(
                (perf_counter() - started) * 1000, 2
            )
        return reports, artifacts_list

    per_image_roi_ms = round(roi_elapsed_ms / max(len(sources), 1), 2)

    ready_indices: list[int] = []
    masked_rois: list[np.ndarray] = []
    gray_inputs: list[np.ndarray] = []
    roi_masks: list[np.ndarray] = []

    for index, (source_image, roi_result) in enumerate(
        zip(source_images, roi_results, strict=True)
    ):
        try:
            detection = select_best_detection(roi_result)
            if not detection:
                reports[index]["roi"] = {
                    "status": "error",
                    "reason": "roi_not_detected",
                    "elapsed_ms": per_image_roi_ms,
                }
                continue
            masked, mask, roi = extract_masked_roi(
                source_image,
                detection,
                shrink_ratio=settings["shrink_ratio"],
                mask_scale=settings["mask_scale"],
            )
            roi.update(status="ok", reason=None, elapsed_ms=per_image_roi_ms)
            reports[index]["roi"] = roi
            artifacts_list[index]["roi_annotated"] = annotate_roi(
                source_image, roi
            )
            artifacts_list[index]["masked_roi"] = masked
            artifacts_list[index]["roi_mask"] = mask

            gray = process_grayscale_clahe(
                masked,
                settings["clahe_clip_limit"],
                settings["clahe_tile_size"],
            )
            artifacts_list[index]["wrinkle_gray"] = gray

            ready_indices.append(index)
            masked_rois.append(masked)
            gray_inputs.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            roi_masks.append(mask)
        except Exception as error:
            reports[index]["roi"] = {
                "status": "error",
                "reason": _error_reason(error),
            }

    for index in range(len(sources)):
        if index not in ready_indices:
            reports[index]["reason"] = reports[index]["roi"].get(
                "reason", "roi_unavailable"
            )
            for role in ("color", "wrinkle", "defect"):
                reports[index]["models"][role] = {
                    "status": "skipped",
                    "reason": "roi_unavailable",
                }

    if not ready_indices:
        for index in range(len(sources)):
            reports[index]["elapsed_ms"] = round(
                (perf_counter() - started) * 1000, 2
            )
        return reports, artifacts_list

    role_inputs = {
        "color": masked_rois,
        "wrinkle": gray_inputs,
        "defect": masked_rois,
    }

    for role in ("color", "wrinkle", "defect"):
        model_path = model_paths.get(role, "")
        if not model_path:
            for index in ready_indices:
                reports[index]["models"][role] = {
                    "status": "skipped",
                    "reason": "model_missing",
                }
            continue

        try:
            batch_results, batch_elapsed_ms = _predict_batch(
                manager,
                role,
                model_path,
                role_inputs[role],
                confidence=_role_option(model_options, role, "confidence"),
                imgsz=_role_option(model_options, role, "imgsz"),
            )
        except Exception as error:
            reason = _error_reason(error)
            for index in ready_indices:
                reports[index]["models"][role] = {
                    "status": "error",
                    "reason": reason,
                }
            continue

        per_image_ms = round(batch_elapsed_ms / max(len(ready_indices), 1), 2)
        for position, index in enumerate(ready_indices):
            try:
                result = batch_results[position]
                if role in {"color", "wrinkle"}:
                    summary = _classification_summary(result, per_image_ms)
                else:
                    summary = _segmentation_summary(
                        result, per_image_ms, roi_masks[position]
                    )
                    artifacts_list[index]["defect_annotated"] = result.plot()
                reports[index]["models"][role] = summary
            except Exception as error:
                reports[index]["models"][role] = {
                    "status": "error",
                    "reason": _error_reason(error),
                }

    for index, report in enumerate(reports):
        model_statuses = report["models"].values()
        has_error = any(item["status"] == "error" for item in model_statuses)
        roi_ok = report["roi"].get("status") == "ok"
        report["ok"] = roi_ok and not has_error
        if not report["ok"] and report.get("reason") is None:
            report["reason"] = "model_inference_failed"
        report["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)

    return reports, artifacts_list


def run_single_model(
    image_path: str | Path,
    role: str,
    model_path: str,
    *,
    input_mode: str = "ready",
    auxiliary_roi_path: str = "",
    parameters: dict | None = None,
    manager: PassionFruitModelManager | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    if role not in ROLE_LABELS:
        raise ValueError(f"未知的模型角色：{role}")
    if input_mode not in {"ready", "raw"}:
        raise ValueError("單模型輸入模式必須是 ready 或 raw。")
    started = perf_counter()
    settings = _settings(parameters)
    manager = manager or PassionFruitModelManager()
    source = Path(image_path).resolve()
    source_image = read_image(source)
    artifacts = {"original": source_image}
    report = {
        "ok": False,
        "source": str(source),
        "mode": "single_model",
        "input_mode": input_mode,
        "role": role,
        "device": manager.device,
        "model_path": str(Path(model_path).resolve()),
        "auxiliary_roi_path": (
            str(Path(auxiliary_roi_path).resolve())
            if auxiliary_roi_path
            else None
        ),
        "parameters": settings,
        "roi": None,
        "model": None,
    }

    model_input = source_image
    roi_mask = None
    if input_mode == "raw" and role != "roi":
        roi, prepared = _prepare_roi(
            source_image, auxiliary_roi_path, manager, settings
        )
        report["roi"] = roi
        if prepared is None:
            report["reason"] = roi["reason"]
            report["model"] = {
                "status": "skipped",
                "reason": "roi_unavailable",
            }
            report["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
            return report, artifacts
        artifacts.update(prepared)
        roi_mask = prepared["roi_mask"]
        model_input = prepared["masked_roi"]
        if role == "wrinkle":
            gray = process_grayscale_clahe(
                model_input,
                settings["clahe_clip_limit"],
                settings["clahe_tile_size"],
            )
            artifacts["wrinkle_gray"] = gray
            model_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    try:
        summary, result = _predict(
            manager,
            role,
            model_path,
            model_input,
            confidence=settings["confidence"] if role == "roi" else None,
            roi_mask=roi_mask,
        )
        if role == "roi":
            detection = summary.pop("detection", None)
            if detection:
                roi = {
                    **detection,
                    "status": "ok",
                    "reason": None,
                    "elapsed_ms": summary["elapsed_ms"],
                }
                report["roi"] = roi
                artifacts["roi_annotated"] = annotate_roi(source_image, roi)
            report["model"] = summary
        else:
            report["model"] = summary
            if role == "defect":
                artifacts["defect_annotated"] = result.plot()
        report["ok"] = report["model"]["status"] == "ok"
        report["reason"] = report["model"].get("reason")
    except Exception as error:
        report["model"] = {"status": "error", "reason": _error_reason(error)}
        report["reason"] = "model_inference_failed"

    report["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
    return report, artifacts


def _validation_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    if not root.exists():
        root.mkdir(parents=True)
    if not root.is_dir():
        raise NotADirectoryError(f"輸出位置不是資料夾：{root}")
    return root


def _save_validation_result_to_dir(
    run_dir: Path,
    report: dict,
    artifacts: dict[str, np.ndarray],
) -> Path:
    run_dir.mkdir()
    artifact_paths = {}
    for name, image in artifacts.items():
        if name == "roi_mask":
            continue
        path = write_image(run_dir / f"{name}.jpg", image)
        artifact_paths[name] = str(path)
    saved_report = {**report, "artifacts": artifact_paths}
    (run_dir / "result.json").write_text(
        json.dumps(saved_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def save_validation_result(
    output_root: str | Path,
    report: dict,
    artifacts: dict[str, np.ndarray],
) -> Path:
    root = _validation_output_root(output_root)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / run_name
    return _save_validation_result_to_dir(run_dir, report, artifacts)


def save_validation_result_batch(
    output_root: str | Path,
    reports: list[dict],
    artifacts_list: list[dict[str, np.ndarray]],
) -> list[Path]:
    root = _validation_output_root(output_root)
    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_%f")
    batch_dir = root / batch_name
    batch_dir.mkdir()
    return [
        _save_validation_result_to_dir(
            batch_dir / f"station_{index:02d}", report, artifacts
        )
        for index, (report, artifacts) in enumerate(
            zip(reports, artifacts_list, strict=True), start=1
        )
    ]
