from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PARAMETERS = {
    "confidence": 0.30,
    "shrink_ratio": 0.00,
    "mask_scale": 1.01,
    "clahe_clip_limit": 2.0,
    "clahe_tile_size": 8,
}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到圖片：{path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"無法讀取圖片：{path}")
    return image


def write_image(path: str | Path, image: np.ndarray, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"輸出檔案已存在：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise ValueError(f"不支援的輸出格式：{extension}")
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"圖片編碼失敗：{path}")
    encoded.tofile(path)
    return path


def validate_parameters(
    confidence: float,
    shrink_ratio: float,
    mask_scale: float,
    clahe_clip_limit: float,
    clahe_tile_size: int,
) -> None:
    if not 0.0 < confidence <= 1.0:
        raise ValueError("ROI confidence 必須大於 0 且不超過 1。")
    if not 0.0 <= shrink_ratio < 0.5:
        raise ValueError("邊界內縮比例必須介於 0（含）與 0.5（不含）之間。")
    if not 0.1 <= mask_scale <= 2.0:
        raise ValueError("圓形遮罩倍率必須介於 0.1 與 2.0 之間。")
    if not 0.1 <= clahe_clip_limit <= 10.0:
        raise ValueError("CLAHE 對比限制必須介於 0.1 與 10.0 之間。")
    if not 2 <= int(clahe_tile_size) <= 32:
        raise ValueError("CLAHE 網格大小必須介於 2 與 32 之間。")


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def select_best_detection(result) -> dict | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    confidences = _to_numpy(boxes.conf).reshape(-1)
    index = int(np.argmax(confidences))
    xyxy = _to_numpy(boxes.xyxy[index]).reshape(-1)
    if xyxy.size != 4:
        raise ValueError("ROI 模型回傳的 bbox 格式不正確。")
    return {
        "index": index,
        "bbox": [float(value) for value in xyxy],
        "confidence": float(confidences[index]),
    }


def clamp_and_shrink_bbox(
    bbox: list[float], image_shape: tuple[int, ...], shrink_ratio: float = 0.0
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    if shrink_ratio:
        pad_x = (x2 - x1) * shrink_ratio
        pad_y = (y2 - y1) * shrink_ratio
        x1, x2 = x1 + pad_x, x2 - pad_x
        y1, y2 = y1 + pad_y, y2 - pad_y
    coordinates = tuple(int(round(value)) for value in (x1, y1, x2, y2))
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise ValueError("ROI bbox 經邊界修正後沒有有效面積。")
    return coordinates


def create_circular_mask(image_shape: tuple[int, ...], mask_scale: float = 1.01) -> np.ndarray:
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("無法為空影像建立遮罩。")
    mask = np.zeros((height, width), dtype=np.uint8)
    radius_x = max(1, int((width / 2) * mask_scale))
    radius_y = max(1, int((height / 2) * mask_scale))
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (radius_x, radius_y),
        0,
        0,
        360,
        255,
        -1,
    )
    return mask


def apply_circular_mask(crop_image: np.ndarray, mask_scale: float = 1.01) -> np.ndarray:
    mask = create_circular_mask(crop_image.shape, mask_scale)
    return cv2.bitwise_and(crop_image, crop_image, mask=mask)


def process_grayscale_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8
) -> np.ndarray:
    if image.size == 0:
        raise ValueError("無法處理空影像。")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_size), int(tile_size)),
    )
    return clahe.apply(gray)


def extract_masked_roi(
    image: np.ndarray,
    detection: dict,
    *,
    shrink_ratio: float = 0.0,
    mask_scale: float = 1.01,
) -> tuple[np.ndarray, np.ndarray, dict]:
    detected_bbox = detection["bbox"]
    x1, y1, x2, y2 = clamp_and_shrink_bbox(
        detected_bbox, image.shape, shrink_ratio
    )
    crop = image[y1:y2, x1:x2]
    mask = create_circular_mask(crop.shape, mask_scale)
    masked = cv2.bitwise_and(crop, crop, mask=mask)
    height, width = image.shape[:2]
    dx1, dy1, dx2, dy2 = detected_bbox
    metadata = {
        "bbox": [round(value, 2) for value in detected_bbox],
        "crop_bbox": [x1, y1, x2, y2],
        "confidence": round(float(detection["confidence"]), 6),
        "roi_area_ratio": round(
            max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1) / (width * height),
            8,
        ),
    }
    return masked, mask, metadata


def annotate_roi(image: np.ndarray, roi: dict) -> np.ndarray:
    annotated = image.copy()
    x1, y1, x2, y2 = (int(round(value)) for value in roi["bbox"])
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 3)
    cv2.putText(
        annotated,
        f"PF {roi['confidence']:.1%}",
        (max(0, x1), max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _resolve_batch_paths(
    input_dir: str | Path, output_dir: str | Path
) -> tuple[Path, Path, list[Path]]:
    source_root = Path(input_dir).resolve()
    destination_root = Path(output_dir).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"輸入資料夾不存在：{source_root}")
    if source_root == destination_root or source_root in destination_root.parents:
        raise ValueError("輸出資料夾不得等於或位於輸入資料夾內。")
    sources = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    collisions = [
        destination_root / source.relative_to(source_root)
        for source in sources
        if (destination_root / source.relative_to(source_root)).exists()
    ]
    report_path = destination_root / "batch_report.json"
    if report_path.exists():
        collisions.append(report_path)
    if collisions:
        preview = "\n".join(str(path) for path in collisions[:5])
        raise FileExistsError(f"輸出資料夾已有同名檔案，未開始處理：\n{preview}")
    return source_root, destination_root, sources


def batch_preprocess(
    detect_roi: Callable[[np.ndarray, float], object],
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    mode: str = "color",
    confidence: float = 0.30,
    shrink_ratio: float = 0.00,
    mask_scale: float = 1.01,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    validate_parameters(
        confidence,
        shrink_ratio,
        mask_scale,
        clahe_clip_limit,
        clahe_tile_size,
    )
    if mode not in {"color", "wrinkle"}:
        raise ValueError("批次模式必須是 color 或 wrinkle。")
    source_root, destination_root, sources = _resolve_batch_paths(
        input_dir, output_dir
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    items = []

    for index, source in enumerate(sources, 1):
        relative_path = source.relative_to(source_root)
        item = {"source": str(relative_path), "status": "error", "reason": None}
        try:
            image = read_image(source)
            detection = select_best_detection(detect_roi(image, confidence))
            if detection is None:
                item.update(status="skipped", reason="roi_not_detected")
            else:
                masked, _, roi = extract_masked_roi(
                    image,
                    detection,
                    shrink_ratio=shrink_ratio,
                    mask_scale=mask_scale,
                )
                output_image = (
                    process_grayscale_clahe(
                        masked, clahe_clip_limit, clahe_tile_size
                    )
                    if mode == "wrinkle"
                    else masked
                )
                destination = destination_root / relative_path
                write_image(destination, output_image)
                item.update(
                    status="ok",
                    reason=None,
                    output=str(relative_path),
                    roi=roi,
                )
        except Exception as error:
            item["reason"] = f"{type(error).__name__}: {error}"
        items.append(item)
        if progress:
            progress(index, len(sources), item["status"])

    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("ok", "skipped", "error")
    }
    report = {
        "ok": counts["error"] == 0,
        "mode": mode,
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "input_dir": str(source_root),
        "output_dir": str(destination_root),
        "parameters": {
            "confidence": confidence,
            "shrink_ratio": shrink_ratio,
            "mask_scale": mask_scale,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_size": int(clahe_tile_size),
        },
        "counts": counts,
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        "items": items,
    }
    (destination_root / "batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
