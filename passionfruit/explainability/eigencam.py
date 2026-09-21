from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_IMGSZ = 1024
DEFAULT_LAYER_INDEX = -3
DEFAULT_CONFIDENCE = 0.25
OCCLUSION_FRACTION = 0.10
OCCLUSION_BGR = (114, 114, 114)
OUTPUT_FILENAMES = {
    "original": "original.jpg",
    "grayscale": "cam_grayscale.png",
    "heatmap": "cam_heatmap.png",
    "overlay": "cam_overlay.png",
    "prediction": "prediction.jpg",
    "comparison": "comparison.png",
    "normalized": "cam_normalized.npy",
    "signed_raw": "cam_signed_raw.npy",
    "positive": "cam_positive.png",
    "negative": "cam_negative.png",
    "absolute": "cam_absolute.png",
    "gray": "cam_gray.png",
    "jet": "cam_jet.png",
    "turbo": "cam_turbo.png",
    "argmax": "cam_argmax_location.png",
    "primary_mask": "primary_prediction_mask.png",
    "occlusion_high": "occlusion_high_cam.jpg",
    "occlusion_low": "occlusion_low_cam.jpg",
    "occlusion_comparison": "occlusion_comparison.png",
    "metrics": "cam_metrics.json",
}


class CamTestError(RuntimeError):
    """An actionable error raised by the EigenCAM test workflow."""


@dataclass(frozen=True)
class PrimaryPrediction:
    class_index: int
    class_name: str
    confidence: float
    mask: np.ndarray


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的整數。")
    return parsed


def confidence_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("必須介於 0 和 1 之間。")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "為 YOLO26 segmentation 模型產生 EigenCAM spatial activation／saliency response，"
            "並執行 SVD polarity 與 occlusion 診斷。EigenCAM 不是 transformer attention，"
            "也不是 class-specific Grad-CAM。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="YOLO26 segmentation .pt 權重"
    )
    parser.add_argument("--image", required=True, type=Path, help="單張測試圖片")
    parser.add_argument(
        "--output", required=True, type=Path, help="CAM 與 audit artifacts 的輸出資料夾"
    )
    parser.add_argument(
        "--imgsz", type=positive_int, default=DEFAULT_IMGSZ, help="Ultralytics 推論尺寸"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help="EigenCAM 運算裝置；MPS 失敗時不會自動回退 CPU",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=DEFAULT_LAYER_INDEX,
        help="model.model.model 的 target layer index，可使用負數索引",
    )
    parser.add_argument(
        "--conf",
        type=confidence_value,
        default=DEFAULT_CONFIDENCE,
        help="YOLO prediction confidence threshold",
    )
    return parser


def load_cam_dependencies():
    """Load the bundled CAM implementation without external checkouts."""
    from passionfruit.third_party import yolo_cam

    root = Path(yolo_cam.__file__).resolve().parent
    try:
        from passionfruit.third_party.yolo_cam.eigen_cam import EigenCAM
        from passionfruit.third_party.yolo_cam.utils.image import show_cam_on_image
    except ModuleNotFoundError as error:
        raise CamTestError(
            "無法載入 YOLO-26-CAM 相依套件"
            f"（缺少 {error.name}）。請在專案環境安裝 requirements.txt，"
            "並確認已依 README 安裝與 PyTorch 配套的 torchvision。"
        ) from error
    except Exception as error:
        raise CamTestError(
            f"載入 YOLO-26-CAM 失敗：{type(error).__name__}: {error}"
        ) from error
    return EigenCAM, show_cam_on_image, root


def validate_input_file(path: Path, label: str, suffix: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CamTestError(f"找不到{label}：{resolved}")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise CamTestError(f"{label}必須是 {suffix} 檔案：{resolved}")
    return resolved


def validate_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise CamTestError(
            "已指定 device=mps，但目前 PyTorch 環境無法使用 MPS。"
            "請檢查 PyTorch／macOS 環境，或明確改用 --device cpu。"
        )


def load_segmentation_model(model_path: Path, device: str, imgsz: int) -> YOLO:
    try:
        model = YOLO(str(model_path))
    except Exception as error:
        raise CamTestError(
            f"無法載入 YOLO 模型 {model_path}：{type(error).__name__}: {error}"
        ) from error
    task = getattr(model, "task", None)
    if task != "segment":
        raise CamTestError(
            f"模型必須是 segmentation model；目前 task={task or 'unknown'}：{model_path}"
        )
    try:
        model.to(device)
    except Exception as error:
        hint = " 建議改用 --device cpu。" if device == "mps" else ""
        raise CamTestError(
            f"無法把模型移到 {device}：{type(error).__name__}: {error}.{hint}"
        ) from error
    model.overrides.update(
        {"task": "segment", "imgsz": imgsz, "device": device, "verbose": False}
    )
    return model


def resolve_target_layer(
    model: YOLO, layer_index: int
) -> tuple[torch.nn.Module, int, int]:
    inner_model = getattr(model, "model", None)
    layers = getattr(inner_model, "model", None)
    if layers is None:
        raise CamTestError(
            "模型沒有可供 EigenCAM 選取的 model.model.model layer sequence。"
        )
    try:
        layer_count = len(layers)
    except TypeError as error:
        raise CamTestError("model.model.model 不是可索引的 layer sequence。") from error
    if not -layer_count <= layer_index < layer_count:
        raise CamTestError(
            f"layer index {layer_index} 超出範圍；此模型共有 {layer_count} 層，"
            f"有效範圍為 {-layer_count} 到 {layer_count - 1}。"
        )
    layer = layers[layer_index]
    if not isinstance(layer, torch.nn.Module):
        raise CamTestError(
            f"layer index {layer_index} 不是 torch.nn.Module，而是 {type(layer).__name__}。"
        )
    return layer, layer_index % layer_count, layer_count


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise CamTestError(f"OpenCV 無法讀取圖片：{image_path}")
    return image


def predict(
    model: YOLO,
    image_bgr: np.ndarray,
    imgsz: int,
    device: str,
    conf: float,
):
    try:
        results = model.predict(
            source=image_bgr,
            imgsz=imgsz,
            device=device,
            conf=conf,
            verbose=False,
            save=False,
        )
        if not results:
            raise CamTestError("YOLO segmentation 推論沒有回傳 Results。")
        return results[0]
    except CamTestError:
        raise
    except Exception as error:
        hint = " 建議改用 --device cpu。" if device == "mps" else ""
        raise CamTestError(
            f"YOLO segmentation 推論失敗：{type(error).__name__}: {error}.{hint}"
        ) from error


def normalize_cam_result(raw_cam: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    if raw_cam is None:
        raise CamTestError("EigenCAM 回傳空值。")
    cam_batch = np.asarray(raw_cam)
    if cam_batch.size == 0 or cam_batch.ndim != 3 or cam_batch.shape[0] < 1:
        raise CamTestError(
            f"EigenCAM 回傳格式不正確，預期 [N,H,W]，實際為 {cam_batch.shape}。"
        )
    grayscale_cam = np.asarray(cam_batch[0], dtype=np.float32)
    if grayscale_cam.shape != expected_shape:
        raise CamTestError(
            f"EigenCAM 尺寸 {grayscale_cam.shape} 與原圖尺寸 {expected_shape} 不一致。"
        )
    if not np.all(np.isfinite(grayscale_cam)):
        raise CamTestError("EigenCAM 灰階結果包含 NaN 或 Infinity。")
    minimum = float(grayscale_cam.min())
    maximum = float(grayscale_cam.max())
    if minimum < -1e-5 or maximum > 1.00001:
        raise CamTestError(
            f"EigenCAM 灰階值未正規化到 [0,1]；實際範圍為 [{minimum}, {maximum}]。"
        )
    return grayscale_cam


def signed_eigencam_projection(activation_batch: np.ndarray) -> np.ndarray:
    """Reproduce YOLO-26-CAM's centered SVD before its ReLU and scaling."""
    activations = np.asarray(activation_batch, dtype=np.float32)
    if activations.ndim != 4 or activations.shape[0] < 1:
        raise CamTestError(
            "target layer activation 必須是 [N,C,H,W]；"
            f"實際為 {activations.shape}。請改用其他 --layer-index。"
        )
    if np.any(np.isinf(activations)):
        raise CamTestError("target layer activation 包含 Infinity。")
    first = activations[0].copy()
    first[np.isnan(first)] = 0
    reshaped = first.reshape(first.shape[0], -1).transpose()
    reshaped = reshaped - reshaped.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(reshaped, full_matrices=True)
    except np.linalg.LinAlgError as error:
        raise CamTestError(f"signed EigenCAM SVD 無法收斂：{error}") from error
    projection = reshaped @ vt[0, :]
    return np.asarray(projection.reshape(first.shape[1:]), dtype=np.float32)


def run_eigencam(
    model: YOLO,
    target_layer: torch.nn.Module,
    image_bgr: np.ndarray,
    device: str,
    eigen_cam_class,
) -> tuple[np.ndarray, np.ndarray]:
    cam = None
    try:
        cam = eigen_cam_class(model, [target_layer], task="seg")
        raw_cam = cam(image_bgr)
        grayscale_cam = normalize_cam_result(raw_cam, image_bgr.shape[:2])
        captured = getattr(cam.activations_and_grads, "activations", None)
        if not captured:
            raise CamTestError("EigenCAM hook 沒有取得 target layer activation。")
        activation_batch = captured[-1].detach().cpu().numpy()
        signed_raw = signed_eigencam_projection(activation_batch)
        return grayscale_cam, signed_raw
    except CamTestError:
        raise
    except Exception as error:
        hint = (
            " MPS 可能不支援此 hook 或 operator；程式沒有改變結果或自動回退，"
            "請明確改用 --device cpu。"
            if device == "mps"
            else " 請嘗試其他 --layer-index，並確認該層輸出是 4D feature tensor。"
        )
        raise CamTestError(
            f"EigenCAM 與 target layer 不相容或計算失敗："
            f"{type(error).__name__}: {error}.{hint}"
        ) from error
    finally:
        activations = getattr(cam, "activations_and_grads", None)
        if activations is not None:
            activations.release()


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise CamTestError("無法 normalize 空白或非有限數值的 CAM map。")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.float32)
    return np.asarray((values - minimum) / (maximum - minimum), dtype=np.float32)


def resize_normalized_map(
    values: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    normalized = normalize_map(values)
    resized = cv2.resize(
        normalized,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return normalize_map(resized)


def signed_diagnostic_maps(
    signed_raw: np.ndarray, target_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    components = {
        "positive": np.maximum(signed_raw, 0),
        "negative": np.maximum(-signed_raw, 0),
        "absolute": np.abs(signed_raw),
    }
    return {
        name: resize_normalized_map(values, target_shape)
        for name, values in components.items()
    }


def cam_statistics(cam: np.ndarray) -> dict[str, Any]:
    maximum_index = int(np.argmax(cam))
    minimum_index = int(np.argmin(cam))
    max_y, max_x = np.unravel_index(maximum_index, cam.shape)
    min_y, min_x = np.unravel_index(minimum_index, cam.shape)
    return {
        "min": float(np.min(cam)),
        "max": float(np.max(cam)),
        "mean": float(np.mean(cam)),
        "median": float(np.median(cam)),
        "argmax_xy": [int(max_x), int(max_y)],
        "argmin_xy": [int(min_x), int(min_y)],
    }


def apply_colormap(cam: np.ndarray, colormap: int | None) -> np.ndarray:
    grayscale = np.uint8(cam * 255.0)
    if colormap is None:
        return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(grayscale, colormap)


def activation_colorbar(width: int, colormap: int | None) -> np.ndarray:
    footer = np.full((68, width, 3), 28, dtype=np.uint8)
    font_scale = max(0.34, min(0.52, width / 900.0))
    left = "Low activation  0.0"
    right = "1.0  High activation"
    cv2.putText(
        footer,
        left,
        (5, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    right_size = cv2.getTextSize(
        right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )[0][0]
    cv2.putText(
        footer,
        right,
        (max(5, width - right_size - 5), 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    gradient = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    gradient = np.repeat(gradient, 28, axis=0)
    if colormap is None:
        gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    else:
        gradient_bgr = cv2.applyColorMap(gradient, colormap)
    footer[34:62] = gradient_bgr
    cv2.rectangle(footer, (0, 33), (width - 1, 62), (230, 230, 230), 1)
    return footer


def with_activation_colorbar(image: np.ndarray, colormap: int | None) -> np.ndarray:
    return cv2.vconcat((image, activation_colorbar(image.shape[1], colormap)))


def diagnostic_visualization(cam: np.ndarray, title: str) -> np.ndarray:
    image = apply_colormap(cam, cv2.COLORMAP_JET)
    header = np.full((38, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        header,
        f"{title} (diagnostic only)",
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat(
        (header, image, activation_colorbar(image.shape[1], cv2.COLORMAP_JET))
    )


def create_cam_images(
    image_bgr: np.ndarray, grayscale_cam: np.ndarray, show_cam_on_image
) -> dict[str, np.ndarray]:
    grayscale_u8 = np.uint8(grayscale_cam * 255.0)
    gray_bgr = apply_colormap(grayscale_cam, None)
    jet_bgr = apply_colormap(grayscale_cam, cv2.COLORMAP_JET)
    turbo_bgr = apply_colormap(grayscale_cam, cv2.COLORMAP_TURBO)
    image_float_bgr = image_bgr.astype(np.float32) / 255.0
    overlay_bgr = show_cam_on_image(
        image_float_bgr,
        grayscale_cam,
        use_rgb=False,
        colormap=cv2.COLORMAP_JET,
    )
    return {
        "grayscale": grayscale_u8,
        "heatmap": with_activation_colorbar(jet_bgr, cv2.COLORMAP_JET),
        "overlay": with_activation_colorbar(overlay_bgr, cv2.COLORMAP_JET),
        "gray": with_activation_colorbar(gray_bgr, None),
        "jet": with_activation_colorbar(jet_bgr, cv2.COLORMAP_JET),
        "turbo": with_activation_colorbar(turbo_bgr, cv2.COLORMAP_TURBO),
        "heatmap_raw": jet_bgr,
        "overlay_raw": overlay_bgr,
    }


def mark_cam_argmax(image_bgr: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
    marked = image_bgr.copy()
    y, x = np.unravel_index(int(np.argmax(grayscale_cam)), grayscale_cam.shape)
    radius = max(8, round(min(image_bgr.shape[:2]) * 0.035))
    thickness = max(2, radius // 5)
    cv2.circle(marked, (int(x), int(y)), radius, (255, 255, 255), thickness + 2)
    cv2.circle(marked, (int(x), int(y)), radius, (0, 0, 255), thickness)
    cv2.drawMarker(
        marked,
        (int(x), int(y)),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=radius * 2,
        thickness=thickness,
    )
    cv2.putText(
        marked,
        f"CAM argmax ({x}, {y})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        marked,
        f"CAM argmax ({x}, {y})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return marked


def create_comparison(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    model_filename: str,
    layer_index: int,
    imgsz: int,
) -> np.ndarray:
    if len(images) != 4 or len(labels) != 4:
        raise ValueError("comparison 必須包含四張圖片與四個標籤。")
    source_height, source_width = images[0].shape[:2]
    scale = min(1.0, 640.0 / max(source_width, source_height))
    panel_width = max(1, round(source_width * scale))
    panel_height = max(1, round(source_height * scale))
    panels = []
    for image, label in zip(images, labels, strict=True):
        panel = cv2.resize(
            image, (panel_width, panel_height), interpolation=cv2.INTER_AREA
        )
        cv2.rectangle(panel, (0, 0), (panel_width, 34), (0, 0, 0), thickness=-1)
        cv2.putText(
            panel,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    body = cv2.hconcat(panels)
    header = np.full((82, body.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        header,
        f"model={model_filename} | target layer={layer_index} | imgsz={imgsz}",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "EigenCAM spatial activation/saliency response; not class-specific",
        (14, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (190, 220, 255),
        1,
        cv2.LINE_AA,
    )
    result = cv2.vconcat((header, body))
    return with_activation_colorbar(result, cv2.COLORMAP_JET)


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(
        np.asarray(mask, dtype=np.float32),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0.5


def class_name(names, class_index: int) -> str:
    return str(names[class_index] if isinstance(names, dict) else names[class_index])


def primary_prediction(
    result, target_shape: tuple[int, int]
) -> PrimaryPrediction | None:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or len(boxes) == 0:
        return None
    mask_data = getattr(masks, "data", None)
    if mask_data is None or len(mask_data) != len(boxes):
        raise CamTestError(
            "segmentation prediction 有 boxes，但沒有一一對應的 masks，"
            "無法執行 occlusion comparison。"
        )
    confidences = boxes.conf.detach().cpu().numpy()
    index = int(np.argmax(confidences))
    class_index = int(boxes.cls[index].item())
    mask = resize_mask(mask_data[index].detach().cpu().numpy(), target_shape)
    return PrimaryPrediction(
        class_index=class_index,
        class_name=class_name(result.names, class_index),
        confidence=float(confidences[index]),
        mask=mask,
    )


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first & second) / union)


def match_primary_prediction(
    result, primary: PrimaryPrediction, target_shape: tuple[int, int]
) -> tuple[float, float | None]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    mask_data = getattr(masks, "data", None)
    if boxes is None or len(boxes) == 0 or mask_data is None:
        return 0.0, None
    best: tuple[float, float] | None = None
    for index in range(len(boxes)):
        if int(boxes.cls[index].item()) != primary.class_index:
            continue
        candidate_mask = resize_mask(
            mask_data[index].detach().cpu().numpy(), target_shape
        )
        iou = mask_iou(primary.mask, candidate_mask)
        confidence = float(boxes.conf[index].item())
        if best is None or (iou, confidence) > (best[1], best[0]):
            best = (confidence, iou)
    return best if best is not None else (0.0, None)


def make_occlusions(
    image_bgr: np.ndarray,
    grayscale_cam: np.ndarray,
    fraction: float = OCCLUSION_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pixel_count = grayscale_cam.size
    selected_count = max(1, int(np.ceil(pixel_count * fraction)))
    order = np.argsort(grayscale_cam.reshape(-1), kind="stable")
    low_flat = order[:selected_count]
    high_flat = order[-selected_count:]
    low_mask = np.zeros(pixel_count, dtype=bool)
    high_mask = np.zeros(pixel_count, dtype=bool)
    low_mask[low_flat] = True
    high_mask[high_flat] = True
    low_mask = low_mask.reshape(grayscale_cam.shape)
    high_mask = high_mask.reshape(grayscale_cam.shape)
    low_occluded = image_bgr.copy()
    high_occluded = image_bgr.copy()
    low_occluded[low_mask] = OCCLUSION_BGR
    high_occluded[high_mask] = OCCLUSION_BGR
    return high_occluded, low_occluded, high_mask, low_mask


def run_occlusion_check(
    model: YOLO,
    original_result,
    image_bgr: np.ndarray,
    grayscale_cam: np.ndarray,
    imgsz: int,
    device: str,
    conf: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    high_image, low_image, high_mask, _ = make_occlusions(image_bgr, grayscale_cam)
    high_result = predict(model, high_image, imgsz, device, conf)
    low_result = predict(model, low_image, imgsz, device, conf)
    primary = primary_prediction(original_result, image_bgr.shape[:2])
    metrics: dict[str, Any] = {
        "method": {
            "fraction": OCCLUSION_FRACTION,
            "selected_pixels": int(np.count_nonzero(high_mask)),
            "total_pixels": int(grayscale_cam.size),
            "fill_bgr": list(OCCLUSION_BGR),
            "matching": "same class with maximum mask IoU to original primary instance",
        }
    }
    primary_mask_u8 = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    if primary is None:
        metrics.update(
            {
                "status": "skipped_no_original_prediction",
                "primary_prediction": None,
                "high_cam": None,
                "low_cam": None,
            }
        )
        high_label = "High-CAM top 10% (no primary score)"
        low_label = "Low-CAM bottom 10% (no primary score)"
    else:
        primary_mask_u8 = primary.mask.astype(np.uint8) * 255
        high_confidence, high_iou = match_primary_prediction(
            high_result, primary, image_bgr.shape[:2]
        )
        low_confidence, low_iou = match_primary_prediction(
            low_result, primary, image_bgr.shape[:2]
        )
        high_drop = primary.confidence - high_confidence
        low_drop = primary.confidence - low_confidence
        metrics.update(
            {
                "status": "ok",
                "primary_prediction": {
                    "class_index": primary.class_index,
                    "class_name": primary.class_name,
                    "confidence": primary.confidence,
                    "mask_pixels": int(np.count_nonzero(primary.mask)),
                },
                "high_cam": {
                    "confidence": high_confidence,
                    "confidence_drop": high_drop,
                    "mask_iou": high_iou,
                },
                "low_cam": {
                    "confidence": low_confidence,
                    "confidence_drop": low_drop,
                    "mask_iou": low_iou,
                },
                "high_drop_minus_low_drop": high_drop - low_drop,
            }
        )
        high_label = f"High-CAM top 10% | conf={high_confidence:.4f}"
        low_label = f"Low-CAM bottom 10% | conf={low_confidence:.4f}"
    comparison = create_occlusion_comparison(
        (original_result.plot(), high_result.plot(), low_result.plot()),
        ("Original prediction", high_label, low_label),
        metrics,
    )
    return metrics, {
        "occlusion_high": high_image,
        "occlusion_low": low_image,
        "occlusion_comparison": comparison,
        "primary_mask": primary_mask_u8,
    }


def create_occlusion_comparison(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    metrics: dict[str, Any],
) -> np.ndarray:
    source_height, source_width = images[0].shape[:2]
    scale = min(1.0, 640.0 / max(source_width, source_height))
    panel_size = (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )
    panels = []
    for image, label in zip(images, labels, strict=True):
        panel = cv2.resize(image, panel_size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (panel_size[0], 34), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    body = cv2.hconcat(panels)
    header = np.full((76, body.shape[1], 3), 28, dtype=np.uint8)
    if metrics.get("status") == "ok":
        primary = metrics["primary_prediction"]
        summary = (
            f"primary={primary['class_name']} {primary['confidence']:.4f} | "
            f"high drop={metrics['high_cam']['confidence_drop']:.4f} | "
            f"low drop={metrics['low_cam']['confidence_drop']:.4f}"
        )
    else:
        summary = "No original prediction: confidence comparison skipped"
    cv2.putText(
        header,
        "Occlusion sanity check: fixed neutral gray, exactly 10% of pixels",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        summary,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (190, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat((header, body))


def write_image(path: Path, image: np.ndarray) -> None:
    try:
        succeeded = cv2.imwrite(str(path), image)
    except Exception as error:
        raise CamTestError(
            f"無法寫入圖片 {path}：{type(error).__name__}: {error}"
        ) from error
    if not succeeded:
        raise CamTestError(f"OpenCV 無法寫入圖片：{path}")


def write_outputs(
    output_directory: Path,
    image_artifacts: dict[str, np.ndarray],
    normalized_cam: np.ndarray,
    signed_raw: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CamTestError(f"無法建立輸出資料夾 {output_directory}：{error}") from error
    for key, image in image_artifacts.items():
        if key in OUTPUT_FILENAMES:
            write_image(output_directory / OUTPUT_FILENAMES[key], image)
    try:
        np.save(
            output_directory / OUTPUT_FILENAMES["normalized"],
            np.asarray(normalized_cam, dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            output_directory / OUTPUT_FILENAMES["signed_raw"],
            np.asarray(signed_raw, dtype=np.float32),
            allow_pickle=False,
        )
        (output_directory / OUTPUT_FILENAMES["metrics"]).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        raise CamTestError(f"無法寫入 CAM numeric/debug data：{error}") from error


def print_cam_summary(
    cam_stats: dict[str, Any],
    signed_raw: np.ndarray,
    diagnostic_maps: dict[str, np.ndarray],
    positive_cam_error: float,
) -> None:
    print("CAM numeric summary:")
    print(f"CAM min: {cam_stats['min']:.8f}")
    print(f"CAM max: {cam_stats['max']:.8f}")
    print(f"CAM mean: {cam_stats['mean']:.8f}")
    print(f"CAM median: {cam_stats['median']:.8f}")
    print(f"CAM argmax (x, y): {tuple(cam_stats['argmax_xy'])}")
    print(f"CAM argmin (x, y): {tuple(cam_stats['argmin_xy'])}")
    print(
        "Signed raw projection: "
        f"shape={signed_raw.shape}, min={signed_raw.min():.8f}, "
        f"max={signed_raw.max():.8f}"
    )
    for name in ("positive", "negative", "absolute"):
        stats = cam_statistics(diagnostic_maps[name])
        print(f"{name.capitalize()} polarity argmax (x, y): {tuple(stats['argmax_xy'])}")
    print(
        "Positive diagnostic vs YOLO-26-CAM max abs error: "
        f"{positive_cam_error:.8f}"
    )


def print_occlusion_summary(metrics: dict[str, Any]) -> None:
    print("Occlusion sanity check:")
    print(
        f"fill BGR = {tuple(metrics['method']['fill_bgr'])}, "
        f"pixels = {metrics['method']['selected_pixels']} / "
        f"{metrics['method']['total_pixels']}"
    )
    if metrics.get("status") != "ok":
        print("Original prediction: none; confidence comparison skipped.")
        return
    primary = metrics["primary_prediction"]
    print(f"primary class: {primary['class_name']} (index={primary['class_index']})")
    print(f"original confidence: {primary['confidence']:.8f}")
    print(f"high-CAM occluded confidence: {metrics['high_cam']['confidence']:.8f}")
    print(f"high-CAM confidence drop: {metrics['high_cam']['confidence_drop']:.8f}")
    print(f"low-CAM occluded confidence: {metrics['low_cam']['confidence']:.8f}")
    print(f"low-CAM confidence drop: {metrics['low_cam']['confidence_drop']:.8f}")
    print(
        "high drop - low drop: "
        f"{metrics['high_drop_minus_low_drop']:.8f} (single-image diagnostic only)"
    )


def run(args: argparse.Namespace) -> Path:
    model_path = validate_input_file(args.model, "模型", ".pt")
    image_path = validate_input_file(args.image, "圖片")
    validate_device(args.device)
    eigen_cam_class, show_cam_on_image, third_party_root = load_cam_dependencies()
    image_bgr = read_image(image_path)
    model = load_segmentation_model(model_path, args.device, args.imgsz)
    original_result = predict(model, image_bgr, args.imgsz, args.device, args.conf)
    prediction_bgr = original_result.plot()
    target_layer, resolved_index, layer_count = resolve_target_layer(
        model, args.layer_index
    )
    layer_type = f"{type(target_layer).__module__}.{type(target_layer).__qualname__}"
    print("CAM target layer:")
    print(
        f"index = {args.layer_index} "
        f"(resolved = {resolved_index}, layers = {layer_count})"
    )
    print(f"type = {layer_type}")
    print(f"YOLO-26-CAM = {third_party_root}")
    print(
        "Interpretation: EigenCAM spatial activation/saliency response; "
        "not transformer attention and not class-specific Grad-CAM."
    )
    grayscale_cam, signed_raw = run_eigencam(
        model, target_layer, image_bgr, args.device, eigen_cam_class
    )
    diagnostic_maps = signed_diagnostic_maps(signed_raw, image_bgr.shape[:2])
    positive_cam_error = float(
        np.max(np.abs(diagnostic_maps["positive"] - grayscale_cam))
    )
    cam_stats = cam_statistics(grayscale_cam)
    print_cam_summary(cam_stats, signed_raw, diagnostic_maps, positive_cam_error)
    cam_images = create_cam_images(image_bgr, grayscale_cam, show_cam_on_image)
    diagnostic_images = {
        name: diagnostic_visualization(cam, name.capitalize())
        for name, cam in diagnostic_maps.items()
    }
    comparison = create_comparison(
        (
            image_bgr,
            prediction_bgr,
            cam_images["heatmap_raw"],
            cam_images["overlay_raw"],
        ),
        ("Original", "YOLO Prediction", "EigenCAM Heatmap", "CAM Overlay"),
        model_path.name,
        args.layer_index,
        args.imgsz,
    )
    occlusion_metrics, occlusion_images = run_occlusion_check(
        model,
        original_result,
        image_bgr,
        grayscale_cam,
        args.imgsz,
        args.device,
        args.conf,
    )
    print_occlusion_summary(occlusion_metrics)
    metrics = {
        "interpretation": (
            "EigenCAM spatial activation/saliency response; not transformer attention "
            "and not class-specific Grad-CAM"
        ),
        "visualization": {
            "cam_zero": "low; black in grayscale and blue in OpenCV COLORMAP_JET",
            "cam_one": "high; white in grayscale and red in OpenCV COLORMAP_JET",
            "inverted": False,
        },
        "cam_normalized": cam_stats,
        "signed_raw": {
            **cam_statistics(signed_raw),
            "shape": list(signed_raw.shape),
            "coordinate_space": "native target-layer feature map",
            "positive_vs_library_max_abs_error": positive_cam_error,
            "diagnostic_argmax_xy": {
                name: cam_statistics(cam)["argmax_xy"]
                for name, cam in diagnostic_maps.items()
            },
        },
        "occlusion": occlusion_metrics,
    }
    image_artifacts = {
        "original": image_bgr,
        "grayscale": cam_images["grayscale"],
        "heatmap": cam_images["heatmap"],
        "overlay": cam_images["overlay"],
        "prediction": prediction_bgr,
        "comparison": comparison,
        "positive": diagnostic_images["positive"],
        "negative": diagnostic_images["negative"],
        "absolute": diagnostic_images["absolute"],
        "gray": cam_images["gray"],
        "jet": cam_images["jet"],
        "turbo": cam_images["turbo"],
        "argmax": mark_cam_argmax(image_bgr, grayscale_cam),
        **occlusion_images,
    }
    output_directory = args.output.expanduser().resolve()
    write_outputs(
        output_directory,
        image_artifacts,
        grayscale_cam,
        signed_raw,
        metrics,
    )
    return output_directory


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_directory = run(args)
    except CamTestError as error:
        parser.exit(1, f"錯誤：{error}\n")
    print(f"完成：結果已寫入 {output_directory}")
    for filename in OUTPUT_FILENAMES.values():
        print(f"- {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
