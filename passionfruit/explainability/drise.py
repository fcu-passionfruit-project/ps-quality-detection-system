from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


EXPECTED_CLASSES = {
    0: "abrasion",
    1: "anthracnose",
    2: "insect_bite",
    3: "insect_track",
}
SIMILARITY_METHOD = "class_match * mask_iou * candidate_confidence"


class DriseError(RuntimeError):
    """A user-actionable D-RISE failure."""


@dataclass(frozen=True)
class InstancePrediction:
    index: int
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: np.ndarray
    mask: np.ndarray


def _class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    try:
        return str(names[class_id] if isinstance(names, dict) else names[class_id])
    except (KeyError, IndexError) as error:
        raise DriseError(f"模型 class names 缺少 class id {class_id}。") from error


def validate_class_contract(names: dict[int, str] | list[str]) -> None:
    actual = {class_id: _class_name(names, class_id) for class_id in EXPECTED_CLASSES}
    if actual != EXPECTED_CLASSES:
        raise DriseError(
            "模型 class mapping 不符合局部瑕疵契約；"
            f"預期 {EXPECTED_CLASSES}，實際 {actual}。"
        )


def mask_to_original_space(
    mask: np.ndarray,
    original_shape: tuple[int, int],
    *,
    context: str,
) -> np.ndarray:
    """Convert a model mask to the original-image pixel coordinate system."""
    array = np.asarray(mask)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise DriseError(
            f"{context} mask 必須是有限值的 2D array；目前 shape={array.shape}。"
        )
    height, width = original_shape
    if height <= 0 or width <= 0:
        raise DriseError(f"原始影像尺寸無效：{original_shape}。")
    if array.shape != original_shape:
        array = cv2.resize(
            array.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    converted = np.asarray(array > 0.5, dtype=bool)
    if converted.shape != original_shape:
        raise DriseError(
            f"{context} mask 無法轉換到原圖座標；"
            f"預期 {original_shape}，實際 {converted.shape}。"
        )
    return converted


def extract_instances(
    result: Any,
    original_shape: tuple[int, int],
    *,
    context: str,
) -> list[InstancePrediction]:
    result_shape = tuple(int(value) for value in result.orig_shape)
    if result_shape != original_shape:
        raise DriseError(
            f"{context} inference 座標系不一致："
            f"result.orig_shape={result_shape}，source shape={original_shape}。"
        )
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    masks = getattr(result, "masks", None)
    mask_data = getattr(masks, "data", None)
    if mask_data is None:
        raise DriseError(f"{context} prediction 有 boxes，但沒有 segmentation masks。")
    if len(mask_data) != len(boxes):
        raise DriseError(
            f"{context} boxes/masks 數量不一致：{len(boxes)} != {len(mask_data)}。"
        )

    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    coordinates = boxes.xyxy.detach().cpu().numpy()
    instances = []
    for index in range(len(boxes)):
        class_id = int(classes[index])
        instances.append(
            InstancePrediction(
                index=index,
                class_id=class_id,
                class_name=_class_name(result.names, class_id),
                confidence=float(confidences[index]),
                box_xyxy=np.asarray(coordinates[index], dtype=np.float64),
                mask=mask_to_original_space(
                    mask_data[index].detach().cpu().numpy(),
                    original_shape,
                    context=f"{context} instance #{index}",
                ),
            )
        )
    return instances


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if first_array.ndim != 2 or second_array.ndim != 2:
        raise DriseError(
            "Mask IoU 只接受 2D masks；"
            f"目前 shapes={first_array.shape}, {second_array.shape}。"
        )
    if first_array.shape != second_array.shape:
        raise DriseError(
            "Mask IoU 座標/尺寸不一致；"
            f"reference={first_array.shape}, candidate={second_array.shape}。"
        )
    first_bool = first_array.astype(bool, copy=False)
    second_bool = second_array.astype(bool, copy=False)
    union = int(np.count_nonzero(first_bool | second_bool))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first_bool & second_bool) / union)


def compute_target_similarity(
    reference: InstancePrediction,
    candidates: Sequence[InstancePrediction],
) -> tuple[float, InstancePrediction | None, float]:
    """Return mask-aware D-RISE affinity for one reference instance.

    Microsoft D-RISE uses localization, objectness, and class affinity. Its
    public DetectionRecord has no segmentation mask, so this project replaces
    box IoU with mask IoU and uses the masked candidate confidence. Candidate
    identity is selected by maximum mask IoU first, then confidence as a stable
    tie-breaker; a higher-confidence but spatially different instance cannot win.
    """
    best: tuple[float, float, InstancePrediction] | None = None
    for candidate in candidates:
        if candidate.class_id != reference.class_id:
            continue
        iou = mask_iou(reference.mask, candidate.mask)
        item = (iou, candidate.confidence, candidate)
        if best is None or item[:2] > best[:2]:
            best = item
    if best is None:
        return 0.0, None, 0.0
    iou, confidence, candidate = best
    return float(iou * confidence), candidate, float(iou)


def select_target_instance(
    instances: Sequence[InstancePrediction],
    target_index: int | None,
    target_class: str | None,
) -> InstancePrediction:
    if not instances:
        raise DriseError("原始圖片沒有任何 prediction，無法選擇 D-RISE target。")
    if target_index is not None:
        if target_index < 0 or target_index >= len(instances):
            available = ", ".join(str(instance.index) for instance in instances)
            raise DriseError(
                f"Target instance index {target_index} does not exist. "
                f"Available instances: {available}"
            )
        target = instances[target_index]
        if target_class is not None and target.class_name != target_class:
            raise DriseError(
                f"target-index {target_index} 是 {target.class_name}，"
                f"與 --target-class {target_class} 不一致。"
            )
        return target
    if target_class is None:
        raise DriseError("必須提供 --target-index，或以 --target-class 選擇唯一 instance。")
    matches = [instance for instance in instances if instance.class_name == target_class]
    if not matches:
        available = ", ".join(sorted({instance.class_name for instance in instances}))
        raise DriseError(
            f"找不到 class={target_class} 的 prediction；現有 classes: {available}。"
        )
    if len(matches) > 1:
        indexes = ", ".join(str(instance.index) for instance in matches)
        raise DriseError(
            f"class={target_class} 有多個 instances（indexes: {indexes}）；"
            "請明確提供 --target-index。"
        )
    return matches[0]


def normalize_saliency(raw: np.ndarray) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise DriseError("D-RISE raw saliency 必須是有限值的 2D array。")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum == minimum:
        return np.zeros(array.shape, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise DriseError(f"已指定 device={requested}，但 CUDA 不可用。")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise DriseError(
            "已指定 device=mps，但目前 PyTorch 環境無法使用 MPS；"
            "請檢查環境，或明確使用 --device cpu。"
        )
    return requested


def _load_microsoft_primitives() -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
    try:
        from vision_explanation_methods.explanations.drise import (
            fuse_mask,
            generate_mask,
        )
    except Exception as error:
        raise DriseError(
            "無法載入 vision-explanation-methods 的 D-RISE primitives。"
            "請依 README 安裝 optional D-RISE dependencies；"
            f"原始錯誤：{type(error).__name__}: {error}"
        ) from error
    return generate_mask, fuse_mask


def _predict(model: YOLO, image_bgr: np.ndarray, args: argparse.Namespace) -> Any:
    try:
        results = model.predict(
            source=image_bgr,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.resolved_device,
            retina_masks=True,
            verbose=False,
        )
    except Exception as error:
        raise DriseError(
            f"YOLO segmentation 推論失敗：{type(error).__name__}: {error}"
        ) from error
    if len(results) != 1:
        raise DriseError(f"單張輸入預期 1 個 Results，實際得到 {len(results)}。")
    return results[0]


def _tensor_to_bgr_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    array = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _annotate_target(image: np.ndarray, target: InstancePrediction) -> np.ndarray:
    output = image.copy()
    color_layer = np.zeros_like(output)
    color_layer[target.mask] = (0, 165, 255)
    output = cv2.addWeighted(output, 1.0, color_layer, 0.45, 0.0)
    x1, y1, x2, y2 = np.rint(target.box_xyxy).astype(int)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 3)
    labels = (
        f"target #{target.index}: {target.class_name}",
        f"conf={target.confidence:.3f}",
    )
    for line_index, label in enumerate(labels):
        cv2.putText(
            output,
            label,
            (8, 24 + line_index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise DriseError(f"無法寫入影像：{path}")


def _unique_output_dir(base: Path, source: Path, target_index: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = base.expanduser().resolve() / f"{stamp}_{source.stem}_target{target_index}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _debug_artifacts(
    output_dir: Path,
    index: int,
    random_mask: np.ndarray,
    masked_input: np.ndarray,
    masked_result: Any,
    similarity: float,
    matched: InstancePrediction | None,
    mask_iou_value: float,
    reference: InstancePrediction,
    candidates: Sequence[InstancePrediction],
) -> None:
    directory = output_dir / "debug" / f"mask_{index:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    _write_image(directory / "random_mask.png", np.rint(random_mask * 255).astype(np.uint8))
    _write_image(directory / "masked_input.jpg", masked_input)
    _write_image(directory / "masked_prediction.jpg", masked_result.plot())
    payload = {
        "mask_index": index,
        "target_similarity": similarity,
        "matched_candidate_index": matched.index if matched is not None else None,
        "matched_candidate_confidence": matched.confidence if matched is not None else None,
        "mask_iou": mask_iou_value,
        "similarity_method": SIMILARITY_METHOD,
        "candidates": [
            {
                "candidate_index": candidate.index,
                "class_id": candidate.class_id,
                "class_name": candidate.class_name,
                "confidence": candidate.confidence,
                "mask_iou": mask_iou(reference.mask, candidate.mask),
                "similarity": (
                    mask_iou(reference.mask, candidate.mask) * candidate.confidence
                    if candidate.class_id == reference.class_id
                    else 0.0
                ),
            }
            for candidate in candidates
        ],
    }
    (directory / "target_similarity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def run(args: argparse.Namespace) -> Path:
    generate_mask, fuse_mask = _load_microsoft_primitives()
    args.resolved_device = resolve_device(args.device)
    if args.imgsz <= 0 or args.num_masks <= 0 or args.mask_res <= 0:
        raise DriseError("--imgsz、--num-masks 與 --mask-res 都必須大於 0。")
    if not 0.0 <= args.conf <= 1.0:
        raise DriseError("--conf 必須介於 0 與 1。")
    if args.save_debug_masks < 0 or args.save_debug_masks > args.num_masks:
        raise DriseError("--save-debug-masks 必須介於 0 與 --num-masks 之間。")

    model_path = args.model.expanduser().resolve()
    source_path = args.source.expanduser().resolve()
    if not model_path.is_file():
        raise DriseError(f"找不到模型：{model_path}")
    if not source_path.is_file():
        raise DriseError(f"找不到來源圖片：{source_path}")
    image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise DriseError(f"OpenCV 無法讀取來源圖片：{source_path}")
    original_shape = image_bgr.shape[:2]

    try:
        model = YOLO(str(model_path))
    except Exception as error:
        raise DriseError(f"無法載入 YOLO 模型：{type(error).__name__}: {error}") from error
    if getattr(model, "task", None) != "segment":
        raise DriseError(f"模型必須是 segmentation task；實際為 {model.task!r}。")
    validate_class_contract(model.names)

    original_result = _predict(model, image_bgr, args)
    instances = extract_instances(original_result, original_shape, context="original")
    if not instances:
        raise DriseError("原始圖片沒有任何 prediction，無法產生 D-RISE。")
    print("Original predictions:")
    for instance in instances:
        print(
            f"  Prediction #{instance.index}: {instance.class_name} "
            f"(class_id={instance.class_id}, conf={instance.confidence:.6f}, "
            f"box={instance.box_xyxy.tolist()})"
        )
    target = select_target_instance(instances, args.target_index, args.target_class)
    if not np.any(target.mask):
        raise DriseError(f"Target instance #{target.index} 的 segmentation mask 為空。")
    output_dir = _unique_output_dir(args.output, source_path, target.index)

    _write_image(output_dir / "prediction.jpg", original_result.plot())
    _write_image(output_dir / "target_instance.jpg", _annotate_target(image_bgr, target))
    _write_image(output_dir / "target_mask.png", target.mask.astype(np.uint8) * 255)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    height, width = original_shape
    padding = int(max(height / args.mask_res, width / args.mask_res))
    weighted_sum = np.zeros(original_shape, dtype=np.float64)
    mask_sum = np.zeros(original_shape, dtype=np.float64)
    score_sum = 0.0
    similarities: list[float] = []

    for mask_index in range(args.num_masks):
        # These are Microsoft's official D-RISE random-mask and masking primitives.
        random_mask = generate_mask(
            (args.mask_res, args.mask_res), original_shape, padding, "cpu"
        )
        masked_tensor = fuse_mask(image_tensor, random_mask)
        masked_bgr = _tensor_to_bgr_uint8(masked_tensor)
        masked_result = _predict(model, masked_bgr, args)
        candidates = extract_instances(
            masked_result, original_shape, context=f"masked #{mask_index}"
        )
        similarity, matched, iou = compute_target_similarity(target, candidates)
        mask_2d = random_mask[0].detach().cpu().numpy().astype(np.float64)
        if mask_2d.shape != original_shape:
            raise DriseError(
                f"Microsoft D-RISE random mask shape mismatch: "
                f"{mask_2d.shape} != {original_shape}。"
            )
        weighted_sum += similarity * mask_2d
        mask_sum += mask_2d
        score_sum += similarity
        similarities.append(similarity)
        if mask_index < args.save_debug_masks:
            _debug_artifacts(
                output_dir,
                mask_index,
                mask_2d,
                masked_bgr,
                masked_result,
                similarity,
                matched,
                iou,
                target,
                candidates,
            )
        print(
            f"D-RISE mask {mask_index + 1}/{args.num_masks}: "
            f"similarity={similarity:.6f}",
            flush=True,
        )

    # Algebraically identical to Microsoft's saliency_fusion(normalize=True),
    # accumulated online to avoid retaining N full-resolution 3-channel masks.
    raw_saliency = weighted_sum - (score_sum / args.num_masks) * mask_sum
    normalized = normalize_saliency(raw_saliency)
    np.save(output_dir / "drise_raw.npy", raw_saliency)
    heatmap = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0.0)
    _write_image(output_dir / "drise_heatmap.png", heatmap)
    _write_image(output_dir / "drise_overlay.png", overlay)

    metadata = {
        "method": "D-RISE",
        "implementation": {
            "mask_generation": "Microsoft vision-explanation-methods generate_mask",
            "mask_application": "Microsoft vision-explanation-methods fuse_mask",
            "saliency_accumulation": "streaming algebraic equivalent of Microsoft saliency_fusion(normalize=True)",
            "raw_saliency": "sum(similarity * mask) - mean(similarity) * sum(mask)",
            "normalization": "min-max for PNG only; drise_raw.npy is pre-normalization",
        },
        "model": str(model_path),
        "source": str(source_path),
        "output_directory": str(output_dir),
        "target_instance_index": target.index,
        "target_class_id": target.class_id,
        "target_class_name": target.class_name,
        "target_confidence": target.confidence,
        "target_box_xyxy": target.box_xyxy.tolist(),
        "original_predictions": [
            {
                "instance_index": instance.index,
                "class_id": instance.class_id,
                "class_name": instance.class_name,
                "confidence": instance.confidence,
                "box_xyxy": instance.box_xyxy.tolist(),
                "mask_pixel_count": int(np.count_nonzero(instance.mask)),
            }
            for instance in instances
        ],
        "similarity_method": SIMILARITY_METHOD,
        "candidate_selection": "same class, maximum mask IoU, confidence tie-breaker",
        "coordinate_space": {
            "name": "original image pixels",
            "shape_hw": list(original_shape),
            "ultralytics_retina_masks": True,
            "mask_resize_fallback": "nearest-neighbor to result.orig_shape",
        },
        "num_masks": args.num_masks,
        "mask_res": args.mask_res,
        "mask_padding": padding,
        "seed": args.seed,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "requested_device": args.device,
        "device": args.resolved_device,
        "mask_generation_device": "cpu",
        "save_debug_masks": args.save_debug_masks,
        "similarity_summary": {
            "minimum": min(similarities),
            "maximum": max(similarities),
            "mean": float(np.mean(similarities)),
            "nonzero_count": int(np.count_nonzero(similarities)),
        },
        "heatmap_polarity": "JET blue=low contribution, red=high contribution to the selected instance",
        "versions": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "vision-explanation-methods": _package_version("vision-explanation-methods"),
            "numpy": np.__version__,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="為單一 YOLO26s-seg prediction instance 產生 D-RISE saliency map。"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--target-class", choices=tuple(EXPECTED_CLASSES.values()))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda:0 等")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--num-masks", type=int, default=1000)
    parser.add_argument("--mask-res", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/drise"))
    parser.add_argument("--save-debug-masks", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir = run(args)
    except DriseError as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1
    print(f"D-RISE 輸出完成：{output_dir}")
    return 0
