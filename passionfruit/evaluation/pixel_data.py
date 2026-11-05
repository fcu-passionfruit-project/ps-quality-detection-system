"""Strict YOLO polygon audit and official letterbox mask restoration."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics.utils.ops import scale_masks

from passionfruit.evaluation.evaluate_experiment_d_area import (
    CLASS_NAMES, EvaluationError, Sample, rasterize_yolo_polygons, sha256_file,
)
from passionfruit.inference.image_processing import create_circular_mask, read_image


def roi_mask(shape: tuple[int, int], mode: str) -> np.ndarray:
    if mode == "django-ellipse":
        return create_circular_mask(shape, mask_scale=1.01) > 0
    if mode == "full-image":
        return np.ones(shape, dtype=bool)
    raise EvaluationError(f"Unknown ROI definition: {mode}")


def load_gt(path: Path, shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], list[int]]:
    if not path.is_file():
        raise EvaluationError(f"遺失標註（不能當成陰性）：{path}")
    counts = [0] * 4
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
            coords = np.asarray([float(v) for v in parts[1:]], dtype=float)
            if cid not in range(4) or len(coords) < 6 or len(coords) % 2:
                raise ValueError("class ID / polygon vertex count")
            pts = coords.reshape(-1, 2)
            if not np.isfinite(pts).all() or np.any((pts < 0) | (pts > 1)):
                raise ValueError("nonfinite / out-of-range coordinates")
            pixels = (pts * [shape[1], shape[0]]).astype(np.int32)
            if len(np.unique(pixels, axis=0)) < 3 or cv2.contourArea(pixels) <= 0:
                raise ValueError("degenerate polygon on original grid")
            counts[cid] += 1
        except (ValueError, IndexError) as error:
            raise EvaluationError(f"{path}:{line_no}: {error}") from error
    return rasterize_yolo_polygons(path, shape), counts


def audit_dataset(data_path: Path) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = data.get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    if tuple(names or []) != CLASS_NAMES or data.get("nc", 4) != 4:
        raise EvaluationError(f"Dataset class contract mismatch: {names}")
    root = data_path.parent
    base = (root / data.get("path", ".")).resolve()
    samples, audit, errors = {}, {}, []
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        declared = data.get(key)
        if not isinstance(declared, str):
            raise EvaluationError(f"Expected directory path for YAML {key}")
        image_dir = (base / declared).resolve()
        resolution = "yaml-relative"
        # Verified Roboflow export layout, identical to existing Experiment D tool.
        if not image_dir.is_dir() and declared == f"../{split}/images" and not data.get("path"):
            image_dir = root / split / "images"
            resolution = "verified Roboflow ../split/images -> YAML-directory/split/images"
        label_dir = image_dir.parent / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise EvaluationError(f"Missing images/labels: {image_dir}, {label_dir}")
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        if not images or len({p.stem for p in images}) != len(images):
            raise EvaluationError(f"Empty split or duplicate image stem: {split}")
        samples[split] = [Sample(p, label_dir / (p.stem + ".txt"), p.name) for p in images]
        entry = dict(images=len(images), image_dir=str(image_dir.resolve()), label_dir=str(label_dir.resolve()),
                     declared_yaml_path=declared, resolution=resolution, empty_labels=0,
                     multi_class_images=0, class_positive_images=[0] * 4, instances=[0] * 4,
                     missing_labels=[], orphan_labels=[], manifest=[], shapes={},
                     roi_fraction_min=1.0, roi_fraction_max=0.0)
        for sample in samples[split]:
            try:
                image = read_image(sample.image_path)
                _, counts = load_gt(sample.label_path, image.shape[:2])
                entry["empty_labels"] += int(not any(counts))
                entry["multi_class_images"] += int(sum(n > 0 for n in counts) > 1)
                entry["class_positive_images"] = [a + int(b > 0) for a, b in zip(entry["class_positive_images"], counts)]
                entry["instances"] = [a + b for a, b in zip(entry["instances"], counts)]
                shape = str(image.shape[:2])
                entry["shapes"][shape] = entry["shapes"].get(shape, 0) + 1
                fraction = float(roi_mask(image.shape[:2], "django-ellipse").mean())
                entry["roi_fraction_min"] = min(entry["roi_fraction_min"], fraction)
                entry["roi_fraction_max"] = max(entry["roi_fraction_max"], fraction)
                entry["manifest"].append(dict(image=str(sample.image_path.resolve()), image_sha256=sha256_file(sample.image_path),
                                              label=str(sample.label_path.resolve()), label_sha256=sha256_file(sample.label_path)))
            except (ValueError, OSError, EvaluationError) as error:
                errors.append(str(error))
                if not sample.label_path.exists():
                    entry["missing_labels"].append(str(sample.label_path))
        stems = {p.stem for p in images}
        entry["orphan_labels"] = [str(p) for p in sorted(label_dir.glob("*.txt")) if p.stem not in stems]
        errors.extend(f"Orphan label: {p}" for p in entry["orphan_labels"])
        audit[split] = entry
    return samples, dict(splits=audit, errors=errors, class_names=CLASS_NAMES,
                         empty_label_policy="Existing empty .txt = legal negative; missing .txt = fatal data error")


def restore_masks(data: torch.Tensor, shape: tuple[int, int], retina: bool = False) -> np.ndarray:
    if data.ndim != 3 or not torch.isfinite(data).all():
        raise EvaluationError(f"Invalid masks tensor: {data.shape}")
    if retina:
        if tuple(data.shape[-2:]) != tuple(shape):
            raise EvaluationError("retina_masks output does not match original image")
        return data.detach().cpu().numpy() > 0.5
    # Official implementation removes symmetric LetterBox padding before resize.
    # Older versions lack mode=; fail rather than silently change interpolation.
    if "mode" not in inspect.signature(scale_masks).parameters:
        raise EvaluationError("This installed scale_masks lacks mode='nearest'; use a supported environment")
    return scale_masks(data.detach().cpu()[None].float(), shape, mode="nearest")[0].numpy() > 0.5
