"""Read-only Django reference and compatible ROI-level Mask postprocessing.

The deployed summary exposes ratios but not union masks. A temporary profiler
observes its return-frame locals to copy the actual unions, without rewriting
its source, AST, globals, OpenCV calls or result tensors.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from passionfruit.evaluation.evaluate_experiment_d_area import CLASS_NAMES, EvaluationError, sha256_file

MASK_MODES = ("scale-masks", "django-deployed")


def load_django_reference(root: Path) -> ModuleType:
    """Load the actual pure helper module, with no sys.path or Django setup.

    detection imports dataset_store and grading, whose module-level code is
    definitions/constants only. No models, hardware, views or settings execute.
    Disable bytecode writes while loading all these original files.
    """
    app = root / "Django_Server/fruit_app"
    source = app / "detection.py"
    name = "_pf_evaluation_django_" + sha256_file(source)[:16]
    if name + ".detection" in sys.modules:
        return sys.modules[name + ".detection"]
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, app / "__init__.py", submodule_search_locations=[str(app)])
        if spec is None or spec.loader is None:
            raise EvaluationError(f"Cannot load original Django helpers: {app}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        return importlib.import_module(name + ".detection")
    finally:
        sys.dont_write_bytecode = previous


def reference_summary(reference: ModuleType, result: Any, roi: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    """Execute unmodified _segmentation_summary and observe its actual unions."""
    target = reference._segmentation_summary
    captured: dict[str, np.ndarray] = {}
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if frame.f_code is target.__code__ and event == "return":
            captured.update({name: mask.copy() for name, mask in frame.f_locals.get("union_masks", {}).items()})
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        summary = target(result, 0.0, roi)
    finally:
        sys.setprofile(previous)
    # The original empty-detection branch does not allocate union_masks.
    for name in CLASS_NAMES:
        captured.setdefault(name, np.zeros(roi.shape, dtype=bool))
    return summary, captured


@dataclass
class DeployedMasks:
    instances: np.ndarray
    unions: list[np.ndarray]
    feature_ratios: dict[str, float]
    consistency: list[dict[str, Any]]


def django_deployed_masks(result: Any, roi: np.ndarray, reference: ModuleType) -> DeployedMasks:
    """Match Django exactly on valid inputs and verify against its real helper.

    Keep dtype until cv2.resize, then strict >0.5, then ROI and class union.
    Rounded feature ratios are separate from unrounded count/ROI metric ratios.
    Evaluation deliberately rejects zero ROI even for an empty prediction,
    whereas the deployed helper checks its denominator only when boxes exist.
    """
    if roi.ndim != 2 or np.count_nonzero(roi) <= 0:
        raise EvaluationError("資料錯誤：有效 ROI 面積為零或非 2D")
    unions = {name: np.zeros(roi.shape, dtype=bool) for name in CLASS_NAMES}
    instances = []
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is not None and len(boxes):
        if masks is None or getattr(masks, "data", None) is None:
            raise EvaluationError("Prediction has boxes without masks")
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidence = boxes.conf.detach().cpu().numpy()
        raw = masks.data.detach().cpu().numpy()
        if len(classes) != len(raw) or len(classes) != len(confidence):
            raise EvaluationError("Prediction boxes/masks/confidences length mismatch")
        for class_index, instance in zip(classes, raw, strict=True):
            name = str(result.names[int(class_index)])
            if name not in unions:
                raise EvaluationError(f"Unknown defect class: {name}")
            # Exact original operation order; do NOT cast fractional masks to uint8.
            resized = cv2.resize(instance, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = (resized > 0.5) & (roi > 0)
            unions[name] |= valid
            instances.append(valid)
    denominator = int(np.count_nonzero(roi))
    ratios = {name: round(min(1.0, np.count_nonzero(union) / denominator), 8) for name, union in unions.items()}
    original, observed = reference_summary(reference, result, roi)
    checks = []
    for name in CLASS_NAMES:
        difference = int(np.count_nonzero(unions[name] ^ observed[name]))
        check = dict(class_name=name, union_xor_pixels=difference,
                     evaluator_feature_ratio=ratios[name], django_feature_ratio=original["area_ratios"][name],
                     ratio_exact_match=ratios[name] == original["area_ratios"][name],
                     raw_ratio=np.count_nonzero(unions[name]) / denominator)
        if difference or not check["ratio_exact_match"]:
            raise EvaluationError(f"Django compatibility mismatch: {check}")
        checks.append(check)
    stack = np.stack(instances) if instances else np.zeros((0, *roi.shape), dtype=bool)
    return DeployedMasks(stack, [unions[name] for name in CLASS_NAMES], ratios, checks)
