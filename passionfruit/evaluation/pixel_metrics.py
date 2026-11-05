"""Class-union pixel metrics; no instance matching or IoU threshold."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from passionfruit.evaluation.evaluate_experiment_d_area import CLASS_NAMES, EvaluationError


def pixel_scores(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Undefined precision is 0 only when GT exists; empty/empty stays N/A."""
    return {
        "Pixel_Precision": tp / (tp + fp) if tp + fp else (0.0 if fn else None),
        "Pixel_Recall": tp / (tp + fn) if tp + fn else None,
        "Pixel_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def union_instances(masks: np.ndarray, classes: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    if masks.shape != (len(classes), *shape):
        raise EvaluationError(f"mask/class shape mismatch: {masks.shape}, {classes.shape}, {shape}")
    if not np.isfinite(classes).all() or np.any(classes != classes.astype(int)) or np.any((classes < 0) | (classes >= 4)):
        raise EvaluationError("Invalid prediction class ID")
    return [np.any(masks[classes == c], axis=0) for c in range(4)]


def measure(gt: np.ndarray, pred: np.ndarray, roi: np.ndarray) -> dict[str, Any]:
    if gt.shape != pred.shape or gt.shape != roi.shape or gt.ndim != 2:
        raise EvaluationError("GT/pred/ROI must share the original 2D coordinate grid")
    roi = roi.astype(bool)
    n = int(roi.sum())
    if n == 0:
        raise EvaluationError("資料錯誤：有效 ROI 面積為零")
    g, p = gt.astype(bool) & roi, pred.astype(bool) & roi
    ga, pa, tp = int(g.sum()), int(p.sum()), int((g & p).sum())
    fp, fn = pa - tp, ga - tp
    signed = (pa - ga) / n
    row = dict(roi_pixels=n, gt_area_pixels=ga, pred_area_pixels=pa,
               intersection_pixels=tp, fp_pixels=fp, fn_pixels=fn,
               gt_area_ratio=ga / n, pred_area_ratio=pa / n,
               absolute_area_error=abs(signed), signed_area_error=signed,
               gt_present=int(ga > 0), pred_present=int(pa > 0),
               relative_area_error=abs(pa - ga) / ga if ga else None,
               gt_pixels_outside_roi=int((gt.astype(bool) & ~roi).sum()),
               pred_pixels_outside_roi=int((pred.astype(bool) & ~roi).sum()))
    row.update(pixel_scores(tp, fp, fn))
    row.update(gt_area_percent=100 * ga / n, pred_area_percent=100 * pa / n,
               absolute_area_error_pp=100 * abs(signed), signed_area_error_pp=100 * signed)
    return row


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Works on live records and csv.DictReader rows, using identical arithmetic."""
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["model_name"], row.get("mask_mode", "scale-masks"), int(row["class_id"]))].append(row)
    output = []
    for (split, model, mode, cid), group in sorted(groups.items()):
        positive = [r for r in group if int(r["gt_present"])]
        negative = [r for r in group if not int(r["gt_present"])]
        def avg(subset, field):
            return sum(float(r[field]) for r in subset) / len(subset) if subset else None
        row = dict(split=split, model_name=model, mask_mode=mode, class_id=cid, class_name=CLASS_NAMES[cid],
                   evaluated_images=len(group), MAE_all=avg(group, "absolute_area_error"),
                   MAE_gt_positive=avg(positive, "absolute_area_error"),
                   MAE_gt_negative=avg(negative, "absolute_area_error"),
                   Mean_Bias_Error=avg(group, "signed_area_error"),
                   gt_positive_images=len(positive), gt_negative_images=len(negative),
                   pred_positive_images=sum(int(r["pred_present"]) for r in group),
                   missed_gt_positive_images=sum(not int(r["pred_present"]) for r in positive),
                   false_positive_gt_negative_images=sum(int(r["pred_present"]) for r in negative),
                   gt_positive_relative_area_error=avg(positive, "relative_area_error"))
        row["false_positive_gt_negative_rate"] = row["false_positive_gt_negative_images"] / len(negative) if negative else None
        for target, source in (("total_gt_pixels", "gt_area_pixels"), ("total_pred_pixels", "pred_area_pixels"),
                               ("total_intersection_pixels", "intersection_pixels"), ("total_fp_pixels", "fp_pixels"), ("total_fn_pixels", "fn_pixels")):
            row[target] = sum(int(r[source]) for r in group)
        row.update({"Micro_" + k: v for k, v in pixel_scores(row["total_intersection_pixels"], row["total_fp_pixels"], row["total_fn_pixels"]).items()})
        for key in ("MAE_all", "MAE_gt_positive", "MAE_gt_negative", "Mean_Bias_Error"):
            row[key + "_pp"] = row[key] * 100 if row[key] is not None else None
        output.append(row)
    return output
