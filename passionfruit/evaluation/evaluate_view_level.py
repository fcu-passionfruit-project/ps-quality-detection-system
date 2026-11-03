"""以決策層需要的「視角層級」指標評估局部瑕疵 segmentation 模型。

XGBoost 決策層只使用每張照片的四類瑕疵面積比例，因此本工具不以 instance mAP
排名，而是評估：
1. 有無判定：每張照片每一類瑕疵「有／無」的 Sensitivity、Specificity、F1。
2. 面積估計：預測面積比例與標註面積比例的 MAE 與 Spearman 相關。
3. 嚴重度一致性：面積比例分成 無／小／中／大 四級後的 quadratic weighted kappa。

confidence threshold 只在 valid 選擇，test 只報告一次並附 bootstrap 95% CI。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from passionfruit.evaluation.evaluate_experiment_d_area import (
    CLASS_NAMES,
    dataset_roi_mask,
    rasterize_yolo_polygons,
)
from passionfruit.inference.image_processing import read_image

THRESHOLDS = (0.01, 0.02, 0.03, 0.04) + tuple(round(value, 2) for value in np.arange(0.05, 0.81, 0.05))
SEVERITY_LEVELS = ("none", "small", "medium", "large")
# valid 的 insect_bite 只有 1 個 GT，不能用來選 threshold，只報告。
SELECTION_CLASSES = ("abrasion", "anthracnose", "insect_track")
BOOTSTRAP_ROUNDS = 2000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class EvaluationError(RuntimeError):
    """輸入資料或模型輸出不符合評估契約。"""


def _split_images(dataset_root: Path, split: str) -> list[Path]:
    image_dir = dataset_root / split / "images"
    if not image_dir.is_dir():
        raise EvaluationError(f"找不到 {split} images：{image_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise EvaluationError(f"{image_dir} 沒有影像。")
    return images


def _label_path(image_path: Path) -> Path:
    return image_path.parents[1] / "labels" / f"{image_path.stem}.txt"


def _gt_area_ratios(image_path: Path) -> dict[str, float]:
    image = read_image(image_path)
    shape = image.shape[:2]
    roi = dataset_roi_mask(shape)
    denominator = int(np.count_nonzero(roi))
    masks = rasterize_yolo_polygons(_label_path(image_path), shape)
    return {name: np.count_nonzero(masks[name] & roi) / denominator for name in CLASS_NAMES}


def severity_bounds(dataset_root: Path) -> dict[str, tuple[float, float]]:
    """以 train split 標註面積（僅正樣本）的 33／67 百分位定義 小／中／大。"""
    positives: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    for image_path in _split_images(dataset_root, "train"):
        for name, ratio in _gt_area_ratios(image_path).items():
            if ratio > 0:
                positives[name].append(ratio)
    bounds = {}
    for name, values in positives.items():
        if len(values) < 3:
            raise EvaluationError(f"train 中 {name} 正樣本不足，無法定義嚴重度分級。")
        low, high = np.percentile(values, [100 / 3, 200 / 3])
        bounds[name] = (float(low), float(high))
    return bounds


def severity_level(ratio: float, bounds: tuple[float, float]) -> int:
    if ratio <= 0:
        return 0
    if ratio <= bounds[0]:
        return 1
    if ratio <= bounds[1]:
        return 2
    return 3


def _predicted_area_ratios(result: Any, shape: tuple[int, int]) -> dict[str, list[float]]:
    """回傳每類在每個 threshold 下的預測 union 面積比例。"""
    roi = dataset_roi_mask(shape)
    denominator = int(np.count_nonzero(roi))
    areas = {name: [0.0] * len(THRESHOLDS) for name in CLASS_NAMES}
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return areas
    if result.masks is None:
        raise EvaluationError(f"{result.path} 有 boxes 但沒有 masks。")
    masks = result.masks.data.cpu().numpy() > 0.5
    if masks.shape[1:] != shape:
        raise EvaluationError(f"{result.path} mask 尺寸 {masks.shape[1:]} 與原圖 {shape} 不符；需 retina_masks=True。")
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    for class_id, name in enumerate(CLASS_NAMES):
        selected = np.flatnonzero(classes == class_id)
        for index, threshold in enumerate(THRESHOLDS):
            kept = selected[confidences[selected] >= threshold]
            if kept.size == 0:
                continue
            union = np.logical_or.reduce(masks[kept], axis=0) & roi
            areas[name][index] = np.count_nonzero(union) / denominator
    return areas


def infer_split(weights: Path, imgsz: int, images: Sequence[Path], device: str) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = tuple(model.names[index] for index in sorted(model.names))
    if names != CLASS_NAMES:
        raise EvaluationError(f"{weights} 類別 {names} 與預期 {CLASS_NAMES} 不符。")
    records = []
    for image_path in images:
        result = model.predict(
            str(image_path), imgsz=imgsz, conf=min(THRESHOLDS), iou=0.7, max_det=300,
            retina_masks=True, device=device, verbose=False,
        )[0]
        shape = result.orig_shape
        records.append({
            "image": image_path.name,
            "gt": _gt_area_ratios(image_path),
            "pred": _predicted_area_ratios(result, tuple(shape)),
        })
    return records


def quadratic_weighted_kappa(truth: Sequence[int], pred: Sequence[int], levels: int = 4) -> float:
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    observed = np.zeros((levels, levels))
    np.add.at(observed, (truth, pred), 1)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    weights = np.subtract.outer(np.arange(levels), np.arange(levels)) ** 2 / (levels - 1) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1 - (weights * observed).sum() / denominator)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a_rank = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    b_rank = np.argsort(np.argsort(b, kind="stable"), kind="stable")
    if np.std(a_rank) == 0 or np.std(b_rank) == 0:
        return float("nan")
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def class_metrics(records: Sequence[dict], name: str, threshold_index: int, bounds: tuple[float, float]) -> dict[str, float]:
    gt = np.array([r["gt"][name] for r in records])
    pred = np.array([r["pred"][name][threshold_index] for r in records])
    gt_pos, pred_pos = gt > 0, pred > 0
    tp = int(np.sum(gt_pos & pred_pos))
    fp = int(np.sum(~gt_pos & pred_pos))
    fn = int(np.sum(gt_pos & ~pred_pos))
    tn = int(np.sum(~gt_pos & ~pred_pos))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {
        "gt_positive_images": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sensitivity, "specificity": specificity, "f1": f1,
        "area_mae_pct": float(np.mean(np.abs(pred - gt)) * 100),
        "area_mae_positive_pct": float(np.mean(np.abs(pred[gt_pos] - gt[gt_pos])) * 100) if gt_pos.any() else float("nan"),
        "area_spearman": spearman(gt, pred),
        "severity_qwk": quadratic_weighted_kappa(
            [severity_level(v, bounds) for v in gt], [severity_level(v, bounds) for v in pred]
        ),
    }


# 每類瑕疵使用的 THRESHOLDS index。
Plan = dict[str, int]


def any_defect_metrics(records: Sequence[dict], plan: Plan) -> dict[str, float]:
    gt = np.array([any(v > 0 for v in r["gt"].values()) for r in records])
    pred = np.array([any(r["pred"][name][plan[name]] > 0 for name in CLASS_NAMES) for r in records])
    tp, fp = int(np.sum(gt & pred)), int(np.sum(~gt & pred))
    fn, tn = int(np.sum(gt & ~pred)), int(np.sum(~gt & ~pred))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"),
    }


def macro_qwk(records: Sequence[dict], plan: Plan, bounds: dict[str, tuple[float, float]]) -> float:
    values = [class_metrics(records, name, plan[name], bounds[name])["severity_qwk"] for name in SELECTION_CLASSES]
    return float(np.nanmean(values))


def uniform_plan(index: int) -> Plan:
    return {name: index for name in CLASS_NAMES}


def select_global_plan(valid_records: Sequence[dict], bounds: dict[str, tuple[float, float]]) -> Plan:
    scores = [macro_qwk(valid_records, uniform_plan(index), bounds) for index in range(len(THRESHOLDS))]
    return uniform_plan(int(np.nanargmax(scores)))


def select_per_class_plan(valid_records: Sequence[dict], bounds: dict[str, tuple[float, float]]) -> Plan:
    """每類各自取 valid QWK 最高的 threshold；同分時取較高 threshold 以減少誤報。

    insect_bite 在 valid 樣本不足，沿用全域 threshold。
    """
    plan = dict(select_global_plan(valid_records, bounds))
    for name in SELECTION_CLASSES:
        scores = np.array([
            class_metrics(valid_records, name, index, bounds[name])["severity_qwk"] for index in range(len(THRESHOLDS))
        ])
        if np.all(np.isnan(scores)):
            continue
        plan[name] = int(np.flatnonzero(np.isclose(scores, np.nanmax(scores)))[-1])
    return plan


def bootstrap_ci(records: Sequence[dict], statistic, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = [records[i] for i in rng.integers(0, len(records), len(records))]
        value = statistic(sample)
        if np.isfinite(value):
            values.append(value)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_difference(base: Sequence[dict], base_plan: Plan, new: Sequence[dict], new_plan: Plan, bounds) -> tuple[float, float, float]:
    """同一批影像重抽樣，回傳 (new - base) Macro QWK 差異的 95% CI 與 new 較佳的比例。"""
    if [r["image"] for r in base] != [r["image"] for r in new]:
        raise EvaluationError("paired bootstrap 需要兩個模型使用相同影像順序。")
    rng = np.random.default_rng(0)
    differences = []
    for _ in range(BOOTSTRAP_ROUNDS):
        indices = rng.integers(0, len(base), len(base))
        difference = macro_qwk([new[i] for i in indices], new_plan, bounds) - macro_qwk([base[i] for i in indices], base_plan, bounds)
        if np.isfinite(difference):
            differences.append(difference)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high), float(np.mean(np.asarray(differences) > 0))


def _model_imgsz(weights: Path) -> int:
    args_path = weights.parents[1] / "args.yaml"
    if not args_path.is_file():
        raise EvaluationError(f"找不到 {args_path}，無法得知訓練 imgsz。")
    return int(yaml.safe_load(args_path.read_text(encoding="utf-8"))["imgsz"])


def _format(value: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def _variant_report(records: dict[str, list[dict]], plan: Plan, bounds) -> dict[str, Any]:
    report: dict[str, Any] = {"thresholds": {name: THRESHOLDS[index] for name, index in plan.items()}, "splits": {}}
    for split, split_records in records.items():
        report["splits"][split] = {
            "classes": {name: class_metrics(split_records, name, plan[name], bounds[name]) for name in CLASS_NAMES},
            "any_defect": any_defect_metrics(split_records, plan),
            "macro_qwk": macro_qwk(split_records, plan, bounds),
        }
    test = records["test"]
    report["test_ci95"] = {
        "macro_qwk": bootstrap_ci(test, lambda s: macro_qwk(s, plan, bounds)),
        "any_defect_f1": bootstrap_ci(test, lambda s: any_defect_metrics(s, plan)["f1"]),
    }
    return report


def evaluate(dataset_root: Path, models: dict[str, Path], output_dir: Path, device: str) -> dict[str, Any]:
    """每個模型各評估「全域 threshold」與「逐類 threshold」兩種設定。

    第一個模型的全域設定視為 baseline，其餘設定都和它做配對比較。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = severity_bounds(dataset_root)
    splits = {split: _split_images(dataset_root, split) for split in ("valid", "test")}
    report: dict[str, Any] = {"severity_bounds_pct": {k: [v * 100 for v in b] for k, b in bounds.items()}, "variants": {}}
    variants: dict[str, tuple[list[dict], Plan]] = {}
    rows = []
    for model_name, weights in models.items():
        imgsz = _model_imgsz(weights)
        records = {split: infer_split(weights, imgsz, images, device) for split, images in splits.items()}
        plans = {"global": select_global_plan(records["valid"], bounds), "per_class": select_per_class_plan(records["valid"], bounds)}
        for mode, plan in plans.items():
            variant = f"{model_name}/{mode}"
            variants[variant] = (records["test"], plan)
            report["variants"][variant] = {"weights": str(weights), "imgsz": imgsz, **_variant_report(records, plan, bounds)}
            for split, data in report["variants"][variant]["splits"].items():
                for name, metrics in data["classes"].items():
                    rows.append({"variant": variant, "split": split, "class": name, "threshold": THRESHOLDS[plan[name]], **metrics})
        with (output_dir / f"{model_name}_per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "image", "class", "gt_area_ratio", *[f"pred_conf_{t}" for t in THRESHOLDS]])
            for split, split_records in records.items():
                for record in split_records:
                    for name in CLASS_NAMES:
                        writer.writerow([split, record["image"], name, record["gt"][name], *record["pred"][name]])

    base_name = next(iter(variants))
    base_records, base_plan = variants[base_name]
    report["paired_vs_baseline"] = {}
    for name, (records, plan) in list(variants.items())[1:]:
        low, high, win = paired_difference(base_records, base_plan, records, plan, bounds)
        report["paired_vs_baseline"][f"{name} - {base_name}"] = {"ci95": [low, high], "p_better": win}

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# 視角層級評估結果", "", "嚴重度分級界線（train 正樣本面積比例 33/67 百分位，單位 %）：", ""]
    lines += ["| 類別 | 小／中 | 中／大 |", "| --- | ---: | ---: |"]
    for name, (low, high) in report["severity_bounds_pct"].items():
        lines.append(f"| {name} | {low:.3f} | {high:.3f} |")
    for variant, data in report["variants"].items():
        lines += ["", f"## {variant}（imgsz={data['imgsz']}）", ""]
        for split in ("valid", "test"):
            split_data = data["splits"][split]
            any_defect = split_data["any_defect"]
            lines += [
                f"### {split}", "",
                f"- 任一瑕疵：Sensitivity {_format(any_defect['sensitivity'])}、Specificity {_format(any_defect['specificity'])}、F1 {_format(any_defect['f1'])}",
                f"- Macro 嚴重度 QWK（{', '.join(SELECTION_CLASSES)}）：{_format(split_data['macro_qwk'])}", "",
                "| 類別 | conf | GT 正樣本圖 | Sens. | Spec. | F1 | 面積 MAE(%) | 正樣本面積 MAE(%) | Spearman | 嚴重度 QWK |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for name, m in split_data["classes"].items():
                lines.append(
                    f"| {name} | {data['thresholds'][name]} | {m['gt_positive_images']} | {_format(m['sensitivity'])} | {_format(m['specificity'])} | {_format(m['f1'])} | "
                    f"{_format(m['area_mae_pct'], 4)} | {_format(m['area_mae_positive_pct'], 4)} | {_format(m['area_spearman'])} | {_format(m['severity_qwk'])} |"
                )
            lines.append("")
        ci = data["test_ci95"]
        lines.append(f"test 95% CI：Macro QWK [{ci['macro_qwk'][0]:.3f}, {ci['macro_qwk'][1]:.3f}]、任一瑕疵 F1 [{ci['any_defect_f1'][0]:.3f}, {ci['any_defect_f1'][1]:.3f}]")
    lines += ["", "## 配對 bootstrap（test，Macro QWK 差異）", "", "| 比較 | 95% CI | 較佳比例 |", "| --- | --- | ---: |"]
    for comparison, paired in report["paired_vs_baseline"].items():
        lines.append(f"| {comparison} | [{paired['ci95'][0]:.3f}, {paired['ci95'][1]:.3f}] | {paired['p_better']:.1%} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="含 train/valid/test 的 YOLO segmentation Dataset 根目錄。")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=WEIGHTS",
                        help="可重複；第一個模型的全域 threshold 結果視為 baseline。")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = {}
    for item in args.model:
        name, separator, path = item.partition("=")
        if not separator or not Path(path).is_file():
            raise SystemExit(f"--model 格式須為 NAME=WEIGHTS 且權重存在：{item}")
        models[name] = Path(path)
    try:
        evaluate(args.dataset, models, args.output, args.device)
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print((args.output / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
