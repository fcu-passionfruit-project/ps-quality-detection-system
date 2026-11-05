"""CLI for fixed-setting local-defect area and micro pixel evaluation.

Run from repository root: python -s -m passionfruit.evaluation.evaluate_local_defect_pixels --help
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops

from passionfruit.evaluation.evaluate_experiment_d_area import CLASS_NAMES, EvaluationError, sha256_file
from passionfruit.evaluation.pixel_data import audit_dataset, load_gt, restore_masks, roi_mask
from passionfruit.evaluation.pixel_metrics import measure, summarize_rows, union_instances
from passionfruit.evaluation.pixel_postprocessing import MASK_MODES, django_deployed_masks, load_django_reference
from passionfruit.evaluation.pixel_report import build_report, make_visuals
from passionfruit.inference.image_processing import read_image

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EvaluationError(f"No records to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_reference(path: Path) -> dict[str, Any]:
    """Recognize a local Git LFS pointer without loading/downloading its target."""
    if not path.is_file():
        return dict(kind="missing", file_sha256=None, target_sha256=None)
    file_sha = sha256_file(path)
    if path.stat().st_size < 1024:
        text = path.read_text(encoding="utf-8")
        if text.startswith("version https://git-lfs.github.com/spec/v1\n"):
            fields = dict(line.split(" ", 1) for line in text.splitlines())
            oid = fields.get("oid", "")
            if not oid.startswith("sha256:") or len(oid) != 71:
                raise EvaluationError(f"Invalid Git LFS oid: {path}")
            return dict(kind="git-lfs-pointer", file_sha256=file_sha, target_sha256=oid[7:], target_size=int(fields["size"]))
    return dict(kind="checkpoint", file_sha256=file_sha, target_sha256=file_sha, target_size=path.stat().st_size)


def deployment_profile(root: Path) -> dict[str, Any]:
    settings = root / "Django_Server/FruitSortingSystem/settings.py"
    detection = root / "Django_Server/fruit_app/detection.py"
    tree = ast.parse(settings.read_text(encoding="utf-8"))
    options = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DETECTION_MODEL_OPTIONS" for t in node.targets):
            options = ast.literal_eval(node.value)
    if options is None or options.get("defect") != {"imgsz": 1024}:
        raise EvaluationError("Django defect settings changed; re-audit deployment preset before use")
    return dict(settings=str(settings), settings_sha256=sha256_file(settings),
                detection=str(detection), detection_sha256=sha256_file(detection), model_options=options)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "training/runs/segment/v22_D_context6/weights/best.pt")
    p.add_argument("--data", type=Path, default=ROOT / "training/packages/local_defects/data.yaml", help="YOLO data.yaml (or dataset directory)")
    p.add_argument("--split", nargs="+", choices=("valid", "val", "test"), default=["valid", "test"])
    p.add_argument("--output", type=Path, required=True, help="New empty output directory; existing artifacts are never overwritten")
    p.add_argument("--device", default="auto", help="auto: CUDA -> MPS -> CPU; explicit unavailable device fails")
    p.add_argument("--preset", choices=("django",), default="django", help="Read-only source audit + frozen current deployment options")
    p.add_argument("--django-root", type=Path, default=ROOT.parent / "PassionFruit_IoT_hardware")
    p.add_argument("--roi-mode", choices=("django-ellipse", "full-image"), default="django-ellipse", help="full-image is an explicitly alternative denominator, not production area error")
    p.add_argument("--mask-mode", choices=MASK_MODES, default="scale-masks", help="scale-masks: original unletterbox evaluation; django-deployed: exact current Django mask postprocessing")
    p.add_argument("--conf", type=float, default=0.25, help="Default deployment 0.25; override is marked sensitivity analysis (no automatic search)")
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--limit", type=int, help="Smoke test: first N sorted images per split; keep same batch as final run")
    p.add_argument("--threads", type=int, default=4, help="CPU torch threads (recorded)")
    return p.parse_args(argv)


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise EvaluationError("Requested MPS is unavailable")
    if (device.startswith("cuda") or device.isdigit()) and not torch.cuda.is_available():
        raise EvaluationError("Requested CUDA is unavailable")
    return device


def prediction_fingerprint(result: Any) -> str:
    """Bind paired modes to identical original boxes/masks, including dtype/shape."""
    digest = hashlib.sha256()
    for value in (getattr(result.boxes, "data", None), getattr(result.masks, "data", None)):
        if value is None:
            digest.update(b"none")
        else:
            array = value.detach().cpu().numpy()
            digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace, argv: list[str] | None) -> int:
    if not 0 < args.conf <= 1 or args.batch < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        raise EvaluationError("Invalid conf, batch, threads or limit")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise EvaluationError(f"Output is nonempty; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mask_mode == "django-deployed" and args.roi_mode != "django-ellipse":
            raise EvaluationError("django-deployed requires django-ellipse; full-image is not a deployed ROI definition")
        profile = deployment_profile(args.django_root.resolve())
        reference = load_django_reference(args.django_root.resolve())
        weights = args.weights.expanduser().resolve()
        if not weights.is_file():
            raise EvaluationError(f"Missing weights: {weights}")
        data = args.data.expanduser().resolve()
        if data.is_dir():
            data /= "data.yaml"
        samples, audit = audit_dataset(data)
        write_json(output / "dataset_audit.json", audit)
        if audit["errors"]:
            raise EvaluationError(f"Dataset audit failed ({len(audit['errors'])} errors); see dataset_audit.json")
        torch.set_num_threads(args.threads)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        device = choose_device(args.device)
        model = YOLO(str(weights))
        names = tuple(model.names[i] for i in range(len(model.names)))
        if model.task != "segment" or names != CLASS_NAMES:
            raise EvaluationError(f"Wrong model task/names: {model.task}, {names}")
        infer = dict(imgsz=1024, conf=args.conf, iou=0.7, max_det=300, retina_masks=False,
                     rect=True, augment=False, agnostic_nms=False, half=False, verbose=False,
                     device=device, save=False)
        dep_weight = args.django_root.resolve() / "Django_Server/models/Defect.pt"
        dep_reference = checkpoint_reference(dep_weight)
        dep_sha = dep_reference["file_sha256"]
        splits = list(dict.fromkeys("valid" if s == "val" else s for s in args.split))
        sources = [Path(__file__), Path(__file__).with_name("pixel_data.py"), Path(__file__).with_name("pixel_metrics.py"),
                   Path(__file__).with_name("pixel_report.py"), Path(__file__).with_name("pixel_postprocessing.py"), Path(__file__).with_name("evaluate_experiment_d_area.py"),
                   ROOT / "passionfruit/inference/image_processing.py", Path(ops.__file__)]
        meta = dict(schema_version=2, mask_mode=args.mask_mode, status="running", created_at=datetime.now().astimezone().isoformat(),
                    model_name=weights.parents[1].name, weights=str(weights), weights_sha256=sha256_file(weights),
                    training_seed=model.ckpt.get("train_args", {}).get("seed"), data=str(data), data_sha256=sha256_file(data),
                    deployment_weight=str(dep_weight), deployment_sha256=dep_sha,
                    deployment_sha_matches=dep_sha == sha256_file(weights), deployment_source=profile,
                    deployment_reference=dep_reference,
                    deployment_target_matches=dep_reference["target_sha256"] == sha256_file(weights),
                    python=sys.version.split()[0], platform=platform.platform(), ultralytics=ultralytics.__version__,
                    torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__, device=device,
                    inference=infer, actual_end2end=bool(getattr(model.model, "end2end", False)), roi_mode=args.roi_mode,
                    scope="smoke" if args.limit else "full", limit=args.limit, batch=args.batch, threads=args.threads,
                    seed=0, deterministic_algorithms=True, setting_label="deployment conf" if args.conf == 0.25 else "confidence sensitivity; not deployment result",
                    command="python -s -m passionfruit.evaluation.evaluate_local_defect_pixels " + shlex.join(argv if argv is not None else sys.argv[1:]),
                    source_sha256={str(p): sha256_file(p) for p in sources},
                    zero_division="P=0 when GT>0 and pred=0; P/R=None when both empty; recall=None when GT=0",
                    interpolation="official scale_masks padding removal, nearest, >0.5" if args.mask_mode == "scale-masks" else "Django: cv2.INTER_NEAREST direct resize -> >0.5 -> ROI -> class OR",
                    area_ratio_policy="Primary metrics remain unrounded pixel_count/ROI; django_deployed_area_ratio reproduces min(1,ratio) and round(...,8)",
                    django_reference="Read-only import of original _segmentation_summary; return-frame union locals observed without code/global patches")
        write_json(output / "run_metadata.json", meta)
        rows, alignments, paths, consistency = [], [], {}, []
        for split in splits:
            selected = samples[split][:args.limit] if args.limit else samples[split]
            for start in range(0, len(selected), args.batch):
                chunk = selected[start:start + args.batch]
                images = [read_image(s.image_path) for s in chunk]
                rois = [roi_mask(im.shape[:2], args.roi_mode) for im in images]
                inputs = [cv2.bitwise_and(im, im, mask=r.astype(np.uint8) * 255) if args.roi_mode == "django-ellipse" else im for im, r in zip(images, rois)]
                results = model.predict(source=inputs, **infer)
                if len(results) != len(chunk):
                    raise EvaluationError("Prediction result count does not match image count")
                for sample, image, roi, model_input, result in zip(chunk, images, rois, inputs, results, strict=True):
                    shape = image.shape[:2]
                    if tuple(result.orig_shape) != shape or not np.array_equal(result.orig_img, model_input):
                        raise EvaluationError("Unexpected original image shape")
                    gt, gt_counts = load_gt(sample.label_path, shape)
                    deployed = django_deployed_masks(result, roi, reference)
                    for cid, check in enumerate(deployed.consistency):
                        consistency.append(dict(split=split, image_name=sample.relative_name, class_id=cid, **check))
                    boxes, masks = result.boxes, result.masks
                    count = len(boxes) if boxes is not None else 0
                    raw_shape = None
                    if count:
                        if masks is None or len(masks.data) != count:
                            raise EvaluationError("Prediction boxes/masks mismatch")
                        classes = boxes.cls.detach().cpu().numpy()
                        confidence = boxes.conf.detach().cpu().numpy()
                        if not np.isfinite(confidence).all() or np.any(confidence < args.conf - 1e-7):
                            raise EvaluationError("Unexpected prediction confidences")
                        raw_shape = list(masks.data.shape)
                        restored = restore_masks(masks.data, shape)
                        scale_unions = union_instances(restored, classes, shape)
                    else:
                        if masks is not None and len(masks.data):
                            raise EvaluationError("Masks exist without boxes")
                        classes = np.empty(0)
                        restored = np.zeros((0, *shape), dtype=bool)
                        scale_unions = [np.zeros(shape, dtype=bool) for _ in CLASS_NAMES]
                    unions = scale_unions if args.mask_mode == "scale-masks" else deployed.unions
                    instances = restored if args.mask_mode == "scale-masks" else deployed.instances
                    cache = Path("mask_cache") / split / (sample.image_path.stem + ".npz")
                    (output / cache).parent.mkdir(parents=True, exist_ok=True)
                    gt_stack = np.stack([gt[name] & roi for name in CLASS_NAMES])
                    pred_stack = np.stack([u & roi for u in unions])
                    scale_stack = np.stack([u & roi for u in scale_unions])
                    django_stack = np.stack(deployed.unions)
                    np.savez_compressed(output / cache, gt=gt_stack, pred=pred_stack, roi=roi,
                                        scale_masks_pred=scale_stack, django_deployed_pred=django_stack)
                    paths[(split, sample.relative_name)] = sample.image_path
                    alignment = dict(split=split, image_name=sample.relative_name, original_height=shape[0], original_width=shape[1],
                                     mask_mode=args.mask_mode, raw_prediction_sha256=prediction_fingerprint(result),
                                     raw_mask_shape=json.dumps(raw_shape), instances=count, restoration=meta["interpolation"])
                    alignments.append(alignment)
                    for cid, name in enumerate(CLASS_NAMES):
                        row = dict(split=split, image_name=sample.relative_name, model_name=meta["model_name"], mask_mode=args.mask_mode, class_id=cid, class_name=name)
                        row.update(measure(gt[name], unions[cid], roi))
                        direct_roi = django_stack[cid]
                        intersecting_instances = sum(bool((instance & gt_stack[cid]).any()) for instance in instances[classes == cid]) if count else 0
                        row.update(pred_instance_count=int((classes == cid).sum()), gt_instance_count=gt_counts[cid],
                                   pred_instances_intersecting_gt_count=intersecting_instances,
                                   roi_definition=args.roi_mode, mask_cache=str(cache),
                                   django_direct_resize_area_ratio=float(direct_roi.sum() / roi.sum()),
                                   django_deployed_area_ratio=deployed.feature_ratios[name],
                                   django_deployed_absolute_area_error=abs(deployed.feature_ratios[name] - row["gt_area_ratio"]),
                                   aligned_minus_django_area_ratio=float((int(scale_stack[cid].sum()) - int(direct_roi.sum())) / roi.sum()),
                                   alignment_xor_pixels=int((scale_stack[cid] ^ direct_roi).sum()))
                        rows.append(row)
                print(f"[{split}] {min(start+args.batch,len(selected))}/{len(selected)}", flush=True)
        write_csv(output / "per_image_class_metrics.csv", rows)
        with (output / "per_image_class_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            summary = summarize_rows(list(csv.DictReader(stream)))
        if summary != summarize_rows(rows):
            raise EvaluationError("CSV roundtrip summary mismatch")
        write_csv(output / "per_class_summary.csv", summary)
        write_csv(output / "mask_alignment.csv", alignments)
        write_csv(output / "django_consistency.csv", consistency)
        verification = dict(status="passed", images=len(alignments), image_class_checks=len(consistency),
                            union_mismatched_pixels=sum(r["union_xor_pixels"] for r in consistency),
                            all_feature_ratios_exact_match=all(r["ratio_exact_match"] for r in consistency),
                            max_feature_rounding_difference=max(abs(r["raw_ratio"]-r["django_feature_ratio"]) for r in consistency),
                            scope="Same inference Results and supplied ROI only; not full end-to-end acceptance")
        write_json(output / "django_consistency.json", verification)
        selections = make_visuals(output, rows, paths)
        meta.update(evaluated_split_counts={s: sum(r["split"] == s for r in rows) // 4 for s in splits},
                    actual_predictor_args=vars(model.predictor.args), csv_roundtrip_summary_verified=True,
                    django_consistency=verification)
        build_report(output, meta, audit, summary, rows, selections)
        meta.update(status="completed", completed_at=datetime.now().astimezone().isoformat())
        write_json(output / "run_metadata.json", meta)
        print(f"Completed: {output}")
        return 0
    except Exception as error:
        write_json(output / "failure.json", dict(status="failed", error_type=type(error).__name__, message=str(error)))
        metadata_path = output / "run_metadata.json"
        if metadata_path.is_file():
            failed_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            failed_meta.update(status="failed", error=str(error))
            write_json(metadata_path, failed_meta)
        raise


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv), argv)


if __name__ == "__main__":
    raise SystemExit(main())
