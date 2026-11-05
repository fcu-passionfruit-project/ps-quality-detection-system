"""Traditional Chinese report and representative class-union visualizations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from passionfruit.inference.image_processing import read_image, write_image


def fmt(value: Any, scale: float = 1.0) -> str:
    return "N/A" if value is None else f"{float(value) * scale:.4f}"


def render_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, roi: np.ndarray,
                 row: dict[str, Any], path: Path) -> None:
    tp, fp, fn = gt & pred, pred & ~gt, gt & ~pred
    panels = [image.copy() for _ in range(4)]
    contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panels[0], contours, -1, (0, 255, 255), 2)
    for panel, mask in ((panels[1], gt), (panels[2], pred)):
        panel[mask] = (panel[mask] * 0.25 + np.array([255, 255, 255]) * 0.75).astype(np.uint8)
    for mask, color in ((tp, (0, 210, 0)), (fp, (0, 0, 255)), (fn, (255, 100, 0))):
        panels[3][mask] = color
    titles = ["Original ROI (yellow=ellipse)", "GT union (white)", "Prediction union (white)", "TP green / FP red / FN blue"]
    output = np.full((560, 1680, 3), 24, dtype=np.uint8)
    cv2.putText(output, f"{row.get('mask_mode', 'scale-masks')} | {row['split']} | {row['class_name']} | {row['image_name'][:90]}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(output, f"GT {row['gt_area_percent']:.3f}% | Pred {row['pred_area_percent']:.3f}% | AE {row['absolute_area_error_pp']:.3f} pp | GT instances {row['gt_instance_count']} | Pred instances {row['pred_instance_count']}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        h, w = panel.shape[:2]
        ratio = min(410 / w, 445 / h)
        resized = cv2.resize(panel, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)
        x, y = i * 420 + 5, 96
        output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(output, title, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    write_image(path, output)


def make_visuals(output: Path, rows: list[dict[str, Any]], sample_paths: dict[tuple[str, str], Path]) -> list[dict[str, Any]]:
    selections = []
    # These rules illustrate outcomes, never select thresholds or ROI definitions.
    rules = (
        ("mostly_covered", lambda r: r["gt_present"] and r["pred_present"], lambda r: r["Pixel_F1"] or 0, True),
        ("single_gt_multiple_predictions_candidate", lambda r: r["gt_instance_count"] == 1 and r["pred_instances_intersecting_gt_count"] >= 2, lambda r: r["intersection_pixels"], True),
        ("overestimate", lambda r: r["gt_present"] and r["signed_area_error"] > 0, lambda r: r["signed_area_error"], True),
        ("underestimate", lambda r: r["gt_present"] and r["pred_present"] and r["signed_area_error"] < 0, lambda r: r["signed_area_error"], False),
        ("false_positive", lambda r: not r["gt_present"] and r["pred_present"], lambda r: r["absolute_area_error"], True),
        ("complete_miss", lambda r: r["gt_present"] and not r["pred_present"], lambda r: r["gt_area_ratio"], True),
    )
    for split in sorted({r["split"] for r in rows}):
        for name, predicate, score, reverse in rules:
            candidates = sorted((r for r in rows if r["split"] == split and predicate(r)), key=score, reverse=reverse)
            if not candidates:
                selections.append(dict(split=split, scenario=name, status="no eligible example"))
                continue
            row = candidates[0]
            with np.load(output / row["mask_cache"]) as cache:
                c = row["class_id"]
                gt, pred, roi = cache["gt"][c], cache["pred"][c], cache["roi"]
            path = output / "visualizations" / f"{split}_{name}_{row['class_name']}.png"
            render_panel(read_image(sample_paths[(split, row["image_name"])]), gt, pred, roi, row, path)
            selections.append(dict(split=split, scenario=name, status="rendered", file=str(path.relative_to(output)),
                                   image=row["image_name"], class_name=row["class_name"]))
    (output / "visualization_index.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    return selections


def build_report(output: Path, meta: dict[str, Any], audit: dict[str, Any], summaries: list[dict[str, Any]],
                 rows: list[dict[str, Any]], selections: list[dict[str, Any]]) -> None:
    mode = meta.get("mask_mode", "scale-masks")
    lines = [f"# 局部瑕疵面積與像素層級評估：{mode}", "", f"Mask 後處理模式：**{mode}**。舊版未記錄 mask_mode 的原始結果屬於 scale-masks；原始檔案保持不變。", "", "## 目的與證據範圍", "",
             "以單張 ROI 圖片、單一瑕疵類別為單位，評估同類別所有實例的像素聯集。目的為檢查可見瑕疵面積特徵，供後續 Image Grade 研究使用。本次沒有訓練模型，也未驗證 XGBoost 分級、整顆 Fruit Grade 或實體表面積。", "",
             "## 模型、環境與資料", "", f"- 模型：`{meta['model_name']}`；權重：`{meta['weights']}`。",
             f"- SHA-256：`{meta['weights_sha256']}`；checkpoint training seed：`{meta['training_seed']}`。",
             f"- Python {meta['python']}；Ultralytics {meta['ultralytics']}；Torch {meta['torch']}；OpenCV {meta['opencv']}；NumPy {meta['numpy']}；device={meta['device']}。",
             f"- Django Defect.pt：`{meta['deployment_weight']}`；本機檔案種類：**{meta['deployment_reference']['kind']}**；其 checkpoint 本體或 LFS 目標 SHA 與本次評估權重相同：**{meta['deployment_target_matches']}**。若為 LFS pointer，本機並無可載入的 Django 權重本體；本次直接讀取指定 checkpoint，不代表 Django runtime 已驗收。",
             f"- YAML：`{meta['data']}`；類別固定 0 abrasion、1 anthracnose、2 insect_bite、3 insect_track，已核對 dataset 與 checkpoint。", "",
             "| split | 實際圖片 | 空標註 | 多類圖片 | 四類 GT-positive 圖片（未裁 ROI） |",
             "|---|---:|---:|---:|---|" ]
    for split, entry in audit["splits"].items():
        lines.append(f"| {split} | {entry['images']} | {entry['empty_labels']} | {entry['multi_class_images']} | {entry['class_positive_images']} |")
    lines += ["", "標註均依 YOLO normalized segmentation polygons 逐行解析，保留全部類別。既存空 .txt 視為合法陰性；缺檔、孤立標註、無效 class、座標、退化 polygon 會記錄並終止評估。完整資料稽核（含 train）不進行任何訓練／擴增。Roboflow YAML 的 ../split/images 在此本機目錄不成立，沿用既有工具核實的 YAML-directory/split/images；實際路徑與解析規則逐 split 記在 dataset_audit.json。", "",
              "## 實際推論與 ROI", "", f"固定推論參數：`{json.dumps(meta['inference'], ensure_ascii=False)}`。",
              "Django settings.py 只指定 defect imgsz=1024；_predict_batch 未指定 conf，YOLO.predict 的預設為 0.25；iou=0.7、max_det=300、retina_masks=False 等由已安裝版本預設提供。本工具明確凍結這些值，完整最終 predictor.args 記在 run_metadata.json。NMS iou 不是像素評估門檻；YOLO26 end-to-end 模式可能不走傳統 NMS，actual_end2end 另行記錄。",
              "推論使用 BGR ROI、橢圓遮罩（django-ellipse 模式）、無 CLAHE，batch 預設 3、rect=True。資料集缺乏每顆果實原始三視角 grouping，因此依排序每三張處理，不能保證與現場三張輸入的 batch padding 完全相同。沒有重新定位或裁切既存 ROI。", "",
              f"面積定義：**{meta['roi_mode']}**。django-ellipse 以 crop 寬高、中心 (W//2,H//2)、半軸 max(1,int(W/2×1.01)) 與 max(1,int(H/2×1.01))，OpenCV filled ellipse 產生。分母是該 binary mask 非零像素數，非 W×H、非 πab 解析面積、非亮度閾值、非精確果皮分割。full-image 是明確命名的替代分母，不能稱為正式系統誤差。",
              "新工具直接重用 inference.create_circular_mask，與 Django _prepare_roi 的幾何公式相同。GT 與 prediction 都裁至同一 mask，ROI=0 為資料錯誤。資料已是 ROI，但缺乏逐圖原始 crop bbox／前處理版本，無法證明歷史 ROI 未被額外 resize 或裁切；因此結果僅支持『在現有 ROI 重建 Django 幾何定義下』的面積估計，不是端到端原始照片特徵誤差。", "",
              "## 兩種 Mask 後處理與正式流程", "",
              "retina_masks=False 時 masks.data 是 letterbox 輸入座標的 N×H×W tensor，並非保證等於原 ROI。scale-masks 使用官方 scale_masks(..., mode='nearest') 去 padding 後還原原圖。django-deployed 則精確重現 _segmentation_summary：原 dtype 直接 cv2.resize(..., INTER_NEAREST)，再 >0.5，再限制 roi_mask>0 並作同類 Boolean OR；不移除 padding。每個 Mask 依 result.names[boxes.cls] 保留類別語意，不使用 bbox 或 polygon 近似。",
              f"本次主指標選用 **{mode}** 的聯集。兩種聯集均由同一份 Results 計算並寫入每圖 cache；alignment_xor_pixels 與 aligned_minus_django_area_ratio 永遠表示 scale-masks 對 django-deployed 的差異，不代表模型改善。正式 Django 原始碼未修改。",
              "正式面積特徵為 round(min(1.0, union_pixels / roi_pixels), 8)。django_deployed_area_ratio 與 django_deployed_absolute_area_error 保留此實際捨入行為；主 pred_area_ratio/MAE 為維持既有公式仍用未捨入像素比，GT 同樣不捨入，兩者不得混稱。",
              f"逐圖直接呼叫原始 Django helper 的一致性結果：**{meta['django_consistency']['image_class_checks']}** 個 image-class；聯集差異像素 **{meta['django_consistency']['union_mismatched_pixels']}**；8 位捨入特徵比例全部精確相同：**{meta['django_consistency']['all_feature_ratios_exact_match']}**。參考函式不回傳 unions，工具以暫時的 return-frame profiler 唯讀取得其執行時 union_masks，不改原函式、AST、globals 或 cv2。詳見 django_consistency.csv/json。",
              "官方方法參考：[Ultralytics scale_masks](https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.scale_masks)；可重現性以 run_metadata 記錄的本機 source SHA 與版本為準，網頁可能更新。", "",
              "## 指標定義與零分母", "",
              "Pᵢ,c 為同類 prediction masks 的 Boolean OR，Gᵢ,c 為同類 GT polygons rasterized 後的 OR；兩者均限制在 Rᵢ。不同類即使重疊亦獨立計算。同像素於同類只計一次，沒有實例匹配或 IoU 0.5 判定。",
              "令 Dᵢ=|Rᵢ|，預測比例=|P|/D，GT比例=|G|/D，AE=||P|−|G||/D，signed error=(|P|−|G|)/D。MAE_all 為全部圖片 AE 的平均；MAE_gt_positive、MAE_gt_negative 分別在 GT>0 與 GT=0 子集取平均；Mean_Bias_Error 為所有圖片 signed error 的平均。ratio×100 為面積百分比；error ratio×100 為百分點 pp（非相對百分誤差）。",
              "TP=|P∩G|、FP=|P−G|、FN=|G−P|。先跨圖片累加，再計算 Micro Pixel Precision=ΣTP/(ΣTP+ΣFP)、Recall=ΣTP/(ΣTP+ΣFN)、F1=2ΣTP/(2ΣTP+ΣFP+ΣFN)。",
              "GT 與 prediction 全空：AE=0、P/R/F1=N/A；有 GT 卻無 prediction：P=0、R=0、F1=0；有 prediction 但全無 GT：P=0、R=N/A、F1=0。子集無樣本為 N/A，無 epsilon。CSV 空欄代表 N/A，JSON 用 null。微平均以整個 split 的 pixel totals 判定零分母，不平均每圖 P/R。", "",
              "## valid/test 結果", "",
              "MAE 與 bias 表中以 pp 顯示；完整 ratio 及所有子集、像素總數在 CSV。",
              "| split | 類別 | N | GT+ | MAE all pp | MAE GT+ pp | MAE GT− pp | bias pp | Micro P | Micro R | Micro F1 | 漏檢 GT+ | 誤檢 GT− |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" ]
    for r in summaries:
        lines.append(f"| {r['split']} | {r['class_name']} | {r['evaluated_images']} | {r['gt_positive_images']} | {fmt(r['MAE_all'],100)} | {fmt(r['MAE_gt_positive'],100)} | {fmt(r['MAE_gt_negative'],100)} | {fmt(r['Mean_Bias_Error'],100)} | {fmt(r['Micro_Pixel_Precision'])} | {fmt(r['Micro_Pixel_Recall'])} | {fmt(r['Micro_Pixel_F1'])} | {r['missed_gt_positive_images']} | {r['false_positive_gt_negative_images']} / {r['gt_negative_images']} |")
    lines += ["", "## 各類別分析", ""]
    for r in summaries:
        tendency = "高估" if r["Mean_Bias_Error"] > 0 else "低估" if r["Mean_Bias_Error"] < 0 else "平均無偏差"
        lines.append(f"- {r['split']} {r['class_name']}：平均{tendency}（{fmt(r['Mean_Bias_Error'],100)} pp），GT+ MAE={fmt(r['MAE_gt_positive'],100)} pp；完全漏檢 {r['missed_gt_positive_images']}/{r['gt_positive_images']}；GT− 誤檢率={fmt(r['false_positive_gt_negative_rate'],100)}%。面積接近仍可能位置錯誤，需與 Micro P/R 同時解讀。")
    affected = sum(r["alignment_xor_pixels"] > 0 for r in rows)
    maximum = max((abs(r["aligned_minus_django_area_ratio"]) for r in rows), default=0)
    lines += ["", f"scale-masks 相較 django-deployed：{affected}/{len(rows)} 個 image-class 的聯集不同；面積比例最大絕對差 {maximum:.8f}（{maximum*100:.4f} pp）。這是後處理差異診斷，不是模型改善。", "",
              "## 人工檢查圖片", "", "白色為指定類別聯集；TP 綠、FP 紅、FN 藍，黃色為 ROI 邊界。one-GT/multiple-predictions 候選要求至少兩個預測各有像素落在該類 GT 內，仍不以實例匹配證明每個預測都屬於同一病灶。若沒有符合情境的案例則明列缺少，不偽造。"]
    for item in selections:
        lines.append(f"- {item['split']} {item['scenario']}：" + (f"[{item['class_name']}]({item['file']})" if item["status"] == "rendered" else "無符合條件案例"))
    lines += ["", "## 與 Mask mAP 的差異、限制與結論", "",
              "Mask mAP 評估各 confidence 下的實例匹配與 IoU 門檻；本工具評估同類聯集的像素重疊及可見面積。實例拆分不直接扣分，但錯類、缺少與多餘像素仍形成 FN/FP。面積 MAE=0 也可能位置完全錯誤。這些指標互補，不能直接聲稱優於 mAP。本文不引用條件尚未逐項核對的歷史 mAP 數值。",
              "MAE_all 可能因大量 GT-negative 圖片而偏小，尤其蟲咬必須同時呈現 GT+ MAE、召回與樣本數。沒有外部場域、獨立果實身份／跨 split 洩漏完整審核、標註者一致性、實體面積真值或分級誤差傳遞實驗。本次圖像集合是現有固定 split，未重切，沒有在 test 挑選 confidence、ROI 或其他超參數。",
              "可支持的結論為：在指定權重、固定 conf、目前資料與幾何 ROI 下，工具量化了各類可見面積誤差及位置重疊；是否達到分級需求，仍需事先定義可接受誤差與下游 XGBoost 實驗。不能由 Pixel Precision/Recall 宣稱整體分級準確率，也不能對失敗類別泛稱具備可靠面積估計能力。", "",
              "## 重現與產物", "", "```bash", meta["command"], "```", "",
              "per_image_class_metrics.csv 每圖四列；per_class_summary.csv 直接由前者重讀重算並驗證一致。mask_cache 保存原圖座標 GT/pred 聯集與 ROI，可稽核像素總數；dataset_audit.json 保存圖／標註 SHA；run_metadata.json 保存權重、程式、環境與實際 args；mask_alignment.csv 保存座標診斷；visualization_index.json 說明案例選擇。", "",
              "## 第四章數據整理建議", "", "1. 方法小節列出 ROI 幾何定義、conf=0.25、逐類聯集與 micro 公式，註明座標還原和現行 Django 面積後處理有差異。",
              "2. valid/test 分表，各類至少列 N、GT+、MAE_all/GT+/GT−（pp）、bias（pp）、Micro P/R，避免少數類低 MAE 被誤解為高品質。",
              "3. 配成功、過估、低估、漏檢與誤檢圖；實例拆分案例標示 GT/pred 個數及 union 指標。",
              "4. 將結論限於單視角可見面積估計，另列 XGBoost 分級驗證、少數類資料量與正式後處理一致性為待驗證項目。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
