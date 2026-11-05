# 模型與實驗評估

`analyze_targeted_crop_experiment.py` 比較 v19／v20／v21；`evaluate_experiment_d_area.py` 比較 D0／D1／D2 的瑕疵面積精度及 threshold。

入口：`python -s -m passionfruit.evaluation.analyze_targeted_crop_experiment --help`、`python -s -m passionfruit.evaluation.evaluate_experiment_d_area --help`。

依賴既有模型、資料集、PyTorch、Ultralytics、Matplotlib；面積評估重用 inference 的遮罩規則。評估產物寫入原本指定的輸出目錄，不放在這裡。

## 固定部署設定的逐類像素評估

`evaluate_local_defect_pixels.py` 固定 conf=0.25、imgsz=1024、iou=0.7、max_det=300、
retina_masks=False，先同類聯集，再計算面積 MAE 與 Micro Pixel Precision/Recall/F1。
`pixel_data.py` 負責完整標註稽核與官方 unletterbox，`pixel_metrics.py` 負責數值，
`pixel_report.py` 負責繁體中文報告與代表圖。部分 parser／hash／常數重用既有 Experiment D 模組。

從 repository root 執行（本機可將 python 換為 `.venv/bin/python`）：

```bash
# Smoke：各 split 前六張，先人工檢視輸出 visualizations/ 再完整跑
python -s -m passionfruit.evaluation.evaluate_local_defect_pixels \
  --weights training/runs/segment/v22_D_context6/weights/best.pt \
  --data training/packages/local_defects/data.yaml --split valid test \
  --output ../local_defect_pixel_analysis_new/smoke --device cpu --limit 6

# 完整 valid/test；每次指定全新空目錄
python -s -m passionfruit.evaluation.evaluate_local_defect_pixels \
  --weights training/runs/segment/v22_D_context6/weights/best.pt \
  --data training/packages/local_defects/data.yaml --split valid test \
  --output ../local_defect_pixel_analysis_new/full --device auto

# 其他模型；使用相同資料、device、batch、ROI 與 conf 以便比較
python -s -m passionfruit.evaluation.evaluate_local_defect_pixels \
  --weights training/runs/segment/v23_D_context6_seed1/weights/best.pt \
  --data training/packages/local_defects/data.yaml --split valid test \
  --output ../local_defect_pixel_analysis_v23 --device cpu
```

`--django-root` 可指定硬體 repository（預設 sibling `PassionFruit_IoT_hardware`），只讀設定與來源，
只讀呼叫原始純 helper 作一致性核對，不初始化 Django 或啟動服務。`--preset django` 為預設。`--roi-mode django-ellipse` 重建正式 crop 幾何遮罩；
`--roi-mode full-image` 是替代分母，不能稱為正式特徵誤差。
`--conf` 非 0.25 會標記為 sensitivity，沒有自動選參。

本工具以官方 `scale_masks(..., mode="nearest")` 去 padding；已驗證 Ultralytics 8.4.122，
缺少 mode 參數的版本明確報錯，不自動安裝／升降級。
Django 現有面積 summary 直接 resize 而未去 padding。新 `--mask-mode` 將這兩種處理明確分開：

```bash
# 精確重現 Django 後處理；第一次先加 --limit 6，完整評估用新的 output 移除 limit
python -s -m passionfruit.evaluation.evaluate_local_defect_pixels \
  --mask-mode django-deployed --split valid test --device cpu \
  --output ../局部瑕疵模型評估分析報告/new_run/django-deployed

# 保留原版 scale_masks 方法（也是預設）；不覆寫任何舊結果
python -s -m passionfruit.evaluation.evaluate_local_defect_pixels \
  --mask-mode scale-masks --split valid test --device cpu \
  --output ../局部瑕疵模型評估分析報告/new_run/scale-masks
```

django-deployed 依原 dtype 直接 INTER_NEAREST resize → >0.5 → ROI 與同類聯集，
`django_deployed_area_ratio` 再套用原碼 min(1,ratio) 與 round(...,8)。主 MAE 保留未捨入像素比的原定義。
`pixel_postprocessing.py` 直接唯讀載入原始 `_segmentation_summary`（不改 Django），
以其 return-frame 實際聯集比對獨立相容實作，每圖四類都必須完全一致。
`django_consistency.csv/json` 記錄像素／特徵一致性；mask cache 同時保存兩模式聯集，
`mask_alignment.csv` 的原始輸出 fingerprint 支持同推論結果的配對比較。
django-deployed 不接受 full-image ROI；資料集歷史 crop／原圖缺漏與現場 batch grouping 的限制仍存在，不能稱為完整端到端驗收。

產物包含 `per_image_class_metrics.csv`、`per_class_summary.csv`、`report.md`、
`dataset_audit.json`、`run_metadata.json`、`mask_alignment.csv`、`mask_cache/`、
`visualizations/` 與 `visualization_index.json`。CSV 空欄為 N/A；summary 從 per-image 重讀重算並驗證。
完整定義與限制見[規格](../../docs/specs/local-defect-pixel-evaluation.md)。
