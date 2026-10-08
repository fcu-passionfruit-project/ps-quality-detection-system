# 程式導覽與路徑遷移

## 先選要做的事

所有指令都從 repository 根目錄執行，先啟用 `PF` 或相容 Python 環境。使用 `python -s -m`，不需安裝本專案，不直接執行巢狀 `.py` 檔案。Windows 可將 `python` 換成 `C:\Users\qoqoo\anaconda3\envs\PF\python.exe`。

| 工作 | 模組入口 | 輸入 | 輸出／寫入位置 |
| --- | --- | --- | --- |
| GUI 前處理及模型驗證 | `apps.inspection_gui` | 使用者選取的照片、模型與參數 | 使用者指定位置的 ROI、JSON、階段圖；第 4 分頁可維護 Manifest 及產生 CSV |
| 決策資料集 | `apps.decision_dataset_cli` | `layout`、`manifest`、`generate` 子指令；預設 DecisionLayerTrain 與 models/manifest.yaml | layout 套用結構；manifest 更新及 .bak；CSV 預設 decision_datasets/，依 version 分目錄 |
| Class-aware sampling | `passionfruit.data_preparation.generate_class_aware_train_list` | `--dataset`、類別曝光倍數 | 訓練圖片清單；`--output` 可指定清單位置，原圖片與標註不變 |
| Targeted crops | `passionfruit.data_preparation.generate_targeted_crops` | `--dataset`、`--weight`、`--experiment C/D`；D 使用 C 的 paired artifacts | 裁切圖與標註、train list、data YAML、manifest 及 validation report；可指定 output-dir、train-list、data-yaml |
| v19／v20／v21 比較 | `passionfruit.evaluation.analyze_targeted_crop_experiment` | `--repo` 下既有三組 run、資料集、best.pt | `training/analysis/v19-v20-v21-targeted-crop`；以 `--output` 改位置 |
| Experiment D 面積評估 | `passionfruit.evaluation.evaluate_experiment_d_area` | `--data`、v22／v23／v24 的 best.pt、validation/test | 預設 `analysis/experiment_d_area_accuracy_corrected_20260904`；report、CSV、圖表與 prediction cache，支援 `--resume` |
| EigenCAM | `passionfruit.explainability.eigencam` | 必填 `--model`、`--image`、`--output` | 熱力圖、遮蔽比較、NumPy 數值及 cam_metrics.json，寫入指定 output |
| D-RISE | `passionfruit.explainability.drise` | 必填 `--model`、`--source`；依需要選 target-index／target-class | 預設 outputs/drise 下獨立執行目錄，可用 `--output` 調整；熱力圖及紀錄 |

例如 `python -s -m passionfruit.explainability.eigencam --help`。除 GUI 外，每個入口都提供 `--help`。完整決策資料集指令見[應用程式手冊](../apps/README.md)。

## 結構與依賴

- `apps/`：GUI 與 CLI 入口。只呼叫核心，不保存另一份演算法。
- `passionfruit/inference/`：模型載入、影像前處理、單模型與多模型 Pipeline。
- `passionfruit/decision_dataset/`：結構、Manifest、CSV，使用 inference。Tkinter adapter 留在 gui.py。
- `passionfruit/data_preparation/`：訓練前的資料處理，不自動啟動訓練。
- `passionfruit/evaluation/`：對既有模型做定量評估，面積評估重用 inference 遮罩。
- `passionfruit/explainability/`：熱力圖與解釋工具；EigenCAM 使用內附 third_party/yolo_cam。
- `tests/`：依上述用途分類的自動化測試。

基礎環境依根目錄 README 安裝 CUDA PyTorch 與 requirements.txt。GUI 需要 Tkinter；影像與推論需要 OpenCV、NumPy、Pillow、PyYAML、Ultralytics；評估圖表需要 Matplotlib；EigenCAM 另需 ttach、torchvision。D-RISE 額外需 vision-explanation-methods 及其執行依賴。

EigenCAM 不再搜尋外部 YOLO-26-CAM checkout。第三方來源及授權現況見[內附程式說明](../passionfruit/third_party/yolo_cam/README.md)。

各工具保留原本參數與 device 預設；例如 v19／v20／v21 分析預設 mps，Windows 使用者應明確傳 `--device cpu` 或可用的 CUDA 裝置。EigenCAM 目前只接受 cpu／mps。不要因目錄整理而假設所有工具的裝置選項相同。

## 程式與產物分開

| 位置 | 用途 | 是否由本次整理搬動 |
| --- | --- | --- |
| tests/ | 自動化測試原始碼 | 依用途分類 |
| training/packages/ | 訓練資料 | 否 |
| training/runs/ | 訓練曲線、args.yaml、results.csv、weights/best.pt | 否 |
| training/test/ | 已訓練模型的測試輸出 | 否 |
| training/analysis/、analysis/、outputs/ | 各工具現行預設分析產物 | 否，輸出預設保持原樣 |
| models/、datasets/、DecisionLayerTrain/ | 模型、資料集及決策照片 | 否 |
| docs/experiments/ | 歷史實驗報告 | 否，保留當時路徑與命令 |

Google Drive 管理的資料與權重不會隨 Git pull 下載。舊 main/ 與 tools/ 已確認只剩可重建快取、desktop.ini，以及與根目錄相同的 VS Code 設定，並已清除。docs/test-doc/ 的既有本機文件也保留。

## 舊路徑對照

新命令統一使用 `python -s -m <模組>`，不保留舊入口。外部 Colab 或個人筆記需依此表更新。歷史報告內的指令用來記錄當時環境，不能直接視為目前操作指令。

| 舊位置 | 新位置 |
| --- | --- |
| main/main.py | apps/inspection_gui.py |
| main/decision_dataset_cli.py | apps/decision_dataset_cli.py |
| main/pipeline.py、model_manager.py、image_processing.py | passionfruit/inference/ 下同名檔案 |
| main/decision_dataset/ | passionfruit/decision_dataset/ |
| tools/generate_class_aware_train_list.py、generate_targeted_crops.py | passionfruit/data_preparation/ 下同名檔案 |
| tools/analyze_targeted_crop_experiment.py、evaluate_experiment_d_area.py | passionfruit/evaluation/ 下同名檔案 |
| tools/yolo26_cam_test.py | passionfruit/explainability/eigencam.py |
| tools/yolo26_drise.py、tools/drise/yolo_seg.py | passionfruit/explainability/drise.py |
| tools/yolo_cam/ | passionfruit/third_party/yolo_cam/ |
| main/README.md | apps/README.md |
| tests/test_*.py | tests/ 對應用途子目錄；GUI 摘要測試移至 apps/ |
| tools/augment_rare_classes.py | 先前已移除；本次移除其過時測試，不提供替代入口 |

## 測試與操作範例

```powershell
python -s -m apps.inspection_gui
python -s -m apps.decision_dataset_cli layout --dry-run
python -s -m apps.decision_dataset_cli generate --version v001 --validate-only
python -s -m passionfruit.data_preparation.generate_class_aware_train_list --help
python -s -m passionfruit.data_preparation.generate_targeted_crops --help
python -s -m passionfruit.evaluation.analyze_targeted_crop_experiment --help
python -s -m passionfruit.evaluation.evaluate_experiment_d_area --help
python -s -m passionfruit.explainability.eigencam --help
python -s -m passionfruit.explainability.drise --help
python -s -m unittest discover -s tests -t .
```

GUI 四個分頁保留；資料集產生仍要求通過驗證的 Model Manifest。自動化測試以暫存資料驗證，不代表模型品質或硬體整合已驗收。

## 此次整理驗證

- 搬移前：32 項測試，6 個錯誤、1 個失敗；部分模組尚未能載入，因此不代表全部測試案例數。
- 已修復舊匯入與 mock 路徑、Windows 跨磁碟權重紀錄、測試 UTF-8 讀取及 OpenCV 單通道維度差異。
- Python 3.14 的 D-RISE 依賴使用 pandas 3.0.6 wheel；較舊 Python 保留 pandas 2.3.2，由 requirements-drise.txt 的環境條件選取。
- GUI 驗證限於建立與四分頁切換，運算行為由暫存資料及 mock 測試驗證；未重跑完整 GPU 模型評估或硬體整合。
- 整理後：51 項自動化測試全部通過，沒有 skip；7 個 CLI 的 `--help` 通過，全部應用與核心模組匯入不啟動 GUI 或載入 YOLO。
- GUI 四分頁建立與切換通過；14,266 個本機資料檔的路徑、大小與修改時間一致，歷史實驗報告的 Git 內容未變。
