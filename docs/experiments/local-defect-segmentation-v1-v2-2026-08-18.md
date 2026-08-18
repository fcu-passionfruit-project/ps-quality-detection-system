# 局部瑕疵 segmentation v1／v2 實驗紀錄

- 實驗日期：2026-08-17 至 2026-08-18
- 紀錄日期：2026-08-18
- 任務：百香果局部瑕疵 instance segmentation
- 結論：兩版模型都學到部分瑕疵特徵，但目前 Dataset 的類別分布、來源識別與 split 完整性不足以支援正式驗收。`models/v2_defect.pt` 保留為實驗權重，不核准為正式 `defect` 模型。

## 實驗目的

比較相同 Dataset 與模型架構下，將訓練 `imgsz` 從 `640` 提高至 `1024` 是否能改善小型局部瑕疵的 segmentation 表現。

這次實驗未形成只改 `imgsz` 的受控 A／B 測試。v2 同時將 `batch` 從固定 `16` 改成自動估算的 `-1`，checkpoint 顯示實際使用 `batch=1`。因此 v1 與 v2 的差異包含輸入尺寸與 batch size，不能把結果完全歸因於解析度。

## Dataset

資料來源為 Roboflow 專案 `local-defects-pk6gn` 的版本 `1`，匯出名稱為 `v1_defect`：

- 匯出時間：2026-08-17 12:38 GMT
- 格式：YOLO26
- 類別：`abrasion`、`anthracnose`、`insect_bite`、`insect_track`
- 前處理：Resize to `640 × 640`，採 Fit within，實際匯出影像會保留長寬比，長邊最多為 `640 px`
- train augmentation：水平／垂直翻轉、90 度旋轉與亮度 ±10%，每張來源影像產生 3 個版本
- 設定檔：`training/packages/v1_defect/data.yaml`

### Split 與 instance 分布

| split | 影像數 | 空標註影像 | `abrasion` | `anthracnose` | `insect_bite` | `insect_track` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1356 | 430 | 1742 | 97 | 36 | 1211 |
| valid | 97 | 31 | 133 | 12 | 1 | 76 |
| test | 97 | 33 | 114 | 16 | 1 | 100 |

`insect_bite` 在 valid 與 test 各只有 1 個 instance，任何 precision、recall 或 AP 都會受單一樣本主導，不能代表泛化能力。train 的 `36` 個 instance 又已包含 augmentation，原始獨立案例數更少。

### Split 來源識別問題

依可解析的 `fruit_<id>` 檔名檢查，已找到下列 Fruit 跨 split：

- train／valid：`fruit_083`、`fruit_122`
- train／test：`fruit_095`、`fruit_135`、`fruit_189`、`fruit_269`

另有 `1433` 張影像檔名不含 Fruit ID：train `1257` 張、valid `86` 張、test `90` 張。這些影像無法只靠匯出檔名驗證是否來自相同 Fruit，因此目前不能排除更多資料洩漏。

## 訓練設定

兩版共同使用 `YOLO26s-seg`、`100 epochs`、`seed=0`、`deterministic=true`、相同 Dataset 與相同 augmentation 設定。完整參數分別保存在：

- `training/runs/segment/v1_defect/args.yaml`
- `training/runs/segment/v2_defect/args.yaml`

| 項目 | v1 | v2 |
| --- | ---: | ---: |
| 模型 | `yolo26s-seg.pt` | `yolo26s-seg.pt` |
| `imgsz` | 640 | 1024 |
| `batch` 設定 | 16 | -1 |
| checkpoint 實際 batch | 16 | 1 |
| epochs | 100 | 100 |
| 總時間 | 3456.64 秒，約 57.6 分鐘 | 22825 秒，約 380.4 分鐘 |
| 相對耗時 | 1.0× | 6.6× |

Roboflow 匯出影像的長邊最多只有 `640 px`。v2 設為 `1024` 時，訓練流程會放大既有像素，不會恢復原始影像已經失去的細節。

## 指標結果

### 訓練保存的 best checkpoint

| 指標 | v1 best | v2 best |
| --- | ---: | ---: |
| Box precision | 0.55709 | 0.25761 |
| Box recall | 0.34715 | 0.29261 |
| Box mAP50 | 0.29295 | 0.22674 |
| Box mAP50-95 | 0.14675 | 0.11986 |
| Mask precision | 0.55196 | 0.24984 |
| Mask recall | 0.31400 | 0.28744 |
| Mask mAP50 | 0.27991 | 0.22871 |
| Mask mAP50-95 | 0.12460 | 0.11357 |

訓練 loss 持續下降，代表模型確實在擬合訓練資料；validation loss 後期回升且 AP 沒有同步改善，顯示泛化開始惡化。v2 增加約 `6.6×` 訓練時間後，best checkpoint 的主要 Box 與 Mask 指標仍低於 v1。

### 固定驗證流程重跑

為了取得可比較的逐類結果，兩個 `best.pt` 都在同一個 valid split、`batch=1`、`workers=0`、相同 Ultralytics `8.4.46` 與 NVIDIA GeForce RTX 3070 上重跑。v2 另外以 `imgsz=640` 驗證一次，觀察相同推論尺寸下的表現。

| 權重／驗證尺寸 | Mask precision | Mask recall | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| v1／640 | 0.55235 | 0.31600 | 0.27669 | 0.12730 |
| v2／1024 | 0.24875 | 0.29073 | 0.22871 | 0.11367 |
| v2／640 | 0.35830 | 0.24534 | 0.24992 | 0.13084 |

v2 在 `640` 推論時的 Mask mAP50-95 比 v1 高 `0.00354`，但 valid 只有 `97` 張，且 `insect_bite` 只有 1 個 instance。這個差距不足以認定 v2 有穩定改善。

### 逐類 Mask 指標

| 權重／驗證尺寸 | 類別 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| v1／640 | `abrasion` | 0.443 | 0.414 | 0.366 | 0.12433 |
| v1／640 | `anthracnose` | 0.462 | 0.574 | 0.488 | 0.26875 |
| v1／640 | `insect_bite` | 1.000 | 0.000 | 0.0199 | 0.01592 |
| v1／640 | `insect_track` | 0.304 | 0.276 | 0.233 | 0.10019 |
| v2／1024 | `abrasion` | 0.390 | 0.391 | 0.324 | 0.11515 |
| v2／1024 | `anthracnose` | 0.435 | 0.417 | 0.432 | 0.28231 |
| v2／1024 | `insect_bite` | 0.000 | 0.000 | 0.000 | 0.00000 |
| v2／1024 | `insect_track` | 0.170 | 0.355 | 0.158 | 0.05722 |

`insect_bite` 的 Recall 皆為 `0`。這項結果主要反映 valid 只有單一 instance，現階段無法判斷模型是否能學習此類別。`abrasion` 與 `insect_track` 已有可量測訊號，但仍有大量漏偵測。

## 混淆矩陣解讀

Ultralytics `8.4.46` 的 segmentation validator 會把預測交給 `ConfusionMatrix.process_batch`。該函式使用預測與 ground truth 的 bounding box IoU 配對，預設圖表不直接衡量 mask polygon 的重疊品質。

- 橫軸是 True class，縱軸是 Predicted class。
- `background` 列代表 false negative，也就是 ground truth 沒有被配對到預測。
- `background` 欄代表 false positive，也就是預測沒有配對到 ground truth。
- 類別對角線偏低代表該類別的 box 配對成功率低，但仍需搭配 Mask precision、recall、mAP50 與 mAP50-95 判讀 segmentation 品質。

原始圖表保存在：

- `training/runs/segment/v1_defect/confusion_matrix.png`
- `training/runs/segment/v1_defect/confusion_matrix_normalized.png`
- `training/runs/segment/v2_defect/confusion_matrix.png`
- `training/runs/segment/v2_defect/confusion_matrix_normalized.png`

## 權重追溯與核准狀態

| 檔案 | SHA-256 |
| --- | --- |
| `training/runs/segment/v1_defect/weights/best.pt` | `3031fdac2261fa4ba154f5c503d457cb46531b93873a7c42aed0c2e1eec109a7` |
| `training/runs/segment/v2_defect/weights/best.pt` | `6733cd31fb70be05c284e0cb59c522d0ed94ad931ab9ebe0dd20dd339e5ffecf` |
| `models/v2_defect.pt` | `6733cd31fb70be05c284e0cb59c522d0ed94ad931ab9ebe0dd20dd339e5ffecf` |

`models/v2_defect.pt` 與 v2 `best.pt` 的 SHA-256 相同，來源已確認。檔案位於 `models/` 不等於通過驗收；目前不把它寫入 `models/manifest.yaml` 的 `approved_for_use` role。

## 診斷結論

1. 兩版模型已有學習訊號，`abrasion`、`anthracnose` 與 `insect_track` 都出現非零 Mask AP。
2. v2 使用的影像沒有超過 `640 px` 的原始細節，`imgsz=1024` 主要增加計算量。
3. v1 與 v2 同時改變 `imgsz` 與 batch size，無法把差異單獨歸因於解析度。
4. `insect_bite` 樣本嚴重不足，valid／test 各 1 個 instance，該類別無法驗收。
5. 已確認至少 6 個具 Fruit ID 的來源跨 split，另有大量影像缺少可追溯 ID。現有 validation／test 結果可能受資料洩漏影響。
6. v2 沒有可靠、全面且足以抵銷成本的改善，因此維持實驗狀態。

## 下一版實驗決策

下一版先重建 Dataset，再測試解析度或模型超參數：

1. 從原始照片建立 `fruit_id + view_id` 對照，無法確認來源的影像先隔離，不直接分入新 Dataset。
2. 先按 Fruit ID 分組切分 train／valid／test，再只對 train 套用 augmentation，阻擋同一 Fruit 或其增強版本跨 split。
3. 補充 `insect_bite` 與 `anthracnose` 的獨立原始案例；valid 與 test 必須各自包含足以計算逐類指標的多個獨立 Fruit，不再接受單一 instance 驗收。
4. 人工抽查四類 polygon，特別核對 `abrasion` 與 `insect_track` 的定義邊界、漏標與重疊標註。抽查結果完成前，不把標註品質視為已確認根因。
5. 若要測試 `imgsz=1024`，從長邊至少 `1024 px` 的原始影像重新匯出，避免先縮至 `640` 再放大。
6. 下一次 A／B 實驗固定相同 model、split、seed、batch、augmentation、epochs、驗證尺寸與 confidence 設定，一次只改一個變因。
7. 超參數凍結後才使用獨立 test split；報告逐類 Box／Mask 指標，並額外量測 `defect_mask_pixels / fruit_roi_pixels` 的誤差與偏差，對齊 Decision Dataset 的正式特徵用途。

正式核准前，模型至少必須滿足下列流程條件：

- Dataset provenance 可追溯，Fruit 跨 split 數量為 `0`。
- 四個類別在 valid 與 test 都有多個獨立案例，沒有無法解讀的單一 instance 類別。
- 新模型在相同驗證條件下與既有 baseline 比較，報告逐類結果，不能只報 macro average。
- test 評估與實際瑕疵面積比例誤差完成後，才透過 Model Manifest 流程更新為 `candidate` 並進一步核准。

## 重現驗證

以下命令是本紀錄使用的固定驗證流程：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -c "from ultralytics import YOLO; m=YOLO(r'training/runs/segment/v1_defect/weights/best.pt'); r=m.val(data=r'training/packages/v1_defect/data.yaml', split='val', imgsz=640, batch=1, workers=0, device=0, plots=False, save_json=False, verbose=True, project=r'training/runs/segment', name='diagnostic_v1', exist_ok=True); print(r.results_dict); print('MAPS', r.seg.maps.tolist())"

C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -c "from ultralytics import YOLO; m=YOLO(r'training/runs/segment/v2_defect/weights/best.pt'); r=m.val(data=r'training/packages/v1_defect/data.yaml', split='val', imgsz=1024, batch=1, workers=0, device=0, plots=False, save_json=False, verbose=True, project=r'training/runs/segment', name='diagnostic_v2', exist_ok=True); print(r.results_dict); print('MAPS', r.seg.maps.tolist())"

C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -c "from ultralytics import YOLO; m=YOLO(r'training/runs/segment/v2_defect/weights/best.pt'); r=m.val(data=r'training/packages/v1_defect/data.yaml', split='val', imgsz=640, batch=1, workers=0, device=0, plots=False, save_json=False, verbose=False, project=r'training/runs/segment', name='diagnostic_v2_at640', exist_ok=True); print(r.results_dict); print('MAPS', r.seg.maps.tolist())"
```

## Repository 邊界

這份紀錄屬於模型 Repository，因為它描述 Dataset、訓練 run、checkpoint、模型評估與後續標註工作。

硬體 Repository `D:\PassionFruit_IoT_hardware` 繼續維護 Django、ESP32 Firmware、硬體安全、資料採集流程與 AI adapter 契約。這次實驗沒有改變四類瑕疵、mask 面積比例、人工覆核或「模型不得直接控制硬體」等既有契約，因此不修改硬體 Repository 的 `DATA_COLLECTION_SPEC.md`、`PROJECT_CONTEXT.md` 或 ADR。
