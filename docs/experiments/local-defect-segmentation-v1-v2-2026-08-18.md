# `v1_defect` Dataset 第 1／2 次 segmentation 訓練實驗紀錄

- 實驗日期：2026-08-17 至 2026-08-18
- 紀錄日期：2026-08-18
- 任務：百香果局部瑕疵 instance segmentation
- 結論：兩次訓練都使用同一份 `training/packages/v1_defect`，模型有學到部分瑕疵特徵，但類別分布與驗證樣本數不足以支援正式驗收。兩個權重都已封存，不核准為正式 `defect` 模型。

## 實驗目的

比較同一份 `v1_defect` Dataset 與相同模型架構下，將訓練 `imgsz` 從 `640` 提高至 `1024` 是否能改善小型局部瑕疵的 segmentation 表現。

這次實驗未形成只改 `imgsz` 的受控 A／B 測試。第 2 次訓練同時將 `batch` 從固定 `16` 改成自動估算的 `-1`，checkpoint 顯示實際使用 `batch=1`。因此兩次訓練的差異包含輸入尺寸與 batch size，不能把結果完全歸因於解析度。

## Dataset

資料來源為 Roboflow 專案 `local-defects-pk6gn` 的版本 `1`，匯出名稱為 `v1_defect`：

- 匯出時間：2026-08-17 12:38 GMT
- 格式：YOLO26
- 類別：`abrasion`、`anthracnose`、`insect_bite`、`insect_track`
- 前處理：Resize to `640 × 640`，採 Fit within，實際匯出影像會保留長寬比，長邊最多為 `640 px`
- train augmentation：水平／垂直翻轉、90 度旋轉與亮度 ±10%，每張來源影像產生 3 個版本
- 設定檔：`training/packages/v1_defect/data.yaml`

Roboflow 只擴增 train，valid 與 test 數量維持不變，這項行為符合模型評估原則。validation／test 必須保留未擴增的固定資料，才不會讓人工合成變化汙染泛化評估。此 Dataset 的主要限制是 rare class 樣本不足。

### Split 與 instance 分布

| split | 影像數 | 空標註影像 | `abrasion` | `anthracnose` | `insect_bite` | `insect_track` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1356 | 430 | 1742 | 97 | 36 | 1211 |
| valid | 97 | 31 | 133 | 12 | 1 | 76 |
| test | 97 | 33 | 114 | 16 | 1 | 100 |

`insect_bite` 在 valid 與 test 各只有 1 個 instance，任何 precision、recall 或 AP 都會受單一樣本主導，不能代表泛化能力。train 的 `36` 個 instance 又已包含 augmentation，原始獨立案例數更少。

## 訓練設定

兩次訓練共同使用 `YOLO26s-seg`、`100 epochs`、`seed=0`、`deterministic=true`、相同 Dataset 與相同 augmentation 設定。兩個 `args.yaml` 的 `data` 都指向 `training/packages/v1_defect/data.yaml`。完整參數分別保存在：

- `training/runs/segment/v1_defect 訓練結果_1/args.yaml`
- `training/runs/segment/v1_defect 訓練結果_2/args.yaml`

| 項目 | 第 1 次訓練 | 第 2 次訓練 |
| --- | ---: | ---: |
| 模型 | `yolo26s-seg.pt` | `yolo26s-seg.pt` |
| `imgsz` | 640 | 1024 |
| `batch` 設定 | 16 | -1 |
| checkpoint 實際 batch | 16 | 1 |
| epochs | 100 | 100 |
| 總時間 | 3456.64 秒，約 57.6 分鐘 | 22825 秒，約 380.4 分鐘 |
| 相對耗時 | 1.0× | 6.6× |

Roboflow 匯出影像的長邊最多只有 `640 px`。第 2 次訓練設為 `1024` 時，訓練流程會放大既有像素，不會恢復原始影像已經失去的細節。

## 指標結果

### 訓練保存的 best checkpoint

| 指標 | 第 1 次 best | 第 2 次 best |
| --- | ---: | ---: |
| Box precision | 0.55709 | 0.25761 |
| Box recall | 0.34715 | 0.29261 |
| Box mAP50 | 0.29295 | 0.22674 |
| Box mAP50-95 | 0.14675 | 0.11986 |
| Mask precision | 0.55196 | 0.24984 |
| Mask recall | 0.31400 | 0.28744 |
| Mask mAP50 | 0.27991 | 0.22871 |
| Mask mAP50-95 | 0.12460 | 0.11357 |

訓練 loss 持續下降，代表模型確實在擬合訓練資料；validation loss 後期回升且 AP 沒有同步改善，顯示泛化開始惡化。第 2 次訓練增加約 `6.6×` 訓練時間後，best checkpoint 的主要 Box 與 Mask 指標仍低於第 1 次訓練。

### 固定驗證流程重跑

為了取得可比較的逐類結果，兩個 `best.pt` 都在同一個 valid split、`batch=1`、`workers=0`、相同 Ultralytics `8.4.46` 與 NVIDIA GeForce RTX 3070 上重跑。第 2 次訓練的權重另外以 `imgsz=640` 驗證一次，觀察相同推論尺寸下的表現。

| 權重／驗證尺寸 | Mask precision | Mask recall | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| 第 1 次／640 | 0.55235 | 0.31600 | 0.27669 | 0.12730 |
| 第 2 次／1024 | 0.24875 | 0.29073 | 0.22871 | 0.11367 |
| 第 2 次／640 | 0.35830 | 0.24534 | 0.24992 | 0.13084 |

第 2 次訓練的權重在 `640` 推論時，Mask mAP50-95 比第 1 次高 `0.00354`，但 valid 只有 `97` 張，且 `insect_bite` 只有 1 個 instance。這個差距不足以認定第 2 次訓練有穩定改善。

### 逐類 Mask 指標

| 權重／驗證尺寸 | 類別 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 第 1 次／640 | `abrasion` | 0.443 | 0.414 | 0.366 | 0.12433 |
| 第 1 次／640 | `anthracnose` | 0.462 | 0.574 | 0.488 | 0.26875 |
| 第 1 次／640 | `insect_bite` | 1.000 | 0.000 | 0.0199 | 0.01592 |
| 第 1 次／640 | `insect_track` | 0.304 | 0.276 | 0.233 | 0.10019 |
| 第 2 次／1024 | `abrasion` | 0.390 | 0.391 | 0.324 | 0.11515 |
| 第 2 次／1024 | `anthracnose` | 0.435 | 0.417 | 0.432 | 0.28231 |
| 第 2 次／1024 | `insect_bite` | 0.000 | 0.000 | 0.000 | 0.00000 |
| 第 2 次／1024 | `insect_track` | 0.170 | 0.355 | 0.158 | 0.05722 |

`insect_bite` 的 Recall 皆為 `0`。這項結果主要反映 valid 只有單一 instance，現階段無法判斷模型是否能學習此類別。`abrasion` 與 `insect_track` 已有可量測訊號，但仍有大量漏偵測。

## 混淆矩陣解讀

Ultralytics `8.4.46` 的 segmentation validator 會把預測交給 `ConfusionMatrix.process_batch`。該函式使用預測與 ground truth 的 bounding box IoU 配對，預設圖表不直接衡量 mask polygon 的重疊品質。

- 橫軸是 True class，縱軸是 Predicted class。
- `background` 列代表 false negative，也就是 ground truth 沒有被配對到預測。
- `background` 欄代表 false positive，也就是預測沒有配對到 ground truth。
- 類別對角線偏低代表該類別的 box 配對成功率低，但仍需搭配 Mask precision、recall、mAP50 與 mAP50-95 判讀 segmentation 品質。

原始圖表保存在：

- `training/runs/segment/v1_defect 訓練結果_1/confusion_matrix.png`
- `training/runs/segment/v1_defect 訓練結果_1/confusion_matrix_normalized.png`
- `training/runs/segment/v1_defect 訓練結果_2/confusion_matrix.png`
- `training/runs/segment/v1_defect 訓練結果_2/confusion_matrix_normalized.png`

## 權重追溯與核准狀態

| 檔案 | SHA-256 |
| --- | --- |
| `training/runs/segment/v1_defect 訓練結果_1/weights/best.pt` | `3031fdac2261fa4ba154f5c503d457cb46531b93873a7c42aed0c2e1eec109a7` |
| `models/封存YOLO模型/0817/v1_defect.pt` | `3031fdac2261fa4ba154f5c503d457cb46531b93873a7c42aed0c2e1eec109a7` |
| `training/runs/segment/v1_defect 訓練結果_2/weights/best.pt` | `6733cd31fb70be05c284e0cb59c522d0ed94ad931ab9ebe0dd20dd339e5ffecf` |
| `models/封存YOLO模型/0817/v1_defect-2.pt` | `6733cd31fb70be05c284e0cb59c522d0ed94ad931ab9ebe0dd20dd339e5ffecf` |

兩個封存權重分別與對應的 `best.pt` SHA-256 相同，來源已確認。模型根目錄目前沒有正式局部瑕疵權重，`models/manifest.yaml` 也維持未核准狀態。

## 診斷結論

1. 兩次訓練的模型已有學習訊號，`abrasion`、`anthracnose` 與 `insect_track` 都出現非零 Mask AP。
2. 第 2 次訓練仍使用 `v1_defect`，影像沒有超過 `640 px` 的原始細節，`imgsz=1024` 主要增加計算量。
3. 兩次訓練同時改變 `imgsz` 與 batch size，無法把差異單獨歸因於解析度。
4. `insect_bite` 樣本嚴重不足，valid／test 各 1 個 instance，該類別無法驗收。
5. 第 2 次訓練沒有可靠、全面且足以抵銷成本的改善，因此維持封存實驗狀態。

## 後續處置

本文件維持為兩個 baseline 的歷史效能紀錄。新版模型採用下列完整改善方案，訓練結果另立文件：

- Roboflow Dataset 不設定強制 Resize，直接以原始解析度匯出，且不在 Roboflow 套用 augmentation。
- 只對 train 中包含 `anthracnose` 或 `insect_bite` 的影像執行 targeted augmentation；valid 與 test 保持原樣。
- 使用 `YOLO26s-seg`、`imgsz=1024`、`cls_pw=0.5`、`copy_paste=0.3`、`copy_paste_mode=mixup`。
- 將新權重與 `v1_defect.pt`、`v1_defect-2.pt` 以相同 valid 指令及 `imgsz=1024` 比較，再對選定權重執行一次 test。
- 新版同時改變 Dataset 與多項訓練設定，因此效能差異代表完整改善方案的結果，不拆分個別措施的貢獻。

正式核准前，模型至少必須滿足下列流程條件：

- 四個類別在 valid 與 test 都有足以解讀逐類指標的多個 instance。
- 新模型在相同驗證條件下與既有 baseline 比較，報告逐類結果，不能只報 macro average。
- test 評估與實際瑕疵面積比例誤差完成後，才透過 Model Manifest 流程更新為 `candidate` 並進一步核准。

## 重現驗證

以下命令是本紀錄使用的固定驗證流程：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -c "from ultralytics import YOLO; m=YOLO(r'training/runs/segment/v1_defect 訓練結果_1/weights/best.pt'); r=m.val(data=r'training/packages/v1_defect/data.yaml', split='val', imgsz=640, batch=1, workers=0, device=0, plots=False, save_json=False, verbose=True, project=r'training/runs/segment', name='diagnostic_v1_run1', exist_ok=True); print(r.results_dict); print('MAPS', r.seg.maps.tolist())"

C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -c "from ultralytics import YOLO; m=YOLO(r'training/runs/segment/v1_defect 訓練結果_2/weights/best.pt'); r=m.val(data=r'training/packages/v1_defect/data.yaml', split='val', imgsz=1024, batch=1, workers=0, device=0, plots=False, save_json=False, verbose=True, project=r'training/runs/segment', name='diagnostic_v1_run2', exist_ok=True); print(r.results_dict); print('MAPS', r.seg.maps.tolist())"

C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -c "from ultralytics import YOLO; m=YOLO(r'training/runs/segment/v1_defect 訓練結果_2/weights/best.pt'); r=m.val(data=r'training/packages/v1_defect/data.yaml', split='val', imgsz=640, batch=1, workers=0, device=0, plots=False, save_json=False, verbose=False, project=r'training/runs/segment', name='diagnostic_v1_run2_at640', exist_ok=True); print(r.results_dict); print('MAPS', r.seg.maps.tolist())"
```

## Repository 邊界

這份紀錄屬於模型 Repository，因為它描述 Dataset、訓練 run、checkpoint 與模型效能評估。

硬體 Repository `D:\PassionFruit_IoT_hardware` 繼續維護 Django、ESP32 Firmware、硬體安全、資料採集流程與 AI adapter 契約。這次實驗沒有改變四類瑕疵、mask 面積比例、人工覆核或「模型不得直接控制硬體」等既有契約，因此不修改硬體 Repository 的 `DATA_COLLECTION_SPEC.md`、`PROJECT_CONTEXT.md` 或 ADR。
