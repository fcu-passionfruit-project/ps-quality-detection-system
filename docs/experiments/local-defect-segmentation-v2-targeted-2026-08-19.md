# `v2_defect` targeted augmentation 與 v3 參數調整實驗紀錄

- 實驗日期：2026-08-19 至 2026-08-20
- 任務：百香果局部瑕疵 instance segmentation
- 模型：`YOLO26s-seg`
- Dataset：`training/packages/v2_defect_argument`
- 訓練紀錄：`training/runs/segment/v2_defect_pw050_cp03`、`training/runs/segment/v3_defect_pw025_no_cp`
- 結論：v2 完整改善方案大幅提升 `anthracnose`，整體 Mask mAP50-95 僅小幅超過最強 baseline；v3 降低 class weighting 並關閉 Copy-Paste 後，precision 提高，但 recall 與 Mask AP 下降，沒有恢復 `abrasion`、`insect_track`。兩者均保留為 candidate 實驗權重，不核准為正式局部瑕疵模型。

## 實驗目的

本次實驗以 `v1_defect` 的兩個封存模型為 baseline，測試以下完整改善方案：

- 使用未在 Roboflow 強制 Resize、未在 Roboflow 套用 augmentation 的 `v2_defect`。
- 只對 train 中包含 `anthracnose` 或 `insect_bite` 的影像執行離線 targeted augmentation。
- 使用 `cls_pw=0.5` 處理 class imbalance。
- 使用 `copy_paste=0.3` 與 `copy_paste_mode=mixup` 增加跨影像 segment 變化。

本次同時改變 Dataset、離線 augmentation、class weighting 與 Copy-Paste，結果代表完整改善方案的整體效果，不能拆解為單一參數的貢獻。

## Dataset 與離線擴增

原始 Dataset 位於 `training/packages/v2_defect`，擴增後 Dataset 位於 `training/packages/v2_defect_argument`。工具只擴增 train，valid 與 test 維持原始影像與標註。

### Split 與 instance 分布

| split | 影像數 | 空標註影像 | `abrasion` | `anthracnose` | `insect_bite` | `insect_track` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train，擴增前 | 433 | 167 | 577 | 54 | 26 | 386 |
| train，擴增後 | 605 | 167 | 878 | 183 | 156 | 573 |
| valid | 93 | 29 | 97 | 10 | 1 | 104 |
| test | 92 | 35 | 98 | 22 | 3 | 100 |

離線工具新增 `172` 張 train 影像：包含 `anthracnose` 的來源影像新增 2 份，包含 `insect_bite` 的來源影像新增 5 份；同時包含兩類的影像採 `insect_bite` 的 5 份設定。完整生成紀錄位於 `training/packages/v2_defect_argument/augmentation_manifest.json`。

valid 只有 1 個 `insect_bite` instance，因此該類的 precision、recall 與 AP 會受單一樣本主導，現階段無法可靠驗收。

## 訓練設定

| 項目 | 設定 |
| --- | --- |
| 模型 | `yolo26s-seg.pt` |
| epochs | 100 |
| `imgsz` | 1024 |
| `batch` 設定 | -1 |
| checkpoint 實際 batch | 1 |
| seed | 0 |
| deterministic | true |
| `cls_pw` | 0.5 |
| `copy_paste` | 0.3 |
| `copy_paste_mode` | `mixup` |
| 總時間 | 9699.18 秒，約 161.7 分鐘 |

完整設定保存在 `training/runs/segment/v2_defect_pw050_cp03/args.yaml`。

## 相同 valid 的 baseline 比較

四個 `best.pt` 使用相同的 `v2_defect_argument` valid、`imgsz=1024`、`batch=1`、`workers=0`、Ultralytics `8.4.46` 與 NVIDIA GeForce RTX 3070 重跑。

| 權重 | Mask precision | Mask recall | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| v1，訓練 `imgsz=640` | 0.55721 | 0.32565 | 0.29086 | 0.13925 |
| v1，訓練 `imgsz=1024` | 0.69692 | 0.32721 | 0.34492 | 0.20013 |
| v2 完整改善方案 | 0.33944 | 0.35960 | 0.36176 | 0.20557 |
| v3，`cls_pw=0.25`、無 Copy-Paste | 0.43408 | 0.32780 | 0.34372 | 0.19723 |

相較最強的 v1 `imgsz=1024` baseline，v2 的 Mask mAP50-95 提高 `0.00545`，約 `2.7%`；Mask mAP50 提高 `0.01684`，約 `4.9%`。Mask recall 提高 `0.03239`，Mask precision 降低 `0.35748`，表示模型提高召回的同時產生更多 false positive。

相較 v2，v3 的 Mask precision 提高 `0.09464`，Mask recall 降低 `0.03180`，Mask mAP50 降低 `0.01803`，Mask mAP50-95 降低 `0.00835`，約下降 `4.1%`。v3 的 Box mAP50-95 為 `0.20779`，v2 為 `0.20855`，兩者幾乎相同；差異主要發生在 mask 品質與召回率。

### 逐類 Mask 指標

| 類別 | 權重 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `abrasion` | v1／1024 | 0.48323 | 0.43382 | 0.37897 | 0.20070 |
| `abrasion` | v2 | 0.38981 | 0.27835 | 0.27780 | 0.10878 |
| `abrasion` | v3 | 0.40367 | 0.30928 | 0.25921 | 0.09857 |
| `anthracnose` | v1／1024 | 0.81253 | 0.50000 | 0.53348 | 0.40583 |
| `anthracnose` | v2 | 0.48345 | 0.84372 | 0.86444 | 0.58425 |
| `anthracnose` | v3 | 0.74089 | 0.80000 | 0.85167 | 0.58434 |
| `insect_bite` | v1／1024 | 1.00000 | 0.00000 | 0.11056 | 0.04422 |
| `insect_bite` | v2 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| `insect_bite` | v3 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| `insect_track` | v1／1024 | 0.49191 | 0.37500 | 0.35667 | 0.14976 |
| `insect_track` | v2 | 0.48451 | 0.31632 | 0.30479 | 0.12927 |
| `insect_track` | v3 | 0.59174 | 0.20192 | 0.26403 | 0.10600 |

v2 的整體 macro mAP 提升主要來自 `anthracnose`。該類 Mask mAP50-95 提高 `0.17842`；`abrasion` 降低 `0.09193`，`insect_track` 降低 `0.02049`。`insect_bite` 的 valid Recall 仍為 0。

v3 保留了 `anthracnose` 的效果，其 Mask mAP50-95 與 v2 幾乎相同，precision 從 `0.48345` 提高至 `0.74089`。`abrasion` recall 略有恢復，但 Mask mAP50-95 進一步降至 `0.09857`；`insect_track` precision 提高、recall 降至 `0.20192`，Mask mAP50-95 降至 `0.10600`。降低 `cls_pw` 並關閉 Copy-Paste 沒有恢復主要類別的 AP，原先假設未獲本次結果支持。

## 訓練曲線

- Mask mAP50-95 在 epoch 67 達到最高 `0.20407`，epoch 100 降至 `0.18555`。
- validation segmentation loss 在 epoch 47 最低，後續逐漸回升。
- 訓練後半段仍持續降低 train loss，但 validation loss 與 Mask mAP50-95 未同步改善，顯示約 epoch 50 至 70 後開始出現過擬合。
- 後續評估使用 `best.pt`，不使用 `last.pt`。

v3 的 train losses 同樣持續下降，validation losses 約在 epoch 60 至 75 後持平或微升。Mask mAP50-95 在 epoch 77 達到最高 `0.19585`，epoch 100 降至 `0.17168`。v3 後續評估同樣使用 `best.pt`。

## `cls_pw` 診斷與調整決策

Ultralytics 依 train instance frequency 計算 inverse-frequency class weights，再以 `cls_pw` 作為次方並正規化，使平均權重為 1.0。以擴增後的 `878／183／156／573` 計算：

| 類別 | `cls_pw=0.5` | `cls_pw=0.25` |
| --- | ---: | ---: |
| `abrasion` | 0.588 | 0.779 |
| `anthracnose` | 1.288 | 1.153 |
| `insect_bite` | 1.395 | 1.200 |
| `insect_track` | 0.728 | 0.867 |

`cls_pw=0.5` 將 `abrasion` 與 `insect_track` 的分類 loss 權重降至 `0.588／0.728`。這項降權與兩類逐類 mAP 退步的方向一致，但本次同時啟用多項改善措施，不能只依相關性宣稱 `cls_pw` 是唯一原因。

v3 將 `cls_pw` 降至 `0.25`，保留少數類別的部分加權，同時減少主要類別被壓低的幅度。相同 valid 重跑後，主要類別 AP 沒有恢復，因此不能將 v2 的退步只歸因於 `cls_pw=0.5`。Ultralytics 設定說明見 <https://docs.ultralytics.com/usage/cfg>。

## Copy-Paste 診斷與調整決策

Ultralytics 的 Copy-Paste 適用於 segmentation，`copy_paste_mode=mixup` 會從另一張影像取得 segments。它不會依本專案的 rare-class 名單優先挑選 `insect_bite`，也沒有百香果表面 ROI 約束，所有 eligible classes 都可能被複製。

在本次使用的 Ultralytics `8.4.46` 實作中，`mixup` 路徑會為含有 segment 的樣本取得另一張影像，再從 IoA 小於 `0.3` 的 eligible instances 中選取約 `round(copy_paste × n)` 個物件。`copy_paste=0.3` 應視為偏強設定。官方功能說明見 <https://docs.ultralytics.com/guides/yolo-data-augmentation#copy-paste-copy_paste>。

由於 Dataset 已完成離線 targeted augmentation，v3 設為：

```text
copy_paste=0.0
```

v3 保留 `anthracnose` 效果，但主要類別沒有恢復，因此暫不建立只新增 Copy-Paste 的下一組對照。若未來仍要量測低強度 Copy-Paste，可使用：

```text
copy_paste=0.1
copy_paste_mode=mixup
```

## Roboflow augmentation 決策

本輪不改用 Roboflow 重新做 rare-class augmentation。Roboflow 可透過 Tag 與 `Filter by Tag` 建立只包含 rare-class 來源影像的 Dataset Version，但 augmentation 仍套用在整張 train 影像與全部標註，且不同擴增倍數需要建立多個 Version 後再合併。

現有離線工具已提供不同類別倍數、固定 seed、polygon 同步轉換與 `augmentation_manifest.json`，可重現性與合併流程較適合目前實驗。Roboflow 行為說明見：

- <https://docs.roboflow.com/datasets/manage-datasets/add-tags-to-images>
- <https://docs.roboflow.com/datasets/image-augmentation>

若未來另做 Roboflow 對照，必須從原始 `v2_defect` 建立 Version，避免對 `v2_defect_argument` 再次擴增。只使用水平／垂直翻轉、90 度旋轉與 Exposure ±10%；不加入 Hue、Grayscale、Crop、Shear、Cutout 或額外 Mosaic，避免破壞瑕疵顏色、小型 polygon 與真實幾何外觀。

## v3 追蹤實驗

v3 沿用 `v2_defect_argument`、模型、split、seed 與解析度，降低 class weighting 並關閉 Copy-Paste。原規劃指令如下：

```powershell
yolo task=segment mode=train `
  model=yolo26s-seg.pt `
  data="training/packages/v2_defect_argument/data.yaml" `
  project="training/runs/segment" `
  name=v3_defect_pw025_no_cp `
  epochs=100 `
  imgsz=1024 `
  batch=-1 `
  device=0 `
  workers=8 `
  seed=0 `
  deterministic=True `
  cls_pw=0.25 `
  copy_paste=0.0
```

### v3 實際訓練設定

| 項目 | v2 | v3 |
| --- | --- | --- |
| Dataset | `v2_defect_argument` | `v2_defect_argument` |
| epochs／`imgsz`／seed | 100／1024／0 | 100／1024／0 |
| `cls_pw` | 0.5 | 0.25 |
| `copy_paste` | 0.3 | 0.0 |
| `copy_paste_mode` | `mixup` | `flip`，但 `copy_paste=0.0`，不生效 |
| checkpoint 實際 batch | 1 | 4 |
| 裝置 | NVIDIA CUDA | Apple MPS |
| Ultralytics | 8.4.46 | 8.4.122 |
| 總時間 | 9699.18 秒，約 161.7 分鐘 | 10698 秒，約 178.3 分鐘 |

v3 實際執行環境同時改變 batch、硬體 backend 與 Ultralytics 版本，因此不能視為只改 `cls_pw` 與 Copy-Paste 的受控 A／B 實驗。相同 validation 重跑可公平比較產生的權重，但無法隔離各項訓練變因的因果貢獻。

### v3 訓練輸出與決策

v3 `results.csv` 在 epoch 100 的 Mask precision、recall、mAP50、mAP50-95 分別為 `0.37725／0.31490／0.34051／0.17168`。以固定 validation 環境重新評估 `best.pt` 後，指標為 `0.43408／0.32780／0.34372／0.19723`。

v3 的 normalized confusion matrix 以 bounding box IoU 進行配對，對角線約為 `abrasion=0.49`、`anthracnose=0.80`、`insect_track=0.31`；唯一的 `insect_bite` ground truth 被判為 `abrasion`。background false negative 約為 `abrasion=0.48`、`anthracnose=0.20`、`insect_track=0.55`。該圖反映 box 配對與分類結果，mask 品質仍以逐類 Mask precision、recall、mAP50 與 mAP50-95 判讀。

實驗決策如下：

1. v3 沒有超越 v2，不取代目前的 v2 candidate。
2. v3 僅作為降低 class weighting、關閉 Copy-Paste 的追蹤實驗權重。
3. v2 與 v3 都未解決 `insect_bite` 無召回，valid 又只有 1 個 instance，暫不依該類單次指標核准模型。
4. 目前沒有候選模型形成明確優勢，暫不執行 test 驗收，也不更新 `models/manifest.yaml`。
5. 若要確認參數的單獨效果，必須固定 v2 的 Ultralytics `8.4.46`、NVIDIA CUDA、`batch=1`、Dataset、seed 與其餘設定，只改 `cls_pw=0.25`、`copy_paste=0.0` 後重訓。

## 權重追溯與狀態

| 檔案 | SHA-256 |
| --- | --- |
| `training/runs/segment/v2_defect_pw050_cp03/weights/best.pt` | `9035be34094c3feff630f0f0d5a0a5f51a504ef673480111761e1a782aed6fbe` |
| `training/runs/segment/v3_defect_pw025_no_cp/weights/best.pt` | `b60c18f607e4183e3597d98c224ed8cff7d8abad2744ec706169584b706a52a1` |

目前權重保留在 training run，尚未更新 `models/manifest.yaml` 為核准模型。這次實驗沒有改變模型輸入輸出契約或硬體控制邊界，因此不修改硬體 Repository 規格文件。
