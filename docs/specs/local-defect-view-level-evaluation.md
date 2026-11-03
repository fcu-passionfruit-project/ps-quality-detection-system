# 局部瑕疵模型的視角層級評估方法

本文件定義局部瑕疵 segmentation 模型的主要評估指標，並整理專題報告中向評審說明「為何不以 mAP 作為主要指標」的論點。實際數值記錄在 `docs/experiments/`，本文件只保存方法。

評估工具：`passionfruit/evaluation/evaluate_view_level.py`

## 1. 評估目標要對齊系統用途

局部瑕疵模型在系統中不是最終輸出。它的輸出會先轉成每張照片的四個特徵，再交給兩階段 XGBoost 判定等級（見 `DecisionLayerTrain/XGboost/兩階段XGBoost模型使用說明.md`）：

```text
照片 → ROI → 局部瑕疵 segmentation → 四類 *_area_ratio → XGBoost → 視角等級 → 三視角取最低 → 果實等級
```

`*_area_ratio` 的定義是「同類所有 mask 的 pixel union ÷ 可見果實 ROI pixel」（`docs/specs/decision-dataset-generator.md`）。因此決策層真正需要的是兩件事：

1. 這張照片**有沒有**某類瑕疵。
2. 該類瑕疵的**面積大約多大**。

決策層不需要知道「這條蟲道是一個 instance 還是三個」，也不在意 mask 邊界是否精準到 IoU 0.95。評估指標應該衡量前兩件事。

## 2. 為何 mAP 不適合作為主要指標

以下是向評審說明的重點。結論不是「mAP 沒用」，而是「mAP 衡量的問題和系統要解決的問題不同」。所以報告中仍會列出 mAP 供參考，只是不用它決定模型好壞。

| mAP 的計算方式 | 在本專題造成的偏差 |
| --- | --- |
| 以 instance 一對一配對（IoU ≥ 門檻才算對） | 細長蟲道常被模型切成 2～5 段。面積幾乎都抓到了，mAP 卻同時記為 1 個漏檢加上數個誤檢。 |
| mAP50-95 要求 IoU 達到 0.95 | 擦傷、蟲道沒有清楚邊界，標註者之間本身就有誤差。嚴格邊界衡量的多半是標註雜訊。 |
| 四類取平均 | valid 中 insect_bite 只有 1 個實例、anthracnose 只有 4 張圖。一個樣本的對錯就能讓整體 mAP 變動 0.1 以上。 |
| 忽略「這張圖有沒有瑕疵」的層級 | 同一張圖有 5 處擦傷，抓到 4 處或 5 處，對分級的影響很小；mAP 卻當成不同的錯誤計分。 |
| 不衡量面積 | 決策層依面積判斷嚴重度，mAP 不反映面積估計誤差。 |

從實驗結果也看得到 mAP 和實際需求脫節：

- 既有實驗 v3–v19 的 Mask mAP50 都落在 0.29～0.37 之間，差異大多在單次訓練的波動範圍內（`docs/experiments/local-defect-segmentation-runs-v3-v19-2026-08-30.md`）。
- Experiment D 的三個 seed，mAP50 標準差只有 0.003，但固定 threshold 下的 TP/FP/FN 卻各相差 38～62 個（`docs/experiments/v22-v23-v24-context6-three-seed-analysis.md`）。排名指標很穩定，不代表實際判定行為穩定。

## 3. 指標設計

所有指標都以**一張照片（一個視角）**為單位，因為 XGBoost 也是逐視角預測，最後才由「三視角取最低」得到果實等級。資料集中多數影像檔名沒有保留 `fruit_id`，因此 segmentation 階段無法直接做果實層級彙總；果實層級的評估放在第 4 級。

### 第 1 級：有無判定（Presence）

對每張照片、每類瑕疵，比較「標註面積 > 0」與「預測面積 > 0」：

- **Sensitivity（靈敏度）**：有瑕疵的照片中，模型抓到的比例。漏判會讓加工果被當成良品出貨，是最需要避免的錯誤。
- **Specificity（特異度）**：沒有瑕疵的照片中，模型正確判為無的比例。誤判會讓良品被降級。
- **F1**。
- **任一瑕疵**：四類合併後的有無判定，對應「這張照片是否需要進一步檢查」。

### 第 2 級：面積估計（Area）

- **面積 MAE（百分點）**：|預測面積比例 − 標註面積比例| 的平均，分別計算全部照片和僅標註為正樣本的照片。
- **Spearman 相關係數**：預測面積與標註面積的排序是否一致。即使絕對值有偏差，只要大小順序正確，XGBoost 仍能學到正確的界線。

固定部署門檻 `0.25` 下的面積誤差（`MAE_GT+`、`MAE_GT−`、`bias`）與像素 Precision／Recall／F1（Dice）另見 README〈局部瑕疵模型評估方式〉及 `docs/experiments/v19-vs-v22-area-pixel-evaluation-2026-11-02.md`。該評估回答「部署設定下 mask 塗得準不準」，本工具回答「選定 threshold 後判級特徵是否一致」，兩者互補。

### 第 3 級：嚴重度一致性（Severity）

把每類的面積比例分成 **無／小／中／大** 四級。小、中、大的界線取 train split 標註正樣本面積的 33 和 67 百分位，只用 train 決定，避免偷看評估資料。接著計算標註等級與預測等級的 **Quadratic Weighted Kappa（QWK）**：

- QWK = 1 表示完全一致，0 表示和隨機猜測一樣。
- 二次加權的意思是：「大」被判成「中」只扣一點分，「大」被判成「無」則重扣。這符合分級的實際代價。
- 主要比較指標是 abrasion、anthracnose、insect_track 三類 QWK 的平均（Macro QWK）。insect_bite 在 valid 只有 1 個正樣本，只報告、不參與選模。

### 第 4 級：端到端等級一致性（規劃中）

在已有專家等級的決策資料集（`decision_datasets/`）上，把新模型產生的面積特徵送入既有 XGBoost，比較預測果實等級與專家等級的 accuracy、混淆矩陣與「加工果漏判率」。這是最終最有說服力的指標，但需要完成三視角照片的專家標註後才能計算。

## 4. 評估流程與統計可信度

1. **threshold 只在 valid 選**：在 conf 0.01～0.80 中選出 valid Macro QWK 最高的全域值，另外每類各自選 QWK 最高的值（insect_bite 沿用全域值）。test 只用選定的值計算一次。
2. **報告 95% 信賴區間**：test 只有 92 張，因此以照片為單位做 2000 次 bootstrap 重抽樣，報告 Macro QWK 與任一瑕疵 F1 的 95% CI。
3. **配對比較**：比較兩個模型時，對同一批 test 照片做配對 bootstrap，報告「新模型 − 舊模型」差異的 95% CI。CI 不跨 0 時，才宣稱新模型較佳。
4. **mAP 仍列為參考**：同時附上 Ultralytics 的 Mask mAP50 和 mAP50-95，方便和文獻比較。

執行方式（從 repository 根目錄）：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -m passionfruit.evaluation.evaluate_view_level `
  --dataset training/packages/v2_defect_balanced `
  --model v19=training/runs/segment/v19_defect_epoch200_pw050_cp01/weights/best.pt `
  --model v25=training/runs/segment/v25_balanced_aug_img800/weights/best.pt `
  --output training/runs/eval/v19_vs_v25
```

每個模型會使用自己 `args.yaml` 中的訓練 `imgsz` 推論。輸出包含 `report.md`、`summary.csv`、`summary.json` 和每張照片的面積比例。

## 5. 評審常見問題

**Q1：為什麼 mAP 只有 0.3 多？模型是不是很差？**

mAP 以 instance 為單位，會把「一條蟲道被切成三段」記為多個錯誤，也會被只有 1～4 個樣本的稀有類別拉低。本系統使用的是每張照片的瑕疵面積，因此我們以有無判定、面積誤差和嚴重度一致性評估模型。mAP 仍列出供參考。

**Q2：為什麼不修改 YOLO backbone，或加入 CBAM、BiFPN、SPD-Conv 等模組？**

根據混淆矩陣，錯誤幾乎都是「瑕疵 ↔ 背景」：以 v19 為例，abrasion 漏檢 68 個、insect_track 漏檢 64 個、背景誤判 69 個，而類別之間互相混淆只有個位數。這表示瓶頸在於資料量少、目標小、標註邊界模糊，而不是特徵不足以區分類別。Attention 類模組主要強化特徵辨識，對這個問題的預期收益有限。訓練資料只有約 270 張含瑕疵的照片，增加參數反而會加重過擬合；既有實驗已觀察到 validation loss 在約 epoch 90～140 開始回升。若之後要調整架構，優先考慮的是針對小目標的 P2（stride 4）偵測頭，而不是 attention 模組。

**Q3：資料不平衡怎麼處理？**

以影像層級重複抽樣，使稀有類別（anthracnose、insect_bite）在每個 epoch 出現的次數接近常見類別，並搭配線上隨機旋轉與翻轉，避免模型只是記住同一張圖。insect_bite 全資料集只有約 30 個實例，目前證據不足以宣稱可辨識，報告中如實標示為「樣本不足」。

**Q4：為什麼要附信賴區間？**

test 只有 92 張照片，部分類別只有個位數樣本。單一數字的差距可能只是抽樣誤差。附上 95% CI，並對兩個模型做配對比較，可以分辨哪些改善是可信的。
