# Experiment B: Class-aware Sampling Analysis

- 分析日期：2026-08-31
- 任務：YOLO26s-seg 百香果局部瑕疵 instance segmentation
- Baseline：`training/runs/segment/v19_defect_epoch200_pw050_cp01`
- Experiment B：`training/runs/segment/v20_B_class_aware`
- 主要問題：固定其他可觀察訓練條件後，class-aware train exposure 是否優於 v19 baseline
- 評估範圍：既有 validation artifact；本次沒有重新訓練、重新 validation 或執行 test split

## Executive Summary

**最終判斷：B. v20 有部分改善，但存在明顯 trade-off，暫時不能取代 v19。**

v20 確實達成了「選定 operating point 的整體 Recall 提高」：

- Box Recall：`0.31903 → 0.38063`，增加 `0.06160`（6.16 個百分點，+19.31%）。
- Mask Recall：`0.32401 → 0.38669`，增加 `0.06268`（6.27 個百分點，+19.35%）。
- bounding-box confusion matrix 的 background unmatched GT：`134 → 123`，減少 11（-8.21%）。

但這不是整體模型品質同步提升：

- Mask Precision：`0.70565 → 0.61755`，下降 `0.08810`（8.81 個百分點）。
- Mask mAP50：`0.36659 → 0.36034`，下降 `0.00625`。
- Mask mAP50-95：`0.21046 → 0.20582`，下降 `0.00464`。
- background unmatched predictions：`69 → 90`，增加 21（+30.43%）。
- Box／Mask 最佳 macro F1 四捨五入後都沒有提高；v20 只是把最佳 threshold 往較低 confidence 移動。
- 最大 achievable macro Recall 反而下降：Box `0.70 → 0.57`、Mask `0.69 → 0.55`（皆為 confidence 0.000 的 curve label）。

逐類結果也不支持「rare classes 全面受益」：

- `anthracnose`：Box／Mask AP50 小幅提高，正確 box match `8 → 9`，unmatched GT `2 → 1`；但 validation 只有 10 個 GT、4 張含此類圖片，不能排除單次波動。
- `insect_bite`：Box／Mask AP50 `0.033 → 0.000`；唯一 GT 兩組都被判成 `abrasion`，4x exposure 沒有產生可驗證的正確辨識。
- `abrasion`：Box／Mask AP50、正確 match 與 FN 明顯改善，但 background FP `31 → 52`，是新增 FP 的主要來源。
- `insect_track`：Box／Mask AP50、正確 match 與 FN 都退步；大型／細長 mask 碎片化仍存在。

因此 v20 的主要效果是 **Recall 上升、Precision 與 background specificity 下降的 P/R trade-off**，而不是同時提高 Recall、mAP、F1 與逐類穩定性。

對 Experiment C 的建議為：**有條件建議進行，但以 `v19 + targeted crop` 為基礎，不以 `v20 + targeted crop` 為基礎。** 這樣 targeted crop 才是唯一新增變因，也避免把 v20 已觀察到的 background FP 問題和 targeted crop 疊加後無法拆解。

## Artifact Review Scope

本次先閱讀 root `AGENTS.md`、`README.md`、`CONTEXT.md`、`main/README.md`、局部瑕疵標註規格與既有局部瑕疵實驗紀錄，再檢查兩個 run 的全部現存 artifact：

- `args.yaml`、`results.csv`、`results.png`、`labels.jpg`
- Box／Mask 的 `P_curve`、`R_curve`、`F1_curve`、`PR_curve`
- `confusion_matrix.png`、`confusion_matrix_normalized.png`
- `val_batch0..2_labels.jpg`、`val_batch0..2_pred.jpg`
- 初始與訓練末段的 `train_batch*.jpg`
- `weights/best.pt`、`weights/last.pt` 的 SHA-256 與 checkpoint metadata
- `training/packages/local_defects/data.yaml`
- `training/packages/local_defects/data_exp_B.yaml`
- `training/packages/local_defects/experiments/train_B_class_aware.txt`
- train／valid／test images 與 labels 的集合、數量、class distribution 與內容 manifest

曲線只有 raster PNG，沒有原始 curve array；因此可報告圖例中的 AP、最佳 F1、threshold 與 maximum Recall，但不虛構精確 curve crossover 或任意 confidence 的數值。

## Experiment Validity Check

### 核心訓練參數

| 參數 | v19 Baseline | v20 Class-aware | 比較結果 |
| --- | --- | --- | --- |
| `model` | `yolo26s-seg.pt` | `yolo26s-seg.pt` | 相同字串 |
| `epochs` | 200 | 200 | 相同 |
| `patience` | 50 | 50 | 相同 |
| `imgsz` | 1024 | 1024 | 相同 |
| `batch` | 4 | 4 | 相同 |
| `device` | `mps` | `mps` | 相同 |
| `seed` | 0 | 0 | 相同 |
| `deterministic` | `true` | `true` | 相同 |
| `cls_pw` | 0.5 | 0.5 | 相同 |
| `copy_paste` | 0.1 | 0.1 | 相同 |
| `copy_paste_mode` | `mixup` | `mixup` | 相同 |
| `optimizer` | `auto` | `auto` | 相同設定；resolved optimizer 未保留在 stripped checkpoint |
| `lr0` | 0.01 | 0.01 | 相同 |
| `lrf` | 0.01 | 0.01 | 相同 |
| `mask_ratio` | 4 | 4 | 相同 |
| `pretrained` | `true` | `true` | 相同 |
| `data` | `local_defects/data.yaml` | `local_defects/data_exp_B.yaml` | 唯一有意義的 args 差異 |
| `split` | `val` | `val` | 相同；不是 test |

完整 `args.yaml` diff 除了 `data`、run `name` 與 `save_dir` 外沒有其他差異。augmentation、loss weights、workers、AMP、mosaic、HSV、flip、IoU、max detections 等其餘記錄參數也相同。

### Dataset 與 split 驗證

| 驗證項目 | 結果 |
| --- | --- |
| v19 train | `../train/images`，433 張 unique images |
| v20 train | 532 list entries、433 張 unique images |
| v20 unique train set 是否等於 v19 | 是，完全相同 |
| v20 repeat factor 分布 | 386 張 1x、21 張 2x、26 張 4x |
| v20 list 是否混入 valid/test | 否，兩者 overlap 都是 0 |
| validation | 兩份 YAML 都是同一個 `../valid/images`，93 張 |
| test | 兩份 YAML 都是同一個 `../test/images`，92 張 |
| class IDs／names | 完全相同：0 `abrasion`、1 `anthracnose`、2 `insect_bite`、3 `insect_track` |
| `nc` | 都是 4 |
| split 是否改變 | 沒有；只改 train list 的重複次數 |

三個 `val_batch*_labels.jpg` 在 v19／v20 間各自具有相同 SHA-256，證明抽樣顯示的 validation 圖、順序與 labels 逐位元相同。Dataset 的現存統計為：

| split | images | null images | `abrasion` | `anthracnose` | `insect_bite` | `insect_track` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train，原始 instances | 433 | 167 | 577 | 54 | 26 | 386 |
| valid，GT instances | 93 | 29 | 97 | 10 | 1 | 104 |
| test，GT instances | 92 | 35 | 98 | 22 | 3 | 100 |

v20 的有效 train exposure 為 `753／122／104／497`，與 `labels.jpg` 顯示一致，證明 Ultralytics 確實讀入 repeated list，而不是退回 433 張原始目錄。

### Sampling intervention 不只改類別比例，也增加總 updates

v20 的 532 entries 比 v19 的 433 images 多 22.86%。在 epochs 與 batch 固定時，這會同時造成：

1. rare-class 相對 exposure 改變；
2. 每 epoch 的 batches 與 optimizer steps 增加；
3. 所有位於 2x／4x 圖片中的其他共存類別也增加 exposure；
4. 完整訓練時間增加。

完整訓練時間由 `15,209.6 s` 增至 `18,495.5 s`，增加 21.60%。因此本報告可以比較「這個 class-aware sampling strategy 的整體實作效果」，但不能把差異嚴格歸因為單純 class prior；總更新次數也是 intervention 的一部分。

null image 沒有被刪除，但相對 exposure 由：

```text
167 / 433 = 38.57%
```

降為：

```text
167 / 532 = 31.39%
```

下降 7.18 個百分點（相對下降 18.61%）。

### 環境與初始權重可比性

| 項目 | v19 | v20 | 判斷 |
| --- | --- | --- | --- |
| Ultralytics checkpoint version | 8.4.122 | 8.4.122 | 相同 |
| device args | MPS | MPS | 相同 |
| Git commit metadata | `7fdf10249c7a2e88eeff8210b66a22d25a6c7a99` | 同左 | 相同 |
| model class／names | SegmentationModel／四類相同 | 同左 | 相同 |
| checkpoint dates | 2026-08-28 | 2026-08-31 | 不同執行時間 |
| PyTorch training version | checkpoint 未記錄 | checkpoint 未記錄 | 無法驗證 |
| initial `yolo26s-seg.pt` SHA-256 | checkpoint 未記錄 | checkpoint 未記錄 | 無法驗證 |

目前 repository root 的 `yolo26s-seg.pt` SHA-256 是 `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4`，但 checkpoints 沒有保存當次載入檔案的 hash，`model.pt_path` 也是空值，因此不能用目前檔案反推兩次歷史 run 一定載入相同 bytes。兩次 run 在同一台路徑、同一 Ultralytics、同一 MPS device、同一 Git commit 且相隔三天，環境相同的證據很強，但 PyTorch 版本與 initial-weight hash 仍是未完全排除的 confounders。

## Overall Metrics

主要比較點是各自 `results.csv` 中 **Mask mAP50-95 最高的 epoch**：

- v19：epoch 186，Mask mAP50-95 `0.21046`
- v20：epoch 161，Mask mAP50-95 `0.20582`

### Box metrics

| Metric | v19 Baseline | v20 Class-aware | Absolute Change | Relative Change |
| --- | ---: | ---: | ---: | ---: |
| Box Precision | 0.69073 | 0.61113 | -0.07960 | -11.52% |
| Box Recall | 0.31903 | 0.38063 | +0.06160 | +19.31% |
| Box mAP50 | 0.36643 | 0.35840 | -0.00803 | -2.19% |
| Box mAP50-95 | 0.21573 | 0.20447 | -0.01126 | -5.22% |

### Mask metrics

| Metric | v19 Baseline | v20 Class-aware | Absolute Change | Relative Change |
| --- | ---: | ---: | ---: | ---: |
| Mask Precision | 0.70565 | 0.61755 | -0.08810 | -12.49% |
| Mask Recall | 0.32401 | 0.38669 | +0.06268 | +19.35% |
| Mask mAP50 | 0.36659 | 0.36034 | -0.00625 | -1.71% |
| Mask mAP50-95 | 0.21046 | 0.20582 | -0.00464 | -2.20% |

### 整體解讀

v20 的 Box 與 Mask Recall 都有實際可見的 6 個百分點改善，並非只有小數點後的噪音方向；但 Precision 同時下降約 8–9 個百分點，而且四個 mAP 指標全部下降。Mask mAP50-95 的 `-0.00464` 很小，單次 seed 下可能落在 run-to-run variance 內，不能宣稱 v20 顯著更差；反過來也不能用 Recall 上升宣稱 v20 整體更好。

如果 class-aware sampling 真正提高了整體 detection／segmentation 能力，理想訊號應至少包含 Recall 與 PR area／mAP 同向上升，或最佳 F1 提高。現有結果沒有出現這種組合，主要是 **Precision ↓ / Recall ↑** 的 operating trade-off。

## Recall Analysis

### Selected best-epoch Recall

| Recall | v19 | v20 | Change |
| --- | ---: | ---: | ---: |
| Box | 0.31903 | 0.38063 | +0.06160 |
| Mask | 0.32401 | 0.38669 | +0.06268 |

在各自最佳 checkpoint 的 Ultralytics selected operating point，v20 的 Recall 明確較高。

### R curve、PR curve 與 F1 curve

| Curve summary | v19 | v20 | 判斷 |
| --- | ---: | ---: | --- |
| Box maximum macro Recall | 0.70 @ 0.000 | 0.57 @ 0.000 | v20 較低 |
| Mask maximum macro Recall | 0.69 @ 0.000 | 0.55 @ 0.000 | v20 較低 |
| Best Box macro F1 | 0.35 @ 0.349 | 0.35 @ 0.293 | F1 未提高，threshold 降低 |
| Best Mask macro F1 | 0.36 @ 0.353 | 0.36 @ 0.291 | F1 未提高，threshold 降低 |
| Box PR area（mAP50） | 0.367 | 0.358 | v20 較低 |
| Mask PR area（mAP50） | 0.367 | 0.360 | v20 較低 |

R curve 並非 v20 在所有 thresholds 都比較高：

- 在 confidence 接近 0 的極低 threshold，v20 的 maximum macro Recall 明顯低於 v19。
- raster curve 顯示越過早期交叉區後，v20 在相當一段實用 confidence 範圍的 macro Recall 較高且下降較慢，這與 best-epoch operating point 的 Recall 上升一致。
- 但因沒有 curve 原始陣列，不能精確報告 crossover threshold 或宣稱「每一個」實用 threshold 都較高。
- v20 的最佳 F1 threshold 從約 0.35 移到約 0.29，代表 confidence calibration／P-R 平衡點改變；最佳 F1 本身沒有提高。

最大 macro Recall 的下降表示即使 threshold 降到接近 0，仍有大量 GT 沒有形成正確 class／IoU match。Mask maximum macro Recall 只有 0.55，不能靠單純調低 deployment threshold 解決。此數值也受 validation 唯一 `insect_bite` 的不穩定性強烈影響，但不改變「v20 沒有全面抬高 recall ceiling」的結論。

## Per-class Analysis

逐類 AP50 來自 best-checkpoint PR curve 圖例。現有 artifact 沒有逐類 AP50-95 的數值，因此不推測。

### Box AP50

| Class | v19 AP50 | v20 AP50 | Change | 判斷 |
| --- | ---: | ---: | ---: | --- |
| `abrasion` | 0.273 | 0.328 | +0.055 | 明顯改善，但 FP 增加 |
| `anthracnose` | 0.813 | 0.848 | +0.035 | 小幅改善，樣本太少 |
| `insect_bite` | 0.033 | 0.000 | -0.033 | 沒有受益；唯一 GT 仍誤分類 |
| `insect_track` | 0.347 | 0.257 | -0.090 | 明顯退步 |

### Mask AP50

| Class | v19 AP50 | v20 AP50 | Change | 判斷 |
| --- | ---: | ---: | ---: | --- |
| `abrasion` | 0.282 | 0.319 | +0.037 | 改善，但與 FP trade-off 並存 |
| `anthracnose` | 0.828 | 0.848 | +0.020 | 小幅改善，證據脆弱 |
| `insect_bite` | 0.033 | 0.000 | -0.033 | 4x exposure 未產生正確辨識 |
| `insect_track` | 0.325 | 0.274 | -0.051 | 退步，fragmentation 未解決 |

### Confusion-matrix correct-match rate

下表是 raw bounding-box confusion matrix 的對角線除以該類 GT 總數，不是 Mask Recall，也不是 PR curve 的官方逐類 Recall：

| Class | v19 correct / GT | v20 correct / GT | Change |
| --- | ---: | ---: | ---: |
| `abrasion` | 28 / 97 = 28.9% | 41 / 97 = 42.3% | +13 correct matches |
| `anthracnose` | 8 / 10 = 80.0% | 9 / 10 = 90.0% | +1 correct match |
| `insect_bite` | 0 / 1 = 0% | 0 / 1 = 0% | 無改善 |
| `insect_track` | 35 / 104 = 33.7% | 28 / 104 = 26.9% | -7 correct matches |

### `abrasion`

- Recall／match 能力確實改善：correct match `28 → 41`，background FN `68 → 53`，Box／Mask AP50 都上升。
- 代價是 background FP `31 → 52`，增加 21；這正好等於 total background FP 的淨增加量。
- prediction panels 中 v20 更常在正常皮孔、亮點、反光與局部色差上輸出 `abrasion`。例如 `val_batch2` 的無標註果面在 v19 沒有 prediction，v20 出現高 confidence `abrasion`；`val_batch1` 也出現額外的正常紋理 `abrasion`。
- 因此 v20 對小型擦傷更積極，部分漏檢轉為命中，但仍大量把正常果皮外觀當成 `abrasion`，而且比 v19 更嚴重。

### `anthracnose`

- 2x sampling 對 validation 指標方向正面：Box AP50 +0.035、Mask AP50 +0.020、correct match +1、FN -1。
- background FP `6 → 7`，沒有爆炸性增加。
- 然而 valid 只有 10 個 instances、4 張含 `anthracnose` 的圖片；一個 match 就會改變 10 個百分點的 confusion-matrix rate。單一 seed 下不能斷言 2x sampling 已穩定改善泛化。
- 合理結論是「有正向訊號，但樣本量不足以確認」。

### `insect_bite`

- 4x exposure 沒有讓模型在現有 validation 開始正確辨識：Box／Mask AP50 都由 0.033 降到 0.000。
- 唯一 GT 在 v19 與 v20 都被判成 `abrasion`，正確 TP 仍為 0。
- confusion matrix 的 background FN 是 0，不代表召回成功；該 GT 是被錯分到另一個 defect class，而不是 unmatched 到 background。
- 原圖中的 GT 很小，且同一果面有明顯、較大的擦傷外觀；這使結果特別容易被單一樣本主導。
- validation 只有 1 個 `insect_bite` GT，不能用這次 0／1 結果宣稱 4x sampling 一定無效；只能說現有 artifact **沒有提供有效的正向證據**。

### `insect_track`

- Box AP50 -0.090、Mask AP50 -0.051、correct match `35 → 28`、background FN `64 → 69`，整體退步。
- true `insect_track` 被預測成 `abrasion` 的次數由 4 增至 7；v20 更積極的 abrasion response 吸收了部分原本的 track GT。
- background FP `32 → 31`，track 自身 FP 沒有增加；主要問題是召回與分類品質下降。
- validation panels 的大型、細長、不規則 `insect_track` 仍只覆蓋局部片段；有些面積比 v19 稍大，但沒有形成一致、完整的 mask，且整體 AP／TP 反而下降。
- 所以不能宣稱 class-aware sampling 改善大型 track fragmentation。

## False Positive / False Negative Analysis

Ultralytics segmentation run 的這張 confusion matrix 使用 bounding-box IoU 配對。橫軸是 True、縱軸是 Predicted；background row 是 unmatched GT，background column 是 unmatched predictions。它不能直接量測 mask polygon 的重疊品質。

### False negatives：unmatched GT → background

| Class | v19 FN | v20 FN | Difference | 判斷 |
| --- | ---: | ---: | ---: | --- |
| `abrasion` | 68 | 53 | -15 | 改善 |
| `anthracnose` | 2 | 1 | -1 | 改善，但只有 10 GT |
| `insect_bite` | 0 | 0 | 0 | 唯一 GT 是誤分類，不是 unmatched |
| `insect_track` | 64 | 69 | +5 | 退步 |
| **Total** | **134** | **123** | **-11** | overall unmatched GT 改善 8.21% |

FN 的改善不是 rare classes 全面下降，而是主要由 `abrasion -15` 驅動；`insect_track` 反向增加 5。若把 misclassification 也視為沒有正確召回，`insect_bite` 兩組仍都是失敗。

### False positives：background → unmatched prediction

| Predicted class | v19 FP | v20 FP | Difference | 判斷 |
| --- | ---: | ---: | ---: | --- |
| `abrasion` | 31 | 52 | +21 | 大幅增加 |
| `anthracnose` | 6 | 7 | +1 | 小幅增加 |
| `insect_bite` | 0 | 0 | 0 | 無變化 |
| `insect_track` | 32 | 31 | -1 | 幾乎相同 |
| **Total** | **69** | **90** | **+21** | 增加 30.43% |

答案是：**是，v20 以更多 background FP 為代價換取 Recall，且新增 FP 幾乎全部來自 `abrasion`。**

這與 null image 相對 exposure 從 38.57% 降到 31.39%、positive entries 相對提高的方向一致。這是相關性，不足以單獨證明因果；但 confusion matrix、Precision 下降與 prediction panels 三種證據一致，因此不能把 FP 增加視為單純 metric noise。

## Training Curve Analysis

### Best epoch、plateau 與時間

| 項目 | v19 | v20 | 判斷 |
| --- | ---: | ---: | --- |
| Mask mAP50-95 best epoch | 186 | 161 | v20 早 25 epochs |
| best Mask mAP50-95 | 0.21046 | 0.20582 | v20 較低 |
| 首次達到自身 best 的 95% | epoch 139 | epoch 88 | v20 epoch-wise 較快 |
| 首次達到自身 best 的 99% | epoch 186 | epoch 88 | v20 很早進入近最佳 plateau |
| 到 best epoch 的 wall time | 14,177.0 s | 14,920.1 s | v20 仍較久，因每 epoch entries 更多 |
| 完整 200 epochs | 15,209.6 s | 18,495.5 s | v20 +21.60% |

v20 以 epoch 數看較快收斂，約 epoch 88 已達自身最終 best 的 99%；epoch 161 的單點最高只比該值高約 0.0017。以 wall clock 看，v20 的最終 best 並沒有更快到達，因每 epoch 多 22.86% samples。結論應是「**較早 plateau，但不是更省算力**」。v20 不需要更多 epoch 才找到可用結果，反而顯示 200 epochs 後半大部分是在帶有 overfitting 的 plateau 中搜尋小幅波動。

### Loss behavior

| Loss | v19 | v20 | 解讀 |
| --- | --- | --- | --- |
| `train/box_loss` | 1.73064 → 1.08579 | 1.67875 → 0.88788 | v20 final 較低，但 exposure／updates 不同 |
| `train/seg_loss` | 3.13180 → 1.43520 | 3.07023 → 1.21244 | v20 final 較低，不代表 validation 較好 |
| `train/cls_loss` | 27.74430 → 0.56177 | 17.44060 → 0.46289 | distribution 與 class exposure 改變，不能直接橫比 |
| `val/box_loss` | min 1.64774 @ 92；final 1.85171 | min 1.77193 @ 116；final 1.87416 | v19 minimum 較好；兩者後期回升 |
| `val/seg_loss` | min 2.26453 @ 98；final 2.54597 | min 2.41378 @ 41；final 2.69413 | v20 更早開始惡化，且全程較高 |
| `val/cls_loss` | min 1.09804 @ 142；final 1.15929 | min 1.37519 @ 107；final 1.55548 | v20 較高且後期 gap 較大 |

以 10-epoch rolling mean 看，v19 val seg loss 的最低區結束於約 epoch 101，v20 則約 epoch 65；20-epoch rolling mean 分別約 epoch 116 與 75。v20 的 segmentation validation gap 更早出現。

兩組 train losses 都一路下降，而 validation loss 在中後段回升，都是 overfitting／generalization gap。v20 的 val seg 與 val cls 較早惡化，但 val box 的回升幅度不比 v19 更大，因此不能簡化成「v20 所有 head 都更嚴重 overfit」。可支持的是：

1. v20 更早進入 plateau；
2. v20 後段持續降低 train loss，卻沒有把 Mask mAP 提高到 v19 以上；
3. class-aware exposure 讓 train cls／seg loss 更低，但 validation cls／seg loss 更高，沒有轉成更好泛化；
4. 因 sampling distribution 與每 epoch updates 不同，不能以較低 train loss 宣稱模型較好。

## Prediction Error Analysis

三組 validation label panels 在兩個 run 間逐位元相同，因此可直接比較同一張圖。這些 panels 只展示 validation 的一部分，不代表 93 張的完整 error audit。

### v20 明顯改善

- `val_batch2` 中部分極小 `abrasion`：v19 完全漏掉或只命中少數，v20 產生更多對應位置的 abrasion predictions，與 `abrasion` FN `68 → 53` 一致。
- 部分 `anthracnose` prediction confidence 與 match 更穩定，和 correct match `8 → 9`、AP50 小幅上升方向一致。
- 某些大型 `insect_track` 圖中，v20 的單一 mask 覆蓋面積比 v19 大，但仍遠小於 GT，不能視為 fragmentation 已解決。

### v20 明顯退步

- 無標註或近似乾淨果面出現更多 `abrasion`，包括正常皮孔、局部亮點、反光與色差；有些 confidence 達 0.7–0.9。
- `val_batch1` 的大型 `insect_track`／多瑕疵果面，v20 出現更多零碎 `abrasion`，而不是完整 track mask。
- `val_batch2` 的無標註果面，v20 額外產生 `abrasion` 或 `insect_track`，v19 則沒有 prediction。

### 兩者都失敗

- `val_batch0` 上方的大型、滿版 `insect_track`：兩者都只抓到少數局部小片，嚴重低估 mask union area。
- 細長、靠近 ROI 邊界的 `insect_track`：兩者都常切成多個碎片，或只覆蓋一小段。
- 非常小、接近正常皮孔尺度的標註：兩者都常漏檢或在附近錯位置產生 prediction。
- 唯一 `insect_bite` 不在顯示的 validation batch panels 中；confusion matrix 顯示兩者都將它判成 `abrasion`。

### 錯誤型態改變

- v19 的主要錯誤偏向漏檢；v20 把一部分 `abrasion` 漏檢轉為命中，但也把更多正常紋理轉為 abrasion FP。
- `insect_track → abrasion` 的 misclassification 從 4 增至 7，表示較積極的 abrasion response 侵蝕 track 分類品質。
- track background FP 沒增加，但 track AP、TP 與 FN 退步；問題不是單純「更多 track proposal」，而是 class discrimination／matching 變差。

## Hypothesis Evaluation

| Hypothesis | 結論 | 證據 |
| --- | --- | --- |
| H1: Rare-class exposure ↑ → rare-class Recall ↑ | **部分支持** | `anthracnose` AP、TP、FN 有小幅正向訊號；`insect_bite` 仍 0 TP、AP50 降至 0。rare classes 沒有全面受益。 |
| H2: Class-aware sampling → overall Recall ↑ | **部分支持** | selected best epoch 的 Box／Mask Recall 各提高約 0.062，total unmatched GT -11；但 maximum macro Recall 下降，且 `insect_track` FN 增加。 |
| H3: Rare positive exposure ↑ → background FP ↑ | **支持** | total background FP `69 → 90`（+30.43%），Precision 明顯下降，visual panels 有更多正常紋理 abrasion FP。 |
| H4: v20 不只是 P/R trade-off，而是真的改善 overall model quality | **不支持** | Box／Mask mAP50 與 mAP50-95 全部下降，最佳 F1 沒提高，`insect_bite`／`insect_track` 沒改善。 |

## v19 vs v20 Decision

### 選項

**B. v20 有部分改善，但存在 trade-off，暫時不能取代 v19。**

選 B 而不是 A 的原因：

1. 首要目標 Recall 的確提高，但不是所有類別提高，也不是 recall ceiling 提高。
2. Mask Precision 下降 0.08810，background FP 增加 21，代價明確且在 visual artifacts 可重現。
3. Mask mAP50 與 mAP50-95 沒有同步改善，最佳 F1 也沒有提高。
4. `anthracnose` 的小幅改善不足以抵銷 validation 樣本量風險。
5. 4x `insect_bite` 沒有可驗證收益；`insect_track` 明顯退步。
6. v20 使用更多 optimizer steps 與 21.60% 更多訓練時間，卻沒有產生更高整體 AP。

v20 值得保留為一個「提高 Recall 但犧牲 specificity」的實驗結果，但不應成為新的 incumbent。

## Recommendation for Experiment C

### 結論

**有條件建議進行 Experiment C，但使用 `v19 + targeted crop`，不使用 `v20 + targeted crop`。**

理由：

- v20 沒有證明 class-aware sampling 能穩定改善 rare classes；它主要改善 `abrasion` recall，且已造成顯著 background FP。
- 若在 v20 上再加 rare-class targeted crop，會同時疊加「repeated full images」與「targeted crops」兩種 positive exposure，結果無法判斷 targeted crop 本身是否有效。
- v19 是目前較高 Precision、mAP 與 `insect_track` AP 的 incumbent；在它上面只加入 targeted crop，才是更乾淨的下一個 controlled experiment。
- targeted crop 直接放大小型 rare defects，理論上比重複整張約 400 px ROI 更可能提高有效 defect pixels；但這仍需實驗證明。

Experiment C 應維持：

1. 同一 valid／test split、class mapping、model、epochs、batch、seed、device、Ultralytics／PyTorch 與 initial-weight hash；
2. 只改 train 中的 rare-class targeted crop，不同時保留 v20 repeated list；
3. 保存 crop manifest、來源 image 對應、class exposure、unique Fruit／View Observation 與 Dataset hash；
4. valid／test 不產生 crop，也不重新分割；
5. 主要成功條件包含 `anthracnose`／`insect_bite` Recall 與 AP 提升、`abrasion`／`insect_track` 不退步、background FP 不顯著增加；
6. 仍應優先補足真實 `insect_bite` validation/test GT，否則 Experiment C 對該類仍無法可靠驗收。

本次沒有修改或建立 Experiment C。

## Limitations

1. validation 的 `insect_bite` 只有 1 個 GT，不能可靠估計該類泛化能力；單一成功或失敗都不能驗收 sampling strategy。
2. 兩組都是單次 `seed=0`。小幅 mAP 差異可能只是 run-to-run variance；`deterministic=true` 也不能替代多 seed mean／standard deviation。
3. checkpoints 證明 Ultralytics 8.4.122、MPS device、Git commit 與記錄參數相同，但沒有保存 PyTorch training version 與 initial-weight SHA-256；這兩項 confounder 無法完全排除。
4. v20 的 train list 增至 532 entries，除了改變 class mix，也增加約 22.86% optimizer steps；不能把結果只歸因於 class prior。
5. PR／P／R／F1 只有 PNG，沒有原始 curve arrays；精確 crossover、任意 threshold 的 per-class P/R 與逐類 AP50-95 無法從現有 artifact 取得。
6. confusion matrix 以 bounding-box IoU matching 為主，不等同 mask polygon quality；大型 mask fragmentation 仍需結合 Mask PR／AP 與 visual panels 判讀。
7. validation panels 只顯示一部分 images，能識別典型錯誤但不能取代全 validation error audit。
8. 現有結果都是 training 過程的 validation。雖然兩份 YAML 的 test split 相同且現存 test 有 92 張，但本次沒有重新跑 test，不能把本報告描述成獨立 test 結果。
9. 既有 Dataset 有已知 fruit-group split、rare-class 樣本量、來源解析度與標註一致性限制；兩個 run 共同受到這些限制。

## Terminal Summary

```text
1. v19 best epoch: 186
   Mask P/R/mAP50/mAP50-95 = 0.70565 / 0.32401 / 0.36659 / 0.21046

2. v20 best epoch: 161
   Mask P/R/mAP50/mAP50-95 = 0.61755 / 0.38669 / 0.36034 / 0.20582

3. Mask Recall: +0.06268 (+19.35%)
4. Mask mAP50-95: -0.00464 (-2.20%)

5. Per-class Mask AP50:
   abrasion      0.282 -> 0.319  (+0.037)
   anthracnose   0.828 -> 0.848  (+0.020)
   insect_bite   0.033 -> 0.000  (-0.033)
   insect_track  0.325 -> 0.274  (-0.051)

6. Background unmatched GT (FN): 134 -> 123 (-11)
   abrasion 68 -> 53; anthracnose 2 -> 1;
   insect_bite 0 -> 0 (but both misclassified); insect_track 64 -> 69

7. Background unmatched predictions (FP): 69 -> 90 (+21)
   abrasion 31 -> 52; anthracnose 6 -> 7;
   insect_bite 0 -> 0; insect_track 32 -> 31

8. Decision: B — v20 improves selected-point Recall but does not outperform v19 overall.
9. Experiment C: conditionally recommended as v19 + targeted crop, not v20 + targeted crop.
10. Report: docs/experiments/v19-vs-v20-class-aware-sampling-analysis.md
```
