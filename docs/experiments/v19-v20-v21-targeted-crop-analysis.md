# Experiment C：Rare-class Targeted Crop 分析

- 分析日期：2026-09-01
- 任務：YOLO26s-seg 百香果局部瑕疵 instance segmentation
- Baseline：`training/runs/segment/v19_defect_epoch200_pw050_cp01`
- Experiment B：`training/runs/segment/v20_B_class_aware`
- Experiment C：`training/runs/segment/v21_C_targeted_crop`
- 主模型：三個 run 各自的 `weights/best.pt`，未使用 `last.pt`
- 本次工作：只讀既有 Dataset、weights 與 training outputs；沒有修改 Dataset、重新訓練或覆寫 run

## 1. Executive Summary

**結論：Experiment C 有價值，但沒有穩定優於 v19 或 v20；下一階段 baseline 仍建議使用 v19。**

Experiment C 最明確的成功是降低 false positives，而不是全面提高 rare-class recall：

- validation 的 all-image bounding-box unmatched predictions 為 `69 / 90 / 54`（v19 / v20 / v21）；v21 比 v20 少 36（-40.00%），也比 v19 少 15（-21.74%）。
- validation 純背景 29 張中，FP predictions 為 `3 / 6 / 2`；test 純背景 35 張中為 `7 / 11 / 8`。C 明顯修復 B 的 background hallucination，但 test 仍比 v19 多 1 個 FP。
- validation Mask Recall 為 `0.32160 / 0.38418 / 0.39694`；v21 取得最高值，同時 Mask Precision `0.65537` 也高於 v20 `0.61681`。
- 但 test Mask Recall 為 `0.30096 / 0.36138 / 0.28042`；v21 最低。C 的 Recall 改善沒有跨 split 重現。
- v21 test Mask mAP50-95 `0.16777` 是三者最高，比 v19 +11.75%、比 v20 +12.21%；但 validation Mask mAP50-95 `0.18795` 卻是最低，比 v19 -10.36%。

在 rare classes 上，C 沒有提供「targeted crop 明確優於 full-image repetition」的完整證據：

- `anthracnose`：v21 validation Mask P/R/AP50 為 `0.77283 / 0.90000 / 0.88303`，優於 v20 的 `0.59167 / 0.90000 / 0.84817`；但 Mask AP50-95 是 `0.52287`，低於 v20 `0.58348`。test 時 v21 的 Recall、AP50 與固定 threshold TP 都略低於 v20，只有 Precision 與 AP50-95 較高。
- `insect_bite`：validation 只有 1 個 GT、test 只有 3 個 GT。三模型在 `conf=0.25` 的 validation/test 正確 TP 都是 0。v21 test Mask AP50 `0.08386` 來自較低 confidence 的 curve behavior，不能視為已成功學會此類別。

在 common classes 上，C 的代價不可忽略：

- `abrasion` 的 test Precision、AP50、AP50-95 與 FP 都優於 v19/v20，但 Recall 只有 `0.26531`，低於 v19 `0.36295` 與 v20 `0.38825`。
- `insect_track` 的 test Mask P/R/AP50/AP50-95 為 `0.35663 / 0.22000 / 0.19465 / 0.08287`；AP50 比 v19 -25.20%、比 v20 -25.69%，固定 threshold TP 也只有 `22/100`。

因此 controlled-experiment 判斷是：targeted crop 比 repeated full-image sampling 更能抑制 FP，且對 `anthracnose` 有部分 quality/specificity 訊號；但它沒有可靠取得 B 的 test Recall，也沒有保住 `insect_track`。這支持「effective defect pixels/scale 確實是因素之一」，但不支持「只要 targeted crop 就能解決 rare-class segmentation」。

## 2. Experiment Setup

### 2.1 Run 與 weight 路徑

| Model | Run | `best.pt` SHA-256 |
| --- | --- | --- |
| v19 | `training/runs/segment/v19_defect_epoch200_pw050_cp01` | `77bd843f8fdd6132167a21d422f1decc155b621e806ffab20f8ac3e0c529e7ce` |
| v20 | `training/runs/segment/v20_B_class_aware` | `6e4ef9932b99d23990638f5ea429d36549787022c16a747c355279a4006e7d85` |
| v21 | `training/runs/segment/v21_C_targeted_crop` | `c4985236e120a118712767e40e0bf55063bdb2df885d74f7786d07ec39a40cf7` |

三個 checkpoint 都是 `SegmentationModel`，class names 完全相同：

```text
0 abrasion
1 anthracnose
2 insect_bite
3 insect_track
```

### 2.2 Training arguments 公平性

三份 `args.yaml` 的完整 diff 只有 `data`、`name`、`save_dir`。其餘可觀察參數一致。

| Parameter | v19 | v20 | v21 |
| --- | --- | --- | --- |
| `model` | `yolo26s-seg.pt` | same | same |
| `imgsz` | 1024 | 1024 | 1024 |
| `epochs` | 200 | 200 | 200 |
| `patience` | 50 | 50 | 50 |
| train `batch` | 4 | 4 | 4 |
| `device` | `mps` | `mps` | `mps` |
| `seed` | 0 | 0 | 0 |
| `deterministic` | `true` | `true` | `true` |
| `cls_pw` | 0.5 | 0.5 | 0.5 |
| `copy_paste` | 0.1 | 0.1 | 0.1 |
| `copy_paste_mode` | `mixup` | `mixup` | `mixup` |
| `optimizer` | `auto` | `auto` | `auto` |
| recorded `lr0` / `lrf` | 0.01 / 0.01 | same | same |
| `weight_decay` | 0.0005 | 0.0005 | 0.0005 |
| warmup epochs / momentum | 3.0 / 0.8 | same | same |
| `data` | `data.yaml` | `data_exp_B.yaml` | `data_exp_C.yaml` |

Ultralytics 8.4.122 的 `optimizer=auto` 會忽略記錄的 `lr0=0.01` 與 `momentum=0.937`。依三個 Dataset entry count、batch、epochs 與該版本的 `build_optimizer` 邏輯，三者 estimated iterations 都低於 10,000，因此都解析為 AdamW，effective `lr0=0.00125`、momentum/beta1 `0.9`，並把 runtime `warmup_bias_lr` 改為 `0.0`。checkpoint 已 strip optimizer，沒有保存 resolved optimizer class；這個結果是依當次精確 Ultralytics 版本與可驗證程式邏輯重建，不是從 optimizer state 讀出。

### 2.3 Augmentation parameters

| Group | 三者共同值 |
| --- | --- |
| HSV | `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4` |
| Geometric | `degrees=0`, `translate=0.1`, `scale=0.5`, `shear=0`, `perspective=0` |
| Flip | `flipud=0`, `fliplr=0.5` |
| Composition | `mosaic=1.0`, `mixup=0`, `cutmix=0`, `copy_paste=0.1`, `copy_paste_mode=mixup` |
| Schedule | `close_mosaic=10` |
| Segmentation | `mask_ratio=4`, `overlap_mask=true` |
| Other recorded args | `bgr=0`, `auto_augment=randaugment`, `erasing=0.4` |

`auto_augment` 與 `erasing` 是共同記錄值，不能當成三組差異。沒有發現 C 額外加入 HSV、rotation、blur 或其他 hidden augmentation。

### 2.4 Split 與 class mapping

| Split | Images | Background images | abrasion | anthracnose | insect_bite | insect_track |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 93 | 29 | 97 | 10 | 1 | 104 |
| test | 92 | 35 | 98 | 22 | 3 | 100 |

三份 data YAML 的 `val` 都解析到同一個 `valid/images`，`test` 都解析到同一個 `test/images`，`nc=4` 與 names 完全一致。現存 split content hashes：

| Split | Image content SHA-256 aggregate | Label content SHA-256 aggregate |
| --- | --- | --- |
| valid | `e224b48e938ed4cf52b112c74830c07a65deff3b5dfc7694d39c1d62f67c0eb5` | `581b209cd2f4e146cd2f009aea8d7b9ca570b8b9402b53569fc0645246bf5f83` |
| test | `5809efd77a019ca10dcd7d16000a06c4e67c05efdb3144b5227b743e45dffade` | `556f2bdac0614e0c461d744adfa834b0bef53b0817c0f021fe66a8bf9a409bb8` |

Experiment C 的 generation validation 另確認 train/valid/test exact image hashes 無 overlap、source manifest generation 前後 0 mismatch、132 crops 全部來自 train，且沒有 target polygon loss 或 malformed labels。

### 2.5 Evaluation protocol

三個 `best.pt` 在 valid/test 使用完全相同設定：

```text
Ultralytics 8.4.122
PyTorch 2.13.0
device=mps
imgsz=1024
evaluation batch=8
workers=0
NMS IoU=0.70
max_det=300
curve confidence floor=0.001
```

`evaluation batch=8` 是 Ultralytics trainer 對 train `batch=4` 的實際 validation batch（detection/segmentation validation 使用 2× train batch），因此能對齊先前 v19/v20 artifacts。新 validation confusion matrix 精確重現舊報告的 v19/v20 FP `69/90` 與 FN `134/123`。

額外 fixed-operating-point audit 使用：

```text
confidence=0.25
NMS IoU=0.70
mask/box TP matching IoU=0.50
class-aware one-to-one greedy matching
```

Ultralytics confusion matrix 則沿用其既有 box IoU `0.45`。官方 P/R 是 curve-derived operating point；固定 threshold 的 TP/FP/FN 是另一口徑，兩者不可直接互相除算。

## 3. Dataset Exposure Comparison

| Class / item | v19 | Exp B | Exp C | C vs B |
| --- | ---: | ---: | ---: | ---: |
| train entries | 433 | 532 | 565 | +33 (+6.20%) |
| unique original train images | 433 | 433 | 433 | same |
| generated crops | 0 | 0 | 132 | +132 |
| abrasion exposure | 577 | 753 | 614 | -139 (-18.46%) |
| anthracnose exposure | 54 | 122 | 128 | +6 (+4.92%) |
| insect_bite exposure | 26 | 104 | 105 | +1 (+0.96%) |
| insect_track exposure | 386 | 497 | 430 | -67 (-13.48%) |
| null exposure | 167 | 167 | 167 | same |

這是本實驗最有辨識力的部分：B/C 的 `anthracnose` 與 `insect_bite` exposure 非常接近，但 B 重複 full image，C 把 rare instance 放大到 crop。C crop 的 target bbox pixel-area share 相對原圖中位數放大 18.305×，crop 寬高中位數 100.5 pixels，訓練時再 resize 到 1024。

同時必須保留一個 confounder：C 比 B 多 6.20% train entries、比 v19 多 30.48%，所以 C 不是「相同 optimizer updates 下只換 framing」；它比較的是兩個完整 Dataset intervention。C 也降低 common-class incidental exposure，這既是設計目的，也是解讀 common-class 退步時的重要因素。

## 4. Training Dynamics

![Training dynamics](../../training/analysis/v19-v20-v21-targeted-crop/figures/training_dynamics.png)

### 4.1 Best、final 與 early stopping

Ultralytics 8.4.122 的 segmentation fitness 是：

```text
fitness = Box mAP50-95 + Mask mAP50-95
```

它不是單一 Mask mAP。v21 的 `best.pt` 因此對應 fitness 最高的 epoch 124，不是 Mask mAP50-95 單項最高的 epoch 129。

| Item | v19 | v20 | v21 |
| --- | ---: | ---: | ---: |
| completed epochs | 200 | 200 | 174 |
| early stopping | no | no | yes |
| best fitness epoch | 186 | 161 | 124 |
| best fitness | 0.42619 | 0.41029 | 0.38144 |
| best Mask mAP50-95 epoch | 186 | 161 | 129 |
| best Mask mAP50-95 in `results.csv` | 0.21046 | 0.20582 | 0.19101 |
| first epoch ≥95% own best Mask AP | 139 | 88 | 119 |
| first epoch ≥99% own best Mask AP | 186 | 88 | 129 |
| wall time to best fitness | 14,177.0 s | 14,920.1 s | 12,134.4 s |
| total wall time | 15,209.6 s | 18,495.5 s | 17,010.3 s |

v21 比 v20 慢進入 epoch-wise near-best plateau（119 vs 88），但比 v19 快（119 vs 139），且最早以 wall time 找到 best fitness。它在 epoch 124 後 50 epochs 沒再改善 fitness，於 epoch 174 安全 early stop。因 C 每 epoch entries 最多，不能只用 epoch 數宣稱「收斂更快」。

### 4.2 Loss

本版本 YOLO26s-seg 的 `results.csv` 沒有 `train/dfl_loss` 或 `val/dfl_loss`；它記錄 `l1_loss` 與 `sem_loss`。因此 DFL 是 unavailable，不以其他 loss 冒充。

| Train loss | v19 start → final | v20 start → final | v21 start → final |
| --- | --- | --- | --- |
| box | 1.73064 → 1.08579 | 1.67875 → 0.88788 | 1.73545 → 1.09597 |
| seg | 3.13180 → 1.43520 | 3.07023 → 1.21244 | 3.30162 → 1.63466 |
| cls | 27.74430 → 0.56177 | 17.44060 → 0.46289 | 14.99990 → 0.80583 |
| l1 | 0.01267 → 0.01041 | 0.01126 → 0.00804 | 0.01355 → 0.00913 |

| Validation loss | v19 min @ epoch → final | v20 min @ epoch → final | v21 min @ epoch → final |
| --- | --- | --- | --- |
| box | 1.64774 @ 92 → 1.85171 | 1.77193 @ 116 → 1.87416 | 1.63916 @ 96 → 1.79278 |
| seg | 2.26453 @ 98 → 2.54597 | 2.41378 @ 41 → 2.69413 | 2.28075 @ 99 → 2.48901 |
| cls | 1.09804 @ 142 → 1.15929 | 1.37519 @ 107 → 1.55548 | 1.54821 @ 146 → 1.57767 |
| l1 | 0.01687 @ 179 → 0.01869 | 0.01822 @ 17 → 0.01986 | 0.01640 @ 1 → 0.01794 |

三者都有 train loss 持續下降、validation loss 在中後段回升的 generalization gap。v20 的 val seg loss 最早惡化；v21 的 raw val seg minimum epoch 與 v19 相近（99 vs 98），但 10-epoch rolling minimum 約在 epoch 87，之後已開始回升。v21 val box loss 是三者最好，但 val cls loss 最差，與 rare/common class discrimination 不穩定一致。

v21 後 20 epochs 的 Mask mAP50-95 範圍是 `0.13457–0.18602`，Mask Precision `0.33005–0.71810`，Mask Recall `0.31292–0.43731`；波動明顯。v21 final epoch Mask P/R/mAP50-95 為 `0.34858 / 0.43731 / 0.16118`，不可拿來取代 epoch 124 的 `best.pt`。

## 5. Overall Validation Results

| Metric | v19 | v20 | v21 |
| --- | ---: | ---: | ---: |
| Mask Precision | 0.70071 | 0.61681 | 0.65537 |
| Mask Recall | 0.32160 | 0.38418 | 0.39694 |
| Mask mAP50 | 0.36681 | 0.36040 | 0.37417 |
| Mask mAP50-95 | 0.20967 | 0.20377 | 0.18795 |
| Box Precision | 0.68911 | 0.61278 | 0.67715 |
| Box Recall | 0.31855 | 0.38063 | 0.34109 |
| Box mAP50 | 0.36664 | 0.35835 | 0.36878 |
| Box mAP50-95 | 0.21582 | 0.20537 | 0.19364 |

| Metric | v20 − v19 | v21 − v19 | v21 − v20 |
| --- | ---: | ---: | ---: |
| Mask Precision | -0.08390 (-11.97%) | -0.04535 (-6.47%) | +0.03855 (+6.25%) |
| Mask Recall | +0.06258 (+19.46%) | +0.07533 (+23.42%) | +0.01275 (+3.32%) |
| Mask mAP50 | -0.00642 (-1.75%) | +0.00736 (+2.01%) | +0.01378 (+3.82%) |
| Mask mAP50-95 | -0.00590 (-2.81%) | -0.02172 (-10.36%) | -0.01582 (-7.76%) |
| Box Precision | -0.07633 (-11.08%) | -0.01196 (-1.74%) | +0.06436 (+10.50%) |
| Box Recall | +0.06208 (+19.49%) | +0.02255 (+7.08%) | -0.03954 (-10.39%) |
| Box mAP50 | -0.00828 (-2.26%) | +0.00214 (+0.58%) | +0.01042 (+2.91%) |
| Box mAP50-95 | -0.01045 (-4.84%) | -0.02218 (-10.28%) | -0.01173 (-5.71%) |

v21 在 validation 取得 B 的 Mask Recall，且 Precision、Mask mAP50、Box Precision、Box mAP50 都優於 B；但 Mask/Box mAP50-95 都更差。這表示 C 在 IoU=0.5 的辨識與 operating point 改善，卻沒有同步提高嚴格 localization quality。

## 6. Overall Test Results

| Metric | v19 | v20 | v21 |
| --- | ---: | ---: | ---: |
| Mask Precision | 0.35381 | 0.34176 | 0.36630 |
| Mask Recall | 0.30096 | 0.36138 | 0.28042 |
| Mask mAP50 | 0.30571 | 0.31460 | 0.31710 |
| Mask mAP50-95 | 0.15013 | 0.14951 | 0.16777 |
| Box Precision | 0.35264 | 0.41730 | 0.37906 |
| Box Recall | 0.30107 | 0.31605 | 0.28824 |
| Box mAP50 | 0.30968 | 0.32083 | 0.32287 |
| Box mAP50-95 | 0.16757 | 0.17017 | 0.18790 |

| Metric | v20 − v19 | v21 − v19 | v21 − v20 |
| --- | ---: | ---: | ---: |
| Mask Precision | -0.01204 (-3.40%) | +0.01250 (+3.53%) | +0.02454 (+7.18%) |
| Mask Recall | +0.06042 (+20.07%) | -0.02055 (-6.83%) | -0.08096 (-22.40%) |
| Mask mAP50 | +0.00888 (+2.91%) | +0.01139 (+3.72%) | +0.00251 (+0.80%) |
| Mask mAP50-95 | -0.00062 (-0.42%) | +0.01763 (+11.75%) | +0.01826 (+12.21%) |
| Box Precision | +0.06466 (+18.34%) | +0.02642 (+7.49%) | -0.03824 (-9.16%) |
| Box Recall | +0.01499 (+4.98%) | -0.01283 (-4.26%) | -0.02781 (-8.80%) |
| Box mAP50 | +0.01115 (+3.60%) | +0.01318 (+4.26%) | +0.00204 (+0.64%) |
| Box mAP50-95 | +0.00260 (+1.55%) | +0.02034 (+12.14%) | +0.01773 (+10.42%) |

test 的結論和 validation 不同：v21 mAP50-95 明顯提高，但 Recall 最低。C 沒有穩定重現 validation 的 Recall advantage；不能只用 test mAP50-95 決定新 baseline。

## 7. Per-class Results

以下 P/R/AP 是官方 curve-derived metrics；`GT/Pred/TP/FP/FN` 是 `conf=0.25`、Mask IoU 0.50 的 fixed audit。

### 7.1 Validation

| Class | Model | Mask P | Mask R | AP50 | AP50-95 | GT | Pred | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abrasion | v19 | 0.47902 | 0.22680 | 0.28160 | 0.11666 | 97 | 61 | 26 | 35 | 71 |
|  | v20 | 0.39861 | 0.39634 | 0.31910 | 0.13461 | 97 | 98 | 37 | 61 | 60 |
|  | v21 | 0.41621 | 0.36082 | 0.29587 | 0.11664 | 97 | 65 | 30 | 35 | 67 |
| anthracnose | v19 | 0.69365 | 0.80000 | 0.82753 | 0.57442 | 10 | 15 | 8 | 7 | 2 |
|  | v20 | 0.59167 | 0.90000 | 0.84817 | 0.58348 | 10 | 15 | 9 | 6 | 1 |
|  | v21 | 0.77283 | 0.90000 | 0.88303 | 0.52287 | 10 | 9 | 8 | 1 | 2 |
| insect_bite | v19 | 1.00000 | 0.00000 | 0.03317 | 0.01658 | 1 | 0 | 0 | 0 | 1 |
|  | v20 | 1.00000 | 0.00000 | 0.00000 | 0.00000 | 1 | 0 | 0 | 0 | 1 |
|  | v21 | 1.00000 | 0.00000 | 0.00000 | 0.00000 | 1 | 0 | 0 | 0 | 1 |
| insect_track | v19 | 0.63020 | 0.25962 | 0.32495 | 0.13102 | 104 | 60 | 29 | 31 | 75 |
|  | v20 | 0.47697 | 0.24038 | 0.27430 | 0.09700 | 104 | 71 | 30 | 41 | 74 |
|  | v21 | 0.43243 | 0.32692 | 0.31779 | 0.11230 | 104 | 54 | 29 | 25 | 75 |

### 7.2 Test

| Class | Model | Mask P | Mask R | AP50 | AP50-95 | GT | Pred | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abrasion | v19 | 0.41085 | 0.36295 | 0.27830 | 0.10694 | 98 | 77 | 33 | 44 | 65 |
|  | v20 | 0.35878 | 0.38825 | 0.28758 | 0.11411 | 98 | 89 | 40 | 49 | 58 |
|  | v21 | 0.49503 | 0.26531 | 0.31286 | 0.13987 | 98 | 49 | 26 | 23 | 72 |
| anthracnose | v19 | 0.65724 | 0.59091 | 0.67552 | 0.38899 | 22 | 21 | 13 | 8 | 9 |
|  | v20 | 0.59197 | 0.72727 | 0.70886 | 0.39111 | 22 | 26 | 16 | 10 | 6 |
|  | v21 | 0.61356 | 0.63636 | 0.67704 | 0.40042 | 22 | 25 | 15 | 10 | 7 |
| insect_bite | v19 | 0.00000 | 0.00000 | 0.00882 | 0.00529 | 3 | 0 | 0 | 0 | 3 |
|  | v20 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 3 | 1 | 0 | 1 | 3 |
|  | v21 | 0.00000 | 0.00000 | 0.08386 | 0.04790 | 3 | 2 | 0 | 2 | 3 |
| insect_track | v19 | 0.34713 | 0.25000 | 0.26022 | 0.09931 | 100 | 54 | 26 | 28 | 74 |
|  | v20 | 0.41629 | 0.33000 | 0.26194 | 0.09281 | 100 | 66 | 28 | 38 | 72 |
|  | v21 | 0.35663 | 0.22000 | 0.19465 | 0.08287 | 100 | 61 | 22 | 39 | 78 |

### 7.3 Rare-class interpretation

#### anthracnose

- v19 → v20：validation Recall +0.10、AP50 +0.02064；test Recall +0.13636、AP50 +0.03335。B 的 repeated sampling 對 anthracnose Recall 有一致正向訊號，但 Precision 較低。
- v19 → v21：validation P/R/AP50 都提高，但 AP50-95 -0.05155；test Recall +0.04545、AP50 幾乎持平、AP50-95 +0.01143。C 更像改善 sensitivity/specificity balance，沒有穩定改善嚴格 mask localization。
- v20 → v21：validation Precision +0.18116、AP50 +0.03485、同為 Recall 0.90，但 AP50-95 -0.06061；test Precision +0.02158、AP50-95 +0.00931，但 Recall -0.09091、TP `16→15`、AP50 -0.03182。

所以 C 對 anthracnose 並非全面優於 B。它在 validation 明顯更少 FP（fixed FP `1` vs `6`），但 test 的 Recall/TP/AP50 略差。valid 只有 10 個、test 22 個 GT；一兩個 instance 已足以改變結論。

#### insect_bite

valid `1` 個 GT、test `3` 個 GT。三模型在 fixed `conf=0.25` 都是 validation TP `0/1`、test TP `0/3`。v21 test AP50 `0.08386` 表示低 confidence ranking 中出現有限正確 overlap，但同時 fixed threshold 有 2 個 prediction、2 個 FP、0 TP。這只能視為弱訊號，不能宣稱 targeted crop 成功。

### 7.4 Common-class preservation

C 沒有全面保留 common classes：

- `abrasion`：C 大幅減少 test predictions/FP，並提高 AP，但 Recall 與 TP 明顯下降。它由 B 的 aggressive response 拉回較保守 operating behavior。
- `insect_track`：validation fixed TP 與 v19 相同且 FP 最少，但 test TP、Recall、AP50、AP50-95 全部最低。C 的 common-class incidental exposure 比 B 少 13.48%，加上 crop scale/context shift，可能共同造成 test degradation。

## 8. Background False Positive Analysis

![Background FP comparison](../../training/analysis/v19-v20-v21-targeted-crop/figures/background_fp_comparison.png)

### 8.1 純背景 images

| Split | Model | Background images | Images with FP | Total FP predictions | FP/image | abrasion | anthracnose | insect_bite | insect_track |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | v19 | 29 | 2 | 3 | 0.10345 | 3 | 0 | 0 | 0 |
|  | v20 | 29 | 4 | 6 | 0.20690 | 5 | 0 | 0 | 1 |
|  | v21 | 29 | 1 | 2 | 0.06897 | 2 | 0 | 0 | 0 |
| test | v19 | 35 | 4 | 7 | 0.20000 | 4 | 2 | 0 | 1 |
|  | v20 | 35 | 9 | 11 | 0.31429 | 7 | 3 | 0 | 1 |
|  | v21 | 35 | 6 | 8 | 0.22857 | 4 | 3 | 0 | 1 |

### 8.2 所有 images 的 unmatched box predictions

此表沿用舊 v19/v20 報告的 Ultralytics confusion-matrix 方法：`conf=0.25`、box IoU 0.45，background column 是 unmatched prediction。

| Split | Model | abrasion FP | anthracnose FP | insect_bite FP | insect_track FP | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| valid | v19 | 31 | 6 | 0 | 32 | 69 |
|  | v20 | 52 | 7 | 0 | 31 | 90 |
|  | v21 | 26 | 1 | 0 | 27 | 54 |
| test | v19 | 39 | 6 | 0 | 29 | 74 |
|  | v20 | 49 | 9 | 1 | 34 | 93 |
|  | v21 | 21 | 8 | 1 | 29 | 59 |

C 明確解決 B 的 background FP 增加：validation total `90→54`，test `93→59`。它甚至低於 v19 的 `69/74`。但這部分來自較保守 prediction distribution；test all-image FN 同時是 v19/v20/v21 `145/130/154`，所以不是免費改善，而是 specificity/recall trade-off。

## 9. Error Analysis

### 9.1 Fixed-threshold total errors

| Split | Model | Predictions | Mask TP | Mask FP | Mask FN | Duplicate predictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| valid | v19 | 136 | 63 | 73 | 149 | 7 |
|  | v20 | 184 | 76 | 108 | 136 | 5 |
|  | v21 | 128 | 67 | 61 | 145 | 8 |
| test | v19 | 152 | 72 | 80 | 151 | 13 |
|  | v20 | 182 | 84 | 98 | 139 | 6 |
|  | v21 | 137 | 63 | 74 | 160 | 9 |

v20 的主要型態是更多 proposals、更多 TP、也更多 FP；v21 則更保守，FP 最少但 test FN 最多。同一 defect 的 duplicate prediction 三者都有，v21 validation 主要是 6 個 `insect_track` duplicates，沒有證據顯示 targeted crop 消除了 fragmentation。

### 9.2 Class confusion

在 fixed `conf=0.25` 且 best-overlap Mask IoU ≥0.50 時：

- validation `insect_track → abrasion`：v19 4、v20 9、v21 3。C 明顯降低 B 的 track-to-abrasion confusion。
- test `insect_track → abrasion`：v19 3、v20 1、v21 4。這個改善沒有跨 split 重現。
- 唯一 validation `insect_bite`：三者都被 `abrasion` 吸收；test 的 3 個 insect_bite 也沒有正確 TP。

### 9.3 代表性 instance patterns

以下是同一 GT instance 以 `conf=0.25`、Mask IoU 0.50 判定的代表案例；檔名只供本機 Dataset audit，不將照片複製到 Git 文件。

| Pattern | Split / image / class | v19 | v20 | v21 |
| --- | --- | --- | --- | --- |
| v19 錯、v20 對、v21 對 | test `fruit_182_img_02...jpg`, abrasion | wrong | IoU 0.709, conf 0.765 | IoU 0.692, conf 0.619 |
| v19 錯、v20 錯、v21 對 | test `img_01_jpg.rf.105b0923...jpg`, insect_track | predicted abrasion, IoU 0.745 | predicted abrasion, IoU 0.478 | correct IoU 0.628, conf 0.271 |
| v19 對、v20 錯、v21 對 | test `img_01_jpg.rf.1fda86d4...jpg`, abrasion | IoU 0.750, conf 0.647 | predicted insect_track, IoU 0.833 | IoU 0.771, conf 0.340 |
| v19 對、v20 對、v21 錯 | test `fruit_135_img_02...jpg`, insect_track | IoU 0.815, conf 0.353 | IoU 0.850, conf 0.900 | miss；best same-class IoU 0.091 |

跨 valid+test 的此四種 pattern 分別有 `19 / 15 / 15 / 22` 個 GT instances。v21 能修正部分 v19/v20 error，但「v19/v20 都對、v21 錯」反而最多，且包含 13 個 test GT（6 insect_track、6 abrasion、1 anthracnose）。這與 test FN 增加一致。

### 9.4 Mask size 與 texture sensitivity

prediction panels 顯示三者對大型、細長、不規則 `insect_track` 仍常只覆蓋局部片段；v21 也會把同一大型 track 切成多個 masks。AP50 與 AP50-95 的落差及 v21 validation `anthracnose` AP50 上升但 AP50-95 下降，顯示 boundary/localization 仍是瓶頸。

C 沒有出現 B 那種全面增加正常果皮 abrasion response 的程度：純背景與 all-image FP 都下降。可是 test 最佳 F1 threshold 移到 0.088，若 deployment 為追求 Recall 而下調 threshold，局部紋理 FP 很可能重新增加；本報告沒有用不同 threshold 取代主要 comparison。

## 10. PR Curve / Threshold Behavior

![Mask PR and F1 comparison](../../training/analysis/v19-v20-v21-targeted-crop/figures/mask_pr_f1_comparison.png)

| Split | Model | Best Mask macro F1 | Confidence | Maximum macro Recall @ conf≈0 |
| --- | --- | ---: | ---: | ---: |
| valid | v19 | 0.36128 | 0.318 | 0.86690 |
|  | v20 | 0.36130 | 0.312 | 0.59190 |
|  | v21 | 0.40754 | 0.206 | 0.64953 |
| test | v19 | 0.33471 | 0.196 | 0.66405 |
|  | v20 | 0.35096 | 0.210 | 0.57203 |
|  | v21 | 0.34456 | 0.088 | 0.85746 |

Experiment B 的 validation 問題不是只靠 threshold 可解：最佳 F1 幾乎等於 v19，PR area/mAP 又較低，且 FP 明顯更多。test 時 B 的最佳 F1 略高，但仍伴隨最高 background FP。

Experiment C 的 confidence calibration/operating point 明顯位移：validation 最佳 threshold 約 0.206，test 約 0.088。validation 最佳 F1 有實質提高，表示不只是 threshold shift；但 test F1 沒有優於 B，且最佳 threshold 跨 split 不穩定。C 的低 selected Recall 部分可透過 threshold 改變，但不能把它描述成「只要調 threshold 就全面優於」。

## 11. v19 vs v20 vs v21 Controlled Comparison

### Q1：Experiment C 是否優於 baseline v19？

- **Overall performance：混合。** validation Mask mAP50 +2.01%、Recall +23.42%，但 mAP50-95 -10.36%；test mAP50-95 +11.75%、Precision +3.53%，但 Recall -6.83%。
- **Rare-class performance：部分正向、證據不足。** anthracnose 有 P/R/AP 正向訊號，但 strict AP 不一致；insect_bite 仍 0 fixed-threshold TP。
- **Common-class performance：沒有全面保留。** abrasion 更精準但 test Recall 下降；insect_track test AP50 -25.20%、TP `26→22`。
- **Background FP：大致優於。** all-image FP validation/test `69→54`、`74→59`；純背景 test `7→8` 則略差。

因此不能回答單純 yes。C 比 v19 更保守、部分 AP 更高，但 recall/common-class robustness 較差。

### Q2：Experiment C 是否優於 Experiment B？

在 rare exposure 近似下，**targeted crop 明顯優於 repeated sampling 的是 specificity/background FP，而不是全面 rare-class segmentation**：

- validation Mask P/R/AP50 都優於 B；test Mask Precision、mAP50、mAP50-95 也優於 B。
- 但 test Mask Recall -22.40%；anthracnose test TP `16→15`、AP50 下降；insect_bite 仍 0 TP；insect_track test AP/TP 更差。

所以 targeted crop 沒有被證明「整體比 full-image repetition 更好」。它提供更好的 FP 控制與部分 localization quality，但犧牲 test recall。

### Q3：C 是否解決 B 的主要副作用？

| B side effect | C outcome |
| --- | --- |
| Precision decline | **部分解決**：C 的 Mask P 在 valid/test 都高於 B |
| mAP decline | **部分解決**：mAP50 與 test mAP50-95 高於 B；validation mAP50-95 更低 |
| background FP increase | **解決**：all-image FP `90→54` valid、`93→59` test |
| common-class degradation | **未解決**：abrasion specificity 改善，但 insect_track test 明顯退步 |

### Q4：推薦哪個下一階段 baseline？

**推薦 v19。**

理由不是 v19 每個 metric 最大，而是它目前提供最穩定、最可解釋的共同基準：

1. v21 沒有在 test 重現 Recall improvement，且 insect_track 明顯退步。
2. insect_bite 樣本太少且三者仍 0 TP，無法用這個核心 target 驗收 C。
3. v21 的 validation/test optimal threshold 差距大，顯示 calibration/distribution sensitivity。
4. v19 沒有 B 的 FP 膨脹，也沒有 C 的 test common-class recall collapse。
5. 用 v19 作下一個 controlled experiment 可保持「只新增一個 Dataset intervention」的乾淨因果解讀。

v21 應保留為 targeted-crop research branch，不應刪除，也不應用 v20 取代它。

## 12. Interpretation

數據部分支持以下假設：

> rare-class 問題不只是出現次數太少，也與 defect 在 full-resolution image 中占據的有效像素比例太低有關。

支持證據：B/C rare exposure 幾乎相同，但 C 的 validation anthracnose Precision/AP50、overall Mask AP50、最佳 F1，以及 validation/test background FP 都優於 B。放大 rare defect pixels 確實改變了模型的 ranking/discrimination，而不是只複製 B 的 recall/FP trade-off。

但結果也顯示有效像素比例不是唯一瓶頸：

- insect_bite 仍沒有可靠 TP，表示 annotation variance、class separability 或樣本 diversity 可能比 exposure 更重要。
- C 的 test insect_track degradation、threshold shift 與 validation strict AP 下降，符合 crop context 不足、train/inference scale mismatch 或 partial-context distribution shift。
- C crops 會保留 crop 內其他 class；common classes 的 exposure 與外觀 framing 因此也被改變，不能把所有差異只歸因於 rare target pixels。
- anthracnose valid/test 的方向並不完全一致，small-n variance 仍大。

## 13. Limitations and Confounders

1. validation/test `insect_bite` 只有 1/3 個 GT，任何 Recall `0→1` 都會有巨大 variance；本次三者 fixed TP 都是 0。
2. valid anthracnose 只有 10 個 GT、test 22 個；一個 instance 可改變 10 或 4.55 個百分點的 empirical recall。
3. 三者都只有 seed 0。`deterministic=true` 不能替代 multi-seed mean/std，MPS operation 也不保證所有 kernel bitwise deterministic。
4. C/B 的 train entries 不同：565 vs 532；epochs 固定不等於 optimizer updates 完全固定。
5. C 的 intervention 同時改變 rare-object scale、crop context、train entry count、common incidental exposure；這些都是 targeted-crop strategy 的組成，但不能拆成單獨因果量。
6. v19/v20 checkpoint metadata 是 Git commit `7fdf102...`，v21 是 `6f0e980...`。Ultralytics/PyTorch/device 與 args 可核對一致，但 repository commit 不同。
7. v21 reproducibility 保存 initial `yolo26s-seg.pt` SHA-256；v19/v20 historical checkpoint 沒保存 initial-weight hash，無法從目前檔案反推歷史 bytes 100% 相同。
8. Experiment C validation 已證明 exact hash/可解析 Fruit ID 無 split overlap，但大量 Roboflow generic filenames 無法可靠還原 Fruit identity；fruit-group split 仍有來源限制。
9. official P/R、AP curves 與 fixed `conf=0.25` counts 是不同 operating definitions；報告已分開，不能混算。
10. prediction panels 只覆蓋一部分 images；完整 counts 來自逐 image audit，但 mask 過大/過小的語意仍需要人工 visual review。

## 14. Recommended Baseline

下一階段 incumbent：**v19 `v19_defect_epoch200_pw050_cp01`**。

保留 v20 作為「Recall/FP trade-off」參考，保留 v21 作為「targeted crop specificity/scale」參考。後續任何新實驗都應同時與 v19、v21 比較，不能只與上一個 run 比。

## 15. Recommended Next Experiment

建議做一個單變因的 **Experiment D：增加 targeted-crop context scale**：

```text
baseline: v19
same crop target instances and crop count as v21
same anthracnose/insect_bite exposure target
only change nominal context_scale: 4.0 -> 6.0
keep context variation, jitter, min/max crop rules, model, seed,
epochs, patience, batch, loss and all augmentations unchanged
```

理由：v21 已證明 targeted crop 能降低 FP 並改善部分 AP，但 test recall、insect_track 與 threshold stability 退步；這組 pattern 比較像 context/scale mismatch，而不是需要立刻再增加 exposure。先增加 context，可測試「保留更多 full-image appearance 是否能保住 common-class performance」，同時仍讓 rare defect 占據比原圖高的有效像素比例。

成功條件應預先固定：

1. test anthracnose TP/Recall/AP 不低於 v21，且至少不劣於 v20 的合理範圍；
2. test insect_track AP50 與 TP 回升到接近 v19；
3. background FP 不回到 v20 水準；
4. validation/test 最佳 F1 threshold 差距縮小；
5. 不把 insect_bite 的單一低-confidence AP 變化當成成功，仍報告 GT/TP/FN。

如果只改 context scale 後仍無法恢復 insect_track/test Recall，應停止 sampling/crop 路線，改做「單一 loss weighting 變因」或先補 rare-class 真實 validation/test diversity；不要把 crop、loss weighting 與新 augmentation 同時堆疊。

## Reproducibility Artifacts

- 分析程式：`tools/analyze_targeted_crop_experiment.py`
- 完整 machine-readable 結果：`training/analysis/v19-v20-v21-targeted-crop/analysis.json`
- aggregate figures：`training/analysis/v19-v20-v21-targeted-crop/figures/`
- 六組 Ultralytics plots：`training/analysis/v19-v20-v21-targeted-crop/ultralytics/{v19,v20,v21}_{val,test}/`

分析程式只讀 Dataset 與 runs。Ultralytics test evaluation 曾自動建立 `test/labels.cache`；完成後已移除，使 Dataset 回復分析前狀態。
