# v19／v21／v22 Targeted-crop Context Scale 分析

- 分析日期：2026-09-02
- 任務：YOLO26s-seg 百香果局部瑕疵 instance segmentation
- Baseline：`v19_defect_epoch200_pw050_cp01`
- Experiment C：`v21_C_targeted_crop`（nominal `context_scale=4.0`）
- Experiment D：`v22_D_context6`（nominal `context_scale=6.0`）
- 正式比較 checkpoint：各 run 的 `weights/best.pt`
- 工作邊界：只讀既有 Dataset、weights 與 training artifacts；未重新訓練、未修改 Dataset／權重／既有 run

## 1. Executive Summary

**Experiment D 最終判定：SUCCESS。Context hypothesis 分類：A. Hypothesis strongly supported。**

D 在 test set 上明確修復 C 的 Recall／common-class generalization regression，而且不是只改善單一指標：

- test Mask Recall：`v19 0.30096 → C 0.28042 → D 0.36618`；D 比 C `+0.08576`（`+30.58%`），比 v19 `+0.06521`（`+21.67%`）。
- test Mask mAP50-95：`0.15013 → 0.16777 → 0.17489`；D 比 C `+0.00713`（`+4.25%`），比 v19 `+0.02476`（`+16.49%`）。
- `insect_track` test Mask mAP50：`0.26022 → 0.19465 → 0.27582`；D 完全恢復並超越 baseline，而非只縮小 regression。
- `anthracnose` test Recall：`0.59091 → 0.63636 → 0.81818`；D 保留並擴大 rare-class gain。
- C 的 validation→test Mask Recall drop 是 `0.11652`，D 降到 `0.01520`；Mask mAP50-95 drop 也由 `0.02019` 降到 `0.01469`。D 的 validation/test 最佳 F1 threshold 差距由 C 的 `0.11812` 降至 `0.02402`。

改善有明確代價：D 相較 C 的 test Mask Precision 由 `0.36630` 降至 `0.35440`（`-0.01190`，`-3.25%`）；固定 `conf=0.25` 的 test mask FP 由 `74` 增至 `90`，純背景 FP predictions 由 `8` 增至 `11`。因此 D 是較好的 ranking/generalization model，但 operating point 比 C 更積極，不能忽略 FP 成本。

`insect_bite` 仍沒有成功：test 只有 3 個 GT，三模型在 `conf=0.25`、Mask IoU 0.50 下全部 `0 TP / 3 FN`。D 的 curve-derived AP 高於 v19、但低於 C；因沒有 fixed-threshold TP，證據不足以判斷 generalization。本文將其標記為 **insufficient evidence**，不把低 confidence AP 波動當成成功或失敗的主要依據。

## 2. Experiment Setup

### 2.1 Run、checkpoint 與 hash

| Model | Run | `best.pt` SHA-256 | `last.pt` SHA-256 |
| --- | --- | --- | --- |
| v19 | `training/runs/segment/v19_defect_epoch200_pw050_cp01` | `77bd843f8fdd6132167a21d422f1decc155b621e806ffab20f8ac3e0c529e7ce` | `7a4848db9f01764ddd21c0d2fb93bf6454755be417c806937fe3023a369fa65a` |
| C | `training/runs/segment/v21_C_targeted_crop` | `c4985236e120a118712767e40e0bf55063bdb2df885d74f7786d07ec39a40cf7` | `259dee14efd5cba566e24a4d206a2b68f1d1a9369b79de97e243de80b65268a4` |
| D | `training/runs/segment/v22_D_context6` | `0165feeafe59f70da11f3e94a52c2300c5cd2ffa36a1e5a3b2185fed1161eac4` | `c7245c657b19e33c38e1f823d6852ef1d6901b1e90059ed71ac95e5aab59a75e` |

所有正式數值只用 `best.pt`；`last.pt` 只確認存在與 hash，沒有參與跨模型比較。三個 checkpoint 都是四類 `SegmentationModel`：

```text
0 abrasion
1 anthracnose
2 insect_bite
3 insect_track
```

### 2.2 Training condition audit

| Condition | v19 | C | D |
| --- | --- | --- | --- |
| model | `yolo26s-seg.pt` | same | same |
| imgsz | 1024 | 1024 | 1024 |
| epochs | 200 | 200 | 200 |
| patience | 50 | 50 | 50 |
| batch | 4 | 4 | 4 |
| seed | 0 | 0 | 0 |
| deterministic | `true` | `true` | `true` |
| cls_pw | 0.5 | 0.5 | 0.5 |
| copy_paste | 0.1 | 0.1 | 0.1 |
| copy_paste_mode | `mixup` | `mixup` | `mixup` |
| optimizer | `auto` | `auto` | `auto` |
| train split | `data.yaml: ../train/images`；433 entries | `train_C_targeted_crop.txt`；565 entries | `train_D_context6.txt`；565 entries |
| valid split | 同一 `../valid/images`；93 images | same | same |
| test split | 同一 `../test/images`；92 images | same | same |
| initial checkpoint | `args.yaml` 記錄 `yolo26s-seg.pt`；歷史 bytes/hash 未保存 | SHA-256 `3da1d83e...c197837a4` | 同 C：`3da1d83e...c197837a4` |

三份 `args.yaml` 的完整 key-by-key diff 只有 `data`、`name`、`save_dir`。其他 augmentation、optimizer 與 schedule 參數一致。`optimizer=auto` 在當時 Ultralytics 8.4.122 條件下解析為 AdamW；checkpoint 已 strip optimizer，這是由同版本程式邏輯重建，不是從 optimizer state 直接讀取。

三個 run 的 checkpoint metadata 都記錄 Ultralytics `8.4.122`，但 repository commit 不同：v19 `7fdf102...`、C `6f0e980...`、D `f858198...`。可觀察 training args、Ultralytics/PyTorch 與資料契約一致；仍無法證明 v19 歷史 initial-weight bytes 與 C/D 完全相同。這是既有 baseline provenance 限制，不是 D 新增的差異。

## 3. Integrity Audit

### 3.1 C／D Dataset intervention

| Dataset item | C | D |
| --- | ---: | ---: |
| original train images | 433 | 433 |
| generated crops | 132 | 132 |
| total train entries | 565 | 565 |
| target crop records | 132 | 132 |
| unique target instances | 80 | 80 |
| anthracnose target records | 54 | 54 |
| insect_bite target records | 78 | 78 |
| nominal context_scale | 4.0 | 6.0 |
| context variation | ±0.5 | ±0.5；paired C realization |
| center jitter | 0.10 | 0.10；paired C realization |

D artifacts 與正式規格、Dataset 報告及題目給定數字一致：

- Dataset validation：`PASS`。
- C deterministic replay：target sequence、rectangle、geometry/pixels、labels 全部 `132/132 SAME`。
- C/D exact stochastic pairing：`131/132`；唯一例外是 sequence 92 的 duplicate rectangle，使用 record-local deterministic retry 一次，沒有使後續 RNG stream cascading。
- D 有 75 records 發生 boundary/clamp，C→D boundary/clamp status changed 23 records；retry 後 duplicate source rectangle 與 exact duplicate generated image 都是 0。
- source train／valid／test manifest unchanged；exact image SHA-256 overlap 皆 0；valid/test source hashes 與 C 完全相同。
- class mapping、`nc=4`、valid/test path 與 Roboflow metadata 均與 baseline 一致。

結論：**沒有發現足以破壞 C↔D controlled comparison 的非預期差異。** C↔D 的唯一主動 Dataset generation 變因是 nominal context scale 4.0→6.0。1/132 retry 與較大的 neighboring exposure 都是事前規格允許且已完整記錄的 downstream effect。

## 4. Dataset／Exposure Comparison

| Class | v19 total | C total | D total | C target | C neighboring | D target | D neighboring |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abrasion | 577 | 614 | 637 | 0 | 37 | 0 | 60 |
| anthracnose | 54 | 128 | 151 | 54 | 20 | 54 | 43 |
| insect_bite | 26 | 105 | 108 | 78 | 1 | 78 | 4 |
| insect_track | 386 | 430 | 445 | 0 | 44 | 0 | 59 |

`target` 是主動建立 crop 的 record count；`neighboring` 是 larger field-of-view 依 label-clipping policy 自然納入的合法 annotation。D 並非刻意把 anthracnose oversample 到 151，target contract 仍是 anthracnose 54 records、insect_bite 78 records。D-C total exposure 差異（abrasion +23、anthracnose +23、insect_bite +3、insect_track +15）全部可由 neighboring exposure 對帳。

結果與 exposure/context change 有合理關聯：較大的 crop 同時降低 target 的極端放大程度、提供更多原圖 surrounding appearance，也增加 common/neighboring instances；D 的 `insect_track` 恢復及 cross-split stability 符合「context/scale mismatch 減少」的預測。但這是單一 seed 的 correlation，無法把改善精確拆成 surrounding context、target pixel scale 或 neighboring exposure 的各別因果效果。

## 5. Evaluation Protocol

既有 C 分析已用同一設定評估 v19/C；本次交叉檢查其 machine-readable results 後，只補跑缺少的 D validation/test。三組正式比較條件如下：

| Setting | Value |
| --- | --- |
| checkpoint | 各 run `weights/best.pt` |
| data YAML | `training/packages/local_defects/data.yaml` |
| split | `val` 與 `test` |
| imgsz | 1024 |
| evaluation batch | 8 |
| device | `mps`（Apple M5 Pro） |
| official curve conf | `None`，Ultralytics curve floor 0.001 |
| NMS IoU | 0.70 |
| max_det | 300 |
| workers | 0 |
| Ultralytics | 8.4.122 |
| PyTorch | 2.13.0 |

另做 fixed-operating-point audit：`conf=0.25`、NMS IoU 0.70、Mask/Box TP matching IoU 0.50、class-aware one-to-one greedy matching。Ultralytics confusion matrix 使用其既有 box IoU 0.45。官方 P/R 是 curve-derived operating point；fixed TP/FP/FN 是另一口徑，不能直接互相除算。

D evaluation 寫入全新的 `training/analysis/v19-v21-v22-context-scale/`；Ultralytics 自動產生的 `test/labels.cache` 已在 evaluation 後移除，Dataset 回復原狀。

## 6. Training Dynamics

Ultralytics 8.4.122 segmentation fitness 為 `Box mAP50-95 + Mask mAP50-95`。因此 best fitness epoch 不一定等於單一 Mask metric 最佳 epoch。

| Item | v19 | C | D |
| --- | ---: | ---: | ---: |
| actual epochs | 200 | 174 | 147 |
| early stopping | no | yes | yes |
| best fitness epoch | 186 | 124 | 97 |
| best fitness | 0.42619 | 0.38144 | 0.38317 |
| best Mask mAP50-95 epoch | 186 | 129 | 97 |
| best Mask mAP50-95 in `results.csv` | 0.21046 | 0.19101 | 0.18956 |
| first epoch ≥95% own best Mask AP | 139 | 119 | 74 |
| first epoch ≥99% own best Mask AP | 186 | 129 | 97 |
| wall time to best fitness | 14,177.0 s | 12,134.4 s | 9,464.2 s |
| total wall time | 15,209.6 s | 17,010.3 s | 14,329.4 s |

D 比 C 更早到達 best/plateau，且最後 20 epochs 的 Mask Precision、mAP50 波動較小：Precision std `0.04961` vs `0.11013`，mAP50 std `0.02055` vs `0.02698`。Recall std 相近（D `0.03329`、C `0.03436`），strict mAP50-95 std 也相近（D `0.01379`、C `0.01325`）。因此可說 D 的 Precision/ranking curve 較穩，但不能宣稱所有 training signals 都更穩定。

### 6.1 Loss summary

| Loss | v19 start→final | C start→final | D start→final |
| --- | --- | --- | --- |
| train box | 1.73064→1.08579 | 1.73545→1.09597 | 1.72494→1.22557 |
| train seg | 3.13180→1.43520 | 3.30162→1.63466 | 3.29779→1.73223 |
| train cls | 27.74430→0.56177 | 14.99990→0.80583 | 16.61720→0.91694 |
| train l1 | 0.01267→0.01041 | 0.01355→0.00913 | 0.01276→0.00944 |
| train sem | 7.52905→0.36604 | 7.06976→0.66572 | 7.11677→0.64888 |

| Validation loss | v19 min epoch / final | C min epoch / final | D min epoch / final |
| --- | --- | --- | --- |
| box | 92 / 1.85171 | 96 / 1.79278 | 114 / 1.75781 |
| seg | 98 / 2.54597 | 99 / 2.48901 | 84 / 2.45806 |
| cls | 142 / 1.15929 | 146 / 1.57767 | 138 / 1.64642 |
| l1 | 179 / 0.01869 | 1 / 0.01794 | 111 / 0.01829 |

本版本 `results.csv` 沒有 DFL loss；它記錄 `l1_loss` 與 `sem_loss`，且 `val/sem_loss` 固定為 0，故不把它當成有辨識力的 validation signal。

三者皆有 train loss 持續下降、validation loss 在中後段回升的 gap。D 的 val seg loss 在 epoch 84 最低、best fitness 在 97，之後未再超越並於 147 early stop；final Mask mAP50-95 `0.15564` 比 best-epoch `0.18956` 低，顯示後期 improvement 消失。C 也有同樣 pattern。D 並未消除 overfitting，只是 best checkpoint 較早出現；跨 split test 結果顯示 D 的 best checkpoint generalize 得比 C 好。

raw train loss 不能直接拿來宣稱 C 優於 D：兩者 crop framing/instance composition 不同，且真正驗收仍是相同 valid/test 的 metrics。

## 7. Validation Metrics

括號為相對於差值左側模型的 relative change；原始值與 absolute difference 同時保留。

| Metric | v19 | C | D | C-v19 | D-v19 | D-C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Box P | 0.68911 | 0.67715 | 0.39385 | -0.01196 (-1.74%) | -0.29526 (-42.85%) | -0.28330 (-41.84%) |
| Box R | 0.31855 | 0.34109 | 0.39117 | +0.02255 (+7.08%) | +0.07262 (+22.80%) | +0.05007 (+14.68%) |
| Box mAP50 | 0.36664 | 0.36878 | 0.37011 | +0.00214 (+0.58%) | +0.00347 (+0.95%) | +0.00133 (+0.36%) |
| Box mAP50-95 | 0.21582 | 0.19364 | 0.19361 | -0.02218 (-10.28%) | -0.02221 (-10.29%) | -0.00003 (-0.02%) |
| Mask P | 0.70071 | 0.65537 | 0.41843 | -0.04535 (-6.47%) | -0.28229 (-40.29%) | -0.23694 (-36.15%) |
| Mask R | 0.32160 | 0.39694 | 0.38138 | +0.07533 (+23.42%) | +0.05977 (+18.59%) | -0.01556 (-3.92%) |
| Mask mAP50 | 0.36681 | 0.37417 | 0.37993 | +0.00736 (+2.01%) | +0.01312 (+3.58%) | +0.00576 (+1.54%) |
| Mask mAP50-95 | 0.20967 | 0.18795 | 0.18959 | -0.02172 (-10.36%) | -0.02008 (-9.58%) | +0.00163 (+0.87%) |

D 的 validation macro Precision 大降需要按 class 解讀：C 對 validation `insect_bite` 沒有 prediction，Ultralytics curve summary 給 P=1；D 有 3 個 fixed-threshold predictions 但 0 TP，該類 P=0，單一 class 就使四類 macro P 降約 0.25。排除這個只有 1 GT 的 class，D 其餘三類 Mask P macro 約 `0.558`，高於 C 約 `0.540`。不過 D validation 純背景 FP `2→8` 是真實退步，不能完全歸因於 macro convention。

## 8. Test Metrics

| Metric | v19 | C | D | C-v19 | D-v19 | D-C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Box P | 0.35264 | 0.37906 | 0.36041 | +0.02642 (+7.49%) | +0.00777 (+2.20%) | -0.01865 (-4.92%) |
| Box R | 0.30107 | 0.28824 | 0.36618 | -0.01283 (-4.26%) | +0.06511 (+21.63%) | +0.07794 (+27.04%) |
| Box mAP50 | 0.30968 | 0.32287 | 0.35094 | +0.01318 (+4.26%) | +0.04126 (+13.32%) | +0.02808 (+8.70%) |
| Box mAP50-95 | 0.16757 | 0.18790 | 0.19454 | +0.02034 (+12.14%) | +0.02697 (+16.10%) | +0.00664 (+3.53%) |
| Mask P | 0.35381 | 0.36630 | 0.35440 | +0.01250 (+3.53%) | +0.00060 (+0.17%) | -0.01190 (-3.25%) |
| Mask R | 0.30096 | 0.28042 | 0.36618 | -0.02055 (-6.83%) | +0.06521 (+21.67%) | +0.08576 (+30.58%) |
| Mask mAP50 | 0.30571 | 0.31710 | 0.34684 | +0.01139 (+3.72%) | +0.04113 (+13.45%) | +0.02974 (+9.38%) |
| Mask mAP50-95 | 0.15013 | 0.16777 | 0.17489 | +0.01763 (+11.75%) | +0.02476 (+16.49%) | +0.00713 (+4.25%) |

D 是三者 test Mask Recall、mAP50、mAP50-95 最高；Mask Precision 與 v19 幾乎相同，略低於 C。Box 與 Mask AP 方向一致，支持改善不只來自單一 mask threshold artifact。

## 9. Per-class Test Analysis

Ultralytics 提供 per-class P、R、mAP50、mAP50-95，以下全部為 Mask metrics。

| Class | Metric | v19 | C | D | C-v19 | D-v19 | D-C |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abrasion | P | 0.41085 | 0.49503 | 0.45900 | +0.08417 (+20.49%) | +0.04815 (+11.72%) | -0.03603 (-7.28%) |
| abrasion | R | 0.36295 | 0.26531 | 0.32653 | -0.09764 (-26.90%) | -0.03642 (-10.03%) | +0.06122 (+23.08%) |
| abrasion | mAP50 | 0.27830 | 0.31286 | 0.32687 | +0.03455 (+12.42%) | +0.04857 (+17.45%) | +0.01401 (+4.48%) |
| abrasion | mAP50-95 | 0.10694 | 0.13987 | 0.15691 | +0.03293 (+30.79%) | +0.04997 (+46.72%) | +0.01704 (+12.18%) |
| anthracnose | P | 0.65724 | 0.61356 | 0.62148 | -0.04369 (-6.65%) | -0.03576 (-5.44%) | +0.00792 (+1.29%) |
| anthracnose | R | 0.59091 | 0.63636 | 0.81818 | +0.04545 (+7.69%) | +0.22727 (+38.46%) | +0.18182 (+28.57%) |
| anthracnose | mAP50 | 0.67552 | 0.67704 | 0.76946 | +0.00152 (+0.23%) | +0.09394 (+13.91%) | +0.09241 (+13.65%) |
| anthracnose | mAP50-95 | 0.38899 | 0.40042 | 0.41844 | +0.01143 (+2.94%) | +0.02945 (+7.57%) | +0.01801 (+4.50%) |
| insect_bite | P | 0.00000 | 0.00000 | 0.00000 | +0.00000 (n/a) | +0.00000 (n/a) | +0.00000 (n/a) |
| insect_bite | R | 0.00000 | 0.00000 | 0.00000 | +0.00000 (n/a) | +0.00000 (n/a) | +0.00000 (n/a) |
| insect_bite | mAP50 | 0.00882 | 0.08386 | 0.01523 | +0.07504 (+851.22%) | +0.00641 (+72.73%) | -0.06863 (-81.84%) |
| insect_bite | mAP50-95 | 0.00529 | 0.04790 | 0.01370 | +0.04261 (+805.57%) | +0.00842 (+159.09%) | -0.03420 (-71.39%) |
| insect_track | P | 0.34713 | 0.35663 | 0.33714 | +0.00950 (+2.74%) | -0.00999 (-2.88%) | -0.01950 (-5.47%) |
| insect_track | R | 0.25000 | 0.22000 | 0.32000 | -0.03000 (-12.00%) | +0.07000 (+28.00%) | +0.10000 (+45.45%) |
| insect_track | mAP50 | 0.26022 | 0.19465 | 0.27582 | -0.06557 (-25.20%) | +0.01560 (+5.99%) | +0.08117 (+41.70%) |
| insect_track | mAP50-95 | 0.09931 | 0.08287 | 0.11053 | -0.01644 (-16.55%) | +0.01122 (+11.30%) | +0.02766 (+33.37%) |

### 9.1 anthracnose

C 相較 v19：Recall `+0.04545`、mAP50 `+0.00152`、mAP50-95 `+0.01143`，但 Precision `-0.04369`。C 是有限 gain。

D 相較 C：P/R/mAP50/mAP50-95 全部改善，其中 Recall `+0.18182`、mAP50 `+0.09241` 最明顯。固定 `conf=0.25` TP/FN 由 C `15/7` 改善到 D `18/4`；D 也優於 v19 `13/9`。這支持 larger context 幫助 anthracnose test generalization，但 test 僅 22 GT、單一 seed，證據強度仍受 small-n 限制。

### 9.2 insect_bite

三模型 official P/R 都是 0，fixed audit 都是 `0 TP / 3 FN`。C 的 AP50 `0.08386` 在 D 降到 `0.01523`，但那是低 confidence curve/ranking 訊號，沒有轉化為 `conf=0.25` 的正確 detection。D 是否保留 rare-class gain：**insufficient evidence**。目前只能說 larger context 沒有解決 insect_bite。

### 9.3 insect_track

C 相較 v19 的 test regression：Recall `-0.03000`（-12.00%）、mAP50 `-0.06557`（-25.20%）、mAP50-95 `-0.01644`（-16.55%），fixed TP `26→22`。

D 將 Recall 提高到 `0.32000`、mAP50 提高到 `0.27582`、mAP50-95 提高到 `0.11053`，fixed TP `22→34`。三項都超越 v19，故判定為 **完全恢復並超越 baseline**，不是部分恢復。代價是 fixed FP `39→51`，Precision 比 C 低 `0.01950`、也比 v19 低 `0.00999`。

### 9.4 abrasion

D 相較 C：Recall 回升 `+0.06122`，mAP50-95 `+0.01704`，但 Precision `-0.03603`。相較 v19，D 的 AP50/AP50-95 與 Precision都更高，Recall 仍低 `0.03642`。因此 context 增加沒有對 common `abrasion` 造成額外 AP 傷害，反而修復部分 Recall；但尚未完全回到 baseline Recall。

## 10. Precision／Recall Trade-off

D vs C 的 test Mask 變化是：

```text
Precision 0.36630 -> 0.35440  (-0.01190, -3.25%)
Recall    0.28042 -> 0.36618  (+0.08576, +30.58%)
mAP50     0.31710 -> 0.34684  (+0.02974, +9.38%)
mAP50-95  0.16777 -> 0.17489  (+0.00713, +4.25%)
```

這個 trade-off 整體值得：Recall 的 absolute gain 約為 Precision loss 的 7.2 倍，且較嚴格的 mAP50-95 同步上升，不是只把 threshold 調低換 Recall。四類中主要 Recall gain 來自 anthracnose `+0.18182`、insect_track `+0.10000`、abrasion `+0.06122`；不是單一 class 拉高 overall。

最佳 Mask F1/threshold：

| Split | v19 | C | D |
| --- | --- | --- | --- |
| validation | 0.361 @ 0.318 | 0.408 @ 0.206 | 0.402 @ 0.187 |
| test | 0.335 @ 0.196 | 0.345 @ 0.088 | 0.364 @ 0.211 |
| absolute threshold gap | 0.122 | 0.118 | 0.024 |

D 的 test best F1 最高，validation/test optimal threshold 也最一致；這強烈支持 C 的 calibration/distribution sensitivity 已改善。正式比較仍使用相同 evaluation 設定，沒有為任一模型個別調 threshold。

## 11. Generalization Analysis

下表為 `validation - test`；正值代表 validation 較高，負值代表 test 較高。

| Metric | v19 | C | D |
| --- | ---: | ---: | ---: |
| Box P | +0.33647 | +0.29809 | +0.03344 |
| Box R | +0.01748 | +0.05285 | +0.02499 |
| Box mAP50 | +0.05696 | +0.04591 | +0.01917 |
| Box mAP50-95 | +0.04825 | +0.00573 | -0.00094 |
| Mask P | +0.34691 | +0.28906 | +0.06402 |
| Mask R | +0.02064 | +0.11652 | +0.01520 |
| Mask mAP50 | +0.06110 | +0.05707 | +0.03309 |
| Mask mAP50-95 | +0.05954 | +0.02019 | +0.01469 |

D 的 Recall 與 AP gaps 最小或接近最小；尤其 C 的 Mask Recall gap `0.11652` 在 D 降為 `0.01520`，同時 D test Recall 真正提高，不是靠降低 validation 才製造較小 gap。D 的 Precision gap 很小部分來自 validation macro Precision 本身下降，故不能單看 Precision gap 宣稱 generalization 改善；更有力的證據是 test Recall/AP 上升、insect_track recovery 及 threshold gap 縮小三者方向一致。

## 12. Error Analysis

### 12.1 Fixed `conf=0.25` Mask TP/FP/FN（test）

| Class | v19 TP/FP/FN | C TP/FP/FN | D TP/FP/FN |
| --- | --- | --- | --- |
| abrasion | 33 / 44 / 65 | 26 / 23 / 72 | 31 / 26 / 67 |
| anthracnose | 13 / 8 / 9 | 15 / 10 / 7 | 18 / 11 / 4 |
| insect_bite | 0 / 0 / 3 | 0 / 2 / 3 | 0 / 2 / 3 |
| insect_track | 26 / 28 / 74 | 22 / 39 / 78 | 34 / 51 / 66 |
| total | 72 / 80 / 151 | 63 / 74 / 160 | 83 / 90 / 140 |

D 比 C 多 20 TP、少 20 FN，但多 16 FP。FN recovery 分散於 abrasion、anthracnose、insect_track；FP increase 最大來自 insect_track。`anthracnose` 不再是「大量幾乎全漏」，D fixed FN 為 4/22；`insect_bite` 仍全漏；`insect_track` FN 由 C 78 降到 D 66，但仍是絕對數最大的錯誤來源之一。

### 12.2 Background and class confusion

| Test error item | v19 | C | D |
| --- | ---: | ---: | ---: |
| pure-background images | 35 | 35 | 35 |
| background images with FP | 4 | 6 | 10 |
| FP predictions on pure background | 7 | 8 | 11 |
| all-image unmatched box FP | 74 | 59 | 88 |
| all-image box background FN | 145 | 154 | 131 |

D 的 larger context 確實減少 all-image background FN：比 C 少 23、比 v19 少 14；但同時產生更多 unmatched FP。D 沒有保留 C 的 FP suppression，這是 D 最主要 downside。

Ultralytics confusion matrix 是 box-level，不是 segmentation-mask confusion。D test 的主要跨 defect class confusion 是 abrasion/insect_track：7 個 true abrasion 被預測為 insect_track，3 個 true insect_track 被預測為 abrasion；另有少量 anthracnose 被判為 insect_bite/insect_track。主要錯誤仍是 defect↔background，而不是四個 defect class 互相混淆。

這些矩陣無法判斷 mask boundary 的臨床／農業語意品質，也不能把 unmatched prediction 全部解讀為 hallucination；可能包含 localization IoU 不足、duplicate/fragment mask 或 annotation boundary 差異。結論限於相同 protocol 下的相對比較。

## 13. Qualitative Prediction Analysis

![Representative test predictions](../../training/analysis/v19-v21-v22-context-scale/qualitative_test_examples.jpg)

上圖使用相同 `conf=0.25`、IoU 0.70，四欄依序為 ground truth、v19、C、D。五列分別代表：

1. v19 correct、C miss、D recover 的 insect_track。
2. C miss、D recover 的 anthracnose。
3. C correct、D miss 的 insect_track 反例。
4. 三者都 miss 的 anthracnose。
5. 三者都 miss 的 insect_bite；其中模型常以其他 defect class 回應附近 appearance。

完整 fixed audit 中，test 有 25 個 GT 是 C miss、D correct：insect_track 16、abrasion 6、anthracnose 3；其中 12 個同時符合 v19 correct、C miss、D recover。反向 C correct、D miss 只有 5 個：insect_track 4、abrasion 1。另有 127 個 GT 三者都 miss，主要是 abrasion 62、insect_track 58，也包括全部 3 個 insect_bite。代表案例支持 aggregate direction，但屬選例，不取代完整 test metrics。

## 14. Experiment D Hypothesis Evaluation

原假設：C crop 太緊，可能使模型依賴局部放大 appearance、缺少 full-image surrounding context，造成 test Recall/common-class generalization 下降；D 把 nominal context scale 4.0→6.0，希望保留 rare gain 並恢復 generalization。

### A. Hypothesis strongly supported

判定依據：

1. D 的 test Mask Recall 比 C +0.08576，且超越 v19 +0.06521。
2. D 的 insect_track Recall/AP/TP 全部從 C regression 恢復，並超越 v19。
3. D 的 anthracnose Recall/mAP50/mAP50-95 全部高於 C 與 v19，保留 rare-class benefit。
4. test Mask mAP50 與 mAP50-95 同步提高，排除「只降 threshold 換 Recall」的簡單解釋。
5. validation→test Recall/AP gap 與 best-F1 threshold gap 明顯縮小，直接符合 generalization hypothesis 的預測。

限制：D 增加 FP，abrasion Recall 尚未完全回到 baseline，insect_bite 無可靠 TP；且只有 seed 0。故「context 6 在這次 controlled run 中修復 C」證據強，但「6.0 普遍優於 4.0」仍需要 seed replication 才能估計 optimization variance。

## 15. Final Verdict

### 15.1 相對 Experiment C

**context_scale 6 優於 context_scale 4。** D 用小幅 Precision/FP 代價換得大幅 Recall，且 Box/Mask AP、anthracnose、insect_track、cross-split stability 都同步改善。D-C 最大改善是 anthracnose Recall `+0.18182`；若以 C 的核心 regression 看，insect_track mAP50 `+0.08117` 且超越 baseline 最具實驗意義。最大退步是 insect_bite curve AP50 `-0.06863`，但因三者都是 0 TP，屬 insufficient evidence；可靠 downside 是 FP 增加與 abrasion/insect_track Precision 小幅下降。

### 15.2 相對 baseline v19

D 不只是「比 C 好但仍低於 v19」。D 的 overall test Mask Recall、mAP50、mAP50-95 均高於 v19，Precision 幾乎相同；anthracnose 與 insect_track 的 Recall/AP 也超越 v19。仍低於 v19 的主要項目是 abrasion Recall `-0.03642`、anthracnose Precision `-0.03576`、insect_track Precision `-0.00999`，以及較高 FP。

### 15.3 Overall

**SUCCESS**

最重要證據：

1. test Mask Recall：D `0.36618`，C `0.28042`，v19 `0.30096`。
2. test Mask mAP50-95：D `0.17489`，C `0.16777`，v19 `0.15013`。
3. insect_track mAP50：D `0.27582`，完全恢復 C `0.19465` 的 regression 並超越 v19 `0.26022`。
4. anthracnose Recall：D `0.81818`，高於 C `0.63636` 與 v19 `0.59091`。
5. C→D test optimal F1 threshold gap `0.118→0.024`，但 D fixed FP `74→90`，顯示成功伴隨 specificity cost。

## 16. Recommended Next Experiment

### 建議立即做

**固定 D Dataset、context_scale=6.0 與全部 training/evaluation 設定，只新增一個 training seed replication（先做 seed=1）。**

理由：D 已同時優於 C 與 v19，現在最重要問題不是 8.0/10.0 是否更高，而是單一 seed 的改善能否重現。下一個 run 只改 training seed，仍用同一 D Dataset、同一 best.pt evaluation protocol；完成後比較 seed 0/1 的 overall/per-class mean、range，以及 FP/Recall trade-off。若 seed=1 方向一致，再視成本補 seed=2；不要一開始把 seed、loss、sampling 一起改。

### 值得之後做

- 在 D 多 seed 確認後，固定模型與 Dataset，做一個共同 operating-threshold／calibration study，目標是在保留大部分 Recall gain 下減少 D 的 FP；三個模型必須使用同一選定規則，不能各自在 test 調參。
- 擴充獨立 valid/test 的真實 `insect_bite` 樣本 diversity。現有 test 3 GT 無法可靠驗收 rare-class intervention；這是 evaluation evidence 問題，不是要求立刻修改目前固定 Dataset。
- 若 D seed replication 穩定但 insect_bite 仍 0 TP，再開一個單變因 loss-weighting experiment；不得同時改 crop/sampling/augmentation。

### 目前不建議做

- 不繼續無限制探索 context_scale 8.0、10.0；6.0 已達成 C 修復目的，先確認 variance 與 FP operating point。
- 不同時堆疊更多 rare exposure、class loss、copy-paste 或新 augmentation；會失去此次乾淨的因果解讀。
- 不以 C 的 insect_bite 高百分比 AP change 作為選模依據；0 TP、3 GT 下百分比極不穩定。

若後續 seed replication 無法重現 D，本次差異的首要替代解釋會是 optimization variance；目前資料較不支持「targeted-crop 方法本身失敗」或「context 6 本身失敗」，因為 D 在主要預註冊方向上跨多個 metrics/classes 一致改善。evaluation noise 對 insect_bite 很強、對 anthracnose中等，對 100 GT 的 insect_track 相對較弱。

## Reproducibility Artifacts

- v19/C 既有 machine-readable evaluation：`training/analysis/v19-v20-v21-targeted-crop/analysis.json`
- D 本次 machine-readable evaluation：`training/analysis/v19-v21-v22-context-scale/v22_evaluation.json`
- D Ultralytics val/test plots：`training/analysis/v19-v21-v22-context-scale/ultralytics/v22_{val,test}/`
- qualitative panel：`training/analysis/v19-v21-v22-context-scale/qualitative_test_examples.jpg`
- C Dataset report：`docs/experiments/v21-experiment-c-targeted-crop-dataset-2026-08-31.md`
- D Dataset report：`docs/experiments/v22-experiment-d-context6-dataset-2026-09-01.md`
