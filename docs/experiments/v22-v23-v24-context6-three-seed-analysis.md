# Experiment D：context_scale=6.0 Three-seed Replication / Stability Analysis

日期：2026-09-04

## 1. Executive Summary

Experiment D 的三個 training seeds（D0/D1/D2）通過完整性稽核，未發現 seed 以外的效能
confounder。116 個 `args.yaml` keys 中，只有預期的 `seed` 與允許不同的 `name`、
`save_dir` 有差異。三個 run 都指向同一份 Experiment D Dataset、同一初始
`yolo26s-seg.pt`，且 checkpoint 內嵌的 Ultralytics 版本均為 `8.4.122`。

正式 test Mask 結果如下：

| Seed | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| D0 / seed 0 | 0.35440 | 0.36618 | 0.34684 | 0.17489 |
| D1 / seed 1 | 0.40103 | 0.32426 | 0.34157 | 0.16308 |
| D2 / seed 2 | 0.35704 | 0.39422 | 0.34718 | 0.17911 |
| **Mean ± sample SD** | **0.37082 ± 0.02619** | **0.36155 ± 0.03521** | **0.34520 ± 0.00315** | **0.17236 ± 0.00831** |

核心結論：

- 三個 D seeds 的 overall Recall 與 Mask mAP50 全部高於 C 與 v19；三-seed mean 也高於
  兩者。mAP50 range 只有 `0.00561`，是最穩定的 ranking improvement。
- overall Mask mAP50-95 的 mean 高於 C/v19，三個 seeds 全部高於 v19，但 D1 略低於 C；
  因此 strict segmentation quality 相對 v19 穩定改善，相對 C 則有 seed sensitivity。
- insect_track 的 C regression 在三個 seeds 都由 Recall/mAP50 修復；mAP50 mean
  `0.26244` 略高於 v19 `0.26022`。strict AP 與固定 threshold TP 的修復較不完整。
- anthracnose Recall 與 mAP50 在三個 seeds 全部高於 C/v19；strict AP 是方向一致但不全數
  高於 C。
- abrasion Recall range `0.13265`、sample SD `0.07143`，是主要 class-level 不穩定項目；
  mAP50/mAP50-95 的方向比 Recall 穩定。
- insect_bite test 只有 3 GT，所有 seeds 在 `conf=0.25` 仍為 0 TP / 3 FN：
  **insufficient evidence**。
- 三個 D seeds 的 validation→test Recall absolute gap 全部遠小於 C；這是 larger context
  修復 generalization regression 最強的 replication evidence。
- fixed `conf=0.25` 的 D TP/FP/FN ranges 分別為 38/62/38，而 mAP50 SD 僅 `0.00315`。
  ranking performance 穩定、fixed-threshold operating behavior 不穩定的現象成立。

正式判定：

- **Experiment D verdict：SUCCESS WITH SEED SENSITIVITY**
- **Context hypothesis evidence：STRONGLY SUPPORTED**
- **Seed replication：STOP**
- **Experiment D 應成為後續實驗的 experimental baseline：YES**
- **下一步：validation-derived confidence-threshold / calibration study**

這些判定不表示可以依 test set 從 D0/D1/D2 挑最高分 checkpoint 作為 deployment model；
experimental baseline 與 deployment checkpoint selection 必須分開。

## 2. Objective

本分析不是找出「最佳 seed」，而是檢驗以下 hypothesis：Experiment C 的
`context_scale=4.0` crop 可能過緊，造成 test Recall/common-class generalization regression；
Experiment D 將 context scale 增至 `6.0`，應保留 rare-class benefit 並修復 generalization。

比較對象：

| Label | Run | Role |
| --- | --- | --- |
| v19 | `v19_defect_epoch200_pw050_cp01` | baseline |
| C | `v21_C_targeted_crop` | Experiment C, context_scale=4.0 |
| D0 | `v22_D_context6` | Experiment D, seed=0 |
| D1 | `v23_D_context6_seed1` | Experiment D, seed=1 |
| D2 | `v24_D_context6_seed2` | Experiment D, seed=2 |

三 seed 只提供 descriptive statistics；sample SD 使用 `ddof=1`。本報告不做正式顯著性
檢定，也不宣稱三點足以可靠估計完整 variance distribution。

## 3. Three-seed Integrity Audit

### 3.1 Training arguments

逐 key 比較三份 `args.yaml`，共有 116 個 union keys：

| Key | D0 | D1 | D2 | Result |
| --- | --- | --- | --- | --- |
| model | `yolo26s-seg.pt` | same | same | PASS |
| data | `training/packages/local_defects/data_exp_D.yaml` | same | same | PASS |
| imgsz | 1024 | 1024 | 1024 | PASS |
| epochs | 200 | 200 | 200 | PASS |
| patience | 50 | 50 | 50 | PASS |
| batch | 4 | 4 | 4 | PASS |
| device | `mps` | `mps` | `mps` | PASS |
| deterministic | true | true | true | PASS |
| cls_pw | 0.5 | 0.5 | 0.5 | PASS |
| copy_paste | 0.1 | 0.1 | 0.1 | PASS |
| copy_paste_mode | `mixup` | `mixup` | `mixup` | PASS |
| optimizer | `auto` | `auto` | `auto` | PASS |
| seed | 0 | 1 | 2 | intended variable |
| name | D0 name | D1 name | D2 name | allowed metadata |
| save_dir | D0 directory | D1 directory | D2 directory | allowed metadata |
| all other 101 settings | same | same | same | PASS |

「all other」包含 learning rate、momentum、weight decay、warmup、AMP、pretrained、mosaic、
mixup、cutmix、HSV、translate/scale/shear/perspective、flip、mask settings、loss weights、
close_mosaic、fraction、rect、multi_scale、validation/NMS 及其餘 Ultralytics arguments。
沒有 unexpected differences。

### 3.2 Dataset, split, and checkpoint identity

| Integrity item | Verified value | Result |
| --- | --- | --- |
| D Dataset SHA-256 | `8d17834c7e768d2b492cdbfcb3c5479756da4b5fe72bc58ef3adcc26a996ba0b` | PASS |
| source Dataset SHA-256 | `be2b7701c799abe1dad31fba2b9deb35c4a8770b118b86e6866d5bd45f6ae42e` | PASS |
| source train split | 433 images; `93cc548941a474d83bdbd0f03f50bb2bb4714a2d3a44d5ea6fd70d90df5e5d56` | PASS |
| generated crops | 132 | PASS |
| total train entries | 565 | PASS |
| train-list SHA-256 | `8d685ded3900d3ff3c9823b21188a5f2033feb0f6bcd015573481e00d9c8c22f` | PASS |
| valid split | 93 images; `639ac68782195fab597ca40e8b9f1a4722f767d1892b58883077a155ed8e70bb` | PASS |
| test split | 92 images; `be82ea7043df3459c2961dd3fdd8927e7818144743eaa158f15e9a87dde7bb42` | PASS |
| crop manifest SHA-256 | `07b57c720725bcd590cab5cd8605f8b487fcfcfd98617c08ce4b96cb3be2f419` | PASS |
| source-splits manifest SHA-256 | `62bf967530434e7bf4bec681874ac53f294df0978989032dd920a296f42069ea` | PASS |
| initial checkpoint path | `yolo26s-seg.pt` in all run args/checkpoints | PASS |
| initial checkpoint SHA-256 | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` | PASS |
| Ultralytics | `8.4.122` embedded in all three `best.pt` | PASS |
| device | `mps` in all three run args/checkpoints | PASS |

Dataset/list/reproducibility artifacts were created before all three training runs. D0/D1/D2 point to
the same `data_exp_D.yaml`; valid/test in that YAML are the unchanged canonical splits. Experiment D's
paired-generation validation status is `PASS`, including deterministic Experiment C replay and split
safety.

Dataset reproducibility records Python `3.12.13`、PyTorch `2.13.0`、Ultralytics `8.4.122`、
macOS arm64. Each checkpoint independently proves the same Ultralytics version, but run directories do
not contain a separate full Python/PyTorch environment snapshot for each training invocation. Therefore,
exact per-run PyTorch build identity cannot be independently reconstructed. This is a provenance
limitation, not an observed confounder. **No seed-external performance confounder was found.**

## 4. Evaluation Protocol

All formal comparisons use `weights/best.pt`; `last.pt` is excluded.

| Setting | Value |
| --- | --- |
| data | `training/packages/local_defects/data.yaml` |
| splits | `val`, `test` |
| imgsz | 1024 |
| batch | 8 |
| device | `mps` |
| workers | 0 |
| official curve conf | `None` (Ultralytics floor 0.001) |
| NMS IoU | 0.70 |
| max_det | 300 |
| fixed audit conf | 0.25 |
| Mask/Box TP matching IoU | 0.50 |
| matching | class-aware one-to-one greedy |

v19/C/D0/D1 use their existing machine-readable evaluations. D2 was evaluated with the identical
protocol into a new directory; no per-seed threshold tuning occurred. D2 `best.pt` SHA-256 is
`f930e6ec4a7fcd1ada75780351844a4abe7983c0dd763e4aacefc8b11e0cdf81`.

## 5. Overall Metrics

### 5.1 Test Mask metrics

| Metric | v19 | C | D0 | D1 | D2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precision | 0.35381 | 0.36630 | 0.35440 | 0.40103 | 0.35704 |
| Recall | 0.30096 | 0.28042 | 0.36618 | 0.32426 | 0.39422 |
| mAP50 | 0.30571 | 0.31710 | 0.34684 | 0.34157 | 0.34718 |
| mAP50-95 | 0.15013 | 0.16777 | 0.17489 | 0.16308 | 0.17911 |

### 5.2 Test Box metrics (supplementary)

| Metric | v19 | C | D0 | D1 | D2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precision | 0.35264 | 0.37906 | 0.36041 | 0.38518 | 0.33152 |
| Recall | 0.30107 | 0.28824 | 0.36618 | 0.30750 | 0.38156 |
| mAP50 | 0.30968 | 0.32287 | 0.35094 | 0.33193 | 0.33966 |
| mAP50-95 | 0.16757 | 0.18790 | 0.19454 | 0.18461 | 0.18705 |

Mask 是主要判據；Box 結果作為補充，沒有改變結論。

## 6. Three-seed Descriptive Statistics

| Metric | D0 | D1 | D2 | Mean | SD | Min | Max | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask Precision | 0.35440 | 0.40103 | 0.35704 | 0.37082 | 0.02619 | 0.35440 | 0.40103 | 0.04663 |
| Mask Recall | 0.36618 | 0.32426 | 0.39422 | 0.36155 | 0.03521 | 0.32426 | 0.39422 | 0.06997 |
| Mask mAP50 | 0.34684 | 0.34157 | 0.34718 | 0.34520 | 0.00315 | 0.34157 | 0.34718 | 0.00561 |
| Mask mAP50-95 | 0.17489 | 0.16308 | 0.17911 | 0.17236 | 0.00831 | 0.16308 | 0.17911 | 0.01603 |

Interpretation by core metric:

- **Recall:** D2 `0.39422` > C `0.28042` and v19 `0.30096`. All 3/3 D seeds exceed C and
  v19. Mean exceeds C by `0.08113` and v19 by `0.06059`.
- **mAP50:** all 3/3 seeds exceed C/v19. Mean exceeds C by `0.02810` and v19 by `0.03948`;
  range `0.00561` is small. This is a stable ranking improvement.
- **mAP50-95:** D2 is the highest D seed at `0.17911`. Mean exceeds C by `0.00459` and v19 by
  `0.02223`; all seeds exceed v19, but D1 is `0.00469` below C. Thus strict segmentation quality is
  robustly above v19 but only directionally/mean-level above C, with visible seed sensitivity.
- **Precision:** mean is only slightly above C and two seeds are below C. Precision is not the main
  evidence for the context hypothesis and should not be described as uniformly improved.

## 7. Per-class Stability

All entries below are official test Mask curve metrics.

| Class | Metric | v19 | C | D0 | D1 | D2 | D mean | D SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abrasion | Precision | 0.41085 | 0.49503 | 0.45900 | 0.50932 | 0.48759 | 0.48530 | 0.02524 |
| abrasion | Recall | 0.36295 | 0.26531 | 0.32653 | 0.21429 | 0.34694 | 0.29592 | 0.07143 |
| abrasion | mAP50 | 0.27830 | 0.31286 | 0.32687 | 0.31468 | 0.34840 | 0.32998 | 0.01707 |
| abrasion | mAP50-95 | 0.10694 | 0.13987 | 0.15691 | 0.12851 | 0.15726 | 0.14756 | 0.01650 |
| anthracnose | Precision | 0.65724 | 0.61356 | 0.62148 | 0.75153 | 0.53993 | 0.63765 | 0.10672 |
| anthracnose | Recall | 0.59091 | 0.63636 | 0.81818 | 0.72727 | 0.90909 | 0.81818 | 0.09091 |
| anthracnose | mAP50 | 0.67552 | 0.67704 | 0.76946 | 0.70748 | 0.75459 | 0.74384 | 0.03236 |
| anthracnose | mAP50-95 | 0.38899 | 0.40042 | 0.41844 | 0.39309 | 0.45573 | 0.42242 | 0.03151 |
| insect_bite | Precision | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| insect_bite | Recall | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| insect_bite | mAP50 | 0.00882 | 0.08386 | 0.01523 | 0.08867 | 0.02971 | 0.04453 | 0.03890 |
| insect_bite | mAP50-95 | 0.00529 | 0.04790 | 0.01370 | 0.04589 | 0.01976 | 0.02645 | 0.01710 |
| insect_track | Precision | 0.34713 | 0.35663 | 0.33714 | 0.34327 | 0.40064 | 0.36035 | 0.03502 |
| insect_track | Recall | 0.25000 | 0.22000 | 0.32000 | 0.35546 | 0.32086 | 0.33211 | 0.02023 |
| insect_track | mAP50 | 0.26022 | 0.19465 | 0.27582 | 0.25546 | 0.25604 | 0.26244 | 0.01159 |
| insect_track | mAP50-95 | 0.09931 | 0.08287 | 0.11053 | 0.08483 | 0.08367 | 0.09301 | 0.01518 |

## 8. Anthracnose Analysis

| Item | v19 | C | D0 | D1 | D2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mask Precision | 0.65724 | 0.61356 | 0.62148 | 0.75153 | 0.53993 |
| Mask Recall | 0.59091 | 0.63636 | 0.81818 | 0.72727 | 0.90909 |
| Mask mAP50 | 0.67552 | 0.67704 | 0.76946 | 0.70748 | 0.75459 |
| Mask mAP50-95 | 0.38899 | 0.40042 | 0.41844 | 0.39309 | 0.45573 |
| fixed TP / FP / FN | 13 / 8 / 9 | 15 / 10 / 7 | 18 / 11 / 4 | 14 / 4 / 8 | 19 / 12 / 3 |

D2 clearly reproduces the high-Recall/high-mAP50 direction. All three D seeds exceed C and v19 in
Recall and mAP50. Fixed TP is more threshold-sensitive: D1 has one fewer TP than C, while D0/D2 exceed
C. Precision also trades against Recall, especially in D2; low Precision cannot erase the ranking and
Recall result, but it matters for deployment calibration.

- **Recall / detection verdict: ROBUST IMPROVEMENT.** Three of three seeds reproduce it.
- **Strict segmentation AP verdict: DIRECTIONALLY ROBUST.** Mean is above C/v19 and D0/D2 are above
  C; D1 is slightly below C but still slightly above v19. It is not a uniform strict-AP win over C.

The test set contains only 22 anthracnose GT, so a single instance changes simple recall materially.

## 9. Insect_track Recovery Analysis

| Item | v19 | C | D0 | D1 | D2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mask Recall | 0.25000 | 0.22000 | 0.32000 | 0.35546 | 0.32086 |
| Mask mAP50 | 0.26022 | 0.19465 | 0.27582 | 0.25546 | 0.25604 |
| Mask mAP50-95 | 0.09931 | 0.08287 | 0.11053 | 0.08483 | 0.08367 |
| fixed TP / FP / FN | 26 / 28 / 74 | 22 / 39 / 78 | 34 / 51 / 66 | 20 / 19 / 80 | 23 / 26 / 77 |

All three seeds repair the large C mAP50 regression: the lowest D mAP50 is `0.25546`, still
`0.06081` above C. D mean `0.26244` is slightly above v19 `0.26022`; D1/D2 individually are only
about `0.004–0.005` below v19. Recall recovery is even clearer: all seeds exceed C and v19, with a
small range `0.03546` and SD `0.02023`.

Strict mAP50-95 recovery is weaker. D0 exceeds v19, whereas D1/D2 sit only slightly above C and below
v19; D mean `0.09301` is above C but below v19. Fixed TP is also seed-sensitive: 34/20/23 for
D0/D1/D2. D1 is below C at `conf=0.25`, and D2 only slightly exceeds C while remaining below v19.

**Verdict: ROBUST RECOVERY of the C regression in Recall/mAP50 ranking performance; partial recovery
for strict AP and seed-sensitive fixed-threshold TP.** Recall recovery is substantially more stable than
strict AP recovery. This directly supports the larger-context hypothesis without claiming complete
strict-mask/deployment recovery.

## 10. Abrasion Seed Sensitivity

D2 Recall `0.34694` is much closer to D0 `0.32653` (difference `0.02041`) than D1 `0.21429`
(difference `0.13265`).

| Metric | D mean | SD | Min | Max | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recall | 0.29592 | 0.07143 | 0.21429 | 0.34694 | 0.13265 |
| mAP50 | 0.32998 | 0.01707 | 0.31468 | 0.34840 | 0.03372 |
| mAP50-95 | 0.14756 | 0.01650 | 0.12851 | 0.15726 | 0.02875 |

Both AP metrics are more stable than Recall. All three mAP50 values exceed C and v19, but the Recall
mean remains below v19 and D1 is below C. mAP50-95 mean exceeds C, yet D1 is below C. Therefore D's
abrasion intervention effect cannot be called a robust Recall improvement.

**Verdict: SEED-SENSITIVE.** AP ranking shows a positive direction, but abrasion Recall is Experiment
D's main class-level instability.

## 11. Insect_bite Limitation

The test set has only 3 insect_bite GT. At fixed `conf=0.25`:

| Run | TP | FP | FN |
| --- | ---: | ---: | ---: |
| v19 | 0 | 0 | 3 |
| C | 0 | 2 | 3 |
| D0 | 0 | 2 | 3 |
| D1 | 0 | 2 | 3 |
| D2 | 0 | 1 | 3 |

D2 remains 0 TP / 3 FN. AP percentage differences are not interpretable evidence with this sample size
and zero fixed TP across all models. **Verdict: insufficient evidence.** Independent valid/test sample
collection is needed before judging this class.

## 12. Fixed-threshold / Calibration Variance

### 12.1 Test audit at `conf=0.25`

| Item | D0 | D1 | D2 | Mean | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP | 83 | 45 | 73 | 67.00 | 38 |
| FP | 90 | 28 | 61 | 59.67 | 62 |
| FN | 140 | 178 | 150 | 156.00 | 38 |
| pure-background FP | 11 | 5 | 9 | 8.33 | 6 |
| background images with FP | 10 | 4 | 7 | 7.00 | 6 |

D2 is closer to D0 than D1 in TP/FN (10 versus 28 difference) and modestly closer to D0 in FP (29
versus 33). It is not a copy of D0: it occupies an intermediate operating regime.

`conf=0.25` is not stable across seeds. D1's much lower FP is coupled to 38 fewer TP and 38 more FN
than D0; it must not be called simply “better.” D2 confirms the continuum: comparable ranking AP can
produce very different counts at one absolute confidence threshold.

### 12.2 Best-F1 confidence behavior

| Seed | Val best F1 | Val threshold | Test best F1 | Test threshold | Absolute threshold gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 | 0.40237 | 0.18719 | 0.36398 | 0.21121 | 0.02402 |
| D1 | 0.34358 | 0.08008 | 0.35379 | 0.14815 | 0.06807 |
| D2 | 0.36116 | 0.15415 | 0.36267 | 0.20521 | 0.05105 |

For reference, C has validation/test thresholds `0.20621/0.08809`, gap `0.11812`. All three D seeds
have smaller cross-split threshold gaps than C. However, D validation optimal thresholds range from
`0.08008` to `0.18719` (range `0.10711`) and test thresholds from `0.14815` to `0.21121`
(range `0.06306`). Absolute score calibration clearly drifts across training seeds.

Thus both observations hold simultaneously:

1. D has more stable val→test threshold transfer than C for every seed.
2. A universal raw confidence value such as 0.25 does not produce stable operating behavior across D
   seeds.

This makes confidence calibration/validation-derived threshold selection the highest-priority next study.

## 13. Generalization Stability

Signed gap is `validation - test`; negative means test is higher.

| Metric gap (val-test) | v19 | C | D0 | D1 | D2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mask Precision | +0.34691 | +0.28906 | +0.06402 | -0.05629 | +0.25125 |
| Mask Recall | +0.02064 | +0.11652 | +0.01520 | -0.00094 | -0.01866 |
| Mask mAP50 | +0.06110 | +0.05707 | +0.03309 | -0.02995 | +0.01501 |
| Mask mAP50-95 | +0.05954 | +0.02019 | +0.01469 | +0.02395 | +0.00411 |

D three-seed absolute-gap descriptive statistics:

| Metric | Mean absolute gap | SD | Min | Max | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mask Precision | 0.12385 | 0.11039 | 0.05629 | 0.25125 | 0.19496 |
| Mask Recall | 0.01160 | 0.00939 | 0.00094 | 0.01866 | 0.01772 |
| Mask mAP50 | 0.02602 | 0.00966 | 0.01501 | 0.03309 | 0.01808 |
| Mask mAP50-95 | 0.01425 | 0.00993 | 0.00411 | 0.02395 | 0.01984 |

D2 Recall gap magnitude `0.01866` is again far below C `0.11652`, and its test Recall is actually
higher than validation. All three D seeds reproduce the Recall-gap repair while also increasing test
Recall, so this is not an artifact of both splits becoming uniformly worse. All three also have smaller
absolute mAP50 gaps than C. Strict AP is not uniform—D1's gap is slightly larger than C—and D2
Precision still has a large gap, although smaller than C.

**Conclusion: validation→test Recall generalization improvement is reproduced in all three seeds and is
the strongest evidence for the context intervention.**

## 14. Training Dynamics

Fitness here follows the prior reports' directly reproducible definition: Box mAP50-95 + Mask
mAP50-95 in `results.csv`.

| Item | D0 | D1 | D2 |
| --- | ---: | ---: | ---: |
| actual epochs | 147 | 134 | 163 |
| best fitness epoch | 97 | 84 | 113 |
| best Mask mAP50-95 epoch | 97 | 84 | 113 |
| best fitness | 0.38317 | 0.38277 | 0.38248 |
| best validation Mask mAP50-95 | 0.18956 | 0.18656 | 0.18513 |
| early stopping | yes | yes | yes |
| time to best | 9,464.2 s | 8,286.8 s | 11,701.7 s |
| total wall time | 14,329.4 s | 13,154.4 s | 16,562.5 s |

Best fitness range is only `0.00069`; best validation Mask mAP50-95 range is `0.00443`. Best epochs
range from 84 to 113, and all runs stop exactly 50 epochs after their best epoch. Different best epochs
are expected training variance, not instability by themselves.

All three show lower final Mask mAP50-95 than their best (`-0.03392`, `-0.02724`, `-0.01047`),
consistent with the previously documented late-training overfit/metric fluctuation pattern. No seed has
an optimization collapse. Training converges to nearly the same best-validation fitness region; the most
consequential variance appears in class-specific Recall/strict AP and especially confidence calibration,
not aggregate best fitness or mAP50 ranking.

## 15. Qualitative Consistency

![C/D0/D1/D2 qualitative comparison](../../training/analysis/v22-v23-v24-context6-three-seed/qualitative_test_examples.jpg)

The panel uses the same test images and fixed `conf=0.25`, NMS 0.70, Mask IoU 0.50 matching setup.
Selected rows illustrate: C miss with 3/3 recovery, C miss with 2/3 recovery, C correct with D
disagreement, stable insect_track/anthracnose recovery, abrasion seed sensitivity, and insect_bite all
miss. They supplement rather than replace the full audit.

Full test GT-instance consistency across D seeds:

| D seeds correct | Overall | abrasion | anthracnose | insect_bite | insect_track |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3/3 | 33 | 10 | 14 | 0 | 9 |
| 2/3 | 36 | 17 | 3 | 0 | 16 |
| 1/3 | 30 | 9 | 3 | 0 | 18 |
| 0/3 | 124 | 62 | 2 | 3 | 57 |

Additional patterns relative to C:

- C miss, all 3 D seeds recover: 3 GT (one each abrasion, anthracnose, insect_track).
- C miss, exactly 2/3 D seeds recover: 15 GT.
- C correct, D seeds inconsistent: 29 GT.
- all D seeds miss all insect_bite GT: 3/3 GT.

The distribution is explicitly tied to `conf=0.25`; it measures fixed-threshold consistency, not the
full ranking curves.

## 16. Three-seed Experiment D Verdict

### **SUCCESS WITH SEED SENSITIVITY**

The core hypothesis succeeds across seeds:

1. all 3/3 seeds improve overall test Recall over C and v19;
2. all 3/3 improve overall Mask mAP50 over C and v19, with very low dispersion;
3. all 3/3 repair the insect_track C regression in Recall/mAP50;
4. all 3/3 reproduce anthracnose Recall/mAP50 improvement;
5. all 3/3 sharply reduce the validation→test Recall gap versus C;
6. none has aggregate optimization collapse.

The result is not labeled ROBUST SUCCESS because material seed sensitivity remains in abrasion Recall,
strict per-class AP, and especially fixed-threshold TP/FP/FN. D1 also falls slightly below C on overall
mAP50-95, although all seeds exceed v19 and the D mean exceeds C. These qualifications do not reduce the
result to PARTIAL SUCCESS because the primary context/generalization directions reproduce in every seed.

## 17. Final Context Hypothesis Verdict

> Increasing targeted-crop context_scale from 4.0 to 6.0 improves Experiment C's generalization
> regression.

### **STRONGLY SUPPORTED**

Can be stated with relatively high confidence within this Dataset/model/split scope:

- overall Recall improves over C and v19 in all three seeds;
- overall Mask mAP50 improves over C and v19 in all three seeds with small variance;
- validation→test Recall gap is far smaller than C in all three seeds;
- insect_track Recall/mAP50 regression is robustly recovered;
- anthracnose Recall/mAP50 direction is robustly improved;
- no seed collapses and all reach a nearly identical best-validation fitness region.

Still cannot be strongly claimed:

- robust abrasion Recall improvement;
- uniform strict per-class mask quality improvement over C;
- insect_track strict AP recovery to v19 in every seed;
- any insect_bite improvement;
- stable deployment behavior at raw `conf=0.25`;
- a universally calibrated confidence scale across seeds.

This evidence level applies only to the current Dataset, fixed splits, YOLO26s-seg initialization and
three seeds. It is not a population-level statistical significance claim.

## 18. Whether to Stop Seed Replication

### **STOP**

Seeds 0/1/2 already answer the main replication question: every seed reproduces overall Recall,
mAP50, insect_track recovery, anthracnose direction and Recall-gap improvement, with no collapse. A
fourth or fifth seed could refine descriptive variance but is unlikely to resolve the now-dominant
deployment uncertainty: confidence calibration. Continuing seed replication has lower information value
than evaluating a validation-locked threshold policy.

Additional seed runs would become warranted only if a later integrity issue is found or a materially
changed Dataset/training setup defines a new experiment.

## 19. Whether D Becomes the New Experimental Baseline

### **YES**

The Experiment D intervention—its `context_scale=6.0` Dataset and otherwise fixed training setup—should
be the new experimental baseline for subsequent controlled studies. This recommendation concerns the
experimental design, not selection of the highest test-scoring seed.

- **Experimental baseline:** use the D Dataset/setup and preserve its data, initialization and training
  contract for the next isolated intervention.
- **Deployment checkpoint selection:** do not choose D0/D1/D2 because one has the best test metric.
  Define selection/calibration rules using validation only, lock them, and evaluate test once. Operational
  error costs and an independent holdout may be needed for final deployment choice.

## 20. Three-seed Summary Statistics

### 20.1 Overall

| Metric | D mean | SD | Min | Max | v19 | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask Precision | 0.37082 | 0.02619 | 0.35440 | 0.40103 | 0.35381 | 0.36630 |
| Mask Recall | 0.36155 | 0.03521 | 0.32426 | 0.39422 | 0.30096 | 0.28042 |
| Mask mAP50 | 0.34520 | 0.00315 | 0.34157 | 0.34718 | 0.30571 | 0.31710 |
| Mask mAP50-95 | 0.17236 | 0.00831 | 0.16308 | 0.17911 | 0.15013 | 0.16777 |

### 20.2 Per-class

| Class | Metric | D mean | SD | Min | Max | v19 | C |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abrasion | Recall | 0.29592 | 0.07143 | 0.21429 | 0.34694 | 0.36295 | 0.26531 |
| abrasion | mAP50 | 0.32998 | 0.01707 | 0.31468 | 0.34840 | 0.27830 | 0.31286 |
| abrasion | mAP50-95 | 0.14756 | 0.01650 | 0.12851 | 0.15726 | 0.10694 | 0.13987 |
| anthracnose | Recall | 0.81818 | 0.09091 | 0.72727 | 0.90909 | 0.59091 | 0.63636 |
| anthracnose | mAP50 | 0.74384 | 0.03236 | 0.70748 | 0.76946 | 0.67552 | 0.67704 |
| anthracnose | mAP50-95 | 0.42242 | 0.03151 | 0.39309 | 0.45573 | 0.38899 | 0.40042 |
| insect_bite* | Recall | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| insect_bite* | mAP50 | 0.04453 | 0.03890 | 0.01523 | 0.08867 | 0.00882 | 0.08386 |
| insect_bite* | mAP50-95 | 0.02645 | 0.01710 | 0.01370 | 0.04589 | 0.00529 | 0.04790 |
| insect_track | Recall | 0.33211 | 0.02023 | 0.32000 | 0.35546 | 0.25000 | 0.22000 |
| insect_track | mAP50 | 0.26244 | 0.01159 | 0.25546 | 0.27582 | 0.26022 | 0.19465 |
| insect_track | mAP50-95 | 0.09301 | 0.01518 | 0.08367 | 0.11053 | 0.09931 | 0.08287 |

\* insect_bite has only 3 test GT and 0 fixed TP for all models; all apparent AP differences are
**insufficient evidence**.

## 21. Recommended Next Experiment

### Validation-derived confidence threshold / calibration study

Keep Dataset and weights frozen. Do not retrain and do not add context, sampling, loss or augmentation
changes.

Proposed protocol:

1. Freeze D0/D1/D2 `best.pt`, the current validation/test splits, `imgsz=1024`, NMS IoU 0.70,
   `max_det=300`, and class-aware Mask IoU 0.50 audit.
2. Pre-register a threshold grid and one objective before examining any new test result. A defensible
   primary objective is overall validation Mask F1; also report per-class Recall/FP trade-offs because
   macro F1 can be noisy for tiny classes.
3. **Seed-specific policy:** select one global confidence threshold for each checkpoint from that
   checkpoint's validation predictions only. Lock all three thresholds, then apply each once to the
   corresponding test predictions.
4. **Shared-policy sensitivity:** independently select one common D threshold that maximizes mean
   validation objective across D0/D1/D2. Lock it and apply it to all three test predictions. This tests
   whether one deployment threshold can be shared across training replications.
5. Compare both locked policies with the pre-existing `conf=0.25` reference using TP/FP/FN,
   per-class Recall, background FP images, and between-seed ranges. Test metrics are reporting endpoints,
   never selection criteria.
6. If score calibration beyond threshold selection is studied (for example monotonic calibration), fit
   it on validation only, preserve ranking, lock all parameters, and use test only for final evaluation.

Success criterion: materially reduce between-seed TP/FN/FP dispersion without sacrificing the replicated
Recall/mAP50 direction. This experiment directly addresses the observed uncertainty and changes neither
model weights nor Dataset.

Do not prioritize `context_scale=8/10`, new loss weighting, more oversampling or new augmentation at this
stage. Three seeds already show that context=6 is stable on the primary hypothesis; mixing in another
training intervention would obscure the calibration question.

## 22. Limitations

- Three seeds support replication and descriptive variance assessment, not reliable estimation of the
  full seed distribution or formal statistical significance.
- The same fixed validation/test splits are reused across experiments; conclusions are split-specific.
- insect_bite has 3 test GT and no fixed TP, so its AP values cannot support an intervention verdict.
- anthracnose has 22 test GT, making instance-level Recall changes coarse.
- Official curve P/R and fixed `conf=0.25` audit are different operating points and must not be
  interchanged.
- Per-run Ultralytics version is embedded in checkpoints, but complete per-run Python/PyTorch build
  snapshots were not stored.
- Qualitative examples were selected to display predefined consistency patterns and are not prevalence
  estimates; the full 223-GT consistency table is the quantitative result.
- Selecting a deployment checkpoint or threshold from these test outcomes would bias the final estimate.

## Reproducibility Artifacts

- D2 evaluation: `training/analysis/v22-v23-v24-context6-three-seed/v24_evaluation.json`
- D2 Ultralytics outputs: `training/analysis/v22-v23-v24-context6-three-seed/ultralytics/v24_{val,test}/`
- Three-seed summary: `training/analysis/v22-v23-v24-context6-three-seed/three_seed_summary.json`
- Summary generation script: `training/analysis/v22-v23-v24-context6-three-seed/build_three_seed_summary.py`
- Qualitative panel: `training/analysis/v22-v23-v24-context6-three-seed/qualitative_test_examples.jpg`
- Panel generation script: `training/analysis/v22-v23-v24-context6-three-seed/make_three_seed_qualitative_panel.py`
- v19/C machine-readable evaluation: `training/analysis/v19-v20-v21-targeted-crop/analysis.json`
- D0 evaluation: `training/analysis/v19-v21-v22-context-scale/v22_evaluation.json`
- D1 evaluation: `training/analysis/v22-context6-seed-replication/v23_evaluation.json`
- Experiment D reproducibility: `training/packages/local_defects/experiments/exp_D_context6/reproducibility.json`
