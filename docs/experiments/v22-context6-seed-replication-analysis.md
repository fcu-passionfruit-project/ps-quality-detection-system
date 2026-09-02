# Experiment D：Context Scale 6 Training Seed Replication

- 分析日期：2026-09-02
- 原 run：`training/runs/segment/v22_D_context6`（training seed 0；下文 `D0`）
- replication run：`training/runs/segment/v23_D_context6_seed1`（training seed 1；下文 `D1`）
- 比較組：`v19_defect_epoch200_pw050_cp01`（`v19`）、`v21_C_targeted_crop`（`C`）
- 正式權重：各 run 的 `weights/best.pt`
- Replication verdict：**STRONG REPLICATION（但 fixed operating point 有明顯 seed-dependent calibration variance）**

## 1. Executive Summary

D1 重現了 Experiment D 最重要的跨模型方向：test Mask Recall `0.32426` 仍高於 C
`0.28042` 與 v19 `0.30096`；Mask mAP50 `0.34157` 仍高於 C `0.31710` 與 v19
`0.30571`。較嚴格的 Mask mAP50-95 為 `0.16308`，低於 C `0.16777` 但只差
`0.00469`，且仍高於 v19 `0.15013`。D0/D1 的 overall Mask mAP50 range 只有
`0.00527`，顯示 ranking performance 收斂到相近區域；Recall/Precision 則互相交換約
`0.04–0.05`，反映 operating-point calibration 對 training seed 較敏感。

核心類別方向也大致重現：D1 anthracnose Recall `0.72727` 與 mAP50 `0.70748` 仍高於
C/v19；insect_track Recall `0.35546` 高於三個既有模型，mAP50 `0.25546` 明顯修復 C
`0.19465` 的 regression，且接近 v19 `0.26022`。但兩者的 strict AP 與固定
`conf=0.25` TP 未完整複製 D0。abrasion 的 D0 Recall recovery 沒有重現；insect_bite
仍為 0 TP / 3 FN，結論是 **insufficient evidence**。

D1 沒有重現 D0 的高 FP downside。在固定 operating point，D0 為
`83 TP / 90 FP / 140 FN`，D1 為 `45 / 28 / 178`；純背景 FP 也由 `11` 降到 `5`。
因此「D 必然以增加 FP 換 Recall」不能跨 seed 保留；較準確的結論是兩個 D seeds 的
curve-derived Recall 都高，但相同 `conf=0.25` 下 calibration 明顯不同。

validation→test Mask Recall gap 在 D1 為 `-0.00094`，絕對值遠小於 C `0.11652`；最佳
F1 threshold gap 為 `0.06807`，也小於 C `0.11812`。larger context 改善 C 的
cross-split generalization regression 因而具有跨 seed 證據。加入 D1 後，Experiment D 的
`SUCCESS` 結論仍成立；「context_scale 6 優於 4」的證據評為 **strengthened**，但應把
abrasion、strict AP 與 fixed-threshold calibration 的 seed sensitivity 寫入限制。

## 2. Replication Objective

本 replication 不要求 D1 精確複製 D0 數值，而是檢驗 D1 是否重現 D 相對 C/v19 的
主要改善方向：overall Recall/AP、anthracnose、insect_track recovery，以及較小的
validation→test generalization gap。這次只讀取既有 Dataset/run artifacts，未重新訓練、
未修改 weights，也未重新生成 Dataset。

## 3. Replication Integrity Audit

### 3.1 Run identity and complete `args.yaml` comparison

兩份 `args.yaml` 都有 116 個相同 keys。逐欄比較後只有三個值不同：training `seed`，以及
不構成實驗變因的 `name`、`save_dir`。其餘 113 個設定完全相同，未發現 training 或
augmentation confounder。

| Item | D0 | D1 | Audit |
| --- | --- | --- | --- |
| model | `yolo26s-seg.pt` | same | PASS |
| data | `training/packages/local_defects/data_exp_D.yaml` | same | PASS |
| imgsz | 1024 | 1024 | PASS |
| epochs | 200 | 200 | PASS |
| patience | 50 | 50 | PASS |
| batch | 4 | 4 | PASS |
| device | `mps` | `mps` | PASS |
| deterministic | true | true | PASS |
| cls_pw | 0.5 | 0.5 | PASS |
| copy_paste | 0.1 | 0.1 | PASS |
| copy_paste_mode | `mixup` | `mixup` | PASS |
| optimizer | `auto` | `auto` | PASS |
| seed | 0 | 1 | intended experimental variable |
| name | `v22_D_context6` | `v23_D_context6_seed1` | run metadata only |
| save_dir | D0 run directory | D1 run directory | run metadata only |
| all other 101 listed settings | same | same | PASS |

「all other」包含 learning-rate/momentum/weight decay/warmup、pretrained、AMP、mosaic、
mixup、cutmix、HSV、translate/scale/shear/perspective、flip、mask settings、loss weights、
close_mosaic、fraction、rect、multi_scale、validation/NMS settings及其餘 Ultralytics arguments。

### 3.2 Dataset and checkpoint identity

| Integrity item | Expected artifact | Current audit | Result |
| --- | ---: | ---: | --- |
| D training-input Dataset SHA-256 | `8d17834c…996ba0b` | same | PASS |
| source Dataset SHA-256 | `be2b7701…6ae42e` | same | PASS |
| source train split | 433 images; `93cc5489…5e5d56` | same | PASS |
| generated crops | 132 | 132 | PASS |
| train entries | 565 | 565 | PASS |
| D train-list SHA-256 | `8d685ded…c8c22f` | same | PASS |
| valid split | 93 images; `639ac687…e70bb` | same | PASS |
| test split | 92 images; `be82ea70…7bb42` | same | PASS |
| crop manifest SHA-256 | `07b57c72…2f419` | same | PASS |
| source-splits manifest SHA-256 | `62bf9675…69ea` | same | PASS |
| original image count | 433 | 433 | PASS |
| initial checkpoint path | `yolo26s-seg.pt` | same in both run args/checkpoints | PASS |
| initial checkpoint SHA-256 | `3da1d83e…837a4` | current artifact same | PASS |

D images/labels、train list、data YAML 與 reproducibility artifacts 的 modification time 都早於
D0/D1 training；重新計算的完整 Dataset digest 也逐位元相符，沒有 seed=1 重新生成 Dataset
的證據。Ultralytics training 可能建立或重用 `labels.cache`，但 cache 不是 Dataset
image/label/list contract，也未改變上述 digest。本次 test evaluation 新建的 test cache 已在
完成後移除。

兩個 `best.pt` 都內嵌 Ultralytics `8.4.122`，且目前/paired reproducibility environment 為
Python `3.12.13`、PyTorch `2.13.0`、macOS arm64。Ultralytics 版本的 run-level parity 可直接
由 checkpoint 證明；run artifacts 沒有各自保存 PyTorch build metadata，因此「兩次 training
使用完全相同 PyTorch build」無法獨立從 run artifact 證明。沒有觀察到版本差異，但這一點是
integrity audit 的證據限制，不補猜數字。

### 3.3 Integrity conclusion

沒有發現 seed 以外的效能 confounder。唯一 provenance limitation 是缺少每個 training run
各自的 PyTorch/environment snapshot；它不構成已知 confounder。

## 4. Evaluation Protocol

正式比較只用各 run 的 `weights/best.pt`。v19/C/D0 沿用既有 machine-readable evaluation；
只補跑缺少的 D1 validation/test：

| Setting | Value |
| --- | --- |
| data | `training/packages/local_defects/data.yaml`（與 D YAML 的 valid/test 相同） |
| split | `val`, `test` |
| imgsz | 1024 |
| batch | 8 |
| device | `mps` |
| workers | 0 |
| official curve confidence | `None`（Ultralytics floor 0.001） |
| NMS IoU | 0.70 |
| max_det | 300 |
| fixed audit confidence | 0.25 |
| Mask/Box TP IoU | 0.50 |
| matching | class-aware one-to-one greedy matching |

沒有替 D1 單獨調 threshold。Ultralytics official P/R 是其 curve-derived operating point；
fixed audit 是相同 `conf=0.25` 的 deployment-oriented audit，兩者不可直接互換。
正式 evaluated checkpoint SHA-256：D0 `0165feea…61eac4`；D1
`a84ffc5f…dae25`。

## 5. Overall Metrics

### 5.1 Test Mask metrics

| Metric | v19 | C | D seed0 | D seed1 | seed1-seed0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precision | 0.35381 | 0.36630 | 0.35440 | 0.40103 | +0.04663 |
| Recall | 0.30096 | 0.28042 | 0.36618 | 0.32426 | -0.04192 |
| mAP50 | 0.30571 | 0.31710 | 0.34684 | 0.34157 | -0.00527 |
| mAP50-95 | 0.15013 | 0.16777 | 0.17489 | 0.16308 | -0.01182 |

D1 的 Recall 高於 C `+0.04384`、高於 v19 `+0.02329`。mAP50 高於 C `+0.02447`、
高於 v19 `+0.03586`。mAP50-95 比 C 低 `0.00469`，但比 v19 高 `0.01295`，即落在
兩者之間且接近 C。這是成功方向的重現，不要求 D1 勝過 D0。

### 5.2 Test Box metrics（補充）

| Metric | v19 | C | D seed0 | D seed1 | seed1-seed0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precision | 0.35264 | 0.37906 | 0.36041 | 0.38518 | +0.02477 |
| Recall | 0.30107 | 0.28824 | 0.36618 | 0.30750 | -0.05868 |
| mAP50 | 0.30968 | 0.32287 | 0.35094 | 0.33193 | -0.01902 |
| mAP50-95 | 0.16757 | 0.18790 | 0.19454 | 0.18461 | -0.00993 |

Box 指標同樣呈現 D1 Precision 較高、Recall 較低；Box mAP50 仍高於 C/v19，strict Box
AP 與 C 接近但稍低。方向與 Mask 解讀一致。

## 6. Seed Variability

目前只有兩個 seeds；以下只報 descriptive mean/range，不將其解讀為完整 variance
distribution。

### 6.1 Overall Mask

| Metric | D seed0 | D seed1 | Mean | Min | Max | Range / absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Precision | 0.35440 | 0.40103 | 0.37772 | 0.35440 | 0.40103 | 0.04663 |
| Recall | 0.36618 | 0.32426 | 0.34522 | 0.32426 | 0.36618 | 0.04192 |
| mAP50 | 0.34684 | 0.34157 | 0.34421 | 0.34157 | 0.34684 | 0.00527 |
| mAP50-95 | 0.17489 | 0.16308 | 0.16899 | 0.16308 | 0.17489 | 0.01182 |

兩個 seeds 的 mAP50 很接近；mAP50-95 有中等差距；P/R 交換更明顯。D 的 two-seed mean
Recall、mAP50、mAP50-95 分別為 `0.34522`、`0.34421`、`0.16899`，都高於 C 與
v19 對應值，但兩個 seeds 不足以建立信賴區間或估計尾端風險。

### 6.2 Per-class Recall and Mask mAP50-95

| Class | Metric | D seed0 | D seed1 | Mean | Range / absolute difference |
| --- | --- | ---: | ---: | ---: | ---: |
| abrasion | Recall | 0.32653 | 0.21429 | 0.27041 | 0.11224 |
| abrasion | mAP50-95 | 0.15691 | 0.12851 | 0.14271 | 0.02840 |
| anthracnose | Recall | 0.81818 | 0.72727 | 0.77273 | 0.09091 |
| anthracnose | mAP50-95 | 0.41844 | 0.39309 | 0.40576 | 0.02535 |
| insect_bite | Recall | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| insect_bite | mAP50-95 | 0.01370 | 0.04589 | 0.02980 | 0.03218 |
| insect_track | Recall | 0.32000 | 0.35546 | 0.33773 | 0.03546 |
| insect_track | mAP50-95 | 0.11053 | 0.08483 | 0.09768 | 0.02570 |

abrasion Recall 是最大 seed range；insect_track Recall 相對穩定且兩個 seeds 都高於 C。
insect_bite 的 AP range 不能視為可靠 seed variance，因 test 只有 3 GT 且兩個 seeds 都是
0 fixed TP。

## 7. Per-class Test Comparison

以下全部是 Mask curve metrics。

| Class | Metric | v19 | C | D seed0 | D seed1 |
| --- | --- | ---: | ---: | ---: | ---: |
| abrasion | P | 0.41085 | 0.49503 | 0.45900 | 0.50932 |
| abrasion | R | 0.36295 | 0.26531 | 0.32653 | 0.21429 |
| abrasion | mAP50 | 0.27830 | 0.31286 | 0.32687 | 0.31468 |
| abrasion | mAP50-95 | 0.10694 | 0.13987 | 0.15691 | 0.12851 |
| anthracnose | P | 0.65724 | 0.61356 | 0.62148 | 0.75153 |
| anthracnose | R | 0.59091 | 0.63636 | 0.81818 | 0.72727 |
| anthracnose | mAP50 | 0.67552 | 0.67704 | 0.76946 | 0.70748 |
| anthracnose | mAP50-95 | 0.38899 | 0.40042 | 0.41844 | 0.39309 |
| insect_bite | P | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| insect_bite | R | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| insect_bite | mAP50 | 0.00882 | 0.08386 | 0.01523 | 0.08867 |
| insect_bite | mAP50-95 | 0.00529 | 0.04790 | 0.01370 | 0.04589 |
| insect_track | P | 0.34713 | 0.35663 | 0.33714 | 0.34327 |
| insect_track | R | 0.25000 | 0.22000 | 0.32000 | 0.35546 |
| insect_track | mAP50 | 0.26022 | 0.19465 | 0.27582 | 0.25546 |
| insect_track | mAP50-95 | 0.09931 | 0.08287 | 0.11053 | 0.08483 |

### 7.1 Direction verdict by class

| Class | Verdict | Evidence |
| --- | --- | --- |
| abrasion | **未重現 D0 的 Recall recovery；整體部分重現** | D1 Recall 低於 C；mAP50 約等於 C且高於 v19，strict AP 高於 v19但低於 C/D0。abrasion 對 seed 最敏感。 |
| anthracnose | **方向重現，但 fixed/strict 指標僅部分重現** | D1 P/R/mAP50 高於 C與v19；strict AP 略低 C `0.00733`、略高 v19 `0.00410`。 |
| insect_bite | **資料不足** | test 只有 3 GT；D1 與其他模型同為 0 TP / 3 FN。curve AP 波動不構成成功/失敗證據。 |
| insect_track | **主要 recovery 方向重現，fixed recovery 未重現** | D1 Recall 高於三者；mAP50 修復 C 大部分 regression且只比 v19低 `0.00476`；strict AP略高 C但低 v19。固定0.25 TP則低於C。 |

### 7.2 Core replication targets

1. **Overall Mask Recall：重現。** D1 `0.32426` > v19 `0.30096` > C `0.28042`；
   D1-D0 為 `-0.04192`。
2. **Overall Mask mAP50-95：部分數值重現、整體方向可接受。** D1 `0.16308` 落在 C
   `0.16777` 與 v19 `0.15013` 之間；距 C 只有 `-0.00469`。
3. **anthracnose：改善方向重現。** D1 Recall `0.72727` 高於 C/v19，mAP50 也高於兩者；
   strict AP 與 fixed TP 未完全重現 D0 的幅度。
4. **insect_track：C regression 被大致修復。** D1 Recall `0.35546` 超越 v19/C/D0；
   mAP50 `0.25546` 接近 v19 `0.26022` 且遠高於 C `0.19465`。不是重新出現 C 等級的
   regression，但 strict AP/fixed TP 顯示 recovery 不如 D0 全面。

## 8. Precision / Recall / FP Trade-off

### 8.1 Fixed `conf=0.25` Mask audit（test）

| Item | v19 | C | D seed0 | D seed1 |
| --- | ---: | ---: | ---: | ---: |
| fixed TP | 72 | 63 | 83 | 45 |
| fixed FP | 80 | 74 | 90 | 28 |
| fixed FN | 151 | 160 | 140 | 178 |
| pure-background FP predictions | 7 | 8 | 11 | 5 |
| background images with FP | 4/35 | 6/35 | 10/35 | 4/35 |

| Class | D0 TP / FP / FN | D1 TP / FP / FN |
| --- | ---: | ---: |
| abrasion | 31 / 26 / 67 | 11 / 3 / 87 |
| anthracnose | 18 / 11 / 4 | 14 / 4 / 8 |
| insect_bite | 0 / 2 / 3 | 0 / 2 / 3 |
| insect_track | 34 / 51 / 66 | 20 / 19 / 80 |

D1 **沒有**在固定 0.25 呈現 D0 的「TP/Recall 增加但 FP 增加」型態，而是更保守：少
38 TP、少 62 FP、多 38 FN。D0 的高 FP 不是跨 seed 必然現象，較像 optimization 導致的
calibration/score-distribution variance。另一方面，兩個 seeds 的 curve Recall 都高於 C，
且 D1 official Precision 較 D0 高 `0.04663`、Recall 低 `0.04192`，mAP50 只低
`0.00527`。因此 aggregate ranking 仍支持 D，但共同 deployment threshold 的穩定性尚未
被證明；不能把 D1 的低 fixed FP 解讀為已解決 FP，因為它同時漏掉更多 GT。

## 9. Generalization Comparison

下表為 signed `validation - test`；負值代表 test 較高。

| Metric gap (val-test) | v19 | C | D seed0 | D seed1 |
| --- | ---: | ---: | ---: | ---: |
| Mask Precision | +0.34691 | +0.28906 | +0.06402 | -0.05629 |
| Mask Recall | +0.02064 | +0.11652 | +0.01520 | -0.00094 |
| Mask mAP50 | +0.06110 | +0.05707 | +0.03309 | -0.02995 |
| Mask mAP50-95 | +0.05954 | +0.02019 | +0.01469 | +0.02395 |

D1 的 Recall absolute gap `0.00094` 明顯小於 C `0.11652`，且 test Recall 本身較高，
不是靠讓兩邊一起變差製造小 gap。D1 mAP50 的 test 值反而高於 validation；strict AP gap
`0.02395` 略大於 C `0.02019`，所以不是所有 cross-split metrics 都改善。

最佳 Mask F1 與 threshold：

| Split | v19 | C | D seed0 | D seed1 |
| --- | --- | --- | --- | --- |
| validation | 0.361 @ 0.318 | 0.408 @ 0.206 | 0.402 @ 0.187 | 0.344 @ 0.080 |
| test | 0.335 @ 0.196 | 0.345 @ 0.088 | 0.364 @ 0.211 | 0.354 @ 0.148 |
| absolute threshold gap | 0.12212 | 0.11812 | 0.02402 | 0.06807 |

D1 threshold gap 雖大於 D0，但仍比 C 小約 42%。Recall gap與 threshold gap 兩項在兩個 D
seeds 都優於 C，支持 larger context 對 C generalization regression 的改善可重現；D1 的
validation F1 較低與 threshold 整體下移則顯示 calibration 並非 seed-invariant。

## 10. Training Dynamics

| Item | D seed0 | D seed1 |
| --- | ---: | ---: |
| actual epochs | 147 | 134 |
| early stopping | yes | yes |
| best fitness epoch | 97 | 84 |
| best fitness | 0.38317 | 0.38277 |
| best Mask mAP50-95 epoch | 97 | 84 |
| best Mask mAP50-95 in `results.csv` | 0.18956 | 0.18656 |
| time to best fitness | 9,464.2 s | 8,286.8 s |
| total wall time | 14,329.4 s | 13,154.4 s |

兩個 seeds 的 best fitness 只差 `0.00040`，best validation Mask mAP50-95 只差
`0.00300`；D1 的 best epoch 早 13 epochs，兩者都在其 best 後 50 epochs early stop。
這是相近 performance region，不是完全不同的 optimization outcome。

| Loss | D0 start→final | D1 start→final |
| --- | --- | --- |
| train box | 1.72494→1.22557 | 1.68445→1.25836 |
| train seg | 3.29779→1.73223 | 3.28581→1.79287 |
| train cls | 16.61720→0.91694 | 18.37730→0.94179 |
| train l1 | 0.01276→0.00944 | 0.01244→0.01011 |
| train sem | 7.11677→0.64888 | 4.61471→0.75276 |

| Validation loss | D0 min epoch / final | D1 min epoch / final |
| --- | --- | --- |
| box | 114 / 1.75781 | 1 / 1.77728 |
| seg | 84 / 2.45806 | 79 / 2.49239 |
| cls | 138 / 1.64642 | 121 / 1.63463 |
| l1 | 111 / 0.01829 | 82 / 0.01863 |

兩者 train losses 持續下降，而 validation seg/l1 在中後期達最低後回升，均有類似 overfit
pattern；沒有一個 seed 顯示完全不同的 collapse。D1 最後 20 epochs 的 Mask metric std：
P `0.15072`、R `0.03488`、mAP50 `0.01698`、mAP50-95 `0.01527`；D0 分別為
`0.04961`、`0.03329`、`0.02055`、`0.01379`。除 D1 Precision（受 small-class macro P
跳動影響）外，late-training AP/Recall variability 接近。不能只依 loss 小差異宣稱不穩定。

## 11. Prediction-level / Qualitative Comparison

![D seed replication qualitative examples](../../training/analysis/v22-context6-seed-replication/qualitative_test_examples.jpg)

上圖在相同 `conf=0.25`、NMS 0.70、Mask match IoU 0.50 下選取六種 pattern，四欄為
ground truth、C、D0、D1。粗線 `TARGET` 是 pattern audit 對應的 GT instance；其餘輪廓
提供同圖 context。圖中同時可見：兩 seed 共同 recover、D0-only recovery、D1-only
recovery，以及 insect_track/anthracnose 的正反例。

完整 prediction-level audit（不是只看選例）：

- D0/D1 都 correct：40 GT instances。
- C miss、D0/D1 都 recover：6（insect_track 4、anthracnose 1、abrasion 1）。
- D0 correct、D1 miss：43；其中 C 也 miss 的有 19。
- D1 correct、D0 miss：5；這 5 個也都是 C miss（insect_track 4、abrasion 1）。
- 兩個 D seeds 與 C 都 miss：130，包括全部 3 個 insect_bite。

這些 pattern 支持共同的 recovery 方向，但也直觀呈現 D1 在固定 0.25 較保守；qualitative
examples 只作補充，不取代 aggregate curve metrics。

## 12. Replication Verdict

### **STRONG REPLICATION**

依預先定義的方向標準，D1 大致同時符合五項：

1. overall test Recall 高於 C 且高於 v19；
2. test Mask mAP50-95 接近 C、仍高於 v19，mAP50 則高於兩者；
3. insect_track 沒有重現 C 的明顯 regression，Recall 更高、mAP50 接近 baseline；
4. anthracnose Recall/mAP50 維持改善方向；
5. validation→test Recall gap 與 optimal-threshold gap 都明顯小於 C。

這個 strong verdict 不代表所有數值完全重現。未重現項目是 abrasion Recall recovery、D0
的 fixed-threshold高 FP/高 TP operating point，以及 D0 對 anthracnose/insect_track strict AP
與 fixed TP 的完整改善幅度。這些差異表示 D 的 ranking/generalization 結論比「固定 0.25
即可穩定部署」更可重現。

## 13. Updated Experiment D Conclusion

1. **Experiment D = SUCCESS 仍成立。** D1 的主要 preregistered direction 沒有回到 C
   regression，且 two-seed mean 的 overall R/mAP50/mAP50-95 都高於 C/v19。
2. **「context_scale 6 優於 4」證據：strengthened。** 現在不只 seed0，在 seed1 也看到
   overall Recall/mAP50、anthracnose 與 insect_track 的主要方向及較小 Recall gap。
3. **larger context 改善 C generalization regression：仍可合理認為成立。** 最有力的跨 seed
   證據是 test Recall、本來 regression 的 insect_track、Recall gap與 threshold gap共同一致。
4. **可跨 seed 保留：** overall Recall > C/v19、mAP50 > C/v19、insect_track curve recovery、
   anthracnose Recall/mAP50 direction、Recall gap明顯小於C、best training performance region
   相近。
5. **只在 seed0 或未能跨 seed 保留：** abrasion Recall recovery；overall strict AP 高於 C；
   insect_track strict AP 超過 baseline；D0 的高 fixed TP/FP operating point；極小的 0.024
   threshold gap。

因此原本的「Context hypothesis = strongly supported」可保留，但需收斂為：
**larger context 對 overall/generalization 與 insect_track regression 的改善具有強的雙 seed
方向證據；個別 class strict AP 與固定 threshold calibration 仍有顯著 seed sensitivity。**

## 14. Recommendation on seed=2 / Next Experiment

雖然本次判為 strong replication，仍**建議補 seed=2**，理由不是 D1 失敗，而是目前只有兩點
不足以估計 variance distribution，且 D1 的 strict AP 剛好落在 C/v19 之間、abrasion 與固定
0.25 operating point 差異大。第三個 seed 能判斷這些是 D1 特例還是穩定的 calibration
variance，會明顯提高畢業專題報告對「context 6 優於 4」的可信度。

下一步優先順序：

1. 固定同一 D Dataset、初始 checkpoint、全部 training/evaluation parameters，只跑 seed=2。
2. 若 seed=2 仍重現 aggregate direction，再停止 seed replication，進入共同
   validation-derived operating-threshold/calibration study；不可在 test 各自調 threshold。
3. insect_bite 應先擴充獨立 valid/test 真實樣本，再評估 intervention；不要依 3 GT 的 AP
   波動改 loss/sampling/augmentation。

本分析不建議立刻改 loss、sampling、augmentation 或繼續掃 context scale，因為那會混入新
變因，且目前最直接的剩餘不確定性是 seed/calibration variance。

## 15. Limitations

- 只有 seed 0/1，mean/range 是描述統計，不能估計完整 variance distribution或做正式推論。
- insect_bite test 只有 3 GT，所有模型 fixed TP 都是 0，證據不足。
- anthracnose test 只有 22 GT，單一 instance 的變化就會明顯改變 Recall。
- official P/R 與 fixed audit 是不同 operating point；D1 顯示 threshold calibration 不能假設
  跨 seed 相同。
- qualitative panel 是依固定 audit pattern 選例，不代表資料分布比例。
- run artifacts 能直接證明相同 Ultralytics 版本，但未各自保存 PyTorch build snapshot。
- 本結論限於目前固定 Dataset/splits、YOLO26s-seg checkpoint與兩個 training seeds，不能外推
  到其他資料版本、架構或 production threshold。

## Reproducibility Artifacts

- D1 machine-readable evaluation：`training/analysis/v22-context6-seed-replication/v23_evaluation.json`
- D1 Ultralytics val/test plots：`training/analysis/v22-context6-seed-replication/ultralytics/v23_{val,test}/`
- qualitative panel：`training/analysis/v22-context6-seed-replication/qualitative_test_examples.jpg`
- panel generation script：`training/analysis/v22-context6-seed-replication/make_qualitative_panel.py`
- v19/C evaluation：`training/analysis/v19-v20-v21-targeted-crop/analysis.json`
- D0 evaluation：`training/analysis/v19-v21-v22-context-scale/v22_evaluation.json`
- Experiment D Dataset report：`docs/experiments/v22-experiment-d-context6-dataset-2026-09-01.md`
- Experiment D reproducibility：`training/packages/local_defects/experiments/exp_D_context6/reproducibility.json`
