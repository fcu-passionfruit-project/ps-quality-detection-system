# v25：類別平衡＋增強參數調整（視角層級評估）

- 日期：2026-10-14
- Run：`training/runs/segment/v25_balanced_aug_img800`
- Baseline：`training/runs/segment/v19_defect_epoch200_pw050_cp01`
- 評估方法：`docs/specs/local-defect-view-level-evaluation.md`
- 評估輸出：`training/runs/eval/v19_vs_v25/`（`report.md`、`summary.csv`、逐張面積）

## 設定

Dataset 為 `v2_defect` 的副本 `training/packages/v2_defect_balanced`。valid／test 和 v19 相同：重現 v19 的 test Mask Recall 0.299，既有紀錄為 0.301。

| 項目 | v19 | v25 |
| --- | --- | --- |
| train 清單 | 433 張原圖 | `train_balanced.txt` 574 筆：含 anthracnose 或 insect_bite 的 47 張各重複 ×4 |
| imgsz／batch | 1024／4 | 800／12（RTX 3070 8 GB 上限） |
| scale | 0.5 | 0.2 |
| degrees／flipud | 0／0 | 180／0.5 |
| mosaic／close_mosaic | 1.0／10 | 0.5／30 |
| cls_pw／copy_paste | 0.5／0.1 | 相同 |
| epochs | 200 | 200，patience 60，epoch 182 早停 |
| Ultralytics | 8.4.122（Mac MPS） | 8.4.46（CUDA） |

平衡清單由 `passionfruit.data_preparation.generate_class_aware_train_list --anthracnose-factor 4 --insect-bite-factor 4` 產生。這次同時改了多個變因，結果只能說明「整組設定」的效果，無法歸因到單一參數。

## 結果（test，92 張）

conf threshold 由 valid Macro QWK 選出：v19 為 0.30，v25 為 0.05。

| 指標 | v19 | v25 |
| --- | ---: | ---: |
| **Macro 嚴重度 QWK**（abrasion、anthracnose、insect_track） | 0.623 [0.438, 0.716] | **0.774** [0.659, 0.837] |
| 任一瑕疵 Sensitivity | 0.754 | **0.930** |
| 任一瑕疵 Specificity | **0.886** | 0.657 |
| 任一瑕疵 F1 | 0.827 [0.740, 0.898] | 0.869 [0.803, 0.926] |
| abrasion QWK／Sens.／Spec. | 0.410／0.513／0.849 | 0.493／0.795／0.679 |
| anthracnose QWK（5 張正樣本） | 0.865 | 0.911 |
| insect_track QWK | 0.595 | **0.919** |
| insect_track 正樣本面積 MAE（百分點） | 13.63 | **5.63** |
| insect_bite（3 張正樣本） | 0／3 | 0／3 |
| 參考：Mask mAP50／mAP50-95 | 0.304／0.151 | 0.357／0.186 |

中括號為 2000 次 bootstrap 的 95% CI。配對 bootstrap 的 v25 − v19 Macro QWK 差異 95% CI 為 **[0.061, 0.268]**，不跨 0。

## 結論

1. **嚴重度一致性有可信的改善**，主要來自 insect_track：面積估計誤差降到約 4 成，大面積蟲害不再被零碎分割而低估。
2. **代價是誤報增加**：任一瑕疵的 Specificity 從 0.886 降到 0.657，主要來自 abrasion 的背景誤檢。方向和 v20 class-aware sampling 一致。若直接部署，會有較多良品被送去降級判斷。
3. v25 在 valid 選到的 conf=0.05 是原搜尋範圍的下限；逐類分析見下節。
4. insect_bite 仍為 0 命中，**證據不足**，應補收樣本或合併類別。
5. 只有單一 seed。Experiment D 顯示 seed 會造成固定 threshold 行為的明顯差異，正式採用前需要至少 3 個 seed。

## 逐類 threshold 追加分析

評估輸出：`training/runs/eval/v19_vs_v25_per_class/`。每類各自取 valid QWK 最高的 conf；insect_bite 沿用全域值。

| 設定（test） | abrasion／anthracnose／insect_track conf | Macro QWK | 任一瑕疵 Sens.／Spec. |
| --- | --- | ---: | --- |
| v19 全域 | 0.30／0.30／0.30 | 0.623 | 0.754／0.886 |
| v19 逐類 | 0.35／0.50／0.05 | 0.726 | 0.825／0.829 |
| v25 全域 | 0.05／0.05／0.05 | 0.774 | 0.930／0.657 |
| v25 逐類 | 0.05／0.60／0.04 | 0.751 | 0.930／0.657 |

與 v19 全域的配對差異 95% CI：v19 逐類為 [0.026, 0.168]，v25 逐類為 [0.025, 0.230]。

1. **insect_track 需要低 conf**：v19 只把 insect_track 降到 0.05，Macro QWK 就從 0.623 升到 0.726。v25 相對 v19 的 insect_track 改善，有相當部分來自 threshold，而不是模型本身。
2. **abrasion 的 QWK 對 threshold 不敏感**：在 conf 0.05～0.40 間都約 0.4，threshold 實際上只在 Sensitivity 和 Specificity 之間取捨，因此以 QWK 選 threshold 不會提高 abrasion threshold。
3. v25 的 abrasion 在 conf 0.20 時，test Sens.／Spec. 為 0.49／0.89，和 v19 在 0.30 的 0.51／0.85 相當；在 0.15 時為 0.62／0.85，同樣的 Specificity 下 Sensitivity 較高。
4. anthracnose 在 valid 只有 4 張正樣本，逐類選出的 0.5～0.6 是過度貼合 valid，test 反而略降。

abrasion 的 threshold 應依「可接受多少良品被誤判」決定，例如要求 valid Specificity ≥ 0.80。這是產品決策，不是單一指標能選出的最佳值。

## 與部署模型 v22 比較

Django 部署的 `Defect.pt` 與 `v22_D_context6/weights/best.pt` SHA-256 相同。Django 未設定 defect confidence，實際使用 Ultralytics 預設的 0.25。評估輸出：`training/runs/eval/v22_vs_v25/`。

| 設定 | valid Macro QWK | test Macro QWK | test 任一瑕疵 Sens.／Spec. | test abrasion Sens.／Spec. |
| --- | ---: | ---: | --- | --- |
| v22 部署現況（全類 0.25） | 0.729 | 0.670 | 0.77／0.74 | 0.49／0.83 |
| v25 全類 0.15 | 0.735 | 0.751 | 0.84／0.77 | 0.62／0.85 |

- 兩者在 valid 幾乎相同。test 上 v25 各項都略好，但配對差異 95% CI 為 [−0.008, 0.185]，跨 0。
- 若各自用 valid 選出的全域 threshold（v22 0.10、v25 0.05）比較，CI 為 [0.030, 0.196]。但 v22 在 0.10 時 test Specificity 只有 0.49，並不是實際會部署的設定。
- 判定：**v25 沒有明確優於部署中的 v22，維持 v22。** 換模型還需要重新產生決策資料、重新訓練 XGBoost，而 v25 只有單一 seed，目前證據不足以支持這個成本。

## 下一步

- 決定 abrasion 的 Specificity 下限，並以 Sensitivity／Specificity 約束選 threshold。
- 用 seed 1、2 重跑 v25 設定，確認改善可重現。
- 以消融實驗拆開「類別平衡」與「增強參數」各自的貢獻。
- 決策資料集完成專家標註後，計算第 4 級端到端等級一致性。
