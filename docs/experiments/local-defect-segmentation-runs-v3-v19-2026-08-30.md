# YOLO26s-seg 局部瑕疵訓練結果分析（v3–v19）

- 分析日期：2026-08-30
- 任務：單一 `YOLO26s-seg` 辨識 `abrasion`、`anthracnose`、`insect_bite`、`insect_track`
- 分析範圍：`training/runs/segment/` 現存的 `args.yaml`、`results.csv`、曲線、confusion matrix、`labels.jpg`、`val_batch*_labels.jpg` 與 `val_batch*_pred.jpg`
- 限制：本次只讀取既有 artifact，沒有重新 validation、重新訓練或修改任何 Dataset／權重。所有 run 的 `args.yaml` 都是 `split=val`，因此本報告沒有獨立 test 結果。

## Current Findings

### 結論摘要

目前表現最好的完整四類 run 是 `v19_defect_epoch200_pw050_cp01`。其 `results.csv` 在 epoch 186 達到：

| 指標 | Box | Mask |
| --- | ---: | ---: |
| Precision | 0.691 | 0.706 |
| Recall | 0.319 | 0.324 |
| mAP50 | 0.366 | 0.367 |
| mAP50-95 | 0.216 | 0.210 |

核心判斷如下：

1. 模型不是單純 underfitting。訓練 loss 持續下降，100 增至 200 epochs 也提高了最佳 mAP；模型確實學到可用特徵。
2. 約 epoch 90–140 後出現 validation localization／segmentation loss 回升，而 train loss 繼續下降，已有 overfitting 跡象。小型 validation set 的 mAP 波動很大，使最佳 mAP 晚於 validation loss 最低點。
3. 主要瓶頸是 Recall，而不是 Precision。最佳 run 的 Mask Precision 0.706、Recall 0.324；Mask F1 的全類最佳值也只有約 0.36（confidence 約 0.353）。即使 confidence 接近 0，Mask Recall 曲線的 macro 值最高也只有約 0.69，表示部分 ground truth 根本沒有被正確提案，不能只靠降低 threshold 解決。
4. Box 與 Mask 沒有巨大落差：最佳 run 的 Box／Mask mAP50 幾乎相同，mAP50-95 只差 0.005。現階段主要問題先發生在找不到目標、類別辨識及 instance 配對，而不是 mask head 單獨崩壞。
5. `anthracnose` 的曲線最好，但 valid 只有 10 個 instances／4 張含該類圖片，數值不穩定；`insect_bite` 最差，valid 只有 1 個 instance，且被判成 `abrasion`。
6. 最佳 run 的 box confusion matrix 有 134 個 unmatched ground truths（false negatives）與 69 個 unmatched predictions（false positives）。兩者都多，但 false negative 更嚴重。
7. validation prediction 圖反覆出現三種典型錯誤：把正常果皮斑點／反光判成 `abrasion` 或 `insect_track`、只覆蓋大型 `insect_track` 的零碎片段、漏掉極小標註。
8. Dataset 比模型容量更像目前的首要瓶頸：類別極度不平衡、rare-class validation 幾乎不可用、來源 ROI 僅約 400 px、部分類別邊界視覺上模糊、已知 Fruit ID 跨 split，而且 class-isolated Dataset 會對同一張圖產生互相矛盾的缺標監督。

### Run 比較：相同 `local_defects` Dataset

下表的 epoch 與指標取每個 `results.csv` 中 Mask mAP50-95 最高的 row，而非最後一個 epoch。`v10` 的名稱雖包含 `detect`，其 `args.yaml` 仍是 `task=segment` 且使用 `yolo26s-seg.pt`。

| run | epochs | `cls_pw` | `copy_paste` | best epoch | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | final Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v7_defect_pw000_no_cp` | 100 | 0.00 | 0.0 | 95 | 0.368 | 0.390 | 0.328 | 0.157 | 0.144 |
| `v8_defect_pw025_no_cp` | 100 | 0.25 | 0.0 | 74 | 0.516 | 0.336 | 0.291 | 0.165 | 0.156 |
| `v9_defect_pw050_no_cp` | 100 | 0.50 | 0.0 | 86 | 0.652 | 0.352 | 0.332 | 0.182 | 0.138 |
| `v10_detect_pw075_no_cp` | 100 | 0.75 | 0.0 | 92 | 0.273 | 0.367 | 0.301 | 0.153 | 0.141 |
| `v16_defect_pw100_no_cp` | 100 | 1.00 | 0.0 | 74 | 0.260 | 0.371 | 0.294 | 0.151 | 0.125 |
| `v17_defect_pw000_cp01` | 100 | 0.00 | 0.1 | 98 | 0.347 | 0.352 | 0.337 | 0.176 | 0.167 |
| `v18_defect_pw050_cp01` | 100 | 0.50 | 0.1 | 95 | 0.375 | 0.379 | 0.348 | 0.186 | 0.173 |
| `v18_defect_pw050_cp03` | 100 | 0.50 | 0.3 | 86 | 0.360 | 0.259 | 0.301 | 0.152 | 0.130 |
| `v19_defect_epoch200_pw050_cp01` | 200 | 0.50 | 0.1 | 186 | 0.706 | 0.324 | 0.367 | **0.210** | 0.171 |
| `v19_defect_epoch200_pw050_cp03` | 196 | 0.50 | 0.3 | 146 | 0.408 | 0.366 | 0.356 | 0.202 | 0.178 |

可支持的參數結論：

- 在 100-epoch、無 Copy-Paste 的 `cls_pw` sweep 中，`cls_pw=0.5` 最好；0.75 與 1.0 都退步，不值得再往更高權重搜尋。
- `copy_paste=0.1` 在 100 epochs 只比不使用 Copy-Paste 高 0.004 Mask mAP50-95，差距小到可能落在單次訓練波動內；`copy_paste=0.3` 則明顯較差。
- 200 epochs 的兩組 run 都比各自 100-epoch 對應 run 的最佳值高，表示 100 epochs 尚未把最佳 checkpoint 搜尋完；但 200 epochs 後段的 validation loss 已惡化，因此應保存／使用 `best.pt`，不可用 `last.pt` 代替。
- `copy_paste=0.1` 的 200-epoch run 仍優於 0.3；目前 incumbent 可保留 0.1，但尚缺一組 200-epoch、`copy_paste=0.0` 的嚴格對照。

### 其他 run 的可比性

- `v3_defect_pw025_no_cp` 使用目前已不存在的 `v2_defect_argument` 路徑，最佳 Mask mAP50-95 為 0.196；它不能和現存 `local_defects` run 視為只差單一參數。
- `v4`、`v5`、`v11`–`v14` 使用 `single_class_per_image_dataset`。其多數最佳 Mask mAP50-95 只有 0.118–0.138；`v4` 例外達 0.182。
- `v4` 與 `v12` 的核心 args 相同，卻分別得到 0.182 與 0.118，說明只記 `seed=0`、`deterministic=True` 仍不足以保證跨時間／環境可重現。Dataset 內容 hash、Ultralytics／PyTorch 版本、初始權重 hash、device 與完整 command 都必須一起固定。
- `v3_defect_pw025_no_cp_batch8` 只跑 3 epochs，是 smoke test，不能判斷 batch 4 與 8 的優劣。
- `v15_defect_SinglePerImg_pw100_no_cp` 沒有 `results.csv`／weights，是未完成 run。
- `v6_defect_abrasion` 是單類 Dataset，不是完整四類模型，不能用來選正式模型。

## Per-class Problems

`v19_defect_epoch200_pw050_cp01` 的 best-checkpoint PR 曲線提供下列逐類 mAP50。`results.csv` 沒有逐類 mAP50-95，因此本次不猜測該值。

| 類別 | Box AP50 | Mask AP50 | box confusion matrix | 判斷 |
| --- | ---: | ---: | --- | --- |
| `abrasion` | 0.273 | 0.282 | 97 個 GT 中 28 正確、68 unmatched、1 判成 `insect_track` | 中低；大量漏檢，也容易把正常紋理當擦傷 |
| `anthracnose` | **0.813** | **0.828** | 10 個 GT 中 8 正確、2 unmatched | 表面最好，但樣本太少，可信區間會很寬 |
| `insect_bite` | 0.033 | 0.033 | 唯一 GT 被判成 `abrasion`，正確數為 0 | 明確最差；目前 validation 無法驗收 |
| `insect_track` | 0.347 | 0.325 | 104 個 GT 中 35 正確、64 unmatched，另有 5 個判成其他類 | 次佳但仍差；大型、細長與不規則區域常只抓到片段 |

類別混淆並非平均分散：

- 最明確的是 `insect_bite → abrasion`。唯一 validation `insect_bite` 是非常小的斑點，且同一張圖有明顯擦傷，視覺與標註語意都不利於模型學到獨立特徵。
- `insect_track → abrasion` 有 4 次，`insect_track → anthracnose` 有 1 次；`abrasion → insect_track` 有 1 次。
- `anthracnose` 和其他類的直接混淆較少，但只有 10 個 GT，不能因此斷言類別已解決。
- confusion matrix 是以 bounding-box IoU 配對，不直接表示 mask polygon 的重疊品質；mask 品質仍以 Mask PR／mAP 判斷。

## Training Curve Analysis

### Underfitting／overfitting

`v19_defect_epoch200_pw050_cp01`：

| 曲線 | train | validation |
| --- | --- | --- |
| box loss | 1.731 → 1.086 | 最低 1.648（epoch 92），結束 1.852 |
| seg loss | 3.132 → 1.435 | 最低 2.265（epoch 98），結束 2.546 |
| cls loss | 27.744 → 0.562 | 最低 1.098（epoch 142），結束 1.159 |

這不是「模型連 training data 都學不動」的 underfitting。相反地，train loss 大幅下降，而 validation box／seg loss 約從 epoch 90–100 後轉為上升，是典型的泛化落差。Mask mAP50-95 仍在 epoch 186 出現單點最高值 0.210，原因較可能是 validation 類別數少、每個 epoch 指標方差大，而不是 validation loss 已重新改善。

`v19_defect_epoch200_pw050_cp03` 更明顯：validation seg loss 在 epoch 67 最低，Mask mAP50-95 在 epoch 146 最佳，之後 50 epochs 沒有刷新 early-stopping fitness，最終於 epoch 196 停止。這組結果支持「後段 plateau + overfitting」，不支持再把 epochs 盲目提高到 300。

100→200 epochs 的效果仍有價值：

- `copy_paste=0.1`：最佳 Mask mAP50-95 由 0.186 提升到 0.210（+0.024）。
- `copy_paste=0.3`：由 0.152 提升到 0.202（+0.050）。

因此下一輪可保留 `epochs=200`、`patience=50` 作為搜尋上限，但選模型必須依 `best.pt` 與逐類結果；增加 epochs 不是主要改善手段。

### Precision／Recall 與 threshold

- 最佳 run 的 Precision 明顯高於 Recall（Mask 0.706 vs 0.324），主要問題是漏檢。
- Mask F1 curve 的 macro 最佳約 0.36，confidence 約 0.353；Box F1 約 0.35，confidence 約 0.349。
- 低 threshold 會增加正常果皮上的 false positive，且 Recall curve 顯示仍有一批目標在任何 threshold 都無法正確召回。
- 後續可以校正 class-specific threshold，但它只能在 FP／FN 間取捨，不能代替 Dataset 與 proposal 能力改善。

## Prediction Error Analysis

從 `v19_defect_epoch200_pw050_cp01/val_batch*_labels.jpg` 與 `val_batch*_pred.jpg` 可觀察到：

1. **正常紋理 false positive**：無標註或近似乾淨的果實上，模型把皮孔、細小亮點、色差及反光判成多個 `abrasion`；也會把成片紋理判成 `insect_track`。
2. **大型 mask 碎片化**：ground truth 的 `insect_track` 常是大片、不規則或細長區域，模型只預測其中幾個局部，造成 box／mask IoU 不足與 defect area ratio 低估。
3. **邊緣誤判**：靠近圓形遮罩邊界、果皮亮邊或黑色背景交界的位置容易出現 `insect_track`／`abrasion` false positive。
4. **微小瑕疵漏檢**：許多 `abrasion` 標註只是數個像素等級的小點；模型常完全漏掉，或在附近另一個視覺相似點產生不匹配的預測。
5. **類別語意混淆**：validation 唯一 `insect_bite` 的 normalized bounding-box area 只有 0.00040，位於同時含擦傷的影像中；最佳模型將其判成 `abrasion`。
6. **instance 切分不一致**：同一片 `insect_track` 在 ground truth 中可能切成多個 polygon／instance，預測也會用不同方式切碎。下游實際需要同類 mask union area，單純 instance mAP 會額外懲罰切分方式差異。

最佳 run 的 raw box confusion matrix：

- true `abrasion` 97：28 TP、68 background FN、1 判成 `insect_track`。
- true `anthracnose` 10：8 TP、2 background FN。
- true `insect_bite` 1：0 TP、1 判成 `abrasion`。
- true `insect_track` 104：35 TP、64 background FN、4 判成 `abrasion`、1 判成 `anthracnose`。
- background false positives 69：31 個 `abrasion`、6 個 `anthracnose`、32 個 `insect_track`。

`copy_paste=0.3` 的 200-epoch run 並未消除這些問題：它有 136 個 background FN、63 個 background FP，且 44／63 個 background FP 是 `insect_track`。提高 Copy-Paste 強度主要改變錯誤分布，沒有帶來更好的整體 Mask mAP50-95。

## Dataset-related Issues

### 類別分布

現存 `training/packages/local_defects` 的直接 label 統計如下：

| split | images | null images | `abrasion` | `anthracnose` | `insect_bite` | `insect_track` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 433 | 167 | 577 | 54 | 26 | 386 |
| valid | 93 | 29 | 97 | 10 | **1** | 104 |
| test | 92 | 35 | 98 | 22 | **3** | 100 |

train 以 `insect_bite=1` 正規化後，instance 比例約為 `22.2 : 2.1 : 1 : 14.8`。`anthracnose` 的漂亮 AP 只來自 4 張 validation 圖；`insect_bite` 的 valid／test 指標分別由 1／3 個 instance 決定。這不足以支援逐類驗收或可靠 hyperparameter selection。

### 目標大小與來源解析度

train polygon 的 normalized bounding-box area 中位數：

| 類別 | median bbox area | p10 | p90 |
| --- | ---: | ---: | ---: |
| `abrasion` | 0.00308 | 0.00059 | 0.02106 |
| `anthracnose` | 0.00324 | 0.00087 | 0.03604 |
| `insect_bite` | 0.00241 | 0.00085 | 0.01042 |
| `insect_track` | 0.04212 | 0.00272 | 0.36285 |

目前 ROI 檔案的尺寸中位數約為 train `404×423`、valid `407×420`；train 範圍約 301–526 px 寬。`imgsz=1024` 已是在放大這些 ROI，並沒有創造新的感測細節。唯一 validation `insect_bite` 的 bbox area 0.00040，在約 400 px 的來源上只有約 8 px 等級的方形範圍。未先提高來源 ROI 細節時，改成 `imgsz=1280` 的優先度很低。

### 標註一致性

- label 檔數與 image 數一致，解析到的 label line 沒有格式錯誤。
- 小型類別的 polygon 點數中位數只有約 10–12，且 validation 圖中可見部分 `abrasion` 是很小、與正常皮孔相似的點。類別定義與最小標註尺寸若不一致，模型會同時受到漏標與錯標影響。
- `insect_track` 的 bbox 面積跨越很大（p10 0.00272、p90 0.36285），且同一病灶可能被分成多個 instance。應先統一連通區、相鄰區與大片表面損傷的 polygon 切分規則。
- 167／433 張 train 圖為 null。negative image 本身有價值，但 validation prediction 顯示正常紋理會產生大量 FP；必須人工複核 null 圖是否真的沒有四類瑕疵，並把高分 FP 作為 hard negatives。

### `single_class_per_image_dataset` 的矛盾監督

此 Dataset 把同一原圖依類別複製，每份只保留一類標註。統計結果：

- train 526 張中有 79 組 exact duplicate，額外重複 93 份，最高同圖 4 份。
- valid 有 17 組 duplicate，test 有 17 組。
- 同一張圖在其中一份標 `abrasion`，另一份只標 `insect_track` 時，未標出的可見瑕疵會被當成 background，形成互相矛盾的 supervision。

這可以解釋為何多數 `single_class_per_image_dataset` run 低於合併多類標註的 `local_defects` run。後續不應再以 class-isolated copies 作為主訓練 Dataset；應在同一 image record 合併所有可見類別的 polygon。

### Split 與資料洩漏風險

- 以 exact file SHA-256 與 Roboflow ID 檢查，`local_defects` 沒有完全相同圖片跨 train／valid／test。
- 但只有少數檔名保留 `fruit_xxx`：train 433 張中 402 張無法由檔名還原 Fruit ID，valid 93 張中 84 張無法還原，test 92 張中 85 張無法還原。
- 在仍可辨識的檔名中，已發現 `fruit_122` 跨 train／valid，`fruit_135` 與 `fruit_189` 跨 train／test；它們是同一 Fruit 的不同 View Observation。

因此目前 split 不是可證明的 fruit-group split，validation／test 可能看到訓練 Fruit 的其他視角。下一個 Dataset Version 應從來源 manifest 重建 stable Fruit ID，並以 Fruit 分組後再固定 split。

## Recommended Experiments

### 先建立可解讀的 Dataset gate

下一輪正式參數實驗前，先產生一個新的、固定不再改動的 Dataset Version：

1. 補充真實 `insect_bite`，並增加 `anthracnose` 的獨立 Fruit 數；valid／test 每類都要有足以解讀的多個 Fruit，而不是靠 augmentation 增加 evaluation 樣本。
2. 由兩位標註者複核 `insect_bite` vs `abrasion`、正常皮孔／反光 vs `abrasion`、`insect_track` instance 切分及 null images。
3. 同一張圖合併四類 polygon，不建立互相缺標的 class-isolated duplicates。
4. 以 Fruit ID group split，保存 source manifest、split 清單與 Dataset hash。
5. 從原始解析度照片裁 ROI；若來源確實只有約 400 px，先改善 source crop，而不是只提高訓練 `imgsz`。

### 建立可重現 incumbent

資料 gate 通過後，先重跑目前最佳設定作為新 baseline：

```text
model=yolo26s-seg.pt
imgsz=1024
epochs=200
patience=50
batch=4
cls_pw=0.5
copy_paste=0.1
copy_paste_mode=mixup
seed=0,1,2（三個獨立 run）
deterministic=True
```

固定同一台 device、Ultralytics／PyTorch 版本、初始權重 SHA-256、Dataset hash 與其餘 args。每組至少報告三個 seed 的 mean／standard deviation；否則 0.004 等級的差異沒有足夠證據。

除標準 Box／Mask 指標外，增加與下游目標一致的 per-class mask-union IoU、Defect Area Ratio MAE／bias，並對每個 Fruit 的三視角分組報告。instance 切分方式不同時，union metric 比單純 instance mAP 更貼近本專案用途。

### 受控 ablation

每次只改一個變因，建議依序執行：

1. **Copy-Paste**：在 200 epochs、`cls_pw=0.5` 下比較 `copy_paste=0.0` vs `0.1`。現有結果已足以淘汰 0.3，但還不能證明 0.1 穩定勝過 0.0。
2. **真實資料 + hard negatives**：保留參數不變，只加入人工確認的漏標修正、正常皮孔／反光／遮罩邊緣 hard negatives，觀察 background FP 與 Recall 是否同時改善。
3. **rare-class targeted augmentation**：只擴增 train 中經複核的 `anthracnose`／`insect_bite`，valid／test 不變；比較無離線擴增 vs 固定倍數擴增。不要和 Copy-Paste、class weight 同時改。
4. **class weighting**：在新版、較平衡 Dataset 上比較 `cls_pw=0.25` vs `0.5`。現有 sweep 不支持 0.75／1.0，先不再測。
5. **mask resolution**：若 detection Recall 改善後仍出現 mask 邊界／面積低估，再比較 `mask_ratio=4` vs `2`。目前 Box／Mask 差距只有約 0.005，優先度低於資料與 detection recall。
6. **來源解析度／imgsz**：只有在高解析 ROI 真正保留更多像素後，才比較 `imgsz=1024` vs `1280`；用現有約 400 px ROI 做此實驗大多只是在測插值與算力。
7. **模型容量**：最後才比較 `YOLO26s-seg` vs `YOLO26m-seg`。目前已有 overfitting 和資料不足，先加大模型較可能記憶資料而非解決 rare class。

### 不需要優先重做的方向

- 不再測 `cls_pw>=0.75`；現有完整 sweep 已一致退步。
- 不再測 `copy_paste=0.3`；100 與 200 epochs 都輸給 0.1。
- 不直接增加到 300 epochs；200-epoch 曲線後段已出現 validation loss 上升。
- 不把 batch 8 的 3-epoch smoke test 當成 batch-size 結論。
- 不只依 macro mAP 選模型；它會被 `anthracnose` 的少量高 AP 掩蓋，且 `insect_bite` 幾乎無法評估。

## Experiment Priority

由「最值得優先實驗」到「最不優先」排列：

| 優先級 | 實驗／處置 | 理由 | 成功判準 |
| ---: | --- | --- | --- |
| 1 | Dataset gate：rare class 真實資料、標註複核、Fruit-group split、高解析 ROI | 直接處理不可驗收的 `insect_bite`、潛在 leakage、矛盾標註與像素不足 | 每類 valid／test 有多個獨立 Fruit；split 可追溯；雙人 audit 通過 |
| 2 | 三 seed 重跑 incumbent（200 ep、`cls_pw=0.5`、CP 0.1） | 先建立可信 variance，避免把單次波動當改善 | 報告 mean±std、逐類指標、union IoU 與 area-ratio error |
| 3 | Hard-negative／漏標修正 ablation | prediction 圖顯示皮孔、反光、遮罩邊緣造成大量 FP，同時存在大量 FN | background FP 與主要類別 FN 同時下降，而不是只交換 P/R |
| 4 | 200-epoch Copy-Paste 0.0 vs 0.1 | 現有 0.1 優勢只有 0.004；0.3 已可淘汰 | 多 seed 下 0.1 的逐類與 macro 指標穩定勝出 |
| 5 | Rare-class targeted augmentation | 可提高 train exposure，但前提是 valid／test 已可評估 | `insect_bite`／`anthracnose` Recall 與 AP 提升，`abrasion`／`insect_track` 不退步 |
| 6 | `cls_pw=0.25` vs `0.5` | 0.5 是現有 sweep 最佳；新版分布改變後才值得重估 | macro 與四類最低表現同步改善 |
| 7 | `mask_ratio=4` vs `2` | 只針對 mask 碎片與小目標；目前 Box／Mask 差距不大 | Mask mAP50-95、union IoU、area MAE 改善且 Box 不變 |
| 8 | 高解析 ROI 後 `imgsz=1024` vs `1280` | 現有 400 px ROI 上調 imgsz 不會新增細節 | 小型類 Recall 提升足以抵銷計算成本 |
| 9 | class-specific threshold calibration | 可調整部署 P/R，但不能創造缺失 proposal | 在固定 validation 上達到業務要求的 per-class P/R |
| 10 | `YOLO26m-seg` 或更大模型 | 資料受限與 overfitting 尚未解決，容量不是首要瓶頸 | 多 seed、獨立 test 與 area metric 都顯著勝過 `s` |

目前不建議核准任何 run 為正式四類 defect 模型。`v19_defect_epoch200_pw050_cp01` 可作為下一版資料上的 incumbent 設定，但 `insect_bite` Recall、background error、split 可追溯性與獨立 test 驗收完成前，只應維持 candidate／實驗狀態。
