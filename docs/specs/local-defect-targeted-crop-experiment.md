# Experiment C／D：Rare-class Targeted Crop 規格

## 實驗邊界

Experiment C 是 `v19_defect_epoch200_pw050_cp01` 原始 `local_defects` train split 加上
rare-class targeted crops。原始 train image 每張只列一次，不讀取
`train_B_class_aware.txt`，valid／test 的圖片、labels 與 split 必須逐位元不變。

class mapping 固定為：

```text
0 abrasion
1 anthracnose
2 insect_bite
3 insect_track
```

只有 class 1 與 class 2 可觸發 crop。預設每個原始 `anthracnose` instance 產生 1 張、
每個 `insect_bite` instance 產生 3 張；兩者都可由 CLI 調整。`abrasion`、
`insect_track` 與 null image 不可主動觸發 crop。

## Crop 與 label contract

1. 將原始 normalized segmentation polygon 轉回原始影像 pixel coordinates。
2. crop 邊長以 target bounding box 最大邊乘 context scale 決定，並套用可調的最小
   crop 尺寸與各軸最大原圖比例；遇到影像邊界時允許 near-square，但 target polygon
   必須完整包含。若 target 本身未橫跨整張原圖，輸出不得退化成完整原圖複本。
3. context scale variation 與 center jitter 使用固定 seed 的 deterministic PRNG。
4. 不進行 resize、HSV、blur、rotation 或其他影像 augmentation；crop 以原始解析度
   PNG 保存，訓練時由 `imgsz=1024` resize。
5. 所有原始 instance 都使用 Sutherland-Hodgman axis-aligned rectangle polygon
   clipping。有效交集保留原 class ID，重新 normalize 到 crop 的 `[0,1]`；少於三點
   或面積小於設定 threshold 的交集移除。
6. 不得只保存 target instance。crop 內仍可見的其他已標註 class 必須保留，避免把
   真實瑕疵變成 background supervision。

## 輸出與驗證

```text
training/packages/local_defects/
├── data_exp_C.yaml
└── experiments/
    ├── train_C_targeted_crop.txt
    └── exp_C_targeted_crop/
        ├── images/
        ├── labels/
        ├── manifest.csv
        ├── source_splits_manifest.csv
        ├── validation_report.json
        └── reproducibility.json
```

`manifest.csv` 必須保存 source／output path、target class／instance／crop index、原圖與
crop 尺寸及座標、實際 context scale、jitter、seed，以及檔名可可靠解析時的
Fruit ID／View ID。無法可靠解析時欄位留空，不得猜測。

validation 至少驗證：

- 原始 train／valid／test manifest 在 generation 前後相同。
- exact image SHA-256 與可可靠解析的 Fruit／View identity 沒有跨 split overlap。
- 每張原始 train image 在新 train list 恰好一次，且每張 generated crop 恰好一次。
- generated source 全部屬於 train；每張 image 都有 label，且沒有 orphan label。
- segmentation class ID、頂點數、面積與 normalized coordinate 合法。
- 每個 target 原始 polygon 完整保留在其 crop label。
- `data_exp_C.yaml` 的 valid、test、nc、names 與原始 `data.yaml` 相同。

`reproducibility.json` 必須保存 source split hashes、train list hash、crop manifest hash、
Experiment C training-input Dataset hash、`yolo26s-seg.pt` SHA-256、seed、完整 generator
參數與 Python／PyTorch／Ultralytics 版本。

## 固定訓練條件

除 train targeted crops 外，固定：`model=yolo26s-seg.pt`、`imgsz=1024`、
`epochs=200`、`patience=50`、`batch=4`、`device=mps`、`seed=0`、
`deterministic=True`、`cls_pw=0.5`、`copy_paste=0.1`、
`copy_paste_mode=mixup`。run 必須寫入 `training/runs/segment/v21_C_targeted_crop`。

## Experiment D：Paired Context Scale

Experiment D 使用 Experiment C 現存 artifacts 作為 pairing ground truth。原始 train
images、target instance sequence、target repetition count、valid／test、label policy 與
training settings 全部不變；唯一主動修改的 Dataset generation 參數是 nominal
`context_scale=4.0` 改為 `6.0`。run 名稱固定為 `v22_D_context6`，Dataset 輸出不得覆寫
Experiment C。

Experiment D 固定的 target contract 是：

```text
crop target records = 132
unique target instances = 80
anthracnose = 54 unique targets / 54 target crop records
insect_bite = 26 unique targets / 78 target crop records
```

`target crop records` 與 `total label exposure` 是不同指標。較大 field-of-view 自然納入
的 neighboring annotations 必須依既有 clipping 規則完整保留；不得刪除合法 labels、
只保留 target、修改 target selection 或調整 crop count 以強制維持 Experiment C 的
total exposure。

### Stochastic replay 與 pairing

產生 D 前必須以 C 的 generator 參數與 global `seed=0` deterministic replay 現存 C，並
逐筆驗證 132/132 target sequence、crop rectangle、image pixels／geometry 與 label bytes。
任何一項無法完整重現時停止，不得建立正式 D。

每筆 D record 的第一個 attempt 必須重用對應 C record 的：

- sampled context variation delta；
- jitter X／Y 的 normalized underlying realization。

D 不複製 C coordinates；它以 `6.0 + variation_delta` 和相同 normalized jitter，重新走
既有 crop geometry、boundary handling 與 polygon clipping。若第一個 D rectangle 與同一
target 先前 repeat 重複，retry stream 必須由 global seed、stable target record ID 與 retry
index 導出，避免改變後續 records。retry 必須記錄 record、原因及最終 pairing 差異。

### Experiment D artifacts 與 validation

```text
training/packages/local_defects/
├── data_exp_D.yaml
└── experiments/
    ├── train_D_context6.txt
    └── exp_D_context6/
        ├── images/
        ├── labels/
        ├── manifest.csv
        ├── paired_parameters.csv
        ├── experiment_c_replay_report.json
        ├── experiment_c_vs_d_audit.json
        ├── source_splits_manifest.csv
        ├── validation_report.json
        └── reproducibility.json
```

validation 必須分開報告 target records、unique targets、generated-crop total label
instances、neighboring instances 及整體 train exposure。另須報告 paired scale／jitter
數量、retry、boundary／clamp、duplicate source rectangle 與 exact duplicate generated
image。必要 Dataset 結構為 433 original train images、132 crops、565 train entries；
valid／test 與 Experiment C 的 source hashes 必須完全相同。
