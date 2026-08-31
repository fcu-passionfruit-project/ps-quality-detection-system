# Experiment C：Rare-class Targeted Crop 規格

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
