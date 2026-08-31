# v21 Experiment C：Rare-class Targeted Crop Dataset

- 建立日期：2026-08-31
- baseline：`v19_defect_epoch200_pw050_cp01`
- 預定 run：`v21_C_targeted_crop`
- 狀態：Dataset 已建立並驗證；尚未開始 200 epochs 訓練

## Controlled Experiment

Experiment C 使用 v19 的 433 張原始 train images，每張只列一次，不讀取 Experiment B
的 `train_B_class_aware.txt`。唯一新增訓練資料是由 train segmentation instances 產生的
targeted crops：`anthracnose` 每個 instance 1 張、`insect_bite` 每個 instance 3 張。
valid 93 張與 test 92 張沿用原始 split，沒有 crop、oversampling、augmentation、label
變更或重新 split。

## 產生參數

```text
seed=0
anthracnose_crops_per_instance=1
insect_bite_crops_per_instance=3
context_scale=4.0
context_scale_variation=0.5
center_jitter=0.10
minimum_crop_size=64
maximum_crop_fraction=0.90
min_polygon_area=1.0 pixel^2
```

每個 target bbox 的最大邊乘 sampled context scale 決定預期 square crop；影像邊界限制
時使用 near-square；各軸另限制在原圖 90% 以內，除非 target 本身更大。crop start 由
deterministic jitter 位移後，再限制於「完整包含 target polygon」的合法範圍。同一
`insect_bite` instance 的三個 crop rectangle 強制互異。實際 132 張 crop 的寬／高中位數
都是 100.5 pixels，範圍分別為 64–403 與 64–446 pixels；實際 context scale 中位數
4.064，範圍 1.833–5.565，較小值來自原圖 90% 上限。沒有 crop 退化成完整原圖；target
bbox pixel-area share 相對原圖至少放大 1.235 倍，中位數 18.305 倍。

所有原始 instance 使用 Sutherland-Hodgman rectangle polygon clipping，再 normalize
到 crop coordinates。target polygon 必須完整保留；crop 內其他 class 的有效交集也保留。
輸出不做 resize、HSV、blur 或 rotation，使用原始 crop resolution 的 PNG。

## Dataset 統計

| 項目 | v19 baseline | Experiment C |
| --- | ---: | ---: |
| train list entries | 433 | 565 |
| unique original train images | 433 | 433 |
| generated crop images | 0 | 132 |
| null exposure | 167 | 167 |

`effective exposure` 定義為實際 train list 所有 entries 中該 class 的 segmentation
instance 總數；Experiment C 沒有 repeated entry。

| Class | v19 images containing | C images containing | v19 instances/exposure | C instances/exposure | Experiment B exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| `abrasion` | 184 | 208 | 577 | 614 | 753 |
| `anthracnose` | 26 | 83 | 54 | 128 | 122 |
| `insect_bite` | 26 | 105 | 26 | 105 | 104 |
| `insect_track` | 123 | 150 | 386 | 430 | 497 |
| `null` | 167 | 167 | 167 | 167 | 167 |

實際產生 54 張 anthracnose crops 與 78 張 insect_bite crops。`anthracnose=128` 高於
理想的 108，是因為同一 crop 必須保留其中仍可見的其他 anthracnose instances；
`insect_bite=105` 比理想 104 多 1，也是其他 target crop 內可見 label 的合法共存。
這是避免錯誤 background supervision 的結果，不是 full-image repetition。

相較 Experiment B，C 的 rare-class exposure 非常接近或略高，但 `abrasion` 與
`insect_track` 的 incidental exposure 明顯較少；C 測試的是放大 rare defect 的有效
pixel scale，不要求複製 B 的 sampling distribution。

## Validation

| 檢查 | 結果 |
| --- | --- |
| train／valid／test exact image SHA-256 overlap | 三組 pair 都是 0 |
| 可可靠解析的 Fruit ID／View ID overlap | 三組 pair 都是 0 |
| generation 前後 source manifest | 0 mismatch |
| generated source 非 train | 0 |
| 原始 full image repetition | 0 |
| missing image／label | 0 |
| malformed segmentation label | 0 |
| class ID 超出 0–3 | 0 |
| normalized coordinate 超出 `[0,1]` | 0 |
| 少於 3 點或近零面積 polygon | 0 |
| target instance 未完整保留 | 0 |
| generated PNG 無法解析 | 0／132 |

只有檔名符合 `fruit_NNN_img_0V` 時才記錄 Fruit ID／View ID；132 張 crops 中 6 張可
可靠解析，126 張留空。原始 Dataset 中 train／valid／test 分別有 402／84／85 張通用
`img_0V_jpg.rf.*` 檔名，無法由檔名證明 fruit-group split；本次不猜測其 Fruit ID。
exact image hashes 與可解析 identities 沒有 overlap，但這個既有來源限制仍需保留。

Ultralytics 8.4.122 的 `check_det_dataset` 與 `YOLODataset` 已成功讀取 565 entries，掃描
結果為 565 images、167 backgrounds、0 corrupt。generated crop 可轉為
`(3, 1024, 1024)` tensor，且 segmentation mask 正常建立；沒有啟動模型訓練。

Experiment C 與 Experiment B 的 6 個 targeted／sampling tests 全部通過。repository
完整 suite 共執行 27 tests，其中 25 個通過；另有兩個本次修改前即可重現的既有問題：
`test_augment_rare_classes` 仍匯入已由 commit `1f995f3` 刪除的舊工具，以及 macOS
`/var`／`/private/var` realpath 差異造成一個 decision-dataset path assertion 失敗。
本次沒有擴大範圍修改這兩個既有問題。

## Reproducibility

| Artifact | SHA-256 |
| --- | --- |
| source train manifest | `93cc548941a474d83bdbd0f03f50bb2bb4714a2d3a44d5ea6fd70d90df5e5d56` |
| source valid manifest | `639ac68782195fab597ca40e8b9f1a4722f767d1892b58883077a155ed8e70bb` |
| source test manifest | `be82ea7043df3459c2961dd3fdd8927e7818144743eaa158f15e9a87dde7bb42` |
| `source_splits_manifest.csv` | `62bf967530434e7bf4bec681874ac53f294df0978989032dd920a296f42069ea` |
| `train_C_targeted_crop.txt` | `9209faf704ef5e3546da0e2704a80475ecca74507dbdac8733e2fcf2bb5d4a02` |
| crop `manifest.csv` | `2412f0654eb6feecbb2b5fefa410fe77a72cf952307f1175599f296ce51f50d6` |
| Experiment C training-input Dataset | `95a0955d0c1d2f869960a577d8020621afda413d360ee7f9a7668fbd485afd7a` |
| `yolo26s-seg.pt` | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` |

環境：macOS 26.6.2 arm64、Python 3.12.13、PyTorch 2.13.0、Ultralytics 8.4.122、
Pillow 12.2.0、PyYAML 6.0.3。相同 seed／參數執行 `--overwrite` 後，Dataset、train list
與 crop manifest hashes 都與第一次 generation 相同。本次 Codex sandbox 內
`torch.backends.mps.is_available()` 回傳 `False`；Dataset smoke test 不需要 GPU，正式
訓練仍依 controlled condition 使用 `device=mps`，執行前應在使用者的互動式 shell 再確認
MPS availability。

## Training Command

```bash
yolo task=segment mode=train \
  model=yolo26s-seg.pt \
  data=training/packages/local_defects/data_exp_C.yaml \
  project=training/runs/segment \
  name=v21_C_targeted_crop \
  imgsz=1024 \
  epochs=200 \
  patience=50 \
  batch=4 \
  device=mps \
  seed=0 \
  deterministic=True \
  cls_pw=0.5 \
  copy_paste=0.1 \
  copy_paste_mode=mixup
```

本次未執行上述 200 epochs training。
