

# 注意：此文件只對應 `single_class_per_image_dataset` ，不得用來描述 `local_defects`

# 局部瑕疵 Roboflow 分開標註指南

## 這次要做什麼

使用同一批百香果影像建立 4 個 Roboflow Project，每個 Project 只標一種瑕疵：

| Project | 只標這個 class |
| --- | --- |
| 炭疽 | `anthracnose` |
| 擦傷 | `abrasion` |
| 畫圖蟲 | `insect_track` |
| 蟲咬 | `insect_bite` |

同一張照片可以出現在 4 個 Project。每個 Project 只處理自己負責的 class，看見其他瑕疵也不要標。

## 標註規則

1. 使用 Instance Segmentation polygon，不使用 bounding box。
2. 一張照片若有多個目標瑕疵，要全部標完。
3. 照片內沒有該 Project 的瑕疵時，確認看過後標記為 Null。
4. class 名稱必須使用表格中的英文名稱，不自行改名。
5. polygon 盡量貼合瑕疵邊緣，標註尺度與判定方式要一致。

例如在 `insect_bite` Project：

- 有蟲咬：標出所有蟲咬 polygon。
- 只有擦傷或炭疽：不要標其他瑕疵，確認沒有蟲咬後標記為 Null。

## Train／Valid／Test 規則

同一張原始照片在 4 個 Project 必須放在相同 split。

```text
fruit_141_img_01

炭疽 Project       → train
擦傷 Project       → train
畫圖蟲 Project     → train
蟲咬 Project       → train
```

不要讓同一張照片在某個 Project 屬於 train，到了另一個 Project 卻變成 valid 或 test。

判斷原始照片時，以 `fruit_xxx_img_xx` 為準，不看 Roboflow 匯出後的 `.rf.<hash>`。 -> `.rf.<hash>`是roboflow匯出照片時會添加的雜湊值。

## Roboflow Version 設定

- 不設定 Resize，保留原始解析度。
- 不使用 Roboflow preprocessing。
- 不使用 Roboflow augmentation。
- 匯出格式使用 YOLO Instance Segmentation。

## 匯出後如何合併

4 個 Dataset 會直接集合成一份 training Dataset：

```text
anthracnose Dataset
+ abrasion Dataset
+ insect_track Dataset
+ insect_bite Dataset
↓
class-isolated combined Dataset
```

合併時遵守以下規則：

- 保留每個 Project 匯出的 `.rf.<hash>` 影像與 label。
- 不刪除相同原圖的不同副本。
- 不把 4 個 Project 的 polygons 合併到同一個 label。
- 不手動修改 class ID，後續由合併程式統一 remap。

## 完成前檢查

- [ ] 每張照片都已檢查，沒有漏標負責的 class。
- [ ] 沒有目標瑕疵的照片已標記為 Null。
- [ ] 全部使用 polygon 標註。
- [ ] class 名稱完全正確。
- [ ] 4 個 Project 的 split 完全一致。
- [ ] Version 沒有 Resize、preprocessing 或 augmentation。
