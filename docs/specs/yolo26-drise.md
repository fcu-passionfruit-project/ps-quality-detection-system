# YOLO26 Segmentation 單一 Instance D-RISE 工具規格

## 解釋目標與邊界

`tools/yolo26_drise.py` 回答「輸入影像哪些 pixels 對原始推論中指定的單一
segmentation instance 有較大貢獻」。它是以隨機遮蔽做 black-box perturbation 的
D-RISE，不是 feature activation、EigenCAM 或 Grad-CAM。

Microsoft `vision-explanation-methods` 是演算法依據。工具直接呼叫其 `generate_mask()`
與 `fuse_mask()`。為避免在 `num_masks=1000`、`imgsz=1024` 時保存所有三通道 full-size
masks，saliency 使用與 Microsoft `saliency_fusion(normalize=True)` 完全相同的 streaming
等式：

```text
raw = sum(similarity_i * mask_i) - mean(similarity) * sum(mask_i)
```

`drise_raw.npy` 保存上式的 normalize 前值；PNG 才進行 min-max normalization。

## YOLO Segmentation Extension

Microsoft `DetectionRecord` 與原始 affinity 只包含 bounding boxes、objectness 與 class
scores。本工具不修改第三方 package，而以獨立 `compute_target_similarity()` 擴充單一
instance segmentation matching：

```text
similarity = class_match * mask_iou(reference_mask, candidate_mask)
             * candidate_confidence
```

每張 masked image 只考慮同 class candidates，先取 Mask IoU 最大者，再以 confidence
打破 IoU 平手。不能由另一個空間上不同、但 confidence 較高的同類 instance 取代 target。

## 座標契約

原始與 masked inference 都使用 `retina_masks=True`。每次 result 必須滿足
`result.orig_shape == source.shape[:2]`。所有 mask 經同一函式轉成原始影像 `(H,W)` 的
boolean mask；若 Ultralytics 回傳較低解析度 mask，只使用 nearest-neighbor resize。
`mask_iou()` 仍會 assertion 兩邊都是相同 shape 的 2D masks，任何 mismatch 都停止。

## Target 與輸出

`--target-index` 直接對應原始 `Results.boxes` / `Results.masks` 的 instance 順序。
`--target-class` 只在該 class 恰有一個 instance 時可單獨使用；多個時必須指定 index。
每次執行在 `--output` 下建立 timestamped 目錄，包含：

- `prediction.jpg`
- `target_instance.jpg`
- `target_mask.png`
- `drise_raw.npy`
- `drise_heatmap.png`
- `drise_overlay.png`
- `metadata.json`

`--save-debug-masks N` 另保存前 N 組 random mask、masked input、masked prediction 與
target similarity JSON；JSON 逐一記錄所有 candidates 的 class、confidence、Mask IoU 與
similarity，供人工驗證 target identity。JET 固定藍色為低貢獻、紅色為高貢獻。

## Device 與重現性

`--device auto` 依 CUDA、MPS、CPU 順序選擇；明確指定不可用的 MPS/CUDA 會報錯，不會
silent fallback。YOLO inference 使用 resolved device；Microsoft random masks 固定在 CPU
生成，使相同 PyTorch seed 的遮罩序列不受 accelerator RNG 影響。metadata 記錄 requested
與 resolved device、所有 D-RISE 參數及套件版本。
