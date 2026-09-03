# YOLO26 Segmentation EigenCAM 測試工具規格

## 用途與解讀邊界

`tools/yolo26_cam_test.py` 用於觀察自訓 YOLO26 segmentation 模型某個 target layer
對單張輸入影像產生的 EigenCAM spatial activation／saliency response。EigenCAM 以
activation 的第一主成分產生空間圖，不是 transformer attention，也不是 class-specific
Grad-CAM；輸出不得被描述成模型判斷某一瑕疵 class 時專門查看的位置。

## 輸入契約

- `--model` 必須指向 Ultralytics 可載入且 `task=segment` 的 `.pt`。
- `--image` 是 OpenCV 可讀取的單張圖片。
- `--imgsz` 預設 `1024`，必須大於零。
- `--device` 只接受 `cpu` 或 `mps`，預設 `cpu`。MPS 不可用或執行失敗時不得自動
  回退 CPU。
- `--layer-index` 是 `model.model.model` 的 Python index，預設 `-3`；執行時必須印出
  原 index、解析後 index、layer 數量與完整 type。
- `--conf` 預設 `0.25`，必須介於 `0` 到 `1`，控制正常 YOLO prediction 的
  postprocessing；它不會使 class-agnostic EigenCAM 變成 class-specific。

第三方 package 必須直接從相對於工具檔案的 `tools/YOLO-26-CAM/yolo_cam` 載入；為相容
GitHub ZIP 的預設目錄名稱，也接受 `tools/YOLO-26-CAM-main/yolo_cam`。不得依賴目前
working directory，也不得複製或修改第三方原始碼。

## Preprocessing 與色彩契約

OpenCV 以 BGR `uint8` 讀取原圖，並把該陣列直接交給 Ultralytics。Ultralytics 的目前
實作負責 letterbox、BGR 轉 RGB、BHWC 轉 BCHW、移至 device、轉 float 及除以 `255`。
工具不得先把原圖強制拉伸成正方形。EigenCAM 透過 YOLO wrapper 推論，因此必須把
指定的 `imgsz` 與 `device` 寫入 model overrides；`conf` 明確傳給另一次正常 YOLO
prediction。

第三方 `show_cam_on_image()` 的 base image 與 colormap 必須使用同一 channel order。
工具全程以 BGR 建立與寫出 OpenCV artifacts，因此呼叫時固定 `use_rgb=False`。
這與 RGB base image 搭配 `use_rgb=True` 後再轉回 BGR 寫檔等價。不得使用 `1-cam` 或
reversed colormap：`cam=0` 在灰階是黑色、JET 是藍色；`cam=1` 在灰階是白色、JET
是紅色。

## Signed SVD polarity 診斷

YOLO-26-CAM 對 centered activation 執行 SVD，取 `VT[0]` 投影後再做 ReLU。Singular
vector 存在數學上的正負號不確定性，因此工具必須從同一次 target-layer hook 取得
activation，重算 ReLU 與 normalization 前的 signed projection。第三方正式 CAM 不得
反相或替換；診斷資料分別定義為：

```python
positive = np.maximum(raw_cam, 0)
negative = np.maximum(-raw_cam, 0)
absolute = np.abs(raw_cam)
```

三者各自 normalize 並 resize 到原圖尺寸。Negative polarity 只能稱為 diagnostic，不能
宣稱比第三方選定的 positive polarity 正確。

## Occlusion sanity check

正常推論以最高 confidence segmentation instance 作為 primary prediction。工具精確選取
normalized CAM 最高與最低各 10% pixels，固定填入中性灰 BGR `(114,114,114)` 後重新
推論。遮蔽後以相同 class 且與原 primary mask IoU 最大的 instance 比較 confidence；若
無匹配 prediction，confidence 記為 `0`。這是單張圖片的 faithfulness sanity check，不能
作為正式模型結論。若原圖沒有 prediction，仍輸出遮蔽圖片，但明確標記 confidence
comparison skipped。

## 輸出契約

成功執行時，核心輸出包含：

1. `original.jpg`：OpenCV 讀取的原始影像。
2. `cam_grayscale.png`：單通道 EigenCAM，固定黑 `=0=low`、白 `=1=high`。
3. `cam_heatmap.png`：對同一 CAM 套用 OpenCV JET，附 activation colorbar。
4. `cam_overlay.png`：CAM heatmap 與原圖疊加，附 activation colorbar。
5. `prediction.jpg`：Ultralytics `Results.plot()` 的 segmentation mask、class 與 confidence。
6. `comparison.png`：依序排列 Original、YOLO Prediction、EigenCAM Heatmap、CAM
   Overlay，並標示 model filename、target layer index、`imgsz`、colorbar 與非
   class-specific 提醒。

Audit artifacts 包含：

- `cam_normalized.npy`、`cam_signed_raw.npy`：正式 normalized CAM 與 native-resolution
  signed SVD projection。
- `cam_positive.png`、`cam_negative.png`、`cam_absolute.png`：三種 signed polarity
  diagnostic。
- `cam_gray.png`、`cam_jet.png`、`cam_turbo.png`：同一 normalized CAM 的三種 colormap。
- `cam_argmax_location.png`：在原圖標記正式 CAM argmax。
- `primary_prediction_mask.png`：occlusion check 使用的原始 primary mask；沒有 prediction
  時是全黑。
- `occlusion_high_cam.jpg`、`occlusion_low_cam.jpg`、`occlusion_comparison.png`：固定
  中性灰遮蔽輸入與推論比較。
- `cam_metrics.json`：CAM 統計、signed polarity 座標與 occlusion metrics。

任何輸入、模型 task、layer、CAM shape／值、MPS 或檔案寫入錯誤都必須以非零 exit code
和具體原因結束，不得把失敗當成沒有瑕疵，也不得靜默改用其他 device 或 layer。
