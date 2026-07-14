# 百香果 Multi-stage Pipeline 檢測系統

本 repository 是百香果專題的模型訓練與檢測層工作區，負責 YOLO 模型訓練、準確率驗證、獨立推論測試，以及未來提供硬體系統呼叫的 Multi-stage Pipeline。

架構內容整理自 Notion 的[「檢測層－Multi-stage Pipeline 架構設計」](https://app.notion.com/p/38d2e71fbc158042bc2bc3e6e43e9bbb)，並依目前正式硬體修正：影像來源是手機相機頁在三個停止拍攝站點取得的 `img_01.jpg`、`img_02.jpg`、`img_03.jpg`，不是舊版 ESP32-CAM 六視角連拍。

## Repository 分工

本 repository 是模型與檢測層的唯一程式來源：

- 訓練與驗證 ROI、顏色、皺褶及局部瑕疵模型。
- 保存模型設定、類別名稱、驗證指標與 dataset manifest。
- 提供不依賴 Django 的獨立推論工具，先確認模型本身能正常運作。
- 未來整理成可匯入的 Python package，供硬體 repository 的 Django adapter 呼叫。

硬體與資料採集系統位於 [`Eten0721/PassionFruit_IoT_hardware`](https://github.com/Eten0721/PassionFruit_IoT_hardware)，負責 Django、手機相機、Dashboard、ESP32、三個 SG90 閘門與 MG996R。模型不得直接控制 GPIO；AI 最終結果必須交回 Django 的 `classify_fruit` 流程。

## 目標 Pipeline

```text
三站原始照片 img_01 / img_02 / img_03
                  |
                  v
Stage 1：YOLO Detect 定位百香果 ROI（imgsz=640）
                  |
                  v
OpenCV：從原始高解析照片裁切 ROI + 圓形遮罩
                  |
          +-------+-------------------+
          |                           |
          v                           v
顏色 YOLO-cls                    灰階 + CLAHE
imgsz=320                            |
          |                           v
          |                    皺褶 YOLO-cls
          |                    imgsz=448
          |                           |
          +-------------+-------------+
                        |
                        v
         局部瑕疵 multi-class YOLO Detect（規劃中）
         蟲咬／猴痘狀腐爛／擦傷，imgsz=640
                        |
                        v
        彙整三站特徵、confidence 與模型版本
                        |
                        v
       決策層：high_medium / low / processing / discard
                        |
                        v
        Django classify_fruit -> ESP32 -> MG996R
```

### Stage 1：ROI 定位

- 任務：從包含軌道背景的原始照片找出百香果 bounding box。
- 模型類型：YOLO Detect。
- 輸入尺寸：`640`。
- 目前正式權重：`main/stage1.pt`。
- 目前類別：`PF`。

Stage 1 只負責定位。裁切時必須回到手機上傳的原始解析度照片，使用偵測座標取得高細節 ROI，不可從已縮小的預覽圖再次裁切。

### OpenCV 中介層

- 依 Stage 1 bounding box 裁切百香果。
- 套用圓形遮罩，降低軌道、反光與雜物對後續分類的干擾。
- 皺褶分支再轉為灰階並套用 CLAHE，提高表皮線條與陰影的局部對比。

目前程式位於 `main/image_processing.py`。

### Stage 2-1：顏色

- 模型類型：YOLO Classification。
- 輸入尺寸：`320`。
- 目前正式權重：`main/stage2_1.pt`。
- 目前權重類別：`bad_color`、`good_color`。
- 顏色是模型判斷依據，訓練時不得啟用 `hsv_h`、`hsv_s`、`hsv_v` 色彩擾動。

### Stage 2-2：皺褶

- 模型類型：YOLO Classification。
- 輸入尺寸：`448`。
- 目前正式權重：`main/stage2_2.pt`。
- 輸入：ROI 經灰階與 CLAHE 處理後，再轉回模型可接受的三通道影像。
- 目前權重類別順序：`slight_wrinkle`、`smooth`、`wrinkle`。

Notion 的早期設計曾以 `0／1／2` 表示「飽滿／輕微／嚴重」，但目前權重的 class index 順序不同。程式必須讀取權重內的 class name，不可自行硬編碼舊數字對照。

### Stage 2-3：局部瑕疵

規劃使用單一 multi-class YOLO Detect，同一次推論辨識多種局部瑕疵，並以 bounding box 或 mask 面積量化嚴重程度。預定優先類別為：

- 蟲咬。
- 猴痘狀同心圓腐爛。
- 大面積擦傷。

炭疽病與蟲爬白痕目前缺少足夠資料，不得在文件或系統中宣稱已完成。相近瑕疵應先集中到同一個多類別模型，避免「每個瑕疵各一個模型」造成重複 backbone、VRAM 浪費與 false positive 疊加。

## 推論與效能策略

- PC 推論服務啟動時一次載入所有正式模型，不可每次收到三張照片才重新載入。
- 同一顆百香果的三張照片，在同一個模型中優先使用 batch inference。
- 不同模型先建立穩定的循序執行基準，再實測 multithreading 是否真的降低總延遲。
- PyTorch／CUDA 與 OpenCV 底層運算可能釋放 GIL，但多執行緒不保證更快，必須記錄每個模型與整條 pipeline 的耗時。
- 暫不使用 multiprocessing 同時載入多份模型，避免重複占用 RAM／VRAM。
- TensorRT、OpenVINO、FP16 與 INT8 屬於 pipeline 正確且準確率通過後的最佳化工作。

## 目前已實作範圍

`main/` 是獨立於 Django 的 Tkinter 驗證工具，目前可以：

1. 選擇一張本機百香果照片。
2. 使用 Stage 1 找出第一個百香果 ROI。
3. 顯示 bounding box、圓形遮罩結果及灰階 CLAHE 預覽。
4. 以兩條 thread 執行顏色與皺褶分類。
5. 顯示兩個模型的 class name 與 confidence。

目前尚未完成：

- 三張照片的 batch pipeline。
- 局部瑕疵正式模型。
- 統一的結構化 `DetectionResult`。
- Django adapter 與自動決策。
- 自動驅動 MG996R；硬體系統目前仍保留人工分類。

## Windows 執行方式

先安裝 Git LFS，clone 後取得正式權重：

```powershell
git lfs install
git lfs pull
```

建立 Python `3.10.20` 環境並安裝模型依賴：

```powershell
conda create -n pf_detection_env python=3.10.20 -y
conda activate pf_detection_env
python -m pip install -r requirements.txt
```

目前 `main.py` 使用相對模型路徑，因此必須從 `main/` 啟動：

```powershell
cd main
python main.py
```

開啟視窗後選擇一張照片，即可測試 ROI、顏色與皺褶模型。只應載入團隊自己訓練並經 manifest 驗證的 `.pt`；不要執行來源不明的 PyTorch 權重。

## 正式權重

| 階段 | 路徑 | Task | `imgsz` | SHA-256 |
|---|---|---|---:|---|
| ROI | `main/stage1.pt` | detect | `640` | `7f99c6d6e696bab4df69c37f2c51e0efe976f057432b573eb7ce48f22b86d35c` |
| 顏色 | `main/stage2_1.pt` | classify | `320` | `8c5ff1eb4bf9b52e1dbfe007b1d577a8ad2da65cb6f0d5716f4a265f1479fefc` |
| 皺褶 | `main/stage2_2.pt` | classify | `448` | `a0732b558cd2cc33d9cc83fd8647cd28adfa89cb17b4115ddb398dfd6c717f8d` |

機器可讀資訊位於 `models/manifest.yaml`。只有通過驗證並準備交付硬體端的正式權重可放在 `main/`，且必須由 Git LFS 追蹤；`runs/` 內的 `last.pt`、實驗 checkpoint 與基礎 YOLO 權重不得提交。

## Dataset 交接規則

原始照片、增強照片、Roboflow 匯出資料與 train／valid／test 影像不進 Git。資料應保存於團隊共用儲存空間，模型 repository 只追蹤 manifest：

```text
dataset_id
建立日期與來源
fruit 數量與影像數量
每顆 fruit 固定三站影像
四個人工分類的分布
train / valid / test 切分
儲存位置與 checksum manifest
```

範本位於 `datasets/manifest.example.yaml`。每份準確率報告必須同時記錄 dataset ID、模型 manifest 版本與 Git commit，避免同一個模型名稱對應到不同資料。

## 未來與硬體系統整合

模型 repository 將提供唯一的檢測實作；硬體 repository 只建立薄 Django adapter。開發期間兩個 repository 放在同一層，未來可用 editable install 連接，但目前不使用 Git submodule。

檢測層預期回傳每站的 ROI、顏色、皺褶、局部瑕疵、confidence、耗時與模型 bundle version。模型載入失敗、找不到百香果或信心不足時，必須回傳明確原因並保留人工覆核，不得自行猜測分類或直接控制 ESP32。

## Git 規則

- 使用 feature branch 與 pull request 協作，不直接覆蓋組員的 `main`。
- 不提交 `.venv/`、照片、dataset、`runs/`、cache 或預測輸出。
- 不提交每次訓練的 `last.pt` 與歷史 checkpoint。
- 正式三個權重使用 Git LFS；更換權重時必須同步更新 `models/manifest.yaml`、SHA-256 與驗證結果。
- 若沒有 Git LFS，clone 後只會取得 pointer，需先安裝 Git LFS 再執行 `git lfs pull`。
