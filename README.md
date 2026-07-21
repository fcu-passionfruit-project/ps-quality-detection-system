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
         炭疽病／畫圖蟲／擦傷／蟲咬，imgsz=640
                        |
                        v
        彙整三站特徵、confidence 與模型版本
                        |
                        v
       決策層：high / medium / low / processing
                        |
                        v
        Django classify_fruit -> ESP32 -> MG996R
```

### Stage 1：ROI 定位

- 任務：從包含軌道背景的原始照片找出百香果 bounding box。
- 目標模型：YOLO26n Detect。
- 輸入尺寸：`640`。
- 目標類別：`PF`。
- 目前沒有隨 repository 發布、可交付硬體端的正式權重。

Stage 1 只負責定位。裁切時必須把偵測座標換算回手機上傳的原始解析度照片，取得高細節 ROI，不可從已縮小的預覽圖再次裁切。每站另保存 `roi_area_ratio = bbox_width × bbox_height / (image_width × image_height)`；它代表百香果 bounding box 在該張照片中的面積比例，供 XGBoost 理解相對大小，不是平方公分或真實表面積。正式欄位為 `station_01_roi_area_ratio`、`station_02_roi_area_ratio`、`station_03_roi_area_ratio`。

### OpenCV 中介層

- 依 Stage 1 bounding box 裁切百香果。
- 套用圓形遮罩，降低軌道、反光與雜物對後續分類的干擾。
- 皺褶分支再轉為灰階並套用 CLAHE，提高表皮線條與陰影的局部對比。

目前程式位於 `main/image_processing.py`。

### Stage 2-1：顏色

- 目標模型：YOLO26n Classification。
- 輸入尺寸：`320`。
- 目標類別：`bad_color`、`good_color`。
- 目前沒有隨 repository 發布、可交付硬體端的正式權重。
- 顏色是模型判斷依據，訓練時不得啟用 `hsv_h`、`hsv_s`、`hsv_v` 色彩擾動。

### Stage 2-2：皺褶

- 目標模型：YOLO26n Classification。
- 輸入尺寸：`448`。
- 輸入：ROI 經灰階與 CLAHE 處理後，再轉回模型可接受的三通道影像。
- 正式目標類別：`smooth`、`wrinkle` 二分類。
- 目前沒有隨 repository 發布、可交付硬體端的正式權重。

先前的三分類 prototype 不再作為正式規格。新模型仍必須從權重讀取 class name，並在 manifest 固定 index 對照，不可由程式自行猜測數字語意。

### Stage 2-3：局部瑕疵

規劃使用單一 multi-class YOLO Detect，同一次推論辨識多種局部瑕疵，並以 bounding box 或 mask 面積量化嚴重程度。預定優先類別為：

- `anthracnose`：炭疽病。
- `insect_track`：畫圖蟲。
- `abrasion`：擦傷。
- `insect_bite`：蟲咬。

上述四種是同學家實地確認會遇到的局部瑕疵；猴痘、芽種不良等目前資料為零，不納入本版規劃。Roboflow 尚未完成 bounding-box 標註與資料量驗證，因此不得固定數字 class index，也不得在文件或系統中宣稱模型已完成。四種瑕疵先集中到同一個多類別模型，避免「每個瑕疵各一個模型」造成重複 backbone、VRAM 浪費與 false positive 疊加。

## 推論與效能策略

- PC 推論服務啟動時一次載入所有正式模型，不可每次收到三張照片才重新載入。
- 同一顆百香果的三張照片，在同一個模型中優先使用 batch inference。
- 不同模型先建立穩定的循序執行基準，再實測 multithreading 是否真的降低總延遲。
- PyTorch／CUDA 與 OpenCV 底層運算可能釋放 GIL，但多執行緒不保證更快，必須記錄每個模型與整條 pipeline 的耗時。
- 暫不使用 multiprocessing 同時載入多份模型，避免重複占用 RAM／VRAM。
- TensorRT、OpenVINO、FP16 與 INT8 屬於 pipeline 正確且準確率通過後的最佳化工作。

## 目前已實作範圍

`main/` 是獨立於 Django 的 Tkinter prototype 驗證工具；只有在開發者自行準備本機 `stage1.pt`、`stage2_1.pt`、`stage2_2.pt` 時，才可執行下列既有流程：

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

建立 Python `3.10.20` 環境並安裝模型依賴：

```powershell
conda create -n pf_detection_env python=3.10.20 -y
conda activate pf_detection_env
python -m pip install -r requirements.txt
```

repository 目前不提供 `.pt` 權重。若只為本機 prototype 驗證，請自行把三個既有權重放在 `main/`；它們受 `.gitignore` 排除，不得加入 commit。`main.py` 使用相對模型路徑，因此必須從 `main/` 啟動：

```powershell
cd main
python main.py
```

開啟視窗後選擇一張照片，即可測試 ROI、顏色與皺褶模型。只應載入團隊自己訓練並經 manifest 驗證的 `.pt`；不要執行來源不明的 PyTorch 權重。

## 正式權重

目前沒有隨 repository 發布的正式權重。原本的三個 prototype 已撤除 Git 追蹤，只保留在需要它們的開發者本機。

機器可讀規劃位於 `models/manifest.yaml`。未來只有通過準確率、類別對照、SHA-256 與硬體整合驗收的正式權重，才可重新設定 Git LFS 規則並提交；`runs/` 內的 `last.pt`、實驗 checkpoint、prototype 與基礎 YOLO 權重不得提交。

## Dataset 交接規則

原始照片、增強照片、Roboflow 匯出資料與 train／valid／test 影像不進 Git。本機原始快照實體保存於 `D:\passion-fruit-datasets\`，repository 根目錄的 ignored `dataset/` junction 指向該位置；模型 repository 只追蹤 `datasets/` 內的 manifest：

```text
dataset_id
建立日期與來源
fruit 數量與影像數量
每顆 fruit 固定三站影像
四個人工分類的分布
train / valid / test 切分
儲存位置與 checksum manifest
```

目前 `pf-20260716-v001` 保存 `2026-07-16` 三站採集的 `327` 顆百香果與 `981` 張照片；三站各 `327` 張，所有照片皆為 `1080 × 1920`。尚未建立 train／valid／test，由後續 Roboflow 流程負責切分。其 manifest 位於 `datasets/pf-20260716-v001.yaml`，原始快照可由 `dataset/pf-20260716-v001/` 存取。

範本位於 `datasets/manifest.example.yaml`。每份準確率報告必須同時記錄 dataset ID、模型 manifest 版本與 Git commit，避免同一個模型名稱對應到不同資料。`dataset/` junction 只提供本機存取，不是第二份備份；凍結快照仍須上傳到 Roboflow 或團隊共用儲存空間。

## 未來與硬體系統整合

模型 repository 將提供唯一的檢測實作；硬體 repository 只建立薄 Django adapter。開發期間兩個 repository 放在同一層，未來可用 editable install 連接，但目前不使用 Git submodule。

檢測層預期回傳每站的 ROI、`roi_area_ratio`、顏色、二分類皺褶、四種局部瑕疵面積比例、confidence、耗時與模型 bundle version。經有效門檻判定有炭疽病時，Decision Layer 直接輸出加工 `processing`；其餘果實才交由 XGBoost 四分類。模型載入失敗、找不到百香果或信心不足時，必須回傳明確原因並保留人工覆核，不得自行猜測分類或直接控制 ESP32。

## Git 規則

- 使用 feature branch 與 pull request 協作，不直接覆蓋組員的 `main`。
- 不提交 `.venv/`、照片、dataset、`runs/`、cache 或預測輸出。
- 不提交 ignored `dataset/` junction 或其指向的原始照片；只提交 `datasets/*.yaml` manifest。
- 不提交每次訓練的 `last.pt` 與歷史 checkpoint。
- Prototype 與未驗收權重一律只留本機；未來正式權重必須經團隊核准後才加入明確的 Git LFS 規則，並同步更新 `models/manifest.yaml`、SHA-256 與驗證結果。
