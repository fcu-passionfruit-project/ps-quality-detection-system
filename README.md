# 即時狀況回報
- 7/25 : 經由實測，上學期訓練的ROI模型在本次裝置化拍攝後的照片辨識上，準確度皆為0.98~1.00，因此可直接採用。

# 百香果 Multi-stage Pipeline 檢測系統

本 repository 是百香果專題檢測層的 GitHub 程式來源，保留 `main/` 模型檢測工具與必要執行文件。資料集、規格文件、模型及訓練產物則由團隊的 [Google Drive 共用資料夾](https://drive.google.com/drive/folders/1USQK25FW7c0lLfCK1j9dGp9byPnZ5UZ9?usp=sharing) 管理。

架構內容整理自 Notion 的[「檢測層－Multi-stage Pipeline 架構設計」](https://app.notion.com/p/38d2e71fbc158042bc2bc3e6e43e9bbb)，並依目前正式硬體修正：影像來源是手機相機頁在三個停止拍攝站點取得的 `img_01.jpg`、`img_02.jpg`、`img_03.jpg`，不是舊版 ESP32-CAM 六視角連拍。

## Repository 分工

專案有兩個主要存取位置：

- [GitHub repository](https://github.com/fcu-passionfruit-project/ps-quality-detection-system)：管理 `main/` 模型檢測工具、執行說明及必要相依套件。
- [Google Drive 共用資料夾](https://drive.google.com/drive/folders/1USQK25FW7c0lLfCK1j9dGp9byPnZ5UZ9?usp=sharing)：供組員共同存取與修改資料、規格、模型及訓練產物。

| 目錄 | 用途 | 主要存取位置 |
| --- | --- | --- |
| `datasets/` | 存放分類標籤及準備上傳 Roboflow 的資料 | Google Drive |
| `docs/` | 存放 Agent skill、參數、定義與規格書 | Google Drive |
| `main/` | 模型檢測工具 | GitHub `main` 分支 |
| `models/` | 存放訓練完成的模型；日期子目錄代表舊模型紀錄 | Google Drive |
| `training/packages/` | 存放模型訓練使用的資料 | Google Drive |
| `training/runs/` | 存放模型訓練結果 | Google Drive |

`main/` 提供不依賴 Django 的獨立推論工具，供團隊先驗證模型，再逐步整理成硬體系統可呼叫的介面。

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
         局部瑕疵 multi-class YOLO Segment（規劃中）
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
- 正式權重由 Google Drive 的 `models/` 管理，執行時透過 GUI 選取。

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

先前的三分類 prototype 不再作為正式規格。新模型仍必須從權重讀取 class name，類別 index 對照與驗收規格由 Google Drive 的 `docs/` 管理，不可由程式自行猜測數字語意。

### Stage 2-3：局部瑕疵

正式目標為單一 multi-class YOLO26n-seg，同一次推論辨識多種局部瑕疵，並以 segmentation mask 面積量化嚴重程度。預定類別為：

- `anthracnose`：炭疽病。
- `insect_track`：畫圖蟲。
- `abrasion`：擦傷。
- `insect_bite`：蟲咬。

上述四種是同學家實地確認會遇到的局部瑕疵；猴痘、芽種不良等目前資料為零，不納入本版規劃。Roboflow 尚未完成 mask 標註與資料量驗證，因此不得固定數字 class index，也不得在文件或系統中宣稱模型已完成。四種瑕疵先集中到同一個多類別模型，避免「每個瑕疵各一個模型」造成重複 backbone、VRAM 浪費與 false positive 疊加。

每站正式輸出 `station_{01|02|03}_{anthracnose|insect_track|abrasion|insect_bite}_area_ratio`，公式為 `defect_mask_pixels / fruit_roi_pixels`，共 `12` 個範圍為 `0～1` 的特徵。未偵測到瑕疵時為 `0`；ROI 或推論失敗時為 `null` 並進入人工覆核。Detect bbox 只能使用 `*_bbox_area_ratio` 作為實驗代理值，不得與正式 mask ratio 混用。現有 `metadata.csv` 有 `13` 筆炭疽 note，分布為中等 `4` 筆、加工 `9` 筆，因此炭疽病不設一票否決。

## 推論與效能策略

- PC 推論服務啟動時一次載入所有正式模型，不可每次收到三張照片才重新載入。
- 同一顆百香果的三張照片，在同一個模型中優先使用 batch inference。
- 不同模型先建立穩定的循序執行基準，再實測 multithreading 是否真的降低總延遲。
- PyTorch／CUDA 與 OpenCV 底層運算可能釋放 GIL，但多執行緒不保證更快，必須記錄每個模型與整條 pipeline 的耗時。
- 暫不使用 multiprocessing 同時載入多份模型，避免重複占用 RAM／VRAM。
- TensorRT、OpenVINO、FP16 與 INT8 屬於 pipeline 正確且準確率通過後的最佳化工作。

## 目前已實作範圍

`main/` 是獨立於 Django 的 Tkinter 驗證工具，正式程式集中於此，提供三個分頁：

1. 批次前處理：遞迴產生彩色 ROI，或供皺褶模型使用的灰階 CLAHE ROI。
2. 單模型驗證：可直接測試模型就緒影像，或以輔助 ROI 模型處理完整原圖。
3. 多模型 Pipeline：依序驗證 ROI、顏色、皺褶及可選的局部瑕疵模型。

所有流程共用 `main/image_processing.py` 的參數與影像處理；每張照片只採用最高 confidence 的 ROI。推論工作在背景執行緒執行，但不同模型維持循序推論基準。驗證結果會保存 JSON、ROI 標註圖與可用的階段影像。

目前尚未完成：

- 三張照片的整顆 fruit batch pipeline。
- 局部瑕疵正式模型。
- Django adapter 與自動決策。
- 自動驅動 MG996R；硬體系統目前仍保留人工分類。

## Windows 執行方式

本 repository 與硬體／Django 共用 `PF` Conda 環境。既有環境已包含 CUDA 版 PyTorch，日常安裝只需同步本 Repository 的直接依賴：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -m pip install -r requirements.txt
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s main\main.py
```

全新重建 `PF` 時固定使用 Python `3.14.4`，並從 PyTorch 官方 CUDA `12.8` index 安裝 GPU wheel：

```powershell
conda create -n PF python=3.14.4 pip -y
conda activate PF
python -s -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -s -m pip install -r requirements.txt
python -s -m pip check
```

`requirements.txt` 不重複列出 PyTorch，避免一般 PyPI 安裝覆蓋 CUDA wheel。指令使用 Python `-s` 停用 user-site，確保程式只讀取 `PF` 內的套件。

目前基準環境為 Python `3.14.4`、Ultralytics `8.4.46`、PyTorch `2.11.0+cu128`、CUDA `12.8` 與 NVIDIA GeForce RTX 3070。程式會優先使用 `cuda:0`，無 CUDA 時才回退 CPU。

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -m unittest discover -s main\tests
```

權重路徑由 GUI 選擇，可直接使用 Google Drive 或其本機同步位置中的 `.pt` 檔案，無須複製到固定位置。`.pt` 仍受 `.gitignore` 排除；只應載入團隊自己訓練並驗證的權重。

## YOLO 訓練工作區

訓練工作區由 Google Drive 的 `training/` 管理：模型訓練使用的資料放在 `training/packages/`，YOLO 訓練結果放在 `training/runs/`。以下相對路徑假設 Google Drive 資料已同步或下載到 repository 根目錄；訓練時必須設定 `project="training/runs"`，並使用能辨識用途與版本的實驗名稱：

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
model.train(
    data="training/packages/stage2-color-v001",
    project="training/runs",
    name="stage2-color-v001",
)
```

訓練完成並通過驗收的模型移至 Google Drive 的 `models/`；舊模型放在日期命名的子目錄中。

## 正式權重

所有訓練完成的權重由 Google Drive 的 `models/` 管理，不隨 GitHub repository 發布。日期命名的子目錄保存舊模型紀錄；目前使用的模型應放在 `models/` 的主要位置，並由 GUI 選取。

`training/runs/` 內的 `last.pt`、實驗 checkpoint、prototype 與基礎 YOLO 權重留在 Google Drive，不提交 GitHub。模型只有通過準確率、類別對照與硬體整合驗收後，才可作為目前使用版本。

## Dataset 交接規則

`datasets/` 存放分類標籤及準備上傳 Roboflow 的資料，統一由 Google Drive 共享與修改，不提交 GitHub。原始照片、增強照片及 train／valid／test 影像也應留在 Google Drive 或 Roboflow，避免 repository 納入大量或頻繁變動的資料。

資料版本、來源、分類定義與切分規格應記錄在 Google Drive 的 `docs/`。每次訓練與準確率報告需標明使用的資料版本及模型版本，避免同名模型對應到不同資料。

## 未來與硬體系統整合

GitHub repository 的 `main/` 提供檢測實作；硬體 repository 只建立薄 Django adapter。開發期間兩個 repository 放在同一層，未來可用 editable install 連接，但目前不使用 Git submodule。

檢測層預期回傳每站的 ROI、`roi_area_ratio`、顏色、二分類皺褶、四種局部瑕疵 mask 面積比例、confidence、耗時與模型 bundle version。所有有效果實均由 XGBoost 結合上述特徵進行四分類，不設炭疽病硬規則。模型載入失敗、找不到百香果或信心不足時，必須回傳明確原因與 `null` 特徵並保留人工覆核，不得自行猜測分類或直接控制 ESP32。

## Git 規則

- 使用 feature branch 與 pull request 協作，不直接覆蓋組員的 `main`。
- GitHub 只提交 `main/` 模型檢測工具及必要的執行文件與相依套件。
- `datasets/`、`docs/`、`models/`、`training/` 由 Google Drive 管理，不提交 GitHub。
- 不提交 `.venv/`、照片、cache、預測輸出、`.pt` 權重或訓練 checkpoint。
