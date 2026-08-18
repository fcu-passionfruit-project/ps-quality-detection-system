# 即時狀況回報

# 百香果 Multi-stage Pipeline 檢測系統

本 repository 是百香果專題檢測層的 GitHub 程式來源，保留 `main/` 模型檢測工具、可執行規格與必要文件。大型資料集、模型及訓練產物由團隊的 [Google Drive 共用資料夾](https://drive.google.com/drive/folders/1USQK25FW7c0lLfCK1j9dGp9byPnZ5UZ9?usp=sharing) 管理。

架構內容整理自 Notion 的[「檢測層－Multi-stage Pipeline 架構設計」](https://app.notion.com/p/38d2e71fbc158042bc2bc3e6e43e9bbb)，並依目前正式硬體修正：影像來源是手機相機頁在三個停止拍攝站點取得的 `img_01.jpg`、`img_02.jpg`、`img_03.jpg`，不是舊版 ESP32-CAM 六視角連拍。

## Repository 分工

專案有兩個主要存取位置：

- [GitHub repository](https://github.com/fcu-passionfruit-project/ps-quality-detection-system)：管理 `main/` 模型檢測工具、執行說明及必要相依套件。
- [Google Drive 共用資料夾](https://drive.google.com/drive/folders/1USQK25FW7c0lLfCK1j9dGp9byPnZ5UZ9?usp=sharing)：供組員共同存取與修改資料、規格、模型及訓練產物。

| 目錄 | 用途 | 主要存取位置 |
| --- | --- | --- |
| `datasets/` | 存放分類標籤及準備上傳 Roboflow 的資料 | Google Drive |
| `docs/` | 存放 Agent 指引與可執行規格；大型參考資料仍放 Google Drive | GitHub／Google Drive |
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
         局部瑕疵 multi-class YOLO Segment
         炭疽病／畫圖蟲／擦傷／蟲咬，imgsz=640
                        |
                        v
        每張照片各產生一列特徵與 confidence
                        |
                        v
       XGBoost：每張照片各預測 0 / 1 / 2 / 3
                        |
                        v
          同一顆果實取三張照片的最低級距
                        |
                        v
        Django classify_fruit -> ESP32 -> MG996R
```

### Stage 1：ROI 定位

- 任務：從包含軌道背景的原始照片找出百香果 bounding box。
- 目標模型：YOLO26n Detect。
- 輸入尺寸：`640`。
- 目標類別：單一 class `PF` 或 `ROI`，兩種命名皆可。
- 正式權重由 Google Drive 的 `models/` 管理，執行時透過 GUI 選取。

Stage 1 只負責定位。裁切時必須把偵測座標換算回手機上傳的原始解析度照片，取得高細節 ROI，不可從已縮小的預覽圖再次裁切。每張照片保存 `fruit_area_ratio = bbox_width × bbox_height / (image_width × image_height)`；它代表百香果 bounding box 在該張照片中的面積比例，供 XGBoost 理解相對大小，不是平方公分或真實表面積。

### OpenCV 中介層

- 依 Stage 1 bounding box 裁切百香果。
- 套用圓形遮罩，降低軌道、反光與雜物對後續分類的干擾。
- 皺褶分支再轉為灰階並套用 CLAHE，提高表皮線條與陰影的局部對比。

目前程式位於 `main/image_processing.py`。

### Stage 2-1：顏色

- 目標模型：YOLO26s Classification。
- 輸入尺寸：`320`。
- 目標類別：`bad_color`、`good_color`。
- 目前沒有隨 repository 發布、可交付硬體端的正式權重。
- 顏色是模型判斷依據，訓練時不得啟用 `hsv_h`、`hsv_s`、`hsv_v` 色彩擾動。

### Stage 2-2：皺褶

- 目標模型：YOLO26s Classification。
- 輸入尺寸：`320`。
- 輸入：ROI 經灰階與 CLAHE 處理後，再轉回模型可接受的三通道影像。
- 目標類別：`false(沒皺褶)`、`true(有皺褶)` 。
- 目前沒有隨 repository 發布、可交付硬體端的正式權重。

先前的三分類 prototype 不再作為正式規格。新模型仍必須從權重讀取 class name，類別 index 對照與驗收規格由 Google Drive 的 `docs/` 管理，不可由程式自行猜測數字語意。

### Stage 2-3：局部瑕疵

- 目標模型：YOLO26s-seg
- 輸入尺寸：`1024`。
- 目標類別：`anthracnose`炭疽病；`insect_track`畫圖蟲；`abrasion`擦傷；`insect_bite`蟲咬。

每張照片正式輸出 `abrasion_area_ratio`、`anthracnose_area_ratio`、`insect_bite_area_ratio` 與 `insect_track_area_ratio`，公式為同類 mask 的 pixel union 除以 `fruit_roi_pixels`。重疊 pixel 只計算一次。未偵測到該瑕疵時為 `0`；ROI 或推論失敗時 CSV 留空並以 `status=error` 記錄原因。

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

- 局部瑕疵正式模型及四個模型的正式 Model Bundle manifest。
- Django adapter 與自動決策。
- XGBoost 訓練與整顆果實最低級距彙整。
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

## 決策資料集工具

組員主要由 GUI 的「4. 決策資料集」分頁操作；自動化與除錯則使用單一入口 `main/decision_dataset_cli.py`。工具只產生一份含 `split` 欄位的主 CSV，不另拆 `train.csv`、`valid.csv` 與 `test.csv`。

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s main\decision_dataset_cli.py --help
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s main\decision_dataset_cli.py generate --version v001
```

操作手冊與目錄分工見 [`main/README.md`](main/README.md)；資料結構、Model Bundle、CSV schema、版本與 XGBoost feature allowlist 的正式契約見 [`docs/specs/decision-dataset-generator.md`](docs/specs/decision-dataset-generator.md)。

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

局部瑕疵 `v1_defect` Dataset 的第 1／2 次訓練設定、逐類指標、Dataset 問題與後續實驗決策，見 [`docs/experiments/local-defect-segmentation-v1-v2-2026-08-18.md`](docs/experiments/local-defect-segmentation-v1-v2-2026-08-18.md)。

## 正式權重

所有訓練完成的權重由 Google Drive 的 `models/` 管理，不隨 GitHub repository 發布。日期命名的子目錄保存舊模型紀錄；目前使用的模型應放在 `models/` 的主要位置，並由 GUI 選取。

`training/runs/` 內的 `last.pt`、實驗 checkpoint、prototype 與基礎 YOLO 權重留在 Google Drive，不提交 GitHub。模型只有通過準確率、類別對照與硬體整合驗收後，才可作為目前使用版本。

## Dataset 交接規則

`datasets/` 存放分類標籤及準備上傳 Roboflow 的資料，統一由 Google Drive 共享與修改，不提交 GitHub。原始照片、增強照片及 train／valid／test 影像也應留在 Google Drive 或 Roboflow，避免 repository 納入大量或頻繁變動的資料。

資料版本、來源、分類定義與切分規格應記錄在 Google Drive 的 `docs/`。每次訓練與準確率報告需標明使用的資料版本及模型版本，避免同名模型對應到不同資料。

## 未來與硬體系統整合

GitHub repository 的 `main/` 提供檢測實作；硬體 repository 只建立薄 Django adapter。目前品質檢測 repository 是位於硬體 repository `external/ps-quality-detection-system/` 下的獨立 Git checkout，兩者各自有 `.git` 與 remote，不使用 Git submodule。

檢測層預期回傳每張照片的 ROI、`fruit_area_ratio`、顏色、二分類皺褶、四種局部瑕疵 mask union 面積比例、confidence、耗時與 Model Bundle version。XGBoost 對每張照片進行四分類，同一顆果實的最終級距取三張照片的最低值，不設炭疽病硬規則。模型載入失敗、找不到百香果或推論失敗時，必須回傳明確原因與空白特徵並保留人工覆核，不得自行猜測分類或直接控制 ESP32。

## Git 規則

- 使用 feature branch 與 pull request 協作，不直接覆蓋組員的 `main`。
- GitHub 提交 `main/` 模型檢測工具、`docs/specs/` 可執行規格及必要的執行文件與相依套件。
- 大型參考資料、`datasets/`、`models/`、`training/` 與 `decision_datasets/` 由 Google Drive 或本機管理，不提交 GitHub。
- 不提交 `.venv/`、照片、cache、預測輸出、`.pt` 權重或訓練 checkpoint。
