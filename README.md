# 即時狀況回報

# 百香果 Multi-stage Pipeline 檢測系統

本 repository 是百香果專題檢測層的 GitHub 程式來源，保留 `main/` 模型檢測工具、可執行規格與必要文件。大型資料集、模型及訓練產物由團隊的 [Google Drive 共用資料夾](https://drive.google.com/drive/folders/1USQK25FW7c0lLfCK1j9dGp9byPnZ5UZ9?usp=sharing) 管理。

目標架構與模型分工以 Notion 的[「檢測層－Multi-stage Pipeline」](https://app.notion.com/p/38d2e71fbc158042bc2bc3e6e43e9bbb)中「Pipeline 架構與分工」為準。影像來源是手機相機頁在三個停止拍攝站點取得的 `img_01.jpg`、`img_02.jpg`、`img_03.jpg`。

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
                  v
顏色 YOLO Classification（imgsz=320，三張照片 batch）
                  |
                  v
灰階 + CLAHE -> 皺褶 YOLO Classification
                  （imgsz=320，三張照片 batch）
                  |
                  v
局部瑕疵 YOLO26s-seg multi-class
炭疽病／畫圖蟲／擦傷／蟲咬（imgsz=1024，三張照片 batch）
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

PC 推論服務啟動時載入全部模型。模型之間依顏色、皺褶、局部瑕疵的順序執行；同一個模型一次處理同一顆 Fruit 的三張照片。OpenCV 前處理可使用 multithreading，模型推論先維持序列式 batch 基準。

### Stage 1：ROI 定位

- 任務：從包含軌道背景的原始照片找出百香果 bounding box。
- 目標模型：YOLO26n Detect。
- 輸入尺寸：`640`。
- 目標類別：單一 class `PF` 或 `ROI`，兩種命名皆可。
- 正式權重由 Google Drive 的 `models/` 管理，執行時透過 GUI 選取。

Stage 1 只負責定位。裁切時必須把偵測座標換算回手機上傳的原始解析度照片，取得高細節 ROI，不可從已縮小的預覽圖再次裁切。Stage 1 回傳 `roi_area_ratio = bbox_width × bbox_height / (image_width × image_height)`；寫入 Decision Dataset 時對應為 `fruit_area_ratio`。它代表百香果 bounding box 在該張照片中的面積比例，供 XGBoost 理解相對大小，不是平方公分或真實表面積。

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
- 模型之間依顏色、皺褶、局部瑕疵循序執行；同一個模型使用三張照片的 batch inference。
- OpenCV 前處理可使用 multithreading；模型推論先建立穩定的序列式 batch 基準，再實測多模型 multithreading 是否降低總延遲。
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

## macOS / Apple Silicon 執行方式

本 repository 亦可在 Apple Silicon Mac 上進行開發、模型測試與 YOLO 訓練。macOS 環境不使用 NVIDIA CUDA，而是透過 PyTorch 的 Metal Performance Shaders（MPS）使用 Apple Silicon GPU。

目前已驗證的本機環境為：

* macOS
* MacBook Pro，Apple M5 Pro
* Python `3.12.13`
* PyTorch `2.13.0`
* Ultralytics `8.4.122`
* GPU backend：Apple Metal Performance Shaders（MPS）
* Python 環境：`venv`
* Python 安裝來源：Homebrew

> 注意：repository 的 `requirements.txt` 目前固定 `ultralytics==8.4.46`，與上述 Mac 本機已驗證的 `8.4.122` 不同。此處記錄的是目前實際使用並驗證過的 Mac 開發環境，不代表 repository 的 dependency baseline 已更新。若要統一團隊使用的 Ultralytics 版本，應另外修改 `requirements.txt` 並重新完成測試，不應只因平台不同而自行變更版本。

### 建立 Python 環境

macOS 使用 Homebrew Python 3.12：

```bash
brew install python@3.12
```

GUI 使用 Tkinter。Homebrew 的 Python 3.12 需另外安裝對應的 Tkinter package：

```bash
brew install python-tk@3.12
```

確認 Python：

```bash
python3.12 --version
```

在 repository root 建立 virtual environment：

```bash
python3.12 -m venv .venv
```

啟用：

```bash
source .venv/bin/activate
```

確認目前使用的是 virtual environment 中的 Python：

```bash
which python
python --version
```

更新 pip：

```bash
python -m pip install --upgrade pip
```

### 安裝 PyTorch

Apple Silicon 不使用 Windows 環境中的 CUDA 12.8 PyTorch wheel。

macOS 安裝 PyTorch：

```bash
python -m pip install torch torchvision
```

確認 PyTorch 與 MPS：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

Apple Silicon 環境正常時應顯示：

```text
MPS available: True
```

### 安裝 Repository Dependencies

repository 的其他 Python dependency 定義於：

```text
requirements.txt
```

安裝前請注意，目前該檔案固定：

```text
ultralytics==8.4.46
```

若本機正在使用其他已驗證版本，例如 `8.4.122`，直接執行：

```bash
python -m pip install -r requirements.txt
```

可能會將 Ultralytics 調整為 repository 固定的版本。

若團隊尚未決定是否統一版本，請先確認目前環境：

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

dependency 安裝或調整完成後可執行：

```bash
python -s -m pip check
```

### 確認 Tkinter

可使用：

```bash
python -c "import tkinter; print('Tk:', tkinter.TkVersion)"
```

若出現：

```text
ModuleNotFoundError: No module named '_tkinter'
```

請確認 Homebrew 已安裝：

```bash
brew install python-tk@3.12
```

並重新建立使用同一套 Homebrew Python 3.12 的 `.venv`。

### 啟動 GUI

在 repository root 啟用 virtual environment：

```bash
source .venv/bin/activate
```

再執行：

```bash
python -s main/main.py
```

模型權重仍由 GUI 選擇，可直接使用 Google Drive 或其他本機同步位置內的 `.pt` 檔案，不需要複製至 repository。

### 執行測試

執行全部 `unittest`：

```bash
python -s -m unittest discover -s main/tests
```

dependency 有異動時另執行：

```bash
python -s -m pip check
```

### 決策資料集 CLI

macOS 使用與 Windows 相同的 Python CLI，只需改用 POSIX 路徑並使用目前啟用的 virtual environment：

```bash
python -s main/decision_dataset_cli.py --help
```

例如產生 `v001`：

```bash
python -s main/decision_dataset_cli.py generate --version v001
```

詳細操作方式仍以 [`main/README.md`](main/README.md) 與 [`docs/specs/decision-dataset-generator.md`](docs/specs/decision-dataset-generator.md) 為準。

### Rare-class Dataset 擴增

Windows 文件中的：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe ...
```

在 macOS 上不適用。

啟用 `.venv` 後改用：

```bash
python -s tools/augment_rare_classes.py \
  --input training/packages/v2_defect \
  --output training/packages/v2_defect_argument \
  --anthracnose-copies 2 \
  --insect-bite-copies 5 \
  --seed 0
```

Dataset 的處理規則與 Windows 相同：工具不修改來源 Dataset，`valid` 與 `test` 原樣複製。

### Experiment B：Class-aware Sampling

Experiment B 只改變 `local_defects` train image 的曝光次數，不修改或複製圖片與 label，也不改變 train／valid／test split。預設規則為 `abrasion=1x`、`anthracnose=2x`、`insect_bite=4x`、`insect_track=1x`、null image `=1x`；同一張圖片含多個 class 時取最大 factor。

在 repository root、已啟用 `.venv` 後產生 train image list：

```bash
python -s tools/generate_class_aware_train_list.py \
  --dataset training/packages/local_defects
```

預設輸出為 `training/packages/local_defects/experiments/train_B_class_aware.txt`。清單可搭配不覆寫正式設定的 `training/packages/local_defects/data_exp_B.yaml` 使用；該 YAML 的 valid、test、class 數量與 class names 和原始 `data.yaml` 相同。

正式訓練命令如下；產生清單不會自動啟動訓練：

```bash
yolo task=segment mode=train \
  model=yolo26s-seg.pt \
  data=training/packages/local_defects/data_exp_B.yaml \
  project=training/runs/segment \
  name=v20_B_class_aware \
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

### Experiment C：Rare-class Targeted Crop

Experiment C 回到 `v19_defect_epoch200_pw050_cp01` 使用的原始 `local_defects`
train split，每張原圖只曝光一次；唯一新增變因是以 segmentation instance 為中心的
`anthracnose`／`insect_bite` targeted crop。它不讀取 Experiment B 的 repeated train
list，也不修改或複製 valid／test。

在 repository root、已啟用 `.venv` 後建立並驗證 Dataset：

```bash
python -s tools/generate_targeted_crops.py \
  --dataset training/packages/local_defects \
  --anthracnose-crops-per-instance 1 \
  --insect-bite-crops-per-instance 3 \
  --context-scale 4.0 \
  --context-scale-variation 0.5 \
  --center-jitter 0.10 \
  --minimum-crop-size 64 \
  --maximum-crop-fraction 0.90 \
  --seed 0
```

工具輸出 `experiments/exp_C_targeted_crop/`、
`experiments/train_C_targeted_crop.txt` 與 `data_exp_C.yaml`，並保存 crop／來源
manifest、驗證報告、Dataset hash、初始權重 hash 與 Python／PyTorch／Ultralytics
版本。既有輸出不會被默默覆蓋；需要以同一組參數重建時明確加上 `--overwrite`。
完整資料契約見
[`docs/specs/local-defect-targeted-crop-experiment.md`](docs/specs/local-defect-targeted-crop-experiment.md)。

Dataset 驗證與 smoke test 通過後，Experiment C 的手動訓練指令為：

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

這個指令必須由開發者在確認 Dataset 統計後手動執行；Dataset generator 不會啟動訓練。

### YOLO 訓練

Apple Silicon 上執行 Ultralytics YOLO 訓練時，GPU device 使用：

```text
device=mps
```

而不是 Windows / NVIDIA 環境中的：

```text
device=0
```

因此 README 內局部瑕疵模型的 Windows 訓練指令：

```text
device=0
```

在 Mac 上應改為：

```text
device=mps
```

例如相同的局部瑕疵訓練實驗，在 macOS 上可執行：

```bash
yolo task=segment mode=train \
  model=yolo26s-seg.pt \
  data=training/packages/v2_defect_argument/data.yaml \
  project=training/runs/segment \
  name=v2_defect_pw050_cp03 \
  epochs=100 \
  imgsz=1024 \
  batch=-1 \
  device=mps \
  workers=8 \
  seed=0 \
  deterministic=True \
  cls_pw=0.5 \
  copy_paste=0.3 \
  copy_paste_mode=mixup \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  degrees=0.0 \
  translate=0.1 \
  scale=0.5 \
  shear=0.0 \
  perspective=0.0 \
  flipud=0.0 \
  fliplr=0.5 \
  bgr=0.0 \
  mosaic=1.0 \
  mixup=0.0 \
  cutmix=0.0 \
  close_mosaic=10
```

除了與運算裝置直接相關的參數外，不應只因 Windows 與 macOS 平台不同，就自行改變 Dataset、augmentation、epochs、imgsz 或其他模型實驗條件，以便不同硬體上的實驗結果仍可比較。

`batch` 可依不同 GPU／記憶體條件另外進行效能實驗，但調整時應留下實驗紀錄。

### CUDA、MPS 與 CPU

兩個開發環境目前使用不同 GPU backend：

| 平台      | GPU                     | PyTorch backend | Ultralytics device    |
| ------- | ----------------------- | --------------- | --------------------- |
| Windows | NVIDIA GeForce RTX 3070 | CUDA            | `device=0` / `cuda:0` |
| macOS   | Apple M5 Pro            | MPS             | `device=mps`          |

因此不得將 CUDA-specific 安裝方式或 `cuda:0` 硬編碼為所有平台共同設定。

若程式需要自動選擇 PyTorch device，應分別考慮：

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

但目前正式 GUI 是否已支援 MPS 自動選擇，仍應以 `main/model_manager.py` 的實際實作為準；README 不應在程式尚未實作前宣稱 GUI 一定會自動使用 MPS。

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

局部瑕疵模型的實驗紀錄：

- `v1_defect` 第 1／2 次歷史 baseline：[`docs/experiments/local-defect-segmentation-v1-v2-2026-08-18.md`](docs/experiments/local-defect-segmentation-v1-v2-2026-08-18.md)
- `v2_defect` targeted augmentation 與 v3 參數調整：[`docs/experiments/local-defect-segmentation-v2-targeted-2026-08-19.md`](docs/experiments/local-defect-segmentation-v2-targeted-2026-08-19.md)

新版局部瑕疵 Dataset 從 Roboflow 以原始解析度匯出至 `training/packages/v2_defect`，不在 Roboflow 套用 preprocessing 或 augmentation。只對 train 的少數類別來源影像產生離線變化：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s tools\augment_rare_classes.py --input training\packages\v2_defect --output training\packages\v2_defect_argument --anthracnose-copies 2 --insect-bite-copies 5 --seed 0
```

工具不修改來源 Dataset；valid 與 test 會原樣複製。v2／v3 訓練結果與後續決策記錄在上述獨立實驗文件。

targeted Dataset 完成並通過檢查後，使用固定參數訓練完整改善方案：

```powershell
yolo task=segment mode=train model=yolo26s-seg.pt data=training/packages/v2_defect_argument/data.yaml project=training/runs/segment name=v2_defect_pw050_cp03 epochs=100 imgsz=1024 batch=-1 device=0 workers=8 seed=0 deterministic=True cls_pw=0.5 copy_paste=0.3 copy_paste_mode=mixup hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=0.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 bgr=0.0 mosaic=1.0 mixup=0.0 cutmix=0.0 close_mosaic=10
```

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
