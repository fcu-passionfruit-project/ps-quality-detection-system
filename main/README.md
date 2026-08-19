# `main/` 操作手冊

`main/` 同時包含既有 YOLO 檢測工具與決策資料集工具，但兩者的核心程式已分開：

- `main.py`、`pipeline.py`、`model_manager.py`、`image_processing.py`：既有 YOLO GUI、Pipeline 與前處理。
- `decision_dataset/`：資料結構、Model Manifest、CSV 產生、共用型別與 GUI adapter。
- `decision_dataset_cli.py`：決策資料集的自動化與除錯入口。
- 正式資料契約：[`../docs/specs/decision-dataset-generator.md`](../docs/specs/decision-dataset-generator.md)。

## GUI

在 `PF` Conda 環境啟動：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s main\main.py
```

第 4 個「決策資料集」分頁包含：

1. 「資料結構」：先預覽，再確認套用；路徑變更會使舊預覽失效。
2. 「Model Manifest」：檢查權重、更新 `candidate`、核准 role 與驗證完整 Bundle。寫入前會保留 `.bak`。
3. 「產生 CSV」：補缺列、只重試錯誤、重跑整顆、重跑單張、替換照片後重跑。安全停止會先完成目前果實並寫入 checkpoint。

Model Manifest 尚未備妥四個 `approved_for_use` role 時，只能做結構整理與驗證，不能正式產生 CSV。

## CLI

所有操作都由同一支程式進入：

```powershell
# 資料結構
python -s main\decision_dataset_cli.py layout --dry-run
python -s main\decision_dataset_cli.py layout --apply

# Manifest
python -s main\decision_dataset_cli.py manifest show
python -s main\decision_dataset_cli.py manifest update --role defect --model models\defect.pt --imgsz 1024 --confidence-threshold 0.25
python -s main\decision_dataset_cli.py manifest approve --role defect
python -s main\decision_dataset_cli.py manifest validate

# CSV
python -s main\decision_dataset_cli.py generate --version v001 --validate-only
python -s main\decision_dataset_cli.py generate --version v001 --mode resume
python -s main\decision_dataset_cli.py generate --version v001 --mode only_errors
python -s main\decision_dataset_cli.py generate --version v001 --mode rerun_fruit --fruit-id 140
python -s main\decision_dataset_cli.py generate --version v001 --mode rerun_view --fruit-id 140 --view-id 2
python -s main\decision_dataset_cli.py generate --version v001 --mode replace_image --fruit-id 140 --view-id 2
```

可在任一指令後加 `--help` 查看完整參數。CLI 與 GUI 都建立同一種 `GenerationRequest`，因此操作規則與輸出一致。
