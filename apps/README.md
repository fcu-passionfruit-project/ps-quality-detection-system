# 應用程式操作手冊

`apps/` 放使用者直接啟動的 GUI 與 CLI，核心運算放在 `passionfruit/`。

- `inspection_gui.py`：前處理、單模型驗證、多模型 Pipeline、決策資料集四個分頁。
- `decision_dataset_cli.py`：資料結構、Model Manifest 與 CSV 自動化入口。
- 依賴方向：`apps` → `passionfruit.decision_dataset`／`passionfruit.inference`；核心不匯入 `apps`。
- [完整程式導覽](../docs/code-guide.md)／[決策資料集規格](../docs/specs/decision-dataset-generator.md)。

所有指令從專案根目錄執行，Windows 也使用 `python -s -m ...`。

## GUI

在 `PF` Conda 環境啟動：

```powershell
C:\Users\qoqoo\anaconda3\envs\PF\python.exe -s -m apps.inspection_gui
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
python -s -m apps.decision_dataset_cli layout --dry-run
python -s -m apps.decision_dataset_cli layout --apply

# Manifest
python -s -m apps.decision_dataset_cli manifest show
python -s -m apps.decision_dataset_cli manifest update --role defect --model models\defect.pt --imgsz 1024 --confidence-threshold 0.25
python -s -m apps.decision_dataset_cli manifest approve --role defect
python -s -m apps.decision_dataset_cli manifest validate

# CSV
python -s -m apps.decision_dataset_cli generate --version v001 --validate-only
python -s -m apps.decision_dataset_cli generate --version v001 --mode resume
python -s -m apps.decision_dataset_cli generate --version v001 --mode only_errors
python -s -m apps.decision_dataset_cli generate --version v001 --mode rerun_fruit --fruit-id 140
python -s -m apps.decision_dataset_cli generate --version v001 --mode rerun_view --fruit-id 140 --view-id 2
python -s -m apps.decision_dataset_cli generate --version v001 --mode replace_image --fruit-id 140 --view-id 2
```

可在任一指令後加 `--help` 查看完整參數。CLI 與 GUI 都建立同一種 `GenerationRequest`，因此操作規則與輸出一致。
