# 決策資料集

`layout.py` 整理及驗證照片結構；`manifest.py` 維護 Model Bundle；`generation.py` 產生、續跑與更新 CSV；`types.py` 保存請求與結果型別；`_shared.py` 保存內部 I/O；`gui.py` 是 Tkinter adapter。

入口為 `python -s -m apps.decision_dataset_cli --help` 或 GUI 第 4 分頁。核心依賴 inference；只由 gui 模組依賴 Tkinter。資料契約見 docs/specs/decision-dataset-generator.md。
