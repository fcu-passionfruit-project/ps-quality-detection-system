# 模型推論

`model_manager.py` 負責載入模型、裝置與模型任務檢查；`image_processing.py` 負責 ROI、圓形遮罩與 CLAHE；`pipeline.py` 負責單模型及多模型推論、特徵與驗證輸出。

由 GUI、決策資料集及評估工具呼叫，本目錄沒有獨立 CLI。依賴 OpenCV、NumPy、PyTorch 與 Ultralytics，不依賴 apps。
