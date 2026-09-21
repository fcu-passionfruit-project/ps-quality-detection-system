# 內附 YOLO CAM 程式

本目錄由原 `tools/yolo_cam/` 原樣搬移，僅調整套件匯入並補 Python package 標記；專案既有文件稱其為 YOLO-26-CAM。

原始 checkout 未附上可驗證的 upstream URL、版本或 LICENSE。本次未推定來源版本或授權，也未增添虛構授權宣告；既有原始碼註記予以保留。Git 歷史可由原 tools/yolo_cam 路徑追溯。

由 `passionfruit.explainability.eigencam` 直接載入，不需另行下載 YOLO-26-CAM-main。內部依賴 torch、torchvision、numpy、opencv-python、ttach；沒有獨立執行入口。
