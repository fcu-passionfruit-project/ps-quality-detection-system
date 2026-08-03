# YOLO 訓練工作區

`training/` 只保存本機訓練資料與實驗輸出；除本說明文件外，內容皆不提交 Git。

```text
training/
├── packages/    # Roboflow 匯出檔與解壓後的訓練資料
└── runs/        # YOLO 訓練結果、圖表與 checkpoint
```

從 repository 根目錄執行訓練，並固定指定輸出目錄與明確的實驗名稱：

```python
model.train(
    data="training/packages/<dataset>/data.yaml",
    project="training/runs",
    name="<用途>-v<版本>",
)
```

`best.pt` 通過準確率、類別對照與硬體整合驗收後，才可重新命名並移至 `models/`。正式模型加入 Git 時，必須同步更新 `models/manifest.yaml`、SHA-256、驗證結果與明確的 Git LFS 規則；`last.pt` 與其他 checkpoint 不得移入 `models/`。

# YOLO訓練語法教學
- detect模型: yolo train data=\訓練包的路徑\data.yaml model=yolo26s.pt epochs=50 imgsz=640 device=0

- classify模型: yolo classify train model=yolo26n-cls.pt epochs=50 imgsz=224 device=0 workers=2 data=[可以把資料包拖曳進terminal會自動填入路徑]

- epochs代表模型要訓練幾輪；imgsz是你餵入的照片尺寸(color和wrinkle皆為320)；device代表訓練裝置(0是預設顯示卡)
