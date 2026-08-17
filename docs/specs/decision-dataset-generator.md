# 決策資料集產生器規格

## 目的

將固定三視角的百香果照片依序送入 ROI、顏色、皺褶與局部瑕疵 YOLO 模型，產生一份供專家標註與後續 XGBoost 訓練使用的主 CSV。

一張照片是一筆 View Observation，也是一列 CSV。同一顆 Fruit 的三列依 `fruit_id + view_id` 識別。XGBoost 未來分別預測三張照片的 Image Grade，Fruit Grade 取三者最低值。

## 輸入契約

```text
DecisionLayerTrain/
├── train/
│   └── fruit_001/
│       ├── img_01.jpg
│       ├── img_02.jpg
│       └── img_03.jpg
├── valid/
└── test/
```

- 必須保留既有 `train/valid/test`，同一個 `fruit_id` 只能出現在一個 split。
- 果實資料夾名稱必須符合 `fruit_<整數>`。
- 每顆果實必須剛好包含 `img_01.jpg`、`img_02.jpg`、`img_03.jpg`。
- 結構錯誤的果實不建立合成 CSV 列，錯誤寫入 `validation_errors.csv`，其他合法果實繼續處理。
- 舊 `上等/中等/下等/加工` 只代表 Legacy Fruit Grade，不可轉成 Image Grade。

GUI 的「資料結構」先預覽搬移計畫，成功後才允許套用；CLI 對應為 `decision_dataset_cli.py layout --dry-run|--apply`。套用時將果實資料夾搬到原 split 直下，寫出 `layout_migration.csv`，並只刪除確認為空的舊級距資料夾。

## Model Bundle 契約

| role | task | 正式 classes | threshold |
| --- | --- | --- | --- |
| `roi` | `detect` | 單一 class `PF` 或 `ROI` | 必填 |
| `color` | `classify` | `bad_color`, `good_color` | `null` |
| `wrinkle` | `classify` | `false(沒皺褶)`, `true(有皺褶)` | `null` |
| `defect` | `segment` | `abrasion`, `anthracnose`, `insect_bite`, `insect_track` | 必填 |

程式依 class name 對應語意，不依賴固定 class index。ROI 模型的單一 class 可命名為 `PF` 或 `ROI`，不允許同時存在兩個 class。每筆 manifest record 必須含 `status`、`path`、`sha256`、`task`、`architecture`、`imgsz`、`classes`、`confidence_threshold`、`size_bytes`。四個 role 都必須是 `approved_for_use`，權重必須位於 repository 的 `models/`，實際 task、classes 與 SHA-256 必須通過驗證。

## CSV Schema

欄位順序固定：

```text
fruit_id,view_id,split,fruit_area_ratio,color_bad,color_confidence,wrinkle,wrinkle_confidence,abrasion_area_ratio,abrasion_max_confidence,anthracnose_area_ratio,anthracnose_max_confidence,insect_bite_area_ratio,insect_bite_max_confidence,insect_track_area_ratio,insect_track_max_confidence,status,error_reason,grade
```

- `fruit_area_ratio`：ROI bounding box 面積除以原圖面積，8 位小數。
- `color_bad`：`bad_color=1`、`good_color=0`。
- `wrinkle`：`true(有皺褶)=1`、`false(沒皺褶)=0`。
- 四個 `*_area_ratio`：同類 instance masks 的 pixel union 除以可見 fruit ROI pixel，8 位小數。重疊 pixel 只算一次。
- 四個 `*_max_confidence`：該類 instance 的最高 confidence，6 位小數。
- 成功但未偵測到某瑕疵時，該類 ratio 與 confidence 都是 `0`。
- ROI、推論或 mask 計算失敗時，全部模型特徵留空，並寫入 `status=error` 與 `error_reason`。
- `grade` 只能留空或填整數文字：`0=加工`、`1=下等`、`2=中等`、`3=上等`。
- CSV 使用 UTF-8 BOM，方便以 Excel 直接開啟繁體中文內容。

第一版 XGBoost baseline feature allowlist：

```text
fruit_area_ratio
color_bad
wrinkle
abrasion_area_ratio
anthracnose_area_ratio
insect_bite_area_ratio
insect_track_area_ratio
```

`fruit_id`、`view_id`、`split`、`status`、`error_reason`、`grade` 與所有 confidence 欄位不可在第一版餵入 XGBoost。

## 輸出與版本

```text
decision_datasets/v001/
├── decision_dataset.csv
├── decision_dataset.schema.md
├── dataset_manifest.json
├── validation_errors.csv
└── processing.log
```

`dataset_manifest.json` 保存來源照片的 `fruit_id:view_id`、split 與 SHA-256，以及完整 Model Bundle snapshot 和實際前處理參數。CSV 不保存照片路徑或 hash。

- 同一 Dataset Version 必須維持完全相同的 Model Manifest hash。
- 更新任一權重時建立新版本。`--labels-from` 只複製來源照片 hash 相同的 grade。
- 五種公開操作模式是 `resume`、`only_errors`、`rerun_fruit`、`rerun_view`、`replace_image`；GUI 與 CLI 共用同一個 `GenerationRequest`。
- `resume` 略過既有成功與錯誤列，只補缺少列；`only_errors` 只重試錯誤列。
- `rerun_fruit` 重跑指定果實三張照片；`rerun_view` 重跑指定單張。
- `replace_image` 更新照片 hash、重跑指定範圍，並清空受影響列的 grade。
- 每顆果實處理完立即原子寫入 CSV。局部失敗繼續批次，最後以非 0 exit code 回報。
- GUI 的安全停止只在果實 checkpoint 邊界生效：完成目前果實最多三張照片並寫入 CSV 後，不再開始下一顆。

## 驗收

1. Layout dry-run 不改動檔案，apply 保留原 split 並產生對照表。
2. 不完整模型、錯誤 task、classes、hash、threshold 或未核准狀態會阻擋正式產生。
3. 同一顆果實的三張照片以 batch inference 執行。
4. 同類瑕疵重疊 mask 採 union，無瑕疵與推論失敗可明確區分。
5. CSV 可續跑、單張重跑、整顆重跑、錯誤重試、照片替換與跨版本複製 grade。
6. 測試須在 `PF` Conda 環境使用 `python -s -m unittest discover -s main\tests` 通過。
