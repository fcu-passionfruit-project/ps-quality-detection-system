# 局部瑕疵逐類像素與面積評估契約

入口為 `python -s -m passionfruit.evaluation.evaluate_local_defect_pixels`，獨立於正式推論。
不訓練、不修改權重、Dataset、split、Django 或決策層欄位。

- `--weights` 接受任意 segmentation checkpoint，但必須符合 Dataset 的四類 index/name 契約。
- `--data` 接受 data.yaml 或其目錄；固定 split 分開計算，不用 train 評估模型。
  資料稽核涵蓋 train/valid/test。支援此專案 Roboflow 匯出的 `../split/images`
  與實際 `YAML-directory/split/images` 差異，逐 split 記錄 resolved path，其他錯誤不得猜測。
- 每張圖固定四列，保留全部 GT polygons。合法空 .txt 為陰性；缺檔、孤立標註、
  非法座標／class／退化 polygon 為錯誤，輸出 dataset_audit.json／failure.json 並停止。
- 同類預測與 GT 各取 Boolean OR，不做 instance matching；不同類別獨立。
- `django` preset 固定 imgsz=1024、conf=0.25、iou=0.7、max_det=300、retina_masks=False、
  rect=True、augment=False、agnostic_nms=False。讀取 sibling Django settings 的 literal options
  核對，並記錄 settings/detection source SHA；只讀載入原始純 helper，不初始化 Django、不啟動硬體或服務。
  YOLO26 end-to-end 時 iou 可能不參與傳統 NMS；它不是本工具的 pixel 評估門檻。
- `--conf` 覆寫明標為 sensitivity；不自動挑 threshold、不在 test 做調參。
- 預設 batch=3、固定排序；既有資料無可靠三視角組合，不保證與現場 batch padding 相同。
- `--device auto` 為 CUDA → MPS → CPU；指定不可用裝置直接失敗，不靜默換裝置。
- `--limit N` 僅取各 split 前 N 張作 smoke，正式評估移除此參數。

`django-ellipse` 重用 inference.create_circular_mask(shape, 1.01)，推論前套用該遮罩，
GT/pred 都限制在同一 ROI，分母為 binary mask 非零像素。它與 Django crop 幾何公式相同，
但沒有逐圖原始 crop provenance，不能證明等同歷史輸出或真實果皮輪廓。
`full-image` 是明確命名的替代分母（不套 ellipse），不得稱為正式面積誤差。
零 ROI 是資料錯誤，不能當成無瑕疵。

`--mask-mode scale-masks` 是相容舊版的預設：retina_masks=False 使用已安裝官方 `scale_masks(..., mode="nearest")` 移除 padding 後還原原圖。
沒有 mode 參數的舊版本須明確報錯，不自行更換 dependency。已驗證環境為 Ultralytics 8.4.122。
retina=True 的內部還原函式只接受原圖尺寸；CLI preset 保持 False。
`--mask-mode django-deployed` 使用原 dtype 的 cv2.resize(INTER_NEAREST) → >0.5 → ROI 限制與同類 OR，
不移除 padding；需要 django-ellipse，不能搭配 full-image 冒稱正式模式。
原始 _segmentation_summary 透過唯讀 import 載入；其相對依賴 dataset_store/grading 只有定義與常數，
不呼叫模型、硬體或 XGBoost。載入期間禁止產生 bytecode，不修改 sys.path。
該函式不回傳聯集，因此暫時以 return-frame profiler 讀取其實際 union_masks，finally 恢復既有 profiler；
不修改 Django 原碼、AST、globals、cv2 或輸入 tensor。每張原始 Results 都驗證聯集逐像素相同及特徵比例精確一致，失敗即停止。

`django_deployed_area_ratio` 完整重現 round(min(1.0, union_pixels/roi_pixels),8)，
`django_deployed_absolute_area_error` 是該特徵對原始 GT ratio 的誤差。
為保持既有指標，主 pred_area_ratio、MAE 仍以未捨入 pixel_count/roi_pixels 計算，不把 rounding 混入主指標。
每份新 CSV、JSON、報告、圖像皆標明 mask_mode；summary 按 mode 分組，不混算。
兩種聯集由同一份 Results 生成並放入 cache，原始 boxes/masks fingerprint 可確認分開執行模式時輸出相同。
既有結果不補寫 mode；比較時明確視為 scale-masks，不能覆寫舊 cache 或歷史報告。

MAE_all 是全部圖片 AE 平均；GT-positive/negative MAE 用各自子集樣本數。
Bias = mean(pred ratio - GT ratio)。ratio×100 為 %，error×100 為 pp。
Pixel P/R/F1 先累加跨圖片 TP/FP/FN 再計算 micro，不能平均每張或每個 instance 的 P/R。
GT/pred 全空時 P/R/F1=N/A；GT>0/pred=0 時 P/R/F1=0；GT=0/pred>0 時 P=0/R=N/A/F1=0。
空子集 N/A，無 epsilon；CSV 空欄、JSON null。

輸出必須是全新空目錄：每圖每類 CSV、從其重讀重算的 summary、完整 source/weight/data SHA、
actual predictor args、原圖座標 mask cache、對齊診斷、代表圖與繁體中文報告。
seed=0、固定排序與 CPU threads，啟用 deterministic algorithms；跨硬體／版本不承諾 bitwise 相同。
檔案與 Git LFS pointer hash 必須區分；pointer 的 OID 可核對版本，不能視為已驗收 Django runtime。

驗收測試見 `tests/evaluation/test_local_defect_pixels.py`；正式全量執行前必須先 smoke 並檢視圖像。
與 Django 的原函式相容性測試見 `tests/evaluation/test_django_deployed_pixels.py`；缺 sibling 原始碼會明確 skip，不能宣稱相容性驗收成功。
同 Results 與指定 ROI 的一致性不代表端到端驗收：資料缺原始照片、歷史 crop bbox、前處理版本與可靠現場三視角 grouping，
也未重新定位果實或驗證現場硬體／部署權重載入。幾何 ROI 相同不能證明歷史影像曾以同規則產生。
不得將良好面積 MAE 或 Micro P/R 直接解讀成 Image Grade／Fruit Grade 準確率。
