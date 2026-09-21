# 模型解釋與熱力圖

`eigencam.py` 產生空間 activation、遮蔽診斷及比較圖；`drise.py` 產生指定 segmentation instance 的 D-RISE 熱力圖。這些是分析工具，不是 tests 自動化測試。

入口：`python -s -m passionfruit.explainability.eigencam --help`、`python -s -m passionfruit.explainability.drise --help`。

EigenCAM 使用 third_party/yolo_cam 與 ttach。D-RISE 額外使用 vision-explanation-methods；安裝方式見根目錄 README。演算法的解讀限制見 docs/specs。
