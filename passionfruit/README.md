# 百香果檢測核心程式

此目錄是專案的 Python 命名空間，不需安裝成套件。

| 子目錄 | 用途 |
| --- | --- |
| inference | 模型載入、影像處理、Pipeline |
| decision_dataset | 決策資料集結構、Manifest 與 CSV |
| data_preparation | 分割訓練資料準備 |
| evaluation | 模型與實驗評估 |
| explainability | EigenCAM、D-RISE |
| third_party | 內附第三方程式 |

依賴由 apps 指向核心。decision_dataset 與 evaluation 可使用 inference；inference 不依賴 GUI 或實驗工具。
