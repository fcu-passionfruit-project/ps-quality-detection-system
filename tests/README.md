# 自動化測試

此處驗證程式正確性，不保存模型訓練或測試結果。

從專案根目錄執行：`python -s -m unittest discover -s tests -t .`。子目錄依 apps、inference、decision_dataset、data_preparation、evaluation、explainability 分類。測試以暫存資料、假模型或 mock 驗證，不使用正式資料集來寫入測試產物。

D-RISE 的兩項第三方原語測試需要額外依賴；缺少時既有測試會報 skip，驗收時應安裝依賴並確認沒有 skip。完整 GPU 模型評估是另一項工作，不能由這套測試取代。
