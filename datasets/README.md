# Dataset manifests

此目錄只保存 dataset manifest 與交接說明，不保存照片、標註影像或 Roboflow 匯出資料。

建立不可修改的 dataset snapshot 後，複製 `manifest.example.yaml`，填入新的 dataset ID、儲存位置、類別分布與 train／valid／test 切分。準確率報告必須同時引用 dataset ID、模型 bundle version 與 Git commit。
