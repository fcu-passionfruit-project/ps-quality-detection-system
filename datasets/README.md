# Dataset manifests

此目錄只保存可追蹤的 dataset manifest 與交接說明，不保存照片、標註影像或 Roboflow 匯出資料。

本機原始照片實體保存在 `D:\passion-fruit-datasets\`，模型 repository 根目錄的 ignored `dataset/` junction 指向該資料夾。建立不可修改的 dataset snapshot 後，複製 `manifest.example.yaml`，填入新的 dataset ID、實體／workspace 路徑、類別分布與 checksum。尚未交給 Roboflow 切分時使用 `split_status: pending_roboflow` 與 `splits: null`，不得填入虛假的 train／valid／test 數量。

準確率報告必須同時引用 dataset ID、模型 bundle version 與 Git commit。Junction 只是本機存取入口，不是第二份備份；原始資料仍須另行上傳到 Roboflow 或團隊共用儲存空間。
