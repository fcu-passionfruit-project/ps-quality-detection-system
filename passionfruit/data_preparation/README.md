# 訓練資料準備

`generate_class_aware_train_list.py` 調整訓練圖片曝光次數；`generate_targeted_crops.py` 建立 C／D 瑕疵裁切資料及紀錄。兩者均不啟動模型訓練。

入口：`python -s -m passionfruit.data_preparation.generate_class_aware_train_list --help`、`python -s -m passionfruit.data_preparation.generate_targeted_crops --help`。

依賴 NumPy、Pillow 與 PyYAML；工具以原始 splits 為輸入，輸出清單、裁切圖片、標註與 YAML。
