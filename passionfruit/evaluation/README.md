# 模型與實驗評估

`analyze_targeted_crop_experiment.py` 比較 v19／v20／v21；`evaluate_experiment_d_area.py` 比較 D0／D1／D2 的瑕疵面積精度及 threshold。

入口：`python -s -m passionfruit.evaluation.analyze_targeted_crop_experiment --help`、`python -s -m passionfruit.evaluation.evaluate_experiment_d_area --help`。

依賴既有模型、資料集、PyTorch、Ultralytics、Matplotlib；面積評估重用 inference 的遮罩規則。評估產物寫入原本指定的輸出目錄，不放在這裡。
