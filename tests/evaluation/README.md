# union-mask、面積與 threshold 指標測試

驗證 union-mask、面積與 threshold 指標。從專案根目錄執行 `python -s -m unittest discover -s tests/evaluation -t .`；完整驗收使用 tests/README.md 的全套指令。

`test_local_defect_pixels.py` 驗證逐類聯集、重疊與拆分不變性、零分母、位置錯誤但面積相同、
micro 與子集分母、CSV roundtrip、GT 異常、ROI 幾何及官方 unletterbox（含奇數 padding）。
指定的 1000-pixel GT / 300+250+200-pixel 預測案例驗證 precision=1、recall=0.75，
並與單一預測 mask 的結果逐欄比較。

`test_django_deployed_pixels.py` 直接呼叫 sibling Django 的原始函式，驗證非整數 Mask 的嚴格 >0.5、
最近鄰縮放、ROI、跨類與重疊聯集、8 位 rounding、空預測、類別語意、原始推論呼叫參數、
已知 crop 的幾何一致性、profiler 恢復及 mode 分組。這些是給定 ROI／Results 的相容性測試，非完整拍攝／定位驗收。

macOS 若既有 migration 測試受 `/var`→`/private/var` 暫存目錄 symlink 影響，可使用
`TMPDIR=/private/tmp python -s -m unittest discover -s tests -t .`；不因此修改正式 layout 行為。
