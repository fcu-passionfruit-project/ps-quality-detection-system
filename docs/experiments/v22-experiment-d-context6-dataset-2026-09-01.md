# Experiment D：Paired Context Scale Dataset

- 建立日期：2026-09-01
- Baseline Dataset：`training/packages/local_defects`
- Pairing reference：`experiments/exp_C_targeted_crop`
- Experiment D：`experiments/exp_D_context6`
- 預定 run：`training/runs/segment/v22_D_context6`
- 狀態：Dataset validation PASS；尚未啟動 YOLO training

## Controlled variable

唯一主動修改的 Dataset generation 參數：

```text
nominal context_scale: 4.0 -> 6.0
```

target selection、target repetition、seed、context variation distribution、center jitter
distribution、minimum crop size、maximum crop fraction、boundary handling、polygon clipping、
crop naming、original train images 與 valid／test 均維持 Experiment C 契約。total class
exposure 的變化來自較大 field-of-view 自然納入 neighboring annotations；沒有刪除或修改
合法 labels。

## Experiment C replay

generator 以 C 的參數與 global `seed=0` 重跑 stochastic sequence，將每筆 accepted
attempt 的 context variation delta、normalized jitter、attempt index 與 rectangle 與既有
manifest 比較，再由 source image 重建 crop pixels 與 label bytes。

| Replay check | Result |
| --- | ---: |
| target record sequence | 132/132 SAME |
| crop rectangles | 132/132 SAME |
| manifest geometry | 132/132 SAME |
| generated image geometry and pixels | 132/132 SAME |
| labels | 132/132 SAME |

## C → D stochastic pairing

D 每筆第一個 attempt 使用 C 對應 record 的 context variation delta 與 normalized jitter，
再由 nominal scale 6.0 重新計算 geometry。sequence 92 的 paired first attempt rectangle
`[0, 0, 238, 238]` 與同一 target 的 sequence 91 重複，因此使用由 seed、stable record ID
與 retry index 導出的 record-local stream retry 一次；後續 records 沒有 RNG cascading。

| Pairing item | Result |
| --- | ---: |
| context variation paired | 131/132 |
| jitter X paired | 131/132 |
| jitter Y paired | 131/132 |
| `D actual scale = C actual scale + 2.0` | 131/132 |
| records with retry | 1 |
| records unable to remain fully paired | 1 |
| D boundary/clamp active | 75 |
| C→D boundary/clamp status changed | 23 |
| duplicate source rectangles after retry | 0 |
| exact duplicate generated crop images | 0 |

## Dataset structure and target exposure

| Item | Count |
| --- | ---: |
| original train images | 433 |
| generated crops | 132 |
| total train entries | 565 |
| unique target instances | 80 |

| Target class | C unique targets | D unique targets | C target records | D target records |
| --- | ---: | ---: | ---: | ---: |
| anthracnose | 54 | 54 | 54 | 54 |
| insect_bite | 26 | 26 | 78 | 78 |

## Total and neighboring exposure

| Class | v19 total | C total | D total | D-C |
| --- | ---: | ---: | ---: | ---: |
| abrasion | 577 | 614 | 637 | +23 |
| anthracnose | 54 | 128 | 151 | +23 |
| insect_bite | 26 | 105 | 108 | +3 |
| insect_track | 386 | 430 | 445 | +15 |

| Class | C crop target | C neighboring | D crop target | D neighboring | D-C neighboring |
| --- | ---: | ---: | ---: | ---: | ---: |
| abrasion | 0 | 37 | 0 | 60 | +23 |
| anthracnose | 54 | 20 | 54 | 43 | +23 |
| insect_bite | 78 | 1 | 78 | 4 | +3 |
| insect_track | 0 | 44 | 0 | 59 | +15 |

## Integrity validation

- original train images preserved once；generated sources 全部來自 train。
- train／valid／test exact image SHA-256 overlap 全部為 0。
- C／D source split manifest SHA-256 相同：
  `62bf967530434e7bf4bec681874ac53f294df0978989032dd920a296f42069ea`。
- malformed labels、out-of-range coordinates、missing image／label、missing target polygon
  全部為 0。
- valid／test、class mapping、`nc` 與 Roboflow metadata 和 baseline 相同。
- 連續兩次 D generation 的完整 artifact tree digest 相同：
  `59459bc1baa20f71fb762cb5dcec7a0ba3eabc684738995e82e2361d81fed9fa`。

## Reproducibility hashes

| Artifact | SHA-256 |
| --- | --- |
| Experiment D training-input Dataset | `8d17834c7e768d2b492cdbfcb3c5479756da4b5fe72bc58ef3adcc26a996ba0b` |
| source Dataset | `be2b7701c799abe1dad31fba2b9deb35c4a8770b118b86e6866d5bd45f6ae42e` |
| train list | `8d685ded3900d3ff3c9823b21188a5f2033feb0f6bcd015573481e00d9c8c22f` |
| crop manifest | `07b57c720725bcd590cab5cd8605f8b487fcfcfd98617c08ce4b96cb3be2f419` |
| paired parameters | `407c9de7bd72b45cd017277b35eac14006c5b487bc759df3350201ff6e868ea3` |
| C replay report | `c2579143e521f79189060dd75eb33b81e48cde3e9fc5a187bb2b981339f40cc1` |
| C vs D audit | `8f447b471b938024c8415f5ccc8eb8d44b0c5931830f645f16290ef1207ad570` |
| initial `yolo26s-seg.pt` | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` |

## Planned training command

本次沒有執行 training。Dataset 通過 review 後，由開發者手動執行：

```bash
yolo task=segment mode=train \
  model=yolo26s-seg.pt \
  data=training/packages/local_defects/data_exp_D.yaml \
  project=training/runs/segment \
  name=v22_D_context6 \
  imgsz=1024 \
  epochs=200 \
  patience=50 \
  batch=4 \
  device=mps \
  seed=0 \
  deterministic=True \
  cls_pw=0.5 \
  copy_paste=0.1 \
  copy_paste_mode=mixup \
  optimizer=auto
```
