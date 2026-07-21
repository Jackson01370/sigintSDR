# sigscan 作業ログ — 2.4GHz ISM 専門家 CNN 再学習（best-val / early-stop / k-fold）

> 実験ノート（人間が経緯を追う）。機械可読の成果物は `runs/ism24_v2/`。
> 更新規律: セッションごとに1エントリ（狙い / やったこと / 分かったこと / 保留 / 成果物 / 次アクション）。
> 常設ルールは `CLAUDE.md`（案A＝合成非混合 / Pattern A 禁止 / 凍結契約 / CC は確定しない）。

---

## エントリ 2026-07-21: v2 再学習（データ 91→192 件・best-val/early-stop/k-fold 導入）

### 狙い
前回 v1（`runs/ism24_v1`・91件・単一 val 18件で final 94.4%）の弱点を潰す:
1. **best-val 保存**（v1 は最終 epoch＝過学習後の重みを保存していた）。
2. **early-stopping**（過学習が始まったら止める）。
3. **k-fold 交差検証**（v1 の val=18件は 1件で 5.6% 動く脆さ。平均±分散で信頼できる評価に）。
既存 `runs/m2_5`（汎用）・`runs/ism24_v1`（前回専門家）は非改変。ルーティングは今回もやらない。

### やったこと
- 実データ件数を再計測（`captures/` 走査・読み取りのみ）: **ble-adv 60 / wifi-24 93 / spurious 39 = 192件**（v1 の 36/23/32=91 から +101）。教師は `method=human` の実データのみ（合成・ルール/CNN 出力を索引段で除外＝案A / Pattern B 純粋）。
- `cnntrain/train_expert.py` に**追加のみ**で改善を実装（既存 sim 経路・v1 の基本経路は不変）:
  - `_train_loop`: best-val 重みを CPU スナップショット保持＋patience による early-stop。既存の凍結 `_epoch_pass` を流用。
  - `stratified_kfold` / `run_expert_kfold`: 層化 k-fold（k=5）を best-val+early-stop で回し、平均±標準偏差・合算混同行列・クラス別 precision/recall を出す（評価専用・checkpoint は残さない）。
  - `run_expert_v2`: k-fold 評価 → 最終 checkpoint（**基準(a)**: 全データの層化 80/20 split で best-val 学習した1本）。
  - CLI: `--kfold` / `--early-stop` / `--best-val` / `--classes` を追加。
- テスト5件追加（`tests/test_train_expert.py`）。全体 **253 passed, 3 skipped**（v2前 248＋5、既存無変更）。
- 実行: `python -m cnntrain.train_expert --classes ble-adv,wifi-24,spurious --out runs/ism24_v2 --kfold 5 --best-val --early-stop 10`（RTX 3080・torch 2.5.1+cu121）。

### 分かったこと（数値は k-fold 平均±分散で読む）
- **k-fold(k=5) val accuracy = 97.4% ± 2.8%**（fold: 97.4 / 100.0 / 92.3 / 100.0 / 97.3）。
- **合算混同行列**（全 fold の val=192 を合算）:
  ```
              ble-adv  wifi-24  spurious
   ble-adv  |    57       2        1        (recall 95.0%, prec 96.6%)
   wifi-24  |     0      93        0        (recall 100.0%, prec 97.9%)
   spurious |     2       0       37        (recall 94.9%, prec 97.4%)
  ```
- **前回の弱点だった ble-adv↔wifi-24 境界は改善**: wifi-24 は 93件全て正解（recall 100%・ble と 1件も混同せず）。境界方向の誤りは ble→wifi の 2件のみ（60件中）。データ倍増（wifi 23→93）が効いた。
- **best-val 保存の効果が明確**: 最終学習で保存されたのは epoch8（val 97.4%・train acc 85%＝過学習前）。early-stop で回った最終 epoch18 の val は 92.3%。**もし v1 同様に最終 epoch を保存していたら −5.1pt** だった。early-stop は best(epoch8) の 10 epoch 後の epoch18 で発火し、無駄な 22 epoch を節約。
- **正直な注記（誇張しない）**:
  - spurious は v1 では「完全分離」だったが、v2 の合算では 2/39 を ble-adv と取り違えた（それでも recall 94.9%）。固定周波数線で本来は易しいクラスであり、易しさが実力を過大評価しやすい点は不変。
  - 難所は依然 ble-adv 周辺（ble→wifi 2・ble→spurious 1・spurious→ble 2）。ただし単一 val ではなく k-fold 平均で 97.4%±2.8% を維持。
  - 「実RF・human確定の held-out で 3 クラスを分離できる」段階であり、それを超える主張（未知環境での汎化・チャネル外変動への頑健性）は未検証。

### 保留・限界
- augmentation は今回も未導入（素の性能＋データ増の効果を切り分けるため。将来課題）。
- unknown クラスは教師0件のため 3 クラスのまま（蓄積後に4クラス化）。
- k-fold は評価専用。デプロイ checkpoint は基準(a)の1本（全データ層化 split の best-val）。

### 成果物
- `runs/ism24_v2/checkpoint.pt`（classes=['ble-adv','wifi-24','spurious']・meta に best_epoch/best_val_acc/kfold サマリ）。
- `runs/ism24_v2/kfold_report.{txt,json}`・`report.{txt,json}`・`history.json`・`train_log.txt`・`kfold_log.txt`。
- 実装: `cnntrain/train_expert.py`（追加のみ）、`tests/test_train_expert.py`（+5件）。

### 次アクション（実装せず申し送り）
(a) バンド別ルーティング（`scheduler`/`config.Band`・専門家の実運用起用）、(b) unknown クラス追加、(c) augmentation、(d) 方式軸2軸化、(e) chirp 等の未識別信号の扱い、(f) ble-adv 境界のさらなるデータ拡充。
