# 作業指示書: capture-engine — review.py を (C) 対応にする最小改修【再実装版】

## 役割
あなたは sigscan プロジェクトの capture-engine エージェント。
本指示は、**CNN 監査で (C) になった記録**を review.py が選別・提示できるようにする**最小改修**。
**確定ロジック（apply_label）とメタ書き戻しは一切変更しない。** 既存の rule<conf_max レビュー導線も無改変。

## 背景（重要・経緯）
- この改修は一度実装されたが、**コミット前に環境事故で失われた**（未コミットのまま clone 復元したため）。
  現在の `review.py` には `--verdict` が無い（`review.py --verdict C` は「unrecognized arguments」エラー）。
  **今回は必ずテストを付け、完了後にコミットする**（同じ轍を踏まない）。
- M3 接続後、記録は監査を通ると `sigscan:method='cnn'` に書き換わり、不整合は global の
  `sigscan:cnn_verdict=='C-conflict'` に記録される。
- 現 review.py の対象は「`sigscan:method=='rule'` かつ `sigscan:confidence < conf_max`（既定 0.5）」のみ。
  このため (C)（method='cnn'）を拾えない。これを `--verdict C` フラグで拾えるようにする。
- **対象データは「今日の実機収集で captures/ に入った実データ」**（旧 2433MHz 記録は失われ、別途再収集済み）。
  現在の captures/ には CNN 監査を通った記録（method='cnn'・global に cnn_verdict/cnn_class/cnn_conf）が
  複数あり、その中に `cnn_verdict=='C-conflict'`（用途ラベル='未識別信号'）の記録が含まれる。
  → 今回はこの**実データで `--verdict C` が実際に (C) を列挙できること**を実地確認する。

## 現在の環境（前回と変わった点）
- **統合 conda 環境 `sigscan`**（Python 3.11・numpy 1.26.4・torch GPU・SoapySDR）で動作。
  Python 実行は **フルパス**（`& $py ...`、`$py = C:\Users\puppy\radioconda\envs\sigscan\python.exe`）。
  ※ Microsoft Store のダミー python 回避のためフルパス運用。
- **pytest ベースラインは 147 passed, 3 skipped**（BW 修正 +4、GPU 対応 +3 を含む最新）。
- captures/ に**実データがある**（前回は 0 バイトで検証不能だったが、今回は実地確認できる）。

## 最重要原則（絶対厳守）
1. **凍結契約不可侵**: spec.py / sigmf_io.py を変更しない。6 継ぎ目のシグネチャ不変。
   `git diff --stat -- spec.py sigmf_io.py` が空。
2. **apply_label は不変**: 人間が確定したときの書き戻し
   （`core:label`＝新ラベル / `sigscan:method='human'` / `sigscan:confidence=1.0` /
   元ラベルを `core:comment` に provenance 保存）は**一切変更しない**。確定の仕組みは既に正しい。
3. **後方互換**: 既定挙動（`rule` かつ `conf<conf_max`）は**無変更**。新機能は**明示フラグでのみ**有効化。
4. **読み取りのみ（選別・提示）**: メタは sigmf_io の read 経路（locale=cp932）で開く。UTF-8 決め打ち禁止。
   選別・提示で captures/ を書き換えない。書き込みは既存 apply_label の**対話確定時のみ**（今回ロジック変更なし）。
5. **しきい値 0.5→0.7 は変更しない**（別途人間承認の別タスク）。今回は触れない。
6. **最小実装**: 画像ビューア内蔵・フィルタ・並べ替え等の新機能は作らない。PNG は**パス表示のみ**。

## 事前確認（実装前に行い報告に含める）
1. review.py の選別箇所（method/confidence で絞っている行）と、`--list`／対話（apply_label 呼び出し）の
   分岐を再確認。選別ロジックを**純関数として切り出せるか**を判断（できれば純関数化してテスト容易に）。
2. 提示行を生成している箇所を特定（`--list` 相当 / 対話表示の両方）。
3. ベースライン pytest（**147 passed, 3 skipped**）を着手前に確認。
4. **今日の実データの確認**: captures/ の `*.sigmf-meta` を数件読み、global に
   `sigscan:cnn_verdict`（'C-conflict' / 'A-consistent' 等）・`sigscan:cnn_class`・`sigscan:cnn_conf` が
   実際に載っていること、`cnn_verdict=='C-conflict'` の記録が**実在するか**を確認して報告
   （実在しなければ、その旨と、代わりに検証可能な記録を報告）。

## 実装内容
### (1) (C) 選別フラグ
- `--verdict C`（内部的に `C-conflict` を意味する）を追加。指定時は `<dir>/*.sigmf-meta` を走査し、
  **global の `sigscan:cnn_verdict == 'C-conflict'`** の記録を対象列にする。
- **フラグ未指定時は従来どおり**（`method=='rule'` かつ `confidence < conf_max`）。挙動を一切変えない。
- 排他で最小に: 「`--verdict C` 指定時は C-conflict のみ／未指定時は従来のみ」。
- global にキーが無い古い記録は安全に除外（エラーにしない）。

### (2) 提示の拡充（`--list` と対話の両方／既存項目は残す）
対象行に、テキストで以下を追加（取れない値は省略＝後方互換、エラーにしない）:
- **CNN 来歴**: `cnn=<cnn_class>@<cnn_conf> verdict=<cnn_verdict>`（global の `sigscan:cnn_*` から）。
- **保持候補／ルール仮説**: `core:comment` の "用途=Unknown(候補つき)" 等、監査前の候補が取れれば提示
  （構造化キーが無ければ comment 文字列の該当部分でよい。取れる範囲で）。
- **PNG パス**: `captures/_images/<base>.png`（ファイルが存在すれば表示。**内蔵表示はしない**）。

### (3) テスト（既存無変更・追加のみ・高速）
- `--verdict C` の選別（純関数化できたらその関数）に対し: (i) `cnn_verdict=='C-conflict'` を拾う、
  (ii) `A-consistent` や verdict 無しを除外する、を検証。
- 既定挙動（`rule` & `conf<conf_max`）が**無変更**であること（既存テストは無改変で緑。
  純関数化したなら従来条件の回帰テストを1つ追加）。
- **apply_label を呼ばない／変更しないこと**を担保（提示・選別のテストで書き戻しが走らないこと）。

## 検証（実装後に必ず行う）
1. `& $py -m py_compile review.py` OK / `& $py -m pytest -q` 全緑（**既存 147 + 新規**）/
   `git diff --stat -- spec.py sigmf_io.py` が**空**。
2. 実走（**確定入力はしない・列挙の確認まで**）:
   - `& $py review.py captures/ --verdict C --list` で、**今日の実データの (C) 記録が列挙される**こと、
     各行に CNN 来歴・候補・PNG パスが出ることを確認。
     before（フラグなし）の件数 → after（`--verdict C`）の件数 を明示（C 記録が実在すれば after>0）。
   - 対話モードで `--verdict C` を起動した場合も、**ラベルを入力せず**に対象が列挙されることだけ確認して抜ける。
3. `& $py review.py captures/`（フラグなし）が**従来どおり**であること（後方互換）。
4. captures/ の `*.sigmf-*` が無変更（件数で明示）。`sigscan:method=='human'` の件数が baseline から**不変**
   （＝この検証では何も確定していない）。

## 完了報告に含めること
- 事前確認の結果（選別箇所 / 純関数化したか / 提示箇所 / **今日の実データに C-conflict 記録が実在したか**）。
- 追加したフラグと選別条件、提示に足した行（CNN 来歴・候補・PNG パス）。
- **apply_label を変更していないことの明言。**
- `--verdict C` 実走の列挙結果（実データの (C) 記録が出た before/after）。実在した記録名も。
- 既定挙動の後方互換確認結果。
- pytest 結果（N passed）・凍結 diff 空・captures 無変更・`method='human'` 件数不変。

## やってはいけないこと（禁止事項の再掲）
- apply_label / メタ書き戻しロジックの変更。確定の意味づけ（`method='human'` / `conf=1.0` / provenance）の変更。
- spec.py / sigmf_io.py / classify / dsp の変更。
- しきい値 0.5→0.7 の変更（別タスク・人間承認制）。
- 画像ビューアの内蔵、フィルタ・並べ替え等の新機能（最小実装に留める）。
- 帯域幅バグの修正（別タスク。今回は触れない）。
- **レビュー中に実際のラベル確定を自動で行うこと**（確定は人間が対話で行う。エージェントは確定しない）。
- メタを UTF-8 決め打ちで開くこと。既存テストの変更・削除。
