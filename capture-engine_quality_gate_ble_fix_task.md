# 作業指示書: capture-engine — 品質ゲート調整（narrow-steady-spur に定常性条件を追加・案1）

## 役割
あなたは sigscan プロジェクトの capture-engine エージェント。
本指示は、先の調査で機序が確定した「品質ゲートが間欠 BLE を弾く問題」への**対症の修正**。
**1点だけ**直す:
- **案1**: `narrow-steady-spur` 判定に**「定常性(persist)」条件**を効かせ、
  **「細くて"定常"」だけをスプリアスとして弾き、「細いが"間欠"」の BLE は弾かない**ようにする。

**これは "steady(定常)" という判定名が、実装では定常性を見ずに"細さ"だけで弾いていた**という
設計と実装のズレを、名前どおりに正す修正。**既存のスプリアス除去（persist が高い定常スプリアスを
弾く挙動）は完全に維持**する。BLE（間欠）だけを巻き添えから救う。

## 背景（調査で確定済みの事実 — これが根拠）
- 実機で、HackRF スプリアス（16MHz 間隔・BW0.5MHz・SNR41-44dB）と BLE（狭帯域・間欠）を観測。
- **決定的な差**: persist（滞在観測中に信号が出た割合）が
  - **スプリアス = persist≈1.00**（全観測に出る＝定常・常時オン）
  - **BLE = persist≈0.26**（19観測中5回＝間欠）
  → 時間構造で両者は**明確に分離できる**。
- しかし現 `narrow-steady-spur` 判定は、調査により **「帯域が細い（BW < narrow 閾）かつ bw_median が
  1ビン級」といった"細さ"条件のみで弾いており、"定常性(persist)"を条件に入れていない**ことが判明。
  → 名前は "steady" なのに定常性を見ていないため、**細い BLE（間欠）も一緒に弾いている**。
- 品質ゲートは6継ぎ目には含まれないが（6継ぎ目= spec.render / sdr / dsp / classify.classify /
  sigmf_io / store.Store）、**収集の心臓部であり、凍結同等の慎重さで扱う**。

## 最重要原則（絶対厳守）
1. **6継ぎ目不可侵**: spec.py / sigmf_io.py / sdr / dsp / classify / store のシグネチャを変更しない。
   `git diff --stat -- spec.py sigmf_io.py` が空。
2. **既存のスプリアス除去を壊さない（最重要）**: persist が高い定常スプリアス（persist≈1.0・
   16MHz 間隔の櫛等）は**引き続き narrow-steady-spur として弾かれる**こと。今回の変更は
   「間欠(persist が低い)信号を除外対象から外す」だけ。**定常スプリアスの除去挙動は不変**。
3. **narrow-steady-spur 以外の drop を変えない**: dc-spike 判定・low-persistence 判定・comb 判定・
   その他の品質ゲートは**一切触らない**（それらは別途・別タスク）。今回は narrow-steady-spur の
   定常性条件追加のみ。
4. **最小実装・スコープ厳守**: persist 条件の追加だけ。閾値の大幅変更・新ゲート・リファクタは
   しない。迷ったら最小。
5. **quality のインターフェース**: quality の評価関数（`quality.evaluate` 等）が既に persist を
   受け取れるか事前確認する。**受け取れるなら引数追加は不要**（内部条件の追加のみ）。もし persist が
   関数に渡っていない場合、シグネチャ追加が要る → その場合は「6継ぎ目ではないが心臓部なので、
   シグネチャ変更が最小で後方互換（デフォルト付き）であることを明示し、報告する」。勝手に大改造しない。
6. **読み取り専用データ**: captures/ の `*.sigmf-*` を書き換えない。メタは cp932 経路。UTF-8 決め打ち禁止。

## 事前確認（実装前に行い、報告に含める）
1. `quality.py` の `narrow-steady-spur` 判定箇所を再掲し、**現在の条件式**を正確に報告
   （どの帯域幅閾で・bw_median をどう使い・"steady" 相当の条件があるか無いか）。
2. **persist が判定関数に渡っているか**を確認:
   - 渡っている → 内部で persist 条件を足すだけ（シグネチャ不変）。
   - 渡っていない → どこで persist が計算され（dwell/scheduler）、quality にどう渡すのが最小かを報告。
     **シグネチャ追加が必要なら、デフォルト引数で後方互換にする案**を提示（実装は承認後でなく、
     この指示の範囲でよいが、シグネチャを変えた事実と後方互換性を完了報告で明示）。
3. persist の定義域（0.0-1.0）と、スプリアス/BLE の実測値（≈1.00 / ≈0.26）を再確認。
   **定常性の閾値をどこに置くか**の候補を報告（例: persist >= 0.9 を「定常」とみなしてスプリアス扱い、
   等。閾は控えめ＝スプリアスを確実に含み BLE を確実に外す値を選ぶ）。
4. `tests/test_quality.py` の現在の期待を読み、narrow-steady-spur に依存するテストを棚卸し
   （ベースライン 154 passed, 3 skipped を着手前に確認）。

## 実装内容
### narrow-steady-spur に定常性(persist)条件を追加（quality.py）
- 現在の「細さ」条件に、**AND で「persist が高い（定常）」条件を加える**。
  - 例（実装者判断で最小に）: `is_narrow_steady_spur = (細い条件) and (persist >= PERSIST_STEADY_THRESH)`。
  - `PERSIST_STEADY_THRESH` は、スプリアス(≈1.00)を確実に含み、BLE(≈0.26)を確実に外す控えめな値
    （例 0.9。事前確認の実測に合わせて決め、報告する）。
- これにより:
  - **定常スプリアス（persist≈1.0・細い）→ 従来どおり narrow-steady-spur で弾く**（除去維持）。
  - **間欠 BLE（persist≈0.26・細い）→ narrow-steady-spur に該当しなくなり、弾かれない**（救済）。
- **他の drop 条件・返り値の形は変えない**。narrow-steady-spur の該当判定に persist の AND を足すだけ。
- 閾値は定数として明示（マジックナンバーを避け、コメントで根拠＝スプリアス/BLE の persist 実測差を残す）。

### テスト（既存無変更・追加のみ／例外は最小）
- 追加:
  - 「細い＋persist高(≈1.0)」→ narrow-steady-spur として弾かれる（**スプリアス除去の維持**を固定）。
  - 「細い＋persist低(≈0.26)」→ narrow-steady-spur に**該当しない**（**BLE 救済**を固定）。
  - 「細い＋persist中間」→ 閾値の境界挙動（閾未満は救済・以上は除去）を1つ。
- **既存テストは無変更で緑**（154 passed 維持）。もし narrow-steady-spur の既存テストが
  「persist を与えず細さだけで弾く」前提なら、その期待が変わり得る → **変える場合は最小で、
  旧期待→新期待・理由を完了報告に列挙**（勝手に消さない・弱めない。定常スプリアス除去の意図は保つ）。
- test_seams（6継ぎ目シグネチャ）は無変更で緑。

## 検証（実装後に必ず行う）
1. `& $py -m py_compile quality.py`（＋触った他ファイル）OK。
2. `& $py -m pytest -q` 全緑（既存 154 + 新規）。test_seams 緑。
   `git diff --stat -- spec.py sigmf_io.py` が**空**。
3. **回帰の実証（report 用）**: 修正前後で、以下を数値/挙動で示す:
   - **定常スプリアス（persist≈1.0・BW0.5MHz）が、修正後も narrow-steady-spur で弾かれる**こと
     （除去維持の証拠）。合成 or 実データの該当ケースで。
   - **間欠 BLE 様（persist≈0.26・狭帯域バースト）が、修正後は narrow-steady-spur で弾かれない**こと
     （救済の証拠）。
   - 可能なら、今日の実データ（16MHz 間隔スプリアスを含む収集）に対し、修正版の quality を通して
     「スプリアスは drop・BLE 様は残る」挙動を確認（captures/ は書き換えず、読み取りで評価）。
4. captures/ の `*.sigmf-*` 無変更（件数明示）。
5. narrow-steady-spur **以外**の drop（dc-spike / low-persistence / comb 等）の挙動が**不変**であること
   （それらのテストが緑のまま＝触っていない証拠）。

## 完了報告に含めること
- 事前確認（narrow-steady-spur の現条件式・persist が関数に渡るか・閾値候補・既存テスト棚卸し）。
- 変更点（persist 条件の追加・閾値と根拠）。（もしシグネチャを足したなら）その事実と後方互換性の明言。
- **既存スプリアス除去の維持の証拠**（persist≈1.0 の定常スプリアスが修正後も弾かれる、before/after）。
- **BLE 救済の証拠**（persist≈0.26 の間欠 BLE が修正後は弾かれない、before/after）。
- narrow-steady-spur **以外**の drop が不変であることの明言。
- **6継ぎ目不変**（spec/sigmf_io diff 空、test_seams 緑）。
- 追加テストの内容と、（あれば）更新した既存テストの一覧（旧→新・理由）。
- pytest 結果（N passed）・captures 無変更。
- 将来課題の再確認: dc-spike は範囲設定で回避・low-persistence 閾調整は別タスク・
  「間欠信号を1周波数滞在で捉える」設計の根本見直しは task の外。

## やってはいけないこと（禁止事項の再掲）
- spec.py / sigmf_io.py / sdr / dsp / classify / store の6継ぎ目シグネチャの変更。
- **既存のスプリアス除去を弱める/壊すこと**（persist≈1.0 の定常スプリアスは引き続き弾く）。
- narrow-steady-spur **以外**の品質ゲート（dc-spike / low-persistence / comb 等）の変更（別タスク）。
- 閾値の大幅変更・新ゲートの追加・リファクタ・スコープ拡張（persist 条件の追加のみ）。
- captures/ の `*.sigmf-*` の書き換え。メタを UTF-8 決め打ちで開くこと。
- 既存テストの安易な削除・弱体化（意図を保った最小修正のみ・要報告）。test_seams の変更。
