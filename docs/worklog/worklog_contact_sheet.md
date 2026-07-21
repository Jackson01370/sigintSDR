# sigscan 作業ログ — 確定レビューのコンタクトシート（表示補助のみ）

> 実験ノート。常設ルールは `CLAUDE.md`。関連: 確定フローは [[worklog_2.4GHz_BLE_groundtruth]]。

---

## エントリ 2026-07-21: コンタクトシート（全PNGを1枚で一覧・自動オープン）

### 狙い
`review.py --suggest --batch-confirm` で一括候補が20件並ぶと、人間は各行の `PNG: …` を
1つずつ Ctrl+クリックして開いていた（20回クリック）。これを、**対象の全 PNG を1枚の
グリッド画像（コンタクトシート）にまとめ、各サムネイルに CC 提案 `[番号] cc_class` を焼き、
対話時のみ自動で開く**ことで解消する。**純粋な見せ方の改善**で、○×の判断・確定・摩擦・
Pattern A 防波堤は一切変更しない。

### やったこと（最小実装・追加中心）
- 新規 `contact_sheet.py`: `sheet_entries`（提示順の ctx→ `[番号] cc_class` エントリ・純関数）と
  `build_contact_sheet`（既存 `captures/_images/*.png` をグリッド配置・欠損は `(画像なし)`
  プレースホルダ）。**`spec.render` を呼ばず既存 PNG を並べるだけ**（凍結表現を迂回しない）。
  matplotlib は build 内で遅延 import（`--open-sheet` 未使用の従来経路で matplotlib を引かない）。
  和文フォント fallback（view_captures 方式）でタイトルの豆腐化を回避。
- `review.py`: `--open-sheet` フラグ（既定オフ）。`run_suggest_review` に `open_sheet/sheet_out/
  open_fn/interactive` を追加（すべて既定は従来挙動）。提示順（batch→rest）でシートを生成し、
  `_maybe_open_contact_sheet` が**対話時のみ**（`_is_interactive`＝stdin/stdout が tty）
  `os.startfile` で開く。ヘッドレス（claude -p・パイプ）では開かない・生成もしない。
  captures/ 配下への出力は bench/ へ退避（`_under_captures` ガード）。既定出力は
  suggestions.csv と同じ bench/<out>/ ディレクトリの `contact_sheet.png`。
  main() は `open_sheet` を **True のときだけ注入**（include_human と同じ方式）＝既存の
  呼び出し・mock シグネチャと後方互換（既存 dispatch テストを無変更で緑に保つ）。
- `collect_review.ps1`: ステップ5（対話の人間○×）にのみ `--open-sheet` を付与。ヘッドレスの
  ステップ4-A（`claude -p`）は review.py を呼ばないため対象外。
- テスト: 新規 `tests/test_contact_sheet.py`（10件）。全体 **274 passed, 3 skipped**（実装前 264＋10）。

### 番号一致（人間が突き合わせられる肝）
シートの `[番号]` は `run_suggest_review` が組む提示順（batch→rest）の 1-based index で、
`_batch_confirm` の一括候補 `[1..N]`（batch 部分）と一致する。単体テストで固定。

### 検証（実データ・確定なし）
`review.run_suggest_review('captures/', 'bench/ble_ch39/suggestions.csv',
pattern='2479MHz_ble_ch39_*', batch_confirm=True, open_sheet=True, interactive=True,
open_fn=spy, apply_fn=spy, input='q')`:
- `bench/ble_ch39/contact_sheet.png`（111KB・実 spectrogram PNG を2枚グリッド）を生成し spy が1回オープン。
- **確定件数 0**（`q` で即終了・apply スパイは捨て・captures/ 非改変）。
- ヘッドレス（interactive=False）ではオープン0回（開かない）。
- captures/ 配下ではなく bench/ 配下に出力（`_under_captures`=False）。

### 保留・限界
- 20件超で1枚が窮屈になる可能性（cols=5 固定）。複数シート分割は将来課題（今回は1枚に詰める）。
- シート上のクリック確定・GUI 化はしない（表示補助のみ・確定は人間の○×）。

### 成果物
- 実装: 新規 `contact_sheet.py`、`review.py`（--open-sheet 配線・確定/摩擦は不変）、`collect_review.ps1`（step5）。
- テスト: 新規 `tests/test_contact_sheet.py`（10件）。
- 凍結契約 diff 空・`captures/` 非改変（シートは bench/ 出力）・○×UI ロジック不変。

### 次アクション（実装せず申し送り）
(a) 20件超のシート複数枚分割、(b) サムネイルに周波数/duty 等の小注記、(c) シートを
review_suggest の CC 分類下地に結合（別タスク）、(d) 確定可否の凡例をシートに描く。
