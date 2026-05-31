# test-docs — 初回ワークオーダー（M1: 契約をロック）

最初に走らせるエージェント。6つの継ぎ目と凍結契約を**テストで固め**、以降の破壊を
即検知できる状態にする。

## ▶ 初回プロンプト（このままセッションに貼る）

```
あなたは test-docs エージェント。まず CONTRACT.md と README.md を読み、リポジトリ構成を
把握して。目的は「凍結した契約と6つの継ぎ目をテストでロックすること」。

以下を pytest で実装し、tests/ 配下に置いて:

1. test_spec.py
   - spec.render(iq, rate) の出力が shape == (spec.IMG_FREQ, spec.IMG_TIME)、
     dtype float32、値域 [0,1] であること（ランダムIQと無音IQ両方で）。
   - spec.spec_summary() を tests/snapshots/spec_summary.json に固定し、一致を検証
     （= 表現仕様が無断で変わったら落ちる）。
2. test_sigmf_io.py
   - write_recording → read_recording の往復で IQ(complex64) が一致。
   - meta に core:datatype=='cf32_le'、core:sample_rate、core:hw が保持される。
   - annotation_from_result が freq_lower/upper_edge と sigscan:confidence/method/snr_db を持つ。
3. test_classify.py
   - 代表的な measurement（例: 2.437GHz/20MHz, 3.55GHz/100MHz, 2.402GHz/2MHz）で
     rule_based のラベルとバンド対応が期待どおり。cnn/llm 未実装時に classify が
     ルール結果へ劣化すること。
4. test_dsp.py
   - 合成トーン/帯域制限ノイズに対し detect_segments が1本検出、measure_signal の
     bw/SNR が妥当な範囲。
5. test_scheduler.py
   - SimBackend で1サイクル（once）が例外なく回り、collect_dir 指定時に SigMF が出る。

さらに:
- 継ぎ目シグネチャの固定テスト（inspect.signature で spec.render / classify.classify /
  sigmf_io.write_recording の引数名を検証）。
- .github/workflows/ci.yml を追加: python 3.12 で pip install + `pytest -q` + `python -m py_compile *.py`。
- README.md にテスト実行方法の節を追記。

制約: 契約ファイル(spec.py / sigmf_io.py)のロジックは変更しない（テストするだけ）。
最後に意図的に1つ継ぎ目を壊して該当テストが落ちることを確認し、元に戻して報告して。
```

## 受け入れ基準
- `pytest -q` が緑。CI が追加され、`py_compile` も通る。
- 継ぎ目シグネチャ／`spec_summary` を変えると対応テストが**落ちる**ことを実証。
- `spec.py`・`sigmf_io.py` のロジック無変更。

## このあと
M1完了・マージ後、capture-engine を起動（依存元はこの後 main から分岐）。
