# sigscan データ契約（CONTRACT）

このファイルは **1本のスレッドで固めた契約** を記す。ここが安定した継ぎ目(seam)に
なるので、以降はこの境界に沿ってマルチエージェント並列化してよい（逆順は不可）。

---

## 1. 正準スペクトログラム表現（`spec.py` が唯一の真実）

収集・評価・学習のすべては **必ず `spec.render(iq, rate)` を通す**。手元STFTを
直に書かない。これで合成・実測・外部データ(WBSig53 等)を同一スケールで混ぜられる。

| 項目 | 値 | 備考 |
|------|----|------|
| 取得レート | 20 MS/s | HackRF 瞬時帯域上限付近・ドウェルと一致 |
| STFT | nfft=512, hop=256, Hann | 50%オーバーラップ |
| 正準画像 | 256 × 256 (freq × time) | CNN/セグメンテーション入力 |
| dB正規化 | 床(5%ile)→0, 床+60dB→1, クリップ | **絶対ゲイン非依存**（受信機差を吸収） |
| バージョン | `SIGSCAN_REP_VERSION = "1.0"` | 変更時は SigMF の生IQから再レンダ |

出力は `float32 [0,1]` の `[256,256]`。表現を変えても **SigMF が生IQを保持**するため
何もロックされない（再レンダで追従）。

---

## 2. SigMF 交換形式（`sigmf_io.py`）

1キャプチャ = `<name>.sigmf-data`（生IQ）+ `<name>.sigmf-meta`（JSON）。
TorchSig / IntelLabs RFML-Framework など SigMF を読むツールと互換。

- `core:datatype` = **`cf32_le`**（complex64 LE）。`numpy complex64.tofile` で直書き。
- `global`: `core:sample_rate`, `core:version`(1.0.0), `core:hw`, `core:recorder`,
  独自 `sigscan:rep_version`, `sigscan:target_src`。
- `captures[0]`: `core:sample_start`, `core:frequency`(中心Hz), `core:datetime`(UTC)。
- `annotations[*]`: `core:sample_start/sample_count`,
  `core:freq_lower_edge`/`core:freq_upper_edge`(絶対Hz), `core:label`,
  `core:comment`, 独自 `sigscan:confidence` / `sigscan:method` / `sigscan:snr_db`。

**出所の正直な記録（重要）**: `core:hw` に実機(`HackRF One`)か合成(`sigscan-sim
(synthetic)`)かを必ず入れる。合成と実測を無自覚に混ぜないための防壁。学習時は
hw でフィルタ/層別化する。

ラベルは弱教師: バンドプラン由来のルール分類（`classify.py`）を annotation 化。
低信頼(confidence<0.5)や `method=rule` のものは後段のレビュー/再ラベル対象。

---

## 3. 安定した継ぎ目（モジュール境界＝契約点）

| 継ぎ目 | API | 役割 |
|--------|-----|------|
| 表現 | `spec.render(iq, rate) -> [256,256] f32` | 単一の真実 |
| 取得 | `sdr.SDRBackend.sweep_power / capture_iq` | Sim/HackRF 抽象 |
| 測定 | `dsp.measure_signal / detect_segments` | 帯域幅・SNR・帯検出 |
| 分類 | `classify.classify(measurement, bands) -> ClassResult` | 3段（rule実装/CNN・LLMフック） |
| 交換 | `sigmf_io.write_recording / read_recording` | SigMF I/O |
| 蓄積 | `store.Store` (SQLite) | 検出ログ |

この6点のシグネチャを変えない限り、各実装は独立に差し替え可能。

---

## 4. 継ぎ目に沿ったエージェント分担（契約確定後に展開）

Claude Code の並列は「経路が互いに独立なとき」最も効く。上の継ぎ目で分けると独立化する。
各エージェントはツール権限を絞り、**git worktree で隔離**、統合時にあなたがレビュー。

| エージェント | 担当継ぎ目 | スコープ |
|--------------|-----------|----------|
| capture-engine | 取得・交換・蓄積 | 自己収集ループ、SigMFデータセット蓄積、重複排除、レビューUI |
| eval-harness | 表現 | WBSig53/MathWorks 学習済みモデルのロード＋`spec.render`入出力アダプタ、**実測に対する**ドメインギャップ測定 |
| cnn-training | 分類(CNN段) | 軽量CNN学習(CPU前提)、`cnn_classify()`実装、SigMF→学習データ変換 |
| llm-vision | 分類(LLM段) | `llm_classify()`、低信頼/未知のスペクトログラムを Gemini/Claude へ |
| test-docs | 全体 | 契約の回帰テスト、ドキュメント、CI |

依存関係の注意（並列でも守る順序）:
- `eval-harness` の **測定**は実測キャプチャに対して行う → `capture-engine` が実データを
  出してから。配線(モデルロード/アダプタ)は先行して並列で可。
- `cnn-training` の学習データは `capture-engine` のSigMF蓄積が前提。
- 全員が `spec.render` と SigMF スキーマを共有（本契約）。ここは凍結。

---

## 5. いま動くもの

```bash
# 自己収集（Sim・自動ラベル付きSigMF出力）
python3 main.py --sim --collect captures/ --collect-snr 8

# 実機
python3 main.py --hardware --collect captures/ --collect-snr 8
```

各検出信号が `captures/<MHz>_<ts>_<n>.sigmf-{data,meta}` として保存され、
`spec.render()` で `[256,256]` の正準表現に展開できる（`sigmf_io.read_recording`→`spec.render`）。
