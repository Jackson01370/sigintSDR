# 指示文まとめ（work-orders）

各ロボットに「最初にやってほしいこと」を1つにまとめたもの。
動かしたいロボットの章の **▶ 貼る文章** の四角を、そのまま Claude Code に貼ってください。

> 使い方: 黒い画面で `claude --agent <名前>` と打って起動 → 下の四角をコピペ。
> （例: `claude --agent test-docs`）

---

## 全体の段取り（00-kickoff）

順番（依存関係を守る）:

1. **test-docs** … 土台をテストで固める（最初に単独）
2. **capture-engine** … 電波を集める（最上流・実データを作る）
3. **llm-vision** … ✅ 完了済み（再実行不要）
4. **eval-harness** … 配線だけ先に（答え合わせは実機データの後）
5. **cnn-training** … 電波あつめが進んでから

準備（1回だけ・済んでいればOK）:
```
git add -A
git commit -m "checkpoint"
```

---

## 1. test-docs（👈 次にやる）

土台がこわれていないか自動でチェックする仕組みを作る。

**▶ 貼る文章:**
```
あなたは test-docs エージェント。まず CONTRACT.md と README.md を読み、リポジトリ構成を把握して。目的は「凍結した契約と6つの継ぎ目をテストでロックすること」。

以下を pytest で実装し、tests/ 配下に置いて:

1. test_spec.py
   - spec.render(iq, rate) の出力が shape == (spec.IMG_FREQ, spec.IMG_TIME)、dtype float32、値域 [0,1] であること（ランダムIQと無音IQ両方で）。
   - spec.spec_summary() を tests/snapshots/spec_summary.json に固定し、一致を検証（= 表現仕様が無断で変わったら落ちる）。
2. test_sigmf_io.py
   - write_recording → read_recording の往復で IQ(complex64) が一致。
   - meta に core:datatype=='cf32_le'、core:sample_rate、core:hw が保持される。
   - annotation_from_result が freq_lower/upper_edge と sigscan:confidence/method/snr_db を持つ。
3. test_classify.py
   - 代表的な measurement（例: 2.437GHz/20MHz, 3.55GHz/100MHz, 2.402GHz/2MHz）で rule_based のラベルとバンド対応が期待どおり。cnn/llm 未実装時に classify がルール結果へ劣化すること。
4. test_dsp.py
   - 合成トーン/帯域制限ノイズに対し detect_segments が1本検出、measure_signal の bw/SNR が妥当な範囲。
5. test_scheduler.py
   - SimBackend で1サイクル（once）が例外なく回り、collect_dir 指定時に SigMF が出る。

さらに:
- 継ぎ目シグネチャの固定テスト（inspect.signature で spec.render / classify.classify / sigmf_io.write_recording の引数名を検証）。
- .github/workflows/ci.yml を追加: python 3.12 で pip install + pytest -q + python -m py_compile *.py。
- README.md にテスト実行方法の節を追記。

制約: 契約ファイル(spec.py / sigmf_io.py)のロジックは変更しない（テストするだけ）。最後に意図的に1つ継ぎ目を壊して該当テストが落ちることを確認し、元に戻して報告して。
```

---

## 2. capture-engine（電波あつめ・最上流）

検出した電波を「自動で名前を付けて」SigMF形式で貯める仕組みを作る。

**▶ 貼る文章:**
```
あなたは capture-engine エージェント。まず CONTRACT.md・README.md・scheduler.py・sigmf_io.py・config.py を読んで。データ契約(spec.py / sigmf_io.py)は凍結、6つの継ぎ目のシグネチャは変えない。

M1として以下を実装して:

1. 重複排除（scheduler.py 内）
   - 同一サイクル/短時間窓で、中心周波数が近接する重複キャプチャの収集を抑制（既存の _build_targets の近接排除と整合させ、収集側にも適用）。

2. dataset.py（新規）— SigMF データセットの管理
   - load_index(dir): ディレクトリ内の *.sigmf-meta を走査し、各レコードの (path, center, bw, label, confidence, method, snr_db, hw, datetime) を一覧化。
   - query(...): hw / label / バンド名 / SNR下限 でフィルタ。
   - dedup(): 同一 (label, 中心±窓) の近接重複を除外。
   - split(val_ratio): train/val を返す。hw（合成/実機）を絶対に混ぜない（split は hw ごとに行い、混在データセットでは sim と real を別グループに保つ）。
   - stats(): バンド別・label別・hw別・SNRヒストグラムを表示する CLI（python3 -m dataset stats captures/）。

3. 低信頼レビュー導線（review.py か dataset のサブコマンド）
   - method=='rule' かつ confidence<0.5 のアノテーションを列挙し、対話で正しい label に修正して .sigmf-meta に書き戻す（sigscan:method を 'human' に更新）。

制約: 依存は numpy のみ。SoapySDR は HackRFBackend 内のみ。すべての SigMF の core:hw を正直に（HackRF One / sigscan-sim (synthetic)）。合成と実機を分離可能に保つ。

検証して報告:
- python3 main.py --sim --collect captures/ --once で SigMF が出る。
- python3 -m dataset stats captures/ がバンド/label/hw の内訳を表示。
- 任意の1件を read_recording → spec.render し (256,256) になることを確認。
```

---

## 3. llm-vision（✅ 完了済み・再実行しない）

完了済み。設定する環境変数は `SIGSCAN_LLM_PROVIDER`（gemini / anthropic / openai）と、
各社のキー（`GEMINI_API_KEY` など）。キー未設定でも安全（Noneで劣化動作）。

---

## 4. eval-harness（答え合わせ・いまは配線だけ）

外部の学習済みモデルを読み込んで、自分の表現につなぐ。**測定は実機データが出てから。**

**▶ 貼る文章:**
```
あなたは eval-harness エージェント。まず CONTRACT.md を読んで。入出力は必ず spec.render() を通す。6つの継ぎ目のシグネチャは変えない。

重要な制約: いまは実機キャプチャが無い。よって M1 では「配線（モデルのロードと入出力アダプタ）」のみを作り、Sim に対する結果は必ず "synthetic-vs-synthetic（本当のギャップではない)" と明示すること。実測に対する測定は後のマイルストーン。

M1として eval/ 配下に実装して:

1. eval/loaders.py
   - TorchSig/WBSig53 の学習済みモデル（github.com/torchdsp/torchsig, torchsig.com）をロードする関数。重みの取得方法・サイズ・ライセンス(DeepSig=CC BY-NC-SA)を docstring と eval/README.md に明記。WebFetch/WebSearch で最新の取得手順を確認してよい。
   - MathWorks/Qoherent の 5G/LTE/WLAN セグメンテーションモデル (github.com/qoherent/spectrogram-segmentation) のロードも同様に。ライセンス確認。
   - 取得が重い/不可の環境向けに、ロード失敗を例外で握りつぶさず明示する。

2. eval/adapters.py
   - sigscan の SigMF / spec.render() 出力を、各外部モデルの期待入力（サイズ・正規化・チャネル）へ写すアダプタ。リサイズ/再正規化はここで吸収。

3. eval/report.py
   - 与えた SigMF 群に推論を回し、sigscan のルールラベルに対する混同/一致を出力。
   - 出力には必ず hw（sim/real）と "synthetic-only（本当のギャップ未測定)" のバナーを付ける。

検証して報告:
- 最低1つの外部モデルをロードし、spec.render() のテンソル1枚に推論が通ること。
- report.py が sim 収集物に対して動き、合成限定である旨を明示出力すること。
- 実測が用意でき次第どう測定に移行するか（手順）を eval/README.md に書く。
```

---

## 5. cnn-training（写真で覚える）

集めた電波の画像で軽いCNNを学習し、2段目を実装する。**電波あつめが進んでから。**

**▶ 貼る文章:**
```
あなたは cnn-training エージェント。まず CONTRACT.md と classify.py を読んで。入力は spec.render() 出力（[256,256] float32）、出力は ClassResult に準拠。classify.classify() のシグネチャは変えない。cnn_classify() はモデル未ロード時に None を返して劣化動作させる。

制約: CPU-only 前提（AMD RX580 は計算に使わない）。小型モデル（compact CNN / MobileNet 級）、現実的なエポック数。学習は「合成で事前学習 → 実機でファインチューニング」を見据え、core:hw で層別。合成と実機を混ぜた評価をしない。評価スプリットは学習から厳密分離。

M1として training/ 配下に実装して:

1. training/data.py
   - capture-engine の dataset.py（無ければ captures/ の SigMF 直読み）から (spec.render()テンソル, label) を作る Dataset。hw でフィルタ/層別できること。
2. training/model.py
   - 小型CNN（入力1ch [256,256] → クラス数）。CPUで現実的な規模に。
3. training/train.py
   - 学習ループ、重み保存(weights/)、SNR別・hw別の精度レポート、混同行列。まずは sim 収集データでエンドツーエンドに通す（パイプライン実証）。
4. classify.cnn_classify() の実装
   - 保存済み重みをロードし、spec.render() 入力から ClassResult(method="cnn") を返す。重みが無ければ None。ラベル空間は SIGNAL_DB / バンドplanと整合させる。

検証して報告:
- sim 収集データで train.py が一周し、weights が保存される。
- cnn_classify() が重みありで ClassResult、なしで None を返す。
- 精度が SNR別・hw別に出る。
```
