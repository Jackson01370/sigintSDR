# capture-engine — 初回ワークオーダー（M1: 収集の堅牢化＋dataset.py）

最上流。実データを産み、cnn-training と eval-harness の測定がここに依存する。

## ▶ 初回プロンプト（このままセッションに貼る）

```
あなたは capture-engine エージェント。まず CONTRACT.md・README.md・scheduler.py・
sigmf_io.py・config.py を読んで。データ契約(spec.py / sigmf_io.py)は凍結、6つの継ぎ目の
シグネチャは変えない。

M1として以下を実装して:

1. 重複排除（scheduler.py 内）
   - 同一サイクル/短時間窓で、中心周波数が近接する重複キャプチャの収集を抑制
     （既存の _build_targets の近接排除と整合させ、収集側にも適用）。

2. dataset.py（新規）— SigMF データセットの管理
   - load_index(dir): ディレクトリ内の *.sigmf-meta を走査し、各レコードの
     (path, center, bw, label, confidence, method, snr_db, hw, datetime) を一覧化。
   - query(...): hw / label / バンド名 / SNR下限 でフィルタ。
   - dedup(): 同一 (label, 中心±窓) の近接重複を除外。
   - split(val_ratio): train/val を返す。**hw（合成/実機）を絶対に混ぜない**
     （split は hw ごとに行い、混在データセットでは sim と real を別グループに保つ）。
   - stats(): バンド別・label別・hw別・SNRヒストグラムを表示する CLI
     （`python3 -m dataset stats captures/`）。

3. 低信頼レビュー導線（review.py か dataset のサブコマンド）
   - method=='rule' かつ confidence<0.5 のアノテーションを列挙し、対話で正しい label に
     修正して .sigmf-meta に書き戻す（sigscan:method を 'human' に更新）。

制約: 依存は numpy のみ。SoapySDR は HackRFBackend 内のみ。すべての SigMF の core:hw を
正直に（HackRF One / sigscan-sim (synthetic)）。合成と実機を分離可能に保つ。

検証して報告:
- `python3 main.py --sim --collect captures/ --once` で SigMF が出る。
- `python3 -m dataset stats captures/` がバンド/label/hw の内訳を表示。
- 任意の1件を read_recording → spec.render し (256,256) になることを確認。
```

## 受け入れ基準
- `dataset.py` が列挙/検索/重複排除/**hw分離split**/統計を提供。
- 低信頼アノテーションのレビュー＆書き戻しが動く。
- 収集→SigMF→`spec.render` の往復が通り、契約に無変更。

## 注意（依存元へ）
このブランチを main にマージしてから cnn-training を分岐させる(学習データの前提)。
