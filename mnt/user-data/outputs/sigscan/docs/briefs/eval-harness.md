# eval-harness — 初回ワークオーダー（M1: 配線のみ。測定は実機待ち）

外部学習済みモデルを sigscan の正準表現に接続する。**M1では配線だけ**。本当の
ドメインギャップ測定は capture-engine の実機キャプチャが出てから。

## ▶ 初回プロンプト（このままセッションに貼る）

```
あなたは eval-harness エージェント。まず CONTRACT.md を読んで。入出力は必ず
spec.render() を通す。6つの継ぎ目のシグネチャは変えない。

重要な制約: いまは実機キャプチャが無い。よって M1 では「配線（モデルのロードと入出力
アダプタ）」のみを作り、Sim に対する結果は必ず "synthetic-vs-synthetic（本当のギャップ
ではない)" と明示すること。実測に対する測定は後のマイルストーン。

M1として eval/ 配下に実装して:

1. eval/loaders.py
   - TorchSig/WBSig53 の学習済みモデル（github.com/torchdsp/torchsig, torchsig.com）を
     ロードする関数。重みの取得方法・サイズ・ライセンス(DeepSig=CC BY-NC-SA)を docstring と
     eval/README.md に明記。WebFetch/WebSearch で最新の取得手順を確認してよい。
   - MathWorks/Qoherent の 5G/LTE/WLAN セグメンテーションモデル
     (github.com/qoherent/spectrogram-segmentation) のロードも同様に。ライセンス確認。
   - 取得が重い/不可の環境向けに、ロード失敗を例外で握りつぶさず明示する。

2. eval/adapters.py
   - sigscan の SigMF / spec.render() 出力を、各外部モデルの期待入力（サイズ・正規化・
     チャネル）へ写すアダプタ。リサイズ/再正規化はここで吸収。

3. eval/report.py
   - 与えた SigMF 群に推論を回し、sigscan のルールラベルに対する混同/一致を出力。
   - 出力には必ず hw（sim/real）と "synthetic-only（本当のギャップ未測定)" のバナーを付ける。

検証して報告:
- 最低1つの外部モデルをロードし、spec.render() のテンソル1枚に推論が通ること。
- report.py が sim 収集物に対して動き、合成限定である旨を明示出力すること。
- 実測が用意でき次第どう測定に移行するか（手順）を eval/README.md に書く。
```

## 受け入れ基準
- 外部モデルのローダ＋アダプタ＋レポート雛形が動く（最低1モデル）。
- Sim 結果に「合成限定・本当のギャップ未測定」が明示される。
- ライセンス（DeepSig CC BY-NC-SA 等）が記録されている。

## 依存
測定フェーズは capture-engine の実機キャプチャ後。配線はそれと並行可。
