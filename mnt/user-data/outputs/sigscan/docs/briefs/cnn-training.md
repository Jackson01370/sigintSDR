# cnn-training — 初回ワークオーダー（M1: 学習パイプライン＋cnn_classify）

`spec.render()` の表現で軽量CNN（CPU前提）を学習し `classify.cnn_classify()` を実装。
capture-engine のデータセット完成後に本格化（M1はパイプライン確立まで）。

## ▶ 初回プロンプト（このままセッションに貼る）

```
あなたは cnn-training エージェント。まず CONTRACT.md と classify.py を読んで。入力は
spec.render() 出力（[256,256] float32）、出力は ClassResult に準拠。classify.classify() の
シグネチャは変えない。cnn_classify() はモデル未ロード時に None を返して劣化動作させる。

制約: CPU-only 前提（AMD RX580 は計算に使わない）。小型モデル（compact CNN / MobileNet 級）、
現実的なエポック数。学習は「合成で事前学習 → 実機でファインチューニング」を見据え、
core:hw で層別。合成と実機を混ぜた評価をしない。評価スプリットは学習から厳密分離。

M1として training/ 配下に実装して:

1. training/data.py
   - capture-engine の dataset.py（無ければ captures/ の SigMF 直読み）から
     (spec.render()テンソル, label) を作る Dataset。hw でフィルタ/層別できること。
2. training/model.py
   - 小型CNN（入力1ch [256,256] → クラス数）。CPUで現実的な規模に。
3. training/train.py
   - 学習ループ、重み保存(weights/)、SNR別・hw別の精度レポート、混同行列。
   - まずは sim 収集データでエンドツーエンドに通す（パイプライン実証）。
4. classify.cnn_classify() の実装
   - 保存済み重みをロードし、spec.render() 入力から ClassResult(method="cnn") を返す。
     重みが無ければ None。ラベル空間は SIGNAL_DB / バンドplanと整合させる。

検証して報告:
- sim 収集データで train.py が一周し、weights が保存される。
- cnn_classify() が重みありで ClassResult、なしで None を返す。
- 精度が SNR別・hw別に出る。
```

## 受け入れ基準
- 学習パイプラインが sim データで end-to-end 動作（実データは後で差し替え）。
- `cnn_classify()` が `classify.classify` に劣化なく組み込まれ、未学習時は None。
- 評価が学習から分離され、SNR別・hw別に報告。

## 依存
学習データは capture-engine の SigMF 蓄積が前提。合成追加は eval-harness 経由の
TorchSig でも可（必ず `spec.render` に正規化）。
