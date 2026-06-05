# 作業指示書: cnn-training 初仕事 — Sim学習パイプラインの火入れ（M1）

## 役割
あなたは sigscan プロジェクトの cnn-training エージェント。
本指示は、CNN（スペクトログラム画像分類器）の学習パイプラインを、
Sim（合成）データで最小構成のまま end-to-end に通す「火入れ」作業である。
精度を出すことが目的ではない。**経路が動くこと**が目的である。

## 背景（なぜ・何を）
- 3段分類器（ルール → CNN → LLM）のうち、CNN 段だけが未着手で、
  学習の仕組みが動くかどうか一度も検証されていない。
- 実データ側には「保存IQが約13msの切り取りで、まばらなバースト信号だと
  ラベルと画像の主役がズレることがある」という収集設計課題が見つかっている。
  その改善効果を測るにも、まず動く学習パイプライン（物差し）が必要。
- Sim-first: 合成データはラベルが生成時の真実（ground truth）なので
  ラベルノイズがゼロ。パイプラインの動作確認に最適。
  DC除去・dwell で成功した「Simで経路確認 → 実機」パターンの再現である。

## 最重要原則（絶対厳守）
1. **凍結契約不可侵**: `spec.py` / `sigmf_io.py` は一切変更しない。
   6継ぎ目（spec.render / sdr backend / dsp.measure_signal, detect_segments /
   classify.classify / sigmf_io read+write / store.Store）のシグネチャ不変。
   - CNN への入力表現は **必ず `spec.render(iq, rate)` → [256,256] float32 [0,1]**
     を使う（正準表現の存在理由そのもの）。独自の前処理・リサイズ・正規化の
     追加は禁止（標準的なデータ拡張を入れたい場合も今回は見送り、将来課題として
     報告に書く）。
2. **classify.classify への組み込みは今回やらない**: CNN は独立モジュールとして
   学習・評価・推論ヘルパまで。3段への接続は次マイルストーン（継ぎ目保護のため）。
3. **SYNTHETIC-ONLY の正直バナー**: レポートとチェックポイントのメタに
   「合成データのみで学習・実環境とのギャップ未測定」を明記
   （eval-harness と同じ正直文化）。
4. **CPU 前提**: torch 2.8.0+cpu / torchvision 0.23.0+cpu。GPU 前提コード禁止。
   小さく速く（火入れであって精度狩りではない）。
5. **量より質**: Sim の教師ラベルは生成時の真実をそのまま使う。
   **ルール分類器の出力をラベル（教師）にしない**（ラベルノイズ防止）。

## 事前確認（実装前に必ず行い、結果を報告に含める）
1. `FILES.md` / `CONTRACT.md` / `AGENTS.md` を読み、モジュール配置の流儀を確認。
   想定は llmvision/, eval/ と同格の新パッケージ `cnntrain/`。流儀が違えば従い、報告。
2. `sdr.SimBackend` と `dsp` の信号生成能力を調べ、**見え方が明確に異なる
   3〜5クラス**を定義できるか確認する。例（あくまで例。実際に生成できるもので決める）:
   - wideband-OFDM風（広帯域の塊）
   - narrowband-burst（BLE風の短い点滅）
   - CW tone（細い連続線）
   - pulse列（周期的な縦線）
   - noise-only（信号なし）
   生成手段（SimBackend 経由か、dsp/numpy で直接合成か）を決めて報告。
3. `dataset.py` の API（load_index / query / split）を確認し、学習データの
   読み込みに再利用できるか判断（再利用できれば実データへの将来展開が楽になる）。
4. torch の CPU 動作確認（import + ダミーテンソルで1回 forward）。
5. 既存テストが現状で全緑（103 passed, 3 skipped）なことを確認してから着手。

## 実装内容

### (1) Sim データ生成 CLI
`python -m cnntrain.simgen --out simdata/ --per-class N --seed S`
- 3〜5クラス・クラス均衡。既定は小さく（per-class 60〜100 程度）。
- **SigMF で保存**（凍結 sigmf_io を使用）。core:hw は既存慣習どおり
  `"sigscan-sim (synthetic)"`（正直なハードウェア表記）。
- 真実ラベルを annotation の sigscan: 名前空間に記録（例: `sigscan:true_class`）。
- クラス名は**方式軸（見え方）**で付ける（用途軸ではない）。これは
  「CNN は方式（見え方）を学び、用途は周波数等の文脈で後段が導く」という
  プロジェクトの確定設計に沿うため。

### (2) データセット読み込み
SigMF → iq → `spec.render` → [256,256] テンソル + ラベル。
train/val 分割（シード固定・再現可能。比率は 80/20 目安）。

### (3) モデル
小さな CNN（自作の数層CNN、または torchvision の軽量モデルを CPU 向けに）。
1 run が CPU で数分〜十数分で終わる規模に抑える。

### (4) 学習 CLI
`python -m cnntrain.train --data simdata/ --epochs E --out runs/<name>/`
- チェックポイント保存（モデル重み + クラス名一覧 + SIGSCAN_REP_VERSION +
  SYNTHETIC-ONLY メタ + 生成シード）。
- 学習経過（loss/acc）のテキストログ出力。

### (5) 評価
val の accuracy と混同行列をレポート（テキスト or JSON）。
レポート冒頭に SYNTHETIC-ONLY バナー（eval-harness の流儀）。

### (6) 推論ヘルパ
チェックポイント読み込み → 画像1枚（[256,256] float32）→
（クラス名, 確信度 softmax）を返す関数。
将来の classify 接続の**準備**であり、接続はしない。

## テスト（高速・既存変更不可）
- T1: simgen が指定クラス・件数の SigMF を作り、真実ラベルが annotation に載る。
- T2: 読み込み（SigMF → spec.render → テンソル）の形状・値域 [0,1]・ラベル対応が正しい。
- T3: 極小データでの 1 エポック・スモーク学習が完走し、チェックポイントが
  保存・再読込できる。
- T4: 推論ヘルパが保存済みチェックポイントで（クラス, 確信度∈[0,1]）を返す。
- 新規テスト全体の実行時間は 1〜2 分以内（CI に優しい規模）。
- 既存テストの変更・削除は不可（追加のみ）。

## 検証（実装後に必ず行う）
1. `python -m py_compile` 対象ファイル全部 → OK。
2. `python -m pytest -q` → 既存 103 passed 維持 + 新規分が全緑。
3. 凍結契約 diff 空: `git diff --stat -- spec.py sigmf_io.py`。
4. end-to-end スモーク: simgen（小）→ train（1〜2 epoch）→ 評価レポート生成、
   が一連で完走すること。所要時間を計って報告。
5. 火入れの成否は「動くこと」。精度は参考値として報告
   （低くても火入れ成功でよい。ただし全サンプル同一クラス予測のような
   明白な不健全があれば、原因の見立てを一言）。

## 完了報告に含めること
- 事前確認の結果（配置の流儀 / 選んだ生成手段 / 定義したクラス一覧と
  その「見え方」の根拠）。
- 変更・追加ファイル一覧と要点。
- end-to-end 実行ログの要約（件数、epoch、所要時間、val 精度、混同行列）。
- SYNTHETIC-ONLY バナーの所在（レポート・チェックポイントのどこに入れたか）。
- テスト結果（N passed / skipped）・凍結 diff 空の明言。
- 詰まった点・将来課題（実データ学習、classify 接続、データ拡張、
  バースト同期キャプチャとの関係 等）。

## やってはいけないこと（禁止事項の再掲）
- spec.py / sigmf_io.py の変更。spec.render を迂回する独自前処理・リサイズ・正規化。
- classify.classify への接続・変更（次マイルストーン）。
- ルール分類器の出力を教師ラベルに使うこと。
- GPU 前提のコード。巨大データ・長時間学習（火入れの範囲を超える精度チューニング）。
- 重量級の新規依存の追加（torch / torchvision / numpy / matplotlib の範囲で）。
- 既存テストの変更・削除。
