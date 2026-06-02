# eval-harness — 外部学習済みモデルの配線と評価

外部の RF 学習済みモデルを sigscan の**正準表現 `spec.render()`** につなぎ、
収集物に推論を回して sigscan のルールラベルと突き合わせるためのハーネス。

> **M1 のスコープ＝配線だけ。** モデルのロード（`loaders.py`）と入出力アダプタ
> （`adapters.py`）を作り、推論が `spec.render()` のテンソルに通ることを確認する。
> **実機キャプチャに対するドメインギャップの測定は後マイルストーン**（CONTRACT.md §4）。
> いま手元にあるのは合成(sim)のみなので、レポートは必ず
> **"synthetic-vs-synthetic（本当のギャップではない）"** と明示する。

---

## いま動くもの

```bash
# 1) この環境でどのモデルがロードできるか自己診断
python -m eval.loaders

# 2) sim 収集物に推論を回して対応表を出す（既定: reference stand-in）
python -m eval.report captures/

# 3) 実モデルを要求（torch + 重みが要る。無ければ明示エラー）
python -m eval.report captures/ --model qoherent-segmentation --weights path/to/weights.pth
#   → ロード不可なら --allow-standin で stand-in に「明示退避」できる
python -m eval.report captures/ --model qoherent-segmentation --allow-standin
```

`reference stand-in` は **numpy だけで動く配線検証用のダミー**。学習済みでも外部
モデルでもなく、スペクトログラムの**占有帯域率**という単純特徴を粗いバケット
（`occ<10% / occ10-50% / occ>50%`）に振るだけ。**信号の有無も変調種別(5G/LTE 等)も
判定しない。** 出力には常に `is_stand_in=True` が立ち、レポートにバナーが出る。
torch が無い環境でも `spec.render → adapter → 推論 → report` の経路を通せる。

---

## モジュール

| ファイル | 役割 |
|----------|------|
| `loaders.py`  | 外部モデルのロード（遅延 import / 重み解決 / 失敗は `ModelUnavailable` で明示） |
| `adapters.py` | `spec.render()` の正準画像・生 IQ を各モデルの入力（サイズ/チャネル/正規化）へ写す |
| `report.py`   | SigMF 群に推論を回し、ルールラベルとの**対応表**を出力（バナー必須） |

設計の核（CONTRACT.md §1 と整合）:
- 画像ドメインのモデルは**必ず `spec.render()` の正準画像を起点**にする（単一の真実）。
- リサイズ・再正規化・チャネル数合わせは**すべて `adapters.py` に吸収**。6 継ぎ目の
  シグネチャ（特に `spec.render`）は触らない。
- 取得が重い/不可の環境では、ロード失敗を**握りつぶさず** `ModelUnavailable` で返す。

---

## 外部モデルの取得手順・サイズ・ライセンス

> ⚠️ ライセンスは各自で最終確認すること。特にデータセット由来の制約（非商用など）は
> 重みにも波及しうる。下表は調査時点の要約。

### 1. TorchSig / Sig53（狭帯域・変調分類）
- リポジトリ: <https://github.com/TorchDSP/torchsig> ・ <https://torchsig.com>
- モデル: **EfficientNet-B4 / XCiT**。入力 = 複素ベースバンド IQ **4096 サンプル**を
  実部/虚部 2ch にした `[2, 4096]`。出力 = **53 クラス**（Sig53）。
- データセット **Sig53 は TorchSig が合成生成**（500万サンプル / 53クラス）。
  → **DeepSig の RadioML(RML2016/2018) とは別物。** RadioML は **CC BY-NC-SA 4.0
  （非商用・継承）**。混同しないこと。
- コードのライセンス: **MIT**。
- 取得:
  ```bash
  pip install torchsig            # Python>=3.10 推奨
  # もしくは: git clone https://github.com/TorchDSP/torchsig && cd torchsig && pip install -e .
  ```
  事前学習チェックポイントはバージョンにより torchsig.com / HuggingFace 配布
  （**数十〜数百 MB**）。ダウンロードした `.pt/.pth` を `--weights` か
  環境変数 `SIGSCAN_TORCHSIG_NB_WEIGHTS` で指定。
- ロード: `loaders.load_torchsig_narrowband(weights=..., num_classes=53)`。

### 2. TorchSig / WBSig53（広帯域・検出/セグメンテーション）
- 同リポジトリ。**複素スペクトログラム（~512×512）**入力の検出網（DETR / Mask R-CNN 系）。
  WBSig53 = 55万サンプル合成 / 53クラス。コード **MIT**。
- 現状 `loaders.load_torchsig_wideband()` は**配線のみ**で、構築 API がバージョン
  依存のため本環境では `ModelUnavailable` を返す（重み入手後に実体化を拡張する）。

### 3. Qoherent / spectrogram-segmentation（5G NR / LTE セグメンテーション）
- リポジトリ: <https://github.com/qoherent/spectrogram-segmentation>
- モデル: **DeepLabv3 + MobileNetV3 backbone**（torchvision 経由）。PyTorch + Lightning。
- 入力: **256×256 RGB** スペクトログラム画像（3ch, **ImageNet 正規化**）。
- 出力: セマンティックセグメンテーション（**noise / 5G NR / LTE**）。学習データは
  **MathWorks の 5G/LTE Toolbox 合成スペクトログラム**（公開）。
- ライセンス: リポジトリ表記 **MIT**（要最終確認）。
- 取得:
  ```bash
  git clone https://github.com/qoherent/spectrogram-segmentation
  pip install torch torchvision pytorch-lightning
  # ノートブックで学習するか、配布チェックポイント(.pt/.pth)を入手
  ```
  `.pt/.pth` を `--weights` か環境変数 `SIGSCAN_QOHERENT_WEIGHTS` で指定。
- ロード: `loaders.load_qoherent_segmentation(weights=..., num_classes=3)`。

### 依存（実モデルを動かす場合のみ）
```bash
# CPU 版 torch（実機計算に GPU は使わない方針と整合）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchvision          # qoherent セグメンテーション用
pip install torchsig             # torchsig narrowband/wideband 用
```
sigscan 本体・stand-in・report は **numpy のみ**で動く（torch は実モデル推論時だけ必須）。

---

## レポートの読み方（誤読を防ぐ注意）

- 外部モデルのクラス空間（53クラス / セグメンテーション）は sigscan の**バンドラベル
  空間とは一致しない**。よってレポートは安易な「正解率」ではなく、
  **行=ルールラベル × 列=外部予測 の対応表（クロス集計）**を出す。
- 必ず付くバナー:
  - `hw 内訳`（sim / real / other の件数）
  - **SYNTHETIC-ONLY**: 入力が sim のみのとき、結果は synthetic-vs-synthetic で
    **本当のドメインギャップは未測定**である旨。
  - **STAND-IN MODEL**: stand-in 使用時、外部モデルの性能ではなく配線確認である旨。

---

## 実測（real）への移行手順 — ドメインギャップを“測定”に変える

`capture-engine` が **HackRF 実機**でキャプチャを出した後に、以下で本来の評価に進む。
継ぎ目（`spec.render` / SigMF スキーマ）は凍結なので、ハーネス側の変更は最小で済む。

1. **実データを収集**（`core:hw = "HackRF One"` が SigMF に正直に入る）:
   ```bash
   python main.py --hardware --collect captures_real/ --collect-snr 8
   ```
2. **実モデルを実体化**（torch + 重みを用意し、`python -m eval.loaders` で
   `[loadable]` になることを確認）。
3. **sim と real を別々に**レポートし、**同一モデル**で突き合わせる:
   ```bash
   python -m eval.report captures/      --model qoherent-segmentation --hw sim  --weights W
   python -m eval.report captures_real/ --model qoherent-segmentation --hw real --weights W
   ```
   - `--hw` 層別を必ず使い、**sim と real を 1 つのレポートに混ぜない**
     （CONTRACT.md §2 の出所防壁。`dataset.split()` も hw を跨がない）。
4. **ギャップ指標を追加**（次マイルストーンの実装ポイント）:
   - 同一クラスに対する **sim 予測分布 vs real 予測分布** の差（例: KL / 総変動距離）。
   - 共通の弱教師ラベルに対する **sim 一致率 − real 一致率**（= 表現ドメインギャップの
     代理指標）。real が揃って初めて意味を持つ。
   - SNR 別・バンド別の層別ギャップ（`dataset.query(snr_min=...)` で抽出）。
5. これらが揃った時点で、レポートのバナーは **SYNTHETIC-ONLY から「sim vs real
   measured」へ切り替える**（`report.py` の `_banner` を real 経路で更新）。

> 重要: それまでは **sim の結果を実性能と読み替えない**。本ハーネスがバナーで
> 繰り返し警告するのはそのため。
