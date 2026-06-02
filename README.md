# sigscan — HackRF 1〜6GHz 自動信号識別（ハイブリッド・スキャン）

HackRF One 1台で 1〜6GHz をスキャンし、検出した信号を自動分類する。
**広域サーベイ**で電波が出ている帯を見つけ、**集中ドウェル**で各帯を捕捉・分類する
ハイブリッド方式。単一デバイスを時分割で回す。

ハードが無くても **シミュレーションで完全に動作**する（`--sim`）。HackRF 実機は
`--hardware` で同じコードパスにそのまま差し替わる。

---

## アーキテクチャ

```
            ┌──────────── HybridScheduler（単一SDRを時分割）────────────┐
  [survey]  │  1〜6GHz を粗スイープ → ノイズ床推定 → アクティブ帯検出     │ survey_interval毎
            │        │                                                  │
  [dwell]   │        ▼  ターゲット = 検出帯(SNR順) ∪ ホットバンド(優先巡回) │
            │  各ターゲットへリチューン → IQ捕捉 → 帯域幅/SNR測定 → 分類   │ 合間に実行
            │        │ → SQLiteログ                                      │
            └────────┴───────────────────────────────────────────────────┘
```

### モジュール構成

| ファイル | 役割 |
|----------|------|
| `main.py` | CLI エントリーポイント |
| `config.py` | SDR/スキャン設定・**1〜6GHzバンドプラン**（日本の割当考慮） |
| `sdr.py` | バックエンド抽象化（`SimBackend` / `HackRFBackend`） |
| `dsp.py` | PSD・ノイズ床・アクティブ帯検出・IQ測定・スペクトログラム（numpyのみ） |
| `classify.py` | 3段分類器（ルールベース実装 + CNN/LLMフック）・信号DB |
| `scheduler.py` | ハイブリッド・スケジューラ（本体）・自己収集・収集側の重複排除 |
| `spec.py` | **データ契約**: 正準スペクトログラム表現（単一の真実） |
| `sigmf_io.py` | **データ契約**: SigMF入出力（自動ラベル付き保存） |
| `store.py` | 検出ログ（SQLite） |
| `dataset.py` | 収集物の索引化・フィルタ・重複排除・train/val分割・統計CLI |
| `review.py` | 低信頼アノテーションの人手レビュー/再ラベル |

### 分類の3段構成

```
Step 1: ルールベース（周波数×帯域幅 → 信号DB照合）  最大 0.85  ← 実装済み
Step 2: CNN推論（スペクトログラム画像）           0.60以上で採用  ← フックのみ
Step 3: LLM Vision（低信頼度・未知信号）                          ← フックのみ
```

CNN/LLM は `classify.py` の `cnn_classify()` / `llm_classify()` を実装すれば有効化。
`--save-spectrograms` で各ドウェルのスペクトログラム PNG を `captures/` に保存する
（CNN学習データ・LLM入力の前段）。

---

## 使い方（シミュレーション・ハード不要）

```bash
pip install numpy matplotlib      # matplotlib は PNG 保存時のみ

python3 main.py --sim --once                 # 1サイクルだけ実行
python3 main.py --sim                         # 連続実行（Ctrl-Cで停止）
python3 main.py --sim --save-spectrograms     # スペクトログラムPNGも保存
python3 main.py --sim --collect captures/      # 自己収集: 自動ラベル付きSigMF出力
python3 main.py --sim --start 2.4e9 --stop 2.5e9   # 2.4GHz帯だけ
```

収集側は近接周波数の重複キャプチャを短時間窓で抑制する（既定30秒、`--collect-dedup-window`
で調整・`0`で無効）。`_build_targets` の近接排除と同じ規則を収集ループにも適用する。

---

## 滞在観測モード ＋ 厳しめの品質ゲート（量より質）

このプロジェクトは「自分の実データが土台」。低品質・断片的なキャプチャを保存すると
それが学習の基準になり全体が崩れる。よって **量より質**: 一瞬かすっただけの単発や、
受信機由来のスプリアスは保存しない。

**滞在観測モード**（`--dwell-seconds` または `--dwell` で有効化）は、1つの対象帯に
指定時間（既定10秒）留まり、IQ を間隔を空けて複数回取得して**バーストを待ち受ける**。
2.4GHz の WiFi/BT は数秒おきに一瞬しか出ないため、約13msの単発観測では取り逃すが、
滞在すれば「持続性」を測れる。最も良く捉えた瞬間（検出マージン最大）を代表として保存し、
出現回数・持続率・SNR分布を集計する。

```bash
python3 main.py --sim --collect captures/ --dwell-seconds 2 --once   # Sim で経路確認
python3 main.py --sim --start 2.4e9 --stop 2.5e9 --dwell-seconds 5    # 2.4GHz に5秒ずつ滞在
```

**品質ゲート**（保存条件）は次を全て満たすものだけ保存する（しきい値は `config.py` の
`QualityConfig`、既定は厳しめ。CLI でも上書き可）:

- **持続性**: 滞在中に複数回はっきり検出された（`--q-min-detections` / `--q-min-persistence`）。
  一瞬かすっただけの単発は破棄。
- **極細スプリアス除外**: 占有が極端に細いだけの山を破棄（`--q-narrow-bw`）。ただし幅だけで
  切らず「同一強度で居座る（低分散・高持続）」かと併用し、バースト性のある正規の狭帯域信号
  （BLE等）は残す。
- **コムスプリアス除外**: 等間隔・同一強度で並ぶ細いピーク列（受信機内部由来。アンテナ無しでも
  出る固定パターン）を検出して破棄。

保存件数が少なくなるのは想定内。足切りを外したいときは `--no-quality-gate`。品質メタ
（観測回数・持続率・SNR統計・スプリアス疑い等）は SigMF の annotation に `sigscan:` 名前空間で
記録される（`global` には `sigscan:capture_mode="dwell"`）。

出力例（持続しない単発は破棄、持続する信号は保存）:
```
  [        detected]  2400.00MHz  BW= 0.2MHz  SNRmax= 43dB  persist=0.60(3/5)  → BLE  [drop:low-persistence(0.60<0.90)]
  [band:Cellular B3 DL] 1842.50MHz  BW=15.0MHz  SNRmax= 27dB  persist=1.00(5/5)  → LTE DL  [SAVE]
```

---

## データセット管理（`dataset.py` / `review.py`）

収集した SigMF を索引化して学習/評価で使える形に整える。**`core:hw`（合成/実機）を
最重要属性として保持し、`split()` は hw 毎に独立して行う**（sim と real を混ぜない）。

```bash
python3 -m dataset stats  captures/                      # バンド/label/hw/SNRの内訳
python3 -m dataset stats  captures/ --dedup              # 近接重複を除いて集計
python3 -m dataset query  captures/ --hw real --snr-min 20 --band WiFi
python3 -m dataset review captures/                      # 低信頼の再ラベル（後述）
```

低信頼レビュー: `method=='rule'` かつ `confidence<0.5` のアノテーションを列挙し、
対話で正しい label に直して `.sigmf-meta` に書き戻す（`sigscan:method` を `human` に更新）。
生IQ（`.sigmf-data`）には触れない。

```bash
python3 review.py captures/            # 対話レビュー
python3 review.py captures/ --list     # 対象を列挙のみ（書き換えなし）
```

出力例：
```
[survey] active=11  2437MHz/29.2MHz/40dB  3550MHz/139.6MHz/39dB ...
  [        detected]  2436.85MHz  BW= 29.2MHz  SNR= 40dB  → WiFi (2.4GHz, 20/40MHz) (0.78/rule)
  [        detected]  3549.85MHz  BW=139.6MHz  SNR= 39dB  → 5G NR (n77/n78 3.5G) (0.80/rule)
  [        detected]  1575.64MHz  BW= 20.0MHz  SNR= 12dB  → GPS/QZSS L1 C/A (0.80/rule)
```

---

## 実機（HackRF）

### Fedora セットアップ

```bash
sudo dnf install python3-soapysdr SoapySDR SoapySDR-devel hackrf

# SoapyHackRF をビルド（ABI不一致の場合）
cd ~/SoapyHackRF && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DSoapySDR_DIR=/usr/local/share/cmake/SoapySDR
make -j4 && sudo make install && sudo ldconfig

# プラグインパス（make() no match 対策）
export SOAPY_SDR_PLUGIN_PATH=/usr/local/lib64/SoapySDR/modules0.8-3

# 確認
SoapySDRUtil --probe="driver=hackrf"
```

### 実行

```bash
python3 main.py --hardware                       # 1〜6GHz 全域
python3 main.py --hardware --lna 32 --vga 24 --amp
python3 main.py --hardware --start 5.7e9 --stop 5.9e9   # 5.8GHz帯

# 滞在観測モード（実機推奨）: 各対象に10秒滞在しバーストを待ち受け、品質ゲートで選別保存
python3 main.py --hardware --start 2.4e9 --stop 2.5e9 --dwell-seconds 10 --collect captures/
```

> **実機での滞在観測**: WiFi/BT はバースト的（数秒おきに一瞬）。`--dwell-seconds 10` 程度で
> 滞在して待ち受けると取り逃しが減る。等間隔・同一強度の細い山（約42dB・0.5MHz幅など）は
> HackRF 内部スプリアスのことが多く、品質ゲートが「同一強度で居座る／等間隔コム」として
> 除外する（アンテナを外して確認するのも有効）。保存件数が少なくても想定内（量より質）。
> 厳しすぎる場合は `--q-min-persistence` 等で緩める。

> 注意: HackRF は1プロセスのみ占有可能。`hackrf_sweep` 等と同時実行しないこと。

---

## バンドプラン（`config.py`）

1〜6GHz の主要な割当を `BAND_PLAN` に定義（GNSS / 携帯sub-6 / ISM・WiFi / 5G NR /
レーダ / アマチュア）。日本固有の ETC/DSRC(5.8GHz)・n79(4.5-4.9GHz)・WiFi W52/53/56
も含む。`priority` でドウェル頻度、`hint` で分類ヒントを与える。新バンドは
`Band(...)` を追加するだけ。信号の精緻化ルールは `classify.py` の `SIGNAL_DB`。

---

## チューニングの勘所

| パラメータ（`config.py`） | 既定 | 説明 |
|---|---|---|
| `survey_bin_hz` | 200kHz | サーベイ分解能（小さいほど精密・遅い） |
| `detect_threshold_db` | 8dB | ノイズ床+これでアクティブ判定 |
| `survey_interval_s` | 12s | サーベイ再実行間隔（合間にドウェル） |
| `max_dwell_per_cycle` | 6 | 1サイクルのドウェル数 |
| `dwell_rate_hz` | 20MHz | 捕捉時IBW（HackRF上限付近） |

滞在観測・品質ゲート（`DwellConfig` / `QualityConfig`。既定は厳しめ）:

| パラメータ | 既定 | 説明 | CLI |
|---|---|---|---|
| `dwell_seconds` | 10s | 1対象帯に滞在する秒数 | `--dwell-seconds` |
| `obs_interval_s` | 0.5s | 滞在中の観測間隔 | `--obs-interval` |
| `detect_snr_db` | 10dB | 1観測で「検出」とみなす検出マージン | `--q-detect-snr` |
| `min_detections` | 3 | 滞在中に必要な最低検出回数 | `--q-min-detections` |
| `min_persistence` | 0.34 | 必要な最低持続率（検出された割合） | `--q-min-persistence` |
| `narrow_bw_hz` | 0.7MHz | これ未満を「極細」とみなす | `--q-narrow-bw` |
| `spur_snr_std_max` / `spur_persistence_min` | 1.5dB / 0.9 | 細い＋同一強度＋ほぼ常時 → スプリアス疑い | — |
| `comb_*` | — | 等間隔・同一強度コムスプリアスの検出条件 | — |
| `enabled` | True | 品質ゲートの有効/無効 | `--no-quality-gate` |

---

## 既知の制約

- HackRF の瞬時帯域は ~20MHz。これより広い信号（5G NR 100MHz等）は1回の捕捉に
  収まらないため、**帯域幅はサーベイ側の値を採用**している。
- 滞在観測の「検出」は、滞在中の最も静かな観測を受信機ノイズ床の基準にして測る。
  そのため**滞在中ずっと帯域を埋め続ける常時信号**（途切れない携帯DL等）は基準が
  立たず過小検出になりうる（単一IBW取得の本質的限界）。この種の常時信号はサーベイ側の
  ログで拾う。バースト信号・狭帯域信号は滞在観測で正しく持続率が測れる。
- サーベイのリチューン・ループは `hackrf_sweep` より遅い。高速化が必要なら
  `sdr.HackRFBackend.sweep_power()` を `hackrf_sweep` のサブプロセス実装に
  差し替える（powerのみで足りるサーベイ用途）。
- 実電波の GPS は通常ノイズ床以下（要逆拡散）。Sim では検証用に微弱可視化している。

---

## テスト / CI

凍結した契約（`spec.py` / `sigmf_io.py`）と6つの継ぎ目を回帰テストでロックしている。
テストは `tests/` 配下にあり、契約のロジックは変更せず **振る舞いとシグネチャだけを固定**する。

```bash
pip install numpy pytest
pytest -q                       # 全テスト実行
pytest -q tests/test_spec.py    # ファイル個別
python -m py_compile *.py       # 構文チェック（CI と同じ）
```

| テスト | 対象継ぎ目 | 何をロックするか |
|--------|-----------|------------------|
| `test_spec.py`      | 表現        | `render()` の shape/dtype/値域、`spec_summary()` をスナップショット固定 |
| `test_sigmf_io.py`  | 交換        | IQ 往復一致、`core:datatype=cf32_le`/sample_rate/hw、annotation の周波数エッジ＆`sigscan:*` |
| `test_classify.py`  | 分類        | 代表バンドのルール分類、CNN/LLM 未実装時のルール劣化 |
| `test_dsp.py`       | 測定        | `detect_segments` の帯検出、`measure_signal` の bw/SNR |
| `test_scheduler.py` | 取得・蓄積  | `SimBackend` で1サイクル(once)、`collect_dir` への SigMF 出力、滞在観測モードの保存/破棄 |
| `test_dwell.py`     | 取得（滞在）| `observe_dwell` の出現回数・持続率・SNR分布の集計、代表選択、deadline/min観測 |
| `test_quality.py`   | 品質ゲート  | 単発/極細スプリアス/コムスプリアスの破棄、正規狭帯域(BLE)の保持、しきい値調整 |
| `test_seams.py`     | 全6継ぎ目   | `inspect.signature` で各 API の引数名を凍結 |

`spec_summary()` を変えると `test_spec.py` が落ちる（= 表現仕様の無断変更検知）。
表現を意図的に変えたときは、SigMF の生IQから再レンダの上で
`tests/snapshots/spec_summary.json` を更新すること。

CI（`.github/workflows/ci.yml`）は Python 3.12 で `pip install` → `python -m py_compile *.py`
→ `pytest -q` を実行する。

## ロードマップ（次の増分）

1. **CNN学習パイプライン** — `--save-spectrograms` で集めた画像 + 合成データで
   小型CNNを学習し `cnn_classify()` を実装（RX580はCPU-only想定なので軽量モデル）。
2. **LLM Vision** — 低信頼度・未知信号のスペクトログラムを Gemini/Claude に投げる
   `llm_classify()`。
3. **継続学習** — 高信頼ドウェルを自動ラベリングして学習データに追加。
4. **ウォーターフォール表示** — サーベイ結果を時系列で可視化。
