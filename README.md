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
| `scheduler.py` | ハイブリッド・スケジューラ（本体）・自己収集 |
| `spec.py` | **データ契約**: 正準スペクトログラム表現（単一の真実） |
| `sigmf_io.py` | **データ契約**: SigMF入出力（自動ラベル付き保存） |
| `store.py` | 検出ログ（SQLite） |

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
```

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

---

## 既知の制約

- HackRF の瞬時帯域は ~20MHz。これより広い信号（5G NR 100MHz等）は1回の捕捉に
  収まらないため、**帯域幅はサーベイ側の値を採用**している。
- サーベイのリチューン・ループは `hackrf_sweep` より遅い。高速化が必要なら
  `sdr.HackRFBackend.sweep_power()` を `hackrf_sweep` のサブプロセス実装に
  差し替える（powerのみで足りるサーベイ用途）。
- 実電波の GPS は通常ノイズ床以下（要逆拡散）。Sim では検証用に微弱可視化している。

---

## ロードマップ（次の増分）

1. **CNN学習パイプライン** — `--save-spectrograms` で集めた画像 + 合成データで
   小型CNNを学習し `cnn_classify()` を実装（RX580はCPU-only想定なので軽量モデル）。
2. **LLM Vision** — 低信頼度・未知信号のスペクトログラムを Gemini/Claude に投げる
   `llm_classify()`。
3. **継続学習** — 高信頼ドウェルを自動ラベリングして学習データに追加。
4. **ウォーターフォール表示** — サーベイ結果を時系列で可視化。
