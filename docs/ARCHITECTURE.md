# sigscan アーキテクチャ / プロジェクト憲章

> このドキュメントは Claude の Project knowledge に常設され、新規チャットのたびに読まれる。
> 目的: セッションを跨いでも全体設計が失われないようにする。**読者は「このプロジェクトを初めて見る Claude（または未来のあなた）」**。
> 記述は**事実ベース**。コードで確認したことだけを断定し、推測には「※未確認」を付す。詳細はコードを見れば分かるよう**入口（file:line）**を示す。
> worklog（`docs/worklog/`）とは役割が違う: worklog は「何をしたか」の時系列、本書は「何であるか」の静的な地図。
> 現状スナップショット時点: pytest **214 passed, 3 skipped**。

---

## 0. 一言で言うと

HackRF One（SDR）で 1〜6GHz をスキャンして電波を自動識別するシステム。信号を検出→**3段分類（ルール分類器→CNN監査→LLM vision）**→SigMF形式で生IQ＋メタを保存し、**「AIが提案・人間が承認」**で ground truth（教師データ）を積み上げる。最終目標は自分の HackRF 実測データで CNN を再学習すること。

---

## 1. 設計思想（最重要・破ってはいけない線）

この4つはプロジェクトの背骨。破ると全てが崩れる。

1. **AIが提案・人間が承認**。ルール/CNN/LLM は提案を出すだけ。最終確定は人間が `review.py` で視覚確認したときのみ（`docs/worklog/worklog_2.4GHz_BLE_groundtruth.md:17`）。
2. **Pattern A 禁止**＝ルール/CNN/LLM の出力を **CNN の学習ラベルにしない**（ラベル汚染防止）。ラベルは弱教師で「後段のレビュー/再ラベル対象」（`CONTRACT.md:43-44`）。○×UIの「摩擦」等はこの Pattern A 化を**コードで**防ぐ実装（`review.py` の `_y_blocked_reason`）。
3. **ground truth は human確定のみ**（`sigscan:method="human"`, `sigscan:confidence=1.0`）。`apply_label` がこれを付与する唯一の経路（`review.py:185-214`）。
4. **量より質**。断片的な単発や受信機由来スプリアスは保存しない。確定できないものは Unknown のまま（`README.md:71-74`）。

派生原則（コードから読み取れる）:
- **表現契約**: 収集・評価・学習の**すべてが `spec.render(iq, rate)` を通す**。手元STFTを直書きしない（`CONTRACT.md:10-11`, `spec.py`）。
- **sim/real 分離**: `core:hw` に実機/合成を必ず記録し、学習時に hw で層別化。合成と実測を無自覚に混ぜない防壁（`CONTRACT.md:39-41`, `dataset.hw_group`）。
- **「意志力なしで回る」**: 運用注意ではなくコードで構造化する（例: `decide_recommend` が duty を引数に取らないことをテストで固定＝`tests/test_review_confirm.py`）。
- **並列化は契約凍結後**（逆順不可）。契約（本 CONTRACT）が固まってから継ぎ目に沿ってエージェント分担（`CONTRACT.md:3-4`, `AGENTS.md:3-4`）。

---

## 2. 凍結契約（変更禁止・条件付き可の区別を正確に）

過去に「main.py は完全凍結」という**誤解**があった。権威定義（CONTRACT.md / AGENTS.md / FILES.md）の実際は以下。

### 2.1 真に凍結 ＝ 変更に人間の明示承認が必要 — **`spec.py` と `sigmf_io.py` の2ファイルのみ**
- `AGENTS.md:64`「契約ファイル（`spec.py`・`sigmf_io.py`）は凍結。変更は人間の明示承認が必要。」
- `CONTRACT.md:80`「全員が `spec.render` と SigMF スキーマを共有（本契約）。ここは凍結。」
- `FILES.md:36-37`「spec.py【凍結】画像の作り方の約束 / sigmf_io.py【凍結】データの保存形式」
- メカニズム: `spec.spec_summary()` を変えると `tests/test_spec.py` のスナップショット照合が落ちる。表現を変えるときは `SIGSCAN_REP_VERSION` を上げ、SigMF の生IQから再レンダ（`CONTRACT.md:19`）。

### 2.2 追加のみ可（additive-only）
- **`main.py`**: 「凍結相当」＝**新CLIフラグ追加のみ**許可、diff は完了報告に全掲（`docs/worklog/...:116`, `capture-engine_dwell_offset_tuning_task.md:55`）。**権威契約は main.py を「凍結」とは規定していない**（6継ぎ目にも含まれない）。
- **`review.py`**: 追加のみ。`apply_label`（確定処理）と既存モードは不変（`tests/test_review.py:7-9`）。

### 2.3 6つの継ぎ目（seam）と凍結署名
`CONTRACT.md:48-59` が定義。`tests/test_seams.py` が `inspect.signature` で**引数名（順序込み）**をロック（`test_seams.py:20-56`）。

| 継ぎ目 | API（凍結署名） | 実装ファイル |
|---|---|---|
| 表現 | `spec.render(iq, rate) -> [256,256] f32` | spec.py【凍結】 |
| 取得 | `sdr.SDRBackend.sweep_power(start_hz,stop_hz,bin_hz)` / `capture_iq(center_hz,rate,n)` | sdr.py |
| 測定 | `dsp.measure_signal(iq,rate,center_hz)` / `detect_segments(freqs_hz,power_db,threshold_db,min_bw_hz,smooth_hz,merge_gap_hz)` | dsp.py |
| 分類 | `classify.classify(measurement,bands,spectrogram_db,png_path,cnn_threshold)` / `rule_based(measurement,bands)` | classify.py |
| 交換 | `sigmf_io.write_recording(...)` / `read_recording(path_base)` / `annotation_from_result(measurement,result)` | sigmf_io.py【凍結】 |
| 蓄積 | `store.Store.log/recent/close` | store.py |

- 分類の戻り値 `ClassResult` のフィールドも凍結: `[label, confidence, method, notes, candidates]`（`test_seams.py:68-71`）。
- 「この6点のシグネチャを変えない限り、各実装は独立に差し替え可能」（`CONTRACT.md:59`）。

### 2.4 自由に編集可（署名・スナップショットは破らない範囲で）
`classify.py`, `dsp.py`, `sdr.py`, `store.py`（＝継ぎ目の**実装**。内部ロジックは差し替え自由）、`scheduler.py`, `dwell.py`, `quality.py`, `config.py`, `view_captures.py`, `dataset.py`, `cnntrain/*`, `eval/*`, `llmvision/*`。

### 2.5 注意点（誤読防止）
- **`observation/` は実在しないファントム**。凍結対象として複数の指示書に列挙されるが、観測ロジックの実体は `dwell.py`（`docs/worklog/...:114`）。
- **サンドボックス作業指示書**（`指示書_*.md`）は `spec.py / sigmf_io.py / main.py / observation/` を「diff 空」と要求するが、これは**そのサンドボックス作業に限った、より厳しいルール**。恒久ポリシー（main.py は追加可）とは別物。
- `FILES.md:9-11` の「`*.py` は触らない・消さない」は主に**削除禁止**の粗い表現。コード変更の【凍結】は spec.py/sigmf_io.py のみ（※粗い表現なので誤読注意）。

---

## 3. 全体パイプライン

```
[survey/サーベイ]  sweep_power(粗掃引) → dsp.detect_segments → アクティブ帯セグメント
        │                                    （survey_interval_s ごとに実行）
        ▼
[dwell/ドウェル]  _build_targets（survey検出＋バンドプラン巡回、最大 max_dwell_per_cycle）
        │
        ├─ 一発 dwell（dwell_mode=False）: capture_iq → measure_signal → classify（LLM 到達可）→ collect保存
        │
        └─ 滞在観測 dwell-observe（dwell_mode=True）: observe_dwell を反復
                 → 品質ゲート evaluate_quality（＋comb横断） → CNN監査（save候補のみ）
                 → SigMF 保存（method=rule/cnn）  ※LLM は到達不能（png_path 無し）
        ▼
[保存]  sigmf_io.write_recording → captures/<MHz>_<ts>_<n>.sigmf-{data,meta}（生IQ＋弱教師ラベル）
        store.Store.log は検出ごとに常時記録（保存可否と独立）
        ▼
[人間確定]  view_captures（PNG生成）→ cnntrain.review_suggest（CC提案・サンドボックス）
        → review.py（--suggest ○×UI で人間が y/n）→ apply_label（method=human, conf=1.0）
```

2つの dwell サブモードの分岐は `scheduler.py:396-402`。survey は `scheduler.py:127-134`、target選択は `_build_targets`（`scheduler.py:137-170`）。

---

## 4. モジュール地図

| ファイル | 役割 | 凍結状態 |
|---|---|---|
| `spec.py` | 正準スペクトログラム表現（STFT nfft=512/hop=256、256×256、dB正規化） | **【凍結】** |
| `sigmf_io.py` | SigMF 読み書き（annotation スキーマ、label/method/confidence/comment） | **【凍結】** |
| `main.py` | CLI エントリ。フラグ→Config→Scheduler 配線 | 凍結相当（追加のみ） |
| `config.py` | 全設定（SDR/Scan/Dwell/Quality/CNN dataclass）＋ BAND_PLAN（31バンド） | 自由 |
| `scheduler.py` | survey/dwell の2経路、target選択、CNN監査配線、SigMF保存 | 自由 |
| `dwell.py` | 滞在観測 `observe_dwell`（持続率・検出マージン・DCスパイク指標） | 自由（収集の心臓部・慎重） |
| `sdr.py` | `SimBackend`/`HackRFBackend`（SoapySDR）、`capture_iq`/`sweep_power` | 実装自由/署名凍結 |
| `dsp.py` | `measure_signal`/`detect_segments`/`welch_psd`/`remove_dc`/`dc_spike_metrics` | 実装自由/署名凍結 |
| `classify.py` | ルール分類器＋CNN監査フック＋LLMフックのオーケストレーション | 実装自由/署名凍結 |
| `quality.py` | 品質ゲート `evaluate_quality`/`flag_comb_spurs` | 自由（凍結同等の慎重さ） |
| `store.py` | SQLite 検出ログ（table `detections`） | 実装自由/署名凍結 |
| `review.py` | 人手レビュー/再ラベル。`apply_label`、`--suggest` ○×UI、`--include-human` 訂正 | 追加のみ |
| `view_captures.py` | SigMF→PNG（周波数軸・検出帯オーバーレイ） | 自由 |
| `dataset.py` | `load_index`/`Record`、`review`/`train` サブコマンド、`band_for_center`/`hw_group` | 自由 |
| `cnntrain/` | CNN 学習・推論・監査・合成データ・道具群（下記§5.2、§8） | 自由 |
| `llmvision/` | LLM vision 段（下記§5.3。完成物・現状維持扱い `FILES.md:41`） | 自由 |
| `eval/` | 外部モデル評価ハーネス（WBSig53等のドメインギャップ測定・M1配線） | 自由 |

主な CLI（`main.py`）: `--sim`/`--hardware`, `--start`/`--stop`, `--focus`, `--once`, `--collect`/`--collect-snr`, `--dwell`/`--dwell-seconds`/`--obs-interval`, `--capture-ms`（取得スナップショット長）, `--dwell-offset-hz`, `--q-*`（品質閾値上書き）, `--no-quality-gate`, `--no-dc-removal`/`--dc-removal`, `--cnn`/`--cnn-checkpoint`, `--save-spectrograms`, `--sim-dc-spike`。
`review.py`: `--conf-max`, `--list`, `--verdict C`, `--pattern`, `--suggest`, `--include-human`。

---

## 5. 三段分類の実装状況

オーケストレーションは `classify.classify`（`classify.py:245-270`）。3段は自然に直列: rule → CNN で確信度が下がれば → LLM が発火。

### 5.1 ルール分類器（`classify.py`）— 実装済み・常時稼働
- `rule_based(measurement, bands)`（`classify.py:80-108`）: center から `_match_band`（バンドプラン、priority で1つに決定）→ `SIGNAL_DB` を帯域幅条件で照合。候補があれば先頭を採用、`snr<6` で confidence×0.6、上限 0.85。バンド一致だがDB候補なしはバンド名をラベルに（conf 0.45/0.30）。バンド外は `UNKNOWN`（conf 0.20）。
- `SIGNAL_DB`（`classify.py:25-50`）: **17件**の5タプル `(band_substr, bw_cond, label, conf, note)`。
- ラベル語彙は計 **48種**（SIGNAL_DB 17 ＋ BAND_PLAN 名 31 − 重複2 ＋ `UNKNOWN`/`NOISE` 2）。`UNKNOWN="未識別信号"`、`NOISE="ノイズ/フロア変動"`（`classify.py:53-54`）。ルール段が自力で出すのは実質 SIGNAL_DB ラベル・バンド名・UNKNOWN（NOISE は LLM 段由来 ※`llmvision/core.py`）。
- バンドの `hint`（`[仮説] use=.../mod=...`）は **label/confidence を左右しない**。notes と LLM 文脈にのみ流れる（`config.py:172-181`）。
- `classify` は rule conf ≥ 0.85 で早期確定（`classify.py:249`）。conf<0.85 のみ後段へ。

### 5.2 CNN（`cnntrain/`）— 実装済みだが**既定 OFF**、合成のみ学習、**現状は単一の全帯域分類器（専門家分割は設計のみ・未実装）**
- **アーキテクチャ** `SmallSpecCNN`（`cnntrain/model.py:12-39`）: Conv(1→8→16→32→32, k3)＋MaxPool/AdaptiveAvgPool → FC(512→64)→Dropout(0.2)→Linear(64,5)。入力 `[B,1,256,256]`（`spec.render` 由来）、出力 **5クラス**、**48,293 params**。
- **クラス（方式軸・sorted 順）**（`cnntrain/classes.py:25-39`）: `cw-tone` / `narrowband-burst` / `noise-only` / `pulse-radar` / `wideband-ofdm`。実データのラベルは**用途軸**（BLE/WiFi/Zigbee）なので軸が違う（照合に期待対応表を使う）。※Kali の記憶にある「7クラス（narrowband-fm/chirp/burst-fsk/noise-floor 等）」はコード（classes.py）・checkpoint と一致しない（早期チャット案の記憶違い ※）。実体は上記5クラス。
- **【重要】単一の全帯域分類器であり、専門家分割は設計方針のみ・未実装（判定=(C)）**:
  - **実装**: CNN は 1 個の checkpoint（`config.CNNConfig.checkpoint`, 既定 `runs/m2_5`, `config.py:162-165`）を `scheduler.py:75-83` で1度だけロードするのみ。`config.Band` dataclass に**モデル指定枠は無い**（`config.py:175-`、フィールドは name/f_lo/f_hi/priority/hint のみ）。**帯域に応じてモデルを切り替えるルーティングは存在しない**。合成学習も中心周波数を 1–6GHz からクラス無相関に振る（`cnntrain/simgen.py`）＝**全帯域を1つのCNNで判定する構成**。git 履歴にも専門家分割/routing の実装コミットは無い。
  - **設計方針（記述はあるが未確定・未実装）**: 「1–6GHz を1つのCNNで判定するのは困難→**帯域ごとの専門CNN（専門家）**に分ける」方針が `BANDPLAN_PROPOSAL.md:98`（「2.4GHz/5GHz 専門家が『これは WiFi ではない』と弾く」）と `BANDPLAN_PROPOSAL.md:144-146`（**§6 未確定・要検討事項**「専門家は周波数で束ねる。境界は 2.4GHz 専門家を完成させてから…検証してから決める」「広帯域を1専門家で扱うか分割か」）に記述。`capture-engine_focus_task.md:13` も「2.4GHz 専門家のデータ収集」と表現。**ただし専門家の境界単位は明示的に「未確定」**で、**コードは未実装**（現状は "困難" とされた単一CNNのまま）。
  - 収集を 2.4GHz 先行にしているのは「2.4GHz 専門家を先に完成させる」方針（`BANDPLAN_PROPOSAL.md:112,144`）と整合する。
- **学習**（`cnntrain/train.py:83-217`）: `run_training`、Adam lr=1e-3、CrossEntropy、8 epoch。**合成のみをデータ層で強制**（`data.load_split` が `hw="sim"` でフィルタ、実データは構造的に除外 `data.py:30-34`）。実データ学習・fine-tune は経路に一切なし。
- **現状の実力（`runs/m2_5` チェックポイントで裏取り）**: val ≈ **95%**（真）、**実RF未経験**（真）。ただし俗説「**学習100%**」は**誤り**＝最終 train_acc **85.8%**（`runs/m2_5/train_log.txt`）。設計上の「火入れ（accuracy 追求ではない）」で非過学習。
- **合成データ**（`cnntrain/simgen.py`）: クラス別 IQ 合成（wideband-ofdm=非周期広帯域 / narrowband-burst=1〜3発 / cw-tone=中心外トーン / pulse-radar=厳密周期 / noise-only）。DC残留線・中心外スプリアスをクラス無相関で注入（CNNが手がかりに使えないように）。**全て凍結 `spec.render` を通す**。合成長 65536(~3.3ms) vs 実dwell 262144(~13ms)＝ドメイン差（※影響未測定）。
- **監査ロジック**（`cnntrain/audit.py:57-104`）: rule(用途) と CNN(方式) を期待対応表 `EXPECTED_REAL`（`expected.py:37-54`, BLE→narrowband-burst / WiFi→wideband-ofdm / Zigbee→両方。**[仮説]**）で照合し verdict を出す:
  - `unmapped`（対応表に無い）: 変更なし・所見のみ。
  - `A-consistent`（CNN∈期待）: conf+0.10（上限0.95）。
  - `B-context-explained`（文脈で説明可、例 2.4GHz WiFi×pulse-radar）: conf−0.05。
  - `C-conflict`（不一致）: conf=min(rule, 1−cnn_conf)。**<0.70 なら用途を Unknown 化**（元ラベルは候補に残し人間へ）。
- **来歴**は `classify.py:180-187` が SigMF **global** に記録: `sigscan:cnn_class/cnn_conf/cnn_verdict/cnn_checkpoint/rule_conf_pre/cnn_conf_post`。
- **発火条件**: `--cnn` 有効かつ rule conf<0.85 のときのみ（既定 `CNNConfig.enabled=False`, `config.py:164`）。checkpoint 既定 `runs/m2_5`（`config.py:165`）。CNN が用途ラベルを勝手に書き換えることはない（C は Unknown 化のみ＝人間判断へ）。
- ※ `classify.py:259-262` の `cnn_classify(spectrogram_db)` は**デッドスタブ**（常に None）。実CNNは上記監査経路。

### 5.3 LLM vision 段（`llmvision/`）— 実装済み＆配線済みだが**既定で休眠**
- **実装は本物**（スタブではない）。プロバイダ: `gemini-2.5-flash` / `claude-haiku-4-5` / `gpt-4o-mini`（`llmvision/client.py:44-48`、env で上書き可）。実 HTTP 呼出（`urllib`）。
- **配線**: `classify.py:264-265` が唯一の本番呼出。トリガは `png_path is not None かつ r.confidence < 0.5`。
- **到達条件（合流が必要）**: **非dwellモード**（`scheduler.dwell()`）＋ **`--save-spectrograms`**（png 生成）＋ rule/CNN 後の conf<0.5 ＋ **プロバイダAPIキー**（`GEMINI_API_KEY` 等）。`--dwell` モードは `png_path` 無しで**到達不能**。既定 `python main.py --sim` では png=None ＝**発火しない**。APIキー未設定なら graceful に None（`core.py:133-135`）。実行環境にキーが設定されているかは※未確認。
- **送信/受信**: スペクトログラム PNG（base64）＋テキストメタ（周波数・BW・SNR・バンド名・ルール所見・参照カタログ）。**生IQは送らない**（`llmvision/core.py:11`）。返りは JSON `{label, confidence, candidates, notes, rationale}` → `ClassResult(method="llm")`。
- `review.py` は llmvision を import しない（人間確定と LLM 段は無関係）。

---

## 6. 品質ゲート（`quality.py` / 閾値は `config.QualityConfig`）

入口 `quality.evaluate_quality(obs, qcfg, comb_spur=False, bw_hz=None)`（`quality.py:35`）。**6継ぎ目ではない**（署名は test_seams でロックされない）が「収集の心臓部」。全ゲートを評価し、発火した理由を累積（短絡しない）。`reasons` が空なら SAVE 候補。

前段の閾値: 1観測が「検出」とみなされる下限 `detect_snr_db=10.0`（`config.py:126`, `dwell.py:113` で使用）＝持続率の母数。

| ゲート | 発火条件（drop 理由） | 閾値（config.py） |
|---|---|---|
| Persistence（単発/低持続） | `n_detect < min_detections` → `transient` / `persistence < min_persistence` → `low-persistence` | `min_detections=3`(:129) / `min_persistence=0.34`(:130) |
| **narrow-steady-spur** | `narrow AND steady AND persistence≥spur_persistence_min`（`quality.py:52-53`） | `narrow_bw_hz=0.7e6`(:134) / `spur_snr_std_max=1.5`(:138) / `spur_persistence_min=0.9`(:139) |
| comb-spur（横断） | `flag_comb_spurs`: 狭帯域候補が等間隔（±`comb_spacing_tol_hz`）・同強度（±`comb_power_tol_db`）で `comb_min_run` 本以上並ぶ | `0.15e6`(:142) / `2.0`(:143) / `3`(:144) |
| **dc-spike** | `dc_excess_mean_db ≥ dc_excess_min_db AND dc_excess_std_db ≤ dc_excess_std_max`（中央集中**かつ**時間不変） | `dc_excess_min_db=12.0`(:152) / `dc_excess_std_max=3.0`(:153)（測定は `dc_band_hz=60e3`/`dc_side_hz=0.8e6`） |

**なぜ間欠BLEは通り、定常な狭帯域スプリアスは落ちるか（narrow-steady-spur の要）**: 両者とも狭帯域なので幅だけでは切らない。追加で「同一強度（`steady`）**かつ**ほぼ常時（`persistence≥0.9`）」を要求する。受信機内部CWスプリアスは強度一定・常時在→両条件成立→drop。BLE adv は間欠バーストで強度変動大（`steady=False`）＆不在の窓多い（`persistence<0.9`）→どちらかで外れて**残る**。BLE は保存に `min_persistence=0.34` を超えれば足りるが、スプリアスと誤認されるには `0.9` 超が必要＝`0.34≤persist<0.9` が「保存に十分・スプリアスには足りない」帯（`MEMORY.md` 参照）。

マスタースイッチ `enabled`（既定 True, `config.py:124`）を OFF にすると常に pass（3つの疑いフラグはメタ用に計算のみ）。verdict フローは `scheduler.dwell_observe_cycle`（`scheduler.py:286-363`）: observe → comb横断 → evaluate → SAVE候補（`collect_dir` かつ passed かつ `snr≥collect_snr_min` かつ 非重複）。

---

## 7. データ資産の現状（読み取り時点のカウント）

### ground truth（`method=human`、確定済み）— 実測カウント
- **`captures/` の human確定 計 37件**: **BLE/Bluetooth (adv?) 35**、**WiFi (2.4GHz, 20/40MHz) 2**。
- 達成チャネル: **ch38(2426) 3 ・ ch37(2402) 17 ・ ch39(2480) 15 = BLE 35**（`docs/worklog/...:13`。WiFi 2 は「棄却対象を正しく記録」した能動再ラベル）。
- `captures/`: `*.sigmf-meta` **118件**（method 内訳: rule 72 / human 37 / cnn 9）。全ラベル文脈: BLE 77 / Zigbee 24 / 未識別信号 14 / WiFi 2 / 移動体衛星通信 1。
- `captures/_review_pending/`: 83件（human **0**＝未確定）。`captures/_images/`: PNG 125枚。

### 合成データ・学習済みモデル
- 合成学習セット `simdata/`: `.sigmf-meta` **300件**（5クラス×60、`core:hw="sigscan-sim (synthetic)"`）。
- CNN checkpoint: **`runs/m2_5/checkpoint.pt`（198,754 B）が唯一の実体**。`runs/m1`・`runs/m2`・`runs/probe_real*` は **0バイトの空placeholder**。config 既定は `runs/m2_5`（実体あり）。

---

## 8. 確定フロー（収集レシピ）と道具

確立した一周（`docs/worklog/...:221`）:
```
collect(13ms) → view_captures.py → cnntrain.review_suggest（CC提案・サンドボックス）→ review.py（--suggest ○×UI・人間確定）
```

- **窓長を用途で分ける**（重要な設計判断）: **BLE収集は既定13ms**（`dwell_samples=1<<18`, `config.py:16`。時間変動が保たれ narrow-steady-spur が BLE を通す）。**duty解析は 400ms**（`--capture-ms 400`。`cnntrain.dutyprobe` の inconclusive を外す）。同じ窓に両立させない。
- **`cnntrain/dutyprobe.py`**: STFT行単位の在時率 duty（burst/continuous 判定の決定的審判）。snapshot<300ms は `inconclusive`（13ms 収集では結論不能＝正常）。measurement であってラベルではない。
- **`cnntrain/review_suggest.py`（サンドボックス・提案専用・SigMF 非改変）**: 各レコードの客観指標＋CCの視覚分類 `cc_class`（ble-adv/wifi/spurious/hopping/unclear）を並べ `bench/` に confirm_sheet を出力。`decide_recommend` は **cc_class 主導・duty 非依存**（13ms=inconclusive でも機能。空欄は黙って skip せず `needs-review`）。スプリアス誤確定ガード: det≈2400.0±0.1MHz または spur_suspect=True → skip 強制。`--auto-classify` で分類タスクリスト生成。
- **`review.py`**:
  - 対象選択3経路: `find_low_confidence`（rule かつ conf<conf_max）/ `find_c_conflict`（CNN監査 C-conflict）/ `--pattern`（ファイル名 glob 狙い撃ち）。`--include-human` で確定済みも呼び戻せる（**訂正経路**）。
  - `--suggest`（○×UI）: 対象は**走査＋フィルタで決め**、suggestions.csv は**提案 lookup 専用**。y=提案ラベルで確定 / n=ラベル選択。**摩擦（安全弁）**: `unclear`/`spurious_warn`/`needs-review`/**提案なし** は y を出さずラベル選択強制（Pattern A 化防止）。
  - `apply_label`: `core:label`/`method=human`/`confidence=1.0` を書き、元ラベルを comment に残す。訂正時は `sigscan:relabel_history` に `{from,to,at}` を append（過去確定を黙って上書きしない）。

---

## 9. HackRF 固有の癖（既知の内部スプリアス）

HackRF 実機にしか出ない固定パターン。**外部データには含まれず、外部学習では直らない**（実測 ground truth でスプリアスとして教える必要がある）。worklog carryover（`docs/worklog/...:211`）が挙げる主なもの:
- **DC残留線**（取得帯域中央=オフセット0Hz）。`dc_removal`（既定ON, `config.py:35`）＋ dc-spike ゲートで対処。
- **40MHz クロック高調波**（**2400 / 2440 / 2480 MHz**）。`review_suggest` は det≈2400.0±0.1 を spurious ガード対象にする（過去に 2400.0 連続線を BLE と誤確定しかけた事故の再発防止）。
- **16MHz コム**（等間隔ピーク列）。comb-spur ゲート（§6）で横断検出。
- 2420 固定線 等。
これらは「中央集中かつ時間不変」「狭帯域かつ定常」「等間隔同強度」といった**時間・周波数構造**で正規信号と切り分ける（品質ゲート§6の設計根拠）。

---

## 10. 現在地と次の一手

### 達成済み
- **BLE adv ground truth 3チャネル・37件（method=human）**（ch37/ch38/ch39）。
- 確定フローの自動化: dutyprobe / review_suggest（視覚主・needs-review）/ review.py の `--capture-ms`・`--pattern`・`--suggest` ○×UI・`--include-human` 訂正経路＋履歴。
- CNN 合成学習パイプライン（val≈95%）と監査経路（A/B/C/unmapped）。

### 未着手 / 未実装 / 未実運用
- **CNN 再学習（本丸・未着手）**: 実測 human確定を教師に CNN を鍛え直す。各クラス数十件が目安。※現状 BLE に偏在（BLE 35 / WiFi 2 のみ）。**専門家分割の下では「2.4GHz 専門家」として学習すれば教師は 2.4GHz の ground truth のみで足り、全帯域が揃うのを待つ必要はない**（帯域を絞ればクラス偏在も緩む・§5.2）。
- **専門家分割CNN（帯域別の専門CNN）— 設計方針あり・未実装（今回の調査で確定）**: 「1–6GHz を1つのCNNで判定するのは困難→2.4GHz/5GHz 等の専門CNN に分ける」方針が `BANDPLAN_PROPOSAL.md:98,144-146`（§6 未確定・要検討事項）にある。**コードは単一の全帯域CNN・帯域別ルーティング無し**（§5.2）。専門家の境界単位は同文書で「未確定・検証してから決める」。CNN再学習はこの分割方針の下で設計する必要がある。
- **LLM vision の実運用**: 実装済みだが既定 OFF・dwell 経路で到達不能・要APIキー（§5.3）。
- **外部IQ事前学習**（将来・厳格な条件付き）: 特徴初期化専用・分類ヘッドは自分の ground truth のみ・Pattern A 厳守（`docs/worklog/...:208-213`）。
- 5GHz 帯展開、他チャネル（ch39 の `--dwell-offset-hz 4e6` 実戦投入 等）。

### 既知の課題・TODO（実装せず記録のみ）
- **BW 過大測定**（`KNOWN_ISSUES.md:7-24`）: `dsp.measure_signal`/`detect_segments` がスプリアス線を過剰併合し帯域幅を過大評価する疑い。未着手の調査課題。
- **captures/*.sigmf-meta が cp932/Shift-JIS エンコード**（UTF-8ではない）。`open()` はロケール既定なので日本語Windowsでは動くが、UTF-8ロケール機では壊れる（※移植性リスク・意図的かは未確認）。
- `runs/m1`・`runs/m2` が 0バイト空 placeholder（config 既定は実体のある m2_5 なので実害なし）。
- `classify.py` の `cnn_classify(spectrogram_db)` はデッドスタブ。
- 合成(3.3ms) vs 実測(13ms) のドメインギャップ未測定（`SYNTHETIC_ONLY_TAG="domain gap unmeasured"`, `classes.py`）。
- `dutyprobe` の位置づけ: 測定（時間占有）であって信号種同定ではない。WiFi/BLE は両方低 duty で分離不能。

---

## 11. 運用の実務（環境・コマンド）

- **Python**: `C:\Users\puppy\radioconda\envs\sigscan\python.exe`（conda 環境 `sigscan`、Python 3.11.x）。フルパス運用。
- **OS/シェル**: Windows + PowerShell。**`&&` 連結禁止（1行ずつ）**。SQLite ログの `--db nul` 禁止（必要なら実ファイル）。
- **テスト**: `& $py -m pytest -q` → 現状 **214 passed, 3 skipped**（torch 未導入環境では CNN/eval 系が skip）。
- よく使うコマンド:
  ```powershell
  & $py main.py --sim --collect captures/ --collect-snr 8                # Sim 収集
  & $py main.py --hardware --start 2.4018e9 --stop 2.4022e9 --focus --dwell-seconds 10 --q-min-persistence 0.2 --collect captures/  # BLE収集(13ms)
  & $py view_captures.py captures/                                        # PNG生成
  & $py -m cnntrain.review_suggest --data captures/ --pattern "2480MHz_*" --out bench/xxx/ --auto-classify  # CC提案
  & $py review.py captures/ --pattern "..._11*" --include-human --suggest bench/xxx/suggestions.csv          # 人間確定(○×UI)
  ```
- 収集(`main.py --hardware`)は物理HackRFが要る＝人間（Kali）の領分。CC はソフト側・提案側を担い、確定はしない。

---

## 12. このドキュメントの更新規律

- 本書は「**何であるか**」の静的な地図。時系列の「何をしたか」は `docs/worklog/` に書く（重複は最小に）。
- 設計思想・凍結契約・継ぎ目・分類段の実装状況・データ資産件数が**変わったとき**に更新する（例: CNN 再学習でクラス/実力が変わる、LLM vision を実運用化する、新しい継ぎ目が増える）。
- 事実と推測を分け、断定は必ずコードで裏を取る。古い数字（ground truth 件数・pytest 数・チェックポイント）は更新時に実測し直す。
</content>
</invoke>
