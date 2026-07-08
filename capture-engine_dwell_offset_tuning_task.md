# 作業指示書: capture-engine — dwell オフセットチューニング（狙った獲物が DC で必ず落ちる構造の解消）

## 役割
あなたは sigscan プロジェクトの capture-engine エージェント。
本指示は、dwell 収集が**ターゲット信号にチューナーを直付けする（= 獲物が必ず DC 位置に乗る）**
設計を改め、**チューナー中心を数 MHz オフセットして獲物を DC から避ける**（ゼロIF 受信機の
オフセットチューニングという SDR の定石）実装。狭帯域・高 SNR の獲物（BLE 等）が
dc-spike ゲートで**構造的に必ず**捨てられる問題を解消する。

## 背景（人間の実測とコード読解で確定済みの事実）
1. **DC 直付けの現行設計**: `scheduler.py:176-177` が `center_hz = target["center"]` を
   そのまま `capture_iq` に渡す。dwell の観測ループ（`dwell.py:88` 付近）も同様の構造。
   → 狙った獲物は必ず IQ の DC 位置に乗る。
2. **帰結（実測）**: 2402MHz（BLE ch37）を狙った dwell で、獲物本体が DC 位置に写った状態で
   dc-spike 判定により破棄（SNR 45dB の明確な信号でも落ちる）。viewer 軸修正後の画像
   `captures/_images/2402MHz_1783315966234_1.png` の t=0〜0.1・2402MHz に「落とされた獲物」が
   淡く写っており、機序の視覚的証拠になっている。
3. **オフセットの有効性の傍証（実測）**: 記録 `2489MHz_1783315909650_0` では、DC から
   約 9.3MHz 離れた位置の狭帯域検出が dc-spike にかからず保存された。
   → DC から離れてさえいれば狭帯域でも保存される。
4. **窓の端は避ける**: 上記 9.3MHz はナイキスト端（±10MHz）に近くロールオフが不安。
   オフセットは 4MHz 程度（判定域から十分遠く、端からも 6MHz 内側）が妥当。
5. **既知の定常スプリアス**: 40MHz クロック高調波（2400/2440/2480MHz、絶対周波数固定）と
   16MHz 櫛（2408+16n、タイル DC 由来）。オフセットしても窓内にこれらが入り込むことはあり、
   **本修正は dc-spike の構造落ちを解くだけで、混獲（狙いと別物の保存）問題は解かない**。

## 最重要原則（絶対厳守）
1. **6継ぎ目のシグネチャ不変**: spec.render / sdr(sweep_power, capture_iq) /
   dsp(measure_signal, detect_segments) / classify.classify / sigmf_io / store。
   **渡す値を変えるのは可、シグネチャ・関数内挙動の変更は不可**。
   `git diff --stat -- spec.py sigmf_io.py` は空であること。
2. **opt-in 設計（既定 0 = 完全に従来挙動）**: オフセット既定値は 0（無効）。
   既存テスト・既存収集挙動は一切変わらない。有効化は config / CLI で明示。
   実機で有効性が実証されたら既定値昇格を人間が別途判断する。
3. **絶対周波数の一貫性（本修正の技術的核心）**: チューナーを実際に合わせた周波数
   f_tune を、**capture_iq / measure_signal / 記録（write_recording に渡す center）の
   すべてに一貫して渡す**。これにより annotation の絶対周波数・SigMF captures の
   core:frequency は自動的に正しくなる。target["center"]（狙い値）自体は書き換えない。
4. **サーベイ（sweep_power）にはオフセットを適用しない**。dwell 収集経路のみ。
5. **quality.py は一切触らない**（ゲート側の調整は別テーマ）。
6. **既存テストは無変更**。追加のみ。
7. **Sim-first**: 実機前に SimBackend で経路を実証する（本プロジェクトの成功パターン）。
8. **最小実装**。迷ったら最小。凝った機能は将来課題コメントに残す。

## 事前確認（実装前に必ず。結果を完了報告に含める）
1. **dwell 収集経路の全洗い出し**: capture_iq を呼ぶ箇所（scheduler.py:177 /
   dwell.py:88 / 他にあれば全て）と、その IQ に対する measure_signal 呼び出し、
   write_recording への center の流れを図示する。center が二重管理されている箇所が
   ないか（片方だけ f_tune にすると絶対周波数がずれる）。
2. **獲物の照合ロジック**: dwell 中・保存時に「検出信号がターゲットのものか」を
   どう選んでいるか（周波数照合か・窓内最強選択か）。窓内最強選択の場合、オフセット後も
   挙動は従来と同型（混獲リスクは従来どおり残る）であることを確認し報告。
   もし「DC 近傍優先」等の中心依存ロジックがあれば**実装前に停止して報告**。
3. **main.py の現行 CLI 構造**: フラグ追加が既存挙動を一切変えない形で可能か。
   main.py は凍結相当の扱いのため、**変更は新フラグ追加のみ・diff を完了報告に全掲**。
4. **SimBackend の DC オフセット擬似**（sdr.py:75 付近の診断機能）の使い方を確認
   （dc_excess を人工的に立てられるか）。
5. **simgen の narrowband-burst の窓内位置分布**（中心固定か、オフセット位置も分布内か）。
   確認のみ・変更禁止。分布外なら「オフセット収集の実データは CNN 合成の位置分布と
   ずれる」旨を KNOWN_ISSUES 向けメモとして報告（対処は将来課題）。

## 実装

### 1. config にパラメータ追加（既定で無効）
- `dwell_offset_hz`（float, 既定 0.0）: dwell 時にチューナー中心へ加算するオフセット。
- `dwell_offset_max_bw_hz`（float, 既定 8e6）: オフセットを適用するターゲット帯域幅の上限。
  広帯域（WiFi 等）にオフセットを適用すると信号端が窓外にはみ出すため、
  **target の帯域幅がこの値以下の狭帯域ターゲットにのみ適用**する。
- 既存 config 構造（cfg.sdr.* の流儀）に合わせて配置。

### 2. dwell 収集経路にオフセット適用
擬似コード（実際の変数名・構造は既存コードに合わせる）:

```python
off = cfg.sdr.dwell_offset_hz
bw = target.get("bw") 等（サーベイ検出の帯域幅。取れない場合の既定は「適用する」side ではなく「適用しない」side に倒す）
f_tune = target["center"] + (off if (off and bw is not None and bw <= cfg.sdr.dwell_offset_max_bw_hz) else 0.0)

iq = backend.capture_iq(f_tune, rate, n)
m  = dsp.measure_signal(iq, rate, f_tune)   # 絶対周波数の一貫性
# 記録時の center も f_tune（IQ の物理中心）を渡す
```

- scheduler.dwell と dwell.py の観測ループの**両方**（および事前確認1で見つかった全経路）に
  同じ規則で適用。適用判断のロジックは 1 箇所（小さなヘルパ関数）に集約し、経路間の
  不一致を作らない。
- `target["center"]`・annotation に入る検出周波数の意味は不変（絶対周波数）。

### 3. main.py に CLI フラグ追加（追加のみ）
- `--dwell-offset-hz`（既定: config 値）。指定時のみ config 値を上書き。
- 既存フラグ・既定挙動は完全不変。diff を完了報告に全掲。

### 4. SigMF へのトレーサビリティ記録
- write_recording の extra_global 経由で `sigscan:dwell_offset_hz`（適用した実効値。
  不適用なら 0.0）を global に記録。annotation の許可キー
  （confidence/method/snr_db）には触らない。
- 旧記録との区別はキーの有無で可能になる。

### 5. dwell の dc_spike 指標（dwell.py:90 付近）は無変更
- オフセットにより獲物が DC から離れるため、dc_excess は真の DC 残留のみを測るようになる。
  これは狙いどおりの副次効果であり、指標側の変更は不要。

## テスト（追加のみ）
1. **ユニット（適用条件）**: (a) offset=0 → f_tune=center（完全従来）。
   (b) offset=4e6・bw=1.5e6 → f_tune=center+4e6。
   (c) offset=4e6・bw=16e6（> max_bw）→ f_tune=center（不適用）。
   (d) bw 不明 → 不適用。
2. **Sim 統合（Sim-first の実証）**: SimBackend + DC オフセット擬似で狭帯域信号を配置し、
   (a) offset=0 で当該ターゲットが dc-spike により drop されることを再現
   （再現できない場合は、dc-spike 判定の実条件をコードから引用して理由を報告。
   テストを緩めて無理に緑にしない）。
   (b) offset=4e6 で保存され、**annotation の絶対周波数が Sim 信号の真の周波数と
   一致（±測定分解能）**すること。ここが崩れると全記録の周波数が静かに壊れるため、
   本テスト群で最重要。
   (c) 広帯域ターゲットでは不適用（f_tune=center）のまま従来どおり動くこと。

## 検証（人間・実機。完了報告には手順の提示まででよい）
- BLE ch38（2426MHz。既知スプリアス 2400/2440/2480 と 2408+16n 櫛から最も遠い）狙いで
  短時間収集を offset なし/あり（--dwell-offset-hz 4e6）で比較:
  1. drop ログ: なし側で dc-spike 落ち、あり側で解消。
  2. あり側の保存記録を viewer で目視: 獲物が DC から −4MHz の位置に写り、
     検出帯マーカー（絶対周波数）は獲物の真の位置に重なる。
  3. low-persistence の挙動が裸で観測できる状態になる（本修正のもう一つの目的）。

## 完了報告に含めること
1. 事前確認 1〜5 の結果（特に center の流れ図と照合ロジックの型）。
2. 変更ファイル一覧と diff 要約。main.py の diff は全掲。
3. pytest 全緑（既存無変更・追加のみ）と、追加テストの一覧。
4. `git diff --stat -- spec.py sigmf_io.py` が空。
5. **本修正が解くもの／解かないものの正直な記載**: dc-spike の構造落ちは解く。
   混獲（窓内の別定常物を保存する）問題と low-persistence 問題は未解決のまま。
6. 実装が本指示書の範囲内である宣言。

## 禁止事項
- quality.py / 6継ぎ目関数の内部・シグネチャ変更。
- spec.render 出力への加工・正規化変更。
- サーベイ（sweep_power）へのオフセット適用。
- 既存テストの変更・削除・弱体化。テストを緩めて緑にする行為。
- 既定値を 0 以外にすること（既定は完全従来挙動）。
- スコープ膨張（獲物追尾・自動オフセット最適化・quality 調整等は将来課題コメントへ）。
