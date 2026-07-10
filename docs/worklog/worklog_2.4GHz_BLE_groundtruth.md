# sigscan 作業ログ — 2.4GHz ISM / BLE ground truth

> このファイルは**人間が読んで経緯を追える**ための実験ノート。`bench/` のような使い捨てサンドボックスや、機械可読のCSV・コードとは別物。「何を・なぜ・どう判断したか」を残す。
>
> **置き場所**: `docs/worklog/`（git 追跡する。使い捨ての `bench/` とは別）。
> **更新規律**: 実験セッションごとに1エントリ追記。各エントリは 狙い / やったこと / 分かったこと / 保留 / 成果物 / 次アクション。

---

## このログのスコープ

BLE advertising の ground truth を 2.4GHz ISM で積む取り組みの記録。
- **主目標**: BLE adv を各チャネルで human確定して ground truth を貯める。**ch38(2426) 3件・ch37(2402) 17件 達成済み（method=human。計20件）**。ch39(2480) が次。
- **副次目標**: dc-spike説の白黒 —「DC上に乗った間欠BLEが dc-spike ゲートで落ちるか」。過去に「必ず落ちる」と結論しオフセット実装の動機にしたが、要再検証に格下げ済み。**→ Entry①②で「白」寄りが補強：オンDCの間欠BLEはdc-spikeで落ちない（落ちるのは別ゲート narrow-steady-spur、しかも13msでは落ちない）**。

### プロジェクト背景（1分版）
HackRF One で 2.4GHz をスキャン→信号自動識別。3段（ルール分類器→CNN監査→LLM vision）、SigMF保存。設計思想は「**AIが提案・人間が承認**」。ルールやCNNやAIの出力を**CNNの学習ラベルにしない**（ラベル汚染防止）。ground truth は人間が `review.py` で視覚確認して確定したときだけ成立。凍結契約が spec.py / sigmf_io.py / main.py / observation/ を守る。

---

## エントリ 2026-07-08 ①: ch37(2402) 初回収集 ＋ CC精確性ベンチマーク

### 狙い
1. ch37 を録って ground truth 2チャネル目を目指す。
2. 同時に dc-spike説の白黒（間欠BLEをオフセット0でDC上に乗せ、ゲートを通るか）。
3. 追加実験: 「人間の視覚分類 vs Claude Code(CC)の視覚分類、どちらが審判(在時率duty)に近いか」をサンドボックスで測る。

### やったこと（時系列）
1. **対照ラン（ゲート無効）** — 存在と幾何の確認用。
   ```
   main.py --hardware --start 2.401e9 --stop 2.403e9 --focus --dwell-seconds 10 --no-quality-gate --collect captures/_review_pending/
   ```
   → 収集16件。検出は全て **2401.1MHz**、幅は 1.2MHz(BLE幅) と 0.2–0.5MHz(狭帯) が混在、**全て persist=1.00**。

2. **ゲートラン（persist0.2）** — 本番想定。
   ```
   main.py --hardware --start 2.401e9 --stop 2.403e9 --focus --dwell-seconds 10 --q-min-persistence 0.2 --collect captures/
   ```
   → 収集11件。1.2MHz幅はSAVE、0.2–0.5MHzの狭帯は一部 `[drop:narrow-steady-spur]` で棄却。

3. **CC精確性ベンチマーク（サンドボックス）** — 指示書を CC に丸投げ。
   - 審判ツール `cnntrain/dutyprobe.py` を新規作成（IQのSTFT行単位で在時率dutyを測る決定的ツール。`spec.stft_db` を import 流用）。合成既知dutyのテスト8件でピン留め（172→180 passed）。
   - CCが対象PNG 27枚を**盲検分類**（duty計算前に burst/continuous を確定・凍結）。
   - duty測定→比較表 `bench/cc_vs_human_2401/benchmark.md` 生成。

### 分かったこと（結論）
1. **dc-spike説は「白」寄りの証拠が出た**。tuner 2401.1 ≒ det 2401.1（オフセットほぼ0＝オンDC）の**間欠1.2MHz信号がゲートを通過**した。落ちたのは狭い0.2–0.5MHzの**定常**スプリアスだけで、理由は `narrow-steady-spur`（dc-spike ではない）。dc-spike判定の「中央集中 **かつ** 時間不変」の2条件設計と整合。ch38(+2MHzでオンDCでなかった)と違い、今回は実際にオンDCで試せた。
2. **2401の信号は間欠（duty≈0.02–0.06）＝連続エミッタ混獲ではない**。連続なら隣の2400線と同じ duty=1.0 になるはず。先週の「連続混獲(イヤホン型)を録っている」懸念は棄却。
3. **連続だった2件（rp_3=1020914 / rp_15=1396911）は det≈2400.0・spur_suspect=True ＝ 40MHzクロック高調波（既知の内部スプリアス）**。duty=1.000。新種の外部汚染ではない。ゲートONなら narrow-steady-spur で落ちる。既知ファミリーと整合。
4. **ターゲティングがズレていた（重要）**。tuner中心が **2401.1** に張り付き、検出帯も [2400.5, 2401.6]（中心2401.05）。ch37の実センタは **2402**。**約1MHz低い所を録っていた**。2401はBLEのadv/dataどちらのチャネルでもない。
5. **精確性ベンチマークは判別力ゼロだった**。(a) 審判が **27/27件 inconclusive**（snapshot 13.1ms << 必要な300ms、adv間隔の隙間を分解不能）。(b) inconclusiveを措いても duty が **自明に二極化**（0.02–0.06 vs 1.000、中間ゼロ）＝人間もCCもほぼ100%当てる＝**2人の判定者を区別する力がない**。「CC 27/27一致」は、CC自身がinconclusiveを返した審判に自明ケースで一致しただけ。**CCが人間より精確という証拠にはならない**。

### 保留・未確定（次セッションが誤解しないために）
- **2401の間欠信号が BLE ch37 adv である確証はない**（2402から約1MHz低い）。→ `review.py` の **human確定は保留**。ch37を録るつもりで別物を録っている可能性が残る。
- **思想変更（CCに視覚判断を委譲するか）は保留**。今回のベンチマークは判別力ゼロで何も採点していない。君の教訓5（実測1回・不十分な状況から根本を断定しない）がそのまま効く。判別力あるベンチマーク（長尺＋hardケース）待ち。

### CCの実力について（所感）
- **自己完結・テスト可・規律明確なタスクでは自己訂正まで含めてほぼ完璧**。特筆: 「git statusでcaptures未変更」を証拠にしようとして、captures/がgitignore対象＝その証拠は無効だと**自分で気づき**、mtime（54ファイルの最新更新が実行帯の72分前・直近30分の書換0件）というより強い証拠に差し替えた。
- **唯一の差**: CCは事前確認で「13ms<300→全件inconclusive確定」と**気づいていながら**そのまま完遂した（指示書が完遂を指示・停止して問い直す権限を与えていなかったため）。「このベンチマークは何も採点できない、先に長尺収集すべきでは？」と**フレーム自体を問い直す初動**は人間側が出した所。＝枠が正しければCCは強力、枠の是非を見抜く判断は人間がまだ価値を出した、という切り分け。

### この日の成果物
- `cnntrain/dutyprobe.py`（審判ツール）— **残す**。長尺収集後に本領を発揮する。
- `tests/test_dutyprobe.py`（+8、既存172無変更 → 180 passed / 3 skipped）。
- `bench/cc_vs_human_2401/`（cc_verdicts.csv / duty_captures.csv / duty_review_pending.csv / benchmark.md）＝**使い捨てサンドボックス**。参考として保持。
- **ground truthへの昇格: なし**。`captures/` は無変更（mtimeで確認）。凍結契約 diff 空。

---

## エントリ 2026-07-08 ②（計画・実行後に結果追記）: ch37 長尺再収集 @ 2402

### なぜ再収集か
①でのズレ2点を直す:
1. ターゲットが2401（ch37実センタ2402より約1MHz低）だった → 2402に合わせ直す。
2. snapshot 13ms（<300ms）で審判が全件inconclusiveだった → 300–500msに延ばす。

### 計画
1. **ターゲット修正**: 窓を **2.4018–2.4022** に絞り、2400高調波(2400)と2401エミッタ(2401)を候補から除外。dwellが2402に寄り、BLE ch37 が DC上(オフセット0)に乗る。
2. **snapshot延長**: **300–500ms**（6–10Mサンプル@20MHz、約48–80MB/件）。inconclusive解消＋隙間が複数見える＝dutyの真のhardケースが生まれる。
   - ※ capture IQ長のCLIノブは要確認（現状 2^18=13ms 固定に見える）。既存フラグが無ければ `--capture-ms` を1本足す小CC案件。
3. **persist0.2据え置き**、DC除去既定ON。
4. **環境**: 既知BLE源1台のみ動作、他の2.4GHz機器オフ。

### 見るべきこと / 判定基準
1. **2402にBLE ch37 advが実在するか**（そもそも出ているか。出ていなければ候補が立たない＝それも発見）。
2. **duty**: 長尺で 0.02–0.05 のバーストか（inconclusive外れる・隙間が複数見えるか）。
3. **dc-spike**: オンDC(2402=DC)の間欠BLEがゲート通過するか（白の確証を2402で取り直す）。
4. これがch37 advと**視覚＋メタで確認できたら** `review.py` で human確定 → **ch37 ground truth**（2チャネル目）。tuner表示ではなく annotation周波数が真値（2402近傍のはず）。

### コマンド（capture長ノブ = `--capture-ms` で解決済み。以下が確定版）
```powershell
$py = "C:\Users\puppy\radioconda\envs\sigscan\python.exe"
```
```powershell
# --capture-ms 400 で1スナップショットを 400ms(=8Mサンプル≒64MB/件)に延長 → duty審判の
# inconclusive(<300ms)を外し adv 間隔の隙間を複数見せる。長尺は dwell が観測ごとの生IQを
# 保持しメモリを食うため dwell-seconds は短め(4s)に(持続率の母数=観測回数は減る点に留意)。
& $py main.py --hardware --start 2.4018e9 --stop 2.4022e9 --focus --dwell-seconds 4 --obs-interval 0.5 --q-min-persistence 0.2 --capture-ms 400 --collect captures/
```
```powershell
& $py view_captures.py captures/
```
```powershell
# duty審判（長尺なら inconclusive=False で burst/continuous を結論可能）。収集後の実 det
# 周波数プレフィックス(例 2402MHz_*)で --pattern 絞り込み可。
& $py -m cnntrain.dutyprobe --data captures/ --out bench/duty_ch37_long.csv
```

### CC実装ノート（2026-07-08・ソフト準備のみ / ハード収集と human確定は Kali）
Entry② の前提ブロッカー「capture長のCLIノブ」を CC が解決。CC が触れたのはソフトのみで、
2402 の実収集・視覚確認・`review.py` human確定は Kali の領分（下の結果欄）。

- **調査**: capture長は `config.SDRConfig.dwell_samples = 1<<18`(≒13ms)。上書きする既存CLIフラグは
  無し（main.py 精査で確認）。`scheduler.py:201,304` が両取得経路とも `cfg.sdr.dwell_samples` を
  読むので、上書き1箇所で survey/dwell 両経路に効く。凍結契約の `observation/` は実在せず、観測は
  `dwell.py`（`observe_dwell(..., n_samples)` は既にパラメータ化済み＝触れる必要なし）。
- **追加した `--capture-ms MS`**（main.py は「凍結相当」＝新フラグ追加のみの許可パターンに従う）:
  ms→サンプル数の変換は純関数 `config.dwell_samples_for_ms`（テスト可能）に置き、main.py は
  `if args.capture_ms is not None: cfg.sdr.dwell_samples = ...` の1行上書き＋情報表示だけ。
  **spec.py / sigmf_io.py / dwell.py / scheduler.py は不変**、凍結契約(spec/sigmf_io) diff 空。
- **検証**: `tests/test_capture_ms.py`(+5、既存無変更 → 185 passed / 3 skipped)。sim end-to-end で
  `--capture-ms 15` → 保存 SigMF の sample_count=300000(=15ms×20MSPS) を確認（ハード不要で
  ノブが保存IQ長まで波及することを実証）。
- **メモリ注意（Kali へ周知）**: `dwell.py` は滞在中の各観測の生IQを全て保持する。400ms=8Mサンプル
  ≈64MB/件。--dwell-seconds 10 / obs-interval 0.5 だと ~20観測 → ピーク ~1.3GB。上のコマンドは
  dwell-seconds 4(~8観測 ≈0.5GB)に抑えた。main.py も 128MB/件超で警告を出す。

### 結果（2026-07-08 夜〜07-09・2手のランで実行）

計画（2402ターゲット・400ms長尺）を実行。2手に分かれた。

**ラン1（イヤホン電源ON・交絡発覚）**
2402近傍が幅4–10MHzの信号に支配され、ルールは大半を "Zigbee/独自2.4G" と誤ラベル、時々SNR38–52dBの強候補が湧いた。当初これを「WiFi級広帯域が支配」と読んだが**誤読**。スペクトログラム目視で正体判明＝**Bluetoothイヤホン（BT Classic/A2DP）の周波数ホッピング(FHSS)**。「幅7–10MHz」は400ms窓がホップ先を1検出に塗り広げた**測定アーティファクト**で、実体は離散バーストの集合。教訓7（イヤホン混獲）の再演。環境管理リストからイヤホンが漏れていた。→ このデータはground truth不可。

**ラン2（イヤホン電源OFF・BLE ch37確認）**
交絡を消すと局面が一変。検出帯に**離散1.1–1.2MHz幅のバースト**（spur_suspect=False・2400線と分離）＝正真正銘のBLE ch37 adv が出た。tuner≈det≈2402（オンDC）。狙い通り2402にch37 advが実在することを確認。別に1件、上下端まで貫く20MHz級のWiFiフレームも混入（ルールはZigbee誤ラベル＝400ms切取アーティファクト）。

**設計衝突を実証（400ms × narrow-steady-spur）**
ラン2のログはほぼ全行が `BLE(adv?) BW=1.0–1.2MHz [drop:narrow-steady-spur]`＝**400msではnarrow-steady-spurがBLEを落とす**（収集4件のみ）。イヤホンOFFで交絡が消えて初めて、これが環境でなくゲート×窓長の衝突だと切り分いた。機序：narrow-steady-spurは「狭帯域 かつ 定常(観測間std小)」で発火。13ms窓ではadvバーストの有無で観測間エネルギーが激しく変動→非定常→BLE通過（Entry①のゲートランで1.2MHzは全SAVE）。400ms窓では1窓に数バーストが入り平均化→観測間std小→定常誤認→BLE棄却。**400msが、ゲートが頼る"時間変動"の手がかりを消した**。

**解決＝窓長を用途で分ける（ゲートは不変）**
標準ルール「narrow-steady-spurはBLE収集失敗を理由に触らない」通り、ゲートは正しく、壊れたのは窓長。よって: (1) **BLE収集は既定13ms**（`--capture-ms`無し。時間変動が保たれBLEが通る。ch38実証済みレシピ）、(2) **400msはゲート無しのduty解析専用**（`--capture-ms`は無駄でなく用途が違うだけ）。「duty審判は長尺要求／narrow-steady-spurは時間変動要求」の矛盾を、同じ窓に両方させないことで解く。

**訂正の記録**：Claudeの「2402を広帯域が支配」は誤読。実体はイヤホンFHSSで、電源OFFで消滅。次セッションが同じ誤読をしないよう明記。

---

## エントリ 2026-07-09: ch37 ground truth 確定 ＋ CC提案ツール導入

### 狙い
ラン2で得たBLE ch37 adv候補を human確定し ground truth 2チャネル目を達成する。確定作業のミス（タイムスタンプ名の取り違え・2400スプリアス誤確定）を減らすため、CC提案ツールをサンドボックスで導入する。

### やったこと
1. **CC提案ツール `cnntrain/review_suggest.py`（サンドボックス・新規）** をCCに実装させた（指示書丸投げ）。対象ごとに duty（dutyprobe流用）・freq/bw/snr・spur_suspect を集め、CCがPNGを視覚分類（ble-adv/wifi/spurious/hopping）＋根拠を付し、`bench/review_suggest_ch37/` にCSVと確定シートを出力。**CCは確定しない・SigMFに書かない**。スプリアスガード（det 2400±0.1 か spur_suspect=True → skip強制）をコードで強制。
2. **`review.py` に `--pattern GLOB` 追加（追加のみ）**。`2402MHz_1783530*` で今日の4件だけをレビュー列に出せる。apply_label・既存モード不変。
3. **人間確定（review.py）**：CC提案を参考に、PNGを自分の目で確認して確定。

### 分かったこと / 結果
1. **ch37 ground truth 達成（2チャネル目）**。BLE adv 3件（`_413566_0` / `_460361_1` / `_864501_3`、det≈2402）を human確定（method=human）。ch38の3件と合わせ**2チャネル・計6件**。プロジェクト最大のボトルネックだったground_truth蓄積が明確に前進。
2. **WiFi 1件（`_778007_2`）を能動再ラベル**。skipせず [4]WiFi(2.4GHz) として確定＝棄却対象のWiFiを"WiFiである"と正しく記録（次に見る人/CNNのため）。
3. **CC提案の的中**：`_0/_1/_3→ble-adv`、`_2→wifi`。人間の目・Claudeの目と**3者一致**。
4. **duty の限界を実証**：WiFi(`_2`)の duty も 0.014 と低く、**duty(burst/continuous)ではWiFiとBLEを分離できない**。決め手は周波数構造（上下貫通 vs 帯域内狭バースト）＝視覚。duty審判は「間欠か連続か」専用で信号種の同定はしない。

### CCの実力について（3点目のデータ）
- **設計思想に隣接する微妙な一線（"提案はするが確定はしない"）をCCが自分で守り切った**。apply_labelを触らず、method=humanを付けず、SigMFに書かず、スプリアスをconfirmにしない——全部指示書通り。**枠が明確なら思想の要に隣接する作業でもCCは逸脱しない**。
- ただし**"CCが人間より精確か"はまだ未決**。今日も自明な4件（明快なBLE vs 明快なWiFi）で判別力あるケースではない。持ち越し#2のまま。

### この日の成果物
- `cnntrain/review_suggest.py`（提案ツール・新規、dutyprobe流用）。
- `tests/test_review_suggest.py`（+9、既存無変更 → 194 passed / 3 skipped）。
- `review.py` に `--pattern`（追加のみ、apply_label不変）。
- `bench/review_suggest_ch37/`（suggestions.csv / confirm_sheet.md / cc_verdicts.csv）＝サンドボックス。
- **ground truth**: BLE ch37 adv 3件 method=human（＋WiFi 1件 method=human）。

---

## エントリ 2026-07-09 ②: ch37 厚み足し（13ms収集・14件確定）＋ review_suggest の穴発覚

### 狙い
Entry②で確定した設計判断（BLE収集は13ms）でch37の厚みを足し、CNN再学習に向けた件数を稼ぐ。同じレシピ（collect→view→review_suggest→review）を13msで回す。

### やったこと
1. **13ms収集**（イヤホンOFF維持・`--capture-ms`無し=既定13ms・dwell10s・persist0.2、窓2.4018–2.4022）→ **14件SAVE**（新バッチ `2402MHz_178356*`）。
2. review_suggest に新バッチを通す（`--pattern "2402MHz_178356*"`、出力 `bench/review_suggest_ch37_b/`）。
3. review.py で人間確定。

### 分かったこと / 結果
1. **13msで narrow-steady-spur × BLE の衝突が消えた**。400msの棄却地獄（Entry②）が消滅し、SNR24–30dB・1.1–1.3MHzのBLEがSAVE連発。**Entry②の窓長判断（収集は13ms）を実証**。
2. **ch37 adv 14件を human確定**（全PNGが検出帯≈2402内の明快な離散バースト・2400線と分離・spur_suspect=False）。**ch37 計17件（Entry③の3＋今回14）・全体計20件**。CNN再学習の厚みに前進。件名周波数表示は tuner中心≈2401.9 だが実信号は annotation 側（2402近傍）が正。
3. **review_suggest の穴発覚（重要）**：13msで duty が inconclusive になり、**CCが視覚分類を埋めず全14件を skip・`cc_class`空欄で提案**＝提案が空振り。原因は**指示書がinconclusive時の視覚分類フォールバックを明示していなかった**（枠の穴。前回400msでは同じ種類の信号を正しく ble-adv にした＝**CCの視覚能力の限界ではない**）。CCは"提案しない(全skip)"側に倒れた＝**安全側の失敗でラベル汚染ゼロ**。今回は人間がPNG目視で確定して回避（全14件を[5]BLEで確定、PNGで離散バースト＋2400線分離を確認済み）。

### CCの実力について（3.5点目のデータ）
- 道具の目的（提案を速く・正確に）は今回**果たせなかった**が、壊れ方は"確定してはいけないものを確定する"でなく"何も提案しない"＝**安全側**。汚染を1件も生んでいない。
- 修正すべきは**指示書/ツール側**（inconclusive時もCCが視覚分類を必ず埋める）であって、CCの視覚判断ではない。前回400msで同種信号を正しく分類済み。

### この日の成果物
- **ground truth**: BLE ch37 adv 14件 method=human（ch37 累計17件）。
- `bench/review_suggest_ch37_b/`（confirm_sheet.md 等＝**空振り提案**。反面教師として保持）。
- コード変更なし（既存の review_suggest / review.py --pattern を使用）。

---

## 未解決・持ち越し
1. ~~**capture IQ長のCLIノブ**~~ → **【解決 2026-07-08 CC】** 既存フラグ無しを確認し `--capture-ms MS` を追加（`config.dwell_samples_for_ms` + main.py 新フラグ。dwell.py/scheduler.py/spec.py/sigmf_io.py 不変、凍結 diff 空、185 passed）。sim で保存IQ長=capture-ms を実証。長尺収集のメモリ注意は Entry② の CC実装ノート参照。
2. **長尺＋hardケースでの人間vsCC精確性ベンチマーク**（未決）。Entry③でCCは"提案はするが確定しない"線を守り、提案は人間確定と3者一致したが、**いずれも自明ケース**（明快なBLE vs 明快なWiFi）で判別力は測っていない。器＝`review_suggest`（提案と人間確定を分離記録）が整ったので、duty 0.5–0.9帯の微妙なバースト・WiFi+BLE重畳など「両者が食い違い得るケース」を混ぜれば初めて優劣が測れる。
3. **ch39(2480)収集**: 40MHz高調波(2480)のド真ん中 → `--dwell-offset-hz 4e6` の初実戦投入候補。
4. **dutyprobe の位置づけ**: あくまで測定（時間占有＝burst/continuous）。**WiFiとBLEは両方低dutyで分離不能**（Entry③実証）＝信号種の同定はしない。BLE/非BLEラベルにも教師にも使わない。
5. **review_suggest の inconclusive 対応**（改修候補・Entry④で判明）: 13ms収集ではdutyがinconclusiveになり、CCが視覚分類を埋めず全skipで出た＝提案空振り。**13msが主レシピなので review_suggest は duty 非依存でも機能する必要**がある。改修＝「dutyが結論不能でも、CCの視覚分類＋メタ(BW/spur_suspect)＋スプリアスガードで recommend を出す」。指示書に「inconclusive時もCCが視覚分類を必ず埋める」を明示。dutyは補助列、主判定は視覚、に寄せる。
6. **外部IQデータセットによる事前学習**（将来検討・厳格な条件付き）。synthetic 100%学習の弱点（実RF未経験）を実測データで埋める方向は筋が良いが、以下を厳守しないと今の思想が崩れる。
   - **使い方は「事前学習(特徴初期化)専用」に限定**。外部大規模IQ（RadioML/DeepSig/公開SigMFコーパス等）で下層の特徴抽出器を初期化し、**上層の分類ヘッドは君のHackRF実測human確定 ground truth だけで学習**（fine-tuning）。「これは何か」を決める層は必ず自分のground truthが握る。
   - **外部ラベルを分類正解に混ぜない＝Pattern A厳守**。外部データのラベルは「自分が視覚確認していない・自分のハードで検証していない」ので、ground truth（confidence=1.0/method=human）に昇格させない。混ぜると積み上げた純度が薄まる。
   - **落とし穴1が本質（外部では直らない）**：今のCNNの主要な誤りは**HackRF固有スプリアスの誤認**（DC残留線・40MHzクロック高調波2400/2440/2480・16MHzコム・2420固定線）。これは君のHackRFにしか出ないので**外部データには一切含まれず、外部学習では直らない**。直すのは実測ground truth（スプリアスを"未識別/スプリアス"として教える）。むしろ他機の癖を覚えて新しい誤りを生むリスクあり。
   - **落とし穴2：ラベル体系の不一致**。外部＝変調ラベル(BPSK/QPSK/GFSK/OFDM…)、君＝用途×変調(BLE adv/WiFi 20/40MHz/Zigbee)。対応は付くが「同じ2.4GHzで隣接する実信号種の区別」は変調ラベルからは学べない。
   - **優先順位は低い**：まず各クラス数十件のhuman確定（君の再学習基準）を貯めるのが先。事前学習+fine-tuningは大きな新規サブシステム（DL・前処理・ラベルマッピング・2段学習）なので、ground truth蓄積という現ボトルネックを解く前に着手すると"willpower無しで回る"から外れる。

---

## 確定した設計判断・教訓（このセッション）
- **窓長は用途で分ける**（Entry②）：BLE収集は既定13ms（時間変動が保たれnarrow-steady-spurがBLEを通す）／duty解析は400ms・ゲート無し。同じ窓に両立させない。ゲート(narrow-steady-spur)は不変。
- **収集前に部屋の全2.4GHz機器を管理**（Entry②）：BluetoothイヤホンのFHSSが2402を占有し交絡（教訓7再演）。イヤホンも管理リストに入れる。
- **CCは"提案はするが確定しない"を守れる**（Entry③）：枠が明確なら思想の要に隣接する作業でも逸脱しない。ただし視覚精確性の人間超えは未実証。
- **確立した収集レシピ**：collect(13ms) → view_captures → review_suggest(CC提案・サンドボックス) → review.py(人間確定・PNG目視) の一周でground truthを積める。
- **review_suggest は視覚主・duty補助であるべき**（Entry④）：13ms主レシピではdutyがinconclusiveになるので、提案の主判定はCCの視覚＋メタ＋スプリアスガード、dutyは補助列。inconclusive時に全skipで倒れたのは指示書の穴で、CCの視覚能力の問題ではない。
