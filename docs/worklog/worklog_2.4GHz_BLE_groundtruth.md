# sigscan 作業ログ — 2.4GHz ISM / BLE ground truth

> このファイルは**人間が読んで経緯を追える**ための実験ノート。`bench/` のような使い捨てサンドボックスや、機械可読のCSV・コードとは別物。「何を・なぜ・どう判断したか」を残す。
>
> **置き場所**: `docs/worklog/`（git 追跡する。使い捨ての `bench/` とは別）。
> **更新規律**: 実験セッションごとに1エントリ追記。各エントリは 狙い / やったこと / 分かったこと / 保留 / 成果物 / 次アクション。

---

## このログのスコープ

BLE advertising の ground truth を 2.4GHz ISM で積む取り組みの記録。
- **主目標**: BLE adv を各チャネルで human確定して ground truth を貯める。ch38(2426) は達成済み（3件、method=human）。ch37(2402)・ch39(2480) が次。
- **副次目標**: dc-spike説の白黒 —「DC上に乗った間欠BLEが dc-spike ゲートで落ちるか」。過去に「必ず落ちる」と結論しオフセット実装の動機にしたが、要再検証に格下げ済み。

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

### 結果（実行後にKaliが記入）
- （観測をここに追記: 実際の det 周波数 / duty / ゲート挙動 / BLE ch37 と確認できたか / human確定に進めたか）

---

## 未解決・持ち越し
1. ~~**capture IQ長のCLIノブ**~~ → **【解決 2026-07-08 CC】** 既存フラグ無しを確認し `--capture-ms MS` を追加（`config.dwell_samples_for_ms` + main.py 新フラグ。dwell.py/scheduler.py/spec.py/sigmf_io.py 不変、凍結 diff 空、185 passed）。sim で保存IQ長=capture-ms を実証。長尺収集のメモリ注意は Entry② の CC実装ノート参照。
2. **長尺＋hardケースでの人間vsCC精確性ベンチマーク**。今回は判別力ゼロ。duty 0.5–0.9帯の微妙なバースト・WiFi+BLE重畳など「両者が食い違い得るケース」を混ぜて初めて判定者の優劣が測れる。
3. **ch39(2480)収集**: 40MHz高調波(2480)のド真ん中 → `--dwell-offset-hz 4e6` の初実戦投入候補。
4. **dutyprobe の位置づけ**: あくまで測定（時間占有）。BLE/非BLEラベルにも教師にも使わない。
