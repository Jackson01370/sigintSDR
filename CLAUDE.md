# CLAUDE.md — sigscan プロジェクト常設ルール

> このファイルは Claude Code が**毎セッション自動で読む**。ここに書かれたことは常に有効。
> 全体設計は `docs/ARCHITECTURE.md`、作業履歴は `docs/worklog/` を参照。

---

## 0. これは何か

HackRF One + discone + LNA で 1〜6GHz をスキャンし、電波を自動識別するシステム。
パイプライン: **ルール分類器 → CNN監査 → LLM vision**（3段目はスケルトンのみ）。データは SigMF 保存。

---

## 1. 絶対の原則（破ってはいけない線）

1. **AIが提案・人間が承認**。あなた（CC）は**提案と道具**を作る。**最終確定は必ず人間**。
2. **Pattern A 禁止**。ルール／CNN／LLM／CC の出力を **CNN の学習ラベルにしない**。ラベル汚染は本プロジェクト最大の禁忌。
3. **ground truth は human確定のみ**（`method=human`, `confidence=1.0`）。人間が PNG を視覚確認して `review.py` で確定したものだけが教師データになりうる。
4. **量より質**。確信が持てないものは `unknown`／`skip` のままにする。無理にラベルを付けない。
5. **意志力なしで回る設計**。穴は運用注意ではなく**コードと構造で塞ぐ**。

### CC が絶対にやらないこと
- `method=human` の付与、`apply_label` の実行、SigMF アノテーションの書き換え
- `captures/` への書込・移動・削除・rename（**読み取りのみ**）
- 提案を確定に格上げすること

---

## 2. 凍結契約

| 対象 | 扱い |
|---|---|
| `spec.py` / `sigmf_io.py` | **真の凍結**。変更禁止（import して呼ぶのは可） |
| `main.py` | **追加のみ可**（新フラグの追加。既存シグネチャ・挙動は不変。diff を全掲すること） |
| 6継ぎ目（seams） | **シグネチャ不変**（`test_seams.py` がロック） |
| その他 | 通常の変更可。ただし既存テストは壊さない |

作業前に必ず `CONTRACT.md` / `AGENTS.md` / `FILES.md` を確認すること（記憶や推測で判断しない）。

---

## 3. 実行環境

```powershell
$py = "C:\Users\puppy\radioconda\envs\sigscan\python.exe"
```

- **毎セッション `$py` を設定し、`& $py ...` で呼ぶ**（Microsoft Store の dummy python が PATH を乗っ取るため、フルパス必須）
- **PowerShell は `&&` 連結不可。コマンドは1行ずつ**
- conda env `sigscan`（Python 3.11 / numpy 1.26.4 / torch 2.5.1+cu121）。旧 venv・radioconda は使わない
- `--db nul` は使わない（必要なら `--db test.db`）

---

## 4. 収集→確定フロー（**完遂義務あり**）

```
collect(13ms) → view_captures → review_suggest(--auto-classify) → [CCが視覚分類] → review.py(--suggest ○×UI・人間)
```

### 4-1. 最重要: `--auto-classify` は「タスクリストを作る」だけでは**終わっていない**

`cnntrain.review_suggest --auto-classify` を実行したら、**必ず続けて最後まで完遂すること**:

1. 生成された `bench/<out>/classify_tasklist.md` を読む
2. **記載された全 PNG を `view` で開いて視覚分類**し、`cc_class` と `cc_rationale` を決める
3. `cc_verdicts.csv` を書く
4. `--verdicts` で**併合して再実行**し、`confirm_sheet.md` を完成させる

**タスクリストを生成した時点で止まらないこと**（`needs-review` のまま人間に投げない）。
ツール自身は画像を見られない（Python スクリプトなので当然）。**視覚分類はあなたにしかできない工程**であり、それを飛ばすと道具が機能しない。

### 4-2. 窓とファイル名の注意
- **SigMF のファイル名は「窓の下端」で決まる**（検出周波数ではない）。例: 窓 2.408–2.416GHz → `2408MHz_<timestamp>_<n>`
- `--pattern` を組む前に、**必ず実ファイル名を確認**する:
  ```powershell
  Get-ChildItem captures\*.sigmf-meta | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name
  ```

### 4-3. 窓長の使い分け（設計判断・変更しない）
- **収集は 13ms（既定）**。観測間の時間変動が保たれ、`narrow-steady-spur` が間欠BLEを正しく通す
- **400ms（`--capture-ms`）は duty 解析専用**（ゲート無効時）。収集に使うと persist が飽和し、ゲートが BLE を「定常」と誤認して落とす

---

## 5. 視覚分類の基準（PNG の読み方）

**判断軸は常に「検出帯（赤帯）の主役が何か」**。帯域外のブロブや、帯域を通過するだけの事象に引きずられない。

| cc_class | 見た目 |
|---|---|
| `ble-adv` | 検出帯**内**に収まる離散ブロブ（BW 1.0–1.6MHz）。間欠。`spur_suspect=False` |
| `wifi` | 検出帯を**上下端まで貫く太い縦帯**が繰り返す（20MHz級）。**BW測定値が3–5MHzでも、窓幅による切り取りアーティファクト**。ルールが "Zigbee" と誤ラベルすることがある |
| `spurious` | **2400.0 / 2440.0 / 2480.0** 付近の連続した水平線（時間軸を貫く）。または `spur_suspect=True` |
| `hopping` | 周波数方向に散らばる離散バースト（BT Classic の FHSS。例: Bluetooth イヤホン） |
| `unclear` | 判断できない。**検出帯に主役が居ない場合もここ**（無理に付けない） |

### スプリアス誤確定ガード（コードで強制済み・弱体化禁止）
`det ≈ 2400.0MHz` または `spur_suspect=True` → `cc_class` に関わらず **`skip` 強制**。

### 摩擦（○×UIの安全弁・省略禁止）
`unclear` / `spurious_warn=True` / `needs-review`（未記入）/ 提案なし → **`y` を提示せず、ラベル選択を強制**。
「y 連打で素通り」は AI 出力がそのまま ground truth になる（＝Pattern A 化）ので、構造で防いでいる。

---

## 6. 2.4GHz ISM 専門家 CNN（現在の主目標）

### 設計思想
1〜6GHz を1つの CNN で見るのは困難なので、**バンド／信号種別ごとに専門家 CNN を分ける**方針。
ただし**専門家の「単位（どの周波数範囲を1つの専門家が担うか）」は未確定**で、`BANDPLAN_PROPOSAL.md §6` に「2.4GHz 専門家を完成させてから、実地の知見で確定（"検証してから決める"）」と明記されている。**まず 2.4GHz 専門家を1つ作る**のが当面の目標。

### バンド別ルーティング：**実装済み**（2026-07-15・案Y）
> 経緯（学びとして残す）: このプロジェクトでは長く「配管は実装済み（`config.py` の `BandPlan.cnn_model`、`classify.py` のルーティング分岐）」と誤記されていた。CC が grep で確認し（`cnn_model`/`BandPlan` は全 `*.py` に0件）、**配管は存在しなかった**ことが判明（Claude の思い込み）。その後、専門家 CNN v2 完成を受けて**新規実装した**。以下が現在の実装:

- **方式は案Y（バンド→checkpoint 対応表）**。`config.Band` dataclass には**モデル枠を足さない**（31バンド全部を変えない）。代わりに `config.py` に対応表 `BAND_CNN_ROUTES = {"ISM 2.4G (WiFi/BT)": "runs/ism24_v2"}` を持つ。ここに無いバンドは**汎用 `runs/m2_5` にフォールバック**（後方互換）。
- **scheduler** が起動時に汎用＋専門家を両方ロード（キャッシュ）し、`_select_cnn_for(center_hz)` で検出周波数→バンド→checkpoint を選ぶ。バンド判定は `classify._match_band` と同一なので、監査に使う checkpoint とルール監査のバンドが構造的に一致。
- **監査マッピング**（`cnntrain/expected.py` / `audit.py`）: 2.4GHz では専門家3クラス ↔ ルールラベルで A-consistent / C-conflict を判定。`ble-adv↔BLE` / `wifi-24↔WiFi` が A-consistent、食い違いは C-conflict→Unknown 化、`Zigbee` 等は unmapped（所見のみ）。**他バンドの汎用監査（方式軸5クラス）は `cnn_classes=None` 既定で従来どおり不変**。
- **専門家は監査（audit）専用**: A-consistent でもルールラベルを専門家クラスに**書き換えない**、C-conflict は Unknown 化のみ、**`method=human` を付与しない**。SigMF の `core:label` を専門家が確定しない。**確定は従来どおり人間の○×**（`review.py`）。＝Pattern A を踏まない。「最後に判断するのは人間」を実運用でも堅持。

#### 実装で判明した重要な発見（spurious 盲点の捕捉）
2.4GHz の human確定データで汎用監査 vs 専門家監査を比較（read-only 観察）したところ、**汎用 CNN は HackRF 内部スプリアス（2439.6MHz・40MHz高調波）を `narrowband-burst` と判定し「A-consistent（BLEとして素通り）」にしていた**＝監査の盲点。**専門家 CNN はこれを `spurious → C-conflict→Unknown` で正しく捕捉し、人間に回す**。数週間かけた spurious 32件の収集が、実運用の監査でここに効いた。
- 限界: WiFi を rule が `Zigbee` と誤ラベルするケースは、専門家が `wifi-24` と分かっていても組み合わせが unmapped 止まり（所見のみ・修正はしない）。将来課題（rule の bw 誤判定対策）。

#### 将来: 5GHz 等の専門家追加は `BAND_CNN_ROUTES` に1行足すだけ
専門家を `review_suggest` の CC 分類下地に結合するのは別の拡張（未実装・将来課題）。

### クラス体系（**案Z・Kali 承認済み 2026-07-15**）
2.4GHz ISM 専門家は **用途ベース**で始める。**unknown は教師0件のため、当面3クラスで開始**し、unknown は蓄積後に4クラス化:

| クラス | v1学習時 | **v2学習時（最新・`method=human` 実測）** |
|---|---|---|
| `ble-adv` | 36件 | **60件**（ch37/38/39。ch39=2480 が高収率90%、ch37/38 は WiFi ch1 と近接し低収率） |
| `wifi-24` | 23件 | **93件** |
| `spurious` | 32件 | **39件**（2440MHz 40MHz高調波ほか） |
| `unknown` | 0件 | 0件（教師なし＝当面クラスから除外、後で追加） |

**専門家 CNN 学習の到達点**:
- v1（`runs/ism24_v1`・91件）: val 94.4% だが過学習・最終epoch保存・val脆弱。
- **v2（`runs/ism24_v2`・192件・現行）**: best-val 保存 + early-stopping + k-fold(5)。**k-fold 平均 97.4%±2.8%**。前回弱点の ble-adv↔wifi-24 境界が改善（wifi-24 は 93件 recall 100%・bleと1件も混同せず）。過学習は best-val 保存（epoch8・train前）で回避。数値は「単一val」でなく「k-fold平均±分散」で読む。到達点は「実RF・human確定 held-out で3クラス分離」段階で、未知環境への汎化は未検証。**入力・学習ループ・checkpoint 形式は `cnntrain/train_expert.py`**。合成データ不使用（Pattern B 純粋）。

**BLE 収集の窓選び（重要な運用知見）**: BLE adv は **ch39(2480MHz) を使う**。ch37(2402)/ch38(2426) は WiFi ch1(2402-2422) の帯域に近接し、強い WiFi に検出が奪われて収率が 10%/0% に落ちる。ch39 は ch1 から 58MHz 離れており収率 ~90%。ただし環境次第（chirp源・スプリアス）で ch39 でも収率が落ちることがある。

- **案Z の意味**: `BANDPLAN_PROPOSAL.md §1` の用途ラベル体系（`WiFi`/`Bluetooth`/`Zigbee`/`WPT`/`Amateur`/`Unknown` × 方式2軸）を土台にしつつ、**実務で発見した `spurious`(HackRF内部) を用途リストに追加**し、当面は**用途1軸**で始める（方式軸・2軸化は将来課題）。`ble-adv` は proposal の `Bluetooth` の下位クラスと位置づける。
- **用途ベースにしたのは、human確定ラベルをそのまま教師にでき、Pattern A を完全回避できるため**。
- Zigbee は**部屋に機器が無い**ため見送り（将来の拡張枠）。
- **汎用CNN（既存 `runs/m2_5`）は方式軸5クラス**（`cw-tone` / `narrowband-burst` / `noise-only` / `pulse-radar` / `wideband-ofdm`、`cnntrain/classes.py`）で、**専門家の用途クラスとは別物**。専門家は**別 checkpoint の新規学習**（既存の再学習・上書きではない）。
  - ※ 過去このファイル/一部文書に「汎用CNN は7クラス（narrowband-fm/chirp/burst-fsk/noise-floor…）」とあったが、これも**誤り**。実体は上記5クラス（CC が `classes.py` で確認）。

---

## 7. HackRF 固有の癖（既知スプリアス・環境）

1. **40MHz クロック高調波**: 2400 / 2440 / 2480MHz。**絶対周波数固定**（tuner を動かしても同じ周波数に出る）
2. **16MHz コム**: 2408 + 16n MHz（DC除去 OFF 時に露出）
3. **2420.0MHz 固定線**: origin 未特定
4. **Bluetooth イヤホン**（BT Classic/A2DP）が FHSS で 2.4GHz を占有し、実験を汚染する。**収集前に電源を落とす**

これらは `narrow-steady-spur` ゲート（狭帯域 かつ 定常）が正しく弾く。**このゲートは BLE 収集失敗を理由に触らない**。

---

## 8. 作業の型（指示書がある場合もない場合も）

1. **事前確認**: `git status` clean か、`pytest` のベースラインを記録（現在 **214 passed, 3 skipped**）
2. **権威定義を読む**（記憶で判断しない）
3. **最小実装**。スコープ膨張禁止。1サイクルで停止
4. **テストで固定**（特に「穴を塞いだ」なら回帰テストを書く）
5. **検証**: 凍結契約の diff が空・`captures/` 非改変・既存テスト無変更
6. **正直に報告**: 既存テストを更新したなら**1件ずつ理由を明示**。推測は「※未確認」と書く。未実装を隠さない

### 出力先
- サンドボックス出力は `bench/` のみ
- ドキュメントは `docs/`
- **`captures/` は読み取り専用**

---

## 9. 困ったときの参照先

| 知りたいこと | 見る場所 |
|---|---|
| 全体設計・モジュール地図・実装状況 | `docs/ARCHITECTURE.md` |
| 何をしてきたか・なぜそう決めたか | `docs/worklog/` |
| 何を変更してよいか | `CONTRACT.md` / `AGENTS.md` / `FILES.md` |
| 未解決の課題 | `docs/worklog/` の「未解決・持ち越し」節 |
