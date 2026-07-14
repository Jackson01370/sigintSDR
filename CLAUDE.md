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

### 設計
1〜6GHz を1つの CNN で見るのは困難なので、**バンド／信号種別ごとに専門家 CNN を分ける**設計。
配管は実装済み（`config.py` の `BandPlan.cnn_model`、`classify.py` のルーティング分岐）だが、**専門家モデルの実体はまだ無い**（現状は汎用7クラス CNN が1つだけ）。

### クラス体系（**案A・Kali 承認済み**）
2.4GHz ISM 専門家は **用途ベース4クラス**:

| クラス | 現状 |
|---|---|
| `ble-adv` | **35件**（ch37/38/39 の3チャネル） |
| `wifi-24` | **23件** |
| `spurious` | **0件** ← 次に収集 |
| `unknown` | 0件 |

- Zigbee は**部屋に機器が無い**ため見送り（将来の拡張枠として残す）
- **用途ベースにしたのは、human確定ラベルをそのまま教師にでき、Pattern A を完全回避できるため**（変調ベース7クラスへのマッピングは、人間が視覚確認していない変換を挟むので採らない）
- 汎用CNN の 7クラス（narrowband-fm / wideband-ofdm / cw-tone / chirp / burst-fsk / noise-floor / unknown）は**別物**。混同しない

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
