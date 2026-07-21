# sigscan 作業ログ — バンド別 CNN ルーティング（案Y・専門家は監査のみ）

> 実験ノート（人間が経緯を追う）。常設ルールは `CLAUDE.md`。
> 前提: 2.4GHz 専門家 `runs/ism24_v2`（用途3クラス・k-fold 97.4%±2.8%、[[worklog_ISM24_expert_CNN_retrain]]）。

---

## エントリ 2026-07-21: バンド別 CNN ルーティング実装（案Y）

### 狙い
単一 checkpoint（汎用 `runs/m2_5`・方式軸5クラス）を全バンドに一律適用していた監査を、
**検出周波数が 2.4GHz ISM 帯なら専門家 `runs/ism24_v2`（用途3クラス）、他バンドは従来どおり
汎用**、に切り替える。専門家は**監査（audit）専用**でラベル確定はしない（Pattern A を踏まない）。

### やったこと（最小実装・追加中心）
- `config.py`: `BAND_CNN_ROUTES = {"ISM 2.4G (WiFi/BT)": "runs/ism24_v2"}` を新設。
  **`config.Band` dataclass は変更しない**（31バンドにモデル枠を足さない＝案Y）。空表なら全汎用。
- `cnntrain/expected.py`: 専門家3クラス↔ルールラベルの期待対応表 `EXPECTED_EXPERT_ISM24`
  （ble-adv↔BLE / wifi-24↔WiFi / spurious↔スプリアス）と、語彙で表を選ぶ `tables_for_cnn_classes`
  を追加。専門家は用途↔用途の直対応のため (B) 文脈規則は空。
- `cnntrain/audit.py`: `audit(..., cnn_classes=None)` を追加（keyword-only・**既定 None は従来=汎用表**
  ＝後方互換）。語彙で期待対応表・文脈規則を切り替える。
- `classify.py`: `_run_cnn_audit` が `ctx.checkpoint.classes` を audit に渡すだけ（1点）。
- `scheduler.py`: 起動時に汎用＋ルート表の専門家を一括ロード（キャッシュ・毎検出でロードし直さない）。
  `_select_cnn_for(center_hz)` が `classify._match_band` と同一判定でバンド→checkpoint を選ぶ。
  来歴表示名を `_ckpt_display_name`（`<run>/checkpoint.pt`→run名）にし、汎用/専門家を区別
  （sigscan:cnn_checkpoint に "m2_5" / "ism24_v2" が載る）。
- テスト: 新規 `tests/test_band_routing.py`（11件）。既存 e2e `test_dwell_e2e_records_cnn_provenance`
  は**合意の上で1件だけ**アサートを付け替え（2.4GHz→専門家3クラス・checkpoint=ism24_v2 を積極固定。
  理由をコメント明示）。全体 **264 passed, 3 skipped**（実装前 253＋11）。

### 監査マッピング（確定した対応）
| 専門家CNN | ルールラベル | 一致時 | 食い違い時 |
|---|---|---|---|
| ble-adv | BLE/Bluetooth (adv?) | A-consistent | C-conflict→Unknown |
| wifi-24 | WiFi (2.4GHz, 20/40MHz) | A-consistent | C-conflict→Unknown |
| spurious | スプリアス(HackRF内部) | A-consistent | C-conflict→Unknown |
- ルール用途が上記3つ以外（Zigbee/独自2.4G 等）は **unmapped**（所見のみ記録・確信度不変＝新たな誤確定を生まない安全側）。

### 分かったこと（read-only 観察・確定なし）
実 human確定 2.4GHz を 12件サンプルし、汎用CNN監査 vs 専門家CNN監査を比較（`scratchpad/observe_routing.py`）:
- **spurious 4件（2439.6MHz・40MHzクロック高調波）**: rule は狭帯域ゆえ **BLE と誤ラベル**。
  汎用CNN=narrowband-burst → 監査 **A-consistent**（＝spurious を BLE として素通り・汎用の盲点）。
  専門家CNN=spurious → 監査 **C-conflict→Unknown**（＝HackRF内部を正しく人間へ回す）。**改善**。
- **BLE 4件**: rule=BLE・汎用=narrowband-burst・専門家=ble-adv とも **A-consistent**（一致・退行なし。専門家の確信度が高い例あり）。
- **WiFi 4件（2400.9MHz）**: rule が帯域幅測定で **Zigbee と誤ラベル**。汎用=wideband-ofdm → **A-consistent**（Zigbee 誤ラベルを追認）。専門家=wifi-24 だが rule=Zigbee は専門家表で **unmapped**（追認せず所見記録のみ）。汎用の false A よりは安全だが**修正はしない**（限界）。
- 集計: 汎用監査 C-conflict=0 / 専門家監査 C-conflict=4 / 「汎用C→専門家A」= 0。
  → 指示書 §142 が例示した「汎用の誤C-conflict を専門家がA に直す」ケースは本サンプルでは出ず、
  代わりに**より重要な逆向きの改善**（汎用が false A-consistent で見逃す spurious を専門家がC-conflictで捕捉）が観察された。

### 保留・限界
- 専門家は用途3クラス外（Zigbee 等）の rule 誤ラベルを**修正できない**（unmapped 止まり）。rule の bw 誤判定（WiFi→Zigbee）はルーティングでは直らない。
- 監査は助言のみ。確定は人間の `review.py` ○×のまま（変更なし）。

### 成果物
- 実装: `config.py` / `cnntrain/expected.py` / `cnntrain/audit.py` / `classify.py` / `scheduler.py`。
- テスト: `tests/test_band_routing.py`（新規11件）、`tests/test_cnntrain_m3.py`（e2e 1件付け替え）。
- 既存モデル非改変（m2_5 / ism24_v1 / ism24_v2 の mtime 不変）。凍結契約 diff 空・`config.Band` 定義不変・`captures/` 非改変。

### 次アクション（実装せず申し送り）
(a) 5GHz 等の専門家追加（`BAND_CNN_ROUTES` に1行）、(b) 専門家を `review_suggest` の CC 分類下地に結合、(c) unknown クラス、(d) chirp 等未識別信号、(e) 監査の確信度閾値のバンド別調整、(f) rule の bw 誤判定（WiFi→Zigbee）対策。
