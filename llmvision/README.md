# llmvision — LLM Vision 段（低信頼/未知信号の最終手段）

`classify.classify()` の 3 段目。ルール段の信頼度が 0.5 未満かつ CNN が決まらない
信号について、スペクトログラム画像と中心周波数・バンド文脈・ルール所見・
参照カタログ (SigIDWiki / Artemis 由来の一般知識) を視覚 LLM に渡して
信号サービスを推定する。

- **公開 API**: `from llmvision import llm_classify`
- **graceful degradation**: 未設定・キー無し・タイムアウト・JSON 失敗 → `None`
- **送信内容**: PNG 画像 + 最小限のメタ（周波数・帯域・SNR・バンド情報）。
  **生 IQ は送らない**。
- **依存**: 標準ライブラリ (`urllib`) のみ。Pillow / matplotlib があれば
  画像縮小・IQ→PNG 生成に使う（無くても路で動く）。

## 対応プロバイダ

| プロバイダ | 既定モデル                       | 必要な環境変数              |
|------------|----------------------------------|-----------------------------|
| Gemini     | `gemini-2.5-flash`               | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| Anthropic  | `claude-haiku-4-5-20251001`      | `ANTHROPIC_API_KEY`         |
| OpenAI     | `gpt-4o-mini`                    | `OPENAI_API_KEY`            |

`SIGSCAN_LLM_PROVIDER=gemini|anthropic|openai` でプロバイダ強制、
`SIGSCAN_LLM_MODEL=...` でモデル切替。未設定なら API キーから自動推測。

## 使い方

### 自動連携（推奨）

`classify.classify()` が低信頼時に自動で `llm_classify()` を呼ぶ。
`main.py --save-spectrograms` で PNG が保存されている場合は
スケジューラが PNG パスを渡せる。プロバイダ未設定なら自動で skip される。

### 直接呼び出し

```python
from llmvision import llm_classify
from config import BAND_PLAN

result = llm_classify(
    "captures/2437_001.png",
    measurement={"center_hz": 2.437e9, "bw_hz": 20e6, "snr_db": 22.0,
                 "occupied_frac": 0.6},
    bands=BAND_PLAN,
    rule_result=None,                # 任意
)
if result:
    print(result.label, result.confidence, result.method)  # method="llm"
```

PNG が手元にない場合は IQ から生成できる（要 matplotlib）:

```python
llm_classify(None, measurement=meas, bands=BAND_PLAN,
             iq=iq_complex64, rate=20e6)
```

## 動作確認（ネットワーク不要）

```bash
python -m llmvision.selftest
```

期待出力:
```
llmvision selftest (offline, no network)
  OK  graceful degradation when provider/keys absent
  OK  parse_response handles plain / fenced / noisy / invalid
  OK  build_user_text contains band / rule / catalog / output spec
  OK  classify() rule path intact + llm_classify legacy signature OK
  OK  payload → ClassResult mapping (method=llm, clamping, noise/unknown)
ALL PASS
```

## 環境変数一覧

| 変数 | 既定 | 説明 |
|------|------|------|
| `SIGSCAN_LLM_PROVIDER` | (自動推測) | `gemini` / `anthropic` / `openai` |
| `SIGSCAN_LLM_MODEL`    | (プロバイダ既定) | モデル名 |
| `SIGSCAN_LLM_TIMEOUT`  | 30 | HTTP タイムアウト秒 |
| `SIGSCAN_LLM_MAX_TOKENS` | 512 | 応答上限 |
| `SIGSCAN_LLM_DEBUG`    | (未設定) | パース失敗時に生応答の先頭を stderr に出力 |
| `SIGSCAN_GEMINI_ENDPOINT` | (公式) | エンドポイント上書き（プロキシ等） |
| `SIGSCAN_ANTHROPIC_ENDPOINT` | (公式) | 同上 |
| `SIGSCAN_OPENAI_ENDPOINT` | (公式) | 同上 |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini |
| `ANTHROPIC_API_KEY` | — | Anthropic |
| `OPENAI_API_KEY`   | — | OpenAI |

## 注意 / プライバシ

- 送信するのは **スペクトログラム PNG** と **計測メタ**（周波数・帯域幅・SNR・
  バンド名・ルール段所見）のみ。**生 IQ・位置情報・ユーザ識別子は送らない**。
- プロバイダ側のログ・データ保持ポリシーは各社の規約に従う。機微情報を含む
  可能性のあるキャプチャは外部送信前にレビューすること。
- 視覚 LLM は「サービス識別」の最終手段。バンドプラン由来の事前情報と
  ルール段所見は**必ず文脈に含める**（`build_request_context` が自動で行う）。

## 失敗モード

| 状況 | 動作 |
|------|------|
| プロバイダ未設定 / APIキー無し | `None` を返す。上位はルール段結果を採用 |
| HTTP 429 / 5xx               | 1 回リトライ後 `None`                 |
| ネットワークエラー / タイムアウト | `None`                                  |
| 応答が JSON でない             | `None`（`SIGSCAN_LLM_DEBUG` で先頭を表示） |
| `label`/`confidence` 欠落      | `None`                                  |

いずれも例外を投げず、上位の分類フローは継続する。
