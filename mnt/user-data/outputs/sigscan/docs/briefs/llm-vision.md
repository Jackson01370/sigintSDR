# llm-vision — 初回ワークオーダー（M1: llm_classify 実装）

低信頼/未知信号のスペクトログラムを視覚LLMに送って識別する `classify.llm_classify()`。
`classify` IF と画像化のみに依存するのでほぼ独立、capture-engine と並行可。

## ▶ 初回プロンプト（このままセッションに貼る）

```
あなたは llm-vision エージェント。まず CONTRACT.md と classify.py を読んで。
classify.classify() のシグネチャは変えない。llm_classify() は ClassResult(method="llm") を
返し、ルール信頼度<0.5 かつ CNN未解決のときだけ呼ばれる前提。API失敗時は None を返して
劣化動作させる。

M1として llmvision/ 配下に実装し、classify.llm_classify() を接続して:

1. llmvision/render.py
   - スペクトログラムPNGを作る（dsp.save_spectrogram_png / spec を利用）。送るのは
     画像と最小限のメタのみ。生IQや個人情報は外部送信しない。
2. llmvision/prompt.py
   - プロンプト生成: 中心周波数・該当バンド・バンドplan由来の候補信号・帯域幅/SNRを文脈に。
     SigIDWiki/Artemis を「参照カタログ」として要約的に織り込む（学習ではなく参照）。
   - 構造化出力（JSON: label, confidence, reasoning）を要求。
3. llmvision/client.py
   - Gemini もしくは Claude の視覚APIクライアント。APIキー/エンドポイントは環境変数
     （例 LLMVISION_API_KEY）。レート制限・タイムアウト・エラーを丁寧に処理。
     最新のAPI仕様は WebFetch で確認してよい。
4. classify.llm_classify(png_path)
   - 上記を束ね、JSONを ClassResult(method="llm") にパース。失敗時 None。

検証して報告:
- サンプルPNGに対し、(モック or 実キーで) llm_classify が ClassResult を返す。
- キー未設定/失敗時に None を返し、classify.classify がルール結果へ劣化する。
```

## 受け入れ基準
- `llm_classify()` がサンプルPNGで `ClassResult(method="llm")` を返す。
- キー無し/エラー時に `None`、`classify` 全体が劣化動作。
- 送信は画像＋最小メタのみ（生IQ非送信）。

## 依存
ほぼ独立（`classify` IF と画像化のみ）。並行で進めてよい。
