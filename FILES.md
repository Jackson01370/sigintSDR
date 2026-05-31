# FILES — このプロジェクトの地図

迷ったらこのファイルを見れば、どれが何か分かります。

---

## ① ぜったいに消さないもの（中身が本体）

- **`*.py`**（`main.py`, `classify.py`, `spec.py` など全部）= プログラム本体
- **`llmvision/` フォルダ** = 「物知りに聞く」部品（作成済み）
- **`.claude/` フォルダ** = ロボットの設定（Claude Code が自動で読む）

この3つは触らない・消さない。これが「作った成果」そのものです。

---

## ② フォルダの全体図

```
sigintSDR/                ← プロジェクトの一番上
│
├─ FILES.md               ← この地図
├─ README.md              ← プロジェクトの説明
├─ CONTRACT.md            ← 「約束ごと（凍結ルール）」
├─ AGENTS.md              ← ロボットたちの動かし方
├─ requirements.txt       ← 必要な部品リスト
├─ worktrees.sh           ← 並列作業の補助
├─ .gitignore             ← セーブしないゴミの一覧
│
├─ main.py                ← 入口（実行するファイル）
├─ config.py              ← 設定と「電波の地図」
├─ sdr.py                 ← 電波を受け取る部分（Sim/実機）
├─ dsp.py                 ← 電波の計算
├─ classify.py            ← 名前を当てる3段階の本体
├─ spec.py                ← 【凍結】画像の作り方の約束
├─ sigmf_io.py            ← 【凍結】データの保存形式
├─ scheduler.py           ← スキャンの司令塔＋収集
├─ store.py               ← 結果の記録（データベース）
│
├─ llmvision/             ← 「物知りに聞く」部品（作成済み・触らない）
│    ├─ __init__.py / prompt.py / client.py
│    ├─ render.py / core.py / selftest.py
│    └─ README.md
│
├─ .claude/
│    └─ agents/           ← ★ロボットの「設定」（自動で読まれる）
│         ├─ capture-engine.md
│         ├─ eval-harness.md
│         ├─ cnn-training.md
│         ├─ llm-vision.md
│         └─ test-docs.md
│
└─ docs/
     └─ briefs/
          └─ work-orders.md   ← ★あなたが「貼る指示文」（全部まとめて1つ）
```

---

## ③ 「同じ名前のファイルが2つある」問題について

以前は、各ロボットについて**同じ名前のファイルが2か所**にありました。これが混乱の元でした。
今回それをやめて、見分けがつくようにしました。

| 種類 | 場所 | 先頭の文字 | あなたの操作 |
|------|------|-----------|-------------|
| ロボットの**設定** | `.claude/agents/test-docs.md` など | `---` で始まる | **触らない**（Claude Codeが自動で読む） |
| あなたが**貼る指示文** | `docs/briefs/work-orders.md`（1つにまとめた） | `# 1. test-docs …` | この中の四角を**コピーして貼る** |

- 設定ファイル（`.claude/agents/` の5つ）は、名前そのままで正しく置けています。**さわらないでください。**
- 指示文は、ばらばらだった6個をやめて **`work-orders.md` 1つ**にまとめました。もう名前がかぶりません。

---

## ④ いまどこまで進んだ？

| 段階 | ロボット | 状態 |
|------|----------|------|
| 「物知りに聞く」部品 | llm-vision | ✅ 完了 |
| 土台のチェック係 | test-docs | 👈 **次にやる** |
| 電波あつめ | capture-engine | これから |
| 外部データで答え合わせ | eval-harness | これから |
| 写真で覚える | cnn-training | これから |
