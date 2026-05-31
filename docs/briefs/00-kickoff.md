# 00 — キックオフ指示書（オーケストレーション）

このディレクトリは各エージェントへの**初回ワークオーダー（マイルストーン1）**。
`.claude/agents/*.md` が常設の役割定義、ここが「最初に何を作るか」。各ブリーフの
`▶ 初回プロンプト` ブロックを、そのエージェントのセッションにそのまま貼る。

前提として **CONTRACT.md が凍結**されている。契約ファイル（`spec.py`・`sigmf_io.py`）は
人間の明示承認なしに変更しない。

---

## 0. 準備（1回だけ）

```bash
# Fedora 上、プロジェクト直下で
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib pytest          # 実機を使うなら別途: sudo dnf install python3-soapysdr SoapySDR hackrf
git init && git add -A && git commit -m "freeze data contract v1"   # サブエージェント/worktreeに必須
```

動作確認: `python3 main.py --sim --once` が走り、検出と分類が出ればOK。

---

## 1. 起動順（依存順を守る）

| 順 | エージェント | 何を | 並列 |
|----|--------------|------|------|
| ① | **test-docs** | 契約をテストでロック | 最初に単独 |
| ② | **capture-engine** | 収集の堅牢化＋`dataset.py`（最上流・実データ源） | — |
| ②並行 | **llm-vision** | LLM段（`classify` IFと画像化のみ依存） | 可 |
| ②並行 | **eval-harness** | **配線のみ**（測定は実機キャプチャ待ち） | 可 |
| ③ | **cnn-training** | capture-engine のデータセット完成後 | — |
| ③ | **eval-harness（測定）** | HackRFで実データ収集後にギャップ測定 | — |

**なぜこの順か**: capture-engine が実データを産み、cnn-training と eval-harness の
「測定」がそれに依存する。test-docs を先に回すと以降の契約破壊を即検知できる。

---

## 2. 起動方法（2通り）

**(A) 委譲（1セッション内・手軽）**
```
@agent-test-docs   <00ブリーフを読んで、test-docs.md の初回プロンプトを実行>
```

**(B) worktreeで並列セッション（本格・推奨）**
```bash
scripts/worktrees.sh setup
# 別ターミナルごとに:
cd ../sigscan-test-docs      && claude --agent test-docs       # ①
cd ../sigscan-capture-engine && claude --agent capture-engine  # ②
cd ../sigscan-llm-vision     && claude --agent llm-vision      # ②並行
```
各セッションで対応するブリーフの `▶ 初回プロンプト` を貼る。

---

## 3. 統合の規律

- 1エージェント=1ブランチ。**依存先（capture-engine）を main にマージしてから**、
  依存元（cnn-training）をその上に分岐/再開（`isolation: worktree` は main から分岐）。
- **契約ファイルは凍結**。test-docs のスナップショットテストが想定外変更を弾く。
- マージ前に必ず人間レビュー。**ボトルネックはレビュー**——同時に走らせるのは
  自分がレビューできる数まで。
- 各エージェントは「契約に触れていない」ことをコミットメッセージに明記。

---

## 4. マイルストーン2以降（参考）

M1完了後の次の波: capture-engine=実機での長時間収集＆データセット拡充 /
cnn-training=実データfine-tune＆SNR別精度改善 / eval-harness=実測ギャップ測定と
転移学習の判断 / llm-vision=未知信号の自動起票 / test-docs=回帰の拡充とCI強化。
