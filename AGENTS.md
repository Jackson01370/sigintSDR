# AGENTS — マルチエージェント運用

CONTRACT.md §4 の分担を Claude Code のサブエージェントに落としたもの。**契約(CONTRACT.md)が
凍結されている前提**で、継ぎ目に沿って並列化する。逆順（契約を固める前に並列化）は不可。

## 定義済みエージェント（`.claude/agents/`）

| エージェント | 継ぎ目 | tools | 色 |
|--------------|--------|-------|----|
| `capture-engine` | 取得・交換・蓄積 | Read/Write/Edit/Bash/Glob/Grep | green |
| `eval-harness` | 表現（外部モデル評価） | +WebFetch/WebSearch | cyan |
| `cnn-training` | 分類(CNN段) | Read/Write/Edit/Bash/Glob/Grep | purple |
| `llm-vision` | 分類(LLM段) | +WebFetch | orange |
| `test-docs` | 全体（テスト/CI） | Read/Write/Edit/Bash/Glob/Grep | blue |

各定義は `isolation: worktree` 付き。サブエージェントとして委譲すると、自動で git
worktree の隔離コピー上で動く（変更が無ければ自動削除、変更があれば残してレビュー可）。

> 注: サブエージェントを使うには git リポジトリ化が必要。最初に
> `git init && git add -A && git commit -m "freeze data contract"` を実行。
> ディスク上で `.claude/agents/*.md` を編集したらセッション再起動（または `/agents`）。

## 2つの動かし方

**(1) 1セッション内で委譲**（手軽・契約が効くか試すのに最適）
```
@agent-capture-engine 収集ループに重複排除と dataset.py を追加して
@agent-test-docs 6つの継ぎ目の回帰テストを pytest で書いて
```
独立タスクは並列委譲できる（「capture-engine と test-docs を並列で」）。ただし
結果が親会話に戻るので、多数同時はコンテキストを食う。

**(2) 1エージェント=1セッションを worktree で並列**（本格並列・推奨）
```
scripts/worktrees.sh setup
# 別ターミナルで:
cd ../sigscan-capture-engine && claude --agent capture-engine
cd ../sigscan-test-docs      && claude --agent test-docs
```

## 依存順（並列でも守る）

```
test-docs ─────────────────────────────────┐ いつでも開始可（契約をテストでロック）
capture-engine ──┬──► (実機SigMF蓄積) ──────┤
                 │                          ▼
eval-harness ────┘ 配線は並行可／**測定は実測キャプチャ待ち**
cnn-training ───────────────► capture-engine のデータセット完成後
llm-vision ─────────────────► ほぼ独立（classify IF と画像化のみ）→ 並行可
```

- **test-docs を最初に**回して契約をテストでロックすると、以降の破壊を即検知できる。
- **capture-engine が最上流**（実データを産む）。
- **eval-harness の「測定」は実機キャプチャが出てから**。それまでは配線（モデルロード／
  `spec.render` アダプタ）のみ並行で進める。Sim 相手の結果は「合成 vs 合成」と明示。
- **cnn-training** は capture-engine の SigMF データセット完成後。
- **llm-vision** は `classify.classify` の IF と画像化だけに依存 → 並行可。

## 統合の規律

- 各エージェントは自分のブランチ/worktree で作業。**依存先（capture-engine）を default
  ブランチへマージしてから**、依存元（cnn-training 等）をその上に分岐/再開する
  （`isolation: worktree` は default ブランチから分岐するため）。
- **契約ファイル（`spec.py`・`sigmf_io.py`）は凍結**。変更は人間の明示承認が必要。
  test-docs のスナップショットテストが想定外変更を弾く。
- **ボトルネックはあなたのレビュー**。生成は数倍速でも統合は人手。レビューできる以上に
  エージェントを広げない。最初は「Explore で並列調査 → 本体で実装」の型が安全。
- サブエージェントは入れ子不可。多段が要るなら chain（順次委譲）か Skills を使う。

## 任意の調整
- 重いエージェント（cnn-training / eval-harness）は `model: opus`、安価で良いものは
  `haiku` に変更可（各 `.claude/agents/*.md` の `model:`）。
- SDRスキルを Claude Code に登録済みなら、関連エージェントに `skills: [sdr-signal-id]`
  を足すとドメイン知識を初期注入できる。
