#!/usr/bin/env bash
# sigscan: エージェント並列実行用 git worktree ヘルパ。
#
# サブエージェント委譲(.claude/agents/*.md の isolation: worktree)とは別に、
# 各エージェントを「独立した Claude Code セッション」として並列で回したい場合に、
# 1エージェント=1 worktree を用意する。
#
# 使い方:
#   scripts/worktrees.sh setup     # 各エージェント用 worktree とブランチを作成
#   scripts/worktrees.sh list      # 一覧
#   scripts/worktrees.sh remove    # すべて削除（マージ済み前提）
#
# setup 後の並列起動例（別ターミナルで）:
#   cd ../sigscan-capture-engine && claude --agent capture-engine
#   cd ../sigscan-eval-harness   && claude --agent eval-harness
set -euo pipefail

AGENTS=(capture-engine eval-harness cnn-training llm-vision test-docs)
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ここは git リポジトリではありません。先に 'git init && git add -A && git commit' を。" >&2
  exit 1
}
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"

cmd="${1:-help}"
case "$cmd" in
  setup)
    for a in "${AGENTS[@]}"; do
      wt="$PARENT/${NAME}-${a}"
      br="agent/${a}"
      if git -C "$ROOT" worktree list | grep -q "$wt"; then
        echo "skip (既存): $wt"
      else
        git -C "$ROOT" worktree add -b "$br" "$wt" >/dev/null
        echo "作成: $wt  (branch $br)"
      fi
    done
    echo "→ 各 worktree で 'claude --agent <name>' を起動して並列作業。"
    ;;
  list)
    git -C "$ROOT" worktree list
    ;;
  remove)
    for a in "${AGENTS[@]}"; do
      wt="$PARENT/${NAME}-${a}"
      if git -C "$ROOT" worktree list | grep -q "$wt"; then
        git -C "$ROOT" worktree remove "$wt" && echo "削除: $wt"
      fi
    done
    ;;
  *)
    sed -n '2,18p' "$0"
    ;;
esac
