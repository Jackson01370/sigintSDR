"""
test_collect_review_dryrun.py — collect_review.ps1 の -DryRun 生成コマンド列を固定する。

ps1 は「オーケストレーションのみ」なので、-DryRun が出すコマンド文字列を検証すれば
回帰を捕まえられる。pwsh(または powershell)が無い環境（Linux CI 等）では skip する。
アサートは ASCII のフラグ/コマンド文字列のみ（コンソール出力エンコーディングに依存しない）。

固定する契約（指示書「collect_review.ps1 を1GHz以下でも」検証項目 1-4）:
1. 既定（Part A 未指定・-NoSuggest なし）で 5 ステップの標準コマンドが出る。Part A の
   追加フラグは一切付かない。
2. Part A の各フラグは指定時のみステップ1に付く。未指定なら付かない。
3. -NoSuggest でステップ3(--auto-classify)が消え、ステップ5が --suggest/--batch-confirm/
   --open-sheet 無しになる。
"""
import os
import shutil
import subprocess

import pytest

PWSH = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(
    PWSH is None, reason="pwsh/powershell 不在（Linux CI 等）のため ps1 DryRun 検証を skip")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "collect_review.ps1")


def _dryrun(*extra):
    """collect_review.ps1 を -DryRun で走らせ標準出力を返す（$py は 'python' に固定）。"""
    cmd = [PWSH, "-NoProfile", "-File", SCRIPT,
           "-Tag", "t", "-Start", "80e6", "-Stop", "90e6",
           "-Max", "10", "-Dwell", "5", "-DryRun", *extra]
    env = dict(os.environ, SIGSCAN_PY="python")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, cwd=REPO)
    return r.stdout + r.stderr


# ---- 1) 既定: 5ステップ標準・Part A フラグ無し ----
def test_default_dryrun_five_canonical_steps():
    out = _dryrun()
    assert ("main.py --hardware --start 80000000 --stop 90000000 --focus "
            "--dwell-seconds 5 --q-min-persistence 0.2 --max-records 10 "
            "--tag t --collect captures/") in out
    assert "view_captures.py captures/ --pattern" in out
    assert "cnntrain.review_suggest" in out and "--auto-classify" in out
    assert ("review.py captures/ --pattern" in out
            and "--suggest" in out and "--batch-confirm" in out
            and "--open-sheet" in out)


def test_default_has_no_partA_flags():
    out = _dryrun()
    for flag in ("--max-minutes", "--dc-guard-hz", "--lna", "--vga",
                 "--q-narrow-bw", "--dwell-offset-hz"):
        assert flag not in out, f"既定で {flag} が付いている（既定不変違反）"


# ---- 2) Part A: 指定したフラグだけ付く ----
def test_partA_flags_only_when_specified():
    out = _dryrun("-DcGuardHz", "500000", "-MaxMinutes", "3",
                  "-QNarrowBw", "100000", "-Lna", "32", "-Vga", "20")
    assert "--dc-guard-hz 500000" in out
    assert "--max-minutes 3" in out
    assert "--q-narrow-bw 100000" in out
    assert "--lna 32" in out
    assert "--vga 20" in out
    # 指定していないものは付かない
    assert "--dwell-offset-hz" not in out


def test_partA_dcguard_zero_is_explicit():
    """-DcGuardHz 0 は「明示的に無効」として --dc-guard-hz 0 を渡す（未指定と区別）。"""
    out = _dryrun("-DcGuardHz", "0")
    assert "--dc-guard-hz 0" in out


def test_partA_flags_go_only_to_step1_not_review():
    """Part A フラグは収集(main.py)にのみ付き、review.py 行には付かない。"""
    out = _dryrun("-DcGuardHz", "500000")
    review_lines = [ln for ln in out.splitlines() if "review.py" in ln]
    assert review_lines
    for ln in review_lines:
        assert "--dc-guard-hz" not in ln


# ---- 3) -NoSuggest: 3/4/4.5 消滅・step5 は --suggest 無し ----
def test_nosuggest_drops_suggest_steps():
    out = _dryrun("-NoSuggest")
    assert "--auto-classify" not in out            # ステップ3 消滅
    assert "review.py captures/ --pattern" in out  # ステップ5 は残る
    # ステップ5 は --suggest 系を一切付けない
    assert "--suggest" not in out
    assert "--batch-confirm" not in out
    assert "--open-sheet" not in out


def test_nosuggest_still_has_collect_and_view():
    """-NoSuggest でも収集(step1)と画像化(step2)は残る。"""
    out = _dryrun("-NoSuggest")
    assert "main.py --hardware" in out
    assert "view_captures.py captures/ --pattern" in out


def test_partA_and_nosuggest_combined_fm_like():
    """1GHz以下の推奨呼び出し（Part A + -NoSuggest）が FM 成功コマンド相当を出す。"""
    out = _dryrun("-MaxMinutes", "3", "-DcGuardHz", "500000",
                  "-Lna", "32", "-Vga", "20", "-QNarrowBw", "100000", "-NoSuggest")
    # 収集に必要フラグが揃う
    for f in ("--max-minutes 3", "--dc-guard-hz 500000", "--lna 32",
              "--vga 20", "--q-narrow-bw 100000", "--max-records 10",
              "--tag t", "--collect captures/"):
        assert f in out
    # CC分類はスキップ・人間の直接入力
    assert "--auto-classify" not in out
    assert "--suggest" not in out
