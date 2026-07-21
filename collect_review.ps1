<#
.SYNOPSIS
  収集〜確定のワンコマンド化ラッパー（オーケストレーションのみ）。

.DESCRIPTION
  収集 → 画像化 → 提案生成 → CC分類 → 人間の○× の5ステップを1コマンドで流す。
  各ステップは既存ツール（main.py / view_captures.py / cnntrain.review_suggest /
  review.py）を「呼ぶだけ」。分類ロジック・確定ロジック・ファイル名生成を
  **再実装しない**。最後の○×は必ず人間が押す（ラッパーは y を代行しない＝Pattern A 防波堤）。

  CC分類(ステップ4)は環境に応じて分岐:
    (A) `claude` CLI が使える → `claude -p` でヘッドレス自動分類・併合（成否と cc_verdicts.csv を確認）
    (B) 使えない/失敗/-NoHeadless → 指示文を再表示し Enter 待ち（人間/CC が別途分類・併合）

.PARAMETER Tag         バッチタグ（必須）。ファイル名に埋め込み --pattern "*<Tag>*" で選択可。
.PARAMETER Start       開始周波数Hz（既定 2.408e9）。
.PARAMETER Stop        終了周波数Hz（既定 2.416e9）。
.PARAMETER Max         収集自動停止の件数（既定 40）。
.PARAMETER Dwell       各対象の滞在秒数（既定 10）。
.PARAMETER QMinPersistence 品質ゲートの最低持続率（既定 0.2）。
.PARAMETER NoQualityGate   品質ゲート無効化（スプリアス収集用スイッチ）。
.PARAMETER Out         出力ディレクトリ（既定 bench/<Tag>/）。
.PARAMETER NoHeadless  ステップ4を常に (B) 待機フォールバックにする（claude -p を使わない）。
.PARAMETER DryRun      実行せずコマンド列だけ表示する（-WhatIf 相当のドライラン）。

.EXAMPLE
  .\collect_review.ps1 -Tag wifi_aug2 -Start 2.408e9 -Stop 2.416e9 -Max 40

.NOTES
  $py はスクリプト内で既定フルパスを設定（環境変数 SIGSCAN_PY で上書き可）。
  常設ルールは CLAUDE.md（$py フルパス必須・PowerShell && 禁止・captures/ の既存ファイルは不改変）。
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Tag,
  [double]$Start = 2.408e9,
  [double]$Stop = 2.416e9,
  [int]$Max = 40,
  [double]$Dwell = 10,
  [double]$QMinPersistence = 0.2,
  [switch]$NoQualityGate,
  [string]$Out = "",
  [switch]$NoHeadless,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# --- $py 解決（環境変数 SIGSCAN_PY で上書き可・既定はフルパス）---
$py = if ($env:SIGSCAN_PY) { $env:SIGSCAN_PY } else { "C:\Users\puppy\radioconda\envs\sigscan\python.exe" }

# --- タグ検証（英数・ハイフン・アンダースコアのみ。命名の唯一点は Python 側だが早期に弾く）---
if ($Tag -notmatch '^[A-Za-z0-9_-]+$') {
  Write-Host "エラー: バッチタグ '$Tag' が不正です（英数・ハイフン・アンダースコアのみ）。" -ForegroundColor Red
  exit 2
}

# --- 出力先の既定（bench/<Tag>）。末尾スラッシュを正規化して子パスの二重 // を防ぐ ---
if ([string]::IsNullOrWhiteSpace($Out)) { $Out = "bench/$Tag" }
$Out = $Out.TrimEnd('/', '\')
$pattern = "*$Tag*"
$verdicts = "$Out/cc_verdicts.csv"
$suggestions = "$Out/suggestions.csv"
$tasklist = "$Out/classify_tasklist.md"

# 品質ゲートのオプション引数（配列でスプラット。&& を使わない）
$gateArgs = @()
if ($NoQualityGate) { $gateArgs = @("--no-quality-gate") }

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  collect_review.ps1 : 収集〜確定ワンコマンド（オーケストレーションのみ）" -ForegroundColor Cyan
Write-Host "  Tag=$Tag  範囲=$Start-$Stop Hz  Max=$Max  Dwell=$Dwell  Out=$Out" -ForegroundColor Cyan
Write-Host "  最後の○×は必ず人間が実施（ラッパーは y を代行しない）" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# --- 各ステップの表示/ドライラン共通ヘルパ ---
function Show-Cmd([string]$label, [string[]]$cmd) {
  Write-Host "`n>>> $label" -ForegroundColor Yellow
  Write-Host "    $($cmd -join ' ')" -ForegroundColor DarkGray
}

# ドライランなら5ステップのコマンド列を並べて終了（実行しない）
if ($DryRun) {
  Write-Host "`n[DryRun] 実行はせず、流れるコマンド列を表示します:" -ForegroundColor Magenta
  Show-Cmd "1. 収集" (@($py, "main.py", "--hardware", "--start", $Start, "--stop", $Stop, "--focus", "--dwell-seconds", $Dwell, "--q-min-persistence", $QMinPersistence) + $gateArgs + @("--max-records", $Max, "--tag", $Tag, "--collect", "captures/"))
  Show-Cmd "2. 画像化" @($py, "view_captures.py", "captures/", "--pattern", "`"$pattern`"")
  Show-Cmd "3. 提案生成" @($py, "-m", "cnntrain.review_suggest", "--data", "captures/", "--pattern", "`"$pattern`"", "--out", $Out, "--auto-classify")
  $branch = if ($NoHeadless) { "(B) 待機フォールバック（-NoHeadless 指定）" } elseif (Get-Command claude -ErrorAction SilentlyContinue) { "(A) claude -p ヘッドレス自動（失敗時 B へ）" } else { "(B) 待機フォールバック（claude CLI 不在）" }
  Write-Host "`n>>> 4. CC分類 → $branch" -ForegroundColor Yellow
  Show-Cmd "5. 人間の○×（ここで人間が y/n を押す）" @($py, "review.py", "captures/", "--pattern", "`"$pattern`"", "--suggest", $suggestions, "--batch-confirm", "--open-sheet")
  Write-Host "`n[DryRun] 実際の収集・確定は行っていません。" -ForegroundColor Magenta
  exit 0
}

# =========================================================================
# 1. 収集（--max-records で自動停止するまで待つ）
# =========================================================================
$collectArgs = @("main.py", "--hardware", "--start", $Start, "--stop", $Stop, "--focus", "--dwell-seconds", $Dwell, "--q-min-persistence", $QMinPersistence) + $gateArgs + @("--max-records", $Max, "--tag", $Tag, "--collect", "captures/")
Show-Cmd "1. 収集" (@($py) + $collectArgs)
& $py @collectArgs
if ($LASTEXITCODE -ne 0) {
  Write-Host "エラー: 収集(main.py)が失敗しました（exit=$LASTEXITCODE）。以降を中止します。" -ForegroundColor Red
  exit 1
}
$collected = @(Get-ChildItem -Path "captures" -Filter "*$Tag*.sigmf-meta" -ErrorAction SilentlyContinue).Count
Write-Host "    収集件数（*$Tag* 一致 .sigmf-meta）: $collected 件" -ForegroundColor Green
if ($collected -eq 0) {
  Write-Host "エラー: 収集0件のため以降（画像化・提案・確定）を実行しません。" -ForegroundColor Red
  exit 1
}

# =========================================================================
# 2. 画像化
# =========================================================================
Show-Cmd "2. 画像化" @($py, "view_captures.py", "captures/", "--pattern", "`"$pattern`"")
# --pattern "*<Tag>*": 新規タグ分だけ描画（+冪等スキップで既存最新はスキップ）＝高速化。
#   全再描画したいときは view_captures.py に --force（このラッパーは新規分のみ描く）。
& $py view_captures.py captures/ --pattern "$pattern"
if ($LASTEXITCODE -ne 0) {
  Write-Host "エラー: 画像化(view_captures.py)が失敗しました（exit=$LASTEXITCODE）。中止します。" -ForegroundColor Red
  exit 1
}

# =========================================================================
# 3. 提案生成（--auto-classify で指示文ブロックも印字される）
# =========================================================================
Show-Cmd "3. 提案生成" @($py, "-m", "cnntrain.review_suggest", "--data", "captures/", "--pattern", "`"$pattern`"", "--out", $Out, "--auto-classify")
& $py -m cnntrain.review_suggest --data captures/ --pattern "$pattern" --out $Out --auto-classify
if ($LASTEXITCODE -ne 0) {
  Write-Host "エラー: 提案生成(review_suggest)が失敗しました（exit=$LASTEXITCODE）。中止します。" -ForegroundColor Red
  exit 1
}
if (-not (Test-Path $tasklist)) {
  Write-Host "エラー: タスクリストが生成されていません（$tasklist）。中止します。" -ForegroundColor Red
  exit 1
}

# =========================================================================
# 4. CC 分類（(A) ヘッドレス自動 / (B) 待機フォールバック）
# =========================================================================
$ccPrompt = "$tasklist を読んで、記載された全PNGを view で視覚分類し、$verdicts に record,cc_class,cc_rationale の CSV を書いてください。書けたら次で併合: python -m cnntrain.review_suggest --data captures/ --pattern `"$pattern`" --out $Out --verdicts $verdicts 。判断基準は CLAUDE.md §5（検出帯=赤帯の主役が何か）。"

$useHeadless = (-not $NoHeadless) -and [bool](Get-Command claude -ErrorAction SilentlyContinue)
if ($useHeadless) {
  Write-Host "`n>>> 4. CC分類 → (A) claude -p ヘッドレス自動" -ForegroundColor Yellow
  try {
    & claude -p $ccPrompt
    if ($LASTEXITCODE -ne 0) { throw "claude -p が非ゼロ終了（exit=$LASTEXITCODE）" }
  } catch {
    Write-Host "    (A) 失敗: $($_.Exception.Message) → (B) 待機フォールバックへ切替" -ForegroundColor DarkYellow
    $useHeadless = $false
  }
  if ($useHeadless -and -not (Test-Path $verdicts)) {
    Write-Host "    (A) 実行後も $verdicts が無い → (B) 待機フォールバックへ切替" -ForegroundColor DarkYellow
    $useHeadless = $false
  }
}

if (-not $useHeadless) {
  $why = if ($NoHeadless) { "-NoHeadless 指定" } elseif (Get-Command claude -ErrorAction SilentlyContinue) { "(A) が使えず" } else { "claude CLI 不在" }
  Write-Host "`n>>> 4. CC分類 → (B) 待機フォールバック（$why）" -ForegroundColor Yellow
  Write-Host "    上記ステップ3が印字した指示文ブロックを CC に渡して、視覚分類→$verdicts→併合を済ませてください。" -ForegroundColor Gray
  Write-Host "    （併合コマンド: $py -m cnntrain.review_suggest --data captures/ --pattern `"$pattern`" --out $Out --verdicts $verdicts）" -ForegroundColor Gray
  Read-Host "CC に上記を渡して分類・併合が済んだら Enter"
}

# 併合を1度だけ流し直して suggestions.csv を最新化（冪等・オーケストレーションのみ）。
# cc_verdicts.csv が無くても verdicts=無 で走り、needs-review として次段へ渡す。
Show-Cmd "4.5 併合確認（suggestions.csv 最新化）" @($py, "-m", "cnntrain.review_suggest", "--data", "captures/", "--pattern", "`"$pattern`"", "--out", $Out, "--verdicts", $verdicts)
& $py -m cnntrain.review_suggest --data captures/ --pattern "$pattern" --out $Out --verdicts $verdicts
if ($LASTEXITCODE -ne 0) {
  Write-Host "エラー: 併合(review_suggest --verdicts)が失敗しました（exit=$LASTEXITCODE）。中止します。" -ForegroundColor Red
  exit 1
}
if (-not (Test-Path $suggestions)) {
  Write-Host "エラー: suggestions.csv が無いため人間の○×へ進めません（$suggestions）。中止します。" -ForegroundColor Red
  exit 1
}

# =========================================================================
# 5. 人間の○×（ここで人間が y/n を押す。ラッパーは代行しない）
# =========================================================================
Show-Cmd "5. 人間の○×（人間が y/n を押す）" @($py, "review.py", "captures/", "--pattern", "`"$pattern`"", "--suggest", $suggestions, "--batch-confirm", "--open-sheet")
Write-Host "    ↓ ここから先は人間の確定操作です（AI は代行しません）。" -ForegroundColor Green
# --open-sheet: 対話の人間○×なので全 PNG を1枚のコンタクトシートにまとめて自動で開く
#   （表示補助のみ・確定は人間）。ヘッドレスのステップ4-A(claude -p)は review.py を
#   呼ばないためシートは付かない（GUI 不要）。
& $py review.py captures/ --pattern "$pattern" --suggest $suggestions --batch-confirm --open-sheet

# =========================================================================
# 終了サマリ
# =========================================================================
Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host "  完了サマリ" -ForegroundColor Cyan
Write-Host "  バッチタグ : $Tag（--pattern `"$pattern`" で再選択可）" -ForegroundColor Cyan
Write-Host "  収集件数   : $collected 件" -ForegroundColor Cyan
Write-Host "  出力先     : $Out（suggestions.csv / confirm_sheet.md / classify_tasklist.md）" -ForegroundColor Cyan
Write-Host "  確定       : 人間が review.py --batch-confirm で実施（AI は代行していない）" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
