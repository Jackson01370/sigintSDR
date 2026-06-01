#!/usr/bin/env bash
# =============================================================================
# sigscan セットアップ — HackRF + SoapySDR の土台を入れる
#
#   対応: Fedora系(dnf) / Ubuntu・Debian・Kali系(apt) / Arch系(pacman) を自動判別
#   方針: まずパッケージで導入 → 動作確認 → モジュールが無ければソースビルドで補完
#         （= どのディストリでも最終的に HackRF が使える状態にする）
#
#   使い方（USB上のプロジェクト直下で）:
#     bash setup.sh           ← 導入を実行（sudo パスワードを聞かれます）
#     bash setup.sh check     ← 何も入れず、現状の確認だけ
#
#   ※ exFAT の USB は実行ビットを保持しないため、必ず「bash setup.sh」で起動。
# =============================================================================
set -uo pipefail

MODE="${1:-install}"
HERE="$(cd "$(dirname "$0")" && pwd)"

c_ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
c_warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
c_err()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
hdr()    { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
# 1. ディストリ判別
# -----------------------------------------------------------------------------
hdr "1. OS を判別"
FAM=""
DISTRO="unknown"
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO="${ID:-unknown}"
    case " ${ID:-} ${ID_LIKE:-} " in
        *" fedora "*|*" rhel "*|*" centos "*) FAM=dnf ;;
        *" debian "*|*" ubuntu "*)            FAM=apt ;;   # Kali は ID_LIKE=debian
        *" arch "*)                            FAM=pacman ;;
    esac
fi
# 保険: コマンドの有無でも判定
if [ -z "$FAM" ]; then
    if   command -v dnf     >/dev/null 2>&1; then FAM=dnf
    elif command -v apt-get >/dev/null 2>&1; then FAM=apt
    elif command -v pacman  >/dev/null 2>&1; then FAM=pacman
    fi
fi
if [ -z "$FAM" ]; then
    c_err "対応するパッケージマネージャ(dnf/apt/pacman)が見つかりません。"
    exit 1
fi
c_ok "ディストリ: ${DISTRO}  /  パッケージ管理: ${FAM}"

# -----------------------------------------------------------------------------
# パッケージ操作ヘルパ
# -----------------------------------------------------------------------------
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

pm_refresh() {
    case "$FAM" in
        apt)    $SUDO apt-get update -y ;;
        dnf)    : ;;                       # dnf は都度メタ更新
        pacman) $SUDO pacman -Sy --noconfirm ;;
    esac
}

# 必須パッケージ（無いと困る）。失敗したら中断。
pm_required() {
    case "$FAM" in
        apt)    $SUDO apt-get install -y "$@" ;;
        dnf)    $SUDO dnf install -y "$@" ;;
        pacman) $SUDO pacman -S --needed --noconfirm "$@" ;;
    esac
}

# 任意パッケージ（名前がディストリ差で違う可能性）。1個ずつ試し、失敗は警告のみ。
pm_optional() {
    local p
    for p in "$@"; do
        case "$FAM" in
            apt)    $SUDO apt-get install -y "$p"            >/dev/null 2>&1 ;;
            dnf)    $SUDO dnf install -y "$p"                >/dev/null 2>&1 ;;
            pacman) $SUDO pacman -S --needed --noconfirm "$p" >/dev/null 2>&1 ;;
        esac
        if [ $? -eq 0 ]; then c_ok "任意パッケージ: $p"; else c_warn "見つからず: $p（後で検証/補完）"; fi
    done
}

# -----------------------------------------------------------------------------
# 検証関数
# -----------------------------------------------------------------------------
has_hackrf_module() {
    command -v SoapySDRUtil >/dev/null 2>&1 || return 1
    SoapySDRUtil --info 2>/dev/null | grep -iq hackrf
}
has_py_soapy() { python3 -c "import SoapySDR" >/dev/null 2>&1; }

# -----------------------------------------------------------------------------
# check モード: 何も入れずに現状確認だけ
# -----------------------------------------------------------------------------
if [ "$MODE" = "check" ]; then
    hdr "現状チェック（インストールはしません）"
    command -v hackrf_info  >/dev/null 2>&1 && c_ok "hackrf_info あり"          || c_warn "hackrf_info なし"
    command -v SoapySDRUtil >/dev/null 2>&1 && c_ok "SoapySDRUtil あり"         || c_warn "SoapySDRUtil なし"
    has_hackrf_module && c_ok "SoapySDR の hackrf モジュールあり"               || c_warn "hackrf モジュールなし"
    has_py_soapy && c_ok "python3 で import SoapySDR 可"                        || c_warn "python3 SoapySDR バインディングなし"
    python3 -c "import numpy" >/dev/null 2>&1 && c_ok "numpy あり"              || c_warn "numpy なし"
    echo; c_ok "確認のみ完了。導入するには引数なしで実行: bash setup.sh"
    exit 0
fi

# -----------------------------------------------------------------------------
# 2. パッケージ導入
# -----------------------------------------------------------------------------
hdr "2. パッケージを導入（sudo パスワードを聞かれます）"
pm_refresh || c_warn "パッケージ情報の更新に失敗（続行）"

case "$FAM" in
  apt)
    pm_required hackrf git cmake g++ pkg-config python3 python3-pip python3-numpy
    pm_optional soapysdr-tools libsoapysdr-dev libhackrf-dev \
                python3-soapysdr soapysdr-module-hackrf soapysdr0.8-module-hackrf
    ;;
  dnf)
    pm_required hackrf git cmake gcc-c++ python3 python3-pip python3-numpy
    pm_optional SoapySDR SoapySDR-devel hackrf-devel \
                python3-soapysdr soapy-hackrf SoapyHackRF
    ;;
  pacman)
    pm_required hackrf git cmake gcc pkgconf python python-pip python-numpy
    pm_optional soapysdr soapyhackrf python-soapysdr
    ;;
esac

# -----------------------------------------------------------------------------
# 3. hackrf モジュールの検証 → 無ければソースビルドで補完
# -----------------------------------------------------------------------------
hdr "3. SoapySDR の HackRF モジュールを確認"
if has_hackrf_module; then
    c_ok "hackrf モジュールを認識"
else
    c_warn "パッケージで入らなかったため、ソースからビルドします"
    # ビルドに必要な開発ファイル（SoapySDR本体 + libhackrf）
    case "$FAM" in
        apt)    pm_optional libsoapysdr-dev libhackrf-dev soapysdr-tools ;;
        dnf)    pm_optional SoapySDR-devel hackrf-devel ;;
        pacman) pm_optional soapysdr ;;
    esac
    TMP="$(mktemp -d)"
    if git clone --depth 1 https://github.com/pothosware/SoapyHackRF.git "$TMP/SoapyHackRF" \
        && cmake -S "$TMP/SoapyHackRF" -B "$TMP/SoapyHackRF/build" \
        && cmake --build "$TMP/SoapyHackRF/build" -j "$(nproc 2>/dev/null || echo 2)" \
        && $SUDO cmake --install "$TMP/SoapyHackRF/build"; then
        $SUDO ldconfig 2>/dev/null || true
        has_hackrf_module && c_ok "ソースビルド成功・モジュール認識" \
                          || c_err "ビルドしたが認識できず（SoapySDR本体の導入を確認してください）"
    else
        c_err "ソースビルドに失敗（ネットワーク/開発ツールを確認してください）"
    fi
    rm -rf "$TMP"
fi

# -----------------------------------------------------------------------------
# 4. 非root で使えるように（udev / plugdev）
# -----------------------------------------------------------------------------
hdr "4. 一般ユーザーで使えるよう設定（udev）"
if getent group plugdev >/dev/null 2>&1; then
    $SUDO usermod -aG plugdev "$USER" 2>/dev/null && c_ok "ユーザー $USER を plugdev に追加" \
        || c_warn "plugdev への追加に失敗（既に所属かも）"
fi
$SUDO udevadm control --reload-rules 2>/dev/null || true
$SUDO udevadm trigger 2>/dev/null || true
c_ok "udev ルールを再読込（HackRF は一度抜き挿ししてください）"

# -----------------------------------------------------------------------------
# 5. 動作確認
# -----------------------------------------------------------------------------
hdr "5. 動作確認"
python3 -c "import numpy" >/dev/null 2>&1 && c_ok "numpy OK" || c_warn "numpy が見えません"
has_py_soapy && c_ok "python3 で SoapySDR import OK" || c_warn "python3 SoapySDR バインディング未検出"

if has_hackrf_module; then c_ok "SoapySDR hackrf モジュール OK"; else c_err "hackrf モジュール未検出"; fi

# HackRF が挿さっていれば実機も確認
if command -v hackrf_info >/dev/null 2>&1; then
    if hackrf_info 2>/dev/null | grep -qi "Serial number\|Found HackRF"; then
        c_ok "HackRF を検出（実機OK）"
    else
        c_warn "HackRF は未接続か未検出（挿してから 'hackrf_info' で確認）"
    fi
fi

# プロジェクトの Sim を1サイクル回す（コードが動くかの最終確認）
if [ -f "$HERE/main.py" ]; then
    hdr "6. プロジェクトのSim動作確認 (main.py --sim --once)"
    if PYTHONIOENCODING=utf-8 python3 "$HERE/main.py" --sim --once >/dev/null 2>&1; then
        c_ok "Sim 実行 OK（プログラム本体は正常）"
    else
        c_warn "Sim 実行に失敗。numpy などの導入を確認してください"
    fi
fi

hdr "完了"
cat <<'EOF'
次の一手:
  1) HackRF を一度抜き差し（udev/グループ反映のため。反映されない場合は一度ログアウト→ログイン）
  2) hackrf_info            … 実機が見えるか確認
  3) SoapySDRUtil --find    … SoapySDR から見えるか確認
  4) python3 main.py --sim --once          … プログラム確認（ハード不要）
  5) python3 main.py --hardware --start 2.4e9 --stop 2.5e9 --collect captures/   … 実機で2.4GHz収集
EOF
