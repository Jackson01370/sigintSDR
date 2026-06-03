"""分類器: 3段構成（ルールベース → CNN → LLM Vision）。

今回はステップ1（ルールベース）を実装し、CNN/LLM は差込口だけ用意する。
measurement（dsp.measure_signal の戻り）を入力に分類結果を返す。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from config import Band


@dataclass
class ClassResult:
    label: str
    confidence: float
    method: str                       # "rule" | "cnn" | "llm"
    notes: str = ""
    candidates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 信号DB: バンド内での帯域幅などから具体ラベルを精緻化するルール群
#   (band_name_substr, bw条件(Hz) or None, label, conf, note)
# ---------------------------------------------------------------------------
SIGNAL_DB: list[tuple] = [
    ("GPS L1",      None,            "GPS/QZSS L1 C/A",       0.80, "BPSK拡散・広帯域・微弱"),
    ("GPS L5",      None,            "GPS/QZSS L5",           0.75, "拡散・~24MHz"),
    ("GPS L2",      None,            "GPS L2",                0.70, ""),
    ("GLONASS",     None,            "GLONASS L1",            0.65, "FDMA"),
    # [混入注意 BANDPLAN_PROPOSAL §X1] 2412/2437/2462/2484MHz は空間伝送型WPT(無線電力
    #   伝送)がWiFiチャネル中心と重なり、強い定常信号として WiFi/レーダと誤認されうる。
    # [混入注意 §X4] 2400-2450MHz は Amateur(2.4G) が ISM と重複し用途確定が難しい
    #   (低確信度→Unknown で吸収)。WPT判定や用途弾きの実ロジックは将来課題（今回は注記のみ）。
    ("ISM 2.4G",    (15e6, 45e6),    "WiFi (2.4GHz, 20/40MHz)", 0.78, "OFDM・矩形ブロック"),
    ("ISM 2.4G",    (None, 3e6),     "BLE/Bluetooth (adv?)",  0.62, "狭帯域・ホッピング"),
    ("ISM 2.4G",    (3e6, 15e6),     "Zigbee/独自2.4G",        0.55, ""),
    # [混入注意 §X2/§X3] 5250-5372.5(W53) / 5470-5725(W56) は気象レーダ(Pulse)が
    #   WiFi(OFDM)と同居しDFS対象。同一帯域に方式の異なる信号が混在するため、用途では
    #   なく方式ラベルでの分離が要。レーダ/WiFi 弁別の実ロジックは将来課題（注記のみ）。
    ("WiFi 5G",     (15e6, 180e6),   "WiFi (5GHz, 20-160MHz)", 0.78, "OFDM"),
    ("ISM 5.8G",    (10e6, 30e6),    "FPVドローン映像 or 5.8G WiFi", 0.55, "アナログ/デジタル映像の可能性"),
    ("ISM 5.8G",    (None, 10e6),    "ETC/DSRC or コードレス",  0.55, "5.8GHz帯狭帯域"),
    ("Cellular B1", (3e6, 60e6),     "LTE/UMTS DL (Band1 2.1G)", 0.80, "OFDM連続ブロック"),
    ("Cellular B3", (3e6, 40e6),     "LTE DL (Band3 1.8G)",   0.78, "OFDM"),
    ("Cellular B7", (3e6, 80e6),     "LTE/NR DL (Band7 2.6G)", 0.75, ""),
    ("5G NR n77",   (40e6, 200e6),   "5G NR (n77/n78 3.5G)",  0.80, "TDD広帯域"),
    ("5G NR n79",   (40e6, 200e6),   "5G NR (n79 4.7G)",      0.72, "TDD"),
    ("Aero/Radar",  None,            "航空監視レーダ等",         0.50, "短パルス・S帯"),
    ("Ham",         None,            "アマチュア無線",           0.45, ""),
]

# ノイズ/未確定の統一ラベル（CNN/LLM が DB 外を返したときの吸収先）
UNKNOWN = "未識別信号"
NOISE = "ノイズ/フロア変動"


def _match_band(center_hz: float, bands: list[Band]) -> Band | None:
    best = None
    for b in bands:
        if b.f_lo <= center_hz <= b.f_hi:
            if best is None or b.priority > best.priority:
                best = b
    return best


def _bw_ok(bw_hz: float, cond) -> bool:
    if cond is None:
        return True
    lo, hi = cond
    if lo is not None and bw_hz < lo:
        return False
    if hi is not None and bw_hz > hi:
        return False
    return True


# ---------------------------------------------------------------------------
# ステップ1: ルールベース
# ---------------------------------------------------------------------------
def rule_based(measurement: dict, bands: list[Band]) -> ClassResult:
    center = measurement["center_hz"]
    bw = measurement["bw_hz"]
    snr = measurement["snr_db"]

    band = _match_band(center, bands)
    if band is None:
        return ClassResult(UNKNOWN, 0.20, "rule",
                           f"バンドプラン外 ({center/1e6:.1f}MHz)")

    # 信号DBで精緻化
    candidates = []
    for name_sub, bw_cond, label, conf, note in SIGNAL_DB:
        if name_sub in band.name and _bw_ok(bw, bw_cond):
            candidates.append((label, conf, note))

    if candidates:
        # SNR が低いほど確信度を割り引く
        label, conf, note = candidates[0]
        if snr < 6:
            conf *= 0.6
        conf = min(conf, 0.85)
        return ClassResult(label, round(conf, 2), "rule",
                           f"{band.name}: {note}".strip(": "),
                           [c[0] for c in candidates])

    # バンドは判るが信号DBに該当なし → バンド名のヒントで返す
    conf = 0.45 if snr >= 8 else 0.30
    return ClassResult(band.name, conf, "rule", band.hint)


# ---------------------------------------------------------------------------
# ステップ2/3: フック（未実装 → None を返して劣化動作）
# ---------------------------------------------------------------------------
def cnn_classify(spectrogram_db) -> ClassResult | None:
    """CNN 推論フック。学習済みモデル導入後にここを実装。"""
    return None


def llm_classify(spectrogram_png_path: str | None,
                 measurement: dict | None = None,
                 bands: list[Band] | None = None,
                 rule_result: "ClassResult | None" = None,
                 *, iq=None, rate: float | None = None) -> ClassResult | None:
    """LLM Vision 段（低信頼度・未知信号を画像で判定）。

    実体は `llmvision` パッケージ。プロバイダ・API キーが未設定なら
    `llmvision.llm_classify` が None を返し、上位は graceful degradation する。
    本関数のシグネチャは後方互換: PNG パス1つだけでも呼べる。
    """
    try:
        from llmvision import llm_classify as _impl
    except Exception:
        return None
    try:
        return _impl(spectrogram_png_path, measurement=measurement,
                     bands=bands, rule_result=rule_result,
                     iq=iq, rate=rate)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# オーケストレーション
# ---------------------------------------------------------------------------
def classify(measurement: dict, bands: list[Band],
             spectrogram_db=None, png_path: str | None = None,
             cnn_threshold: float = 0.60) -> ClassResult:
    r = rule_based(measurement, bands)
    if r.confidence >= 0.85:
        return r

    if spectrogram_db is not None:
        c = cnn_classify(spectrogram_db)
        if c and c.confidence >= cnn_threshold:
            return c

    if png_path is not None and r.confidence < 0.5:
        l = llm_classify(png_path, measurement=measurement,
                         bands=bands, rule_result=r)
        if l:
            return l

    return r
