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
    """CNN 推論フック（旧・スペクトログラム経路。常に None＝劣化動作）。

    M3 の CNN 監査は本フックではなく、下の CNN 監査コンテキスト経由で
    凍結 spec.render（infer.classify_iq）を通して行う（新たな前処理を作らない）。
    本スタブは従来挙動（spectrogram_db を渡しても rule のまま）を保つため残す。
    """
    return None


# ---------------------------------------------------------------------------
# ステップ2: CNN 監査コンテキスト（既定 OFF。set されない限り classify は従来と
#   完全に同一の出力）。
#   凍結シグネチャ classify.classify(...) を変えずに IQ/rate を 2 段目へ渡すための、
#   作業指示が明示的に許可した「オプショナルなコンテキスト設定関数」方式。単一
#   スレッドのスケジューラが各信号の直前に set し、直後に clear する。torch を引く
#   cnntrain.infer は **本コンテキストが set されたときだけ遅延 import** される
#   （OFF 運用・torch 無し環境で classify は壊れない＝禁止事項3）。
# ---------------------------------------------------------------------------
@dataclass
class CNNAuditContext:
    checkpoint: object            # cnntrain.infer.Checkpoint（型注釈で torch を引かない）
    iq: object                    # 生 IQ（complex64 配列等）
    rate: float
    center_hz: float
    checkpoint_name: str = ""     # 来歴記録用（ファイル名で可）
    provenance: dict | None = None    # 監査後に classify が埋める（呼び出し側が SigMF へ）
    decision: object = None           # AuditDecision（テスト・デバッグ用）


_cnn_ctx: "CNNAuditContext | None" = None


def set_cnn_context(ctx: "CNNAuditContext | None") -> None:
    """次の classify() 呼び出しで使う CNN 監査コンテキストを設定する。

    呼ばない/None を渡す限り classify は従来挙動（CNN 段は完全にスキップ）。
    """
    global _cnn_ctx
    _cnn_ctx = ctx


def clear_cnn_context() -> None:
    """CNN 監査コンテキストを解除する（classify 呼び出し後に必ず呼ぶ）。"""
    global _cnn_ctx
    _cnn_ctx = None


def _run_cnn_audit(r: ClassResult, measurement: dict,
                   ctx: CNNAuditContext) -> ClassResult:
    """ルール結果 r を CNN 所見で監査し、確信度調整・(C)で Unknown 化して返す。

    torch を引く cnntrain.infer は **ここで遅延 import**（CNN 有効時のみ）。
    凍結 spec.render を通すため infer.classify_iq を再利用する（新前処理なし）。
    """
    from cnntrain import infer           # torch を引く: CNN 有効時のみ
    from cnntrain import audit as _audit

    cnn_class, cnn_conf = infer.classify_iq(ctx.checkpoint, ctx.iq, ctx.rate)
    center = measurement.get("center_hz")
    decision = _audit.audit(r.label, r.confidence, cnn_class, cnn_conf,
                            center_hz=center)

    # 来歴（SigMF 用。呼び出し側＝scheduler が extra_global に載せる）。
    # 調整前後の確信度が追えること（rule_conf_pre / cnn_conf_post）。
    ctx.decision = decision
    ctx.provenance = {
        "sigscan:cnn_class": cnn_class,
        "sigscan:cnn_conf": round(float(cnn_conf), 3),
        "sigscan:cnn_verdict": decision.verdict,
        "sigscan:cnn_checkpoint": ctx.checkpoint_name,
        "sigscan:rule_conf_pre": round(float(r.confidence), 3),
        "sigscan:cnn_conf_post": round(float(decision.conf_after), 3),
    }
    return _apply_cnn_decision(r, decision)


def _apply_cnn_decision(r: ClassResult, decision) -> ClassResult:
    """AuditDecision を ClassResult に反映（純粋な field 操作）。

    大原則: CNN の判定だけで用途ラベルを書き換えない。(C) は Unknown 化のみで、
    元ラベルは candidates に残し、人間判断（review.py）へ回す。
    """
    conf = round(float(decision.conf_after), 2)
    tag = (f"[CNN監査:{decision.verdict} "
           f"{decision.cnn_class}@{decision.cnn_conf:.2f}]")
    notes = (r.notes + " " + tag).strip() if r.notes else tag

    # unmapped: 期待対応表に用途が無い → ラベル・確信度・method はそのまま（所見のみ）。
    if decision.verdict == "unmapped":
        return ClassResult(r.label, r.confidence, r.method, notes,
                           list(r.candidates))

    # (C) かつ <0.7: 用途を Unknown に落とす（元ラベルを候補に残す）。method=cnn。
    if decision.to_unknown:
        cands = list(r.candidates)
        if r.label not in cands:
            cands = [r.label] + cands
        return ClassResult(UNKNOWN, conf, "cnn",
                           notes + " → 用途=Unknown(候補つき)", cands)

    # (A)/(B)/(C≥0.7): ラベル維持・確信度のみ調整。method=cnn（CNN が監査・調整）。
    return ClassResult(r.label, conf, "cnn", notes, list(r.candidates))


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

    # ステップ2: CNN 監査（既定 OFF。コンテキストが set されたときのみ）。
    #   ルール結果を CNN 所見で監査し、確信度を調整（(C) で Unknown 化）する。
    #   ここを通った後の r.confidence を 3段目 LLM のトリガが見る＝自然に直列
    #   （CNN で確信度が 0.5 未満まで下がれば既存の LLM トリガが発火する）。
    if _cnn_ctx is not None:
        r = _run_cnn_audit(r, measurement, _cnn_ctx)

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
