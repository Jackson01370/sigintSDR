"""品質ゲート（保存条件）— 量より質。

低品質・断片的なキャプチャを土台にすると、それが学習の基準（お手本）になり
全体が使い物にならなくなる。よって保存は厳しめに足切りする。判断材料は
`dwell.DwellObservation`（滞在観測の統計）で、しきい値は `config.QualityConfig`。

保存するのは以下を全て満たすものだけ:
  1) 持続性: 滞在中に複数回はっきり検出された（単発・かすりは破棄）。
  2) 極細スプリアス除外: 占有が極端に細いだけの山を破棄。ただし幅だけで切らず、
     「同一強度で居座る」かどうかと併用し、バースト性のある正規の狭帯域信号
     (BLE等)を誤って捨てない。
  3) コムスプリアス除外: 等間隔・同一強度で並ぶ細いピーク列（受信機内部由来。
     アンテナ無しでも出る固定パターン）を検出して破棄。

品質メタ（観測回数・持続率・SNR統計・スプリアス疑いフラグ等）は保存する SigMF の
annotation に `sigscan:` 名前空間で記録する。凍結契約 sigmf_io.write_recording の
annotation キー許可リストは変えられないため、書き出し後に meta JSON を最小限
patch する（review.py と同じく、ロケール既定エンコーディングで往復互換を保つ）。
依存は標準ライブラリのみ。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field


@dataclass
class QualityVerdict:
    passed: bool
    reasons: list = field(default_factory=list)   # 破棄理由（空なら合格）
    is_spur_suspect: bool = False                  # 単独での極細・居座りスプリアス疑い
    is_comb_spur: bool = False                      # 等間隔コムスプリアスの一員


def evaluate_quality(obs, qcfg, comb_spur: bool = False,
                     bw_hz=None) -> QualityVerdict:
    """滞在観測 1 件の保存可否を判定する。

    comb_spur は flag_comb_spurs によるクロスターゲット判定結果を渡す。
    bw_hz: 「極細」判定に使う占有帯域幅(Hz)。None なら obs.bw_rep_hz。検出帯では
        サーベイ実測(detect_segments)の帯域幅の方が堅牢なため、呼び出し側が渡せる。
    qcfg.enabled が False ならゲートを無効化（常に合格）するが、スプリアス疑い
    フラグはメタ記録用に計算する。
    """
    bw = obs.bw_rep_hz if bw_hz is None else float(bw_hz)
    narrow = bw < qcfg.narrow_bw_hz
    steady = obs.snr_std_db <= qcfg.spur_snr_std_max
    # 単独スプリアス疑い: 細い + ほぼ一定強度 + ほぼ常時 → 受信機内部スプリアス。
    spur_suspect = (narrow and steady
                    and obs.persistence >= qcfg.spur_persistence_min)

    reasons: list[str] = []

    # 1) 持続性（単発・かすりを破棄）
    if obs.n_detect < qcfg.min_detections:
        reasons.append(
            f"transient(detect={obs.n_detect}<{qcfg.min_detections})")
    if obs.persistence < qcfg.min_persistence:
        reasons.append(
            f"low-persistence({obs.persistence:.2f}<{qcfg.min_persistence:.2f})")

    # 2) 極細スプリアス（同一強度で居座る narrow を破棄。
    #    バースト性のある narrow=BLE等は steady でないので残る）
    if spur_suspect:
        reasons.append("narrow-steady-spur")

    # 3) クロスターゲット等間隔コムスプリアス
    if comb_spur:
        reasons.append("comb-spur")

    if not qcfg.enabled:
        return QualityVerdict(True, [], spur_suspect, comb_spur)
    return QualityVerdict(len(reasons) == 0, reasons, spur_suspect, comb_spur)


def flag_comb_spurs(observations, qcfg, bw_list=None) -> list[bool]:
    """等間隔・同一強度の細いピーク列（受信機コムスプリアス）を検出する。

    細い(narrow_bw 未満)観測を中心周波数で並べ、隣接間隔がほぼ一定で、かつ
    ピーク強度がほぼ揃っているランを探す。ランの本数が comb_min_run 以上なら、
    その全メンバを comb スプリアスとして True にする。

    bw_list: 各 observation の「極細」判定に使う占有帯域幅(Hz)。None なら
        各 obs.bw_rep_hz。検出帯ではサーベイ実測の帯域幅を渡せる。
    returns: observations と同順・同長の bool 列。
    """
    n = len(observations)
    flags = [False] * n

    def _bw(i, o):
        return o.bw_rep_hz if bw_list is None else float(bw_list[i])

    cand = sorted(((i, o) for i, o in enumerate(observations)
                   if _bw(i, o) < qcfg.narrow_bw_hz),
                  key=lambda t: t[1].center_hz)
    m = len(cand)
    if m < qcfg.comb_min_run:
        return flags

    centers = [o.center_hz for _, o in cand]
    powers = [o.peak_db_rep for _, o in cand]
    gaps = [centers[j + 1] - centers[j] for j in range(m - 1)]   # 長さ m-1

    # gap 列を「最初の間隔と一致し続ける」ランに分割する。
    # gap[j..k] が等間隔 ⇔ ピーク [j..k+1] が等間隔列。
    j = 0
    while j < len(gaps):
        k = j
        while (k + 1 < len(gaps)
               and abs(gaps[k + 1] - gaps[j]) <= qcfg.comb_spacing_tol_hz):
            k += 1
        peak_lo, peak_hi = j, k + 1                  # この等間隔列に属するピーク範囲
        run_powers = powers[peak_lo:peak_hi + 1]
        homogeneous = (max(run_powers) - min(run_powers)) <= qcfg.comb_power_tol_db
        if (peak_hi - peak_lo + 1) >= qcfg.comb_min_run and homogeneous:
            for r in range(peak_lo, peak_hi + 1):
                flags[cand[r][0]] = True
        j = k + 1
    return flags


def quality_annotation_meta(obs, verdict) -> dict:
    """保存する SigMF の annotation に載せる品質メタ（全キー sigscan: 名前空間）。"""
    return {
        "sigscan:dwell_obs": int(obs.n_obs),
        "sigscan:dwell_detect": int(obs.n_detect),
        "sigscan:persistence": round(float(obs.persistence), 3),
        "sigscan:snr_max_db": round(float(obs.snr_max_db), 1),
        "sigscan:snr_mean_db": round(float(obs.snr_mean_db), 1),
        "sigscan:snr_std_db": round(float(obs.snr_std_db), 2),
        "sigscan:bw_median_hz": round(float(obs.bw_median_hz), 1),
        "sigscan:spur_suspect": bool(verdict.is_spur_suspect or verdict.is_comb_spur),
        "sigscan:quality_pass": bool(verdict.passed),
    }


def add_quality_to_meta(path_base: str, quality: dict, ann_index: int = 0) -> dict:
    """書き出し済み .sigmf-meta の指定 annotation に品質メタを追記する。

    凍結契約 sigmf_io.write_recording を呼んだ「後」に最小限 patch する。生IQ
    (.sigmf-data) には触れない。読み書きとも sigmf_io と同じ（ロケール既定）
    エンコーディング・整形(indent=2, ensure_ascii=False)に合わせ、patch 後の meta を
    sigmf_io.read_recording が再び読めることを保証する（review.py と同じ規律）。
    returns: 更新後の meta。
    """
    meta_path = path_base + ".sigmf-meta"
    with open(meta_path) as f:
        meta = json.load(f)
    anns = meta.get("annotations") or []
    if 0 <= ann_index < len(anns):
        anns[ann_index].update(quality)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta
