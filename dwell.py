"""滞在観測（dwell 観測の長時間化）。

1つの対象周波数に一定時間留まり、IQ を複数回取得してバーストを待ち受ける。
2.4GHz の WiFi/BT は数秒おきに一瞬しか出ないため、約13msの単発観測では
取り逃す。滞在して反復観測し、

  * 最も良く捉えた瞬間（最大の検出マージン）を代表として残す
  * 観測の統計（出現回数・持続率=検出された割合・検出マージンの分布）を集計する

ことで、後段の品質ゲート（quality.py）が「持続性」を根拠に保存可否を判断できる
ようにする。依存は numpy のみ。取得は継ぎ目 `backend.capture_iq` を、測定は継ぎ目
`dsp.measure_signal` をそのまま使う（どちらもシグネチャは変更しない）。

検出マージンについて:
  dsp.measure_signal の in-band SNR は、信号が取得帯域(IBW)を埋めると床も一緒に
  持ち上がり過小評価される（広帯域の WiFi/携帯など）。一方、信号の在/不在は
  「絶対ピーク電力」に表れる（在: 高い / 不在=ノイズのみ: 低い）。そこで滞在中の
  ピーク電力の最小値を受信機ノイズ床の推定として使い、

      det_snr = max(in-band SNR, peak_db - 滞在中の最小peak_db)

  を「検出マージン」と定義する。狭帯域信号は in-band SNR 側で、広帯域バーストは
  ピーク上昇側で捉えられ、両者を統一的に扱える。
  （注: 滞在中ずっと帯域を埋め続ける常時信号=途切れない携帯DL等は、不在の静かな
   観測が無いためマージンが立たず過小検出になりうる。これは単一IBW取得の本質的
   限界で、その種の常時信号はサーベイ側のログで拾う。）
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np

import dsp


@dataclass
class DwellObservation:
    """1対象帯の滞在観測の集計結果。"""
    center_hz: float
    target_src: str
    n_obs: int                  # 総観測回数
    n_detect: int               # はっきり検出された回数（出現回数）
    persistence: float          # 持続率 = n_detect / n_obs
    snr_max_db: float           # 検出マージンの最大（最も良く捉えた瞬間）
    snr_mean_db: float          # 検出マージンの平均
    snr_std_db: float           # 検出マージンのばらつき（小さい=同一強度で居座る）
    snr_detect_mean_db: float   # 検出された観測だけの平均マージン
    bw_rep_hz: float            # 代表観測の占有帯域幅
    bw_median_hz: float         # 全観測の中央値帯域幅
    occupied_frac_rep: float
    peak_db_rep: float
    noise_ref_db: float         # 検出に使った受信機ノイズ床推定
    best: dict                  # 代表観測の measure_signal 結果
    best_iq: np.ndarray         # 代表観測の生IQ（保存に使う）
    snr_series: list = field(default_factory=list)   # 各観測の検出マージン(dB)
    bw_series: list = field(default_factory=list)     # 各観測の帯域幅(Hz)
    # DCスパイク（中央スパイク）指標。dc_excess = 中央集中の強さ(dB)。
    #   平均が大きい=中央集中して細い、観測間の std が小さい=時間不変。両方揃えば
    #   DCオフセット由来の中央スパイクを疑う（quality.py で判定）。
    dc_excess_mean_db: float = 0.0       # 中央集中の強さ(dB)の平均
    dc_excess_std_db: float = 0.0        # 中央集中の強さ(dB)の観測間ばらつき
    dc_excess_series: list = field(default_factory=list)  # 各観測の dc_excess(dB)


def observe_dwell(backend, center_hz: float, rate: float, n_samples: int,
                  dcfg, qcfg, target_src: str = "", noise_floor_db=None,
                  time_fn=time.time, sleep_fn=time.sleep) -> DwellObservation:
    """1対象帯に dcfg.dwell_seconds 留まり、反復観測して統計を集計する。

    観測は obs_interval_s 間隔で行い、最低 min_observations 回・最大
    max_observations 回までで打ち切る。各観測の検出マージンが qcfg.detect_snr_db
    以上を「検出」と数える（→ 持続率）。代表は検出マージン最大の観測。

    noise_floor_db: 受信機ノイズ床の絶対基準(dB, measure_signal の peak_db スケール)。
        None なら滞在中の最小 peak_db を基準に自己推定する。
    time_fn / sleep_fn はテストから差し替え可能（実時間に依存せず回せる）。
    生IQ は代表分のみ保持する（全観測を抱えるとメモリを食うため）。
    """
    deadline = time_fn() + max(0.0, float(dcfg.dwell_seconds))
    obs_list: list[dict] = []
    iqs: list[np.ndarray] = []
    dc_excess: list[float] = []
    k = 0

    # --- 収集フェーズ: 反復観測して測定値を貯める ---
    while True:
        iq = backend.capture_iq(center_hz, rate, n_samples)
        m = dsp.measure_signal(iq, rate, center_hz)
        # DCスパイク指標: 中央(DC)が両脇よりどれだけ突出して細いか(dB)。
        #   観測ごとに測り、後段で平均(中央集中)と std(時間不変性)を取る。
        dcm = dsp.dc_spike_metrics(iq, rate, dc_band_hz=qcfg.dc_band_hz,
                                   side_hz=qcfg.dc_side_hz)
        obs_list.append(m)
        dc_excess.append(float(dcm["dc_excess_db"]))
        iqs.append(np.asarray(iq, dtype=np.complex64))
        k += 1
        if k >= dcfg.max_observations:
            break
        if time_fn() >= deadline and k >= dcfg.min_observations:
            break
        sleep_fn(max(0.0, float(dcfg.obs_interval_s)))

    # --- 集計フェーズ: 検出マージンを計算し統計を取る ---
    peak = np.asarray([m["peak_db"] for m in obs_list], dtype=float)
    inband_snr = np.asarray([m["snr_db"] for m in obs_list], dtype=float)
    bw_arr = np.asarray([m["bw_hz"] for m in obs_list], dtype=float)

    # 受信機ノイズ床基準: 指定が無ければ滞在中の最小ピーク（=最も静かな観測）。
    noise_ref = float(noise_floor_db) if noise_floor_db is not None else float(peak.min())
    det_snr = np.maximum(inband_snr, peak - noise_ref)   # 検出マージン

    detected = det_snr >= qcfg.detect_snr_db
    n_detect = int(detected.sum())

    best_i = int(np.argmax(det_snr))                      # 最も良く捉えた瞬間
    best = obs_list[best_i]
    det_vals = det_snr[detected]

    dc_arr = np.asarray(dc_excess, dtype=float)

    return DwellObservation(
        center_hz=float(center_hz),
        target_src=target_src,
        n_obs=int(k),
        n_detect=n_detect,
        persistence=float(n_detect / k) if k else 0.0,
        snr_max_db=float(det_snr.max()),
        snr_mean_db=float(det_snr.mean()),
        snr_std_db=float(det_snr.std()),
        snr_detect_mean_db=float(det_vals.mean()) if det_vals.size else 0.0,
        bw_rep_hz=float(best["bw_hz"]),
        bw_median_hz=float(np.median(bw_arr)),
        occupied_frac_rep=float(best.get("occupied_frac", 0.0)),
        peak_db_rep=float(best["peak_db"]),
        noise_ref_db=noise_ref,
        best=best,
        best_iq=iqs[best_i],
        snr_series=[round(float(x), 2) for x in det_snr],
        bw_series=[float(x) for x in bw_arr],
        dc_excess_mean_db=float(dc_arr.mean()),
        dc_excess_std_db=float(dc_arr.std()),
        dc_excess_series=[round(float(x), 2) for x in dc_arr],
    )
