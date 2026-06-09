"""cnntrain (1/6): Sim データ生成 CLI（torch 非依存）。

5 つの **方式軸（見え方）クラス** の合成 IQ を numpy で直接合成し、凍結
`sigmf_io.write_recording` で SigMF として保存する。

M2 実測アライン（M1.5プローブ+人間の画像確認で確定した乖離を反映）:
  * DC 残留線を全クラス一律（クラス無相関）に注入（_inject_dc_residual）。実BLEの
    cw-tone 誤認の真犯人＝中心の DC 残留線を再現。注入は IQ(時間領域)で行い凍結
    spec.render を通す（画像描き込みは禁止）。強度は実測 dc_excess に較正。
  * wideband-ofdm を「全時間持続の塊」から「非周期・可変幅の広帯域バースト列」へ
    再定義（実WiFiパケット様）。pulse-radar（厳密周期・短パルス）との対比は維持。
  * cw-tone は中心から |off|>0.5MHz に制約（注入DC線と分離）。

M2.5 形状の実測再アライン（M2スポットライト9枚の画像確認で測れた形状ギャップを反映）:
  * narrowband-burst を短く疎らに（_short_sparse_burst_envelope, 1〜3発・各窓の~3%）。
    在窓 BLE が cw-tone でなく narrowband-burst に届くことを狙う。帯域1〜2.5MHzは維持。
  * wideband-ofdm をほぼ全幅(≈18〜20MHz)化し、時間配置を「少数+長沈黙 / 密集クラスタ」
    の2モードに（実WiFi画像）。pulse-radar との弁別は時間構造に委ねる（周期 vs 非周期）。
  * 中心外スプリアス線も全クラス一律（クラス無相関）に注入（_inject_spurious_line）。
    「単独の細線が在るだけでは cw-tone と断定しない」を学べるように。実測 prominence
    に較正。DC 注入(M2)・pulse-radar 定義は不変。

生成手段に SimBackend を使わず numpy 直接合成を選んだ理由:
  * SimBackend の環境は固定の周波数割当で、クラス均衡な生成・パルス列・per-class の
    細かい制御に向かない（_default_sim_signals は環境スナップショット用）。
  * とはいえ合成イディオムは sdr.SimBackend を踏襲する（帯域制限ノイズ=FFTマスク、
    CW=複素正弦）。生 IQ は必ず凍結 spec.render / sigmf_io を通る。

真実ラベル(ground truth)の記録（凍結契約の範囲で）:
  * core:label        = 正準クラス名（合成では label == 真実。ルール出力ではない）
  * sigscan:method    = "sim-truth"（ルール由来でないことを明示）
  * sigscan:confidence= 1.0（生成時の真実は確実）
  * sigscan:true_class= 同じクラス名を **global** に冗長記録（extra_global 経由）。
      write_recording は annotation の sigscan: 名前空間に confidence/method/snr_db
      しか通さない（凍結）。true_class を annotation に入れる凍結互換の経路が無いため、
      1ファイル=1クラスである性質を使い global スコープに記録する。
  * core:hw           = "sigscan-sim (synthetic)"（既存慣習＝正直なハードウェア表記）

CLI:
    python -m cnntrain.simgen --out simdata/ --per-class 80 --seed 0
"""
from __future__ import annotations

import os
import sys

import numpy as np

import sigmf_io
from cnntrain import classes

RATE = classes.GEN_RATE_HZ
N = classes.GEN_SAMPLES

HW_SYNTHETIC = "sigscan-sim (synthetic)"


# ---------------------------------------------------------------------------
# 合成プリミティブ（sdr.SimBackend のイディオムを踏襲）
# ---------------------------------------------------------------------------
def _complex_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """基準の複素ガウシアンノイズ（各成分 N(0,1)）。"""
    return (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)).astype(np.complex64)


def _bandlimited(n: int, rate: float, off_hz: float, bw_hz: float,
                 rng: np.random.Generator) -> np.ndarray:
    """帯域制限ノイズ（OFDM/拡散/広帯域の代用）。単位標準偏差に正規化して返す。

    sdr.SimBackend.capture_iq と同じ FFT マスク方式。
    """
    white = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    spec_ = np.fft.fftshift(np.fft.fft(white))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / rate))
    spec_[np.abs(f - off_hz) > bw_hz / 2] = 0
    bl = np.fft.ifft(np.fft.ifftshift(spec_))
    bl = bl / (np.std(bl) + 1e-9)
    return bl.astype(np.complex64)


def _cw(n: int, rate: float, off_hz: float) -> np.ndarray:
    """連続波（CW）トーン = 単一複素正弦。振幅 1。"""
    t = np.arange(n) / rate
    return np.exp(2j * np.pi * off_hz * t).astype(np.complex64)


def _amp(snr_db: float) -> float:
    """SNR(dB) → ノイズ床(振幅~1)に対する信号振幅スケール。"""
    return float(10.0 ** (snr_db / 20.0))


# ---------------------------------------------------------------------------
# DC 残留線の注入（M2: 実測アライン）
# ---------------------------------------------------------------------------
# ゼロIF受信機は remove_dc 後も中心(0Hzオフセット)に弱い残留線を残す。実 captures は
# dc_removed=True でも dsp.dc_spike_metrics の dc_excess_db が概ね [1.0, 4.25]dB
# (10-90pct)・中央値~1.4・最大~10.8 に分布する（事前確認3で実測・ラベル不使用）。
# 合成データには DC が無かったため M1.5 で CNN が「中心の細線」を cw-tone の手掛かりに
# 誤用した（画像確認で確定した真犯人）。これを是正するため、**全クラスに同一分布で**
# （クラスと無相関に）DC 残留を注入する。注入は IQ(時間領域)で行い凍結 spec.render を
# 通す（画像への線の描き込みは禁止）。amp はノイズ標準偏差(=1)に対する相対値で、
# dc_excess は相対dB＝絶対ゲイン非依存なので合成にそのまま較正できる。
# 事前確認3の amp→dc_excess 較正:
#   amp 0.03->1.0dB / 0.05->2.4dB / 0.065->3.8dB / 0.08->4.8dB / 0.16->~10dB
DC_INJECT_PROB = 0.8                  # 注入確率（0.7〜0.9 の範囲）
DC_AMP_MAIN = (0.025, 0.06)           # 主分布 → excess ~ [0.9, 3.0]dB（実測の低偏重に整合）
DC_AMP_TAIL = (0.07, 0.16)            # 強テール → excess 最大~10dB（実測max10.8に対応）
DC_TAIL_PROB = 0.12                   # テール（強め）に入る確率
DC_DRIFT_HZ = 8e3                     # ゆっくりしたドリフト（DC帯 |f|<60kHz 内に収まる）
DC_EXCESS_CAL_RANGE = (1.0, 4.25)     # 報告用: 実測10-90pctの目標域


def _inject_dc_residual(iq: np.ndarray, rate: float, rng: np.random.Generator):
    """中心(0Hz)に弱い DC 残留線を確率 DC_INJECT_PROB で注入する。

    複素の微小オフセット + ゆっくりしたドリフト（remove_dc 後の残留を模す）。
    クラスに依らず同一分布から引く（=識別の手掛かりにならない）。
    returns: (iq, injected: bool, amp: float|None)
    """
    if rng.random() >= DC_INJECT_PROB:
        return iq.astype(np.complex64), False, None
    if rng.random() < DC_TAIL_PROB:
        amp = float(rng.uniform(*DC_AMP_TAIL))
    else:
        amp = float(rng.uniform(*DC_AMP_MAIN))
    f_drift = float(rng.uniform(-DC_DRIFT_HZ, DC_DRIFT_HZ))
    phi = float(rng.uniform(0, 2 * np.pi))
    t = np.arange(iq.size) / rate
    dc = amp * np.exp(1j * (2 * np.pi * f_drift * t + phi))
    return (iq + dc.astype(np.complex64)).astype(np.complex64), True, amp


# ---------------------------------------------------------------------------
# 中心外スプリアス線の注入（M2.5: 実測アライン第2周）
# ---------------------------------------------------------------------------
# M2 スポットライト画像の人間確認で、画面中央**ではない**位置（例 2413.5/2432.1MHz）に
# 細い持続線（実在のスプリアス or 狭帯域 CW、DC ではない）が見つかった。これが M2 で
# 再定義した「cw-tone=中心外の細線」に合致し、残存 cw-tone 誤予測の一因になりうる。
# 対策: DC と同じ思想で、中心外の弱い持続 CW 線も **全クラスに無相関で** 注入し、
# 「単独の細線が在るだけでは cw-tone と断定しない（主たる構造で判断する）」を学べる
# ようにする。注入は IQ(時間領域)で行い凍結 spec.render を通す（画像描き込み禁止）。
# 事前確認4の実測（中心外 narrow prominence: 中央値3.0 / p75 3.6 / max10.7dB、強い線は
# 6/64 記録）に較正。amp→prominence: 0.05->2.5 / 0.07->3.9 / 0.10->6.0 / 0.15->8.9dB。
SPUR_INJECT_PROB = 0.3                 # 注入確率（小さめ 0.2〜0.4）
SPUR_AMP_MAIN = (0.04, 0.08)           # 主分布 → prominence ~ [2, 4.5]dB（実測中央付近）
SPUR_AMP_TAIL = (0.10, 0.18)           # 強テール → prominence ~ [6, 10.5]dB（実測max対応）
SPUR_TAIL_PROB = 0.25                  # 強い線に入る確率（全体の ~7.5% ≒ 実測 6/64）
SPUR_OFF_MIN = 0.8e6                   # 中心から十分離す（DC帯・cw下限0.6MHzより外）
SPUR_PROM_CAL_RANGE = (3.0, 3.6)       # 報告用: 実測 median..p75


def _inject_spurious_line(iq: np.ndarray, rate: float, rng: np.random.Generator):
    """中心外の弱い持続 CW スプリアス線を確率 SPUR_INJECT_PROB で注入する。

    cw-tone の主信号（強い中心外トーン）とは違い **弱い**（実測 prominence に較正）。
    クラスに依らず同一分布から引く（=識別の手掛かりにならない）。
    returns: (iq, injected: bool, amp: float|None)
    """
    if rng.random() >= SPUR_INJECT_PROB:
        return iq.astype(np.complex64), False, None
    if rng.random() < SPUR_TAIL_PROB:
        amp = float(rng.uniform(*SPUR_AMP_TAIL))
    else:
        amp = float(rng.uniform(*SPUR_AMP_MAIN))
    mag = float(rng.uniform(SPUR_OFF_MIN, 0.42 * rate))
    off = mag if rng.random() < 0.5 else -mag
    phi = float(rng.uniform(0, 2 * np.pi))
    t = np.arange(iq.size) / rate
    line = amp * np.exp(1j * (2 * np.pi * off * t + phi))
    return (iq + line.astype(np.complex64)).astype(np.complex64), True, amp


# ---------------------------------------------------------------------------
# 時間エンベロープ（pulse-radar=周期 / wideband-ofdm=非周期 を明確に分離）
# ---------------------------------------------------------------------------
def _periodic_pulse_envelope(n: int, rng: np.random.Generator):
    """pulse-radar 用: 厳密に周期的・短パルス・一様幅。returns (env, starts, lengths)。"""
    npulse = int(rng.integers(6, 14))
    pri = max(1, n // npulse)                  # 一定の繰返し間隔
    plen = max(64, int(pri * float(rng.uniform(0.06, 0.14))))   # 一様幅
    phase0 = int(rng.integers(0, pri))
    env = np.zeros(n, dtype=np.float32)
    starts: list[int] = []
    k = 0
    while True:
        s = phase0 + k * pri
        if s >= n:
            break
        env[s:min(s + plen, n)] = 1.0
        starts.append(s)
        k += 1
    return env, starts, [plen] * len(starts)


def _irregular_burst_envelope(n: int, rng: np.random.Generator):
    """wideband-ofdm 用（M2.5: 実 WiFi 画像へ再アライン）: 非周期の広帯域縦縞列。

    M2 の「窓全体への均等散布」をやめ、画像確認の 2 パターンを反映する:
      モード(i)  少数(2〜4 本) + 長い沈黙（ランダム位置）
      モード(ii) 密集クラスタ（窓の一部に固まる）+ 死んだ時間（無信号区間）
    縦縞 1 本の幅は窓の 3〜6%（実測 0.4〜0.8ms / 13ms 窓に対応。spec.render が時間軸を
    256 列に正規化するので絶対 ms ではなく窓に対する割合で合わせる）。
    pulse-radar（厳密周期・短パルス）とは **時間構造** で弁別（こちらは非周期）。
    returns (env, starts, lengths)。
    """
    env = np.zeros(n, dtype=np.float32)
    starts: list[int] = []
    lengths: list[int] = []

    def _width() -> int:
        return max(256, int(float(rng.uniform(0.03, 0.06)) * n))   # 縦縞幅 3〜6%

    def _add(pos: int, blen: int):
        pos = int(max(0, min(pos, n - 1)))
        end = min(pos + blen, n)
        env[pos:end] = 1.0
        starts.append(pos)
        lengths.append(end - pos)

    if rng.random() < 0.5:
        # モード(i): 少数 + 長い沈黙。
        k = int(rng.integers(2, 5))               # 2〜4 本
        for _ in range(k):
            bl = _width()
            _add(int(rng.integers(0, max(1, n - bl))), bl)
    else:
        # モード(ii): 密集クラスタ + 死んだ時間。クラスタは窓の一部(18〜40%)に固まる。
        k = int(rng.integers(3, 7))               # 3〜6 本
        span = max(2 * _width(), int(float(rng.uniform(0.18, 0.40)) * n))
        cstart = int(rng.integers(0, max(1, n - span)))
        for _ in range(k):
            bl = _width()
            _add(cstart + int(rng.integers(0, max(1, span - bl))), bl)

    order = np.argsort(starts)
    starts = [int(starts[i]) for i in order]
    lengths = [int(lengths[i]) for i in order]
    return env, starts, lengths


def _short_sparse_burst_envelope(n: int, rng: np.random.Generator,
                                 n_range: tuple = (1, 4),
                                 len_frac: tuple = (0.025, 0.05)):
    """narrowband-burst 用（M2.5: 実 BLE 画像へ再アライン）: 短いバースト 1〜3 発・疎ら。

    実 BLE は 1 発 ≈0.3〜0.6ms（窓の ~3%）と短く、窓に 1〜3 発だけ。M2 の長め(20〜40%)
    1 発から実測へ寄せる。returns (env, starts, lengths)。
    """
    nburst = int(rng.integers(*n_range))          # 1〜3 発
    env = np.zeros(n, dtype=np.float32)
    starts: list[int] = []
    lengths: list[int] = []
    for _ in range(nburst):
        bl = max(128, int(float(rng.uniform(*len_frac)) * n))
        s = int(rng.integers(0, max(1, n - bl)))
        env[s:s + bl] = 1.0
        starts.append(s)
        lengths.append(bl)
    order = np.argsort(starts)
    starts = [int(starts[i]) for i in order]
    lengths = [int(lengths[i]) for i in order]
    return env, starts, lengths


# ---------------------------------------------------------------------------
# クラス別ジェネレータ: (iq, info) を返す。info は annotation 用の off/bw/snr。
# off=None は「帯域全体 or 信号なし」（freq edges を付けない）。
# DC 残留は generate() で全クラス一律に注入する（クラスと無相関）。
# ---------------------------------------------------------------------------
def _gen_wideband_ofdm(rng: np.random.Generator):
    """実WiFi様（M2.5再アライン）: ほぼ全幅(≈18〜19.6MHz)・非周期の広帯域縦縞列。

    画像確認: 実 WiFi は **キャプチャ全幅(20MHz, 画面上端〜下端)を埋める**（M2 の
    12〜16MHz は狭すぎた）。縦縞 1 本は窓の 3〜6%、時間配置は「少数+長沈黙」か
    「密集クラスタ+死時間」の 2 モード。帯域は pulse-radar と同程度だが、弁別は
    **時間構造**（非周期 vs 厳密周期）に委ねる（pulse-radar 定義は不変）。
    """
    snr = float(rng.uniform(15, 28))
    bw = float(rng.uniform(0.90, 0.98) * RATE)        # ほぼ全幅（18.0〜19.6MHz）
    off = float(rng.uniform(-0.02, 0.02) * RATE)      # 中心付近（全幅なので僅か）
    wide = _bandlimited(N, RATE, off, bw, rng)
    env, starts, lengths = _irregular_burst_envelope(N, rng)
    # バースト間の濃淡: 縦縞毎に振幅をふらつかせる（実WiFiの可変電力を模す）。
    gain = env.copy()
    for s, l in zip(starts, lengths):
        gain[s:s + l] *= float(rng.uniform(0.6, 1.0))
    iq = _complex_noise(N, rng) + _amp(snr) * (wide * gain)
    return iq.astype(np.complex64), dict(off=off, bw=bw, snr=snr)


def _gen_narrowband_burst(rng: np.random.Generator):
    """実BLE様（M2.5再アライン）: 狭帯域(1〜2.5MHz維持)・短いバースト 1〜3 発・疎ら。

    画像確認: 実 BLE は帯域 ≈2MHz(M2 で妥当・維持)だが 1 発 ≈0.3〜0.6ms と短く窓に
    1〜3 発だけ。M2 の長め(20〜40%)1 発から実測へ寄せ、在窓 BLE が cw-tone でなく
    narrowband-burst に分類されることを狙う。
    """
    snr = float(rng.uniform(16, 30))
    bw = float(rng.uniform(1.0e6, 2.5e6))             # 帯域は維持
    off = float(rng.uniform(-0.35, 0.35) * RATE)
    sig = _bandlimited(N, RATE, off, bw, rng)
    env, _starts, _lengths = _short_sparse_burst_envelope(N, rng)
    iq = _complex_noise(N, rng) + _amp(snr) * (sig * env)
    return iq.astype(np.complex64), dict(off=off, bw=bw, snr=snr)


def _gen_cw_tone(rng: np.random.Generator):
    """連続波トーン（M2制約）: 中心から |offset| > 0.5MHz に置く。

    注入される DC 残留線（中心0Hzの細線）と重ならないようにし、「中心の細線=
    アーティファクト / 中心外の細線=cw-tone」を CNN が学べるようにする。
    """
    snr = float(rng.uniform(18, 32))
    mag = float(rng.uniform(0.6e6, 0.35 * RATE))      # |off| >= 0.6MHz > 0.5MHz
    off = mag if rng.random() < 0.5 else -mag
    iq = _complex_noise(N, rng) + _amp(snr) * _cw(N, RATE, off)
    return iq.astype(np.complex64), dict(off=off, bw=2e4, snr=snr)


def _gen_pulse_radar(rng: np.random.Generator):
    """レーダ様: 厳密に周期的・短パルス・一様幅の広帯域縦縞（wideband-ofdm と対比）。"""
    snr = float(rng.uniform(18, 32))
    env, _starts, _lengths = _periodic_pulse_envelope(N, rng)
    # パルス中は広帯域（ほぼ全帯域）→ 周期的な縦縞になる。
    wide = _bandlimited(N, RATE, 0.0, RATE * 0.9, rng)
    iq = _complex_noise(N, rng) + _amp(snr) * (wide * env)
    return iq.astype(np.complex64), dict(off=0.0, bw=RATE * 0.9, snr=snr)


def _gen_noise_only(rng: np.random.Generator):
    iq = _complex_noise(N, rng)
    return iq.astype(np.complex64), dict(off=None, bw=0.0, snr=0.0)


_GENERATORS = {
    "wideband-ofdm": _gen_wideband_ofdm,
    "narrowband-burst": _gen_narrowband_burst,
    "cw-tone": _gen_cw_tone,
    "pulse-radar": _gen_pulse_radar,
    "noise-only": _gen_noise_only,
}


# ---------------------------------------------------------------------------
# 生成本体
# ---------------------------------------------------------------------------
def _annotation(cls: str, center_hz: float, info: dict) -> dict:
    """1 件の SigMF annotation（真実ラベル）。off=None なら freq edges を省く。"""
    ann = dict(label=cls, method="sim-truth", confidence=1.0,
               snr_db=round(info["snr"], 1),
               comment=f"synthetic ground-truth class={cls}")
    off = info.get("off")
    if off is not None:
        bw = max(float(info.get("bw", 0.0)), 1.0)
        ann["freq_lower_edge"] = center_hz + off - bw / 2
        ann["freq_upper_edge"] = center_hz + off + bw / 2
    return ann


def generate(out_dir: str, per_class: int = 80, seed: int = 0,
             class_names: list[str] | None = None,
             verbose: bool = False) -> list[str]:
    """合成 SigMF データセットを生成して書き出す。

    out_dir       : 出力先（無ければ作成）
    per_class     : 1 クラスあたりの件数（クラス均衡）
    seed          : 生成シード（再現可能）
    class_names   : 生成するクラス（既定: classes.CLASSES の 5 クラス全部）
    returns       : 書き出したベースパス一覧（拡張子なし）
    """
    cls_list = list(class_names) if class_names else list(classes.CLASSES)
    for c in cls_list:
        if c not in _GENERATORS:
            raise ValueError(f"未知クラス: {c}（既知: {sorted(_GENERATORS)}）")

    os.makedirs(out_dir, exist_ok=True)

    # SeedSequence で各ファイルに独立かつ再現可能な乱数ストリームを割り当てる。
    total = len(cls_list) * per_class
    children = np.random.SeedSequence(seed).spawn(total)

    bases: list[str] = []
    k = 0
    for cls in cls_list:
        for i in range(per_class):
            rng = np.random.default_rng(children[k])
            k += 1
            iq, info = _GENERATORS[cls](rng)
            # DC 残留線を全クラス一律（同一分布・クラス無相関）に注入する。
            iq, dc_injected, dc_amp = _inject_dc_residual(iq, RATE, rng)
            # 中心外スプリアス線も全クラス一律（クラス無相関）に注入する（M2.5）。
            iq, spur_injected, spur_amp = _inject_spurious_line(iq, RATE, rng)
            # 中心周波数は 1〜6GHz から **クラスと独立に** 抽選する。
            # 「CNN は方式(見え方)を学び、用途は周波数で後段が導く」設計に従い、
            # クラスと周波数を相関させない（火入れの誠実さ）。
            center_hz = float(rng.uniform(1e9, 6e9))
            base = os.path.join(out_dir, f"{cls}_{i:04d}")
            extra_global = {
                "sigscan:rep_version": classes.REP_VERSION,
                "sigscan:true_class": cls,           # 真実ラベル（global 冗長記録）
                "sigscan:synthetic_only": classes.SYNTHETIC_ONLY_TAG,
                "sigscan:gen_seed": int(seed),
                "sigscan:gen_index": int(k - 1),
                "sigscan:dc_injected": bool(dc_injected),       # M2: DC残留注入の有無
                "sigscan:spur_injected": bool(spur_injected),   # M2.5: 中心外スプリアス
            }
            if dc_amp is not None:
                extra_global["sigscan:dc_amp"] = round(float(dc_amp), 4)
            if spur_amp is not None:
                extra_global["sigscan:spur_amp"] = round(float(spur_amp), 4)
            sigmf_io.write_recording(
                base, iq, center_hz=center_hz, sample_rate=RATE,
                annotations=[_annotation(cls, center_hz, info)],
                hw=HW_SYNTHETIC, recorder="sigscan-cnntrain",
                description=f"synthetic {cls} for CNN fire-test",
                extra_global=extra_global,
            )
            bases.append(base)
            if verbose:
                print(f"  [{k:4d}/{total}] {os.path.basename(base)}  "
                      f"SNR={info['snr']:4.1f}dB  center={center_hz/1e6:7.1f}MHz")
    return bases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _force_utf8():
    # Windows cp932 等で日本語/Unicode を print してもクラッシュしないように。
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8()
    import argparse
    p = argparse.ArgumentParser(
        prog="cnntrain.simgen",
        description="CNN 火入れ用の合成 SigMF データを生成（方式軸 5 クラス・均衡）")
    p.add_argument("--out", required=True, help="出力ディレクトリ")
    p.add_argument("--per-class", type=int, default=80, dest="per_class",
                   help="1 クラスあたりの件数（既定 80）")
    p.add_argument("--seed", type=int, default=0, help="生成シード（既定 0）")
    p.add_argument("--classes", nargs="*", default=None,
                   help=f"生成クラス（既定: 全 {len(classes.CLASSES)} クラス）")
    p.add_argument("-q", "--quiet", action="store_true", help="件ごとの表示を抑制")
    args = p.parse_args(argv)

    cls_list = args.classes or list(classes.CLASSES)
    print("=" * 72)
    print("  cnntrain.simgen — 合成データ生成（SYNTHETIC-ONLY）")
    print("=" * 72)
    for line in classes.SYNTHETIC_ONLY_LINES:
        print("  !! " + line)
    print("-" * 72)
    print(f"  出力     : {args.out}")
    print(f"  クラス   : {', '.join(cls_list)}")
    print(f"  per-class: {args.per_class}  seed: {args.seed}")
    print(f"  rate     : {RATE/1e6:.1f} MS/s   N: {N} samples/file")
    print("-" * 72)

    bases = generate(args.out, per_class=args.per_class, seed=args.seed,
                     class_names=args.classes, verbose=not args.quiet)
    print("-" * 72)
    print(f"  完了: {len(bases)} 件（= {len(cls_list)} クラス × {args.per_class}）")
    print(f"  各クラスの『見え方』:")
    for c in cls_list:
        print(f"    - {c:18s}: {classes.look_of(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
