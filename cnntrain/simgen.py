"""cnntrain (1/6): Sim データ生成 CLI（torch 非依存）。

5 つの **方式軸（見え方）クラス** の合成 IQ を numpy で直接合成し、凍結
`sigmf_io.write_recording` で SigMF として保存する。生成手段に SimBackend を
使わず numpy 直接合成を選んだ理由:
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
# クラス別ジェネレータ: (iq, info) を返す。info は annotation 用の off/bw/snr。
# off=None は「帯域全体 or 信号なし」（freq edges を付けない）。
# ---------------------------------------------------------------------------
def _gen_wideband_ofdm(rng: np.random.Generator):
    snr = float(rng.uniform(15, 28))
    bw = float(rng.uniform(12e6, 16e6))
    off = float(rng.uniform(-0.08, 0.08) * RATE)
    iq = _complex_noise(N, rng) + _amp(snr) * _bandlimited(N, RATE, off, bw, rng)
    return iq.astype(np.complex64), dict(off=off, bw=bw, snr=snr)


def _gen_narrowband_burst(rng: np.random.Generator):
    snr = float(rng.uniform(16, 30))
    bw = float(rng.uniform(1.0e6, 2.5e6))
    off = float(rng.uniform(-0.35, 0.35) * RATE)
    sig = _bandlimited(N, RATE, off, bw, rng)
    # 時間ゲート: 連続した 1 バースト（全長の 20〜40%）だけ点く。
    frac = float(rng.uniform(0.2, 0.4))
    blen = max(1, int(N * frac))
    start = int(rng.integers(0, max(1, N - blen)))
    env = np.zeros(N, dtype=np.float32)
    env[start:start + blen] = 1.0
    iq = _complex_noise(N, rng) + _amp(snr) * (sig * env)
    return iq.astype(np.complex64), dict(off=off, bw=bw, snr=snr)


def _gen_cw_tone(rng: np.random.Generator):
    snr = float(rng.uniform(18, 32))
    off = float(rng.uniform(-0.35, 0.35) * RATE)
    iq = _complex_noise(N, rng) + _amp(snr) * _cw(N, RATE, off)
    return iq.astype(np.complex64), dict(off=off, bw=2e4, snr=snr)


def _gen_pulse_radar(rng: np.random.Generator):
    snr = float(rng.uniform(18, 32))
    npulse = int(rng.integers(6, 14))
    pri = max(1, N // npulse)               # パルス繰返し間隔（サンプル）
    duty = float(rng.uniform(0.06, 0.14))
    plen = max(64, int(pri * duty))
    env = np.zeros(N, dtype=np.float32)
    phase0 = int(rng.integers(0, pri))
    for k in range(npulse + 1):
        s = phase0 + k * pri
        if s >= N:
            break
        env[s:min(s + plen, N)] = 1.0
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
            }
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
