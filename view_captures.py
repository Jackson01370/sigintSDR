"""保存済み SigMF キャプチャを画像(PNG)にして目で確認するためのビューア。

使い方（プロジェクト直下で）:
    python view_captures.py captures/                 # captures/ 内の全件を画像化
    python view_captures.py captures/ --out images/   # 出力先を指定（既定: <dir>/_images）
    python view_captures.py captures/ --limit 6       # 先頭6件だけ

各レコードにつき1枚の PNG を作る。中身は左に「スペクトログラム」（縦=周波数, 横=時間,
明るいほど強い）、右に「平均スペクトル」（その帯域のどこにエネルギーがあるか）。
画像の作り方は凍結契約 spec.render と同じ正準表現に合わせる（学習でAIが見るのと同じ絵）。

依存: numpy, matplotlib。プロジェクトの spec.py / sigmf_io.py をそのまま使う
（どちらも変更しない・読み取るだけ）。Windows(cp932)でも文字化けしないよう英字主体。
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np

import sigmf_io
import spec


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def _meta_summary(meta: dict) -> dict:
    """meta(JSON)から表示に使う情報を安全に取り出す。"""
    g = meta.get("global", {}) or {}
    anns = meta.get("annotations", []) or []
    a = anns[0] if anns else {}
    center = float(g.get("core:frequency", a.get("core:freq_lower_edge", 0.0)) or 0.0)
    # center は global に無い場合があるので annotation の上下端から復元
    lo = a.get("core:freq_lower_edge")
    hi = a.get("core:freq_upper_edge")
    if (not center) and lo is not None and hi is not None:
        center = (float(lo) + float(hi)) / 2.0
    return {
        "rate": float(g.get("core:sample_rate", spec.CAPTURE_RATE_HZ) or spec.CAPTURE_RATE_HZ),
        "hw": str(g.get("core:hw", "?")),
        "center": center,
        "label": str(a.get("core:description", a.get("label", "")) or ""),
        "confidence": a.get("sigscan:confidence"),
        "method": a.get("sigscan:method"),
        "snr_db": a.get("sigscan:snr_db"),
        "persistence": a.get("sigscan:persistence"),
        "spur": a.get("sigscan:spur_suspect"),
        "bw_hz": (float(hi) - float(lo)) if (lo is not None and hi is not None) else None,
    }


def render_one(base: str, out_path: str) -> dict:
    """1レコード(base.sigmf-*)を読み、PNG を1枚書き出す。要約 dict を返す。"""
    import matplotlib
    matplotlib.use("Agg")  # 画面なしで画像ファイルだけ作る
    import matplotlib.pyplot as plt

    iq, meta = sigmf_io.read_recording(base)
    info = _meta_summary(meta)
    rate = info["rate"]

    # 凍結契約と同じ正準スペクトログラム（縦256=周波数, 横256=時間, 0..1）
    img = spec.render(iq, rate)

    # 平均スペクトル（周波数ごとの平均パワー）。縦軸＝周波数に合わせるため転置側を平均。
    spectrum = img.mean(axis=1)
    freqs_mhz = (np.linspace(-rate / 2, rate / 2, img.shape[0]) + info["center"]) / 1e6

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2.4, 1]})

    # 左: スペクトログラム
    extent = [0, 1, freqs_mhz[0], freqs_mhz[-1]]
    axL.imshow(img, aspect="auto", origin="lower", cmap="viridis", extent=extent)
    axL.set_xlabel("time (normalized)")
    axL.set_ylabel("frequency (MHz)")
    cen = info["center"] / 1e6
    bw = (info["bw_hz"] / 1e6) if info["bw_hz"] else None
    title = f"{cen:.1f} MHz"
    if bw:
        title += f"  BW~{bw:.1f} MHz"
    if info["label"]:
        title += f"\n{info['label']}"
    axL.set_title(title, fontsize=10)

    # 右: 平均スペクトル（横=パワー, 縦=周波数）
    axR.plot(spectrum, freqs_mhz, color="#1f77b4")
    axR.set_xlabel("avg power (norm)")
    axR.set_ylim(freqs_mhz[0], freqs_mhz[-1])
    axR.grid(True, alpha=0.3)
    axR.set_title("mean spectrum", fontsize=10)

    # 下に出所・品質メタを注記
    foot = f"hw={info['hw']}"
    if info["snr_db"] is not None:
        foot += f"  SNR={info['snr_db']}dB"
    if info["persistence"] is not None:
        foot += f"  persist={info['persistence']}"
    if info["method"]:
        foot += f"  method={info['method']}"
    if info["spur"] is not None:
        foot += f"  spur_suspect={info['spur']}"
    fig.text(0.01, 0.005, foot, fontsize=8, color="#444")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return info


def main(argv=None) -> int:
    _force_utf8()
    p = argparse.ArgumentParser(
        description="保存済み SigMF キャプチャをスペクトログラム画像(PNG)にする")
    p.add_argument("dir", help="SigMF を格納したディレクトリ（例: captures/）")
    p.add_argument("--out", default=None, help="画像の出力先（既定: <dir>/_images）")
    p.add_argument("--limit", type=int, default=None, help="先頭から処理する件数")
    args = p.parse_args(argv)

    metas = sorted(glob.glob(os.path.join(args.dir, "*.sigmf-meta")))
    if not metas:
        print(f"SigMF が見つかりません: {args.dir}")
        return 1
    if args.limit:
        metas = metas[: args.limit]

    out_dir = args.out or os.path.join(args.dir, "_images")
    os.makedirs(out_dir, exist_ok=True)

    print(f"画像化: {len(metas)} 件 → {out_dir}")
    ok = 0
    for mp in metas:
        base = mp[: -len(".sigmf-meta")]
        name = os.path.basename(base)
        out_path = os.path.join(out_dir, name + ".png")
        try:
            info = render_one(base, out_path)
            tag = info["label"] or "(no label)"
            print(f"  OK  {name}.png   {info['center']/1e6:.1f}MHz  {tag}")
            ok += 1
        except Exception as e:  # noqa: BLE001 - 1件失敗で全体を止めない
            print(f"  NG  {name}: {e.__class__.__name__}: {e}")

    print(f"\n完了: {ok}/{len(metas)} 枚を {out_dir} に保存しました。")
    print("画像をエクスプローラーで開いて確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
