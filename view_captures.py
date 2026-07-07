"""保存済み SigMF キャプチャを画像(PNG)にして目で確認するためのビューア。

使い方（プロジェクト直下で）:
    python view_captures.py captures/                 # captures/ 内の全件を画像化
    python view_captures.py captures/ --out images/   # 出力先を指定（既定: <dir>/_images）
    python view_captures.py captures/ --limit 6       # 先頭6件だけ

各レコードにつき1枚の PNG を作る。中身は左に「スペクトログラム」（縦=周波数, 横=時間,
明るいほど強い）、右に「平均スペクトル」（その帯域のどこにエネルギーがあるか）。
画像の作り方は凍結契約 spec.render と同じ正準表現に合わせる（学習でAIが見るのと同じ絵）。

依存: numpy, matplotlib。プロジェクトの spec.py / sigmf_io.py をそのまま使う
（どちらも変更しない・読み取るだけ）。用途ラベルは日本語をそのまま表示する
（コンソールは UTF-8 強制、PNG タイトルは和文フォントへのフォールバックで対応）。
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np

import dsp
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
    # center は IQ の物理中心。SigMF 標準では core:frequency は captures 要素に入る
    #   （sigmf_io.write_recording もそう書く）ため、まず captures[0] を見る。global 直下は
    #   旧データ互換のフォールバック、annotation 上下端からの復元は最後の手段。
    caps = meta.get("captures", []) or []
    c0 = caps[0] if caps else {}
    lo = a.get("core:freq_lower_edge")
    hi = a.get("core:freq_upper_edge")
    center = float(c0.get("core:frequency", 0.0) or 0.0)      # 第一: IQ の物理中心（正）
    if not center:
        center = float(g.get("core:frequency", 0.0) or 0.0)   # 第二: 旧データ互換（global保持）
    if (not center) and lo is not None and hi is not None:
        center = (float(lo) + float(hi)) / 2.0                # 第三: annotationから復元（最後の手段）
    return {
        "rate": float(g.get("core:sample_rate", spec.CAPTURE_RATE_HZ) or spec.CAPTURE_RATE_HZ),
        "hw": str(g.get("core:hw", "?")),
        "center": center,
        # 用途ラベル: sigmf_io.write_recording は annotation の core:label に書く
        "label": str(a.get("core:label", a.get("core:description", a.get("label", ""))) or ""),
        "confidence": a.get("sigscan:confidence"),
        "method": a.get("sigscan:method"),
        "snr_db": a.get("sigscan:snr_db"),
        "persistence": a.get("sigscan:persistence"),
        "spur": a.get("sigscan:spur_suspect"),
        "bw_hz": (float(hi) - float(lo)) if (lo is not None and hi is not None) else None,
        # 検出帯（annotation の下端/上端, Hz）。マーカー描画用。無ければ None。
        "det_lo": (float(lo) if lo is not None else None),
        "det_hi": (float(hi) if hi is not None else None),
        # CNN 監査来歴（M3 以降の記録のみ global に載る。古い記録は None）
        "cnn_verdict": g.get("sigscan:cnn_verdict"),
        "cnn_class": g.get("sigscan:cnn_class"),
        "cnn_conf": g.get("sigscan:cnn_conf"),
    }


def label_text(info: dict) -> str:
    """一覧行・PNG タイトルに出すラベル文字列を組み立てる（純関数）。

    _meta_summary の戻り値から「用途ラベル + CNN 監査来歴」を1行にする。
    例: `未識別信号  [CNN:C-conflict cnn=noise-only@0.47]`
    CNN 来歴の無い古い記録はラベルのみ、ラベルも無ければ "(no label)"
    （後方互換: どのキーが欠けてもエラーにしない）。
    """
    text = str(info.get("label") or "") or "(no label)"
    verdict = info.get("cnn_verdict")
    if not verdict:
        return text
    cnn = f"CNN:{verdict}"
    if info.get("cnn_class"):
        cnn += f" cnn={info['cnn_class']}"
        if info.get("cnn_conf") is not None:
            try:
                cnn += f"@{float(info['cnn_conf']):.2f}"
            except (TypeError, ValueError):
                pass
    return f"{text}  [{cnn}]"


def _title_font_family() -> list:
    """PNG タイトル用のフォント列。和文ラベルが豆腐(□)にならないための表示時対処。

    既定の DejaVu Sans を先頭に保ち（英数字は従来どおり）、インストール済みの
    和文フォントを1つだけ後ろに足す。matplotlib>=3.6 のグリフ単位フォールバックで
    和文グリフのみ和文フォントで描かれる。見つからなければ既定のまま（従来挙動）。
    """
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return ["sans-serif"]
    fam = [n for n in ("DejaVu Sans",) if n in installed]
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic",
                 "Noto Sans CJK JP", "IPAexGothic", "IPAGothic"):
        if name in installed:
            fam.append(name)
            break
    return fam or ["sans-serif"]


def render_one(base: str, out_path: str, flatten_dc: bool = False) -> dict:
    """1レコード(base.sigmf-*)を読み、PNG を1枚書き出す。要約 dict を返す。

    flatten_dc=True なら、画像化の前に時間変動DC追従のハイパス(dsp.remove_dc に rate を
    渡す)で中央のDCスパイク残留を平坦化する。表示時のみの処理で、保存済み生IQ
    (.sigmf-data)には一切触れない（凍結 spec.render もそのまま）。
    """
    import matplotlib
    matplotlib.use("Agg")  # 画面なしで画像ファイルだけ作る
    import matplotlib.pyplot as plt

    iq, meta = sigmf_io.read_recording(base)
    info = _meta_summary(meta)
    rate = info["rate"]

    if flatten_dc:
        # 既存キャプチャの中央DCスパイク残留を表示時に平坦化（生IQは不変）。
        iq = dsp.remove_dc(iq, rate=rate)

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
    # 保存の根拠になった検出帯[det_lo, det_hi]を半透明の横帯＋上下端の細線で重畳（軸=周波数）。
    #   「検出帯」と「画像の実体」のズレが一目で見える（dwell の混獲記録の発見用）。
    #   spec.render の画像ピクセルには一切触れない（目盛りと注記の重畳のみ）。
    det_lo = info.get("det_lo")
    det_hi = info.get("det_hi")
    if det_lo is not None and det_hi is not None:
        axL.axhspan(det_lo / 1e6, det_hi / 1e6, alpha=0.15, color="#ff7f0e")
        axL.axhline(det_lo / 1e6, color="#ff7f0e", lw=0.6, alpha=0.7)
        axL.axhline(det_hi / 1e6, color="#ff7f0e", lw=0.6, alpha=0.7)
    cen = info["center"] / 1e6
    bw = (info["bw_hz"] / 1e6) if info["bw_hz"] else None
    # タイトルは二本立て: IQ 中心(tuner)と検出中心(det=(lo+hi)/2)を併記。det の無い記録は
    #   従来様式（tuner のみ＋BW）。
    if det_lo is not None and det_hi is not None:
        det_cen = (det_lo + det_hi) / 2.0 / 1e6
        title = f"tuner {cen:.1f} MHz | det {det_cen:.1f} MHz"
    else:
        title = f"{cen:.1f} MHz"
    if bw:
        title += f"  BW~{bw:.1f} MHz"
    if info["label"] or info.get("cnn_verdict"):
        title += f"\n{label_text(info)}"
    axL.set_title(title, fontsize=10, fontfamily=_title_font_family())

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
    p.add_argument("--flatten-dc", action="store_true",
                   help="表示時に中央のDCスパイク残留を平坦化（時間変動DC追従ハイパス）。"
                        "保存済み生IQには触れない")
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
            info = render_one(base, out_path, flatten_dc=args.flatten_dc)
            tag = label_text(info)
            print(f"  OK  {name}.png   {info['center']/1e6:.1f}MHz  {tag}")
            ok += 1
        except Exception as e:  # noqa: BLE001 - 1件失敗で全体を止めない
            print(f"  NG  {name}: {e.__class__.__name__}: {e}")

    print(f"\n完了: {ok}/{len(metas)} 枚を {out_dir} に保存しました。")
    print("画像をエクスプローラーで開いて確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
