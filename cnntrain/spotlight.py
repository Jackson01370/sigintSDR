"""cnntrain (M2): スポットライト — イレギュラーな電波を目立つ場所に集める発見の漏斗。

ユーザー発案。プローブ（推論のみ）の結果から「人手で見るべき候補」を 2 タイプ抽出し、
captures/_images/_spotlight/ に PNG（spec.render 経由＝view_captures と同等の見た目）と
spotlight_report.txt を出力する。**自動隔離はしない**（隔離・ラベル修正の判断は人間）。

2 タイプ（いずれも CLI で閾値・top-N を調整可）:
  * タイプ1（自白型）: ルールラベルが空/未識別、または rule confidence < 閾値
    （既定 0.5）。注意: 現収集物は rule conf≈0.62 が大量のため、閾値 0.7 だと洪水。
    既定を 0.5 にして「本当に自信のないものだけ」拾う（設計判断）。
  * タイプ2（監査型）: 期待対応表[仮説]との不一致 かつ CNN confidence ≥ 閾値
    （既定 0.9）。確信度降順。電子レンジ誤ラベル発見がこの原型。
両タイプとも top-N（既定 15）で上限し洪水を防ぐ。

原則: 実データは読み取り専用（*.sigmf-* は触らない）。PNG は新規レンダのみ。
凍結 spec.render を迂回しない（view_captures.render_one を再利用）。

CLI:
    python -m cnntrain.spotlight --data captures/ --checkpoint runs/m2/checkpoint.pt \
        --out captures/_images/_spotlight/ [--type1-conf 0.5 --type2-cnn 0.9 --top-n 15]
"""
from __future__ import annotations

import json
import os
import sys

import spec
from cnntrain import probe


SPOTLIGHT_HEADER = [
    "ここは『発見の漏斗(funnel)』。スポットライトは人手で見るべき候補を集めるだけ。",
    "隔離・ラベル修正の判断は人間が行う（自動隔離・自動上書きはしない）。",
    "SIM-TRAINED MODEL / REAL-DATA PROBE 由来。一致は accuracy ではない（軸が違う照合）。",
]


def _is_empty_label(label: str | None) -> bool:
    if not label:
        return True
    s = label.strip().lower()
    return s in ("", "(ラベルなし)", "(none)", "unknown", "unidentified", "未識別")


def select_spotlight(records, type1_conf_thr: float = 0.5,
                     type2_cnn_thr: float = 0.9, top_n: int = 15) -> dict:
    """ProbeRecord 群から タイプ1/タイプ2 候補を選ぶ（各 top_n 上限）。

    returns: dict(type1=[ProbeRecord...], type2=[ProbeRecord...])
    """
    # タイプ1（自白型）: ラベル空 or rule conf < 閾値。確信度の低い順（怪しい順）。
    type1 = [r for r in records
             if _is_empty_label(r.label)
             or (r.rule_confidence is not None and r.rule_confidence < type1_conf_thr)]
    type1.sort(key=lambda r: (r.rule_confidence if r.rule_confidence is not None
                              else -1.0, r.file))
    type1 = type1[:top_n]

    # タイプ2（監査型）: 期待と不一致 かつ CNN 高確信。確信度降順。
    type2 = [r for r in records
             if r.matched is False and r.confidence >= type2_cnn_thr]
    type2.sort(key=lambda r: (-r.confidence, r.file))
    type2 = type2[:top_n]
    return dict(type1=type1, type2=type2)


def _reason_type1(r, thr: float) -> str:
    if _is_empty_label(r.label):
        return "自白型: ルールラベルが空/未識別 → 人手で正体確認"
    rc = r.rule_confidence
    return (f"自白型: ルール '{r.label}' の確信度 {rc:.2f} < {thr} "
            f"→ 弱教師として怪しい、人手確認候補")


def _reason_type2(r, thr: float) -> str:
    exp = "/".join(r.expected) if r.expected else "-"
    return (f"監査型: 期待[仮説]={exp} と不一致なのに CNN '{r.pred}' を高確信"
            f"({r.confidence:.2f}≥{thr}) → ラベル誤り/未知信号の候補"
            f"（電子レンジ発見の原型）")


def _render_png(base_path: str, out_path: str) -> bool:
    """view_captures.render_one を再利用して spec.render 経由の PNG を書く。

    実データ(*.sigmf-*)は読み取りのみ。matplotlib 不在なら False。
    """
    try:
        import view_captures
        view_captures.render_one(base_path, out_path)
        return True
    except Exception:
        return False


def run_spotlight(data_dir: str, ckpt_path: str, out_dir: str,
                  type1_conf_thr: float = 0.5, type2_cnn_thr: float = 0.9,
                  top_n: int = 15, render: bool = True,
                  probe_result=None) -> dict:
    """プローブ → 候補抽出 → PNG + レポート出力。returns サマリ dict。

    probe_result を渡せばプローブを再実行しない（CLI 連携用）。
    """
    res = probe_result or probe.run_probe(data_dir, ckpt_path, top_n=top_n)
    sel = select_spotlight(res.records, type1_conf_thr=type1_conf_thr,
                           type2_cnn_thr=type2_cnn_thr, top_n=top_n)
    os.makedirs(out_dir, exist_ok=True)

    # 候補（ユニークなファイル）を PNG 化（重複レンダ回避）。
    entries = []
    for r in sel["type1"]:
        entries.append(("type1", r, _reason_type1(r, type1_conf_thr)))
    for r in sel["type2"]:
        entries.append(("type2", r, _reason_type2(r, type2_cnn_thr)))

    rendered: dict[str, bool] = {}
    if render:
        for _typ, r, _reason in entries:
            if r.file in rendered:
                continue
            base = os.path.join(data_dir, r.file)         # 読み取りのみ
            png = os.path.join(out_dir, r.file + ".png")
            rendered[r.file] = _render_png(base, png)

    txt_path, json_path = _write_report(out_dir, res, sel, entries, rendered,
                                        type1_conf_thr, type2_cnn_thr, top_n,
                                        render)
    return dict(n_type1=len(sel["type1"]), n_type2=len(sel["type2"]),
                report_txt=txt_path, report_json=json_path,
                rendered=sum(1 for v in rendered.values() if v),
                out_dir=out_dir)


def _write_report(out_dir, res, sel, entries, rendered,
                  t1_thr, t2_thr, top_n, render) -> tuple[str, str]:
    line = "=" * 76
    L = [line, "  cnntrain スポットライト（発見の漏斗）", line, "  " + "!" * 72]
    for s in SPOTLIGHT_HEADER:
        L.append("  !! " + s)
    L.append("  " + "!" * 72)
    m = res.ckpt_meta
    L.append(f"  checkpoint : {res.checkpoint}  (SYNTHETIC-ONLY 学習, "
             f"sim_val={m.get('final_val_acc','-')})")
    L.append(f"  data       : {res.data_dir}   records={res.n_total}")
    L.append(f"  閾値       : タイプ1 rule_conf<{t1_thr}（既定0.5: conf≈0.62 の洪水回避）"
             f"  / タイプ2 CNN_conf≥{t2_thr}   top-N={top_n}")
    L.append(line)

    L.append("")
    L.append(f"[タイプ1 自白型]  {len(sel['type1'])} 件（ルールが自信なし/未識別）")
    if not sel["type1"]:
        L.append("  (該当なし — 現収集物は rule conf がおおむね閾値以上)")
    for r in sel["type1"]:
        rc = f"{r.rule_confidence:.2f}" if r.rule_confidence is not None else " - "
        L.append(f"  {r.center_mhz:8.1f}MHz  rule='{r.label}'(conf={rc})  "
                 f"CNN={r.pred}({r.confidence:.2f})  {r.file}")
        L.append(f"      理由: {_reason_type1(r, t1_thr)}")

    L.append("")
    L.append(f"[タイプ2 監査型]  {len(sel['type2'])} 件（期待[仮説]不一致 × CNN高確信）")
    if not sel["type2"]:
        L.append("  (該当なし)")
    for r in sel["type2"]:
        exp = "/".join(r.expected) if r.expected else "-"
        L.append(f"  {r.center_mhz:8.1f}MHz  rule='{r.label}'  期待={exp}  "
                 f"CNN={r.pred}({r.confidence:.2f})  {r.file}")
        L.append(f"      理由: {_reason_type2(r, t2_thr)}")

    if render:
        ok = sum(1 for v in rendered.values() if v)
        L.append("")
        L.append(f"[PNG]  {ok}/{len(rendered)} 枚を {out_dir} に出力（spec.render 経由・"
                 "view_captures と同等）。元レコードは不変。")
    L.append("")
    L.append("  次アクション（人間）: PNG を確認し、隔離(_review_pending/)・ラベル修正を判断。")

    txt = "\n".join(L)
    txt_path = os.path.join(out_dir, "spotlight_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    def _entry_json(typ, r, reason):
        return dict(type=typ, file=r.file, center_mhz=r.center_mhz,
                    rule_label=r.label, rule_confidence=r.rule_confidence,
                    cnn_pred=r.pred, cnn_confidence=r.confidence,
                    expected=r.expected, reason=reason,
                    png=(r.file + ".png") if rendered.get(r.file) else None)
    payload = dict(
        funnel="discovery only; human decides isolation/relabel (no auto-isolation)",
        match_is_not_accuracy=True,
        checkpoint=res.checkpoint, data_dir=res.data_dir, n_total=res.n_total,
        thresholds=dict(type1_rule_conf_lt=t1_thr, type2_cnn_conf_ge=t2_thr,
                        top_n=top_n),
        type1=[_entry_json("type1", r, _reason_type1(r, t1_thr))
               for r in sel["type1"]],
        type2=[_entry_json("type2", r, _reason_type2(r, t2_thr))
               for r in sel["type2"]],
    )
    json_path = os.path.join(out_dir, "spotlight_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return txt_path, json_path


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8()
    import argparse
    p = argparse.ArgumentParser(
        prog="cnntrain.spotlight",
        description="イレギュラーな電波を集める発見の漏斗（推論のみ・自動隔離なし）")
    p.add_argument("--data", required=True, help="SigMF データディレクトリ（captures/）")
    p.add_argument("--checkpoint", required=True, help="チェックポイント(.pt)")
    p.add_argument("--out", required=True,
                   help="出力先（例: captures/_images/_spotlight/）")
    p.add_argument("--type1-conf", type=float, default=0.5, dest="t1",
                   help="タイプ1: rule confidence 下限（既定 0.5）")
    p.add_argument("--type2-cnn", type=float, default=0.9, dest="t2",
                   help="タイプ2: CNN confidence 閾値（既定 0.9）")
    p.add_argument("--top-n", type=int, default=15, dest="top_n",
                   help="各タイプの上限件数（既定 15）")
    p.add_argument("--no-render", action="store_true",
                   help="PNG を作らずレポートのみ")
    args = p.parse_args(argv)

    summary = run_spotlight(args.data, args.checkpoint, args.out,
                            type1_conf_thr=args.t1, type2_cnn_thr=args.t2,
                            top_n=args.top_n, render=not args.no_render)
    print(f"スポットライト: タイプ1={summary['n_type1']} 件 / "
          f"タイプ2={summary['n_type2']} 件 / PNG={summary['rendered']} 枚")
    print(f"  レポート: {summary['report_txt']}")
    print(f"           {summary['report_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
