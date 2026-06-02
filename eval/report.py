"""eval-harness (3/3): 推論レポート（合成限定・ドメインギャップ未測定）。

与えた SigMF 群に外部モデル（または reference stand-in）の推論を回し、sigscan の
ルールラベルとの **対応表（クロス集計）** を出力する。

重要な誠実さの制約（CONTRACT.md §4）:
  * いま手元にあるのは合成(sim)キャプチャのみ。よって結果は必ず
    "synthetic-vs-synthetic（本当のギャップではない）" と明示する。
  * 出力には hw（sim/real）内訳と、stand-in 使用時のバナーを必ず付ける。
  * 外部モデルのクラス空間は sigscan のバンドラベルと一致しないため、ここで出すのは
    「一致率」ではなく **対応表**。安易な accuracy は出さない（誤読防止）。

CLI:
    python -m eval.report captures/                       # 既定: reference-standin
    python -m eval.report captures/ --model qoherent-segmentation --allow-standin
    python -m eval.report captures/ --hw sim --limit 50
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import sigmf_io
import spec
from dataset import load_index, Record
from eval import adapters
from eval import loaders
from eval.loaders import LoadedModel, ModelUnavailable


# ---------------------------------------------------------------------------
# 1 件ぶんの推論結果
# ---------------------------------------------------------------------------
@dataclass
class Inference:
    path: str
    hw_group: str               # sim / real / other
    rule_label: str             # sigscan のルールラベル（弱教師）
    ext_pred: str               # 外部モデル/stand-in の予測クラス名
    extra: dict = field(default_factory=dict)


@dataclass
class Report:
    model_summary: str
    is_stand_in: bool
    n_total: int
    n_inferred: int
    hw_counts: dict              # {sim: n, real: n, ...}
    crosstab: dict               # {rule_label: {ext_pred: count}}
    rule_labels: list            # 行ラベル順
    ext_labels: list             # 列ラベル順
    errors: list = field(default_factory=list)   # (path, message)


# ---------------------------------------------------------------------------
# 推論ループ
# ---------------------------------------------------------------------------
def _infer_record(model: LoadedModel, rec: Record):
    """1 レコードを read_recording → spec.render → adapt → predict。"""
    # read_recording はロケール既定で meta を開く（凍結契約の符号化に一致）。
    iq, meta = sigmf_io.read_recording(rec.path)
    rate = float(meta.get("global", {}).get("core:sample_rate",
                                             spec.CAPTURE_RATE_HZ))
    x = adapters.adapt_for(model.spec, iq=iq, rate=rate)
    out = model.predict(x)
    return out


def run_report(sigmf_dir: str, model_name: str = "reference-standin",
               allow_standin: bool = False, hw: str | None = None,
               limit: int | None = None, **load_kwargs) -> Report:
    """SigMF 群に推論を回し Report を返す（合成限定・対応表）。"""
    ds = load_index(sigmf_dir)
    if hw:
        ds = ds.query(hw=hw)
    records = list(ds)
    if limit is not None:
        records = records[:limit]

    model = loaders.load_model(model_name, allow_standin=allow_standin,
                               **load_kwargs)

    hw_counts: dict[str, int] = {}
    crosstab: dict[str, dict[str, int]] = {}
    rule_labels: list[str] = []
    ext_labels: list[str] = []
    inferred = 0
    errors: list[tuple[str, str]] = []

    for rec in records:
        hw_counts[rec.hw_group] = hw_counts.get(rec.hw_group, 0) + 1
        try:
            out = _infer_record(model, rec)
        except Exception as e:  # noqa: BLE001 - 1件の失敗で全体を止めない
            errors.append((rec.path, f"{e.__class__.__name__}: {e}"))
            continue
        inferred += 1
        rl = rec.label or "(ラベルなし)"
        ep = str(out.get("pred_name", "?"))
        if rl not in crosstab:
            crosstab[rl] = {}
            rule_labels.append(rl)
        crosstab[rl][ep] = crosstab[rl].get(ep, 0) + 1
        if ep not in ext_labels:
            ext_labels.append(ep)

    rule_labels.sort()
    ext_labels.sort()
    return Report(
        model_summary=model.spec.summary(),
        is_stand_in=model.is_stand_in,
        n_total=len(records), n_inferred=inferred,
        hw_counts=hw_counts, crosstab=crosstab,
        rule_labels=rule_labels, ext_labels=ext_labels, errors=errors,
    )


# ---------------------------------------------------------------------------
# 出力（バナー必須）
# ---------------------------------------------------------------------------
def _banner(rep: Report, out) -> None:
    line = "=" * 72
    print(line, file=out)
    print("  sigscan eval-harness レポート", file=out)
    print(line, file=out)
    # hw 内訳
    hw_str = ", ".join(f"{k}={v}" for k, v in sorted(rep.hw_counts.items())) \
        or "(なし)"
    print(f"  hw 内訳: {hw_str}", file=out)
    only_sim = set(rep.hw_counts) <= {"sim"} and rep.hw_counts
    # 合成限定バナー（必須）
    print("  " + "!" * 68, file=out)
    if only_sim:
        print("  !! SYNTHETIC-ONLY: 入力はすべて sim（合成）。", file=out)
        print("  !! 結果は synthetic-vs-synthetic — **本当のドメインギャップは未測定**。",
              file=out)
    else:
        print("  !! 注意: real を含むが、本 M1 は配線検証。ドメインギャップは別途測定。",
              file=out)
    print("  !! 実測評価への移行手順は eval/README.md を参照。", file=out)
    # stand-in バナー（必須・該当時）
    if rep.is_stand_in:
        print("  !! STAND-IN MODEL: 外部学習済みモデルではなく numpy スタンドイン。",
              file=out)
        print("  !! 出力は占有度ベースの粗い分類で、変調種別(5G/LTE等)は判定しない。",
              file=out)
        print("  !! → これは『配線が通ること』の確認であり、外部モデルの性能ではない。",
              file=out)
    print("  " + "!" * 68, file=out)
    print(f"  model : {rep.model_summary}", file=out)
    print(f"  対象  : {rep.n_total} records / 推論成功 {rep.n_inferred}", file=out)
    print(line, file=out)


def print_report(rep: Report, file=None) -> None:
    out = file or sys.stdout
    _banner(rep, out)

    if not rep.crosstab:
        print("\n(推論できたレコードがありません)", file=out)
        if rep.errors:
            _print_errors(rep, out)
        return

    print("\n[対応表]  行 = sigscan ルールラベル(弱教師)  ×  列 = 外部予測", file=out)
    print("  ※ クラス空間が異なるため『一致率』ではなく対応の集計。\n", file=out)

    # 列ヘッダ
    col_w = max(10, *(len(c) for c in rep.ext_labels)) if rep.ext_labels else 10
    col_w = min(col_w, 18)
    header = " " * 34 + "".join(f"{c[:col_w]:>{col_w+2}}" for c in rep.ext_labels)
    print(header, file=out)
    for rl in rep.rule_labels:
        row = rep.crosstab.get(rl, {})
        cells = "".join(f"{row.get(c, 0):>{col_w+2}}" for c in rep.ext_labels)
        print(f"  {rl[:32]:<32}{cells}", file=out)

    # 列合計
    totals = {c: sum(rep.crosstab[rl].get(c, 0) for rl in rep.rule_labels)
              for c in rep.ext_labels}
    tot_cells = "".join(f"{totals[c]:>{col_w+2}}" for c in rep.ext_labels)
    print(f"  {'(列合計)':<32}{tot_cells}", file=out)

    if rep.errors:
        _print_errors(rep, out)

    print("\n[まとめ]", file=out)
    print("  * 外部予測の分布は上表の列合計どおり。", file=out)
    if rep.is_stand_in:
        print("  * これは stand-in による配線検証。外部モデル本体ではない。", file=out)
    print("  * sim 限定のため、実機との差（ドメインギャップ）は本レポートでは測れない。",
          file=out)


def _print_errors(rep: Report, out) -> None:
    print(f"\n[推論失敗 {len(rep.errors)} 件]", file=out)
    for path, msg in rep.errors[:10]:
        print(f"  - {path}: {msg}", file=out)
    if len(rep.errors) > 10:
        print(f"  ... 他 {len(rep.errors) - 10} 件", file=out)


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
        prog="eval.report",
        description="SigMF 群に外部モデル/stand-in の推論を回し対応表を出す（合成限定）")
    p.add_argument("dir", help="SigMF を格納したディレクトリ")
    p.add_argument("--model", default="reference-standin",
                   choices=loaders.available_models(),
                   help="使う外部モデル（既定: reference-standin）")
    p.add_argument("--allow-standin", action="store_true", dest="allow_standin",
                   help="外部モデルがロード不可なら stand-in に明示退避する")
    p.add_argument("--hw", default=None, help="hw フィルタ: sim/real/other")
    p.add_argument("--limit", type=int, default=None, help="先頭 N 件に限定")
    p.add_argument("--weights", default=None, help="外部モデルの重みパス")
    args = p.parse_args(argv)

    load_kwargs = {}
    if args.weights:
        load_kwargs["weights"] = args.weights
    try:
        rep = run_report(args.dir, model_name=args.model,
                         allow_standin=args.allow_standin, hw=args.hw,
                         limit=args.limit, **load_kwargs)
    except ModelUnavailable as e:
        print("モデルをロードできませんでした（--allow-standin で stand-in に退避可）:\n",
              file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2
    print_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
