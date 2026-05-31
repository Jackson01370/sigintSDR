"""SigMF データセットの管理（収集物の一覧化・フィルタ・重複排除・分割・統計）。

`captures/` に貯まった `*.sigmf-meta` を走査して 1 レコード = 1 SigMF Recording
として索引化し、学習/評価で使える形に整える。依存は numpy のみ。

設計の要点:
  * `core:hw`（HackRF One / sigscan-sim (synthetic)）を最重要属性として保持する。
    合成と実機を無自覚に混ぜないための防壁（CONTRACT.md §2）。`split()` は hw 毎に
    独立して行い、sim と real を同一データ点として跨がせない。
  * バンド名は config.BAND_PLAN（分類器と同じ割当）から中心周波数で引く。

CLI:
    python3 -m dataset stats  captures/        # バンド/label/hw/SNR の内訳
    python3 -m dataset query  captures/ --hw sim --snr-min 20 --band WiFi
    python3 -m dataset review captures/         # 低信頼アノテーションの再ラベル
"""
from __future__ import annotations
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

from config import BAND_PLAN, Band

# 出所(hw)の正準グループ。混在禁止の判定はこの2群で行う。
HW_SIM_PREFIX = "sigscan-sim"
HW_REAL = "HackRF One"


def hw_group(hw: str) -> str:
    """core:hw 文字列を 'sim' / 'real' / 'other' のグループに正規化する。"""
    if not hw:
        return "other"
    if hw.startswith(HW_SIM_PREFIX) or "synthetic" in hw.lower():
        return "sim"
    if hw == HW_REAL or "hackrf" in hw.lower():
        return "real"
    return "other"


def band_for_center(center_hz: float, bands: list[Band] | None = None) -> str | None:
    """中心周波数が属するバンド名を返す（分類器 _match_band と同じ優先度規則）。"""
    bands = BAND_PLAN if bands is None else bands
    best: Band | None = None
    for b in bands:
        if b.f_lo <= center_hz <= b.f_hi:
            if best is None or b.priority > best.priority:
                best = b
    return best.name if best else None


# ---------------------------------------------------------------------------
# 1 レコード = 1 SigMF Recording
# ---------------------------------------------------------------------------
@dataclass
class Record:
    path: str          # ベースパス（拡張子なし。read_recording に渡せる）
    center: float      # 中心周波数 Hz（captures[0].core:frequency）
    bw: float          # 占有帯域幅 Hz（annotation のエッジ差）
    label: str | None
    confidence: float | None
    method: str | None  # rule / cnn / llm / human
    snr_db: float | None
    hw: str            # core:hw（出所の正直な記録）
    datetime: str | None

    @property
    def hw_group(self) -> str:
        return hw_group(self.hw)

    @property
    def band(self) -> str | None:
        return band_for_center(self.center)

    @property
    def meta_path(self) -> str:
        return self.path + ".sigmf-meta"

    @property
    def data_path(self) -> str:
        return self.path + ".sigmf-data"


def _pick_annotation(meta: dict) -> dict:
    """索引化に使う代表アノテーションを選ぶ（ラベル付きを優先、無ければ先頭）。"""
    anns = meta.get("annotations") or []
    for a in anns:
        if a.get("core:label"):
            return a
    return anns[0] if anns else {}


def _record_from_meta(base: str, meta: dict) -> Record:
    g = meta.get("global", {})
    caps = meta.get("captures") or [{}]
    cap0 = caps[0]
    ann = _pick_annotation(meta)

    center = float(cap0.get("core:frequency", 0.0))
    lo = ann.get("core:freq_lower_edge")
    hi = ann.get("core:freq_upper_edge")
    bw = float(hi - lo) if (lo is not None and hi is not None) else 0.0

    def _num(x):
        return float(x) if x is not None else None

    return Record(
        path=base,
        center=center,
        bw=bw,
        label=ann.get("core:label"),
        confidence=_num(ann.get("sigscan:confidence")),
        method=ann.get("sigscan:method"),
        snr_db=_num(ann.get("sigscan:snr_db")),
        hw=g.get("core:hw", ""),
        datetime=cap0.get("core:datetime"),
    )


def load_index(dirpath: str) -> "Dataset":
    """ディレクトリ内の *.sigmf-meta を走査し Dataset（Record 一覧）を返す。

    壊れた meta はスキップし stderr に警告する（索引化を止めない）。
    """
    records: list[Record] = []
    for meta_path in sorted(glob.glob(os.path.join(dirpath, "*.sigmf-meta"))):
        base = meta_path[: -len(".sigmf-meta")]
        try:
            # sigmf_io.read_recording と同じ（ロケール既定）エンコーディングで開く。
            # 凍結契約が meta を書いた符号化と一致させ、往復互換を保つ。
            with open(meta_path) as f:
                meta = json.load(f)
            records.append(_record_from_meta(base, meta))
        except Exception as e:   # noqa: BLE001 - 索引化を止めない
            print(f"[load_index] skip {meta_path}: {e}", file=sys.stderr)
    return Dataset(records)


# ---------------------------------------------------------------------------
# データセット（Record の集合 + フィルタ/重複排除/分割/統計）
# ---------------------------------------------------------------------------
class Dataset:
    def __init__(self, records: list[Record]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, i):
        return self.records[i]

    # --- フィルタ ---
    def query(self, hw: str | None = None, label: str | None = None,
              band: str | None = None, snr_min: float | None = None) -> "Dataset":
        """hw / label / バンド名 / SNR下限 でフィルタした Dataset を返す。

        hw     : 'sim' / 'real' / 'other'（グループ）か、core:hw の部分一致文字列。
        label  : core:label の部分一致（大文字小文字無視）。
        band   : バンド名の部分一致（大文字小文字無視）。
        snr_min: sigscan:snr_db >= snr_min（snr 不明のレコードは除外）。
        """
        out = self.records

        if hw is not None:
            key = hw.lower()
            if key in ("sim", "real", "other"):
                out = [r for r in out if r.hw_group == key]
            else:
                out = [r for r in out if key in (r.hw or "").lower()]

        if label is not None:
            key = label.lower()
            out = [r for r in out if r.label and key in r.label.lower()]

        if band is not None:
            key = band.lower()
            out = [r for r in out if r.band and key in r.band.lower()]

        if snr_min is not None:
            out = [r for r in out if r.snr_db is not None and r.snr_db >= snr_min]

        return Dataset(out)

    def groups_by_hw(self) -> dict[str, "Dataset"]:
        """hw グループ（sim/real/other）ごとの Dataset を返す。"""
        groups: dict[str, list[Record]] = {}
        for r in self.records:
            groups.setdefault(r.hw_group, []).append(r)
        return {k: Dataset(v) for k, v in groups.items()}

    # --- 重複排除 ---
    def dedup(self, window_hz: float = 1e6) -> "Dataset":
        """同一 (label, 中心±window_hz) の近接重複を除外する。

        ラベル毎に SNR 降順で走査し、既採用と window 内に重なる中心は捨てる
        （= 各クラスタで最も SNR の高い 1 件を残す）。hw は跨がない
        （sim と real は別レコードとして扱い、互いを重複とみなさない）。
        """
        kept: list[Record] = []
        ordered = sorted(
            self.records,
            key=lambda r: (r.snr_db if r.snr_db is not None else -1e9,
                           r.confidence if r.confidence is not None else -1.0),
            reverse=True,
        )
        for r in ordered:
            dup = False
            for k in kept:
                if (k.label == r.label and k.hw_group == r.hw_group
                        and abs(k.center - r.center) < window_hz):
                    dup = True
                    break
            if not dup:
                kept.append(r)
        # 元の並び（パス昇順）に戻して返す
        kept.sort(key=lambda r: r.path)
        return Dataset(kept)

    # --- train/val 分割（hw を絶対に混ぜない） ---
    def split(self, val_ratio: float = 0.2, seed: int = 0) -> tuple["Dataset", "Dataset"]:
        """train/val に分割する。hw グループ毎に独立して分割し、sim と real を
        混ぜない（CONTRACT.md §2 の防壁）。

        各 hw グループ内で seed 固定シャッフル → 先頭 round(n*val_ratio) 件を val。
        戻り値の train/val はそれぞれ各 hw を比率どおり含むが、1 つの sim レコードと
        real レコードが「同じ点」として train/val 境界を跨ぐことはない。
        純 real での評価が欲しければ先に query(hw='real') してから split する。
        """
        train: list[Record] = []
        val: list[Record] = []
        for grp_name, grp in sorted(self.groups_by_hw().items()):
            recs = sorted(grp.records, key=lambda r: r.path)   # 安定な基準順
            n = len(recs)
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_val = int(round(n * val_ratio))
            val_idx = set(int(i) for i in perm[:n_val])
            for i, r in enumerate(recs):
                (val if i in val_idx else train).append(r)
        train.sort(key=lambda r: r.path)
        val.sort(key=lambda r: r.path)
        return Dataset(train), Dataset(val)

    # --- 統計表示 ---
    def stats(self, file=None) -> None:
        """バンド別・label別・hw別の内訳と SNR ヒストグラムを表示する。"""
        out = file or sys.stdout
        n = len(self.records)
        print(f"== SigMF dataset: {n} records ==\n", file=out)
        if n == 0:
            print("(レコードがありません)", file=out)
            return

        # hw 別
        print("[hw（出所）]", file=out)
        _print_counts(_count_by(self.records, lambda r: f"{r.hw_group:5s} {r.hw}"), out)

        # バンド別
        print("\n[バンド別]", file=out)
        _print_counts(_count_by(self.records,
                                lambda r: r.band or "(バンドプラン外)"), out)

        # label 別
        print("\n[label別]", file=out)
        _print_counts(_count_by(self.records,
                                lambda r: r.label or "(ラベルなし)"), out)

        # method 別
        print("\n[method別]", file=out)
        _print_counts(_count_by(self.records,
                                lambda r: r.method or "(なし)"), out)

        # SNR ヒストグラム
        snrs = [r.snr_db for r in self.records if r.snr_db is not None]
        print(f"\n[SNRヒストグラム]  (n={len(snrs)})", file=out)
        _print_histogram(snrs, out)


# ---------------------------------------------------------------------------
# 表示ヘルパ
# ---------------------------------------------------------------------------
def _count_by(records, keyfn) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in records:
        k = keyfn(r)
        counts[k] = counts.get(k, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _print_counts(items: list[tuple[str, int]], out, bar_w: int = 24) -> None:
    if not items:
        print("  (なし)", file=out)
        return
    top = max(c for _, c in items)
    for name, c in items:
        bar = "#" * max(1, int(round(bar_w * c / top))) if top else ""
        print(f"  {c:4d}  {bar:<{bar_w}}  {name}", file=out)


def _print_histogram(values, out, bins: int = 10, bar_w: int = 30) -> None:
    if not values:
        print("  (SNR データなし)", file=out)
        return
    v = np.asarray(values, dtype=float)
    counts, edges = np.histogram(v, bins=bins)
    top = int(counts.max()) if counts.size else 0
    for i in range(len(counts)):
        c = int(counts[i])
        bar = "#" * int(round(bar_w * c / top)) if top else ""
        print(f"  [{edges[i]:6.1f},{edges[i+1]:6.1f}) {c:4d}  {bar}", file=out)


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


def _cmd_stats(args) -> int:
    ds = load_index(args.dir)
    if args.hw or args.label or args.band or args.snr_min is not None:
        ds = ds.query(hw=args.hw, label=args.label, band=args.band,
                      snr_min=args.snr_min)
    if args.dedup:
        before = len(ds)
        ds = ds.dedup(window_hz=args.dedup_window)
        print(f"(dedup: {before} -> {len(ds)} records)\n")
    ds.stats()
    return 0


def _cmd_query(args) -> int:
    ds = load_index(args.dir).query(hw=args.hw, label=args.label,
                                    band=args.band, snr_min=args.snr_min)
    if args.dedup:
        ds = ds.dedup(window_hz=args.dedup_window)
    print(f"== {len(ds)} records ==")
    for r in ds:
        conf = f"{r.confidence:.2f}" if r.confidence is not None else "  - "
        snr = f"{r.snr_db:5.1f}" if r.snr_db is not None else "  -  "
        print(f"  {r.center/1e6:9.2f}MHz  BW={r.bw/1e6:6.1f}MHz  SNR={snr}dB  "
              f"{conf}/{r.method or '-':5s}  [{r.hw_group}]  "
              f"{r.label or '(none)'}  <{os.path.basename(r.path)}>")
    return 0


def _cmd_review(args) -> int:
    import review
    return review.run_review(args.dir, conf_max=args.conf_max)


def main(argv=None) -> int:
    _force_utf8()
    import argparse
    p = argparse.ArgumentParser(prog="dataset",
                                description="SigMF データセットの管理")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_filters(sp):
        sp.add_argument("dir", help="SigMF を格納したディレクトリ")
        sp.add_argument("--hw", default=None,
                        help="出所フィルタ: sim/real/other か core:hw 部分一致")
        sp.add_argument("--label", default=None, help="label 部分一致")
        sp.add_argument("--band", default=None, help="バンド名 部分一致")
        sp.add_argument("--snr-min", type=float, default=None, dest="snr_min",
                        help="SNR下限 dB")
        sp.add_argument("--dedup", action="store_true",
                        help="近接重複(label,中心±窓)を除外してから集計")
        sp.add_argument("--dedup-window", type=float, default=1e6, dest="dedup_window",
                        help="重複排除の周波数窓 Hz（既定 1e6）")

    sp_stats = sub.add_parser("stats", help="バンド/label/hw/SNR の内訳を表示")
    add_filters(sp_stats)
    sp_stats.set_defaults(func=_cmd_stats)

    sp_query = sub.add_parser("query", help="フィルタ結果を1件ずつ列挙")
    add_filters(sp_query)
    sp_query.set_defaults(func=_cmd_query)

    sp_review = sub.add_parser("review", help="低信頼アノテーションを対話で再ラベル")
    sp_review.add_argument("dir", help="SigMF を格納したディレクトリ")
    sp_review.add_argument("--conf-max", type=float, default=0.5, dest="conf_max",
                           help="この信頼度未満の rule アノテーションを対象（既定0.5）")
    sp_review.set_defaults(func=_cmd_review)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
