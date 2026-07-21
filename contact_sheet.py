"""確定レビューのコンタクトシート（表示補助のみ・確定/分類には使わない）。

`review.py --suggest --batch-confirm --open-sheet` で、レビュー対象の**既存 PNG**
（`captures/_images/*.png`＝凍結 `spec.render` 生成物）を1枚のグリッド画像に並べ、
各サムネイルに CC の提案ラベル `[番号] cc_class` を焼き込む。人間は1枚を眺めて
ターミナルの○×一覧と突き合わせられる。

原則（逸脱禁止）:
  * **`spec.render` を呼ばない**（凍結表現を迂回した独自画像化をしない）。既存 PNG を読むだけ。
  * **`captures/` に書かない**。出力は呼び出し側が `bench/` 配下のパスを渡す。
  * matplotlib は `build_contact_sheet` 内で**遅延 import**（`review` の軽さ・
    `--open-sheet` を使わない従来経路で matplotlib を引かない）。
  * これは見せ方の補助。○×の判断・確定・摩擦・Pattern A 防波堤には一切関与しない。
"""
from __future__ import annotations

import os


def _title_font_family() -> list:
    """タイトル用フォント列（和文が豆腐□にならないための表示時対処）。

    view_captures._title_font_family と同じ方針: DejaVu Sans を先頭に、インストール
    済みの和文フォントを1つだけ後ろに足す（matplotlib>=3.6 のグリフ単位フォール
    バック）。見つからなければ既定のまま。**表示のみ**（保存 PNG のピクセルには無関係）。
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


def caption_for(index: int, cc_class: str | None, has_suggestion: bool = True) -> str:
    """サムネイルのキャプション文字列 `[番号] cc_class` を組む（純関数・テスト可能）。

    cc_class 未記入（needs-review）や提案なしは `(提案なし)`。番号は○×UI の
    一括候補/個別確認の番号と一致させるための 1-based index。
    """
    cc = (cc_class or "").strip()
    label = cc if cc else "(提案なし)"
    return f"[{index}] {label}"


def sheet_entries(ordered_ctxs: list[dict]) -> list[dict]:
    """提示順の ctx 群 → シート用エントリ列（番号は提示順に 1-based で一致）。

    `ordered_ctxs` は `run_suggest_review` が組む提示順（batch → rest）。返す各 entry の
    `index` は `_batch_confirm` の一括候補番号（batch 部分の [1..N]）と一致する。
    純関数（matplotlib 不要）。allow_y は枠色のヒント（確定可否の判断には使わない）。
    """
    entries: list[dict] = []
    for i, ctx in enumerate(ordered_ctxs, 1):
        cc = (ctx.get("cc_class") or "").strip()
        entries.append(dict(
            index=i,
            record=ctx.get("record"),
            cc_class=cc,
            caption=caption_for(i, cc, bool(ctx.get("has_suggestion"))),
            png=ctx.get("png"),
            allow_y=bool(ctx.get("allow_y")),
        ))
    return entries


def _png_is_readable(png) -> bool:
    """png が実在するファイルパスなら True（`(画像未生成…)` 等のプレースホルダは False）。"""
    return bool(png) and isinstance(png, str) and os.path.isfile(png)


def build_contact_sheet(entries: list[dict], out_path: str,
                        cols: int = 5, thumb_in: float = 2.3) -> str | None:
    """entries をグリッド画像にして out_path に書き、パスを返す（0件なら None）。

    各セル: 既存 PNG を imshow ＋ キャプション `[番号] cc_class`。PNG 欠損・読込失敗は
    `(画像なし)` プレースホルダ（欠損で落ちない）。allow_y は緑枠、個別確認（allow_y
    =False）は橙枠で「y 可/要ラベル選択」を色で示す（最小の親切・判断には使わない）。
    matplotlib は**ここで遅延 import**（Agg・GUI 不要・保存のみ）。out_path の親
    ディレクトリは作成する（呼び出し側が `bench/` 配下を渡す＝`captures/` に書かない）。
    """
    if not entries:
        return None
    import matplotlib
    matplotlib.use("Agg")                 # 画面なし・ファイル保存のみ
    matplotlib.rcParams["font.family"] = _title_font_family()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n = len(entries)
    ncol = max(1, min(cols, n))
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * thumb_in, nrow * (thumb_in + 0.35)),
                             squeeze=False)
    cells = [axes[r][c] for r in range(nrow) for c in range(ncol)]
    for k, ax in enumerate(cells):
        if k >= n:
            ax.axis("off")               # 余りセルは非表示
            continue
        e = entries[k]
        drew = False
        if _png_is_readable(e.get("png")):
            try:
                ax.imshow(mpimg.imread(e["png"]), aspect="auto")
                drew = True
            except Exception:
                drew = False
        if not drew:
            ax.imshow([[0.15]], cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.text(0.5, 0.5, "(画像なし)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        # 枠色: y 可=緑 / 個別確認(unclear/warn/提案なし)=橙（色は補助・判断はしない）。
        color = "#2e9e4f" if e.get("allow_y") else "#e08a1e"
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(2.2)
        ax.set_title(e["caption"], fontsize=8, color=color)

    fig.suptitle("コンタクトシート（表示補助・確定は人間の○×）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
