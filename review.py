"""低信頼アノテーションの人手レビュー導線。

弱教師ラベル（`method=='rule'` かつ `confidence<0.5`）を列挙し、対話で正しい
label に直して `.sigmf-meta` に書き戻す。書き戻し時に `sigscan:method` を
`'human'` に更新し、`sigscan:confidence` を 1.0（人手＝確定）に上げる。元の
ルールラベルは provenance として comment に残す。

生IQ（`.sigmf-data`）には一切触れない。meta JSON のみを最小限書き換える。
依存は numpy 不要・標準ライブラリと classify/config のみ。

CLI:
    python3 review.py captures/                 # 対話レビュー
    python3 review.py captures/ --list          # 対象を列挙のみ（書き換えなし）
    python3 review.py captures/ --verdict C      # CNN 監査で (C)=C-conflict の記録を対象
    python3 -m dataset review captures/         # 同じ導線（dataset サブコマンド）
"""
from __future__ import annotations
import csv
import datetime
import glob
import json
import os
import sys

from config import BAND_PLAN
from classify import SIGNAL_DB, UNKNOWN, NOISE, SPURIOUS

REVIEW_METHOD = "human"

# CNN 監査 (C) = 「ルールと CNN が食い違い、確定を人手に委ねる」verdict の内部値。
# global の sigscan:cnn_verdict に格納される（cnntrain M3 が書き込む）。
C_CONFLICT = "C-conflict"

# cc_class(視覚分類) → 確定ラベル文字列の写像（1箇所に定義）。hopping/unclear は
# 写像なし＝○×UI で y(提案確定)を出せない（ラベル選択を強制）。spurious は写像あり
# だが、y 許可は「スプリアス警告つきレコードを spurious として確定する＝正しい方向」
# のときだけ（_y_blocked_reason の方向性ガードで制御）。
CC_CLASS_TO_LABEL = {
    "ble-adv": "BLE/Bluetooth (adv?)",
    "wifi": "WiFi (2.4GHz, 20/40MHz)",
    "spurious": SPURIOUS,
}


def cc_class_to_label(cc_class: str | None) -> str | None:
    """cc_class を確定ラベル文字列へ写像。写像が無ければ None。"""
    return CC_CLASS_TO_LABEL.get((cc_class or "").strip())


def _now_iso() -> str:
    """UTC ISO タイムスタンプ（sigmf_io と同形式）。relabel 履歴の at 用。"""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def candidate_labels() -> list[str]:
    """再ラベルの選択肢（信号DB由来のラベル + バンド名 + 統一ラベル）。"""
    labels: list[str] = []
    for entry in SIGNAL_DB:           # (band_sub, bw_cond, label, conf, note)
        labels.append(entry[2])
    for b in BAND_PLAN:
        labels.append(b.name)
    labels += [SPURIOUS, UNKNOWN, NOISE]
    # 重複を順序保持で除去
    seen: set[str] = set()
    uniq: list[str] = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def find_low_confidence(dirpath: str, conf_max: float = 0.5,
                        pattern: str | None = None,
                        include_human: bool = False) -> list[dict]:
    """method=='rule' かつ confidence<conf_max のアノテーションを列挙。

    pattern 指定時はファイル名(ベース)が pattern に一致する *.sigmf-meta だけを走査
    する（既定 None＝全 *.sigmf-meta で従来どおり）。conf_max による絞りは不変。

    include_human=True のとき、method=='human'（確定済み）のレコードも confidence に
    関わらず対象に含める（訂正経路）。既定 False＝従来どおり rule のみ（後方互換）。

    returns: [{meta_path, ann_index, meta, ann, center, bw, label, confidence,
               method, snr_db, hw, datetime}, ...]
    """
    items: list[dict] = []
    glob_pat = (pattern + ".sigmf-meta") if pattern else "*.sigmf-meta"
    for meta_path in sorted(glob.glob(os.path.join(dirpath, glob_pat))):
        try:
            # sigmf_io と同じ（ロケール既定）エンコーディングで開き往復互換を保つ。
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:   # noqa: BLE001
            print(f"[review] skip {meta_path}: {e}", file=sys.stderr)
            continue
        g = meta.get("global", {})
        caps = meta.get("captures") or [{}]
        cap0 = caps[0]
        for i, ann in enumerate(meta.get("annotations") or []):
            method = ann.get("sigscan:method")
            conf = ann.get("sigscan:confidence")
            if method == "rule":
                if conf is None or float(conf) >= conf_max:
                    continue
            elif include_human and method == "human":
                pass   # 訂正経路: 確定済みを confidence に関わらず対象に含める
            else:
                continue
            lo = ann.get("core:freq_lower_edge")
            hi = ann.get("core:freq_upper_edge")
            bw = float(hi - lo) if (lo is not None and hi is not None) else 0.0
            items.append(dict(
                meta_path=meta_path, ann_index=i, meta=meta, ann=ann,
                center=float(cap0.get("core:frequency", 0.0)), bw=bw,
                label=ann.get("core:label"),
                confidence=float(conf) if conf is not None else 0.0,
                method=method,
                snr_db=ann.get("sigscan:snr_db"), hw=g.get("core:hw", ""),
                datetime=cap0.get("core:datetime"),
            ))
    return items


def is_c_conflict(meta: dict) -> bool:
    """global の sigscan:cnn_verdict が 'C-conflict' なら True（純関数・読み取りのみ）。

    キーが無い古い記録・verdict が別値（'A-consistent' 等）は False。
    例外にはしない（安全に除外する）。
    """
    g = meta.get("global") or {}
    return g.get("sigscan:cnn_verdict") == C_CONFLICT


def _candidate_hint(comment) -> str | None:
    """core:comment から監査前の保持候補（"用途=..." 以降）を取り出す。

    構造化キーは無いので comment 文字列の該当部分を返す。marker が無ければ None。
    """
    if not comment:
        return None
    marker = "用途="
    idx = comment.find(marker)
    if idx < 0:
        return None
    return comment[idx:].strip()


def find_c_conflict(dirpath: str, pattern: str | None = None) -> list[dict]:
    """global の sigscan:cnn_verdict=='C-conflict' の記録のアノテーションを列挙。

    読み取りのみ（captures/ は書き換えない）。sigmf_io と同じロケール既定
    エンコーディングで meta を開く（UTF-8 決め打ちしない）。global にキーが
    無い古い記録は安全に除外する。pattern 指定時はファイル名(ベース)一致の
    *.sigmf-meta だけを走査（既定 None＝従来どおり）。

    returns: find_low_confidence と互換の item に CNN 来歴（cnn_class/cnn_conf/
             cnn_verdict）・comment・method（='cnn'）を足した dict のリスト。
    """
    items: list[dict] = []
    glob_pat = (pattern + ".sigmf-meta") if pattern else "*.sigmf-meta"
    for meta_path in sorted(glob.glob(os.path.join(dirpath, glob_pat))):
        try:
            # find_low_confidence と同じ（ロケール既定）エンコーディングで開く。
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:   # noqa: BLE001
            print(f"[review] skip {meta_path}: {e}", file=sys.stderr)
            continue
        if not is_c_conflict(meta):
            continue
        g = meta.get("global", {})
        caps = meta.get("captures") or [{}]
        cap0 = caps[0]
        for i, ann in enumerate(meta.get("annotations") or []):
            lo = ann.get("core:freq_lower_edge")
            hi = ann.get("core:freq_upper_edge")
            bw = float(hi - lo) if (lo is not None and hi is not None) else 0.0
            conf = ann.get("sigscan:confidence")
            items.append(dict(
                meta_path=meta_path, ann_index=i, meta=meta, ann=ann,
                center=float(cap0.get("core:frequency", 0.0)), bw=bw,
                label=ann.get("core:label"),
                confidence=float(conf) if conf is not None else 0.0,
                snr_db=ann.get("sigscan:snr_db"), hw=g.get("core:hw", ""),
                datetime=cap0.get("core:datetime"),
                method=ann.get("sigscan:method"),
                cnn_class=g.get("sigscan:cnn_class"),
                cnn_conf=g.get("sigscan:cnn_conf"),
                cnn_verdict=g.get("sigscan:cnn_verdict"),
                comment=ann.get("core:comment"),
            ))
    return items


def _png_path_for(meta_path: str) -> str | None:
    """captures/_images/<base>.png が存在すればそのパスを返す（無ければ None）。

    内蔵表示はしない（パス表示のみ）。base は meta ファイル名から拡張子を除いたもの。
    """
    base = os.path.basename(meta_path)
    if base.endswith(".sigmf-meta"):
        base = base[: -len(".sigmf-meta")]
    png = os.path.join(os.path.dirname(meta_path), "_images", base + ".png")
    # 実在すれば絶対パスで返す（`\`/`/` 混在を解消し、端末で Ctrl+クリックして開ける形に）。
    return os.path.abspath(png) if os.path.exists(png) else None


def apply_label(meta_path: str, ann_index: int, new_label: str,
                method: str = REVIEW_METHOD, confidence: float = 1.0,
                record_history: bool = False, at: str | None = None) -> dict:
    """meta JSON の指定アノテーションを再ラベルし書き戻す（生IQには触れない）。

    core:label を new_label に、sigscan:method を method('human') に、
    sigscan:confidence を confidence(1.0) に更新。元ラベルは comment に残す。
    sigmf_io.write_recording と同じ整形（indent=2, ensure_ascii=False）で保存。

    record_history=True のとき、annotation の sigscan:relabel_history に
    {from,to,from_method,at} を **append** する（訂正の履歴。過去の確定を黙って
    上書きしない）。既定 False＝従来挙動（履歴キーを足さない・完全後方互換）。
    returns: 更新後の annotation dict。
    """
    # 読み書きとも sigmf_io（ロケール既定エンコーディング）に合わせ、書き戻した
    # meta を sigmf_io.read_recording が再度読めることを保証する。
    with open(meta_path) as f:
        meta = json.load(f)
    anns = meta.get("annotations") or []
    if not (0 <= ann_index < len(anns)):
        raise IndexError(f"annotation index {ann_index} out of range in {meta_path}")
    ann = anns[ann_index]
    old_label = ann.get("core:label")
    old_method = ann.get("sigscan:method")

    ann["core:label"] = str(new_label)
    ann["sigscan:method"] = method
    ann["sigscan:confidence"] = float(confidence)
    note = f"human-relabeled (was '{old_label}' via {old_method})"
    existing = ann.get("core:comment")
    ann["core:comment"] = f"{existing} | {note}" if existing else note
    if record_history:
        hist = ann.get("sigscan:relabel_history")
        if not isinstance(hist, list):
            hist = []
        hist.append({"from": old_label, "to": str(new_label),
                     "from_method": old_method, "at": at or _now_iso()})
        ann["sigscan:relabel_history"] = hist

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return ann


def _extra_lines(item: dict) -> str:
    """CNN 来歴・保持候補・PNG パスの追加行を組む。

    後方互換のため、CNN 監査を通っていない（cnn_verdict が無い）既定 rule 経路の
    item では常に空文字を返す（＝従来表示のまま）。取れない値は省略する。
    """
    if not item.get("cnn_verdict"):
        return ""
    lines: list[str] = []
    cls = item.get("cnn_class")
    conf = item.get("cnn_conf")
    conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
    lines.append(f"  cnn    : cnn={cls}@{conf_s} verdict={item['cnn_verdict']}")
    cand = _candidate_hint(item.get("comment"))
    if cand:
        lines.append(f"  候補   : {cand}")
    png = _png_path_for(item["meta_path"])
    if png:
        lines.append(f"  png    : {png}")
    return "\n" + "\n".join(lines)


def _print_item(item: dict, print_fn) -> None:
    from dataset import band_for_center, hw_group   # 遅延 import（循環回避）
    band = band_for_center(item["center"]) or "(バンドプラン外)"
    snr = item["snr_db"]
    snr_s = f"{snr:.1f}dB" if snr is not None else "?"
    method = item.get("method") or "rule"
    print_fn(
        f"\n  file   : {os.path.basename(item['meta_path'])}\n"
        f"  freq   : {item['center']/1e6:.3f} MHz  (band: {band}, hw: {hw_group(item['hw'])})\n"
        f"  bw/snr : {item['bw']/1e6:.1f} MHz / {snr_s}\n"
        f"  label  : '{item['label']}'  (confidence={item['confidence']:.2f}, method={method})"
        + _extra_lines(item)
    )


def run_review(dirpath: str, conf_max: float = 0.5,
               input_fn=input, print_fn=print, verdict: str | None = None,
               pattern: str | None = None, include_human: bool = False) -> int:
    """対話で低信頼アノテーションを再ラベルする。

    input_fn / print_fn は差し替え可能（テスト用）。input_fn が空文字や 's' を
    返すとスキップ、'q' で終了。数値で候補ラベル選択、それ以外は自由入力ラベル。

    verdict=='C' 指定時は対象を rule 低信頼ではなく CNN 監査 (C)=C-conflict の
    記録に切り替える（選別のみ変更。確定＝apply_label のロジックは不変）。
    pattern 指定時はファイル名(ベース)一致のレコードだけに絞る（選別のみ・確定不変）。
    """
    if verdict == "C":
        items = find_c_conflict(dirpath, pattern=pattern)
        header = (f"CNN監査(C)レビュー: sigscan:cnn_verdict=='{C_CONFLICT}' の記録 "
                  f"{len(items)} 件（対象ディレクトリ: {dirpath}）")
    else:
        items = find_low_confidence(dirpath, conf_max=conf_max, pattern=pattern,
                                    include_human=include_human)
        _scope = (f"confidence<{conf_max}" if conf_max != float("inf") else "全信頼度")
        if pattern:
            _scope += f" / pattern='{pattern}'"
        if include_human:
            _scope += " / +human(訂正対象)"
        header = (f"低信頼レビュー: method='rule' かつ {_scope} の"
                  f"アノテーション {len(items)} 件（対象ディレクトリ: {dirpath}）")
    cands = candidate_labels()

    print_fn(header)
    if not items:
        return 0

    print_fn("\n候補ラベル:")
    for i, c in enumerate(cands):
        print_fn(f"  [{i:2d}] {c}")
    print_fn("\n操作: 番号=その候補に確定 / 自由入力=任意ラベル / 空・s=スキップ / q=終了\n")

    changed = 0
    for n, item in enumerate(items, 1):
        print_fn(f"--- [{n}/{len(items)}] ---")
        _print_item(item, print_fn)
        try:
            ans = input_fn("  正しい label を入力 > ")
        except EOFError:
            print_fn("\n入力終了。レビューを中断します。")
            break
        ans = (ans or "").strip()

        if ans.lower() == "q":
            print_fn("レビューを終了します。")
            break
        if ans == "" or ans.lower() == "s":
            print_fn("  → スキップ")
            continue

        if ans.isdigit() and 0 <= int(ans) < len(cands):
            new_label = cands[int(ans)]
        else:
            new_label = ans   # 自由入力ラベル

        apply_label(item["meta_path"], item["ann_index"], new_label)
        changed += 1
        print_fn(f"  → '{item['label']}' を '{new_label}' に修正（method=human）")

    print_fn(f"\n完了: {changed} 件を再ラベルしました。")
    return 0


# ===========================================================================
# ○×UI（--suggest）: CC 提案を y/n で確定。摩擦=安全弁で y 連打素通りを防ぐ。
# ===========================================================================
def _read_suggestions(path: str) -> list[dict]:
    """review_suggest の suggestions.csv を読む（先頭 '#' コメント行を飛ばす）。"""
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def _first_band_ann_index(meta: dict):
    """band を持つ最初の annotation の (index, ann)。無ければ (0,ann0) / (None,None)。"""
    anns = meta.get("annotations") or []
    for i, a in enumerate(anns):
        if (a.get("core:freq_lower_edge") is not None
                and a.get("core:freq_upper_edge") is not None):
            return i, a
    if anns:
        return 0, anns[0]
    return None, None


def _y_blocked_reason(cc_class: str | None, spurious_warn: bool,
                      recommend: str | None) -> str:
    """○×UIで y(提案確定)を出してよいか。出せない理由文字列を返す（''＝y提示可）。

    安全弁（Pattern A 化＝AI出力が素通りで ground truth 化するのを防ぐ）。スプリアス
    ガードは「方向性」を持つ（無効化ではない）:
      - spurious_warn かつ cc_class!='spurious' → 危険な方向（スプリアス疑いを実信号
        として確定しようとしている）＝ブロック
      - spurious_warn かつ cc_class=='spurious' → 正しい方向（提案どおり spurious と
        して確定）＝許可
    次は不変でブロック（今回の高速化でも一切緩めない）:
      - cc_class 未記入(needs-review) / cc_class=='unclear' / 確定ラベル写像なし
    """
    if recommend == "needs-review" or not (cc_class or "").strip():
        return "視覚分類 未記入(needs-review)"
    if cc_class == "unclear":
        return "CCが unclear と判定"
    if spurious_warn and cc_class != "spurious":
        return ("スプリアス警告 + 非spurious提案（危険な方向: "
                "スプリアス疑いのレコードを実信号として確定しようとしている）")
    if cc_class_to_label(cc_class) is None:
        return f"cc_class='{cc_class}' はラベル写像なし"
    return ""


def _shortcut_labels(cands: list[str]) -> list[tuple[int, str]]:
    """ショートカットに出す (番号, 表示名) を候補表から動的に引く（番号ハードコード禁止）。"""
    out: list[tuple[int, str]] = []
    for label, name in ((SPURIOUS, "spurious"),
                        (CC_CLASS_TO_LABEL["ble-adv"], "BLE"),
                        (CC_CLASS_TO_LABEL["wifi"], "WiFi")):
        if label in cands:
            out.append((cands.index(label), name))
    return out


def _confirm_prompt_line(cands: list[str], allow_y: bool) -> str:
    """1行のショートカットプロンプト。48行の一覧は既定で出さない（? で展開）。

    y は許可時のみ表示（摩擦条件に該当すれば出さない）。頻出ラベルの番号は
    _shortcut_labels で実際の候補番号を引く（表示番号＝入力番号が必ず一致）。
    """
    parts = (["y=確定"] if allow_y else [])
    parts += [f"{i}={name}" for i, name in _shortcut_labels(cands)]
    parts += ["s=スキップ", "?=全ラベル一覧", "q=終了"]
    return "[" + " / ".join(parts) + "]"


def _print_full_list(cands: list[str], print_fn) -> None:
    """従来の48行ラベル一覧（? を押したときだけ出す）。"""
    print_fn("\n候補ラベル:")
    for i, c in enumerate(cands):
        print_fn(f"  [{i:2d}] {c}")


def _prompt_confirm(cands: list[str], allow_y: bool, proposed_label,
                    input_fn, print_fn):
    """1レコードの確定プロンプト（一覧は既定非表示・? で展開）。

    returns (kind, label): kind ∈ {'confirm','label','skip','quit'}
      - y(許可時のみ) → ('confirm', proposed_label)
      - 番号 → その候補 / 自由入力 → 任意ラベル（従来どおり）
      - s・空 → skip / q → quit / ? → 全一覧を印字して再プロンプト
    """
    while True:
        line = _confirm_prompt_line(cands, allow_y)
        try:
            ans = input_fn(f"  {line} > ")
        except EOFError:
            return ("quit", None)
        raw = (ans or "").strip()
        low = raw.lower()
        if low == "q":
            return ("quit", None)
        if raw == "" or low == "s":
            return ("skip", None)
        if raw == "?":
            _print_full_list(cands, print_fn)
            continue
        if low == "y":
            if allow_y:
                return ("confirm", proposed_label)
            print_fn("  ⚠ このレコードは y 不可（摩擦）。"
                     "番号/自由入力でラベルを選ぶか s/? を使ってください。")
            continue
        if raw.isdigit() and 0 <= int(raw) < len(cands):
            return ("label", cands[int(raw)])
        return ("label", raw)   # 自由入力ラベル（従来どおり）


def _build_suggest_ctx(item: dict, suggest: dict) -> dict:
    """1レコードの表示・判定コンテキストを組む（個別確認・一括確定で共用）。

    reason/allow_y（摩擦の方向性ガードの結果）まで確定させる。表示に必要な
    生値（center/bw/snr/persist/現在ラベル）と提案（cc_class/proposed_label/
    det/bw/duty/根拠）を1つの dict にまとめる。
    """
    meta_path = item["meta_path"]
    record = os.path.basename(meta_path)
    if record.endswith(".sigmf-meta"):
        record = record[: -len(".sigmf-meta")]
    png = _png_path_for(meta_path)           # 実在なら絶対パス、無ければ None
    row = suggest.get(record)                # 提案 lookup（無ければ「提案なし」）
    if row is not None:
        cc_class = (row.get("cc_class") or "").strip()
        cc_rationale = (row.get("cc_rationale") or "").strip()
        spurious_warn = (row.get("spurious_warn") == "True")
        recommend = (row.get("recommend") or "").strip()
        proposed_label = cc_class_to_label(cc_class)
        # PNG はローカル実解決を最優先（suggestions.csv の png 列は auto-classify を画像生成前に
        # 走らせると "(なし)" や相対パスで古びるため、実在するローカル画像を上書きさせない）。
        # ローカルに無い場合のみ、CSV に実在パスがあれば絶対化して採用する。
        if png is None:
            csv_png = (row.get("png") or "").strip()
            if csv_png and csv_png != "(なし)" and os.path.exists(csv_png):
                png = os.path.abspath(csv_png)
        det = row.get("det_freq_mhz", "?")
        bw_sug = row.get("bw_mhz", "?")
        duty = (f"{row.get('duty', '?')}"
                f"{'(inconclusive)' if row.get('duty_inconclusive') == 'True' else ''}")
        reason = _y_blocked_reason(cc_class, spurious_warn, recommend)
    else:
        cc_class = cc_rationale = recommend = ""
        spurious_warn = False
        proposed_label = None
        det = bw_sug = duty = "?"
        reason = "提案なし（suggestions.csv に該当エントリなし）"
    # 画像が実在しなければ「(なし)」を素通しせず、次にすべき操作を明示する。
    if png is None:
        png = "(画像未生成: view_captures.py 実行)"
    return dict(
        meta_path=meta_path, ann_index=item["ann_index"], record=record,
        cur_label=item.get("label"), cur_method=item.get("method") or "rule",
        cur_conf=item.get("confidence"),
        center=item.get("center"), bw_item=item.get("bw"),
        snr=item.get("snr_db"),
        persist=(item.get("ann") or {}).get("sigscan:persistence"),
        png=png, has_suggestion=(row is not None),
        cc_class=cc_class, cc_rationale=cc_rationale,
        spurious_warn=spurious_warn, recommend=recommend,
        proposed_label=proposed_label, det=det, bw_sug=bw_sug, duty=duty,
        reason=reason, allow_y=(reason == ""),
    )


def _print_suggest_header(ctx: dict, n: int, total: int, print_fn) -> None:
    """個別確認の1レコード見出し（従来の表示を踏襲）。"""
    if ctx["has_suggestion"]:
        cc_line = (f"  CC提案 : {ctx['proposed_label'] or '(写像なし)'}   "
                   f"[cc_class={ctx['cc_class'] or '未記入'}]")
        rationale_line = f"  根拠   : {ctx['cc_rationale'] or '-'}"
        extra = f" duty={ctx['duty']} spur={ctx['spurious_warn']}"
    else:
        cc_line = "  CC提案 : (提案なし＝suggestions.csv に該当エントリなし)"
        rationale_line = None
        extra = ""
    snr, persist = ctx["snr"], ctx["persist"]
    lines = [f"--- [{n}/{total}] ---",
             f"  file   : {ctx['record']}.sigmf-meta",
             f"  PNG    : {ctx['png']}", cc_line]
    if rationale_line:
        lines.append(rationale_line)
    lines.append(f"  客観   : tuner={ctx['center']/1e6:.3f}MHz "
                 f"det_bw={ctx['bw_item']/1e6:.2f}MHz "
                 f"SNR={snr if snr is not None else '?'}dB "
                 f"persist={persist if persist is not None else '?'}{extra}")
    lines.append(f"  現在   : label='{ctx['cur_label']}' "
                 f"(confidence={ctx['cur_conf']}, method={ctx['cur_method']})"
                 + ("  ← 訂正対象" if ctx['cur_method'] == "human" else ""))
    print_fn("\n".join(lines))


def _review_one(ctx: dict, cands: list[str], n: int, total: int,
                input_fn, print_fn, apply_fn) -> str:
    """1レコードを個別確認して確定/スキップ/終了する。returns 'changed'|'skip'|'quit'。"""
    _print_suggest_header(ctx, n, total, print_fn)
    if not ctx["allow_y"]:
        print_fn(f"  ⚠ {ctx['reason']} → 提案確定(y)は不可。"
                 "ラベルを選択してください（skip 可）")
    kind, new_label = _prompt_confirm(cands, ctx["allow_y"],
                                      ctx["proposed_label"], input_fn, print_fn)
    if kind == "quit":
        print_fn("レビューを終了します。")
        return "quit"
    if kind == "skip":
        print_fn("  → スキップ")
        return "skip"
    if kind == "confirm":
        apply_fn(ctx["meta_path"], ctx["ann_index"], ctx["proposed_label"],
                 record_history=True)
        print_fn(f"  → 確定: '{ctx['cur_label']}' → '{ctx['proposed_label']}' "
                 f"(method=human, cc_class={ctx['cc_class']})")
        return "changed"
    apply_fn(ctx["meta_path"], ctx["ann_index"], new_label, record_history=True)
    print_fn(f"  → 確定: '{ctx['cur_label']}' → '{new_label}' (method=human)")
    return "changed"


def _batch_confirm(batch: list[dict], input_fn, print_fn, apply_fn):
    """一括候補（y 許可の明快な提案）を一覧表示し y/i/n を受ける。

    returns (result, n_confirmed): result ∈ {'confirm','individual','abort'}。
      confirm → 一覧を CC提案どおり一括確定した / individual → 1件ずつ回す /
      abort → 何も確定しない。**迷うものは一切ここに来ない**（呼び出し側で仕分け済み）。
    """
    if not batch:
        print_fn("\n一括確定候補: 0 件（全件が個別確認へ）")
        return ("individual", 0)
    print_fn(f"\n=== 一括確定候補（CC提案どおり y 可）: {len(batch)} 件 ===")
    for i, c in enumerate(batch, 1):
        print_fn(f"  [{i}] {c['record']}  cc={c['cc_class']} → {c['proposed_label']}  "
                 f"det={c['det']}MHz bw={c['bw_sug']}MHz duty={c['duty']}")
        print_fn(f"       PNG: {c['png']}  根拠: {c['cc_rationale'] or '-'}")
    try:
        ans = input_fn(f"\n上記 {len(batch)} 件を CC提案どおり確定しますか？ "
                       "[y=一括確定 / i=1件ずつ個別確認 / n=中止] > ")
    except EOFError:
        return ("abort", 0)
    ans = (ans or "").strip().lower()
    if ans == "y":
        counts: dict[str, int] = {}
        for c in batch:
            apply_fn(c["meta_path"], c["ann_index"], c["proposed_label"],
                     record_history=True)
            counts[c["proposed_label"]] = counts.get(c["proposed_label"], 0) + 1
        print_fn(f"  → 一括確定 {len(batch)} 件（method=human）")
        for label, k in sorted(counts.items()):
            print_fn(f"     {label}: {k} 件")
        return ("confirm", len(batch))
    if ans == "i":
        return ("individual", 0)
    return ("abort", 0)


def run_suggest_review(dirpath: str, suggest_csv: str,
                       input_fn=input, print_fn=print,
                       include_human: bool = False, apply_fn=None,
                       conf_max: float = float("inf"),
                       verdict: str | None = None,
                       pattern: str | None = None,
                       batch_confirm: bool = False) -> int:
    """○×UI: CC 提案を確定する（摩擦=安全弁つき・一覧は既定非表示・? で展開）。

    **対象集合は走査＋フィルタで決める**（run_review と同一経路。suggestions.csv は
    対象集合の決定に一切関与しない）:
      verdict=='C' → find_c_conflict(dirpath, pattern)
      それ以外     → find_low_confidence(dirpath, conf_max, pattern, include_human)
    suggestions.csv は **各レコードに提案を紐付ける lookup 専用**（キー=ファイル名ベース）。

    y=CC提案ラベルで確定 / 番号=ラベル選択 / s=スキップ / ?=全一覧 / q=終了。
    摩擦（提案素通り＝Pattern A 化を構造で防ぐ）＝ _y_blocked_reason の方向性ガード:
      unclear / needs-review(未記入) / 提案なし / **spurious警告+非spurious提案** は
      y を出さない。spurious警告+spurious提案（正しい方向）だけは y を許可する。
    batch_confirm=True: y 可の明快な提案を一括確定できる（迷うものは個別確認へ回す。
      **黙って捨てない**）。既定 False＝従来どおり1件ずつ。
    確定は apply_label（method=human, confidence=1.0, 履歴 record_history=True）。
    """
    apply_fn = apply_fn or apply_label
    # --- 対象集合＝走査＋フィルタ（既存の選別ロジックを再利用）---
    if verdict == "C":
        items = find_c_conflict(dirpath, pattern=pattern)
    else:
        items = find_low_confidence(dirpath, conf_max=conf_max, pattern=pattern,
                                    include_human=include_human)
    # --- suggestions.csv は提案の lookup（record 名→行）専用 ---
    suggest = {(r.get("record") or "").strip(): r
               for r in _read_suggestions(suggest_csv)
               if (r.get("record") or "").strip()}
    cands = candidate_labels()
    scope = f"pattern={pattern!r} include_human={include_human}"
    if verdict == "C":
        scope += " verdict=C"
    print_fn(f"○×UIレビュー: 対象 {len(items)} 件（{scope} / 提案元 {suggest_csv}）")

    ctxs = [_build_suggest_ctx(item, suggest) for item in items]
    changed = 0
    individual = ctxs
    if batch_confirm:
        # 仕分け: y 許可の明快なものだけ一括候補。迷うもの（unclear / needs-review /
        # 提案なし / spurious警告+非spurious）は必ず個別確認へ（一括に混ぜない）。
        batch = [c for c in ctxs if c["allow_y"]]
        rest = [c for c in ctxs if not c["allow_y"]]
        result, nconf = _batch_confirm(batch, input_fn, print_fn, apply_fn)
        if result == "abort":
            print_fn("\n中止しました（何も確定していません）。")
            return 0
        changed += nconf
        # 一括後も個別グループは必ず回す。i 選択時は一括候補も個別へ回す。
        individual = (batch + rest) if result == "individual" else rest

    if batch_confirm and individual:      # 仕分け後の個別グループを明示（従来モードでは出さない）
        print_fn(f"\n--- 個別確認: {len(individual)} 件 ---")
    for n, ctx in enumerate(individual, 1):
        st = _review_one(ctx, cands, n, len(individual),
                         input_fn, print_fn, apply_fn)
        if st == "quit":
            break
        if st == "changed":
            changed += 1

    print_fn(f"\n完了: {changed} 件を確定しました。")
    return 0


def _cmd_list(dirpath: str, conf_max: float, verdict: str | None = None,
              pattern: str | None = None, include_human: bool = False) -> int:
    if verdict == "C":
        items = find_c_conflict(dirpath, pattern=pattern)
        print(f"CNN監査(C)記録 {len(items)} 件 (sigscan:cnn_verdict=='{C_CONFLICT}'):")
    else:
        items = find_low_confidence(dirpath, conf_max=conf_max, pattern=pattern,
                                    include_human=include_human)
        _scope = (f"confidence<{conf_max}" if conf_max != float("inf") else "全信頼度")
        if pattern:
            _scope += f", pattern='{pattern}'"
        if include_human:
            _scope += ", +human(訂正対象)"
        print(f"低信頼アノテーション {len(items)} 件 (method='rule', {_scope}):")
    for item in items:
        base = os.path.basename(item['meta_path'])
        head = (f"  {item['center']/1e6:9.3f}MHz  conf={item['confidence']:.2f}  "
                f"'{item['label']}'  <{base}>")
        # 訂正経路（--include-human）で拾った確定済みは訂正対象として明示。
        # rule/cnn 経路は method!='human' なので付かない（後方互換）。
        if item.get("method") == "human":
            head += "  [method=human ← 訂正対象]"
        # CNN 来歴・保持候補・PNG パスは C 経路（cnn_verdict あり）でのみ付加。
        # 既定 rule 経路は cnn_verdict を持たないので従来行のまま（後方互換）。
        if item.get("cnn_verdict"):
            conf = item.get("cnn_conf")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
            head += f"  cnn={item.get('cnn_class')}@{conf_s} verdict={item['cnn_verdict']}"
            cand = _candidate_hint(item.get("comment"))
            if cand:
                head += f"  {cand}"
            png = _png_path_for(item['meta_path'])
            if png:
                head += f"  [{png}]"
        print(head)
    return 0


def main(argv=None) -> int:
    _force_utf8()
    import argparse
    p = argparse.ArgumentParser(prog="review",
                                description="低信頼アノテーションの人手レビュー/再ラベル")
    p.add_argument("dir", help="SigMF を格納したディレクトリ")
    p.add_argument("--conf-max", type=float, default=None, dest="conf_max",
                   help="この信頼度未満の rule アノテーションを対象（既定0.5）。"
                        "--pattern 単独時は信頼度を無視、併用時は AND")
    p.add_argument("--list", action="store_true",
                   help="対象を列挙するだけで書き換えない")
    p.add_argument("--verdict", choices=["C"], default=None,
                   help="CNN監査で (C)=C-conflict になった記録を対象にする"
                        "（rule低信頼の代わり。未指定なら従来どおり）")
    p.add_argument("--pattern", default=None, metavar="GLOB",
                   help="ファイル名(ベース)が GLOB に一致するレコードだけを対象にする"
                        "（例 \"2402MHz_1783530*\"）。特定レコードを狙い撃つ選別フィルタ。"
                        "--conf-max と併用で AND、単独なら信頼度に関わらず一致ファイルを対象")
    p.add_argument("--suggest", default=None, metavar="CSV",
                   help="review_suggest の suggestions.csv を読み CC提案を ○×UIで確定。"
                        "y=提案ラベルで確定 / 番号=ラベル選択 / ?=全一覧。安全弁: unclear/"
                        "未記入/『warn+非spurious提案』は y 不可（ラベル選択強制）。warn+spurious"
                        "提案は y 可（正しい方向）。--list 併用時は列挙のみ（確定しない）")
    p.add_argument("--batch-confirm", action="store_true", dest="batch_confirm",
                   help="--suggest 併用時のみ有効。y 可の明快な提案を一括確定できる"
                        "（迷うもの＝unclear/未記入/warn+非spurious は個別確認へ回す）。"
                        "既定オフ＝従来どおり1件ずつ")
    p.add_argument("--include-human", action="store_true", dest="include_human",
                   help="method=human の確定済みレコードも対象に含める（訂正経路）。"
                        "--pattern と併用で特定の誤確定を呼び戻す。既定は rule のみ（従来）")
    args = p.parse_args(argv)
    # --conf-max 実効値: 明示指定を最優先。--pattern 単独なら信頼度を無視(inf)。
    #   どちらも無ければ従来既定 0.5（＝既存の対象選択・挙動は不変）。
    if args.conf_max is not None:
        conf_max = args.conf_max
    elif args.pattern is not None:
        conf_max = float("inf")
    else:
        conf_max = 0.5
    # include_human は True のときだけ渡す（既定 False は各関数の既定に委ね、既存の
    #   呼び出し・mock シグネチャとの後方互換を保つ）。
    ih = {"include_human": True} if args.include_human else {}
    if args.list:                       # --list は読み取りのみ（--suggest 併用でも列挙）
        return _cmd_list(args.dir, conf_max, verdict=args.verdict,
                         pattern=args.pattern, **ih)
    if args.suggest is not None:        # ○×UI（対話確定）: 対象は走査+フィルタ・CSVは提案lookup
        return run_suggest_review(args.dir, args.suggest, conf_max=conf_max,
                                  verdict=args.verdict, pattern=args.pattern,
                                  batch_confirm=args.batch_confirm, **ih)
    if args.batch_confirm:              # --batch-confirm は --suggest 専用（単独は無視）
        print("警告: --batch-confirm は --suggest と併用時のみ有効です（無視します）。")
    return run_review(args.dir, conf_max=conf_max, verdict=args.verdict,
                      pattern=args.pattern, **ih)


if __name__ == "__main__":
    raise SystemExit(main())
