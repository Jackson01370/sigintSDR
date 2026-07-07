# 作業指示書: capture-engine — view_captures.py 周波数軸修正（center が lower_edge に化けるバグ）

## 役割
あなたは sigscan プロジェクトの capture-engine エージェント。
本指示は **表示ツール view_captures.py のみ**の最小修正。目的は、画像の周波数軸・タイトルが
**誤った基準（annotation の freq_lower_edge）**で描かれているバグを直し、
「画像で確認してから次へ」という本プロジェクト最重要の規律の土台を回復すること。

## 背景（人間の実測とコード読解で機序確定済み — 疑う余地なし）

view_captures.py の `_meta_summary`（40行目付近）:

```python
center = float(g.get("core:frequency", a.get("core:freq_lower_edge", 0.0)) or 0.0)
```

問題は三段:

1. `core:frequency` を **global (`g`) から探しているが、SigMF 標準ではこのキーは
   captures 配列の要素に入る**。本プロジェクトの sigmf_io もそう書いている
   （実例: `meta["captures"][0]["core:frequency"]` に 2401908502.12 が入っており、
   global には無い）。よって `g.get` は**常に失敗**する。
2. その `g.get` の**デフォルト引数に `a.get("core:freq_lower_edge", 0.0)` を
   詰め込んでいる**ため、失敗時のフォールバックが「annotation の下端周波数」になる。
   結果、**center = freq_lower_edge**。
3. 直後の復元分岐 `if (not center) and lo is not None and hi is not None:` は、
   center が lower_edge（truthy）のため**決して実行されない死にコード**。

帰結: 軸（135行目 `freqs_mhz = linspace(-rate/2, rate/2) + center`）とタイトルが
常に lower_edge 基準となり、**真の周波数から (IQ中心 − lower_edge) だけ全画像がずれる**。
狙いどおりの信号が保存された記録でも −BW/2 ずれ、dwell で狙いと違う検出が保存された
記録では数 MHz ずれる。

実測証拠（記録 `captures/2402MHz_1783315966234_1`）:
- タイトル表示 2406.6 MHz = annotation freq_lower_edge 2406.551 の丸め（バグの直接痕跡）。
- IQ の物理中心 = captures[0]["core:frequency"] = 2401.909 MHz。
- 画像内に IQ インバランス由来のイメージペア（DC を挟んで対称の鏡像バースト）があり、
  その対称中心を表示軸から逆算すると実 2401.75 MHz ≈ captures の値と一致。
  → **captures[0]["core:frequency"] が物理的に正しい軸基準であることの独立な裏付け**。

## 最重要原則（絶対厳守）
1. **6継ぎ目不可侵**: spec.render / sdr / dsp / classify.classify / sigmf_io / store の
   シグネチャ・挙動を変更しない。`git diff --stat -- spec.py sigmf_io.py` が空であること。
2. **変更は view_captures.py（と新規テスト）のみ**。quality.py / scheduler.py / dwell.py は
   一切触らない（dwell のオフセットチューニングは別タスク）。
3. **spec.render の出力（スペクトログラム画像そのもの）には一切手を加えない**。
   本修正は軸ラベル・extent・タイトル・マーカー重畳という「目盛りと注記」だけの話。
4. **render_one のシグネチャは維持**（spotlight が再利用しているため）。戻り値・引数を変えない。
   info 辞書へのキー追加は可（既存キーの意味変更は不可）。
5. **既存テストは無変更**。追加のみ。
6. **最小実装**。配色・レイアウト・機能の美化はしない。迷ったら最小。

## 事前確認（実装前に必ず）
1. `render_one` の呼び出し元を全検索（spotlight 含む）し、シグネチャ互換の影響範囲を把握。
2. 実メタを1件読み、captures / global / annotations のキー配置を目で確認
   （locale は cp932。UTF-8 決め打ち禁止）。
3. `--flatten-dc` の経路が本修正と独立であることを確認（触らない）。

## 実装

### 1. `_meta_summary` の center 取得を正す（優先順位を明示的な段階に分解）

デフォルト引数への詰め込みをやめ、以下の優先順位で:

```python
caps = meta.get("captures", []) or []
c0 = caps[0] if caps else {}
center = float(c0.get("core:frequency", 0.0) or 0.0)      # 第一: IQの物理中心（正）
if not center:
    center = float(g.get("core:frequency", 0.0) or 0.0)   # 第二: 旧データ互換（globalに持つ場合）
lo = a.get("core:freq_lower_edge")
hi = a.get("core:freq_upper_edge")
if (not center) and lo is not None and hi is not None:
    center = (float(lo) + float(hi)) / 2.0                # 第三: annotationから復元（最後の手段）
```

あわせて info に検出帯を追加（マーカー描画用。無ければ None）:

```python
"det_lo": (float(lo) if lo is not None else None),
"det_hi": (float(hi) if hi is not None else None),
```

### 2. 軸は IQ 中心基準のまま（式は既存どおり）、検出帯マーカーを重畳

- 135行目の軸計算式は変更不要（center の中身が正しくなることで軸が直る）。
- 左パネルに annotation 検出帯 [det_lo, det_hi] を重畳:
  半透明の横帯（`axhspan(det_lo/1e6, det_hi/1e6, alpha≈0.15)`）＋上下端に細線。
  det_lo/det_hi が None のときは描かない。
  → 「保存の根拠になった検出帯」と「画像の実体」のズレが一目で見える
  （dwell の混獲記録を発見する道具になる）。

### 3. タイトルを二本立てに

- IQ 中心（tuner）と検出中心（det = (lo+hi)/2）を併記。例:
  `tuner 2401.9 MHz | det 2408.0 MHz  BW~2.9 MHz`
- det が無い記録は従来様式（tuner のみ＋BW）。

## テスト（追加のみ）
`_meta_summary` のユニットテストを新規追加（既存テストの配置慣習に合わせる）:
1. captures[0] に core:frequency がある標準メタ → center がその値（annotation に
   lower/upper があっても captures 優先であることを含めて検証）。
2. captures に無く global にある → global の値。
3. どちらにも無く annotation の lo/hi のみ → (lo+hi)/2。
4. det_lo / det_hi が annotation から正しく取れる／無ければ None。

## 検証（人間の目視。完了報告に再生成画像のパスを含めること）
1. `python view_captures.py captures/` で再生成。
2. 記録 `2402MHz_1783315966234_1` の新画像で:
   - イメージペアの対称中心が軸上で ≈2401.9 MHz（=DC）に来る。
   - 検出帯マーカーが 2406.55–2409.44 MHz に出て、明るいバーストと重なる。
   - 細い定常線が ≈2400.0 MHz に来る。
3. 別の1件（狙いどおり保存された狭帯域記録）で、検出帯マーカーが DC 近傍に来る。

## 完了報告に含めること
1. 変更ファイル一覧と diff 要約（view_captures.py と新規テストのみのはず）。
2. pytest 全緑（既存テスト無変更で追加のみ）。
3. `git diff --stat -- spec.py sigmf_io.py` が空。
4. 実装が本指示書の範囲内である宣言（スコープ外に触れていない）。

## 禁止事項
- quality.py / scheduler.py / dwell.py / 6継ぎ目の変更。
- spec.render 出力（画像ピクセル）への加工・リサイズ・正規化変更。
- 既存テストの変更・削除・弱体化。
- メタ読み込みの UTF-8 決め打ち（cp932 を維持）。
- スコープ膨張（新オプション追加・配色変更・リファクタ等）。
