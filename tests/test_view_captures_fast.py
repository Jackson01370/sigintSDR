"""test_view_captures_fast.py — view_captures の高速化（--pattern + 冪等スキップ）。

ロックする契約（追加のみ・既存無変更）:
  F1 pattern フィルタ: select_metas が pattern 一致分だけ選ぶ。未指定は全件。0件で正常。
  F2 冪等スキップ: _needs_render が PNG新しい→スキップ / 無い・古い→描画 / --force→描画 /
     mtime 取得失敗→描画（保守的）。data の mtime も見る（新しい方）。
  F3 後方互換: --pattern 無し / --force で全描画。--limit の既存挙動が不変。
  F4 captures 非書込: 描画は _images にのみ書き、SigMF（meta/data）を変更しない。
  F5 スモーク: 1件 描画 → 2回目スキップ → --force で再描画。

captures/ は触らない（tmp のみ）。spec.render は迂回しない（描画は render_one 経由）。
"""
import os
import time

import numpy as np
import pytest

import view_captures as vc


def _touch(p):
    open(p, "w").close()


def _mk_meta_data(d, name):
    """mtime 判定用の空 meta/data ペアを作る（描画はしない）。"""
    _touch(os.path.join(d, name + ".sigmf-meta"))
    _touch(os.path.join(d, name + ".sigmf-data"))


def _write_sigmf(dirpath, name, seed=0, n=8192):
    """render 可能な実 SigMF を書く（スモーク用）。"""
    import sigmf_io
    rng = np.random.default_rng(seed)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    base = os.path.join(dirpath, name)
    sigmf_io.write_recording(
        base, iq, center_hz=2.44e9, sample_rate=20e6,
        annotations=[dict(freq_lower_edge=2.433e9, freq_upper_edge=2.447e9,
                          label="BLE/Bluetooth (adv?)", confidence=0.6,
                          method="rule")])
    return base


# ===========================================================================
# F1: select_metas（pattern / limit / 0件）
# ===========================================================================
def test_select_metas_pattern_all_and_nomatch(tmp_path):
    d = str(tmp_path)
    for nm in ("2400MHz_tagA_1", "2400MHz_tagA_2", "2481MHz_tagB_1"):
        _touch(os.path.join(d, nm + ".sigmf-meta"))
    allm = [os.path.basename(m) for m in vc.select_metas(d)]
    assert allm == ["2400MHz_tagA_1.sigmf-meta", "2400MHz_tagA_2.sigmf-meta",
                    "2481MHz_tagB_1.sigmf-meta"]                 # 未指定=全件
    tagA = [os.path.basename(m) for m in vc.select_metas(d, pattern="*tagA*")]
    assert tagA == ["2400MHz_tagA_1.sigmf-meta", "2400MHz_tagA_2.sigmf-meta"]
    assert vc.select_metas(d, pattern="*nomatch*") == []         # 0件（一致なし）
    assert len(vc.select_metas(d, limit=2)) == 2                 # limit=先頭2件


# ===========================================================================
# F2: _needs_render（スキップ / 描画 / force / 保守的）
# ===========================================================================
def test_needs_render_skip_render_force(tmp_path):
    d = str(tmp_path)
    base = os.path.join(d, "rec")
    _mk_meta_data(d, "rec")
    png = os.path.join(d, "rec.png")
    assert vc._needs_render(base, png) is True                   # PNG 無し → 描画
    _touch(png)
    os.utime(png, (time.time() + 10, time.time() + 10))          # PNG を新しく
    assert vc._needs_render(base, png) is False                  # 新しい → スキップ
    assert vc._needs_render(base, png, force=True) is True       # --force → 描画
    os.utime(png, (1, 1))                                        # PNG を古く
    assert vc._needs_render(base, png) is True                   # 古い → 描画


def test_needs_render_uses_newer_of_meta_data(tmp_path):
    d = str(tmp_path)
    base = os.path.join(d, "rec")
    _mk_meta_data(d, "rec")
    png = os.path.join(d, "rec.png"); _touch(png)
    os.utime(png, (100, 100))
    os.utime(base + ".sigmf-meta", (50, 50))
    os.utime(base + ".sigmf-data", (200, 200))                  # data が PNG より新しい
    assert vc._needs_render(base, png) is True                   # data(200)>png(100)→描画


def test_needs_render_conservative_on_error(tmp_path):
    d = str(tmp_path)
    base = os.path.join(d, "ghost")                             # meta/data 無し
    png = os.path.join(d, "ghost.png"); _touch(png)
    # PNG はあるが SigMF が無い → mtime 取得失敗 → 保守的に描画（描かれない事故を避ける）。
    assert vc._needs_render(base, png) is True


# ===========================================================================
# F5 + F3: main() スモーク（描画→スキップ→force）・後方互換
# ===========================================================================
def test_main_render_then_skip_then_force(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    d = tmp_path / "caps"; d.mkdir()
    _write_sigmf(str(d), "2440MHz_tagX_1", seed=1)
    _write_sigmf(str(d), "2481MHz_tagY_1", seed=2)
    imgdir = d / "_images"
    # 1回目: pattern で tagX のみ → 1件描画（tagY は対象外）。
    assert vc.main([str(d), "--pattern", "*tagX*"]) == 0
    out = capsys.readouterr().out
    assert (imgdir / "2440MHz_tagX_1.png").exists()
    assert not (imgdir / "2481MHz_tagY_1.png").exists()
    assert "描画 1 件" in out
    # 2回目: 同条件 → 冪等スキップ（描画0）。
    vc.main([str(d), "--pattern", "*tagX*"])
    out2 = capsys.readouterr().out
    assert "描画 0 件" in out2 and "スキップ 1 件" in out2
    # --force: 全再描画（後方互換）。
    vc.main([str(d), "--pattern", "*tagX*", "--force"])
    out3 = capsys.readouterr().out
    assert "描画 1 件" in out3 and "skip=OFF(--force)" in out3


def test_main_pattern_no_match_returns_0(tmp_path, capsys):
    d = tmp_path / "caps"; d.mkdir()
    _write_sigmf(str(d), "2440MHz_tagX_1", seed=1)
    assert vc.main([str(d), "--pattern", "*zzz*"]) == 0          # 0件でも正常終了
    assert "パターン一致なし" in capsys.readouterr().out


def test_main_no_pattern_renders_all_backcompat(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    d = tmp_path / "caps"; d.mkdir()
    _write_sigmf(str(d), "2440MHz_a", seed=1)
    _write_sigmf(str(d), "2481MHz_b", seed=2)
    assert vc.main([str(d)]) == 0                                # 未指定=全件
    out = capsys.readouterr().out
    assert "描画 2 件" in out and "pattern=(全件)" in out
    assert (d / "_images" / "2440MHz_a.png").exists()
    assert (d / "_images" / "2481MHz_b.png").exists()


# ===========================================================================
# F4: captures（SigMF）非改変
# ===========================================================================
def test_main_does_not_modify_sigmf(tmp_path):
    pytest.importorskip("matplotlib")
    d = tmp_path / "caps"; d.mkdir()
    base = _write_sigmf(str(d), "2440MHz_tagX_1", seed=1)
    meta_p, data_p = base + ".sigmf-meta", base + ".sigmf-data"
    m0, d0 = os.path.getmtime(meta_p), os.path.getmtime(data_p)
    s0 = os.path.getsize(meta_p)
    vc.main([str(d)])
    # SigMF は不変（mtime/サイズ）。PNG は _images にのみ出る。
    assert os.path.getmtime(meta_p) == m0 and os.path.getmtime(data_p) == d0
    assert os.path.getsize(meta_p) == s0
    assert (d / "_images" / "2440MHz_tagX_1.png").exists()
