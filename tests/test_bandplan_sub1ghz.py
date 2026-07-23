"""
test_bandplan_sub1ghz.py — 1GHz以下バンド（指示書 Part B）を固定する。

6) 新バンドが _match_band で正しく引ける（80MHz→FM放送, 922MHz→LPWA 等）
7) 既存31バンドの一致結果が不変（特に 2.4GHz ISM）。900MHz は従来どおりバンド外。
8) エントリの妥当性（f_lo<f_hi、既存最下限960MHzより下＝既存と重複しない）
"""
import classify
from config import BAND_PLAN


# 新規追加した1GHz以下バンドの (代表周波数Hz, 期待バンド名)
SUB1_CASES = [
    (80e6,  "FM放送"),
    (127e6, "航空無線(VHF/AM)"),
    (160e6, "業務/防災無線(VHF)"),
    (314e6, "特定小電力 315MHz帯"),
    (428e6, "特定小電力 426MHz帯"),
    (435e6, "アマチュア 430MHz帯"),
    (600e6, "地上デジタルTV(UHF)"),
    (922e6, "LPWA 920MHz帯"),
]
NEW_NAMES = {name for _, name in SUB1_CASES}

# 既存31バンド（Part B 追加前のスナップショット）。これらは不変であること。
ORIGINAL_31_NAMES = {
    "GPS L5 / QZSS L5", "GPS L2", "GPS L1 / QZSS L1", "GLONASS L1",
    "Cellular B3 DL 1.8G", "Cellular B1 DL 2.1G", "Cellular B7/n7 2.6G",
    "5G NR n77/n78 3.5G", "5G NR n79 4.7G", "ISM 2.4G (WiFi/BT)",
    "WiFi 5G W52/W53", "WiFi 5G W56 (DFS)", "ISM 5.8G (FPV/ETC)",
    "WiFi 6E edge", "Aero/Radar S-band", "Ham 1.2G", "Ham 2.4G",
    "航空無線航行 DME/TACAN", "Lバンド各種レーダー", "Cバンド気象/航行レーダー",
    "電波高度計等(航空)", "インマルサット/移動体衛星↓", "Iridium/移動体衛星↑",
    "気象ラジオゾンデ", "衛星・ロケット追跡管制↑", "衛星・ロケット追跡管制↓",
    "移動体衛星通信", "ITS DSRC/ETC", "産業用ドローン(5.7G)",
    "FPVドローン映像(5.8G)", "ロボット用無線(2.4G)",
}

# 既存バンドの代表一致（Part B で不変であること・2.4GHz を含む）
EXISTING_MATCH_CASES = [
    (1176.5e6, "GPS L5 / QZSS L5"),
    (1575.42e6, "GPS L1 / QZSS L1"),
    (2140e6, "Cellular B1 DL 2.1G"),
    (2437e6, "ISM 2.4G (WiFi/BT)"),     # 2.4GHz 最重要（518件のhuman確定資産）
    (2480e6, "ISM 2.4G (WiFi/BT)"),
    (3550e6, "5G NR n77/n78 3.5G"),
    (5180e6, "WiFi 5G W52/W53"),
    (5805e6, "ISM 5.8G (FPV/ETC)"),
]


# ---- 6) 新バンドが正しく引ける ----
def test_new_sub1ghz_bands_match():
    for f, name in SUB1_CASES:
        b = classify._match_band(f, BAND_PLAN)
        assert b is not None and b.name == name, f"{f/1e6:.1f}MHz → {b}"


# ---- 7) 既存が不変 ----
def test_existing_band_names_all_present():
    """既存31バンド名が全て残っている（削除・改名なし）。"""
    names = {b.name for b in BAND_PLAN}
    missing = ORIGINAL_31_NAMES - names
    assert not missing, f"既存バンドが消えた: {missing}"


def test_existing_band_matches_unchanged():
    """既存バンドの一致結果が Part B 追加後も不変（特に 2.4GHz ISM）。"""
    for f, name in EXISTING_MATCH_CASES:
        b = classify._match_band(f, BAND_PLAN)
        assert b is not None and b.name == name, f"{f/1e6:.1f}MHz → {b}"


def test_900mhz_still_outside_band_plan():
    """900MHz は依然どのバンドにも属さない（test_classify の前提を維持）。"""
    assert classify._match_band(900e6, BAND_PLAN) is None


# ---- 8) 妥当性 ----
def test_all_bands_flo_lt_fhi():
    for b in BAND_PLAN:
        assert b.f_lo < b.f_hi, f"f_lo>=f_hi: {b.name}"


def test_new_bands_below_existing_minimum_no_overlap():
    """新規は全て f_hi<=既存最下限(960MHz)＝既存31バンドと周波数が重ならない。"""
    existing = [b for b in BAND_PLAN if b.name not in NEW_NAMES]
    existing_min_flo = min(b.f_lo for b in existing)
    assert existing_min_flo == 960.0e6            # 既存最下限は DME/TACAN 960MHz
    for b in BAND_PLAN:
        if b.name in NEW_NAMES:
            assert b.f_hi <= existing_min_flo, f"{b.name} が既存下限に食い込む"


def test_new_bands_mutual_overlap_at_most_shared_endpoint():
    """新規バンド同士は端点共有までは許容するが内部で重ならない（区間が交差しない）。"""
    news = sorted((b for b in BAND_PLAN if b.name in NEW_NAMES),
                  key=lambda b: b.f_lo)
    for a, c in zip(news, news[1:]):
        assert a.f_hi <= c.f_lo, f"{a.name} と {c.name} が重複"
