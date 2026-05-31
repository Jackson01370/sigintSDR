"""sigscan 設定: SDR パラメータ・スイープ/ドウェル設定・1〜6GHz バンドプラン。

周波数はすべて Hz。HackRF One の対応範囲は 1MHz〜6GHz、瞬時帯域は最大 ~20MHz。
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# SDR / 取得パラメータ
# ---------------------------------------------------------------------------
@dataclass
class SDRConfig:
    # ドウェル（集中捕捉）時のサンプルレート。HackRF は ~20MSPS が上限。
    dwell_rate_hz: float = 20e6
    # 1回のドウェルで取得する IQ サンプル数（スペクトログラム/帯域幅測定用）。
    dwell_samples: int = 1 << 18          # 262144 ≒ 13ms @20MSPS
    # サーベイ（粗スイープ）1ステップあたりのサンプル数。
    survey_samples: int = 1 << 14         # 16384

    # ゲイン（HackRF: lna 0-40 step8 / vga 0-62 step2 / amp on/off）
    lna_gain: float = 24.0
    vga_gain: float = 20.0
    amp_on: bool = False

    # SoapySDR リチューン後の整定待ち（秒）。PLL ロック待ち。
    retune_settle_s: float = 0.015


# ---------------------------------------------------------------------------
# スキャン制御パラメータ
# ---------------------------------------------------------------------------
@dataclass
class ScanConfig:
    start_hz: float = 1.0e9
    stop_hz: float = 6.0e9

    # サーベイの周波数分解能（FFTビン幅相当）。粗くて速い設定。
    survey_bin_hz: float = 200e3
    # サーベイ1ステップでカバーする帯域（瞬時帯域以下に）。
    survey_step_hz: float = 18e6

    # 検出しきい値: ノイズ床 + これ(dB) を超えたら「アクティブ」。
    detect_threshold_db: float = 8.0
    # アクティブ帯とみなす最小帯域幅。これ未満のスパイクは無視。
    min_segment_bw_hz: float = 50e3

    # サーベイを再実行する間隔（秒）。この合間にドウェルを回す。
    survey_interval_s: float = 12.0
    # 1サイクルでドウェルするターゲット数の上限。
    max_dwell_per_cycle: int = 6

    # スペクトログラム PNG を保存するか（CNN/LLM 連携の前段）。
    save_spectrograms: bool = False
    spectrogram_dir: str = "captures"


# ---------------------------------------------------------------------------
# バンドプラン（1〜6GHz・日本の割当を考慮）
#   priority: 大きいほどドウェル頻度を上げる
#   hint    : ルールベース分類のヒント
# ---------------------------------------------------------------------------
@dataclass
class Band:
    name: str
    f_lo: float
    f_hi: float
    priority: int = 1
    hint: str = ""

    @property
    def center(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)

    @property
    def width(self) -> float:
        return self.f_hi - self.f_lo


BAND_PLAN: list[Band] = [
    # --- GNSS ---
    Band("GPS L5 / QZSS L5",   1176.0e6, 1177.0e6, 2, "CDMA拡散, 帯域~24MHz, 常時微弱"),
    Band("GPS L2",             1227.0e6, 1228.0e6, 1, "CDMA拡散"),
    Band("GPS L1 / QZSS L1",   1574.0e6, 1577.0e6, 3, "CDMA拡散, BPSK, 常時微弱・広帯域ノイズ状"),
    Band("GLONASS L1",         1598.0e6, 1606.0e6, 1, "FDMA"),
    # --- 携帯 sub-6（日本）---
    Band("Cellular B3 DL 1.8G", 1805.0e6, 1880.0e6, 2, "LTE OFDM, 5/10/15/20MHzブロック"),
    Band("Cellular B1 DL 2.1G", 2110.0e6, 2170.0e6, 3, "LTE/UMTS OFDM, 連続ブロック"),
    Band("Cellular B7/n7 2.6G", 2620.0e6, 2690.0e6, 2, "LTE/NR OFDM"),
    Band("5G NR n77/n78 3.5G",  3300.0e6, 3800.0e6, 3, "5G NR TDD, 100MHz級の広帯域"),
    Band("5G NR n79 4.7G",      4500.0e6, 4900.0e6, 2, "5G NR TDD（日本 4.5-4.9）"),
    # --- ISM / WiFi / BT ---
    Band("ISM 2.4G (WiFi/BT)",  2400.0e6, 2483.5e6, 3, "WiFi 20/40MHz・BLEホップ・Zigbee・電子レンジ"),
    Band("WiFi 5G W52/W53",     5150.0e6, 5350.0e6, 3, "WiFi 20/40/80/160MHz OFDM"),
    Band("WiFi 5G W56 (DFS)",   5470.0e6, 5725.0e6, 2, "WiFi + 気象レーダ(DFS)混在"),
    Band("ISM 5.8G (FPV/ETC)",  5725.0e6, 5875.0e6, 2, "FPVドローン映像・ETC/DSRC・コードレス"),
    Band("WiFi 6E edge",        5925.0e6, 6000.0e6, 1, "WiFi 6E 下端"),
    # --- レーダ・航空 ---
    Band("Aero/Radar S-band",   2700.0e6, 2900.0e6, 1, "航空監視レーダ等, 短パルス"),
    # --- アマチュア ---
    Band("Ham 1.2G",           1240.0e6, 1300.0e6, 1, "アマチュア"),
    Band("Ham 2.4G",           2400.0e6, 2450.0e6, 1, "アマチュア(ISM重複)"),
]


@dataclass
class Config:
    sdr: SDRConfig = field(default_factory=SDRConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    bands: list[Band] = field(default_factory=lambda: list(BAND_PLAN))
