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

    # DCスパイク除去（DCオフセット補正 / DC offset correction）。受信の最も入口で
    #   複素IQの平均(複素DCオフセット)を引き、取得帯域の中央(オフセット0Hz)に出る
    #   ゼロIF受信機由来の時間不変スパイク(DCスパイク)を消す。捨てるのではなく信号
    #   から除く方式（他のSDRソフトと同様）。実機(HackRF)では既定で有効。CLI から
    #   --no-dc-removal で無効化できる。合成(Sim)は元々DCが無いので Backend 側で
    #   既定オフ（診断用 --sim-dc-spike で注入したものを --dc-removal で消せる）。
    dc_removal: bool = True


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
# 滞在観測（dwell 観測の長時間化）
#   1つの対象周波数に一定時間留まり、IQ を複数回取得してバーストを待ち受ける。
#   2.4GHz の WiFi/BT は数秒おきに一瞬しか出ないため、約13msの単発観測では
#   取り逃す。滞在して反復観測することで「持続性」を測れるようにする。
# ---------------------------------------------------------------------------
@dataclass
class DwellConfig:
    # 各対象帯に滞在する秒数（スイープ全体ではなく1帯あたり）。
    dwell_seconds: float = 10.0
    # 滞在中の観測の間隔（秒）。バーストを跨いで待ち受けるため間を空ける。
    obs_interval_s: float = 0.5
    # 滞在中の最低観測回数（持続率の母数を確保。短い滞在でも下限を保証）。
    min_observations: int = 4
    # 暴走防止の観測回数上限（dwell_seconds が長くても打ち切る）。
    max_observations: int = 80


# ---------------------------------------------------------------------------
# 品質ゲート（保存条件）— 既定は「厳しめ」に倒す（量より質）
#   低品質・断片的なキャプチャを土台にすると学習の基準が崩れる。一瞬かすった
#   だけの単発や受信機由来スプリアスは保存しない。しきい値はすべてここで調整可。
# ---------------------------------------------------------------------------
@dataclass
class QualityConfig:
    enabled: bool = True
    # 1観測で「はっきり検出」とみなす SNR 下限(dB)。収集の足切りより厳しめ。
    detect_snr_db: float = 10.0
    # 持続性: 滞在中にこの回数以上はっきり検出され、かつ持続率がこれ以上。
    #   一瞬かすっただけの単発（detections 不足/低持続率）は破棄する。
    min_detections: int = 3
    min_persistence: float = 0.34
    # 極細スプリアス対策: 占有帯域がこれ未満を「極細」とみなす。
    #   ただし幅だけで切らず、下の steady 判定（同一強度で居座る）と併用して、
    #   バースト性のある正規の狭帯域信号(BLE等)を誤って捨てない。
    narrow_bw_hz: float = 0.7e6
    # 「同一強度で居座る」判定: SNR のばらつきがこれ以下（ほぼ一定）かつ
    #   持続率がこれ以上（ほぼ常時）なら、受信機内部スプリアスを疑う。
    #   narrow_bw と組み合わさったときのみ破棄する（広帯域の常時信号は正規）。
    spur_snr_std_max: float = 1.5
    spur_persistence_min: float = 0.9
    # クロスターゲットのコムスプリアス検出（等間隔・同一強度の細いピーク列）。
    #   アンテナ無しでも出る固定パターン＝複数帯にまたがる等間隔・同強度ピーク。
    comb_spacing_tol_hz: float = 0.15e6   # 隣接間隔の一致許容
    comb_power_tol_db: float = 2.0        # ピーク強度の一致許容
    comb_min_run: int = 3                 # この本数以上並んだら全て破棄
    # DCスパイク除外（DCオフセット由来の中央スパイク）。ゼロIF受信機(HackRF等)が
    #   取得帯域のちょうど中央(中心周波数=オフセット0Hz)に出す、本物ではない細い線。
    #   「中央集中(dc_excess)・時間不変(dc_excess の観測間ばらつき小)・細い」が揃った
    #   ときのみ破棄する。中央からオフセットした信号・広帯域信号(WiFi等)・時間変動する
    #   バースト(BLE等)は対象外（本物として残す）。
    dc_band_hz: float = 60e3              # 中央バンドの半幅（|offset|<=これをDCとみなす）
    dc_side_hz: float = 0.8e6            # 比較する両脇リングの外縁（dc_band〜これ）
    dc_excess_min_db: float = 12.0       # 中央が両脇よりこれ以上高ければ「中央集中」
    dc_excess_std_max: float = 3.0       # 観測間の excess ばらつきがこれ以下なら「時間不変」


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
    dwell: DwellConfig = field(default_factory=DwellConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    bands: list[Band] = field(default_factory=lambda: list(BAND_PLAN))
