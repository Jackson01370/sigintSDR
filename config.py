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

    # dwell オフセットチューニング（ゼロIF受信機の定石）。dwell 収集時にチューナー中心を
    #   この Hz だけずらし、狙った狭帯域の獲物を取得帯域の中央(DC=0Hz)から避ける。獲物が
    #   DC に乗ると、DC残留線や DC位置の固定スパイク(40MHzクロック高調波・16MHz櫛等)と重なり
    #   dc-spike ゲートで構造的に落ちるため。既定 0=無効（完全に従来挙動）。有効化は CLI
    #   --dwell-offset-hz。サーベイ(sweep_power)には適用しない＝dwell 収集経路のみ。
    dwell_offset_hz: float = 0.0
    # オフセットを適用するターゲット帯域幅の上限(Hz)。これより広い帯域(WiFi等)にオフセットを
    #   掛けると信号端が取得帯域(窓)の外へはみ出すため、狭帯域ターゲットにのみ適用する。
    dwell_offset_max_bw_hz: float = 8e6

    # DC残留ガード（ゼロIF受信機の DC 残留対策・dwell 経路のみ）。dwell の測定で、
    #   チューナー中心 ± dc_guard_hz(Hz) の範囲を「窓の主役」候補から除外し、残った中で
    #   最強の信号（＝次に強い本物）を拾う。1GHz以下では、チューナ中心に張り付く DC 残留
    #   （LO漏れのドリフトで DC 除去後も残る細い線）が常に窓の最強となって本物の放送局を
    #   奪うため。判定は取得IQの中心相対(0Hz=チューナ中心)ゆえ構造的にチューナ相対で、
    #   絶対周波数固定のスプリアス(2400/2440/2480の40MHz高調波等)とは別物。offset併用時は
    #   オフセット適用後の実チューナ中心が基準になる（capture 中心が動けばガードも動く）。
    #   既定 0=無効（完全に従来挙動＝1ビットも変わらない）。有効化は CLI --dc-guard-hz。
    #   サーベイ(sweep_power)には適用しない＝dwell 収集経路のみ（DC 残留は capture_iq に
    #   だけ現れ sweep_power には載らない・dwell_offset_hz と同じ流儀）。
    dc_guard_hz: float = 0.0


def dwell_samples_for_ms(capture_ms: float, rate_hz: float) -> int:
    """1回のドウェル取得(スナップショット)の長さ ms を IQ サンプル数に変換する。

    n = round(capture_ms/1000 * rate_hz)。rate は SDRConfig.dwell_rate_hz(既定 20MSPS)。
    例) 13.107ms→262144(既定 2^18) / 300ms→6,000,000 / 400ms→8,000,000(≈64MB, complex64)。

    duty 審判(cnntrain.dutyprobe)が BLE adv 間隔(20-100ms)の隙間を分解するには
    snapshot>=300ms が要る。既定の 13ms では全件 inconclusive になる（在時率が結論不能）。
    capture_ms<=0 は不正(ValueError)。返り値は 1 以上を保証する。
    """
    if not (capture_ms > 0):
        raise ValueError(f"capture_ms は正の値である必要があります: {capture_ms!r}")
    return max(int(round(capture_ms / 1000.0 * rate_hz)), 1)


# ---------------------------------------------------------------------------
# スキャン制御パラメータ
# ---------------------------------------------------------------------------
@dataclass
class ScanConfig:
    start_hz: float = 1.0e9
    stop_hz: float = 6.0e9

    # 帯域フォーカス: True かつ start/stop 指定時、ターゲット候補の合流点
    #   （_build_targets の出口）で中心周波数が [start, stop] 外の候補を1点だけ
    #   除外し、指定帯域に張り付く。これでバンドプラン巡回由来の範囲外目標と
    #   サーベイ端の食み出し検出の両方が同じ関所で消える。既定 False（従来どおり
    #   バンドプラン全巡回・挙動不変）。CLI --focus で有効化。範囲は既存の
    #   start/stop を流用し、新たな範囲指定は作らない。
    band_focus: bool = False

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
# CNN 分類器（3段分類器の 2 段目＝監査役）— 既定 OFF（挙動不変）
#   有効時のみ、滞在観測で保存候補になった信号の IQ を凍結 spec.render 経由で
#   CNN に通し、ルール × CNN × バンドプラン文脈の整合チェックで確信度を調整する。
#   checkpoint はディレクトリ（中の checkpoint.pt を補完）でもファイルでも可。
# ---------------------------------------------------------------------------
@dataclass
class CNNConfig:
    enabled: bool = False
    checkpoint: str = "runs/m2_5"


# ---------------------------------------------------------------------------
# バンドプラン（1〜6GHz・日本の割当を考慮）
#   priority: 大きいほどドウェル頻度を上げる
#   hint    : ルールベース分類のヒント（表示/LLM文脈用の自由文字列。機械可読の
#             用途/方式ラベルではない。"[仮説] use=.../mod=..." は地図上の予測で
#             あって確定ラベルではなく、classify の確信度決定には用いない）
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
    # R3: 上限を 3400MHz へ拡張（一次情報「2700-3400 船舶の航行用等レーダー」に合わせる）。
    #   name は classify.SIGNAL_DB の "Aero/Radar" 部分一致が参照するため不変に保つ
    #   （リネームすると精緻化ルールが無言で外れる）。priority も既存どおり据え置き
    #   （task(1) は f_hi 拡張のみ指示。受信現実性による再調整は将来）。
    Band("Aero/Radar S-band",   2700.0e6, 3400.0e6, 1, "[仮説] use=Radar mod=Pulse/Chirp-LFM / 船舶航行用・空港監視ASR・気象等のSバンドレーダ, 一次情報2700-3400"),
    # --- アマチュア ---
    Band("Ham 1.2G",           1240.0e6, 1300.0e6, 1, "アマチュア"),
    Band("Ham 2.4G",           2400.0e6, 2450.0e6, 1, "アマチュア(ISM重複)"),

    # =======================================================================
    # 追加バンド（BANDPLAN_PROPOSAL §3 A/B/C）— 1〜6GHz「完全な地図」化。
    #   hint 中の "[仮説] use=.../mod=..." は総務省一次情報に基づく地図上の予測で
    #   あり、確定ラベルではない。hint は classify では notes / LLM 文脈にしか
    #   流れず（label/confidence の決定経路は不変）、確信度<0.7→Unknown→review.py
    #   での人間判断という確定フローはそのまま。用途/方式の機械可読マッピング
    #   （BAND_LABEL_HINT 等の参照表）は最小実装方針により今回見送り＝将来課題。
    #   既存バンド(ISM2.4G/5.8G, WiFi W53/W56 等)と重複/隣接するものは「同じ帯域に
    #   異なる信号が同居する」現実を地図化するため意図的に残す。二重処理は起きない:
    #   _match_band は priority(strict >)で1つに決め、巡回の _build_targets.add() は
    #   近接 center を吸収する。
    # =======================================================================
    # A. レーダ系（最大の穴）
    Band("航空無線航行 DME/TACAN", 960.0e6, 1215.0e6, 1, "[仮説] use=Radar mod=Pulse / DME/TACAN測距(航空機-地上局), 一次情報960-1400"),
    Band("Lバンド各種レーダー", 1215.0e6, 1400.0e6, 1, "[仮説] use=Radar mod=Pulse / 航空路監視ARSR等, 準天頂・地球探査と共用"),
    Band("Cバンド気象/航行レーダー", 5250.0e6, 5372.5e6, 2, "[仮説] use=Radar mod=Pulse/Chirp-LFM / 公共機関の気象レーダ, WiFi W53と重複しDFS対象"),
    Band("電波高度計等(航空)", 4200.0e6, 4400.0e6, 1, "[仮説] use=Radar mod=FMCW / 航空機電波高度計"),
    # B. 衛星・移動体衛星系
    Band("インマルサット/移動体衛星↓", 1525.0e6, 1559.0e6, 1, "[仮説] use=SatComm mod=CW-Narrow/OFDM / 移動体衛星通信(微弱), 一次情報1525-1559"),
    Band("Iridium/移動体衛星↑", 1618.25e6, 1660.5e6, 1, "[仮説] use=SatComm mod=CW-Narrow / Iridium 1616-1626.5(微弱・専用アンテナ要)"),
    Band("気象ラジオゾンデ", 1670.0e6, 1690.0e6, 1, "[仮説] use=SatComm mod=CW-Narrow / 気球による高層気象観測, 一次情報1670-1690"),
    Band("衛星・ロケット追跡管制↑", 2025.0e6, 2110.0e6, 1, "[仮説] use=SatComm mod=OFDM/PhaseCoded / S帯TT&C(上り), 一次情報2025-2110"),
    Band("衛星・ロケット追跡管制↓", 2200.0e6, 2300.0e6, 1, "[仮説] use=SatComm mod=OFDM/PhaseCoded / S帯TT&C(下り), 一次情報2200-2300"),
    Band("移動体衛星通信", 2500.0e6, 2535.0e6, 1, "[仮説] use=SatComm mod=OFDM / 移動体衛星通信サービス, 一次情報2500-2535"),
    # C. 特殊業務・その他
    Band("ITS DSRC/ETC", 5770.0e6, 5850.0e6, 2, "[仮説] use=ETC-DSRC mod=OFDM/CW-Narrow / 高速料金所ETC・路車間, 道路近くで受信現実性あり"),
    Band("産業用ドローン(5.7G)", 5650.0e6, 5755.0e6, 1, "[仮説] use=Drone-FPV mod=OFDM / ロボット用無線(無人移動体画像伝送), 1W"),
    Band("FPVドローン映像(5.8G)", 5725.0e6, 5875.0e6, 1, "[仮説] use=Drone-FPV mod=OFDM/FMCW / FPVレース等映像伝送, ISM5.8Gと重複"),
    Band("ロボット用無線(2.4G)", 2483.5e6, 2494.0e6, 1, "[仮説] use=Drone-FPV mod=OFDM / 無人移動体画像伝送, 2.4G ISM上端に隣接"),

    # =======================================================================
    # 1GHz以下（指示書「1GHz以下の受信対応」Part B）— Diamond D1300AM の受信範囲
    #   25-1300MHz 内。従来 BAND_PLAN は 1〜6GHz で範囲を切っていたため 1GHz以下が
    #   丸ごと未記載で、FM放送・LPWA 等が全て「未識別信号(バンドプラン外)」表示に
    #   なっていた。ここで地図を 1GHz以下へ延長する。全エントリは f_hi<=928MHz で、
    #   既存最下限バンド(960MHz DME/TACAN)より下＝既存31バンドと重複せず、既存の
    #   一致結果(_match_band)を変えない。priority は控えめ(1)で、既定 1-6GHz 巡回への
    #   影響を最小化（--focus + --start/--stop で 1GHz以下に張り付く運用を想定）。
    #   1GHzでレンジを切っていた経緯: BANDPLAN_PROPOSAL §7 参照。
    #   ※ 1GHz以下では LNA(2.4GHz専用フィルタ内蔵)を外す運用（コード変更不要）。
    # =======================================================================
    Band("FM放送", 76.0e6, 95.0e6, 1, "[地図] use=Broadcast mod=FM / 国内FM放送(V-Low含む). 連続・広帯域・常時"),
    Band("航空無線(VHF/AM)", 118.0e6, 137.0e6, 1, "[地図] use=Aero-Voice mod=AM / 航空管制VHF音声. D1300AMはAM対応"),
    Band("業務/防災無線(VHF)", 150.0e6, 174.0e6, 1, "[地図] use=LandMobile mod=FM/Digital / 陸上移動業務・消防防災等"),
    # 300MHz帯 特定小電力: 国内割当は 312-315.25MHz(テレメータ/テレコントロール)＋315MHz
    #   RKE。提案 315-316 は割当上端(315.25)より上に外れるため 312-316 に補正（下端を
    #   実割当の312へ、315MHz帯リモコンを含める）。
    Band("特定小電力 315MHz帯", 312.0e6, 316.0e6, 1, "[地図] use=SRD mod=ASK/OOK / リモコン・テレメータ(300MHz帯特定小電力). 間欠・狭帯域"),
    # 400MHz帯 特定小電力: 426/429MHz帯(429MHz帯上端~429.7375)。提案 426-430 は上端が
    #   アマチュア430MHz帯と接触するため 426-429.75 に補正（重複回避・実割当に整合）。
    Band("特定小電力 426MHz帯", 426.0e6, 429.75e6, 1, "[地図] use=SRD mod=FM/Digital / 426/429MHz帯特定小電力(音声・データ). 間欠・狭帯域"),
    Band("アマチュア 430MHz帯", 430.0e6, 440.0e6, 1, "[地図] use=Amateur mod=FM/SSB/Digital / 国内アマチュア70cm帯"),
    Band("地上デジタルTV(UHF)", 470.0e6, 710.0e6, 1, "[地図] use=Broadcast mod=OFDM(ISDB-T) / 地デジ物理ch13-52. 連続・広帯域・常時"),
    Band("LPWA 920MHz帯", 920.0e6, 928.0e6, 1, "[地図] use=LPWA/RFID mod=LoRa/FSK/Wi-SUN / 国内920MHz帯(RFID/センサ網). 間欠・狭帯域"),
]


# ---------------------------------------------------------------------------
# バンド別 CNN ルーティング表（案Y）— バンド名 → 専門家 checkpoint
#   単一グローバル checkpoint（CNNConfig.checkpoint＝汎用 runs/m2_5・方式軸5クラス）を
#   全バンドに一律適用する既定に対し、「専門家 CNN があるバンドだけ」ここに書いて
#   上書きする。ここに無いバンドは従来どおり汎用へフォールバック（後方互換）。
#
#   * キーは BAND_PLAN の **実際のバンド name** に一致させる（_match_band が返す name）。
#   * 値は checkpoint のディレクトリ or ファイル（scheduler が中の checkpoint.pt を補完）。
#   * config.Band dataclass は **変更しない**（31バンド全部にモデル枠を足さない＝案Y）。
#     将来 5GHz 専門家等を足すときは、ここに 1 行追加するだけで済む。
#   * 空 dict なら全バンド汎用＝現状と完全に同一挙動。
#
#   専門家は **監査（audit）専用**。ここでの切替は「どの CNN で監査するか」だけで、
#   ラベル確定フロー（review.py の人手○×）には一切触れない（Pattern A を踏まない）。
BAND_CNN_ROUTES: dict[str, str] = {
    "ISM 2.4G (WiFi/BT)": "runs/ism24_v2",   # 2.4GHz ISM 専門家（用途3クラス）
}


@dataclass
class Config:
    sdr: SDRConfig = field(default_factory=SDRConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    dwell: DwellConfig = field(default_factory=DwellConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    bands: list[Band] = field(default_factory=lambda: list(BAND_PLAN))
