"""LLM Vision 段のプロンプト構築と参照信号カタログ。

カタログは SigIDWiki / Artemis（sigidwiki.com, www.disk91.com/Artemis）に公開されている
一般的な信号特徴（周波数・帯域幅・変調・視覚的特徴）を **参照** として要約したもの。
LLM が訓練済み知識を使ってラベリングする際の「鞍点」になる事前情報を与え、
バンドプラン外の信号でも合理的な候補を出させる狙い。

注: ここに置く情報は教育的な要約に留め、座標やキーワードは sigscan の他モジュール
（`config.BAND_PLAN`・`classify.SIGNAL_DB`）と矛盾しないこと。
"""
from __future__ import annotations
import json
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 参照カタログ（1〜6GHz 中心）
#
# (band_hint, freq_mhz_range, typical_bw, modulation, visual_marker, label)
# ---------------------------------------------------------------------------
SIGNAL_CATALOG: list[dict] = [
    # GNSS
    dict(label="GPS L1 C/A",         freq_mhz=(1574, 1577),  bw="2 MHz (C/A) / 20 MHz スカートあり",
         modulation="BPSK / CDMA 拡散",
         visual="床下〜ノイズ床ぎりぎり, 広がりはあるが構造が見えにくい"),
    dict(label="QZSS L1",            freq_mhz=(1574, 1577),  bw="〜20 MHz",
         modulation="BPSK 拡散",
         visual="GPS L1 とほぼ同位相帯, わずかに強い場合あり（日本上空）"),
    dict(label="GPS L2C",            freq_mhz=(1227, 1228),  bw="2 MHz",
         modulation="BPSK 拡散",
         visual="極微弱・狭帯域"),
    dict(label="GPS L5",             freq_mhz=(1175, 1177),  bw="24 MHz",
         modulation="BPSK 拡散",
         visual="広く低い拡散, 端がなだらか"),
    dict(label="GLONASS L1",         freq_mhz=(1598, 1606),  bw="0.5 MHz × 多チャネル",
         modulation="FDMA",
         visual="等間隔の細い縦縞 (FDMA channels)"),
    # Cellular
    dict(label="LTE/UMTS DL B1 (2.1G)", freq_mhz=(2110, 2170), bw="5/10/15/20 MHz ブロック",
         modulation="OFDM (DL)",
         visual="矩形のスペクトル, 端が垂直, 1 ms フレームで時間的に連続"),
    dict(label="LTE DL B3 (1.8G)",   freq_mhz=(1805, 1880),  bw="5-20 MHz",
         modulation="OFDM",
         visual="矩形ブロック, B1 と同様"),
    dict(label="LTE/NR DL B7/n7 (2.6G)", freq_mhz=(2620, 2690), bw="5-20 MHz",
         modulation="OFDM",
         visual="矩形ブロック"),
    dict(label="5G NR n77/n78 (3.5G)", freq_mhz=(3300, 3800), bw="40-100 MHz",
         modulation="OFDM TDD",
         visual="広い矩形, TDD バーストで時間方向に明暗の縞 (DL/UL)"),
    dict(label="5G NR n79 (4.7G)",   freq_mhz=(4500, 4900),  bw="40-100 MHz",
         modulation="OFDM TDD",
         visual="広い矩形, TDD 縞"),
    # WiFi / BT / ISM
    dict(label="WiFi 2.4 GHz (802.11g/n/ax 20/40 MHz)", freq_mhz=(2400, 2483.5),
         bw="20 / 40 MHz", modulation="OFDM",
         visual="矩形ブロック, バースト的, 中央キャリア無し"),
    dict(label="Bluetooth Classic / BLE (FH)", freq_mhz=(2400, 2483.5),
         bw="1 MHz (Classic) / 2 MHz (BLE)", modulation="GFSK/PSK ホッピング",
         visual="細い縦線が時間で位置をジャンプ (FHSS), 平均化すると ISM 全体にうっすら"),
    dict(label="Zigbee / IEEE 802.15.4", freq_mhz=(2405, 2480), bw="2 MHz",
         modulation="OQPSK",
         visual="狭めの矩形, 16 チャネルのいずれか"),
    dict(label="電子レンジ漏洩 (2.4G)", freq_mhz=(2400, 2483.5), bw="〜20 MHz",
         modulation="非変調 漏洩",
         visual="50/60 Hz の周期的なバースト, 時間軸でドット状"),
    dict(label="WiFi 5 GHz (W52/W53/W56)", freq_mhz=(5150, 5725), bw="20/40/80/160 MHz",
         modulation="OFDM",
         visual="広矩形, バースト"),
    dict(label="気象レーダ (W56 DFS)", freq_mhz=(5250, 5725), bw="数 MHz, 短パルス",
         modulation="パルスレーダ",
         visual="非常に短いパルス, 時間軸に線, WiFi と混在"),
    dict(label="ETC / DSRC", freq_mhz=(5770, 5850), bw="〜5 MHz",
         modulation="ASK/OOK 系",
         visual="狭帯域, 短いバースト"),
    dict(label="FPV ドローン映像 (5.8 GHz)", freq_mhz=(5725, 5875), bw="6-20 MHz",
         modulation="アナログ FM 映像 / DJI OcuSync",
         visual="アナログ: 中央キャリアと側波帯のお椀型. DJI: OFDM 矩形バースト"),
    # Misc
    dict(label="Aero/監視レーダ S-band", freq_mhz=(2700, 2900), bw="2-5 MHz",
         modulation="短パルス",
         visual="極短パルス, 時間方向に飛び石"),
    dict(label="Amateur (1.2/2.4 GHz)", freq_mhz=(1240, 2450), bw="変動",
         modulation="多様 (FM/SSB/digital)",
         visual="低 duty, 局所的"),
    # Generic
    dict(label="未識別の OFDM 様", freq_mhz=(1000, 6000), bw="〜",
         modulation="OFDM 様",
         visual="平坦な矩形, 端が立つ"),
    dict(label="未識別の狭帯域", freq_mhz=(1000, 6000), bw="< 200 kHz",
         modulation="FSK/PSK 等",
         visual="細い縦線"),
    dict(label="未識別のチャープ/掃引", freq_mhz=(1000, 6000), bw="変動",
         modulation="チャープ / 周波数掃引",
         visual="斜めの線（時間に対して周波数が変化）"),
]


# ---------------------------------------------------------------------------
# プロンプト本文
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "あなたは無線スペクトル解析の専門家です。与えられたスペクトログラム画像と中心"
    "周波数・帯域文脈を見て、最も可能性の高い信号サービスを 1 つ特定してください。"
    "ラベルは参照カタログ (SigIDWiki / Artemis 由来の一般知識) に整合させ、"
    "推測の根拠を画像から読み取った視覚的特徴で説明してください。"
    "確度が低い場合は confidence を下げ、未知/雑音の可能性が高ければ"
    "label を '未識別信号' か 'ノイズ/フロア変動' にしてください。"
    "出力は厳密に JSON のみ。前後に説明や ``` を付けないこと。"
)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label":      {"type": "string"},
        "confidence": {"type": "number"},
        "candidates": {"type": "array", "items": {"type": "string"}},
        "notes":      {"type": "string"},
        "rationale":  {"type": "string"},
    },
    "required": ["label", "confidence"],
}


@dataclass
class PromptContext:
    """LLM に渡す状況: 周波数・帯域幅・SNR・ルール段の所見・該当バンド情報。"""
    center_hz: float
    bw_hz: float
    snr_db: float
    occupied_frac: float = 0.0
    band_name: str = ""
    band_hint: str = ""
    rule_label: str = ""
    rule_confidence: float = 0.0
    rule_notes: str = ""
    rule_candidates: list[str] | None = None
    rep_summary: dict | None = None


def _format_catalog(center_mhz: float, window_mhz: float = 200.0) -> str:
    """周波数近傍のカタログだけ抜き出してテキスト化（プロンプト長削減）。"""
    lines: list[str] = []
    for entry in SIGNAL_CATALOG:
        lo, hi = entry["freq_mhz"]
        if hi < center_mhz - window_mhz or lo > center_mhz + window_mhz:
            # 近傍以外でも generic（未識別系）は残す
            if not entry["label"].startswith("未識別"):
                continue
        lines.append(
            f"- {entry['label']}  "
            f"[{entry['freq_mhz'][0]:.0f}-{entry['freq_mhz'][1]:.0f} MHz, "
            f"bw={entry['bw']}, mod={entry['modulation']}, "
            f"見た目={entry['visual']}]"
        )
    return "\n".join(lines) if lines else "(該当帯のカタログ無し)"


def build_user_text(ctx: PromptContext, window_mhz: float = 200.0) -> str:
    """添付画像とともに送る本文。"""
    center_mhz = ctx.center_hz / 1e6
    bw_mhz = ctx.bw_hz / 1e6
    catalog = _format_catalog(center_mhz, window_mhz)
    rule_part = (
        f"ルール段の所見: label='{ctx.rule_label}', conf={ctx.rule_confidence:.2f}, "
        f"notes='{ctx.rule_notes}'"
        if ctx.rule_label else "ルール段の所見: なし"
    )
    if ctx.rule_candidates:
        rule_part += " / candidates=" + ", ".join(ctx.rule_candidates[:4])

    band_part = (
        f"バンドプラン該当: {ctx.band_name} ({ctx.band_hint})"
        if ctx.band_name else "バンドプラン該当: なし（バンド外）"
    )

    rep = ctx.rep_summary or {}
    rep_part = (
        f"画像生成設定: STFT nfft={rep.get('nfft','?')} hop={rep.get('hop','?')}, "
        f"取得レート={rep.get('rate_hz','?')} Hz, "
        f"画像サイズ={rep.get('img',[256,256])}"
    )

    return (
        "## 計測条件\n"
        f"- 中心周波数: {center_mhz:.3f} MHz\n"
        f"- 推定占有帯域: {bw_mhz:.3f} MHz  (occupied_frac={ctx.occupied_frac:.2f})\n"
        f"- SNR: {ctx.snr_db:.1f} dB\n"
        f"- {band_part}\n"
        f"- {rule_part}\n"
        f"- {rep_part}\n"
        "\n## 周辺の参照カタログ (SigIDWiki / Artemis 由来の一般知識)\n"
        f"{catalog}\n"
        "\n## 出力形式 (JSON only)\n"
        "```\n"
        '{\n'
        '  "label":      "<信号サービスの短い日本語ラベル>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "candidates": ["<次点>", "<次々点>"],\n'
        '  "notes":      "<日本語の補足>",\n'
        '  "rationale":  "<画像から読んだ視覚的特徴を1〜2文>"\n'
        '}\n'
        "```\n"
        "※ JSON のみ出力。コードフェンス・前置き・補足説明は一切付けないこと。"
    )


def parse_response(text: str) -> dict | None:
    """LLM 応答テキストから JSON を抽出。``` フェンス・前後の散文を許容。"""
    if not text:
        return None
    t = text.strip()
    # ```json ... ``` フェンスの除去
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        if t.endswith("```"):
            t = t[: -3]
    # 最初に { が出てから最後の } まで切り出す
    lo = t.find("{")
    hi = t.rfind("}")
    if lo < 0 or hi <= lo:
        return None
    blob = t[lo: hi + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "label" not in obj or "confidence" not in obj:
        return None
    return obj
