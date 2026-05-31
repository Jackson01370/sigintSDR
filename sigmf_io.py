"""sigscan データ契約 (2/2): SigMF 入出力。

SigMF Recording = `<name>.sigmf-data`（生IQ）+ `<name>.sigmf-meta`（JSON）。
自動ラベルは annotations に格納する。これにより TorchSig / IntelLabs
RFML-Framework など SigMF を読む既存ツールと最初から互換になる。

datatype は cf32_le（complex64 LE = I,Q interleaved float32）。numpy の
complex64.tofile はこの並びでそのまま書ける。
"""
from __future__ import annotations
import json
import datetime
import numpy as np

SIGMF_VERSION = "1.0.0"
DATATYPE = "cf32_le"


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def write_recording(path_base: str, iq, center_hz: float, sample_rate: float,
                    annotations: list[dict] | None = None,
                    hw: str = "HackRF One", recorder: str = "sigscan",
                    description: str = "", extra_global: dict | None = None) -> dict:
    """SigMF Recording を2ファイルで書き出す。

    annotations: 各要素 {freq_lower_edge, freq_upper_edge, label,
                 comment?, confidence?, method?, snr_db?}。
                 サンプル範囲はキャプチャ全体に付与する。
    returns: meta dict
    """
    iq = np.asarray(iq, dtype=np.complex64)
    iq.tofile(path_base + ".sigmf-data")     # complex64 -> cf32_le

    glob = {
        "core:datatype": DATATYPE,
        "core:sample_rate": float(sample_rate),
        "core:version": SIGMF_VERSION,
        "core:hw": hw,
        "core:recorder": recorder,
        "core:num_channels": 1,
    }
    if description:
        glob["core:description"] = description
    if extra_global:
        glob.update(extra_global)

    captures = [{
        "core:sample_start": 0,
        "core:frequency": float(center_hz),
        "core:datetime": _utc_now_iso(),
    }]

    n = int(iq.size)
    anns: list[dict] = []
    for a in (annotations or []):
        ann = {"core:sample_start": 0, "core:sample_count": n}
        if "freq_lower_edge" in a:
            ann["core:freq_lower_edge"] = float(a["freq_lower_edge"])
        if "freq_upper_edge" in a:
            ann["core:freq_upper_edge"] = float(a["freq_upper_edge"])
        if a.get("label") is not None:
            ann["core:label"] = str(a["label"])
        if a.get("comment"):
            ann["core:comment"] = str(a["comment"])
        for k in ("confidence", "method", "snr_db"):   # ラベリング根拠を残す
            if a.get(k) is not None:
                ann[f"sigscan:{k}"] = a[k]
        anns.append(ann)

    meta = {"global": glob, "captures": captures, "annotations": anns}
    with open(path_base + ".sigmf-meta", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def read_recording(path_base: str) -> tuple[np.ndarray, dict]:
    """SigMF Recording を読み込み (iq, meta) を返す。"""
    with open(path_base + ".sigmf-meta") as f:
        meta = json.load(f)
    dt = meta["global"].get("core:datatype", DATATYPE)
    if dt != DATATYPE:
        raise ValueError(f"未対応 datatype: {dt}（cf32_le のみ対応）")
    iq = np.fromfile(path_base + ".sigmf-data", dtype=np.complex64)
    return iq, meta


def annotation_from_result(measurement: dict, result) -> dict:
    """measure_signal の結果 + 分類結果 から SigMF annotation 1件を作る。"""
    center = measurement["center_hz"]
    bw = max(measurement.get("bw_hz", 0.0), 1.0)
    return dict(
        freq_lower_edge=center - bw / 2,
        freq_upper_edge=center + bw / 2,
        label=result.label,
        comment=result.notes,
        confidence=round(float(result.confidence), 3),
        method=result.method,
        snr_db=round(float(measurement.get("snr_db", 0.0)), 1),
    )
