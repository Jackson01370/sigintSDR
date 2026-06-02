"""eval-harness (1/3): 外部学習済みモデルのロード。

このモジュールは「答え合わせ用」の外部モデルを sigscan に **配線** するための
ロード関数を提供する。M1 はあくまで配線（モデルのロードと入出力アダプタ）であり、
実機キャプチャに対するドメインギャップ測定は後マイルストーン（CONTRACT.md §4）。

設計方針:
  * 重い依存（torch / torchsig / torchvision）は **遅延 import**。未導入なら例外を
    握りつぶさず `ModelUnavailable` で明示する（取得が重い/不可の環境向け）。
  * 重み(state_dict)は **明示パス or 環境変数 or 既知キャッシュ** から探す。
    見つからなければダウンロード手順を添えて `ModelUnavailable` を投げる。
  * torch が無い環境でも配線（adapter→推論→report）を検証できるよう、
    numpy だけで動く **reference stand-in** を用意する。これは学習済みでも
    外部モデルでもなく、`is_stand_in=True` / `is_pretrained=False` を必ず立てる。
    report 側はこのフラグを見てバナーで明示する（誠実さの防壁）。

--------------------------------------------------------------------------
外部モデルの出所・ライセンス・入力仕様（取得手順は eval/README.md に詳述）
--------------------------------------------------------------------------
TorchSig / Sig53（狭帯域分類）   github.com/TorchDSP/torchsig, torchsig.com
  * コード: MIT License。
  * モデル: EfficientNet-B4 / XCiT。入力 = 複素ベースバンド IQ 4096 サンプルを
    実部/虚部の 2ch にした [2, 4096]。出力 = 53 クラス。
  * Sig53 データセットは TorchSig が合成生成（500万サンプル / 53クラス）。
    → DeepSig の RadioML(RML2016/2018) とは別物。RadioML は CC BY-NC-SA 4.0
      （非商用）。混同しないこと。
  * 重み: バージョンにより torchsig.com / HuggingFace 配布。数十〜数百 MB。
    本ローダは重みファイルパスを要求し、無ければ取得手順付きで例外。

TorchSig / WBSig53（広帯域検出・セグメンテーション）
  * コード: MIT。複素スペクトログラム（~512x512）入力の検出/セグメンテーション網。
    出力 = 53 クラス。WBSig53 は 55万サンプル合成。

Qoherent spectrogram-segmentation   github.com/qoherent/spectrogram-segmentation
  * コード: MIT License（リポジトリ表記）。PyTorch + Lightning。
  * モデル: DeepLabv3 + MobileNetV3 backbone（torchvision 経由）。
  * 入力: 256x256 RGB スペクトログラム画像（3ch, ImageNet 正規化）。
  * 出力: セマンティックセグメンテーション（noise / 5G NR / LTE）。学習データは
    MathWorks の 5G/LTE Toolbox 合成スペクトログラム（公開）。
  * 重み: ノートブックで学習 or 配布チェックポイント。本ローダはパス要求。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ===========================================================================
# 例外（握りつぶさない・明示する）
# ===========================================================================
class ModelUnavailable(RuntimeError):
    """外部モデルをロードできない（依存未導入 / 重み欠如 / 構築失敗）。

    取得が重い・不可の環境ではこの例外で **明示的に** 失敗させる。呼び出し側で
    捕捉して stand-in に切り替えるかどうかを決める（黙って成功扱いにしない）。
    """


# ===========================================================================
# モデル仕様（adapter が参照する入力契約）
# ===========================================================================
@dataclass
class ModelSpec:
    """外部モデルの入力契約と出所メタ（adapters.py / report.py が参照）。"""
    name: str
    family: str                 # torchsig-narrowband | torchsig-wideband |
                                # qoherent-segmentation | reference-standin
    input_kind: str             # "iq" | "spectrogram" | "rgb-spectrogram"
    input_size: tuple           # iq: (n,) / 画像: (H, W)
    channels: int
    normalization: str          # 正規化の説明（ImageNet / unit-energy 等）
    num_classes: int
    class_names: list[str] = field(default_factory=list)
    # 出所・ライセンス（report バナー / README と一致させる）
    source: str = ""
    license: str = ""
    weights_source: str = ""
    weights_size: str = ""
    is_pretrained: bool = False
    is_stand_in: bool = False

    def summary(self) -> str:
        tag = " [STAND-IN]" if self.is_stand_in else (
            " [pretrained]" if self.is_pretrained else " [random-init]")
        return (f"{self.name}{tag}  family={self.family}  "
                f"in={self.input_kind}{self.channels}x{self.input_size}  "
                f"classes={self.num_classes}  license={self.license or '?'}")


@dataclass
class LoadedModel:
    """ロード済みモデル。`predict(x)` で adapter 出力に推論を回す統一窓口。"""
    spec: ModelSpec
    backend: str                # "torch" | "numpy"
    _forward: Callable[[np.ndarray], dict]   # 内部推論関数

    @property
    def is_stand_in(self) -> bool:
        return self.spec.is_stand_in

    def predict(self, x_adapted: np.ndarray) -> dict:
        """adapter が整形した入力に推論を回し、統一フォーマットの dict を返す。

        returns: {family, is_stand_in, pred_name, pred_index, scores?, mask?}
        """
        out = self._forward(np.asarray(x_adapted))
        out.setdefault("family", self.spec.family)
        out.setdefault("is_stand_in", self.spec.is_stand_in)
        return out


# ===========================================================================
# 共通: torch の遅延 import（無ければ明示例外）
# ===========================================================================
def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ModelUnavailable(
            "PyTorch が見つかりません。外部モデル（torchsig / qoherent）の推論には "
            "torch が必須です。CPU 版なら:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "torch 不要で配線だけ検証するなら load_reference_standin(...) を使う。\n"
            f"(import error: {e})"
        ) from e
    import torch
    return torch


def _resolve_weights(weights: Optional[str], env_var: str,
                     family: str, how_to_get: str) -> str:
    """重みパスを解決（引数 → 環境変数）。無ければ取得手順付きで例外。"""
    path = weights or os.environ.get(env_var)
    if not path:
        raise ModelUnavailable(
            f"{family} の重みが指定されていません。"
            f"weights=... か 環境変数 {env_var} でローカルパスを与えてください。\n"
            f"取得手順: {how_to_get}"
        )
    if not os.path.exists(path):
        raise ModelUnavailable(
            f"{family} の重みが見つかりません: {path}\n取得手順: {how_to_get}"
        )
    return path


# ===========================================================================
# TorchSig: 狭帯域分類（Sig53 / EfficientNet-B4）
# ===========================================================================
TORCHSIG_NB_CLASSES_HINT = 53   # Sig53


def load_torchsig_narrowband(weights: Optional[str] = None,
                             num_classes: int = TORCHSIG_NB_CLASSES_HINT,
                             device: str = "cpu") -> LoadedModel:
    """TorchSig 狭帯域分類器（EfficientNet-B4, Sig53）をロード。

    入力契約: 複素ベースバンド IQ 4096 サンプル → [2, 4096]（実部/虚部）。
    重み: weights= か 環境変数 SIGSCAN_TORCHSIG_NB_WEIGHTS。

    依存未導入・重み欠如は ModelUnavailable（握りつぶさない）。
    """
    torch = _require_torch()
    how = ("pip install torchsig（github.com/TorchDSP/torchsig, Python>=3.10）→ "
           "torchsig.com / HuggingFace の Sig53 EfficientNet-B4 チェックポイントを "
           "ダウンロードしてパス指定。Sig53=TorchSig合成(MIT)。")
    try:
        # torchsig のバージョン差を吸収（モデル構築 API はバージョンで揺れる）。
        try:
            from torchsig.models.iq_models.efficientnet.efficientnet import (
                efficientnet_b4,
            )
            net = efficientnet_b4(pretrained=False, num_classes=num_classes)
        except Exception:
            from torchsig.models import efficientnet_b4  # 新しめのレイアウト
            net = efficientnet_b4(num_classes=num_classes)
    except Exception as e:  # noqa: BLE001
        raise ModelUnavailable(
            f"torchsig の狭帯域モデルを構築できません。{how}\n(import/build error: {e})"
        ) from e

    wpath = _resolve_weights(weights, "SIGSCAN_TORCHSIG_NB_WEIGHTS",
                             "torchsig-narrowband", how)
    try:
        state = torch.load(wpath, map_location=device)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        net.load_state_dict(state, strict=False)
        net.eval().to(device)
    except Exception as e:  # noqa: BLE001
        raise ModelUnavailable(
            f"torchsig 狭帯域の重みロードに失敗: {wpath}\n(error: {e})") from e

    spec = ModelSpec(
        name="torchsig-sig53-efficientnet_b4", family="torchsig-narrowband",
        input_kind="iq", input_size=(4096,), channels=2,
        normalization="per-sample unit-energy (実部/虚部 2ch)",
        num_classes=num_classes, source="github.com/TorchDSP/torchsig",
        license="code: MIT / Sig53: TorchSig-synthetic",
        weights_source="torchsig.com / HuggingFace", weights_size="~数十-数百MB",
        is_pretrained=True,
    )

    def _forward(x: np.ndarray) -> dict:
        import torch as _t
        t = _t.as_tensor(np.asarray(x, dtype=np.float32))
        if t.ndim == 2:
            t = t.unsqueeze(0)            # [2,4096] -> [1,2,4096]
        with _t.no_grad():
            logits = net(t.to(device))
        probs = _t.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return dict(pred_index=idx, pred_name=f"class_{idx}",
                    scores=probs, kind="classifier")

    return LoadedModel(spec=spec, backend="torch", _forward=_forward)


# ===========================================================================
# TorchSig: 広帯域検出（WBSig53, 複素スペクトログラム）
# ===========================================================================
def load_torchsig_wideband(weights: Optional[str] = None,
                           num_classes: int = 53,
                           device: str = "cpu") -> LoadedModel:
    """TorchSig 広帯域検出/セグメンテーション（WBSig53, ~512x512 複素スペクトログラム）。

    NOTE: WBSig53 検出網はバージョン差が大きく、構築 API も流動的。本関数は
    torch/torchsig と重みが揃わない環境では必ず ModelUnavailable を投げ、
    「配線は用意したが本環境では実体化できない」ことを明示する。
    """
    _require_torch()
    how = ("pip install torchsig → WBSig53 検出器（DETR/Mask R-CNN 系）の "
           "チェックポイントを取得。複素スペクトログラム ~512x512 入力。")
    # 構築 API がバージョン依存のため、ここでは安全側に倒して明示失敗。
    raise ModelUnavailable(
        "torchsig 広帯域(WBSig53)検出器のローダは配線のみ実装。本環境では未実体化。\n"
        f"取得手順: {how}\n"
        "（M1 の検証は狭帯域 or reference-standin で行う。実体化は重み入手後に "
        "load_torchsig_wideband を拡張する。）"
    )


# ===========================================================================
# Qoherent: 5G NR / LTE スペクトログラム・セグメンテーション
# ===========================================================================
QOHERENT_CLASSES = ["noise/background", "5G NR", "LTE"]


def load_qoherent_segmentation(weights: Optional[str] = None,
                               num_classes: int = 3,
                               class_names: Optional[list[str]] = None,
                               device: str = "cpu") -> LoadedModel:
    """Qoherent の 5G/LTE スペクトログラム・セグメンテーション網をロード。

    DeepLabv3 + MobileNetV3（torchvision）。入力 256x256 RGB（ImageNet 正規化）。
    重み: weights= か 環境変数 SIGSCAN_QOHERENT_WEIGHTS。
    """
    torch = _require_torch()
    class_names = class_names or QOHERENT_CLASSES
    how = ("git clone github.com/qoherent/spectrogram-segmentation → "
           "ノートブックで学習 or 配布チェックポイントを取得し、その .pt/.pth を "
           "weights= で指定。torchvision の deeplabv3_mobilenet_v3_large を使用。")
    try:
        from torchvision.models.segmentation import (
            deeplabv3_mobilenet_v3_large,
        )
        net = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
    except Exception as e:  # noqa: BLE001
        raise ModelUnavailable(
            "torchvision の deeplabv3_mobilenet_v3_large を構築できません "
            f"（pip install torchvision）。{how}\n(error: {e})"
        ) from e

    wpath = _resolve_weights(weights, "SIGSCAN_QOHERENT_WEIGHTS",
                             "qoherent-segmentation", how)
    try:
        state = torch.load(wpath, map_location=device)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        net.load_state_dict(state, strict=False)
        net.eval().to(device)
    except Exception as e:  # noqa: BLE001
        raise ModelUnavailable(
            f"qoherent セグメンテーションの重みロードに失敗: {wpath}\n(error: {e})"
        ) from e

    spec = ModelSpec(
        name="qoherent-deeplabv3-mobilenetv3", family="qoherent-segmentation",
        input_kind="rgb-spectrogram", input_size=(256, 256), channels=3,
        normalization="ImageNet mean/std (RGB)",
        num_classes=num_classes, class_names=class_names,
        source="github.com/qoherent/spectrogram-segmentation",
        license="MIT (要確認)", weights_source="repo notebook / checkpoint",
        weights_size="~数十MB", is_pretrained=True,
    )

    def _forward(x: np.ndarray) -> dict:
        import torch as _t
        t = _t.as_tensor(np.asarray(x, dtype=np.float32))
        if t.ndim == 3:
            t = t.unsqueeze(0)            # [3,256,256] -> [1,3,256,256]
        with _t.no_grad():
            out = net(t.to(device))["out"][0].cpu().numpy()  # [C,H,W]
        seg = np.argmax(out, axis=0)                          # [H,W] クラス
        counts = np.bincount(seg.ravel(), minlength=num_classes)
        idx = int(np.argmax(counts))
        return dict(pred_index=idx,
                    pred_name=class_names[idx] if idx < len(class_names)
                    else f"class_{idx}",
                    mask=seg, class_counts=counts, kind="segmentation")

    return LoadedModel(spec=spec, backend="torch", _forward=_forward)


# ===========================================================================
# Reference stand-in（numpy のみ・配線検証用・学習済みではない）
# ===========================================================================
# 占有帯域率ベースの粗いバケット。**変調種別も信号有無も判定しない**
# （5G/LTE 等を当てない／純ノイズの presence detection もしない）。
STANDIN_CLASSES = ["occ<10%", "occ10-50%", "occ>50%"]


def load_reference_standin(emulates: str = "qoherent-segmentation",
                           num_classes: int = 3) -> LoadedModel:
    """numpy だけで動く配線検証用スタンドイン（**学習済みでも外部モデルでもない**）。

    目的は adapter→推論→report の経路を torch 無し環境でも通すこと。
    スペクトログラムの **占有度/帯域幅** という単純特徴を粗いクラスに写すだけで、
    変調種別（5G/LTE/WiFi 等）は一切判定しない。出力は必ず is_stand_in=True。

    emulates: どの family の入力形に合わせるか（adapter 選択用）。
      "qoherent-segmentation" -> [3,256,256] を受ける
      "torchsig-narrowband"   -> [2,4096] を受ける
    """
    if emulates == "torchsig-narrowband":
        spec = ModelSpec(
            name="reference-standin(nb)", family="torchsig-narrowband",
            input_kind="iq", input_size=(4096,), channels=2,
            normalization="(stand-in) per-sample unit-energy",
            num_classes=len(STANDIN_CLASSES), class_names=list(STANDIN_CLASSES),
            source="(none - sigscan reference stand-in)",
            license="N/A", is_pretrained=False, is_stand_in=True,
        )

        def _forward(x: np.ndarray) -> dict:
            x = np.asarray(x, dtype=np.float32)
            iq2 = x[0] if x.ndim == 3 else x          # [2,4096]
            comp = iq2[0] + 1j * iq2[1]
            mag = np.abs(np.fft.fftshift(np.fft.fft(comp)))
            return _occupancy_predict(mag[None, :])    # 1行スペクトル

    else:  # 既定: セグメンテーション family（spec.render 画像を直接受ける）
        spec = ModelSpec(
            name="reference-standin(seg)", family="qoherent-segmentation",
            input_kind="rgb-spectrogram", input_size=(256, 256), channels=3,
            normalization="(stand-in) passthrough",
            num_classes=len(STANDIN_CLASSES), class_names=list(STANDIN_CLASSES),
            source="(none - sigscan reference stand-in)",
            license="N/A", is_pretrained=False, is_stand_in=True,
        )

        def _forward(x: np.ndarray) -> dict:
            x = np.asarray(x, dtype=np.float32)
            img = x[0] if x.ndim == 4 else x          # [3,256,256]
            chan = img[0] if img.ndim == 3 else img   # 1ch ぶんで占有度
            return _occupancy_predict(chan)

    return LoadedModel(spec=spec, backend="numpy", _forward=_forward)


def _occupancy_predict(img2d: np.ndarray) -> dict:
    """スペクトル像から「占有帯域率」を求め粗いクラスに振る（決定的・学習なし）。

    周波数プロファイル（時間方向平均）を **画像内 min-max 正規化** してから、
    正規化値 > 0.5 の周波数ビン割合を占有率とする。min-max 正規化により入力の
    絶対スケール（ImageNet 正規化済み画像 / 生 FFT マグニチュード）に依存しない。

    注意（誠実さ）: これは帯域の埋まり具合だけを見る素朴な指標。信号の有無や
    変調種別は判定しない。純ノイズはコントラストが微小でも min-max で増幅され
    中位の占有率に化けうる（presence detection はしない）。
    """
    a = np.asarray(img2d, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    # freq 軸（長い方を周波数とみなす）に沿って時間平均 → プロファイル
    freq_profile = a.mean(axis=1) if a.shape[0] >= a.shape[1] else a.mean(axis=0)
    lo = float(freq_profile.min()) if freq_profile.size else 0.0
    hi = float(freq_profile.max()) if freq_profile.size else 0.0
    rng = hi - lo
    if rng < 1e-9 or freq_profile.size == 0:
        occ = 0.0
    else:
        norm = (freq_profile - lo) / rng
        occ = float(np.mean(norm > 0.5))
    if occ < 0.10:
        idx = 0
    elif occ < 0.50:
        idx = 1
    else:
        idx = 2
    scores = np.zeros(len(STANDIN_CLASSES), dtype=np.float32)
    scores[idx] = 1.0
    return dict(pred_index=idx, pred_name=STANDIN_CLASSES[idx],
                scores=scores, occupancy=round(occ, 3), kind="standin")


# ===========================================================================
# ディスパッチャ
# ===========================================================================
_LOADERS: dict[str, Callable[..., LoadedModel]] = {
    "torchsig-narrowband": load_torchsig_narrowband,
    "torchsig-wideband": load_torchsig_wideband,
    "qoherent-segmentation": load_qoherent_segmentation,
    "reference-standin": load_reference_standin,
}


def available_models() -> list[str]:
    return list(_LOADERS)


def load_model(name: str, *, allow_standin: bool = False,
               **kwargs: Any) -> LoadedModel:
    """名前で外部モデルをロード。`allow_standin=True` なら ModelUnavailable 時に
    対応する family を emulate する reference-standin に **明示的に** 退避する
    （退避したことは戻り値の spec.is_stand_in と report バナーで分かる）。
    """
    if name not in _LOADERS:
        raise ValueError(f"未知のモデル名: {name}（{available_models()}）")
    if name == "reference-standin":
        return load_reference_standin(**{k: v for k, v in kwargs.items()
                                         if k in ("emulates", "num_classes")})
    try:
        return _LOADERS[name](**kwargs)
    except ModelUnavailable:
        if not allow_standin:
            raise
        emulates = name if name in ("torchsig-narrowband",
                                    "qoherent-segmentation") else \
            "qoherent-segmentation"
        sub = load_reference_standin(emulates=emulates)
        # 退避した事実を名前にも刻む（誠実さ）。
        sub.spec.name = f"reference-standin(fallback<-{name})"
        return sub


# ===========================================================================
# 自己診断 CLI: どのモデルがこの環境でロードできるかを表示
# ===========================================================================
def _selfcheck() -> int:
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    print("== eval.loaders self-check ==")
    try:
        import torch
        print(f"torch: OK ({torch.__version__})")
    except Exception as e:  # noqa: BLE001
        print(f"torch: 未導入 ({e.__class__.__name__})")
    for name in available_models():
        try:
            m = load_model(name)
            print(f"[loadable] {name}: {m.spec.summary()}")
        except ModelUnavailable as e:
            head = str(e).splitlines()[0]
            print(f"[unavailable] {name}: {head}")
        except Exception as e:  # noqa: BLE001
            print(f"[error] {name}: {e.__class__.__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
