# 作業指示書: cnn-training — CNN 学習の GPU 対応（device 自動選択・最小改修）

## 役割
あなたは sigscan プロジェクトの cnn-training エージェント。
本指示は、CPU 前提で書かれた `cnntrain/train.py` に **GPU（CUDA）自動選択を最小限で足す**改修。
「CUDA が使えれば GPU、無ければ従来どおり CPU」にするだけ。**CPU 後方互換を絶対に保つ**。
学習ロジック・ハイパラ・checkpoint 形式・評価は変えない。**device の選択と、モデル/データを
その device に載せる処理だけ**を足す。

## 背景
- 現状 `cnntrain/train.py` は **CPU 決め打ち**（train.py:120 で `device : cpu` を文字列表示、
  コメントも「CPU 前提」）。M1 火入れ時の意図的な設計（当時 GPU 環境なし）。
- 新環境に **RTX 3080 + torch 2.5.1+cu121（CUDA 利用可）** が入り、`torch.cuda.is_available()` が True。
  だが学習は CPU で走る（コードが device を選んでいない）。
- **これはバグ修正ではなく小さな新機能追加**。実利は現状小さい（300件×8epoch が CPU 40.8s → GPU で
  数秒〜十数秒程度）が、将来データが増えたとき効く。今回は「最小で GPU を使えるようにする」だけ。
- `cnntrain.train` は独立パッケージ。**凍結契約（spec.render / sigmf_io）には触れない**構造。
  学習済みモデルは `runs/<name>/checkpoint.pt` に保存され、収集の `--cnn`（`main.py --cnn
  --cnn-checkpoint runs/m2_5`）がこれを読む。**checkpoint の保存形式・キー・読込互換を壊さないこと**が最重要
  （壊すと `--cnn` 収集が動かなくなる）。

## 最重要原則（絶対厳守）
1. **凍結契約不可侵**: `spec.py` / `sigmf_io.py` を変更しない。6継ぎ目のシグネチャ不変。
   `git diff --stat -- spec.py sigmf_io.py` が空。CNN 入力は従来どおり凍結 `spec.render` 経由（不変）。
2. **CPU 後方互換を絶対維持**: CUDA が無い環境（CI・他 PC・CPU only）で、**従来と完全に同じに動く**こと。
   GPU 前提の決め打ちにしない。`torch.cuda.is_available()` が False なら従来の CPU 経路そのまま。
3. **checkpoint 形式・互換を壊さない（最重要）**: 保存する checkpoint の中身（state_dict のキー、
   メタ情報、rep_version 等）を変えない。**GPU で学習しても、保存時は CPU に戻して保存する**
   （`model.to("cpu")` してから state_dict を保存、または `map_location` で吸収できる形にする）。
   狙いは「GPU 学習した checkpoint も、CPU 環境の推論（`--cnn` 収集や既存テスト）でそのまま読める」こと。
   → **保存済み checkpoint を CPU でも GPU でも読めること**を必ず確認する。
4. **学習の数値的挙動・ハイパラを変えない**: batch/lr/epochs/seed/val_ratio/モデル構造は不変。
   device が変わっても**同じ seed で学習が再現される**設計を保つ(乱数シードの扱いに注意。CPU/GPU で
   完全 bit 一致まで求めないが、val accuracy が従来水準(≈95%)から大きくずれないこと)。
5. **最小実装・スコープ厳守**: device 選択と `.to(device)` の付与だけ。**それ以外の最適化を足さない**
   （num_workers 変更、AMP/mixed precision、pin_memory、batch サイズ変更、cudnn.benchmark 等は
   今回やらない。将来課題コメントに留める）。迷ったら最小。
6. **読み取り専用データ**: simdata/ / captures/ の `*.sigmf-*` を書き換えない。

## 事前確認（実装前に行い、報告に含める）
1. `cnntrain/train.py` を読み、報告:
   - 現在 device 相当を扱っている箇所（train.py:120 の表示、モデル生成・データ供給・学習ループで
     tensor がどこに載るか）。CPU 決め打ちがどこで効いているか。
   - **checkpoint を保存している箇所**（何を・どのキーで保存しているか。torch.save の内容）。
   - **checkpoint を読み込む側**（`--cnn` 収集が使う推論経路。おそらく `cnntrain/` 内の infer/classify
     連携や `main.py --cnn`）が、どう checkpoint をロードするか（`map_location` の有無）。
2. `--device` のような CLI 引数は現状無い（`main.py --help` にも無し）。**今回 CLI 引数は足さず自動選択**にするか、
   `--device auto|cpu|cuda` を足すか、を判断（推奨: **引数を足さず自動選択**＝最小。ただし
   「CUDA があっても CPU で回したい」需要が将来あるなら `--device` も一案。今回は最小＝自動選択を推奨）。
3. 既存テストの棚卸し（ベースライン 147 passed, 3 skipped を着手前に確認）:
   - `tests/test_cnntrain_m1.py` 〜 `m3.py`（学習・推論・spotlight・classify 連携）で、
     device や checkpoint 形式に依存するテストがあるか。**CPU 環境で緑のまま維持**できるかを確認。
   - `tests/test_seams.py`（シグネチャ）は無変更で緑（当然）。

## 実装内容
### device 自動選択の追加（cnntrain/train.py）
- 学習開始時に device を決める:
  `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- **モデルとデータ（入力 tensor・ラベル）を `.to(device)` で載せる**（学習ループ内で batch を device へ）。
- 表示（train.py:120）を、決め打ち `cpu` から**実際の device**に変える
  （例: `device : cuda` / `cuda:0 (NVIDIA GeForce RTX 3080)` 等。CPU 時は従来どおり `cpu`）。
- **checkpoint 保存は CPU 化してから**: `model.to("cpu")` の state_dict を保存する（または保存後に
  device に戻す）。保存する dict のキー・メタは一切変えない。目的は「GPU 学習 checkpoint を
  CPU 推論で読める」互換の保証。
- 乱数シード: 既存の seed 設定を尊重。CUDA を使う場合に必要なら `torch.cuda.manual_seed_all(seed)` を
  **追加**（既存の CPU seed 設定は消さない）。完全 bit 再現までは求めないが、val が従来水準を保つこと。

### 推論側（読み込み）の互換（必要な場合のみ・最小）
- `--cnn` 収集や推論が checkpoint を読む箇所で、**`map_location` が無くて GPU 保存物を CPU で
  読めない懸念**がある場合のみ、`torch.load(..., map_location="cpu")` を追加（最小）。
  ただし本指示の保存側で CPU 化して保存するなら、読込側は無改変で足りる可能性が高い。
  **まず保存側 CPU 化で対応し、読込側は「必要な場合のみ」最小修正**（変えたら報告）。
- 推論を GPU で回すかは**今回スコープ外**（収集の CNN 監査は軽量。CPU 推論のままで可）。
  学習だけ GPU 対応する。推論の device 最適化は将来課題。

### テスト（既存無変更・追加のみ）
- 追加: 「device 選択関数（切り出せるなら純関数）が、CUDA 有無で正しい device を返す」テスト
  （CUDA 無し CI では cpu を返すことの確認。cuda 分岐は環境依存なのでスキップ/条件付きに）。
- 追加: 「GPU 学習を模した checkpoint（または CPU 保存 checkpoint）が CPU で読み込めて推論できる」
  往復テスト（checkpoint 互換の担保）。CUDA が無い環境では CPU 保存物で代替。
- **既存テストは無変更で緑**（CPU 環境で 147 passed 維持）。test_seams 無変更。

## 検証（実装後に必ず行う）
1. `python -m py_compile cnntrain/train.py`（＋読込側を触ったらそれも）OK。
2. `& $py -m pytest -q` 全緑（既存 147 + 新規）。**CPU 環境でも緑**であること（後方互換）。test_seams 緑。
3. `git diff --stat -- spec.py sigmf_io.py` が空。
4. **実学習で GPU が使われることの確認**: `& $py -m cnntrain.train --data simdata/ --epochs 8
   --out runs/m2_5_gpu/` を実走し、表示が `device : cuda`（RTX 3080）になり、学習が完了、
   **val accuracy が従来水準（≈95%）**であることを確認（大きく劣化しないこと）。
   ※ 既存 `runs/m2_5/` を上書きせず、確認用に別 out（`runs/m2_5_gpu/`）に出すのが安全。
5. **checkpoint 互換の確認（最重要）**: GPU 学習で作った checkpoint を、`--cnn` 収集または
   推論経路が**CPU で読めて動く**ことを確認（`map_location` 無しで読めるか。読めれば互換 OK）。
   可能なら `& $py main.py --sim --dwell --dwell-seconds 2 --cnn --cnn-checkpoint runs/m2_5_gpu --once`
   等の**実機を掴まない sim 経路**で、GPU 製 checkpoint が読み込めることを確認（実機収集はしない）。
6. simdata/ / captures/ の `*.sigmf-*` 無変更（件数明示）。

## 完了報告に含めること
- 事前確認（device 決め打ち箇所・checkpoint 保存/読込の仕組み・map_location 有無）。
- 変更点（device 自動選択の追加、`.to(device)`、表示の実 device 化、checkpoint の CPU 化保存）。
- **CPU 後方互換の明言**（CUDA 無し環境で従来どおり・147 passed 維持）。
- **checkpoint 互換の明言**（GPU 製 checkpoint が CPU 推論・`--cnn` で読めること。読込側を変えたなら明記）。
- **凍結契約不変**（spec/sigmf_io diff 空、test_seams 緑、CNN 入力は spec.render のまま）。
- 実学習の device 表示（cuda/RTX 3080）と val accuracy（従来水準か）、学習時間の CPU 比。
- 追加テストの内容。（あれば）触れた読込側の最小修正。
- 将来課題: 推論の GPU 化・AMP・num_workers・pin_memory 等は**今回スコープ外**である旨。

## やってはいけないこと（禁止事項の再掲）
- `spec.py` / `sigmf_io.py` の変更。6継ぎ目シグネチャの変更。
- **checkpoint の保存形式・キー・メタの変更**（`--cnn` 収集・既存テストが読めなくなる）。
- **CPU 後方互換を壊すこと**（GPU 前提の決め打ち。CUDA 無し環境で落ちる実装）。
- 学習ハイパラ（batch/lr/epochs/seed/val_ratio）・モデル構造・評価ロジックの変更。
- スコープ拡張（AMP/mixed precision・num_workers 変更・pin_memory・cudnn.benchmark・
  推論の GPU 化・バッチ最適化等）。**今回は device 選択と .to(device) と CPU 化保存だけ**。迷ったら最小。
- simdata/ / captures/ の `*.sigmf-*` の書き換え。既存テストの変更・削除。test_seams の変更。
