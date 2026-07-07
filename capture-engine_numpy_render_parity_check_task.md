# 作業指示書: capture-engine — 収集前の数値一致確認（numpy 1.26.4 vs 2.2.3 で spec.render が一致するか）

## 役割
あなたは sigscan プロジェクトの capture-engine エージェント。
本指示は **収集を再開する前の安全確認**。実機収集は radioconda の Python(3.12, numpy 2.2.3) で行う予定だが、
学習側・合成データ・既存モデルは venv(3.9, numpy 1.26.4) で作られている。両者で凍結契約
**`spec.render`（およびその依存 `stft_db` 等）が同一 IQ に対して同じ数値を出すか**を検証し、
numpy バージョン差による分布ズレ（synthetic/real の domain gap 再発）が無いことを確認する。
**プロジェクトのコード・データ・凍結契約は一切変更しない（読み取り専用の検証）。**

## 背景
- 環境診断の結果、2つの Python が併存:
  - **venv (3.9.25)**: numpy 1.26.4 + torch(GPU)。sim/CNN/pytest 用。SoapySDR 無し（実機不可）。
  - **radioconda (3.12.9)**: numpy 2.2.3 + SoapySDR。実機収集可。torch 無し。
- 収集は radioconda で回す方針（収集に torch 不要）。ただし収集時に IQ→画像化する `spec.render` は
  凍結契約であり、**全コンポーネントが同一スケールで混ざる前提**（合成・実測・外部データ）。
  numpy 1.x と 2.x で数値が僅かでも異なると、「2.x で録った実データ」と「1.x で作った合成/モデル」の間に
  分布ギャップが生じ得る。これは本プロジェクトが最も警戒する失敗モード（CNN の synthetic/real gap）。
- `spec.render`（spec.py）は凍結契約。`stft_db` は FFT（`np.fft`）・`np.hanning`・`percentile`・
  dB 変換・[256,256] リサイズ等の numpy 演算で構成。これらが 1.x/2.x で bit 一致するかが焦点。

## 最重要原則（絶対厳守）
1. **コード・データを変更しない**: `spec.py` / `sigmf_io.py` / その他 `*.py` / captures/ / simdata/ を
   一切編集しない。これは検証であって実装ではない。
2. **凍結契約に触れない**: `spec.render` / `stft_db` の実装を変えない。呼ぶだけ。
3. **両 Python で同じコードを呼ぶ**: 比較は「同一の spec.py を、venv の python と radioconda の python で
   それぞれ import して実行し、出力配列を突き合わせる」形。**コードのコピー改変や再実装をしない**
   （プロジェクトの spec.py をそのまま両インタプリタから使う）。
4. **決定論的な入力で比較**: ランダム性を排除するため、**固定シードで生成した同一 IQ**（複数種類）を
   使う。IQ は float32/complex64 で両環境に同一バイトで渡す（後述）。
5. **一時ファイルは repo 外・非コミット**（scratchpad 等）。captures/ に書かない。
   メタを読む用事は無い想定だが、読むなら sigmf_io の cp932 経路（UTF-8 決め打ち禁止）。
6. **正直に報告**: 「たぶん一致」で流さず、**実際の数値差（最大絶対誤差・不一致要素数）**で判定する。
   一致・不一致どちらでも事実を報告。環境を変えるコマンド（インストール等）は実行しない。

## 検証内容（実施して報告に含める）

### (A) 比較スクリプトの作成（repo 外・両 Python から同一 spec.py を使う）
1. scratchpad（repo 外）に、次を行う小スクリプトを2段構えで用意（**プロジェクトの spec.py を import して使う**。
   spec.py を書き換えない・コピーして改変しない）:
   - **段1（生成・保存）**: 固定シード（例 seed=0,1,2）で複数種類の代表 IQ を numpy で生成し、
     **同一バイトのバイナリ（.npy）として保存**する。代表 IQ は少なくとも:
     (i) 帯域制限ノイズ（狭帯域バースト様）、(ii) 単一トーン（CW様）、(iii) 広帯域ノイズ、
     (iv) 実際の収集長（262144 サンプル ≈ 13ms）に合わせた長さのもの。
     生成は片方の Python（例 venv）で一度だけ行い、.npy を確定させる（両環境で同じ入力を保証）。
   - **段2（各環境で render して保存）**: venv の python と radioconda の python の**それぞれ**で、
     保存済み .npy を読み込み → プロジェクトの `spec.render`（および可能なら `stft_db` の生 dB 配列）に
     通し → 出力配列を .npy で保存する（例 `out_venv_iq0.npy`, `out_rconda_iq0.npy`）。
     **同じ spec.py を両者が import すること**（sys.path をプロジェクト直下に通す）。
2. 実行方法（読み取り専用・環境変更なし）:
   - venv 側: `python <script> --mode render --in <iq.npy> --out <out_venv.npy>`
   - radioconda 側: `C:\Users\puppy\radioconda\python.exe <script> --mode render --in <iq.npy> --out <out_rconda.npy>`
   - Device open も --hardware も行わない（これは純粋に IQ→画像の数値比較）。

### (B) 数値比較
3. venv 出力 と radioconda 出力を突き合わせ、IQ 種類ごとに報告:
   - **最大絶対誤差** `np.max(np.abs(a - b))`
   - **完全一致か** `np.array_equal(a, b)`（bit 一致）
   - 不一致なら、不一致要素数・相対誤差（`np.max(np.abs(a-b)/(np.abs(b)+eps))`）・
     差が出ている箇所の傾向（全体に微小か、特定周波数ビンか、リサイズ境界か等）。
   - `spec.render` の最終 [256,256] 出力だけでなく、可能なら中間の `stft_db`（dB 配列）でも比較し、
     差が FFT 段で出るのか正規化/リサイズ段で出るのかを切り分け。
4. 判定基準（報告に明記）:
   - **bit 完全一致** → 「numpy 2.x 収集で分布ズレ無し。radioconda で収集して安全」。
   - **微小差（例: 最大絶対誤差が float32 の丸め誤差 ~1e-6 オーダー以下）** → 実質同一。
     ただし「完全一致ではない」ことは明記し、CNN 入力として無視できる根拠（dB スケール・
     [0,1] 正規化後の差の大きさ）を示す。
   - **無視できない差（正規化後 [0,1] で目に見えるオーダーの差、または特定ビンの系統差）** →
     「そのまま radioconda 収集は危険」と判定し、対処（後述の環境統合案）を提案。

### (C) numpy 以外の差の確認（副次）
5. 参考として、両環境の関連ライブラリのバージョン差を報告（読むだけ）:
   numpy（1.26.4 vs 2.2.3）、Python（3.9 vs 3.12）、（あれば）scipy 等 spec/dsp が使うもの。
   差が出た場合に「numpy が主因か、他要因か」の切り分け材料にする。

## 検証（タスク要件）
1. `git status --short` で**作業ツリーがクリーン**（このタスクで何も変更していないこと）を確認・明示。
   spec.py 含むコード・captures/・simdata/ を書き換えていないこと。
2. 実行したのは**読み取り専用の比較のみ**（spec.py を import して呼んだだけ・環境変更なし・
   Device open / --hardware なし）であることを明記。
3. pytest 不要（コード未変更）。

## 完了報告に含めること
- (A) 比較の方法（どの IQ を・どう生成し・両 Python でどう render したか。spec.py は無改変で import した旨）。
- (B) IQ 種類ごとの **最大絶対誤差・bit 一致可否**、および差が出た段（FFT/正規化/リサイズ）の切り分け。
- (C) numpy/Python/scipy 等のバージョン差。
- **総合判定**: 「bit 一致 / 実質同一(微小差) / 無視できない差」のどれか、および
  「radioconda で収集して安全か否か」の明確な結論。
- 無視できない差だった場合の対処提案（例: 収集も numpy 1.x に揃える案 = conda で
  numpy 1.26 系＋SoapySDR の env を1本立てる、等。実装・インストールはしない・提案のみ）。
- 「コード・データ未変更／作業ツリー clean／読み取り専用のみ」の明言。

## やってはいけないこと（禁止事項の再掲）
- `spec.py` / `sigmf_io.py` / その他コード / captures/ / simdata/ の書き換え。
- `spec.render` / `stft_db` の実装変更、またはコピーして改変した版で比較すること
  （**必ずプロジェクトの spec.py そのものを両 Python から import する**）。
- インストール・環境変更・設定書き換えの実行（対処は提案のみ）。
- `--hardware` 収集や SoapySDR.Device open（本タスクは IQ→画像の数値比較のみ）。
- captures/ への書き込み。メタを UTF-8 決め打ちで開くこと。
- スコープ拡張（数値一致確認以外の「ついでの検証」）。迷ったら最小。
