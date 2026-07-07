# 作業指示書: capture-engine — 新Windows環境の HackRF 実機環境 診断（収集再開の準備）

## 役割
あなたは sigscan プロジェクトの capture-engine エージェント。
本指示は **環境診断が主体**。新しい Windows PC（乗り換え直後）で、HackRF One を使った
実機収集（`python main.py --hardware ...`）が動く状態かを**点検し、何が足りないかを特定して報告**する。
**プロジェクトのコード（*.py）・captures/・凍結契約には一切触れない。** 不足が見つかった場合、
導入は**手順を提示するに留め**、インストーラの実行や物理操作（USB 抜き差し等）は人間が行う。

## 背景
- 旧環境（別 Windows）では HackRF One + PothosSDR/SoapySDR で収集できていた
  （収集コマンド: `python main.py --hardware --start 2.4e9 --stop 2.5e9 --focus --dwell-seconds 10 --collect captures/`）。
- 新 PC（i9-11900K / RTX 3080 / Win64）にコードは clone 済み、Python 3.9.25 venv・numpy 1.26.4・
  torch(GPU) 導入済み、pytest 147 passed。**ただし実機ドライバ層（SoapySDR / SoapyHackRF / HackRF USB ドライバ）が
  新 PC に入っているかは未確認。**
- `requirements.txt` の方針: SoapySDR は pip ではなく別途導入（Windows は PothosSDR 同梱が定番）。
  SoapyHackRF のビルド/導入は README 参照。
- 収集コード側の SDR 層は `sdr.py`（`sweep_power` / `capture_iq`）。ここが SoapySDR 経由で HackRF を掴む。

## 最重要原則（絶対厳守）
1. **コード・データに触れない**: `*.py`（`sdr.py` 含む）/ `spec.py` / `sigmf_io.py` / captures/ の
   `*.sigmf-*` を一切編集しない。これは環境診断であって実装ではない。
2. **勝手にインストールしない**: システムへの導入（PothosSDR インストーラ実行、ドライバ導入、
   Zadig 等での USB ドライバ差し替え）は**人間が行う**。エージェントは「何を・どの順で入れるか」の
   **手順と根拠を提示するだけ**。pip での安全なパッケージ導入（もし必要で、かつ venv 内に閉じるもの）は
   提案として示すが、実行判断は人間に委ねる。
3. **物理操作はしない/できない**: USB 抜き差し、アンテナ接続等は人間。エージェントは指示のみ。
4. **診断は非破壊**: 環境を変えるコマンド（インストール・設定書き換え）は実行しない。
   状態を**読むだけ**のコマンド（バージョン確認、デバイス列挙、import 可否）に限定する。
5. **正直に報告**: 「たぶん入っている」で流さない。実際に確認コマンドを叩き、出力（成否）で判断する。
   確認できないことは「確認できなかった」と明記。

## 診断内容（実施して報告に含める）

### (A) Python 側から SDR ライブラリが見えるか
1. venv が有効な前提で、SoapySDR が Python から import できるかを確認（**読むだけ**）:
   - `python -c "import SoapySDR; print(SoapySDR.getAPIVersion())"` の成否と出力。
   - 失敗するなら、その ImportError の内容を報告（モジュール自体が無いのか、DLL 依存が壊れているのか）。
2. `sdr.py` が import している SDR 関連モジュール名を確認し、それらが import 可能かを個別に報告
   （`sdr.py` を**読む**のは可。編集は不可）。

### (B) SoapySDR ランタイムと HackRF ドライバの有無
3. `SoapySDRUtil` が PATH にあるか、あれば情報を取得（**読むだけ**）:
   - `SoapySDRUtil --info`（インストール状況・モジュール検索パス）。
   - `SoapySDRUtil --find`（接続デバイスの列挙）。HackRF が挿さっていれば `driver=hackrf` 等が出る想定。
   - コマンド自体が無い場合は「SoapySDRUtil が PATH に無い（＝PothosSDR 未導入 or PATH 未通し）」と報告。
4. `hackrf_info`（HackRF 純正ツール）が PATH にあれば実行し、デバイス認識とファーム情報を報告
   （**読むだけ**）。無ければ「hackrf_info 無し」と報告。

### (C) 収集コードが実機モードで起動できるかの入口確認（デバイスを掴む手前まで）
5. `python main.py --help` を実行し、`--hardware` を含む実機収集オプションが現行コードにあるか、
   起動方法を報告（**--help のみ。実際の収集や --hardware 実行はしない**。デバイスを掴むと排他や
   Resource busy を招くため、この診断では収集を開始しない）。
6. 実機を掴まない範囲で、`sdr.py` の SDR 初期化経路（どの driver 文字列で SoapySDR.Device を開くか、
   sample_rate や bandwidth の既定）を**読んで**報告。実際に Device を open する検証は**しない**
   （open は人間が収集時に行う）。

### (D) 環境の全体像
7. 参考情報として報告（読むだけ）:
   - `python --version` / venv パス / `pip show SoapySDR`（pip 管理なら）で SoapySDR の出所。
   - OS / PATH に PothosSDR 系のパスが含まれるか（`where SoapySDRUtil` 等）。

## 何が足りないかの判定と、導入手順の提示（提案のみ・実行は人間）
8. (A)〜(D) の結果から、**収集を動かすために不足しているもの**を列挙し、**導入手順を順序立てて提示**する。
   典型的な想定（実際の診断結果に合わせて具体化すること）:
   - **PothosSDR（SoapySDR + SoapyHackRF 同梱）が未導入** → 公式 PothosSDR インストーラの入手先と
     インストール手順、インストール後に PATH を通す必要の有無、venv の Python から SoapySDR を
     import できるようにする方法（PothosSDR 同梱 Python バインディングと venv の関係に注意）を提示。
   - **HackRF の USB ドライバ未導入** → Windows で HackRF を認識させる手順（必要なら Zadig 等）を、
     手順として提示（実行は人間）。
   - **PATH 未通し** → どのディレクトリを PATH に追加すべきかを提示。
   - venv の Python から SoapySDR が見えない場合の対処（PothosSDR の Python と venv の食い違いは
     よくある罠。回避策を複数提示し、どれを採るかは人間が選ぶ）。
   **各手順に「なぜそれが必要か」を一言添える。** インストーラ実行・ドライバ差し替え・USB 操作は
   人間が行う旨を明記。

## 検証（診断後）
1. `git status --short` で**作業ツリーがクリーン**（このタスクで何も変更していないこと）を確認・明示。
   コード・データ・設定を書き換えていないこと。
2. 実行したのは**読み取り専用の確認コマンドのみ**であることを明記（インストール・設定変更・収集開始をしていない）。
3. pytest 不要（コード未変更）。

## 完了報告に含めること
- (A) SoapySDR の Python import 可否と出力（失敗ならエラー内容）。
- (B) SoapySDRUtil / hackrf_info の有無と、デバイス列挙結果（HackRF が見えたか）。
- (C) `main.py --help` の実機オプション有無と起動方法、`sdr.py` の driver/rate 既定。
- (D) 環境の全体像（Python/venv/SoapySDR の出所/PATH）。
- **不足しているものの列挙と、順序立てた導入手順（実行は人間）＋各手順の理由。**
- 「コード・データ未変更／作業ツリー clean／読み取り専用コマンドのみ実行」の明言。
- もし HackRF が既に認識され、SoapySDR も import できるなら「収集可能な状態」と判定し、
  次に人間が打つべき収集コマンド（狭い範囲の例）を提示。

## やってはいけないこと（禁止事項の再掲）
- `*.py`（`sdr.py` 含む）/ `spec.py` / `sigmf_io.py` / captures/ の書き換え。
- **システムへのインストール・ドライバ導入・設定変更の実行**（手順提示のみ。実行は人間）。
- **`--hardware` での収集の実行**（デバイスを掴むと排他/Resource busy を招く。診断は掴む手前まで）。
- SoapySDR.Device を実際に open して掴む検証（人間が収集時に行う）。
- USB 抜き差し等の物理操作の指示を「済ませた前提」で進めること（状態は人間に確認する）。
- 環境を変えるコマンドの実行全般。captures/ の `*.sigmf-*` 参照時に UTF-8 決め打ちで開くこと。
