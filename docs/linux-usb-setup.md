# USBで持ち運び → どのLinux機でも動かす手引き

このプロジェクトを USB に入れて、Fedora / Ubuntu / Kali / Arch などどの Linux 機に挿しても
動かすための手順。考え方は2つに分かれる。

- **コード**は USB で持ち運べる（gitで管理されているので丸ごとコピーでOK）。
- **HackRFを動かす土台**（SoapySDR等）は、挿した各マシンに1回ずつ入れる
  → これを `setup.sh` 一発で済ませる。

---

## 0. 置き場所（最初に1回）

`setup.sh` をプロジェクトの一番上（`README.md` と同じ場所）に置く。この手引きは
`docs/linux-usb-setup.md` に置く。Windows で次を実行してセーブ:

```
git add -A
git commit -m "add cross-distro linux setup.sh and USB guide"
```

---

## 1. USB を準備（exFAT）

- USB を **exFAT** でフォーマットしておくと、Windows でも Linux でも読み書きできる。
  （Windows: USBを右クリック →「フォーマット」→ ファイルシステムで exFAT を選択）
- 既に FAT32/exFAT のUSBならそのままでよい（このプロジェクトのファイルは小さい）。

## 2. プロジェクトを USB にコピー

- `sigintSDR` フォルダごと USB にドラッグ＆ドロップ。
- `captures/`（電波データ）と `.venv/` は重いので**コピー不要**（無くても動く）。
  コードと設定だけで十分。

---

## 3. Linux 機で土台を入れる（各マシン1回）

1. USB を Linux 機に挿す。たいてい自動でマウントされる。場所の例:
   `/run/media/あなたの名前/USBの名前/sigintSDR` または `/media/...`。
   分からなければファイルマネージャでUSBを開き、右クリック「端末で開く」。
2. 端末でプロジェクトの中に入る:
   ```
   cd /run/media/$USER/USBの名前/sigintSDR     # 実際のパスに置き換え
   ```
3. セットアップを実行（**`bash` を付けて起動**。exFATは実行ビットを持てないため）:
   ```
   bash setup.sh
   ```
   - 途中で **sudo パスワード**を聞かれる（管理者権限が必要）。
   - OS（Fedora/Ubuntu/Kali/Arch）を自動で見分けて、必要なものを入れる。
   - パッケージで入らない部分は**自動でソースからビルド**して補う。

> 何も入れずに今の状態だけ見たいときは: `bash setup.sh check`

## 4. つないで確認

```
hackrf_info          # HackRF が見えるか（シリアル番号が出ればOK）
SoapySDRUtil --find  # SoapySDR から見えるか
python3 main.py --sim --once   # プログラム自体の確認（ハード不要）
```

うまくいったら、実機で2.4GHz帯を収集:
```
python3 main.py --hardware --start 2.4e9 --stop 2.5e9 --collect captures/
```

---

## 5. つまずいたら

- **`hackrf_info` で「Resource busy」やデバイスが見えない**
  → HackRF を一度抜いて挿し直す。`setup.sh` 直後は、権限反映のため
  **一度ログアウト→ログイン**が必要なこともある。
- **git が "dubious ownership" と警告する**（USB上のリポジトリでよく出る）
  → 表示されたコマンド、または次を実行:
  ```
  git config --global --add safe.directory /run/media/$USER/USBの名前/sigintSDR
  ```
- **exFAT が読めない（古いLinux）**
  → 通常は新しめのカーネルで自動対応。古い場合のみ exfat ドライバを追加導入。
- **`bash setup.sh check` で hackrf モジュールが「なし」のまま**
  → `setup.sh`（引数なし）を再実行。ソースビルドにはネット接続が必要。

---

## メモ

- データ（`captures/`）は機械ごとにローカルに溜まる。複数機で集めたデータを1つに
  まとめたいときは、各機の `captures/` をコピーして集約し、`python3 -m dataset stats`
  で確認できる。**合成(sim)と実機データは混ぜない**設計なので、出所(`core:hw`)は
  自動で区別される。
- Python の部品はシステムに入れる方針（SoapySDR のバインディングがシステム側にあるため）。
  仮想環境(venv)を使いたい場合は `--system-site-packages` 付きで作ること。
