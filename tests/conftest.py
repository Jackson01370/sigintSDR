"""
pytest 共通設定。

tests/ はリポジトリ直下のモジュール(spec.py / sigmf_io.py ...)を import するので、
リポジトリルートを sys.path 先頭に入れておく（pytest の import モードに依存しない）。
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
