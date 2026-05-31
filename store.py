"""検出ログ（SQLite）。軽量。"""
from __future__ import annotations
import sqlite3
import time


class Store:
    def __init__(self, path: str = "sigscan.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, center_hz REAL, bw_hz REAL,
                snr_db REAL, label TEXT, confidence REAL,
                method TEXT, notes TEXT
            )""")
        self.conn.commit()

    def log(self, m: dict, result) -> None:
        self.conn.execute(
            "INSERT INTO detections (ts,center_hz,bw_hz,snr_db,label,confidence,method,notes)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), m["center_hz"], m["bw_hz"], m["snr_db"],
             result.label, result.confidence, result.method, result.notes))
        self.conn.commit()

    def recent(self, n: int = 20):
        cur = self.conn.execute(
            "SELECT ts,center_hz,bw_hz,snr_db,label,confidence,method"
            " FROM detections ORDER BY id DESC LIMIT ?", (n,))
        return cur.fetchall()

    def close(self):
        self.conn.close()
