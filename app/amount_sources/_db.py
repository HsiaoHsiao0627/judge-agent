# -*- coding: utf-8 -*-
"""_db.py — 共用的唯讀 SQLite 連線工具（TPL/RSA 比較池共用）。"""
import sqlite3


def open_readonly(db_path: str):
    """唯讀連線，允許跨執行緒使用（check_same_thread=False）。
    這是安全的：FastAPI 用執行緒池處理請求，啟動時建立的連線可能被不同執行緒
    的請求使用；因為這裡永遠只有讀取、沒有寫入，關掉 SQLite 預設的執行緒歸屬檢查
    不會有資料損毀風險。若未來這個連線也要寫入，這個假設就不成立，需要重新設計。"""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
